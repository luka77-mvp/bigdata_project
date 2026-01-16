from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler, StandardScaler
from pyspark.sql.functions import col
import happybase
import time

# --- 1. 初始化 Spark ---
spark = SparkSession.builder \
    .appName("ETL_With_Performance_Metrics") \
    .enableHiveSupport() \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# --- 2. 连接 HBase ---
print("正在连接 HBase...")
connection = happybase.Connection('node01')
connection.open()

# 确保表存在
if b'user_recs' not in connection.tables():
    connection.create_table('user_recs', {'recs': dict()})
if b'movie_info' not in connection.tables():
    connection.create_table('movie_info', {'info': dict()})
if b'model_metrics' not in connection.tables():
    connection.create_table('model_metrics', {'metrics': dict()})

recs_table = connection.table('user_recs')
movie_table = connection.table('movie_info')
metrics_table = connection.table('model_metrics')

# ==========================================
# 3. Model A: ALS 推荐模型 (优化版 + 性能记录)
# ==========================================
print("\nModel A正在优化训练 ALS 模型...")
start_time_a = time.time()

ratings = spark.sql("SELECT userId, movieId, rating FROM movie_db.ratings")
train_count_a = ratings.count()

# 划分数据集
(training, test) = ratings.randomSplit([0.8, 0.2])

# 定义 ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", 
          coldStartStrategy="drop", nonnegative=True)

# 定义网格
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [20, 50]) \
    .addGrid(als.regParam, [0.05, 0.1]) \
    .build()

# 定义评估器
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# 交叉验证
cv = CrossValidator(estimator=als,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=3) 

print("正在进行 Grid Search (寻找最优参数)...")
model_cv = cv.fit(training)
best_model_als = model_cv.bestModel

best_rank = best_model_als.rank
best_reg = best_model_als._java_obj.parent().getRegParam()
rmse = evaluator.evaluate(best_model_als.transform(test))

training_time_a = time.time() - start_time_a

print(f"最优参数: Rank={best_rank}, RegParam={best_reg}")
print(f"测试集 RMSE: {rmse:.4f}")
print(f"训练耗时: {training_time_a:.2f} 秒")

# 保存性能指标到 HBase
metrics_table.put(b'model_a', {
    b'metrics:name': 'ALS 协同过滤推荐'.encode('utf-8'),
    b'metrics:rmse': f'{rmse:.4f}'.encode('utf-8'),
    b'metrics:rank': str(best_rank).encode('utf-8'),
    b'metrics:regParam': f'{best_reg:.4f}'.encode('utf-8'),
    b'metrics:train_samples': str(train_count_a).encode('utf-8'),
    b'metrics:training_time': f'{training_time_a:.2f}'.encode('utf-8')
})
print("性能指标已保存到 HBase")

print("保存推荐结果到 HBase...")
user_recs = best_model_als.recommendForAllUsers(5)
pdf_recs = user_recs.toPandas()

batch = recs_table.batch()
for index, row in pdf_recs.iterrows():
    uid = str(row['userId'])
    recs_str = ",".join([f"{r['movieId']}:{r['rating']:.2f}" for r in row['recommendations']])
    batch.put(uid, {'recs:list': recs_str})
batch.send()
print("完成！")

print("正在保存模型文件到 HDFS...")
als_model_path = "hdfs:///bigdata_project/models/als_model"
try:
    import subprocess
    subprocess.run(['hdfs', 'dfs', '-rm', '-r', '/bigdata_project/models/als_model'], 
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    print("已删除旧模型")
except:
    pass

best_model_als.save(als_model_path)
print("模型文件已保存到 HDFS！")

# ==========================================
# 4. 数据同步（基础信息）
# ==========================================
print("\n正在将电影元数据从 Hive 同步到 HBase...")
df_movies_basic = spark.sql("""
    SELECT id, title, genres, budget, revenue
    FROM movie_db.movies 
    WHERE id IS NOT NULL
""")
pdf_movies_basic = df_movies_basic.toPandas()

batch = movie_table.batch()
for index, row in pdf_movies_basic.iterrows():
    mid = str(row['id'])
    batch.put(mid, {
        'info:title': str(row['title']),
        'info:genres': str(row['genres']) if row['genres'] else '-',
        'info:budget': str(row['budget']),
        'info:revenue': str(row['revenue'])
    })
batch.send()
print(f"完成！同步了 {len(pdf_movies_basic)} 条数据。")

# ==========================================
# 5. Model D: K-Means
# ==========================================
print("\nModel D正在训练 K-Means 聚类模型...")
start_time_d = time.time()

df_movies = spark.sql("""
    SELECT 
        id, 
        title,
        concat(genres, ' ', keywords) as text,
        budget,
        popularity,
        vote_average
    FROM movie_db.movies 
    WHERE genres IS NOT NULL 
    AND budget > 0 
    AND popularity > 0
""")
train_count_d = df_movies.count()
print(f"有效聚类数据量: {train_count_d} 条")

print("提取文本特征...")
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df_movies)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=300)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="text_features")
idfModel = idf.fit(featurizedData)
textData = idfModel.transform(featurizedData)

print(" 标准化数值特征...")
assembler_numeric = VectorAssembler(
    inputCols=['budget', 'popularity', 'vote_average'],
    outputCol='numeric_features_raw',
    handleInvalid='skip'
)
numericData = assembler_numeric.transform(textData)

scaler = StandardScaler(inputCol="numeric_features_raw", outputCol="features", withMean=True, withStd=True)
scalerModel = scaler.fit(numericData)
finalData = scalerModel.transform(numericData)

print("评估当前配置 (K=5, 仅数值特征)...")
kmeans = KMeans().setK(5).setSeed(42).setMaxIter(30).setInitMode("k-means||")
model_km = kmeans.fit(finalData)
predictions = model_km.transform(finalData)

evaluator_clustering = ClusteringEvaluator()
silhouette = evaluator_clustering.evaluate(predictions)

training_time_d = time.time() - start_time_d

print(f"轮廓系数 (Silhouette Score) = {silhouette:.4f}")
print(f"训练耗时: {training_time_d:.2f} 秒")


metrics_table.put(b'model_d', {
    b'metrics:name': 'K-Means 电影聚类'.encode('utf-8'),
    b'metrics:silhouette': f'{silhouette:.4f}'.encode('utf-8'),
    b'metrics:k': '5'.encode('utf-8'),
    b'metrics:train_samples': str(train_count_d).encode('utf-8'),
    b'metrics:training_time': f'{training_time_d:.2f}'.encode('utf-8')
})
print("性能指标已保存到 HBase")

print("\n分析聚类特征并写入 HBase...")
cluster_names = {}
cluster_stats = []

for cluster_id in range(5):
    cluster_data = predictions.filter(col("prediction") == cluster_id)
    count = cluster_data.count()
    if count > 0:
        avg_budget = cluster_data.agg({"budget": "avg"}).first()[0]
        avg_popularity = cluster_data.agg({"popularity": "avg"}).first()[0]
        avg_vote = cluster_data.agg({"vote_average": "avg"}).first()[0]
        
        cluster_stats.append({
            'id': cluster_id,
            'count': count,
            'budget': avg_budget,
            'popularity': avg_popularity,
            'vote': avg_vote
        })

cluster_stats.sort(key=lambda x: x['budget'], reverse=True)

for idx, stats in enumerate(cluster_stats):
    cluster_id = stats['id']
    count = stats['count']
    avg_budget = stats['budget']
    avg_popularity = stats['popularity']
    avg_vote = stats['vote']
  
    # 超高预算（1亿美元以上）
    if avg_budget > 100000000:
        if avg_popularity > 80:
            name = "超级大片"
        else:
            name = "高投资制作"
    
    # 高预算（5000万-1亿）
    elif avg_budget > 50000000:
        if avg_popularity > 40:
            name = "热门商业片"
        else:
            name = "高成本制作"
    
    # 中高预算（2500万-5000万）
    elif avg_budget > 25000000:
        if avg_popularity > 15 and avg_vote > 6.5:
            name = "口碑佳作"
        elif avg_popularity < 5:
            name = "冷门中等片"
        elif avg_vote > 6.0:
            name = "中等制作精品"
        else:
            name = "中等制作"
    
    # 中小预算（1000万-2500万）- 修正区间
    elif avg_budget > 10000000:
        if avg_popularity > 20 and avg_vote > 6.5:
            name = "热门中小片"  # 热度高且评分好
        elif avg_vote > 6.8:
            name = "中小成本佳片"  # 评分很高
        elif avg_popularity > 15:
            name = "中小成本热片"  # 热度较高
        else:
            name = "中小成本"
    
    # 低预算（500万-1000万）
    elif avg_budget > 5000000:
        if avg_vote > 7.0:
            name = "独立佳片"
        elif avg_popularity > 15:
            name = "小成本热片"
        elif avg_vote > 6.5:
            name = "小成本精品"
        else:
            name = "小成本制作"
    
    # 超低预算（<500万）
    else:
        if avg_vote > 7.0:
            name = "独立佳片"
        elif avg_popularity < 5 and avg_vote < 5.5:
            name = "小众低分片"
        else:
            name = "超低成本"
    
    cluster_names[cluster_id] = name
    print(f"  聚类 {cluster_id}: {name} (数量: {count}, 平均预算: ${avg_budget:,.0f}, 热度: {avg_popularity:.1f}, 评分: {avg_vote:.1f})")

# 7. 保存结果到 HBase
pdf_clusters = predictions.select("id", "prediction").toPandas()

batch = movie_table.batch()
for index, row in pdf_clusters.iterrows():
    mid = str(row['id'])
    cluster_id = int(row['prediction'])
    cluster_name = cluster_names.get(cluster_id, f"类别{cluster_id}")
    
    batch.put(mid, {
        'info:cluster': str(cluster_id),
        'info:cluster_name': cluster_name
    })
batch.send()

print(f"完成！已保存 {len(pdf_clusters)} 部电影的聚类结果。")
print("所有离线任务已完成。")

connection.close()
spark.stop()
