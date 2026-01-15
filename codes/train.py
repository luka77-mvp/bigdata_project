from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import log, exp, col
import happybase
import time
import subprocess

def delete_hdfs_model(path):
    """删除 HDFS 上的旧模型"""
    try:
        subprocess.run(['hdfs', 'dfs', '-rm', '-r', path], 
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print(f">>> 已删除旧模型: {path}")
    except:
        pass

# --- 初始化 ---
spark = SparkSession.builder \
    .appName("Train_Models_With_Performance_Metrics") \
    .enableHiveSupport() \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 连接 HBase
print(">>> [系统] 正在连接 HBase...")
connection = happybase.Connection('node01')
connection.open()

# 确保表存在
if b'model_metrics' not in connection.tables():
    connection.create_table('model_metrics', {'metrics': dict()})

metrics_table = connection.table('model_metrics')

# 使用 HDFS 路径（分布式存储）
base_path = "hdfs:///bigdata_project/models"

print("=" * 80)
print("🚀 开始执行模型优化训练任务 (Grid Search + Cross Validation)")
print("=" * 80)

# ==========================================
# Model B: 票房预测模型（优化版 + 性能记录）
# ==========================================
print("\n>>> [Model B] 正在准备数据...")
start_time_b = time.time()

df_b = spark.sql("""
    SELECT budget, popularity, runtime, revenue 
    FROM movie_db.movies 
    WHERE revenue > 100000 AND budget > 100000 
    AND popularity > 0.1 AND runtime > 60
""")

train_count_b = df_b.count()

# 对数变换
df_b = df_b.withColumn("log_budget", log(col("budget"))) \
           .withColumn("log_popularity", log(col("popularity"))) \
           .withColumn("log_revenue", log(col("revenue")))

assembler_b = VectorAssembler(
    inputCols=['log_budget', 'log_popularity', 'runtime'], 
    outputCol='features', handleInvalid="skip"
)
data_b = assembler_b.transform(df_b)
train_b, test_b = data_b.randomSplit([0.8, 0.2], seed=42)

# --- 核心优化部分 ---
print(">>> [Model B] 配置超参数网格...")
rf_b = RandomForestRegressor(featuresCol='features', labelCol='log_revenue')

param_grid_b = ParamGridBuilder() \
    .addGrid(rf_b.numTrees, [50, 100]) \
    .addGrid(rf_b.maxDepth, [10, 15]) \
    .build()

evaluator_b = RegressionEvaluator(
    labelCol="log_revenue", predictionCol="prediction", metricName="rmse"
)

cv_b = CrossValidator(
    estimator=rf_b,
    estimatorParamMaps=param_grid_b,
    evaluator=evaluator_b,
    numFolds=3,
    parallelism=2
)

print(">>> [Model B] 开始交叉验证训练 (这可能需要几分钟)...")
cv_model_b = cv_b.fit(train_b)
best_model_b = cv_model_b.bestModel

best_trees_b = best_model_b.getNumTrees
best_depth_b = best_model_b.getOrDefault('maxDepth')

# 计算测试集性能
test_rmse_b = evaluator_b.evaluate(best_model_b.transform(test_b))

# 计算 R² (决定系数)
r2_evaluator = RegressionEvaluator(labelCol="log_revenue", predictionCol="prediction", metricName="r2")
r2_b = r2_evaluator.evaluate(best_model_b.transform(test_b))

training_time_b = time.time() - start_time_b

print(f">>> [Model B] 最优参数: Trees={best_trees_b}, Depth={best_depth_b}")
print(f">>> [Model B] 测试集 RMSE: {test_rmse_b:.4f}")
print(f">>> [Model B] 测试集 R²: {r2_b:.4f}")
print(f">>> [Model B] 训练耗时: {training_time_b:.2f} 秒")

# 保存性能指标到 HBase
metrics_table.put(b'model_b', {
    b'metrics:name': '随机森林票房预测'.encode('utf-8'),
    b'metrics:rmse': f'{test_rmse_b:.4f}'.encode('utf-8'),
    b'metrics:r2': f'{r2_b:.4f}'.encode('utf-8'),
    b'metrics:trees': str(best_trees_b).encode('utf-8'),
    b'metrics:depth': str(best_depth_b).encode('utf-8'),
    b'metrics:train_samples': str(train_count_b).encode('utf-8'),
    b'metrics:training_time': f'{training_time_b:.2f}'.encode('utf-8')
})
print(">>> [Model B] 性能指标已保存到 HBase")

# 保存最优模型
model_b_path = f"{base_path}/revenue_model"
delete_hdfs_model(model_b_path)
best_model_b.save(model_b_path)
print(f">>> [Model B] 最优模型已保存到 {model_b_path}")

# ==========================================
# Model C: 成败分类模型（优化版 + 性能记录）
# ==========================================
print("\n>>> [Model C] 正在准备数据...")
start_time_c = time.time()

df_c = spark.sql("""
    SELECT budget, popularity, runtime, revenue 
    FROM movie_db.movies 
    WHERE budget > 100000 AND runtime IS NOT NULL
""")

train_count_c = df_c.count()

df_c = df_c.withColumn("label", (df_c["revenue"] > df_c["budget"]).cast("double"))
df_c = df_c.withColumn("log_budget", log(col("budget"))) \
           .withColumn("log_popularity", log(col("popularity")))

assembler_c = VectorAssembler(
    inputCols=['log_budget', 'log_popularity', 'runtime'], 
    outputCol='features', handleInvalid="skip"
)
data_c = assembler_c.transform(df_c)
train_c, test_c = data_c.randomSplit([0.8, 0.2], seed=42)

# --- 核心优化部分 ---
print(">>> [Model C] 配置超参数网格...")
rf_c = RandomForestClassifier(featuresCol='features', labelCol='label')

param_grid_c = ParamGridBuilder() \
    .addGrid(rf_c.numTrees, [50, 100]) \
    .addGrid(rf_c.maxDepth, [10, 15]) \
    .build()

evaluator_c = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)

cv_c = CrossValidator(
    estimator=rf_c,
    estimatorParamMaps=param_grid_c,
    evaluator=evaluator_c,
    numFolds=3,
    parallelism=2
)

print(">>> [Model C] 开始交叉验证训练...")
cv_model_c = cv_c.fit(train_c)
best_model_c = cv_model_c.bestModel

best_trees_c = best_model_c.getNumTrees
best_depth_c = best_model_c.getOrDefault('maxDepth')

# 计算测试集性能
accuracy = evaluator_c.evaluate(best_model_c.transform(test_c))

# 计算 F1 分数
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
f1_score = f1_evaluator.evaluate(best_model_c.transform(test_c))

training_time_c = time.time() - start_time_c

print(f">>> [Model C] 最优参数: Trees={best_trees_c}, Depth={best_depth_c}")
print(f">>> [Model C] 验证集准确率: {accuracy:.4f}")
print(f">>> [Model C] F1 分数: {f1_score:.4f}")
print(f">>> [Model C] 训练耗时: {training_time_c:.2f} 秒")

# 保存性能指标到 HBase
metrics_table.put(b'model_c', {
    b'metrics:name': '随机森林成败分类'.encode('utf-8'),
    b'metrics:accuracy': f'{accuracy:.4f}'.encode('utf-8'),
    b'metrics:f1': f'{f1_score:.4f}'.encode('utf-8'),
    b'metrics:trees': str(best_trees_c).encode('utf-8'),
    b'metrics:depth': str(best_depth_c).encode('utf-8'),
    b'metrics:train_samples': str(train_count_c).encode('utf-8'),
    b'metrics:training_time': f'{training_time_c:.2f}'.encode('utf-8')
})
print(">>> [Model C] 性能指标已保存到 HBase")

# 保存最优模型
model_c_path = f"{base_path}/classify_model"
delete_hdfs_model(model_c_path)
best_model_c.save(model_c_path)
print(f">>> [Model C] 最优模型已保存到 {model_c_path}")

print("\n" + "=" * 80)
print("✅ 所有模型训练完成！")
print("=" * 80)

connection.close()
spark.stop()
