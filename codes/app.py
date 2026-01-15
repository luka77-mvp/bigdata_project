from flask import Flask, render_template_string, request, session, redirect, url_for
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType
import happybase
import time
import math
import random
import numpy as np
from contextlib import contextmanager

app = Flask(__name__)

# ==========================================
# HBase 连接管理器（消除冗余连接代码）
# ==========================================
HBASE_HOST = 'node01'

@contextmanager
def get_hbase_conn():
    """HBase 连接上下文管理器，自动处理连接关闭"""
    conn = None
    try:
        conn = happybase.Connection(HBASE_HOST)
        conn.open()
        yield conn
    finally:
        if conn: conn.close()

def format_currency(value, default='$0'):
    """格式化货币显示"""
    try:
        return "${:,.0f}".format(float(value))
    except:
        return default
app.secret_key = 'bigdata_project_secret_key_final'

# ==========================================
# 1. 系统初始化
# ==========================================
print(">>> [System] 正在启动 Spark 引擎 (Local Mode)...")
spark = SparkSession.builder \
    .appName("Web_App_Final_Realtime_ALS") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 全局变量：存储 ALS 模型的电影特征向量 {movieId: [vector...]}
MOVIE_FACTORS_DICT = {} 

print(">>> [System] 正在加载离线模型...")
try:
    # 使用 HDFS 路径（分布式存储）
    # 1. 加载 Model B (票房预测)
    model_b_path = "hdfs:///bigdata_project/models/revenue_model"
    loaded_model_b = RandomForestRegressionModel.load(model_b_path)
    
    # 2. 加载 Model C (成败分类)
    model_c_path = "hdfs:///bigdata_project/models/classify_model"
    loaded_model_c = RandomForestClassificationModel.load(model_c_path)

    # 3. 加载 Model A (ALS) 并提取特征向量
    print(">>> [System] 正在加载 ALS 模型并提取特征向量...")
    model_a_path = "hdfs:///bigdata_project/models/als_model"
    
    try:
        loaded_model_als = ALSModel.load(model_a_path)
        item_factors_df = loaded_model_als.itemFactors.toPandas()
        for index, row in item_factors_df.iterrows():
            MOVIE_FACTORS_DICT[int(row['id'])] = np.array(row['features'])
        print(f">>> [Success] ALS 向量加载成功！缓存了 {len(MOVIE_FACTORS_DICT)} 部电影特征。")
    except Exception as e:
        print(f">>> [Warning] 未找到 ALS 模型: {e}")

    print(">>> [Success] 所有模型从 HDFS 加载完成！")

except Exception as e:
    print(f">>> [Warning] 模型加载过程中出现错误: {e}")
    loaded_model_b = None
    loaded_model_c = None

# ==========================================
# 2. 索引构建 + 题材缓存
# ==========================================
ALL_MOVIE_IDS = []
GENRE_CACHE = {}

def init_movie_index():
    """构建全量索引并缓存题材信息"""
    global ALL_MOVIE_IDS, GENRE_CACHE
    print(">>> [System] 正在连接 HBase 构建全量索引...")
    ids = []
    genres_dict = {}
    try:
        with get_hbase_conn() as connection:
            if b'movie_info' in connection.tables():
                table = connection.table('movie_info')
                for key, data in table.scan(columns=[b'info:genres']):
                    movie_id = key.decode('utf-8')
                    ids.append(movie_id)
                    genres = data.get(b'info:genres', b'').decode('utf-8')
                    genres_dict[movie_id] = genres
                
                ALL_MOVIE_IDS = sorted(ids, key=lambda x: int(x) if x.isdigit() else 0)
                GENRE_CACHE = genres_dict
                print(f">>> [Success] 索引构建完成！共发现 {len(ALL_MOVIE_IDS)} 部电影。")
                print(f">>> [Success] 题材缓存完成！")
            else:
                print(">>> [Warning] HBase 表 'movie_info' 不存在")
    except Exception as e:
        print(f">>> [Error] 无法连接 HBase 或表不存在: {e}")
        ALL_MOVIE_IDS = []
        GENRE_CACHE = {}

init_movie_index()

# ==========================================
# 3. 辅助功能
# ==========================================
GENRE_MAP = {
    'Action': '动作', 'Adventure': '冒险', 'Animation': '动画', 'Comedy': '喜剧',
    'Crime': '犯罪', 'Documentary': '纪录片', 'Drama': '剧情', 'Family': '家庭',
    'Fantasy': '奇幻', 'History': '历史', 'Horror': '恐怖', 'Music': '音乐',
    'Mystery': '悬疑', 'Romance': '爱情', 'Science Fiction': '科幻', 'TV Movie': '电视电影',
    'Thriller': '惊悚', 'War': '战争', 'Western': '西部', 'Foreign': '外语'
}

def translate_genres(genre_str):
    if not genre_str or genre_str == '-': return '未知'
    parts = genre_str.replace('|', ',').split(',')
    translated = []
    for p in parts:
        p = p.strip()
        translated.append(GENRE_MAP.get(p, p)) 
    return ', '.join(translated)


# ==========================================
# 4. 图表数据获取（4个图表版本）
# ==========================================
def get_chart_data():
    """获取4个图表的数据"""
    genre_counts = {}
    scatter_data = []
    cluster_counts = {}
    budget_by_cluster = {}
    
    try:
        with get_hbase_conn() as conn:
            if b'movie_info' not in conn.tables(): 
                return [], [], [], []
            
            table = conn.table('movie_info')
            
            for key, data in table.scan():
                g_str = data.get(b'info:genres', b'').decode('utf-8')
                if g_str and g_str != '-':
                    main_genre = g_str.replace('|', ',').split(',')[0].strip()
                    cn_genre = GENRE_MAP.get(main_genre, main_genre)
                    if cn_genre:
                        genre_counts[cn_genre] = genre_counts.get(cn_genre, 0) + 1
                
                try:
                    bud = float(data.get(b'info:budget', b'0'))
                    rev = float(data.get(b'info:revenue', b'0'))
                    if bud > 10000 and rev > 10000 and rev < bud * 100:
                        scatter_data.append([bud, rev])
                except: pass
                
                cluster_name = data.get(b'info:cluster_name', b'').decode('utf-8')
                if cluster_name:
                    cluster_counts[cluster_name] = cluster_counts.get(cluster_name, 0) + 1
                
                try:
                    bud = float(data.get(b'info:budget', b'0'))
                    if bud > 0 and cluster_name:
                        if cluster_name not in budget_by_cluster:
                            budget_by_cluster[cluster_name] = []
                        budget_by_cluster[cluster_name].append(bud)
                except: pass
    except Exception as e:
        print(f"Chart Data Error: {e}")
        return [], [], [], []
    
    # 处理数据
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    pie_data = [{"name": k, "value": v} for k, v in sorted_genres[:10]]
    
    if len(scatter_data) > 2000: 
        scatter_data = random.sample(scatter_data, 2000)
    
    cluster_pie_data = [{"name": k, "value": v} for k, v in cluster_counts.items()]
    
    # 计算每个聚类的平均预算
    budget_bar_data = []
    for cluster_name in sorted(budget_by_cluster.keys()):
        avg_budget = sum(budget_by_cluster[cluster_name]) / len(budget_by_cluster[cluster_name])
        budget_bar_data.append({"name": cluster_name, "value": int(avg_budget)})
    
    return pie_data, scatter_data, cluster_pie_data, budget_bar_data

def get_model_metrics():
    """从 HBase 读取模型性能指标"""
    metrics = []
    try:
        with get_hbase_conn() as conn:
            if b'model_metrics' not in conn.tables():
                return []
            
            table = conn.table('model_metrics')
            
            for model_key in [b'model_a', b'model_b', b'model_c', b'model_d']:
                try:
                    row = table.row(model_key)
                    if row:
                        metric = {
                            'model_id': model_key.decode('utf-8'),
                            'name': row.get(b'metrics:name', b'').decode('utf-8'),
                            'train_samples': row.get(b'metrics:train_samples', b'0').decode('utf-8'),
                            'training_time': row.get(b'metrics:training_time', b'0').decode('utf-8')
                        }
                        
                        if model_key == b'model_a':
                            metric['rmse'] = row.get(b'metrics:rmse', b'').decode('utf-8')
                            metric['rank'] = row.get(b'metrics:rank', b'').decode('utf-8')
                            metric['regParam'] = row.get(b'metrics:regParam', b'').decode('utf-8')
                        elif model_key == b'model_b':
                            metric['rmse'] = row.get(b'metrics:rmse', b'').decode('utf-8')
                            metric['r2'] = row.get(b'metrics:r2', b'').decode('utf-8')
                            metric['trees'] = row.get(b'metrics:trees', b'').decode('utf-8')
                            metric['depth'] = row.get(b'metrics:depth', b'').decode('utf-8')
                        elif model_key == b'model_c':
                            metric['accuracy'] = row.get(b'metrics:accuracy', b'').decode('utf-8')
                            metric['f1'] = row.get(b'metrics:f1', b'').decode('utf-8')
                            metric['trees'] = row.get(b'metrics:trees', b'').decode('utf-8')
                            metric['depth'] = row.get(b'metrics:depth', b'').decode('utf-8')
                        elif model_key == b'model_d':
                            metric['silhouette'] = row.get(b'metrics:silhouette', b'').decode('utf-8')
                            metric['k'] = row.get(b'metrics:k', b'').decode('utf-8')
                        
                        metrics.append(metric)
                except Exception as e:
                    print(f"Error reading {model_key}: {e}")
                    continue
    except Exception as e:
        print(f"HBase connection error: {e}")
        return []
    
    return metrics


# ==========================================
# 5. 数据库逻辑
# ==========================================
def get_cluster_badge(cluster_name):
    """根据聚类名称返回对应的徽章颜色"""
    if "超级" in cluster_name or "高投资" in cluster_name: return "badge-red"
    if "商业" in cluster_name or "高成本" in cluster_name: return "badge-orange"
    if "数据缺失" in cluster_name or "小众低分" in cluster_name: return "badge-secondary"
    if "中等" in cluster_name: return "badge-yellow"
    if "中小" in cluster_name: return "badge-purple"
    if "热门" in cluster_name or "网络" in cluster_name: return "badge-blue"
    if "口碑" in cluster_name or "佳片" in cluster_name or "佳作" in cluster_name or "精品" in cluster_name: return "badge-green"
    if "独立" in cluster_name or "文艺" in cluster_name: return "badge-teal"
    if "小成本" in cluster_name: return "badge-purple"
    if "冷门" in cluster_name: return "badge-secondary"
    if "未分类" in cluster_name: return "badge-secondary"
    return "badge-secondary" 

def format_movie_data(mid, data):
    title = data.get(b'info:title', b'Unknown').decode('utf-8')
    genres_cn = translate_genres(data.get(b'info:genres', b'-').decode('utf-8'))
    budget_str = data.get(b'info:budget', b'0').decode('utf-8')
    cluster_name = data.get(b'info:cluster_name', b'').decode('utf-8')
    
    if not cluster_name: 
        analysis_text = "未归类 (Model D Missing)"
        badge_class = "badge-secondary"
    else:
        analysis_text = cluster_name 
        badge_class = get_cluster_badge(cluster_name)

    return {
        'id': mid, 'title': title, 'genres': genres_cn,
        'budget': format_currency(budget_str), 'analysis': analysis_text, 'badge_class': badge_class
    }

def search_movie_by_id(target_id):
    try:
        with get_hbase_conn() as conn:
            table = conn.table('movie_info')
            row = table.row(str(target_id).encode('utf-8'))
            if not row: return [] 
            return [format_movie_data(str(target_id), row)]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def get_hbase_detail(movie_id):
    try:
        with get_hbase_conn() as conn:
            table = conn.table('movie_info')
            row = table.row(str(movie_id).encode('utf-8'))
            if not row: return None
            return {
                'title': row.get(b'info:title', b'Unknown').decode('utf-8'),
                'genres': translate_genres(row.get(b'info:genres', b'-').decode('utf-8')),
                'budget': format_currency(row.get(b'info:budget', b'0').decode('utf-8')),
                'revenue': format_currency(row.get(b'info:revenue', b'0').decode('utf-8')),
                'cluster_id': row.get(b'info:cluster', b'-').decode('utf-8'),
                'cluster_name': row.get(b'info:cluster_name', b'Unknown').decode('utf-8')
            }
    except Exception as e:
        return None

def _fetch_movies_from_hbase(page_ids):
    """从 HBase 批量获取电影数据的内部函数"""
    movies = []
    try:
        with get_hbase_conn() as conn:
            table = conn.table('movie_info')
            row_keys = [str(mid).encode('utf-8') for mid in page_ids]
            rows = table.rows(row_keys)
            data_dict = {key.decode('utf-8'): data for key, data in rows}
            
            for mid in page_ids:
                if mid in data_dict:
                    movies.append(format_movie_data(mid, data_dict[mid]))
    except Exception as e:
        print(f"HBase Error: {e}")
    return movies

def get_movies_by_page(page_num=1, page_size=20, genre_filter=None):
    """分页获取电影数据，支持题材筛选"""
    if not ALL_MOVIE_IDS: init_movie_index()
    if not ALL_MOVIE_IDS: return [], 0
    
    # 根据筛选条件确定 ID 列表
    if genre_filter and genre_filter != 'all':
        source_ids = [mid for mid in ALL_MOVIE_IDS if genre_filter in GENRE_CACHE.get(mid, '')]
    else:
        source_ids = ALL_MOVIE_IDS
    
    # 计算分页
    total_count = len(source_ids)
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
    page_num = max(1, min(page_num, total_pages))
    
    start_idx = (page_num - 1) * page_size
    page_ids = source_ids[start_idx:start_idx + page_size]
    
    movies = _fetch_movies_from_hbase(page_ids)
    if not movies and genre_filter:
        return [], 0
    
    return movies, total_pages


# ==========================================
# 6. 预测逻辑 (Model B & C)
# ==========================================
def _prepare_prediction_data(budget, pop, runtime):
    """准备预测所需的特征数据（消除 Model B/C 的重复代码）"""
    b = float(budget); p = float(pop); r = float(runtime)
    safe_b = max(b, 1000.0); safe_p = max(p, 0.1)
    data = [(math.log(safe_b), math.log(safe_p), r)]
    df = spark.createDataFrame(data, ["log_budget", "log_popularity", "runtime"])
    vec_df = VectorAssembler(inputCols=['log_budget', 'log_popularity', 'runtime'], outputCol='features').transform(df)
    return b, p, r, vec_df

def predict_revenue_real(budget, pop, runtime):
    if not loaded_model_b: return "模型未加载"
    try:
        b, p, r, vec_df = _prepare_prediction_data(budget, pop, runtime)
        raw_res = math.exp(loaded_model_b.transform(vec_df).select("prediction").first()[0])
        final_res = raw_res
        if p < 5: final_res *= 0.3
        elif p < 15: final_res *= 0.6
        if b < 5000000 and p < 10: final_res = min(final_res, b * 5.0)
        if p < 100: final_res = min(final_res, b * 20.0)
        return "{:,.2f}".format(max(final_res, b * 0.5))
    except Exception as e: return f"Error: {e}"

def predict_classify_real(budget, pop, runtime):
    if not loaded_model_c: return "模型未加载"
    try:
        b, p, r, vec_df = _prepare_prediction_data(budget, pop, runtime)
        if b < 1000000 and p < 5: return "亏损 (Loss)"
        if b > 100000000 and p < 30: return "亏损 (Loss)"
        res = loaded_model_c.transform(vec_df).select("prediction").first()[0]
        return "盈利 (Profit)" if res == 1.0 else "亏损 (Loss)"
    except Exception as e: return f"Error: {e}"

# ==========================================
# 7. 实时推荐逻辑 (Model A - Realtime)
# ==========================================
def calculate_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    if vec1 is None or vec2 is None: return -1
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0: return 0
    return dot_product / (norm_a * norm_b)

def run_realtime_als(new_uid, liked_mids):
    """真正的实时推荐：根据输入电影的向量，寻找相似向量"""
    if not MOVIE_FACTORS_DICT:
        print(">>> [Warning] 向量字典为空，无法推荐。")
        return [], 0
        
    start = time.time()
    
    target_mids = []
    try:
        parts = liked_mids.replace('，', ',').split(',')
        for p in parts:
            if p.strip().isdigit():
                target_mids.append(int(p.strip()))
    except: pass
        
    if not target_mids: return [], 0

    user_vector = None
    valid_count = 0
    
    for mid in target_mids:
        if mid in MOVIE_FACTORS_DICT:
            vec = MOVIE_FACTORS_DICT[mid]
            if user_vector is None:
                user_vector = vec
            else:
                user_vector = user_vector + vec 
            valid_count += 1
            
    if user_vector is None:
        return [], 0.1 
        
    user_vector = user_vector / valid_count
    
    candidates = []
    
    for mid, vec in MOVIE_FACTORS_DICT.items():
        if mid in target_mids: continue
            
        score = calculate_similarity(user_vector, vec)
        if score > 0.6:
            candidates.append((mid, score))
            
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_5 = candidates[:5]
    
    recs_list = []
    try:
        with get_hbase_conn() as conn:
            table = conn.table('movie_info')
            
            for mid, score in top_5:
                row = table.row(str(mid).encode('utf-8'))
                if row:
                    recs_list.append({
                        'id': mid, 
                        'title': row.get(b'info:title', b'Unknown').decode('utf-8'),
                        'genres': translate_genres(row.get(b'info:genres', b'-').decode('utf-8')),
                        'score': "{:.1f}".format(score * 5.0)
                    })
    except Exception as e:
        print(f"HBase Error: {e}")

    return recs_list, time.time() - start


# ==========================================
# 8. 前端界面 (HTML)
# ==========================================
html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>电影数据分析平台</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body { font-family: "Microsoft YaHei", sans-serif; background: #f4f6f9; display: flex; height: 100vh; margin: 0; }
        .sidebar { width: 250px; background: #2c3e50; color: white; display: flex; flex-direction: column; }
        .sidebar h2 { padding: 25px; text-align: center; border-bottom: 1px solid #4b545c; margin: 0; background: #222d32; }
        .nav-item { padding: 18px 25px; color: #b8c7ce; text-decoration: none; display: block; border-left: 4px solid transparent; transition: 0.3s; }
        .nav-item:hover { background: #34495e; }
        .nav-item.active { background: #3c8dbc; color: white; border-left-color: #fff; }
        
        .content { flex: 1; padding: 40px; overflow-y: auto; }
        .header { margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 15px; }
        
        .card { background: white; padding: 30px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-top: 4px solid #ccc; margin-bottom: 25px; }
        .card-blue { border-top-color: #007bff; } .card-green { border-top-color: #28a745; }
        .card-yellow { border-top-color: #ffc107; } .card-red { border-top-color: #dc3545; }
        .card-purple { border-top-color: #6f42c1; }
        
        .form-group { margin-bottom: 15px; }
        label { display: block; font-weight: bold; margin-bottom: 5px; color: #333; }
        input { padding: 10px; width: 300px; border: 1px solid #ccc; border-radius: 4px; }
        .hint { display: block; font-size: 12px; color: #888; margin-top: 4px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 4px; font-weight: bold; margin-top: 10px; }
        button:hover { background: #0056b3; }
        
        .loader { display: none; color: #ffc107; font-weight: bold; margin-top: 10px; }
        .result-box { background: #f8f9fa; border: 1px solid #e9ecef; padding: 15px; margin-top: 20px; border-radius: 4px; }
        
        .db-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
        .db-table th { background: #f1f3f5; padding: 10px; border-bottom: 2px solid #dee2e6; text-align: left; }
        .db-table td { padding: 10px; border-bottom: 1px solid #e9ecef; color: #333; vertical-align: middle; }
        .db-table tr:hover { background-color: #f8f9fa; }
        
        .badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; color: white; font-weight: bold; display: inline-block; min-width: 80px; text-align: center; }
        .badge-red { background: #e74c3c; } 
        .badge-orange { background: #e67e22; }
        .badge-yellow { background: #f39c12; }
        .badge-blue { background: #3498db; } 
        .badge-green { background: #27ae60; }
        .badge-teal { background: #20c997; } 
        .badge-purple { background: #6f42c1; }
        .badge-secondary { background: #6c757d; }

        .pagination { display: flex; gap: 5px; margin-top: 25px; align-items: center; }
        .page-link { padding: 6px 12px; border: 1px solid #dee2e6; color: #007bff; text-decoration: none; border-radius: 4px; background: white; font-size: 14px; }
        .page-link:hover { background: #e9ecef; }
        .page-link.active { background: #007bff; color: white; border-color: #007bff; }
        
        .search-bar { display: flex; gap: 10px; margin-bottom: 20px; align-items: flex-end; }
        
        .chart-container { display: flex; flex-wrap: wrap; gap: 20px; }
        .chart-box { width: 45%; min-width: 400px; height: 400px; border: 1px solid #eee; padding: 10px; }
    </style>
    <script>function load(){ document.getElementById('loader').style.display='block'; }</script>
</head>
<body>
    <div class="sidebar">
        <h2>电影数据分析平台</h2>
        <a href="/?tab=recsys" class="nav-item {% if tab=='recsys' %}active{% endif %}">Model A: 电影推荐</a>
        <a href="/?tab=predict" class="nav-item {% if tab=='predict' %}active{% endif %}">Model B: 票房预测</a>
        <a href="/?tab=classify" class="nav-item {% if tab=='classify' %}active{% endif %}">Model C: 投资成败预测</a>
        <a href="/?tab=cluster" class="nav-item {% if tab=='cluster' %}active{% endif %}">Model D: 电影聚类</a>
        <a href="/?tab=database" class="nav-item {% if tab=='database' %}active{% endif %}">数据库</a>
        <a href="/?tab=charts" class="nav-item {% if tab=='charts' %}active{% endif %}">可视化图表</a>
    </div>

    <div class="content">
        <div class="header"><h1>系统功能区</h1></div>

        {% if tab == 'charts' %}
        <div class="card card-green">
            <h3>全量数据可视化分析</h3>
            <p>基于 HBase 数据库的全量扫描与分析结果。</p>
            
            <div style="margin-bottom:30px;">
                <h4 style="margin-bottom:15px; color:#333;">模型性能指标</h4>
                <table class="db-table">
                    <thead>
                        <tr>
                            <th width="20%">模型名称</th>
                            <th width="15%">训练样本数</th>
                            <th width="30%">性能指标</th>
                            <th width="20%">模型参数</th>
                            <th width="15%">训练耗时</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for m in model_metrics %}
                        <tr>
                            <td><strong>{{ m.name }}</strong></td>
                            <td>{{ m.train_samples }}</td>
                            <td>
                                {% if m.model_id == 'model_a' %}
                                    <span style="color:#007bff; font-weight:bold;">RMSE: {{ m.rmse }}</span>
                                {% elif m.model_id == 'model_b' %}
                                    <span style="color:#ffc107; font-weight:bold;">RMSE: {{ m.rmse }}</span><br>
                                    <span style="color:#28a745; font-weight:bold;">R²: {{ m.r2 }}</span>
                                {% elif m.model_id == 'model_c' %}
                                    <span style="color:#dc3545; font-weight:bold;">准确率: {{ m.accuracy }}</span><br>
                                    <span style="color:#6f42c1; font-weight:bold;">F1: {{ m.f1 }}</span>
                                {% elif m.model_id == 'model_d' %}
                                    <span style="color:#20c997; font-weight:bold;">轮廓系数: {{ m.silhouette }}</span>
                                {% endif %}
                            </td>
                            <td>
                                {% if m.model_id == 'model_a' %}
                                    Rank: {{ m.rank }}<br>RegParam: {{ m.regParam }}
                                {% elif m.model_id == 'model_b' or m.model_id == 'model_c' %}
                                    Trees: {{ m.trees }}<br>Depth: {{ m.depth }}
                                {% elif m.model_id == 'model_d' %}
                                    K: {{ m.k }}
                                {% endif %}
                            </td>
                            <td>{{ m.training_time }} 秒</td>
                        </tr>
                        {% else %}
                        <tr>
                            <td colspan="5" style="text-align:center; padding:20px; color:#999;">
                                暂无性能数据，请先运行 <code>etl2.py</code> 和 <code>train2.py</code>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            
            <!-- 4个可视化图表 -->
            <div class="chart-container">
                <div id="pieChart" class="chart-box"></div>
                <div id="clusterPieChart" class="chart-box"></div>
                <div id="scatterChart" class="chart-box"></div>
                <div id="budgetBarChart" class="chart-box"></div>
            </div>
            <script>
                // 1. 题材分布饼图
                var pieChart = echarts.init(document.getElementById('pieChart'));
                var pieOption = {
                    title: { text: '热门电影题材分布', left: 'center' },
                    tooltip: { trigger: 'item' },
                    series: [{
                        type: 'pie', radius: '50%',
                        data: {{ pie_data | tojson }},
                        emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
                    }]
                };
                pieChart.setOption(pieOption);

                // 2. 聚类分布饼图（新增）
                var clusterPieChart = echarts.init(document.getElementById('clusterPieChart'));
                var clusterPieOption = {
                    title: { text: 'K-Means 聚类分布', left: 'center' },
                    tooltip: { trigger: 'item', formatter: '{b}: {c}部 ({d}%)' },
                    series: [{
                        type: 'pie', radius: '50%',
                        data: {{ cluster_pie_data | tojson }},
                        emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
                    }]
                };
                clusterPieChart.setOption(clusterPieOption);

                // 3. 预算vs票房散点图
                var scatterChart = echarts.init(document.getElementById('scatterChart'));
                var scatterOption = {
                    title: { text: '预算与票房关系分析', left: 'center' },
                    tooltip: { formatter: function(p){ return '预算: $' + p.data[0].toLocaleString() + '<br>票房: $' + p.data[1].toLocaleString(); } },
                    xAxis: { name: '预算($)', type: 'value', splitLine:{show:false} },
                    yAxis: { name: '票房($)', type: 'value', splitLine:{show:false} },
                    series: [{
                        symbolSize: 5,
                        data: {{ scatter_data | tojson }},
                        type: 'scatter',
                        itemStyle: { color: '#007bff' }
                    }]
                };
                scatterChart.setOption(scatterOption);

                // 4. 聚类平均预算柱状图（新增）
                var budgetBarChart = echarts.init(document.getElementById('budgetBarChart'));
                var budgetBarData = {{ budget_bar_data | tojson | safe }};
                var budgetBarOption = {
                    title: { text: '各聚类平均预算对比', left: 'center' },
                    tooltip: { 
                        trigger: 'axis',
                        formatter: function(params) {
                            return params[0].name + '<br>平均预算: $' + params[0].value.toLocaleString();
                        }
                    },
                    xAxis: { 
                        type: 'category', 
                        data: budgetBarData.map(d => d.name),
                        axisLabel: { interval: 0, rotate: 30, fontSize: 10 }
                    },
                    yAxis: { 
                        name: '平均预算($)', 
                        type: 'value',
                        axisLabel: {
                            formatter: function(value) {
                                return '$' + (value / 1000000).toFixed(0) + 'M';
                            }
                        }
                    },
                    series: [{
                        type: 'bar',
                        data: budgetBarData.map(d => d.value),
                        itemStyle: { 
                            color: function(params) {
                                var colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#27ae60', '#20c997', '#6f42c1', '#6c757d'];
                                return colors[params.dataIndex % colors.length];
                            }
                        }
                    }]
                };
                budgetBarChart.setOption(budgetBarOption);
            </script>
        </div>
        {% endif %}

        {% if tab == 'recsys' %}
        <div class="card card-blue">
            <h3>Model A: ALS 协同过滤推荐</h3>
            <p style="color:#666; font-size:14px; margin-bottom:15px;">
               <strong>实时推荐引擎</strong>: 基于 ALS 协同过滤算法。<br>
               <strong>建议输入 5-10 部喜欢的电影</strong>，推荐效果会更准确。<br>
               <span style="color:#e67e22;">协同过滤基于用户行为相似度，不是题材相似度。</span>
            </p>
            <form method="post" onsubmit="load()">
                <input type="hidden" name="tab" value="recsys">
                <div class="form-group"><label>新用户 ID (任意):</label><input type="number" name="new_uid" value="{{ new_uid }}" required></div>
                <div class="form-group">
                    <label>喜欢的电影 ID (多个用逗号隔开):</label>
                    <input type="text" name="liked_mids" value="{{ liked_mids }}" placeholder="例如: 12, 19995" required>
                    <span class="hint">推荐测试 ID: 12 (海底总动员), 19995 (阿凡达), 597 (泰坦尼克号), 862 (玩具总动员), 278 (肖申克的救赎), 680 (低俗小说), 13 (阿甘正传), 122 (指环王)</span>
                </div>
                <button type="submit">启动实时推荐</button>
                <div id="loader" class="loader">正在计算向量相似度...</div>
            </form>
            {% if als_recs %}
            <div class="result-box">
                <p><strong>计算耗时:</strong> {{ duration }}s</p>
                <table class="db-table">
                    <tr><th>ID</th><th>电影名</th><th>题材</th><th>推荐指数</th></tr>
                    {% for r in als_recs %}
                    <tr>
                        <td>{{ r.id }}</td>
                        <td>{{ r.title }}</td>
                        <td>{{ r.genres }}</td>
                        <td><span class="badge badge-blue">{{ r.score }}</span></td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
        </div>
        {% endif %}

        {% if tab == 'predict' %}
        <div class="card card-yellow">
            <h3>Model B: 票房预测</h3>
            <form method="post">
                <input type="hidden" name="tab" value="predict">
                <div class="form-group"><label>预算 ($):</label><input type="number" name="budget" required></div>
                <div class="form-group"><label>热度 (0.1-500):</label><input type="number" step="0.1" name="pop" required></div>
                <div class="form-group"><label>时长 (min):</label><input type="number" name="runtime" required></div>
                <button type="submit">预测票房</button>
            </form>
            {% if revenue %}
            <div class="result-box" style="border-left: 5px solid #ffc107;"><h3>预测: ${{ revenue }}</h3></div>
            {% endif %}
        </div>
        {% endif %}

        {% if tab == 'classify' %}
        <div class="card card-red">
            <h3>Model C: 投资成败预测</h3>
            <form method="post">
                <input type="hidden" name="tab" value="classify">
                <div class="form-group"><label>预算 ($):</label><input type="number" name="c_budget" required></div>
                <div class="form-group"><label>热度 (0.1-500):</label><input type="number" step="0.1" name="c_pop" required></div>
                <div class="form-group"><label>时长 (min):</label><input type="number" name="c_runtime" required></div>
                <button type="submit">运行分类器</button>
            </form>
            {% if cls_res %}
            <div class="result-box" style="border-left: 5px solid #dc3545;"><h3>结果: {{ cls_res }}</h3></div>
            {% endif %}
        </div>
        {% endif %}

        {% if tab == 'cluster' %}
        <div class="card card-green">
            <h3>Model D: K-Means 电影聚类分析</h3>
            <p style="color:#666; margin-bottom:20px;">基于电影题材、预算、热度等特征的智能分类</p>
            <form method="post">
                <input type="hidden" name="tab" value="cluster">
                <div class="form-group">
                    <label>电影 ID:</label>
                    <input type="text" name="cluster_id" value="{{ cluster_id }}"  required>
                    <span class="hint">测试 ID: 19995 (阿凡达), 597 (泰坦尼克), 12 (海底总动员)</span>
                </div>
                <button type="submit">查询聚类结果</button>
            </form>
            {% if cluster_info %}
            <div class="result-box">
                <h4>电影信息</h4>
                <p><strong>片名:</strong> {{ cluster_info.title }}</p>
                <p><strong>题材:</strong> {{ cluster_info.genres }}</p>
                <p><strong>预算:</strong> {{ cluster_info.budget }} | <strong>票房:</strong> {{ cluster_info.revenue }}</p>
                <hr>
                <h4>聚类分析结果</h4>
                <p><strong>所属类别:</strong> <span style="background:#28a745; color:white; padding:4px 12px; border-radius:3px; font-size:16px;">{{ cluster_info.cluster_name }}</span></p>
                <p><strong>聚类 ID:</strong> {{ cluster_info.cluster_id }}</p>
                <p style="color:#666; font-size:14px; margin-top:10px;">
                    该电影通过 K-Means 算法被归类为"{{ cluster_info.cluster_name }}"，
                    与同类电影在题材、预算、市场热度等方面具有相似特征。
                </p>
            </div>
            {% endif %}
        </div>
        {% endif %}
        {% if tab == 'database' %}
        <div class="card card-purple">
            <h3>HBase 电影数据库</h3>
            
            <form method="post" class="search-bar">
                <input type="hidden" name="tab" value="database">
                <div style="flex-grow:1;">
                    <label style="font-size:12px;">按 ID 精确搜索:</label>
                    <input type="text" name="search_id" value="{{ search_id }}" placeholder="输入电影 ID (如: 19995)" style="width:100%;">
                </div>
                <button type="submit" style="margin:0; height:40px;">搜索</button>
                {% if search_id %}
                <a href="/?tab=database" style="padding:10px; color:#666; text-decoration:underline;">清除搜索</a>
                {% endif %}
            </form>
            
            <div style="margin-bottom:20px; margin-top:15px;">
                <label style="font-size:12px; font-weight:bold; margin-bottom:5px; display:block;">按题材筛选:</label>
                <select id="genreFilter" onchange="filterByGenre()" style="padding:8px; border:1px solid #ccc; border-radius:4px; width:200px; font-size:14px;">
                    <option value="all" {% if genre_filter == 'all' %}selected{% endif %}>全部题材</option>
                    <option value="Action" {% if genre_filter == 'Action' %}selected{% endif %}>动作</option>
                    <option value="Adventure" {% if genre_filter == 'Adventure' %}selected{% endif %}>冒险</option>
                    <option value="Animation" {% if genre_filter == 'Animation' %}selected{% endif %}>动画</option>
                    <option value="Comedy" {% if genre_filter == 'Comedy' %}selected{% endif %}>喜剧</option>
                    <option value="Crime" {% if genre_filter == 'Crime' %}selected{% endif %}>犯罪</option>
                    <option value="Documentary" {% if genre_filter == 'Documentary' %}selected{% endif %}>纪录片</option>
                    <option value="Drama" {% if genre_filter == 'Drama' %}selected{% endif %}>剧情</option>
                    <option value="Family" {% if genre_filter == 'Family' %}selected{% endif %}>家庭</option>
                    <option value="Fantasy" {% if genre_filter == 'Fantasy' %}selected{% endif %}>奇幻</option>
                    <option value="History" {% if genre_filter == 'History' %}selected{% endif %}>历史</option>
                    <option value="Horror" {% if genre_filter == 'Horror' %}selected{% endif %}>恐怖</option>
                    <option value="Music" {% if genre_filter == 'Music' %}selected{% endif %}>音乐</option>
                    <option value="Mystery" {% if genre_filter == 'Mystery' %}selected{% endif %}>悬疑</option>
                    <option value="Romance" {% if genre_filter == 'Romance' %}selected{% endif %}>爱情</option>
                    <option value="Science Fiction" {% if genre_filter == 'Science Fiction' %}selected{% endif %}>科幻</option>
                    <option value="Thriller" {% if genre_filter == 'Thriller' %}selected{% endif %}>惊悚</option>
                    <option value="War" {% if genre_filter == 'War' %}selected{% endif %}>战争</option>
                    <option value="Western" {% if genre_filter == 'Western' %}selected{% endif %}>西部</option>
                </select>
                <script>
                    function filterByGenre() {
                        var genre = document.getElementById('genreFilter').value;
                        window.location.href = '/?tab=database&genre=' + encodeURIComponent(genre);
                    }
                </script>
            </div>
            
            <p style="color:#666; font-size:12px;">
                {% if search_id %}
                    搜索结果: {{ all_movies|length }} 条
                {% elif genre_filter != 'all' %}
                    筛选结果: 共 {{ total_pages }} 页
                {% else %}
                    共 {{ total_pages }} 页。
                {% endif %}
            </p>
            
            <table class="db-table">
                <thead>
                    <tr>
                        <th width="8%">ID</th>
                        <th width="30%">电影名称</th>
                        <th width="25%">题材</th>
                        <th width="15%">预算</th>
                        <th width="22%">聚类</th>
                    </tr>
                </thead>
                <tbody>
                    {% for m in all_movies %}
                    <tr>
                        <td><strong>{{ m.id }}</strong></td>
                        <td>{{ m.title }}</td>
                        <td><span style="color:#555;">{{ m.genres }}</span></td>
                        <td>{{ m.budget }}</td>
                        <td><span class="badge {{ m.badge_class }}">{{ m.analysis }}</span></td>
                    </tr>
                    {% else %}
                    <tr><td colspan="5" style="text-align:center; padding:20px; color:#999;">暂无数据或 ID 不存在</td></tr>
                    {% endfor %}
                </tbody>
            </table>
            
            {% if not search_id %}
            <div class="pagination-container">
                <div class="pagination">
                    <a href="/?tab=database&page=1&genre={{ genre_filter }}" class="page-link">&laquo; 首页</a>
                    {% for p in range(page_start, page_end + 1) %}
                        <a href="/?tab=database&page={{ p }}&genre={{ genre_filter }}" class="page-link {% if p == current_page %}active{% endif %}">{{ p }}</a>
                    {% endfor %}
                    <a href="/?tab=database&page={{ total_pages }}&genre={{ genre_filter }}" class="page-link">尾页 &raquo;</a>
                    <span style="font-size:12px; color:#888; margin-left:10px;">Page {{ current_page }} / {{ total_pages }}</span>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}

    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    tab = request.args.get('tab', 'recsys')
    if request.form.get('tab'): tab = request.form.get('tab')

    new_uid = request.form.get('new_uid', '')
    liked_mids = request.form.get('liked_mids', '')
    search_id = request.form.get('search_id', '') 
    cluster_id = request.form.get('cluster_id', '')
    
    als_recs = []; duration = 0; revenue = None; cls_res = None; cluster_info = None
    all_movies = []; current_page = 1; total_pages = 1; page_start = 1; page_end = 1
    pie_data = []; scatter_data = []; model_metrics = []
    cluster_pie_data = []; budget_bar_data = []
    genre_filter = request.args.get('genre', 'all')

    if request.method == 'POST':
        if tab == 'recsys' and new_uid and liked_mids:
            als_recs, duration = run_realtime_als(new_uid, liked_mids)
            duration = f"{duration:.2f}"
        elif tab == 'predict':
            b = request.form.get('budget'); p = request.form.get('pop'); r = request.form.get('runtime')
            revenue = predict_revenue_real(b, p, r)
        elif tab == 'classify':
            b = request.form.get('c_budget'); p = request.form.get('c_pop'); r = request.form.get('c_runtime')
            cls_res = predict_classify_real(b, p, r)
        elif tab == 'cluster' and cluster_id:
            cluster_info = get_hbase_detail(cluster_id)

    if tab == 'charts':
        pie_data, scatter_data, cluster_pie_data, budget_bar_data = get_chart_data()
        model_metrics = get_model_metrics()

    if tab == 'database':
        if search_id:
            all_movies = search_movie_by_id(search_id)
            total_pages = 1
        else:
            try: current_page = int(request.args.get('page', 1))
            except: current_page = 1
            filter_param = None if genre_filter == 'all' else genre_filter
            all_movies, total_pages = get_movies_by_page(current_page, 20, filter_param)
            page_start = max(1, current_page - 2)
            page_end = min(total_pages, current_page + 2)

    return render_template_string(html_template, tab=tab, 
                                  new_uid=new_uid, liked_mids=liked_mids, als_recs=als_recs, duration=duration,
                                  search_id=search_id, cluster_id=cluster_id, cluster_info=cluster_info,
                                  revenue=revenue, cls_res=cls_res,
                                  all_movies=all_movies, current_page=current_page, total_pages=total_pages,
                                  page_start=page_start, page_end=page_end,
                                  pie_data=pie_data, scatter_data=scatter_data, genre_filter=genre_filter,
                                  cluster_pie_data=cluster_pie_data, budget_bar_data=budget_bar_data,
                                  model_metrics=model_metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)
