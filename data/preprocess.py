import pandas as pd
import json
import numpy as np
import os
import subprocess
import sys

print("=" * 80)
print("🚀 一键式数据处理 + HDFS上传 + Hive导入")
print("   使用 MovieLens 真实评分数据")
print("=" * 80)

# ==========================================
# 1. 读取原始数据
# ==========================================
print("\n>>> [步骤 1] 读取原始数据...")

required_files = ['tmdb_5000_movies.csv', 'tmdb_5000_credits.csv', 'ratings.csv', 'links.csv']
for f in required_files:
    if not os.path.exists(f):
        print(f"❌ 错误：没找到 {f}，请检查路径！")
        sys.exit(1)

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
ratings_raw = pd.read_csv('ratings.csv')
links = pd.read_csv('links.csv')

print(f"   ✅ TMDB 电影数据: {len(movies)} 部")
print(f"   ✅ MovieLens 评分数据: {len(ratings_raw)} 条 ({ratings_raw['userId'].nunique()} 用户)")
print(f"   ✅ ID 映射表: {len(links)} 条")

# ==========================================
# 2. 合并电影数据
# ==========================================
print("\n>>> [步骤 2] 合并电影元数据...")

credits_renamed = credits[['movie_id', 'cast', 'crew']]
movies = movies.merge(credits_renamed, left_on='id', right_on='movie_id', how='left')
print(f"   ✅ 合并后数据量: {len(movies)} 部")

# ==========================================
# 3. 处理 MovieLens 评分数据（核心步骤）
# ==========================================
print("\n>>> [步骤 3] 映射 MovieLens 评分到 TMDB ID...")

# 获取 TMDB 电影 ID 集合
tmdb_movie_ids = set(movies['id'].values)
print(f"   TMDB 电影 ID 数量: {len(tmdb_movie_ids)}")

# 清理 links 数据（去除 NaN）
links_clean = links.dropna(subset=['tmdbId'])
links_clean['tmdbId'] = links_clean['tmdbId'].astype(int)
print(f"   有效映射数量: {len(links_clean)}")

# 创建 MovieLens ID -> TMDB ID 的映射字典
ml_to_tmdb = dict(zip(links_clean['movieId'], links_clean['tmdbId']))

# 映射评分数据
ratings_raw['tmdbId'] = ratings_raw['movieId'].map(ml_to_tmdb)

# 过滤：只保留能映射且在 TMDB 数据集中存在的电影
ratings_mapped = ratings_raw.dropna(subset=['tmdbId'])
ratings_mapped['tmdbId'] = ratings_mapped['tmdbId'].astype(int)
ratings_mapped = ratings_mapped[ratings_mapped['tmdbId'].isin(tmdb_movie_ids)]

print(f"   ✅ 映射后评分数量: {len(ratings_mapped)} 条")
print(f"   ✅ 覆盖用户数: {ratings_mapped['userId'].nunique()}")
print(f"   ✅ 覆盖电影数: {ratings_mapped['tmdbId'].nunique()}")

# 生成最终评分文件（使用 TMDB ID）
ratings_final = ratings_mapped[['userId', 'tmdbId', 'rating']].copy()
ratings_final.columns = ['userId', 'movieId', 'rating']

# ==========================================
# 4. 清洗电影 JSON 数据
# ==========================================
print("\n>>> [步骤 4] 解析 JSON 格式的复杂列...")

def get_names(obj):
    try:
        if pd.isna(obj): return ""
        L = json.loads(obj)
        return "|".join([i['name'] for i in L[:5]])
    except: return ""

def get_director(obj):
    try:
        if pd.isna(obj): return ""
        L = json.loads(obj)
        for i in L:
            if i.get('job') == 'Director':
                return i.get('name', '')
        return ""
    except: return ""

movies['genres_str'] = movies['genres'].apply(get_names)
movies['keywords_str'] = movies['keywords'].apply(get_names)
movies['cast_str'] = movies['cast'].apply(get_names)
movies['director'] = movies['crew'].apply(get_director)
print("   ✅ JSON 解析完成")

# ==========================================
# 5. 数据清洗
# ==========================================
print("\n>>> [步骤 5] 数据清洗...")

movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.date

mean_revenue = movies[movies['revenue'] > 0]['revenue'].mean()
mean_budget = movies[movies['budget'] > 0]['budget'].mean()

movies['revenue'] = movies['revenue'].replace(0, mean_revenue)
movies['budget'] = movies['budget'].replace(0, mean_budget)
movies['popularity'] = movies['popularity'].fillna(0)
movies['vote_average'] = movies['vote_average'].fillna(0)
movies['vote_count'] = movies['vote_count'].fillna(0)
movies['runtime'] = movies['runtime'].fillna(0)

def clean_text(text):
    if isinstance(text, str):
        return text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
    return text

movies['title'] = movies['title'].apply(clean_text)
movies['genres_str'] = movies['genres_str'].apply(clean_text).replace('', '-')
movies['keywords_str'] = movies['keywords_str'].apply(clean_text).replace('', '-')
movies['director'] = movies['director'].apply(clean_text).replace('', 'Unknown')

print("   ✅ 数据清洗完成")

# ==========================================
# 6. 保存清洗后的数据
# ==========================================
print("\n>>> [步骤 6] 保存处理后的数据...")

clean_cols = ['id', 'title', 'budget', 'revenue', 'popularity', 
              'vote_average', 'vote_count', 'runtime', 
              'genres_str', 'keywords_str', 'director', 'release_date']

movies_clean = movies[clean_cols].copy()
movies_clean.columns = ['id', 'title', 'budget', 'revenue', 'popularity',
                        'vote_average', 'vote_count', 'runtime',
                        'genres', 'keywords', 'director', 'release_date']

# 保存电影数据
movies_clean.to_csv('clean_movies.txt', index=False, sep='\t', header=False)
print(f"   ✅ 电影数据: clean_movies.txt ({len(movies_clean)} 部)")

# 保存真实评分数据
ratings_final.to_csv('real_ratings.txt', index=False, sep=',', header=False)
print(f"   ✅ 真实评分: real_ratings.txt ({len(ratings_final)} 条)")

# ==========================================
# 7. 上传到 HDFS
# ==========================================
print("\n>>> [步骤 7] 上传文件到 HDFS...")

hdfs_dir = "/bigdata_project/data"

subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', hdfs_dir], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for f in ['clean_movies.txt', 'real_ratings.txt']:
    subprocess.run(['hdfs', 'dfs', '-rm', '-f', f'{hdfs_dir}/{f}'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = subprocess.run(['hdfs', 'dfs', '-put', f, hdfs_dir], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✅ {f} 上传成功")
    else:
        print(f"   ❌ {f} 上传失败: {result.stderr}")
        sys.exit(1)

# ==========================================
# 8. 导入到 Hive
# ==========================================
print("\n>>> [步骤 8] 导入数据到 Hive...")

hive_sql = """
CREATE DATABASE IF NOT EXISTS movie_db;

DROP TABLE IF EXISTS movie_db.movies;
DROP TABLE IF EXISTS movie_db.ratings;

CREATE TABLE movie_db.movies (
    id INT, title STRING, budget DOUBLE, revenue DOUBLE,
    popularity DOUBLE, vote_average DOUBLE, vote_count INT,
    runtime DOUBLE, genres STRING, keywords STRING,
    director STRING, release_date DATE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\\t' STORED AS TEXTFILE;

LOAD DATA INPATH '/bigdata_project/data/clean_movies.txt' INTO TABLE movie_db.movies;

CREATE TABLE movie_db.ratings (
    userId INT, movieId INT, rating DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE;

LOAD DATA INPATH '/bigdata_project/data/real_ratings.txt' INTO TABLE movie_db.ratings;

SHOW TABLES IN movie_db;
"""

with open('load_to_hive.sql', 'w') as f:
    f.write(hive_sql)

result = subprocess.run(['hive', '-f', 'load_to_hive.sql'], capture_output=True, text=True)
if result.returncode == 0:
    print("   ✅ Hive 表创建和数据加载成功")
else:
    print(f"   ❌ Hive 执行失败: {result.stderr}")
    sys.exit(1)

# ==========================================
# 9. 数据统计
# ==========================================
print("\n" + "=" * 80)
print("📊 数据统计")
print("=" * 80)

print(f"\n电影数据: {len(movies_clean)} 部")
print(f"真实评分数据:")
print(f"  • 用户数: {ratings_final['userId'].nunique()}")
print(f"  • 电影数: {ratings_final['movieId'].nunique()}")
print(f"  • 评分数: {len(ratings_final):,} 条")
print(f"  • 数据来源: MovieLens (真实用户评分)")

print(f"\nHive 表: movie_db.movies, movie_db.ratings")

print("\n" + "=" * 80)
print("✅ 全部完成！使用真实 MovieLens 评分数据")
print("=" * 80)

# 清理临时文件
try:
    os.remove('load_to_hive.sql')
except: pass
