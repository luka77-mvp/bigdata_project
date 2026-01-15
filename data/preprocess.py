import pandas as pd
import json
import numpy as np
import os
import subprocess
import sys

print("=" * 80)
print("🚀 一键式数据处理 + HDFS上传 + Hive导入")
print("=" * 80)

# ==========================================
# 1. 读取原始数据
# ==========================================
print("\n>>> [步骤 1] 读取原始数据...")

if not os.path.exists('tmdb_5000_movies.csv'):
    print("❌ 错误：没找到 tmdb_5000_movies.csv，请检查路径！")
    sys.exit(1)

if not os.path.exists('tmdb_5000_credits.csv'):
    print("❌ 错误：没找到 tmdb_5000_credits.csv，请检查路径！")
    sys.exit(1)

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print(f"   ✅ 读取电影数据: {len(movies)} 部")
print(f"   ✅ 读取演职员数据: {len(credits)} 条")

# ==========================================
# 2. 合并数据（避免列名冲突）
# ==========================================
print("\n>>> [步骤 2] 合并数据...")

credits_renamed = credits[['movie_id', 'cast', 'crew']]
movies = movies.merge(credits_renamed, left_on='id', right_on='movie_id', how='left')

print(f"   ✅ 合并后数据量: {len(movies)} 部")

# ==========================================
# 3. 清洗 JSON 数据
# ==========================================
print("\n>>> [步骤 3] 解析 JSON 格式的复杂列...")

def get_names(obj):
    """提取 JSON 数组中的 name 字段，用竖线分隔"""
    try:
        if pd.isna(obj):
            return ""
        L = json.loads(obj)
        return "|".join([i['name'] for i in L[:5]])
    except:
        return ""

def get_director(obj):
    """从 crew 中提取导演名字"""
    try:
        if pd.isna(obj):
            return ""
        L = json.loads(obj)
        for i in L:
            if i.get('job') == 'Director':
                return i.get('name', '')
        return ""
    except:
        return ""

print("   正在解析 genres...")
movies['genres_str'] = movies['genres'].apply(get_names)

print("   正在解析 keywords...")
movies['keywords_str'] = movies['keywords'].apply(get_names)

print("   正在解析 cast...")
movies['cast_str'] = movies['cast'].apply(get_names)

print("   正在解析 director...")
movies['director'] = movies['crew'].apply(get_director)

print("   ✅ JSON 解析完成")

# ==========================================
# 4. 数据清洗
# ==========================================
print("\n>>> [步骤 4] 数据清洗...")

# 处理日期
print("   处理日期格式...")
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.date

# 填补空值
print("   填补缺失值...")
mean_revenue = movies[movies['revenue'] > 0]['revenue'].mean()
mean_budget = movies[movies['budget'] > 0]['budget'].mean()

movies['revenue'] = movies['revenue'].replace(0, mean_revenue)
movies['budget'] = movies['budget'].replace(0, mean_budget)

movies['popularity'] = movies['popularity'].fillna(0)
movies['vote_average'] = movies['vote_average'].fillna(0)
movies['vote_count'] = movies['vote_count'].fillna(0)
movies['runtime'] = movies['runtime'].fillna(0)

print(f"   ✅ 均值填补: revenue={mean_revenue:,.0f}, budget={mean_budget:,.0f}")

# 清理文本
print("   清理文本特殊字符...")
def clean_text(text):
    if isinstance(text, str):
        return text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
    return text

movies['title'] = movies['title'].apply(clean_text)
movies['genres_str'] = movies['genres_str'].apply(clean_text)
movies['keywords_str'] = movies['keywords_str'].apply(clean_text)
movies['director'] = movies['director'].apply(clean_text)

# 处理空字符串
movies['genres_str'] = movies['genres_str'].replace('', '-')
movies['keywords_str'] = movies['keywords_str'].replace('', '-')
movies['director'] = movies['director'].replace('', 'Unknown')

print("   ✅ 文本清理完成")

# ==========================================
# 5. 保存清洗后的电影数据
# ==========================================
print("\n>>> [步骤 5] 保存清洗后的电影数据...")

clean_cols = [
    'id', 'title', 'budget', 'revenue', 'popularity', 
    'vote_average', 'vote_count', 'runtime', 
    'genres_str', 'keywords_str', 'director', 'release_date'
]

movies_clean = movies[clean_cols].copy()
movies_clean.columns = [
    'id', 'title', 'budget', 'revenue', 'popularity',
    'vote_average', 'vote_count', 'runtime',
    'genres', 'keywords', 'director', 'release_date'
]

output_file = 'clean_movies.txt'
movies_clean.to_csv(output_file, index=False, sep='\t', header=False)

print(f"   ✅ 已保存: {output_file}")
print(f"   ✅ 数据量: {len(movies_clean)} 部电影")

# ==========================================
# 6. 生成虚拟评分数据
# ==========================================
print("\n>>> [步骤 6] 生成虚拟评分数据...")

n_users = 1000
n_top_movies = 2000

movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
top_movies = movies.nlargest(n_top_movies, 'popularity')['id'].values

print(f"   选择热度前 {n_top_movies} 部电影")
print(f"   生成 {n_users} 个虚拟用户的评分...")

ratings_list = []

for user_id in range(1, n_users + 1):
    num_seen = np.random.randint(20, 50)
    movies_seen = np.random.choice(top_movies, num_seen, replace=False)
    
    for movie_id in movies_seen:
        rating = np.random.choice([3, 4, 5, 1, 2], p=[0.3, 0.4, 0.2, 0.05, 0.05])
        ratings_list.append([user_id, int(movie_id), float(rating)])
    
    if user_id % 200 == 0:
        print(f"   进度: {user_id}/{n_users} 用户")

ratings_df = pd.DataFrame(ratings_list, columns=['userId', 'movieId', 'rating'])

ratings_file = 'fake_ratings.txt'
ratings_df.to_csv(ratings_file, index=False, sep=',', header=False)

print(f"   ✅ 已保存: {ratings_file}")
print(f"   ✅ 评分数量: {len(ratings_df):,} 条")

# ==========================================
# 7. 上传到 HDFS
# ==========================================
print("\n>>> [步骤 7] 上传文件到 HDFS...")

hdfs_dir = "/bigdata_project/data"

# 创建 HDFS 目录
print(f"   创建 HDFS 目录: {hdfs_dir}")
subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', hdfs_dir], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 删除旧文件（如果存在）
print("   删除旧文件（如果存在）...")
subprocess.run(['hdfs', 'dfs', '-rm', '-f', f'{hdfs_dir}/clean_movies.txt'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['hdfs', 'dfs', '-rm', '-f', f'{hdfs_dir}/fake_ratings.txt'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 上传新文件
print(f"   上传 {output_file} 到 HDFS...")
result = subprocess.run(['hdfs', 'dfs', '-put', output_file, hdfs_dir], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"   ✅ {output_file} 上传成功")
else:
    print(f"   ❌ {output_file} 上传失败: {result.stderr}")
    sys.exit(1)

print(f"   上传 {ratings_file} 到 HDFS...")
result = subprocess.run(['hdfs', 'dfs', '-put', ratings_file, hdfs_dir], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"   ✅ {ratings_file} 上传成功")
else:
    print(f"   ❌ {ratings_file} 上传失败: {result.stderr}")
    sys.exit(1)

# ==========================================
# 8. 导入到 Hive
# ==========================================
print("\n>>> [步骤 8] 导入数据到 Hive...")

# 创建 Hive SQL 脚本
hive_sql = """
-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS movie_db;

-- 删除旧表（如果存在）
DROP TABLE IF EXISTS movie_db.movies;
DROP TABLE IF EXISTS movie_db.ratings;

-- 创建电影表
CREATE TABLE movie_db.movies (
    id INT,
    title STRING,
    budget DOUBLE,
    revenue DOUBLE,
    popularity DOUBLE,
    vote_average DOUBLE,
    vote_count INT,
    runtime DOUBLE,
    genres STRING,
    keywords STRING,
    director STRING,
    release_date DATE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\\t'
STORED AS TEXTFILE;

-- 加载电影数据
LOAD DATA INPATH '/bigdata_project/data/clean_movies.txt' 
INTO TABLE movie_db.movies;

-- 创建评分表
CREATE TABLE movie_db.ratings (
    userId INT,
    movieId INT,
    rating DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

-- 加载评分数据
LOAD DATA INPATH '/bigdata_project/data/fake_ratings.txt' 
INTO TABLE movie_db.ratings;

-- 显示表信息
SHOW TABLES IN movie_db;
"""

# 保存 SQL 脚本
sql_file = 'load_to_hive.sql'
with open(sql_file, 'w') as f:
    f.write(hive_sql)

print(f"   ✅ 已生成 Hive SQL 脚本: {sql_file}")

# 执行 Hive SQL
print("   正在执行 Hive SQL...")
print("   ⚠️  注意：会删除旧表并重新创建！")

result = subprocess.run(['hive', '-f', sql_file], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("   ✅ Hive 表创建和数据加载成功")
    # 显示部分输出
    if "OK" in result.stdout:
        print("   ✅ Hive 命令执行成功")
else:
    print(f"   ❌ Hive 执行失败")
    print(f"   错误信息: {result.stderr}")
    sys.exit(1)

# ==========================================
# 9. 验证数据
# ==========================================
print("\n>>> [步骤 9] 验证 Hive 表数据...")

verify_sql = """
SELECT COUNT(*) as movie_count FROM movie_db.movies;
SELECT COUNT(*) as rating_count FROM movie_db.ratings;
"""

verify_file = 'verify_hive.sql'
with open(verify_file, 'w') as f:
    f.write(verify_sql)

result = subprocess.run(['hive', '-f', verify_file], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("   ✅ 数据验证成功")
    print(f"   输出:\n{result.stdout}")
else:
    print(f"   ⚠️  验证失败，但数据可能已经加载")

# ==========================================
# 10. 数据统计
# ==========================================
print("\n" + "=" * 80)
print("📊 数据统计")
print("=" * 80)

print(f"\n电影数据:")
print(f"  • 总数: {len(movies_clean)} 部")
print(f"  • 预算范围: ${movies_clean['budget'].min():,.0f} - ${movies_clean['budget'].max():,.0f}")
print(f"  • 票房范围: ${movies_clean['revenue'].min():,.0f} - ${movies_clean['revenue'].max():,.0f}")

print(f"\n评分数据:")
print(f"  • 用户数: {ratings_df['userId'].nunique()}")
print(f"  • 电影数: {ratings_df['movieId'].nunique()}")
print(f"  • 评分数: {len(ratings_df):,} 条")

print(f"\nHDFS 路径:")
print(f"  • 电影数据: {hdfs_dir}/clean_movies.txt")
print(f"  • 评分数据: {hdfs_dir}/fake_ratings.txt")

print(f"\nHive 表:")
print(f"  • 数据库: movie_db")
print(f"  • 电影表: movie_db.movies")
print(f"  • 评分表: movie_db.ratings")

print("\n" + "=" * 80)
print("✅ 全部完成！数据已成功导入 Hive")
print("=" * 80)

print("\n下一步:")
print("  1. 运行: spark-submit --master yarn etl2.py")
print("  2. 运行: spark-submit --master yarn train2.py")
print("  3. 运行: python app2.py")

# 清理临时文件
print("\n>>> 清理临时文件...")
try:
    os.remove(sql_file)
    os.remove(verify_file)
    print("   ✅ 临时文件已清理")
except:
    pass

