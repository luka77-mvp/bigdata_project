# 电影数据分析平台

基于 Hadoop 生态的电影大数据分析与机器学习平台，实现电影推荐、票房预测、投资成败分类和电影聚类等功能。

## 技术栈

- **计算框架**: Apache Spark (MLlib)
- **数据仓库**: Apache Hive
- **NoSQL 数据库**: Apache HBase
- **分布式存储**: HDFS
- **Web 框架**: Flask
- **可视化**: ECharts

## 项目结构

```
├── codes/
│   ├── app.py          # Flask Web 应用（模型服务 + 可视化）
│   ├── etl.py          # ETL 流程 + ALS推荐模型 + K-Means聚类
│   └── train.py        # 票房预测模型 + 成败分类模型训练
├── data/
│   ├── preprocess.py   # 数据预处理 + HDFS上传 + Hive导入
│   ├── tmdb_5000_movies.csv    # TMDB 电影原始数据
│   └── tmdb_5000_credits.csv   # TMDB 演职员原始数据
```

## 功能模块

| 模型 | 算法 | 功能描述 |
|------|------|----------|
| Model A | ALS 协同过滤 | 基于用户行为的电影推荐 |
| Model B | 随机森林回归 | 电影票房预测 |
| Model C | 随机森林分类 | 电影投资成败预测 |
| Model D | K-Means 聚类 | 电影类型自动分类 |

## 运行步骤

### 1. 数据预处理

```bash
cd data
python preprocess.py
```

该脚本会自动完成：
- 清洗 TMDB 原始数据
- 生成模拟用户评分数据
- 上传数据到 HDFS
- 创建 Hive 表并导入数据

### 2. 模型训练

```bash
# ETL + ALS推荐 + K-Means聚类
spark-submit --master yarn codes/etl.py

# 票房预测 + 成败分类模型
spark-submit --master yarn codes/train.py
```

### 3. 启动 Web 服务

```bash
python codes/app.py
```

访问 `http://localhost:5000` 即可使用平台。

## 数据流

```
TMDB CSV → preprocess.py → HDFS → Hive
                                    ↓
                              etl.py / train.py
                                    ↓
                            HBase + HDFS (模型)
                                    ↓
                                 app.py
                                    ↓
                               Web 界面
```

## 环境要求

- Hadoop 集群 (HDFS + YARN)
- Apache Spark 2.x+
- Apache Hive
- Apache HBase
- Python 3.x
- 依赖包: pyspark, happybase, flask, pandas, numpy
