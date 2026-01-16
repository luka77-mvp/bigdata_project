# 基于Spark的电影数据分析与推荐系统

## 项目简介

本项目是《分布式系统应用设计》课程的大作业，基于Hadoop+Spark分布式架构，实现电影数据分析与推荐系统。系统整合TMDB电影元数据和MovieLens真实用户评分数据，构建了四个核心机器学习模型。

## 团队成员

| 姓名 | 角色 | 主要职责 |
|------|------|---------|
| 李伯犀 | 组长 | ALS推荐算法、系统后端开发、项目管理 |
| 刘梓晗 | 成员 | 票房预测模型、数据预处理、Hive数据仓库 |
| 贺鹏宇 | 成员 | 投资分类模型、集群环境搭建 |
| 刘子正 | 成员 | K-Means聚类、前端可视化 |

## 技术栈

- **分布式存储**: Hadoop HDFS 3.3.6
- **分布式计算**: Apache Spark 3.5.2
- **数据仓库**: Apache Hive 3.1.2
- **NoSQL数据库**: Apache HBase 2.4.17
- **机器学习**: Spark MLlib
- **Web框架**: Flask 3.1.2
- **可视化**: ECharts

## 系统功能

### Model A: ALS协同过滤推荐
- 基于用户喜好的电影实时推荐
- RMSE: 0.8581

### Model B: 票房预测
- 随机森林回归预测电影票房
- R²: 0.3126

### Model C: 投资成败分类
- 随机森林分类判断投资盈亏
- 准确率: 80.75%

### Model D: K-Means聚类
- 电影智能分类（超级大片、热门商业片、口碑佳作等）
- 轮廓系数: 0.4506


## 项目结构

```
bigdata_project/
├── codes/                        # 核心代码
│   ├── app.py                    # Flask Web应用主程序
│   ├── etl.py                    # ETL处理 + ALS + K-Means训练
│   └── train.py                  # 随机森林模型训练
├── data/                         # 数据文件
│   ├── tmdb_5000_movies.csv      # TMDB电影元数据
│   ├── tmdb_5000_credits.csv     # 演员导演数据
│   ├── ratings.csv               # MovieLens评分数据
│   ├── links.csv                 # ID映射表
│   ├── preprocess.py             # 数据预处理脚本
│   ├── clean_movies.txt          # 清洗后的电影数据（也会上传到hdfs）
│   └── real_ratings.txt          # 处理后的评分数据（也会上传到hdfs）
└── README.md
```

## 集群配置

| 节点 | 配置 | 角色 |
|------|------|------|
| Node01 | 4G/2core | NameNode, ResourceManager, Spark Master, HMaster |
| Node02 | 2G/1core | DataNode, NodeManager, Spark Worker, HRegionServer |
| Node03 | 2G/1core | DataNode, NodeManager, Spark Worker, HRegionServer |


## 数据集

- **TMDB 5000 Movie Dataset**: 4,803部电影元数据
- **MovieLens Dataset**: 100,004条真实用户评分

## 模型性能

| 模型 | 指标 | 数值 | 训练样本 | 耗时 |
|------|------|------|---------|------|
| ALS推荐 | RMSE | 0.8581 | 70,194 | 48s |
| 票房预测 | R² | 0.3126 | 4,550 | 51s |
| 投资分类 | Accuracy | 80.75% | 4,740 | 28s |
| 电影聚类 | Silhouette | 0.4506 | 4,802 | 6s |

## 界面预览

系统提供Web可视化界面，包含：
- 电影推荐功能
- 票房预测功能
- 投资分类功能
- 聚类查询功能
- 数据库浏览
- ECharts可视化图表

## License

本项目仅用于课程学习，数据集来源于Kaggle和GroupLens。
