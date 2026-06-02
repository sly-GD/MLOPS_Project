# 课程实践项目 — 基于 MiniMind 的中文文本分类多系统对比 & MLOps 全流程实践

本项目基于 [MiniMind](https://github.com/jingyaogong/minimind) 小型语言模型（~64M 参数）的 SFT 对话数据与模型架构，完成两个课程实践项目：（1）中文文本主题分类任务在 scikit-learn / Apache Spark MLlib / PyTorch 三种计算系统上的算法对比；（2）基于 MLflow 的 MiniMind SFT 训练全生命周期 MLOps 实践。

---

## 目录

- [项目 1：中文文本主题分类多系统算法对比](#项目-1中文文本主题分类多系统算法对比)
  - [任务描述](#任务描述)
  - [数据集](#数据集)
  - [三种框架实现方案](#三种框架实现方案)
  - [对比维度与实验设计](#对比维度与实验设计)
  - [技术栈](#技术栈)
  - [分工](#分工)
- [项目 2：基于 MLflow 的 MiniMind SFT 训练 MLOps 实践](#项目-2基于-mlflow-的-minimind-sft-训练-mlops-实践)
  - [任务描述](#任务描述-1)
  - [数据集](#数据集-1)
  - [MLOps 实践流程](#mlops-实践流程)
  - [技术栈](#技术栈-1)
  - [分工](#分工-1)

---

## 项目 1：中文文本主题分类多系统算法对比

### 任务描述

以中文文本**主题分类**为统一任务，基于 MiniMind 的 SFT 对话数据构建标注样本，在三种不同计算系统（单机传统 ML、分布式计算、深度学习）上实现**逻辑回归**分类算法，从分类精度、训练时间、推理速度、内存/显存消耗、模型大小等维度进行系统级对比，分析不同计算范式的优劣势与适用场景。

**类别体系**（共 6 类，基于用户首轮 query 的关键词规则自动标注）：

| 类别 | 说明 | 关键词示例 |
|------|------|-----------|
| `identity` | 询问模型身份/来源 | 真实来源、真实身份、开发背景、哪家公司 |
| `tech_discussion` | 技术/算法/原理讨论 | 如何平衡、怎么处理、原理、算法、为什么 |
| `recommendation` | 推荐类（电影/书籍/音乐） | 推荐、有没有看过、好电影、好书 |
| `casual_chat` | 闲聊/问候/日常 | 今天、过得怎么样、在忙什么、聊聊 |
| `domain_knowledge` | 领域知识问答（医疗/科学/AI伦理） | 医疗、伦理、人工智能、量子、科学 |
| `capability` | 询问模型能力/边界 | 能不能、是否支持、可以做什么、多语言 |

### 数据集

**来源**：`mini_deepseek_mind/dataset/sft_t2t_mini.jsonl`（约 90 万条多轮对话）

**构建流程**：
1. 提取每条对话的**首条 user query**
2. 使用关键词规则自动匹配 6 个主题类别
3. 未匹配的归入 `other`，最终按类别分层采样 10 万条（保证各类别均衡）
4. 统一按 **8:1:1** 划分为训练集 / 验证集 / 测试集

**格式**（CSV）：

```
text,label
"你的真实来源是什么？",identity
"如何处理数据中的缺失值？",tech_discussion
"你最近有没有看过什么有趣的电影？",recommendation
...
```

**自动标注脚本**位于 `project1/label_data.py`，基于 MiniMind 的 BPE tokenizer 分词后匹配关键词。

### 三种框架实现方案

#### 框架 1：scikit-learn（单机传统 ML）

```
project1/
├── sklearn/
│   ├── train_sklearn.py       # 训练 + 评估
│   ├── predict_sklearn.py     # 推理测试
│   └── requirements.txt
```

**实现路径**：
1. 加载 CSV 数据，划分训练/验证/测试集
2. 文本向量化：复用 MiniMind 的 BPE tokenizer 分词 → `TfidfVectorizer(max_features=5000)`
3. 分类器：`LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)` / 可选 `LinearSVC`、`RandomForestClassifier`
4. 评估：accuracy、precision、recall、F1（macro）、混淆矩阵

**资源记录**：`time` 计时、`psutil` 内存监控、`joblib.dump` 模型大小

#### 框架 2：Apache Spark MLlib（分布式计算）

```
project1/
├── spark/
│   ├── train_spark.py         # 训练 + 评估
│   ├── predict_spark.py       # 推理测试
│   └── requirements.txt
```

**实现路径**：
1. `spark.read.csv` 加载数据，`StringIndexer` 编码标签
2. 文本向量化：`Tokenizer` → `HashingTF(numFeatures=5000)` → `IDF`（分布式词频/逆文档频率）
3. 分类器：`LogisticRegression(maxIter=100, regParam=0.01)` / 可选 `RandomForestClassifier`
4. 评估：`MulticlassClassificationEvaluator(metricName="f1")`、混淆矩阵
5. 通过 `--master local[*]` 模拟分布式，可选 `yarn` / `standalone` 模式观测扩展性

**资源记录**：Spark UI（ Stages / Task 耗时）、`spark.executor.memory` 配置对比

#### 框架 3：PyTorch（深度学习）

```
project1/
├── pytorch/
│   ├── train_torch.py         # 训练 + 评估
│   ├── predict_torch.py       # 推理测试
│   ├── model.py               # 分类模型定义
│   └── requirements.txt
```

**实现路径**（提供两档方案，按实际情况选择）：

| 方案 | 模型 | 参数量 | 训练方式 | 硬件需求 |
|------|------|--------|---------|---------|
| A（轻量） | EmbeddingBag + Linear 分类头 | ~5M | 从零训练，类 FastText | CPU 可运行 |
| B（深度） | MiniMind backbone + 分类头 | ~65M | 加载预训练权重，全参微调 | GPU (≥6GB) |

**方案 A 流程**：
1. MiniMind BPE tokenizer 分词 → `nn.EmbeddingBag(vocab_size=6400, embedding_dim=256)`
2. 池化后接 `nn.Linear(256, 6)` 分类头
3. 损失：`CrossEntropyLoss`，优化器：`AdamW(lr=1e-3)`

**方案 B 流程**：
1. 加载 `mini_deepseek_mind/out/pretrain_768.pth` 预训练权重
2. 冻结/部分解冻 transformer 层，替换 lm_head 为 6 分类头
3. 损失：`CrossEntropyLoss`，优化器：`AdamW(lr=5e-5)`，混合精度 `bfloat16`

**资源记录**：`torch.cuda.max_memory_allocated()`、`time`、`thop.profile` 计算 FLOPs

### 对比维度与实验设计

| 对比维度 | scikit-learn | Spark MLlib | PyTorch |
|---------|-------------|-------------|---------|
| 训练时间 | CPU 单机计时 | Spark UI Stage 耗时 | GPU/CPU 计时 |
| 推理时间 | 单条/批量延迟 | 分布式批量延迟 | 单条/批量延迟 |
| 峰值内存 | `psutil` RSS | Spark Executor 内存 | CPU RSS / GPU 显存 |
| 模型大小 | `.pkl` 文件大小 | Pipeline 序列化大小 | `.pth` 文件大小 |
| 分类精度 | Accuracy / F1 | Accuracy / F1 | Accuracy / F1 |
| 代码行数 | — | — | — |

**控制变量**：
- 同一数据集（固定 10 万条，固定 8:1:1 划分）
- 同一任务（6 分类）
- 同一评估脚本（`project1/evaluate.py`），统一计算指标
- 同一硬件环境记录（CPU 型号、内存、GPU 型号）

**最终产出**：
- `project1/report/` 目录：实验报告（Markdown），含对比表格、柱状图、结论分析

### 技术栈

| 组件 | 技术 |
|------|------|
| 数据标注 | Python（关键词规则），复用 MiniMind BPE tokenizer |
| scikit-learn | scikit-learn >= 1.6.1, joblib, psutil |
| Spark MLlib | PySpark >= 3.5, pyspark.ml.feature, pyspark.ml.classification |
| PyTorch | torch >= 2.6.0, transformers >= 4.57.6, einops |
| 实验对比 | time, psutil, pandas, matplotlib, seaborn |

### 分工

| 成员 | 职责 |
|------|------|
| **A** | 数据标注脚本 + scikit-learn 方案实现 + 对比实验报告撰写 |
| **B** | Spark MLlib 方案实现 + 分布式环境搭建 + 资源数据采集 |
| **C** | PyTorch 方案实现（含 MiniMind 模型加载）+ GPU 资源数据采集 |

---

## 项目 2：基于 MLflow 的 MiniMind SFT 训练 MLOps 实践

### 任务描述

以 MiniMind 小型语言模型的**指令微调（SFT）**为主线，利用 **MLflow** 平台完成完整的 MLOps 全生命周期管理：从数据版本追踪、多组超参数实验对比、模型注册与版本管理，到最终将最佳模型部署为在线推理服务。

核心目标：证明 MLflow 可以有效管理深度学习训练流程中的实验混乱、模型版本失控、部署重复劳动等痛点。

### 数据集

**来源**：`mini_deepseek_mind/dataset/sft_t2t_mini.jsonl`（约 1.6 GB，90 万条多轮对话）

**预处理**：
1. 使用 `datasets` 库加载 JSONL 文件
2. 过滤过短（< 10 tokens）或过长（> 2048 tokens）对话
3. 按 9:1 划分训练集 / 验证集，保存为 Parquet 格式
4. 使用 **MLflow Dataset** 或 **DVC** 记录数据版本

### MLOps 实践流程

```
数据准备 → 实验追踪 → 模型注册 → 模型部署
  │            │            │           │
  │ DVC/       │ MLflow     │ MLflow    │ ModelScope
  │ MLflow     │ Tracking   │ Model     │ / Flask
  │ Dataset    │            │ Registry  │
```

#### 阶段一：数据准备与版本管理

```
project2/
├── data_prep.py           # 数据加载、清洗、划分
├── data_prep.ipynb        # 数据探索分析（EDA）
└── data/
    ├── train.parquet
    └── val.parquet
```

- 使用 `datasets` 库加载 JSONL，转换为 Parquet 格式
- 使用 **DVC**（`dvc init && dvc add data/train.parquet`）追踪数据版本
- 或在 MLflow 中注册 Dataset：`mlflow.data.from_pandas(df, source="sft_t2t_mini.jsonl")`

#### 阶段二：实验追踪（MLflow Tracking）

**使用 MLflow Tracking 记录每次 SFT 训练的全过程。**

```
project2/
├── train_sft_mlflow.py    # SFT 训练 + MLflow 日志
├── mlflow_ui.sh           # 启动 MLflow UI
└── mlruns/                # MLflow 本地实验存储（自动生成）
```

**记录内容**：

| 类别 | 具体项 | MLflow API |
|------|--------|-----------|
| 参数 | learning_rate, batch_size, num_epochs, lora_r, weight_decay | `mlflow.log_param()` |
| 指标 | train_loss, val_loss, accuracy, perplexity | `mlflow.log_metric()` |
| 产物 | 模型权重 (.pth), tokenizer, 训练配置 JSON | `mlflow.log_artifact()` |
| 模型 | 注册为 MLflow Model 格式 | `mlflow.pytorch.log_model()` |

**对比实验设计**（至少 6 组）：

| 实验 | lr | batch_size | epochs | LoRA | 备注 |
|------|-----|-----------|--------|------|------|
| 1 | 1e-5 | 16 | 2 | 无 | baseline |
| 2 | 5e-5 | 16 | 2 | 无 | 更高 lr |
| 3 | 1e-5 | 32 | 2 | 无 | 更大 batch |
| 4 | 1e-5 | 16 | 4 | 无 | 更多 epoch |
| 5 | 1e-4 | 16 | 2 | r=8 | LoRA 微调 |
| 6 | 5e-5 | 32 | 4 | r=16 | 综合最优 |

每次实验自动记录到 MLflow，可在 UI 中可视化对比。

**启动 MLflow UI**：

```bash
mlflow ui --port 5000
```

#### 阶段三：模型注册（MLflow Model Registry）

1. 从 MLflow Tracking 中选择最优实验 run
2. 注册模型：`mlflow.register_model("runs:/<run_id>/model", "MiniMind-SFT")`
3. 管理版本：
   - **v1**：baseline（实验 1）
   - **v2**：高 lr（实验 2）
   - **v3**：LoRA 微调（实验 5）
   - **v4**：最优组合（实验 6）→ **Staging → Production**
4. 阶段转换（Staging → Production），模拟 CI/CD 流程

#### 阶段四：模型部署

**方案 A（推荐）：ModelScope 部署**

```
project2/
├── deploy_modelscope.py   # 模型推送至 ModelScope
└── modelscope_demo.py     # 调用 ModelScope API 测试
```

- 将最优模型权重上传至 ModelScope
- 配置 ModelScope 推理 API，提供 RESTful 接口
- 编写 demo 客户端调用验证

**方案 B：本地 Flask API 部署**

```
project2/
├── deploy_flask.py        # Flask 推理服务
├── client_test.py         # 客户端测试脚本
└── Dockerfile             # Docker 容器化
```

```
docker build -t minimind-sft-api .
docker run -p 5001:5001 minimind-sft-api
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"text": "什么是机器学习？"}'
```

### 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch >= 2.6.0, transformers >= 4.57.6 |
| 实验管理 | MLflow >= 2.20, wandb / swanlab（可选辅助） |
| 数据版本 | DVC >= 3.0 |
| 数据加载 | datasets >= 3.6.0 |
| 部署 | Flask, Docker, ModelScope |
| 可视化 | MLflow UI, matplotlib, streamlit |
| 代码质量 | black, ruff |

### 分工

| 成员 | 职责 |
|------|------|
| **A** | 数据预处理与版本管理（DVC/MLflow Dataset）+ SFT 训练实验脚本 + MLflow Tracking 集成 |
| **B** | 多组超参数实验执行 + MLflow UI 对比分析 + 模型注册与版本管理 |
| **C** | 模型部署（Flask API / Docker / ModelScope）+ 客户端测试 + 整体流程文档 |

---

## 项目关联概览

```
MiniMind SFT 数据 (sft_t2t_mini.jsonl) ←── 项目 2 核心数据
       │
       ├── 项目 1: 提取 query → 自动标注 → 3 框架分类对比
       │     (scikit-learn / Spark MLlib / PyTorch)
       │
       └── 项目 2: MiniMind SFT 训练 → MLflow 全流程管理
             (追踪 / 注册 / 部署 / 监控)
```

两个项目共享 MiniMind 的数据与 tokenizer 生态。项目 1 聚焦于"同一任务跨系统的算法对比"，项目 2 聚焦于"深度学习训练的工程化管理"，互不依赖但数据同源，便于团队协作与成果整合。
