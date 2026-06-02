# 项目 1 执行计划 — 中文文本主题分类多系统算法对比

## 环境快照

| 项目 | 值 |
|------|-----|
| OS | Linux |
| Python | 3.11.15 (conda env: `mlops_project`) |
| CPU | 16 核 |
| RAM | 7.7 GB |
| GPU | RTX 3060 6GB (CUDA 13.0) |
| 硬盘 | 909 GB 可用 |
| Java | ❌ 未安装（Spark 需要 OpenJDK 17） |
| 数据集 | `minimind_data/sft_t2t_mini.jsonl`（905,718 行 / 1.7 GB） |
| PyTorch 方案 | **A**（EmbeddingBag + Linear，~5M 参数，从零训练） |
| 数据规模 | **10 万条**（6 类均衡，8:1:1 切分） |
| 执行方式 | **单人串行** |

---

## 阶段划分

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5 ──→ Phase 6 ──→ Phase 7
  环境搭建      数据标注      公共工具      sklearn        Spark        PyTorch     对比报告     验证跑通
```

---

## Phase 0：环境与依赖准备

### Step 0.1 — 安装 Java（Spark 前置依赖）

- 为什么：PySpark 需要 JVM，当前系统无 Java
- 安装：`apt install openjdk-17-jdk-headless`（无 GUI，~120 MB）
- 验证：`java -version` 输出 `openjdk version "17.x.x"`

### Step 0.2 — 安装 Python 依赖

```bash
pip install numpy pandas scikit-learn joblib psutil matplotlib seaborn
pip install pyspark
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets einops
```

| 包 | 用途 | 预估大小 |
|----|------|---------|
| numpy, pandas | 数据处理 | 50 MB |
| scikit-learn | 框架 1 | 30 MB |
| pyspark | 框架 2 | 250 MB |
| torch | 框架 3 | 2.1 GB |
| transformers | BPE tokenizer | 已含 |
| joblib | 模型持久化 | 内置 |
| psutil | 内存监控 | 内置 |
| matplotlib, seaborn | 图表 | 30 MB |
| einops | 深度学习工具 | <1 MB |

### Step 0.3 — 目录结构验证

```
project1/
├── data/                  ← 标注后的 CSV 数据集
├── common/               ← 共享工具
│   ├── utils.py          # BPE tokenizer 封装、路径常量
│   ├── evaluate.py       # 统一评估：accuracy/precision/recall/F1/混淆矩阵
│   └── monitor.py        # 资源监控器（时间 / CPU RSS / GPU 显存）
├── sklearn/
│   ├── train_sklearn.py
│   ├── predict_sklearn.py
│   └── requirements.txt
├── spark/
│   ├── train_spark.py
│   ├── predict_spark.py
│   └── requirements.txt
├── pytorch/
│   ├── train_torch.py
│   ├── predict_torch.py
│   └── model.py
├── artifacts/             ← 三个框架产出的模型文件
├── report/
│   ├── report.md          # 最终实验报告
│   └── figures/           # 对比图表
└── PLAN.md                ← 本文档
```

---

## Phase 1：数据标注与切分（label_data.py）

### 输入/输出

- 输入：`minimind_data/sft_t2t_mini.jsonl`（905,718 条多轮对话）
- 输出：`project1/data/train.csv`（80k）、`val.csv`（10k）、`test.csv`（10k）

### 关键算法

1. **query 提取**：每条对话的 `conversations[0].content`（首轮 user query）
2. **关键词分类**（6 类 + `other`）：

| 类别 | 匹配关键词 |
|------|-----------|
| `identity` | 真实来源、真实身份、开发背景、哪家公司、谁开发 |
| `tech_discussion` | 如何平衡、怎么处理、原理、算法、为什么、数学、训练 |
| `recommendation` | 推荐、有没有看过、好电影、好书、好听的 |
| `casual_chat` | 今天、过得怎么样、在忙什么、聊聊、你好吗 |
| `domain_knowledge` | 医疗、伦理、人工智能、量子、科学、基因 |
| `capability` | 能不能、是否支持、可以做什么、多语言 |

3. **分层采样**：每个类取 min(available, 100k/6 ≈ 16666) 条，合并后随机 shuffle
4. **切分**：8:1:1 train/val/test，保存为 CSV

### 边界处理

- 空 query → 跳过
- 长度 < 3 字符 → 跳过
- 多关键词匹配 → 按首次匹配的类别
- 无匹配 → 归入 `other`（采样时丢弃，因为 not in 目标 6 类）

### 资源开销

- JSONL 逐行流式读取（不一次性全部加载到内存）
- 分词使用 MiniMind 自己的 BPE tokenizer（通过 `transformers` 加载 `mini_deepseek_mind/model/tokenizer.json`）

---

## Phase 2：公共工具层（common/）

### `common/evaluate.py` — 统一评估器

三个框架的预测结果统一用同一函数计算指标：

```python
def evaluate(y_true: List[str], y_pred: List[str], save_path: str = None) -> dict:
    # accuracy
    # precision (macro)
    # recall (macro)
    # f1 (macro)
    # confusion matrix (matplotlib 保存)
```

### `common/monitor.py` — 资源监控器

**CPU 模式**（sklearn, Spark CPU）：

```python
class CPUMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
    def start(self): ...
    def stop(self): ...
    def peak_memory_mb(self) -> float: ...  # RSS 峰值
    def elapsed_seconds(self) -> float: ...
```

**GPU 模式**（PyTorch，可选 Fallback）：

```python
class GPUMonitor(CPUMonitor):
    def peak_gpu_memory_mb(self) -> float:
        return torch.cuda.max_memory_allocated() // 1024**2
```

### `common/utils.py` — 共享工具

- `get_tokenizer()` — 加载 MiniMind BPE tokenizer
- `CLASSES` — 类别列表常量
- `DATA_DIR`, `ARTIFACTS_DIR` — 路径常量
- `create_timestamp()` — 时间戳生成

---

## Phase 3：scikit-learn 方案（CPU 单机）

### 文件

- `sklearn/train_sklearn.py` — 训练 + 评估
- `sklearn/predict_sklearn.py` — 推理测试

### 流程

```python
# 1. 加载 CSV
df_train = pd.read_csv("data/train.csv")

# 2. 文本向量化
tokenizer = get_tokenizer()
def tokenize(text):
    return " ".join(tokenizer.tokenize(text))  # BPE 分词后用空格连接

vectorizer = TfidfVectorizer(
    max_features=5000,
    tokenizer=lambda x: x.split(),
    preprocessor=lambda x: x          # 已分词，不再预处理
)
X_train = vectorizer.fit_transform(df_train["text"].apply(tokenize))

# 3. 训练
model = LogisticRegression(C=1.0, multi_class="multinomial", solver="lbfgs", max_iter=1000)
# 套 CPUMonitor → 记录 time / memory
model.fit(X_train, y_train)

# 4. 保存
joblib.dump(model, "artifacts/sklearn_model.pkl")
joblib.dump(vectorizer, "artifacts/sklearn_vectorizer.pkl")

# 5. 评估 → 调用 common.evaluate()
```

### 预期性能（80k 训练，5k 特征，16 核）

| 指标 | 估值 |
|------|------|
| 训练时间 | < 30 秒 |
| 峰值内存 | < 500 MB RSS |
| 模型大小 | < 2 MB |

### 风险

- `lbfgs` solver 在 6 类 softmax 下收敛良好，但 80k×5000 的 sparse matrix 约 80 MB 稀疏表示
- 备选 solver（`sag`）可提速但精度一致

---

## Phase 4：Apache Spark MLlib 方案（分布式计算）

### 文件

- `spark/train_spark.py` — 训练 + 评估
- `spark/predict_spark.py` — 推理测试

### 流程

```python
# 0. SparkSession
spark = SparkSession.builder \
    .appName("TopicClassification") \
    .master("local[*]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# 1. 加载
df = spark.read.csv("data/train.csv", header=True, inferSchema=True)

# 2. 文本向量化
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# 3. 标签编码
labelIndexer = StringIndexer(inputCol="label", outputCol="label_index")

# 4. 分类器
lr = LogisticRegression(maxIter=100, regParam=0.01, family="multinomial")

# 5. Pipeline + 训练
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, labelIndexer, lr])
# 注：此处 Tokenizer 会空格分词（HashingTF 输入）。为保持与 sklearn 公平，需传已 BPE 分词的列
# 方案：在 pandas 侧预处理 text → "bpe_tokens" 列，写入 CSV，Spark 直接读该列
model = pipeline.fit(df)

# 6. 保存
model.write().overwrite().save("artifacts/spark_model")
```

### ⚠️ 关键设计决策：BPE 分词一致性问题

- sklearn/TF-IDF 用了 BPE 分词后的空格文本（单词为 BPE subword）
- Spark/Tokenizer 默认按空格分词 → 和 BPE 天然兼容
- **解决方案**：数据集 CSV 中加一列 `bpe_text`（由 label_data.py 预先 BPE 分词），三个框架都读这列
- Spark 的 `Tokenizer` 读 `bpe_text` 列保持与 sklearn TF-IDF 相同的特征空间

### 预期性能（local[*]，16 核，2g executor）

| 指标 | 估值 |
|------|------|
| 训练时间 | 1-3 分钟 |
| 峰值内存（driver） | ~1 GB RSS |
| 模型大小 | ~4 MB |

### 风险

- Spark local 模式没有真正的分布式优势，训练可能慢于 sklearn
- 这是**设计的对比目标** — 我们要在报告中解释为什么"分布式框架在小数据集上的开销 > 收益"
- `HashingTF` 有哈希碰撞风险（5000 特征下碰撞率极低）

---

## Phase 5：PyTorch 方案 A（EmbeddingBag + Linear）

### 文件

- `pytorch/model.py` — `TextClassifier` 模型定义
- `pytorch/train_torch.py` — 训练 + 评估
- `pytorch/predict_torch.py` — 推理测试

### 模型架构

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size=6400, embed_dim=256, num_classes=6):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", sparse=True)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, offsets):
        x = self.embedding(input_ids, offsets)   # (batch, embed_dim)
        return self.classifier(x)                 # (batch, 6)
```

### 训练配置

| 参数 | 值 |
|------|-----|
| vocab_size | 6400（MiniMind 词表） |
| embed_dim | 256 |
| 优化器 | AdamW lr=1e-3 |
| 损失 | CrossEntropyLoss |
| batch_size | 64 |
| epoch | 10（early stopping on val_loss） |
| 混合精度 | bfloat16（GPU） / fp32（CPU fallback） |
| 学习率调度 | CosineAnnealingLR |

### 分词器对接

- 用 MiniMind BPE tokenizer 把 text 转为 token IDs
- `nn.EmbeddingBag` 支持可变长度输入（通过 `input_ids` + `offsets`）
- DataLoader 不要求 padding → 内存高效

### 预期性能（GPU：RTX 3060）

| 指标 | 估值 |
|------|------|
| 训练时间（10 epoch） | 30-60 秒 |
| 峰值显存 | < 1 GB |
| CPU 峰值内存 | < 2 GB |
| 模型大小 | ~6 MB（vocab * embed + embed * 6） |
| FLOPs | ~10M 级（thop 测量） |

---

## Phase 6：对比报告与可视化

### 目录 `report/`

```
report/
├── report.md              # 完整实验报告
└── figures/
    ├── training_time.png
    ├── inference_latency.png
    ├── peak_memory.png
    ├── model_size.png
    ├── accuracy_f1.png
    ├── confusion_matrix_sklearn.png
    ├── confusion_matrix_spark.png
    └── confusion_matrix_torch.png
```

### report.md 结构

```markdown
# 中文文本主题分类多系统算法对比实验报告

## 1. 实验概述
## 2. 硬件环境
## 3. 数据集
## 4. 各框架实现
### 4.1 scikit-learn
### 4.2 Apache Spark MLlib
### 4.3 PyTorch
## 5. 对比结果
### 5.1 分类精度对比（表格）
### 5.2 训练时间对比（柱状图）
### 5.3 推理速度对比
### 5.4 峰值内存/显存对比
### 5.5 模型大小对比
## 6. 分析与结论
## 7. 附录：控制变量说明
```

### 对比表格模板

| 维度 | scikit-learn | Spark MLlib | PyTorch |
|------|-------------|-------------|---------|
| Accuracy | xx% | xx% | xx% |
| F1 (macro) | x.xxx | x.xxx | x.xxx |
| Training Time (s) | xx.x | xxx.x | xx.x |
| Inference (1k samples, s) | x.xx | xx.xx | x.xx |
| Peak Memory (MB) | xxx | xxx | xxx |
| Model Size (MB) | x.x | x.x | x.x |
| Code Lines | ~xx | ~xx | ~xx |

---

## Phase 7：烟囱测试 → 全量跑通

### 7.1 烟囱测试（1 万条子集）

- 先用前 10,000 条数据跑通三个框架 pipeline
- 验证每个步骤的输出格式正确
- 确认 metrics 在合理范围内

### 7.2 全量执行顺序

```
smoke_test: 1k → 验证通过
    ↓
Phase 3 sklearn: 80k train → evaluate on test → save artifacts
    ↓
Phase 4 Spark: 80k train → evaluate on test → save artifacts
    ↓
Phase 5 PyTorch: 80k train → evaluate on test → save artifacts
    ↓
Phase 6: collect metrics → generate report → save figures
```

### 7.3 验证检查清单

- [ ] 三个框架 test set 一致
- [ ] evaluate.py 统一产出 metrics
- [ ] 资源监控正确记录
- [ ] 混淆矩阵正常（对角线主导）
- [ ] 图表可读
- [ ] report.md 完整

---

## 风险登记表

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Java 安装失败 | 低 | 高 | 使用 SDKMAN 或 conda install -c conda-forge openjdk |
| PySpark OOM | 中 | 中 | 减小 executor memory / 用 spark.driver.memory + local[4] 限制并行 |
| GPU 显存不足 | 低 | 中 | 方案 A 显存 <1G，6G 绰绰有余 |
| BPE tokenizer 加载失败 | 低 | 高 | 使用 HF `AutoTokenizer` 而非手动加载 tokenizer.json |
| 标注样本均衡性差 | 中 | 中 | 某些类别匹配不到 16666 条时，降低采样数，记录实际样本分布 |
| HashingTF 碰撞 | 低 | 低 | 设 numFeatures=5000，碰撞概率 < 0.1% |
| Spark Kernel 死掉 | 低 | 高 | 设 `spark.driver.extraJavaOptions="-Xss4m"` |

---

## 时间估算（单人串行）

| Phase | 内容 | 预估耗时 |
|-------|------|---------|
| 0 | 环境安装 | 30 分钟（pip + apt） |
| 1 | 数据标注脚本 + 运行 | 30 分钟 |
| 2 | 公共工具编写 | 15 分钟 |
| 3 | sklearn 方案 | 20 分钟 |
| 4 | Spark 方案 | 30 分钟 |
| 5 | PyTorch 方案 | 45 分钟 |
| 6 | 对比报告 + 图表 | 30 分钟 |
| 7 | 调试 + 全量跑通 | 2-3 小时 |
| **总计** | | **5-6 小时** |
