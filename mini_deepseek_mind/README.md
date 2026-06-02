<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind)](LICENSE)
![Muon Optimizer](https://img.shields.io/badge/Optimizer-AdamW%20%2B%20Muon-blue)
![mHC](https://img.shields.io/badge/Residual-mHC-green)
![CSA/HCA](https://img.shields.io/badge/Attention-CSA%2FHCA-orange)

</div>

<div align="center">
  <h3>"大道至简"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

---

**MiniMind** 是一个从零开始训练超小语言模型（~64M 参数）的开源项目，主线结构对齐 Qwen3 / Qwen3-MoE 生态。仅需约 3 元 GPU 租用成本 + 2 小时即可从零训练出一个可对话的 LLM。

本项目在原版 MiniMind 基础上引入了 **DeepSeek-V4 级别的架构升级**：

- **Muon 优化器**（结合 AdamW）替代纯 AdamW 训练策略
- **mHC（流形约束超链接）** 增强残差连接，稳定深层信号传播
- **CSA/HCA 混合注意力**（Compressed Sparse / Heavily Compressed Attention）压缩长序列 KV cache

---

### 特性

- 极简 Transformer Decoder-Only 结构（Dense + MoE），对齐 Qwen3 生态
- 全流程代码：Pretrain → SFT → LoRA → DPO → PPO/GRPO/CISPO → Agentic RL → 知识蒸馏
- 全部核心算法 PyTorch 原生实现，不依赖第三方高层封装
- 支持 Tool Calling、Adaptive Thinking（`<think>`标签）、YaRN 长文本外推
- 兼容 transformers/vllm/ollama/llama.cpp 等推理引擎
- 支持 DDP / DeepSpeed 多卡训练，SwanLab/WandB 可视化
- 提供 OpenAI 兼容 API 服务端与 Streamlit WebUI

### 已发布模型

| 模型 | 参数量 | 发布时间 |
|------|--------|---------|
| minimind-3 | 64M | 2026.04 |
| minimind-3-moe | 198M-A64M | 2026.04 |
| minimind2 | 104M | 2025.04 |

---

### 架构升级亮点

#### Muon 优化器

参考 DeepSeek-V4 §2.4，使用 Muon（Hybrid Newton-Schulz 迭代）替代大部分参数的 AdamW 优化器。92% 的 hidden 2D 权重矩阵走 Muon，embedding / RMSNorm / mHC 参数走 AdamW。

#### mHC：流形约束超链接

参考 DeepSeek-V4 §2.2，替换标准残差连接为流形约束的超链接：

```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)
```

- 残差扩张系数 n_hc=4，残差流维度从 d 扩张到 4d
- A_l 经 Sigmoid 约束非负有界
- B_l 经 Sinkhorn-Knopp 算法（20 次迭代）投影到双随机矩阵流形，谱范数 ≤ 1
- C_l 经 2·Sigmoid 约束非负有界
- 参数动态生成（input-dependent + static biases）

#### CSA：压缩稀疏注意力

参考 DeepSeek-V4 §2.3.1，将标准注意力替换为压缩 + 稀疏的 MQA 注意力：

- **双序列重叠压缩**：每 m=4 个 KV entries 压缩为 1 个，前后 block 重叠 m 个 entry
- **Lightning Indexer**：低秩查询 + 多 head 索引器，为每个 query 选择 top-k 个压缩 block
- **滑动窗口**：保留最近 n_win=64 个未压缩 KV entry 用于局部建模
- **Partial RoPE**：仅对 head 最后 64 维施加位置编码
- **分组输出投影**：8 heads 分 4 组投影后再合并
- **MQA**：所有 query head 共享同一个压缩 K,V

#### HCA：重度压缩注意力

参考 DeepSeek-V4 §2.3.2，简化版 CSA：

- 单序列压缩（m'=32），无重叠，无索引器
- 在压缩后的 KV 上做密集注意力

#### 层间分配策略

```
Layer 0-1: 标准注意力 (full)
Layer 2:   CSA
Layer 3:   HCA
Layer 4:   CSA
Layer 5:   HCA
Layer 6:   CSA
Layer 7:   HCA
```

---

### 快速开始

```bash
git clone <your-repo-url>
cd minimind && pip install -r requirements.txt
```

**推理**：
```bash
# 下载模型后
python eval_llm.py --load_from ./minimind-3
```

**训练**（推荐使用交互式流水线）：
```bash
python run_pipeline.py
```

或分步执行：
```bash
cd trainer
python train_pretrain.py                   # 预训练
python train_full_sft.py                   # SFT
python train_lora.py                       # LoRA（可选）
python train_dpo.py                        # DPO（可选）
python train_grpo.py --loss_type cispo     # RLAIF（可选）
python train_agent.py                      # Agentic RL（可选）
python train_distillation.py               # 知识蒸馏（可选）
```

所有脚本均支持 `--from_resume 1` 断点续训。

---

### 优化器策略：AdamW + Muon

| 训练阶段 | 默认 lr | Muon lr | 使用 Muon? |
|---------|---------|---------|-----------|
| Pretrain | 5e-4 | 0.033 | 是（~92% 参数） |
| Full SFT | 1e-5 | 6.7e-4 | 是 |
| Distillation | 5e-6 | 3.3e-4 | 是 |
| LoRA | 1e-4 | - | 否（低秩空间收益小） |
| DPO | 4e-8 | - | 否（lr 极低） |
| PPO/GRPO/Agent | 3e-7 | - | 否（lr 极低） |

Muon 通过 Hybrid Newton-Schulz 迭代对 2D 权重矩阵做正交化更新，加速全参数高 lr 阶段的收敛。

---

### 架构配置

| 特性 | Dense | MoE |
|------|-------|-----|
| 参数量 | 60M | 198M-A64M |
| hidden_size | 768 | 768 |
| num_layers | 8 | 8 |
| q_heads / kv_heads | 8 / 4 | 8 / 4 |
| head_dim | 96 | 96 |
| vocab_size | 6400 | 6400 |
| 优化器 | Muon + AdamW | Muon + AdamW |
| 残差连接 | **mHC** (n_hc=4) | **mHC** (n_hc=4) |
| 注意力 | **CSA/HCA** 交叠（可开关） | **CSA/HCA** 交叠（可开关） |

---

### 数据

下载地址：[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

核心文件放入 `./dataset/`：
- `pretrain_t2t_mini.jsonl` (1.2GB, 必备)
- `sft_t2t_mini.jsonl` (1.6GB, 必备)
- `rlaif.jsonl` / `agent_rl.jsonl` / `dpo.jsonl`（可选）

---

### 项目结构

```
├── model/
│   └── model_minimind.py         # 模型定义：含 mHC、CSA、HCA
├── trainer/
│   ├── train_pretrain.py / train_full_sft.py
│   ├── train_lora.py / train_dpo.py
│   ├── train_ppo.py / train_grpo.py
│   ├── train_agent.py / train_distillation.py
│   └── trainer_utils.py         # create_optimizer (Muon + AdamW)
├── scripts/                      # 推理与服务
├── dataset/                      # 数据
├── docs/                         # 文档/论文
│   ├── DeepSeek_V4.pdf           # 参考论文
│   └── why_use_muon.md           # Muon 改动说明
├── logs/                         # 训练日志
├── backup/                       # 旧版本备份
├── muon.py                       # Muon 优化器实现
├── run_pipeline.py               # 交互式训练流水线
├── eval_llm.py                   # CLI 推理入口
├── README.md / README_en.md
└── requirements.txt
```

---

### 模型结构一览

```
MiniMindForCausalLM (70M params)
└── MiniMindModel
    ├── embed_tokens: Embedding(6400, 768)
    ├── MHCBlock × 8 (sinkhorn_iters=20, n_hc=4)    ← mHC 残差替换
    ├── layers × 8
    │    ├── Layer 0-1:  Attention (标准 GQA)
    │    ├── Layer 2,4,6:  CSAAttention               ← CSA 压缩稀疏注意力
    │    ├── Layer 3,5,7:  HCAAttention               ← HCA 重度压缩注意力
    │    ├── input_layernorm / post_attention_layernorm
    │    └── mlp: FeedForward (gate/down/up)
    ├── norm: RMSNorm(768)
    └── RoPE 预计算缓冲区

优化器:
├── Muon 组  (91.5% = 58.98M): hidden 2D 权重
└── AdamW 组 (7.7% = 4.93M): embed / norms / mHC / lm_head
```

---

### 实验

单卡 NVIDIA 3090 训练成本估算：

| 模型 | Pretrain | SFT |
|------|----------|-----|
| minimind-3 (64M) | ~1.2h / 1.6￥ | ~1.1h / 1.4￥ |
| minimind-3 + Muon | 接近 | 接近 |
| minimind-3 + Muon + mHC | +~10% 时间 | +~10% 时间 |
| minimind-3 + Muon + CSA/HCA | +~2x 时间 | +~2x 时间 |

> 本项目基于 Apache 2.0 协议开源。原始项目由 [jingyaogong](https://github.com/jingyaogong/minimind) 开发。
> 架构改进参考 DeepSeek-V4: [arXiv](https://arxiv.org/abs/2505.xxxxx)。
