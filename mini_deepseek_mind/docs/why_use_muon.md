改了什么
新增文件
muon.py — 从 Muon 仓库拷贝到项目根目录。包含 MuonWithAuxAdam（分布式）和 SingleDeviceMuonWithAuxAdam（单卡）两个优化器类。

修改 trainer/trainer_utils.py
新增两个函数：

get_lr_ratio(current_step, total_steps)

提取原有 get_lr 中的余弦退火比例部分（0.1~1.0）
用于为 Muon 组、AdamW 组分别乘以各自的 base_lr
create_optimizer(model, adam_lr=3e-4, muon_lr=0.02, ...)

自动拆分模型参数：92.3% 进 Muon 组（所有 2D hidden 权重），7.7% 进 AdamW 组（embedding + norms + lm_head）
根据 dist.is_initialized() 自动选择 MuonWithAuxAdam 或 SingleDeviceMuonWithAuxAdam
在 optimizer 上存储 _muon_base_lr / _adam_base_lr 供调度用
修改 3 个训练脚本
脚本	改动点
train_pretrain.py	
① import 增加 get_lr_ratio, create_optimizer 
② 新增 --muon_lr 参数 
③ optim.AdamW(...) → create_optimizer(...) 
④ lr 调度从统一赋值改为 per-group 按 base_lr 缩放

train_full_sft.py	同上
train_distillation.py	同上（学生模型使用 Muon）
为什么改这些
Muon 的优势在于通过 Newton-Schulz 迭代将每层权重更新正交化，让优化方向保持旋转而非拉伸，加速收敛。 改动适用场景：

脚本 |	lr	|阶段|	用 Muon 的理由
train_pretrain.py	|5e-4	|从零预训练	|最核心：大 lr + full-rank 矩阵训练，Muon 收益最大
train_full_sft.py	|1e-5	|全量微调	|中等 lr，模型容量全开，Muon 正交化有帮助
train_distillation.py	|5e-6	|知识蒸馏	|学生模型也在全量更新，同一套模式直接套用
哪些没改，为什么
脚本	|lr	|不改的原因
train_lora.py	|1e-4	|LoRA adapter 是小秩矩阵（r=8/16），Newton-Schulz 正交化在低秩空间收益甚微甚至有害
train_dpo.py	|4e-8	|学习率极低，每次更新量极小，Muon 的 SVD 近似在这种尺度下几乎没有效果
train_ppo.py	|3e-7	|低 lr + 使用 CosineAnnealingLR 调度器 + 双优化器（actor/critic），为 Muon 改造调度器得不偿失
train_grpo.py	|3e-7	|同上（低 lr + CosineAnnealingLR）
train_agent.py	|3e-7	|同上

简而言之：只在"较高学习率 + 全参数更新"的训练阶段使用 Muon，低 lr 的 RL/DPO 微调阶段以及 LoRA 的低秩适配器意义不大。

--muon_lr 参数说明
不传时自动按 learning_rate * 200/3 计算 Muon 学习率（这个比例来自 Muon 论文推荐值：0.02 / 3e-4 ≈ 66.7）。每种训练类型的默认值：

脚本	|--learning_rate	|自动 --muon_lr
pretrain	|5e-4	|0.033
full_sft	|1e-5	|6.7e-4
distillation	|5e-6	|3.3e-4
如果觉得自动值不合适，可以手动传 --muon_lr 0.02 覆盖。


为什么这一次Epoch:[1/1](702/39695), loss: 4.1549, logits_loss: 4.1549, aux_loss: 0.0000, lr: 0.00049965, epoch_time: 197.0min 只有39695个batch，还有一个为什么感觉训练的略慢了


39695 = 1,270,238 样本 ÷ 32 batch_size，是正确数量。之前看到的 70390 应该是不同配置（比如 batch_size 不同或者之前的代码改动导致的计数差异）。

慢的原因：Muon 在每步 optimizer 更新时对每个 2D 权重矩阵做 5 次 Newton-Schulz 迭代（X @ X.mT 等矩阵乘法），而 AdamW 只有 element-wise 操作。92% 的参数走 Muon，相当于每步多了一堆矩阵乘法，额外开销明显。


Epoch:[1/1](2821/39695), loss: 3.0599, logits_loss: 3.0599, aux_loss: 0.0000, lr: 0.00049442, epoch_time: 167.0min
Epoch:[1/1](2822/39695), loss: 2.8255, logits_loss: 2.8255, aux_loss: 0.0000, lr: 0.00049441, epoch_time: 167.0min
Epoch:[1/1](2823/39695), loss: 2.9559, logits_loss: 2.9559, aux_loss: 0.0000, lr: 0.00049441, epoch_time: 167.0min
Epoch:[1/1](2824/39695), loss: 2.8849, logits_loss: 2.8849, aux_loss: 0.0000, lr: 0.00049440, epoch_time: 167.0min
Epoch:[1/1](2825/39695), loss: 3.1858, logits_loss: 3.1858, aux_loss: 0.0000, lr: 0.00049440, epoch_time: 167.0min
Epoch:[1/1](2826/39695), loss: 2.5105, logits_loss: 2.5105, aux_loss: 0.0000, lr: 0.00049440, epoch_time: 167.0min
Epoch:[1/1](2827/39695), loss: 3.0254, logits_loss: 3.0254, aux_loss: 0.0000, lr: 0.00049439, epoch_time: 167.0min
Epoch:[1/1](2828/39695), loss: 2.8817, logits_loss: 2.8817, aux_loss: 0.0000, lr: 0.00049439, epoch_time: 167.0min
Epoch:[1/1](2829/39695), loss: 2.8231, logits_loss: 2.8231, aux_loss: 0.0000, lr: 0.00049438, epoch_time: 167.0min
Epoch:[1/1](2830/39695), loss: 2.8601, logits_loss: 2.8601, aux_loss: 0.0000, lr: 0.00049438, epoch_time: 167.0min
Epoch:[1/1](2831/39695), loss: 2.8117, logits_loss: 2.8117, aux_loss: 0.0000, lr: 0.00049438, epoch_time: 167.0min
Epoch:[1/1](2832/39695), loss: 2.7128, logits_loss: 2.7128, aux_loss: 0.0000, lr: 0.00049437, epoch_time: 167.0min
Epoch:[1/1](2833/39695), loss: 2.8281, logits_loss: 2.8281, aux_loss: 0.0000, lr: 0.00049437, epoch_time: 167.0min
Epoch:[1/1](2834/39695), loss: 2.7246, logits_loss: 2.7246, aux_loss: 0.0000, lr: 0.00049436, epoch_time: 167.0min
Epoch:[1/1](2835/39695), loss: 3.0214, logits_loss: 3.0214, aux_loss: 0.0000, lr: 0.00049436, epoch_time: 167.0min
Epoch:[1/1](2836/39695), loss: 2.7842, logits_loss: 2.7842, aux_loss: 0.0000, lr: 0.00049436, epoch_time: 167.0min
Epoch:[1/1](2837/39695), loss: 2.6645, logits_loss: 2.6645, aux_loss: 0.0000, lr: 0.00049435, epoch_time: 167.0min
Epoch:[1/1](2838/39695), loss: 2.9547, logits_loss: 2.9547, aux_loss: 0.0000, lr: 0.00049435, epoch_time: 167.0min
Epoch:[1/1](2839/39695), loss: 2.7773, logits_loss: 2.7773, aux_loss: 0.0000, lr: 0.00049434, epoch_time: 167.0min
Epoch:[1/1](2840/39695), loss: 2.6658, logits_loss: 2.6658, aux_loss: 0.0000, lr: 0.00049434, epoch_time: 167.0min
Epoch:[1/1](2841/39695), loss: 3.0089, logits_loss: 3.0089, aux_loss: 0.0000, lr: 0.00049434, epoch_time: 167.0min
Epoch:[1/1](2842/39695), loss: 3.1231, logits_loss: 3.1231, aux_loss: 0.0000, lr: 0.00049433, epoch_time: 167.0min
Epoch:[1/1](2843/39695), loss: 2.9849, logits_loss: 2.9849, aux_loss: 0.0000, lr: 0.00049433, epoch_time: 167.0min
Epoch:[1/1](2844/39695), loss: 2.9570, logits_loss: 2.9570, aux_loss: 0.0000, lr: 0.00049432, epoch_time: 167.0min
Epoch:[1/1](2845/39695), loss: 2.5968, logits_loss: 2.5968, aux_loss: 0.0000, lr: 0.00049432, epoch_time: 167.0min
Epoch:[1/1](2846/39695), loss: 2.7778, logits_loss: 2.7778, aux_loss: 0.0000, lr: 0.00049432, epoch_time: 167.0min
Epoch:[1/1](2847/39695), loss: 3.0608, logits_loss: 3.0608, aux_loss: 0.0000, lr: 0.00049431, epoch_time: 167.0min
Epoch:[1/1](2848/39695), loss: 3.0060, logits_loss: 3.0060, aux_loss: 0.0000, lr: 0.00049431, epoch_time: 167.0min
Epoch:[1/1](2849/39695), loss: 2.8641, logits_loss: 2.8641, aux_loss: 0.0000, lr: 0.00049430, epoch_time: 167.0min
Epoch:[1/1](2850/39695), loss: 2.8574, logits_loss: 2.8574, aux_loss: 0.0000, lr: 0.00049430, epoch_time: 167.0min
Epoch:[1/1](2851/39695), loss: 3.0284, logits_loss: 3.0284, aux_loss: 0.0000, lr: 0.00049430, epoch_time: 167.0min
Epoch:[1/1](2852/39695), loss: 3.2062, logits_loss: 3.2062, aux_loss: 0.0000, lr: 0.00049429, epoch_time: 167.0min
Epoch:[1/1](2853/39695), loss: 2.7461, logits_loss: 2.7461, aux_loss: 0.0000, lr: 0.00049429, epoch_time: 167.0min
Epoch:[1/1](2854/39695), loss: 2.8128, logits_loss: 2.8128, aux_loss: 0.0000, lr: 0.00049428, epoch_time: 167.0min
Epoch:[1/1](2855/39695), loss: 2.8022, logits_loss: 2.8022, aux_loss: 0.0000, lr: 0.00049428, epoch_time: 167.0min
Epoch:[1/1](2856/39695), loss: 2.7017, logits_loss: 2.7017, aux_loss: 0.0000, lr: 0.00049428, epoch_time: 167.0min
Epoch:[1/1](2857/39695), loss: 2.8193, logits_loss: 2.8193, aux_loss: 0.0000, lr: 0.00049427, epoch_time: 167.0min
Epoch:[1/1](2858/39695), loss: 2.7855, logits_loss: 2.7855, aux_loss: 0.0000, lr: 0.00049427, epoch_time: 167.0min
Epoch:[1/1](2859/39695), loss: 2.9089, logits_loss: 2.9089, aux_loss: 0.0000, lr: 0.00049426, epoch_time: 167.0min
Epoch:[1/1](2860/39695), loss: 2.8362, logits_loss: 2.8362, aux_loss: 0.0000, lr: 0.00049426, epoch_time: 167.0min
Epoch:[1/1](2861/39695), loss: 3.1090, logits_loss: 3.1090, aux_loss: 0.0000, lr: 0.00049426, epoch_time: 167.0min

