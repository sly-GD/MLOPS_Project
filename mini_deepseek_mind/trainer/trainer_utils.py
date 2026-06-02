"""
训练工具函数集合：

本文件封装了 MiniMind 所有训练脚本共享的基础设施，包括：
  1. 模型参数量统计（区分 Dense / MoE 架构）
  2. 分布式训练初始化（DDP）
  3. 随机种子设置
  4. 余弦退火学习率调度
  5. 模型权重加载与 tokenizer 初始化
  6. Checkpoint 保存/恢复（含 wandb id、优化器、scaler 等完整状态）
  7. Batch Sampler 跳过逻辑（用于断点续训跳过已完成的 step）
  8. 外部 Reward Model 包装器（用于 PPO/GRPO 阶段打分）
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM


"""
============================================
模型参数量统计
============================================
对于 MoE 架构，需要区分：
  - total: 所有参数的总和（包括所有 expert 的参数量）
  - active: 每次前向传播实际激活的参数（base + n_active experts + shared experts）
  
计算公式：
  base = total - (expert_param × n_routed_experts) - (shared_expert_param × n_shared_experts)
  active = base + (expert_param × n_active_experts_per_tok) + (shared_expert_param × n_shared_experts)

打印格式："Model Params: 64.00M"（Dense）或 "Model Params: 198.00M-A64.00M"（MoE）
"""
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 取第一个 expert（mlp.experts.0）的参数量作为单个 expert 的参数量
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


"""
============================================
判断当前进程是否为主进程（rank 0），用于日志打印和模型保存
"""
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


"""
============================================
余弦退火学习率调度（Cosine Annealing）
============================================
公式：
    lr(current_step) = lr_max × (0.1 + 0.45 × (1 + cos(π × current_step / total_steps)))

特点：
  - learning rate 最大值为 lr_max
  - 最小值约为 lr_max × 0.1
  - 余弦形状平滑下降，在训练后期衰减更快
  - 相比 StepLR，这种"热身衰减"的方式通常收敛效果更好
"""
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def get_lr_ratio(current_step, total_steps):
    return 0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps))


def create_optimizer(model, adam_lr=3e-4, muon_lr=0.02, weight_decay=0.1, momentum=0.95, adam_betas=(0.9, 0.95), adam_eps=1e-10):
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

    hidden_2d = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and 'embed' not in n and 'lm_head' not in n and 'mhc' not in n:
            hidden_2d.append(p)
        else:
            other_params.append(p)

    param_groups = [
        dict(params=hidden_2d, use_muon=True,
             lr=muon_lr, momentum=momentum, weight_decay=weight_decay),
        dict(params=other_params, use_muon=False,
             lr=adam_lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay),
    ]

    use_dist = dist.is_initialized() and dist.get_world_size() > 1
    OptimizerClass = MuonWithAuxAdam if use_dist else SingleDeviceMuonWithAuxAdam
    optimizer = OptimizerClass(param_groups)
    optimizer._muon_base_lr = muon_lr
    optimizer._adam_base_lr = adam_lr
    return optimizer


"""
============================================
分布式训练初始化（DDP — Distributed Data Parallel）
============================================
通过环境变量 RANK 判断是否处于分布式环境：
  - 若 RANK 未设置（非分布式）：返回 0，不做任何操作
  - 若 RANK 已设置（分布式）：
    1. 调用 dist.init_process_group(backend="nccl") 初始化 NCCL 通信后端
    2. 从 LOCAL_RANK 环境变量获取当前进程的本地 GPU 编号
    3. 设置当前进程使用的 CUDA 设备

环境变量由 torchrun 或 mp.spawn 自动设置：
  RANK        — 全局进程编号
  LOCAL_RANK  — 当前节点内的进程编号
  WORLD_SIZE  — 总进程数
"""
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


"""
============================================
随机种子设置
============================================
固定 Python random、numpy、torch（CPU + GPU）的随机种子，
确保实验可复现。

额外设置：
  - cudnn.deterministic = True   保证卷积等操作确定性
  - cudnn.benchmark = False      关闭自动调优（benchmark 会引入随机性）
"""
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""
============================================
Checkpoint 管理（保存 / 加载）

双重文件策略：
  1. xxx.pth          — 仅模型权重（half + cpu），供推理加载使用
  2. xxx_resume.pth   — 完整状态（权重 + optimizer + scaler + epoch + step + wandb_id），供续训使用

保存流程（model != None）：
  1. 解包 DDP / torch.compile 包装，拿到原始模型
  2. 保存 .pth（仅权重，fp16 + cpu）
  3. 保存 _resume.pth（含完整训练状态）

加载流程（model == None）：
  1. 若 _resume.pth 存在则加载并返回
  2. 若 GPU 数量发生变化，自动调整 step 数（保证学习率调度对齐）：
      调整公式：step = step × old_world_size / new_world_size

算法背景：
  resume 恢复时，如果 world_size 变了，每个 step 看到的样本量不同，
  需要按比例调整 step 数以保持 training progress 对齐。
"""
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='./checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # 解包 DDP / torch.compile，获取原始 model 的 state_dict
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}  # fp16 节省磁盘
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)  # 原子写入，防止断电写损坏

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 保存完整训练状态
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        # 额外参数（如 scaler）通过 kwargs 传入
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


"""
============================================
模型 & Tokenizer 初始化
============================================
加载流程：
  1. 使用 HuggingFace AutoTokenizer 从 model/ 目录加载 BPE tokenizer
  2. 创建 MiniMindForCausalLM 实例
  3. 若 from_weight != 'none'，从 {save_dir}/{from_weight}_*.pth 加载预训练权重
  4. 统计并打印参数量
  5. 将模型移至指定设备并返回

参数：
  from_weight — 权重文件名前缀（如 'pretrain'、'full_sft' 等）
                设为 'none' 则从随机初始化开始
"""
def init_model(lm_config, from_weight='pretrain', tokenizer_path='./model', save_dir='./out', device='cuda'):
    tokenizer_path = os.path.abspath(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


"""
============================================
SkipBatchSampler — 跳过前 N 个 batch 的采样器
============================================
用途：
  断点续训时，某些 step 已经训练过了，需要跳过这些 batch。

实现逻辑：
  - 遍历 sampler 中的 index，每 batch_size 个组成一个 batch
  - 跳过前 skip_batches 个完整的 batch
  - 从第 skip_batches+1 个 batch 开始 yield

注意：
  - 如果数据总数不是 batch_size 的整数倍，最后一个不足的 batch 只有当
    skip_batches 已经跳过足够数量时才会被 yield
"""
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


"""
============================================
Reward Model 包装器（用于 PPO/GRPO 阶段）
============================================
功能：
  封装一个 HuggingFace 模型作为奖励模型（RM），为对话生成结果打分。

使用方式：
  reward_model = LMForRewardModel("path/to/reward-model")
  score = reward_model.get_score(messages, response)

get_score 流程：
  1. 将历史对话和最新 query 拼接为完整上下文
  2. 构造 user/assistant 格式的评价 prompt
  3. 调用内部模型的 get_score 方法得到分数
  4. 将分数裁剪到 [-3.0, 3.0] 范围

注意：
  这里使用的是完整的 AutoModel（非 AutoModelForSequenceClassification），
  因为一些 reward model 本身暴露了 get_score 接口（如 internlm2-1.8b-reward）。
"""
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
