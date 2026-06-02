import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, get_lr_ratio, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, create_optimizer

warnings.filterwarnings('ignore')


"""
============================================
预训练（Pre-training）主循环
============================================
核心逻辑：
  1. 从 DataLoader 获取 batch 数据 (input_ids, labels)
  2. 前向传播 → 计算语言建模 loss (交叉熵)
  3. 梯度累积 + 梯度裁剪 → 反向传播更新参数
  4. 周期性打印日志、保存 checkpoint
"""
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # 计算当前 step 的余弦退火学习率并设置（per-group）
        ratio = get_lr_ratio(epoch * iters + step, args.epochs * iters)
        for param_group in optimizer.param_groups:
            if param_group.get('use_muon', False):
                param_group['lr'] = optimizer._muon_base_lr * ratio
            else:
                param_group['lr'] = optimizer._adam_base_lr * ratio

        # 混合精度上下文：前向传播，得到 loss（语言建模loss + MoE辅助loss）
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps  # 累积步数归一化

        # 反向传播（scaler 缩放梯度以适配 fp16）
        scaler.scale(loss).backward()

        # 累积到 accumulation_steps 步后才更新参数
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 优化器更新权重
            scaler.update()         # 更新 scaler 系数

            optimizer.zero_grad(set_to_none=True)  # 清零梯度

        # 打印训练日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            muon_lr_val = next((g['lr'] for g in optimizer.param_groups if g.get('use_muon')), 0.0)
            adam_lr_val = next((g['lr'] for g in optimizer.param_groups if not g.get('use_muon')), current_lr)
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, muon_lr: {muon_lr_val:.8f}, adam_lr: {adam_lr_val:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "muon_lr": muon_lr_val, "adam_lr": adam_lr_val, "epoch_time": eta_min})

        # 定期保存模型权重和断点续训文件
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    # epoch 结束时处理末尾不足 accumulation_steps 的残余梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":


    print('torch:', torch.__version__)
    print('cuda:', torch.cuda.is_available())
    print('cuda devices:', torch.cuda.device_count())





    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率（AdamW组）")
    parser.add_argument('--muon_lr', default=None, type=float, help="Muon组学习率，不设置则=learning_rate*200/3")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_mhc', default=1, type=int, choices=[0, 1], help="是否使用mHC（0=否，1=是）")
    parser.add_argument('--use_csa_hca', default=0, type=int, choices=[0, 1], help="是否使用CSA/HCA混合注意力（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe), use_mhc=bool(args.use_mhc), use_csa_hca=bool(args.use_csa_hca))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    """
    混合精度训练（Mixed Precision）：
    - bfloat16: 梯度缩放不需要 GradScaler，直接 forward/backward 即可
    - float16: 需要用 GradScaler 防止下溢出
    - CPU 模式: 不使用 autocast
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    """
    加载流程：
    1) init_model 创建 MiniMindForCausalLM 实例，若 from_weight != 'none' 则加载 .pth 权重
    2) PretrainDataset: 加载 jsonl 文件，每条样本 tokenize 后拼成 [BOS] text [EOS] [PAD]...
       labels 中 PAD 位置设为 -100（计算 loss 时忽略）
    3) 分布式训练时使用 DistributedSampler 分片数据
    4) 优化器: AdamW，GradScaler 仅在 float16 模式下启用
    """
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    muon_lr = args.muon_lr if args.muon_lr is not None else args.learning_rate * 200 / 3
    optimizer = create_optimizer(model, adam_lr=args.learning_rate, muon_lr=muon_lr)

    # ========== 6. 从ckp恢复状态 ==========
    """
    断点续训：恢复模型权重、优化器状态、scaler 状态、当前 epoch 和 step
    """
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    """
    逐 epoch 训练流程：
    - 每个 epoch 重新 shuffle 数据
    - 断点续训时跳过已完成的 step（用 SkipBatchSampler）
    - DataLoader 自动做 batch 组装
    """
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
