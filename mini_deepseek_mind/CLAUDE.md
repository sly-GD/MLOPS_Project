# MiniMind Project Context

## Overview

MiniMind: from-scratch tiny LLM (~64M params) aligned with Qwen3/Qwen3-MoE. Full pipeline: Pretrain → SFT → LoRA → DPO → PPO/GRPO/CISPO → Agentic RL → Distillation.

Key modification: **Muon optimizer** + AdamW mixed training strategy (see `why_use_muon.md`).

## Commands

```bash
# Interactive training pipeline (recommended)
python run_pipeline.py

# Individual training (cd trainer/ first)
python train_pretrain.py                    # pretrain
python train_full_sft.py                    # full SFT
python train_lora.py                        # LoRA
python train_dpo.py                         # DPO
python train_grpo.py --loss_type cispo      # GRPO/CISPO
python train_agent.py                       # Agentic RL
python train_distillation.py                # distillation

# Inference
python eval_llm.py --load_from ./minimind-3
python eval_llm.py --weight full_sft

# Resume training
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1

# Distributed training
torchrun --nproc_per_node N train_xxx.py
```

## Key Files

| File | Purpose |
|------|---------|
| `model/model_minimind.py` | MiniMind Dense + MoE model definition, compatible with `transformers.PreTrainedModel` |
| `trainer/trainer_utils.py` | Shared utils: `create_optimizer` (Muon+AdamW split), `get_lr_ratio`, checkpoint, DDP init |
| `muon.py` | Muon optimizer (`MuonWithAuxAdam` for dist, `SingleDeviceMuonWithAuxAdam` for single GPU) |
| `trainer/train_pretrain.py` | Pretraining loop with per-group Muon/AdamW lr scheduling |
| `trainer/train_full_sft.py` | Full SFT (masked loss on assistant tokens only) |
| `dataset/lm_dataset.py` | PretrainDataset (next-token) |
| `dataset/im_dataset.py` | SFT/InstructDataset (masked loss) |
| `run_pipeline.py` | Interactive multi-stage pipeline runner with rich UI |
| `eval_llm.py` | CLI inference with LoRA, thinking, tool call support |

## Muon Optimizer Details

`create_optimizer(model, adam_lr, muon_lr)` in `trainer_utils.py` splits params:
- **Muon group** (~92.3%): all 2D hidden weights (nn.Linear.weight with ndim==2)
- **AdamW group** (~7.7%): embeddings, norms, lm_head, biases, 1D params

Per-step lr scheduling in training scripts:
```python
ratio = get_lr_ratio(step, total_steps)
for group in optimizer.param_groups:
    if group.get('use_muon', False):
        group['lr'] = optimizer._muon_base_lr * ratio
    else:
        group['lr'] = optimizer._adam_base_lr * ratio
```

Muon uses Newton-Schulz iteration (5 steps) to orthogonalize 2D weight updates. Expect ~20-30% slower per-step but faster convergence.

## Architecture

- Decoder-only Transformer, Pre-Norm + RMSNorm, SwiGLU, RoPE (YaRN support)
- `q_heads=8, kv_heads=4, max_pos=32768, rope_theta=1e6`
- MoE variant: 4 experts / top-1 routing (no shared expert)
- Tokenizer: BPE+ByteLevel, vocab size 6400
- Model config in `MiniMindConfig` supports CSA/HCA attention, mHC connections (experimental)

## Conventions

- All losses computed in `train_epoch()` inside each `train_*.py`
- Checkpoint format: `{weight_name}_{dimension}.pth` (e.g. `full_sft_768.pth`)
- Resume checkpoint: `{weight_name}_{dimension}_resume.pth` (full optimizer+scaler state)
- Arguments use `argparse` with consistent names across scripts
- MoE suffix for weights: `_moe` appended after dimension (e.g. `pretrain_768_moe.pth`)
- `use_wandb` flag enables SwanLab/WandB logging
- `use_compile` flag enables `torch.compile` (CUDA only)
- Dataset format: JSONL, conversations with `role`/`content`/`tool_calls` fields
