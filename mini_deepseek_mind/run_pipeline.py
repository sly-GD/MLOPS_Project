"""
MiniMind 训练流水线启动脚本

交互式引导完成多阶段训练流程：
  1. 展示推荐训练顺序
  2. 询问已完成的阶段（断点续训 / 跳过）
  3. 选择要运行的阶段并配置参数
  4. 按顺序执行各阶段训练脚本
  5. 实时显示进度条和训练日志
"""

import subprocess
import sys
import os
import re
import time
from pathlib import Path
from threading import Thread
from queue import Queue, Empty
from datetime import datetime

from rich.console import Console, Group
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint
from rich.rule import Rule
from rich.markdown import Markdown

console = Console()

PROJECT_ROOT = Path(__file__).parent

# ============================================================
# 阶段定义：每个阶段的元信息、依赖关系、默认参数
# ============================================================
STAGES = [
    {
        "id": "pretrain",
        "name": "预训练 (Pre-training)",
        "desc": "从零学习语言规律，所有 token 参与 next-token prediction loss 计算",
        "script": "trainer/train_pretrain.py",
        "depends_on": None,
        "output_weight": "pretrain",
        "defaults": {
            "data_path": "../dataset/pretrain_t2t_mini.jsonl",
            "max_seq_len": 340,
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 5e-4,
            "accumulation_steps": 8,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_weight": "none",
            "from_resume": 0,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "最大序列长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数", "int"),
            ("learning_rate", "学习率", "float"),
            ("accumulation_steps", "梯度累积步数", "int"),
        ],
    },
    {
        "id": "full_sft",
        "name": "全参数 SFT (Supervised Fine-Tuning)",
        "desc": "监督微调，仅对 assistant 回复部分计算 loss",
        "script": "trainer/train_full_sft.py",
        "depends_on": "pretrain",
        "output_weight": "full_sft",
        "defaults": {
            "data_path": "../dataset/sft_t2t_mini.jsonl",
            "max_seq_len": 768,
            "batch_size": 16,
            "epochs": 2,
            "learning_rate": 1e-5,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_weight": "pretrain",
            "from_resume": 0,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "最大序列长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数", "int"),
            ("learning_rate", "学习率", "float"),
            ("accumulation_steps", "梯度累积步数", "int"),
        ],
    },
    {
        "id": "lora",
        "name": "LoRA 微调",
        "desc": "低秩适配微调，仅训练 adapter 参数",
        "script": "trainer/train_lora.py",
        "depends_on": "full_sft",
        "output_weight": "lora",
        "defaults": {
            "data_path": "../dataset/lora_medical.jsonl",
            "max_seq_len": 340,
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 1e-4,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_weight": "full_sft",
            "from_resume": 0,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "最大序列长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数（LoRA 通常多一些）", "int"),
            ("learning_rate", "学习率", "float"),
        ],
    },
    {
        "id": "dpo",
        "name": "DPO 偏好优化",
        "desc": "Direct Preference Optimization，直接优化偏好对齐",
        "script": "trainer/train_dpo.py",
        "depends_on": "full_sft",
        "output_weight": "dpo",
        "defaults": {
            "data_path": "../dataset/dpo.jsonl",
            "max_seq_len": 1024,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 4e-8,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_weight": "full_sft",
            "from_resume": 0,
            "beta": 0.15,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "最大序列长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数", "int"),
            ("learning_rate", "学习率（建议 <=5e-8）", "float"),
            ("beta", "DPO beta 参数", "float"),
        ],
    },
    {
        "id": "ppo",
        "name": "PPO 强化学习",
        "desc": "Actor-Critic + GAE，需外部 reward model 打分",
        "script": "trainer/train_ppo.py",
        "depends_on": "full_sft",
        "output_weight": "ppo_actor",
        "defaults": {
            "data_path": "../dataset/rlaif.jsonl",
            "max_seq_len": 768,
            "max_gen_len": 1024,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 3e-7,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_weight": "full_sft",
            "from_resume": 0,
            "reward_model_path": "../../internlm2-1_8b-reward",
            "clip_epsilon": 0.2,
            "kl_coef": 0.02,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "Prompt 最大长度", "int"),
            ("max_gen_len", "生成最大长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数", "int"),
            ("learning_rate", "Actor 学习率", "float"),
            ("reward_model_path", "Reward 模型路径", "path"),
        ],
    },
    {
        "id": "grpo",
        "name": "GRPO / CISPO 强化学习",
        "desc": "组相对策略优化，无需 critic 网络",
        "script": "trainer/train_grpo.py",
        "depends_on": "full_sft",
        "output_weight": "grpo",
        "defaults": {
            "data_path": "../dataset/rlaif.jsonl",
            "max_seq_len": 768,
            "max_gen_len": 1024,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 3e-7,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_weight": "full_sft",
            "from_resume": 0,
            "reward_model_path": "../../internlm2-1_8b-reward",
            "num_generations": 6,
            "loss_type": "cispo",
            "beta": 0.1,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "Prompt 最大长度", "int"),
            ("max_gen_len", "生成最大长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数", "int"),
            ("learning_rate", "学习率", "float"),
            ("num_generations", "每个 prompt 生成数", "int"),
            ("loss_type", "Loss 类型 (grpo/cispo)", "choice"),
            ("reward_model_path", "Reward 模型路径", "path"),
        ],
    },
    {
        "id": "agent",
        "name": "Agent 工具调用 RL",
        "desc": "多轮工具调用强化学习，模拟工具环境",
        "script": "trainer/train_agent.py",
        "depends_on": "full_sft",
        "output_weight": "agent",
        "defaults": {
            "data_path": "../dataset/agent_rl.jsonl",
            "max_seq_len": 1024,
            "max_gen_len": 768,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 3e-7,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_weight": "full_sft",
            "from_resume": 0,
            "reward_model_path": "../../internlm2-1_8b-reward",
            "num_generations": 4,
            "loss_type": "cispo",
            "beta": 0.1,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "最大序列长度", "int"),
            ("max_gen_len", "单次最大生成长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数", "int"),
            ("learning_rate", "学习率", "float"),
            ("num_generations", "每个 prompt 生成数", "int"),
            ("loss_type", "Loss 类型 (grpo/cispo)", "choice"),
        ],
    },
    {
        "id": "distillation",
        "name": "知识蒸馏 (Distillation)",
        "desc": "Teacher 指导学生模型，CE + KL 混合 loss",
        "script": "trainer/train_distillation.py",
        "depends_on": "full_sft",
        "output_weight": "full_dist",
        "defaults": {
            "data_path": "../dataset/sft_t2t_mini.jsonl",
            "max_seq_len": 340,
            "batch_size": 32,
            "epochs": 6,
            "learning_rate": 5e-6,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "from_student_weight": "full_sft",
            "from_teacher_weight": "full_sft",
            "from_resume": 0,
            "alpha": 0.5,
            "temperature": 1.5,
        },
        "key_params": [
            ("data_path", "训练数据路径", "path"),
            ("max_seq_len", "最大序列长度", "int"),
            ("batch_size", "Batch size", "int"),
            ("epochs", "训练轮数", "int"),
            ("learning_rate", "学习率", "float"),
            ("alpha", "CE loss 权重 (总 loss=α*CE+(1-α)*KL)", "float"),
            ("temperature", "蒸馏温度", "float"),
        ],
    },
]

STAGE_MAP = {s["id"]: s for s in STAGES}


# ============================================================
# 工具函数
# ============================================================

def stage_completed(stage_id):
    stage = STAGE_MAP[stage_id]
    output_path = PROJECT_ROOT / "out" / f"{stage['output_weight']}_{stage['defaults'].get('hidden_size', 768)}.pth"
    if stage.get("defaults", {}).get("use_moe", 0):
        output_path = PROJECT_ROOT / "out" / f"{stage['output_weight']}_{stage['defaults'].get('hidden_size', 768)}_moe.pth"
    return output_path.exists()


def find_completed_stages():
    completed = []
    for s in STAGES:
        if stage_completed(s["id"]):
            completed.append(s["id"])
    return completed


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    clear_screen()
    title = Text("""
    ╔═══════════════════════════════════════════╗
    ║         MiniMind 训练流水线               ║
    ║    从零训练小语言模型 (~64M 参数)          ║
    ╚═══════════════════════════════════════════╝
    """, style="bold cyan")
    console.print(title)


def print_pipeline_diagram():
    table = Table(show_header=False, border_style="dim", padding=(0, 1))
    table.add_column("步骤", style="bold yellow", width=6)
    table.add_column("阶段", style="bold white", width=28)
    table.add_column("说明", style="dim white", width=50)

    for i, s in enumerate(STAGES):
        dep = s["depends_on"]
        dep_str = f"  ← 需 {dep}" if dep else "  ← 从零开始"
        status = ""
        if stage_completed(s["id"]):
            status = " ✅"
        table.add_row(f" {i+1}.", f"{s['name']}{status}", s["desc"])

    console.print(Panel(table, title="[bold cyan]推荐训练流水线顺序[/]", border_style="cyan"))
    rprint()


# ============================================================
# 参数配置
# ============================================================

def configure_stage_params(stage, defaults):
    params = dict(defaults)

    rprint(f"\n[bold yellow]▶ 配置 {stage['name']} 参数[/]")
    rprint(f"  当前工作目录: {PROJECT_ROOT}")

    for key, label, param_type in stage["key_params"]:
        default_val = params.get(key)
        if param_type == "int":
            val = IntPrompt.ask(f"  {label}", default=int(default_val))
            params[key] = val
        elif param_type == "float":
            val = FloatPrompt.ask(f"  {label}", default=float(default_val))
            params[key] = val
        elif param_type == "path":
            val = Prompt.ask(f"  {label}", default=str(default_val))
            params[key] = val
        elif param_type == "choice":
            val = Prompt.ask(f"  {label}", default=str(default_val))
            params[key] = val
        else:
            val = Prompt.ask(f"  {label}", default=str(default_val))
            params[key] = val

    if "from_weight" in params:
        override = Prompt.ask(f"  基于权重文件前缀（即 ../out/xxx_768.pth 中的 xxx）", default=str(params["from_weight"]))
        params["from_weight"] = override

    use_wandb = Confirm.ask("  使用 SwanLab/WandB 记录?", default=False)
    params["use_wandb"] = "store_true" if use_wandb else ""

    use_compile = Confirm.ask("  使用 torch.compile 加速?", default=False)
    params["use_compile"] = 1 if use_compile else 0

    return params


def build_command(stage, params):
    script_path = PROJECT_ROOT / stage["script"]
    cmd = [sys.executable, str(script_path)]

    for key, value in params.items():
        if key == "use_wandb" and value == "store_true":
            cmd.append("--use_wandb")
        elif key == "use_moe":
            cmd.extend([f"--{key}", str(int(value))])
        elif key == "use_compile":
            cmd.extend([f"--{key}", str(value)])
        elif key == "from_resume":
            cmd.extend([f"--{key}", str(value)])
        elif isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    return cmd


# ============================================================
# 训练执行器（流式输出 + 进度解析）
# ============================================================

def stream_subprocess(cmd, stage_name):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=PROJECT_ROOT,
    )

    progress_queue = Queue()
    line_queue = Queue()

    epoch_pattern = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\)")
    loss_pattern = re.compile(r"loss: ([\d.]+)")

    def reader_thread():
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip("\n")
            if not line:
                continue
            line_queue.put(line)

            m = epoch_pattern.search(line)
            if m:
                current_step = int(m.group(3))
                total_steps = int(m.group(4))
                current_epoch = int(m.group(1))
                total_epochs = int(m.group(2))
                progress_queue.put(("step", current_step, total_steps, current_epoch, total_epochs))

            m = loss_pattern.search(line)
            if m:
                progress_queue.put(("loss", float(m.group(1))))

        process.wait()
        progress_queue.put(("done", process.returncode))

    t = Thread(target=reader_thread, daemon=True)
    t.start()
    return process, progress_queue, line_queue


def run_training(stage, params):
    cmd = build_command(stage, params)
    stage_name = stage["name"]

    rprint(f"\n[bold green]▶ 开始运行: {stage_name}[/]")
    rprint(f"  命令: {' '.join(str(x) for x in cmd)}")
    rprint(f"  [dim]{'='*60}[/]\n")

    process, progress_queue, line_queue = stream_subprocess(cmd, stage_name)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    task_id = progress.add_task(f"[cyan]{stage_name}", total=100)
    current_loss = None
    display_lines = []
    max_display = 8

    with Live(progress, console=console, refresh_per_second=10) as live:
        while True:
            try:
                item = progress_queue.get(timeout=0.1)
            except Empty:
                continue

            if item[0] == "step":
                _, current_step, total_steps, current_epoch, total_epochs = item
                pct = (current_step / total_steps) * 100
                pct = min(pct, 100)
                total_pct = ((current_epoch - 1) / total_epochs) * 100 + pct / total_epochs
                progress.update(task_id, completed=total_pct,
                                description=f"[cyan]Epoch {current_epoch}/{total_epochs} Step {current_step}/{total_steps}")
            elif item[0] == "loss":
                current_loss = item[1]
            elif item[0] == "done":
                returncode = item[1]
                if returncode == 0:
                    progress.update(task_id, completed=100, description=f"[green]✔ {stage_name} 完成!")
                    time.sleep(1)
                else:
                    progress.update(task_id, description=f"[red]✘ {stage_name} 失败 (code={returncode})")
                    time.sleep(1)
                return returncode == 0

            while not line_queue.empty():
                line = line_queue.get_nowait()
                display_lines.append(line)
                if len(display_lines) > max_display:
                    display_lines = display_lines[-max_display:]


# ============================================================
# 主流程
# ============================================================

def main():
    print_header()
    print_pipeline_diagram()

    completed_stages = find_completed_stages()
    if completed_stages:
        rprint(f"[bold green]已检测到已完成的阶段: {', '.join(completed_stages)}[/]\n")
    else:
        rprint("[dim]未检测到已保存的权重文件，从头开始。[/]\n")

    rprint("\n[bold]请选择要运行的阶段[/] [dim](输入编号，逗号分隔；输入 'all' 运行全部)[/]")
    rprint("  [dim]提示: 依赖前置阶段的阶段会自动跳过（若前置未完成）[/]")

    for i, s in enumerate(STAGES):
        status = ""
        if stage_completed(s["id"]):
            status = " ✅"
        needs = f"  (需: {s['depends_on']})" if s["depends_on"] else ""
        rprint(f"  [bold]{i+1}.[/] {s['name']}{status}{' [dim]'+needs+'[/]' if needs else ''}")

    raw = Prompt.ask("\n[bold yellow]输入选择[/]", default="1,2")
    if raw.strip().lower() == "all":
        selected_ids = [s["id"] for s in STAGES]
    else:
        indices = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(STAGES):
                    indices.append(idx)
        selected_ids = [STAGES[i]["id"] for i in indices]

    if not selected_ids:
        rprint("[red]未选择任何阶段，退出。[/]")
        return

    rprint(f"\n[bold]选中的阶段: {', '.join(selected_ids)}[/]")

    use_global_params = Confirm.ask("\n对所有选中阶段使用相同的基础参数?（hidden_size=768, num_hidden_layers=8, use_moe=0）", default=True)

    global_hidden = 768
    global_layers = 8
    global_moe = 0
    if use_global_params:
        global_hidden = IntPrompt.ask("  hidden_size (模型维度)", default=768)
        global_layers = IntPrompt.ask("  num_hidden_layers (层数)", default=8)
        global_moe = Confirm.ask("  使用 MoE 架构?", default=False)

    stages_to_run = []
    prerequisites_met = True

    for sid in selected_ids:
        stage = STAGE_MAP[sid]
        dep = stage["depends_on"]

        if dep and dep not in selected_ids and not stage_completed(dep):
            rprint(f"[yellow]⚠ 前置阶段 '{dep}' 未选择且未完成，跳过 {stage['name']}[/]")
            continue

        if stage_completed(sid):
            redo = Confirm.ask(f"[yellow]⚠ {stage['name']} 已有完成输出，是否重新训练?[/]", default=False)
            if not redo:
                continue

        params = dict(stage["defaults"])
        if use_global_params:
            params["hidden_size"] = global_hidden
            params["num_hidden_layers"] = global_layers
            params["use_moe"] = 1 if global_moe else 0

        params = configure_stage_params(stage, params)
        stages_to_run.append((stage, params))

    if not stages_to_run:
        rprint("[red]没有需要运行的阶段。[/]")
        return

    rprint(f"\n[bold]准备运行 {len(stages_to_run)} 个阶段...[/]")

    success_count = 0
    for stage, params in stages_to_run:
        ok = run_training(stage, params)
        if ok:
            success_count += 1
            rprint(f"[green]✔ {stage['name']} 运行成功[/]")
        else:
            rprint(f"[red]✘ {stage['name']} 运行失败[/]")
            cont = Confirm.ask("是否继续后续阶段?", default=False)
            if not cont:
                break

    rprint(f"\n[bold]{'='*50}[/]")
    rprint(f"[bold]流水线运行完毕: {success_count}/{len(stages_to_run)} 阶段成功[/]")

    if success_count == len(stages_to_run):
        rprint("\n[green]所有阶段完成![/]")
        rprint("[dim]可用以下命令测试模型:[/]")
        rprint("  python eval_llm.py")
        rprint("  python scripts/web_demo.py")
        rprint("  python scripts/serve_openai_api.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]用户中断，退出。[/]")
        sys.exit(0)
