# -*- coding: utf-8 -*-
"""
训练主循环 —— 采样 ↔ 训练 **交替跑**，这就是"单卡分时复用"的落地。

    python -m train.run --rounds 20 --select arm_base --n 8

每一轮干两件事，各起一个进程：

    ① python -m train.collect   →  起 vLLM，采一批轨迹，写成 jsonl，**退出**（显存还回去）
    ② python -m train.trainer   →  读 jsonl，更新 LoRA，存成新 adapter
    回到 ①，vLLM 重启时挂上**新的 adapter**

为什么用子进程而不是一个进程里来回切：
    vLLM 和训练同时在显存里峰值会撞车。两个进程各自进出，
    显存天然不会重叠 —— 代价是每轮 vLLM 启动 20~40 秒，
    摊到一批几百条轨迹上不到 5%。

⚠️ 每轮结束都会检查**组内零方差率**并打印。
   本项目的第一监控指标不是 loss，是它 ——
   一旦连续几轮都 > 0.8，说明这批题对当前策略"全会"或"全不会"，
   **训练其实是空转的**，loss 曲线看不出来，但攒出来的 adapter 没进步。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]


def sh(cmd, log_path: Path) -> int:
    """跑一个子进程，输出同时进终端和日志文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace")
        f.write(p.stdout)
        print(p.stdout, end="")
    return p.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20, help="采样↔训练交替多少轮")
    ap.add_argument("--select", default="arm_base")
    ap.add_argument("--n", type=int, default=8, help="每题采样条数")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--init-adapter", default="", help="起始 adapter（SFT 产物）；空 = 从基座开始")
    ap.add_argument("--run-name", default="run0")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    # ⚠️ 必须在这里也加一份并**透传给 trainer** —— 否则光在 trainer 里加参数，
    #    从 train.run 跑的时候会**静默走默认值**（命令里写 lata 也没用），
    #    而且不报任何错。本项目反复踩的就是这个形状的坑（2026-09-18 差点又踩一次）。
    ap.add_argument("--length-norm", default="mean", choices=["mean", "lata"],
                    help="GRPO 长度归一化：mean=÷L（默认）｜ lata=÷√L（参考实现 0.125→0.185）")
    # ⭐ 四臂消融的另一根轴。同样必须**在这里加一份并透传给 collect** ——
    #    只在 collect 里加参数的话，从 train.run 跑的时候写 --ds 也没用、还不报错。
    ap.add_argument("--ds", type=int, default=0,
                    help="动态采样：加采直到攒够 K 个有方差的组（0=关，=基线行为）")
    ap.add_argument("--ds-max-pass", type=int, default=6)
    args = ap.parse_args(argv)

    run_dir = _PROJECT / "runs" / args.run_name
    ckpt_dir = run_dir / "ckpt"
    roll_dir = run_dir / "rollouts"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[run] {run_dir}")
    print(f"[run] {args.rounds} 轮 ｜ 选题 {args.select} ｜ 每题 {args.n} 条")

    py = sys.executable
    adapter = args.init_adapter
    zero_var_history = []

    for r in range(args.rounds):
        tag = f"{r:03d}"
        roll = roll_dir / f"step_{tag}.jsonl"
        ckpt_out = ckpt_dir / f"ckpt_{tag}"

        print(f"\n{'='*70}\n[run] 第 {r} 轮 —— 采样\n{'='*70}")
        cmd = [py, "-m", "train.collect",
               "--model", args.model, "--select", args.select,
               "--n", str(args.n), "--engine", args.engine,
               "--max-seq-tokens", str(args.max_seq_tokens),
               "--out", str(roll)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.ds:
            cmd += ["--ds", str(args.ds), "--ds-max-pass", str(args.ds_max_pass)]
        if adapter:
            cmd += ["--adapter", str(adapter)]
        if sh(cmd, log_dir / f"collect_{tag}.log") != 0:
            print(f"[run] ❌ 第 {r} 轮采样失败，日志 {log_dir / f'collect_{tag}.log'}")
            return 1

        # 读一眼这一轮的采样体检
        summ = roll.with_suffix(".summary.json")
        if summ.exists():
            s = json.loads(summ.read_text(encoding="utf-8"))
            zero_var_history.append(s.get("zero_var_rate", 0.0))

        print(f"\n{'='*70}\n[run] 第 {r} 轮 —— 训练\n{'='*70}")
        cmd = [py, "-m", "train.trainer", "grpo",
               "--model", args.model, "--rollouts", str(roll),
               "--out", str(ckpt_out),
               "--length-norm", args.length_norm,
               # ⚠️ 必须把采样用的上限原样传给训练 —— 传漏了长轨迹会在编码时被静默截断
               "--max-seq-tokens", str(args.max_seq_tokens)]
        if adapter:
            cmd += ["--adapter-in", str(adapter)]
        if sh(cmd, log_dir / f"train_{tag}.log") != 0:
            print(f"[run] ❌ 第 {r} 轮训练失败，日志 {log_dir / f'train_{tag}.log'}")
            return 1
        adapter = ckpt_out

        if len(zero_var_history) >= 3 and all(z >= 0.8 for z in zero_var_history[-3:]):
            print(f"\n[run] ⚠️ 连续 3 轮组内零方差率 ≥ 0.8（{[round(z,2) for z in zero_var_history[-3:]]}）")
            print("[run]    这批题对当前策略不是全对就是全错 —— **训练在空转**，loss 看不出来。")
            print("[run]    对策：换难度档（--select exp_medium）、或调 n，别硬跑到底。")

    (run_dir / "done.json").write_text(json.dumps({
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_adapter": str(adapter),
        "zero_var_history": zero_var_history,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[run] ✅ 跑完 {args.rounds} 轮")
    print(f"[run] 最终 adapter：{adapter}")
    print(f"[run] 零方差率历史：{[round(z, 3) for z in zero_var_history]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
