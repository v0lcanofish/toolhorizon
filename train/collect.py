# -*- coding: utf-8 -*-
"""
上卡第一段：**采样**（起 vLLM → 收数据 → 退出，显存还给训练）。

    python -m train.collect --select train --n 8 --out data/rollouts/step_000.jsonl

为什么采样和训练是两个进程、用文件交接：
    单卡 24 GB 上，vLLM 和训练**不能同时驻留**（峰值会撞在一起）。
    本项目的做法是分时复用：采样跑完就退出，训练再上。
    用文件交接比同进程 colocate 简单得多，而且**不可能踩**那个经典坑 ——
    "vLLM 进程没真死 → 训练启动 OOM，但 nvidia-smi 看着是空的"。
    ⭐ 顺带好处：每条轨迹的原始对话都落盘了，观测指标可以**离线重算**，
       不用为了补一个指标再租一次卡。

选哪些题（`--select`）—— E6 实验（S/O 配比）就靠这个开关：
    train         31 道 S 类（有区分度：做对给分、不做给 0）
    probe         19 道 O 类（单边信号：不做给满分、乱做给 0）
    arm_none      31 + 0        ← E6 三臂之一
    arm_base      31 + 19       ← 现状
    arm_double    31 + 19×2     ← 惩罚样本加倍，看"欠调用"会不会更严重
    exp_easy / exp_medium      扩题集的简单档 / 困难档

⚠️ 跑这个之前，先跑 `scripts/gpu_smoke.py` —— 30 分钟确认环境和速度，
   不要一上来就采几百条然后发现 20 分钟才跑完一批。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT = Path(__file__).resolve().parents[1]

TASK_SPLIT = _PROJECT / "data" / "task_split.json"
EXPANDED = _PROJECT / "data" / "tasks_expanded.json"


# ---------------------------------------------------------------- 选题


def select_tasks(which: str, limit: int = 0) -> List[Tuple[int, Any]]:
    """按实验需要挑题，返回 [(task_id, task), ...]。"""
    from env import TASKS

    split = json.loads(TASK_SPLIT.read_text(encoding="utf-8"))
    train_ids = [r["task"] for r in split["train"]]
    probe_ids = [r["task"] for r in split["probe_overcall"]]

    if which == "train":
        ids = train_ids
    elif which == "probe":
        ids = probe_ids
    elif which == "arm_none":
        ids = train_ids
    elif which == "arm_base":
        ids = train_ids + probe_ids
    elif which == "arm_double":
        # ⭐ E6 第三臂：把"惩罚型样本"加倍。
        #    假设是"惩罚信号来得比奖励信号早且密 → 模型变保守 → 欠调用"，
        #    那么把 O 类题翻倍，欠调用应当**更严重**。
        ids = train_ids + probe_ids + probe_ids
    elif which in ("exp_easy", "exp_medium"):
        return _expanded("easy" if which == "exp_easy" else "medium", limit)
    else:
        raise ValueError(f"未知的 --select：{which}")

    items = [(i, TASKS[i]) for i in ids]
    if limit:
        items = items[:limit]
    return items


def _expanded(difficulty: str, limit: int) -> List[Tuple[int, Any]]:
    from tau_bench.types import Action, Task

    exp = json.loads(EXPANDED.read_text(encoding="utf-8"))
    out = []
    for j, raw in enumerate(exp["tasks"]):
        if raw.get("difficulty") != difficulty:
            continue
        t = Task(
            user_id=raw["user_id"],
            actions=[Action(name=a["name"], kwargs=a["kwargs"]) for a in raw["actions"]],
            outputs=raw.get("outputs", []),
            instruction=raw["instruction"],
        )
        out.append((100000 + j, t))          # 用大编号和原题隔开，避免 task_id 撞车
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- 主流程


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录；空 = 用基座")
    ap.add_argument("--select", default="train")
    ap.add_argument("--n", type=int, default=8, help="每题采样条数（GRPO 的 group size）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 道题（0 = 全部）")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from train.rollout_batch import RolloutConfig, run_grouped_rollout, save_rollouts
    from train.tokenize import load_tokenizer, tool_schemas

    # 🔴 tokenizer 必须和模型同源（本地那份词表 151,665 ≠ Qwen2.5 的 151,936）
    tok = load_tokenizer(model_path=args.model)
    tools = tool_schemas()
    items = select_tasks(args.select, args.limit)
    n_traj = len(items) * args.n
    print(f"[collect] 选题 {args.select}：{len(items)} 道 × {args.n} 条 = {n_traj} 条轨迹")
    print(f"[collect] adapter：{args.adapter or '（无，用基座）'}")

    t0 = time.time()
    if args.engine == "vllm":
        from train.engine import VLLMEngine
        engine = VLLMEngine(args.model, tok,
                            lora_path=args.adapter or None,
                            gpu_memory_utilization=args.gpu_mem,
                            max_model_len=args.max_seq_tokens)
    else:
        from transformers import AutoModelForCausalLM
        from train.engine import HFEngine
        from train.modeling import load_model
        m = load_model(args.model, dtype="bfloat16", device_map="auto")
        if args.adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, args.adapter)
        engine = HFEngine(m, tok, device="cuda")
    print(f"[collect] 引擎就绪，用时 {time.time() - t0:.1f}s")

    cfg = RolloutConfig(
        n_group=args.n, max_turns=args.max_turns,
        max_seq_tokens=args.max_seq_tokens,
    )
    t1 = time.time()
    batch = run_grouped_rollout(items, engine, cfg, tok, tools)
    dt = time.time() - t1

    n = save_rollouts(batch, args.out, step=0)
    s = batch.summary()
    stats = engine.stats()
    summary = {
        "select": args.select, "n_tasks": len(items), "n_group": args.n,
        "n_saved": n, "adapter": args.adapter, "engine": args.engine,
        "seconds": dt, "sec_per_trajectory": dt / max(1, n),
        "new_tokens": stats.total_new_tokens,
        "tok_per_sec": stats.total_new_tokens / max(1e-9, dt),
        **s,
    }
    Path(args.out).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[collect] 采完 {n} 条，用时 {dt:.1f}s（{dt/max(1,n):.2f}s/条，"
          f"{summary['tok_per_sec']:.0f} tok/s）")
    print(f"[collect] ⭐ 组内零方差率 {s['zero_var_rate']:.3f}"
          f"（>0.8 说明这批题对当前策略太难或太易，这一步训练基本白跑）")
    print(f"[collect] reward 均值 {s['reward_mean']:.3f} ｜ 通过率 {s['pass_rate']:.3f}")
    print(f"[collect] 平均轮数 {s['turns_mean']:.2f} ｜ 工具调用 {s['tool_calls_mean']:.2f}")
    print(f"[collect] 格式崩 {s['malformed_rate']:.3f} ｜ 截断 {s['truncated_rate']:.3f}")
    print(f"[collect] 终止方式 {s.get('terminated_by', {})}")
    print(f"[collect] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
