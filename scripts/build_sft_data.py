# -*- coding: utf-8 -*-
"""
H2-a · 造 SFT 数据：能复用的直接复用，缺的才合成。

背景（为什么要这一步）：
    gold action 序列里 **没有 respond 动作**（实测：158 个动作里出现 0 次）。
    直接重放 gold 得到的轨迹，结尾是空的 → 模型学不到「什么时候该收尾」。

发现的出路：
    τ-bench 自带 **268 条真实成功轨迹**（gpt-4o 84 + sonnet 184），
    它们本身就已经是「完整对话 + 收尾回复」——**开箱即用，几乎不用转换**。

本脚本做的事：
    1. 从 268 条里筛出【训练集 31 道】的部分        → sft_seed.jsonl
    2. 报告训练集里【没有真实轨迹】的题（要另外补）  → 打印出来

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 D:/anaconda/python.exe scripts/build_sft_data.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TRAJ_DIR = (PROJECT.parent / "reference-repos" / "agentic-grpo-longhorizon"
            / "tau-bench" / "historical_trajectories")
SPLIT = PROJECT / "data" / "task_split.json"
OUT = PROJECT / "data" / "sft_seed.jsonl"

FILES = ["gpt-4o-airline.json", "sonnet-35-new-airline.json"]


def main() -> int:
    print("=" * 76)
    print("H2-a · 造 SFT 数据（复用真实轨迹）")
    print("=" * 76)

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_ids = {r["task"] for r in split["train"]}
    with_outputs = {t: None for t in split.get("tasks_with_outputs", [])}

    rows, per_task = [], defaultdict(list)
    for fn in FILES:
        src = fn.replace("-airline.json", "")
        for x in json.loads((TRAJ_DIR / fn).read_text(encoding="utf-8")):
            if x.get("reward") != 1.0:
                continue
            tid = int(x["task_id"])
            if tid not in train_ids:            # 探针集的轨迹不用于训练
                continue
            msgs = x["traj"]

            # --- 完整性检查：开头必须是 system，结尾必须有一条非空 assistant
            assert msgs[0]["role"] == "system", f"task{tid}: 首条不是 system"
            finals = [m for m in msgs if m["role"] == "assistant"
                      and (m.get("content") or "").strip()]
            if not finals:
                continue                         # 没有收尾回复的，跳过
            last = msgs[-1]
            ok_end = last["role"] == "assistant" or last["role"] == "tool"

            rows.append({
                "task_id": tid,
                "source": src,
                "n_messages": len(msgs),
                "n_assistant": sum(1 for m in msgs if m["role"] == "assistant"),
                "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs),
                "final_resp_chars": len(finals[-1].get("content") or ""),
                "messages": msgs,
            })
            per_task[tid].append(src)

    # ---------------------------------------------------------------- 报告
    print(f"训练集 {len(train_ids)} 道 ｜ 可用真实轨迹 {len(rows)} 条"
          f" ｜ 覆盖 {len(per_task)} 道\n")

    print("-" * 76)
    print("每道题有几条可用轨迹")
    print("-" * 76)
    n1 = sum(1 for v in per_task.values() if len(v) == 1)
    multi = sorted(((t, len(v)) for t, v in per_task.items()
                    if len(v) > 1), key=lambda kv: -kv[1])
    print(f"   只有 1 条 : {n1} 道")
    print(f"   有多条    : {len(multi)} 道（最多 {multi[0][1] if multi else 0} 条）")

    missing = sorted(train_ids - set(per_task))
    print()
    print("-" * 76)
    print(f"⚠️  训练集里【没有真实轨迹】的 {len(missing)} 道 → 需要另外补")
    print("-" * 76)
    print(f"   task {missing}")

    # ---------------------------------------------------------------- 统计
    lens = [r["final_resp_chars"] for r in rows]
    print()
    print("-" * 76)
    print("收尾回复长度")
    print("-" * 76)
    print(f"   中位 {sorted(lens)[len(lens)//2]} 字符 ｜ "
          f"范围 {min(lens)} – {max(lens)}")
    by_src = Counter(r["source"] for r in rows)
    print(f"   来源分布: {dict(by_src)}")

    # ---------------------------------------------------------------- 存盘
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=" * 76)
    print(f"✅ 写出 {len(rows)} 条 → {OUT}")
    print(f"   还差 {len(missing)} 道题的轨迹要合成（下一步）")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
