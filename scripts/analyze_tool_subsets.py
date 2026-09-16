# -*- coding: utf-8 -*-
"""
探针 4：不同工具子集下，还有多少题是「有区分度」的？

verify_oracle.py 的 D 组测出一个反直觉的事实：
  砍掉 cancel_reservation 后，reward 通过率**一点没降**（仍然 46/50）。

原因是 Env.calculate_reward()（base.py:125-165）并不是拿 agent 的结果去比对
「标准的数据库终态」，而是**用同一套 tools_map 把 gold 重放一遍**再比哈希。
工具被砍掉后，agent 调它失败、gold 重放它也失败 —— 两边同样 no-op，哈希照样相等。

→ 所以砍工具的代价不是「reward 归零」，而是**题目失去区分度**：
  不管 policy 做什么，数据库都不变，reward 恒为 1.0。

这个脚本把「失去区分度」量化出来，用来定工具子集。

跑法：
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/analyze_tool_subsets.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import build_env, TASKS, ALL_TOOL_NAMES   # noqa: E402
from env.bootstrap import TAU_BENCH                 # noqa: F401,E402
from tau_bench.types import RESPOND_ACTION_NAME     # noqa: E402

N = len(TASKS)

# 候选方案
CONFIGS = {
    "全量 14 个（基准）": sorted(ALL_TOOL_NAMES),
    "砍 think + list_all_airports（12）": sorted(ALL_TOOL_NAMES - {"think", "list_all_airports"}),
    "README 的 14→7 方案": sorted({
        "get_user_details", "get_reservation_details", "search_direct_flight",
        "search_onestop_flight", "update_reservation_flights",
        "update_reservation_baggages", "calculate",
    }),
    "只保留 gold 里出现过的（11）": sorted({
        "book_reservation", "calculate", "cancel_reservation",
        "get_reservation_details", "get_user_details", "search_direct_flight",
        "send_certificate", "transfer_to_human_agents",
        "update_reservation_baggages", "update_reservation_flights",
        "update_reservation_passengers",
    }),
}


def measure(tool_names):
    """返回 (有区分度的题数, noop 能拿满分的题数, 逐题明细)"""
    n_disc = 0
    n_noop = 0
    detail = []
    for i, task in enumerate(TASKS):
        # ① 重放 gold 后数据库有没有变
        env = build_env(i, tool_names=tool_names)
        env.reset(task_index=i)
        h_fresh = env.get_data_hash()
        for a in task.actions:
            env.step(a)
        changed = env.get_data_hash() != h_fresh

        # ② 什么都不做能不能拿满分
        env2 = build_env(i, tool_names=tool_names)
        env2.reset(task_index=i)
        noop_reward = env2.calculate_reward().reward

        n_disc += int(changed)
        n_noop += int(noop_reward == 1.0)
        detail.append({"idx": i, "db_changed": changed, "noop": noop_reward})
    return n_disc, n_noop, detail


print("=" * 78)
print("探针 4 · 工具子集 → 题目区分度")
print("=" * 78)
print(f"共 {N} 题。理想情况：有区分度 = {N}，noop 满分 = 0。")
print()

summary = {}
for name, tools in CONFIGS.items():
    print(f"跑：{name} （{len(tools)} 个工具）...")
    n_disc, n_noop, detail = measure(tools)
    summary[name] = {
        "n_tools": len(tools),
        "n_discriminative": n_disc,
        "n_noop_pass": n_noop,
        "tools": tools,
        "detail": detail,
    }

print()
print("-" * 78)
print(f"{'方案':<36}{'工具数':>6}{'有区分度':>10}{'noop满分':>10}{'废题率':>9}")
print("-" * 78)
base = summary["全量 14 个（基准）"]["n_discriminative"]
for name, s in summary.items():
    waste = (N - s["n_discriminative"]) / N * 100
    mark = ""
    if s["n_discriminative"] < base:
        mark = f"  ← 比基准少 {base - s['n_discriminative']} 题"
    print(f"{name:<36}{s['n_tools']:>6}{s['n_discriminative']:>10}"
          f"{s['n_noop_pass']:>10}{waste:>8.0f}%{mark}")
print("-" * 78)
print()

# 各方案下变成废题的题号
base_detail = {d["idx"]: d["db_changed"] for d in summary["全量 14 个（基准）"]["detail"]}
for name, s in summary.items():
    if name.startswith("全量"):
        continue
    lost = [d["idx"] for d in s["detail"]
            if base_detail.get(d["idx"]) and not d["db_changed"]]
    if lost:
        print(f"「{name}」新变成无区分度的题（{len(lost)} 道）：{lost}")
print()

out = PROJECT / "data" / "tool_subsets.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"结果已存：{out}")
print()
print("=" * 78)
