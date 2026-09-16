# -*- coding: utf-8 -*-
"""
探针 3：为什么 19 道题「什么都不做」能拿满分？

verify_oracle.py 的 C 组测出 19/50 道题 noop 得分 1.0，其中甚至有 8 条 gold 动作的题。
猜测：task.actions 里混了**只读动作**（get_*/search_*/calculate），
      这些动作不改变数据库 → 重放 gold 与「什么都不做」的最终数据库状态相同 → 哈希相同 → 满分。

本脚本把这些题的 gold 动作按读/写拆开算，并直接测「重放 gold 到底改没改数据库」。

跑法：
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/analyze_hackable.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import build_env, TASKS                      # noqa: E402
from env.bootstrap import TAU_BENCH                   # noqa: F401,E402
from tau_bench.types import RESPOND_ACTION_NAME       # noqa: E402

# 只读工具：不改 data["reservations"] / data["users"]
READ_ONLY = {
    "get_reservation_details",
    "get_user_details",
    "search_direct_flight",
    "search_onestop_flight",
    "list_all_airports",
    "calculate",
    "think",
}

N = len(TASKS)
rows = []

print("=" * 78)
print("探针 3 · reward 可 hack 面归因")
print("=" * 78)
print()

print("逐题测量（重放 gold 后数据库哈希是否变化）...")
for i, task in enumerate(TASKS):
    env = build_env(i)
    env.reset(task_index=i)
    h_fresh = env.get_data_hash()

    for action in task.actions:
        env.step(action)
    h_after = env.get_data_hash()

    names = [a.name for a in task.actions if a.name != RESPOND_ACTION_NAME]
    n_read = sum(1 for n in names if n in READ_ONLY)
    n_write = len(names) - n_read

    rows.append(
        {
            "idx": i,
            "user_id": task.user_id,
            "n_actions": len(names),
            "n_read": n_read,
            "n_write": n_write,
            "n_outputs": len(task.outputs),
            "db_changed": h_fresh != h_after,
            "nonempty_actions": [n for n in names if n not in READ_ONLY],
        }
    )

print("完成。")
print()

# ---------------------------------------------------------------- 汇总

neutral = [r for r in rows if not r["db_changed"]]
changed = [r for r in rows if r["db_changed"]]

print("-" * 78)
print("① 重放 gold 后数据库是否真的变了")
print("-" * 78)
print(f"   变了（有区分度）    ：{len(changed):>2} / {N}")
print(f"   没变（无区分度）    ：{len(neutral):>2} / {N}   ← 这些题「什么都不做」= 满分")
print()

# ---------------------------------------------------------------- 无区分度题的构成

print("-" * 78)
print("② 无区分度的 {} 道题，gold 里都是些什么动作".format(len(neutral)))
print("-" * 78)
print(f"{'idx':>4} {'user_id':<26} {'总动作':>6} {'只读':>5} {'写':>4}  动作构成")
print("-" * 78)
for r in neutral:
    comp = dict(Counter(r["nonempty_actions"])) if r["nonempty_actions"] else "（全只读）"
    print(f"{r['idx']:>4} {r['user_id']:<26} {r['n_actions']:>6} "
          f"{r['n_read']:>5} {r['n_write']:>4}  {comp}")
print()

# ---------------------------------------------------------------- 有区分度题的构成

print("-" * 78)
print("③ 有区分度的题，读写构成")
print("-" * 78)
all_read = sum(r["n_read"] for r in rows)
all_write = sum(r["n_write"] for r in rows)
print(f"   全 50 题 gold 动作合计：只读 {all_read} 条 / 写 {all_write} 条 "
      f"（只读占 {all_read/(all_read+all_write)*100:.0f}%）")
print()

# 交叉表：写动作数 vs 有无区分度
print("   写动作数 → 是否无区分度：")
ct = Counter()
for r in rows:
    ct[(r["n_write"], not r["db_changed"])] += 1
for w in sorted({k[0] for k in ct}):
    n_neutral = ct.get((w, True), 0)
    n_changed = ct.get((w, False), 0)
    print(f"     写动作 {w:>2} 条：有区分度 {n_changed:>2} 题，无区分度 {n_neutral:>2} 题")
print()

# ---------------------------------------------------------------- 结论

print("-" * 78)
print("④ 判读")
print("-" * 78)
zero_write = [r for r in neutral if r["n_write"] == 0]
nonzero_write = [r for r in neutral if r["n_write"] > 0]

print(f"   无区分度题中，写动作数 = 0 的：{len(zero_write)} 道"
      f" → 这些题的 gold 全是只读动作，本来就改不动数据库，可以理解")
print(f"   无区分度题中，写动作数 > 0 的：{len(nonzero_write)} 道"
      f" → ⚠️ 这些更可疑，写了却没改成功")
if nonzero_write:
    print()
    for r in nonzero_write:
        print(f"      task[{r['idx']:>2}] {r['user_id']:<26} "
              f"写动作 {r['n_write']} 条：{r['nonempty_actions']}")
print()

# ---------------------------------------------------------------- 存盘

out = PROJECT / "data" / "hackable_analysis.json"
out.write_text(
    json.dumps(
        {
            "read_only_tools": sorted(READ_ONLY),
            "n_tasks": N,
            "n_db_changed": len(changed),
            "n_db_neutral": len(neutral),
            "neutral_task_indices": [r["idx"] for r in neutral],
            "changed_task_indices": [r["idx"] for r in changed],
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(f"结果已存：{out}")
print()
print("=" * 78)
