# -*- coding: utf-8 -*-
"""
每道题的「奖励信号形状」探针 —— 零方差来源的第一层分解。

要回答的问题：
  「什么都不做能拿满分」的题，和「乱做也能拿满分」的题，
   是不是同一回事？ —— 【不是】，处方完全相反。

四路探针（全部零模型、零 GPU，只跑环境）：
  P0  noop      什么都不做
  P1  gold      按 gold 动作重放（+合成 respond）
  P2  foreign   注入【别的题】的 gold 写动作（模拟「过调用」）
  P3  gold_only 只重放 gold 里的只读动作，不做写（模拟「欠调用」）

分类（三选一，互斥）：
  N  null      noop=1  且 foreign=1   →  怎么都能过：真·无信号
  O  one-sided noop=1  且 foreign=0   →  只惩罚过调用：单边信号 ★
  S  signal    noop=0  且 gold=1      →  有区分度：正常训练题
  X  其他                              →  需要人工看

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/probe_signal_shape.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import build_env, TASKS                                              # noqa: E402
from env.bootstrap import TAU_BENCH                                           # noqa: F401,E402
from tau_bench.types import Action, RESPOND_ACTION_NAME                       # noqa: E402

WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
    "send_certificate",
}
N = len(TASKS)


# ---------------------------------------------------------------- 原语


def run(task_index: int, actions) -> float | str:
    """在原子上跑一串动作，返回 reward；动作非法返回 'ERR:<类型>'。"""
    env = build_env(task_index)
    env.reset(task_index=task_index)
    for a in actions:
        try:
            env.step(a)
        except Exception as e:                     # noqa: BLE001
            return f"ERR:{type(e).__name__}"
    return env.calculate_reward().reward


def gold_actions(i, *, with_respond=True):
    t = TASKS[i]
    acts = list(t.actions)
    if with_respond and t.outputs:
        acts.append(Action(name=RESPOND_ACTION_NAME,
                           kwargs={"content": " ".join(t.outputs)}))
    return acts


def readonly_actions(i):
    """gold 里去掉全部写动作（模拟欠调用）。"""
    return [a for a in TASKS[i].actions if a.name not in WRITE_TOOLS]


def write_actions_pool():
    """全部题目的 gold 写动作池，用于构造 foreign 探针。"""
    pool = []
    for i, t in enumerate(TASKS):
        for a in t.actions:
            if a.name in WRITE_TOOLS:
                pool.append((i, a))
    return pool


def pick_foreign(i, pool, k=3):
    """给第 i 题挑 k 个【别的题】的写动作。"""
    out = []
    step = max(1, len(pool) // 7)
    j = (i * step + 3) % len(pool)
    while len(out) < k and len(out) < len(pool):
        src, a = pool[j % len(pool)]
        if src != i:
            out.append(a)
        j += 1
    return out


# ---------------------------------------------------------------- 主流程


def main() -> int:
    print("=" * 78)
    print("每道题的奖励信号形状 · 四路探针（零模型 / 零 GPU）")
    print("=" * 78)

    pool = write_actions_pool()
    print(f"题目 {N} 道 ｜ gold 写动作池 {len(pool)} 条\n")

    rows = []
    for i in range(N):
        t = TASKS[i]
        n_write = sum(1 for a in t.actions if a.name in WRITE_TOOLS)

        r_noop = run(i, [])
        r_gold = run(i, gold_actions(i))
        r_ro = run(i, readonly_actions(i))

        # foreign：挑别题的写动作，取「最能破坏」的结果（min）
        fr = [run(i, [a]) for a in pick_foreign(i, pool, k=3)]
        num = [x for x in fr if isinstance(x, (int, float))]
        r_foreign = (min(num) if num else "ERR")

        # ---- 分类
        if r_noop == 1.0 and r_foreign == 1.0:
            cls = "N"          # 怎么都能过 → 真·无信号
        elif r_noop == 1.0 and r_foreign == 0.0:
            cls = "O"          # 只惩罚过调用 → 单边信号
        elif r_noop == 0.0 and r_gold == 1.0:
            cls = "S"          # 有区分度
        else:
            cls = "X"

        rows.append({
            "task": i, "user_id": t.user_id, "n_actions": len(t.actions),
            "n_write": n_write, "n_outputs": len(t.outputs),
            "r_noop": r_noop, "r_gold": r_gold, "r_readonly": r_ro,
            "r_foreign": r_foreign, "foreign_probe": fr, "class": cls,
        })

    # ------------------------------------------------------------ 汇总
    from collections import Counter
    cnt = Counter(r["class"] for r in rows)
    print("-" * 78)
    print("分类结果")
    print("-" * 78)
    name = {"S": "S 有区分度（正常训练题）",
            "O": "O 单边信号（只惩罚过调用）★",
            "N": "N 真·无信号（怎么都能过）",
            "X": "X 未分类（需人工看）"}
    for k in "SONX":
        if cnt[k]:
            print(f"  {name[k]:<32} {cnt[k]:>3} / {N}  ({cnt[k]/N*100:5.1f}%)")
    print()

    print("-" * 78)
    print("逐题明细（按类分组）")
    print("-" * 78)
    print(f"{'cls':<4}{'task':>5}{'写动作':>8}{'outputs':>9}"
          f"{'noop':>7}{'gold':>7}{'只读':>7}{'foreign':>9}")
    for k in "SONX":
        for r in rows:
            if r["class"] != k:
                continue
            print(f"{r['class']:<4}{r['task']:>5}{r['n_write']:>8}{r['n_outputs']:>9}"
                  f"{str(r['r_noop']):>7}{str(r['r_gold']):>7}"
                  f"{str(r['r_readonly']):>7}{str(r['r_foreign']):>9}")

    # ------------------------------------------------------------ 存盘
    out = PROJECT / "data" / "signal_shape.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_tasks": N,
        "counts": dict(cnt),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"逐题结果已存：{out}")
    print()
    print("=" * 78)
    print("读法：")
    print("  O 类（单边信号）是【资产】——它是「过调用」的天然探针，")
    print("  训练里剔除、评测里保留。")
    print("  N 类（真·无信号）才是纯毒药，连探针价值都没有。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
