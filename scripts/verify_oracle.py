# -*- coding: utf-8 -*-
"""
Stage 0 出口判据之一：gold 重放 oracle 必须得 reward = 1.0。

同时测另外三件事，它们决定后面的设计：
  C. 「什么都不做」能不能拿分        → reward hack 面
  D. 砍掉 gold 里用到的工具会怎样    → 验证 README 的 14→7 方案
  E. 用工具子集跑 oracle 是否等价    → 确认子集化本身不破坏奖励

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/verify_oracle.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import build_env, TASKS, GOLD_TOOLS, SAFE_TO_CUT, ALL_TOOL_NAMES   # noqa: E402
from env.bootstrap import TAU_BENCH                                          # noqa: F401,E402
from tau_bench.types import Action, RESPOND_ACTION_NAME                      # noqa: E402

N = len(TASKS)


# ---------------------------------------------------------------- 单题跑法


def run(task_index: int, *, mode: str, tool_names=None) -> float:
    """
    mode:
      "replay"    —— 按 gold 动作序列逐条重放
      "replay+resp" —— 重放 gold，再补一条 respond（内容 = task.outputs 拼接）
      "noop"      —— 什么都不做，直接结算
    """
    env = build_env(task_index, tool_names=tool_names)
    env.reset(task_index=task_index)
    task = TASKS[task_index]

    if mode.startswith("replay"):
        for action in task.actions:
            env.step(action)
    if mode == "replay+resp" and task.outputs:
        env.step(
            Action(
                name=RESPOND_ACTION_NAME,
                kwargs={"content": " ".join(task.outputs)},
            )
        )

    return env.calculate_reward().reward


# ---------------------------------------------------------------- 主流程


def main() -> int:
    print("=" * 72)
    print("Stage 0 出口判据 · gold 重放 oracle")
    print("=" * 72)
    print(f"题目数：{N}   工具数：{len(ALL_TOOL_NAMES)}")
    print()

    results = {}

    # ---- A / B：全量工具下的 oracle
    print("跑 A（gold 重放）/ B（gold 重放 + 合成 respond）/ C（什么都不做）...")
    a = [run(i, mode="replay") for i in range(N)]
    b = [run(i, mode="replay+resp") for i in range(N)]
    c = [run(i, mode="noop") for i in range(N)]
    results["A_replay"] = a
    results["B_replay_respond"] = b
    results["C_noop"] = c

    n_a, n_b, n_c = sum(a), sum(b), sum(c)

    print()
    print("-" * 72)
    print("结果")
    print("-" * 72)
    print(f"A  gold 重放                 通过 {int(n_a):>3} / {N}   "
          f"({n_a/N*100:5.1f}%)")
    print(f"B  gold 重放 + 合成 respond   通过 {int(n_b):>3} / {N}   "
          f"({n_b/N*100:5.1f}%)")
    print(f"C  什么都不做                 通过 {int(n_c):>3} / {N}   "
          f"({n_c/N*100:5.1f}%)   ← 这些是 reward hack 面")
    print()

    # A 里失败的题是谁
    fail_a = [i for i in range(N) if a[i] != 1.0]
    if fail_a:
        print(f"A 未通过的 {len(fail_a)} 道：")
        for i in fail_a:
            t = TASKS[i]
            print(f"  task[{i:>2}] {t.user_id:<26} gold 动作 {len(t.actions)} 条，"
                  f"outputs {len(t.outputs)} 条")
        print()

    # C 里通过的题是谁 —— 这部分最关键
    pass_c = [i for i in range(N) if c[i] == 1.0]
    if pass_c:
        print(f"⚠️  C「什么都不做」却能拿满分的 {len(pass_c)} 道：")
        for i in pass_c:
            t = TASKS[i]
            print(f"  task[{i:>2}] {t.user_id:<26} gold 动作 {len(t.actions)} 条，"
                  f"outputs {len(t.outputs)} 条")
        print()
        print("  这些题在 GRPO 里是毒药：组内不管采样成什么样，")
        print("  「不调工具」永远是对的 → 组内零方差 → 无梯度；")
        print("  一旦有方差，梯度方向是「学会别调工具」——")
        print("  正好会**人为制造出**我们要研究的『欠调用』现象。")
        print()

    # ---- D：砍掉 gold 里用到的工具会怎样
    print("-" * 72)
    print("D  砍工具的影响（README 的 14→7 方案验证）")
    print("-" * 72)
    probe_tool = "cancel_reservation"
    subset = sorted(ALL_TOOL_NAMES - {probe_tool})
    d = [run(i, mode="replay", tool_names=subset) for i in range(N)]
    n_d = sum(d)
    broke = [i for i in range(N) if a[i] == 1.0 and d[i] != 1.0]
    print(f"   砍掉 {probe_tool} 后，gold 重放通过 {int(n_d)} / {N}")
    print(f"   原本能过、现在挂掉的题：{len(broke)} 道 → task{broke}")
    print()

    # ---- E：用子集跑是否等价
    print("-" * 72)
    print("E  工具子集本身是否破坏奖励（保留全部 gold 用到的工具）")
    print("-" * 72)
    e = [run(i, mode="replay", tool_names=sorted(GOLD_TOOLS)) for i in range(N)]
    n_e = sum(e)
    same = all(e[i] == a[i] for i in range(N))
    print(f"   只给 GOLD_TOOLS（{len(GOLD_TOOLS)} 个）跑 gold 重放：通过 {int(n_e)} / {N}")
    print(f"   与全量工具结果是否逐题一致：{'是 ✅' if same else '否 ❌'}")
    print()

    cut_subset = sorted(ALL_TOOL_NAMES - SAFE_TO_CUT)
    f = [run(i, mode="replay", tool_names=cut_subset) for i in range(N)]
    n_f = sum(f)
    same_f = all(f[i] == a[i] for i in range(N))
    print(f"   砍掉 SAFE_TO_CUT={sorted(SAFE_TO_CUT)}（{len(cut_subset)} 个工具）："
          f"通过 {int(n_f)} / {N}，逐题一致：{'是 ✅' if same_f else '否 ❌'}")
    print()

    # ---- 存盘
    out = PROJECT / "data" / "oracle_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_tasks": N,
                "A_replay": a,
                "B_replay_respond": b,
                "C_noop": c,
                "D_cut_cancel_reservation": d,
                "E_gold_tools_only": e,
                "summary": {
                    "A_pass": int(n_a),
                    "B_pass": int(n_b),
                    "C_pass": int(n_c),
                    "hackable_tasks": pass_c,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"逐题结果已存：{out}")

    print()
    print("=" * 72)
    if n_b == N:
        print(f"✅ Stage 0 判据达成：gold 重放 + 合成 respond 后 {N}/{N} 全通过")
    else:
        print(f"❌ 判据未达成：B 只过了 {int(n_b)}/{N}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
