# -*- coding: utf-8 -*-
"""
块 H4 · 观测器自测 —— 确认 10 个指标都算得出来、且口径正确。

两种验法，缺一不可：
  ① 合成轨迹：手工构造已知行为的轨迹，**期望值是手算出来的**，精确验证口径
  ② 真实轨迹：用假策略真的跑一遍环境，确认在真数据上不炸

跑法（纯 CPU，零 GPU）：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/eval_observe.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import TASKS                                          # noqa: E402
from env.rollout import Episode, run_episode                    # noqa: E402
from observe import compute, render, tool_use_of                # noqa: E402
from eval_mock import NoopPolicy, ScriptedPolicy                # noqa: E402


# ---------------------------------------------------------------- 合成轨迹


def fake_ep(task_id: int, tools, reward: float = 0.0, n_respond: int = 1,
            args=None) -> Episode:
    """手工造一条轨迹：只关心工具调用序列，便于手算期望值。

    tools: ["cancel_reservation", ...]；args: 与 tools 等长的参数字符串列表
           （不传则每次参数都不同 —— 即"对不同的对象各做一次"，抖动应为 0）
    """
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "go"}]
    cid = 0
    for k, name in enumerate(tools):
        cid += 1
        a = args[k] if args else json.dumps({"obj": k})
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{cid}", "type": "function",
                                     "function": {"name": name, "arguments": a}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{cid}",
                     "name": name, "content": "ok"})
    for _ in range(n_respond):
        msgs.append({"role": "assistant", "content": "done"})
        msgs.append({"role": "user", "content": "ok"})
    return Episode(task_id=task_id, messages=msgs, reward=reward,
                   n_turns=len(tools) + n_respond,
                   n_tool_calls=len(tools), done=True, terminated_by="user_stop")


WRITE = "cancel_reservation"
READ = "get_reservation_details"


# ---------------------------------------------------------------- ① 合成轨迹验口径


def test_synthetic() -> list:
    print("\n① 合成轨迹：口径精确验证")
    print("-" * 76)

    # S 类题（gold 有写动作） id=0,1,2 ；O 类题（gold 无写动作） id=100,101
    meta = {0: {"class": "S", "n_write": 1},
            1: {"class": "S", "n_write": 1},
            2: {"class": "S", "n_write": 2},
            100: {"class": "O", "n_write": 0},
            101: {"class": "O", "n_write": 0}}

    errs = []
    cases = [
        # 名字, 轨迹, 期望的指标子集
        ("全都照做（该写就写）",
         [fake_ep(0, [WRITE]), fake_ep(1, [READ, WRITE]), fake_ep(2, [WRITE, WRITE]),
          fake_ep(100, [READ]), fake_ep(101, [])],
         {"under_call_rate": 0.0, "over_call_rate": 0.0, "jitter_mean": 0.0}),

        ("全都不动手",
         [fake_ep(0, []), fake_ep(1, []), fake_ep(2, []),
          fake_ep(100, []), fake_ep(101, [])],
         {"under_call_rate": 1.0, "over_call_rate": 0.0}),

        ("在探针题上乱写（过调用）",
         [fake_ep(0, [WRITE]), fake_ep(1, [WRITE]), fake_ep(2, [WRITE, WRITE]),
          fake_ep(100, [WRITE]), fake_ep(101, [WRITE])],
         {"under_call_rate": 0.0, "over_call_rate": 1.0}),

        # 抖动 = 对【同一个对象】重复操作。下面每次参数都一样 → 全是无效重复
        ("抖动：对同一个对象反复操作",
         [fake_ep(0, [READ] * 4, args=['{"id":"A"}'] * 4),      # 4 次同参 → 重复 3
          fake_ep(1, [READ] * 2, args=['{"id":"B"}'] * 2),      # 2 次同参 → 重复 1
          fake_ep(2, [WRITE] * 2, args=['{"id":"C"}'] * 2),     # 2 次同参 → 重复 1
          fake_ep(100, [READ]),                                  # 1 次 → 0
          fake_ep(101, [])],                                     # 0 次 → 0
         {"jitter_mean": (3/4 + 1/2 + 1/2 + 0 + 0) / 5}),
    ]

    for name, eps, expect in cases:
        m = compute(eps, meta)
        d, h = m["degradation"], m["health"]
        got = dict(d)
        got["jitter_mean"] = round(d["jitter_mean"], 4)
        ok = True
        for k, v in expect.items():
            if abs(got.get(k, -999) - v) > 1e-6:
                ok = False
                errs.append(f"[{name}] {k}: 期望 {v}, 实得 {got.get(k)}")
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}")
        print(f"       欠调用={d['under_call_rate']:.2f} ({d['under_call_n']})  "
              f"过调用={d['over_call_rate']:.2f} ({d['over_call_n']})  "
              f"抖动={d['jitter_mean']:.3f}")

    # 抖动的手算：以「工具名+参数」为签名，重复次数/总数
    #   0号 4 次同参 → 3 次重复 → 0.75
    #   1号 2 次同参 → 1 次重复 → 0.50
    #   2号 2 次同参 → 1 次重复 → 0.50
    #   100号 1 次 → 0 ； 101号 0 次 → 0
    expected_jitter = (0.75 + 0.5 + 0.5 + 0.0 + 0.0) / 5
    print(f"\n  抖动指数手算校验：期望 {expected_jitter:.3f}")
    m = compute(cases[3][1], meta)
    got_j = m["degradation"]["jitter_mean"]
    if abs(got_j - expected_jitter) > 1e-6:
        errs.append(f"抖动手算对不上：期望 {expected_jitter:.4f} 实得 {got_j:.4f}")
        print(f"                    ❌ 实得 {got_j:.4f}")
    else:
        print(f"                    ✅ 实得 {got_j:.4f}")

    # 零方差率：把同题多采样组成"组"
    print("\n  组内零方差率（GRPO 的组 = 同一道题的 n 条采样）")
    grp_same = [fake_ep(0, [WRITE], reward=1.0) for _ in range(4)]
    grp_mix = [fake_ep(1, [WRITE], reward=1.0), fake_ep(1, [WRITE], reward=1.0),
               fake_ep(1, [WRITE], reward=0.0), fake_ep(1, [WRITE], reward=0.0)]
    m = compute(grp_same + grp_mix, meta)
    zvr = m["health"]["zero_var_rate"]
    print(f"     2 个组（一组全同、一组混合）→ 零方差率 = {zvr:.2f}（期望 0.50）")
    if abs(zvr - 0.5) > 1e-6:
        errs.append(f"零方差率：期望 0.5 实得 {zvr}")
    else:
        print("     ✅ 正确")

    return errs


# ---------------------------------------------------------------- ② 真实轨迹


def test_real() -> list:
    print("\n② 真实轨迹：用假策略真的跑一遍环境")
    print("-" * 76)

    split = json.loads((PROJECT / "data" / "task_split.json").read_text(encoding="utf-8"))
    meta = {}
    for r in split["train"]:
        meta[r["task"]] = {"class": "S", "n_write": r["n_write"]}
    for r in split["probe_overcall"]:
        meta[r["task"]] = {"class": "O", "n_write": 0}

    errs = []
    scenarios = [
        ("照本宣科", ScriptedPolicy, 0.0, 0.0),
        ("什么都不做", NoopPolicy, 1.0, 0.0),
    ]
    for name, policy_cls, exp_under, exp_over in scenarios:
        # 训练集跑 6 道 + 探针集跑 4 道（够算指标即可，不必全跑）
        ids = [r["task"] for r in split["train"][:6]] + \
              [r["task"] for r in split["probe_overcall"][:4]]
        eps = [run_episode(TASKS[i], policy_cls(TASKS[i]), task_id=i) for i in ids]
        m = compute(eps, meta)
        d = m["degradation"]
        ok = (abs(d["under_call_rate"] - exp_under) < 1e-6
              and abs(d["over_call_rate"] - exp_over) < 1e-6)
        print(f"  {'✅' if ok else '❌'} {name:<10} "
              f"欠调用={d['under_call_rate']:.2f} ({d['under_call_n']})  "
              f"过调用={d['over_call_rate']:.2f} ({d['over_call_n']})  "
              f"抖动={d['jitter_mean']:.3f}  零方差率={m['health']['zero_var_rate']:.2f}")
        if not ok:
            errs.append(f"[{name}] 欠/过调用率不符："
                        f"期望 {exp_under}/{exp_over}，实得 "
                        f"{d['under_call_rate']}/{d['over_call_rate']}")
        if name == "照本宣科":
            print()
            print(render(m))
            print()
    return errs


# ---------------------------------------------------------------- main


def main() -> int:
    print("=" * 76)
    print("块 H4 · 观测器自测（10 个指标）")
    print("=" * 76)

    errs = test_synthetic() + test_real()

    print("\n" + "=" * 76)
    if errs:
        print("❌ 自检未通过：")
        for e in errs:
            print("   -", e)
        return 1
    print("✅ 观测器自检全绿")
    print("   · 10 个指标全部算得出来")
    print("   · 口径经【手算 + 合成轨迹】精确验证（含抖动指数、零方差率）")
    print("   · 真实轨迹上跑通（照本宣科 → 欠调用 0；什么都不做 → 欠调用 1.0）")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
