# -*- coding: utf-8 -*-
"""
块 T5 · 用户模拟器判据（零模型 / 零 GPU / 断网可跑）。

判据不是"我觉得它像" —— 而是拿**真实数据回放**：

    115 条真实轨迹里有 685 对【agent 问 → 用户答】。
    把 agent 的那些问题**原样喂给模拟器**，看它答出什么。

四个数：
    ① 兜底率            答不上来、只能糊弄的比例          **判据 < 30%**
    ② 关键信息命中率    真实回答里的预定号/user_id，模拟器答不答得出
    ③ 收尾正确率        agent 说"还有别的需要吗"，模拟器该收尾
    ④ 确定性            同一输入多次调用结果一致

⭐ 对照组：旧的占位版（永远回 "Okay, please go ahead."）也跑一遍同样的 685 对。
   **没有对照的数字没有意义** —— "命中率 40%" 听着低，但占位版是 0%。

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/eval_user_sim.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import TASKS                                              # noqa: E402
from env.tau_env import load_data_readonly                         # noqa: E402
from env.user_sim import SlotUserSim, STOP, detect_intent          # noqa: E402

SFT = PROJECT / "data" / "sft_final.jsonl"
RE_RES = re.compile(r"\b[A-Z0-9]{6}\b")
RE_UID = re.compile(r"\b[a-z]+_[a-z]+_\d+\b")

_fails = []


def check(cond, label, detail=""):
    print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond


def load_pairs():
    """从真实轨迹里抽出【agent 问 → 用户答】配对，按 task 分组。"""
    rows = [json.loads(l) for l in SFT.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_task = {}
    for r in rows:
        tid = r["task_id"]
        if not (0 <= tid < len(TASKS)):     # 只看 50 道真题；合成题没有对应 task 对象
            continue
        ms = r["messages"]
        for i, m in enumerate(ms):
            if m["role"] == "user" and i > 1:
                prev = ms[i - 1]
                if prev["role"] == "assistant" and prev.get("content"):
                    by_task.setdefault(tid, []).append((prev["content"], m["content"]))
    return by_task


class PlaceholderSim:
    """旧的占位版 —— 对照组。永远回同一句话，零信息。"""

    def reset(self, instruction=None):
        pass

    def step(self, content):
        return "Okay, please go ahead."


def ids_in(text):
    return set(RE_RES.findall(text)) | set(RE_UID.findall(text))


def main() -> int:
    print("=" * 78)
    print("块 T5 · 用户模拟器判据（真实数据回放）")
    print("=" * 78)

    data = load_data_readonly()
    by_task = load_pairs()
    n_pairs = sum(len(v) for v in by_task.values())
    print(f"\n   回放语料：{len(by_task)} 道题 ｜ {n_pairs} 对【agent 问 → 用户答】")

    results = {}
    # ⚠️ 回放时 max_turns 要放到很大：真实轨迹有的长达 27 个 user 轮，
    #    用运行时的 20 轮上限会让模拟器在后半段一律回 STOP，把指标压低 ——
    #    那是**评测口径的假象**，不是模拟器不行。
    for name, make in (("SlotUserSim（新）", lambda t: SlotUserSim(t, data, max_turns=10 ** 6)),
                       ("占位版（对照）", lambda t: PlaceholderSim())):
        n_fb = n_tot = 0
        id_total = id_hit = 0
        stop_total = stop_ok = 0
        per_intent = Counter()
        samples = []

        for tid, pairs in by_task.items():
            sim = make(TASKS[tid])
            sim.reset(TASKS[tid].instruction)
            for q, real_a in pairs:
                got = sim.step(q)
                n_tot += 1
                intent = detect_intent(q)
                per_intent[intent] += 1

                if isinstance(sim, SlotUserSim) and sim.fallbacks and intent == "fallback":
                    n_fb += 1
                elif not isinstance(sim, SlotUserSim):
                    # 占位版：没有"兜底"概念，但它对**所有**问题都是同一句空话
                    n_fb += 1

                # 关键信息：真实回答里有的 ID，模拟器答不答得出
                want = ids_in(real_a.replace(STOP, ""))
                if want:
                    id_total += 1
                    if want & ids_in(got):
                        id_hit += 1
                        if len(samples) < 3:
                            samples.append((q[:60], real_a[:70], got[:70]))

                # 收尾：agent 在问"还有别的吗"，就该回 STOP
                if intent == "stop":
                    stop_total += 1
                    if STOP in got:
                        stop_ok += 1

        results[name] = {
            "fallback_rate": n_fb / max(1, n_tot),
            "id_hit_rate": id_hit / max(1, id_total),
            "id_total": id_total,
            "stop_rate": stop_ok / max(1, stop_total),
            "per_intent": per_intent,
            "samples": samples,
        }

    print("\n① 对照表")
    print(f"   {'模拟器':<18}{'兜底率':>10}{'关键信息命中率':>16}{'收尾正确率':>12}")
    for name, r in results.items():
        print(f"   {name:<18}{r['fallback_rate']:>9.1%}{r['id_hit_rate']:>15.1%}"
              f"{r['stop_rate']:>11.1%}")
    print(f"      （关键信息命中率的样本量：{results['SlotUserSim（新）']['id_total']} 对"
          f"含 ID 的真实回答）")

    new, old = results["SlotUserSim（新）"], results["占位版（对照）"]
    check(new["fallback_rate"] < 0.30, "① 兜底率 < 30%",
          f"{new['fallback_rate']:.1%}（判据线 30%）")
    check(new["id_hit_rate"] > 0.60, "② 关键信息命中率 > 60%",
          f"{new['id_hit_rate']:.1%}")
    check(new["id_hit_rate"] > old["id_hit_rate"] + 0.5, "③ 显著优于占位版对照",
          f"新 {new['id_hit_rate']:.1%} vs 占位 {old['id_hit_rate']:.1%}")
    check(new["stop_rate"] > old["stop_rate"] + 0.9, "④ 收尾正确率远高于占位版",
          f"新 {new['stop_rate']:.1%} vs 占位 {old['stop_rate']:.1%}")

    print("\n⑤ 确定性：同一输入多次调用结果一致")
    sim = SlotUserSim(TASKS[0], data)
    sim.reset(TASKS[0].instruction)
    a = [sim._reply("Which reservation would you like to change?", "") for _ in range(3)]
    check(len(set(a)) == 1, "三次调用逐字符相同", repr(a[0][:50]))

    print("\n⑥ 意图分布（agent 都在问什么）")
    for k, v in new["per_intent"].most_common(8):
        print(f"   {k:<18} {v:4d}")

    print("\n⑦ 抽样肉眼检查（真实答 vs 模拟器答）")
    for q, real_a, got in new["samples"]:
        print(f"   Q   : {q}")
        print(f"   真实: {real_a}")
        print(f"   模拟: {got}")
        print()

    print("=" * 78)
    if _fails:
        print(f"❌ 判据未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    print("✅ T5 用户模拟器判据通过")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
