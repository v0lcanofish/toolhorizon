# -*- coding: utf-8 -*-
"""
块 H3 · mock 自测 —— 不花钱、不占 GPU，把整条链路验一遍。

为什么必须先做这个（而不是直接租卡）：
    租卡是按小时烧钱的。**如果在卡上才发现循环写错了，那是在烧钱调 bug。**
    这里用一个"照本宣科的假策略"把循环跑通，把所有能提前发现的问题挡在租卡之前。

验四件事：
    ① 多轮循环跑得通    —— 消息序列、role 顺序、终止条件
    ② reward 算得对     —— 照本宣科 = 满分；什么都不做 = 0 分
    ③ 格式和 SFT 对齐   —— 生成的消息结构必须和 data/sft_final.jsonl 一致
    ④ 零方差机制可见    —— 组内 reward 全同时 advantage 全 0（项目核心命题的第一手验证）

跑法（纯 CPU）：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/eval_mock.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import TASKS                                              # noqa: E402
from env.rollout import run_episode                                # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME            # noqa: E402

SPLIT = PROJECT / "data" / "task_split.json"
SFT = PROJECT / "data" / "sft_final.jsonl"


# ---------------------------------------------------------------- 假策略


class ScriptedPolicy:
    """照本宣科的假策略：按顺序吐出 gold 动作，吐完了就说收尾话。

    ⚠️ 收尾话不是随便说的 —— 对带 `outputs` 的题（本 benchmark 有 4 道），
       环境的 outputs 检查会做**字符串包含**比对，不收尾话里嵌进关键信息就拿不到分。
       （这是块 H1 发现的硬约束，H3 第一次跑 mock 时立刻又踩到了）
    """

    def __init__(self, task, final_text: str = ""):
        self.actions = list(task.actions)
        self.i = 0
        if final_text:
            self.final_text = final_text
        elif task.outputs:
            self.final_text = ("Here is the information you asked for: "
                               + ", ".join(task.outputs) + ".")
        else:
            self.final_text = "All set. Anything else I can help with?"

    def __call__(self, messages):
        if self.i < len(self.actions):
            a = self.actions[self.i]
            self.i += 1
            return a
        return Action(name=RESPOND_ACTION_NAME,
                      kwargs={"content": self.final_text})


class NoopPolicy:
    """什么都不做，直接说收尾。"""

    def __init__(self, task):
        pass

    def __call__(self, messages):
        return Action(name=RESPOND_ACTION_NAME,
                      kwargs={"content": "I can't help with that right now."})


# ---------------------------------------------------------------- 检查


def check_roles(ep) -> str:
    """消息序列的两个硬约束：
       ① 首条必须是 system，第二条必须是 user
       ② assistant 后面必须紧跟 tool 或 user（不能凭空出现两条 assistant）"""
    r = [m["role"] for m in ep.messages]
    if r[0] != "system":
        return "首条不是 system"
    if r[1] != "user":
        return "第二条不是 user"
    for i in range(len(r) - 1):
        if r[i] == "assistant" and r[i + 1] not in ("tool", "user"):
            return f"第 {i} 条 assistant 后面跟的是 {r[i+1]}"
    return ""


def check_format(ep) -> str:
    """和 SFT 数据逐字段比 —— 字段名/位置对不上，训练就断了。"""
    for m in ep.messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            if list(m.keys()) != ["role", "content", "tool_calls"]:
                return f"assistant(工具轮) 字段不对: {list(m.keys())}"
            tc = m["tool_calls"][0]
            if set(tc.keys()) != {"id", "type", "function"}:
                return f"tool_calls 字段不对: {set(tc.keys())}"
        if m["role"] == "tool":
            if set(m.keys()) != {"role", "tool_call_id", "name", "content"}:
                return f"tool 消息字段不对: {set(m.keys())}"
    return ""


def main() -> int:
    print("=" * 78)
    print("块 H3 · mock 自测（零模型 / 零 GPU）")
    print("=" * 78)

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_ids = [r["task"] for r in split["train"]]
    probe_ids = [r["task"] for r in split["probe_overcall"]]

    # ---------------- ① 全 50 道跑一遍照本宣科
    print("\n① 多轮循环：照本宣科（执行 gold）跑全 50 道")
    eps, bad_role, bad_fmt = [], [], []
    for i in range(len(TASKS)):
        ep = run_episode(TASKS[i], ScriptedPolicy(TASKS[i]), task_id=i)
        eps.append(ep)
        e = check_roles(ep)
        if e:
            bad_role.append((i, e))
        e = check_format(ep)
        if e:
            bad_fmt.append((i, e))

    n_pass = sum(1 for e in eps if e.reward == 1.0)
    print(f"   消息序列合法     {len(eps)-len(bad_role)}/{len(eps)}")
    print(f"   字段格式对齐 SFT {len(eps)-len(bad_fmt)}/{len(eps)}")
    print(f"   reward = 1.0     {n_pass}/{len(eps)}")
    if bad_role:
        print("   ⚠️ role 问题：", bad_role[:3])
    if bad_fmt:
        print("   ⚠️ 格式问题：", bad_fmt[:3])

    by_term = Counter(e.terminated_by for e in eps)
    print(f"   终止方式         {dict(by_term)}")

    # ---------------- ② reward 判据
    print("\n② reward 判据：照本宣科 vs 什么都不做")

    print("   什么都不做（跑训练集的 31 道）...")
    noop_scores = {}
    for i in train_ids:
        ep = run_episode(TASKS[i], NoopPolicy(TASKS[i]), task_id=i)
        noop_scores[i] = ep.reward
    n_noop_zero = sum(1 for v in noop_scores.values() if v == 0.0)
    print(f"   训练集 31 道里，什么都不做得 0 分的：{n_noop_zero}/31")

    print("   什么都不做（跑探针集的 19 道）...")
    probe_scores = {}
    for i in probe_ids:
        ep = run_episode(TASKS[i], NoopPolicy(TASKS[i]), task_id=i)
        probe_scores[i] = ep.reward
    n_probe_one = sum(1 for v in probe_scores.values() if v == 1.0)
    print(f"   探针集 19 道里，什么都不做得满分的：{n_probe_one}/19")

    print("   照本宣科（跑训练集的 31 道）...")
    gold_scores = {}
    for i in train_ids:
        ep = run_episode(TASKS[i], ScriptedPolicy(TASKS[i]), task_id=i)
        gold_scores[i] = ep.reward
    n_gold_one = sum(1 for v in gold_scores.values() if v == 1.0)
    print(f"   训练集 31 道里，照本宣科得满分的：{n_gold_one}/31")

    # ---------------- ③ 零方差机制
    print("\n③ 零方差机制（项目核心命题的第一手验证）")

    def advantage(rewards):
        """GRPO 的组内相对优势（先不除 std，看原始差值）"""
        import statistics as st
        m = st.mean(rewards)
        sd = st.pstdev(rewards)
        if sd < 1e-8:
            return [0.0] * len(rewards), sd
        return [(r - m) / (sd + 1e-8) for r in rewards], sd

    # 用 8 道【纯 S 类】题构组——否则混进 outputs 题会干扰"全对/全错"的构造
    clean = [i for i in train_ids if TASKS[i].outputs == []][:8]
    cases = [
        ("训练集 + 照本宣科（全对）",
         [gold_scores[i] for i in clean]),
        ("训练集 + 什么都不做（全错）",
         [noop_scores[i] for i in clean]),
        ("混采：4 道照本宣科 + 4 道不做",
         [gold_scores[t] for t in clean[:4]] + [noop_scores[t] for t in clean[4:]]),
    ]
    print(f"   {'场景':<34}{'reward 分布':<22}{'std':>8}{'adv 全 0?':>12}")
    for name, rs in cases:
        adv, sd = advantage(rs)
        all_zero = all(abs(a) < 1e-9 for a in adv)
        dist = f"{sum(1 for r in rs if r==1.0)}对/{sum(1 for r in rs if r==0.0)}错"
        print(f"   {name:<34}{dist:<22}{sd:>8.4f}{('是 ← 零梯度' if all_zero else '否'):>12}")

    # ---------------- 断言
    print("\n" + "=" * 78)
    errs = []
    if n_pass != len(eps):
        errs.append(f"照本宣科没全满分：{n_pass}/{len(eps)}")
    if bad_role:
        errs.append(f"{len(bad_role)} 道消息序列不合法")
    if bad_fmt:
        errs.append(f"{len(bad_fmt)} 道字段格式和 SFT 不一致")
    if n_noop_zero != 31:
        errs.append(f"训练集里什么都不做应全 0 分，实际 {n_noop_zero}/31")
    if n_probe_one != 19:
        errs.append(f"探针集里什么都不做应全满分，实际 {n_probe_one}/19")

    if errs:
        print("❌ 自检未通过：")
        for e in errs:
            print("   -", e)
        return 1

    print("✅ mock 自测全绿")
    print(f"   · 多轮循环：50 道跑通，消息序列合法，字段与 SFT 一致")
    print(f"   · reward 判据：照本宣科 50/50 满分 ｜ 训练集不做全 0 分 ｜ 探针集不做全满分")
    print(f"   · 零方差机制：全对/全错时 advantage 恒为 0 → **零梯度**（项目命题成立）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
