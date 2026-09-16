# -*- coding: utf-8 -*-
"""
探针 1：把 τ-bench airline 的 50 题 gold 轨迹结构量出来。

为什么要有这个脚本：
  README 里写了"工具 14 → 7"，但 Env.calculate_reward() 是靠**重放 gold actions**
  算 gt_data_hash 的（base.py:125-165）。工具出现在 gold 里却被砍掉会怎样，得先量出来。

⚠️ 本脚本第三节给出的「reward 恒为 0」推断是**错的**，2026-09-15 已被
   scripts/verify_oracle.py 的 D 组实测推翻（砍掉 cancel_reservation 后通过率
   仍是 46/50，一点没降）。真实机制见 scripts/analyze_tool_subsets.py：
   重放和 agent 调用走的是**同一个 tools_map**，工具缺失时两边同样 no-op，
   哈希照样相等 → 代价不是 reward 归零，而是**题目失去区分度**。

本脚本不依赖 litellm（参考仓库只读，用桩模块注入，不改它的源码）。

跑法：
  PYTHONIOENCODING=utf-8 D:/anaconda/python.exe scripts/probe_tau_bench.py
"""

import sys
import types
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- 路径

HERE = Path(__file__).resolve().parent          # ToolHorizon/scripts
PROJECT = HERE.parent                            # ToolHorizon
from env.bootstrap import TAU_BENCH                            # noqa: E402

if not TAU_BENCH.exists():
    sys.exit(f"[FATAL] 找不到 tau-bench：{TAU_BENCH}")

# tau-bench 目录本身要进 sys.path，才能 `import tau_bench`
sys.path.insert(0, str(TAU_BENCH))


# ------------------------------------------------- 桩掉 litellm（只读仓库，不改源码）

def stub_litellm() -> None:
    """
    tau_bench/envs/user.py 第 5 行是模块顶层的 `from litellm import completion`，
    本地没装 litellm，一 import 就炸。而 import tau_bench 又会连带执行 user.py。

    参考仓库要求只读，所以这里往 sys.modules 里塞一个假 litellm，
    让它能 import 过去。真正用到 completion 的是 LLMUserSimulationEnv，
    我们后续要用规则模拟器替换掉它，不会走到。
    """
    if "litellm" in sys.modules:
        return

    fake = types.ModuleType("litellm")

    def _completion(*args, **kwargs):
        raise RuntimeError(
            "桩 litellm 被调用了 —— 说明有代码在真的想走 LLM 用户模拟器。"
            "Stage 0 应当用规则式 RuleBasedUserSim，不该走到这里。"
        )

    fake.completion = _completion
    fake.provider_list = []
    sys.modules["litellm"] = fake


stub_litellm()


# ---------------------------------------------------------------- 导入

print("=" * 68)
print("探针 1 · τ-bench airline gold 轨迹结构")
print("=" * 68)
print(f"Python     : {sys.version.split()[0]}")
print(f"tau-bench  : {TAU_BENCH}")
print()

from tau_bench.envs.airline.tasks_test import TASKS            # noqa: E402
from tau_bench.envs.airline.tools import ALL_TOOLS             # noqa: E402

print(f"[OK] import 成功：{len(TASKS)} 个 task，{len(ALL_TOOLS)} 个 tool")
print()

# ---------------------------------------------------------------- 1. 工具分布

tool_counter = Counter()       # 每个 tool 在 gold 里出现几次
respond_count = 0              # respond 动作数
all_tool_names = {t.get_info()["function"]["name"] for t in ALL_TOOLS}

for task in TASKS:
    for action in task.actions:
        if action.name == "respond":
            respond_count += 1
        else:
            tool_counter[action.name] += 1

print("-" * 68)
print("① 全部 tool 在 gold 轨迹里的出现次数（按次数降序）")
print("-" * 68)
print(f"{'tool 名':<32}{'gold 出现次数':>12}   用到的题数")
print("-" * 68)

# 每题的 tool 集合，用来算"哪些题依赖某个 tool"
task_tools = [set(a.name for a in t.actions if a.name != "respond") for t in TASKS]

for name in sorted(all_tool_names, key=lambda n: -tool_counter[n]):
    n = tool_counter[name]
    n_tasks = sum(1 for s in task_tools if name in s)
    flag = "  ← gold 里从未用到的 tool" if n == 0 else ""
    print(f"{name:<32}{n:>12}   {n_tasks:>3} 题{flag}")

print("-" * 68)
print(f"respond 动作总数：{respond_count}")
print()

# ---------------------------------------------------------------- 2. 每题步数

lengths = [len([a for a in t.actions]) for t in TASKS]
tool_only = [len([a for a in t.actions if a.name != "respond"]) for t in TASKS]
sorted_l = sorted(lengths)


def pct(xs, p):
    """简单百分位（最近邻，不插值）—— 样本量小的时候够用。"""
    if not xs:
        return 0
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return sorted(xs)[k]


print("-" * 68)
print("② 每题的 gold 动作步数")
print("-" * 68)
print(f"gold 动作总数（含 respond）：p50={pct(lengths,50)}  p90={pct(lengths,90)}  "
      f"max={max(lengths)}  min={min(lengths)}")
print(f"只看 tool 调用        ：p50={pct(tool_only,50)}  p90={pct(tool_only,90)}  "
      f"max={max(tool_only)}  min={min(tool_only)}")
print()

step_hist = Counter(lengths)
print("步数直方图：")
for step in sorted(step_hist):
    bar = "█" * step_hist[step]
    print(f"  {step:>2} 步 | {bar} {step_hist[step]}")
print()

# ---------------------------------------------------------------- 3. 砍工具可行性
# README 的候选：保留 7 个，砍掉 book/cancel/send_certificate/transfer/list_airports/think
KEEP = [
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "search_onestop_flight",
    "update_reservation_flights",
    "update_reservation_baggages",
    "calculate",
]
CUT = sorted(all_tool_names - set(KEEP))

print("-" * 68)
print("③ 砍工具可行性（README 的 14 → 7 方案）")
print("-" * 68)
print(f"计划保留 {len(KEEP)} 个：{', '.join(KEEP)}")
print(f"计划砍掉 {len(CUT)} 个：{', '.join(CUT)}")
print()

broken = []       # 砍掉后会重放失败的题
for i, t in enumerate(TASKS):
    hit = task_tools[i] & set(CUT)
    if hit:
        broken.append((i, sorted(hit), t.user_id))

print(f"⚠️  砍掉这些 tool 后，**{len(broken)} / {len(TASKS)} 题**的 gold 重放会失败")
print()

if broken:
    cut_hit = Counter()
    for _, hit, _ in broken:
        for h in hit:
            cut_hit[h] += 1
    print("被牵连的 tool 及牵连题数：")
    for name, n in cut_hit.most_common():
        total = tool_counter[name]
        print(f"  {name:<28} 牵连 {n:>2} 题（该 tool 在 gold 中共出现 {total} 次）")
    print()

    print("前 10 道会被牵连的题：")
    for i, hit, uid in broken[:10]:
        print(f"  task[{i:>2}] {uid:<26} 用到被砍的：{', '.join(hit)}")
    print()

    print("👉 结论：README 要砍的 7 个工具里，有 5 个真的出现在 gold 中。")
    print("   ⚠️ 但**不要**据此推断「reward 会恒为 0」—— 那是本脚本早期版本的错误推断，")
    print("      实测见 scripts/analyze_tool_subsets.py：")
    print("      README 的 14→7 方案让有区分度的题从 30 掉到 14（废题率 40% → 72%）。")
else:
    print("👉 结论：README 的 14→7 方案成立，gold 重放不受影响。")

# ---------------------------------------------------------------- 4. gold 是不是完整轨迹

print("-" * 68)
print("④ task.actions 的语义：是完整轨迹，还是「数据库变更 oracle」？")
print("-" * 68)

n_with_outputs = sum(1 for t in TASKS if len(t.outputs) > 0)
out_lens = [len(t.outputs) for t in TASKS]
print(f"有 outputs 的题：{n_with_outputs} / {len(TASKS)}")
print(f"outputs 条数分布：{dict(sorted(Counter(out_lens).items()))}")
print()

print("outputs 举例（每题要求最终回复里必须出现的字符串）：")
for i in (0, 1, 2):
    t = TASKS[i]
    print(f"  task[{i}] {t.user_id}  outputs={t.outputs}")
print()

print("gold 里出现过的动作名（去重）：")
seen_names = Counter(a.name for t in TASKS for a in t.actions)
print(f"  {sorted(seen_names)}")
print(f"  其中 respond 出现 {seen_names.get('respond', 0)} 次")
print()

print("0 步题（gold 无任何动作）长什么样 —— 看是不是「纯查询」题：")
zero_step = [i for i in range(len(TASKS)) if len(TASKS[i].actions) == 0]
for i in zero_step[:3]:
    t = TASKS[i]
    instr = t.instruction[:150].replace("\n", " ")
    print(f"  task[{i:>2}] {t.user_id}")
    print(f"          instruction: {instr}...")
    print(f"          outputs    : {t.outputs}")
print(f"  （0 步题共 {len(zero_step)} 道）")
print()

print("👉 判读：")
if seen_names.get("respond", 0) == 0 and n_with_outputs > 0:
    print("   gold 里 0 个 respond，但 {0} 道题有 outputs 要求".format(n_with_outputs))
    print("   → task.actions **不含最终回复**（respond 恒为 0），")
    print("     所以「重放 gold 直接得到 SFT 数据」不成立：轨迹没有收尾的答复。")
    print()
    print("   ⚠️ 修正：本脚本早期版本说「gold 只记录落库变更」也是错的。")
    print("      实测 scripts/analyze_hackable.py：50 题 gold 共 158 条动作中")
    print("      只读动作 98 条（62%），写动作 60 条。gold 是**读写混合的参考动作序列**。")
else:
    print("   与预期不符，需要重新判读。")

print()
print("=" * 68)
print("探针 1 结束")
print("=" * 68)
