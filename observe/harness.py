# -*- coding: utf-8 -*-
"""
评测 harness —— 三分类口径（covered / uncovered / unseen）。

━━━ 为什么不能只报一个 pass@1 ━━━

longhorizon 的诊断文档白纸黑字承认：

    train ∩ eval = 40/50 = 80%   ｜   unseen 只有 10 题（统计上不可靠）

也就是说，它报出来的"泛化"其实**八成是在背答案**，
而真正测泛化的那 10 道题，n 太小，抖动比信号大。

本项目扩题到 800 道之后，第一次可以**把这三件事分开量**：

    covered    训练时见过的原题（31 道）        → 高 = 记住了，不稀奇
    uncovered  同难度、同类型，但没训过        → ⭐ 真正的"学会了没有"
    unseen     更难的档 / 没训过的类型          → ⭐ 泛化边界

    ⭐ **covered 和 uncovered 的差距，就是"训练集泄漏"的量。**

━━━ 口径 ━━━

pass@1 = 每题采 N 条，**全对才算过**（τ-bench 的通行口径，比"有一条对"严）
另外报组内零方差率 —— 它才是"这一步训练有没有用"的直接指标。

跑法（纯 CPU，用假策略验证 harness 本身）：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 -m observe.harness --mock
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from train.rollout_batch import RolloutConfig, compute_advantages, run_grouped_rollout

TASK_SPLIT = _PROJECT / "data" / "task_split.json"
EXPANDED = _PROJECT / "data" / "tasks_expanded.json"

SPLITS = ("covered", "uncovered", "unseen")


# ---------------------------------------------------------------- 分片


@dataclass
class TaskItem:
    task_id: str
    task: Any
    kind: str = ""
    difficulty: str = ""
    src: str = ""


def _expanded_tasks(kinds: set, difficulties: set, limit: int) -> List[TaskItem]:
    """从扩题集里取一部分（按 kind/difficulty 过滤）。"""
    from tau_bench.types import Action, Task

    if not EXPANDED.exists():
        return []
    exp = json.loads(EXPANDED.read_text(encoding="utf-8"))
    out: List[TaskItem] = []
    for j, raw in enumerate(exp["tasks"]):
        if kinds and raw.get("kind") not in kinds:
            continue
        if difficulties and raw.get("difficulty") not in difficulties:
            continue
        out.append(TaskItem(
            task_id=f"exp{j}",
            task=Task(
                user_id=raw["user_id"],
                actions=[Action(name=a["name"], kwargs=a["kwargs"]) for a in raw["actions"]],
                outputs=raw.get("outputs", []),
                instruction=raw["instruction"],
            ),
            kind=raw.get("kind", ""),
            difficulty=raw.get("difficulty", ""),
            src="扩题器",
        ))
        if len(out) >= limit:
            break
    return out


def build_splits(limit: int = 30) -> Dict[str, List[TaskItem]]:
    """
    构造三个分片。

    ⚠️ 分片规则是**结构性的**，不依赖随机：
       covered   = `task_split.json` 里的 31 道训练题
       uncovered = 扩题集 **easy 档**（同难度、新用户/新订单组合，没训过）
       unseen    = 扩题集 **medium 档**（步骤更多、信息依赖更长）

    ⚠️ 实测（2026-09-17）**"换类型"这个轴不存在**：
       扩题集只有四种 kind（cancel / change_baggages / change_passenger /
       send_certificate），它们用到的工具全部 ⊆ 训练集 31 道用过的工具集。
       也就是说 —— **扩题器扩的是"实例"，不是"任务类型"**。
       这条要如实写进报告：泛化测得到的是"新实例"，不是"新能力"。
       → 所以第三个分片改用**难度**当轴（easy 训练过、medium 是能力边界外）。

    Returns:
        {split_name: [TaskItem, ...]}
    """
    from env import TASKS

    split = json.loads(TASK_SPLIT.read_text(encoding="utf-8"))
    train_ids = [r["task"] for r in split["train"]]

    covered = [TaskItem(f"tau{i}", TASKS[i], kind="tau-bench", src="原题") for i in train_ids]

    uncovered = _expanded_tasks(set(), {"easy"}, limit)
    unseen = _expanded_tasks(set(), {"medium"}, limit)

    return {"covered": covered, "uncovered": uncovered, "unseen": unseen}


# ---------------------------------------------------------------- 评测


@dataclass
class SplitResult:
    name: str
    n_tasks: int
    n_samples: int
    pass_at_1: float          # 全对才算过
    pass_any: float           # 有一条对就算过
    reward_mean: float
    zero_var_rate: float
    turns_mean: float
    tool_calls_mean: float
    terminated: Dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def evaluate_split(
    name: str,
    items: Sequence[TaskItem],
    engine,
    n_samples: int = 4,
    cfg: RolloutConfig = None,
    tokenizer=None,
    tools=None,
) -> SplitResult:
    """跑一个分片，返回该分片的指标。"""
    cfg = cfg or RolloutConfig(n_group=n_samples)
    if not items:
        return SplitResult(name, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    batch = run_grouped_rollout(
        [(it.task_id, it.task) for it in items], engine, cfg, tokenizer, tools,
    )
    eps = batch.episodes
    rewards = [e.reward for e in eps]
    return SplitResult(
        name=name,
        n_tasks=len(batch.groups),
        n_samples=len(eps),
        pass_at_1=st.mean([1.0 if g.mean >= 1.0 - 1e-9 else 0.0 for g in batch.groups]),
        pass_any=st.mean([1.0 if any(r >= 1.0 - 1e-9 for r in g.rewards) else 0.0
                          for g in batch.groups]),
        reward_mean=st.mean(rewards),
        zero_var_rate=batch.zero_var_rate,
        turns_mean=st.mean([e.n_turns for e in eps]),
        tool_calls_mean=st.mean([e.n_tool_calls for e in eps]),
        terminated=dict(Counter(e.terminated_by for e in eps)),
    )


def evaluate(engine, n_samples: int = 4, limit: int = 30,
             tokenizer=None, tools=None) -> Dict[str, SplitResult]:
    from train.tokenize import load_tokenizer, tool_schemas

    tok = tokenizer or load_tokenizer()
    tools = tools if tools is not None else tool_schemas()
    splits = build_splits(limit=limit)
    return {name: evaluate_split(name, items, engine, n_samples,
                                 tokenizer=tok, tools=tools)
            for name, items in splits.items()}


# ---------------------------------------------------------------- 报告


def render(results: Dict[str, SplitResult]) -> str:
    L = ["分片        题数   样本   pass@1(全对)  pass@any   reward均值  组内零方差率  平均轮数  工具调用",
         "-" * 96]
    for name in SPLITS:
        r = results.get(name)
        # ⚠️ "没跑这个分片" 和 "这个分片真的没题" 是两件事，**不能用同一句话**。
        #    2026-09-18 实测踩过：eval_sft_gate 只跑 gate 分片时，
        #    covered 会显示成"扩题集里没有符合条件的题" —— 而 covered 是 31 道训练原题，
        #    根本不来自扩题集。**这句话是错的，而且不报错**，正是本项目一直在治的"静默失真"。
        if r is None:
            L.append(f"{name:<11} {'—':>4}   （本次未运行这个分片）")
            continue
        if r.n_tasks == 0:
            L.append(f"{name:<11} {'—':>4}   （空分片：扩题集里没有符合条件的题）")
            continue
        L.append(f"{name:<11} {r.n_tasks:>4} {r.n_samples:>6} "
                 f"{r.pass_at_1:>12.3f} {r.pass_any:>10.3f} {r.reward_mean:>11.3f} "
                 f"{r.zero_var_rate:>13.3f} {r.turns_mean:>9.2f} {r.tool_calls_mean:>9.2f}")
    cov, unc = results.get("covered"), results.get("uncovered")
    if cov and unc and cov.n_tasks and unc.n_tasks:
        gap = cov.pass_at_1 - unc.pass_at_1
        L.append("")
        L.append(f"⭐ 训练集泄漏量（covered − uncovered）= {gap:+.3f}")
        L.append("   这个差越大，说明分数里'背答案'的成分越多，越不能当泛化。")
    return "\n".join(L)


# ---------------------------------------------------------------- mock 自测


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="用假策略跑（零模型/零 GPU）")
    ap.add_argument("--n", type=int, default=4, help="每题采样条数")
    ap.add_argument("--limit", type=int, default=30, help="每个分片最多多少题")
    args = ap.parse_args(argv)

    if not args.mock:
        print("真模型评测请走 train/collect.py 的引擎；这里只做 CPU 自测。")
        return 2

    from env import TASKS
    from train.engine import ScriptedEngine
    from train.tokenize import load_tokenizer, tool_schemas

    tok, tools = load_tokenizer(), tool_schemas()

    def make_scripted(pool):
        """
        照本宣科的假策略：从 prompt 里数已经调了几个工具，决定下一步。

        ⚠️ 两个踩过的坑：
           ① 匹配必须用**完整 instruction**：50 道题里有 8 组的前 60 字符完全相同
              （task 41/42/43 都以 "You are Anya Garcia (with ID: anya_garcia_5901)…" 开头），
              用前缀匹配会张冠李戴 —— 实测直接让 6/31 掉分。
           ② `pool` 必须包含**扩题集的新题**，不能只有 50 道原题，
              否则 uncovered / unseen 两个分片全部返回兜底句、通过率 0，
              看起来像"泛化彻底失败"，其实是假引擎不认得那些题。
        """
        def scripted(prompt: str) -> str:
            n_done = prompt.count("<tool_response>")
            for t in pool:
                if t.instruction in prompt:
                    if n_done < len(t.actions):
                        a = t.actions[n_done]
                        args_ = json.dumps(a.kwargs, ensure_ascii=False)
                        return ('<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>'
                                % (a.name, json.dumps(args_, ensure_ascii=False)))
                    outs = getattr(t, "outputs", None) or []
                    return ("Here is the information you asked for: " + ", ".join(outs) + "."
                            if outs else "All set. Anything else I can help you with?")
            return "Sorry, could you repeat that?"
        return scripted

    pool = [it.task for it in build_splits(limit=args.limit)["covered"]] + \
           [it.task for it in build_splits(limit=args.limit)["uncovered"]] + \
           [it.task for it in build_splits(limit=args.limit)["unseen"]]
    scripted = make_scripted(pool)

    print("=" * 96)
    print("块 H5 · 评测 harness 自测（三分类口径，零模型）")
    print("=" * 96)
    fails = []

    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)

    # ① 分片构造
    splits = build_splits(limit=args.limit)
    for k, v in splits.items():
        print(f"   {k:<11} {len(v)} 题")
    check(len(splits["covered"]) == 31, f"covered = 31 道训练原题（实际 {len(splits['covered'])}）")
    check(len(splits["uncovered"]) > 0, f"uncovered 非空（{len(splits['uncovered'])}）")
    check(len(splits["unseen"]) > 0, f"unseen 非空（{len(splits['unseen'])}）")

    # ⭐ 分片之间必须**不相交** —— 否则"泄漏量"这个数没有意义
    ids = {k: {it.task_id for it in v} for k, v in splits.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                inter = ids[a] & ids[b]
                check(not inter, f"{a} ∩ {b} = ∅", f"交集 {len(inter)}")
    # covered 是原题、uncovered/unseen 是扩题，ID 空间天然隔开；
    # 真正的泄漏风险是**同一 (user, reservation)** 出现在两边，这里也查一下
    covered_users = {getattr(it.task, "user_id", "") for it in splits["covered"]}
    leak = [it.task_id for it in splits["uncovered"]
            if getattr(it.task, "user_id", "") in covered_users]
    check(not leak,
          f"uncovered 里没有和训练集撞 user 的题（撞了 {len(leak)} 道）",
          "撞 user 意味着'同一个人的另一张订单'，算不算泄漏是个判断，不是事实")

    # ② 照本宣科：三个分片都应该全对
    print("\n   照本宣科（假策略执行 gold）…")
    res_gold = evaluate(ScriptedEngine(scripted), n_samples=args.n,
                        limit=args.limit, tokenizer=tok, tools=tools)
    print(render(res_gold))
    print("\n   ⭐ 照本宣科没拿满分的，是**被长度上限掐掉的**还是**真做错了**？")
    for name in SPLITS:
        r = res_gold[name]
        if not r.n_tasks:
            continue
        n_over = r.terminated.get("context_overflow", 0)
        over_rate = n_over / max(1, r.n_samples)
        # 排除超长之后的通过率 —— 这才是"策略本身做对了吗"
        adjusted = r.pass_at_1 / max(1e-9, 1 - over_rate)
        check(adjusted > 0.95,
              f"{name}：排除超长后照本宣科全对（{adjusted:.3f}）",
              f"原始 {r.pass_at_1:.3f}，超长 {n_over}/{r.n_samples} = {over_rate:.1%}")
        if over_rate > 0:
            print(f"      ⚠️ {name} 有 {over_rate:.1%} 的**正确轨迹**因为超过 S_max=8192 被判 0 分")
            print(f"         这不是策略的错，是长度预算的错 —— 训练时它会变成奖励噪声")

    # ⭐ 金标准策略下"泄漏量"必须 ≈ 0 —— gold 不依赖有没有训练过。
    #    这条断言的意义：证明这个指标**测的是策略对训练集的依赖**，
    #    而不是别的东西（分片难度不均、引擎不认得新题之类）。
    cov_g, unc_g = res_gold["covered"], res_gold["uncovered"]
    if cov_g.n_tasks and unc_g.n_tasks:
        def adj(r):
            over = r.terminated.get("context_overflow", 0) / max(1, r.n_samples)
            return r.pass_at_1 / max(1e-9, 1 - over)

        gap_raw = cov_g.pass_at_1 - unc_g.pass_at_1
        gap_adj = adj(cov_g) - adj(unc_g)
        # ⭐ 必须用**修正掉长度截断之后**的数比 —— 否则 covered 那 12.9% 的超长
        #    会凭空变成"负泄漏"，把指标变成噪声。
        check(abs(gap_adj) < 0.05,
              f"金标准策略下泄漏量 ≈ 0（修正后 {gap_adj:+.3f}）",
              f"未修正时是 {gap_raw:+.3f} —— 差值全部来自 covered 的 12.9% 超长截断")
        print("      ↑ 这条断言说明：**不修正长度截断，泄漏量指标会凭空多出十几个百分点的噪声**")

    # ③ 什么都不做：covered 应该全错（这些题 gold 有写动作），且零方差率 = 1
    print("\n   什么都不做…")
    noop = ScriptedEngine(lambda p: "I can't help with that right now.")
    res_noop = evaluate(noop, n_samples=args.n, limit=args.limit,
                        tokenizer=tok, tools=tools)
    print(render(res_noop))
    cov = res_noop["covered"]
    check(cov.pass_at_1 < 0.1, f"covered：什么都不做 pass@1 ≈ 0（{cov.pass_at_1:.3f}）")
    check(cov.zero_var_rate == 1.0,
          f"covered：零方差率 = 1.0（{cov.zero_var_rate:.2f}）",
          "全错 → std=0 → advantage 全 0 → **这一步训练是白跑的**")

    print("\n" + "=" * 96)
    if fails:
        print(f"❌ harness 自测未通过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ 评测 harness 自测全绿（三分类口径可用、分片不相交、指标算得出）")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
