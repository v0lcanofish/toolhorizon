# -*- coding: utf-8 -*-
"""
四臂消融上卡前的**离线自测** —— 纯 CPU、不碰 GPU、不花钱。

    为什么必须先跑这个：
      租卡是按小时烧钱的。四臂要跑 20+ 小时，**如果在卡上才发现 DS 的循环写错了，
      那是在烧钱调 bug**，而且三个臂的结果全部作废要重跑。

    这里用一个**注入的假采样器**把 DS 的循环整个跑一遍（不加载模型、不跑 τ-bench），
    把能提前发现的问题全挡在租卡之前。

验六件事：
    ① DS 关掉（--ds 0）= 基线，逐字节不变（防"关掉 DS 却偷偷改了行为"）
    ② DS 加采确实能攒够 K 个有方差的组
    ③ 到顶保护：攒不够也不会死循环（防采样成本失控）
    ④ ⭐ 同一道题加采两遍**不能并组**  ← 最容易静默出错的一条
    ⑤ summary 口径：zero_var_rate 按"采到的全部"算，不是按进训练的那批
    ⑥ S_max 不写死：能从 summary 读出来，也能被命令行覆盖

跑法（纯 CPU）：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/selftest_ds.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

from train.engine import GenStats                                          # noqa: E402
from train.rollout_batch import (                                          # noqa: E402
    GroupResult, RolloutBatch, compute_advantages, run_ds_passes, save_rollouts,
)
from train.tokenize import load_tokenizer                                  # noqa: E402
from train.trainer import batches_from_records                             # noqa: E402

OK, FAIL = "[OK]", "[!!]"
_fails = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    print(f"   {OK if cond else FAIL} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond


# ---------------------------------------------------------------- 假零件


class _Ep:
    """Episode 的最小替身 —— summary() 只读这几个字段。"""

    def __init__(self, reward: float):
        self.reward = reward
        self.n_turns = 3
        self.n_tool_calls = 2
        self.terminated_by = "user_stop"


class _Sample:
    """Trajectory 的最小替身 —— save_rollouts 只读这几个字段。"""

    def __init__(self, idx: int, reward: float, msgs):
        self.sample_idx = idx
        self.n_malformed = 0
        self.prompt_tokens = 10
        self.msgs = msgs
        self.ep = _Ep(reward)


def make_batch(task_ids, rewards_by_task, pass_idx: int, msgs):
    """照 RolloutBatch 的形状造一批，advantage 用**真函数**算（不手写假值）。"""
    groups = []
    for tid in task_ids:
        rewards = rewards_by_task[tid]
        advs, sd, zero = compute_advantages(rewards)
        samples = [_Sample(i, r, msgs) for i, r in enumerate(rewards)]
        groups.append(GroupResult(
            task_id=tid, rewards=rewards, advantages=advs, std=sd,
            zero_variance=zero, episodes=[s.ep for s in samples],
            samples=samples, pass_idx=pass_idx,
        ))
    return RolloutBatch(groups=groups, gen_stats=GenStats())


class FakeEngine:
    def stats(self):
        return GenStats()


def make_rollout_fn(pass0_var: int, later_var: int, n_tasks: int = 6, n_group: int = 4):
    """
    假采样器：第 0 遍有 pass0_var 个组有方差，之后每遍有 later_var 个。
    这样 DS 循环要走几遍是**确定的**，可以直接对账。
    """
    state = {"i": 0, "msgs": None}

    def fn(task_items, engine, cfg, tokenizer, tools, pass_idx=0):
        n_var = pass0_var if pass_idx == 0 else later_var
        rewards = {}
        for j in range(n_tasks):
            tid = 100 + j
            if j < n_var:
                # 有方差：一半对一半错 → std>0
                rewards[tid] = [1.0, 0.0] * (n_group // 2) if n_group % 2 == 0 \
                    else [1.0] * (n_group // 2) + [0.0] * (n_group - n_group // 2)
            else:
                rewards[tid] = [0.0] * n_group       # 全错 = 零方差
        state["i"] += 1
        return make_batch(sorted(rewards), rewards, pass_idx, state["msgs"])

    fn.state = state
    return fn


# ---------------------------------------------------------------- 主流程


def main() -> int:
    print("=" * 78)
    print("四臂消融 · DS 循环离线自测（纯 CPU）")
    print("=" * 78)

    # 用真 SFT 数据里的消息结构 —— 自己编一个 {system,user} 的短列表有风险，
    # tokenize.encode 是照着 SFT 的格式写的（trainer 自测也是这么取样本的）。
    sft = PROJECT / "data" / "sft_final.jsonl"
    msgs = json.loads(sft.read_text(encoding="utf-8").splitlines()[0])["messages"]

    # ---------------- ① DS 关掉 = 基线
    print("\n① DS 关掉（k=0）= 基线：只采一遍，一条不筛")
    fn = make_rollout_fn(pass0_var=2, later_var=2)
    fn.state["msgs"] = msgs
    per_pass, kept, passes = run_ds_passes(
        [(100, None)], FakeEngine(), None, None, None, k=0, max_pass=6, rollout_fn=fn)
    check(passes == 1, f"只采了 1 遍（实测 {passes}）")
    check(len(per_pass) == 1, f"per_pass 只有 1 批（实测 {len(per_pass)}）")
    check(fn.state["i"] == 1, f"假采样器只被调用 1 次（实测 {fn.state['i']}）",
          "k=0 时绝不能偷偷加采 —— 否则基线臂就不可比了")
    all_g = [g for b in per_pass for g in b.groups]
    check(len(all_g) == 6, f"采到 6 个组（实测 {len(all_g)}）")
    check(len(kept) == 2, f"其中有方差的 2 个（实测 {len(kept)}）",
          "k=0 时 kept 只是**记账**，不影响存盘")

    # ---------------- ② 加采到够 K
    print("\n② DS 加采：第 0 遍 2 个、之后每遍 1 个，K=4 → 应当采 3 遍")
    fn = make_rollout_fn(pass0_var=2, later_var=1)
    fn.state["msgs"] = msgs
    per_pass, kept, passes = run_ds_passes(
        [(100, None)], FakeEngine(), None, None, None, k=4, max_pass=6, rollout_fn=fn)
    check(passes == 3, f"采了 3 遍（实测 {passes}）",
          "2 + 1 + 1 = 4，正好够 K")
    check(len(kept) == 4, f"攒到 4 个有方差的组（实测 {len(kept)}）")

    # ---------------- ③ 到顶保护
    print("\n③ 到顶保护：永远攒不够也不会死循环（防采样成本失控）")
    fn = make_rollout_fn(pass0_var=0, later_var=0)      # 一个都攒不到
    fn.state["msgs"] = msgs
    per_pass, kept, passes = run_ds_passes(
        [(100, None)], FakeEngine(), None, None, None, k=4, max_pass=5, rollout_fn=fn)
    check(passes == 5, f"停在 max_pass=5 而不是无限采（实测 {passes}）")
    check(len(kept) == 0, f"一个都没攒到（实测 {len(kept)}）",
          "卡上会打印警告，说明这批题对当前策略没有中间地带")

    # ---------------- ④ ⭐ 同一道题加采两遍不能并组
    print("\n④ ⭐ 同一道题加采两遍**不能并组**（最容易静默出错的一条）")
    print("      p0 全对（零方差）｜ p1 = [1,0,0,0]（有方差）")
    # 第 0 遍：task 77 全对；第 1 遍：task 77 一半一半
    p0 = make_batch([77], {77: [1.0, 1.0, 1.0, 1.0]}, pass_idx=0, msgs=msgs)
    p1 = make_batch([77], {77: [1.0, 0.0, 0.0, 0.0]}, pass_idx=1, msgs=msgs)
    both = RolloutBatch(groups=p0.groups + p1.groups, gen_stats=GenStats())
    tmp = PROJECT / "data" / "_tmp_selftest_ds.jsonl"
    n_written = save_rollouts(both, tmp)
    check(n_written == 8, f"落盘 8 条（实测 {n_written}）")

    recs = [json.loads(l) for l in tmp.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = {r.get("group_key") for r in recs}
    check(len(keys) == 2, f"产出 2 个不同的 group_key（实测 {len(keys)}：{sorted(keys)}）",
          "缺了 group_key 的话两遍会共用 task_id=77")

    tok = load_tokenizer()
    bs, advs = batches_from_records(recs, tok, [], max_seq_tokens=128)
    check(len(bs) == 8, f"trainer 侧仍然是 8 个 batch（实测 {len(bs)}）")
    # 关键断言：把 8 条按 pass 分开看，p0 那 4 条的 advantage **必须全 0**
    n_p0 = sum(1 for r in recs if r["group_key"].endswith("#p0"))
    a_p0 = [a for r, a in zip(recs, advs) if r["group_key"].endswith("#p0")]
    a_p1 = [a for r, a in zip(recs, advs) if r["group_key"].endswith("#p1")]
    check(all(abs(a) < 1e-12 for a in a_p0),
          f"p0 那 {n_p0} 条的 advantage 全 0（零方差组就该没梯度）",
          f"{[round(a, 3) for a in a_p0]}")
    check(any(abs(a) > 1e-12 for a in a_p1),
          f"p1 那 {len(a_p1)} 条有非零 advantage", f"{[round(a, 3) for a in a_p1]}")
    print("      ⭐ 若两遍被并成一组（16→8 条一起算 advantage），p0 会**白捡**到非零优势 ——")
    print("         零方差的那一遍被'洗白'成有信号，而 loss 曲线完全看不出来。")
    tmp.unlink(missing_ok=True)

    # ---------------- ⑤ summary 口径
    print("\n⑤ summary 口径：zero_var_rate 按**采到的全部**算")
    sampled = RolloutBatch(groups=[g for b in [p0, p1] for g in b.groups],
                           gen_stats=GenStats())
    train_only = RolloutBatch(groups=[g for g in sampled.groups if not g.zero_variance],
                              gen_stats=GenStats())
    check(abs(sampled.zero_var_rate - 0.5) < 1e-9,
          f"按采到的全部算 = 0.5（实测 {sampled.zero_var_rate}）")
    check(abs(train_only.zero_var_rate) < 1e-9,
          f"若按进训练的那批算 = 0.0（实测 {train_only.zero_var_rate}）",
          "← 这就是必须先定口径的原因：DS 臂会恒等于 0，和基线没法比")

    # ---------------- ⑥ S_max 不写死
    print("\n⑥ S_max 不写死：从 summary 读 / 命令行覆盖 / 都没有就明说")
    import analyze_rollouts as ar                                   # noqa: E402
    d = PROJECT / "data" / "_tmp_selftest_smax"
    (d.with_suffix(".jsonl")).write_text("", encoding="utf-8")
    (d.with_suffix(".summary.json")).write_text(
        json.dumps({"max_seq_tokens": 16384}), encoding="utf-8")
    check(ar.resolve_s_max(d.with_suffix(".jsonl")) == 16384,
          "从 summary.json 读到 16384（= 9/18 那轮的真实值，不是写死的 8192）")
    check(ar.resolve_s_max(d.with_suffix(".jsonl"), cli=8192) == 8192,
          "命令行 --s-max 能覆盖")
    d.with_suffix(".summary.json").unlink(missing_ok=True)
    check(ar.resolve_s_max(d.with_suffix(".jsonl")) == 8192,
          "字段缺失时退回 8192 **并打印警告**（不静默用错的数）")
    d.with_suffix(".jsonl").unlink(missing_ok=True)

    # ---------------- ⑦ ReplayEpisode 与真 Episode 的字段同步
    print("\n⑦ ReplayEpisode 与 env.rollout.Episode 字段同步（防静默漂移）")
    print("      replay 用的是自造的最小替身（为了不把 τ-bench 依赖拖进分析脚本），")
    print("      ⚠️ 所以 Episode 加了字段时这里**不会报错**，只会算出错的指标。用断言兜住。")
    from dataclasses import fields as _fields
    from observe.replay import ReplayEpisode
    from env.rollout import Episode as RealEpisode
    rf = {f.name for f in _fields(ReplayEpisode)}
    ef = {f.name for f in _fields(RealEpisode)}
    NEEDED = {"task_id", "messages", "reward", "n_turns", "n_tool_calls", "terminated_by"}
    check(NEEDED <= rf, f"替身覆盖了 compute() 读的全部字段（{sorted(NEEDED)}）",
          f"缺 {sorted(NEEDED - rf)}" if NEEDED - rf else "")
    # `done` 是 env 的运行态标记（是否正常终止），观测器不读它 —— **显式列出来**，
    # 免得以后 Episode 真加了别的字段也被这条断言放过。
    ALLOWED_GAP = {"done"}
    gap = ef - rf - ALLOWED_GAP
    check(not gap, "两边没有意外差异", f"新多出来：{sorted(gap)} —— "
          f"要么补进 ReplayEpisode，要么确认观测器不用它" if gap else "只差 done（已登记）")

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 自测未通过（{len(_fails)} 项）：")
        for f in _fails:
            print(f"   · {f}")
        print("⛔ 别上卡 —— 上卡只会把同样的错烧成 GPU 小时。")
        return 1
    print("✅ 全部通过。可以上卡跑四臂了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
