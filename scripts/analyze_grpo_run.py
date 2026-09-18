# -*- coding: utf-8 -*-
"""
GRPO 25 轮跑完之后的**逐轮曲线**与判读。

    python scripts/analyze_grpo_run.py                       # 默认读 data/grpo_vanilla_steps.json
    python scripts/analyze_grpo_run.py --steps some.json

━━━ 这个脚本要回答的三个问题 ━━━

① **零方差率往哪走**（项目的第一监控指标）
② **工具调用往哪走** —— 项目的核心命题是「工具使用退化谱」（欠调用 / 过调用 / 抖动），
   所以要看的是它**相对 gold 的位置**，不是绝对值大小
③ **变慢是环境问题还是模型问题** —— 用 new_tokens 和 tok_per_sec 拆开

⚠️ 图上用英文标签：本机 matplotlib 没有 CJK 字体，中文会渲染成豆腐块
   （而且 `savefig` **只 warning 不报错** —— 图"看起来就是那样"）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT = _PROJECT / "data" / "grpo_vanilla_steps.json"


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=str(DEFAULT))
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    d = json.loads(Path(a.steps).read_text(encoding="utf-8"))
    S = d["steps"]
    gold = d.get("_gold_reference", {})
    sft = d.get("_sft_start", {})
    n = len(S)

    def col(k):
        return [s[k] for s in S]

    print("=" * 96)
    print(f"GRPO 逐轮 ｜ {n} 轮 ｜ {d['_run']}")
    print("=" * 96)
    hdr = f"{'step':>4}{'零方差':>9}{'通过率':>9}{'工具调用':>10}{'轮数':>8}{'截断':>8}{'格式崩':>8}{'秒/条':>8}{'k tokens':>10}"
    print(hdr)
    print("-" * 96)
    for s in S:
        print(f"{s['step']:>4}{pct(s['zero_var_rate']):>9}{pct(s['pass_rate']):>9}"
              f"{s['tool_calls_mean']:>10.2f}{s['turns_mean']:>8.1f}"
              f"{pct(s['truncated_rate']):>8}{pct(s['malformed_rate']):>8}"
              f"{s['sec_per_trajectory']:>8.2f}{s['new_tokens']/1000:>10.0f}")

    # ---- 三段对照：把"开头被 SFT 惯性带偏"和"稳态"分开，别拿第一步当结论
    head, tail = S[:3], S[-8:]
    avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)

    print("\n" + "=" * 96)
    print("① ⭐ 零方差率 —— 项目的第一监控指标")
    print("=" * 96)
    print(f"   SFT 起点（GRPO 之前）      {pct(sft.get('zero_var_rate', float('nan')))}")
    print(f"   前 3 轮                    {pct(avg(head, 'zero_var_rate'))}")
    print(f"   后 8 轮（稳态）            {pct(avg(tail, 'zero_var_rate'))}")
    peak = max(S, key=lambda s: s["zero_var_rate"])
    print(f"   峰值                       {pct(peak['zero_var_rate'])}（step {peak['step']}）")
    print("   ⇒ 训练**没有**把零方差压下去，反而把它推高了。")

    print("\n" + "=" * 96)
    print("② ⭐⭐ 工具调用 —— 项目核心命题（工具使用退化谱）")
    print("=" * 96)
    print(f"   gold 真实成功轨迹          {gold.get('tool_calls_mean')}（中位 {gold.get('tool_calls_median')}）")
    print(f"   SFT 之后（GRPO 之前）      {sft.get('tool_calls_mean')}"
          f"   = gold 的 {sft.get('tool_calls_mean', 0)/gold.get('tool_calls_mean', 1):.2f} 倍 → **过调用**")
    print(f"   前 3 轮                    {avg(head, 'tool_calls_mean'):.2f}"
          f"   = gold 的 {avg(head, 'tool_calls_mean')/gold.get('tool_calls_mean', 1):.2f} 倍")
    print(f"   后 8 轮（稳态）            {avg(tail, 'tool_calls_mean'):.2f}"
          f"   = gold 的 {avg(tail, 'tool_calls_mean')/gold.get('tool_calls_mean', 1):.2f} 倍 → **欠调用**")

    print("\n" + "=" * 96)
    print("③ 变慢是环境问题还是模型问题")
    print("=" * 96)
    print(f"   new_tokens  {col('new_tokens')[0]/1000:.0f}k → {avg(tail,'new_tokens')/1000:.0f}k"
          f"   （×{avg(tail,'new_tokens')/col('new_tokens')[0]:.1f}）")
    print(f"   tok/sec     {col('tok_per_sec')[0]:.0f} → {avg(tail,'tok_per_sec'):.0f}"
          f"   （×{avg(tail,'tok_per_sec')/col('tok_per_sec')[0]:.2f}）← 基本没变 = 硬件没问题")
    print(f"   秒/条       {col('sec_per_trajectory')[0]:.2f} → {avg(tail,'sec_per_trajectory'):.2f}")
    print("   ⇒ 变慢 100% 来自「生成的 token 变多了」，不是机器慢了。")

    print("\n" + "=" * 96)
    print("⚠️ 一个被推翻的代理指标")
    print("=" * 96)
    print("   从 rollout 文件体积读到：11.16MB(step19) → 10.40MB(step21)，")
    print("   当时猜「过调用在收敛」。实测 tool_calls：", end="")
    print(f"{S[19]['tool_calls_mean']:.2f} → {S[21]['tool_calls_mean']:.2f} —— **基本平，没收敛**。")
    print("   ⭐ 体积反映的是 token 总量，不是工具调用次数：")
    print(f"      token 涨了 {avg(tail,'new_tokens')/col('new_tokens')[0]:.1f} 倍，"
          f"而工具调用**降了** {col('tool_calls_mean')[0]/avg(tail,'tool_calls_mean'):.1f} 倍。")

    if not a.no_fig:
        _figure(S, gold, sft, _PROJECT / "reports" / "figs")
    return 0


def _figure(S, gold, sft, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    x = [s["step"] for s in S]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0][0]
    ax.plot(x, [s["zero_var_rate"] for s in S], "o-", color="#c0392b", lw=2, ms=4)
    if "zero_var_rate" in sft:
        ax.axhline(sft["zero_var_rate"], ls="--", color="gray",
                   label=f"SFT start {sft['zero_var_rate']:.3f}")
        ax.legend()
    ax.set_title("(1) Zero-variance rate  UP = worse", fontsize=12, weight="bold")
    ax.set_ylabel("fraction of groups with no gradient")
    ax.set_ylim(0.6, 1.03)
    ax.grid(alpha=.3)

    ax = axes[0][1]
    ax.plot(x, [s["tool_calls_mean"] for s in S], "o-", color="#2471a3", lw=2, ms=4,
            label="GRPO")
    ax.axhline(gold["tool_calls_mean"], ls="--", color="green",
               label=f"gold mean {gold['tool_calls_mean']}")
    ax.axhline(gold["tool_calls_median"], ls=":", color="green", alpha=.6,
               label=f"gold median {gold['tool_calls_median']}")
    if "tool_calls_mean" in sft:
        ax.scatter([-0.8], [sft["tool_calls_mean"]], color="orange", zorder=5, s=70)
        ax.annotate("SFT\n(over-calling)", (-0.8, sft["tool_calls_mean"]),
                    textcoords="offset points", xytext=(12, -4), fontsize=9, color="orange")
    ax.set_xlim(-1.6, max(x) + 0.5)
    ax.set_title("(2) Tool calls per trajectory  -- over -> under", fontsize=12, weight="bold")
    ax.set_ylabel("mean tool calls")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    ax = axes[1][0]
    ax.plot(x, [s["pass_rate"] for s in S], "o-", color="#1e8449", lw=2, ms=4, label="pass rate")
    ax.plot(x, [s["truncated_rate"] for s in S], "s-", color="#b9770e", lw=1.6, ms=3,
            label="truncated")
    ax.plot(x, [s["turns_mean"] / 20 for s in S], "^-", color="#7d3c98", lw=1.4, ms=3,
            label="turns / 20")
    ax.set_title("(3) Pass rate flat  |  truncation UP", fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    ax = axes[1][1]
    ax.plot(x, [s["new_tokens"] / 1000 for s in S], "o-", color="#c0392b", lw=2, ms=4,
            label="new tokens (k)")
    ax.set_ylabel("new tokens (k) per round", color="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(x, [s["tok_per_sec"] for s in S], "s-", color="#566573", lw=1.6, ms=3,
             label="tok/sec")
    ax2.set_ylabel("tok/sec  (flat = hardware OK)", color="#566573")
    ax2.set_ylim(0, 1600)
    ax.set_title("(4) Slowdown = more tokens, not slower HW", fontsize=12, weight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.3)

    for row in axes:
        for ax_ in row:
            ax_.set_xlabel("GRPO round")
    fig.suptitle("ToolHorizon / runs/vanilla  --  GRPO 25 rounds (first %d shown)" % len(S),
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = outdir / "fig_grpo_vanilla_curve.png"
    fig.savefig(p, dpi=140)
    print(f"\n   图 → {p.relative_to(_PROJECT.parents[1])}")


if __name__ == "__main__":
    raise SystemExit(main())
