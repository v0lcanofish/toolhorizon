# -*- coding: utf-8 -*-
"""
四臂消融 · 对照表 + 对比图。

    python scripts/analyze_arms.py                      # 自动找 runs/ 下的四个臂
    python scripts/analyze_arms.py --rounds 15          # 只比前 15 轮（臂跑的轮数不同也能比）
    python scripts/analyze_arms.py --runs-root runs     # 指定 runs 目录

四个臂（名字以实际目录为准，缺哪个就跳过哪个）：
    ① arm_vanilla     length_norm=mean   ds=0
    ② arm_ds          length_norm=mean   ds=K
    ③ vanilla         length_norm=lata   ds=0    ⚠️ **目录名叫 vanilla 但开了 LATA**
    ④ arm_lata_ds     length_norm=lata   ds=K

━━━ 为什么要有「有效更新轨迹数」这一列 ━━━

该项目 GRPO 是**每条轨迹一次 optimizer.step()**、分母是**该轨迹自己的长度**，
所以零 advantage 的轨迹 = 一次**空转**的 step（梯度恒为 0）。
于是"这一轮真正学到了多少"不能用 loss 看，得看：

    有效更新轨迹数 = (1 − 零方差率) × 组数 × 每组条数

基线臂**稳态约 8–10 条/轮（≈1 组/轮）** —— 248 次 step 里约 240 次是空转。
（⚠️ 2026-09-20 订正：本文件初版写的是「~1 条/轮 —— 247 次空转」，**量级错了 10 倍** ——
 把 31 组里的 1 组错写成了 248 条里的 1 条。实测 arm_vanilla 末轮 (1−0.968)×31×8 = 8 条。）
这正是 9/17 调研里那句「把更新集中在很小且有偏的子集上」的可测版本。

⚠️ 图上标签用英文：本机 matplotlib 没有 CJK 字体，中文会渲染成豆腐块
   （而且 savefig **只 warning 不报错** —— 图"看起来就是那样"）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]

# 顺序 = 展示顺序。第一个是基准，后面都和它比。
ARMS = [
    ("arm_vanilla", "vanilla",  "mean", "ds=0"),
    ("arm_ds",      "mean+DS",  "mean", "ds=K"),
    ("vanilla",     "LATA",     "lata", "ds=0  ← 目录名不反映配置"),
    ("arm_lata_ds", "LATA+DS",  "lata", "ds=K"),
]


def pct(x) -> str:
    return "  ——  " if x is None else f"{x*100:.1f}%"


def load_arm(runs_root: Path, name: str):
    """读一个臂的逐轮 summary。找不到就返回 None（臂还没跑 / 目录名对不上）。"""
    d = runs_root / name / "rollouts"
    if not d.is_dir():
        return None
    out = []
    for f in sorted(d.glob("step_*.summary.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            continue
        s["step"] = int(re.search(r"step_(\d+)", f.name).group(1))
        s["eff_updates"] = eff_updates(s)       # ⭐ 主指标，塞进行里好统一取
        out.append(s)
    return sorted(out, key=lambda s: s["step"]) or None


def col(rows, k, default=None):
    return [r.get(k, default) for r in rows]


def eff_updates(s):
    """
    这一轮**真正产生梯度**的轨迹数。
    = (1 − 零方差率) × 组数 × 每组条数 —— 与臂无关，四臂可直接比。
    """
    zg, ng = s.get("zero_var_rate"), s.get("n_group")
    n_grp = s.get("n_groups") or s.get("n_tasks")
    if zg is None or not ng or not n_grp:
        return None
    return (1.0 - zg) * n_grp * ng


def tail_mean(rows, k, n=8):
    vals = [r[k] for r in rows[-n:] if r.get(k) is not None]
    return st.mean(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(_PROJECT / "runs"))
    ap.add_argument("--rounds", type=int, default=0,
                    help="只用前 N 轮（0 = 各臂全用）。臂轮数不同时必须指定，否则不可比")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    runs_root = Path(a.runs_root)

    data = {}
    for name, label, ln, note in ARMS:
        rows = load_arm(runs_root, name)
        if rows is None:
            print(f"   (跳过 {name} —— {runs_root / name} 下没有 step_*.summary.json)")
            continue
        if a.rounds:
            rows = [r for r in rows if r["step"] < a.rounds]
        data[name] = (label, ln, note, rows)

    if not data:
        raise SystemExit(f"{runs_root} 下没找到任何臂 —— 先跑 scripts/run_arms.sh")

    n_now = {k: len(v[3]) for k, v in data.items()}
    common = min(n_now.values())
    print("=" * 104)
    print(f"四臂消融 ｜ runs-root={runs_root}")
    print(f"各臂轮数：{n_now} ｜ **按最少的 {common} 轮对齐**（臂跑完的轮数不同时，"
          f"拿长的比短的是作弊）")
    print("=" * 104)

    def f2(v):
        return "   ——  " if v is None else f"{v:.2f}"

    # ---------------- 主表
    print(f"\n{'臂':<14}{'配置':<11}{'轮数':>5}{'零方差(稳态)':>13}{'通过率(稳态)':>13}"
          f"{'工具调用(稳态)':>15}{'有效更新/轮':>12}{'通过率(全程)':>13}")
    print("-" * 104)
    for name, (label, ln, note, rows) in data.items():
        rows = rows[:common]
        print(f"{name:<14}{label:<11}{len(rows):>5}"
              f"{pct(tail_mean(rows,'zero_var_rate')):>13}"
              f"{pct(tail_mean(rows,'pass_rate')):>13}"
              f"{f2(tail_mean(rows,'tool_calls_mean')):>15}"
              f"{f2(tail_mean(rows,'eff_updates')):>12}"
              f"{pct(tail_mean(rows,'pass_rate',n=len(rows))):>13}")

    # ---------------- 工具调用相对 gold
    gold, g = None, None
    ref = _PROJECT / "data" / "grpo_vanilla_steps.json"
    if ref.exists():
        try:
            gold = json.loads(ref.read_text(encoding="utf-8")).get("_gold_reference", {})
        except Exception:                                       # noqa: BLE001
            pass
    if gold and gold.get("tool_calls_mean"):
        g = gold["tool_calls_mean"]
        print(f"\n工具调用相对 gold（均值 {g} / 中位 {gold.get('tool_calls_median')}）：")
        for name, (label, ln, note, rows) in data.items():
            rows = rows[:common]
            v = tail_mean(rows, "tool_calls_mean")
            if v is None:
                continue
            side = "过调用" if v > g else "**欠调用**"
            print(f"   {name:<14}{v:>6.2f}  = gold 的 {v/g:.2f}×   {side}")

    # ---------------- 末轮快照（防"窗口均值掩盖末值"）
    # ⚠️ 2026-09-20 加：工具调用尤其危险 —— 四臂都从 ~16 一路掉穿 gold，
    #    用「后 8 轮均值」读会得出「arm_vanilla 还在过调用侧」，
    #    看末轮才发现它掉得比谁都深。窗口均值会给出**反号**的结论。
    print("\n末轮快照（⚠️ 上表的『稳态』是窗口均值，会掩盖末值 —— 工具调用上两者能给出反号结论）：")
    print(f"   {'臂':<14}{'末轮':>5}{'零方差':>9}{'通过率':>9}{'工具调用':>10}{'vs gold':>10}"
          f"{'有效更新':>10}")
    for name, (label, ln, note, rows) in data.items():
        rows = rows[:common]
        if not rows:
            continue
        L = rows[-1]
        v = L.get("tool_calls_mean")
        ratio = f"{v / g:.2f}×" if (v is not None and gold and gold.get("tool_calls_mean")) else "  —"
        print(f"   {name:<14}{L['step']:>5}{pct(L.get('zero_var_rate')):>9}"
              f"{pct(L.get('pass_rate')):>9}{f2(v):>10}{ratio:>10}{f2(eff_updates(L)):>10}")

    # ---------------- DS 记账
    dsed = {k: v for k, v in data.items() if any(s.get("ds_k") for s in v[3])}
    if dsed:
        print("\nDS 记账（加采的代价 —— 这一列本身就是 E-E「过滤的代价」的实测）：")
        print(f"   {'臂':<14}{'目标 K':>7}{'加采遍数':>9}{'采样放大':>9}{'进训练组数':>11}"
              f"{'第一遍零方差':>13}")
        def last(rows, k, spec):
            v = rows[-1].get(k) if rows else None
            return "  —" if v is None else format(v, spec)

        for name, (label, ln, note, rows) in dsed.items():
            rows = rows[:common]
            print(f"   {name:<14}{last(rows,'ds_k','.0f'):>7}{last(rows,'ds_passes','.1f'):>9}"
                  f"{last(rows,'ds_amplify','.2f') + '×' if rows and rows[-1].get('ds_amplify') is not None else '  —':>9}"
                  f"{last(rows,'ds_train_groups','.1f'):>11}"
                  f"{pct(tail_mean(rows,'ds_pass0_zero_var_rate')):>13}")
        print("   ⭐『第一遍零方差』是**和基线臂同口径**的数（同一批题、只采一遍）——")
        print("      用它证明四个臂面对的题目难度是一样的，差异只来自算法。")

    # ---------------- 判据
    print("\n" + "=" * 104)
    print("怎么读这张表")
    print("=" * 104)
    print("  · **有效更新/轮**是这张表的主角：基线臂稳态约 8–10 条，任何臂只要把它抬起来就是有效的。")
    print("  · 🆕 **先看『末轮快照』再看主表**：主表是**窗口均值**，会掩盖末值。工具调用上两者")
    print("    能给出**反号**结论（例：arm_vanilla 窗口均值 1.03× 像在过调用侧，末轮 0.51× 其实掉最深）。")
    print("  · **零方差率**四臂若都在 0.9+ → 说明病根不在算法轴，在**难度校准**（31 道题")
    print("    对当前策略太难），四个臂只是四条平线。")
    print("  · **工具调用**看它落在 gold 的哪一侧 —— 过调用/欠调用是创新点的落点。")
    print("  · 单次运行、单 seed、1.5B ⇒ 曲线上的小波动不能当趋势读。")

    if not a.no_fig:
        _figure(data, common, gold, _PROJECT / "reports" / "figs")
    return 0


def _figure(data, common, gold, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    colors = ["#c0392b", "#2471a3", "#7d3c98", "#1e8449"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    series = {name: v[3][:common] for name, v in data.items()}

    ax = axes[0][0]
    for (name, (label, ln, note, _)), c in zip(data.items(), colors):
        r = series[name]
        ax.plot(col(r, "step"), col(r, "zero_var_rate"), "o-", color=c, lw=1.8, ms=3,
                label=f"{name} ({label})")
    ax.set_ylim(0.6, 1.03)
    ax.set_title("(1) Zero-variance rate  UP = worse", fontsize=12, weight="bold")
    ax.set_ylabel("fraction of groups with no gradient")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[0][1]
    for (name, (label, ln, note, _)), c in zip(data.items(), colors):
        r = series[name]
        ax.plot(col(r, "step"), col(r, "tool_calls_mean"), "o-", color=c, lw=1.8, ms=3,
                label=name)
    if gold and gold.get("tool_calls_mean"):
        ax.axhline(gold["tool_calls_mean"], ls="--", color="green",
                   label=f"gold {gold['tool_calls_mean']}")
        ax.axhline(gold["tool_calls_median"], ls=":", color="green", alpha=.6)
    ax.set_title("(2) Tool calls  -- which side of gold?", fontsize=12, weight="bold")
    ax.set_ylabel("mean tool calls")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[1][0]
    for (name, (label, ln, note, _)), c in zip(data.items(), colors):
        r = series[name]
        ax.plot(col(r, "step"), col(r, "pass_rate"), "o-", color=c, lw=1.8, ms=3,
                label=name)
    ax.set_title("(3) Pass rate", fontsize=12, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[1][1]
    for (name, (label, ln, note, _)), c in zip(data.items(), colors):
        r = series[name]
        y = [s.get("eff_updates") for s in r]
        ax.plot(col(r, "step"), y, "o-", color=c, lw=1.8, ms=3, label=name)
    ax.set_title("(4) Effective updates per round  (the real metric)",
                 fontsize=12, weight="bold")
    ax.set_ylabel("trajectories with non-zero advantage")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    for row in axes:
        for ax_ in row:
            ax_.set_xlabel("round")
    fig.suptitle("ToolHorizon / four-arm ablation", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = outdir / "fig_four_arms.png"
    fig.savefig(p, dpi=140)
    print(f"\n   图 → {p.relative_to(_PROJECT.parents[1])}")


if __name__ == "__main__":
    raise SystemExit(main())
