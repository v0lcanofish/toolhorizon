# -*- coding: utf-8 -*-
"""
块 H7 · 四臂消融的**起点 vs 末值**对比图（给报告 §二/§六 配图）。

    python scripts/make_figs_h7.py
    → reports/figs/fig_h7_start_vs_final.png

━━━ 为什么单独做这一张（`analyze_arms.py` 已经有 fig_four_arms.png 了）━━━

那张是**逐轮曲线**，能看形状但看不出"四条臂各自停在哪"。
报告里最硬的两个数恰恰是**末值**：

  · 工具调用停在 gold 的哪一侧（`arm_ds` 1.00× 是唯一一条没掉穿的）
  · 有效更新数被托到了多少（基线衰减到 8 条，DS 臂 32–48）

而**窗口均值会给出反号结论**（见 analyze_arms.py 的「末轮快照」一节），
所以这张图**只用两个点**：起点（step 0）和末值（四臂对齐的最后一轮），
中间不平均、不插值 —— 免得又被均值骗一次。

⚠️ 图上标签用英文：本机 matplotlib 没有 CJK 字体，中文会渲染成豆腐块
   （savefig 只 warning 不报错 —— 图"看起来就是那样"）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]

# 展示顺序 = analyze_arms.py 的 ARMS 顺序，别乱
ARMS = [
    ("arm_vanilla", "vanilla (÷L)"),
    ("arm_ds",      "DS (÷L)"),
    ("vanilla",     "LATA (÷√L)"),
    ("arm_lata_ds", "LATA+DS (÷√L)"),
]
COLORS = ["#c0392b", "#2471a3", "#7d3c98", "#1e8449"]


def load_arm(name: str):
    d = _PROJECT / "runs" / name / "rollouts"
    if not d.is_dir():
        return None
    rows = []
    for f in sorted(d.glob("step_*.summary.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            continue
        s["step"] = int(re.search(r"step_(\d+)", f.name).group(1))
        rows.append(s)
    return sorted(rows, key=lambda s: s["step"]) or None


def eff_updates(s):
    """这一轮**真正产生梯度**的轨迹数 = (1 − 零方差率) × 组数 × 每组条数。"""
    zg, ng = s.get("zero_var_rate"), s.get("n_group")
    n_grp = s.get("n_groups") or s.get("n_tasks")
    if zg is None or not ng or not n_grp:
        return None
    return (1.0 - zg) * n_grp * ng


def main() -> int:
    data = {}
    for name, label in ARMS:
        rows = load_arm(name)
        if rows:
            data[name] = (label, rows)
        else:
            print(f"   (跳过 {name} —— 没有 step_*.summary.json)")

    if not data:
        raise SystemExit("runs/ 下没找到任何臂")

    # ⚠️ 四臂对齐到**最少轮**（arm_lata_ds 被容器重启打断，只有 12 轮）
    last = min(len(rows) for _, rows in data.values())
    print(f"四臂对齐到前 {last} 轮（末值取 step_{last - 1:03d}）")

    gold = None
    ref = _PROJECT / "data" / "grpo_vanilla_steps.json"
    if ref.exists():
        try:
            gold = json.loads(ref.read_text(encoding="utf-8")).get("_gold_reference", {})
        except Exception:                                       # noqa: BLE001
            pass
    g = gold.get("tool_calls_mean") if gold else None

    labels = [data[n][0] for n, _ in ARMS if n in data]
    names = [n for n, _ in ARMS if n in data]
    colors = [c for (n, _), c in zip(ARMS, COLORS) if n in data]

    tc0 = [data[n][1][0].get("tool_calls_mean") for n in names]
    tc1 = [data[n][1][last - 1].get("tool_calls_mean") for n in names]
    ef0 = [eff_updates(data[n][1][0]) for n in names]
    ef1 = [eff_updates(data[n][1][last - 1]) for n in names]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(names))
    w = 0.36

    # ---- (1) 工具调用：起点 vs 末值，gold 基准线
    ax1.bar(x - w / 2, tc0, w, label=f"round 0 (start)", color="#bdc3c7",
            edgecolor="black", linewidth=.6)
    ax1.bar(x + w / 2, tc1, w, label=f"round {last-1} (final)", color=colors,
            edgecolor="black", linewidth=.6)
    if g:
        ax1.axhline(g, ls="--", color="green", lw=1.8, zorder=5)
        ax1.text(len(names) - 0.45, g + 0.25, f"gold = {g}", color="green",
                 fontsize=10, weight="bold", ha="right")
    for xi, (a, b) in enumerate(zip(tc0, tc1)):
        if a is not None:
            ax1.text(xi - w / 2, a + 0.15, f"{a:.1f}", ha="center", fontsize=8.5)
        if b is not None:
            ax1.text(xi + w / 2, b + 0.15, f"{b:.1f}", ha="center", fontsize=8.5,
                     weight="bold")
            if g:
                ax1.text(xi + w / 2, b + 0.95, f"{b/g:.2f}×", ha="center",
                         fontsize=8.5, color=colors[xi], weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("mean tool calls per trajectory")
    ax1.set_title("(1) Tool calls: where did each arm STOP?\n"
                  "all four fell from ~16 — only DS stayed ON the gold line",
                  fontsize=11.5, weight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=.3)
    ax1.set_ylim(0, max([v for v in tc0 + tc1 if v is not None] + [g or 0]) * 1.25)

    # ---- (2) 有效更新数：起点 vs 末值
    ax2.bar(x - w / 2, ef0, w, label=f"round 0 (start)", color="#bdc3c7",
            edgecolor="black", linewidth=.6)
    ax2.bar(x + w / 2, ef1, w, label=f"round {last-1} (final)", color=colors,
            edgecolor="black", linewidth=.6)
    for xi, (a, b) in enumerate(zip(ef0, ef1)):
        if a is not None:
            ax2.text(xi - w / 2, a + 1.0, f"{a:.0f}", ha="center", fontsize=8.5)
        if b is not None:
            ax2.text(xi + w / 2, b + 1.0, f"{b:.0f}", ha="center", fontsize=8.5,
                     weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("trajectories with non-zero advantage")
    ax2.set_title("(2) Effective updates per round\n"
                  "248 optimizer.step() calls, ~240 of them idle in the baselines",
                  fontsize=11.5, weight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=.3)

    fig.suptitle("ToolHorizon / four-arm ablation — START vs FINAL "
                 f"(aligned to first {last} rounds)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _PROJECT / "reports" / "figs" / "fig_h7_start_vs_final.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"   图 → {out.relative_to(_PROJECT.parents[1])}")

    # 顺手把数打出来，免得图里看错
    print()
    print(f"   {'臂':<16}{'工具调用 起→末':>22}{'vs gold':>10}{'有效更新 起→末':>20}")
    for i, n in enumerate(names):
        r = f"{tc1[i]/g:.2f}×" if (g and tc1[i] is not None) else "—"
        print(f"   {labels[i]:<16}{tc0[i]:>10.2f} → {tc1[i]:<8.2f}{r:>10}"
              f"{ef0[i]:>11.0f} → {ef1[i]:<7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
