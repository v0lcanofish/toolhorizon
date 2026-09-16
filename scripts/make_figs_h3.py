# -*- coding: utf-8 -*-
"""
块 H3 可视化：多轮循环 + 零方差机制。

  fig7_rollout_zerovar.png

跑法：D:/anaconda/python.exe scripts/make_figs_h3.py  （纯 CPU，不需要 GPU）
"""

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    split = json.loads((PROJECT / "data" / "task_split.json").read_text(encoding="utf-8"))
    train = [r["task"] for r in split["train"]]
    probe = [r["task"] for r in split["probe_overcall"]]

    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.32)

    # ---------------- ① 多轮循环示意
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    ax0.set_xlim(0, 10); ax0.set_ylim(0, 10)
    ax0.set_title("① 一次 episode 长什么样", fontsize=13, fontweight="bold", pad=10)

    boxes = [
        ("system", "航空政策 + 工具定义", "#455a64", 8.85),
        ("user", "任务指令", "#1565c0", 7.55),
        ("assistant", "说话 / 调工具", "#2e7d32", 6.25),
        ("tool", "工具返回（role=tool）", "#6a1b9a", 4.95),
        ("user", "用户回话（role=user）", "#1565c0", 3.65),
        ("assistant", "收尾那句话", "#2e7d32", 2.35),
    ]
    for role, desc, c, y in boxes:
        ax0.add_patch(mp.FancyBboxPatch((0.6, y - 0.42), 8.6, 0.84,
                                        boxstyle="round,pad=0.08",
                                        fc=c, ec=c, lw=1.6))
        ax0.text(0.95, y, role, va="center", fontsize=10.5,
                 color="white", fontweight="bold")
        ax0.text(2.5, y, desc, va="center", fontsize=10, color="white")
    # 循环箭头
    ax0.annotate("", xy=(9.6, 6.25), xytext=(9.6, 3.65),
                 arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=2.2,
                                 connectionstyle="arc3,rad=-0.55"))
    ax0.text(9.85, 4.95, "循环", rotation=90, va="center", ha="center",
             fontsize=10.5, color="#c62828", fontweight="bold")
    ax0.text(5.0, 1.1, "★ 只有 assistant 那几句算 loss\n（user / tool 的都不算）",
             ha="center", fontsize=10.5, color="#1b5e20", fontweight="bold")

    # ---------------- ② reward 判据
    ax1 = fig.add_subplot(gs[1])
    cats = ["训练集 31 道\n（照本宣科）", "训练集 31 道\n（什么都不做）", "探针集 19 道\n（什么都不做）"]
    vals = [1.0, 0.0, 1.0]
    cols = ["#2e7d32", "#c62828", "#ef6c00"]
    bars = ax1.bar(cats, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.04,
                 f"{'满分' if v else '0 分'}", ha="center",
                 fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.32)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_ylabel("reward", fontsize=12)
    ax1.set_title("② reward 判据成立 50/50", fontsize=13, fontweight="bold", pad=10)
    ax1.tick_params(axis="x", labelsize=9.5)

    # ---------------- ③ 零方差机制（用卡片，因为两条 std=0 画成柱子看不见）
    ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.set_title("③ 零方差机制（项目核心命题）", fontsize=13, fontweight="bold", pad=10)

    cards = [
        ("8 道全对", "8 对 / 0 错", "std = 0.00", "advantage 全 0  →  零梯度",
         "#c62828", "#ffebee"),
        ("8 道全错", "0 对 / 8 错", "std = 0.00", "advantage 全 0  →  零梯度",
         "#c62828", "#ffebee"),
        ("4 对 4 错", "4 对 / 4 错", "std = 0.50", "advantage ≠ 0  →  有梯度",
         "#2e7d32", "#e8f5e9"),
    ]
    y = 8.6
    for name, dist, sd, adv, ec, fc in cards:
        ax2.add_patch(mp.FancyBboxPatch((0.2, y - 1.25), 9.6, 2.3,
                                        boxstyle="round,pad=0.12",
                                        fc=fc, ec=ec, lw=2.0))
        ax2.text(0.7, y + 0.62, name, fontsize=12, fontweight="bold", color=ec,
                 va="center")
        ax2.text(3.6, y + 0.62, dist, fontsize=10.5, color="#37474f", va="center")
        ax2.text(0.7, y - 0.10, sd, fontsize=11, fontweight="bold",
                 color="#37474f", va="center")
        ax2.text(3.6, y - 0.10, adv, fontsize=11, fontweight="bold",
                 color=ec, va="center")
        y -= 2.9

    ax2.text(5.0, 0.55, "全对和全错，advantage 都是 0",
             ha="center", fontsize=10.5, color="#b71c1c", fontweight="bold")

    fig.suptitle("图 H3-1｜多轮循环跑通 + 零方差机制实证（零模型 / 零 GPU）",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：全对和全错拿到的 advantage 都是 0 —— 这不是 bug，是 GRPO 的结构性质，"
             "也正是「没信号」的来源。",
             ha="center", fontsize=11.5, color="#b71c1c", fontweight="bold")
    
    out = FIGS / "fig7_rollout_zerovar.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  ✅ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
