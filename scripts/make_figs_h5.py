# -*- coding: utf-8 -*-
"""
块 H5 的出图脚本。

    python scripts/make_figs_h5.py

两张图，回答两个问题：
    图 H5-1  「接话的那个人」修好没有？         ← 左：新旧对照；右：它在整条链上的位置
    图 H5-2  「上卡之前，到底验了多少东西？」    ← 全链路自检的成绩单

数字全部来自真实脚本输出（reports/H5-自测输出-2026-09-17.md），不是编的。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "reports" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#263238"
MUTED = "#78909c"
BAD = "#c62828"
GOOD = "#2e7d32"
BLUE = "#1565c0"


# ================================================================ 图 1
def fig1():
    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)

    # ---------------- 左：新旧对照
    ax = fig.add_subplot(gs[0, 0])
    labels = ["答不上来\n糊弄过去", "关键信息\n答得出来", "该收尾时\n收尾"]
    old = [100.0, 0.0, 0.0]
    new = [0.2, 62.9, 100.0]
    x = range(len(labels))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], old, w, label="旧的占位版（哑巴）",
                color="#ef9a9a", edgecolor=BAD, linewidth=1.4)
    b2 = ax.bar([i + w / 2 for i in x], new, w, label="新的槽位版",
                color="#a5d6a7", edgecolor=GOOD, linewidth=1.4)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2.5,
                    f"{b.get_height():.1f}%", ha="center", fontsize=11.5,
                    fontweight="bold", color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0, 138)
    ax.set_ylabel("比例（%）", fontsize=11.5)
    ax.set_title("① 拿 568 对真实【AI 问 → 真人答】回放考它",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.legend(fontsize=11, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 1.0))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.text(2.62, 100, "越低越好", ha="center", fontsize=9.5, color=MUTED)
    ax.annotate("修好之前，AI 问什么\n都只得到「好的，请继续」",
                xy=(0 - w / 2, 101), xytext=(1.62, 126),
                fontsize=9.8, color=BAD, ha="center",
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))

    # ---------------- 右：为什么这件事致命
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("② 为什么它是致命的：让 AI 走不到答案",
                  fontsize=13.5, fontweight="bold", pad=12)

    steps = [
        ("顾客开口", "「我的账号是 aarav_garcia_1177，\n我想改我的航班」", "#e3f2fd", BLUE),
        ("AI 查了一下", "名下有两张订单：M05KNL、UHDAHF", "#e3f2fd", BLUE),
        ("AI 必须问一句", "「请问您要改哪一张？」", "#fff8e1", "#f9a825"),
        ("哑巴接话", "「好的，请继续。」  ← 没有答案", "#ffebee", BAD),
        ("结果", "AI 永远拿不到订单号 → 这道题【任何模型都解不了】", "#ffebee", BAD),
    ]
    y = 8.9
    for i, (head, body, bg, ec) in enumerate(steps):
        h = 1.32
        ax2.add_patch(FancyBboxPatch((0.5, y - h), 9.0, h,
                                     boxstyle="round,pad=0.08,rounding_size=0.18",
                                     facecolor=bg, edgecolor=ec, linewidth=1.5))
        ax2.text(0.85, y - 0.42, head, fontsize=10.8, fontweight="bold",
                 color=ec, va="center")
        ax2.text(0.85, y - 0.96, body, fontsize=10.0, color=INK, va="center")
        y -= h + 0.22

    ax2.text(0.5, 0.62,
             "训练集 31 道题里，有 12 道（38.7%）第一步就要问顾客要信息。\n"
             "这 12 道题会一直拿 0 分 → 组内全错 → 梯度恒为 0。\n"
             "更危险的是：跑出来的成绩很难看，会让人以为是【模型不行】。",
             fontsize=10.2, color=BAD, va="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3e0",
                       edgecolor=BAD, linewidth=1.3))

    fig.suptitle("图 H5-1 · 会说话的用户模拟器 —— 把「任何模型都解不了」的题救回来",
                 fontsize=15.5, fontweight="bold", y=0.985)
    p = OUT / "fig9_user_sim.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("→", p)


# ================================================================ 图 2
def fig2():
    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

    # ---------------- 左：自检成绩单
    ax = fig.add_subplot(gs[0, 0])
    rows = [
        ("全链路彩排\neval_gpu_pipeline.py", 16, "collect -> jsonl -> train -> adapter 接得上"),
        ("分组采样\neval_train_logic.py", 29, "两路算法算 loss mask 逐 token 对账"),
        ("训练器\ntrainer.py selftest", 14, "零方差 -> 零梯度；A 取反损失取反"),
        ("训练步\neval_train_step.py", 13, "分块 CE 与官方实现误差 0.00e+00"),
        ("评测 harness\nobserve.harness", 13, "三分类口径、分片不相交"),
        ("用户模拟器\neval_user_sim.py", 5, "拿 568 对真实对话回放"),
    ]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    y = range(len(rows))
    ax.barh(list(y), vals, color="#90caf9", edgecolor=BLUE, linewidth=1.3, height=0.62)
    for i, (n, v, note) in enumerate(rows):
        ax.text(v + 0.6, i, f"{v} 条", va="center", fontsize=11.5,
                fontweight="bold", color=INK)
        ax.text(0.5, i - 0.36, note, va="center", fontsize=9.3, color=MUTED)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 35)
    ax.set_xlabel("通过的断言条数（全部 0 失败）", fontsize=11)
    ax.set_title("① 上卡之前，本地已经验了 90 条断言",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------- 右：钱花在哪
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("② 为什么值得先验：钱只花在「真结果」上",
                  fontsize=13.5, fontweight="bold", pad=12)

    blocks = [
        ("本地 CPU（已花）", "0 元", "90 条断言 / 6 个自检脚本\n全部逻辑、格式、数值对账", GOOD, "#e8f5e9"),
        ("上卡 smoke（下一步）", "约 ¥1–3", "30 分钟：显存、速度、峰值显存\n→ 拿到四个数就关机", "#f9a825", "#fff8e1"),
        ("正式训练", "约 ¥50–150", "30 GPU 时，分 3 次租\n每次只跑真实验、不调代码", BLUE, "#e3f2fd"),
    ]
    y = 8.7
    for head, cost, body, ec, bg in blocks:
        h = 2.15
        ax2.add_patch(FancyBboxPatch((0.4, y - h), 9.2, h,
                                     boxstyle="round,pad=0.1,rounding_size=0.2",
                                     facecolor=bg, edgecolor=ec, linewidth=1.6))
        ax2.text(0.8, y - 0.5, head, fontsize=11.5, fontweight="bold", color=ec, va="center")
        ax2.text(9.2, y - 0.5, cost, fontsize=12.5, fontweight="bold",
                 color=ec, va="center", ha="right")
        ax2.text(0.8, y - 1.45, body, fontsize=10.0, color=INK, va="center")
        y -= h + 0.35

    fig.suptitle("图 H5-2 · 上卡前的准备 —— 把「只有上了卡才会暴露」的问题提前挡掉",
                 fontsize=15.5, fontweight="bold", y=0.985)
    p = OUT / "fig10_h5_readiness.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("→", p)


if __name__ == "__main__":
    fig1()
    fig2()
