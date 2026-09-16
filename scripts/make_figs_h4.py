# -*- coding: utf-8 -*-
"""块 H4 可视化：观测器。""

跑法：D:/anaconda/python.exe scripts/make_figs_h4.py
"""

import json
import sys
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


def main() -> int:
    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 1.15], wspace=0.26)

    # ---------------- ① 10 个指标，分两类
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    ax0.set_xlim(0, 10); ax0.set_ylim(0, 10)
    ax0.set_title("① 盯 10 个指标（分两类）", fontsize=13, fontweight="bold", pad=10)

    group_a = [
        ("★ 组内零方差率", "第一监控指标"),
        ("reward 均值 / 标准差", ""),
        ("平均轮数 / 工具调用数", ""),
        ("轨迹长度 / 消息数", ""),
    ]
    group_b = [
        ("欠调用率", "该写却没写"),
        ("过调用率", "不该写却写了"),
        ("抖动指数", "对同一对象重复"),
        ("空转轮数", "无写任务上的白说"),
    ]
    group_c = [("KL 散度 / 策略熵", "训练时由训练脚本提供")]

    def block(y, title, items, ec, fc, tc):
        h = 0.62 * len(items) + 0.95
        ax0.add_patch(mp.FancyBboxPatch((0.2, y - h), 9.6, h,
                                        boxstyle="round,pad=0.12",
                                        fc=fc, ec=ec, lw=2.0))
        ax0.text(0.55, y - 0.5, title, fontsize=11.5, fontweight="bold",
                 color=tc, va="center")
        yy = y - 1.05
        for name, note in items:
            ax0.text(0.85, yy, "· " + name, fontsize=10.2, color="#263238", va="center")
            if note:
                ax0.text(6.1, yy, note, fontsize=9.2, color="#78909c", va="center")
            yy -= 0.62
        return y - h - 0.55

    y = 9.6
    y = block(y, "A· 训练健康度（从轨迹算）", group_a, "#1565c0", "#e3f2fd", "#0d47a1")
    y = block(y, "B· ★ 工具退化谱（本项目核心创新）", group_b, "#ef6c00", "#fff3e0", "#e65100")
    y = block(y, "C· 需要模型（留接口）", group_c, "#6a1b9a", "#f3e5f5", "#4a148c")

    # ---------------- ② 抖动定义被测试逼着改对
    ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
    ax1.set_title("② 测试逼出来的一个定义修正", fontsize=13, fontweight="bold", pad=10)

    ax1.text(0.3, 9.3, "真实 task[2] 的 gold 动作：", fontsize=10.5,
             color="#37474f", fontweight="bold", va="center")
    for k in range(5):
        ax1.add_patch(mp.FancyBboxPatch((0.5 + k * 1.85, 8.35), 1.7, 0.6,
                                        boxstyle="round,pad=0.06",
                                        fc="#00897b", ec="#00695c", lw=1.4))
        ax1.text(1.35 + k * 1.85, 8.65, "改航班", ha="center", va="center",
                 color="white", fontsize=9.5, fontweight="bold")
    ax1.text(5.0, 7.85, "5 次！但改的是 5 个【不同】预订 → 完全正确",
             ha="center", fontsize=10, color="#00695c", fontweight="bold")

    # 旧定义
    ax1.add_patch(mp.FancyBboxPatch((0.3, 5.5), 9.4, 1.9,
                                    boxstyle="round,pad=0.12",
                                    fc="#ffebee", ec="#c62828", lw=2))
    ax1.text(0.6, 7.0, "× 旧定义：只看工具名", fontsize=11, fontweight="bold",
             color="#c62828", va="center")
    ax1.text(0.6, 6.45, "(总调用数 − 不同工具数) / 总调用数", fontsize=9.8,
             color="#37474f", va="center")
    ax1.text(0.6, 5.95, "→ 判成 0.80 高抖动   可它明明是对的",
             fontsize=10.2, color="#c62828", fontweight="bold", va="center")

    # 新定义
    ax1.add_patch(mp.FancyBboxPatch((0.3, 3.1), 9.4, 1.9,
                                    boxstyle="round,pad=0.12",
                                    fc="#e8f5e9", ec="#2e7d32", lw=2))
    ax1.text(0.6, 4.6, "√ 新定义：看【工具名 + 参数】", fontsize=11,
             fontweight="bold", color="#2e7d32", va="center")
    ax1.text(0.6, 4.05, "重复签名数 / 总调用数", fontsize=9.8,
             color="#37474f", va="center")
    ax1.text(0.6, 3.55, "→ 5 个不同预订 = 5 个不同签名 → 抖动 0.00 ✓",
             fontsize=10.2, color="#2e7d32", fontweight="bold", va="center")

    ax1.text(5.0, 2.3, "★ 「抖动」的真正含义是\n对同一个对象反复做同一件事\n而不是同一个工具调了多次",
             ha="center", fontsize=11, color="#b71c1c", fontweight="bold")
    ax1.text(5.0, 0.9, "（这个是自测跑出来的 —— 光看代码想不到）",
             ha="center", fontsize=9.8, color="#78909c")

    # ---------------- ③ 仪表盘长什么样
    ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.set_title("③ 训练日志里长这样（mock 数据）", fontsize=13,
                  fontweight="bold", pad=10)

    log = (
        "── 训练健康度 ──────────────────\n"
        "  轨迹 10 条 ｜ 组 10 个\n"
        "  ★ 组内零方差率   1.000  ! 题太难\n"
        "  reward 均值/标准差 1.000 / 0.000\n"
        "  通过率             1.000\n"
        "  平均轮数 / 工具调用 8.80 / 1.60\n"
        "  终止方式  {user_stop: 9, tool: 1}\n"
        "── 工具退化谱 ──────────────────\n"
        "  欠调用率   0.000   (0/6)\n"
        "  过调用率   0.000   (0/4)\n"
        "  抖动指数   0.000\n"
        "  空转轮数   6.25"
    )
    ax2.add_patch(mp.FancyBboxPatch((0.2, 0.8), 9.6, 7.9,
                                    boxstyle="round,pad=0.15",
                                    fc="#263238", ec="#263238"))
    ax2.text(0.55, 8.3, log, fontsize=8.6, color="#b0bec5", va="top")
    ax2.text(5.0, 0.35, "★ 分母一并报出来 (0/6) —— 否则不知道在多少道上算的",
             ha="center", fontsize=9.5, color="#b71c1c", fontweight="bold")

    fig.suptitle("图 H4-1｜观测器：训练时盯什么、怎么算", fontsize=15,
                 fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：租卡一旦开始就在烧钱，不知道该看什么 = 跑完只有一堆 loss 数字，"
             "出不了「三条退化曲线」这个交付物。",
             ha="center", fontsize=11.5, color="#1b5e20", fontweight="bold")
    out = FIGS / "fig8_observer.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  √ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
