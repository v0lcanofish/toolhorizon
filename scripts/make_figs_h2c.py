# -*- coding: utf-8 -*-
"""
块 H2-c 可视化：扩题器的产出。

  fig6_expanded.png   800 道新题的构成 + 验证有效率

跑法：python3.9 scripts/make_figs_h2c.py
"""

import json
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

KIND_CN = {"cancel": "取消预订", "change_passenger": "改乘客",
           "change_baggages": "改行李额", "send_certificate": "发证书"}
COLS = ["#1565c0", "#00897b", "#ef6c00", "#6a1b9a"]


def main() -> int:
    p = PROJECT / "data" / "tasks_expanded.json"
    if not p.exists():
        print("  ⚠️ 缺 tasks_expanded.json"); return 1
    d = json.loads(p.read_text(encoding="utf-8"))
    tasks = d["tasks"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # ① 题型分布
    kc = Counter(t["kind"] for t in tasks)
    labels = [KIND_CN[k] for k, _ in kc.most_common()]
    vals = [v for _, v in kc.most_common()]
    b = axes[0].bar(labels, vals, color=COLS[:len(vals)], width=0.6)
    for bb, v in zip(b, vals):
        axes[0].text(bb.get_x() + bb.get_width() / 2, v + 1, str(v),
                     ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, max(vals) * 1.22)
    axes[0].set_ylabel("题数", fontsize=12)
    axes[0].set_title("① 题型分布", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=11)

    # ② 难度分布
    dc = Counter(t["difficulty"] for t in tasks)
    dl = [d for d in ["easy", "medium"] if d in dc]
    dv = [dc[d] for d in dl]
    dn = {"easy": "easy（单动作）", "medium": "medium（2 动作）"}
    b2 = axes[1].bar([dn[k] for k in dl], dv,
                     color=["#43a047", "#fb8c00"][:len(dl)], width=0.5)
    for bb, v in zip(b2, dv):
        axes[1].text(bb.get_x() + bb.get_width() / 2, v + 1,
                     f"{v}\n({v/len(tasks)*100:.0f}%)", ha="center",
                     fontsize=11.5, fontweight="bold")
    axes[1].set_ylim(0, max(dv) * 1.28)
    axes[1].set_title("② 难度分布", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=11)

    # ③ 验证漏斗
    tried = d["tried"]
    passed = len(tasks)
    drops = d.get("drops", {})
    stages = [("尝试生成", tried, "#90a4ae"),
              ("通过环境验证", passed, "#2e7d32")]
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    y = 4.4
    for name, v, c in stages:
        w = v / tried * 8.4
        ax.barh([y], [max(w, 0.15)], color=c, height=0.75)
        ax.text(0.1 + max(w, 0.15) / 2, y, f"{v}", va="center", ha="center",
                color="white", fontsize=13, fontweight="bold")
        ax.text(-0.15, y, name, va="center", ha="right", fontsize=11.5)
        y -= 1.15

    ax.text(5, 2.05, f"有效率 {passed/tried*100:.1f}%", ha="center",
            fontsize=13.5, fontweight="bold", color="#2e7d32")
    drop_txt = "丢弃原因：\n" + "\n".join(
        f"  · {k}  {v}" for k, v in sorted(drops.items(), key=lambda kv: -kv[1]))
    ax.text(0.2, 1.0, drop_txt, va="top", fontsize=10.5, color="#b71c1c")
    ax.set_title("③ 验证漏斗（生成不是即用）", fontsize=13, fontweight="bold")

    fig.suptitle(f"图 H2-2｜扩题器：{passed} 道新题，全部通过环境验证",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：造题的关键不是「生成」，是「每道题都丢回环境跑两遍」——"
             "什么都不做必须 0 分、gold 重放必须满分，两条都过才留下。",
             ha="center", fontsize=11.5, color="#1b5e20", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = FIGS / "fig6_expanded.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  ✅ {out}")
    print(f"     {passed} 道 ｜ 有效率 {passed/tried*100:.1f}% ｜ "
          f"题型 {dict(kc)} ｜ 难度 {dict(dc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
