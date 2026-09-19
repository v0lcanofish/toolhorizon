# -*- coding: utf-8 -*-
"""
块 H2-a/b 的可视化：SFT 数据的构成。

  fig5_sft_data.png   115 条 SFT 轨迹的覆盖情况与构成

跑法：python3.9 scripts/make_figs_h2.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

C_REAL_4O = "#1565c0"
C_REAL_SN = "#00897b"
C_SYNTH = "#ef6c00"


def main() -> int:
    split = json.loads((PROJECT / "data" / "task_split.json").read_text(encoding="utf-8"))
    train_ids = sorted(r["task"] for r in split["train"])

    rows = [json.loads(l) for l in
            (PROJECT / "data" / "sft_final.jsonl").open(encoding="utf-8")]

    per_task = defaultdict(list)
    for r in rows:
        per_task[r["task_id"]].append(r["source"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2),
                                   gridspec_kw={"width_ratios": [1.9, 1]})

    # ---------------- 左：31 道题的覆盖情况
    xs = list(range(len(train_ids)))
    counts = [len(per_task.get(t, [])) for t in train_ids]
    colors = []
    for t, c in zip(train_ids, counts):
        srcs = set(per_task.get(t, []))
        if not srcs or srcs == {"synth"}:
            colors.append(C_SYNTH)
        else:
            colors.append(C_REAL_SN if "sonnet-35-new" in srcs else C_REAL_4O)
    bars = ax1.bar(xs, counts, color=colors, width=0.7)
    for x, c in zip(xs, counts):
        ax1.text(x, c + 0.15, str(c), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(t) for t in train_ids], fontsize=9, rotation=90)
    ax1.set_xlabel("训练集的 31 道题（task id）", fontsize=12)
    ax1.set_ylabel("可用轨迹条数", fontsize=12)
    ax1.set_ylim(0, max(counts) + 1.6)
    ax1.set_title("① 每道题有几条 SFT 轨迹", fontsize=13, fontweight="bold")

    import matplotlib.patches as mp
    ax1.legend(handles=[
        mp.Patch(color=C_REAL_4O, label="真实轨迹（gpt-4o）"),
        mp.Patch(color=C_REAL_SN, label="真实轨迹（sonnet）"),
        mp.Patch(color=C_SYNTH, label="合成（gold 重放 + 补收尾）"),
    ], fontsize=10, loc="upper right")

    ax1.text(0.5, -0.34,
             "★ 橙色 = 那 7 道没有真实轨迹的题，用【gold 重放 + 补收尾】合成 → 31 道全覆盖",
             transform=ax1.transAxes, ha="center", fontsize=10.5,
             color=C_SYNTH, fontweight="bold")

    # ---------------- 右：来源构成 + 收尾长度
    src = Counter(r["source"] for r in rows)
    labels = ["sonnet 真实", "gpt-4o 真实", "合成"]
    vals = [src.get("sonnet-35-new", 0), src.get("gpt-4o", 0), src.get("synth", 0)]
    cols = [C_REAL_SN, C_REAL_4O, C_SYNTH]
    wedges, _, autot = ax2.pie(
        vals, colors=cols, autopct=lambda p: f"{int(round(p*sum(vals)/100))}",
        startangle=90, textprops={"fontsize": 12, "color": "white",
                                  "fontweight": "bold"})
    ax2.legend(wedges, [f"{l}  {v} 条" for l, v in zip(labels, vals)],
               loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=10.5,
               frameon=False)
    ax2.set_title("② 115 条的来源", fontsize=13, fontweight="bold", pad=34)

    fig.suptitle("图 H2-1｜SFT 数据构成：115 条轨迹，31 道题全覆盖",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：金标准里没有「收尾那句话」，但 τ-bench 自带的 268 条真实成功轨迹本身就是"
             "完整对话——108 条直接复用，剩下 7 道重放 gold + 补收尾合成。",
             ha="center", fontsize=11.5, color="#1b5e20", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    out = FIGS / "fig5_sft_data.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  ✅ {out}")

    # ---- 控制台摘要
    print()
    print(f"  轨迹总数 {len(rows)} ｜ 覆盖 {len(per_task)}/31 道")
    print(f"  来源：{dict(src)}")
    print(f"  每道题最少 {min(counts)} 条，最多 {max(counts)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
