# -*- coding: utf-8 -*-
"""
块 H1 的三张可视化 —— 把「搞清数据」这一步变成看得见的东西。

  fig1_signal_shape.png   每道题的奖励信号形状（四路探针）
  fig2_expand_space.png   扩题器的组合空间 vs 目标
  fig3_token_dist.png     真实轨迹 token 分布 + 构成

跑法：python3.9 scripts/make_figs_h1.py   （需要 matplotlib）
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGS = PROJECT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

C_S = "#2e7d32"      # 有区分度
C_O = "#ef6c00"      # 单边信号
C_N = "#c62828"      # 真无信号
C_GREY = "#90a4ae"


def load(name):
    p = PROJECT / "data" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ================================================================ fig 1
def fig_signal_shape():
    d = load("signal_shape.json")
    if not d:
        print("  ⚠️ 缺 signal_shape.json"); return
    rows = d["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2),
                                   gridspec_kw={"width_ratios": [1, 1.6]})

    # --- 左：分类占比
    cnt = Counter(r["class"] for r in rows)
    labels = ["S 有区分度", "O 单边信号", "N 真·无信号"]
    vals = [cnt.get("S", 0), cnt.get("O", 0), cnt.get("N", 0)]
    colors = [C_S, C_O, C_N]
    bars = ax1.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v}\n({v/50*100:.0f}%)",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 38)
    ax1.set_ylabel("题目数（共 50 道）", fontsize=12)
    ax1.set_title("① 50 道题的信号形状分类", fontsize=13, fontweight="bold")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.annotate("★ 这里是 0\n没有一道题\n是「怎么都对」",
                 xy=(2, 0), xytext=(2, 22),
                 ha="center", fontsize=11, color=C_N, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_N, lw=1.8))

    # --- 右：逐题四路探针结果（排序后的堆叠条）
    order = sorted(rows, key=lambda r: ({"S": 0, "O": 1}.get(r["class"], 2), r["task"]))
    probes = ["r_noop", "r_readonly", "r_foreign", "r_gold"]
    pname = ["什么都不做", "只做查询", "乱做(过调用)", "完整正确"]
    pcolor = ["#607d8b", "#8d6e63", "#c62828", "#2e7d32"]
    xs, ys = [], []
    for i, r in enumerate(order):
        for j, p in enumerate(probes):
            v = r.get(p)
            xs.append(i)
            ys.append(j)
            ax2.scatter(i, j, s=52,
                        c=(pcolor[j] if v == 1.0 else "#ffffff"),
                        edgecolors=pcolor[j], linewidths=1.6, zorder=3)
    ax2.set_yticks(range(len(probes)))
    ax2.set_yticklabels(pname, fontsize=11)
    ax2.set_xlabel("50 道题（按类别排序）", fontsize=12)
    ax2.set_title("② 每道题在四种策略下的得分（实心=拿满分 1.0）",
                  fontsize=13, fontweight="bold")
    nS = cnt.get("S", 0)
    ax2.axvline(nS - 0.5, color="k", ls="--", lw=1.4, zorder=2)
    ax2.text(nS / 2 - 0.5, 3.55, f"S 类 {nS} 道\n做完才满分",
             ha="center", fontsize=11, color=C_S, fontweight="bold")
    ax2.text(nS + (50 - nS) / 2 - 0.5, 3.55, f"O 类 {50-nS} 道\n不做就满分",
             ha="center", fontsize=11, color=C_O, fontweight="bold")
    ax2.set_ylim(-0.6, 3.95)
    ax2.set_xlim(-1, 50)
    ax2.grid(axis="y", ls=":", alpha=0.35)

    fig.suptitle("图 H1-1｜τ-bench airline 50 道题的奖励信号形状（四路探针实测）",
                 fontsize=14.5, fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：没有一道题是真·无信号 —— 19 道「不做满分但乱做 0 分」是单边信号，"
             "只惩罚过调用，不奖励任何正确行为。",
             ha="center", fontsize=11.5, color="#b71c1c", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    out = FIGS / "fig1_signal_shape.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  ✅ {out}")


# ================================================================ fig 2
def fig_expand_space():
    d = load("expandable_space.json")
    if not d:
        print("  ⚠️ 缺 expandable_space.json"); return
    by = d["by_tool"]
    order = sorted(by.items(), key=lambda kv: -kv[1])

    fig, ax = plt.subplots(figsize=(11, 5.6))
    names = [k for k, _ in order]
    vals = [v for _, v in order]
    colors = ["#1565c0"] * len(vals)
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.62)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 900, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=11, fontweight="bold")
    ax.axvline(400, color=C_N, ls="--", lw=2)
    ax.text(420, 0.15, "目标 400 道", color=C_N,
            fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    ax.set_xlabel("可构造的题目实例数（对数轴）", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(100, 200000)
    ax.set_title("图 H1-2｜扩题器的组合空间 vs 目标", fontsize=14, fontweight="bold")

    txt = (f"保守口径（每条预订最多出 4 类题 + 发证书）\n"
           f"2000 × 4 + 313 = {d['conservative']:,} 道\n"
           f"→ 是目标 400 的 {d['conservative']/400:.1f} 倍")
    ax.text(0.985, 0.06, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11.5, bbox=dict(boxstyle="round,pad=0.6", fc="#e8f5e9",
                                     ec=C_S, lw=1.6))
    fig.text(0.5, 0.015,
             "一句话：题量不是瓶颈 —— 真正的瓶颈是「过环境 oracle 验证后的有效率」。",
             ha="center", fontsize=11.5, color="#1b5e20", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIGS / "fig2_expand_space.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  ✅ {out}")


# ================================================================ fig 3
def fig_token_dist():
    d = load("token_stats.json")
    if not d:
        print("  ⚠️ 缺 token_stats.json"); return
    vals = d["all_tot"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    # --- 左：直方图 + 分位数
    ax1.hist(vals, bins=34, color="#5c6bc0", edgecolor="white", alpha=0.9)
    top = ax1.get_ylim()[1]
    ax1.axvline(8192, color="k", ls="-", lw=2.2, alpha=0.75)
    ax1.text(8300, top * 0.72, "S_max=8192\n标称上限", color="k",
             fontsize=10.5, fontweight="bold", ha="left")
    for lab, v, c, y in [("p50", d["total"]["p50"], C_S, 0.95),
                         ("p90", d["total"]["p90"], C_O, 0.80),
                         ("max", d["total"]["max"], C_N, 0.95)]:
        ax1.axvline(v, color=c, ls="--", lw=2)
        ax1.text(v - 250, top * y, f"{lab}={v:,.0f} ", color=c,
                 fontsize=11, fontweight="bold", ha="right")
    ax1.set_xlabel("总 token 数（含 14 工具 schema）", fontsize=12)
    ax1.set_ylabel("轨迹条数", fontsize=12)
    ax1.set_title(f"真实成功轨迹 token 分布（n={d['n']}）",
                  fontsize=13, fontweight="bold")

    # --- 右：单条轨迹的 token 构成
    comp = [("tool 返回", 3574, "#c62828"),
            ("system", 1265, "#ef6c00"),
            ("assistant（被训练）", 492, "#2e7d32"),
            ("user", 112, "#1565c0")]
    tot = sum(v for _, v, _ in comp)
    left = 0.0
    for name, v, c in comp:
        ax2.barh([0], [v], left=left, color=c, height=0.5)
        if v / tot > 0.04:
            ax2.text(left + v / 2, 0, f"{v/tot*100:.0f}%", ha="center",
                     va="center", color="white", fontsize=12, fontweight="bold")
        left += v
    ax2.set_xlim(0, tot)
    ax2.set_yticks([])
    ax2.set_xlabel("token 数", fontsize=12)
    ax2.set_title("单条 24 消息轨迹的构成", fontsize=13, fontweight="bold")
    ax2.legend(handles=[Patch(facecolor=c, label=n) for n, _, c in comp],
               loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=10.5)

    fig.suptitle("图 H1-3｜真实轨迹长度实测 —— 修正了原先 2.3 倍的低估",
                 fontsize=14.5, fontweight="bold")
    fig.text(0.5, 0.015,
             "一句话：工具 schema 占 3249 token 是漏算的大头；p90=10287 已超 S_max，必须截断；"
             "被训练的 assistant token 只占 12%。",
             ha="center", fontsize=11.5, color="#b71c1c", fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    out = FIGS / "fig3_token_dist.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  ✅ {out}")


# ================================================================ fig 4
def fig_task_split():
    d = load("task_split.json")
    if not d:
        print("  ⚠️ 缺 task_split.json"); return
    train = {r["task"]: r for r in d["train"]}
    probe = {r["task"]: r for r in d["probe_overcall"]}

    SUB_COLOR = {"zero_action": "#8e24aa", "transfer_only": "#5e35b1",
                 "readonly_only": "#3949ab"}
    SUB_NAME = {"zero_action": "零动作", "transfer_only": "只有转人工",
                "readonly_only": "只查不改"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.5, 8.4),
                                   gridspec_kw={"height_ratios": [1.5, 1]})

    # ---- 上：50 道题的格子图（10 列 × 5 行）
    for i in range(50):
        r, c = divmod(i, 10)
        y = 4 - r
        if i in train:
            fc, ec, lab = "#2e7d32", "#1b5e20", "S"
        else:
            st = probe[i]["subtype"]
            fc, ec, lab = SUB_COLOR[st], "#00000055", "O"
        ax1.add_patch(plt.Rectangle((c, y), 0.92, 0.92, fc=fc, ec=ec, lw=1.6))
        ax1.text(c + 0.46, y + 0.58, f"{i}", ha="center", va="center",
                 color="white", fontsize=11, fontweight="bold")
        ax1.text(c + 0.46, y + 0.22, lab, ha="center", va="center",
                 color="#ffffffcc", fontsize=8.5)
    ax1.set_xlim(-0.4, 10.2)
    ax1.set_ylim(-1.15, 5.25)
    ax1.axis("off")
    ax1.set_title("块 H1-D｜50 道题的去向：31 道进训练集，19 道当评测探针",
                  fontsize=14, fontweight="bold", pad=8)
    ax1.text(5, -0.62, "绿色 = 训练集（S 有区分度）　｜　紫色系 = 评测探针（O 单边信号）",
             ha="center", fontsize=11, color="#37474f")
    ax1.text(5, -1.02,
             "★ 探针题不进训练：它们奖励与策略无关，会毒化 GRPO；但它们是唯一能测「过调用」的一批",
             ha="center", fontsize=10.5, color="#b71c1c", fontweight="bold")

    # ---- 下：两侧子分组（两个并排的小条形图）
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    ax2.axis("off")

    kt = Counter(r["kind"] for r in d["train"])
    kp = Counter(r["subtype"] for r in d["probe_overcall"])

    # 训练集
    rows_t = [("write  要改数据库", kt.get("write", 0), "#2e7d32"),
              ("communicate_only  只汇报", kt.get("communicate_only", 0), "#00acc1")]
    # 探针集
    rows_p = [("readonly_only  只查不改", kp.get("readonly_only", 0), "#3949ab"),
              ("zero_action  零动作", kp.get("zero_action", 0), "#8e24aa"),
              ("transfer_only  只有转人工", kp.get("transfer_only", 0), "#5e35b1")]

    def mini(ax, title, rows, x0):
        ax.set_xlim(0, 34); ax.set_ylim(-0.6, len(rows) - 0.25)
        ax.invert_yaxis()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([n for n, _, _ in rows], fontsize=11)
        for i, (_, v, c) in enumerate(rows):
            ax.barh(i, v, color=c, height=0.5)
            ax.text(v + 0.7, i, f"{v} 道", va="center", fontsize=11.5,
                    fontweight="bold", color=c)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=8)

    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax2.get_subplotspec(), wspace=0.45)
    mini(fig.add_subplot(gs[0]), "训练集 31 道的构成", rows_t, 0)
    mini(fig.add_subplot(gs[1]), "评测探针 19 道的构成（按 gold 动作）", rows_p, 1)

    fig.text(0.5, 0.012,
             "一句话：同一份 50 道题，训练剔除 19 道、评测保留 19 道 —— "
             "分流的唯一依据是「什么都不做能不能拿满分」。",
             ha="center", fontsize=11.5, color="#1b5e20", fontweight="bold")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = FIGS / "fig4_task_split.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  ✅ {out}")


if __name__ == "__main__":
    print("生成块 H1 可视化 ...")
    fig_signal_shape()
    fig_expand_space()
    fig_token_dist()
    fig_task_split()
    print("完成。输出目录：", FIGS)
