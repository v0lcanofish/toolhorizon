# -*- coding: utf-8 -*-
"""
解剖一轮 rollout —— 回答"**为什么零方差率这么高**"。

━━━ 为什么要有这个脚本（2026-09-18）━━━

probe1 那一轮跑出来：零方差率 **0.903**，248 条轨迹里 **224 条 advantage=0**（不产生梯度）。
但同一份输出里还有一行对不上：**平均才 10.44 轮，而 max_turns 是 40**。
如果 58% 是撞轮数上限，平均值该在 23 以上 —— 所以**撞的不是轮数，是 token 预算**。

光看聚合数字定不了性。这个脚本把 jsonl 拆开，回答四个问题：
  ① 到底撞的是哪个上限（max_turns 还是 context_overflow）
  ② 被截断的轨迹长什么样（轮数/token 数/是哪几道题）
  ③ 如果它们能跑完，通过率会是多少（反事实估算）
  ④ 有信号的是哪几道题（GRPO 就靠这几道在学）

跑法：
    python scripts/analyze_rollouts.py runs/probe1/rollouts/step_000.jsonl
    python scripts/analyze_rollouts.py            # 不给路径就找最新的一份
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent

# 撞上这两个 = 被强制判 0 分（见 train/rollout_batch.py:_finish）
TRUNCATED = ("max_turns", "context_overflow")


def pick_latest() -> Path:
    cands = sorted((_PROJECT / "runs").glob("*/rollouts/step_*.jsonl"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("runs/ 下找不到任何 step_*.jsonl —— 先跑一轮 collect")
    return cands[-1]


def load(path: Path):
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    if not rows:
        raise SystemExit(f"{path} 是空的")
    return rows


def bar(frac: float, width: int = 28) -> str:
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    path = Path(argv[0]).resolve() if argv else pick_latest()

    # ⚠️ 相对路径直接 relative_to 会抛 ValueError —— 而命令行传的就是相对路径
    #    （`python scripts/analyze_rollouts.py runs/probe1/...`）。
    #    本地造数据测出来的，不测的话上卡必崩。
    try:
        shown = path.relative_to(_PROJECT)
    except ValueError:
        shown = path

    rows = load(path)
    n = len(rows)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    print("=" * 78)
    print(f"rollout 解剖：{shown}")
    print(f"  {n} 条轨迹 ｜ {len(by_task)} 道题 ｜ 每题 {n // max(1, len(by_task))} 条")
    print("=" * 78)

    # ---------------------------------------------------------------- ① 终止方式
    print("\n① 终止方式（撞上 max_turns / context_overflow 的，reward 被强制判 0）\n")
    c = Counter(r["terminated_by"] for r in rows)
    for k, v in c.most_common():
        flag = "  ⛔强制判0" if k in TRUNCATED else ""
        print(f"   {k:<18} {v:>4}  ({v / n:5.1%}) {bar(v / n)}{flag}")

    # ---------------------------------------------------------------- ② 分组画像
    print("\n② 各类终止方式长什么样\n")
    print(f"   {'终止方式':<18}{'reward均值':>10}{'轮数中位':>10}{'轮数最大':>10}"
          f"{'prompt中位':>12}{'prompt最大':>12}")
    for k, _ in c.most_common():
        sub = [r for r in rows if r["terminated_by"] == k]
        turns = [r["n_turns"] for r in sub]
        pt = [r["prompt_tokens"] for r in sub]
        print(f"   {k:<18}{st.mean([r['reward'] for r in sub]):>10.3f}"
              f"{st.median(turns):>10.0f}{max(turns):>10}"
              f"{st.median(pt):>12.0f}{max(pt):>12}")

    # ---------------------------------------------------------------- ③ 诊断
    print("\n③ 诊断：撞的是轮数上限还是 token 预算？\n")
    S_MAX = 8192
    trunc = [r for r in rows if r["terminated_by"] in TRUNCATED]
    if not trunc:
        print("   没有被截断的轨迹 —— 零方差率不是截断造成的，得往别处找原因")
    else:
        mt = [r for r in trunc if r["terminated_by"] == "max_turns"]
        co = [r for r in trunc if r["terminated_by"] == "context_overflow"]
        print(f"   截断 {len(trunc)} 条 = max_turns {len(mt)} + context_overflow {len(co)}")
        if co:
            pt = [r["prompt_tokens"] for r in co]
            over = sum(1 for x in pt if x >= S_MAX * 0.9)
            print(f"   → **主因是 token 预算**：{len(co)} 条撞 context_overflow，"
                  f"其中 {over} 条的 prompt 已 ≥ 90% 的 S_max({S_MAX})")
            print(f"      prompt_tokens 中位 {st.median(pt):.0f} ｜ 最大 {max(pt)}")
            print(f"      ⭐ 可修：`--max-seq-tokens` 提到 16384（实测训练峰值才 5.15 GB，"
                  f"24 GB 显存扛得住）")
        if mt:
            print(f"   → 另有 {len(mt)} 条撞 max_turns(40)")
        # 轮数证据
        print(f"\n   反证检查：截断轨迹的轮数中位 {st.median([r['n_turns'] for r in trunc]):.0f}"
              f"（若接近 40 则主因是轮数；明显偏低则主因是 token）")

    # ---------------------------------------------------------------- ④ 反事实
    print("\n④ 反事实：要是它们能跑完，通过率会是多少？\n")
    ok = [r for r in rows if r["terminated_by"] not in TRUNCATED]
    if ok and trunc:
        p_ok = sum(1 for r in ok if r["reward"] >= 1) / len(ok)
        print(f"   跑完的轨迹 {len(ok)} 条，其中通过率 {p_ok:.1%}")
        print(f"   当前全程通过率 {sum(1 for r in rows if r['reward'] >= 1) / n:.1%}")
        print(f"   ⚠️ **如果**截断那 {len(trunc)} 条能跑到同样水平，"
              f"总通过率会到 {p_ok:.1%}（现在的 {p_ok / max(1e-9, sum(1 for r in rows if r['reward'] >= 1) / n):.1f} 倍）")
        print(f"   ⚠️ 但这是**上限估计**，不是预测 —— 被截断的轨迹本来就更可能是弱轨迹。")

    # ---------------------------------------------------------------- ⑤ 题层面
    print("\n⑤ 零方差拆开看 —— **全错和全对是两种相反的东西，混在一起会看不出方向**\n")
    zv_all_fail, zv_all_pass, sig = [], [], []
    for k, v in by_task.items():
        rs = {x["reward"] for x in v}
        if len(rs) > 1:
            sig.append(k)
        elif next(iter(rs)) >= 1.0 - 1e-9:
            zv_all_pass.append(k)      # 全对 → 零方差，但说明"已经会了"
        else:
            zv_all_fail.append(k)      # 全错 → 零方差，这才是"不会做"
    print(f"   全错（不会做）  {len(zv_all_fail):>3} 道  {bar(len(zv_all_fail) / len(by_task))}")
    print(f"   全对（已经会了）{len(zv_all_pass):>3} 道  {bar(len(zv_all_pass) / len(by_task))}")
    print(f"   有信号（能学）  {len(sig):>3} 道  {bar(len(sig) / len(by_task))}")
    if zv_all_pass:
        print(f"   ⭐ 全对的题（模型已稳定做对，对训练没贡献但是**学会了的证据**）："
              f"{sorted(zv_all_pass)}")

    print("\n⑥ 逐题表（**看截断是分散在所有题上，还是集中在几道题**）\n")
    print(f"   {'题号':<8}{'通过/8':>8}{'截断/8':>8}   状态")
    for k in sorted(by_task, key=lambda x: (not str(x).isdigit(), str(x))):
        v = by_task[k]
        p = sum(1 for x in v if x["reward"] >= 1)
        t = sum(1 for x in v if x["terminated_by"] in TRUNCATED)
        st_ = ("⭐有信号" if k in sig else
               ("全对" if k in zv_all_pass else "全错"))
        print(f"   {str(k):<8}{p:>8}{t:>8}   {st_}")

    n_trunc_tasks = len({r["task_id"] for r in rows
                         if r["terminated_by"] in TRUNCATED})
    print(f"\n   截断波及 {n_trunc_tasks}/{len(by_task)} 道题"
          f"（平均每题 {len(trunc) / max(1, n_trunc_tasks):.1f} 条）")

    print("\n" + "=" * 78)
    print("下一步怎么定：")
    print("  · 截断**分散在所有题**上 → 加 --max-seq-tokens 16384 重跑一轮（治标）")
    print("  · 有信号的题 < 5 道       → 光加预算不够，得回去加 SFT（治本）")
    print("  · 若「全对」的题明显变多  → 那是好事，说明 SFT 真的学进去了")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
