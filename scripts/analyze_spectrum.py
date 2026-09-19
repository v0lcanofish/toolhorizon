# -*- coding: utf-8 -*-
"""
⭐ 工具使用退化谱 —— **本项目的核心创新点**，从落盘的 rollout 离线重算（0 GPU）。

    python scripts/analyze_spectrum.py runs/arm_vanilla
    python scripts/analyze_spectrum.py runs/arm_vanilla runs/arm_ds runs/vanilla
    python scripts/analyze_spectrum.py --all           # 自动找 runs/ 下的臂

━━━ 为什么要有这个（2026-09-19）━━━

退化谱在 `observe/metrics.py` 里**两周前就实现好了**（欠调用 / 过调用 / 抖动 / 空转，
jitter 的定义还专门为"对 5 个不同预订做同一件事 ≠ 抖动"改过），
但训练路径和分析脚本**一次都没调用它** —— H6 报告里这个核心创新点
只报了 `tool_calls_mean` 一个数。这是"写了但没接进流程"的第 7 个实例。

━━━ ⚠️ 必须先知道的一条口径（否则会误读）━━━

退化谱的四个指标**不是同一批题上测的**：

    欠调用  ← 只能测 **gold 有写动作**的题（S 类，`n_write > 0`）
    过调用  ← 只能测 **gold 无写动作**的题（O 类探针，`n_write == 0`）
    抖动/空转 ← 所有轨迹

而四臂跑的是 `--select train`（31 道 S 类）。所以：

    **过调用这一维在四臂的数据上测不出来** —— O 类题只有 task 44 一道混在里面。

脚本会把分母打出来，并且**在分母太小时明确警告**，不让你拿噪声当结论。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))

from observe.replay import load_run, load_task_meta                    # noqa: E402

ARMS = ["arm_vanilla", "arm_ds", "vanilla", "arm_lata_ds"]
# 分母小于这个数就认为这一维"测不出来"（一道题 × 8 条 = 8 条，太小）
MIN_DENOM = 48          # = 6 道题 × 8 条


def avg(rows, path, n=8):
    """后 n 轮的均值。path 是 'degradation.jitter_mean' 这种点号路径。"""
    vals = []
    for _, m in rows[-n:]:
        v = m
        for k in path.split("."):
            v = (v or {}).get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def head(rows, path, n=3):
    return avg(rows[:n], path, n=n)


def fmt(v, spec=".3f"):
    return "  ——  " if v is None else format(v, spec)


def report(name: str, rows) -> None:
    print("\n" + "=" * 96)
    print(f"▶ {name} ｜ {len(rows)} 轮")
    print("=" * 96)

    h0, h1 = head(rows, "degradation.under_call_n", 1), avg(rows, "degradation.under_call_n", 1)
    _ = h0
    print(f"\n{'轮':>4}{'欠调用率':>11}{'抖动':>9}{'写调用/条':>11}{'工具调用/条':>12}"
          f"{'空转轮数':>10}{'零方差率':>11}")
    print("-" * 96)
    for step, m in rows:
        d, h = m["degradation"], m["health"]
        print(f"{step:>4}{fmt(d.get('under_call_rate')):>11}{fmt(d.get('jitter_mean')):>9}"
              f"{fmt(d.get('write_calls_mean'), '.2f'):>11}"
              f"{fmt(h.get('tool_calls_mean'), '.2f'):>12}"
              f"{fmt(d.get('idle_turns_mean'), '.1f'):>10}"
              f"{fmt(h.get('zero_var_rate')):>11}")

    print(f"\n起点(前3轮) → 稳态(后8轮)：")
    for label, path in (("欠调用率", "degradation.under_call_rate"),
                        ("抖动", "degradation.jitter_mean"),
                        ("写调用/条", "degradation.write_calls_mean"),
                        ("工具调用/条", "health.tool_calls_mean"),
                        ("零方差率", "health.zero_var_rate")):
        a, b = head(rows, path), avg(rows, path)
        if a is None or b is None:
            continue
        arrow = "↑" if b > a else ("↓" if b < a else "=")
        print(f"   {label:<12}{a:>8.3f}  {arrow}  {b:>8.3f}")

    # ---- 分母诊断（这一段是防止误读的关键）
    d_last = rows[-1][1]["degradation"]
    print(f"\n   ⚠️ 分母诊断（这两个比例是在**不同池子**上算的，别混着读）：")
    print(f"      欠调用 分母 = {d_last.get('under_call_n')} 条  ← gold 有写动作的题（S 类）")
    print(f"      过调用 分母 = {d_last.get('over_call_n')} 条  ← gold **无**写动作的题（O 类探针）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run 目录（默认自动找 runs/ 下的臂）")
    ap.add_argument("--all", action="store_true", help="自动找 runs/ 下的全部臂")
    ap.add_argument("--rounds", type=int, default=0)
    a = ap.parse_args()

    roots = [Path(x) for x in a.runs]
    if a.all or not roots:
        roots = [(_PROJECT / "runs" / n) for n in ARMS]
    roots = [r for r in roots if (r / "rollouts").is_dir()]
    if not roots:
        raise SystemExit("找不到任何 run 目录 —— 先跑 scripts/run_arms.sh")

    meta = load_task_meta()
    # ---- 先算"这批数据里，退化谱各维到底有没有分母"
    n_sw = sum(1 for v in meta.values() if v.get("n_write", 0) > 0)
    n_snw = sum(1 for v in meta.values() if v.get("n_write", 0) == 0)

    print("=" * 96)
    print("工具使用退化谱 ⭐ 核心创新点 ｜ 从落盘 rollout 离线重算（0 GPU）")
    print("=" * 96)
    print(f"task_split 里：n_write>0（欠调用可测）{n_sw} 道 ｜ "
          f"n_write==0（过调用可测）{n_snw} 道")

    warn = []
    for r in roots:
        rows = load_run(r, a.rounds or None, meta)
        if not rows:
            print(f"\n   (跳过 {r.name} —— 没有 step_*.jsonl)")
            continue
        report(r.name, rows)
        n_oc = rows[-1][1]["degradation"].get("over_call_n", "0/0")
        try:
            got = int(str(n_oc).split("/")[1])
        except Exception:                                      # noqa: BLE001
            got = 0
        if got * 8 < MIN_DENOM and rows[-1][1]["degradation"].get("over_call_rate") is not None:
            warn.append((r.name, n_oc))

    if warn:
        print("\n" + "=" * 96)
        print("⚠️ 「过调用」这一维**测不出来**，别当结论用")
        print("=" * 96)
        for name, denom in warn:
            print(f"   {name:<14} 过调用分母只有 {denom} 条题（{MIN_DENOM // 8} 道以下 = 噪声）")
        print("""
   原因：过调用只能在 **gold 无写动作** 的题上测，而四臂跑的是 `--select train`
         （31 道 S 类），O 类探针题只有 task 44 一道混在里面。

   ⭐ 要让这一维可测，只需补一次**探针集评测**（不用重训）：
        用最终 adapter 在 `--select probe`（19 道 O 类）上采一轮 ≈ 8 分钟 GPU，
        然后 analyze_spectrum.py 就能算出真正的 over_call_rate。
        这一步之前一直被漏掉 —— 而它是「过调用」唯一的硬定义。""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
