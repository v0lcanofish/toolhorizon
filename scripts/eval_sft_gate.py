# -*- coding: utf-8 -*-
"""
SFT 出口判据评测 —— 拿**真 adapter** 在 medium 档上算 pass@1，自动判 [15%, 50%]。

━━━ 为什么必须单独有这么一个脚本（2026-09-18）━━━

H5 的硬判据写着：

    SFT 完在 medium 档上 pass@1 ∈ [15%, 50%]
      < 5%  → 立刻降到 easy 档，不要头铁
      > 50% → 题太简单，GRPO 会组内全对，同样是白跑

**但 `observe/harness.py` 只有 `--mock` 模式**（第 229 行直接 `return 2`）：
它能在 CPU 上验证 harness 本身对不对，**却没法拿真模型算 pass@1**。

于是会撞上这个局面：租卡 → 跑完 SFT（1.5 GPU时）→ **判据没入口可验** →
要么瞎猜着往下跑 GRPO（12 GPU时），要么再租一次卡补评测。
**这就是那种"本地 20 分钟不做，卡上多花 ¥25"的活。**

━━━ 跑法 ━━━

卡上（SFT 前先量基座，SFT 后再量一次 —— 两次相减才是 SFT 的增量）：

    cd /root/autodl-tmp/ToolHorizon
    source scripts/env.sh

    # ① 先看要花多少（不跑）
    python scripts/eval_sft_gate.py --dry-run

    # ② SFT 之前：量基座，拿到 before
    python scripts/eval_sft_gate.py --model "$TOOLHORIZON_TOKENIZER" \\
        --out reports/sft_gate_base.json

    # ③ SFT 之后：量 adapter
    python scripts/eval_sft_gate.py --model "$TOOLHORIZON_TOKENIZER" \\
        --adapter models/adapter_sft --out reports/sft_gate_sft.json

本地 CPU（零模型、零 GPU，验的是**这套判据逻辑本身**）：

    PYTHONIOENCODING=utf-8 python3.13 \\
        scripts/eval_sft_gate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from observe.harness import build_splits, evaluate, render, SplitResult  # noqa: E402

# ---------------------------------------------------------------- 判据

# ⚠️⚠️ **2026-09-18 更正：判据该量哪个分片，原文档写得不清楚。**
#
# 原话是「SFT 完在 medium 档上 pass@1 ∈ [15%, 50%]」。
# 但这句话的**目的**是判断"GRPO 有没有活干"：通过率 0 → 组内全错 → std=0 → 一步学不动。
# 而 GRPO 采样的是 `--select arm_base` = **31 道 S 类训练原题**（= harness 的 `covered`），
# **不是 medium**。所以真正决定"GRPO 有没有信号"的是 `covered`，不是 `unseen`。
#
# 结论：**两个都量、都报**，别赌哪个对。
#   covered → "GRPO 有没有信号"（主）
#   unseen  → "SFT 有没有泛化"（副）
# 两个口径打架时（比如 covered 过了、unseen 没过）以 covered 为准做训练决策，
# 但**必须把 unseen 的差距写进报告**，那是泛化边界，不能装作没看见。
JUDGED = [
    ("covered", "训练分片", "GRPO 实际采样的 31 道 S 类原题"),
    ("unseen", "medium 泛化分片", "扩题集 medium 档，没训过"),
]

# 主判据用哪个分片（列表里第一个存在的结果）
PRIMARY_SPLIT = "covered"

PASS_LOW = 0.15
PASS_HIGH = 0.50
HARD_FLOOR = 0.05

# 实测（2026-09-17 · 4090D）：单条轨迹 4.14 s，vLLM 启动 119.8 s
SEC_PER_EPISODE = 4.14
VLLM_INIT_SEC = 119.8
YUAN_PER_HOUR = 1.8          # 4090D 区间 1.5–2.0，取中值估


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """
    比例的 95% 置信区间（Wilson 区间）。

    ⚠️ **为什么必须报区间而不是光报一个点**：
       gate 分片是 30 道题。p=0.30 时，点估计的标准误 ≈ sqrt(.3*.7/30) ≈ 0.084，
       95% 区间差不多是 ±0.16 —— **判据带只有 [0.15, 0.50] 这么宽，区间一撑就顶到边**。

       也就是说：**跑出 0.14 判"不过"、跑出 0.16 判"过"，这个差别在噪声里，不该当结论。**
       用 Wilson 而不是正态近似，是因为它在 p 靠近 0 或 1 时不会给出越界的区间
       （基座大概率落在 p≈0 那一侧，正态近似在那里会算出负的下界）。

    返回 (low, high)。
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def verdict(pass_at_1: float, zero_var_rate: float) -> dict:
    """
    纯函数 —— 判据只有这一个地方定义，自测直接打它的边界。

    ⚠️ 为什么把判据抽成纯函数：**租一次卡只能验一次**。
       如果判据散在 main() 里，跑出 0.149 还是 0.151 就得靠人眼看，
       而边界值恰恰是最需要机械判的（差 0.002 结论就从"过"变"不过"）。
    """
    if pass_at_1 < HARD_FLOOR:
        return {
            "level": "FAIL_HARD",
            "headline": f"pass@1 = {pass_at_1:.3f} < {HARD_FLOOR} —— SFT 没把通过率抬起来",
            "action": "立刻降到 easy 档（--select exp_easy），不要头铁往下跑 GRPO",
            "reason": "组内大概率全错 → std=0 → advantage 全 0 → 一步都学不动，纯烧钱",
        }
    if pass_at_1 < PASS_LOW:
        return {
            "level": "WARN_LOW",
            "headline": f"pass@1 = {pass_at_1:.3f}，落在 [{HARD_FLOOR}, {PASS_LOW}) 之间",
            "action": "可以试跑 1 轮 GRPO 看零方差率；若 ≥0.8 超过 3 轮就换档",
            "reason": "能学，但组内全错的组偏多，有效梯度比正常少",
        }
    if pass_at_1 > PASS_HIGH:
        return {
            "level": "FAIL_EASY",
            "headline": f"pass@1 = {pass_at_1:.3f} > {PASS_HIGH} —— 题对这个策略太简单了",
            "action": "换更难的档，或把 medium 换成步骤更多的题",
            "reason": "组内大概率全对 → std=0 → 同样是白跑（和全错是一回事，只是方向相反）",
        }
    return {
        "level": "PASS",
        "headline": f"pass@1 = {pass_at_1:.3f}，落在判据区间 [{PASS_LOW}, {PASS_HIGH}] 内",
        "action": "可以往下跑 GRPO",
        "reason": "组内既有对也有错 → std>0 → advantage 有信号",
    }


# ---------------------------------------------------------------- 成本预估


def estimate(n_tasks: int, n_samples: int, splits: int) -> dict:
    """跑之前先算清楚要花多少钱 —— 这是这个脚本的主要用途之一。"""
    episodes = n_tasks * n_samples * splits
    sec = episodes * SEC_PER_EPISODE + VLLM_INIT_SEC
    hours = sec / 3600.0
    return {
        "n_tasks": n_tasks,
        "n_samples": n_samples,
        "n_splits": splits,
        "episodes": episodes,
        "est_minutes": round(sec / 60.0, 1),
        "est_hours": round(hours, 3),
        "est_yuan": round(hours * YUAN_PER_HOUR, 2),
    }


# ---------------------------------------------------------------- 报告组装


def build_report(model: str, adapter: str, engine: str, n: int, limit: int,
                 elapsed: float, results: dict) -> dict:
    """
    组装落盘的那份 JSON。

    ⚠️ 单独抽出来是为了**让自测能打到真代码**：如果这段逻辑长在 run_real() 里，
    自测就只能测到纯函数，而"字段名写错 / to_dict 漏字段"这类 bug
    恰恰发生在这一段 —— 上卡跑完 10 分钟才发现 JSON 是坏的，就得重跑。
    """
    gates = {}
    for name, short, desc in JUDGED:
        if name not in results:
            continue
        g: SplitResult = results[name]
        k = round(g.pass_at_1 * g.n_tasks)
        lo, hi = wilson_ci(k, g.n_tasks)
        gates[name] = {
            "split": name, "short": short, "desc": desc,
            "pass_at_1": g.pass_at_1,
            "n_tasks": g.n_tasks,
            "n_trajectories": g.n_samples,
            "ci95": [round(lo, 3), round(hi, 3)],
            # 区间跨过判据边界 → 这个结论不稳，别拿它做决定
            "ci_straddles_boundary": lo < PASS_LOW < hi or lo < PASS_HIGH < hi,
            "zero_var_rate": g.zero_var_rate,
            "reward_mean": g.reward_mean,
            "thresholds": {"low": PASS_LOW, "high": PASS_HIGH,
                           "hard_floor": HARD_FLOOR},
            **verdict(g.pass_at_1, g.zero_var_rate),
        }
    primary = PRIMARY_SPLIT if PRIMARY_SPLIT in gates else next(iter(gates), None)
    # 两个口径打架 → 必须显式记下来，不能只报主判据
    disagree = (len(gates) > 1 and
                len({g["level"] for g in gates.values()}) > 1)
    return {
        "model": model,
        "adapter": adapter or None,
        "engine": engine,
        "n_samples": n,
        "limit": limit,
        "elapsed_sec": round(elapsed, 1),
        "splits": {k: r.to_dict() for k, r in results.items()},
        "primary_split": primary,
        "gates_disagree": disagree,
        "gates": gates,
    }


# ---------------------------------------------------------------- 真跑


def run_real(args) -> int:
    from train.tokenize import load_tokenizer, tool_schemas
    from observe.harness import evaluate_split

    splits = build_splits(limit=args.limit)
    if args.splits == "all":
        todo = ["covered", "uncovered", "unseen"]
    else:
        todo = [s.strip() for s in args.splits.split(",") if s.strip()]

    est = estimate(sum(len(splits[s]) for s in todo), args.n, 1)
    print("=" * 78)
    print("SFT 出口判据评测")
    print("=" * 78)
    print(f"  模型    ：{args.model}")
    print(f"  adapter ：{args.adapter or '（无 —— 量的是基座）'}")
    print(f"  引擎    ：{args.engine}")
    print(f"  分片    ：{', '.join(todo)}")
    for s in todo:
        print(f"      {s:<11} {len(splits[s])} 题 × {args.n} 条")
    print(f"  ⏱️  预估 ：约 {est['est_minutes']} 分钟 ｜ ≈ ¥{est['est_yuan']}")
    print(f"      （{est['episodes']} 条轨迹 × {SEC_PER_EPISODE}s + vLLM 启动 {VLLM_INIT_SEC}s）")
    print("=" * 78)

    if args.dry_run:
        print("\n--dry-run：只算账，不跑。去掉 --dry-run 即真跑。")
        return 0

    tok = load_tokenizer(model_path=args.model)
    tools = tool_schemas()

    t0 = time.time()
    if args.engine == "vllm":
        from train.engine import VLLMEngine
        engine = VLLMEngine(args.model, tok, lora_path=args.adapter or None,
                            max_model_len=args.max_seq_tokens)
    else:
        from train.engine import HFEngine
        from train.modeling import load_model
        m = load_model(args.model, dtype="bfloat16", device_map="auto")
        if args.adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, args.adapter)
        engine = HFEngine(m, tok, device="cuda")
    print(f"\n[引擎就绪] {time.time() - t0:.1f}s")

    results = {}
    for s in todo:
        t1 = time.time()
        r = evaluate_split(s, splits[s], engine, n_samples=args.n,
                           tokenizer=tok, tools=tools)
        results[s] = r
        print(f"  [{s}] 完成，用时 {time.time() - t1:.1f}s")

    print()
    print(render(results))

    report = build_report(args.model, args.adapter, args.engine, args.n,
                          args.limit, time.time() - t0, results)

    print()
    print("─" * 78)
    for name, g in report["gates"].items():
        mark = "★主判据" if name == report["primary_split"] else " 副指标"
        print(f"🚦 {mark} ｜ {g['short']}（{g['desc']}）")
        print(f"   {g['headline']}")
        print(f"   结论：{g['action']}")
        print(f"   原因：{g['reason']}")
        print(f"   ⚠️ 零方差率 = {g['zero_var_rate']:.3f}"
              f"（>0.8 = 这批题对该策略'不是全对就是全错'，训练在空转）")
        print(f"   📊 统计噪声：{round(g['pass_at_1'] * g['n_tasks'])}/{g['n_tasks']} 题全对"
              f"，95% CI = [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]")
        if g["ci_straddles_boundary"]:
            print(f"   ⚠️ **区间跨过了判据边界** —— 这个结论落在噪声里，别当定论。"
                  f"要定的话加 --limit {g['n_tasks'] * 2} 重测")
        else:
            print(f"   ✅ 区间整个落在判据"
                  f"{'内' if g['level'] == 'PASS' else '外'}，结论稳")
        print()
    if report["gates_disagree"]:
        print("⚠️⚠️ **两个口径打架** —— 主判据看训练分片（GRPO 采样的是它），")
        print("     但泛化分片的差距**必须写进报告**，不能装作没看见。")
    print("─" * 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"报告已写出：{out}")
    pk = report["primary_split"]
    return 0 if (pk and report["gates"][pk]["level"] == "PASS") else 1


# ---------------------------------------------------------------- 自测（零模型）


def run_self_test() -> int:
    """
    ⚠️ 这个自测**验不了"真模型跑得动吗"**（那只能上卡验）。
    它验的是这个脚本自己的两处：**判据函数的边界** 和 **分片接得上**。

    为什么边界值必须机械测：租一次卡只能验一次，
    跑出 0.149 还是 0.151 结论就反过来，靠人眼看会出事。
    """
    print("=" * 78)
    print("eval_sft_gate 自测（零模型 / 零 GPU）")
    print("=" * 78)
    fails = []

    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)

    # ① 判据函数的**边界** —— 每个档位上下各打一发
    print("\n   ① 判据边界（这是租卡那次唯一能验的东西，必须机械判）")
    cases = [
        (0.000, "FAIL_HARD"),   # 基座大概率在这
        (0.049, "FAIL_HARD"),
        (0.050, "WARN_LOW"),    # 下界：>= 0.05 就不再是硬失败
        (0.149, "WARN_LOW"),
        (0.150, "PASS"),        # 判据下界
        (0.300, "PASS"),
        (0.500, "PASS"),        # 判据上界（含）
        (0.501, "FAIL_EASY"),   # 上界外
        (0.900, "FAIL_EASY"),
    ]
    for p, want in cases:
        got = verdict(p, 0.0)["level"]
        check(got == want, f"pass@1={p:.3f} → {want}", f"实际 {got}")

    # ② 判据区间必须和常量自洽（防止改了一个忘改另一个）
    check(verdict(PASS_LOW, 0)["level"] == "PASS", f"下界 {PASS_LOW} 判过")
    check(verdict(PASS_HIGH, 0)["level"] == "PASS", f"上界 {PASS_HIGH} 判过")
    check(PASS_LOW < PASS_HIGH, "区间非空")
    check(HARD_FLOOR < PASS_LOW, "硬底线低于判据下界")

    # ①b 置信区间 —— 判据的"确定感"全在这上面，两端边界必须对
    print("\n   ①b 置信区间（30 题时区间有多宽？这决定结论能不能当定论）")
    lo0, hi0 = wilson_ci(0, 30)
    check(lo0 == 0.0 and 0 < hi0 < 0.15, "k=0/30：下界钉在 0，上界不夸张",
          f"[{lo0:.3f}, {hi0:.3f}]")
    lo1, hi1 = wilson_ci(30, 30)
    check(hi1 == 1.0 and 0.85 < lo1 < 1.0, "k=30/30：上界钉在 1，下界不夸张",
          f"[{lo1:.3f}, {hi1:.3f}]")
    lo3, hi3 = wilson_ci(9, 30)          # p = 0.3
    check(lo3 < 0.3 < hi3, "k=9/30：区间包含点估计", f"[{lo3:.3f}, {hi3:.3f}]")
    check(hi3 - lo3 > 0.20,
          "⭐ 30 题时区间宽 > 0.20 —— 这就是为什么边界值不能当定论",
          f"宽度 {hi3 - lo3:.3f}")
    # 区间永远不许越界（正态近似在 p≈0 时会算出负下界，Wilson 不会）
    bad_ci = [wilson_ci(k, n) for n in (1, 5, 30, 100) for k in range(n + 1)]
    check(all(0.0 <= a <= b <= 1.0 for a, b in bad_ci),
          "所有 (k,n) 组合下区间都在 [0,1] 内（Wilson 不越界）",
          f"试了 {len(bad_ci)} 组")

    # ② 分片接得上（且 gate 分片非空 —— 空分片会让判据永远 FAIL_HARD）
    print("\n   ② 分片")
    sp = build_splits(limit=30)
    for k in ("covered", "uncovered", "unseen"):
        check(len(sp[k]) > 0, f"{k} 非空", f"{len(sp[k])} 题")
    for name, short, _ in JUDGED:
        check(len(sp[name]) > 0,
              f"⭐ 判据分片 {name}（{short}）非空",
              f"{len(sp[name])} 题 —— 空了的话判据会永远 FAIL_HARD")

    # ④ 成本预估算得对（防止手滑把趟数算错，多烧钱）
    print("\n   ③ 成本预估")
    e = estimate(30, 4, 1)
    check(e["episodes"] == 120, "30 题 × 4 条 × 1 分片 = 120 条", str(e["episodes"]))
    check(abs(e["est_minutes"] - (120 * SEC_PER_EPISODE + VLLM_INIT_SEC) / 60) < 0.1,
          "分钟数与实测参数自洽", f"{e['est_minutes']} 分钟")
    check(0 < e["est_yuan"] < 5, "单分片评测花费 < ¥5", f"¥{e['est_yuan']}")

    # ④ 干跑一遍**真代码路径** —— 前面三项都是纯函数，这一项才碰到
    #    evaluate_split / build_report / json.dumps 这条真正会跑的链路。
    #    为什么必须有：昨天那类 bug 全是"逻辑写对了，但没接进流程"。
    print("\n   ④ 干跑管线（真 evaluate_split + build_report，不只是纯函数）")
    from train.engine import ScriptedEngine
    from observe.harness import evaluate_split
    from train.tokenize import load_tokenizer, tool_schemas

    tok, tools = load_tokenizer(), tool_schemas()
    tiny = build_splits(limit=1)
    bad = ScriptedEngine(lambda p: "I can't help with that right now.")
    # 两个判据分片各跑一遍（coverd 也要，因为主判据是它）
    r = evaluate_split("covered", tiny["covered"], bad,
                       n_samples=2, tokenizer=tok, tools=tools)
    r2 = evaluate_split("unseen", tiny["unseen"], bad,
                        n_samples=2, tokenizer=tok, tools=tools)
    rep = build_report("self-test", "", "scripted", 2, 1, 0.0,
                       {"covered": r, "unseen": r2})

    check(r.pass_at_1 == 0.0, "干跑：乱答策略 pass@1 = 0", f"实际 {r.pass_at_1}")
    check(r.zero_var_rate > 0.99,
          "干跑：全错 → 零方差率 ≈ 1", f"实际 {r.zero_var_rate:.3f}")
    check(set(rep["gates"]) == {"covered", "unseen"},
          "干跑：两个口径都出了判据", str(list(rep["gates"])))
    check(rep["primary_split"] == PRIMARY_SPLIT,
          f"干跑：主判据落在 {PRIMARY_SPLIT}", str(rep["primary_split"]))
    check(all(g["level"] == "FAIL_HARD" for g in rep["gates"].values()),
          "干跑：两个口径都正确落到 FAIL_HARD")
    check(rep["gates_disagree"] is False,
          "干跑：两口径结论一致时不报打架")
    check(rep["gates"]["covered"]["n_trajectories"] == 62,
          "干跑：样本数落对了（covered 31 题 × 2 条）",
          str(rep["gates"]["covered"]["n_trajectories"]))
    try:
        blob = json.dumps(rep, ensure_ascii=False)
        ok = all(k in blob for k in ("pass_at_1", "thresholds", "zero_var_rate"))
        check(ok, "干跑：JSON 可序列化且关键字段齐全", f"{len(blob)} 字符")
    except TypeError as ex:
        check(False, "干跑：JSON 可序列化", f"序列化失败：{ex}")

    print()
    print("=" * 78)
    if fails:
        print(f"❌ 自测未通过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ 自测全绿：判据边界正确、分片接得上、成本算得对")
    print("   ⚠️ 但它**验不了真模型** —— 那只能上卡验。")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="零模型自测（本地 CPU）")
    ap.add_argument("--dry-run", action="store_true", help="只算成本，不跑")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录；空 = 量基座")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--splits", default="all",
                    help="all（默认，三片都跑）｜ 或逗号分隔，如 covered,unseen")
    ap.add_argument("--n", type=int, default=4, help="每题采样条数")
    ap.add_argument("--limit", type=int, default=30, help="每片最多多少题")
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    ap.add_argument("--out", default=str(_PROJECT / "reports" / "sft_gate.json"))
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
