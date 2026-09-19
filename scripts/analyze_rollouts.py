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


def load_read_only_tools() -> set:
    """
    只读工具清单 —— **从 data/hackable_analysis.json 读，不写死**。

    写死的话，哪天工具集改了（12/14 个那两个版本），这里的"读/写"分类会静默错位，
    而输出看起来完全正常。拿不到就返回空集并**吵一声**（那样所有工具都会被算成"写"，
    结论会反过来 —— 必须让人看见）。
    """
    p = _PROJECT / "data" / "hackable_analysis.json"
    try:
        return set(json.loads(p.read_text(encoding="utf-8"))["read_only_tools"])
    except Exception as e:                                    # noqa: BLE001
        print(f"   ⚠️ 读不到只读工具清单（{p.name}: {e}）——")
        print("      下面所有工具都会被算成『写工具』，结论方向可能反。别照抄这一节。")
        return set()


def tool_calls_of(rec) -> list:
    """从一条轨迹的 messages 里取出所有工具调用名（顺序保留）。"""
    out = []
    for m in rec.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            out.append(tc.get("function", {}).get("name", "?"))
    return out


def resolve_s_max(rollout_path: Path, cli: int = 0) -> int:
    """
    取**这一轮真正用的** S_max，优先从旁边的 summary.json 读。

    ⚠️ 为什么不能写死：这里原本硬编码 8192，而 9/18 那轮用的是 16384 ——
       于是"截断是不是撞 token 预算"会被算错（prompt ≥ 0.9×S_max 的比例虚高）。
       这就是"写死的数字 + 分析脚本 = 数字口径不一致"那个形状。
       真值拿不到就**明说这是假设值**，不静默用一个错的数。
    """
    if cli:
        print(f"   （S_max = {cli}，由 --s-max 指定）")
        return cli
    summ = rollout_path.with_suffix(".summary.json")
    if summ.exists():
        try:
            v = json.loads(summ.read_text(encoding="utf-8")).get("max_seq_tokens")
            if v:
                print(f"   （S_max = {v}，取自 {summ.name}）")
                return int(v)
        except Exception:
            pass
    print("   ⚠️ 旁边 summary.json 里没有 max_seq_tokens，**按 8192 假设** ——")
    print("      这一轮的截断诊断可能不准；请用 `--s-max <实际值>` 显式传入。")
    return 8192


def bar(frac: float, width: int = 28) -> str:
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    # 顺手挑出 --s-max（这个脚本原本是纯位置参数的用法，不动它）
    s_max_cli, rest, i = 0, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--s-max" and i + 1 < len(argv):
            s_max_cli = int(argv[i + 1]); i += 2; continue
        if a.startswith("--s-max="):
            s_max_cli = int(a.split("=", 1)[1]); i += 1; continue
        rest.append(a); i += 1
    argv = rest
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
    S_MAX = resolve_s_max(path)
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

    # ══════════════════════════════════════════════════════════════════
    # ⑦⑧⑨ 工具行为解剖
    #
    # ⭐ 为什么必须有这三节：光看 tool_calls_mean 一个数，**分不清**
    #    「学会了少调工具」和「放弃尝试 / 空转」—— 而这两件事的含义完全相反。
    #    H6 报告里那条保留意见（gold 7.1 是成功轨迹的统计，我们 pass 只有 15%）
    #    就是冲这个来的，这三节是它的解药。
    # ══════════════════════════════════════════════════════════════════

    ok_rows = [r for r in rows if r["reward"] >= 1.0 - 1e-9]
    bad_rows = [r for r in rows if r["reward"] < 1.0 - 1e-9]

    print("\n⑦ 工具调用 **按结局拆**（同一策略、同一分布下比 —— 修掉取样偏差）\n")
    print("   ⚠️ gold 的 7.1 是**成功轨迹**的统计。我们 pass_rate 才 ~15%，")
    print("      拿失败轨迹去比它本身有偏。正确的对照是下面这两行之比。\n")
    _mean = lambda rs, k: (st.mean([r[k] for r in rs]) if rs else float("nan"))
    print(f"   {'':<8}{'条数':>6}{'工具调用':>10}{'轮数':>8}{'截断率':>9}{'格式崩':>9}")
    for label, rs in (("成功", ok_rows), ("失败", bad_rows)):
        if not rs:
            print(f"   {label:<8}{0:>6}   —— 这一轮没有{label}轨迹")
            continue
        tr = sum(1 for r in rs if r["terminated_by"] in TRUNCATED) / len(rs)
        print(f"   {label:<8}{len(rs):>6}{_mean(rs,'n_tool_calls'):>10.2f}"
              f"{_mean(rs,'n_turns'):>8.1f}{tr:>9.1%}{_mean(rs,'n_malformed'):>9.2f}")
    if ok_rows and bad_rows:
        d = _mean(ok_rows, "n_tool_calls") - _mean(bad_rows, "n_tool_calls")
        verdict = ("成功轨迹**调得更多** → 少调不是「学会做对」的路径"
                   if d > 0.5 else
                   "成功轨迹**调得更少** → 与「欠调用 = 在学正确行为」一致" if d < -0.5 else
                   "两者**基本持平** → 总 tool_calls 下降**不能**归因于「成功需要少调」")
        print(f"\n   ⇒ 成功 − 失败 = {d:+.2f} ｜ {verdict}")
        print("      ⭐ 这一行才是「欠调用」这个说法成不成立的判据。")

    # ---------------- ⑧ 读工具 vs 写工具
    print("\n⑧ 掉的是**读**工具还是**写**工具？\n")
    READ_ONLY = load_read_only_tools()
    per_tool, per_tool_ok, per_tool_bad = Counter(), Counter(), Counter()
    for r in rows:
        names = tool_calls_of(r)
        per_tool.update(names)
        # ⚠️ 别写成 `r in ok_rows` —— 那是按 dict **相等**判断，两条内容相同的轨迹会串。
        if r["reward"] >= 1.0 - 1e-9:
            per_tool_ok.update(names)
        else:
            per_tool_bad.update(names)
    if READ_ONLY:
        rd = sum(v for k, v in per_tool.items() if k in READ_ONLY)
        wr = sum(v for k, v in per_tool.items() if k not in READ_ONLY)
        tot = max(1, rd + wr)
        print(f"   读工具 {rd:>6}（{rd/tot:5.1%}）｜ 写工具 {wr:>6}（{wr/tot:5.1%}）")
        print(f"   {'':<26}{'读':>8}{'写':>8}")
        for label, cnt in (("成功轨迹", per_tool_ok), ("失败轨迹", per_tool_bad)):
            a = sum(v for k, v in cnt.items() if k in READ_ONLY)
            b = sum(v for k, v in cnt.items() if k not in READ_ONLY)
            print(f"   {label:<26}{a:>8}{b:>8}")
        print("\n   ⇒ 若**写**工具掉得比读工具狠 → 更像『学会不动手』（可能是题目结构喂出来的）；")
        print("     读**写**一起掉 → 更像整体退化 / 空转。")
        print(f"\n   逐工具计数：{dict(per_tool.most_common())}")

    # ---------------- ⑨ 不调工具的那些 turn 在干什么
    print("\n⑨ **不调工具的 assistant turn** 在干什么？（决定叫『欠调用』还是『空转』）\n")
    text_turns, n_tool_turns, empty_turns = [], 0, 0
    for r in rows:
        for m in r.get("messages") or []:
            if m.get("role") != "assistant":
                continue
            if m.get("tool_calls"):
                n_tool_turns += 1
            else:
                c = (m.get("content") or "").strip()
                text_turns.append(len(c))
                if len(c) <= 5:                 # 空的 / 只有标点 = 没说话也没做事
                    empty_turns += 1
    n_text = len(text_turns)
    print(f"   assistant turn 总数 {n_text + n_tool_turns}"
          f" ｜ 带工具调用 {n_tool_turns} ｜ **不带工具调用 {n_text}**")
    if n_text:
        print(f"   其中近乎空的（≤5 字符）{empty_turns}（{empty_turns/n_text:.1%}）")
        print(f"   文本长度：均值 {st.mean(text_turns):.0f} 字符 ｜ 中位 {st.median(text_turns):.0f}"
              f" ｜ p90 {sorted(text_turns)[int(0.9*(n_text-1))]} ｜ 最大 {max(text_turns)}")
        print(f"   这些 turn 贡献的字符总量 {sum(text_turns):,}")
        verdict = ("**输出很啰嗦** → 是『少做事、多说话』，不是少做事"
                   if st.median(text_turns) > 200 else
                   "**输出很短** → 更像空转 / 复读，不是有效推理")
        print(f"\n   ⇒ {verdict}")

    print("\n" + "=" * 78)
    print("下一步怎么定：")
    print("  · 截断**分散在所有题**上 → 加 --max-seq-tokens 16384 重跑一轮（治标）")
    print("  · 有信号的题 < 5 道       → 光加预算不够，得回去加 SFT（治本）")
    print("  · 若「全对」的题明显变多  → 那是好事，说明 SFT 真的学进去了")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
