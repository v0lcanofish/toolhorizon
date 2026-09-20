# -*- coding: utf-8 -*-
"""
T8 · 真实轨迹的 token 长度分布实测（Stage 0 出口判据之一）。

为什么必须实测：README 的显存账建立在「轨迹长度」上，而 longhorizon 标称 S_max=8192/16K。
标称值不能拿来算显存 —— 必须用**真实成功轨迹**量一遍。

数据源：tau-bench/historical_trajectories/ 里 gpt-4o / sonnet 的 airline 成功轨迹（reward=1）
方法：用 Qwen2.5-1.5B 的 tokenizer + apply_chat_template(tools=...) 还原模型真正看到的 token 序列

⚠️ 解释器必须是有 torch/transformers 的那个环境（与跑 τ-bench 的环境可能不同）

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.9 scripts/token_stats.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

TOK_DIR = PROJECT / "models" / "qwen1.5b-tokenizer"
TRAJ_DIR = (PROJECT.parent / "reference-repos" / "agentic-grpo-longhorizon"
            / "tau-bench" / "historical_trajectories")


def pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[i]


def mean(v):
    return sum(v) / len(v) if v else 0.0


def load_schema(name):
    p = PROJECT / "data" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ⚠️ 为什么手工渲染而不是 apply_chat_template：
#    曾遇到 conda base 的 jinja2 是 2.11.3，而 transformers 4.57 需要 jinja2 ≥ 3.1
#    （chat_template_utils 里用了 jinja2.pass_eval_context）。
#    升级 jinja2 有连带风险，**不动环境**，改成按 Qwen2.5 的模板手工拼串。
#    拼法与官方模板一致：工具定义注入 system 段；每轮 <|im_start|>role ... <|im_end|>

TOOLS_HDR = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_FTR = "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"


def _block(role, body):
    return f"<|im_start|>{role}\n{body}<|im_end|>\n"


def render(messages, tools, tok):
    """按 Qwen2.5 模板手工还原 token 序列。"""
    try:
        parts = []
        msgs = messages
        # 第一条 system（含工具定义）
        if msgs and msgs[0].get("role") == "system":
            sys_body = msgs[0].get("content") or ""
            msgs = msgs[1:]
        else:
            sys_body = (f"You are a helpful assistant.{TOOLS_HDR}"
                        f"{json.dumps(tools, ensure_ascii=False, indent=4)}{TOOLS_FTR}"
                        if tools else "You are a helpful assistant.")
        if tools:
            sys_body += TOOLS_HDR + json.dumps(tools, ensure_ascii=False, indent=4) + TOOLS_FTR
        parts.append(_block("system", sys_body))

        for m in msgs:
            role = m.get("role")
            body = m.get("content") or ""
            if role == "assistant":
                tc = m.get("tool_calls")
                if tc:
                    calls = []
                    for c in tc:
                        fn = c.get("function") or {}
                        calls.append({"name": fn.get("name", ""),
                                      "arguments": fn.get("arguments", "{}")})
                    body = (body or "") + "".join(
                        f"\n<tool_call>\n{json.dumps(c, ensure_ascii=False)}\n</tool_call>"
                        for c in calls)
                parts.append(_block("assistant", body))
            elif role == "tool":
                parts.append(_block("user", f"<tool_response>\n{body}\n</tool_response>"))
            else:
                parts.append(_block(role, body))
        text = "".join(parts)
        return tok.encode(text, add_special_tokens=False)
    except Exception as e:                                # noqa: BLE001
        print(f"  ⚠️ 渲染失败: {type(e).__name__}: {e}")
        return None


def main() -> int:
    from transformers import AutoTokenizer

    print("=" * 78)
    print("T8 · 真实轨迹 token 长度分布实测")
    print("=" * 78)

    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
    print(f"tokenizer: Qwen2.5-1.5B ｜ vocab = {tok.vocab_size:,}\n")

    schema14 = load_schema("tool_schema_14.json")
    schema12 = load_schema("tool_schema_12.json")

    def schema_tokens(s):
        """schema 的固定开销 = 有工具定义 vs 无工具定义的 system 段之差。"""
        if not s:
            return None
        base = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "x"}]
        with_t = len(render(base, s, tok))
        without = len(render(base, None, tok))
        return with_t - without

    n14, n12 = schema_tokens(schema14), schema_tokens(schema12)
    print("-" * 78)
    print("工具 schema 的固定开销")
    print("-" * 78)
    print(f"  14 个工具: {len(json.dumps(schema14, ensure_ascii=False)):>6,} 字符 "
          f"→ {n14:>6,} token")
    if n12 is not None:
        print(f"  12 个工具: {len(json.dumps(schema12, ensure_ascii=False)):>6,} 字符 "
              f"→ {n12:>6,} token")
    print()

    # ---------------------------------------------------------- 真实轨迹
    all_tot, all_asst, all_msgs = [], [], []
    for fn in ["gpt-4o-airline.json", "sonnet-35-new-airline.json"]:
        p = TRAJ_DIR / fn
        if not p.exists():
            print(f"  ⚠️ 缺文件 {fn}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        ok = [x for x in data if x.get("reward") == 1.0]
        n_ok = 0
        for x in ok:
            ids = render(x["traj"], schema14, tok)
            if ids is None:
                continue
            n_ok += 1
            all_tot.append(len(ids))
            all_asst.append(sum(
                1 for m in x["traj"]
                if m.get("role") == "assistant"
                for _ in tok((m.get("content") or "")).get("input_ids", [])
            ))
            all_msgs.append(len(x["traj"]))
        print(f"  {fn:<32} 成功 {len(ok):>3} 条，成功渲染 {n_ok}")

    if not all_tot:
        print("❌ 没有可渲染的轨迹")
        return 1

    # ---------------------------------------------------------- 结果
    print()
    print("-" * 78)
    print(f"真实成功轨迹的 token 长度（n = {len(all_tot)}，含 14 工具 schema）")
    print("-" * 78)
    hdr = f"{'指标':<12}{'总 token':>12}{'assistant token':>18}{'消息数':>10}"
    print(hdr)
    for label, vals in [("p50", [pct(all_tot, 50), pct(all_asst, 50), pct(all_msgs, 50)]),
                        ("p90", [pct(all_tot, 90), pct(all_asst, 90), pct(all_msgs, 90)]),
                        ("max", [max(all_tot), max(all_asst), max(all_msgs)]),
                        ("mean", [mean(all_tot), mean(all_asst), mean(all_msgs)])]:
        print(f"{label:<12}{vals[0]:>12,.0f}{vals[1]:>18,.0f}{vals[2]:>10.1f}")

    print()
    print(f"  assistant token 占比："
          f"{mean(all_asst)/mean(all_tot)*100:.1f}%  "
          f"（其余是 system / user / tool 结果，不过 loss 但仍要过 forward）")

    # ---------------------------------------------------------- 对照标称
    print()
    print("=" * 78)
    print("对照与含义")
    print("=" * 78)
    print(f"  longhorizon 标称 S_max = 8192 (+12288) = 20,480")
    print(f"  实测成功轨迹中位数 = {pct(all_tot,50):,.0f} token")
    print(f"  → 标称值是实测中位数的 {8192/pct(all_tot,50):.1f}×，"
          f"是实测 p90 的 {8192/pct(all_tot,90):.1f}×")
    print()
    print("  对显存账的影响：")
    print(f"    若按标称 8192 算 logits: 8192 × {tok.vocab_size} × 4B "
          f"= {8192*tok.vocab_size*4/1e9:.2f} GB / 张量")
    print(f"    若按实测 p90  {pct(all_tot,90):,.0f}: "
          f"{pct(all_tot,90)*tok.vocab_size*4/1e9:.2f} GB / 张量")

    out = PROJECT / "data" / "token_stats.json"
    out.write_text(json.dumps({
        "n": len(all_tot), "vocab": tok.vocab_size,
        "schema_tokens_14": n14, "schema_tokens_12": n12,
        "total": {"p50": pct(all_tot, 50), "p90": pct(all_tot, 90),
                  "max": max(all_tot), "mean": mean(all_tot)},
        "assistant": {"p50": pct(all_asst, 50), "p90": pct(all_asst, 90),
                      "max": max(all_asst), "mean": mean(all_asst)},
        "n_messages": {"p50": pct(all_msgs, 50), "max": max(all_msgs),
                       "mean": mean(all_msgs)},
        "all_tot": all_tot,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"结果已存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
