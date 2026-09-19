# -*- coding: utf-8 -*-
"""
块 H5 · 训练逻辑自测（零模型 / 零 GPU / 断网可跑）。

为什么在租卡之前必须跑这个：
    租卡是按小时烧钱的。**训练脚本里最贵的 bug 不是崩溃，是"看起来在训"。**
    比如 loss mask 错位几个 token —— 训练照跑，loss 照降，
    只是模型顺便学会了"把用户说的话也复述一遍"，你要到评测时才发现。

验七件事：
    ① 两路 mask 对账      字符偏移法 vs 增量差分法，**必须逐 token 一致**
    ② mask 语义          解回文本，确认算 loss 的正好是 agent 说的话
    ③ 负例（关键）       故意错位 1 个 token，看对账**抓不抓得住**
    ④ 渲染一致性         SFT 和 rollout 用的是同一份 prompt（含工具 schema）
    ⑤ 分组采样           lockstep 跑通；全对组 adv 全 0；混采组 adv 有正有负
    ⑥ 截断与格式崩       撞上限给 0 分；半截 tool_call 能被计数（不是静默吞掉）
    ⑦ 接观测器           observe.metrics.compute 能吃这批 episode，算出零方差率

跑法（纯 CPU）：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/eval_train_logic.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import TASKS                                                    # noqa: E402
from train.engine import ScriptedEngine                                  # noqa: E402
from train.tokenize import (                                             # noqa: E402
    DEFAULT_TOOLS, assistant_char_spans, encode, encode_by_incremental,
    load_tokenizer, render, tool_schemas,
)
from train.rollout_batch import (                                        # noqa: E402
    RolloutConfig, compute_advantages, load_rollouts, run_grouped_rollout, save_rollouts,
)

SFT = PROJECT / "data" / "sft_final.jsonl"
SPLIT = PROJECT / "data" / "task_split.json"

OK, FAIL = "[OK]", "[!!]"
_fails = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    print(f"   {OK if cond else FAIL} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond


# ---------------------------------------------------------------- 假引擎


def gold_text(action) -> str:
    """把 τ-bench 的 Action 变成模型会吐的那段文本（模板里的格式）。"""
    args = json.dumps(action.kwargs, ensure_ascii=False)
    if action.name == "respond":
        return action.kwargs.get("content", "")
    return ('<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>'
            % (action.name, json.dumps(args, ensure_ascii=False)))


class GoldEngine(ScriptedEngine):
    """
    照本宣科的假引擎：从 prompt 里**数已经出现了几个工具调用**，
    据此决定下一个动作是 gold 的第几个 —— 这样它能在 lockstep 里同时服务所有题。

    （比"给每条轨迹配一个计数器"稳：prompt 本身就是状态，不会和真实轨迹走岔。）

    ⚠️ 踩过两次同一个坑（都跟"在 prompt 里数标签"有关）：
       ① 数裸 `<tool_call>` → 模板里写着 "within <tool_call></tool_call> XML tags"，从 2 起
       ② 数 `<tool_call>\n{"name"` → 模板里**自带一段示例**：
          `return a json object ... :\\n<tool_call>\\n{"name": <function-name>, ...`
          于是从 1 起 —— 假引擎一上来就跳到 gold 的第 2 个动作，静默跑偏。
       ③ 数 `</tool_call><|im_end|>` → 示例那段结尾也是这个，照样不唯一。
       ✅ 最终用 `<tool_response>`：**每个真正执行过的工具调用恰好产生一个**，
          而模板的工具说明里根本没有这个字。
    """

    MARK = "<tool_response>"

    def __init__(self, tasks):
        self.index = {t.instruction[:60]: t for t in tasks}
        super().__init__(self._respond)

    def _respond(self, prompt: str) -> str:
        task = None
        for key, t in self.index.items():
            if key in prompt:
                task = t
                break
        if task is None:
            return "Sorry, could you repeat that?"
        k = prompt.count(self.MARK)
        if k < len(task.actions):
            return gold_text(task.actions[k])
        outs = task.outputs
        return ("Here is the information you asked for: " + ", ".join(outs) + "."
                if outs else "All set. Anything else I can help with?")


class NoopEngine(ScriptedEngine):
    """什么都不做，直接说收尾话。"""

    def __init__(self):
        super().__init__(lambda p: "I can't help with that right now.")


class BrokenEngine(ScriptedEngine):
    """格式崩：吐出半截 tool_call（模拟被截断 / 模型没学会闭合标签）。"""

    def __init__(self):
        super().__init__(lambda p: '<tool_call>\n{"name": "get_user_details", "argu')


# ---------------------------------------------------------------- 用例数据


def load_cases(tokenizer, tools):
    """SFT 数据 + 合成轨迹，凑一批"结构各不相同"的用例。"""
    rows = [json.loads(l) for l in SFT.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [(r["task_id"], r["messages"]) for r in rows[:8]]

    # 合成一条：工具轮 + 纯话术轮 + 带工具的多轮，专门覆盖边界
    t0 = TASKS[0]
    synth = [
        {"role": "system", "content": rows[0]["messages"][0]["content"]},
        {"role": "user", "content": t0.instruction},
        {"role": "assistant", "content": "Sure, let me look that up."},
        {"role": "user", "content": "Thanks."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_user_details",
                                      "arguments": '{"user_id": "x_1"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_user_details",
         "content": '{"name": "X"}'},
        {"role": "assistant", "content": "All done."},
    ]
    cases.append((-1, synth))
    return cases


# ---------------------------------------------------------------- ① ② ③


def test_mask(tokenizer, tools, cases):
    print("\n① 两路 mask 对账（字符偏移法 vs 增量差分法）")
    n_token, n_mismatch, ratios = 0, 0, []
    for tid, msgs in cases:
        e1 = encode(msgs, tokenizer, tools)
        m2 = encode_by_incremental(msgs, tokenizer, tools)
        n_token += e1.n_tokens
        if e1.loss_mask != m2:
            n_mismatch += 1
            print(f"      ⚠️ task {tid}: 两路 mask 不一致 "
                  f"(差 {sum(1 for a, b in zip(e1.loss_mask, m2) if a != b)} 个 token)")
        ratios.append(e1.loss_ratio)
    check(n_mismatch == 0, f"{len(cases)} 条轨迹两路 mask 逐 token 一致",
          f"共 {n_token} token，loss 占比 {sum(ratios)/len(ratios):.1%}")

    print("\n② mask 语义：解回文本，确认算 loss 的正好是 agent 说的话")
    tid, msgs = cases[0]
    enc = encode(msgs, tokenizer, tools)
    n_asst = sum(1 for m in msgs if m["role"] == "assistant")
    check(len(enc.spans) == n_asst, f"区间数 = assistant 消息数（{enc.spans} vs {n_asst}）")

    leaked = []
    for a, b, _cs, _ce in enc.spans:
        s = tokenizer.decode(enc.input_ids[a:b], skip_special_tokens=False)
        body = s.replace("<|im_end|>", "")
        # 每条都必须"起于 assistant 内容、收于 <|im_end|>"，且不是工具返回
        if s.count("<|im_start|>") or "<tool_response>" in s:
            leaked.append(s[:60])
    check(not leaked, "没有一条 loss 区间混进 role 标记或工具返回", str(leaked[:2]))

    covered = sum(b - a for a, b, _c, _d in enc.spans)
    check(covered == enc.n_loss_tokens, "区间长度之和 = mask 的 1 的个数",
          f"{covered} == {enc.n_loss_tokens}")

    print("\n③ 负例：把 mask 故意错位 1 个 token，对账抓不抓得住")
    _, msgs = cases[0]
    e1 = encode(msgs, tokenizer, tools)
    m2 = encode_by_incremental(msgs, tokenizer, tools)
    shifted = list(m2[1:]) + [0]                    # 整体左移一格
    check(shifted != e1.loss_mask,
          "错位 1 token 会被发现（说明①的断言不是永远为真）",
          f"错位后有 {sum(1 for a, b in zip(shifted, e1.loss_mask) if a != b)} 个 token 不同")

    # 再狠一点：只错一个 token（边界最容易漏的那种）
    one = list(m2)
    idx = next(i for i, v in enumerate(one) if v == 1)
    one[idx] = 0
    check(one != e1.loss_mask, "少一个 token 也会被发现")


# ---------------------------------------------------------------- ④


def test_render_consistency(tokenizer, tools):
    print("\n④ 渲染一致性：SFT 数据和 rollout 用的必须是同一份 prompt")
    rows = [json.loads(l) for l in SFT.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = rows[0]
    msg = r["messages"][:2]                          # system + 第一条 user

    a = render(msg, tools, tokenizer, add_generation_prompt=True)
    b = render(msg, tools, tokenizer, add_generation_prompt=True)
    check(a == b, "同一个函数两次调用逐字节一致（确定性）")

    no_tools = render(msg, [], tokenizer, add_generation_prompt=True)
    check(a != no_tools, "不传 tools 渲染出来的 prompt 不一样（说明 schema 真的进了 prompt）",
          f"带 tools {len(a)} 字符 vs 不带 {len(no_tools)} 字符，差 {len(a)-len(no_tools)}")

    check(DEFAULT_TOOLS and len(DEFAULT_TOOLS) == 12, "工具子集是 12 个（14 − think − list_all_airports）")

    # 结尾必须是生成提示，否则模型不知道该接着说话
    check(a.endswith("<|im_start|>assistant\n"), "prompt 结尾是生成提示",
          repr(a[-24:]))


# ---------------------------------------------------------------- ⑤ ⑥


def test_grouped_rollout(tokenizer, tools):
    print("\n⑤ 分组采样（lockstep）")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_ids = [r["task"] for r in split["train"]]
    # 挑 4 道【纯 S 类且无 outputs】的题：gold 能跑满分、不说 outputs 也不会掉分
    clean = [i for i in train_ids if TASKS[i].outputs == []][:4]
    items = [(i, TASKS[i]) for i in clean]

    cfg = RolloutConfig(n_group=8, max_turns=40)
    batch = run_grouped_rollout(items, GoldEngine(TASKS), cfg, tokenizer, tools)

    check(len(batch.groups) == 4, f"组数 = 4（实际 {len(batch.groups)}）")
    check(all(len(g.rewards) == 8 for g in batch.groups), "每组 8 条轨迹")
    check(all(not g.zero_variance or g.std == 0 for g in batch.groups), "零方差组的 std 确实是 0")

    gold_ok = sum(1 for g in batch.groups if g.mean == 1.0)
    check(gold_ok == 4, f"照本宣科 4 道题全部满分（实际 {gold_ok}/4）")
    check(batch.zero_var_rate == 1.0,
          "⭐ 全对 → 零方差率 100%（这正是要监控的'白跑一步'）",
          f"zero_var_rate={batch.zero_var_rate:.2f}")
    check(all(abs(a) < 1e-9 for g in batch.groups for a in g.advantages),
          "全对组 advantage 恒为 0 → 零梯度")

    print("\n   混采：一半照本宣科、一半什么都不做")
    mixed = []
    for i in clean[:2]:
        mixed.append((i, TASKS[i]))
    b_gold = run_grouped_rollout([mixed[0]], GoldEngine(TASKS), cfg, tokenizer, tools)
    b_noop = run_grouped_rollout([mixed[0]], NoopEngine(), cfg, tokenizer, tools)

    rewards = b_gold.groups[0].rewards + b_noop.groups[0].rewards
    advs, sd, zero = compute_advantages(rewards)
    check(not zero and sd > 0, f"混采组 std={sd:.3f} > 0")
    check(any(a > 0 for a in advs) and any(a < 0 for a in advs),
          "优势有正有负（梯度真正指向'做对'）",
          f"adv 取值 {sorted(set(round(a,3) for a in advs))}")

    print("\n⑥ 截断与格式崩（都要能被看见，不能静默吞掉）")

    # 预算装不下固定开销时，必须**炸掉**，不能静默把所有轨迹判成超长
    raised = ""
    try:
        run_grouped_rollout([(clean[0], TASKS[clean[0]])], GoldEngine(TASKS),
                            RolloutConfig(n_group=1, max_seq_tokens=2048), tokenizer, tools)
    except ValueError as e:
        raised = str(e).splitlines()[0]
    check(bool(raised), "预算装不下固定开销时直接报错（不静默截断）", raised[:90])

    trunc = run_grouped_rollout(
        [(clean[0], TASKS[clean[0]])], GoldEngine(TASKS),
        RolloutConfig(n_group=2, max_turns=1), tokenizer, tools)
    n_trunc = sum(1 for e in trunc.episodes if e.terminated_by == "max_turns")
    check(n_trunc == 2, f"撞 max_turns 的轨迹被正确标记（{n_trunc}/2）")
    check(all(e.reward == 0.0 for e in trunc.episodes),
          "⭐ 截断轨迹给 0 分，不给部分奖励（否则等于奖励'啰嗦到被截断'）")

    broken = run_grouped_rollout(
        [(clean[0], TASKS[clean[0]])], BrokenEngine(),
        RolloutConfig(n_group=2, max_turns=4), tokenizer, tools)
    n_mal = sum(s.n_malformed for s in broken.groups[0].samples)
    kinds = {k for s in broken.groups[0].samples for k in s.parse_kinds}
    check(n_mal > 0, f"半截 <tool_call> 被计为 malformed（{n_mal} 次）", f"解析类型={sorted(kinds)}")
    check(broken.episodes[0].reward == 0.0, "格式崩拿不到分")

    return batch, items


# ---------------------------------------------------------------- ⑦


def test_observer_and_io(batch, items, tokenizer, tools):
    print("\n⑦ 接观测器 + 落盘读回")
    try:
        from observe.metrics import compute as obs_compute, render as obs_render
        split = json.loads(SPLIT.read_text(encoding="utf-8"))
        task_meta = {r["task"]: r for r in split["train"] + split["probe_overcall"]}
        m = obs_compute(batch.episodes, task_meta, extra={"entropy": 0.0, "kl": 0.0})
        check("zero_var_rate" in m["health"], "观测器算出了组内零方差率",
              f"{m['health']['zero_var_rate']:.2f}")
        check("under_call_rate" in m["degradation"], "退化谱指标在（欠调用等）")
        print("      " + obs_render(m).replace("\n", "\n      "))
    except Exception as e:                                    # noqa: BLE001
        check(False, "观测器接入", f"{type(e).__name__}: {e}")

    out = PROJECT / "data" / "_tmp_h5_rollouts.jsonl"
    n = save_rollouts(batch, out, step=0)
    back = load_rollouts(out)
    check(n == len(batch.episodes) == len(back), f"jsonl 往返条数一致（{n}）")
    check(back[0]["messages"] == batch.episodes[0].messages, "消息逐字段一致")
    check(all("advantage" in r and "group_std" in r for r in back), "每条都带 advantage 与组 std")
    out.unlink(missing_ok=True)


# ---------------------------------------------------------------- main


def main() -> int:
    print("=" * 78)
    print("块 H5 · 训练逻辑自测（零模型 / 零 GPU）")
    print("=" * 78)

    tokenizer = load_tokenizer()
    tools = tool_schemas()
    check(len(tools) == 12, "工具 schema 取到 12 个")

    cases = load_cases(tokenizer, tools)
    test_mask(tokenizer, tools, cases)
    test_render_consistency(tokenizer, tools)
    batch, items = test_grouped_rollout(tokenizer, tools)
    test_observer_and_io(batch, items, tokenizer, tools)

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 自检未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    print("✅ H5 训练逻辑自测全绿")
    print("   · loss mask：两路算法逐 token 对账一致，错 1 个 token 也抓得住")
    print("   · 渲染：SFT 与 rollout 共用同一份 prompt（含 12 个工具 schema）")
    print("   · 分组采样：全对组 adv 全 0 ｜ 混采组 adv 有正负 ｜ 截断给 0 ｜ 格式崩可见")
    print("   · 观测器与落盘：指标算得出、jsonl 往返一致")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
