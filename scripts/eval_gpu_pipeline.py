# -*- coding: utf-8 -*-
"""
块 H5 · **上卡前的全链路彩排**（零 GPU / 零下载）。

`scripts/gpu_smoke.py` 要在租来的卡上跑，可它本身也可能有 bug。
这个脚本做的事是：**把那条链路在本地用假引擎走一遍**，
把"只有上了卡才会暴露"的问题（文件格式、字段名、数据读回、形状对不上）提前挡掉。

走的就是 `train/run.py` 那一圈的**真实函数**：

    collect： 选题 → run_grouped_rollout → save_rollouts → jsonl
    train  ： load_rollouts → batches_from_records → grpo_step → 反向 → 存 adapter

差别只有一个：策略引擎从 vLLM 换成**照本宣科的假引擎**。
正因为只差这一个，链路上任何别的毛病都会在这里现形。

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/eval_gpu_pipeline.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

import torch                                                          # noqa: E402

from env import TASKS                                                 # noqa: E402
from train.collect import select_tasks                                # noqa: E402
from train.engine import ScriptedEngine                               # noqa: E402
from train.modeling import attach_lora, build_tiny_model, collate, forward_logp  # noqa: E402
from train.rollout_batch import (                                     # noqa: E402
    RolloutConfig, load_rollouts, run_grouped_rollout, save_rollouts,
)
from train.tokenize import load_tokenizer, tool_schemas               # noqa: E402
from train.trainer import batches_from_records, grpo_step, sft_step   # noqa: E402

_fails = []


def check(cond, label, detail=""):
    print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond


NOOP_TEXT = "I can't help with that right now."


class MixedEngine(ScriptedEngine):
    """
    混合假策略（替代 vLLM）：**同一道题的 4 条采样里，一半做对、一半什么都不做**。

    ⚠️ 这里有一个不写出来就会踩的坑：**确定性的假引擎会让同一组的 n 条完全一样**，
       于是组内 std 恒为 0、advantage 恒为 0 —— 看起来"跑通了"，
       实际上**一条非零 advantage 的路径都没走到**，训练段等于没验。

    做法：靠**样本序号**决定（第 1、3 条不做，第 0、2 条做）。
       第 2 轮之后靠 prompt 里有没有 NOOP_TEXT 认出"这条是偷懒那条"，保持一致。
    """

    def __init__(self, pool):
        self.pool = pool
        self.turn0_count = {}
        super().__init__(self._respond)

    def _respond(self, prompt: str) -> str:
        for t in self.pool:
            if t.instruction not in prompt:
                continue
            if NOOP_TEXT in prompt:              # 认出来了：这条从头偷懒到尾
                return NOOP_TEXT
            n = prompt.count("<tool_response>")
            if n == 0:                           # 第一轮，按样本序号分派
                k = self.turn0_count.get(t.instruction, 0)
                self.turn0_count[t.instruction] = k + 1
                if k % 2 == 1:
                    return NOOP_TEXT
            if n < len(t.actions):
                a = t.actions[n]
                s = json.dumps(a.kwargs, ensure_ascii=False)
                return ('<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>'
                        % (a.name, json.dumps(s, ensure_ascii=False)))
            outs = getattr(t, "outputs", None) or []
            return ("Here is the information you asked for: " + ", ".join(outs) + "."
                    if outs else "All set. Anything else I can help you with?")
        return "Sorry, could you repeat that?"


def main() -> int:
    print("=" * 78)
    print("块 H5 · 上卡前全链路彩排（假引擎替代 vLLM，其余全是真的）")
    print("=" * 78)

    tok = load_tokenizer()
    tools = tool_schemas()
    run_dir = PROJECT / "runs" / "_selftest"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================ ① collect 段
    print("\n① collect 段：选题 → 采样 → 落盘")
    items = select_tasks("train", limit=4)
    check(len(items) == 4, f"选题拿到 {len(items)} 道")

    # 每一组里一半做对、一半不做 —— 这样组内才有方差，advantage 才有正有负
    pool = [t for _, t in items]
    eng = MixedEngine(pool)
    cfg = RolloutConfig(n_group=4, max_seq_tokens=8192)
    batch = run_grouped_rollout(items, eng, cfg, tok, tools)

    roll = run_dir / "step_000.jsonl"
    n = save_rollouts(batch, roll, step=0)
    check(n == 16, f"写出 {n} 条（4 道 × 4 条）")
    check(roll.exists() and roll.stat().st_size > 0, f"文件非空（{roll.stat().st_size/1024:.0f} KB）")

    # 落盘的字段必须是训练段真会读的那些 —— 少一个 KeyError，多一个没人管
    rec = json.loads(roll.read_text(encoding="utf-8").splitlines()[0])
    need = {"task_id", "sample_idx", "reward", "advantage", "messages"}
    missing = need - set(rec)
    check(not missing, "落盘字段齐全", f"缺 {sorted(missing)}" if missing else
          f"共 {len(rec)} 个字段")
    check(isinstance(rec["messages"], list) and rec["messages"][0]["role"] == "system",
          "messages 是完整对话（首条 system）")

    # ============================================================ ② train 段
    print("\n② train 段：读回 → 拼批 → 算 advantage → 前向反向")
    back = load_rollouts(roll)
    check(len(back) == n, f"读回 {len(back)} 条")

    bs, advs = batches_from_records(back, tok, tools, max_seq_tokens=2048)
    check(len(bs) == len(back), f"每条轨迹一个 batch（{len(bs)}）")
    check(len(set(round(a, 6) for a in advs)) > 1,
          "advantage 有区分度（不是所有样本一个值）",
          f"取值 {sorted(set(round(a, 3) for a in advs))}")

    # 分组必须按 task_id 分对：同一道题的 4 条应当 advantage 加起来 ≈ 0
    from collections import defaultdict
    grp = defaultdict(list)
    for r, a in zip(back, advs):
        grp[r["task_id"]].append(a)
    sums = {k: round(sum(v), 6) for k, v in grp.items()}
    check(all(abs(v) < 1e-6 for v in sums.values()),
          "每组 advantage 之和 ≈ 0（组内归一化的必然结果）", str(sums))

    torch.manual_seed(0)
    model = build_tiny_model()
    # ⚠️ 必须在 attach_lora **之前**把基座权重抄下来 ——
    #    `get_peft_model` 是**原地包装**的，事后 `model.state_dict()` 拿到的
    #    已经是带 `base_layer` / `lora_A` 的 PEFT 状态字典，load 回裸模型会直接报错。
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    lora = attach_lora(model, r=8, alpha=16)
    lora.train()
    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-3)

    hist = []
    for b, a in zip(bs, advs):
        opt.zero_grad()
        loss, stats = grpo_step(lora, b, advantage=a)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    check(all(x == x for x in hist), "全部 loss 有限（无 NaN）")
    check(len(hist) == len(bs), f"跑完 {len(hist)} 步")

    # ============================================================ ③ 存/读 adapter
    print("\n③ 存 adapter → 读回 → 确认能接着训（run.py 每轮都要走这一步）")
    ck = run_dir / "ckpt_000"
    lora.save_pretrained(str(ck))
    check(ck.exists() and any(ck.iterdir()), f"adapter 落盘（{sum(f.stat().st_size for f in ck.iterdir())/1024:.0f} KB）")

    from peft import PeftModel
    # ⚠️ 必须让第二个基座和第一个**权重完全相同** —— `build_tiny_model()` 每次调用
    #    都是新的随机初始化。这正是本脚本第一次跑挂掉的原因（差 0.74）：
    #    adapter 权重是对的，挂在了一个**不同的基座**上。
    #    真实场景里对应的坑是"把 LoRA 挂到错的基座模型"，同样**静默失效、不报错**。
    base2 = build_tiny_model()
    base2.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(base2, str(ck), is_trainable=True)
    with torch.no_grad():
        lp1 = forward_logp(lora, bs[0])
        lp2 = forward_logp(reloaded, bs[0])
    check(torch.allclose(lp1, lp2, atol=1e-5),
          "读回的 adapter 与原模型输出一致（存/读没丢东西）",
          f"最大差 {float((lp1-lp2).abs().max()):.2e}")

    # 反向验证：故意挂到**没对齐权重的**基座上，必须能看出不对
    base3 = build_tiny_model()
    wrong = PeftModel.from_pretrained(base3, str(ck), is_trainable=True)
    with torch.no_grad():
        lp3 = forward_logp(wrong, bs[0])
    check(not torch.allclose(lp1, lp3, atol=1e-3),
          "挂错基座时输出确实不同（说明上一条断言不是永远为真）",
          f"最大差 {float((lp1-lp3).abs().max()):.2e} —— 报错都没有，只会静默变差")

    # ============================================================ ④ 零方差那一步
    print("\n④ ⭐ 组内全对时，这一步训练应当**完全没有梯度**")
    allone = [dict(r, reward=1.0) for r in back[:4]]
    b2, a2 = batches_from_records(allone, tok, tools, max_seq_tokens=1024)
    check(all(abs(x) < 1e-12 for x in a2), "全对 → advantage 全 0")
    lora.zero_grad()
    l0, _ = grpo_step(lora, b2[0], advantage=a2[0])
    l0.backward()
    gnz = [p for n_, p in lora.named_parameters()
           if "lora_" in n_ and p.grad is not None and p.grad.abs().sum() > 0]
    check(abs(l0.item()) < 1e-9 and len(gnz) == 0,
          "loss=0 且梯度全 0 → 这一步是**空转**（loss 曲线看不出来）")

    # 清理
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 彩排未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    print("✅ 全链路彩排通过 —— collect → jsonl → train → adapter 接得上")
    print("   上卡时唯一的变化：把假引擎换成 VLLMEngine（run.py 已经这么做了）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
