# -*- coding: utf-8 -*-
"""
块 H5 · 训练步自测（**真跑 torch，但零 GPU、零下载**）。

用的是一个**随机初始化的小 Qwen2**（`build_tiny_model`，约 1,950 万参数，本地现造）。
验的是训练**逻辑**，不是模型效果 —— 逻辑跟模型多大、权重是什么无关，
所以没必要为了"提前验一下"去拉 3GB 权重。

验六件事：
    ① 分块交叉熵 == transformers 内置损失      （证明省显存不是靠算错）
    ② 单点解析解：某个位置的 loss **精确等于** −log p(该 token)
       ⭐ 这条是**专治 shift-by-one** 的 —— 偏一位就必然对不上
    ③ 负例：故意不 shift，②的等式必须失败   （证明②不是永远为真）
    ④ 梯度只流到 LoRA，基座冻结；漏 enable_input_require_grads 会怎样
    ⑤ mask 真的生效：非 loss 位置不出现在 target 里
    ⑥ 能训：小 batch 过拟合，loss 单调下降

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/eval_train_step.py
"""

import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

import torch                                                            # noqa: E402
import torch.nn.functional as F                                         # noqa: E402

from train.modeling import (                                            # noqa: E402
    IGNORE_INDEX, attach_lora, backbone_and_head, build_tiny_model,
    chunked_ce_loss, collate, enable_grad_checkpointing, forward_loss,
    hf_reference_loss, shift_for_lm, trainable_report,
)
from train.tokenize import (                                            # noqa: E402
    encode, load_tokenizer, tool_schemas,
)

SFT = PROJECT / "data" / "sft_final.jsonl"
_fails = []


def check(cond, label, detail=""):
    print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        _fails.append(label)
    return cond


def load_batch(tokenizer, tools, n=3, max_len=512):
    rows = [json.loads(l) for l in SFT.read_text(encoding="utf-8").splitlines() if l.strip()]
    encs = []
    for r in rows[:n]:
        e = encode(r["messages"], tokenizer, tools, max_length=max_len, truncate_side="left")
        encs.append(e)
    return collate(encs, pad_id=tokenizer.pad_token_id or 0)


def main() -> int:
    print("=" * 78)
    print("块 H5 · 训练步自测（真跑 torch / 零 GPU / 零下载）")
    print("=" * 78)

    torch.manual_seed(0)
    tokenizer = load_tokenizer()
    tools = tool_schemas()
    model = build_tiny_model()
    model.eval()

    batch = load_batch(tokenizer, tools, n=3, max_len=512)
    print(f"\n   batch: {tuple(batch.input_ids.shape)} ｜ 有效 loss token "
          f"{batch.n_loss_tokens} / {batch.input_ids.numel()} "
          f"（{batch.n_loss_tokens / batch.input_ids.numel():.1%}）")

    # ---------------------------------------------------------- ① 分块 CE 对账
    print("\n① 分块交叉熵 vs transformers 内置损失（同一个 batch）")
    inputs, attn, targets = shift_for_lm(batch)
    backbone, head = backbone_and_head(model)
    with torch.no_grad():
        hidden = backbone(input_ids=inputs, attention_mask=attn, use_cache=False).last_hidden_state
        mine = chunked_ce_loss(hidden, targets, head.weight, chunk_size=64)
        ref = hf_reference_loss(model, batch)
    rel = abs(mine.item() - ref.item()) / max(1e-9, abs(ref.item()))
    check(rel < 1e-4, f"两者数值一致（分块 {mine.item():.6f} vs 内置 {ref.item():.6f}）",
          f"相对误差 {rel:.2e}")

    # 分块大小不该改变结果
    with torch.no_grad():
        mine2 = chunked_ce_loss(hidden, targets, head.weight, chunk_size=7)
    check(abs(mine2.item() - mine.item()) < 1e-4, "换 chunk_size 结果不变（7 vs 64）")

    # ---------------------------------------------------------- ② shift 精确解析解
    print("\n② ⭐ shift-by-one 的精确检验（单点解析解）")
    e = encode(json.loads(SFT.read_text(encoding='utf-8').splitlines()[0])["messages"],
               tokenizer, tools, max_length=512, truncate_side="left")
    m = torch.tensor([e.loss_mask], dtype=torch.long)
    # 只留一个 loss 位置，其余全 0。
    # ⚠️ 必须挑中间的位置：max_length 截断会让**第 0 个 token 本身就是 assistant token**
    #    （左截断从一句话中间切开），拿它当锚点的话 `input_ids[:, :pos]` 会是空的。
    nz = [int(x) for x in m.nonzero()[:, 1].tolist()]
    pos = nz[len(nz) // 2]
    assert pos >= 1
    m1 = torch.zeros_like(m)
    m1[0, pos] = 1
    b1 = collate([type(e)(input_ids=e.input_ids, loss_mask=m1[0].tolist(),
                        spans=[], text="", truncated=False)], tokenizer.pad_token_id or 0)
    with torch.no_grad():
        got = forward_loss(model, b1).item()
        # 手算：−log softmax(位置 pos−1 的 logits)[token pos]
        out = model(input_ids=b1.input_ids[:, :pos], use_cache=False)
        logp = F.log_softmax(out.logits[0, pos - 1].float(), dim=-1)
        exact = -logp[b1.input_ids[0, pos]].item()
    check(abs(got - exact) < 1e-4,
          f"单点 loss == −log p(该 token)（{got:.6f} vs {exact:.6f}）",
          f"位置 {pos}")

    # ---------------------------------------------------------- ③ 负例
    print("\n③ 负例：故意不 shift，②的等式必须失败")
    with torch.no_grad():
        out_full = model(input_ids=b1.input_ids, use_cache=False)
        logp0 = F.log_softmax(out_full.logits[0, pos].float(), dim=-1)
        wrong = -logp0[b1.input_ids[0, pos]].item()          # 用「同一位置」预测（错的做法）
    check(abs(wrong - exact) > 1e-3,
          "不 shift 算出来的值确实不同（说明②真的能抓出偏位）",
          f"错法 {wrong:.6f} vs 正法 {exact:.6f}")

    # ---------------------------------------------------------- ④ 梯度流
    print("\n④ 梯度：只该流到 LoRA，基座冻结")
    model.train()
    lora = attach_lora(model, r=8, alpha=16)
    rep = trainable_report(lora)
    check(rep["ratio"] < 0.05, f"可训练参数只有 {rep['ratio_pct']}",
          f"{rep['trainable']:,} / {rep['total']:,}")

    loss = forward_loss(lora, batch)
    loss.backward()
    named = [(n, p) for n, p in lora.named_parameters() if "lora_" in n and p.grad is not None]
    g_nz = [n for n, p in named if p.grad.abs().sum() > 0]
    g_zero = [n for n, p in named if p.grad.abs().sum() == 0]
    check(len(g_nz) > 0, f"LoRA 参数拿到非零梯度（{len(g_nz)}/{len(named)} 个张量）")
    # ⚠️ 梯度是 0 的必然只剩 lora_A —— LoRA 的 B 初始化为 0，
    #    所以第一步 d(loss)/d(A) = Bᵀ·(下游梯度) = 0，**这是设计如此，不是 bug**。
    #    （也正是"第一步只有 B 在动"这个现象的来源）
    check(all("lora_A" in n for n in g_zero),
          f"梯度为 0 的只有 lora_A（{len(g_zero)} 个）—— B 初始为 0 的必然结果",
          "；".join(n.split(".")[-2] for n in g_zero[:3]) if g_zero else "无")
    check(len([p for n, p in lora.named_parameters()
               if "lora_" not in n and p.requires_grad]) == 0,
          "基座参数 requires_grad=False（没被训）")

    # 反向验证：漏掉 enable_input_require_grads 的症状
    lora.zero_grad()
    base_model = build_tiny_model()
    peft_no_gc = attach_lora(base_model, r=8, alpha=16)
    peft_no_gc.gradient_checkpointing_enable()          # ⚠️ 故意不调 enable_input_require_grads
    peft_no_gc.train()
    try:
        l2 = forward_loss(peft_no_gc, batch)
        l2.backward()
        g2 = [p.grad for n, p in peft_no_gc.named_parameters()
              if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0]
        symptom = "梯度全空（loss 不会降，但不报错）" if not g2 else "梯度正常"
        print(f"      · 只开 grad ckpt、不调 enable_input_require_grads → {symptom}")
    except Exception as ex:                                  # noqa: BLE001
        print(f"      · 只开 grad ckpt → 直接报错：{type(ex).__name__}")

    # ---------------------------------------------------------- ⑤ mask 生效
    print("\n⑤ mask 真的生效：非 loss 位置不出现在 target 里")
    keep = (targets != IGNORE_INDEX)
    check(int(keep.sum()) == int(batch.loss_mask[:, 1:].sum()),
          "target 里的有效位置数 == loss_mask 右移一位的和")
    # 抽查：每个有效 target 指向的那个 token，本身必须是 assistant token
    # ⚠️ 下标要 +1：targets[:, t] 对应的是 input_ids[:, t+1]（shift 过了）
    ii = keep.nonzero()
    ok = all(batch.loss_mask[b, t + 1].item() == 1 for b, t in ii.tolist())
    check(ok, "每个有效 target 指向的 token 都标了 loss_mask")

    # ---------------------------------------------------------- ⑥ 能训
    print("\n⑥ 能训：小 batch 过拟合，loss 应明显下降")
    # 起点 loss ≈ ln(vocab) = 11.93 —— 随机初始化的模型对每个 token 都均匀猜。
    # 刻意用**很短**的序列：2 层 64 维的迷你模型 + LoRA r=8 只有 1.6 万可训练参数，
    # 让它去背 3×512 个 token 是超出容量的（那样只能掉几个点，看不出信号）。
    # 背 2×128 个 token 才在它能力范围内 —— 验的是"梯度在把 loss 往下推"，
    # 不是"这个迷你模型有多强"。
    small = load_batch(tokenizer, tools, n=2, max_len=128)
    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=5e-3)
    lora.train()
    hist = []
    for step in range(150):
        opt.zero_grad()
        l = forward_loss(lora, small)
        l.backward()
        opt.step()
        hist.append(l.item())
    head, tail = hist[:20], hist[-20:]
    gain = st.mean(head) - st.mean(tail)
    noise = st.pstdev(tail)
    # ⭐ 判据不用"降了百分之几"（拍出来的阈值没有意义），用两条有依据的：
    #    ① 绝对收益 > 0.5 nat —— 0.5 nat 意味着困惑度降了 e^0.5 ≈ 1.65 倍，
    #       是肉眼可辨的"学到了"，不是浮点抖动；
    #    ② 收益 > 5 倍窗口内噪声 —— 否则"在降"可能只是这 20 步碰巧偏低。
    check(gain > max(0.5, 5 * noise),
          f"loss 从 {st.mean(head):.4f} 降到 {st.mean(tail):.4f}（降 {gain:.3f} nat）",
          f"尾窗噪声 σ={noise:.4f}，收益是它的 {gain/max(noise,1e-9):.1f} 倍"
          f"（起点 ≈ ln(151665) = {math.log(151665):.2f}）")
    # ⚠️ 不要求"最后一步是全程最低" —— 那等于要求 SGD 不抖。
    #    要验的是"确实推下去了"且"没炸"，两条分开验。
    check(all(x == x and x < math.inf for x in hist), "全程无 NaN / Inf")
    check(max(tail) < max(head), "后 20 步整体低于前 20 步（不是在原地打转）")

    print("\n" + "=" * 78)
    if _fails:
        print(f"❌ 自检未通过（{len(_fails)} 项）：")
        for f in _fails:
            print("   -", f)
        return 1
    print("✅ H5 训练步自测全绿")
    print("   · 分块 CE 与内置损失数值一致（省显存不是靠算错）")
    print("   · shift-by-one 有精确解析解兜底，偏一位必被抓")
    print("   · 梯度只进 LoRA；mask 真的遮挡；小 batch 能过拟合")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
