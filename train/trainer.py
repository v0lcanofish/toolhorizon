# -*- coding: utf-8 -*-
"""
训练器 —— SFT 预热（Stage 1）与 GRPO 主训练（Stage 2）共用一套骨架。

    python -m train.trainer selftest   # ⭐ 零 GPU / 零下载，本地 CPU 验全部逻辑
    python -m train.trainer sft        # Stage 1（租卡）
    python -m train.trainer grpo       # Stage 2（租卡）

为什么 SFT 和 GRPO 放一个文件：
    两者的**数据通路完全相同**（messages → token → loss mask → 分块前向），
    只差一个"每个 token 乘什么权重"。
    分成两个文件迟早会出现"改了一边的 mask 忘了另一边"。

━━━ 本文件要保证的三件事 ━━━

  ① **零方差 ⇒ 零梯度** —— 组内全对/全错时 advantage 恒 0，
     loss 恒 0、梯度恒 0。selftest 里有断言，这是项目核心命题的第一手证据。

  ② **SFT 和 GRPO 用同一条 token 化通路** —— 都走 train/tokenize.encode，
     都要传同一份 tools。任何一处漏传，训练和推理就变成两个任务。

  ③ **显存不靠运气** —— logits 从不出现在 [B, S, vocab] 这个形状上，
     全程只对"要算 loss 的那 9.4% 位置"分块计算（见 train/modeling.py）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from train.losses import compute_advantages, grpo_loss
from train.modeling import (
    Batch, attach_lora, collate, enable_grad_checkpointing, forward_logp,
    forward_loss, shift_for_lm, trainable_report,
)
from train.tokenize import encode, load_tokenizer, tool_schemas

_PROJECT = Path(__file__).resolve().parents[1]

IGNORE = -100


# ---------------------------------------------------------------- 配置


@dataclass
class TrainConfig:
    """超参默认值对齐 longhorizon 的 vanilla_grpo.yaml。"""

    # 数据
    max_seq_tokens: int = 8192
    # SFT
    sft_lr: float = 1e-4
    sft_epochs: int = 3
    # GRPO
    grpo_lr: float = 5e-6
    kl_coef: float = 0.01          # longhorizon 用 0.01 + low_var_kl
    clip_eps: float = 0.2
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    # 优化
    grad_accum: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42


# ---------------------------------------------------------------- 数据


def batches_from_records(
    records: Sequence[Dict[str, Any]],
    tokenizer,
    tools,
    max_seq_tokens: int = 8192,
    tokenizer_already_encoded: bool = True,
) -> Tuple[List[Batch], List[float]]:
    """
    rollout jsonl 记录 → [Batch] + 每组 advantage。

    记录必须带 `messages`（train/rollout_batch.save_rollouts 写的格式）。
    advantage 从记录里的 reward **重新算一遍**（而不是直接读文件里那个），
    这样"数据在磁盘上躺了多久"都不会影响梯度 —— 顺带也验证了落盘/读回没串。
    """
    by_group: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault(r["task_id"], []).append(r)

    batches, advs = [], []
    for tid, group in by_group.items():
        rewards = [r["reward"] for r in group]
        a, _sd, _zero = compute_advantages(rewards)
        for rec, ai in zip(group, a):
            enc = encode(rec["messages"], tokenizer, tools, max_length=max_seq_tokens)
            batches.append(collate([enc], pad_id=tokenizer.pad_token_id or 0))
            advs.append(float(ai))
    return batches, advs


# ---------------------------------------------------------------- 训练步


def sft_step(model, batch: Batch, chunk_size: int = 256):
    """Stage 1：普通交叉熵，权重全是 1。"""
    return forward_loss(model, batch, chunk_size=chunk_size)


def grpo_step(
    model,
    batch: Batch,
    advantage: float,
    ref_model=None,
    kl_coef: float = 0.0,
    chunk_size: int = 256,
):
    """
    Stage 2：GRPO。

    ⚠️ 本项目的 rollout 与训练是**两段式的**（分时复用显存），
       每批数据只更新一次 → 没有 old_logp → ratio ≡ 1。
       这里如实反映这一点，不假装在做 PPO 多轮裁剪。
    """
    loss_mask = batch.loss_mask[:, 1:].float()
    logp = forward_logp(model, batch, chunk_size=chunk_size)

    ref_logp = None
    if ref_model is not None and kl_coef > 0:
        with torch.no_grad():
            ref_logp = forward_logp(ref_model, batch, chunk_size=chunk_size)

    adv = torch.tensor([advantage], dtype=torch.float32, device=logp.device)
    return grpo_loss(logp, loss_mask, adv, old_logp=None,
                     ref_logp=ref_logp, kl_coef=kl_coef)


# ---------------------------------------------------------------- IO


def save_adapter(model, out_dir) -> str:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(p))
    return str(p)


def append_metrics(path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ================================================================ selftest
# ⭐ 零 GPU / 零下载：用随机初始化的小 Qwen2 把训练逻辑全部验一遍。
#    验的是**逻辑**，不是模型效果 —— 所以没必要为了"提前验一下"去拉 3GB 权重。


def selftest() -> int:
    from train.modeling import build_tiny_model

    fails = []

    def check(cond, label, detail=""):
        print(f"   {'[OK]' if cond else '[!!]'} {label}" + (f"  —— {detail}" if detail else ""))
        if not cond:
            fails.append(label)
        return cond

    print("=" * 78)
    print("块 H5 · 训练器自测（真跑 torch / 零 GPU / 零下载）")
    print("=" * 78)

    torch.manual_seed(0)
    tok = load_tokenizer()
    tools = tool_schemas()
    sft = _PROJECT / "data" / "sft_final.jsonl"
    rows = [json.loads(l) for l in sft.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---------------- ① 端到端：SFT 数据 → batch → loss → 反向
    print("\n① SFT 通路（真数据，截到 768 token）")
    encs = [encode(r["messages"], tok, tools, max_length=768) for r in rows[:4]]
    b = collate(encs, pad_id=tok.pad_token_id or 0)
    model = build_tiny_model()
    lora = attach_lora(model, r=8, alpha=16)
    rep = trainable_report(lora)
    check(rep["ratio"] < 0.05, f"LoRA 可训练参数 {rep['ratio_pct']}")

    lora.train()
    l = sft_step(lora, b)
    l.backward()
    g = [p.grad for n, p in lora.named_parameters() if "lora_" in n and p.grad is not None]
    check(len(g) > 0 and any(x.abs().sum() > 0 for x in g), "反向传播拿到非零梯度")
    check(torch.isfinite(l).item(), f"loss 有限（{l.item():.4f}）")
    check(b.n_loss_tokens > 0 and b.n_loss_tokens < b.input_ids.numel(),
          f"loss 只覆盖部分 token（{b.n_loss_tokens}/{b.input_ids.numel()}）")

    # ---------------- ② ⭐ 核心命题：零方差 ⇒ 零梯度
    print("\n② ⭐ 零方差 ⇒ 零梯度（本项目的核心命题，做成可断言）")
    for name, rewards in (("全对", [1.0] * 8), ("全错", [0.0] * 8)):
        advs, sd, zero = compute_advantages(rewards)
        check(zero and all(abs(a) < 1e-12 for a in advs),
              f"{name}组：std={sd:.4f} → advantage 全 0", f"{advs[:3]}")

    lora.zero_grad()
    loss0, stats = grpo_step(lora, b, advantage=0.0)
    loss0.backward()
    gnz = [p.grad for n, p in lora.named_parameters()
           if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0]
    check(abs(loss0.item()) < 1e-9, f"advantage=0 → loss ≡ 0（{loss0.item():.3e}）")
    check(len(gnz) == 0, f"advantage=0 → **梯度全为 0**（{len(gnz)} 个张量非零）",
          "这一步训练真的什么都没学到，而 loss 曲线上只是'平了一下'")

    # ---------------- ③ 非零 advantage 真的在推
    print("\n③ 非零 advantage：损失有闭式解，且梯度方向跟着 advantage 走")
    lora.zero_grad()
    lp, _ = grpo_step(lora, b, advantage=1.0)
    lp.backward()
    gnz = [p for n, p in lora.named_parameters()
           if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0]
    check(len(gnz) > 0, f"A=+1 时梯度非零（{len(gnz)} 个张量）")
    # ratio ≡ 1 时 L = −(Σ m·A)/Σm = −A —— **有闭式解**，直接对账，不靠感觉
    check(abs(lp.item() + 1.0) < 1e-6,
          f"L(A=+1) 精确等于 −A = −1（实测 {lp.item():.8f}）")
    neg, _ = grpo_step(lora, b, advantage=-1.0)
    check(abs(neg.item() - 1.0) < 1e-6,
          f"L(A=−1) 精确等于 +1（实测 {neg.item():.8f}）→ 梯度方向真的跟着 A 走")

    # ---------------- ④ 优化：advantage 为正时，训练会提高这批数据的 logprob
    print("\n④ 走几步之后：A>0 的样本 logprob 应当上升")
    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-3)
    lora.train()
    with torch.no_grad():
        before = forward_logp(lora, b)
    for _ in range(20):
        opt.zero_grad()
        ls, _ = grpo_step(lora, b, advantage=1.0)
        ls.backward()
        opt.step()
    with torch.no_grad():
        after = forward_logp(lora, b)
    m = b.loss_mask[:, 1:].float()
    d_before = float((before * m).sum() / m.sum())
    d_after = float((after * m).sum() / m.sum())
    check(d_after > d_before, f"平均 logprob 上升（{d_before:.4f} → {d_after:.4f}）")

    # ---------------- ⑤ 落盘/读回 → advantage 重算一致
    print("\n⑤ 从磁盘读回 rollout 记录，advantage 重算应当一致")
    from train.rollout_batch import load_rollouts, save_rollouts
    recs = [{"step": 0, "task_id": 0, "sample_idx": i, "reward": 1.0 if i < 3 else 0.0,
             "messages": rows[0]["messages"]} for i in range(6)]
    tmp = _PROJECT / "data" / "_tmp_trainer_selftest.jsonl"
    class _G:  # 只为复用 save_rollouts 的形状
        pass
    with tmp.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    back = load_rollouts(tmp)
    bs, advs = batches_from_records(back, tok, tools, max_seq_tokens=512)
    check(len(bs) == 6, f"读回 6 条 → 生成 6 个 batch（{len(bs)}）")
    expect, _sd, _z = compute_advantages([r["reward"] for r in back])
    check(all(abs(a - e) < 1e-9 for a, e in zip(advs, expect)),
          "重算的 advantage 与预期一致", f"{[round(a,3) for a in advs]}")
    tmp.unlink(missing_ok=True)

    print("\n" + "=" * 78)
    if fails:
        print(f"❌ 自测未通过（{len(fails)} 项）：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ 训练器自测全绿")
    print("   · SFT 通路：数据→token→mask→分块前向→反向，梯度非零")
    print("   · ⭐ 零方差 ⇒ advantage 全 0 ⇒ **梯度全 0**（核心命题，已断言）")
    print("   · A 取反损失取反；A>0 走几步后 logprob 上升")
    print("   · rollout jsonl 落盘/读回一致，advantage 重算一致")
    print("=" * 78)
    return 0


# ================================================================ CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ToolHorizon 训练器")
    ap.add_argument("mode", choices=["selftest", "sft", "grpo"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft-data", default=str(_PROJECT / "data" / "sft_final.jsonl"))
    ap.add_argument("--rollouts", default=str(_PROJECT / "data" / "rollouts" / "step_latest.jsonl"))
    ap.add_argument("--adapter-in", default="",
                    help="从已有 adapter 接着训（GRPO 每轮要带上上一轮的）；空 = 新建 LoRA")
    ap.add_argument("--out", default=str(_PROJECT / "models" / "adapter"))
    ap.add_argument("--metrics", default=str(_PROJECT / "reports" / "train_metrics.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if args.mode == "selftest":
        return selftest()

    # 下面两条只在租卡当天走 —— 本地没有 cuda / 没下权重
    from train.modeling import load_model

    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    # 🔴 tokenizer 必须和模型同源 —— 否则 token id 对不上，模型看到的是乱码，
    #    但 loss 照样会降（它在拟合乱码），要到评测才发现。详见 train/tokenize.py
    tok = load_tokenizer(model_path=args.model)
    tools = tool_schemas()
    print(f"[trainer] 加载 {args.model} …")
    model = load_model(args.model, dtype="bfloat16")
    if args.adapter_in:
        # ⭐ 接着上一轮的 adapter 训 —— GRPO 是**不断迭代**的，
        #    每轮采样的策略必须是上一轮训完的那个，否则 advantage 就对不上了。
        from peft import PeftModel
        lora = PeftModel.from_pretrained(model, args.adapter_in, is_trainable=True)
        print(f"[trainer] 从 adapter 接着训：{args.adapter_in}")
    else:
        lora = attach_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)
    lora = lora.to(args.device)

    # 🔴 **必须开梯度检查点** —— 2026-09-17 在 4090D 上实测踩到：
    #    没开的话，一条 8192 token 的轨迹前向就把 24 GB 吃满（PyTorch 分配 22.5 GB）→ OOM。
    #    原理：不检查点时要保存**每一层**的中间激活（28 层 × 8192 token），
    #          开了之后只存层边界，反向时重算 —— 激活从 ~20 GB 降到 ~2-3 GB。
    #    代价是前向多跑一遍，换取装得下。
    enable_grad_checkpointing(lora)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[trainer] LoRA 可训练参数 {trainable_report(lora)['ratio_pct']}（已开梯度检查点）")
    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad],
                            lr=cfg.sft_lr if args.mode == "sft" else cfg.grpo_lr)
    lora.train()

    if args.mode == "sft":
        rows = [json.loads(l) for l in Path(args.sft_data).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        encs = [encode(r["messages"], tok, tools, max_length=cfg.max_seq_tokens) for r in rows]
        print(f"[trainer] SFT {len(encs)} 条轨迹")
        step = 0
        for ep in range(cfg.sft_epochs):
            for i in range(0, len(encs), cfg.grad_accum):
                t0 = time.time()
                opt.zero_grad()
                total = 0.0
                for e in encs[i:i + cfg.grad_accum]:
                    b = collate([e], pad_id=tok.pad_token_id or 0).to(args.device)
                    loss = sft_step(lora, b) / cfg.grad_accum
                    loss.backward()
                    total += loss.item()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in lora.parameters() if p.requires_grad], cfg.max_grad_norm)
                opt.step()
                step += 1
                append_metrics(args.metrics, {"stage": "sft", "step": step, "epoch": ep,
                                              "loss": total, "sec": time.time() - t0})
                if step % 5 == 0 or step == 1:
                    print(f"[sft] ep{ep} step{step} loss={total:.4f}")
        print(f"[trainer] adapter → {save_adapter(lora, args.out)}")
        return 0

    # ---- GRPO
    from train.rollout_batch import load_rollouts
    recs = load_rollouts(args.rollouts)
    bs, advs = batches_from_records(recs, tok, tools, cfg.max_seq_tokens)
    print(f"[trainer] GRPO {len(bs)} 条轨迹，{len(set(r['task_id'] for r in recs))} 组")
    n_zero = sum(1 for a in advs if abs(a) < 1e-12)
    print(f"[trainer] ⚠️ advantage 为 0 的轨迹：{n_zero}/{len(advs)}"
          f"（这些轨迹**不产生任何梯度**）")
    for step, (b, a) in enumerate(zip(bs, advs), 1):
        b = b.to(args.device)
        opt.zero_grad()
        loss, stats = grpo_step(lora, b, advantage=a, kl_coef=0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in lora.parameters() if p.requires_grad], cfg.max_grad_norm)
        opt.step()
        append_metrics(args.metrics, {"stage": "grpo", "step": step, "adv": a, **stats})
    print(f"[trainer] adapter → {save_adapter(lora, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
