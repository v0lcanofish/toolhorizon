# -*- coding: utf-8 -*-
"""
模型与损失 —— 训练脚本里唯一容易「悄悄错」的地方都在这。

本文件解决三个显存刺客（README 列过，这里是它们的解药）：

  ① **logits 张量**：S=8192 × vocab 151,665 × fp32 = **4.97 GB 单个张量**
     → `chunked_ce_loss()`：只对**真正要算 loss 的位置**（实测只占 9.4%）算 logits，
       再按 chunk 切成 256 个位置一批。峰值从 4.97 GB 降到约 0.3 GB。

  ② **`logits_to_keep` 缺失**：轨迹里只有约 9.4% 的 token 是 assistant 产出的。
     不做 mask-only 就是 10 倍浪费。
     → `collate()` 把不参与 loss 的位置全设成 -100，损失函数**跳过它们**。

  ③ **PEFT + 梯度检查点的 `enable_input_require_grads()`**：
     开了 gradient checkpointing 之后，如果忘了这个，LoRA 的梯度会全是 None，
     而训练**不报错**，只是 loss 不降。`enable_grad_checkpointing()` 里一起做了。

⚠️ 还有一个「不报错但全错」的坑，单独说：**shift-by-one**
   因果语言模型是「用位置 t 的输出预测 token t+1」。
   所以 loss 的 mask 必须是 `loss_mask[:, 1:]`，**不是 `loss_mask`**。
   写错的话训练照跑、loss 照降，只是模型学的是"复读上一个 token"。
   → `scripts/eval_train_logic.py` 里有一条断言专门盯这个。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

# Qwen2 系列的 LoRA 目标模块（注意力 + MLP 全打）
QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------- 模型


def build_tiny_model(vocab_size: int = 151665, hidden_size: int = 64,
                     num_layers: int = 2, num_heads: int = 4, num_kv_heads: int = 2):
    """
    造一个**随机初始化的小 Qwen2**，用来在 CPU 上验训练循环 —— 不需要下载任何权重。

    ⚠️ 这不是"凑合"，是刻意的：验证的是**训练逻辑**
       （mask 对不对、梯度流不流、shift 有没有偏），
       这些跟权重是什么、模型多大完全无关。
       而随权重一起下载来的 3GB 才是真正浪费租卡前的时间。
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg)


def load_model(model_path: str, dtype: str = "bfloat16", device_map: Optional[str] = None):
    """加载真实模型（租卡当天用）。"""
    from transformers import AutoModelForCausalLM

    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32,
                   "float16": torch.float16}[dtype]
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device_map,
    )


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.0,
                target_modules: Optional[Sequence[str]] = None):
    """
    挂 LoRA。**只训 adapter，基座冻结** —— 显存账的关键（README：84 倍）。

    ⭐ 顺带解决一个概念问题：「参考模型」不用另存一份 ——
       把 adapter 关掉（`model.disable_adapter()`）就是基座本身，
       这正是 GRPO 里算 KL / 参考 logprob 要的东西。
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules or QWEN_LORA_TARGETS),
        task_type="CAUSAL_LM", bias="none",
    )
    return get_peft_model(model, cfg)


def enable_grad_checkpointing(model) -> None:
    """
    开梯度检查点 —— ⚠️ **必须同时 `enable_input_require_grads()`**。

    漏了它的症状：LoRA 梯度全 None，loss 一动不动，但**不报错**。
    （这就是 README 里"报错后手贱关 grad ckpt → 激活 ×3–5"那条的另一面）
    """
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        model.get_input_embeddings().register_forward_hook(
            lambda _m, _i, out: out.requires_grad_(True)
        )


def backbone_and_head(model) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    拆出「主干（不含 lm_head）」和「lm_head」。

    为什么不用 `model(...)` 直接拿 loss：
        那样 transformers 会**先把 [B,S,vocab] 的 logits 全算出来**再挑要用的位置，
        等于为了 9.4% 的有效 token 付 100% 的显存。我们要自己控节奏。

    兼容 PeftModel（外面套了一层）和裸 CausalLM。
    """
    base = model
    if hasattr(base, "get_base_model"):        # PeftModel
        base = base.get_base_model()
    backbone = getattr(base, "model", None)    # Qwen2ForCausalLM.model
    if backbone is None:
        raise TypeError(f"认不出主干：{type(base).__name__}")
    head = model.get_output_embeddings()
    if head is None:
        raise TypeError("拿不到 lm_head")
    return backbone, head


def trainable_report(model) -> Dict[str, Any]:
    """看一眼到底在训什么 —— 参数对不上是"花了钱没训到东西"的头号原因。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / max(1, total),
        "ratio_pct": f"{trainable / max(1, total):.4%}",
    }


# ---------------------------------------------------------------- 拼批


@dataclass
class Batch:
    """一个 micro-batch 的张量。"""

    input_ids: torch.Tensor        # [B, S]
    attention_mask: torch.Tensor   # [B, S]
    loss_mask: torch.Tensor        # [B, S]  1 = 参与 loss

    @property
    def n_loss_tokens(self) -> int:
        return int(self.loss_mask.sum().item())

    def to(self, device) -> "Batch":
        return Batch(self.input_ids.to(device),
                     self.attention_mask.to(device),
                     self.loss_mask.to(device))


def collate(encoded: List[Any], pad_id: int, device=None) -> Batch:
    """
    [Encoded] → 一个 padded batch（右填充）。

    ⚠️ 右填充 + attention_mask 是对的，但**不要**顺手把 pad 位置和 loss 位置搞混：
        pad 的 loss_mask 必须是 0，否则模型会去学"预测 padding"。
    """
    if not encoded:
        raise ValueError("空的 batch")
    max_len = max(len(e.input_ids) for e in encoded)
    B = len(encoded)

    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, e in enumerate(encoded):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        attn[i, :n] = 1
        mask[i, :n] = torch.tensor(e.loss_mask, dtype=torch.long)

    b = Batch(ids, attn, mask)
    return b.to(device) if device is not None else b


def shift_for_lm(batch: Batch):
    """
    ⭐ **shift-by-one**：把「第 t 个位置的输出」和「第 t+1 个 token」对上。

    因果 LM 的定义就是「看着前 t 个，预测第 t+1 个」。
    所以：
        输入           input_ids[:, :-1]     （位置 0 .. S-2）
        监督目标       input_ids[:, 1:]      （token 1 .. S-1）
        参与 loss 的    loss_mask[:, 1:]     ← ⚠️ 也是错的常见来源

    写反了这一位的后果：训练照常、loss 照降，只是模型学会"复读上一个 token"。

    Returns:
        (inputs, targets) —— targets 里不参与 loss 的位置已经设成 -100
    """
    inputs = batch.input_ids[:, :-1]
    attn = batch.attention_mask[:, :-1]
    targets = batch.input_ids[:, 1:].clone()
    keep = batch.loss_mask[:, 1:] == 1
    targets[~keep] = IGNORE_INDEX
    return inputs, attn, targets


# ---------------------------------------------------------------- 损失


def chunked_ce_loss(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head_weight: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    分块交叉熵 —— 只对**有效位置**算 logits，再按 chunk 分批。

    省在哪：
        · 跳过 -100 的位置 → 实测只算 9.4% 的 token（近 10 倍省）
        · 每 chunk 的 logits 只有 [chunk, vocab]，
          256 × 151,665 × 4 B ≈ 155 MB，而不是整条的 4.97 GB

    ⚠️ 数值口径：logits 强制转 fp32 再算 ——
       和 transformers 内置损失一致（它也是 `.float()` 之后调 CrossEntropyLoss）。
       bf16 直接算 CE 会因为 log-sum-exp 精度不够而抖动。
    """
    flat_h = hidden.reshape(-1, hidden.size(-1))
    flat_t = targets.reshape(-1)
    idx = (flat_t != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        # 全被 mask 掉时不能返回 0（那样它跟计算图没关系，backward 会报错）
        return flat_h.sum() * 0.0

    total = None
    for i in range(0, idx.numel(), chunk_size):
        sel = idx[i:i + chunk_size]
        logits = flat_h[sel] @ lm_head_weight.t()          # [n, V]
        l = F.cross_entropy(logits.float(), flat_t[sel], reduction="sum")
        total = l if total is None else total + l
    return total / idx.numel()


def chunked_token_logp(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head_weight: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    逐 token 的 log p(target | 前缀)，**同样只对有效位置算**。

    和 chunked_ce_loss 是一套东西的两个出口：
        CE   = −mean(logp)              ← SFT 用
        logp = 每个 token 各自的值       ← GRPO 算 ratio / KL 用

    Returns:
        [B, S] 的张量；**被 mask 掉的位置是 0**（不是 -inf，方便直接乘 advantage）
    """
    B, S = targets.shape
    flat_h = hidden.reshape(-1, hidden.size(-1))
    flat_t = targets.reshape(-1)
    out = torch.zeros(flat_t.numel(), dtype=torch.float32, device=flat_h.device)
    idx = (flat_t != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return out.view(B, S)
    for i in range(0, idx.numel(), chunk_size):
        sel = idx[i:i + chunk_size]
        logits = (flat_h[sel] @ lm_head_weight.t()).float()
        lp = F.log_softmax(logits, dim=-1)
        out[sel] = lp.gather(1, flat_t[sel].unsqueeze(1)).squeeze(1)
    return out.view(B, S)


def forward_logp(model, batch: Batch, chunk_size: int = 256) -> torch.Tensor:
    """一趟前向，返回 [B, S-1] 的逐 token logprob（已 shift）。"""
    backbone, head = backbone_and_head(model)
    inputs, attn, targets = shift_for_lm(batch)
    out = backbone(input_ids=inputs, attention_mask=attn, use_cache=False)
    return chunked_token_logp(out.last_hidden_state, targets, head.weight, chunk_size)


def forward_loss(model, batch: Batch, chunk_size: int = 256) -> torch.Tensor:
    """一趟前向 + 损失。**训练循环调这个**。"""
    backbone, head = backbone_and_head(model)
    inputs, attn, targets = shift_for_lm(batch)
    out = backbone(input_ids=inputs, attention_mask=attn, use_cache=False)
    return chunked_ce_loss(out.last_hidden_state, targets,
                           head.weight, chunk_size=chunk_size)


def hf_reference_loss(model, batch: Batch) -> torch.Tensor:
    """
    用 transformers 自己的实现算同一个 batch 的 loss —— **只给对账用**。

    存在的意义：证明 chunked_ce_loss 不是"看起来对"。
    两者在同一个 batch 上必须数值相等（浮点误差 1e-4 以内）。
    ⚠️ 它在 S=8192 时会造出 4.97 GB 的张量，**只在自测的小 batch 上用**。
    """
    inputs, attn, targets = shift_for_lm(batch)
    out = model(input_ids=inputs, attention_mask=attn, use_cache=False)
    logits = out.logits.float()
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


__all__ = [
    "build_tiny_model", "load_model", "attach_lora", "enable_grad_checkpointing",
    "backbone_and_head", "trainable_report",
    "Batch", "collate", "shift_for_lm",
    "chunked_ce_loss", "chunked_token_logp", "forward_logp",
    "forward_loss", "hf_reference_loss",
    "IGNORE_INDEX", "QWEN_LORA_TARGETS",
]
