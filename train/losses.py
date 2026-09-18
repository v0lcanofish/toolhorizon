# -*- coding: utf-8 -*-
"""
GRPO 的目标函数 —— **本项目的核心命题就写在这个文件里**。

    A_i = (r_i − mean(group)) / (std(group) + eps)
    L   = − (1/Σm) Σ_t  m_t · A_i · min(ρ_t·A_i , clip(ρ_t)·A_i)  +  β·KL_t
    ρ_t = exp(logp_t − logp_t^old)

三件事必须说清楚，否则这个文件会被误读：

① **零方差 ⇒ 零梯度，是数学恒等式，不是"训练不稳定"**
   组内 reward 全相同（全对 / 全错）时 std=0 → A 全是 0 → L≡0 → 梯度全 0。
   而 loss 曲线上只是"这一步没动"，**看不出来**。
   → 所以观测器的第一指标是「组内零方差率」，不是 loss。
   → `zero_variance_report()` 把这个恒等式做成可断言的。

② **advantage 是按「整条轨迹」广播到每个 token 上的**
   本项目的奖励是**结果奖励**（数据库改没改对），不是过程奖励。
   所以一条轨迹里每个 token 拿同一个 A_i。
   （过程奖励 = 真 PRM。**2026-09-18 更正**：这行原先写的是
     "longhorizon 已证明 rule-based PRM-Lite 反而更差，本项目不做" —— **那个判读是错的**。
     参考实现 README 的真实数字：vanilla 0.125 ｜ PRM-Lite 单独 **0.140** ｜
     LATA 单独 0.185 ｜ **两者联合 0.240**。PRM-Lite 单独用**比 vanilla 好**，
     它的毛病是"单独用收益小"，不是"更差"。
     参考实现的原话是 **"Neither works well in isolation"** —— 价值在**信号传递**：
     PRM 产生局部信号，LATA 的 √L 提供传输通道。
     本项目目前两者都没做；LATA 已在下方 `grpo_loss(length_norm=)` 留了开关。）

③ **KL 用 low_var_kl**（`exp(Δ) − Δ − 1`），和 longhorizon 的配置对齐（coef 0.01）。
   它是无偏的且恒 ≥ 0，比 `Δ` 那个形式数值稳。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["compute_advantages", "grpo_loss", "kl_low_var", "zero_variance_report"]


# ---------------------------------------------------------------- advantage


def compute_advantages(rewards, eps: float = 1e-8):
    """
    组内归一化优势。返回 (advantages, std, zero_variance)。

    ⚠️ 和 train/rollout_batch.py 里那份是同一个公式的两处实现 ——
       自测脚本会断言两者一致，防止哪天改了一边忘了另一边。
    """
    n = len(rewards)
    if n == 0:
        return [], 0.0, True
    m = sum(rewards) / n
    var = sum((r - m) ** 2 for r in rewards) / n      # 总体标准差（pstdev）
    sd = math.sqrt(var)
    if sd < eps:
        return [0.0] * n, sd, True
    return [(r - m) / (sd + eps) for r in rewards], sd, False


def zero_variance_report(advantages) -> dict:
    """
    ⭐ 把核心命题做成可断言的量。
    """
    n = len(advantages)
    return {
        "n": n,
        "all_zero": all(abs(a) < 1e-12 for a in advantages) if n else True,
        "max_abs": max((abs(a) for a in advantages), default=0.0),
    }


# ---------------------------------------------------------------- KL


def kl_low_var(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """
    low_var_kl：`exp(Δ) − Δ − 1`，Δ = ref_logp − logp。

    恒 ≥ 0（因为 e^x ≥ x+1），且 Δ=0 时为 0 —— 所以它天然是"偏离参考多少"的度量。
    比直接用 Δ 稳：Δ 是线性的，会正负抵消，跑着跑着看不出偏离。
    """
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0


# ---------------------------------------------------------------- 主损失


def grpo_loss(
    logp: torch.Tensor,              # [B, S-1] 当前策略的逐 token logprob（mask 外为 0）
    loss_mask: torch.Tensor,         # [B, S-1] 1 = 算 loss
    advantages: torch.Tensor,        # [B]      每条轨迹一个（结果奖励）
    old_logp: Optional[torch.Tensor] = None,
    ref_logp: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    length_norm: str = "mean",        # "mean" = ÷L（原版）｜ "lata" = ÷√L
) -> tuple[torch.Tensor, dict]:
    """
    GRPO 损失。

    ⚠️ 当 `old_logp is None` 时 ρ ≡ 1，clip 恒等 —— 那就是**纯 REINFORCE + 基线**，
       也就是"每批数据只训一遍"的情形。本项目的采样/训练是分时复用的两段式，
       默认就是这种（每批 rollout 只更新一次），所以 clip 默认不起作用。
       要开多次内层 epoch 才需要传 old_logp。
    """
    m = loss_mask.float()
    adv = advantages.unsqueeze(1)                     # [B,1] → 广播到该轨迹的每个 token

    if old_logp is None:
        # ⚠️ 不能写 torch.ones_like(logp) —— 那样损失和模型**脱钩**，
        #    backward() 会直接报 "does not require grad"。
        #    写 exp(logp − logp.detach()) 数值恒等于 1，但计算图保留下来，
        #    于是"零 advantage → 梯度精确为 0"才是**真的被验证**，
        #    而不是"压根没建图"。
        ratio = torch.exp(logp - logp.detach())
    else:
        ratio = torch.exp(torch.clamp(logp - old_logp, -20, 20))

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv

    # ⭐ 长度归一化 —— **LATA 就落在这一行**。
    n_tok = m.sum().clamp(min=1.0)
    #   "mean" ÷L（原版）：长轨迹的每 token 梯度按 1/L 衰减，
    #     参考实现的原话是这等于在训练模型"trade quantity for quality"——
    #     学成"说短话 + 频繁试错"，第 150 步后崩。**这正是我们担心的事。**
    #   "lata" ÷√L：长度涨 4 倍时每 token 梯度只减半（而非四分之一），
    #     保住"多轮推理"的边际激励。参考实现实测 vanilla 0.125 → LATA 单独 0.185。
    denom = n_tok.sqrt() if length_norm == "lata" else n_tok
    pg = -(torch.min(surr1, surr2) * m).sum() / denom

    # ⚠️ 记统计数字必须 no_grad —— 否则 PyTorch 会警告
    #    "Converting a tensor with requires_grad=True to a scalar"
    #    （2026-09-17 在卡上刷屏，不影响结果但很难看）
    with torch.no_grad():
        stats = {
            "ratio_mean": float((ratio * m).sum() / m.sum().clamp(min=1.0)),
            "n_tokens": int(m.sum().item()),
            "adv_abs_mean": float(advantages.abs().mean()) if advantages.numel() else 0.0,
        }

    loss = pg
    if ref_logp is not None and kl_coef > 0:
        kl = (kl_low_var(logp, ref_logp) * m).sum() / m.sum().clamp(min=1.0)
        loss = loss + kl_coef * kl
        stats["kl"] = float(kl)

    stats["pg"] = float(pg)
    return loss, stats
