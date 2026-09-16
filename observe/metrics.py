# -*- coding: utf-8 -*-
"""
观测器 —— 训练时盯什么、怎么算。

为什么训练前就要写好：
    租卡一旦开始就在烧钱。**不知道该看什么，跑完 6 小时只能拿到一堆 loss 数字**，
    而本项目的最终交付物是「三条工具退化曲线」——没有观测器就出不来。

两类指标，**数据来源不同，必须分开**：

  类 A｜从轨迹就能算（零模型，纯 CPU）
      · 组内零方差率          —— 第一监控指标：>80% 说明题太难、没信号
      · reward 均值 / 标准差
      · agent 轮数、轨迹长度、工具调用数
      · ⭐ 工具退化谱 4 条     —— 本项目的核心创新

  类 B｜需要模型（训练时由训练脚本提供）
      · KL 散度（当前策略 vs 参考模型）
      · 策略熵
      → 本模块只留接口（extra 参数），不在 CPU 上算

⚠️ 一条已实测的限制（块 H1 发现）：
    「过调用」只能在【探针集 19 道】上测，「欠调用」只能在【训练集 31 道】上测。
    原因：探针题的 gold 里没有写动作，**只读动作不改数据库**，
         所以"查了"和"压根没查"在这些题上拿一样的分 —— 它们区分不了欠调用。
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# 写动作 = 会改数据库的那些。只读动作（get_*/search_*）不算。
WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
    "send_certificate",
}


# ---------------------------------------------------------------- 口径


@dataclass
class ToolUse:
    """从一条轨迹里拆出来的工具使用情况。"""

    calls: List[str] = field(default_factory=list)      # 按顺序的工具名
    sigs: List[str] = field(default_factory=list)       # 工具名+#参数（判"对同一对象重复"）

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_write(self) -> int:
        return sum(1 for c in self.calls if c in WRITE_TOOLS)

    @property
    def n_distinct(self) -> int:
        return len(set(self.calls))

    @property
    def jitter(self) -> float:
        """
        抖动指数：**对同一个对象反复做同一件事** 的程度。

            0.0 = 没有一次无效重复
            1.0 = 全都在重复（每一次都跟之前某一次完全相同）

        ⚠️ 这里必须看【参数】，不能只看工具名 —— 这是自测时发现的：
           真实 task[2] 的 gold 是 update_reservation_flights × 5，
           但那是对 **5 个不同预订** 操作，**完全正确**。
           只看工具名会把它判成 0.8 的高抖动 —— 定义就错了。

        所以签名取「工具名 + 参数 JSON」：
          5× update_flights(不同 reservation_id) → 5 个不同签名 → 抖动 0 ✅
          3× get_user_details(同一个 user_id)   → 1 个签名重复 2 次 → 抖动 2/3
        """
        if not self.sigs:
            return 0.0
        return (len(self.sigs) - len(set(self.sigs))) / len(self.sigs)


def tool_use_of(ep) -> ToolUse:
    """从 Episode 的 messages 里抽出工具调用序列（名字 + 参数签名）。"""
    calls, sigs = [], []
    for m in ep.messages:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                calls.append(name)
                sigs.append(f"{name}|{fn.get('arguments', '')}")
    return ToolUse(calls=calls, sigs=sigs)


# ---------------------------------------------------------------- 主入口


def compute(
    episodes: Iterable,
    task_meta: Dict[int, Dict[str, Any]],
    extra: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    算全套指标。

    Args:
        episodes:  List[rollout.Episode]
        task_meta: {task_id: {"class": "S"/"O", "n_write": int, ...}}
                   来自 data/task_split.json
        extra:     类 B 指标（KL / 熵），训练时传进来；CPU 上不传

    Returns:
        {
          "health":  {...},        # 训练健康度
          "degradation": {...},    # 工具退化谱
          "by_group": [...],       # 逐组明细（零方差率要按组算）
        }
    """
    eps = list(episodes)
    if not eps:
        return {"health": {}, "degradation": {}, "by_group": []}

    # ---- 按 task_id 分组（GRPO 的"组"= 同一道题的 n 条采样）
    groups: Dict[int, List[Any]] = defaultdict(list)
    for e in eps:
        groups[e.task_id].append(e)

    # ================================================================
    # 类 A · 训练健康度
    # ================================================================
    rewards = [e.reward for e in eps]

    # 组内零方差率 —— 第一监控指标
    zero_var_groups, group_rows = 0, []
    for tid, g in groups.items():
        rs = [e.reward for e in g]
        sd = st.pstdev(rs) if len(rs) > 1 else 0.0
        is_zero = sd < 1e-8
        zero_var_groups += int(is_zero)
        group_rows.append({
            "task_id": tid,
            "n": len(rs),
            "reward_mean": st.mean(rs),
            "reward_std": sd,
            "zero_variance": is_zero,
            "cls": task_meta.get(tid, {}).get("class", "?"),
        })

    health = {
        "n_episodes": len(eps),
        "n_groups": len(groups),
        # ⭐ 第一监控指标
        "zero_var_rate": zero_var_groups / len(groups) if groups else 0.0,
        # reward
        "reward_mean": st.mean(rewards),
        "reward_std": st.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "pass_rate": sum(1 for r in rewards if r >= 1.0 - 1e-9) / len(rewards),
        # 行为
        "turns_mean": st.mean([e.n_turns for e in eps]),
        "tool_calls_mean": st.mean([e.n_tool_calls for e in eps]),
        "msg_len_mean": st.mean([len(e.messages) for e in eps]),
        # 终止方式分布
        "terminated_by": dict(Counter(e.terminated_by for e in eps)),
    }

    # ================================================================
    # 类 A · ⭐ 工具退化谱（本项目的核心创新）
    # ================================================================

    # ---- 欠调用：在"该写"的任务上，一个写动作都没做
    #      只能在训练集（gold 有写动作）上测
    should_write = [e for e in eps
                    if task_meta.get(e.task_id, {}).get("n_write", 0) > 0]
    if should_write:
        n_under = sum(1 for e in should_write
                      if tool_use_of(e).n_write == 0)
        under_rate = n_under / len(should_write)
    else:
        n_under, under_rate = 0, 0.0

    # ---- 过调用：在"不需要写"的任务上，做了写动作
    #      只能在探针集（gold 里没有写动作）上测
    should_not_write = [e for e in eps
                        if task_meta.get(e.task_id, {}).get("n_write", 0) == 0]
    if should_not_write:
        n_over = sum(1 for e in should_not_write
                     if tool_use_of(e).n_write > 0)
        over_rate = n_over / len(should_not_write)
    else:
        n_over, over_rate = 0, 0.0

    # ---- 抖动：全部轨迹的平均
    jitters = [tool_use_of(e).jitter for e in eps]

    # ---- 空转：无写动作任务上白说了几轮
    idle = [e.n_turns for e in should_not_write] or [0]

    degradation = {
        # 分母一并报出来 —— 否则不知道这个比例是在多少道题上算的
        "under_call_rate": under_rate,
        "under_call_n": f"{n_under}/{len(should_write)}",
        "over_call_rate": over_rate,
        "over_call_n": f"{n_over}/{len(should_not_write)}",
        "jitter_mean": st.mean(jitters),
        "idle_turns_mean": st.mean(idle),
        # 工具使用总览
        "write_calls_mean": st.mean([tool_use_of(e).n_write for e in eps]),
        "tool_dist": dict(Counter(c for e in eps for c in tool_use_of(e).calls)),
    }

    # ================================================================
    # 类 B · 训练时才有（KL / 熵）
    # ================================================================
    if extra:
        health.update(extra)

    return {
        "health": health,
        "degradation": degradation,
        "by_group": sorted(group_rows, key=lambda r: r["task_id"]),
    }


# ---------------------------------------------------------------- 打印


def render(m: Dict[str, Any]) -> str:
    """把指标渲染成一段可读文本（训练日志里直接 print）。"""
    h, d = m.get("health", {}), m.get("degradation", {})
    if not h:
        return "(无数据)"

    zv = h["zero_var_rate"]
    flag = "  ⚠️ 题太难/没信号" if zv > 0.8 else ""
    L = [
        "── 训练健康度 " + "─" * 46,
        f"  轨迹 {h['n_episodes']} 条 ｜ 组 {h['n_groups']} 个",
        f"  ⭐ 组内零方差率   {zv:.3f}{flag}",
        f"  reward 均值/标准差 {h['reward_mean']:.3f} / {h['reward_std']:.3f}",
        f"  通过率            {h['pass_rate']:.3f}",
        f"  平均轮数 / 工具调用 {h['turns_mean']:.2f} / {h['tool_calls_mean']:.2f}",
        f"  终止方式          {h['terminated_by']}",
        "── 工具退化谱 " + "─" * 46,
        f"  欠调用率  {d['under_call_rate']:.3f}   ({d['under_call_n']})",
        f"  过调用率  {d['over_call_rate']:.3f}   ({d['over_call_n']})",
        f"  抖动指数  {d['jitter_mean']:.3f}",
        f"  空转轮数  {d['idle_turns_mean']:.2f}",
    ]
    if "kl" in h:
        L.insert(8, f"  KL / 熵            {h.get('kl', 0):.4f} / {h.get('entropy', 0):.4f}")
    return "\n".join(L)


__all__ = ["compute", "render", "tool_use_of", "ToolUse", "WRITE_TOOLS"]
