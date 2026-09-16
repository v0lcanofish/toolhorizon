# -*- coding: utf-8 -*-
"""
规则式用户模拟器 —— 零 LLM。

为什么不用 τ-bench 自带的 LLMUserSimulationEnv：
  longhorizon 自己的诊断文档写了 "user simulator 是 eval 噪声的主要来源"。
  换成规则模拟器后，**reward 方差 100% 来自 policy**，归因链干净得多。
  这是研究增量，不只是工程妥协。

当前阶段（Stage 0 第一步 · 占位版）：
  只需要满足 Env 的构造契约（Env.__init__ 里就会调 load_user）。
  实测发现 50 道题的 gold 动作里 **respond 出现 0 次**（见 scripts/probe_tau_bench.py），
  所以跑 gold 重放 oracle 时 step() 根本不会被调到。
  真正的 slot 解析版（按 instruction 抽出偏好，关键词匹配回答）在下一步做。

跑法：由 env/tau_env.py 内部实例化，不需要单独跑。
"""

import re
from typing import Optional

from env.bootstrap import TAU_BENCH  # noqa: F401  必须先于 tau_bench 导入

from tau_bench.envs.user import BaseUserSimulationEnv  # noqa: E402

USER_ID_RE = re.compile(r"user id is ([a-z_0-9]+)", re.IGNORECASE)


class RuleBasedUserSim(BaseUserSimulationEnv):
    """
    规则用户模拟器。

    接口契约（来自 BaseUserSimulationEnv）：
        reset(instruction) -> str   开场白
        step(content)      -> str   对 agent 这句话的回应
        get_total_cost()   -> float 永远 0（不花 token）

    step() 返回的字符串里如果含 "###STOP###"，Env 会判定 episode 结束
    （见 tau_bench/envs/base.py:100）。
    """

    def __init__(self, max_turns: int = 8) -> None:
        self.instruction: str = ""
        self.user_id: Optional[str] = None
        self.turn: int = 0
        self.max_turns = max_turns

    # ------------------------------------------------------------ 内部

    def _extract_user_id(self, instruction: str) -> Optional[str]:
        m = USER_ID_RE.search(instruction)
        return m.group(1) if m else None

    # ------------------------------------------------------------ 契约

    def reset(self, instruction: Optional[str] = None) -> str:
        self.instruction = instruction or ""
        self.user_id = self._extract_user_id(self.instruction)
        self.turn = 0

        if self.user_id:
            return f"Hi, I need some help with my flights. My user id is {self.user_id}."
        return "Hi, I need some help with my flights."

    def step(self, content: str) -> str:
        self.turn += 1
        if self.turn >= self.max_turns:
            return "Thanks, that's all I needed. ###STOP###"

        # TODO(Stage 0 第二步)：slot 解析版。
        #   做法：用一次性 DeepSeek 调用把 instruction 解析成 slot dict 缓存成 JSON，
        #   运行时按关键词匹配返回对应 slot；匹配不到返回固定兜底句。
        #   降级链：规则模拟器 → 回放 historical_trajectories 的 user 轮 → easy 档任务
        return "Okay, please go ahead."

    def get_total_cost(self) -> float:
        return 0.0


__all__ = ["RuleBasedUserSim"]
