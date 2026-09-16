# -*- coding: utf-8 -*-
"""
多轮 episode 循环 —— 训练和评测共用的采样入口。

env 只提供了 step()，**循环要自己写**。这个文件就是那个循环。

一次 episode 长这样：
    system（政策 + 工具定义）
      ↓
    user（任务指令）
      ↓
    ┌─→ assistant 说了什么 / 调了什么工具
    │      ↓
    │   环境返回：工具结果（role=tool） 或 用户回话（role=user）
    └──┘  直到 done
      ↓
    reward = 数据库有没有改对（+ outputs 检查）

⚠️ 消息格式必须和 SFT 数据（data/sft_final.jsonl）**逐字段一致**，
   否则「SFT 预热 → GRPO」这一步会断掉。

跑法：被训练/评测脚本 import，不单独跑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from tau_bench.types import Action, RESPOND_ACTION_NAME

from env.tau_env import build_env_custom, reward_of

_PROMPT_FILE = Path(__file__).resolve().parents[1] / "data" / "system_prompt.txt"


def load_system_prompt() -> str:
    """读系统提示（取自真实成功轨迹，含 τ-bench 的航空政策）。"""
    if not _PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"缺 {_PROMPT_FILE}\n"
            "它应该由 scripts/ 下的脚本从 historical_trajectories 里提取生成。"
        )
    return _PROMPT_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 数据结构


@dataclass
class Episode:
    """一次采样出来的轨迹。"""

    task_id: int
    messages: List[Dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    n_turns: int = 0                  # agent 说了几轮
    n_tool_calls: int = 0
    done: bool = False                # 是否正常终止（False = 撞上 max_turns）
    terminated_by: str = ""           # user_stop | terminate_tool | max_turns

    @property
    def assistant_turns(self) -> List[Dict[str, Any]]:
        """只有这些是要算 loss 的。"""
        return [m for m in self.messages if m["role"] == "assistant"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reward": self.reward,
            "n_turns": self.n_turns,
            "n_tool_calls": self.n_tool_calls,
            "done": self.done,
            "terminated_by": self.terminated_by,
            "messages": self.messages,
        }


# ---------------------------------------------------------------- 策略接口

# 策略就是一个函数：给当前消息列表，返回下一个动作。
# 真模型（vLLM）、假模型（照本宣科）、SFT 模型都实现成这个签名。
Policy = Callable[[List[Dict[str, Any]]], Action]


# ---------------------------------------------------------------- 循环


def run_episode(
    task,
    policy: Policy,
    task_id: int = -1,
    max_turns: int = 40,
    system_prompt: str = "",
) -> Episode:
    """
    跑一次 episode。

    Args:
        task:          tau_bench.types.Task
        policy:        给消息列表 → 下一个 Action
        max_turns:     硬上限（防弱策略无限循环把显存撑爆）
        system_prompt: 系统提示；空则用任务自带的（τ-bench 的 wiki 政策）
    """
    env = build_env_custom(task)
    env.reset(task_index=0)

    # ---- 系统提示
    # 取自真实成功轨迹的第 0 条（τ-bench 的航空政策，6155 字符）。
    # ⚠️ 必须和 SFT 数据里用的是【同一份】——否则格式对不上，训练会断。
    if not system_prompt:
        system_prompt = load_system_prompt()

    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.instruction},
    ]

    ep = Episode(task_id=task_id, messages=msgs)
    resp = None
    call_id = 0

    for _ in range(max_turns):
        action = policy(msgs)
        ep.n_turns += 1

        resp = env.step(action)

        # ---- agent 这一轮是"调工具"还是"说话"？
        if action.name == RESPOND_ACTION_NAME:
            msgs.append({"role": "assistant",
                         "content": action.kwargs.get("content", "")})
            # 用户回话 → role=user（**不进 loss**）
            msgs.append({"role": "user", "content": resp.observation})
        else:
            call_id += 1
            ep.n_tool_calls += 1
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{task_id}_{call_id}",
                    "type": "function",
                    "function": {"name": action.name,
                                 "arguments": _dump_args(action.kwargs)},
                }],
            })
            msgs.append({"role": "tool", "tool_call_id": f"call_{task_id}_{call_id}",
                         "name": action.name,
                         # 工具结果可能很长（航班列表 JSON）——原样保留，
                         # 裁剪是后面单独的一个消融轴（见 README 的 D-7）
                         "content": resp.observation})

        if resp.done:
            ep.done = True
            ep.terminated_by = (
                "terminate_tool" if action.name in getattr(env, "terminate_tools", [])
                else "user_stop"
            )
            break

    if not ep.done:
        ep.terminated_by = "max_turns"

    # ⚠️ 必须用 reward_of：calculate_reward() 不幂等，见 tau_env.reward_of 的注释
    ep.reward = reward_of(env, resp)
    return ep


def _dump_args(kwargs: Dict[str, Any]) -> str:
    import json
    return json.dumps(kwargs, ensure_ascii=False)


__all__ = ["run_episode", "Episode", "Policy"]
