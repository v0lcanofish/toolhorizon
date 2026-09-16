# -*- coding: utf-8 -*-
"""
τ-bench airline 环境工厂。

做了三件事，都是对 vendored 仓库的**外部包装**（参考仓库保持只读）：
  1. 猴补丁 Env.__init__ 里的 load_user → 换成规则式用户模拟器
     （base.py:74 写死了 self.user = load_user(...)，没给出注入口）
  2. 把 tools 变成可选子集
  3. 统一 task_index 的传入方式（Env.reset 默认 random.randint(0, len) 有 off-by-one，
     见 base.py:70 —— 会 IndexError，所以永远显式传 task_index）

跑法：被 scripts/ 下的脚本 import，不单独跑。
"""

import copy
from typing import Any, Dict, List, Optional, Sequence, Type

from env.bootstrap import TAU_BENCH  # noqa: F401  必须先于 tau_bench 导入
from env.user_sim import RuleBasedUserSim

import tau_bench.envs.base as tb_base                          # noqa: E402
from tau_bench.envs.base import Env                            # noqa: E402
from tau_bench.envs.airline.data import load_data              # noqa: E402
from tau_bench.envs.airline.rules import RULES                 # noqa: E402
from tau_bench.envs.airline.wiki import WIKI                   # noqa: E402
from tau_bench.envs.airline.tools import ALL_TOOLS             # noqa: E402
from tau_bench.envs.airline.tasks_test import TASKS            # noqa: E402
from tau_bench.envs.tool import Tool                           # noqa: E402


# ---------------------------------------------------------------- 工具子集

TOOLS_BY_NAME = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
ALL_TOOL_NAMES = set(TOOLS_BY_NAME)

# 实测（scripts/probe_tau_bench.py）：出现在 gold 动作里的工具。
# ⚠️ 这些**必须保留** —— Env.calculate_reward() 靠重放 gold 动作算 gt_data_hash，
#    工具不在 tools_map 里就重放失败，reward 恒为 0。
GOLD_TOOLS = {
    "book_reservation",
    "calculate",
    "cancel_reservation",
    "get_reservation_details",
    "get_user_details",
    "search_direct_flight",
    "send_certificate",
    "transfer_to_human_agents",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}

# 实测在 gold 里出现 0 次的工具。
#   think              → 返回空串，纯粹占一轮
#   list_all_airports  → wiki 里已有机场表，信息冗余
#   search_onestop_flight → ⚠️ 虽然在 gold 里 0 次，但 agent **需要**它
#                           （task[0] 的 instruction 明说 "one stopover also fine"），
#                           gold 不记搜索类动作 ≠ 搜索不需要发生。默认保留。
AGENT_ONLY_TOOLS = {"think", "list_all_airports", "search_onestop_flight"}

# 无争议可砍（0 次 gold + 纯冗余）
SAFE_TO_CUT = {"think", "list_all_airports"}


def get_tools(names: Optional[Sequence[str]] = None) -> List[Type[Tool]]:
    """按名字取工具类列表；names=None 表示全给。"""
    if names is None:
        return list(ALL_TOOLS)
    unknown = set(names) - ALL_TOOL_NAMES
    if unknown:
        raise ValueError(f"未知工具名：{sorted(unknown)}")
    return [TOOLS_BY_NAME[n] for n in names]


# ---------------------------------------------------------------- 数据加载

_MASTER_DATA: Optional[Dict[str, Any]] = None


def load_data_cached() -> Dict[str, Any]:
    """
    语义与 tau_bench 的 load_data 完全一致（每次返回一份独立可改的数据），
    但只解析一次磁盘上的 JSON，之后走 deepcopy。

    为什么需要：flights+reservations+users ≈ 5 MB JSON，
    每建一个 Env 要 load 一次、reset 再一次、calculate_reward 再一次。
    50 题 × 多配置下不缓存要跑好几分钟。
    """
    global _MASTER_DATA
    if _MASTER_DATA is None:
        _MASTER_DATA = load_data()
    return copy.deepcopy(_MASTER_DATA)


# ---------------------------------------------------------------- 猴补丁

_ORIGINAL_LOAD_USER = tb_base.load_user          # 留个把手，方便还原


def patch_load_user(user_sim_factory=RuleBasedUserSim) -> None:
    """
    把 base.load_user 换成返回规则模拟器。

    base.py 第 8 行是 `from tau_bench.envs.user import load_user`，
    名字被绑进了 base 模块的命名空间，所以要补丁 tb_base.load_user 而不是 user.load_user。
    """
    tb_base.load_user = lambda **kwargs: user_sim_factory()


patch_load_user()


# ---------------------------------------------------------------- 工厂


def build_env(
    task_index: int,
    tool_names: Optional[Sequence[str]] = None,
    user_sim_factory=RuleBasedUserSim,
) -> Env:
    """
    造一个 airline env，**必须显式给 task_index**。

    Args:
        task_index: 0..49
        tool_names: 工具子集；None = 全部 14 个
    """
    if not (0 <= task_index < len(TASKS)):
        raise IndexError(f"task_index 越界：{task_index}，合法范围 0..{len(TASKS)-1}")

    if tool_names is None:
        tools = list(ALL_TOOLS)
    else:
        tools = get_tools(tool_names)

    env = Env(
        data_load_func=load_data_cached,
        tools=tools,
        tasks=list(TASKS),
        wiki=WIKI,
        rules=RULES,
        user_strategy="llm",        # 会被上面的猴补丁吞掉，不会真的建 LLM 模拟器
        user_model="unused-by-rule-sim",
        user_provider=None,
        user_api_base=None,
        task_index=task_index,
    )
    # MockAirlineDomainEnv 在子类里设的这个；我们绕过它直接建 Env，得自己设
    env.terminate_tools = ["transfer_to_human_agents"]
    return env


def reward_of(env, resp=None) -> float:
    """
    安全地取 reward —— 因为 calculate_reward() **不是幂等的**。

    ⚠️ 踩过的坑（2026-09-16，块 H3 的 mock 自测抓到的）：
       calculate_reward() 内部会 `self.data = 重新加载数据()` 然后重放 gold 动作，
       **调用结束后 env.data 被留在「gold 已执行」的状态**。
       所以它只能调一次：

           第 1 次 calculate_reward() = 0.0   ← 对（什么都不做该 0 分）
           第 2 次 calculate_reward() = 1.0   ← 错！变成假满分
           第 3 次 calculate_reward() = 1.0

    为什么这很危险：如果训练循环里不小心调了两次（比如 step() 返回 done 时内部已算过一次），
    **所有题都会变成满分 → 组内全对 → std=0 → advantage 全 0 → 训练完全无效**，
    而且 loss 曲线上看不出来。

    正确用法：
      · episode 正常结束（done） → 用 step() 返回的 resp.reward（内部刚算过，是第一次）
      · 撞上 max_turns 截断      → 自己调一次 calculate_reward()（也是第一次）
    """
    if resp is not None and getattr(resp, "done", False):
        return float(resp.reward)                    # 已经算过，别再算
    return float(env.calculate_reward().reward)      # 截断的情况，第一次调用


def build_env_custom(
    task,
    tool_names: Optional[Sequence[str]] = None,
    user_sim_factory=RuleBasedUserSim,
) -> Env:
    """
    用【自定义题目】建 env —— 扩题器验证专用。

    为什么需要单独一个函数：
        build_env() 里的 tasks=list(TASKS) 是写死的 50 道题，
        task_index 只能在这 50 道里选。扩题器造出来的新题不在里面，
        所以要把任务列表换成 [task] 单元素列表。

    Args:
        task: tau_bench.types.Task（user_id / actions / instruction / outputs）
    """
    tools = list(ALL_TOOLS) if tool_names is None else get_tools(tool_names)

    env = Env(
        data_load_func=load_data_cached,
        tools=tools,
        tasks=[task],               # ← 唯一的区别：任务列表只有这一道
        wiki=WIKI,
        rules=RULES,
        user_strategy="llm",
        user_model="unused-by-rule-sim",
        user_provider=None,
        user_api_base=None,
        task_index=0,
    )
    env.terminate_tools = ["transfer_to_human_agents"]
    return env


__all__ = [
    "build_env",
    "build_env_custom",
    "reward_of",
    "get_tools",
    "TASKS",
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "ALL_TOOL_NAMES",
    "GOLD_TOOLS",
    "AGENT_ONLY_TOOLS",
    "SAFE_TO_CUT",
    "RuleBasedUserSim",
]
