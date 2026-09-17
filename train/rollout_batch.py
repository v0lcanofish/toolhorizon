# -*- coding: utf-8 -*-
"""
分组批量采样 —— GRPO 的"组"就是在这里形成的。

GRPO 的 advantage 是**组内相对**的：
    A_i = (r_i − mean(group)) / (std(group) + eps)
所以"组"必须是**同一道题的 n 条采样**，而且必须来自**当前策略**（不能用历史轨迹回放）。
这个文件干的就是：把 B 道题 × n 条 = B×n 条轨迹，**同步推进**着采出来。

为什么同步（lockstep）而不是一条一条跑：
    引擎（vLLM）是按批吞吐的，一条一条跑等于把 GPU 当 CPU 用。
    每轮把所有还活着的轨迹凑成一个 batch 喂进去，是单卡上唯一现实的打法。

⚠️ 三个必须和 env/rollout.py 保持一致的地方（这个文件里都调用同一份代码，不重写）：
    ① 消息怎么写进 msgs      → env.rollout.step_messages
    ② reward 怎么取          → env.tau_env.reward_of（calculate_reward 不幂等！）
    ③ 系统提示是哪一份       → env.rollout.load_system_prompt

⭐ 零方差是**一等公民**：不是"算出来顺便看看"，而是每个组都显式记录
   `zero_variance`，因为它是本项目的核心命题（全对/全错 → 零梯度 → 白跑一步）。
"""

from __future__ import annotations

import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from env.rollout import Episode, load_system_prompt, step_messages
from env.tau_env import build_env_custom, reward_of
from train.engine import BaseEngine, GenStats, ParseInfo, parse_action
from train.tokenize import load_tokenizer, render, tool_schemas

_PROJECT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 配置


@dataclass
class RolloutConfig:
    """采样配置。默认值对齐 longhorizon 的 vanilla_grpo.yaml（n=8, temp=0.7）。"""

    n_group: int = 8                 # 每道题采几条（GRPO 的 group size）
    max_turns: int = 40              # 单条轨迹的硬上限（防弱策略无限循环）
    max_seq_tokens: int = 8192       # S_max：prompt + 生成 的总预算
    max_new_tokens: int = 512        # 单轮最多生成多少
    temperature: float = 0.7
    top_p: float = 0.9
    truncated_reward: float = 0.0    # 撞 max_turns / 超长的轨迹给多少分

    # ⚠️ 为什么截断给 0 而不是"已经做对的部分功"：
    #    README 写死了这条 —— **不给部分奖励，否则等于奖励"啰嗦到被截断"**。
    #    弱策略一旦发现"说废话也能得分"，就会朝那个方向漂。

    @property
    def max_prompt_tokens(self) -> int:
        """prompt 预算 = S_max − 这一轮要生成的长度。**推导出来，不手填。**"""
        return self.max_seq_tokens - self.max_new_tokens


# ---------------------------------------------------------------- 一条在跑的轨迹


@dataclass
class Trajectory:
    """一条正在推进的轨迹（还没跑完时，状态在这里）。"""

    task_id: int
    sample_idx: int
    env: Any
    msgs: List[Dict[str, Any]]
    ep: Episode
    resp: Any = None
    finished: bool = False
    n_malformed: int = 0
    parse_kinds: List[str] = field(default_factory=list)
    prompt_tokens: int = 0

    @property
    def key(self) -> str:
        return f"task{self.task_id}_s{self.sample_idx}"


@dataclass
class GroupResult:
    """一道题的 n 条轨迹 + 组内 advantage。**GRPO 训练的最小单位**。"""

    task_id: int
    rewards: List[float]
    advantages: List[float]
    std: float
    zero_variance: bool
    episodes: List[Episode]
    samples: List[Trajectory]

    @property
    def mean(self) -> float:
        return st.mean(self.rewards) if self.rewards else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rewards": self.rewards,
            "advantages": self.advantages,
            "std": self.std,
            "zero_variance": self.zero_variance,
            "reward_mean": self.mean,
            "n_malformed": sum(s.n_malformed for s in self.samples),
            "terminated_by": [e.terminated_by for e in self.episodes],
        }


@dataclass
class RolloutBatch:
    """一次采样（若干组）的整体结果。"""

    groups: List[GroupResult]
    gen_stats: GenStats

    @property
    def episodes(self) -> List[Episode]:
        return [e for g in self.groups for e in g.episodes]

    @property
    def zero_var_rate(self) -> float:
        """⭐ 第一监控指标：多少比例的组是"全对或全错"（→ 零梯度 → 白跑一步）。"""
        if not self.groups:
            return 0.0
        return sum(1 for g in self.groups if g.zero_variance) / len(self.groups)

    def summary(self) -> Dict[str, Any]:
        eps = self.episodes
        if not eps:
            return {}
        rewards = [e.reward for e in eps]
        return {
            "n_groups": len(self.groups),
            "n_episodes": len(eps),
            "zero_var_rate": self.zero_var_rate,
            "reward_mean": st.mean(rewards),
            "pass_rate": sum(1 for r in rewards if r >= 1.0 - 1e-9) / len(rewards),
            "turns_mean": st.mean([e.n_turns for e in eps]),
            "tool_calls_mean": st.mean([e.n_tool_calls for e in eps]),
            "malformed_rate": (
                sum(len(s.parse_kinds) and s.n_malformed / max(1, len(s.parse_kinds))
                    for s in [x for g in self.groups for x in g.samples])
                / max(1, len(eps))
            ),
            "truncated_rate": sum(
                1 for e in eps if e.terminated_by in ("max_turns", "context_overflow")
            ) / len(eps),
        }


# ---------------------------------------------------------------- advantage


def compute_advantages(rewards: Sequence[float], eps: float = 1e-8):
    """
    GRPO 的组内相对优势 —— **本项目核心命题就藏在这个式子里**。

        A_i = (r_i − mean) / (std + eps)

    当组内 reward 全相同时（全对或全错）std=0 → 分子也全是 0 → **A 全 0**。
    这不是"训练不稳定"，是**这一步真的什么都没学到**（零梯度），
    但 loss 曲线上往往只是"平了一下"，看不出来。

    Returns:
        (advantages, std, zero_variance)
    """
    n = len(rewards)
    if n == 0:
        return [], 0.0, True
    m = st.mean(rewards)
    sd = st.pstdev(rewards) if n > 1 else 0.0
    if sd < eps:
        return [0.0] * n, sd, True
    return [(r - m) / (sd + eps) for r in rewards], sd, False


# ---------------------------------------------------------------- 主循环


def _finish(t: Trajectory, why: str, cfg: RolloutConfig) -> None:
    """
    收尾一条轨迹并结算 reward。

    ⚠️ 这里必须走 reward_of —— `calculate_reward()` **不幂等**，
       调第二次会把所有题变成假满分（详见 env/tau_env.py 的注释）。
    ⚠️ 截断的轨迹（max_turns / 超长）**根本不去调 calculate_reward**：
       直接给 truncated_reward（默认 0），理由见 RolloutConfig。
       顺带一个好处：完全绕开了不幂等的坑。
    """
    t.finished = True
    t.ep.terminated_by = why

    if why in ("max_turns", "context_overflow"):
        t.ep.done = False
        t.ep.reward = float(cfg.truncated_reward)
        return

    t.ep.done = True
    t.ep.reward = reward_of(t.env, t.resp)


def run_grouped_rollout(
    task_items: Sequence[Tuple[int, Any]],
    engine: BaseEngine,
    cfg: Optional[RolloutConfig] = None,
    tokenizer=None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> RolloutBatch:
    """
    采样一批：task_items 里每道题 × cfg.n_group 条，**同步推进**。

    Args:
        task_items: [(task_id, tau_bench.types.Task), ...]
        engine:     策略引擎（ScriptedEngine / VLLMEngine / HFEngine 都行）
        tokenizer:  用于算 prompt token 数、判断超长；None 则加载本地那份
        tools:      工具 schema —— **必须和 SFT 用同一份**（见 train/tokenize.py 顶部）
    """
    cfg = cfg or RolloutConfig()
    tok = tokenizer or load_tokenizer()
    tools = tools if tools is not None else tool_schemas()
    sys_prompt = load_system_prompt()

    # ---- 建轨道
    trajs: List[Trajectory] = []
    for tid, task in task_items:
        for k in range(cfg.n_group):
            env = build_env_custom(task)
            env.reset(task_index=0)          # ⚠️ 必须显式传：默认 random.randint 有 off-by-one
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": task.instruction},
            ]
            ep = Episode(task_id=tid, messages=msgs)
            trajs.append(Trajectory(tid, k, env, msgs, ep))

    # ---- 同步推进
    while True:
        pending = [t for t in trajs if not t.finished]
        if not pending:
            break

        prompts: List[str] = []
        keep: List[Trajectory] = []
        for t in pending:
            if t.ep.n_turns >= cfg.max_turns:
                _finish(t, "max_turns", cfg)
                continue

            p = render(t.msgs, tools, tok, add_generation_prompt=True)
            ntok = len(tok(p, add_special_tokens=False)["input_ids"])
            t.prompt_tokens = ntok
            if ntok + cfg.max_new_tokens > cfg.max_prompt_tokens:
                if t.ep.n_turns == 0:
                    # ⭐ 防呆：第一轮就装不下，不是"策略啰嗦"，是**配置错了**。
                    #    实测固定开销（system 1265 + 12 工具 schema 2360 = 3625 token）
                    #    已经吃掉 S_max=8192 的 44%，拍脑袋填的 prompt 预算很容易比它还小。
                    #    这时候静默把所有轨迹判成"超长"会让人以为是模型的问题——
                    #    所以这里直接炸掉，把真实数字甩到脸上。
                    raise ValueError(
                        f"prompt 预算装不下固定开销：task {t.task_id} 第一轮就要 "
                        f"{ntok} token，加上 max_new_tokens={cfg.max_new_tokens} 超过 "
                        f"max_prompt_tokens={cfg.max_prompt_tokens}"
                        f"（= S_max {cfg.max_seq_tokens} − {cfg.max_new_tokens}）。\n"
                        f"  → 要么调大 max_seq_tokens，要么砍 system 提示 / 工具 schema。"
                    )
                # 再生成一轮就铁定超长 → 现在就掐掉，别浪费一次前向
                _finish(t, "context_overflow", cfg)
                continue

            prompts.append(p)
            keep.append(t)

        if not prompts:
            continue

        outs = engine.generate(
            prompts, n=1,
            temperature=cfg.temperature, top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
        )

        for t, texts in zip(keep, outs):
            text = texts[0] if texts else ""
            action, info = parse_action(text)
            t.parse_kinds.append(info.kind)
            if not info.ok:
                t.n_malformed += 1

            t.ep.n_turns += 1
            resp = t.env.step(action)
            t.resp = resp
            step_messages(t.msgs, t.ep, action, resp, t.task_id)

            if resp.done:
                term = ("terminate_tool"
                        if action.name in getattr(t.env, "terminate_tools", [])
                        else "user_stop")
                _finish(t, term, cfg)

    # ---- 组装分组 + advantage
    by_task: Dict[int, List[Trajectory]] = {}
    for t in trajs:
        by_task.setdefault(t.task_id, []).append(t)

    groups: List[GroupResult] = []
    for tid, ts in by_task.items():
        ts.sort(key=lambda x: x.sample_idx)
        rewards = [t.ep.reward for t in ts]
        advs, sd, zero = compute_advantages(rewards)
        groups.append(GroupResult(
            task_id=tid,
            rewards=rewards,
            advantages=advs,
            std=sd,
            zero_variance=zero,
            episodes=[t.ep for t in ts],
            samples=ts,
        ))

    groups.sort(key=lambda g: g.task_id)
    return RolloutBatch(groups=groups, gen_stats=engine.stats())


# ---------------------------------------------------------------- 落盘


def save_rollouts(batch: RolloutBatch, path, step: int = 0) -> int:
    """
    把一次采样写成 jsonl —— **训练进程从这里读数据**。

    为什么走磁盘而不是传内存：
        本项目的单卡策略是**分时复用**（vLLM 采样完就退出，显存还给训练）。
        两个进程用文件交接，比同进程共存简单得多，也不可能踩
        "vLLM 没真死 → 训练 OOM 但 nvidia-smi 看着是空的"那个坑。
    ⭐ 顺带的好处：每条轨迹的 messages 原样存下来 → 观测器可以**离线重算**任何指标，
        不用为了补一个指标重跑一遍 GPU。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for g in batch.groups:
            for s in g.samples:
                rec = {
                    "step": step,
                    "task_id": g.task_id,
                    "sample_idx": s.sample_idx,
                    "reward": s.ep.reward,
                    "advantage": g.advantages[s.sample_idx],
                    "group_std": g.std,
                    "zero_variance_group": g.zero_variance,
                    "n_turns": s.ep.n_turns,
                    "n_tool_calls": s.ep.n_tool_calls,
                    "terminated_by": s.ep.terminated_by,
                    "n_malformed": s.n_malformed,
                    "prompt_tokens": s.prompt_tokens,
                    "messages": s.msgs,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def load_rollouts(path) -> List[Dict[str, Any]]:
    """读回采样结果（训练进程用）。"""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


__all__ = [
    "RolloutConfig",
    "Trajectory",
    "GroupResult",
    "RolloutBatch",
    "compute_advantages",
    "run_grouped_rollout",
    "save_rollouts",
    "load_rollouts",
]
