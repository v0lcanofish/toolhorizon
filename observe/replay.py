# -*- coding: utf-8 -*-
"""
从**落盘的 rollout jsonl** 离线重算观测指标 —— 不需要 GPU、不需要重跑采样。

━━━ 为什么要有这个（2026-09-19）━━━

`train/rollout_batch.py` 的 `save_rollouts` docstring 里早就写了这个设计意图：

    ⭐ 顺带的好处：每条轨迹的 messages 原样存下来 → 观测器可以**离线重算**任何指标，
       不用为了补一个指标重跑一遍 GPU。

**但这句话两周没被兑现过。** `observe/metrics.py` 里退化谱
（欠调用 / 过调用 / 抖动 / 空转）完整实现了、还有断言护着，
可**训练路径和分析脚本一次都没调用它** —— 于是 H6 报告里
「工具使用退化谱」这个**核心创新点**只报了 `tool_calls_mean` 一个数。

这个模块把那条路接上：jsonl → Episode → `metrics.compute()`。

━━━ 怎么用 ━━━

    from observe.replay import load_run, load_task_meta
    meta = load_task_meta()
    rows = load_run(Path("runs/arm_vanilla"))        # → [(step, metrics), ...]

命令行见 `scripts/analyze_spectrum.py`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from observe.metrics import compute

_PROJECT = Path(__file__).resolve().parents[1]
TASK_SPLIT = _PROJECT / "data" / "task_split.json"


# ---------------------------------------------------------------- 轨迹


@dataclass
class ReplayEpisode:
    """
    和 `env.rollout.Episode` **同形**的最小替身。

    ⚠️ 为什么不用真的 Episode：`env.rollout` 会拉起 τ-bench 的 import
       （还要求 Python ≥3.10）。观测器本来就是 duck-typing 的
       （`metrics.tool_use_of(ep)` 只读 `ep.messages`），没必要为了算指标
       把整个环境依赖拖进来 —— 分析脚本要能在任何 Python 上跑。

    ⚠️ 代价：**字段会漂移**。`env.rollout.Episode` 改了字段这里不会知道。
       所以 `scripts/selftest_ds.py` 里有一条断言，真的 import 两份来对字段名。
    """

    task_id: int
    messages: List[Dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    n_turns: int = 0
    n_tool_calls: int = 0
    terminated_by: str = ""
    # ⭐ 分组键。`--ds` 会对同一道题加采多遍，只有它能分清遍次。
    #    None 会退化成 task_id —— 但那样多遍又会并成一组，所以在 __post_init__ 里兜住。
    group_key: Any = None

    def __post_init__(self):
        if self.group_key is None:
            self.group_key = self.task_id


def episodes_from_records(records) -> List[ReplayEpisode]:
    """rollout jsonl 的记录 → ReplayEpisode 列表。"""
    out = []
    for r in records:
        out.append(ReplayEpisode(
            task_id=r["task_id"],
            messages=r.get("messages") or [],
            reward=float(r.get("reward", 0.0)),
            n_turns=int(r.get("n_turns", 0)),
            n_tool_calls=int(r.get("n_tool_calls", 0)),
            terminated_by=r.get("terminated_by", ""),
            group_key=r.get("group_key", r["task_id"]),
        ))
    return out


# ---------------------------------------------------------------- 题元信息


def load_task_meta(split_path: Optional[Path] = None) -> Dict[int, Dict[str, Any]]:
    """
    从 `data/task_split.json` 读题元信息 → `{task_id: {"class": "S"/"O", "n_write": n, ...}}`

    ⭐ 这两个字段**本来就在文件里**（不是这里现造的）：`class` 是 S/O 分流，
       `n_write` 是 gold 里有几个写动作 —— 退化谱的欠调用/过调用就靠它分池。
    """
    sp = json.loads(Path(split_path or TASK_SPLIT).read_text(encoding="utf-8"))
    meta: Dict[int, Dict[str, Any]] = {}
    for key in ("train", "probe_overcall"):
        for r in sp.get(key, []):
            meta[int(r["task"])] = dict(r)
    return meta


# ---------------------------------------------------------------- 一轮 / 一整个 run


def load_round(rollout_jsonl: Path, task_meta=None) -> Tuple[int, Dict[str, Any]]:
    """算一个 step 的指标。返回 (step, metrics)。"""
    import re
    rows = [json.loads(l) for l in
            Path(rollout_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    step = int(re.search(r"step_(\d+)", Path(rollout_jsonl).name).group(1))
    eps = episodes_from_records(rows)
    return step, compute(eps, task_meta if task_meta is not None else load_task_meta())


def load_run(run_dir: Path, rounds: Optional[int] = None, task_meta=None) -> List[Tuple[int, Dict[str, Any]]]:
    """
    算一整个 run 的逐轮指标。

    ⚠️ 每个 step_*.jsonl 有 ~10MB，25 轮就是 250MB 要解析 ——
       **别在训练还在跑的时候跑这个**（会跟采样抢 CPU）。
    """
    run_dir = Path(run_dir)
    meta = task_meta if task_meta is not None else load_task_meta()
    files = sorted(run_dir.glob("rollouts/step_*.jsonl"))
    if rounds:
        files = files[:rounds]
    return [load_round(f, meta) for f in files]
