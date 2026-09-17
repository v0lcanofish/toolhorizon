# -*- coding: utf-8 -*-
"""
策略引擎 —— 把「模型」抽象成一个函数，方便三种实现互换：

    ScriptedEngine   假模型，照本宣科（CPU，零显存，本地自测用）
    HFEngine         transformers.generate（CPU 调试 / 或 GPU 上小批量兜底）
    VLLMEngine       vLLM 离线批推理（租卡当天用，吞吐最高）

三个都实现同一个签名：
    generate(prompts: List[str], n: int) -> List[List[str]]      # [prompt][sample]

⭐ 为什么强调「同一个签名」：
    和项目二（EvidenceTrace）的编排引擎一个思路 ——
    **先用假模型把逻辑验对，再花钱接真模型**。
    本地 CPU 自测跑的就是 ScriptedEngine，一行代码不用改。

另外负责一件事：**把模型吐的文本解析成环境能吃的 Action**。
    解析失败率本身就是重要的观测指标（README 风险 R2「工具调用格式崩」）——
    所以这里不静默吞掉异常，一律记进 ParseInfo 让上层统计。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# 延迟导入 tau_bench（要 env.bootstrap 先把 sys.path 接好）
from env.bootstrap import TAU_BENCH  # noqa: F401
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402

# 模板里的工具调用格式（见 tokenizer_config.json 的 chat_template）
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>", re.S)
_LOOSE_OPEN_RE = re.compile(r"<tool_call>", re.S)


# ---------------------------------------------------------------- 解析


@dataclass
class ParseInfo:
    """一次解析的体检结果。用来统计「格式崩了没有」。"""

    kind: str = "respond"          # tool_call | respond | malformed
    reason: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.kind != "malformed"


def parse_action(text: str) -> tuple[Action, ParseInfo]:
    """
    模型输出文本 → Action。

    规则（故意宽松，因为要观测"崩"而不是"崩了就崩了"）：
        ① 有完整 `<tool_call>{...}</tool_call>` → 取第一个，JSON 能解 → 工具动作
        ② 有完整标签但 JSON 坏掉             → malformed（不猜、不修）
        ③ 有 `<tool_call>` 但没闭合           → malformed（截断了）
        ④ 什么都没有                          → 当成 respond，整段文本就是话术

    ⚠️ 为什么不"尽力修复"坏 JSON：
        修复会把「格式能力」这一维的退化曲线抹平 —— 而**那正是本项目要观测的东西**。
    """
    raw = text or ""
    m = _TOOL_CALL_RE.search(raw)
    if m:
        body = m.group("body")
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            return (Action(name=RESPOND_ACTION_NAME, kwargs={"content": raw.strip()}),
                    ParseInfo(kind="malformed", reason=f"JSON 解不开：{e}", raw=raw))
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
        if not isinstance(name, str) or not name:
            return (Action(name=RESPOND_ACTION_NAME, kwargs={"content": raw.strip()}),
                    ParseInfo(kind="malformed", reason="缺 name 字段", raw=raw))
        if isinstance(args, str):                      # 有的模型会把 arguments 再套一层字符串
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return (Action(name=RESPOND_ACTION_NAME, kwargs={"content": raw.strip()}),
                        ParseInfo(kind="malformed", reason="arguments 是坏字符串", raw=raw))
        if not isinstance(args, dict):
            return (Action(name=RESPOND_ACTION_NAME, kwargs={"content": raw.strip()}),
                    ParseInfo(kind="malformed", reason="arguments 不是对象", raw=raw))
        return Action(name=name, kwargs=args), ParseInfo(kind="tool_call", raw=raw)

    if _LOOSE_OPEN_RE.search(raw):
        return (Action(name=RESPOND_ACTION_NAME, kwargs={"content": raw.strip()}),
                ParseInfo(kind="malformed", reason="<tool_call> 没闭合（多半被截断）", raw=raw))

    return (Action(name=RESPOND_ACTION_NAME, kwargs={"content": raw.strip()}),
            ParseInfo(kind="respond", raw=raw))


# ---------------------------------------------------------------- 引擎基类


@dataclass
class GenStats:
    n_calls: int = 0
    n_sequences: int = 0
    total_new_tokens: int = 0

    def merge(self, other: "GenStats") -> None:
        self.n_calls += other.n_calls
        self.n_sequences += other.n_sequences
        self.total_new_tokens += other.total_new_tokens


class BaseEngine:
    """策略引擎接口。"""

    name = "base"

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> List[List[str]]:
        raise NotImplementedError

    def stats(self) -> GenStats:
        return GenStats()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- 假引擎


class ScriptedEngine(BaseEngine):
    """
    照本宣科的假引擎 —— **本地自测专用，零模型零显存**。

    传进来的 responder 收到 prompt，返回一段文本（会被 parse_action 解析）。
    想让假模型"坏掉"也很简单：返回半个 `<tool_call>` 就行，
    正好用来测「格式崩了会不会被看见」。
    """

    name = "scripted"

    def __init__(self, responder: Callable[[str], str]):
        self.responder = responder
        self._stats = GenStats()

    def generate(self, prompts, n=1, temperature=0.7, top_p=0.9, max_tokens=512):
        out: List[List[str]] = []
        for p in prompts:
            base = self.responder(p)
            out.append([base] * n)          # 假引擎没随机性，n 条一样
        self._stats.n_calls += 1
        self._stats.n_sequences += len(out) * n
        self._stats.total_new_tokens += sum(len(o.split()) for row in out for o in row)
        return out

    def stats(self) -> GenStats:
        return self._stats


class QueueEngine(BaseEngine):
    """按预设队列吐文本的假引擎 —— 用来精确构造"第几步调什么工具"。"""

    name = "queue"

    def __init__(self, script: List[str], fallback: str = "All set. Anything else I can help with?"):
        self.script = list(script)
        self.i = 0
        self.fallback = fallback

    def generate(self, prompts, n=1, temperature=0.7, top_p=0.9, max_tokens=512):
        out = []
        for _ in prompts:
            if self.i < len(self.script):
                t = self.script[self.i]
                self.i += 1
            else:
                t = self.fallback
            out.append([t] * n)
        return out


# ---------------------------------------------------------------- 真引擎


class HFEngine(BaseEngine):
    """
    transformers.generate。**只在 CPU 调试或小批量兜底时用**，吞吐远不如 vLLM。

    真训练默认走 VLLMEngine（见 train/collect.py）。
    """

    name = "hf"

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._stats = GenStats()
        self.stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>")]
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def generate(self, prompts, n=1, temperature=0.7, top_p=0.9, max_tokens=512):
        import torch

        outs: List[List[str]] = []
        for p in prompts:
            ids = self.tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"]
            ids = ids.to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    ids,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=self.pad_id,
                    eos_token_id=self.stop_ids,
                )
            row = []
            for g in gen:
                new = g[ids.shape[1]:]
                txt = self.tokenizer.decode(new, skip_special_tokens=True)
                row.append(txt)
                self._stats.total_new_tokens += int(new.shape[0])
            outs.append(row)
        self._stats.n_calls += 1
        self._stats.n_sequences += sum(len(r) for r in outs)
        return outs

    def stats(self) -> GenStats:
        return self._stats


class VLLMEngine(BaseEngine):
    """
    vLLM 离线批推理。**租卡当天才用得上**（本地没装 vllm，import 放在 __init__ 里）。

    为什么用离线 `LLM` 而不是起 `vllm serve` 服务：
        本项目的取舍是**分时复用**（vLLM 和训练不同时驻留显存，省约 40%）。
        离线模式就是一个普通进程 —— 采完数据就退出，显存干净还给训练。
        起服务反而要管进程生命周期，容易踩"vLLM 没真死 → 训练 OOM 但 nvidia-smi 看着是空的"。

    ⚠️ 必须传 `stop=["<|im_end|>"]`：
        我们自己渲染 prompt（train/tokenize.render），不走 vLLM 的 chat template，
        所以停止符也得自己声明 —— 否则模型会一路自问自答下去。
    """

    name = "vllm"

    def __init__(
        self,
        model_path: str,
        tokenizer,
        lora_path: Optional[str] = None,
        lora_rank: int = 16,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 12288,
    ):
        from vllm import LLM, SamplingParams  # 延迟导入
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self.tokenizer = tokenizer
        self.llm = LLM(
            model=model_path,
            enable_lora=lora_path is not None,
            max_lora_rank=lora_rank if lora_path else None,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
        )
        # ⭐ 权重同步就靠这个对象：训练存完 adapter，采样进程重启时把新的
        #    lora_path 塞进来即可。代价约 5 秒（adapter 只有 37 MB）。
        self._lora_request = (LoRARequest("adapter", 1, lora_path) if lora_path else None)
        self._stats = GenStats()

    def generate(self, prompts, n=1, temperature=0.7, top_p=0.9, max_tokens=512):
        sp = self._SamplingParams(
            n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, stop=["<|im_end|>"],
        )
        reqs = self.llm.generate(prompts, sp, use_tqdm=False,
                                 lora_request=self._lora_request)
        outs: List[List[str]] = []
        for r in reqs:
            row = [o.text for o in r.outputs]
            outs.append(row)
            self._stats.total_new_tokens += sum(len(o.token_ids) for o in r.outputs)
        self._stats.n_calls += 1
        self._stats.n_sequences += sum(len(r) for r in outs)
        return outs

    def stats(self) -> GenStats:
        return self._stats


__all__ = [
    "parse_action",
    "ParseInfo",
    "BaseEngine",
    "ScriptedEngine",
    "QueueEngine",
    "HFEngine",
    "VLLMEngine",
    "GenStats",
]
