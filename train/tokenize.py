# -*- coding: utf-8 -*-
"""
消息 ↔ token 的唯一入口 —— **SFT 和 rollout 必须共用这一个文件**。

为什么"唯一入口"是硬要求：
    训练时喂给模型的 prompt，和 rollout 时喂给模型的 prompt，
    只要差一个 token，SFT 预热出来的策略就和 GRPO 里采样的就不是同一个分布。
    **格式漂移是比超参更隐蔽的杀手**——loss 曲线看不出来，只有 final 分数会莫名其妙地低。

本文件解决三件事：
    ① 渲染    messages → 文本（含工具 schema，由 chat template 负责）
    ② 对齐    messages → token 序列 + **逐 token 的 loss mask**
    ③ 防呆    两路独立算法算 mask，互相对账（见 scripts/eval_train_logic.py）

⚠️ 本项目最容易踩的一个坑（写完这个文件才确认下来）：
    **工具 schema 不在 system 消息里**（实测 data/system_prompt.txt 里
    `book_reservation` 出现 0 次），它只能靠 `apply_chat_template(tools=...)` 渲染进去。
    → 于是「传不传 tools」会改变**系统提示的全文**，
      一处传、一处不传，训练和推理就变成了两个任务。
    所以本文件里所有渲染都强制要求传 tools。

跑法：被 SFT / rollout / 自测脚本 import，不单独跑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = _PROJECT / "models" / "qwen1.5b-tokenizer"

# 训练用的工具子集：14 − think − list_all_airports = 12（见 README v3 修订）
DEFAULT_TOOLS = [
    "book_reservation",
    "calculate",
    "cancel_reservation",
    "get_reservation_details",
    "get_user_details",
    "search_direct_flight",
    "search_onestop_flight",
    "send_certificate",
    "transfer_to_human_agents",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
]


# ---------------------------------------------------------------- tokenizer


def load_tokenizer(path: Optional[str] = None, model_path: Optional[str] = None):
    """
    加载 tokenizer。

    优先级：`model_path` > `path` > 环境变量 `TOOLHORIZON_TOKENIZER` > 本地默认目录。

    Args:
        path:       本地 tokenizer 目录（默认 `models/qwen1.5b-tokenizer`，11MB，CPU 自测用）
        model_path: **要训的模型**的路径/名字。

    ⚠️ 环境变量 `TOOLHORIZON_TOKENIZER` 是给**租来的机器**用的：
       打包时刻意不带本地 tokenizer（它和 Qwen2.5 不同源），
       卡上要用的是刚下好的那个模型自带的。setup 脚本会把它 export 出来，
       这样所有自测脚本在不改一行代码的前提下就用上对的 tokenizer。

    🔴 **训练时一定要传 `model_path`** —— 这个坑不报错但会毁掉一切：

        本地那份 tokenizer 是 **151,665** 个 token；
        Qwen2.5-1.5B-Instruct 自带的是 **151,936** 个（同一段文字切出来的 id 可能不一样）。
        用本地这份去 tokenize、喂给那个模型 —— **数字对不上，模型看到的是乱码**，
        而 loss 照样会降（它会去拟合这个乱码），**到评测时才发现**。

    所以规则是：**tokenizer 必须和模型同源**。
    本地这份只用于 CPU 自测（那时模型也是本地造的，两边一致）。
    """
    import os

    from transformers import AutoTokenizer

    p = str(model_path or path
            or os.environ.get("TOOLHORIZON_TOKENIZER")
            or TOKENIZER_DIR)
    # 判断"这看起来是个本地路径吗" —— 是本地路径且不存在，就直接报清楚，
    # 别丢给 transformers 去猜（它会当成 HuggingFace repo id，报一个看不懂的错）
    _looks_local = p.startswith(("/", "./", "../", "~")) or (len(p) > 2 and p[1] == ":")
    if _looks_local and not Path(p).exists():
        raise FileNotFoundError(
            f"找不到 tokenizer：{p}\n"
            "  本地自测用：项目里的 models/qwen1.5b-tokenizer\n"
            "  租来的机器上：设 TOOLHORIZON_TOKENIZER=<下载好的模型目录>\n"
            "  （打包时刻意不带本地那份 —— 它和 Qwen2.5 词表不同源，用了会静默毁掉训练）"
        )
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def check_tokenizer_matches(model_path: str, verbose: bool = True) -> dict:
    """
    对账：模型自带的 tokenizer 和本地那份是不是同一个。

    ⚠️ 租卡当天**必须先跑这个**。不一样的话，之前用本地 tokenizer 量出来的
       所有 token 数（p50=6,692、固定开销 3,625、S_max 预算…）都要重算。

    ⚠️ **本地那份可能压根不存在** —— 租来的机器上是刻意不带的（它和 Qwen2.5 不同源）。
       这不是错误，只是"没有对照物可对"，跳过即可。
       （2026-09-17 踩：第一版没处理这种情况，在卡上直接把整个烟雾测试炸停了）
    """
    remote = load_tokenizer(model_path=model_path)
    info: Dict[str, Any] = {"model_vocab": len(remote), "model_path": model_path}

    try:
        local = load_tokenizer()
    except Exception as e:                                   # noqa: BLE001
        info.update({"local_vocab": None, "compared": False,
                     "reason": f"{type(e).__name__}: {str(e)[:100]}"})
        if verbose:
            print(f"  ⚠️ 本地没有对照 tokenizer（{type(e).__name__}）—— 跳过对账")
            print(f"  → 训练用的是模型自带的：词表 {len(remote)}")
            print(f"     （这是租来的机器的正常情况，不影响后续）")
        return info

    same = len(local) == len(remote)
    probe = "You want to change your upcoming reservation from ATL to PHL on 2024-05-20."
    ids_same = local(probe)["input_ids"] == remote(probe)["input_ids"]
    info.update({
        "local_vocab": len(local),
        "vocab_same": same,
        "sample_ids_same": ids_same,
        "compared": True,
    })
    if verbose:
        print(f"  本地 tokenizer 词表 {info['local_vocab']} ｜ {model_path} 词表 {info['model_vocab']}")
        if same and ids_same:
            print("  ✅ 一致，之前量的 token 数可以直接用")
        else:
            print("  ❌ **不一致**：必须改用模型自带的那份，并重跑 scripts/token_stats.py")
            print("     （train/collect.py 和 train/trainer.py 已经会自动用模型那份）")
    return info


# ---------------------------------------------------------------- 工具 schema


def tool_schemas(names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    取工具 schema（OpenAI 格式，正是 chat template 期望的形状）。

    返回的是 `{"type": "function", "function": {...}}` ——
    直接来自 τ-bench 的 `Tool.get_info()`，和真实轨迹里送给 gpt-4o 的是同一份。
    """
    import sys

    if str(_PROJECT) not in sys.path:
        sys.path.insert(0, str(_PROJECT))
    from env import ALL_TOOLS  # noqa: E402  延迟导入：避免 tokenize 层依赖 env

    by_name = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
    if names is None:
        names = DEFAULT_TOOLS
    unknown = set(names) - set(by_name)
    if unknown:
        raise ValueError(f"未知工具名：{sorted(unknown)}")
    return [by_name[n].get_info() for n in names]


# ---------------------------------------------------------------- 渲染


def render(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tokenizer=None,
    add_generation_prompt: bool = False,
) -> str:
    """
    messages → 文本。**所有** prompt 构造都必须走这里。

    Args:
        tools: 工具 schema；**训练和 rollout 都必须传同一份**（见文件头的警告）。
               model=None 时会即时从 τ-bench 取。
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if tools is None:
        tools = tool_schemas()
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def header_suffix(messages: List[Dict[str, Any]], tools, tokenizer) -> str:
    """
    取「生成提示后缀」= `<|im_start|>assistant\\n`。

    不硬编码 —— 从模板自己身上量出来：
        render(msgs, gen=True) − render(msgs, gen=False)
    换 tokenizer / 换模板版本时自动跟着变。
    """
    plain = render(messages, tools, tokenizer, add_generation_prompt=False)
    with_gen = render(messages, tools, tokenizer, add_generation_prompt=True)
    if not with_gen.startswith(plain):
        raise AssertionError("模板不满足前缀假设：add_generation_prompt 不是纯追加")
    return with_gen[len(plain):]


# ---------------------------------------------------------------- 对齐


def assistant_char_spans(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tokenizer=None,
) -> List[Tuple[int, int]]:
    """
    算出「每一条 assistant 消息」在渲染文本里的**字符区间**。

    原理（不假设模板长什么样，只用它的前缀性质）：

        R(k) = render(messages[:k], gen=False)      # 前 k 条渲染出来
        R(i) 是 R(i+1) 的前缀

        assistant i 的起点 = len( render(messages[:i], gen=True) )
                             ↑ = R(i) + '<|im_start|>assistant\\n'，即"轮到它说话了"
        assistant i 的终点 = len( R(i+1) ) - 1
                             ↑ 减 1 是砍掉尾随换行；`<|im_end|>` **留在 loss 里**，
                               因为"学会收尾"本身就是 SFT 要教的东西
                               （本 benchmark 的正解必须主动汇报，不会收尾直接丢分）
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if tools is None:
        tools = tool_schemas()

    spans: List[Tuple[int, int]] = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        before_gen = render(messages[:i], tools, tokenizer, add_generation_prompt=True)
        through_i = render(messages[:i + 1], tools, tokenizer, add_generation_prompt=False)
        start = len(before_gen)
        end = len(through_i) - 1          # 砍尾随 '\n'
        if end <= start:
            raise AssertionError(
                f"第 {i} 条 assistant 渲染出来是空的（start={start}, end={end}）"
            )
        spans.append((start, end))
    return spans


@dataclass
class Encoded:
    """一条轨迹的 token 化结果。"""

    input_ids: List[int]
    loss_mask: List[int]           # 1 = 算 loss（assistant 自己产出的 token）
    spans: List[Tuple[int, int, int, int]]   # (tok_start, tok_end, char_start, char_end)
    text: str
    truncated: bool = False

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def n_loss_tokens(self) -> int:
        return sum(self.loss_mask)

    @property
    def loss_ratio(self) -> float:
        return self.n_loss_tokens / max(1, self.n_tokens)


def encode(
    messages: List[Dict[str, Any]],
    tokenizer=None,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_length: Optional[int] = None,
    truncate_side: str = "left",
) -> Encoded:
    """
    messages → token + loss mask。

    loss mask 用**字符偏移**判定：某个 token 的字符区间完全落在某条 assistant
    消息的区间里，它才参与 loss。其余（system / user / tool 返回）一律 0。

    ⚠️ 为什么不用「数 token 个数」那种简单办法：
        `1 token ≠ 1 字符`，中英混排 / 特殊 token 一搅，边界必错。
        错几个 token 不会报错，只会让模型学到"把用户的话也复述一遍"。

    Args:
        max_length:  超过就截断。**按 p90=10,287 实测，8K 下必须截断**。
        truncate_side: 'left' = 砍最老的 tool 返回（推荐）；
                       'right' = 砍尾巴（会把最后的收尾回复砍掉，教学效果差）
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if tools is None:
        tools = tool_schemas()

    text = render(messages, tools, tokenizer, add_generation_prompt=False)
    spans_char = assistant_char_spans(messages, tools, tokenizer)

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids: List[int] = list(enc["input_ids"])
    offsets: List[Tuple[int, int]] = list(enc["offset_mapping"])
    if len(ids) != len(offsets):
        raise AssertionError(f"id 数与 offset 数不一致：{len(ids)} vs {len(offsets)}")

    # ---- 逐 token 判 mask
    mask = [0] * len(ids)
    for j, (ts, te) in enumerate(offsets):
        if te <= ts:
            # 字符长度为 0 的 token：通常是 tokenizer 自己插的特殊 token。
            # 它们不属于任何 assistant 区间，默认不算 loss。
            continue
        for (cs, ce) in spans_char:
            if ts >= cs and te <= ce:
                mask[j] = 1
                break

    # ---- 截断
    truncated = False
    if max_length is not None and len(ids) > max_length:
        truncated = True
        if truncate_side == "left":
            cut = len(ids) - max_length
            ids = ids[cut:]
            mask = mask[cut:]
        else:
            ids = ids[:max_length]
            mask = mask[:max_length]

    # ---- 转成 token 级区间（供断言与可视化用）
    tok_spans: List[Tuple[int, int, int, int]] = []
    for (cs, ce) in spans_char:
        idx = [j for j, (ts, te) in enumerate(offsets) if te > ts and ts >= cs and te <= ce]
        if idx:
            tok_spans.append((idx[0], idx[-1] + 1, cs, ce))

    if truncated:
        # 截断后 token 下标整体左移，区间跟着挪
        shift = len(enc["input_ids"]) - len(ids)
        tok_spans = [
            (max(0, a - shift), max(0, b - shift), c, d) for (a, b, c, d) in tok_spans
        ]
        tok_spans = [(a, b, c, d) for (a, b, c, d) in tok_spans if b > a]

    return Encoded(
        input_ids=ids,
        loss_mask=mask,
        spans=tok_spans,
        text=text,
        truncated=truncated,
    )


def encode_by_incremental(
    messages: List[Dict[str, Any]],
    tokenizer=None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """
    **第二路算法**：用「增量差分」算 loss mask —— 只为对账用。

    做法：对每条 assistant 消息 i，分别 tokenize
        A = render(messages[:i], gen=True)          ← 轮到它说话
        B = render(messages[:i+1], gen=False)       ← 说完 + 收尾
    若 B 以 A 的 token 序列为前缀，则 [len(A), len(B)-1) 就是它的 loss 区间。

    和 `encode()` 的字符偏移法**完全独立**（一个走字符、一个走 token）。
    两者算出来必须一模一样 —— 这是 scripts/eval_train_logic.py 的核心断言。
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if tools is None:
        tools = tool_schemas()

    full = render(messages, tools, tokenizer, add_generation_prompt=False)
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    mask = [0] * len(ids)

    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        a_text = render(messages[:i], tools, tokenizer, add_generation_prompt=True)
        b_text = render(messages[:i + 1], tools, tokenizer, add_generation_prompt=False)
        a_ids = tokenizer(a_text, add_special_tokens=False)["input_ids"]
        b_ids = tokenizer(b_text, add_special_tokens=False)["input_ids"]
        if len(a_ids) > len(b_ids) or b_ids[: len(a_ids)] != a_ids:
            raise AssertionError(
                f"增量法失败：第 {i} 条 assistant 的 B 不以 A 为前缀。"
                "说明 tokenizer 在拼接处不满足前缀性质，需要改用字符偏移法。"
            )
        for j in range(len(a_ids), len(b_ids) - 1):     # −1：砍尾随 '\n'
            mask[j] = 1
    return mask


# ---------------------------------------------------------------- 解码（调试/可视化）


def spans_to_readable(
    enc: Encoded,
    tokenizer,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """把 loss 区间解回文本，用来肉眼确认「算 loss 的到底是不是 agent 说的话」。"""
    out = []
    for k, (a, b, cs, ce) in enumerate(enc.spans):
        if k >= limit:
            break
        out.append({
            "idx": k,
            "tok_span": (a, b),
            "n_tokens": b - a,
            "text": tokenizer.decode(enc.input_ids[a:b], skip_special_tokens=False),
        })
    return out


__all__ = [
    "load_tokenizer",
    "tool_schemas",
    "render",
    "header_suffix",
    "assistant_char_spans",
    "encode",
    "encode_by_incremental",
    "spans_to_readable",
    "Encoded",
    "DEFAULT_TOOLS",
    "TOKENIZER_DIR",
]
