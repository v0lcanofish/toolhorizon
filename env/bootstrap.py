# -*- coding: utf-8 -*-
"""
导入前置：把 vendored 的 τ-bench 接上 sys.path，并桩掉 litellm。

任何要用 τ-bench 的模块，**第一行**先 import 本模块：

    from env.bootstrap import TAU_BENCH   # noqa: F401  (必须先于 tau_bench)

为什么要单独一个模块：
  1. τ-bench 不在 site-packages 里，是 vendored 在 reference-repos 下的，要手动进 sys.path
  2. tau_bench/envs/user.py 第 5 行是模块顶层的 `from litellm import completion`，
     本地没装 litellm，一 import 就炸；而 import tau_bench 又必然连带执行 user.py
  3. 参考仓库要求**只读**，所以不改它的源码，改用 sys.modules 注入桩

关于 Python 版本（踩过的坑，2026-09-15）：
  τ-bench 用了 3.10+ 语法（envs/airline/env.py:22 的 match/case、
  model_utils/api/api.py 里的 X | None 注解），**Python 3.9 跑不了**，
  连 import 都过不去（SyntaxError）。本项目统一用：
      D:/anaconda/envs/myenv/python.exe   （Python 3.13 + pydantic）
"""

import sys
import types
from pathlib import Path

# ---------------------------------------------------------------- 路径

import os

PROJECT = Path(__file__).resolve().parents[1]


def _find_tau_bench() -> Path:
    """
    找 τ-bench 在哪。按优先级试四处，避免写死路径。

    ⚠️ 为什么必须可配置：τ-bench 是**别人的仓库**（MIT 许可），不能打包进本项目。
       所以每个人 clone 的位置都不一样，写死路径 = 别人跑不起来。

    顺序：
      1. 环境变量 TAU_BENCH_PATH   —— 最明确，推荐
      2. 项目同级目录 ../tau-bench  —— 把两个仓库放一起时的常见布局
      3. 项目内的 vendor/tau-bench —— 想固定版本可以放这里
      4. 原开发机的 reference-repos 布局 —— 兼容本机
    """
    cands = []
    if os.environ.get("TAU_BENCH_PATH"):
        cands.append(Path(os.environ["TAU_BENCH_PATH"]))
    cands += [
        PROJECT.parent / "tau-bench",
        PROJECT / "vendor" / "tau-bench",
        PROJECT.parent / "reference-repos" / "agentic-grpo-longhorizon" / "tau-bench",
    ]
    for c in cands:
        if (c / "tau_bench").is_dir():
            return c.resolve()
    raise RuntimeError(
        "找不到 τ-bench。请任选一种方式让它可见：\n"
        "  1) 设环境变量：export TAU_BENCH_PATH=/path/to/tau-bench\n"
        "  2) 放到项目同级：<ToolHorizon 的父目录>/tau-bench\n"
        "  3) 放到项目内：ToolHorizon/vendor/tau-bench\n"
        "获取方式：git clone https://github.com/sierra-research/tau-bench\n"
        "\n已尝试过的路径：\n  " + "\n  ".join(str(c) for c in cands)
    )


TAU_BENCH = _find_tau_bench()

if str(TAU_BENCH) not in sys.path:
    sys.path.insert(0, str(TAU_BENCH))


# ---------------------------------------------------------------- litellm 桩


class _StubCompletion:
    """桩 litellm.completion：被真调用到就报错，说明有代码想走 LLM 用户模拟器。"""

    @staticmethod
    def __call__(*args, **kwargs):
        raise RuntimeError(
            "桩 litellm 被调用了。Stage 0 应当走规则式 RuleBasedUserSim，"
            "不该有任何代码路径去调 LLM 用户模拟器。"
        )


def _install_litellm_stub() -> None:
    if "litellm" in sys.modules:
        return
    fake = types.ModuleType("litellm")
    fake.completion = _StubCompletion()
    fake.provider_list = []
    sys.modules["litellm"] = fake


_install_litellm_stub()


# ---------------------------------------------------------------- 便捷出口

__all__ = ["PROJECT", "TAU_BENCH"]
