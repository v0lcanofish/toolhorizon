# -*- coding: utf-8 -*-
"""
把"上卡要用的东西"打成一个包 —— 免得手工挑文件漏一个，上了卡才发现。

    python scripts/make_upload_bundle.py
    → 生成 dist/toolhorizon_upload.tar.gz

装什么、不装什么，都是有理由的：

  ✅ 装  env/ train/ observe/ scripts/        —— 全部代码
  ✅ 装  data/                                —— 题、SFT 数据、槽位、分流清单（5.8 MB）
  ✅ 装  requirements.txt                     —— 环境复现
  ✅ 装  tau_bench/（只装 tau_bench 子目录）    —— 环境本体（8.9 MB）
  ❌ 不装 historical_trajectories/（51 MB）    —— 只在**造 SFT 数据**时用过一次，
                                                训练和采样都不碰它
  ❌ 不装 models/                             —— 本地那份 tokenizer 和 Qwen2.5 **不同源**
                                                （151,665 vs 151,936），上卡要用模型自带的
  ❌ 不装 reference-repos/ 其余部分            —— 参考仓库只读，训练用不上
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TAU_BENCH = PROJECT.parent / "reference-repos" / "agentic-grpo-longhorizon" / "tau-bench"
OUT = PROJECT / "dist" / "toolhorizon_upload.tar.gz"

INCLUDE_DIRS = ["env", "train", "observe", "scripts", "data"]
INCLUDE_FILES = ["requirements.txt", "README.md", "smoke-test-清单.md",
                 "GPU-上卡清单.md", "AutoDL租卡步骤.md", "今天做什么.md"]

SKIP_PATTERNS = ("__pycache__", ".pyc", ".pyo", "runs", "dist")


def skip(p: Path) -> bool:
    return any(s in p.parts or p.name.endswith(s) for s in SKIP_PATTERNS)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    total = 0

    with tarfile.open(OUT, "w:gz") as tar:
        def add(p: Path, arc: str):
            nonlocal n_files, total
            if skip(p):
                return
            tar.add(p, arcname=arc)
            if p.is_file():
                n_files += 1
                total += p.stat().st_size

        for d in INCLUDE_DIRS:
            src = PROJECT / d
            if not src.is_dir():
                print(f"  ⚠️ 缺目录 {d}，跳过")
                continue
            for p in src.rglob("*"):
                if p.is_file() and not skip(p):
                    add(p, str(Path("ToolHorizon") / p.relative_to(PROJECT)))

        for f in INCLUDE_FILES:
            src = PROJECT / f
            if src.is_file():
                add(src, str(Path("ToolHorizon") / f))

        # τ-bench：只装 tau_bench 子目录，**不要** historical_trajectories（51MB，用不上）
        if TAU_BENCH.is_dir():
            tb = TAU_BENCH / "tau_bench"
            if tb.is_dir():
                for p in tb.rglob("*"):
                    if p.is_file() and not skip(p):
                        add(p, str(Path("tau-bench") / "tau_bench"
                                    / p.relative_to(tb)))
            print(f"  ✅ 装入 τ-bench 的 tau_bench/（不含 historical_trajectories）")
        else:
            print(f"  ⚠️ 没找到 τ-bench：{TAU_BENCH}")
            print(f"     上卡后自己 clone 一份，并设 TAU_BENCH_PATH 指过去：")
            print(f"     git clone https://github.com/sierra-research/tau-bench")

    size_mb = OUT.stat().st_size / 1024 ** 2
    print(f"\n→ {OUT}")
    print(f"  {n_files} 个文件，压缩后 {size_mb:.1f} MB")
    print(f"\n上卡步骤（烟雾测试 2026-09-17 已跑过，下面是接着跑的）：")
    print(f"  0. 传上去：  scp {OUT.name} root@<机器>:~/   →   tar xzf   →   cd ToolHorizon")
    print(f"  1. 恢复环境：source scripts/env.sh")
    print(f"  2. 量基座：  python scripts/eval_sft_gate.py --dry-run          # 先看要花多少钱")
    print(f"               python scripts/eval_sft_gate.py \\")
    print(f"                   --model \"$TOOLHORIZON_TOKENIZER\" --out reports/sft_gate_base.json")
    print(f"  3. SFT 预热：python -m train.trainer sft --model \"$TOOLHORIZON_TOKENIZER\" \\")
    print(f"                   --sft-data data/sft_final.jsonl --out models/adapter_sft")
    print(f"  4. 验判据：  python scripts/eval_sft_gate.py \\")
    print(f"                   --model \"$TOOLHORIZON_TOKENIZER\" --adapter models/adapter_sft \\")
    print(f"                   --out reports/sft_gate_sft.json")
    print(f"  5. 彩排一轮：python -m train.run --rounds 1 --select train --n 8 --limit 4 \\")
    print(f"                   --model \"$TOOLHORIZON_TOKENIZER\" --run-name rehearsal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
