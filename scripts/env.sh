#!/bin/bash
# =============================================================================
# 开机恢复 —— **下次开机后，第一件事就是 source 这个文件**。
#
#   cd /root/autodl-tmp/ToolHorizon
#   source scripts/env.sh
#
# 它把所有环境变量一次设好，省得每次重新敲、也省得漏设一个然后跑到一半报错。
#
# 为什么需要它：**AutoDL 关机再开机，环境变量不会自动恢复**
# （它是 shell 会话级的，会话一关就没了），
# 但**磁盘上的东西全在**（模型、代码、pip 装的包都保留）。
# =============================================================================

# 项目根目录（不管从哪 source 都能定位）
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_MODELS="$(dirname "$_HERE")/models"

# ---- τ-bench 在哪 ----
# 注：其实不设也能自动找到（env/bootstrap.py 会试 ../tau-bench），
#     但显式设上更稳，出问题也好排查。
export TAU_BENCH_PATH="${TAU_BENCH_PATH:-$(dirname "$_HERE")/tau-bench}"

# ---- tokenizer 用哪份 ----
# 🔴 必须和要训的模型同源。本地那份和 Qwen2.5 词表虽然一样，
#    但云上根本没带它，所以这里指到下载好的模型上。
if [ -f "$_MODELS/MODEL_PATH.txt" ]; then
    export TOOLHORIZON_TOKENIZER="$(cat "$_MODELS/MODEL_PATH.txt")"
elif [ -d "$_MODELS/Qwen2.5-1.5B-Instruct" ]; then
    export TOOLHORIZON_TOKENIZER="$_MODELS/Qwen2.5-1.5B-Instruct"
fi

# ---- 显存碎片整理（2026-09-17 那次 OOM 的报错信息里推荐的）----
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- 自检：把关键路径都验一遍，缺什么当场说清楚 ----
echo "── 环境已就绪 ──────────────────────────────────"
_ok() { echo "  ✓ $1"; }
_bad() { echo "  ✗ $1"; _FAIL=1; }

[ -n "$TAU_BENCH_PATH" ] && [ -d "$TAU_BENCH_PATH/tau_bench" ] \
    && _ok "τ-bench: $TAU_BENCH_PATH" \
    || _bad "τ-bench 找不到：$TAU_BENCH_PATH（应该和 ToolHorizon 并排）"

if [ -n "$TOOLHORIZON_TOKENIZER" ] && [ -d "$TOOLHORIZON_TOKENIZER" ]; then
    _ok "tokenizer: $TOOLHORIZON_TOKENIZER"
else
    _bad "模型/tokenizer 找不到，模型是不是被清掉了？"
    echo "     重下：跑一遍 scripts/autodl_setup.sh 的第 ④ 步"
fi

[ -d "$_HERE/data" ] && [ -f "$_HERE/data/task_split.json" ] \
    && _ok "数据: $_HERE/data" \
    || _bad "数据目录不完整：$_HERE/data"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    _ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
else
    echo "  ! 看不见 GPU —— 如果你是无卡模式开机，这是正常的；正式跑训练要切 GPU 模式"
fi

echo "────────────────────────────────────────────────"
if [ -n "${_FAIL:-}" ]; then
    echo "⚠️ 有项目没就绪，先别跑训练。上面的 ✗ 就是问题所在。"
else
    echo "✅ 全部就绪。可以跑训练了："
    echo
    echo "   # SFT 预热"
    echo "   python -m train.trainer sft --model \"\$TOOLHORIZON_TOKENIZER\" \\"
    echo "     --sft-data data/sft_final.jsonl --out models/adapter_sft"
    echo
    echo "   # GRPO 主训练（先试跑 1 轮确认管线）"
    echo "   python -m train.run --rounds 1 --select train --n 8 --limit 4 \\"
    echo "     --model \"\$TOOLHORIZON_TOKENIZER\" --run-name rehearsal"
fi
echo
