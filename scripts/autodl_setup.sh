#!/bin/bash
# =============================================================================
# AutoDL 一键初始化 —— 在**刚租好的机器**上跑这一条，剩下的它自己干。
#
#   bash scripts/autodl_setup.sh
#
# 它做五件事（每件都会打印结果，失败会明确告诉你卡在哪）：
#   ① 检查 Python 版本（τ-bench 要 3.10+，低了直接报错）
#   ② 开 AutoDL 学术加速（下模型 / pip 都靠它，不开会卡死在 HuggingFace）
#   ③ 装依赖（peft / vllm 等）
#   ④ 下 Qwen2.5-1.5B-Instruct（约 3GB，带镜像兜底）
#   ⑤ 自检：把项目的自测脚本跑一遍，确认代码在卡上也是好的
#
# ⭐ 省钱提示：**这些活全都能在「无卡模式」下做完**（¥0.1/小时），
#    做完关机，再切到 GPU 模式跑 smoke —— 能省掉一半的准备时间钱。
# =============================================================================

set -u

# ---------- 可调 ----------
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
WORK="${WORK:-/root/autodl-tmp}"          # AutoDL 的数据盘，比系统盘大
# --------------------------

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; NC=$'\033[0m'
ok()   { echo "${GRN}  ✓${NC} $*"; }
warn() { echo "${YEL}  !${NC} $*"; }
die()  { echo "${RED}  ✗ $*${NC}"; exit 1; }
step() { echo; echo "── $* ──────────────────────────────────"; }

echo "================================================================"
echo " ToolHorizon · AutoDL 环境初始化"
echo "================================================================"

# ---------------------------------------------------------------- ① Python
step "① 检查 Python 版本"
PY="$(command -v python || true)"
[ -n "$PY" ] || die "找不到 python。先确认镜像选的是 PyTorch 镜像。"
VER="$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "  当前 python: $PY  ($VER)"
$PY -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' \
  || die "Python $VER 太老。τ-bench 用了 3.10+ 语法，3.9 连 import 都过不去（SyntaxError）。
     → 换一个 Python ≥3.10 的镜像重租，别在这个上面耗。"

# ---------------------------------------------------------------- ② 学术加速
step "② 开学术加速（下模型要靠它）"
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  ok "已开启（AutoDL 学术加速）"
else
  warn "没找到 /etc/network_turbo —— 可能不是 AutoDL，或者镜像不一样"
  warn "退而求其次用 HF 镜像站"
  export HF_ENDPOINT=https://hf-mirror.com
fi

# ---------------------------------------------------------------- ③ 依赖
step "③ 装依赖"
$PY -c "import torch;print('  torch',torch.__version__,'| cuda',torch.version.cuda)" \
  || die "torch 没装好。镜像选错了，重租一个 PyTorch 镜像。"

$PY -m pip install -q peft transformers pydantic matplotlib 2>&1 | tail -3
ok "peft / transformers / pydantic / matplotlib 就绪"

if $PY -c "import vllm" 2>/dev/null; then
  ok "vllm 已存在（镜像自带）：$($PY -c 'import vllm;print(vllm.__version__)')"
else
  echo "  正在装 vllm（比较大，几分钟）…"
  $PY -m pip install -q vllm 2>&1 | tail -5
  $PY -c "import vllm;print('  ✓ vllm',vllm.__version__)" \
    || warn "vllm 装不上 —— 先别管，smoke 脚本会用 HF 引擎兜底跑，
    但正式训练会慢很多。把报错贴出来。"
fi

# ---------------------------------------------------------------- ④ 模型
step "④ 下模型（约 3 GB）"
mkdir -p "$WORK/models"
LOCAL="$WORK/models/$(basename "$MODEL_ID")"

if [ -d "$LOCAL" ] && ls "$LOCAL"/*.safetensors >/dev/null 2>&1; then
  ok "模型已存在，跳过：$LOCAL"
else
  echo "  下到 $LOCAL"
  if $PY - <<PYEOF
import os, sys
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
try:
    from huggingface_hub import snapshot_download
    snapshot_download("$MODEL_ID", local_dir="$LOCAL",
                      allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
    sys.exit(0)
except Exception as e:
    print("  下载失败：", type(e).__name__, e)
    sys.exit(1)
PYEOF
  then
    ok "模型下载完成"
  else
    warn "HuggingFace 下不动。试 ModelScope（国内快）："
    $PY -m pip install -q modelscope 2>&1 | tail -2
    if $PY -c "
from modelscope import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$LOCAL')
" 2>&1 | tail -3; then
      ok "ModelScope 下载完成"
    else
      die "两个源都下不动。检查网络代理，或手动指定 HF 镜像。"
    fi
  fi
fi

# 记下来，后面 smoke 直接用本地路径（省掉重复解析）
echo "$LOCAL" > "$WORK/models/MODEL_PATH.txt"
ok "模型路径记在 $WORK/models/MODEL_PATH.txt"

# ---------------------------------------------------------------- ⑤ 自检
step "⑤ 自检：代码在卡上是不是好的"
cd "$(dirname "$0")/.." || die "找不到项目目录"
export TAU_BENCH_PATH="${TAU_BENCH_PATH:-$(pwd)/../tau-bench}"
echo "  TAU_BENCH_PATH=$TAU_BENCH_PATH"

# ⚠️ 关键：打包时**刻意没带**本地 tokenizer（它的词表 151,665 和 Qwen2.5 的 151,936
#    不同源，用了会让模型看到乱码而 loss 照样降）。这里指到刚下好的模型自带的 tokenizer，
#    所有自测脚本就会自动用对的那份。
export TOOLHORIZON_TOKENIZER="$LOCAL"
echo "  TOOLHORIZON_TOKENIZER=$TOOLHORIZON_TOKENIZER"
if [ ! -d "$TOOLHORIZON_TOKENIZER" ]; then
  die "模型目录不存在：$TOOLHORIZON_TOKENIZER —— 第 ④ 步是不是没成功？"
fi

$PY scripts/eval_train_logic.py >/tmp/h5_logic.log 2>&1 \
  && ok "训练逻辑自测通过（29 条断言）" \
  || { warn "训练逻辑自测没过，日志在 /tmp/h5_logic.log"; tail -20 /tmp/h5_logic.log; }

$PY -m train.trainer selftest >/tmp/h5_trainer.log 2>&1 \
  && ok "训练器自测通过（14 条断言）" \
  || { warn "训练器自测没过，日志在 /tmp/h5_trainer.log"; tail -20 /tmp/h5_trainer.log; }

# ---------------------------------------------------------------- 收尾
echo
echo "================================================================"
echo " ${GRN}初始化完成${NC}"
echo "================================================================"
echo
echo " 下一步（**切到 GPU 模式之后**）跑烟雾测试："
echo
echo "   export TAU_BENCH_PATH=$TAU_BENCH_PATH"
echo "   python scripts/gpu_smoke.py --n 5 --model \"\$(cat $WORK/models/MODEL_PATH.txt)\""
echo
echo " 拿到 reports/gpu_smoke.json 就关机，把它发回来。"
echo
