#!/usr/bin/env bash
# ==============================================================================
# 等臂A（arm_easy_m）跑完 → 接力跑臂B（arm_easy_lata）
#
# 为什么需要这个独立脚本（2026-09-20）：
#   `run_tier.sh` 里的自动选档判据用的是「通过率 ∈ [15%, 50%]」——**尺子拿错了**。
#   那条判据是从 SFT 出口闸门搬来的，问的是"模型够不够格去训"；
#   而臂B 要问的是「**有没有足够信号让 LATA 的重加权有作用**」，
#   那个答案看的是**零方差率**，不是通过率。
#
#   实测（起点 adapter_sft10，31 题 × 8 条）：
#       exp_easy   零方差 0.613  通过率 0.129   ← 零方差降了 19.3pp，通过率几乎没动
#       exp_medium 零方差 1.000  通过率 0.000   ← 248 条一条没对，比旧题集还难
#   按旧判据：两个都不在 [15%,50%] ⇒ OK=0 ⇒ **臂B 被误跳过**。
#   按正确的判据（零方差率明显低于基线 0.806）：**easy 明确达标**。
#
# 用法（在 AutoDL 的 Terminal 里）：
#     tmux new -s armb
#     cd /root/autodl-tmp/ToolHorizon
#     bash scripts/run_arm_b.sh
#     Ctrl+B 然后 D
# ==============================================================================
set -euo pipefail

cd /root/autodl-tmp/ToolHorizon
source scripts/env.sh >/dev/null 2>&1 || true

ROUNDS="${ROUNDS:-10}"
N="${N:-8}"
LIMIT="${LIMIT:-31}"
ADAPTER="${ADAPTER:-models/adapter_sft10}"
SMAX="${SMAX:-16384}"
TIER="${TIER:-easy}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-300}"          # 最多等 5 小时，防死等

ARM_A="arm_${TIER}_m"
ARM_B="arm_${TIER}_lata"

# ---- 预检 ------------------------------------------------------------------
[ -n "${TOOLHORIZON_TOKENIZER:-}" ] || { echo "⛔ TOOLHORIZON_TOKENIZER 没设 —— 先 source scripts/env.sh"; exit 1; }
[ -d "$ADAPTER" ] || { echo "⛔ 起点 adapter 不存在：$ADAPTER"; exit 1; }

# ⭐ 防重复启动：如果 run_tier.sh 其实已经跑过臂B 了，这里就别再跑一遍
if [ -d "runs/$ARM_B" ]; then
    echo "⏭  runs/$ARM_B 已存在 —— **不重复跑**。"
    echo "    如果那是跑到一半失败的残骸，先 \`mv runs/$ARM_B runs/${ARM_B}_old\` 再重来。"
    exit 0
fi

if [ -z "${TMUX:-}" ]; then
    echo "⛔ 不在 tmux 里。先 \`tmux new -s armb\`，再重跑本脚本。"
    [ "${ALLOW_NO_TMUX:-0}" = "1" ] || exit 1
fi

# ---- 等臂A -----------------------------------------------------------------
echo "━━━ 等 $ARM_A 跑完 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
waited=0
until [ -f "runs/$ARM_A/done.json" ]; do
    sleep 60
    waited=$((waited + 1))
    if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
        echo "⛔ 等了 ${MAX_WAIT_MIN} 分钟还没等到 runs/$ARM_A/done.json —— 停下。"
        echo "    臂A 可能是挂了：看 runs/$ARM_A/logs/ 最后一份 train_*.log。"
        exit 1
    fi
    if [ $((waited % 15)) -eq 0 ]; then
        n=$(ls runs/$ARM_A/ckpt 2>/dev/null | wc -l)
        echo "   …已等 ${waited} 分钟 ｜ $ARM_A 当前 $n 轮"
    fi
done
echo "✅ $ARM_A 已完成（done.json 出现）"

# ⭐ 等完之后**再查一次** —— 本脚本启动时查过一次，但那是在臂A 跑完之前。
#    万一 run_tier.sh 那次其实判了 OK=1（它已经在跑臂B），这里就会撞车：
#    train.run 用 mkdir(exist_ok=True)，**两个进程会同时往同一个 runs/ 目录写、不报错**。
if [ -d "runs/$ARM_B" ]; then
    echo "⏭  runs/$ARM_B 在等待期间出现了 —— 说明 run_tier.sh 自己已经在跑臂B。"
    echo "    本脚本**退出，不重复启动**。"
    exit 0
fi

echo "   开始臂B"
echo "   臂A 的零方差率曲线：$(python -c "
import json;d=json.load(open('runs/$ARM_A/done.json'));print([round(z,3) for z in d.get('zero_var_history',[])])" 2>/dev/null || echo '（读不到）')"

# ---- 臂B -------------------------------------------------------------------
echo
echo "=============================================================================="
echo "▶ 臂B $ARM_B ｜ select=exp_${TIER} ｜ length_norm=lata(÷√L) ｜ ds=0 ｜ $ROUNDS 轮"
echo "  与臂A 唯一差别 = ÷L vs ÷√L（同题/同起点/同轮数）⇒ 干净对照"
echo "=============================================================================="
python -m train.run \
    --rounds "$ROUNDS" --select "exp_${TIER}" --limit "$LIMIT" --n "$N" \
    --model "$TOOLHORIZON_TOKENIZER" --init-adapter "$ADAPTER" \
    --max-seq-tokens "$SMAX" --length-norm lata \
    --run-name "$ARM_B"

echo
echo "=============================================================================="
echo "✅ 臂B 跑完。取数（纯 CPU，之后可以关机）："
echo "   A=$ARM_A B=$ARM_B"
echo "   python -m json.tool runs/\$B/rollouts/step_009.summary.json | head -32"
echo "   然后把 runs/\$A 和 runs/\$B 的 step_*.summary.json 打包发回。"
echo "=============================================================================="
