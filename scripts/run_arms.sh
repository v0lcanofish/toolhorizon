#!/usr/bin/env bash
# ==============================================================================
# 四臂消融 · 三个缺的臂**接力跑**（第 4 臂已经有了，见下面「已有臂」）
#
#   臂①  arm_vanilla     length_norm=mean   ds=0     ← 零代码，最想知道答案的那个
#   臂②  arm_ds          length_norm=mean   ds=K
#   臂④  arm_lata_ds     length_norm=lata   ds=K
#
#   已有臂  runs/vanilla = **+LATA**（`--length-norm lata`）—— 注意目录名叫 vanilla
#          但实际开了 LATA，是命名陷阱。**不要重跑它**，正式对照里它就是臂③。
#
# 用法（在 AutoDL 的 Terminal 里）：
#     tmux new -s arms
#     cd /root/autodl-tmp/ToolHorizon
#     bash scripts/run_arms.sh
#     Ctrl+B 然后 D        ← 挂起，之后关浏览器/关机都不影响
#
# 覆盖默认值（可选）：
#     ROUNDS=15 DSK=4 ADAPTER=models/adapter_sft10 bash scripts/run_arms.sh
#
# ⚠️ 这个脚本**故意**在任何一个臂失败时停下来 —— 失败多半是配置问题，
#    继续跑只会把同样的错再烧 8 小时。
# ==============================================================================
set -euo pipefail

cd /root/autodl-tmp/ToolHorizon

# ---- 可调参数 --------------------------------------------------------------
ROUNDS="${ROUNDS:-15}"                 # 15 轮：稳态从 step 7 就成型，够看
N="${N:-8}"
SELECT="${SELECT:-train}"              # = 31 道 S 类题（= E6 的 arm_none 臂）
ADAPTER="${ADAPTER:-models/adapter_sft10}"
SMAX="${SMAX:-16384}"
DSK="${DSK:-4}"                        # DS 目标组数
DSMAXPASS="${DSMAXPASS:-6}"
# ---------------------------------------------------------------------------

echo "=============================================================================="
echo "四臂消融 · 三臂接力 ｜ 每臂 $ROUNDS 轮 × $N 条 × ${SELECT}"
echo "起点 adapter：$ADAPTER"
echo "=============================================================================="

# ---- 上卡前检查：任何一条不过就**别开始**（防跑一半才发现） -----------------
echo "[预检]"
source scripts/env.sh >/dev/null 2>&1 || true

fail=0
[ -n "${TOOLHORIZON_TOKENIZER:-}" ] || { echo "   ⛔ TOOLHORIZON_TOKENIZER 没设 —— 环境变量不跨开关机，先 source scripts/env.sh"; fail=1; }
[ -d "$ADAPTER" ] || { echo "   ⛔ 起点 adapter 不存在：$ADAPTER（四个臂必须同一个起点，错了整张表作废）"; fail=1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
    || { echo "   ⛔ 看不到 GPU —— 是不是用了「无卡模式」开机？"; fail=1; }
[ -d "runs/vanilla" ] || echo "   ⚠️ runs/vanilla（臂③ +LATA）不在，四臂对照会缺一个"
for a in arm_vanilla arm_ds arm_lata_ds; do
    [ -d "runs/$a" ] && { echo "   ⛔ runs/$a 已存在 —— 要重跑先 \`mv runs/$a runs/${a}_old\`，别直接覆盖"; fail=1; }
done
[ "$fail" = 0 ] || { echo "[预检] 未通过，**没有开始**。"; exit 1; }
echo "   ✓ 环境 / adapter / GPU / 目录 都就绪"
echo "   ✓ GPU：$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
if [ -z "${TMUX:-}" ]; then
    echo
    echo "   ⛔ 不在 tmux 里。**关浏览器 = 训练被杀 = 前面烧的卡全白费。**"
    echo "      先 \`tmux new -s arms\`，再在这个会话里重跑本脚本。"
    [ "${ALLOW_NO_TMUX:-0}" = "1" ] || exit 1
    echo "      （ALLOW_NO_TMUX=1 已设，继续 —— 后果自负）"
fi

echo "[预估] ⚠️ summary.json 里的 \`seconds\` **只是采样时间、不含训练** ——"
echo "       拿它当「每轮耗时」会低估约 1/3。真实墙钟：9/18 那轮 25 轮 ≈ 13 GPU 时。"
echo "       故 ${ROUNDS} 轮 ≈ 6.5 小时/臂（DS 臂因加采会更久）。"
echo "       判据：\`ls -lt runs/<臂>/ckpt/\` 相邻两行的时间差 = 每轮真实墙钟耗时。"
echo

# ---- 单臂 ------------------------------------------------------------------
run_arm () {
    local name="$1" ln="$2" ds="$3"
    echo
    echo "=============================================================================="
    echo "▶ 臂 $name ｜ length_norm=$ln ｜ ds=$ds"
    echo "=============================================================================="
    local extra=()
    if [ "$ds" != "0" ]; then
        extra=(--ds "$ds" --ds-max-pass "$DSMAXPASS")
    fi
    python -m train.run \
        --rounds "$ROUNDS" --select "$SELECT" --n "$N" \
        --model "$TOOLHORIZON_TOKENIZER" --init-adapter "$ADAPTER" \
        --max-seq-tokens "$SMAX" --length-norm "$ln" \
        --run-name "$name" "${extra[@]}"
}

# ---- 接力 ------------------------------------------------------------------
run_arm arm_vanilla    mean 0            || { echo "⛔ 臂① arm_vanilla 失败 —— 停下，先看 runs/arm_vanilla/logs/"; exit 1; }
run_arm arm_ds         mean "$DSK"       || { echo "⛔ 臂② arm_ds 失败 —— 停下。臂①的结果是好的，可在 tmux 里单独补跑 ②④。"; exit 1; }
run_arm arm_lata_ds    lata "$DSK"       || { echo "⛔ 臂④ arm_lata_ds 失败 —— 停下。①② 已在盘上。"; exit 1; }

echo
echo "=============================================================================="
echo "✅ 三个臂全部跑完。四臂齐了："
echo "   ① runs/arm_vanilla     vanilla"
echo "   ② runs/arm_ds          +dynamic sampling"
echo "   ③ runs/vanilla         +LATA        （已有，注意目录名不叫 arm_lata）"
echo "   ④ runs/arm_lata_ds     +LATA+DS"
echo
echo "下一步（纯 CPU，不用再动卡）："
echo "   for d in arm_vanilla arm_ds vanilla arm_lata_ds; do \\"
echo "       echo \"== \$d\"; ls runs/\$d/rollouts/*.summary.json | head -1 | xargs cat; done"
echo "   然后把四份 summary 贴回来出对照表。"
echo "=============================================================================="
