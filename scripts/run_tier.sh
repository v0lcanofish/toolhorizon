#!/usr/bin/env bash
# ==============================================================================
# 难度档实验 · **一次启动跑完全程**（标定 → 自动选档 → 接力跑臂）
#
#   ① 标定 exp_easy     纯采样，不训练（248 条，约 10-20 分钟）
#   ② 标定 exp_medium   纯采样，不训练
#   ③ 自动选档：通过率**落在 [15%, 50%] 的那一档**（项目原定的 SFT 出口判据）
#   ④ 臂A  arm_<档>_m       length_norm=mean  ds=0  ← 与 arm_vanilla 逐字相同，只换 --select
#   ⑤ 臂B  arm_<档>_lata    length_norm=lata  ds=0  ← **仅当 ③ 选中的档达标才跑**
#
# 为什么要有 ①②③：`exp_medium`（2 个动作）跟现在的 train 档（n_write 均值 1.7）
#   几乎一样难，`exp_easy`（1 个动作）也未必就简单 —— **没有先验理由认为换档一定更好**，
#   所以先花 30 分钟量一下，再决定 5 小时的臂跑在哪一档上。
#
# 为什么 ⑤ 有条件：现在这批题 96% 全错 ⇒ advantage 几乎恒为 0 ⇒ LATA 的重加权
#   几乎没有东西可作用。**只有换到有信号的档，⑤ 才回答得了问题**；档没选对就跑 ⑤
#   是白烧 3.5 小时。
#
# 用法（在 AutoDL 的 Terminal 里）：
#     tmux new -s tier
#     cd /root/autodl-tmp/ToolHorizon
#     bash scripts/run_tier.sh
#     Ctrl+B 然后 D        ← 挂起，关浏览器不影响
#
# 覆盖默认值：
#     ROUNDS=10 LO=0.15 HI=0.50 bash scripts/run_tier.sh
#
# ⚠️ 本脚本在任何一个阶段失败时**停下来** —— 失败多半是配置问题，
#    继续跑只会把同样的错再烧几小时。
# ==============================================================================
set -euo pipefail

cd /root/autodl-tmp/ToolHorizon

# ---- 可调参数 --------------------------------------------------------------
ROUNDS="${ROUNDS:-10}"
N="${N:-8}"
LIMIT="${LIMIT:-31}"                   # 每档取前 N 道（和已有臂同为 31 道 × 8 条 = 248）
ADAPTER="${ADAPTER:-models/adapter_sft10}"
SMAX="${SMAX:-16384}"
LO="${LO:-0.15}"                       # 目标通过率下界
HI="${HI:-0.50}"                       # 目标通过率上界
# ---------------------------------------------------------------------------

echo "=============================================================================="
echo "难度档实验 ｜ 标定 → 选档 → 接力 ｜ 每臂 $ROUNDS 轮 × $LIMIT 题 × $N 条"
echo "起点 adapter：$ADAPTER"
echo "=============================================================================="

# ---- 预检 ------------------------------------------------------------------
echo "[预检]"
source scripts/env.sh >/dev/null 2>&1 || true

fail=0
[ -n "${TOOLHORIZON_TOKENIZER:-}" ] || { echo "   ⛔ TOOLHORIZON_TOKENIZER 没设 —— 先 source scripts/env.sh"; fail=1; }
[ -d "$ADAPTER" ] || { echo "   ⛔ 起点 adapter 不存在：$ADAPTER（必须与四臂同起点，错了整张表作废）"; fail=1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
    || { echo "   ⛔ 看不到 GPU —— 是不是用了「无卡模式」开机？"; fail=1; }
[ "$fail" = 0 ] || { echo "[预检] 未通过，**没有开始**。"; exit 1; }
echo "   ✓ 环境 / adapter / GPU 就绪"
echo "   ✓ GPU：$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

if [ -z "${TMUX:-}" ]; then
    echo
    echo "   ⛔ 不在 tmux 里。**关浏览器 = 训练被杀 = 前面烧的卡全白费。**"
    echo "      先 \`tmux new -s tier\`，再在这个会话里重跑本脚本。"
    [ "${ALLOW_NO_TMUX:-0}" = "1" ] || exit 1
    echo "      （ALLOW_NO_TMUX=1 已设，继续 —— 后果自负）"
fi

# ---- 标定 ------------------------------------------------------------------
calib () {
    local tier="$1"
    local d="runs/calib_${tier}"
    if [ -f "$d/rollouts/step_000.summary.json" ]; then
        echo "   ⏭  已有 $d 的标定结果，跳过采样（要重测先删掉这个目录）"
        return 0
    fi
    mkdir -p "$d/rollouts"
    echo "   ▶ 标定 exp_${tier}（纯采样，不训练）…"
    python -m train.collect \
        --model "$TOOLHORIZON_TOKENIZER" --adapter "$ADAPTER" \
        --select "exp_${tier}" --limit "$LIMIT" --n "$N" \
        --max-seq-tokens "$SMAX" \
        --out "$d/rollouts/step_000.jsonl" 2>&1 | tee "$d/collect.log"
}

echo
echo "━━━ ① 标定 exp_easy ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
calib easy
echo
echo "━━━ ② 标定 exp_medium ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
calib medium

# ---- 选档 ------------------------------------------------------------------
echo
echo "━━━ ③ 选档 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SEL=$(python - "runs/calib_easy/rollouts/step_000.summary.json" \
             "runs/calib_medium/rollouts/step_000.summary.json" "$LO" "$HI" <<'PY'
import json, sys

def load(p):
    s = json.load(open(p, encoding="utf-8"))
    return s["pass_rate"], s["zero_var_rate"], s.get("n_tasks")

easy = load(sys.argv[1])
medi = load(sys.argv[2])
lo, hi = float(sys.argv[3]), float(sys.argv[4])
mid = (lo + hi) / 2

print(f"   {'档':<10}{'题数':>5}{'通过率':>10}{'零方差率':>10}")
for name, (p, z, n) in (("exp_easy", easy), ("exp_medium", medi)):
    print(f"   {name:<10}{n:>5}{p:>9.1%}{z:>10.1%}")

def dist(p):                      # 到目标区间的距离：落在区间内 = 0
    return 0.0 if lo <= p <= hi else (lo - p if p < lo else p - hi)

cands = [("easy", easy[0]), ("medium", medi[0])]
# 先取"落在区间内"的；都不在就取"离区间最近"的
inwin = [c for c in cands if lo <= c[1] <= hi]
pick = min(inwin, key=lambda c: abs(c[1] - mid)) if inwin \
       else min(cands, key=lambda c: dist(c[1]))
ok = "1" if inwin else "0"

print()
if inwin:
    print(f"   ✅ 选中 **exp_{pick[0]}**（通过率 {pick[1]:.1%} 落在 [{lo:.0%}, {hi:.0%}] 内）")
    print(f"      其余档：{', '.join(f'{n}={p:.1%}' for n, p in cands if n != pick[0])}")
else:
    print(f"   ⚠️ **两个档没有一个落进 [{lo:.0%}, {hi:.0%}]** —— 仍然选最接近的 exp_{pick[0]}"
          f"（通过率 {pick[1]:.1%}）")
    print(f"      含义：换难度档**没有解决**难度校准问题，④ 的结果会是又一次『没信号』。")
    print(f"      ⑤（LATA 臂）**将被跳过** —— 没有信号就没有东西可以重加权。")
print(f"__TIER__={pick[0]}")
print(f"__OK__={ok}")
PY
)
echo "$SEL" | sed 's/^__.*//' | sed '/^$/d'
TIER="$(echo "$SEL" | sed -n 's/^__TIER__=//p')"
OK="$(echo "$SEL" | sed -n 's/^__OK__=//p')"

[ -n "$TIER" ] || { echo "⛔ 选档失败（拿不到 TIER）—— 停下"; exit 1; }
echo "   → TIER=$TIER  OK=$OK"

ARM_A="arm_${TIER}_m"
ARM_B="arm_${TIER}_lata"
for a in "$ARM_A" "$ARM_B"; do
    [ -d "runs/$a" ] && { echo "   ⛔ runs/$a 已存在 —— 要重跑先 \`mv runs/$a runs/${a}_old\`，别直接覆盖"; exit 1; }
done

# ---- 接力跑臂 --------------------------------------------------------------
run_arm () {
    local name="$1" ln="$2"
    echo
    echo "=============================================================================="
    echo "▶ 臂 $name ｜ select=exp_${TIER} ｜ length_norm=$ln ｜ ds=0 ｜ $ROUNDS 轮"
    echo "=============================================================================="
    python -m train.run \
        --rounds "$ROUNDS" --select "exp_${TIER}" --limit "$LIMIT" --n "$N" \
        --model "$TOOLHORIZON_TOKENIZER" --init-adapter "$ADAPTER" \
        --max-seq-tokens "$SMAX" --length-norm "$ln" \
        --run-name "$name"
}

echo
echo "━━━ ④ 臂A（÷L 基线，与 arm_vanilla 同配置，只换 --select）━━━━━━━━━━━━━━━━━━"
run_arm "$ARM_A" mean || { echo "⛔ 臂A 失败 —— 停下，先看 runs/$ARM_A/logs/"; exit 1; }

if [ "$OK" = "1" ]; then
    echo
    echo "━━━ ⑤ 臂B（÷√L / LATA）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    run_arm "$ARM_B" lata || { echo "⛔ 臂B 失败 —— 停下。臂A 的结果已在盘上。"; exit 1; }
else
    echo
    echo "━━━ ⑤ 臂B **已跳过** ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   原因：标定时两档零方差率都太高（没有信号），LATA 的重加权无对象可作用。"
    echo "   这不是失败 —— 它本身就是『难度校准』这个诊断的**阴性结果**，值得写进报告。"
fi

echo
echo "=============================================================================="
echo "✅ 跑完。拿结果的两条命令（纯 CPU，之后可以关机）："
echo "   D=runs/$ARM_A/rollouts"
echo "   python -m json.tool \$D/step_009.summary.json | head -40"
echo
echo "   标定数据（也请一起带回）："
echo "   python -m json.tool runs/calib_easy/rollouts/step_000.summary.json"
echo "   python -m json.tool runs/calib_medium/rollouts/step_000.summary.json"
echo "=============================================================================="
