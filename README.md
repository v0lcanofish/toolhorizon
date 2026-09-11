# ToolHorizon · 长链路多工具智能体 RL 稳定性研究

> 建立 2026-09-11 ｜ 取代原 A（GRPO-Tamer）+ B（When2Search）双项目结构
> 参考天花板：`agentic-grpo-longhorizon`（已本地 clone，2×A800 / 7B / 72B 模拟器，**只读**）
> **当前状态：设计定稿，Stage 0 未开工**（按双轨排期，9/19 起全速）

---

## 一句话

> **longhorizon 证明了 2×A800 上需要 PRM-Lite + LATA 才能救回来的坍塌，在 1/10 算力的小模型上是什么形态？**
> 我把它的结论降级复现，并补上它没看的第四维——**工具行为本身的退化**。

---

## 定位：不做第二个 longhorizon

### 资源对比（全部设计的出发点）

| | longhorizon | 本项目 |
|---|---|---|
| 显存 | 2×A800 = **160 GB** | 1×4090 = **24 GB**（且从未验证） |
| 模型 | 7B policy + 72B-AWQ 模拟器 | 1.5B + **规则模拟器（零 LLM）** |
| 工具 | 14 个 | **7 个** |
| 轨迹 | 标称 16K | S_max 8192（**实测成功轨迹 p50 仅 2.5K**） |
| 时间 | 4 个月 | **6 周**（每天 4h，晚上） |
| 起点 | 有 GPU + veRL 经验 | **0 行 RL 代码** |

**差约 10 倍算力、1/3 时间、更低起点 → 照抄必死。**

### 三个不可动摇的判断

1. **不照抄**。它做的是"2×A800 上坍塌怎么修"，我们做"这些结论降级后还成立吗"
2. **它的资源门槛本身就是缝隙**。劣势（没算力）正好是研究变量
3. **它漏了第四维**。它诊断的三大病因（group reward 饱和 / 训练集泄漏 / 长推理能力下降）**都不含工具使用行为**——14 个工具白摆着，没当观察对象

---

## 创新点

| | 创新点 | 一句话 | 为什么没人做 |
|---|---|---|---|
| **A（主）** | **工具使用退化谱** | 欠调用（学会猜）/ 过调用（搜索成瘾）/ 抖动（工具间横跳）——定义 + 量化 + 前兆指标 | longhorizon 有 14 个工具却没分析调用分布；EMTIR-GRPO 只做奖励修正不做诊断 |
| **B（附）** | **坍塌模式的规模迁移** | 它三大病因在 1.5B 上是否成立？哪条增强、哪条消失、有没有新的 | 它只做 7B，**无任何小模型对照** |
| C（可选） | 坍塌的阶段迁移 | SFT→GRPO→课程重标定，坍塌第一次出现在哪 | 时间不一定放得下 |

---

## 场景：τ-bench Airline（本地数据全有，零新下载）

### 已验证的本地资产（实测，非估算）

| 资产 | 路径（相对 `reference-repos/agentic-grpo-longhorizon/`） | 实测 |
|---|---|---|
| 航班/订单/用户库 | `tau-bench/tau_bench/envs/airline/data/{flights,reservations,users}.json` | 2.1 / 2.4 / 0.6 MB |
| **任务定义** | `.../airline/tasks.py` | **50 题，每题带 instruction + gold action 序列** |
| 环境状态机 | `.../airline/env.py` + `rules.py` + `wiki.md` | 纯 Python，确定性 |
| **真实轨迹** | `tau-bench/historical_trajectories/*.json` | **51 MB，airline 成功 268 条**（gpt-4o 84 + sonnet 184） |
| 消融报告 | `agentic-grpo-longhorizon/docs/ablation/` | 4 臂完整指标 |

**实测轨迹长度**（gpt-4o-airline 成功轨迹，脚本实测）：
```
p50 ≈ 2,512 token ｜ p90 ≈ 4,009 ｜ max ≈ 4,464
单条 24 消息：system 1 / user 6 / assistant 11 / tool 6
```
> ⚠️ 它标称 16K，**真实成功轨迹中位数只有 2.5K** → 显存需求比标称小一个量级。

### 工具 14 → 7

**保留**：`get_user_details`、`get_reservation_details`、`search_direct_flight`、`search_onestop_flight`、`update_reservation_flights`、`update_reservation_baggages`、`calculate`

**砍掉**：`book/cancel/send_certificate/transfer/list_airports/think`

> 完整 tool schema ≈ 13,528 字符 ≈ **3,400 token，每一轮都付**。对 1.5B 的 8K context 是 40% 固定税。

### 用户模拟器：规则式，零 LLM

`tau_bench/envs/user.py` 的 `BaseUserSimulationEnv` 是抽象类，只需实现 `reset(instruction)` / `step(content)`。

- **做法**：`tasks.py` 的 instruction 用一次性 DeepSeek 调用解析成 slot dict 缓存 JSON；运行时关键词匹配返回 slot；匹配不到返回固定兜底句
- **降级 1**：直接**回放** `historical_trajectories` 的 user 轮（268 条真实成功轨迹，免费）
- **降级 2**：难度降到 easy 档
- ⚠️ **必踩的坑**：`tau_bench/envs/user.py` 模块顶层 `from litellm import completion`，本地没装 → 改成 lazy import

> **这不只是工程妥协，是研究增量**：longhorizon 自己的诊断文档写"user simulator 是 eval 噪声的主要来源"。换规则模拟器后，**reward 方差 100% 来自 policy**，归因链干净得多。

---

## 技术架构

### 模型配置（推荐）

```
Qwen2.5-1.5B-Instruct + LoRA r=16 / α=32 (bf16 base)
S_max = 8192 (prompt 4096 + response 4096)
micro-batch 4 ｜ group n=8 ｜ train_batch 4 prompt ｜ 梯度累积 2
峰值 ≈ 11.3 GB 训练 / 12.4 GB rollout  →  留 2 倍余量
```

> **不用 QLoRA**：NF4 与 vLLM 的 bf16 数值不一致 → PPO ratio 是垃圾。

### 显存账（胜负手）

| | 全参 fp32 AdamW | LoRA bf16 |
|---|---|---|
| 0.5B | 6.93 GB | 2.10 GB |
| **1.5B** | **19.46 GB** ⚠️ 起床剩 3GB | **4.30 GB** ✅ |

LoRA 把优化器+梯度从 18.5 GB 压到 0.22 GB（**84 倍**）。

**四个内存刺客**（按危险度）：
1. **logits 张量**：S=8192 × vocab **151936** × fp32 = **4.98 GB 单个张量** → **必须用 fused/chunked CE**
2. **`logits_to_keep` 缺失**：轨迹只有 ~10% 是 assistant token，不做 mask-only logits 就是 10 倍浪费
3. **PEFT + grad ckpt 的 `enable_input_require_grads()`**：报错后手贱关 grad ckpt → 激活 ×3–5
4. **被 mask 的 token 照样吃激活**：tool observation 占 ~51% token，不过 loss 但必须过 forward/backward

### 分时复用（单卡核心决策）

```
vLLM 起服务 → 收满 rollout 批次 → kill → 训练 → 存 adapter(37MB) → 重启 vLLM --lora-modules
```
峰值 = max(rollout 12.4, train 11.3) = **12.4 GB**，省约 40%。

- **代价**：vLLM init 20–40s → 大批量收数据摊薄到 <5%
- **代价 2**：轻微 off-policy → **本身是一个消融轴**
- ❌ **排除**同进程 colocate（要 NCCL/shm + FSDP + ContextVar 隔离）——这正是"不吹工程深度"要避开的

### 成本

**全项目 GPU < ¥300**（60–90 GPU 小时 × ¥1.5–2.5/h，含 3 倍调试余量）

---

## 三阶段管线

### Stage 0 · 纯 CPU（~40% 工作量，**租卡前就能做完**）★

- 规则式用户模拟器 `RuleBasedUserSim`
- **扩题器**：采样 (user, reservation) → 模板化 instruction + gold action → **500–2000 个难度分级任务**（修掉 longhorizon"只有 50 题、train/eval 重合 80%"的死结）
- **Gold 轨迹合成**：在 env 里直接执行 `tasks.py` 的 actions → **零模型零 API 的无限完美 SFT 数据**
- 奖励函数 + 评测 harness（pass@1、**组内零方差率**、avg_turns、per-turn 长度）
- **Mock-policy 自测**：照搬 `Yu-Agentic-RL/scripts/grpo/search_agent.py --mock`
- **精确 token 统计**：只下 Qwen tokenizer（~11MB，不需权重）纯 CPU 跑，**租卡前验证显存账**

**出口判据**：`eval.py --mock` 全绿 ｜ gold 轨迹 oracle reward=1.0 ｜ token 长度分布实测出来

### Stage 1 · SFT 预热（GPU ≈ 1.5h）

- 输入：Stage 0 合成 gold 轨迹 2–5K 条，**Hermes 格式（`<tool_call>` 标签，不用 JSON）**
- **为什么必须有**：1.5B base 在 airline 上 pass rate 大概率≈0 → 组内全 0 → std=0 → advantage=0 → **一步学不动**
- 配置：LoRA r=16，lr 1e-4，3 epoch，峰值 ≈ 8 GB
- **硬判据**：medium 档 pass@1 ∈ **[15%, 50%]**；**< 5% → 立刻降级 easy 档，不要头铁**

### Stage 2 · GRPO 主训练（GPU ≈ 15h × 4 臂）

- 输入：SFT adapter + medium 档 500 题（400 train / 100 held-out）
- 配置：n=8，lr 5e-6，KL coef 0，S_max=8192
- **现成资产**：`Yu-Agentic-RL/scripts/grpo/grpo_update.py` 里 **outcome-variance gating（= DAPO dynamic sampling）和 LATA（√L 归一化）已实现**，`compute_advantages()` 有 `gate=True` 开关
- **消融 4 臂**：① vanilla ② +dynamic sampling ③ +LATA ④ +两者
- 耗时：~3–5 min/step，250 步 ≈ 12–20 GPU 小时

### Stage 3 · 课程重标定（GPU ≈ 6h）

用 Stage 2 最优 checkpoint 跑 hard 档，筛出 pass rate 落在 **20–80% 的"当前能力边界"任务**，组成新训练集再训一轮。

> 把"数据难度分布"从固定超参变成**可主动控制的实验变量**。产出**能力边界漂移曲线**。

**不做**：训真 PRM（rule-based PRM-Lite 在 longhorizon 已证明失败，error_rate 反而最高 0.365）。

---

## 实验矩阵

| 实验 | 目的 | 产出 |
|---|---|---|
| **E1 复现验证** | longhorizon 三大病因在 1.5B 上成立吗 | 三病因逐条成立/不成立 + 差异说明 |
| **E2 工具退化谱** ⭐ | 工具调用分布随训练怎么漂移 | **欠调用/过调用/抖动三条曲线 + 前兆指标** |
| **E3 消融 4 臂** | dynamic sampling / LATA 各自贡献 | 消融对比表 |
| **E4 规模对比** | 1.5B vs longhorizon 7B 的坍塌模式差异 | 对照表（引用其公开数字） |
| **E5 课程重标定** | 难度分布是不是坍塌主因 | 能力边界漂移曲线 |

**评测口径**（抄 longhorizon 的 pass@k）：N=4 采样，区分 **covered / uncovered / unseen** → 识别训练集泄漏。
扩题后 train/test **真正不相交**，泛化指标 n 从它的 34 涨到 200+（**对它的直接改进**）。

---

## 里程碑

| 周 | 时间 | 内容 | 出口 |
|---|---|---|---|
| **W0** | 9/11–9/18 | ⚠️ **白天面试优先**，晚上只做设计确认 + Stage 0 起步 | 设计文档定稿 ✅ |
| **W1** | 9/19–9/25 | **Stage 0 全部（CPU）** + 首次 AutoDL smoke test | mock 全绿 + token 分布实测 |
| **W2** | 9/26–10/2 | Stage 1 SFT + Stage 2 首跑（vanilla 臂） | 第一条 reward 曲线 |
| **W3** | 10/3–10/9 | Stage 2 消融 4 臂 | 消融对比表 |
| **W4** | 10/10–10/16 | E1 复现验证 + E2 工具退化谱 | 三病因结论 + 退化曲线 |
| **W5** | 10/17–10/23 | Stage 3 课程重标定 + E4 规模对比 | 漂移曲线 |
| **W6** | 10/24–10/31 | 收口：README + 技术博客 + 简历条目 | **10/31 硬边界交付** |

---

## 风险与降级链

| 风险 | 概率 | 对策 |
|---|---|---|
| **R1 不是坍塌，是"没信号"** | **~50%** | **"组内零方差率"当第一监控指标**，>80% 说明题太难；扩题器出难度分档；SFT 出口卡 15–50% |
| R2 工具调用格式崩 | ~30% | Hermes 标签而非 JSON；rollout 上 guided decoding；把"格式约束是否掩盖格式坍塌"作为消融轴 |
| R3 规则模拟器不够用 | ~25% | 降级链：规则 → **回放 268 条真实轨迹** → easy 档 |
| R4 GPU 环境跑不起来 | ~20% | 首次租卡只做 **30 分钟 smoke test**；确认 `nvidia-smi` 满 24GB（防 4090D / 虚拟化切分） |

**降级阶梯（写死）**：
```
规则模拟器 → 回放模拟器 → easy 档任务
medium 档 → easy 档
4 臂消融 → 2 臂（vanilla + 全开）
S_max 8192 → 4096
1.5B → 0.5B 全参
```

**显存打脸点**：① 没上 fused/chunked CE → 4.98 GB 单张量必 OOM（1 号杀手）② 忘 `logits_to_keep` ③ vocab 是 151936 不是 128256 ④ vLLM 进程没真死 → 训练启动 OOM 且 `nvidia-smi` 看着是空的 ⑤ 弱策略循环 → 轨迹膨胀到 8–12K

**单卡多轮 rollout 真障碍**：
- 进程切换固定成本（vLLM init 20–40s）→ 大批量收数据摊薄
- **长尾轨迹拖死整批**（straggler）→ per-trajectory 硬超时 + **截断轨迹给 0 奖励**（不给部分奖励，否则奖励"啰嗦到被截断"）
- 权重同步 → 存 adapter(37MB) + 重启 vLLM `--lora-modules`，约 5 秒

---

## ⛔ 红线：不能吹的东西

以下 longhorizon 的工程深度，**我们不做、也不写进简历**：
- ContextVar 并发隔离（async multi-turn rollout 的 env 状态共享）
- `bypass_mode` + fused CE 消除 FSDP log_prob 重算的 20–40GB 峰值
- Render-Twice-Diff 多轮 loss mask 方案
- 异步训练 pipeline / vLLM V1 适配
- 2×A800 / 7B / 72B-AWQ simulator

**简历上必须诚实区分**：「复现自 longhorizon」的部分 vs「我的增量」的部分。

**陷阱清单**（看着美好但做不出来）：
- **直接上 veRL**（单卡要 FSDP + colocate；**读它用于面试，不要跑它**）
- **用 1.5B 当 LLM 用户模拟器**（退化成不连贯文本，把混杂因子请回来）
- **离线 RL / 历史轨迹当 GRPO rollout**（GRPO 的 group 必须来自**当前策略**采样；回放只能用于 SFT 和模拟器）
- **14 个工具全保留**（schema 3400 token × 每轮）
- **tool 返回原始 JSON**（占轨迹 51%，裁剪成必需字段可砍 40% 的 S）

---

## 验证方法

**Stage 0 出口（全 CPU，租卡前）**：
```bash
cd 代码库/projects/ToolHorizon
D:/anaconda/python.exe eval.py --mock          # 期望全绿
D:/anaconda/python.exe synth_gold.py --n 100   # 期望 oracle reward=1.0
D:/anaconda/python.exe token_stats.py          # 期望 p50≈2500, p90≈4000, max≈4500
```

**GPU smoke test（首次租卡，30 分钟内）**：
```bash
nvidia-smi                                      # 确认满 24GB
python -c "import vllm; ..."                    # 加载 1.5B
python rollout_smoke.py --n 5                   # 5 条 rollout 跑通
```

**Stage 1 出口**：medium 档 pass@1 ∈ [15%, 50%]
**Stage 2 出口**：4 臂消融曲线 + **组内零方差率 < 80%**
**最终验收**：E1–E5 五张表/图 + README + 技术博客

---

## 目录规划（开工后建）

```
ToolHorizon/
├── README.md              ← 本文件（权威设计）
├── env/                   ← τ-bench 包装 + 规则用户模拟器
├── data/                  ← 扩题器产物 + gold 轨迹
├── reward/                ← 奖励函数 + 评测 harness
├── train/                 ← SFT + GRPO（复用 Yu-Agentic-RL 的 grpo_update.py）
├── observe/               ← 四指标观测器 + 工具退化指标 ⭐
├── scripts/               ← mock 自测 / token 统计 / smoke test
└── reports/               ← 实验图表
```

---

## 相关文件

- `reference-repos/agentic-grpo-longhorizon/` — 参考天花板（τ-bench + 消融报告，**只读**）
- `reference-repos/Yu-Agentic-RL/scripts/grpo/grpo_update.py` — GRPO 主体（gating + LATA 已实现）
- `reference-repos/Yu-Agentic-RL/scripts/grpo/search_agent.py` — 多轮 episode 循环 + `--mock` 模板
- `projects/GRPO-Tamer/` — 原项目 A（**已归档**，仅保留数据笔记）
- `docs/主作品增量方案.md` — 原增量方案（**已归档**）
