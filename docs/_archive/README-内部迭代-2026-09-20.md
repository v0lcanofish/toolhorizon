# ToolHorizon · 长链路多工具智能体 RL 稳定性研究

> 建立 2026-09-11 ｜ 取代原 A（GRPO-Tamer）+ B（When2Search）双项目结构
> 参考天花板：`agentic-grpo-longhorizon`（已本地 clone，2×A800 / 7B / 72B 模拟器，**只读**）
> **当前状态：Stage 0 进行中（2026-09-15 开工），v4 修订见下**

---

## 🔴 v5 修订（2026-09-16 晚）：块 H3 mock 自测，抓到一个会毁掉训练的 bug

> 脚本 `scripts/eval_mock.py`（零模型 / 零 GPU）｜ 新增 `env/rollout.py`（多轮循环）

**`eval.py --mock` 全绿**：
```
① 多轮循环    50 道跑通 ｜ 消息序列合法 50/50 ｜ 字段与 SFT 数据一致 50/50
② reward 判据 照本宣科 50/50 满分 ｜ 训练集不做 31/31 得 0 分 ｜ 探针集不做 19/19 满分
③ 零方差机制 全对 → std=0 → advantage 全 0（零梯度）✅
```

### 🔥 最重要的：`calculate_reward()` **不是幂等的**

```
第 1 次 calculate_reward() = 0.0   ← 对
第 2 次 calculate_reward() = 1.0   ← 错！变成假满分
第 3 次 calculate_reward() = 1.0
```

**机制**：它内部会 `self.data = 重新加载数据()` 然后**重放 gold 动作**算 `gt_data_hash`。
调用结束后 **`env.data` 被留在「gold 已执行过」的状态** ——
于是第二次调用时，"当前哈希"已经是 gold 状态，两边相等 → **reward 恒为 1.0**。

> ⚠️ **这是本项目最危险的一个坑**：训练循环里只要不小心调两次
> （比如 `step()` 返回 `done` 时内部已经算过一次），
> **所有题都会变成满分 → 组内全对 → `std=0` → advantage 全 0 → 训练完全无效**，
> **而且 loss 曲线上完全看不出来。**

**已固化**：`env/tau_env.py` 的 `reward_of(env, resp)` ——
`done` 时用 `step()` 返回的 `resp.reward`（内部刚算过，是第一次）；
只有撞 `max_turns` 截断时才自己调一次。

### 第二个坑：假策略的收尾话必须带 `outputs`

第一版 mock 只过 46/50 —— 4 道带 `outputs` 的题失败，
因为假策略说的是固定的 `"All set. Anything else I can help with?"`。

**这正是 v4 里 H1 发现的硬约束**（收尾必须嵌入关键信息，环境做字符串包含检查）——
**换了个场景又撞上一次。**

### 新增资产

| 文件 | 作用 |
|---|---|
| `env/rollout.py` | **多轮 episode 循环**（训练/评测共用的采样入口） |
| `env/tau_env.py::reward_of` | 安全的 reward 取值（绕开不幂等） |
| `data/system_prompt.txt` | 系统提示（取自真实轨迹，6155 字符，含全部业务规则） |
| `scripts/eval_mock.py` | mock 自测（Stage 0 出口判据） |

---

## 🔴 v4 修订（2026-09-16）：信号形状四路探针，修正 v3 的一处判断

> 脚本 `scripts/probe_signal_shape.py`（零模型/零 GPU/可复现）｜ 报告 `reports/信号形状探针-2026-09-16.md`

**四路探针**：`noop`（什么都不做）/ `gold`（重放）/ `readonly`（只做只读）/ **`foreign`（注入别题的 gold 写动作 = 模拟过调用）**

| 类别 | 判定 | 题数 | 占比 |
|---|---|---:|---:|
| **S 有区分度** | `noop=0` 且 `gold=1` | **31** | 62.0% |
| **O 单边信号** | `noop=1` 且 `foreign=0` | **19** | 38.0% |
| **N 真·无信号** | `noop=1` 且 `foreign=1` | **0** | **0.0%** |

### ⚠️ 修正 v3 的两处表述

| v3 写的 | 实测 |
|---|---|
| "**20 道**（40%）的题 gold 改不动数据库" | **19 道（38%）**（v3 自己在下一行也写了 noop 通过 19 道，两处口径没对齐） |
| "reward 与策略行为**无关**、恒为 1.0" | ❌ **错**。注入别题 gold 写动作后，**19 道全部掉到 0.0** |

> ✅ **正确表述**：这 19 道题的 reward **与策略行为有关，但是单向的**——
> **只惩罚"乱做"，不奖励任何"正确行为"**。因为题目本就不需要动作，
> 不存在"做对了"这个状态，只存在"做错了"。
>
> ⭐ **没有一道题是真·无信号。**

### ⭐⭐ 由此得出的「欠调用」机制

```
S 类（31 道，62%）：做对了 → +1 ；不做 → 0     →  教"该动就动"（难、稀疏）
O 类（19 道，38%）：不做   → +1 ；乱做 → 0     →  教"别乱动"（易、密集）
```

**对弱策略，惩罚信号来得比奖励信号早、比奖励信号密 → 模型变保守 → 欠调用。**

> 不是"没信号所以不学"，是**"有信号，但只有惩罚方向"**。

### 新实验 D：S/O 配比 → 退化方向 ⭐

只改 S:O 配比，三臂对照（31+19 / 31+0 / 31+38），观测工具调用数漂移。
**把"欠调用"从"模型学坏了"变成"数据配比导致的、可预测、可控的现象"。**

### ⚠️ 附带修正：轨迹 token 长度被低估 2.3 倍

| 9/15 写的 | 2026-09-16 重测（n=268，**含工具 schema**） |
|---|---|
| p50 ≈ 2,512 token | **p50 = 6,692 ｜ p90 = 10,287 ｜ max = 23,936** |

**根因**：9/15 只算了消息本身，**漏掉工具 schema 的固定税 3,249 token**。
后果：**p90 = 10,287 > S_max 8192 → 必须做截断**（原以为"比标称小一个量级"）。
另：**tool 返回占 65.7%**、assistant token 只占 **12%** → 裁剪 tool 字段比调 S_max 更划算。
详见 `reports/token分布实测-2026-09-16.md`。

### 附加发现：第四种信号形状

`task[44]`（`n_write=0` 但 `noop=0`）：正解是**不改 DB 但必须汇报**（`outputs=["4"]`）。
**「纯沟通型任务」在 τ-bench 里只有 1 道（2%）** → 扩题器不生成 `outputs` 会把它稀释到 0。

---

## 🔴 v3 修订（2026-09-15 晚）：Stage 0 第一批实测，推翻 v1/v2 的四处设计

> 全部数字本地跑出、脚本可复现 ｜ 完整报告：`reports/Stage0-实测发现-2026-09-15.md`

| # | 原文 | 实测后 |
|---|---|---|
| 1 | 「工具 **14 → 7**」 | ❌ **作废**，改 **12 个**（14 − `think` − `list_all_airports`）。README 方案会让有区分度的题从 30 掉到 **14**（废题率 40%→72%） |
| 2 | 「gold 轨迹合成：重放 actions → 零成本无限量 SFT 数据」 | ❌ **不成立**。gold 里 `respond` 出现 **0 次**，重放出来的轨迹没有收尾回复，不可训练 |
| 3 | 「tool schema ≈ 13,528 字符 ≈ 3,400 token」 | 实测 **9,144 字符 ≈ 2,345 token**，原数字高估 **48%** |
| 4 | 「τ-bench airline **50 题**」 | 实测**有效题只有 30 道**。`20/50` 的题 gold 改不动数据库 |

### ⭐ 最重要的一条：40% 的题没有区分度

| 测量 | 结果 |
|---|---|
| gold 重放 + 合成 respond | **50 / 50** ✅（Stage 0 出口判据达成） |
| 重放 gold 后数据库**没变**的题 | **20 / 50（40%）** |
| **「什么都不做」直接拿满分 1.0 的题** | **19 / 50（38%）** |

20 道废题构成：16 道 gold 里**一条写动作都没有**（含 7 道 0 动作题）+ 4 道只有 `transfer_to_human_agents`（该工具在 `terminate_tools` 里，`calculate_reward` 重放时**会跳过它**）。

**规律非常干净——只要 gold 里有一条真写动作，题目就有区分度**（写动作 0 条 → 16/16 废；≥2 条 → 21/21 全有效）。

**为什么致命**：放进 GRPO 组内归一化 `A_i = (r_i - mean) / (std + eps)`——
全组不动 → `std=0` → 零梯度；一旦有 rollout 乱动 → 它拿 0 → `std>0` → 梯度指向**「别调工具」**。

> ⚠️ **这等于人为制造出我们要研究的「欠调用（学会猜）」现象——不是模型学坏的，是数据喂出来的。**
>
> **这条把项目核心命题从设想变成实测**：不是坍塌了所以没信号，是**数据侧的样本结构天生在推"别动"**。
> 而且 **longhorizon 没报告这条**——它说「50 题是核心约束」，实际**有区分度的只有 31 道**。

> 🔴 **v4 修正（2026-09-16）**：本段的"20 道 / 40%"应为 **19 道 / 38%**；
> 且"reward 与策略行为**无关**"是**错的**——它们是**单边信号**（乱做会掉分），
> 详见顶部 v4 修订与 `reports/信号形状探针-2026-09-16.md`。

### 一个反直觉的机制（顺手推翻了本仓脚本的早期推断）

砍掉 `cancel_reservation`（11 道题在用）后，gold 重放通过率 **一点没降**（仍 46/50）。
因为 `calculate_reward` 不是比对「标准终态」，而是**用同一套 `tools_map` 把 gold 重放一遍**再比哈希。
工具被砍 → agent 调它失败、gold 重放它也失败 → **两边同样 no-op，哈希照样相等**。

> **砍工具的代价是「题目失去区分度」，不是「reward 归零」。**

### 环境（踩坑，已固化进 `env/bootstrap.py`）

- **Python 3.9 跑不了**：`envs/airline/env.py:22` 的 `match` 是 3.10+ 语法，连 import 都过不去
  → 统一用同一个 **Python ≥ 3.10** 的环境（本项目实测 3.13 + pydantic 2.13.5）
- **litellm**：`envs/user.py:5` 顶层 import，用 `sys.modules` **注入桩**（参考仓库保持只读，未改一行）
- **`load_user` 写死**在 `base.py:74` → 猴补丁
- **`reset()` off-by-one**：`random.randint(0, len(tasks))` 会 IndexError → 永远显式传 `task_index`

**Stage 0 出口判据进度**：
- [x] gold 重放 + 合成 respond → **50/50** ✅
- [x] **mock-policy 自测全绿** ✅（v5，`scripts/eval_mock.py`）
- [x] **token 长度分布实测** ✅（v4，p50=6,692）

---

## 🔄 v2 修订（2026-09-11 晚）：硬件升级 + 三个认知纠正

**用户可租 A800** → 原"1×4090 降级复现"的定位作废。**1×A800 先 smoke test，确认后再决定加卡。**

深挖 longhorizon 仓库后，**推翻了 v1 的三个前提**：

| v1 说的 | 实际（已核配置文件） | 影响 |
|---|---|---|
| longhorizon 是 **7B 全参** | **7B + LoRA r=16/α=32** | "对齐它"= 用 LoRA，**比全参便宜得多** |
| 上下文 S_max = 8192 | **8192 + 12288 = 20480** | 显存需求比 v1 算的大 |
| KL coef = 0 | **0.01, low_var_kl** | 它用了 KL 约束 |

**它的真实超参**（`configs/train/grpo/vanilla_grpo.yaml`）：
```
n=8 ｜ train_batch_size=4（=32 轨迹/步）｜ lr 5e-6
temperature 0.7 ｜ top_p 0.9 ｜ max_turns 15+15
optimizer_offload=true ｜ param_offload=true ｜ use_fused_kernels=true
bypass_mode=true ｜ calculate_log_probs=true ｜ tensor_model_parallel_size=2
```

**它的三阶段基线（可直接当 E1 对照，不用自己跑 base）**：

| 阶段 | pass^1 | 备注 |
|---|---|---|
| Base 7B | 0.160 | |
| SFT | 0.145 | **SFT 后反而降了！** reasoning p50 63 → 22 token（模式压缩） |
| GRPO step150（峰值） | **0.225** | |
| GRPO step200（坍塌） | **0.175** | avg_turns 7.3→5.08，error_rate 0.010→0.200 |

**实际训到 225 步就终止**（配置写 500），终止依据：grad_norm 衰减到 0.005-0.01 + pass^1 触顶回落。

### 🎯 深挖发现的最大机会（v1 完全没看到）

它的诊断报告白纸黑字承认：

> **"τ-bench airline 官方只有 50 个 task。数据量小是本项目最核心的结构性约束。"**
> `train ∩ eval = 40/50 = 80%` 重合
> **"train.parquet 仅 40 task……无法扩大 train pool。所以「加数据」不是可选解法"**

而 `experiments/sft_collect_airline/summary.json` 实测暴露：

```
num_tasks_with_success:  19 / 50  = 38%     ← 只有 38% 的任务采得出成功轨迹
dropped_tasks:           31 个             ← 62% 被丢弃
n_train_trajectories:    45                ← GRPO 训练集 ≈ 40-45 条
unseen_task_ids:         10 个             ← "泛化"指标只有 10 题（统计不可靠）
elapsed_seconds:         25,913 ≈ 7.2 小时  ← 用 72B best-of-16 采这些数据花的
```

**→ 它被迫放弃的方向，恰好是我们能做而它做不到的**：

| | longhorizon | 我们 |
|---|---|---|
| 造 SFT 数据 | 72B 模型 best-of-16，**7.2 小时** | **程序化执行 gold action 序列，0 成本无限量** |
| 训练集 | **40 条** | 扩题器生成 **400+** |
| 泛化指标 | **10 道 unseen 题** | **200+ 道真不相交** |

**这不是重复它，是解掉它自己承认解不掉的约束。** 叙事："它说加数据不是可选解法，我证明了那是它数据构造方法的限制，不是问题本身。"

### 🧰 白捡的资产：它的配置可直接复用

```
configs/train/grpo/vanilla_grpo.yaml      ← baseline，超参全在
configs/train/grpo/{lata,prm_lite,turn_discount}.yaml   ← 消融臂
configs/train/mock/mock_grpo.yaml         ← ★ 单卡 5 步管线验证（n_gpus_per_node=1）
configs/tool_config/tau_bench_airline_tools.yaml
scripts/train/grpo/{run_vanilla.sh,build_grpo_parquet.py}
verl/                                     ← vendored 的完整 veRL（0.6.1）
docs/vanilla_grpo/vanilla_grpo_diagnosis.md   ← ★ 三大病因 + 14 种 reward hacking 模式
```

### ⚠️ 卡点（详见 `smoke-test-清单.md`）

1. **`setup.sh` 第 5 步有 Python 语法错误**（行首多空格 + 说注释没注释）→ `set -e` 下整脚本崩
2. **`experiments/sft_lora_merged` 不存在**（运行产物）→ smoke test 时把 `model.path` 改指 base 模型
3. **`CUDA_HOME` 必须指系统 CUDA**（torch 是 cu126，系统工具链是 12.4）→ 开机先 `ls /usr/local/`

**下一步 → `smoke-test-清单.md`（1 卡 ≤2 小时，验证环境 + 拿真实速度数字）**

---

## 📌 以下 v1 内容（4090 版）部分已过时，保留作推演记录

> v1 是在"只有 1×4090 24GB"的前提下推的。**硬件部分已作废**，但
> 「场景选择理由 / 工具 14→7 / 用户模拟器设计 / 风险清单 / 红线」仍然有效。

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

**实测轨迹长度**（n=268，gpt-4o 84 + sonnet 184，**含 14 工具 schema**，2026-09-16 重测）：
```
p50 = 6,692 token ｜ p90 = 10,287 ｜ max = 23,936 ｜ 平均 21 条消息
其中 assistant token 只占 12%（752 / 6,692）；tool 返回占 65.7%
工具 schema 固定开销：14 个 = 3,249 token ｜ 12 个 = 3,062 token
```
> 🔴 **本节已修正（2026-09-16）**：原写 "p50 ≈ 2,512" 是**漏算了工具 schema** 的数字
> （2,512 ≈ gpt-4o 不含 schema 的 2,624）。含 schema 后**标称 8192 与实测中位数是同一量级（1.2×），
> 不是"小一个量级"**；且 **p90 = 10,287 已超过 S_max 8192 → 必须做截断**。
> 详见 `reports/token分布实测-2026-09-16.md`。

### 工具 14 → 12 ⚠️（v3 已修正，原「14 → 7」作废）

**保留 12 个**：11 个 gold 里出现过的（`book_reservation` / `calculate` / `cancel_reservation` /
`get_reservation_details` / `get_user_details` / `search_direct_flight` / `send_certificate` /
`transfer_to_human_agents` / `update_reservation_baggages` / `update_reservation_flights` /
`update_reservation_passengers`）+ `search_onestop_flight`

**砍掉 2 个**：`think`（返回空串，纯占轮次）、`list_all_airports`（wiki 里已有）

**为什么不能按原方案砍到 7 个**（实测）：砍掉的工具若出现在 gold 里，会让该题**失去区分度**——
reward 与策略行为无关、恒为 1.0。原方案让有区分度的题从 30 → **14**（废题率 40% → 72%）。

`search_onestop_flight` 虽然 gold 里 0 次、砍了也不掉区分度，但 **agent 真的需要它**
（task[0] 的 instruction 明写 "one stopover also fine"）——
**gold 不记录搜索类动作 ≠ 搜索不需要发生**。

> 实测 tool schema：14 个 = **9,144 字符 ≈ 2,345 token**；12 个 = 8,561 ≈ 2,195。
> 对 8K context 是 **29% 固定税**（原文写 40%，高估了）。

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
- ~~**Gold 轨迹合成**：在 env 里直接执行 `tasks.py` 的 actions → **零模型零 API 的无限完美 SFT 数据**~~
  ⚠️ **v3 实测推翻**：gold 里 `respond` 出现 **0 次**，重放出来的轨迹**没有收尾回复**，
  直接当 SFT 数据不可训练。可行路径待定（补合成回复？只拿 gold 当 reward 校验？）
- ⭐ **19 道单边信号题的处理**（v4 已定方案，最高优先级）：**训练剔除、评测保留作「过调用」探针**
  （它们是 50 题里唯一能测过调用的一批；见 v4 修订与 `reports/信号形状探针-2026-09-16.md`）
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
| **E6 S/O 配比** ⭐ v4 新增 | 惩罚型样本占比是否决定退化方向 | 三臂（31+19 / 31+0 / 31+38）的工具调用数漂移曲线 |
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

**Stage 0 出口（全 CPU，租卡前）** ⚠️ **解释器必须是 3.10+**（3.9 跑不了 τ-bench）：
```bash
cd 代码库/projects/ToolHorizon
PY="python3.13"

# ★ 已跑通（2026-09-15）：gold 重放 oracle
PYTHONIOENCODING=utf-8 $PY scripts/verify_oracle.py
#   → A gold 重放 46/50 ｜ B 重放+合成 respond 50/50 ✅ ｜ C 什么都不做 19/50

PYTHONIOENCODING=utf-8 $PY scripts/analyze_hackable.py       # 40% 无区分度归因
PYTHONIOENCODING=utf-8 $PY scripts/analyze_tool_subsets.py   # 工具子集对比

# 待做：
# $PY eval.py --mock          # 期望全绿
# $PY token_stats.py          # 期望 p50≈2500, p90≈4000, max≈4500
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
├── env/                   ← ✅ 已建：τ-bench 包装
│   ├── bootstrap.py       ←   sys.path 接入 + litellm 桩（必须先 import）
│   ├── user_sim.py        ←   规则式用户模拟器（当前占位版，slot 解析待做）
│   └── tau_env.py         ←   env 工厂 + 工具子集 + 数据缓存
├── data/                  ← ✅ 已建：实测产物
│   ├── oracle_results.json        ← gold 重放 oracle 逐题结果
│   ├── hackable_analysis.json     ← 无区分度题归因
│   └── tool_subsets.json          ← 工具子集对比
├── reward/                ← 奖励函数 + 评测 harness（待做）
├── train/                 ← SFT + GRPO（复用 Yu-Agentic-RL 的 grpo_update.py）
├── observe/               ← 四指标观测器 + 工具退化指标 ⭐（待做）
├── scripts/               ← ✅ 已建：5 个探针/验证脚本
│   ├── probe_tau_bench.py         ← 工具分布 + gold 结构
│   ├── probe_tasks_compare.py     ← tasks.py vs tasks_test.py
│   ├── verify_oracle.py           ← ★ Stage 0 出口判据
│   ├── analyze_hackable.py        ← ★ 40% 无区分度归因
│   └── analyze_tool_subsets.py    ← ★ 工具子集方案
└── reports/               ← ✅ Stage0-实测发现-2026-09-15.md
```

---

## 相关文件

- `reference-repos/agentic-grpo-longhorizon/` — 参考天花板（τ-bench + 消融报告，**只读**）
- `reference-repos/Yu-Agentic-RL/scripts/grpo/grpo_update.py` — GRPO 主体（gating + LATA 已实现）
- `reference-repos/Yu-Agentic-RL/scripts/grpo/search_agent.py` — 多轮 episode 循环 + `--mock` 模板
- `projects/GRPO-Tamer/` — 原项目 A（**已归档**，仅保留数据笔记）
- `docs/主作品增量方案.md` — 原增量方案（**已归档**）
