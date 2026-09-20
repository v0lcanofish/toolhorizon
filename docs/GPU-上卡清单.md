# GPU 上卡清单 · ToolHorizon

> 建立 2026-09-17 ｜ **照着从上往下走，每步都有"成功长什么样"和"挂了怎么办"**
> 配套：`scripts/gpu_smoke.py`（第一步跑的）、`scripts/make_upload_bundle.py`（打包）

---

## 0 · 这次租卡的目标（先说清楚，免得跑偏）

**第一次只干一件事：花 30 分钟拿到真实速度数字，然后关机。**

具体要拿到四个数：
1. 显存是不是满血 24 GB（防 4090D / 虚拟化切分）
2. 跑一条 rollout 要几秒
3. 生成速度 tok/s
4. 训练一步的**峰值显存**

**为什么先只要这四个数**：后面所有决策（采样批量多大、训练多少步、要不要第二张卡、总预算多少）
都依赖它们。**先花 1~2 块钱买这四个数，比直接开跑几小时划算得多。**
拿到数字后再定下一步方案。

---

---

## 0.5 · ✅ 实测数字（2026-09-17 · RTX 4090D 24GB）

> 第一次租卡的成果。**下面每一个数字都是真跑出来的，不是估算。**
> 命令：`python scripts/gpu_smoke.py`

| 指标 | 实测 | 评价 |
|---|---|---|
| 显存总量 | **24.0 GB**（24564 MiB） | 满血，没被阉割 |
| 权重加载 | 5.2 秒 ｜ 占 2.88 GB | 磁盘很快 |
| vLLM 启动 | 119.8 秒 | 每轮一次，可接受 |
| ⭐ **采样：每条轨迹** | **4.14 秒** | |
| 生成速度 | 327.5 tok/s | 正常 |
| ⭐ **训练：一步（8192 token）** | **2.1 秒** | 前向+反向 |
| ⭐ **训练峰值显存** | **5.15 GB** | ✅ 判据 ≤ 13 GB |
| 显存占用率 | **21%** | 余量 4 倍 |

**换算成实际节奏**：

```
一轮 31 题 × 8 条 = 248 条
  采样        17.1 分钟
  vLLM 启动    2   分钟
  训练         8.7 分钟（248 × 2.1 秒）
  ─────────────────────
  每轮        约 28 分钟
```

### 🎯 卡型结论：**4090D 够用，不需要 A800**

| | 4090D | A800 |
|---|---|---|
| 显存 | 24 GB | 80 GB |
| 价格 | ~¥1.5–2/h | ~¥5–8/h |
| 我们实际用 | **5.15 GB** | 5.15 GB |
| 结论 | ✅ **够用 + 便宜 3–4 倍** | ❌ 租它是烧钱 |

**A800 只在两种情况用**：
1. 4090D / 3090 / A5000 全都抢不到时（**方案完全不变**，代码不用改）
2. 以后要补 **7B 规模对比**实验（可选增量，7B LoRA 峰值约 18–20 GB，24 GB 会很紧）

> 算力上也占不到便宜：我们训的是 1.5B 小模型，瓶颈在显存不在算力，
> A800 的算力优势在这个尺度上体现不出来。

### 🔴 这次抓到的 4 个「只有真卡才暴露」的 bug

| # | 问题 | 如果没抓到会怎样 |
|---|---|---|
| 1 | tokenizer 路径在云上找不到 | 烟雾测试直接崩 |
| 2 | `reports/` 目录不存在 | 报告写不出去，数字全丢 |
| 3 | `load_model` 从错的模块导入 | 训练脚本启动即崩 |
| 4 | ⭐ **漏开梯度检查点** | **租卡 → 采样 17 分钟 → 训练 OOM 崩掉，白烧一轮** |

**第 4 个最值钱**：设计文档里写明要开梯度检查点，函数也写了，
**但训练主路径从来没调用过** —— 一条 8192 token 的轨迹前向就把 24 GB 吃满
（PyTorch 分配 22.5 GB）。开了之后激活从 ~20 GB 降到 ~2–3 GB，峰值 5.15 GB。

> **这类 bug 本地 CPU 永远测不出来** —— 无 GPU 时那段代码根本不执行。
> 这正是"先花 ¥2 跑烟雾测试"的价值所在。

---

## 0.6 · 下次开机怎么恢复（照着抄）

```bash
# ① 开机（GPU 模式）→ 进 JupyterLab → Terminal
cd /root/autodl-tmp/ToolHorizon
source scripts/env.sh          # ← 一条命令恢复所有环境变量（含自检）
```

**关机 ≠ 释放，这是两件事**：

| 操作 | 数据盘（模型/代码/checkpoint） | 系统盘（pip 装的包） |
|---|---|---|
| **关机** | ✅ 全保留 | ✅ 全保留 |
| **释放** ⚠️ | ❌ 全删 | ❌ 全删 |

**点「关机」，别点「释放」。**

**最坏情况也不可怕**：代码和数据你本机有完整副本；模型 3 GB 重跑一次 setup 就有（约 25 分钟）。

### ⚠️ 长跑必须挂 tmux，否则关掉浏览器就断

```bash
tmux new -s train          # 开会话
# 在里面跑训练…
# Ctrl+B 然后按 D          → 挂起（训练继续，可以关浏览器）
tmux attach -t train       # 回来看
tmux ls                    # 看有哪些会话
```

---

## 1 · 传文件

本地先打包（在 `代码库/projects/ToolHorizon` 下）：

```bash
python3.13 scripts/make_upload_bundle.py
# → dist/toolhorizon_upload.tar.gz   164 个文件 / 1.4 MB
```

传上去并解开：

```bash
scp dist/toolhorizon_upload.tar.gz root@<机器IP>:~/
ssh root@<机器IP>
mkdir -p ~/work && tar xzf toolhorizon_upload.tar.gz -C ~/work && cd ~/work/ToolHorizon
```

**包里有什么、为什么**：

| | |
|---|---|
| ✅ `env/ train/ observe/ scripts/ data/` | 全部代码 + 题 + SFT 数据 + 槽位（5.8 MB） |
| ✅ `tau-bench/tau_bench/` | 环境本体（8.9 MB）—— **只装了 `tau_bench` 子目录** |
| ❌ 没装 `historical_trajectories/` | 51 MB，只在**造 SFT 数据**时用过一次，训练采样都不碰 |
| ❌ 没装本地 `models/` tokenizer | 云上用**模型自带**的那份（实测两份一致，见 §3 的说明） |

**让代码找到 τ-bench**（包里的目录名是 `tau-bench`，和默认查找路径对得上）：

```bash
export TAU_BENCH_PATH=~/work/tau-bench
```

---

## 2 · 装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 里是 CPU 能装的部分。**GPU 相关的两个必须自己装对版本**：

```bash
# 1) torch —— 跟着镜像自带的 CUDA 版本走，别硬套
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 2) vLLM —— 版本必须和 torch/CUDA 对得上，装错会在启动时报一堆看不懂的错
pip install vllm
```

✅ **成功长什么样**：`python -c "import torch, vllm, transformers, peft; print('ok')"` 不报错。

⚠️ **AutoDL 用户**：镜像一般已经带好 torch + CUDA，**先别升级 torch**，
只要 `pip install vllm` 能装上、能 import，就不要再动。

---

## 3 · 第一步：跑烟雾测试 ★

```bash
cd ~/work/ToolHorizon
python scripts/gpu_smoke.py --n 5
```

它分四段跑，**每段独立计时、独立报错**，哪段挂了一眼就知道该修什么：

| 段 | 在验什么 | 成功长什么样 |
|---|---|---|
| ① 硬件 | 显存满血吗、torch 看得见卡吗 | 打印 `显存总量 24.0 GB` + 卡型号 |
| ② 加载 | 1.5B 权重进来要多久 | `加载用时 X s ｜ 权重占显存 Y GB` |
| ③ 采样 ★ | 起 vLLM，跑 5 条真轨迹 | **每条轨迹 X 秒 / X tok/s / 采样峰值显存 X GB** |
| ④ 训练 | 训一步，看峰值显存 | **训练峰值显存 ≤ 13 GB** |

### ✅ 第 ② 段的 tokenizer 对账（2026-09-17 实测结论）

```
本地 tokenizer 词表 151665 ｜ Qwen2.5-1.5B-Instruct 词表 151665
✅ 一致，之前量的 token 数可以直接用
```

**⚠️ 更正一处早先的误判**：这里原本写着"两个数必须一致，不一致会静默毁掉训练"，
并把本地词表 **151,665** 和 **151,936** 当成"两份不同的 tokenizer"。

**那个判断是错的** —— 两个数根本不是同一个量：

| | 数值 | 是什么 |
|---|---|---|
| tokenizer 的 token 数 | **151,665** | 本地那份与 Qwen2.5 **完全相同** |
| `config.json` 里的 `vocab_size` | 151,936 | 模型嵌入表**预留了 271 个空槽** |

**本来就该不相等。** 实测 `vocab_same: true`、`sample_ids_same: true`。

> 不过这个误判催生的防御措施是对的，保留：
> **`train/collect.py` / `train/trainer.py` 强制用模型自带的 tokenizer。**
> 原则没错 —— tokenizer 必须和模型同源。只是在 Qwen2.5 这个例子里，两份恰好一样。

### 跑完取回报告

`reports/gpu_smoke.json` 会落盘。**把那个 JSON 取回**，据此算：
- 一轮采样要几分钟
- 训练 250 步要几小时
- 总共要烧多少 GPU 小时 ≈ 多少钱

---

## 4 · 第二步：全链路彩排（零成本，但要在卡上跑一次）

烟雾测试只验了单条轨迹。跑训练之前，先在卡上把**完整的 collect → train 一圈**走一遍：

```bash
# 先 SFT 预热（§5.1），再：
python -m train.run --rounds 1 --select train --n 8 --limit 4 --run-name rehearsal
```

`--limit 4` 让它只跑 4 道题、1 轮，**几分钟就完**。

✅ **成功长什么样**：`runs/rehearsal/` 下出现 `rollouts/step_000.jsonl` 和 `ckpt/ckpt_000/`，
终端打出"组内零方差率"那一行。

⚠️ 这一步的意义：**把"路径写错 / 磁盘满了 / adapter 存不下"这类问题在 1 轮里暴露掉**，
不要等到跑了 20 轮才发现。

---

## 5 · 正式开跑（拿到 smoke 数字、方案定了之后）

### 5.1 Stage 1 · SFT 预热（必须做，不是可选项）

```bash
cd /root/autodl-tmp/ToolHorizon
source scripts/env.sh                    # 先恢复环境（含自检）

python -m train.trainer sft \
  --model "$TOOLHORIZON_TOKENIZER" \
  --sft-data data/sft_final.jsonl \
  --out models/adapter_sft
```

**为什么必须有**：1.5B 基座在航空客服上的通过率大概率≈0。
通过率 0 意味着**组内全错 → std=0 → advantage 全 0 → 一步都学不动**。
SFT 的作用是把通过率抬到"有的对有的错"的区间里，GRPO 才有活干。

**硬判据**：SFT 完在 medium 档上 pass@1 ∈ **[15%, 50%]**。
- **< 5% → 立刻降到 easy 档，不要头铁**（降级链见 README）
- \> 50% → 题太简单了，GRPO 会组内全对，同样是白跑

#### 5.1b 判据怎么量 ⭐（2026-09-18 新增，**别跳过这步**）

判据有入口了 —— `scripts/eval_sft_gate.py`。**SFT 前后各跑一次**：

```bash
# ① 先看要花多少钱（零成本，建议每次都先 --dry-run）
python scripts/eval_sft_gate.py --dry-run

# ② SFT 之前：量基座，拿到 before
python scripts/eval_sft_gate.py --model "$TOOLHORIZON_TOKENIZER" \
    --out reports/sft_gate_base.json

# ③ SFT 之后：量 adapter，拿到 after + 自动判 [15%, 50%]
python scripts/eval_sft_gate.py --model "$TOOLHORIZON_TOKENIZER" \
    --adapter models/adapter_sft --out reports/sft_gate_sft.json
```

每次约 **10 分钟 / ¥0.3**（30 题 × 4 条 = 120 条轨迹）。
**两次相减才是 SFT 的增量** —— 只量 after 的话，"SFT 有没有用"这个问题答不了。

**为什么必须有这一步**：在它之前，`observe/harness.py` **只有 `--mock` 模式**，
拿真 adapter 算不出 pass@1。也就是 —— 跑完 SFT 不知道该不该往下跑 GRPO，
只能瞎猜。猜错的代价是那次 12 GPU 时的 GRPO 白烧。

⚠️ **统计噪声（别把这个数当定论）**：30 道题时，pass@1 的 95% 区间宽约 **0.31**，
而判据带本身只有 0.35 宽。所以 —— **落在边界上的值（0.14 / 0.16）不算结论**，
脚本会自己提示"区间跨过了判据边界"，那时候加 `--limit 60` 重测。

### 5.2 Stage 2 · GRPO 主训练

```bash
tmux new -s train                        # ⚠️ 必须挂 tmux，否则关浏览器就断

python -m train.run \
  --rounds 25 --select arm_base --n 8 \
  --model "$TOOLHORIZON_TOKENIZER" \
  --init-adapter models/adapter_sft \
  --run-name vanilla
```

**每一轮干两件事**（两个进程轮着来，显存不重叠）：
```
① train.collect   → 起 vLLM 采样 50 题 × 8 条，写 jsonl，退出
② train.trainer   → 读 jsonl，更新 LoRA，存成新 adapter
回到 ①，vLLM 重启时挂上新的 adapter
```

**盯什么（不是 loss）**：终端每轮会打 **组内零方差率**。

> 连续 3 轮 ≥ 0.8 → 脚本会自己警告。
> 含义是这批题对当前策略"不是全对就是全错"，**训练在空转**，而 loss 曲线看不出来。
> 对策：换难度档（`--select exp_medium`）或调 n，别硬跑到底。

### 5.3 E6 实验（本项目的核心实验）

只改"惩罚型样本"的比例，看工具调用数怎么漂：

```bash
python -m train.run --rounds 20 --select arm_none   --n 8 --run-name e6_none    # 31 道 S 类
python -m train.run --rounds 20 --select arm_base   --n 8 --run-name e6_base    # 31 + 19
python -m train.run --rounds 20 --select arm_double --n 8 --run-name e6_double  # 31 + 38
```

**假设**：S 类（做对给分）教"该动就动"，O 类（乱做扣分）教"别乱动"；
惩罚信号来得比奖励信号早且密 → 模型变保守 → **欠调用**。
三臂对照，如果 double 那臂的欠调用明显更重，**假设成立**。

---

## 6 · 挂了的排查表

| 症状 | 大概率原因 | 怎么办 |
|---|---|---|
| `setup` 时 `CUDA_HOME` 报错 | torch 是 cu126，系统工具链是 12.4 | `ls /usr/local/` 看实际版本，指过去 |
| vLLM 启动报版本冲突 | vLLM 和 torch/CUDA 版本对不上 | `pip install vllm==<匹配版本>`，别硬升级 torch |
| 训练启动 OOM，但 `nvidia-smi` 看着是空的 | **vLLM 进程没真死** | `pkill -f vllm`；`collect.py` 跑完会自己退出，真死了才有这问题 |
| 显存只有 22 GB 出头 | 4090D 或虚拟化切分 | **别往下跑**，显存账要重算 |
| 模型一直自问自答不停止 | 忘了 `stop=["<|im_end|>"]` | `VLLMEngine` 里已经写死了，报这个说明你在用别的引擎 |
| 采样特别慢（>60s/条） | 弱策略陷入循环，轨迹膨胀到 8–12K | `--max-turns` 调小；或先 SFT 再采样 |
| 磁盘满 | adapter 每轮 37 MB，20 轮 ≈ 740 MB | `df -h`；把旧 checkpoint 删掉 |

---

## 7 · 预算表（2026-09-17 已按实测更新）

| 阶段 | 计划 GPU 时 | 实测 | 备注 |
|---|---|---|---|
| 第一次 smoke | 0.5 h | ⬜ 待填 | **只做这个就先关机** |
| Stage 1 SFT | 1.5 h | ⬜ | 115 条 × 3 epoch |
| Stage 2 GRPO 单臂 | 6 h | ⬜ | 20 轮 × (采样+训练) |
| 3 臂消融 | 18 h | ⬜ | E6 三臂 |
| 课程重标定 | 4 h | ⬜ | Stage 3 |
| **合计** | **~57 h** | ✅ 实测过 | **≈ ¥100**（4090D ¥1.5–2/h） |

### 实测后的分次租卡计划（每次都有 checkpoint 落地）

| 第几次 | 干什么 | GPU 时 | 花费 |
|---|---|---|---|
| 1 | SFT 预热 → 存 `models/adapter_sft` | 1.5 | ¥3 |
| 2 | GRPO 首跑 25 轮 → **第一条曲线** | 12 | ¥25 |
| 3 | E6 三臂消融（核心实验） | 35 | ¥60 |
| 4 | 课程重标定 + 评测 | 9 | ¥15 |

**关键**：`models/` 和 `runs/` 都在数据盘，**关机保留** → 可以中断续跑，不会前功尽弃。

**分三次租**：①只跑 smoke ②跑 1 臂拿曲线 ③补消融。

---

## 8 · 红线（写在这，免得跑嗨了忘）

- ❌ **不要**把 adapter 挂到不同基座上 —— 不报错，只会静默变差
  （`scripts/eval_gpu_pipeline.py` 里有这条的反向验证）
- ❌ **不要**用本地那份 tokenizer 训 Qwen2.5 —— 词表不同源
- ❌ **不要**在 smoke 通过之前跑训练 —— 那是拿钱买"环境到底行不行"这个信息
- ✅ **每跑完一段就关机**，别让它空转
