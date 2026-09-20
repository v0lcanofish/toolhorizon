# A800 Smoke Test 清单（1 卡，目标 ≤2 小时）

> 定稿 2026-09-11 ｜ **租卡是按小时计费，卡上现想一步就是一元钱。**
> 目标：用最少的钱验证环境能跑通 + 拿到真实的速度/显存数字，**用于决定后续租几张、租多久**。
> 依据：`reference-repos/agentic-grpo-longhorizon/` 的 `setup.sh` / `requirements.txt` / `run_vanilla.sh`

---

## 〇、先看结论：三个已探明的卡点

### 卡点 1：`setup.sh` 第 5 步有语法错误，会直接崩

```python
 snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ', cache_dir='./models')
#↑ 行首多一个空格，且注释说"先注释"但没注释掉
```
Python 会报 `IndentationError: unexpected indent` → `set -e` 下整脚本退出。
**修法**：跑之前先删掉这一行（72B 我们不需要）。

### 卡点 2：SFT checkpoint 不存在

配置里的 `model.path: experiments/sft_lora_merged` **是运行产物，仓库里没有**（我们实测过 `experiments/` 只有 3 个目录）。
**修法（smoke test 专用）**：把 mock 配置的 `model.path` 改成 **base 模型** `./models/Qwen/Qwen2.5-7B-Instruct`。smoke test 只验管线通不通，不验效果。

### 卡点 3：`CUDA_HOME` 必须指向系统 CUDA 12.4

`run_vanilla.sh` 里的原话：
```bash
# PyTorch 是 2.7.0+cu126(自带 CUDA 12.6 runtime),但系统 CUDA 工具链是 12.4
# CUDA_HOME 必须指向实际存在的系统目录,供 triton/deepspeed 找 ptxas
export CUDA_HOME=/usr/local/cuda-12.4
export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas
```
**开机后先 `ls /usr/local/ | grep cuda` 确认实际版本**，不一定是 12.4。

---

## 一、开机前（本地就能做，不花 GPU 钱）

- [ ] AutoDL 选实例：**1×A800 80GB**，镜像选 **CUDA 12.4+**，PyTorch 2.x 基础镜像
- [ ] 确认 `nvidia-smi` 显示的是**完整 80GB**（防虚拟化切分/4090D 那种坑）
- [ ] 磁盘：Qwen2.5-7B ≈ **15GB**，veRL + 依赖 ≈ 10GB → **系统盘至少 50GB**
- [ ] 准备好本清单 + `reference-repos/agentic-grpo-longhorizon/` 整个目录（要传上去）

> ⚠️ **传仓库**：`reference-repos/` 被 `.gitignore` 排除，不在 GitHub 上。需要**打包上传**或从 GitHub 重新 clone
> （`git clone https://github.com/qiqihezh/agentic-grpo-longhorizon`，国内加 `https://ghproxy.net/` 前缀）

---

## 二、卡上步骤（按顺序，每步有验收标准）

### Step 0 · 验机器（2 分钟）
```bash
nvidia-smi                        # 验收：确认 80GB、显存没被占
ls /usr/local/ | grep cuda        # 验收：记下 CUDA 版本，后面要用
df -h                             # 验收：系统盘 ≥50GB 可用
```
**记下**：GPU 型号、显存、CUDA 版本、可用磁盘。

### Step 1 · 建环境（15–25 分钟，网络快的话）
```bash
conda create -n agentrl python=3.10 -y
conda activate agentrl
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
**验收**：打印 `2.7.0+cu126 True`。**`False` 就是白租，立刻排查。**

### Step 2 · 装依赖（20–40 分钟，**最耗时**）
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation     # 要编译，慢
cd /root/verl && pip install -e .                # veRL editable，不要 pip install verl
cd /root/tau-bench && pip install -e .
```
**验收**：`python -c "import verl, vllm, flash_attn; print('ok')"`
> ⚠️ **加速**：`flash-attn` 编译很慢，可以先跳过——smoke test 用不上它（配置里是 `flash_attention_2`，但 base 模型用默认 attn 也能跑）。
> **建议：Step 2 先跳过 flash-attn，跑通再说。**

### Step 3 · 下模型（10–20 分钟，看带宽）
```bash
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./models')
"
```
**验收**：`du -sh ./models/Qwen/Qwen2.5-7B-Instruct` ≈ 15GB

### Step 4 · 光跑环境（**不加载模型，5 分钟**）★ 性价比最高
```bash
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP="True"
python -c "
from tau_bench.envs.airline.env import MockAirlineDomainEnv
env = MockAirlineDomainEnv()
res = env.reset(0)
print('task 0 instruction:', res.observation[:200])
"
```
**验收**：打印出任务描述。
> 💡 **这一步纯 CPU**，但它是**验证 tau-bench 装对了 + env 能跑**的最快方式。
> 如果 `litellm` 报 import 错，检查 `tau_bench/envs/user.py` 的顶层 import（已知坑）。

### Step 5 · 加载 7B + 单次前向（5 分钟）
```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p='./models/Qwen/Qwen2.5-7B-Instruct'
tok=AutoTokenizer.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='cuda')
print('显存占用:', round(torch.cuda.memory_allocated()/1e9,2), 'GB')
out=m.generate(**tok('你好', return_tensors='pt').to('cuda'), max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
"
```
**验收**：能生成文本 + 打印显存。
**记下这个数字**——7B bf16 权重应该 ≈ **15.2GB**。对不上说明 dtype 没生效。

### Step 6 · vLLM 加载 + 测速（10 分钟）★ **关键数字**
```bash
export VLLM_USE_V1=1
python -c "
import time
from vllm import LLM, SamplingParams
llm = LLM(model='./models/Qwen/Qwen2.5-7B-Instruct',
          tensor_parallel_size=1, gpu_memory_utilization=0.5,
          max_model_len=24576, enforce_eager=True)
sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, n=8)
t=time.time()
outs = llm.generate(['请写一段关于航班的介绍']*4, sp)
dt=time.time()-t
ntok=sum(len(o.token_ids) for out in outs for o in out.outputs)
print(f'生成 {ntok} token 用时 {dt:.1f}s = {ntok/dt:.0f} tok/s')
"
```
**验收 + 记下**：**tokens/s**。
> 这个数字决定训练时长估算：每步 32 条轨迹 × ~2500 token ≈ 80K token。
> 实测 tok/s → 算出每步生成要多久 → 乘以 225 步 = vanilla GRPO 的总时长。

### Step 7 · 跑 5 步 mock（可选，30–60 分钟）★ **真正的验收**
把 `configs/train/mock/mock_grpo.yaml` 改两处：
```yaml
model.path: ./models/Qwen/Qwen2.5-7B-Instruct        # 原来是 experiments/sft_lora_merged
trainer.n_gpus_per_node: 1
```
然后（**注意：需要先有 parquet 数据**）：
```bash
bash scripts/train/grpo/build_grpo_parquet.py --help   # 先看它怎么造训练数据
python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs --config-name=mock_grpo
```
**验收**：5 步跑完不报错。
> ⚠️ 这一步依赖 `experiments/vanilla_mock/train.parquet`，**当前不存在**，要先跑 `build_grpo_parquet.py`。
> **如果 Step 6 顺利且时间不够，Step 7 可以留到下一次租卡。**

---

## 三、必须记下来带回的数字

| 数字 | 用途 | 从哪来 |
|---|---|---|
| GPU 型号 / 显存 / CUDA 版本 | 确认机器对得上 | Step 0 |
| 环境搭建总耗时 | 估算后续每次开机成本 | Step 1–2 |
| **7B bf16 权重实测显存** | 验证显存预算 | Step 5 |
| **vLLM tok/s** | **算训练总时长 → 决定租几张** | Step 6 |
| 7B 在 tau-bench 上的 step 耗时 | 同上 | Step 7（可选） |
| 遇到的每个报错 + 解法 | 下次不用重踩 | 全程 |

---

## 四、成本红线

- **Step 0–3 合计约 1–1.5 小时**（含下载）。如果 2 小时内没到 Step 4，**停下来重新评估**——大概率是网络或镜像问题，别硬烧
- **每步骤做完就记数字**，不要等全部跑完再回忆
- **不确定就先关机**：AutoDL 关机不计费（只计存储）

---

## 五、smoke test 通过后要回答的问题

1. 1×A800 够不够？还是必须 2 张？
2. vanilla GRPO 跑 225 步要多少 GPU 小时 → 多少钱？
3. 4 条消融臂的总成本是多少 → 预算够不够？
4. **创新点定哪个**（用户 9/11 决定：等 smoke test 后再说）

---

## 附：为什么这个 smoke test 特别重要

它同时验证三件**从未被验证过**的事：
1. 用户**从未跑过任何 GPU 训练**（本地一直 CPU）
2. AutoDL **注册了但从没用过**
3. 7B 级模型的整套 veRL 栈**能不能在单卡跑起来**

**任何一个不成立，整个项目计划都要改。所以这 1–2 小时是整个 6 周里性价比最高的时间。**
