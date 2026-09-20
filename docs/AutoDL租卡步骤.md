# AutoDL 租卡步骤 · 从零到跑通

> 建立 2026-09-17 ｜ 当前状态：**已注册 + 已实名 + 未充值**
> 目标：**花 ¥3 左右，30 分钟拿到四个数，然后关机。**

---

## 先看这张图，知道我们要去哪

```
① 充值 ¥10
      ↓
② 租一张 4090（先选「无卡模式」开机！） ← 省钱的关键
      ↓
③ 传文件 + 跑一键初始化           ← 下模型、装依赖都在这步，无卡模式 ¥0.1/h
      ↓
④ 关机 → 切换「GPU 模式」开机
      ↓
⑤ 跑烟雾测试（约 15 分钟）      ← 这一步才真正烧显卡的钱
      ↓
⑥ 下载 gpu_smoke.json → 关机
```

⭐ **省钱的核心在第 ② 步**：AutoDL 可以「**无卡模式**」开机——**没有显卡，但环境是全的**，
只要 **¥0.1/小时**。下模型（3GB）、装 vllm（几百 MB）、跑自检，这些**都不需要显卡**。
全在无卡模式做完，再切 GPU 模式只跑那 15 分钟测试。

---

## ① 充值 ¥10

控制台右上角 → 账户充值。

**充多少**：第一次 **¥10 就够**（无卡模式准备约 ¥0.3 + 显卡 15 分钟约 ¥0.5）。
不用多充，跑通了我们再决定后面充多少。

---

## ② 租卡（这里有 3 个地方容易选错）

进 **算力市场**，按下面选：

| 选什么 | 选成什么 | 选错的后果 |
|---|---|---|
| **地区** | 哪个有货选哪个，**不用挑** | — |
| **GPU 型号** | **RTX 4090**（不是 4090D！） | 4090D 是阉割版，显存一样但算力低 |
| **主机数量** | 1 | — |
| **镜像** ⭐ | **PyTorch 2.4.0 ｜ Python 3.12 ｜ CUDA 12.1** | ⚠️ **Python 必须 ≥3.10**。选到 3.8 的镜像整个项目跑不了，要重租 |
| **开机方式** ⭐ | **「无卡模式开机」** | 直接 GPU 开机的话，下模型的半小时也在按显卡价烧钱 |

> ⚠️ **如果 4090 没货**：等一会儿，或者换地区。**不要**退而求其次选 3090（24GB 但更慢）
> 或 A100（贵 3 倍，这次用不上）。

> ⚠️ **镜像如果找不到 2.4.0**：选 **PyTorch 2.1.0 ｜ Python 3.10 ｜ CUDA 12.1** 也行。
> 关键是 **Python ≥ 3.10** 和 **是 PyTorch 镜像**（不是 TensorFlow）。

**租的时候记下这三个东西**（在「容器实例」卡片上）：
```
SSH 登录指令：ssh -p 12345 root@region-1.autodl.com
SSH 密码：    xxxxxxxx
JupyterLab：  点「快捷工具」里的 JupyterLab
```

---

## ③ 传文件 + 初始化（无卡模式下做）

### 3.1 先进 JupyterLab

控制台 → **容器实例** → 找到你的实例 → 点 **JupyterLab**。

左边是文件树。**进到 `/root/autodl-tmp/` 这个目录**（数据盘，空间大；
`/root` 是系统盘只有 30GB，放模型会满）。

### 3.2 传文件上去

**方法 A（推荐，最简单）**：在 JupyterLab 左边文件树里，进到 `autodl-tmp`，
**直接把 `toolhorizon_upload.tar.gz` 拖进浏览器窗口**。

**方法 B**：用你本机终端 `scp`（把端口和地址换成你卡片上的）：
```bash
scp -P 12345 "dist/toolhorizon_upload.tar.gz" root@region-1.autodl.com:/root/autodl-tmp/
```

### 3.3 开终端，解压

在 JupyterLab 里点 **Terminal**（或者用 SSH 登进去），粘这几行：

```bash
cd /root/autodl-tmp
tar xzf toolhorizon_upload.tar.gz
cd ToolHorizon
```

解开之后你应该看到 `env/ train/ observe/ scripts/ data/` 和 `../tau-bench/`。

### 3.4 跑一键初始化

```bash
bash scripts/autodl_setup.sh
```

**它会自己干五件事**，每件都打印结果：
1. 检查 Python 版本（低于 3.10 直接报错让你换镜像）
2. 开 AutoDL 学术加速（**不开的话下模型会卡死在 HuggingFace**）
3. 装依赖（peft / vllm）
4. 下 Qwen2.5-1.5B（约 3GB，HuggingFace 走不通会自动换 ModelScope）
5. **跑一遍项目自检**——确认代码在卡上也是好的

⏱️ **大概 10–20 分钟**（主要花在下模型上）。无卡模式 ¥0.1/小时，这步花不到 ¥0.1。

**看到 `初始化完成` 四个字就是好了。**

---

## ④ 关机 → 切 GPU 模式

控制台 → 容器实例 → 点 **关机**。

等状态变成「已关机」后，点 **开机** → 这次选 **「GPU 模式」**。

⚠️ **注意**：无卡模式下改的东西**是保留的**，不用担心白干。

---

## ⑤ 跑烟雾测试（这一步才真正烧显卡的钱）

开机后进 JupyterLab → Terminal：

```bash
cd /root/autodl-tmp/ToolHorizon
export TAU_BENCH_PATH=/root/autodl-tmp/tau-bench
python scripts/gpu_smoke.py --n 5 --model "$(cat /root/autodl-tmp/models/MODEL_PATH.txt)"
```

**它会分四段跑**，大约 10–15 分钟：

```
① 硬件   显存是不是满血 24 GB
② 加载   1.5B 权重进来要多久 + tokenizer 对账
③ 采样 ★ 起 vLLM 跑 5 条真轨迹 → 「每条几秒」「每秒多少 token」
④ 训练   训一步，看峰值显存
```

跑完会在 `reports/gpu_smoke.json` 生成报告。

---

## ⑥ 关机，取回报告

```bash
cat reports/gpu_smoke.json
```

**把这一坨复制下来**，据此算：
- 一轮采样要几分钟
- 训练 250 步要几小时
- 总共要烧多少 GPU 小时 ≈ 多少钱

**然后就可以关机了。**

---

## 卡住了怎么办

| 现象 | 怎么办 |
|---|---|
| `python: command not found` | 镜像是 Miniconda 的，先 `source /root/miniconda3/etc/profile.d/conda.sh` 或 `conda activate base` |
| 初始化脚本说 Python 版本太老 | **换镜像重租**，别耗 |
| 下模型卡住不动 | 停掉，手动跑 `source /etc/network_turbo` 再重跑脚本 |
| `nvidia-smi` 在无卡模式下不存在 | **正常**，无卡模式就是没显卡。切 GPU 模式就有了 |
| 显存显示只有 22 GB | 是 4090D，**显存账要重算**（本项目按 24GB 设计） |
| 找不到 `tau-bench` 目录 | `export TAU_BENCH_PATH=/root/autodl-tmp/tau-bench` 再跑 |
| 🔴 **训练跑到一半进程没了** | 见下面「长跑的头号杀手」 |

---

## ⚠️ 长跑的头号杀手：宿主机重启

**2026-09-20 实测事故**：四臂消融的最后一条臂跑到第 12 轮（凌晨 03:45）突然中断。

怎么判断是**宿主机重启**而不是自己的代码崩：

```bash
tmux ls                     # → no server running on /tmp/tmux-0/default
nvidia-smi                  # → 0% / 0MiB / No running processes
ls -lt runs/<臂>/ckpt/      # → 最后一个 ckpt 停在某时刻，之后几小时无动静
```

⭐ **`tmux: no server running` 是最硬的信号** ——
**进程自己崩溃不会杀掉 tmux 服务端，只有重启才会。**

**关键认知**：

- **tmux 只挡"关浏览器"，挡不住宿主机重启。** 它挡掉 99% 的中断，但那 1% 会把前面烧的卡全废掉。
- **重启后训练不会自动恢复**，必须手动重跑。
- 所以长跑要按"随时可能被打断"来设计：
  - 每轮都存 checkpoint（本项目本来逐轮存）
  - 报告里写清楚**实际跑了多少轮**，别按计划轮数写
  - 想真正抗打断得加 systemd 单元或开机自启脚本 —— 本项目**没做**，如实标注

**恢复方式**：从最后一个 `ckpt_NNN` 接着跑，或把 `--rounds` 调小重跑补齐剩余轮数。

---

## 三个检查点

1. **租好了** → 记下卡型 + 镜像（对照 §② 确认没选错）
2. **初始化跑完了** → 确认没有报错
3. **烟雾测试跑完了** → 取回 `gpu_smoke.json`

**中间任何一步报错，把屏幕上红色的原文整段留下**，再对照上表定位。
