# -*- coding: utf-8 -*-
"""
上卡第一条命令 —— **30 分钟确认环境 + 拿到真实速度数字**。

    cd <项目目录>
    python scripts/gpu_smoke.py

为什么要单独一个脚本：
    租卡按小时烧钱。**一上来就跑训练，出了问题是"是环境的错还是代码的错"分不清**，
    只能靠重跑来二分，那是拿钱买信息。
    这个脚本把"能不能跑"和"跑多快"拆成四段，每段独立计时、独立报错，
    哪一段挂了一眼就知道该修什么。

四段：
    ① 硬件   nvidia-smi —— 显存是不是满血？（防 4090D / 虚拟化切分，README 风险 R4）
    ② 加载   1.5B 权重进来要多久、占多少显存
    ③ 采样   起 vLLM，跑 5 条真实 rollout —— 拿到 tok/s 和 每条秒数 ★最关键
    ④ 训练   训一步，记**峰值显存**（判据：≤ 13 GB）

⭐ ③ 的数字直接决定预算：把"每条秒数"乘上计划的总轨迹数，
   就是整个项目要烧多少 GPU 小时。**租第二张卡之前先看这个数。**

跑法：
    python scripts/gpu_smoke.py --n 5
    python scripts/gpu_smoke.py --n 5 --skip-train     # 只想快速确认环境
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

REPORT = PROJECT / "reports" / "gpu_smoke.json"
# ⚠️ 打包时 reports/ 不在清单里（图太重），所以脚本必须自己建 ——
#    2026-09-17 踩：没建，跑完最后一步写报告直接 FileNotFoundError，
#    前面辛苦跑出来的数字全打印在屏幕上了但没落盘。
REPORT.parent.mkdir(parents=True, exist_ok=True)


class _SkipStep(Exception):
    """内部信号：这一段的开关关掉了，直接跳过（不是错误）。"""


def banner(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)
    sys.stdout.flush()


def mem_gb():
    """当前已分配的显存（GB）。"""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 ** 3


def peak_gb():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def device_used_gb() -> float:
    """
    **设备级**显存占用（整张卡用了多少），跨进程有效。

    ⚠️ 为什么不能用 peak_gb()：vLLM 跑在**子进程**里（它自己会 spawn），
       父进程的 `torch.cuda.memory_allocated()` 只能看到自己，
       于是在 vLLM 采样时读出来永远是 **0.0**。
       （2026-09-17 踩：第一次烟雾测试 sample 峰值显存报的就是 0.0）
       `torch.cuda.mem_get_info()` 问的是驱动，谁占的都算，才准。
    """
    import torch
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 ** 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=5, help="跑几条 rollout")
    ap.add_argument("--tasks", type=int, default=2, help="用几道题（会 ×n 条）")
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-sample", action="store_true",
                    help="跳过 ③ 采样，直接量 ④ 训练显存（用 SFT 数据造 batch，1 分钟搞定）")
    args = ap.parse_args(argv)

    result = {"model": args.model, "when": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------------------------------------------------------- ① 硬件
    banner("① 硬件 —— 显存是不是满血？")
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        print(out)
        import re
        m = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", out)
        if m:
            used, total = int(m.group(1)), int(m.group(2))
            result["gpu_total_gb"] = round(total / 1024, 1)
            print(f"→ 显存总量 {total/1024:.1f} GB")
            if total < 22000:
                print("  ⚠️ 小于 22 GB —— 可能是 4090D 或虚拟化切分。"
                      "24GB 的显存账要重算，先别往下跑。")
        import torch
        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        result["device"] = torch.cuda.get_device_name(0)
        print(f"→ torch {torch.__version__} ｜ cuda {torch.version.cuda} ｜ {result['device']}")
        if not torch.cuda.is_available():
            print("  ❌ torch 看不见 CUDA。先 `pip install` 对应版本，别往下跑。")
            return 1
    except Exception as e:                                   # noqa: BLE001
        print(f"❌ 硬件检查失败：{type(e).__name__}: {e}")
        return 1

    # ---------------------------------------------------------- ② 加载
    banner("② 加载模型")
    try:
        import torch
        from train.modeling import load_model
        from train.tokenize import check_tokenizer_matches, load_tokenizer, tool_schemas
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model = load_model(args.model, dtype="bfloat16", device_map="cuda")
        dt = time.time() - t0
        result["load_sec"] = round(dt, 1)
        result["load_mem_gb"] = round(mem_gb(), 2)
        print(f"→ 加载用时 {dt:.1f}s ｜ 权重占显存 {mem_gb():.2f} GB")
        # 🔴 先对账 tokenizer —— 本地那份词表和 Qwen2.5 的**不一样**（151,665 vs 151,936）。
        #    不同源的话 token id 对不上，模型看到的是乱码，但 loss 照样降，评测时才发现。
        result["tokenizer"] = check_tokenizer_matches(args.model)
        tok = load_tokenizer(model_path=args.model)
        tools = tool_schemas()
        print(f"→ 训练用 tokenizer 词表 {len(tok)} ｜ 工具 {len(tools)} 个")
        del model
        torch.cuda.empty_cache()
    except Exception as e:                                   # noqa: BLE001
        print(f"❌ 加载失败：{type(e).__name__}: {e}")
        return 1

    # ---------------------------------------------------------- ③ 采样 ★
    batch = None
    banner("③ 采样 —— 最关键的三个数字")
    try:
        if args.skip_sample:
            # 只量训练显存的话不用重新采样，④ 会用 SFT 数据造 batch —— 省 3 分钟 + 一次 vLLM 启动
            raise _SkipStep
        import torch
        from env import TASKS
        from train.rollout_batch import RolloutConfig, run_grouped_rollout

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        engine = None
        init_s = 0.0
        engine_kind = "vllm"
        try:
            from train.engine import VLLMEngine
            engine = VLLMEngine(args.model, tok, gpu_memory_utilization=0.85,
                                max_model_len=args.max_seq_tokens)
            init_s = time.time() - t0
            print(f"→ vLLM 启动 {init_s:.1f}s")
        except Exception as ve:                              # noqa: BLE001
            # ⚠️ vLLM 的轮子是跟着 CUDA 大版本走的（现在默认 cu130）。
            #    如果镜像的 CUDA / 驱动和它对不上，import 或启动就会炸，
            #    典型报错是 `libcudart.so.13`。
            #    **这时候不该让整个烟雾测试失败** —— 退回 HF 引擎照样能拿到
            #    显存、流程、每条轨迹耗时这些关键数字，只是吞吐会悲观很多。
            engine_kind = "hf"
            print(f"⚠️ vLLM 起不来：{type(ve).__name__}: {str(ve)[:200]}")
            print("   → 退回 transformers 引擎。吞吐数字会**明显偏悲观**，")
            print("     但显存账和流程正确性照样能验。想修 vLLM 的话，")
            print("     先看报错里有没有 libcudart / cuda 字样 —— 那是镜像 CUDA 版本的问题。")
            from train.engine import HFEngine
            from train.modeling import load_model as _lm
            _m = _lm(args.model, dtype="bfloat16", device_map="cuda")
            engine = HFEngine(_m, tok, device="cuda")
        result["engine"] = engine_kind

        items = [(i, TASKS[i]) for i in range(min(args.tasks, len(TASKS)))]
        cfg = RolloutConfig(n_group=args.n, max_seq_tokens=args.max_seq_tokens)
        t0 = time.time()
        batch = run_grouped_rollout(items, engine, cfg, tok, tools)
        dt = time.time() - t0

        n_ep = len(batch.episodes)
        st = engine.stats()
        result.update({
            "vllm_init_sec": round(init_s, 1),
            "rollout_sec": round(dt, 1),
            "n_episodes": n_ep,
            "sec_per_episode": round(dt / max(1, n_ep), 2),
            "new_tokens": st.total_new_tokens,
            "tok_per_sec": round(st.total_new_tokens / max(1e-9, dt), 1),
            "peak_mem_rollout_gb": round(device_used_gb(), 2),
            "zero_var_rate": batch.zero_var_rate,
            "reward_mean": batch.summary().get("reward_mean", 0.0),
        })
        print(f"→ {n_ep} 条轨迹用了 {dt:.1f}s")
        print(f"  ⭐ 每条轨迹 {dt/max(1,n_ep):.2f}s")
        print(f"  ⭐ 生成速度 {result['tok_per_sec']} tok/s")
        print(f"  ⭐ 采样峰值显存 {result['peak_mem_rollout_gb']} GB")
        print(f"→ 零方差率 {batch.zero_var_rate:.2f} ｜ reward 均值 "
              f"{result['reward_mean']:.2f}（基座模型大概率很低，正常）")

        # ---- 预算换算
        per = dt / max(1, n_ep)
        for label, ntraj in (("一轮 31 题 ×8 条", 31 * 8), ("一轮 + 探针 50 题 ×8", 50 * 8)):
            mins = per * ntraj / 60
            print(f"  · {label}：约 {mins:.1f} 分钟采样")
        del engine
        torch.cuda.empty_cache()
    except _SkipStep:
        print("  （--skip-sample：已跳过，④ 会用 SFT 数据造 batch）")
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"❌ 采样失败：{type(e).__name__}: {e}")
        print("   排查顺序：① vLLM 版本和 torch 版本对得上吗 ② 显存够不够 "
              "③ 是不是忘了 stop=['<|im_end|>']（会一直自问自答）")
        result["sample_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    # ---------------------------------------------------------- ④ 训练
    if not args.skip_train:
        banner("④ 训练一步 —— 记峰值显存（判据 ≤ 13 GB）")
        try:
            import torch
            from train.modeling import (attach_lora, collate,
                                        enable_grad_checkpointing, load_model,
                                        trainable_report)
            from train.tokenize import encode
            from train.trainer import grpo_step

            # 🔴 **必须开梯度检查点** —— 2026-09-17 在 4090D 上实测踩到：
            #    没开的话一条 8192 token 的轨迹前向就把 24 GB 吃满（PyTorch 分配 22.5 GB）→ OOM。
            #    原理：不检查点要保存**每一层**的中间激活（28 层 × 8192 token），
            #          开了只存层边界、反向时重算 → 激活从 ~20 GB 降到 ~2-3 GB。
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            torch.cuda.empty_cache()
            model = load_model(args.model, dtype="bfloat16", device_map="cuda")
            lora = attach_lora(model, r=16, alpha=32).to("cuda")
            enable_grad_checkpointing(lora)
            lora.train()
            print(f"→ LoRA 可训练参数 {trainable_report(lora)['ratio_pct']}（已开梯度检查点）")

            if batch is not None:
                msgs = batch.episodes[0].messages
            else:
                # --skip-sample：拿一条真实 SFT 轨迹造 batch，序列长度分布一致
                import json as _json
                _p = PROJECT / "data" / "sft_final.jsonl"
                _rows = [_json.loads(l) for l in _p.read_text(encoding="utf-8").splitlines()
                         if l.strip()]
                msgs = _rows[0]["messages"]
                print(f"  （用 SFT 数据造 batch：{len(msgs)} 条消息）")

            # ⭐ 逐档降序列长度重试 —— **保证一定能拿到一个数字**，
            #    而不是撞上 OOM 就什么都测不到（第一次就是这么白跑的）
            ok = False
            for seq_len in (args.max_seq_tokens, 4096, 2048):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    enc = encode(msgs, tok, tools, max_length=seq_len)
                    b = collate([enc], pad_id=tok.pad_token_id or 0).to("cuda")
                    print(f"→ 试 {b.input_ids.shape[1]} token"
                          f"（算 loss 的 {b.n_loss_tokens}）…")
                    t0 = time.time()
                    loss, _ = grpo_step(lora, b, advantage=1.0)
                    loss.backward()
                    dt = time.time() - t0
                    result.update({
                        "train_seq_tokens": int(b.input_ids.shape[1]),
                        "train_step_sec": round(dt, 1),
                        "peak_mem_train_gb": round(peak_gb(), 2),
                        "loss": float(loss.item()),
                    })
                    print(f"→ 一步（前向+反向）{dt:.1f}s ｜ loss {loss.item():.4f}")
                    print(f"⭐ 训练峰值显存 {result['peak_mem_train_gb']} GB"
                          f"（序列 {b.input_ids.shape[1]}）")
                    if result["peak_mem_train_gb"] > 13:
                        print("  ⚠️ 超过 13 GB —— 检查是不是漏了分块 CE / 梯度检查点")
                    else:
                        print("  ✅ 在预算内")
                    ok = True
                    break
                except torch.OutOfMemoryError:
                    print(f"  ⚠️ {seq_len} token 装不下，降一档重试")
                    lora.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            if not ok:
                print("  ❌ 降到 2048 仍然 OOM —— 梯度检查点可能没真正生效")
        except Exception as e:                               # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"❌ 训练步失败：{type(e).__name__}: {e}")
            # ⚠️ 记进报告 —— 2026-09-17 踩：报错只打在屏幕上没进 JSON，
            #    结果拿到的报告里只有 ①② 的数据，看不出 ③④ 到底为什么没跑
            result["train_error"] = f"{type(e).__name__}: {str(e)[:300]}"

    # ---------------------------------------------------------- 汇总
    banner("汇总（这些数字决定租不租第二张卡）")
    for k in ("gpu_total_gb", "load_sec", "vllm_init_sec", "sec_per_episode",
              "tok_per_sec", "peak_mem_rollout_gb", "train_step_sec",
              "peak_mem_train_gb"):
        if k in result:
            print(f"  {k:<24} {result[k]}")
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ 报告写进 {REPORT}")
    print("→ 取回这份 JSON，据此定采样批量、训练步数和总预算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
