# -*- coding: utf-8 -*-
"""
H2-b · 合成缺失的轨迹（训练集里那 7 道没有真实轨迹的题）。

为什么要合成：
    268 条真实轨迹只覆盖了训练集 31 道里的 24 道。
    剩下 7 道（task 3/8/19/22/23/25/33）没有现成数据 —— 但 SFT 需要每道题都见过。

怎么合成：
    1. 在环境里【重放 gold 动作】，把每一步的工具返回抓下来
    2. 拼成对话：system → user(指令) → [assistant(工具调用) → tool(返回)]×n → assistant(收尾)
    3. ⚠️ 收尾那句话是【硬约束】：
       - task[8] 有 outputs=['327','1000','1786'] → 收尾必须包含这三个数
       - 其余 6 道无 outputs → 用模板（轮换几种，避免模型背同一句）

⭐ 判据：合成出的轨迹丢回环境重放，reward 必须 = 1.0

跑法（必须用 myenv，要 τ-bench）：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/synth_sft_data.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import build_env, TASKS                                     # noqa: E402
from env.bootstrap import TAU_BENCH                                  # noqa: F401,E402
from tau_bench.types import Action, RESPOND_ACTION_NAME               # noqa: E402

TRAJ_DIR = (PROJECT.parent / "reference-repos" / "agentic-grpo-longhorizon"
            / "tau-bench" / "historical_trajectories")

MISSING = [3, 8, 19, 22, 23, 25, 33]
OUT = PROJECT / "data" / "sft_synth.jsonl"

# 收尾模板：轮换用，避免所有轨迹结尾一模一样
DONE_TEMPLATES = [
    "I've successfully completed all the changes you requested. Is there anything else I can help you with?",
    "All done! Your requests have been processed. Let me know if there's anything else I can do for you.",
    "Everything you asked for has been taken care of. Feel free to reach out if you need further assistance.",
]


def system_prompt() -> str:
    """系统提示从真实轨迹里取——同一领域共用一份。"""
    for fn in ["gpt-4o-airline.json", "sonnet-35-new-airline.json"]:
        d = json.loads((TRAJ_DIR / fn).read_text(encoding="utf-8"))
        for x in d:
            if x.get("reward") == 1.0:
                m = x["traj"][0]
                if m.get("role") == "system":
                    return m["content"]
    raise RuntimeError("取不到 system prompt")


def synth(task_index: int, sysp: str) -> dict:
    """在环境里重放 gold，抓工具返回，拼成完整对话。"""
    task = TASKS[task_index]
    env = build_env(task_index)
    env.reset(task_index=task_index)

    msgs = [{"role": "system", "content": sysp},
            {"role": "user", "content": task.instruction}]

    for k, a in enumerate(task.actions):
        resp = env.step(a)
        obs = getattr(resp, "observation", None)
        if obs is None:
            obs = str(resp)
        # 工具调用轮：content 为空，与真实轨迹一致（实测真实轨迹里也是这样）
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{task_index}_{k}",
                "type": "function",
                "function": {"name": a.name,
                             "arguments": json.dumps(a.kwargs, ensure_ascii=False)},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"call_{task_index}_{k}",
                     "name": a.name, "content": obs})

    # ---- 收尾回复：有 outputs 的必须嵌进去（硬约束）
    if task.outputs:
        text = "Here is the information you asked for: " + \
               ", ".join(task.outputs) + "."
    else:
        text = DONE_TEMPLATES[task_index % len(DONE_TEMPLATES)]

    # ⚠️ 踩坑（第一版就栽在这）：光把收尾写进 messages 没用 ——
    #    环境的 outputs 检查看的是 env.actions（真的执行过的动作），
    #    必须【真的 step 一次 respond】，否则 task[8] 会判 0 分。
    env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": text}))
    msgs.append({"role": "assistant", "content": text})

    reward = env.calculate_reward().reward
    return {"task_id": task_index, "source": "synth",
            "n_messages": len(msgs),
            "n_assistant": sum(1 for m in msgs if m["role"] == "assistant"),
            "n_tool_calls": len(task.actions),
            "final_resp_chars": len(text),
            "outputs": task.outputs,
            "reward_check": reward,
            "messages": msgs}


def main() -> int:
    print("=" * 76)
    print("H2-b · 合成缺失轨迹（训练集里没有真实轨迹的 7 道）")
    print("=" * 76)

    sysp = system_prompt()
    print(f"system prompt 取自真实轨迹，长度 {len(sysp)} 字符\n")

    rows, bad = [], []
    for i in MISSING:
        r = synth(i, sysp)
        rows.append(r)
        mark = "✅" if r["reward_check"] == 1.0 else "❌"
        print(f"  {mark} task[{i:>2}]  动作 {r['n_tool_calls']} 条  "
              f"消息 {r['n_messages']} 条  reward={r['reward_check']}  "
              f"outputs={r['outputs'] or '—'}")
        if r["reward_check"] != 1.0:
            bad.append(i)

    print()
    if bad:
        print(f"❌ {len(bad)} 道合成后重放没拿满分：{bad}")
        print("   → 判据没达成，不能写盘")
        return 1

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 76)
    print(f"✅ 判据达成：{len(rows)}/{len(rows)} 条合成轨迹重放后 reward = 1.0")
    print(f"   已写出 → {OUT}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
