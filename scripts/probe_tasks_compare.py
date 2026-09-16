# -*- coding: utf-8 -*-
"""
探针 2：tasks.py vs tasks_test.py 对不上，要查清哪个是真的。

背景：探针 1 发现 tasks_test.TASKS 里有 7 道题的 gold 动作数是 0，
但它们的 instruction 明显要求改数据库（"cancel your flights..."）。
可疑点：longhorizon 可能改过这个 vendored 副本。

这个脚本只做一件事：把两份任务定义逐题对齐，看差异在哪。

跑法：
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/probe_tasks_compare.py
"""

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
from env.bootstrap import TAU_BENCH                            # noqa: E402
sys.path.insert(0, str(TAU_BENCH))

# 桩掉 litellm（参考仓库只读）
fake = types.ModuleType("litellm")
fake.completion = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("桩被调用"))
fake.provider_list = []
sys.modules.setdefault("litellm", fake)

from tau_bench.envs.airline.tasks_test import TASKS          # noqa: E402  已补全的 pydantic 版

AIRLINE = TAU_BENCH / "tau_bench" / "envs" / "airline"

print("=" * 68)
print("探针 2 · tasks.py vs tasks_test.py")
print("=" * 68)

# tasks.py 是 dict 格式的裸数据，用 exec 读进来（避免 import 触发 airline/__init__ 的 match 语法）
ns = {}
exec((AIRLINE / "tasks.py").read_text(encoding="utf-8"), ns)
raw_tasks = ns["tasks"]

print(f"tasks.py      : {len(raw_tasks):>3} 条（dict 格式）")
print(f"tasks_test.py : {len(TASKS):>3} 条（pydantic Task 格式）")
print()

# 对齐：以 tasks_test 为准，按 user_id 找 tasks.py 里的对应条目
raw_by_uid = {}
for t in raw_tasks:
    raw_by_uid.setdefault(t["user_id"], []).append(t)

print("-" * 68)
print("逐题对比 gold 动作数")
print("-" * 68)
print(f"{'idx':>4} {'user_id':<26} {'test 动作数':>10} {'raw 动作数':>10}  差异")
print("-" * 68)

mismatch = []
for i, t in enumerate(TASKS):
    cands = raw_by_uid.get(t.user_id, [])
    if not cands:
        raw_n = "无此 user_id"
        diff = "⚠️ tasks.py 里找不到这个人"
        mismatch.append((i, t.user_id, len(t.actions), None, diff))
    else:
        # 同一个人可能有多道题，取动作数最接近的比
        raw_n = min(len(c["actions"]) for c in cands)
        same = len(t.actions) == raw_n
        diff = "" if same else "⚠️ 动作数不一致"
        if not same:
            mismatch.append((i, t.user_id, len(t.actions), raw_n, diff))
    print(f"{i:>4} {t.user_id:<26} {len(t.actions):>10} {str(raw_n):>10}  {diff}")

print("-" * 68)
print(f"共 {len(mismatch)} 题对不上")
print()

# 重点：0 步题在 tasks.py 里是什么样
zero_step = [i for i, t in enumerate(TASKS) if len(t.actions) == 0]
print("-" * 68)
print(f"重点核查：tasks_test 里 {len(zero_step)} 道 0 步题，在 tasks.py 里有动作吗？")
print("-" * 68)
for i in zero_step:
    t = TASKS[i]
    cands = raw_by_uid.get(t.user_id, [])
    raw_ns = [len(c["actions"]) for c in cands]
    print(f"  task[{i:>2}] {t.user_id:<26} test=0 步 | tasks.py={raw_ns}")
    if cands and any(n > 0 for n in raw_ns):
        names = [a["name"] for c in cands for a in c["actions"]]
        print(f"            → tasks.py 里的动作：{names}")
print()

print("=" * 68)
print("探针 2 结束")
print("=" * 68)
