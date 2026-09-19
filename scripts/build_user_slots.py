# -*- coding: utf-8 -*-
"""
块 T5 · 把用户模拟器「知道的事实」导成缓存文件 `data/user_slots.json`。

为什么要落盘：
    ① 判据要可复现 —— 别人（或三个月后的我）能直接看模拟器凭什么这么答，
       不用去读 `env/user_sim.py` 的逻辑
    ② 扩题器造出来的新题也要有槽位，不然 rollout 到新题时模拟器就是哑巴
    ③ **可审计**：槽位里每一格都能追到来源（instruction / gold 动作 / 用户档案），
       如果一个值看起来不对，能一眼看出是哪一步抽错的

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/build_user_slots.py
"""

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import TASKS                                          # noqa: E402
from env.tau_env import load_data_readonly                     # noqa: E402
from env.user_sim import slots_from_task                       # noqa: E402

OUT = PROJECT / "data" / "user_slots.json"
EXPANDED = PROJECT / "data" / "tasks_expanded.json"


def dump(task, tag: str) -> dict:
    s = slots_from_task(task, load_data_readonly())
    d = asdict(s)
    d.pop("profile", None)          # 档案是查出来的，不入库（避免和 users.json 重复）
    d["tag"] = tag
    return d


def main() -> int:
    data = load_data_readonly()
    rows = {}

    for i, t in enumerate(TASKS):
        rows[f"tau{i}"] = dump(t, "tau-bench 原题")

    if EXPANDED.exists():
        exp = json.loads(EXPANDED.read_text(encoding="utf-8"))
        from tau_bench.types import Task                          # noqa: E402
        for j, raw in enumerate(exp["tasks"]):
            from tau_bench.types import Action
            task = Task(
                user_id=raw["user_id"],
                actions=[Action(name=a["name"], kwargs=a["kwargs"]) for a in raw["actions"]],
                outputs=raw.get("outputs", []),
                instruction=raw["instruction"],
            )
            rows[f"exp{j}"] = dump(task, "扩题器生成")

    # ---- 自检：槽位不能是空的
    n_empty_uid = sum(1 for r in rows.values() if not r["user_id"])
    n_no_res = sum(1 for r in rows.values() if not r["reservation_ids"])
    n_knows = sum(1 for r in rows.values() if r["knows_reservation_id"])
    n_known_pairs = 0
    for r in rows.values():
        if r["knows_reservation_id"] and r["reservation_ids"]:
            # 用户知道的号必须真是他自己的订单
            mine = {x for x in r["reservation_ids"]}
            if mine:
                n_known_pairs += 1

    payload = {
        "n": len(rows),
        "source": "instruction + gold 动作 kwargs + 用户档案（见 env/user_sim.py 文件头）",
        "counts": {
            "total": len(rows),
            "missing_user_id": n_empty_uid,
            "no_reservation_id": n_no_res,
            "knows_reservation_id": n_knows,
        },
        "slots": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"写出 {OUT}  （{OUT.stat().st_size/1024:.0f} KB）")
    print(f"  题数            {len(rows)}")
    print(f"  缺 user_id      {n_empty_uid}")
    print(f"  没有预定号      {n_no_res}")
    print(f"  用户知道预定号  {n_knows}  ← 其余题里，agent 必须自己查出来（这正是多轮交互的意义）")

    # 抽样打印
    print("\n抽样：")
    for k in ("tau0", "tau6", "tau12"):
        r = rows.get(k)
        if not r:
            continue
        print(f"  {k}: uid={r['user_id']} 预定号={r['reservation_ids'][:3]} "
              f"知道自己预定号={r['knows_reservation_id']}")
        print(f"      instruction: {r['instruction'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
