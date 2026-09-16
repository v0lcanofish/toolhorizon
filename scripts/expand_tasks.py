# -*- coding: utf-8 -*-
"""
H2-c · 扩题器 —— 程序化造新题。

为什么要造题：
    原始 50 道太少了（训练集才 31 道）。模型见不到足够多的题，学不出泛化。
    实测本地数据（500 用户 / 2000 预订 / 300 航班）能拼出 8000+ 道。

⭐ 核心原则：**不是生成即用，每道题都要过环境验证。**
    对每道新造的题，在环境里跑两次：
      · 什么都不做  → reward 必须 = 0.0   （这题要"有区分度"）
      · gold 重放   → reward 必须 = 1.0   （gold 动作本身是对的）
    两条都过才留下，否则丢弃。

三类难度档：
    easy    单动作，指令里直接给 ID
    medium  2–3 个动作
    hard    指令不给 ID，agent 得自己先查（gold 里带查询动作）

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 D:/anaconda/envs/myenv/python.exe scripts/expand_tasks.py --n 200
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env import build_env_custom                                          # noqa: E402
from env.tau_env import load_data_cached                                  # noqa: E402
from tau_bench.types import Task, Action                                  # noqa: E402

OUT = PROJECT / "data" / "tasks_expanded.json"

FIRST = ["Aarav", "Chen", "Sofia", "Omar", "Yara", "Raj", "Ivan", "Mia", "Noah", "Zara",
         "Liam", "Aisha", "Diego", "Emma", "Hiro", "Nina", "Sam", "Tara", "Leo", "Ana"]
LAST = ["Garcia", "Rossi", "Silva", "Kim", "Brown", "Ahmed", "Jackson", "Martin",
        "Nguyen", "Patel", "Lopez", "Cohen", "Khan", "Wang", "Singh", "Muller"]


# ---------------------------------------------------------------- 造题原语


def op_cancel(r, rng):
    """取消预订（单动作）"""
    instr = (f"Your user id is {r['user_id']}. "
             f"You want to cancel your upcoming reservation to {r['destination']}.")
    acts = [Action(name="cancel_reservation",
                   kwargs={"reservation_id": r["reservation_id"]})]
    return instr, acts


def op_change_passenger(r, rng):
    """改乘客姓名（单动作）"""
    if not r["passengers"]:
        return None
    ps = [dict(p) for p in r["passengers"]]
    old_first = ps[0].get("first_name", "")
    # 同样要保证新名 ≠ 原名，否则动作不改变数据库
    cand = [f for f in FIRST if f != old_first]
    ps[0]["first_name"] = rng.choice(cand or FIRST)
    ps[0]["last_name"] = rng.choice(LAST)
    instr = (f"Your user id is {r['user_id']}. "
             f"You want to fix the spelling of the passenger name on your "
             f"reservation to {r['destination']}.")
    acts = [Action(name="update_reservation_passengers",
                   kwargs={"reservation_id": r["reservation_id"], "passengers": ps})]
    return instr, acts


def op_change_baggages(r, rng, users):
    """改行李额（单动作）—— payment_id 必须属于该用户"""
    u = users.get(r["user_id"])
    if not u or not u.get("payment_methods"):
        return None

    # ⚠️ 踩坑（第二次）：certificate 不能用于【修改】预订 ——
    #    环境会返回 "Error: certificate cannot be used to update reservation"，
    #    动作被拒 → 数据库没变 → gold 重放同样是 no-op → 两边哈希相同 → 这题 reward 恒为 1.0。
    #    （和 H1 那个「动作失败不改变区分度、只让题目失效」是同一类机制）
    usable = [pid for pid, pm in u["payment_methods"].items()
              if pm.get("source") != "certificate"]
    if not usable:
        return None
    pay_id = rng.choice(usable)

    # ⚠️ 踩坑：新值必须 ≠ 原值，否则动作等于没改数据库 → 这题没区分度
    #    （实测现有预订里 nonfree 全是 0，total 有 44% 落在我原来的取值范围内）
    cur_total = r.get("total_baggages", 0)
    cur_nonfree = r.get("nonfree_baggages", 0)
    total = rng.choice([t for t in (1, 2, 3, 4, 5, 6) if t != cur_total] or [cur_total + 1])
    nonfree = rng.choice([n for n in (0, 1, 2) if n != cur_nonfree and n <= total] or [0])
    if total == cur_total and nonfree == cur_nonfree:
        return None

    instr = (f"Your user id is {r['user_id']}. "
             f"You want to change the number of checked bags on your reservation "
             f"to {r['destination']} to {total} total.")
    acts = [Action(name="update_reservation_baggages",
                   kwargs={"reservation_id": r["reservation_id"],
                           "total_baggages": total, "nonfree_baggages": nonfree,
                           "payment_id": pay_id})]
    return instr, acts


def op_send_certificate(u, rng):
    """发补偿证书（单动作，只跟用户有关）"""
    amount = rng.choice([50, 100, 150, 200, 250, 300])
    instr = (f"Your user id is {u['id']}. "
             f"You had a bad experience and want a ${amount} travel certificate.")
    acts = [Action(name="send_certificate",
                   kwargs={"user_id": u["id"], "amount": amount})]
    return instr, acts


OPS = {
    "cancel": op_cancel,
    "change_passenger": op_change_passenger,
    "change_baggages": op_change_baggages,
    "send_certificate": op_send_certificate,
}


# ---------------------------------------------------------------- 组装 + 验证


def _call(kind, r, u, rng):
    """统一调用入口：send_certificate 吃 user，其余吃 reservation。"""
    if kind == "send_certificate":
        return op_send_certificate(u, rng)
    if kind == "change_baggages":
        return op_change_baggages(r, rng, {r["user_id"]: u})
    return OPS[kind](r, rng)


def make_task(kind, difficulty, rng, reservations, users):
    """按难度档组装一道题。返回 Task 或 None。"""
    rid, r = rng.choice(reservations)
    u = users.get(r["user_id"])
    if u is None:
        return None

    got = _call(kind, r, u, rng)
    if not got:
        return None
    instr, acts = got

    if difficulty == "medium":
        # 拼一个同预订、不同类型的第二动作
        pool = [k for k in ("cancel", "change_passenger", "change_baggages")
                if k != kind and k != "send_certificate"]
        if not pool:
            return None
        k2 = rng.choice(pool)
        got2 = _call(k2, r, u, rng)
        if not got2:
            return None
        instr = instr + " " + got2[0].split(". ", 1)[-1]
        acts = acts + got2[1]

    return Task(user_id=r["user_id"], instruction=instr, actions=acts, outputs=[])


def validate(task):
    """⭐ 两道判据：noop 必须 0 分，gold 重放必须满分。"""
    e1 = build_env_custom(task)
    e1.reset(task_index=0)
    r_noop = e1.calculate_reward().reward

    e2 = build_env_custom(task)
    e2.reset(task_index=0)
    try:
        for a in task.actions:
            e2.step(a)
    except Exception:                                     # noqa: BLE001
        return None, None
    r_gold = e2.calculate_reward().reward
    return r_noop, r_gold


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="目标题数（验证通过的口径）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-try", type=int, default=8000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = load_data_cached()
    reservations = list(data["reservations"].items())
    users = {uid: {"id": uid, **u} for uid, u in data["users"].items()}

    print("=" * 76)
    print("H2-c · 扩题器")
    print("=" * 76)
    print(f"数据：{len(reservations)} 条预订 ｜ {len(users)} 个用户")
    print(f"目标：{args.n} 道（必须过验证）｜ seed={args.seed}\n")

    kinds = list(OPS)
    kept, tried, drops = [], 0, Counter()
    dropped_detail = []
    while len(kept) < args.n and tried < args.max_try:
        tried += 1
        kind = rng.choice(kinds)
        # send_certificate 是【用户级】动作，跟预订级动作拼不到一起 → 只出 easy
        diff = "easy" if kind == "send_certificate" else rng.choice(["easy", "easy", "medium"])
        t = make_task(kind, diff, rng, reservations, users)
        if t is None:
            drops["造题失败"] += 1
            continue
        r_noop, r_gold = validate(t)
        if r_gold is None:
            drops["动作执行报错"] += 1
            continue
        if r_gold != 1.0:
            drops["gold 重放没满分"] += 1
            continue
        if r_noop != 0.0:
            drops["没区分度(不做也满分)"] += 1
            dropped_detail.append({
                "kind": kind, "difficulty": diff, "user_id": t.user_id,
                "instruction": t.instruction,
                "actions": [(a.name, a.kwargs) for a in t.actions],
                "r_noop": r_noop, "r_gold": r_gold,
            })
            continue
        kept.append({"kind": kind, "difficulty": diff,
                     "user_id": t.user_id, "instruction": t.instruction,
                     "actions": [{"name": a.name, "kwargs": a.kwargs}
                                 for a in t.actions],
                     "outputs": t.outputs})
        if len(kept) % 25 == 0:
            print(f"  已通过 {len(kept)} 道（试了 {tried} 次）")

    print()
    print("-" * 76)
    print("结果")
    print("-" * 76)
    print(f"  尝试            {tried:>5} 次")
    print(f"  ✅ 通过验证      {len(kept):>5} 道"
          f"   （有效率 {len(kept)/max(tried,1)*100:.1f}%）")
    print(f"  ❌ 丢弃          {tried - len(kept):>5} 道")
    if drops:
        print("     丢弃原因：")
        for k, v in drops.most_common():
            print(f"       {k:<24} {v:>5}")

    print()
    print("  难度分布：", dict(Counter(r["difficulty"] for r in kept)))
    print("  题型分布：", dict(Counter(r["kind"] for r in kept)))

    OUT.write_text(json.dumps({
        "n": len(kept), "seed": args.seed, "tried": tried,
        "drops": dict(drops), "dropped_detail": dropped_detail, "tasks": kept,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"✅ 已写出 {len(kept)} 道 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
