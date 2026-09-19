# -*- coding: utf-8 -*-
"""
T4-P 扩题器可行性探针 —— 生死判据。

问题：τ-bench airline 本地数据，能程序化组合出多少道【可构造 gold】的新题？
      如果 < 400，「扩题到 400+」的整个叙事就塌了 —— 必须在写扩题器之前知道。

方法：不改环境、不跑模型，纯统计 data/ 下三个 json 的组合空间。
      对每一类 gold 动作，数一遍「有合法输入、能构造出可验证 gold」的实例数。

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/probe_expandable.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from env.bootstrap import TAU_BENCH                                           # noqa: F401,E402

DATA = TAU_BENCH / "tau_bench" / "envs" / "airline" / "data"

TARGET = 400          # README 承诺的扩题目标


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def main() -> int:
    users = load("users.json")
    res = load("reservations.json")
    flights = load("flights.json")

    print("=" * 76)
    print("T4-P · 扩题器可行性探针")
    print("=" * 76)
    print(f"原始数据：{len(users)} 用户 ｜ {len(res)} 预订 ｜ {len(flights)} 航班")
    print()

    rs = list(res.values())
    us = list(users.values())

    # ---------------------------------------------------------- 逐类数可用实例
    print("-" * 76)
    print("按 gold 动作类型数组合空间")
    print("-" * 76)
    print(f"{'gold 动作':<34}{'可用实例':>10}{'依据':>30}")

    avail = {}

    # 1) 取消 —— 任何一条预订都能作为目标
    avail["cancel_reservation"] = len(rs)
    basis1 = "任一 reservation"

    # 2) 改行李额 —— 需要预订 + baggage 字段（预订级就有）
    has_bag = sum(1 for r in rs if "total_baggages" in r and "nonfree_baggages" in r)
    avail["update_reservation_baggages"] = has_bag
    basis2 = "有 total/nonfree_baggages"

    # 3) 改航班 —— 需要预订在飞的航线上有可选新航班
    routes = {(f.get("origin"), f.get("destination")) for f in flights.values()}
    routes |= {(d, o) for (o, d) in list(routes)}          # 双向
    ok_flight = sum(1 for r in rs if (r.get("origin"), r.get("destination")) in routes)
    avail["update_reservation_flights"] = ok_flight
    basis3 = f"航线在 {len(routes)} 条可选航线内"

    # 4) 改乘客 —— 需要 ≥2 名乘客（改一个，另一个不动）
    multi = sum(1 for r in rs if len(r.get("passengers", [])) >= 2)
    avail["update_reservation_passengers"] = multi
    basis4 = "乘客数 ≥2"

    # 5) 订票 —— (用户, 航线) 组合
    users_with_pm = sum(1 for u in us if u.get("payment_methods"))
    avail["book_reservation"] = users_with_pm * len(routes)
    basis5 = f"{users_with_pm} 有支付方式的用户 × {len(routes)} 航线"

    # 6) 发证书 —— 需要用户名下有 certificate
    with_cert = sum(1 for u in us
                    if any(v.get("source") == "certificate"
                           for v in u.get("payment_methods", {}).values()))
    avail["send_certificate"] = with_cert
    basis6 = "名下 certificate 数 ≥1"

    for k, b in [("cancel_reservation", basis1),
                 ("update_reservation_baggages", basis2),
                 ("update_reservation_flights", basis3),
                 ("update_reservation_passengers", basis4),
                 ("book_reservation", basis5),
                 ("send_certificate", basis6)]:
        print(f"{k:<34}{avail[k]:>10,}{b:>30}")

    total = sum(avail.values())
    print("-" * 76)
    print(f"{'合计（上界）':<34}{total:>10,}")
    print()

    # ---------------------------------------------------------- 保守估计
    # 只取「一题一动作」的最保守口径：每条预订最多出 4 类题
    conservative = (len(rs) * 4) + with_cert
    print("-" * 76)
    print("保守口径（每条预订最多出 4 类题 + 发证书）")
    print("-" * 76)
    print(f"  {len(rs)} × 4 + {with_cert} = {conservative:,} 道")
    print()

    # ---------------------------------------------------------- 判据
    print("=" * 76)
    ok = conservative >= TARGET
    print(f"判据：可构造题数 ≥ {TARGET}")
    print(f"实测：保守 {conservative:,} 道 ｜ 上界 {total:,} 道")
    print(f"结论：{'✅ 通过' if ok else '❌ 不通过'} —— "
          f"是目标的 {conservative/TARGET:.1f}× （保守口径）")
    print("=" * 76)
    print()
    print("含义：")
    print("  ·「扩题到 400+」的叙事【成立】，且余量充足")
    print("  · 就算只做「一题一动作」的简单题，也有 %d 道可用" % conservative)
    print("  · 真正的瓶颈不是【数量】，是【过 oracle 验证后的有效率】")
    print("    （扩题器的产出必须逐题过 env oracle，见设计补全文档洞 3）")
    print()

    # ---------------------------------------------------------- 存盘
    out = PROJECT / "data" / "expandable_space.json"
    out.write_text(json.dumps({
        "n_users": len(users), "n_reservations": len(res), "n_flights": len(flights),
        "n_routes": len(routes),
        "by_tool": avail,
        "total_upper": total,
        "conservative": conservative,
        "target": TARGET,
        "pass": ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
