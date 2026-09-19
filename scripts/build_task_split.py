# -*- coding: utf-8 -*-
"""
T2 · 把信号形状分析结果，变成训练/评测可执行的【分流清单】。

输入：data/signal_shape.json   （probe_signal_shape.py 的产出）
输出：data/task_split.json

分流的唯一依据是 class 字段：
    S（有区分度）  → 训练集
    O（单边信号）  → 评测探针集（专测「过调用」）
    N（真无信号）  → 若有，两者都剔除（当前实测为 0 道）

为什么不能合成一份：
    O 类在【训练】里是毒药 —— 奖励与策略行为无关，全组不动就 std=0 零梯度，
    一旦有 rollout 乱动，梯度指向「别调工具」。
    但它们在【评测】里是资产 —— 是全部 50 道题里唯一能测「过调用」的一批。
    所以：训练剔除、评测保留。

跑法：
  cd 代码库/projects/ToolHorizon
  PYTHONIOENCODING=utf-8 python3.13 scripts/build_task_split.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SRC = PROJECT / "data" / "signal_shape.json"
DST = PROJECT / "data" / "task_split.json"

TRANSFER_TOOL = "transfer_to_human_agents"


def load_tasks():
    """拿原始 task 对象，用来判「只有转人工」这种子类。"""
    sys.path.insert(0, str(PROJECT))
    from env import TASKS                                          # noqa: E402
    return TASKS


def main() -> int:
    if not SRC.exists():
        print(f"❌ 缺输入 {SRC}\n   先跑 scripts/probe_signal_shape.py")
        return 1

    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = data["rows"]
    TASKS = load_tasks()

    print("=" * 74)
    print("T2 · 题分流清单")
    print("=" * 74)
    print(f"输入 {SRC.name}：{len(rows)} 道题 ｜ 分类 {data['counts']}\n")

    train, probe, drop = [], [], []
    for r in rows:
        i = r["task"]
        rec = {
            "task": i,
            "user_id": r["user_id"],
            "class": r["class"],
            "n_write": r["n_write"],
            "n_outputs": r["n_outputs"],
            "r_noop": r["r_noop"],
            "r_gold": r["r_gold"],
            "r_foreign": r["r_foreign"],
        }
        if r["class"] == "S":
            # 训练集：再标一个子类，方便扩题时对齐配比
            rec["kind"] = "communicate_only" if r["n_write"] == 0 else "write"
            train.append(rec)
        elif r["class"] == "O":
            # 探针集：按 gold 里的动作构成分子类
            # ⚠️ 坑：不能用 `names <= {TRANSFER_TOOL}` —— 空集是任何集合的子集，
            #    「零动作题」会被误判成 transfer_only（本脚本第一版就踩了，
            #    所以显示 9 道而真实只有 2 道）。
            #    必须先显式判空。
            names = [a.name for a in TASKS[i].actions]
            if not names:
                rec["subtype"] = "zero_action"          # gold 里一条动作都没有
            elif set(names) == {TRANSFER_TOOL}:
                rec["subtype"] = "transfer_only"        # 只有转人工
            else:
                rec["subtype"] = "readonly_only"        # 只有只读动作，不改库
            probe.append(rec)
        else:
            drop.append(rec)

    # ---------------------------------------------------------------- 报告
    print("-" * 74)
    print("分流结果")
    print("-" * 74)
    print(f"{'用途':<22}{'题数':>6}   说明")
    print(f"{'训练集 (train)':<22}{len(train):>6}   S 类：不做 0 分、做对满分")
    print(f"{'评测探针 (probe)':<22}{len(probe):>6}   O 类：不做满分、乱做 0 分 → 专测过调用")
    if drop:
        print(f"{'剔除 (drop)':<22}{len(drop):>6}   N 类：怎么都能过")
    else:
        print(f"{'剔除 (drop)':<22}{0:>6}   ——（实测为 0 道）")
    print()

    # 子分组
    kc = Counter(r["kind"] for r in train)
    print("训练集子分组：")
    print(f"   write（要改数据库）      {kc.get('write', 0):>3} 道")
    print(f"   communicate_only（只汇报） {kc.get('communicate_only', 0):>3} 道"
          f"   ← 最稀缺的一类")
    kp = Counter(r["subtype"] for r in probe)
    print("探针集子分组（按 gold 里的动作构成）：")
    print(f"   zero_action（一条动作都没有） {kp.get('zero_action', 0):>3} 道")
    print(f"   transfer_only（只有转人工）   {kp.get('transfer_only', 0):>3} 道")
    print(f"   readonly_only（只查不改）      {kp.get('readonly_only', 0):>3} 道")
    print()

    # 带 outputs 的题（这些题"必须汇报"才有分）
    with_out = [r["task"] for r in rows if r["n_outputs"] > 0]
    print(f"带 outputs（必须汇报才算完成）的题：{with_out}  共 {len(with_out)} 道")
    print(f"   → 占全部 50 道的 {len(with_out)/len(rows)*100:.0f}%")
    print()

    # ---------------------------------------------------------------- 断言
    assert len(train) + len(probe) + len(drop) == len(rows), "分流总数对不上"
    assert not (set(r["task"] for r in train) & set(r["task"] for r in probe)), \
        "训练集和探针集有重叠"
    for r in train:
        assert r["r_noop"] == 0.0 and r["r_gold"] == 1.0, f"task{r['task']} 不满足 S 判据"
    for r in probe:
        assert r["r_noop"] == 1.0 and r["r_foreign"] == 0.0, f"task{r['task']} 不满足 O 判据"

    out = {
        "source": SRC.name,
        "n_total": len(rows),
        "counts": {"train": len(train), "probe": len(probe), "drop": len(drop)},
        "criteria": {
            "train": "class==S  (noop=0 且 gold=1)",
            "probe": "class==O  (noop=1 且 foreign=0)",
            "drop": "class==N  (noop=1 且 foreign=1)",
        },
        "train": train,
        "probe_overcall": probe,
        "drop": drop,
        "tasks_with_outputs": with_out,
        "note": (
            "probe_overcall 专测「过调用」：在这 19 道题上，只要模型执行了任何写动作，"
            "就说明它在没事找事。训练集不含它们。"
        ),
    }
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print(f"✅ 断言全过：训练 {len(train)} + 探针 {len(probe)} + 剔除 {len(drop)} = {len(rows)}")
    print(f"清单已存：{DST}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
