# -*- coding: utf-8 -*-
"""
规则式用户模拟器 —— 零 LLM，但**真的会说话**。

━━━ 为什么要有这个文件（v2，2026-09-17 重写）━━━

v1 是占位版：无论 agent 问什么，永远回同一句 "Okay, please go ahead."。
它让 pipeline 能跑通，但会**静默毁掉训练**：

    真实轨迹的 685 个 user 轮里，16.4% 含预订号、13.4% 含 user_id，
    长度 p50 = 98 字符；占位版是固定 24 字符、**零信息**。
    而训练集 31 道题里有 12 道（38.7%）的 gold **第一步就要用到
    instruction 里没给的 ID** —— 只能靠问用户拿到。

    → 占位版下这 12 道题**任何策略都解不了** → 组内全对概率为 0、全错概率极高
      → std=0 → 零梯度白跑。而且它会污染 SFT 的出口判据 pass@1 ∈ [15%,50%]，
      让人误以为是模型不行。

⚠️ 这个坑 H3 当时没暴露，因为照本宣科的假策略**根本不听用户说话**。

━━━ 信息从哪来 ━━━

两个来源，都不是新造的数据：
  ① `task.instruction` —— 用户自己一上来说的话
  ② **`task.actions`（gold 动作）的 kwargs** —— ⭐ 关键：
     gold 里就带着"用户本该提供的那个答案"（要改的那张订单号就在
     `update_reservation_flights.reservation_id` 里）。
     所以模拟器不是凭空编，而是**把 gold 已经蕴含的信息，按对话节奏放出来**。

    ⚠️ 诚实边界：这意味着本模拟器是一个**知道答案的合作用户**，
       比真人更配合。它给出的是 pass rate 的**上界**，不是无偏估计。
       这一点必须写进报告，不能当成"我复现了真人"。

━━━ 怎么判它够不够好 ━━━

`scripts/eval_user_sim.py` 拿 **685 对真实【agent 问 → 用户答】** 回放：
  · 兜底率           —— 模拟器答不上来、只能糊弄的比例（判据 < 30%）
  · 关键信息命中率   —— 真实回答里出现的预定号/user_id，模拟器答不答得出来

跑法：由 env/tau_env.py 内部实例化，不需要单独跑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from env.bootstrap import TAU_BENCH  # noqa: F401  必须先于 tau_bench 导入

from tau_bench.envs.user import BaseUserSimulationEnv  # noqa: E402

# ---------------------------------------------------------------- 正则

RE_USER_ID = re.compile(r"\b[a-z]+_[a-z]+_\d+\b")
RE_RES_ID = re.compile(r"\b[A-Z0-9]{6}\b")
RE_FLIGHT = re.compile(r"\b[A-Z]{2}\d{3,4}\b")
RE_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
RE_AIRPORT = re.compile(r"\b[A-Z]{3}\b")

# τ-bench 里 user 轮的结束标记（见 tau_bench/envs/base.py：含它就判定 episode 结束）
STOP = "###STOP###"

# 从 gold 动作 kwargs 的 **键名** 归类到槽位
KEY_MAP = {
    "reservation_id": "reservation_ids",
    "user_id": "user_ids",
    "payment_id": "payment_ids",
    "cabin": "cabins",
    "insurance": "insurances",
    "flight_type": "flight_types",
    "total_baggages": "baggages",
    "nonfree_baggages": "baggages",
    "origin": "origins",
    "destination": "destinations",
    "date": "dates",
    "flight_number": "flight_numbers",
}


# ---------------------------------------------------------------- 槽位


@dataclass
class UserSlots:
    """用户「知道」的事实。全部来自 instruction + gold 动作 + 用户档案，无一处编造。"""

    user_id: str = ""
    reservation_ids: List[str] = field(default_factory=list)
    user_ids: List[str] = field(default_factory=list)
    flight_numbers: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    origins: List[str] = field(default_factory=list)
    destinations: List[str] = field(default_factory=list)
    airports: List[str] = field(default_factory=list)
    cabins: List[str] = field(default_factory=list)
    payment_ids: List[str] = field(default_factory=list)
    insurances: List[str] = field(default_factory=list)
    flight_types: List[str] = field(default_factory=list)
    baggages: List[str] = field(default_factory=list)
    passengers: List[Dict[str, str]] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    # ⭐ 用户**记不记得自己的预定号** —— 决定他答"就是 XXX"还是"我不记得了，你查一下"
    knows_reservation_id: bool = False


_OWNED: Optional[Dict[str, List[str]]] = None


def owned_reservations(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    {user_id: [reservation_id, ...]} —— 谁名下有哪些订单。

    为什么要这个：判断"用户知不知道自己的预定号"不能靠正则猜
    （instruction 里的 6 位大写串有大量误报，实测 24 道题里错 6 道）。
    真正的判据是：**instruction 里出现的那个号，是不是这个用户自己的订单**。
    """
    global _OWNED
    if _OWNED is None:
        idx: Dict[str, List[str]] = {}
        for rid, r in data.get("reservations", {}).items():
            idx.setdefault(r.get("user_id", ""), []).append(rid)
        _OWNED = idx
    return _OWNED

    def get(self, key: str) -> List[str]:
        v = getattr(self, key, [])
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def _walk(node, sink: Dict[str, List[str]], passengers: List[Dict[str, str]]) -> None:
    """递归走 gold 动作的 kwargs，按键名归类叶子值。"""
    if isinstance(node, dict):
        # 乘客条目单独收（first_name/last_name/dob 散在 dict 里）
        if "first_name" in node or "last_name" in node:
            p = {k: str(v) for k, v in node.items()
                 if k in ("first_name", "last_name", "dob")}
            if p and p not in passengers:
                passengers.append(p)
        for k, v in node.items():
            if k in KEY_MAP and not isinstance(v, (dict, list)):
                sink.setdefault(KEY_MAP[k], []).append(str(v))
            _walk(v, sink, passengers)
    elif isinstance(node, list):
        for x in node:
            _walk(x, sink, passengers)


def slots_from_task(task, data: Optional[Dict[str, Any]] = None) -> UserSlots:
    """
    从一道题里抽出用户掌握的事实。

    Args:
        task:  τ-bench Task（instruction / actions / user_id）
        data:  τ-bench 的全量数据（用来查用户档案：姓名/邮箱/地址/生日）
    """
    s = UserSlots(user_id=getattr(task, "user_id", "") or "",
                  instruction=getattr(task, "instruction", "") or "")

    sink: Dict[str, List[str]] = {}
    _walk([a.kwargs for a in getattr(task, "actions", [])], sink, s.passengers)

    for k, vs in sink.items():
        # 去重但**保序** —— 顺序是有意义的：gold 里第一次出现的那个
        # 往往就是用户最先会说的那个
        seen, uniq = set(), []
        for v in vs:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        setattr(s, k, uniq)

    # instruction 里提到的 ID 也算用户知道的
    instr = s.instruction
    for pat, key in ((RE_RES_ID, "reservation_ids"), (RE_USER_ID, "user_ids"),
                     (RE_FLIGHT, "flight_numbers"), (RE_DATE, "dates")):
        for m in pat.findall(instr):
            cur = getattr(s, key)
            if m not in cur:
                cur.append(m)
    s.airports = sorted(set(RE_AIRPORT.findall(instr)) | set(s.origins) | set(s.destinations))

    if data:
        if s.user_id and s.user_id in data.get("users", {}):
            s.profile = data["users"][s.user_id]
        # ⭐ 精确判定"用户记不记得预定号"：instruction 里的号必须真是他自己的订单
        mine = set(owned_reservations(data).get(s.user_id, []))
        spoken = set(RE_RES_ID.findall(s.instruction))
        known = mine & spoken
        s.knows_reservation_id = bool(known)
        if known:
            # 用户嘴上说的那个号优先 —— 那是他真正关心的那张订单
            s.reservation_ids = ([r for r in s.reservation_ids if r in mine]
                                 or list(known))
    return s


# ---------------------------------------------------------------- 意图规则

# ⚠️ 顺序即优先级 —— 先判"确认"，再判"要信息"。
#    因为 agent 常问 "Would you like me to change the reservation to ...?"
#    这句话里 reservation 和 confirm 同时出现，而**正确反应是"同意"**。
#
#    规则不是拍脑袋来的：从 685 对真实【agent 问 → 用户答】里统计出来的
#    高频问法（reservation 402 / flight 322 / change 224 / passenger 205 /
#    class 177 / payment 174 / confirm 160 / cancel 143 / which 123 ...）。
INTENTS = [
    ("stop", ["anything else", "is there anything", "help with today",
              "have a great", "thank you for your patience"]),
    # ⚠️ identity 必须排在 confirm **前面**：
    #    agent 常写 "provide your user ID **so I can proceed with** ..."，
    #    里面有 "proceed with" —— 排在后面就会被判成"求确认"，
    #    于是用户回一句"好的请继续"，而真实用户是在报 ID。
    #    （这条是拿真实数据回放时抓出来的，不是想出来的）
    ("identity", ["user id", "your id", "reservation id", "reservation number",
                  "booking id", "which reservation", "which booking",
                  "provide your", "retrieve your", "look up your", "identify you"]),
    ("passenger", ["passenger", "date of birth", "dob", "traveling with", "who is flying",
                   "traveler", "travelling", "second passenger", "companion"]),
    ("payment", ["payment", "credit card", "certificate", "gift card", "pay for",
                 "charged", "refund"]),
    ("insurance", ["insurance"]),
    ("baggage", ["baggage", "bags", "luggage"]),
    ("cabin", ["cabin", "class", "economy", "business", "upgrade"]),
    ("flight", ["which flight", "flight option", "which option", "departure",
                "onestop", "one stop", "direct flight", "red-eye", "nonstop"]),
    ("cancel", ["cancel"]),
    ("change", ["change", "update", "modify", "rebook"]),
    ("book", ["book", "booking", "reserve"]),
    ("cost", ["how much", "total cost", "the price", "cost of", "total price"]),
    ("confirm", ["would you like me to", "shall i", "should i", "do you want me to",
                 "would you like to proceed", "please confirm", "go ahead with",
                 "is that okay", "does that work", "are you okay with",
                 "let me know if you"]),
    ("profile", ["your name", "full name", "email", "address", "phone"]),
]


# agent 汇报"办完了" → 用户该收尾。
# 真实轨迹里这个模式非常干净：agent 说 "The booking ... has been successfully
# completed. Here are the details ..." → 用户 "Thank you for your help. That's all
# for now."
# ⚠️ 加问号护栏：agent 常写 "I've successfully found two reservations — which one?"
#    那是**在问问题**，不是在汇报完成，不能触发收尾。
DONE_PATTERNS = ["successfully", "has been updated", "has been cancelled",
                 "has been booked", "has been changed", "have been updated",
                 "is now confirmed", "you're all set", "you are all set",
                 "i've updated", "i have updated", "i've cancelled",
                 "i have cancelled", "i've booked", "i have booked"]


def detect_intent(text: str) -> str:
    """把 agent 的一句话归类到意图。**纯关键词，确定性，无随机。**"""
    raw = text or ""
    t = raw.lower()
    # ⭐ 先判"agent 列出多个航班选项"：
    #    这时用户的正确反应是**挑一个**，不是"好的请继续"。
    #    （"Here are two available nonstop flights: 1. HAT266 ... 2. HAT123"）
    if len(set(RE_FLIGHT.findall(raw))) >= 2:
        return "choose_flight"
    if any(k in t for k in DONE_PATTERNS) and not raw.rstrip().endswith("?"):
        return "stop"
    for name, kws in INTENTS:
        if any(k in t for k in kws):
            return name
    return "fallback"


# ---------------------------------------------------------------- 模拟器


class SlotUserSim(BaseUserSimulationEnv):
    """
    槽位式用户模拟器。

    Args:
        task:      τ-bench Task —— 信息源（instruction + gold 动作）
        data:      全量数据，用于查用户档案
        max_turns: 硬上限；到了就收尾（防弱策略把 episode 拖长）
    """

    def __init__(self, task=None, data=None, max_turns: int = 20) -> None:
        self.task = task
        self.slots = slots_from_task(task, data) if task is not None else UserSlots()
        self.max_turns = max_turns
        self.turn = 0
        self.answered = 0          # 真正给出信息或明确回应的次数
        self.fallbacks = 0         # 糊弄过去的次数（兜底句触发率的分母/分子）
        self.intents: List[str] = []

    # ------------------------------------------------------------ 契约

    def reset(self, instruction: Optional[str] = None) -> str:
        self.turn = 0
        self.answered = 0
        self.fallbacks = 0
        self.intents = []
        if instruction:
            # Env 会把 task.instruction 传进来 —— 那就是用户的开场白本身
            return instruction
        return self.slots.instruction or "Hi, I need some help with my flights."

    def step(self, content: str) -> str:
        self.turn += 1
        if self.turn >= self.max_turns:
            return f"Thanks, that's all I needed. {STOP}"

        intent = detect_intent(content)
        self.intents.append(intent)
        reply = self._reply(intent, content)

        if intent == "fallback":
            self.fallbacks += 1
        else:
            self.answered += 1
        return reply

    def get_total_cost(self) -> float:
        return 0.0

    # ------------------------------------------------------------ 应答

    def _reply(self, intent: str, q: str) -> str:
        s = self.slots

        if intent == "stop":
            return f"Thank you so much for your help! {STOP}"

        if intent == "confirm":
            return "Yes, please go ahead."

        if intent == "choose_flight":
            # ⭐ 用户挑的是**金标准里的那一班** —— 这才是"gold 可达"的保证。
            #    信息仍然是通过对话流出来的（agent 问 → 用户挑），不是硬塞进 instruction。
            opts = list(dict.fromkeys(RE_FLIGHT.findall(q)))
            gold = [f for f in s.flight_numbers if f in opts]
            if gold:
                return f"I'll go with Flight Number {gold[0]}, please."
            return "Whichever is the cheapest option, please."

        if intent == "identity":
            t = q.lower()
            wants_uid = any(w in t for w in ("user id", "your id"))
            wants_res = any(w in t for w in (
                "reservation id", "reservation number", "booking id",
                "which reservation", "which booking", "the reservation"))
            parts = []
            if wants_uid and s.user_id:
                parts.append(f"My user ID is {s.user_id}.")
            if wants_res and s.knows_reservation_id and s.reservation_ids:
                parts.append(f'The reservation ID is "{s.reservation_ids[0]}".')
            elif wants_res:
                parts.append("I don't have my reservation ID handy right now.")
            if not parts:                      # 泛泛地问身份
                if s.user_id:
                    parts.append(f"My user ID is {s.user_id}.")
                elif s.knows_reservation_id and s.reservation_ids:
                    parts.append(f'The reservation ID is "{s.reservation_ids[0]}".')
            if not parts:
                return self._fallback()
            if wants_res and not s.knows_reservation_id:
                parts.append("My profile should have it.")
            return " ".join(parts)

        if intent == "passenger":
            if s.passengers:
                p = s.passengers[0]
                name = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                if p.get("dob"):
                    return f"The passenger is {name}, date of birth {p['dob']}."
                return (f"The passenger's name is {name}. I can't remember the date "
                        "of birth, but it's in my profile.")
            return ("It's just me on this reservation. My details are in my profile"
                    + (f" under {s.user_id}." if s.user_id else "."))

        if intent == "payment":
            if s.payment_ids:
                return ("Please use the payment on file: "
                        + ", ".join(f'"{x}"' for x in s.payment_ids[:3]) + ".")
            return "Use whichever payment method is on my profile, please."

        if intent == "insurance":
            if s.insurances:
                v = s.insurances[0]
                return f"I'd like insurance = {v} for this booking."
            return "No insurance needed, thank you."

        if intent == "baggage":
            if s.baggages:
                return f"I'd like {s.baggages[0]} checked bag(s) in total."
            return "No extra baggage for me, thank you."

        if intent == "cabin":
            if s.cabins:
                return f"I'd like {s.cabins[0]} class."
            return "Economy is fine, as long as it's the cheapest option."

        if intent == "flight":
            if s.flight_numbers:
                return (f'My original flight is {s.flight_numbers[0]}'
                        + (f" on {s.dates[0]}" if s.dates else "") + ".")
            return ("I'd like the cheapest available option"
                    + (f" from {s.origins[0]} to {s.destinations[0]}"
                       if s.origins and s.destinations else "") + ".")

        if intent == "user_id":
            return (f"My user id is {s.user_id}." if s.user_id
                    else self._fallback())

        if intent == "profile":
            p = s.profile
            name = p.get("name", {})
            full = f"{name.get('first_name','')} {name.get('last_name','')}".strip()
            bits = [b for b in (full, p.get("email"), p.get("dob")) if b]
            return ("My details are: " + ", ".join(bits) + "." if bits else self._fallback())

        if intent == "cancel":
            if s.reservation_ids:
                return f'Yes, please cancel reservation "{s.reservation_ids[0]}".'
            return "Yes, please cancel it."

        if intent in ("change", "book"):
            if s.origins and s.destinations:
                return (f"I'd like to {intent} from {s.origins[0]} to "
                        f"{s.destinations[0]}, and I'd prefer the cheapest option.")
            return f"Yes, please go ahead and {intent} it."

        if intent == "cost":
            return "That works for me — please go ahead."

        return self._fallback()

    @staticmethod
    def _fallback() -> str:
        return ("I'm not sure about that — I don't have it in front of me. "
                "Could you check on your end?")


# 向后兼容：老代码 import 的是 RuleBasedUserSim
RuleBasedUserSim = SlotUserSim


__all__ = ["SlotUserSim", "RuleBasedUserSim", "slots_from_task", "UserSlots",
           "detect_intent", "STOP"]
