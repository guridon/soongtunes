"""A sign the change extractor got backwards, put right from the message's own words
(docs/prereg-sign-rule.md). No LLM call.

On test2 the delta extractor wrote an exclusion for a genre the user asked for on 21 of the
40 values it lost: "아이돌은?", "OST는 어때?", "인디로 골라줘", "그중에 아이돌만". Three went
the other way: "이별 노래는 빼고" came back as add.

Two narrow rules, on the extractor's operations before they are linked:
- **exclude → add:** the value is named in the message, the message has no negation word
  anywhere, and it asks for something (a question mark at the end, 어때, 골라, (으)로 바꿔,
  만, 듣고 싶, 추천, 틀어, 들려). A message that mixes signs ("발라드 빼고 아이돌은?") is
  left alone, and so is an exclusion the LLM inferred from context ("다른 것도 추천해줄래?"
  → exclude the artist shown before): the user didn't name it.
- **add → exclude:** the value is in the message and a negation word follows it within
  WINDOW characters (spaces dropped): "이별 노래는 빼고", "아이돌 말고 다른 걸로".
"""

import re

NEG = re.compile(
    r"빼|말고|말구|제외|싫|없이|없는|아닌|아니고|아니라|별로|지겨|질려|질린|그만|됐어|"
    r"안\s*들|피해|거르|걸러|금지|노노|패스|ㄴㄴ"
)
ASK = re.compile(
    r"[?？]\s*$|어때|어떨까|골라|로\s*바꿔|만\s*$|만\s*(?:골라|보여|추천|틀어|들려|줘)|"
    r"듣고\s*싶|추천|틀어|들려"
)
WINDOW = 8
KINDS = {"artist", "genre", "mood"}


def _compact(s: str) -> str:
    return "".join(s.split())


def _named(value: str, msg: str) -> bool:
    """The value, or its first two characters ("인디" for 인디음악), is in the message."""
    v = _compact(value)
    return v in msg or (len(v) >= 2 and v[:2] in msg)


def fix(raw: dict, message: str) -> tuple[dict, list[list[str]]]:
    """(the extractor's output with signs put right, [[old op, new op, value], ...])."""
    ops = raw.get("ops") or []
    flips: list[list[str]] = []
    if not ops:
        return raw, flips
    msg = _compact(message)
    no_neg, asks = NEG.search(message) is None, ASK.search(message) is not None
    out = []
    for o in ops:
        value = o.get("value") or ""
        if o.get("kind") not in KINDS or not value:
            out.append(o)
            continue
        new = o.get("op")
        if new == "exclude" and no_neg and asks and _named(value, msg):
            new = "add"
        elif new == "add":
            at = msg.find(_compact(value))
            after = msg[at + len(_compact(value)) :][:WINDOW] if at >= 0 else ""
            if after and NEG.search(after):
                new = "exclude"
        if new != o.get("op"):
            flips.append([o.get("op"), new, value])
            o = {**o, "op": new}
        out.append(o)
    return ({**raw, "ops": out} if flips else raw), flips
