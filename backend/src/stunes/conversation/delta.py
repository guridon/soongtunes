"""hybrid_delta / hybrid_delta_recent (docs/prereg-context-2x2.md): the LLM reads the stored state
and outputs only this turn's changes to it, as operations; code applies them.

The full-state extractors (`SlotExtractor` with state_context) restate the whole state every
turn and lost stored exclusions doing it. The same split is the change-output design of IC-DST
(Findings EMNLP 2022) and Diable (Findings ACL 2023), on value sets here: an operation adds or
removes one value, not a whole slot.
"""

import orjson

from stunes.conversation.extract import _EMPTY, EXAMPLES, RECENT, Op, Slots, SongMention
from stunes.llm import ChatModel

OPS = ["add", "replace", "cancel", "exclude", "new_topic", "no_change"]
KINDS = ["artist", "genre", "mood", "year"]
VALUE_KINDS = ("artist", "genre", "mood")
PREF_KIND = {"artist": "artist", "genre": "genre", "mood": "tag"}  # op kind -> Preference.kind
_NULLABLE_STR = {"type": ["string", "null"]}
_NULLABLE_INT = {"type": ["integer", "null"]}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string", "enum": ["find_song", "recommend", "reset", "other"]},
        "songs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"title": {"type": "string"}, "artist": _NULLABLE_STR},
                "required": ["title", "artist"],
            },
        },
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {"type": "string", "enum": OPS},
                    "kind": {"anyOf": [{"type": "string", "enum": KINDS}, {"type": "null"}]},
                    "value": _NULLABLE_STR,
                    "old": _NULLABLE_STR,
                    "year_from": _NULLABLE_INT,
                    "year_to": _NULLABLE_INT,
                },
                "required": ["op", "kind", "value", "old", "year_from", "year_to"],
            },
        },
        "similar_to_previous": {"type": "boolean"},
        "limit": _NULLABLE_INT,
        "search_text": {"type": "string"},
    },
    "required": ["intent", "songs", "ops", "similar_to_previous", "limit", "search_text"],
}

# One source of truth: [현재 조건]. The history only helps read the latest message. The rules
# name no value from the evaluation set's templates; the genre names are the list.
SYSTEM = """너는 음악 검색 서비스의 대화 상태 갱신기다. [현재 조건]은 직전 턴까지 검색에 쓴 조건이고, 이것이 기준이다.
최신 발화가 그 조건을 어떻게 바꾸는지만 연산 목록(ops)으로 출력한다. [현재 조건]을 다시 옮겨 적지 않는다. 코드가 연산을 [현재 조건]에 적용한 결과가 새 검색 조건이 된다.

입력
- [현재 조건]: 지금까지의 조건.
- [직전 추천 목록]: 직전 턴에 보여 준 곡, 번호순. "N번째 곡"은 여기서 찾는다.
- [직전 대화]: 최신 발화를 해석하는 데만 쓴다.
- [최신 발화]

연산 (ops의 각 항목: op, kind, value, old, year_from, year_to)
- add: 조건을 하나 더한다.
- replace: 같은 종류의 조건을 다른 값으로 바꾼다. old는 바뀌는 기존 값, value는 새 값.
  - 가수와 장르는 하나씩만 둔다. [현재 조건]에 이미 있는 종류의 다른 값을 말하면 add가 아니라 replace다.
  - 분위기는 "도", "함께", "추가로"처럼 더한다는 말이 있을 때만 add, 아니면 replace다.
- exclude: 제외 조건을 하나 더한다. "말고/빼고/빼줘/제외/싫어/별로"의 대상은 언제나 exclude다. [현재 조건]에 없는 값이어도 exclude다.
- cancel: [현재 조건]에 있는 조건을 없던 일로 한다 ("상관없어", "아무거나", "없어도 돼", "취소"). 그 값을 value에 넣는다. 한 종류를 통째로 없애면 value는 null이고, 이때 그 종류의 제외 조건은 그대로 둔다.
  - 제외했던 값을 다시 허용하면 그 제외 조건을 cancel한다.
  - "빼줘"는 cancel이 아니라 exclude다.
- new_topic: 이전 조건을 제외까지 모두 버리고 새로 찾는다. 새 조건은 같은 ops에 add로 넣는다.
- no_change: 조건이 그대로일 때만 쓴다 (같은 조건으로 곡만 더 원할 때). 이때는 다른 연산을 넣지 않는다.
- kind: artist / genre / mood / year. new_topic과 no_change는 kind, value가 null.
- genre의 value는 반드시 다음 목록의 표기 중에서만 고른다: {genres}
- mood: 분위기·감정·상황·테마를 짧은 핵심어로 (예: "비 오는 날", "새벽", "신나는", "드라이브"). '…' 느낌처럼 따옴표로 감싸면 따옴표 안 문자열 그대로. 장르 이름이 따옴표 안에서 '느낌'으로 쓰이면 mood.
- year: value는 null, year_from / year_to에 숫자. "90년대" → 1990, 1999. "2010년 이후" → 2010, null. 다른 kind면 둘 다 null.
- "N번째 곡의 가수", "N번 곡과 같은 장르"처럼 목록의 곡을 가리키면 조건이 바뀐 것이다 (no_change가 아니다). [직전 추천 목록]의 그 곡에서 값을 찾아 add하거나, 같은 종류가 이미 있으면 replace한다.

나머지 필드
- intent: find_song=특정 곡을 찾음 / recommend=조건에 맞는 곡 추천 / reset=새로 검색·초기화 / other=음악과 무관
- songs: 사용자가 특정 곡을 지목하면 {{title, artist}}. 따옴표 안 제목은 그대로 옮긴다. 아티스트를 모르면 null.
- similar_to_previous: "이런", "비슷한", "더", "방금 그 노래" 처럼 직전 결과를 기준으로 하면 true.
- limit: "5곡", "다섯 개" → 5. 없으면 null.
- search_text: 연산을 적용한 조건과 최신 발화를 합쳐, 혼자 읽어도 뜻이 통하는 한 문장 검색 질의.
- 예시마다 [현재 조건]이 따로 있다. 예시끼리는 서로 다른 대화다."""


def _op(op, kind=None, value=None, old=None, year_from=None, year_to=None) -> dict:
    return {
        "op": op,
        "kind": kind,
        "value": value,
        "old": old,
        "year_from": year_from,
        "year_to": year_to,
    }


def _answer(intent="recommend", ops=(), songs=(), similar=False, limit=None, text="") -> dict:
    return {
        "intent": intent,
        "songs": list(songs),
        "ops": list(ops),
        "similar_to_previous": similar,
        "limit": limit,
        "search_text": text,
    }


# The served extractor's three first-turn examples, as changes to an empty state, then five with
# a stored state: cancel against exclude, a cancelled exclusion, a song in the list, no change,
# a new topic. Their wording avoids the evaluation set's templates; the listed songs and artists
# are made up, so a copied name links to nothing. (state, utterance, answer, list)
_FIRST = [
    _answer(
        "find_song",
        [_op("no_change")],
        songs=[{"title": "Our July In The Rain (Acoustic)", "artist": "He Is We"}],
        text="He Is We Our July In The Rain (Acoustic)",
    ),
    _answer(
        ops=[_op("add", "mood", "새벽감성"), _op("exclude", "genre", "발라드")],
        similar=True,
        limit=5,
        text="새벽감성 느낌의 발라드가 아닌 노래",
    ),
    _answer(
        ops=[
            _op("add", "artist", "아이유"),
            _op("add", "mood", "신나는"),
            _op("add", "year", year_from=1990, year_to=1999),
        ],
        text="1990년대 아이유의 신나는 노래",
    ),
]
EXAMPLES_DELTA = [("", q, a, "") for (q, _), a in zip(EXAMPLES, _FIRST, strict=True)] + [
    (
        "장르: 댄스\n분위기: 신나는",
        "신나는 거 아니어도 돼. 아이돌 노래는 싫어",
        _answer(
            ops=[_op("cancel", "mood", "신나는"), _op("exclude", "genre", "아이돌")],
            text="아이돌이 아닌 댄스 노래",
        ),
        "",
    ),
    (
        "장르: 재즈\n제외 분위기: 이별",
        "재즈 대신 R&B로 하고, 이별 노래도 이제 괜찮아",
        _answer(
            ops=[_op("replace", "genre", "R&B/Soul", old="재즈"), _op("cancel", "mood", "이별")],
            text="R&B/Soul 노래",
        ),
        "",
    ),
    (
        "장르: 발라드\n제외 분위기: 이별",
        "2번 곡 부른 사람 노래로 찾아줘",
        _answer(
            ops=[_op("add", "artist", "예시가수나")],
            text="예시가수나의 발라드 노래",
        ),
        "1. 예시곡 하나 - 예시가수가 (장르: 발라드)\n2. 예시곡 둘 - 예시가수나 (장르: 발라드, 인디음악)",
    ),
    (
        "가수: 넬\n연도: 2000~2009",
        "이 조건 그대로 계속 틀어줘",
        _answer(ops=[_op("no_change")], similar=True, text="2000년대 넬 노래"),
        "",
    ),
    (
        "가수: 넬\n제외 장르: 발라드",
        "그건 그만하고 클래식 틀어줘",
        _answer(ops=[_op("new_topic"), _op("add", "genre", "클래식")], text="클래식 음악"),
        "",
    ),
]


class DeltaExtractor:
    """Same call as `SlotExtractor.extract(message, history, state, results)`. Returns Slots
    whose `ops` hold the changes; its value lists hold what needs linking (added and new
    positives in artists / genres / moods, added exclusions in exclude_*), and year_from /
    year_to the year an add or replace sets."""

    def __init__(self, llm: ChatModel, genres: list[str], recent: bool = False):
        self._llm = llm
        self._system = SYSTEM.format(genres=", ".join(genres)) + (RECENT if recent else "")
        self._window = 2 if recent else 12  # messages; 12 is the full-state arms' window

    @staticmethod
    def _framed(message: str, context: str, state: str, results: str) -> str:
        return (
            f"[현재 조건]\n{state or _EMPTY}\n\n[직전 추천 목록]\n{results or _EMPTY}\n\n"
            f"[직전 대화]\n{context or _EMPTY}\n\n[최신 발화]\n{message}"
        )

    async def extract(
        self,
        message: str,
        history: list[dict[str, str]],
        state: str = "",
        results: str = "",
    ) -> tuple[Slots, dict]:
        messages = [{"role": "system", "content": self._system}]
        for st, q, a, listed in EXAMPLES_DELTA:
            messages += [
                {"role": "user", "content": self._framed(q, "", st, listed)},
                {"role": "assistant", "content": orjson.dumps(a).decode()},
            ]
        context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-self._window :])
        messages.append({"role": "user", "content": self._framed(message, context, state, results)})
        raw = await self._llm.complete_json(messages, SCHEMA, name="state_change")
        slots = to_slots(raw)
        slots.search_text = slots.search_text or message
        slots.prompt_chars = sum(len(m["content"]) for m in messages)
        return slots, raw


def to_slots(raw: dict) -> Slots:
    ops = [
        Op(
            o.get("op", ""),
            o.get("kind"),
            o.get("value"),
            o.get("old"),
            o.get("year_from"),
            o.get("year_to"),
        )
        for o in raw.get("ops") or []
        if o.get("op") in OPS
    ]
    slots = Slots(
        intent=raw.get("intent", "recommend"),
        songs=[
            SongMention(s["title"], s.get("artist"))
            for s in raw.get("songs") or []
            if s.get("title")
        ],
        similar_to_previous=bool(raw.get("similar_to_previous")),
        limit=raw.get("limit"),
        search_text=raw.get("search_text", ""),
        ops=ops,
    )
    lists = {
        ("artist", True): slots.artists,
        ("genre", True): slots.genres,
        ("mood", True): slots.moods,
        ("artist", False): slots.exclude_artists,
        ("genre", False): slots.exclude_genres,
        ("mood", False): slots.exclude_moods,
    }
    for o in ops:
        if o.kind == "year" and o.op in ("add", "replace"):
            slots.year_from, slots.year_to = o.year_from, o.year_to
        elif o.op in ("add", "replace", "exclude") and o.value and o.kind in VALUE_KINDS:
            lists[(o.kind, o.op != "exclude")].append(o.value)
    return slots
