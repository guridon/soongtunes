"""LLM slot extraction: one utterance (+ recent history) -> a typed search request.

The LLM only fills slots; linking, retrieval, ranking and verification are deterministic
code. vLLM's json_schema structured output guarantees the shape, so there is no
JSON-repair path.
"""

from dataclasses import dataclass, field

import orjson

from stunes.llm import ChatModel

_STR_LIST = {"type": "array", "items": {"type": "string"}}
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
                "properties": {"title": {"type": "string"}, "artist": {"type": ["string", "null"]}},
                "required": ["title", "artist"],
            },
        },
        "artists": _STR_LIST,
        "genres": _STR_LIST,
        "moods": _STR_LIST,
        "exclude_artists": _STR_LIST,
        "exclude_genres": _STR_LIST,
        "exclude_moods": _STR_LIST,
        "year_from": _NULLABLE_INT,
        "year_to": _NULLABLE_INT,
        "similar_to_previous": {"type": "boolean"},
        "limit": _NULLABLE_INT,
        "search_text": {"type": "string"},
    },
    "required": [
        "intent",
        "songs",
        "artists",
        "genres",
        "moods",
        "exclude_artists",
        "exclude_genres",
        "exclude_moods",
        "year_from",
        "year_to",
        "similar_to_previous",
        "limit",
        "search_text",
    ],
}

SYSTEM = """너는 음악 검색 서비스의 요청 파서다. 사용자의 최신 발화를 직전 대화와 함께 읽고, 스키마에 맞는 JSON만 출력한다.

필드 규칙
- intent: find_song=특정 곡을 찾음 / recommend=조건에 맞는 곡 추천 / reset=새로 검색·초기화 / other=음악과 무관
- songs: 사용자가 특정 곡을 지목하면 {{title, artist}}. 따옴표 안 제목은 그대로 옮긴다. 아티스트를 모르면 null.
- artists: 언급된 아티스트 이름 원문 그대로 (예: "아이유", "BTS").
- genres: 반드시 다음 목록의 표기 중에서만 고른다: {genres}
- moods: 분위기·감정·상황·테마. 사용자가 '…' 느낌/스타일/분위기처럼 따옴표로 감싸면 따옴표 안 문자열을 그대로 넣는다.
  그 외에는 짧은 핵심어로 줄인다 (예: "비 오는 날", "새벽", "신나는", "드라이브").
- exclude_*: "말고/빼고/제외/싫어/별로" 등 부정의 대상만.
- year_from / year_to: "90년대" → 1990, 1999. "2010년 이후" → 2010, null. 없으면 null.
- similar_to_previous: "이런", "비슷한", "더", "방금 그 노래" 처럼 직전 결과를 기준으로 하면 true.
- limit: "5곡", "다섯 개" → 5. 없으면 null.
- search_text: 직전 대화를 반영해 혼자 읽어도 뜻이 통하는 한 문장 검색 질의.
장르 이름이 따옴표 안에서 '느낌'으로 쓰이면 genres가 아니라 moods로 넣는다."""

# hybrid_ctxsplit: the LLM marks a new topic, so the rules can drop the stored exclusions there
# and keep them everywhere else
SCHEMA_TOPIC = {
    **SCHEMA,
    "properties": {**SCHEMA["properties"], "new_topic": {"type": "boolean"}},
    "required": [*SCHEMA["required"], "new_topic"],
}

EXAMPLES = [
    (
        "He Is We의 'Our July In The Rain (Acoustic)'을 검색해줘.",
        {
            "intent": "find_song",
            "songs": [{"title": "Our July In The Rain (Acoustic)", "artist": "He Is We"}],
            "artists": [],
            "genres": [],
            "moods": [],
            "exclude_artists": [],
            "exclude_genres": [],
            "exclude_moods": [],
            "year_from": None,
            "year_to": None,
            "similar_to_previous": False,
            "limit": None,
            "search_text": "He Is We Our July In The Rain (Acoustic)",
        },
    ),
    (
        "이런 '새벽감성' 느낌 좋다. 발라드 말고 비슷한 거 5곡만",
        {
            "intent": "recommend",
            "songs": [],
            "artists": [],
            "genres": [],
            "moods": ["새벽감성"],
            "exclude_artists": [],
            "exclude_genres": ["발라드"],
            "exclude_moods": [],
            "year_from": None,
            "year_to": None,
            "similar_to_previous": True,
            "limit": 5,
            "search_text": "새벽감성 느낌의 발라드가 아닌 노래",
        },
    ),
    (
        "90년대 아이유 노래 중에 신나는 거",
        {
            "intent": "recommend",
            "songs": [],
            "artists": ["아이유"],
            "genres": [],
            "moods": ["신나는"],
            "exclude_artists": [],
            "exclude_genres": [],
            "exclude_moods": [],
            "year_from": 1990,
            "year_to": 1999,
            "similar_to_previous": False,
            "limit": None,
            "search_text": "1990년대 아이유의 신나는 노래",
        },
    ),
]


@dataclass
class SongMention:
    title: str
    artist: str | None


@dataclass
class Op:
    """One change to the stored state (hybrid_delta*)."""

    op: str  # add / replace / cancel / exclude / new_topic / no_change
    kind: str | None = None  # artist / genre / mood / year
    value: str | None = None
    old: str | None = None  # replace: the value being replaced
    year_from: int | None = None
    year_to: int | None = None


@dataclass
class Slots:
    intent: str = "recommend"
    songs: list[SongMention] = field(default_factory=list)
    artists: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    exclude_artists: list[str] = field(default_factory=list)
    exclude_genres: list[str] = field(default_factory=list)
    exclude_moods: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    similar_to_previous: bool = False
    limit: int | None = None
    search_text: str = ""
    new_topic: bool = False
    # hybrid_delta*: this turn's changes to the stored state (the lists above then hold only the
    # values to link: added and new positives, added exclusions)
    ops: list["Op"] = field(default_factory=list)
    # characters sent to the LLM for this turn, examples included
    prompt_chars: int = 0

    def move_genre_exclusions(self, genres: set[str]) -> None:
        """An exclusion in the mood or artist slot that is exactly a genre name is a genre
        ("아이돌 말고" came back as a mood or an artist on dev)."""
        for name in ("exclude_moods", "exclude_artists"):
            values = getattr(self, name)
            moved = [v for v in values if v in genres]
            if moved:
                setattr(self, name, [v for v in values if v not in genres])
                self.exclude_genres += [v for v in moved if v not in self.exclude_genres]

    @classmethod
    def from_json(cls, data: dict) -> "Slots":
        return cls(
            intent=data.get("intent", "recommend"),
            songs=[
                SongMention(s["title"], s.get("artist"))
                for s in data.get("songs", [])
                if s.get("title")
            ],
            **{
                k: data.get(k, [])
                for k in (
                    "artists",
                    "genres",
                    "moods",
                    "exclude_artists",
                    "exclude_genres",
                    "exclude_moods",
                )
            },
            year_from=data.get("year_from"),
            year_to=data.get("year_to"),
            similar_to_previous=bool(data.get("similar_to_previous")),
            limit=data.get("limit"),
            search_text=data.get("search_text", ""),
            new_topic=bool(data.get("new_topic")),
        )


# hybrid_llmstate (docs/prereg-context-eval.md): the LLM itself tracks the conversation's state,
# with no stored preferences behind it. The baseline the VirtualQuery state has to beat. The
# rules name no genre or artist: a named example leaked into first-turn slots on dev. Its
# few-shot turns carry an explicit empty history: sent bare, the cumulative rule read them as
# earlier conversation (아이유, 1990s and 발라드 showed up on first turns on dev), and moved into
# the system text they no longer taught the extraction (dropped moods, open decades).
CUMULATIVE = """

누적 모드: 슬롯에는 최신 발화의 조건만이 아니라, 직전 대화 전체를 반영해 지금 유효한 조건을 모두 넣는다.
- 사용자가 바꾼 조건은 새 값으로 바꾼다 (다른 장르를 말하면 그 장르만 남긴다).
- "상관없어"처럼 취소한 조건은 뺀다.
- 제외(exclude_*)는 사용자가 다시 원하기 전까지 유지한다.
- 새 주제로 넘어가면 이전 조건은 버린다.
- "더 보여줘"처럼 조건을 말하지 않으면 직전 조건을 그대로 넣는다.
- "N번째 곡"은 직전 대화의 추천 목록에서 찾는다."""


# hybrid_ctx: the stored state goes into the context. The LLM reads the conditions the last
# turn searched with and the numbered last results, and restates the whole state; the pipeline
# takes its slots as the state. On the test run the history-only LLM dropped the untouched
# genre on add and exclude turns (32-33%): its history holds the user's words and three titles,
# not the conditions that were applied.
STATE_CONTEXT = """
- [현재 조건]은 직전 턴까지 검색에 실제로 쓴 조건이다. 최신 발화가 건드리지 않은 조건은 그대로 슬롯에 옮긴다.
- 최신 발화가 조건을 취소하면 빼고, 같은 종류의 새 값을 말하면 바꾼다. 새 주제로 넘어가면 [현재 조건]을 제외까지 모두 버린다.
- "N번째 곡"은 [직전 추천 목록]의 N번 곡이다. 그 곡의 장르를 말하면 genres에, 그 곡의 가수를 말하면 artists에 그 값을 넣는다."""

# hybrid_ctxsplit: the exclusions are kept by rule, so the LLM only marks where a new topic starts
TOPIC = """
- new_topic: 최신 발화가 [현재 조건]을 이어가지 않고 새로 찾기 시작하면 true. 조건을 바꾸거나 더하거나 빼거나 취소하면 false."""

# hybrid_ctx_recent / hybrid_delta_recent: the history is cut to the last exchange
RECENT = """
- [직전 대화]에는 마지막 한 번의 주고받기만 있다. 그 전 대화는 [현재 조건]에 정리돼 있다."""

_EMPTY = "(없음)"


class SlotExtractor:
    def __init__(
        self,
        llm: ChatModel,
        genres: list[str],
        cumulative: bool = False,
        state_context: bool = False,
        split: bool = False,
        recent: bool = False,
    ):
        """`split` (hybrid_ctxsplit, implies state_context): the LLM also marks a new topic, and
        a genre name in the mood or artist exclusion slot is moved to the genre slot.
        `recent` (hybrid_ctx_recent, with state_context): the history is only the last exchange,
        the user's last message and the reply."""
        self._llm = llm
        self._system = SYSTEM.format(genres=", ".join(genres))
        self._split = split
        self._genres = set(genres)
        self._schema = SCHEMA_TOPIC if split else SCHEMA
        state_context = state_context or split
        self._state_context = state_context
        cumulative = cumulative or state_context
        self._cumulative = cumulative
        if state_context:
            self._system += (
                CUMULATIVE
                + STATE_CONTEXT
                + (TOPIC if split else "")
                + (RECENT if recent else "")
                + "\n- 예시마다 [현재 조건]과 [직전 대화]가 따로 있다. 예시끼리는 서로 다른 대화다."
            )
        elif cumulative:
            self._system += (
                CUMULATIVE + "\n- 예시마다 [직전 대화]가 따로 있다. 예시끼리는 서로 다른 대화다."
            )
        # the cumulative extractor reads the whole stored history (12 messages, 6 turns): it has
        # no state to fall back on
        self._window = 2 if recent else (12 if cumulative else 6)

    def _framed(self, message: str, context: str, state: str, results: str) -> str:
        head = (
            f"[현재 조건]\n{state or _EMPTY}\n\n[직전 추천 목록]\n{results or _EMPTY}\n\n"
            if self._state_context
            else ""
        )
        return f"{head}[직전 대화]\n{context or _EMPTY}\n\n[최신 발화]\n{message}"

    async def extract(
        self,
        message: str,
        history: list[dict[str, str]],
        state: str = "",
        results: str = "",
    ) -> tuple[Slots, dict]:
        """`state` and `results` (state_context only): the conditions the last turn searched
        with and its numbered results, as text."""
        messages = [{"role": "system", "content": self._system}]
        if self._cumulative:
            # the examples in the same framing as the request, each with an empty history: sent
            # bare, the cumulative rule read them as this conversation's earlier turns
            for q, a in EXAMPLES:
                messages += [
                    {"role": "user", "content": self._framed(q, "", "", "")},
                    {
                        "role": "assistant",
                        "content": orjson.dumps(
                            (a | {"new_topic": False}) if self._split else a
                        ).decode(),
                    },
                ]
            context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-self._window :])
            messages.append(
                {"role": "user", "content": self._framed(message, context, state, results)}
            )
            raw = await self._llm.complete_json(messages, self._schema, name="music_request")
            slots = Slots.from_json(raw)
            slots.search_text = slots.search_text or message
            slots.prompt_chars = sum(len(m["content"]) for m in messages)
            if self._split:
                slots.move_genre_exclusions(self._genres)
            return slots, raw
        for q, a in EXAMPLES:
            messages += [
                {"role": "user", "content": q},
                {"role": "assistant", "content": orjson.dumps(a).decode()},
            ]
        if history:
            context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-self._window :])
            messages.append(
                {"role": "user", "content": f"[직전 대화]\n{context}\n\n[최신 발화]\n{message}"}
            )
        else:
            messages.append({"role": "user", "content": message})
        raw = await self._llm.complete_json(messages, SCHEMA, name="music_request")
        slots = Slots.from_json(raw)
        if not slots.search_text:
            slots.search_text = message
        return slots, raw
