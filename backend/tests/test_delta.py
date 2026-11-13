import asyncio

from stunes.conversation.delta import EXAMPLES_DELTA, DeltaExtractor, to_slots
from stunes.conversation.extract import Op, Slots
from stunes.conversation.pipeline import HybridRecommender, _Linked
from stunes.conversation.state import Preference, SessionState
from stunes.retrieval.linker import Linked


class _Linker:
    async def artist(self, name):
        return Linked("artist", f"a:{name}", name, "exact")

    def genre(self, name):
        return Linked("genre", name, name, "exact")

    async def tags(self, mood, k=2, context_tags=None):
        return [Linked("tag", mood, mood, "exact")]


def _rec() -> HybridRecommender:
    rec = object.__new__(HybridRecommender)
    rec._linker = _Linker()
    return rec


def _state() -> SessionState:
    return SessionState(
        "s",
        turn=2,
        prefs=[
            Preference("genre", "재즈", "재즈", True, 1),
            Preference("tag", "잔잔한", "잔잔한", True, 1),
            Preference("tag", "이별", "이별", False, 1),
        ],
        year_from=1990,
        year_to=1999,
    )


def _apply(slots: Slots, linked: _Linked | None = None):
    return asyncio.run(_rec()._apply_delta(_state(), linked or _Linked(), slots))


def _keys(prefs):
    return {(p.kind, p.key, p.positive) for p in prefs}


def test_an_exclusion_is_added_and_the_rest_stays():
    slots = Slots(ops=[Op("exclude", "genre", "발라드")])
    how, (_, remove, prefs), year = _apply(
        slots, _Linked(neg_genres=[Linked("genre", "발라드", "발라드", "exact")])
    )
    assert how == "delta"
    assert _keys(prefs) == {
        ("genre", "재즈", True),
        ("tag", "잔잔한", True),
        ("tag", "이별", False),
        ("genre", "발라드", False),
    }
    assert remove == [] and year == (1990, 1999)


def test_cancel_takes_either_sign_and_a_kind_wide_cancel_keeps_exclusions():
    _, (_, remove, prefs), _ = _apply(Slots(ops=[Op("cancel", "mood", "이별")]))
    assert ("tag", "이별", False) not in _keys(prefs)
    assert remove == [("tag", "이별")]
    _, (_, _, prefs), _ = _apply(Slots(ops=[Op("cancel", "mood")]))
    assert _keys(prefs) == {("genre", "재즈", True), ("tag", "이별", False)}


def test_replace_swaps_the_named_value_or_the_whole_kind():
    linked = _Linked(genres=[Linked("genre", "R&B/Soul", "R&B/Soul", "exact")])
    _, (_, remove, prefs), _ = _apply(
        Slots(ops=[Op("replace", "genre", "R&B/Soul", old="재즈")]), linked
    )
    assert ("genre", "R&B/Soul", True) in _keys(prefs)
    assert remove == [("genre", "재즈")]
    # an old value that matches nothing still replaces the kind's positives
    _, (_, remove, _), _ = _apply(
        Slots(ops=[Op("replace", "genre", "R&B/Soul", old="블루스")]), linked
    )
    assert remove == [("genre", "재즈")]


def test_year_operations():
    assert _apply(Slots(ops=[Op("cancel", "year")]))[2] == (None, None)
    assert _apply(Slots(ops=[Op("replace", "year", year_from=2000, year_to=2009)]))[2] == (
        2000,
        2009,
    )


def test_new_topic_drops_everything_then_adds():
    linked = _Linked(genres=[Linked("genre", "클래식", "클래식", "exact")])
    how, (_, remove, prefs), year = _apply(
        Slots(ops=[Op("new_topic"), Op("add", "genre", "클래식")]), linked
    )
    assert how == "new_topic"
    assert _keys(prefs) == {("genre", "클래식", True)}
    assert len(remove) == 3 and year == (None, None)


def test_no_change_and_an_empty_answer_keep_the_state_and_differ_in_the_record():
    how, (_, remove, prefs), year = _apply(Slots(ops=[Op("no_change")]))
    assert (how, remove, len(prefs), year) == ("no_change", [], 3, (1990, 1999))
    how, (_, remove, prefs), _ = _apply(Slots(ops=[]))
    assert (how, remove, len(prefs)) == ("failed_empty", [], 3)


def test_to_slots_puts_values_to_link_in_the_lists():
    slots = to_slots(
        {
            "intent": "recommend",
            "ops": [
                {"op": "add", "kind": "mood", "value": "신나는", "old": None,
                 "year_from": None, "year_to": None},
                {"op": "exclude", "kind": "genre", "value": "아이돌", "old": None,
                 "year_from": None, "year_to": None},
                {"op": "cancel", "kind": "artist", "value": "넬", "old": None,
                 "year_from": None, "year_to": None},
                {"op": "add", "kind": "year", "value": None, "old": None,
                 "year_from": 1990, "year_to": 1999},
            ],
        }
    )  # fmt: skip
    assert (slots.moods, slots.exclude_genres, slots.artists) == (["신나는"], ["아이돌"], [])
    assert (slots.year_from, slots.year_to) == (1990, 1999)
    assert [o.op for o in slots.ops] == ["add", "exclude", "cancel", "add"]


class _LLM:
    def __init__(self):
        self.messages = self.schema = None

    async def complete_json(self, messages, schema, name):
        self.messages, self.schema = messages, schema
        return {"intent": "recommend", "ops": [{"op": "no_change"}]}


def test_delta_extractor_frames_examples_and_cuts_history_when_recent():
    llm = _LLM()
    history = [{"role": "user", "content": f"m{i}"} for i in range(6)]
    slots, _ = asyncio.run(
        DeltaExtractor(llm, ["재즈"], recent=True).extract("더 줘", history, "장르: 재즈", "1. 곡")
    )
    assert "ops" in llm.schema["required"]
    assert len(llm.messages) == 1 + 2 * len(EXAMPLES_DELTA) + 1
    assert "마지막 한 번의 주고받기" in llm.messages[0]["content"]
    last = llm.messages[-1]["content"]
    assert last.startswith("[현재 조건]\n장르: 재즈\n\n[직전 추천 목록]\n1. 곡")
    assert "m3" not in last and "user: m4\nuser: m5" in last
    # a stored-state example carries its own state
    assert any(m["content"].startswith("[현재 조건]\n가수: 넬") for m in llm.messages[1:-1])
    assert [o.op for o in slots.ops] == ["no_change"] and slots.prompt_chars > 0


def test_a_cancel_of_a_value_the_state_lacks_is_an_exclusion():
    # "아이돌은 빼줘" read as cancel 아이돌
    _, (_, remove, prefs), _ = _apply(Slots(ops=[Op("cancel", "genre", "아이돌")]))
    assert ("genre", "아이돌", False) in _keys(prefs)
    assert remove == []


def test_a_new_genre_replaces_the_stored_one_even_as_add():
    # "힙합은 어때?" read as add 랩/힙합
    linked = _Linked(genres=[Linked("genre", "랩/힙합", "랩/힙합", "exact")])
    _, (_, remove, prefs), _ = _apply(Slots(ops=[Op("add", "genre", "랩/힙합")]), linked)
    assert {k for k in _keys(prefs) if k[0] == "genre"} == {("genre", "랩/힙합", True)}
    assert remove == [("genre", "재즈")]
    # moods are a list: an add keeps the stored one
    linked = _Linked(tags=[Linked("tag", "새벽", "새벽", "exact")])
    _, (_, _, prefs), _ = _apply(Slots(ops=[Op("add", "mood", "새벽")]), linked)
    assert {("tag", "잔잔한", True), ("tag", "새벽", True)} <= _keys(prefs)
