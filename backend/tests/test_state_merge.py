from stunes.conversation.extract import Slots
from stunes.conversation.pipeline import HybridRecommender, _Linked
from stunes.conversation.state import Preference, SessionState
from stunes.retrieval.linker import Linked


def _state(*prefs: Preference) -> SessionState:
    return SessionState("s", turn=1, prefs=list(prefs))


def test_new_positive_of_a_kind_replaces_old_and_negatives_accumulate():
    state = _state(
        Preference("genre", "발라드", "발라드", True, 1),
        Preference("artist", "1", "아이유", False, 1),
    )
    linked = _Linked(
        genres=[Linked("genre", "댄스", "댄스", "exact")],
        neg_tags=[Linked("tag", "새벽", "새벽", "exact")],
    )
    upsert, remove, prefs = HybridRecommender._merge(state, linked)
    keys = {(p.kind, p.key, p.positive) for p in prefs}
    assert keys == {("genre", "댄스", True), ("artist", "1", False), ("tag", "새벽", False)}
    assert ("genre", "발라드") in remove
    assert {(p.kind, p.key) for p in upsert} == {("genre", "댄스"), ("tag", "새벽")}


def test_saying_something_positively_lifts_its_negative():
    state = _state(Preference("tag", "감성", "감성", False, 1))
    upsert, remove, prefs = HybridRecommender._merge(
        state, _Linked(tags=[Linked("tag", "감성", "감성", "exact")])
    )
    assert [(p.key, p.positive) for p in prefs] == [("감성", True)]
    assert remove == []  # same (kind, key): an upsert flips the sign in place
    assert [(p.key, p.positive) for p in upsert] == [("감성", True)]


def test_positive_tags_are_capped_to_the_newest():
    state = _state(*(Preference("tag", t, t, True, 1) for t in ["a", "b"]))
    linked = _Linked(tags=[Linked("tag", t, t, "exact") for t in ["c", "d"]])
    _, remove, prefs = HybridRecommender._merge(state, linked)
    # tags this turn replace earlier positive tags; then the newest MAX_TAGS are kept
    assert [p.key for p in prefs if p.kind == "tag"] == ["c", "d"]
    assert set(remove) == {("tag", "a"), ("tag", "b")}


def test_replace_takes_the_restated_slots_as_the_whole_state():
    # hybrid_ctx: the extractor read the stored state, so what it leaves out is dropped
    state = _state(
        Preference("genre", "인디", "인디", True, 1),
        Preference("tag", "데이트", "데이트", True, 1),
        Preference("genre", "포크", "포크", False, 1),
    )
    linked = _Linked(
        genres=[Linked("genre", "인디", "인디", "exact")],
        neg_genres=[Linked("genre", "포크", "포크", "exact")],
    )
    upsert, remove, prefs = HybridRecommender._replace(state, linked)
    assert {(p.kind, p.key, p.positive) for p in prefs} == {
        ("genre", "인디", True),
        ("genre", "포크", False),
    }
    assert remove == [("tag", "데이트")]
    assert upsert == prefs


def test_replace_keeps_a_value_said_both_ways_as_positive_and_caps_tags():
    linked = _Linked(
        tags=[Linked("tag", t, t, "exact") for t in ["a", "b", "c", "d"]],
        neg_tags=[Linked("tag", "a", "a", "exact")],
    )
    _, _, prefs = HybridRecommender._replace(_state(), linked)
    assert [(p.key, p.positive) for p in prefs] == [("b", True), ("c", True), ("d", True)]


def _split_state() -> SessionState:
    return _state(
        Preference("genre", "인디", "인디", True, 1),
        Preference("tag", "따뜻한", "따뜻한", False, 1),
        Preference("genre", "발라드", "발라드", False, 1),
    )


def test_split_keeps_stored_exclusions_the_restatement_left_out():
    # "트로트는?": the LLM restated the new genre only; the exclusions stay by rule
    linked = _Linked(genres=[Linked("genre", "트로트", "트로트", "exact")])
    how, (_, remove, prefs) = HybridRecommender._update("split", _split_state(), linked, Slots())
    assert how == "replace"
    assert {(p.kind, p.key, p.positive) for p in prefs} == {
        ("genre", "트로트", True),
        ("tag", "따뜻한", False),
        ("genre", "발라드", False),
    }
    assert remove == [("genre", "인디")]


def test_split_lifts_an_exclusion_said_positively():
    linked = _Linked(genres=[Linked("genre", "발라드", "발라드", "exact")])
    _, (_, _, prefs) = HybridRecommender._update("split", _split_state(), linked, Slots())
    assert ("genre", "발라드", True) in {(p.kind, p.key, p.positive) for p in prefs}
    assert ("genre", "발라드", False) not in {(p.kind, p.key, p.positive) for p in prefs}


def test_split_drops_everything_on_a_marked_new_topic():
    linked = _Linked(artists=[Linked("artist", "7", "요조", "exact")])
    how, (_, remove, prefs) = HybridRecommender._update(
        "split", _split_state(), linked, Slots(new_topic=True)
    )
    assert how == "new_topic"
    assert [(p.kind, p.key) for p in prefs] == [("artist", "7")]
    assert set(remove) == {("genre", "인디"), ("tag", "따뜻한"), ("genre", "발라드")}


def test_split_keeps_the_state_when_nothing_is_restated():
    # "더 없어?" read as `other`: no condition, so the stored state stands
    how, (upsert, remove, prefs) = HybridRecommender._update(
        "split", _split_state(), _Linked(), Slots(intent="other")
    )
    assert how == "kept"
    assert (upsert, remove) == ([], [])
    assert len(prefs) == 3


def test_replace_still_drops_on_an_empty_restatement():
    # hybrid_ctx as tested on test: unchanged
    how, (_, remove, prefs) = HybridRecommender._update(
        "replace", _split_state(), _Linked(), Slots(intent="other")
    )
    assert how == "replace"
    assert prefs == []
    assert len(remove) == 3
