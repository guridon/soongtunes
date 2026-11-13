from stunes.conversation import signs


def one(message: str, op: str, value: str) -> list[list[str]]:
    return signs.fix({"ops": [{"op": op, "kind": "genre", "value": value}]}, message)[1]


def test_exclusion_on_an_ask_becomes_positive():
    assert one("아이돌은?", "exclude", "아이돌") == [["exclude", "add", "아이돌"]]
    assert one("인디로 골라줘", "exclude", "인디음악") == [["exclude", "add", "인디음악"]]
    assert one("그중에 아이돌만", "exclude", "아이돌")


def test_exclusion_with_a_negation_word_stays():
    assert one("발라드는 빼줘", "exclude", "발라드") == []
    assert one("발라드 빼고 아이돌은?", "exclude", "아이돌") == []  # mixed signs: left alone


def test_inferred_exclusion_stays():
    # the LLM excluded the artist shown before; the user never named it
    assert one("같은 장르의 OST 중에서 다른 것도 추천해줄래?", "exclude", "박효신") == []


def test_positive_followed_by_a_negation_word_becomes_exclusion():
    assert one("이별 노래는 빼고", "add", "이별") == [["add", "exclude", "이별"]]
    assert one("아이돌 말고 다른 걸로", "add", "아이돌") == [["add", "exclude", "아이돌"]]
    assert one("발라드 말고 댄스로 바꿔줘", "add", "댄스") == []


def test_other_ops_untouched():
    raw = {"ops": [{"op": "no_change", "kind": None, "value": None}]}
    assert signs.fix(raw, "아이돌은?") == (raw, [])
