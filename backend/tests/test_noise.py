from stunes.eval.noise import KINDS, apply, compose, decompose, to_keys


def test_hangul_round_trip():
    for ch in "아이유밤편지뷁":
        assert compose(*decompose(ch)) == ch


def test_english_mode_keys():
    assert to_keys("아이유") == "dkdldb"
    assert to_keys("밤편지") == "qkavuswl"
    assert to_keys("괜찮아") == "rhoscksgdk"  # compound vowel and final are two keys each


def test_same_seed_same_edit_and_one_edit_only():
    text = "비 오는 밤에 듣기 좋은 잔잔한 노래 추천해줘"
    for kind in KINDS:
        a, b = apply(kind, text, "12:1"), apply(kind, text, "12:1")
        assert a == b and a.applied and a.text != text
    typo = apply("typo", text, "12:1")
    diff = [(x, y) for x, y in zip(typo.text, text, strict=True) if x != y]
    assert len(diff) == 1  # one syllable changed, length kept


def test_spacing_keeps_the_letters():
    text = "아이유 밤편지 같은 노래 찾아줘"
    for seed in map(str, range(30)):
        n = apply("spacing", text, seed)
        assert n.text.replace(" ", "") == text.replace(" ", "") and n.text != text


def test_no_content_token_is_left_alone():
    n = apply("typo", "IU 2 곡", "1")
    assert not n.applied and n.text == "IU 2 곡"
