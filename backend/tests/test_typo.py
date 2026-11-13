import asyncio

from stunes.eval.noise import to_keys
from stunes.retrieval import typo
from stunes.retrieval.bm25 import EntityHit
from stunes.retrieval.linker import EntityLinker


def test_keys_round_trip():
    for w in ["아이유", "방탄소년단", "볼빨간사춘기", "랩/힙합", "쌀쌀함", "꽃", "뉴진스"]:
        assert typo.from_keys(to_keys(w)) == w
        assert typo.looks_hanyoung(to_keys(w))
    assert not typo.looks_hanyoung("IU") and not typo.looks_hanyoung("아이유")


def test_closest_by_jamo():
    names = [("1", "아이유"), ("2", "아이린"), ("3", "아이비")]
    assert typo.closest("아이요", names, 0.2) == "1"  # one jamo off
    assert typo.closest("완전다른", names, 0.2) is None


class FakeBM25:
    def __init__(self, names):
        self.names = names  # key -> name

    async def link(self, name, kind, k=5):
        hits = [
            EntityHit(kind, key, n, 0, 1.0, n.lower() == name.strip().lower())
            for key, n in self.names.items()
            if n.lower() == name.strip().lower() or (k > 5)
        ]
        return hits[:k]


def test_linker_fallbacks():
    lk = EntityLinker(FakeBM25({"7": "아이유", "8": "아이린"}), None, None, ["발라드", "랩/힙합"])
    assert lk.genre("발라드").via == "exact"
    assert lk.genre(to_keys("발라드")).via == "keys"
    g = lk.genre("발라두")  # a typo the alias table doesn't hold
    assert (g.key, g.via) == ("발라드", "fuzzy")
    a = asyncio.run(lk.artist(to_keys("아이유")))
    assert (a.key, a.via) == ("7", "keys")
    a = asyncio.run(lk.artist("아이요"))
    assert (a.key, a.via) == ("7", "fuzzy")


def test_read_back_sentence():
    text, changed = typo.read_back(to_keys("아이유") + " 노래 " + to_keys("발라드") + "로")
    assert text == "아이유 노래 발라드로" and len(changed) == 2
    for clean in ["Crucial Star 노래", "BTS DJ remix", "love and peace", "IU 좋아"]:
        assert typo.read_back(clean) == (clean, [])
