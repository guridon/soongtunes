import asyncio
from types import SimpleNamespace

from stunes.conversation.extract import Slots
from stunes.conversation.pipeline import HybridRecommender
from stunes.retrieval.bm25 import EntityHit
from stunes.retrieval.linker import EntityLinker

GENRES = ["발라드", "랩/힙합", "포크/블루스", "포크/블루스/컨트리", "아이돌"]


class KindBM25:
    """Exact-name hits per kind, like the entities index for these tests."""

    def __init__(self, names):
        self.names = names  # kind -> {key: name}

    async def link(self, name, kind, k=5):
        q = name.strip().lower()
        hits = [
            EntityHit(kind, key, n, 0, 1.0, n.lower() == q)
            for key, n in self.names.get(kind, {}).items()
            if n.lower() == q
        ]
        return hits[:k]


class NoVectors:
    async def active(self, kind):
        return None


def linker():
    bm25 = KindBM25(
        {
            "artist": {"10": "바닐라 어쿠스틱", "11": "요조"},
            "tag": {"겨울": "겨울", "봄": "봄", "봄노래": "봄노래", "사랑": "사랑"},
        }
    )
    return EntityLinker(bm25, NoVectors(), None, GENRES), bm25


def test_genre_slash_parts():
    lk, _ = linker()
    assert lk.genre("힙합/랩").key == "랩/힙합"
    assert lk.genre("힙/힙합").key == "랩/힙합"
    assert lk.genre("포크/블루스/컨트리").via == "exact"  # a genre of its own stays
    assert lk.genre_exact("발라두") is None  # no typo fallback here


def test_tag_generic_suffix():
    lk, _ = linker()
    for mention, tag in [("겨울 느낌", "겨울"), ("사랑 느낌도 같이", "사랑"), ("봄노래", "봄노래")]:
        got = asyncio.run(lk.tags(mention))
        assert [x.key for x in got] == [tag]
    assert asyncio.run(lk.tag_exact("겨울 분위기"))
    assert not asyncio.run(lk.tag_exact("바닐라 어쿠스틱"))


def test_values_filed_under_the_wrong_type_move():
    lk, bm25 = linker()
    fake = SimpleNamespace(_linker=lk, _bm25=bm25, _s2t=False, _reranker=None)
    slots = Slots(artists=["아이돌"], genres=["요조"], moods=["바닐라 어쿠스틱", "겨울 느낌"])
    linked, _ = asyncio.run(
        HybridRecommender._link(fake, slots, SimpleNamespace(last_results=[]), "")
    )
    assert [x.key for x in linked.genres] == ["아이돌"]
    assert sorted(x.key for x in linked.artists) == ["10", "11"]
    assert [x.key for x in linked.tags] == ["겨울"]
    neg = Slots(exclude_artists=["발라드"], exclude_moods=["요조"])
    linked, _ = asyncio.run(
        HybridRecommender._link(fake, neg, SimpleNamespace(last_results=[]), "")
    )
    assert [x.key for x in linked.neg_genres] == ["발라드"]
    assert [x.key for x in linked.neg_artists] == ["11"]
