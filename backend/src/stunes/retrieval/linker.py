"""Entity linking: surface forms from the LLM -> canonical graph keys.

v1 linked by embedding similarity, but the dump had no name-embedding indexes, so in
practice v1 fell back to case-insensitive exact match. Here each type has its own method,
compared against the others on v1-set mentions with known answers (docs/results.md,
"Entity linking: which method per type"):
- genres: name, alias table, generic suffix (every method links clean mentions at 1.00);
- artists: BM25 over names, accepted when mention and name contain each other (embedding
  rerank of the BM25 candidates: +0.020, under the bar to switch);
- tags: exact name, else BM25's top hit first, then dense (BM25 top 1 beat dense-first by
  +0.039, with fewer wrong links).
Genres and artists the method can't link get two fallbacks (retrieval/typo.py): a word typed
in English mode read back as Hangul, then jamo edit distance <= 0.2 of the length.
Since 11-07 a tag mention with a generic word after it ("겨울 느낌") links the word before it
when that is exactly a tag, and a genre written with "/" in another order links by its parts
(results.md, "말했지만 개체 연결에서 빠진 값").
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from itertools import chain, zip_longest

from stunes import metrics
from stunes.llm import Embedder
from stunes.retrieval import typo
from stunes.retrieval.bm25 import BM25
from stunes.retrieval.vector import VectorIndex

logger = logging.getLogger(__name__)

# Melon splits several genres into domestic/foreign codes with the same display name
# (e.g. 록/메탈 = GN0600 and GN1000), so genres are linked by *name*.
GENRE_ALIASES = {
    "팝": "POP",
    "pop": "POP",
    "팝송": "POP",
    "케이팝": "아이돌",
    "k-pop": "아이돌",
    "kpop": "아이돌",
    "록": "록/메탈",
    "락": "록/메탈",
    "rock": "록/메탈",
    "메탈": "록/메탈",
    "metal": "록/메탈",
    "힙합": "랩/힙합",
    "랩": "랩/힙합",
    "hiphop": "랩/힙합",
    "hip-hop": "랩/힙합",
    "rap": "랩/힙합",
    "알앤비": "R&B/Soul",
    "r&b": "R&B/Soul",
    "rnb": "R&B/Soul",
    "소울": "R&B/Soul",
    "soul": "R&B/Soul",
    "인디": "인디음악",
    "indie": "인디음악",
    "트로트": "성인가요",
    "edm": "EDM",
    "일렉트로닉": "일렉트로니카",
    "electronic": "일렉트로니카",
    "일렉": "일렉트로니카",
    "포크": "포크/블루스",
    "블루스": "포크/블루스",
    "컨트리": "포크/블루스/컨트리",
    "jpop": "J-POP",
    "j-pop": "J-POP",
    "제이팝": "J-POP",
    "ost": "OST",
    "드라마 ost": "OST",
    "재즈": "재즈",
    "jazz": "재즈",
    "클래식": "클래식",
    "classic": "클래식",
    "뉴에이지": "뉴에이지",
    "ccm": "CCM",
    "동요": "어린이/태교",
    "태교": "어린이/태교",
    "국악": "국악",
    "뮤지컬": "뮤지컬",
}

_NORM = re.compile(r"[\s'\"`’‘“”]+")
# a generic noun after a mood word, with "도" / "같이" after it
_TAG_SUFFIX = re.compile(r"\s*(?:느낌|노래|분위기|곡|음악|스타일)(?:도)?(?:\s*같이)?$")


def _norm(s: str) -> str:
    return _NORM.sub("", s).lower()


@dataclass(frozen=True)
class Linked:
    kind: str  # artist | genre | tag | song
    key: str
    name: str
    via: str  # exact | alias | bm25 | vector | context | keys | fuzzy | suffix
    score: float = 1.0


FUZZY_MAX_RATIO = 0.2  # chosen on the dev half (docs/prereg-link-methods.md)


class EntityLinker:
    def __init__(
        self,
        bm25: BM25,
        vectors: VectorIndex,
        embedder: Embedder,
        genres: list[str],
        tag_min_similarity: float = 0.55,
    ):
        self._bm25 = bm25
        self._vectors = vectors
        self._embedder = embedder
        self._genres = {_norm(g): g for g in genres}
        self._tag_min_sim = tag_min_similarity

    def genre(self, mention: str) -> Linked | None:
        hit = self._genre(mention)
        if hit is None and typo.looks_hanyoung(mention):
            back = self._genre(typo.from_keys(mention))
            hit = back and Linked("genre", back.key, back.name, "keys")
        if hit is None:
            key = typo.closest(mention, [(g, g) for g in self._genres.values()], FUZZY_MAX_RATIO)
            hit = key and Linked("genre", key, key, "fuzzy")
        return hit or None

    def genre_exact(self, mention: str) -> Linked | None:
        """A genre by name or alias only, no typo fallbacks: for moving a value the LLM filed
        under another type."""
        return self._genre(mention)

    def _genre(self, mention: str) -> Linked | None:
        n = _norm(mention)
        if n in self._genres:
            return Linked("genre", self._genres[n], self._genres[n], "exact")
        alias = GENRE_ALIASES.get(n) or GENRE_ALIASES.get(mention.strip().lower())
        if alias and _norm(alias) in self._genres:
            return Linked("genre", alias, alias, "alias")
        # "힙합/랩" for 랩/힙합: the parts that link must all name one genre
        if "/" in mention:
            parts = {self._genre(p) for p in mention.split("/") if p.strip()}
            keys = {p.key for p in parts if p}
            if len(keys) == 1:
                return Linked("genre", *keys, *keys, "alias")
        # "발라드 음악", "록 장르" -> strip the generic suffix and retry once
        stripped = re.sub(r"(음악|장르|노래|곡|계열|스타일)$", "", mention.strip()).strip()
        if stripped and stripped != mention.strip():
            return self._genre(stripped)
        return None

    async def artist(self, mention: str) -> Linked | None:
        hit = await self._artist(mention)
        if hit is None and typo.looks_hanyoung(mention):
            back = await self._artist(typo.from_keys(mention))
            hit = back and Linked("artist", back.key, back.name, "keys", back.score)
        if hit is None:
            cands = await self._bm25.link(mention, "artist", k=20)
            key = typo.closest(mention, [(c.key, c.name) for c in cands], FUZZY_MAX_RATIO)
            name = next((c.name for c in cands if c.key == key), "")
            hit = key and Linked("artist", key, name, "fuzzy")
        return hit or None

    async def artist_exact(self, mention: str) -> Linked | None:
        """An artist whose name is exactly the mention, else None."""
        hits = await self._bm25.link(mention, "artist", k=1)
        if hits and hits[0].exact:
            return Linked("artist", hits[0].key, hits[0].name, "exact", hits[0].score)
        return None

    async def _artist(self, mention: str) -> Linked | None:
        hits = await self._bm25.link(mention, "artist", k=5)
        if not hits:
            return None
        top = hits[0]
        if top.exact:
            return Linked("artist", top.key, top.name, "exact", top.score)
        # without an exact match, require the mention to be (nearly) contained in the name
        if _norm(mention) in _norm(top.name) or _norm(top.name) in _norm(mention):
            return Linked("artist", top.key, top.name, "bm25", top.score)
        return None

    async def tag_exact(self, mention: str) -> bool:
        """The mention, or the mention without its generic suffix, is exactly a tag name."""
        m = mention.strip()
        for form in dict.fromkeys([m, _TAG_SUFFIX.sub("", m)]):
            hits = await self._bm25.link(form, "tag", k=1) if form else []
            if hits and hits[0].exact:
                return True
        return False

    async def tags(
        self, mention: str, k: int = 2, context_tags: list[str] | None = None
    ) -> list[Linked]:
        """A mood phrase -> up to k tags.

        exact tag name wins outright. Otherwise, when the user refers back ("이런 X 느낌"),
        the closest tag *of the previous results* comes first, then global candidates:
        dense handles paraphrases ("마음을 울리는" -> 감성), BM25 spelling variants
        ("비오는 날" -> 비오는날).
        """
        has_vectors = await self._vectors.active("tag") is not None
        q_task = (
            asyncio.ensure_future(self._embedder.embed_query(f"음악 태그: {mention}"))
            if has_vectors
            else None
        )
        lexical = await self._bm25.link(mention, "tag", k=5)
        if lexical and lexical[0].exact:
            if q_task:
                q_task.cancel()
            return [Linked("tag", lexical[0].key, lexical[0].name, "exact", lexical[0].score)]
        # "겨울 느낌", "봄 노래", "사랑 느낌도 같이": the tag is the word before the generic one
        stripped = _TAG_SUFFIX.sub("", mention.strip())
        if stripped and stripped != mention.strip():
            base = await self._bm25.link(stripped, "tag", k=1)
            if base and base[0].exact:
                if q_task:
                    q_task.cancel()
                return [Linked("tag", base[0].key, base[0].name, "suffix", base[0].score)]

        contextual: list[Linked] = []
        dense: list[Linked] = []
        if q_task is not None:
            q = await q_task
            hits, ranked = await asyncio.gather(
                self._vectors.search("tag", q, k=5),
                self._vectors.rank_among("tag", q, context_tags or []),
                return_exceptions=True,
            )
            # the dense side is an improvement, not a requirement: when a search hits its
            # statement timeout under load, link with exact/BM25 alone instead of failing the
            # turn (troubleshooting #19)
            if isinstance(hits, BaseException):
                logger.warning("dense tag linking dropped: %r", hits)
                metrics.ERRORS.labels("link_tag_dense").inc()
                hits = []
            if isinstance(ranked, BaseException):
                logger.warning("contextual tag linking dropped: %r", ranked)
                metrics.ERRORS.labels("link_tag_context").inc()
                ranked = []
            dense = [Linked("tag", t, t, "vector", s) for t, s in hits if s >= self._tag_min_sim]
            contextual = [Linked("tag", t, t, "context", s) for t, s in ranked[:1]]
        # BM25's top hit leads; the rest interleave dense and contained BM25 names
        first = [Linked("tag", h.key, h.name, "bm25", h.score) for h in lexical[:1]]
        lex = [
            Linked("tag", h.key, h.name, "bm25", h.score)
            for h in lexical[1:]
            if _norm(mention) in _norm(h.name) or _norm(h.name) in _norm(mention)
        ]
        out: list[Linked] = []
        for cand in chain(contextual, first, chain.from_iterable(zip_longest(dense, lex))):
            if cand is not None and all(c.key != cand.key for c in out):
                out.append(cand)
        return out[:k]

    async def song(self, title: str, artist: str | None) -> Linked | None:
        hits = await self._bm25.find_song(title, artist, k=3)
        if not hits:
            return None
        cards = await self._bm25.get_songs([hits[0][0]])
        card = cards.get(hits[0][0])
        if card is None:
            return None
        # accept only if the title really matches; BM25 always returns *something*
        if _norm(title) not in _norm(card["title"]) and _norm(card["title"]) not in _norm(title):
            return None
        return Linked("song", hits[0][0], card["title"], "bm25", hits[0][1])
