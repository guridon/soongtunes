"""Parameterized Cypher templates for candidate generation and verification.

The LLM never writes Cypher on this path: it fills slots, and the
slots become parameters of fixed templates. Query text is assembled only from the constant
fragments below, never from user input.
"""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from stunes.db import Graph

if TYPE_CHECKING:
    from stunes.retrieval.ppr import PPR


@dataclass
class Conditions:
    """Accumulated search conditions; all keys are canonical graph keys."""

    tags: list[str] = field(default_factory=list)  # Tag.name (soft: rank by #matches)
    genres: list[str] = field(default_factory=list)  # Genre.name (hard)
    artists: list[str] = field(default_factory=list)  # Artist.artist_id (hard)
    neg_tags: list[str] = field(default_factory=list)
    neg_genres: list[str] = field(default_factory=list)
    neg_artists: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)  # Song.song_id never to return
    year_from: int | None = None
    year_to: int | None = None

    def params(self) -> dict:
        return {
            "tags": self.tags,
            "genres": self.genres,
            "artists": self.artists,
            "neg_tags": self.neg_tags,
            "neg_genres": self.neg_genres,
            "neg_artists": self.neg_artists,
            "exclude": self.exclude,
            "year_from": self.year_from,
            "year_to": self.year_to,
        }

    @property
    def has_positive(self) -> bool:
        return bool(self.tags or self.genres or self.artists)

    @property
    def has_hard(self) -> bool:
        return bool(
            self.genres
            or self.artists
            or self.neg_genres
            or self.neg_artists
            or self.neg_tags
            or self.year_from
            or self.year_to
        )


_HARD_FILTERS = """
  AND NOT s.song_id IN $exclude
  AND ($genres = [] OR EXISTS { (s)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $genres })
  AND ($artists = [] OR EXISTS { (s)-[:PERFORMED_BY]->(a:Artist) WHERE a.artist_id IN $artists })
  AND NOT EXISTS { (s)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $neg_genres }
  AND NOT EXISTS { (s)-[:PERFORMED_BY]->(a:Artist) WHERE a.artist_id IN $neg_artists }
  AND NOT EXISTS { (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $neg_tags }
  AND ($year_from IS NULL OR toInteger(left(s.issue_date, 4)) >= $year_from)
  AND ($year_to IS NULL OR toInteger(left(s.issue_date, 4)) <= $year_to)
"""

# Plan choice is cost-based on anchor size (Tag/Genre.song_count, precomputed):
#   small anchor  -> expand from it and rank every candidate exactly
#   big anchor    -> walk songs in popularity order (index-backed ORDER BY) and stop at
#                    scan_k matches. Tag '감성' (121k songs): 1.6s expanding -> ~40ms scanning.
#                    A rare tag (332 songs) is the opposite: ~2.5s scanning, so it expands.
# Crossover, warm, k=100 (troubleshooting #18): 4,357 songs expand 138 / scan 262 ms;
# 6,054 -> 141 / 48 ms; 7,092 -> 147 / 43 ms; two tags, 9,778 -> 1,317 / 399 ms.
# (It was 20,000, set from the two extremes alone.)
ANCHOR_MAX_SONGS = 5_000

_RANK_BY_TAG_HITS = """
WITH s, COUNT { (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags } AS tag_hits
ORDER BY tag_hits DESC, s.popularity DESC, s.song_id
LIMIT $k
RETURN s.song_id AS song_id, tag_hits
"""

_EXPAND = {
    "artist": """
        MATCH (a0:Artist) WHERE a0.artist_id IN $artists
        MATCH (s:Song)-[:PERFORMED_BY]->(a0)
        WITH DISTINCT s
        WHERE true {filters}
    """,
    "genre": """
        MATCH (g0:Genre) WHERE g0.name IN $genres
        MATCH (s:Song)-[:HAS_GENRE]->(g0)
        WITH DISTINCT s
        WHERE ($tags = [] OR EXISTS { (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags }) {filters}
    """,
    "tag": """
        MATCH (t0:Tag) WHERE t0.name IN $tags
        MATCH (s:Song)-[:HAS_TAG]->(t0)
        WITH DISTINCT s
        WHERE true {filters}
    """,
}

# The scan walks at most SCAN_DEPTH songs, then the plan expands from the smallest anchor if
# it found fewer than k. Each anchor alone is common among popular songs, their combination
# need not be: "운동" (47k songs) and 재즈 (48k) together, or a genre within one decade. An
# unbounded walk then ran into the 3 s budget: in the 2x2 eval run the graph source was lost
# on 605 of 7,985 turns, 45% of the tag + genre + decade turns (results.md, "여러 조건이
# 겹칠 때의 그래프 검색").
SCAN_DEPTH = 20_000  # chosen on the dev half of recorded calls (prereg-graph-plan-link-retype.md)

_SCAN = """
MATCH (s:Song) WHERE s.popularity >= 0
WITH s ORDER BY s.popularity DESC LIMIT $scan_depth
WITH s WHERE ($tags = [] OR EXISTS { (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags }) {filters}
WITH s LIMIT $scan_k
"""

_ANCHOR_SIZES = """
CALL () { MATCH (t:Tag) WHERE t.name IN $tags RETURN coalesce(sum(t.song_count), 0) AS tags }
CALL () { MATCH (g:Genre) WHERE g.name IN $genres RETURN coalesce(sum(g.song_count), 0) AS genres }
RETURN tags, genres
"""

# Q3 in the spec: songs that share playlists with the seeds. A popular seed sits in
# thousands of playlists, so only its most-liked playlists are expanded (supernode cap).
_COOCCUR = """
MATCH (seed:Song) WHERE seed.song_id IN $seeds
CALL (seed) {
  MATCH (seed)<-[:INCLUDES]-(p:Playlist)
  WHERE NOT p.playlist_id IN $exclude_playlists
  WITH p ORDER BY toInteger(p.like_cnt) DESC LIMIT $max_playlists
  RETURN p
}
MATCH (p)-[:INCLUDES]->(s:Song)
WHERE NOT s.song_id IN $seeds
WITH s, count(DISTINCT p) AS co
WHERE true {filters}
WITH s, co, COUNT { (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags } AS tag_hits
ORDER BY tag_hits DESC, co DESC, s.popularity DESC, s.song_id
LIMIT $k
RETURN s.song_id AS song_id, co
"""

WEIGHTED_DEPTH = 1_000
# "auto": degree from two seeds up, count for one. Lists alone on the PPR dev split, Recall@50
# count vs degree: 1 seed 0.136 vs 0.130, 2 seeds 0.148 vs 0.181, 3 seeds 0.158 vs 0.208,
# 5 seeds 0.161 vs 0.247 (stunes.bench.seed_sweep). One seed has no second seed to agree with,
# and 1/degree then just favours its smallest playlists.
COOCCUR_WEIGHTS = {"count", "degree", "auto"}

_TAG_HITS = """
UNWIND $ids AS id
MATCH (s:Song {song_id: id})
RETURN id AS song_id, COUNT { (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags } AS tag_hits
"""

_VIOLATIONS = """
UNWIND $ids AS id
MATCH (s:Song {song_id: id})
WITH s,
  ($genres <> [] AND NOT EXISTS { (s)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $genres }) AS no_genre,
  ($artists <> [] AND NOT EXISTS { (s)-[:PERFORMED_BY]->(a:Artist) WHERE a.artist_id IN $artists }) AS no_artist,
  EXISTS { (s)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $neg_genres } AS bad_genre,
  EXISTS { (s)-[:PERFORMED_BY]->(a:Artist) WHERE a.artist_id IN $neg_artists } AS bad_artist,
  EXISTS { (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $neg_tags } AS bad_tag,
  ($year_from IS NOT NULL AND coalesce(toInteger(left(s.issue_date, 4)), 0) < $year_from) AS too_old,
  ($year_to IS NOT NULL AND coalesce(toInteger(left(s.issue_date, 4)), 9999) > $year_to) AS too_new
WHERE no_genre OR no_artist OR bad_genre OR bad_artist OR bad_tag OR too_old OR too_new
RETURN s.song_id AS song_id,
  [r IN [['missing_genre', no_genre], ['missing_artist', no_artist], ['excluded_genre', bad_genre],
         ['excluded_artist', bad_artist], ['excluded_tag', bad_tag], ['before_year', too_old],
         ['after_year', too_new]] WHERE r[1] | r[0]] AS reasons
"""


# what the agent allows a graph source (agent/graph.py SOURCE_TIMEOUT_S); the server stops
# there too, so abandoned candidate queries don't keep running (troubleshooting #18)
SOURCE_TIMEOUT_S = 3.0


class GraphRetriever:
    def __init__(
        self,
        graph: Graph,
        timeout_s: float = SOURCE_TIMEOUT_S,
        llm_tags: bool = False,
        weighted: "PPR | None" = None,
    ):
        self._g = graph
        self._timeout = timeout_s
        self._llm_tags = llm_tags
        # in-memory copy of the playlist graph for cooccurring(weights="degree")
        self.weighted = weighted

    def _q(self, query: str) -> str:
        # LLM-predicted tags are their own edge type (llm_features/tagger.py project), so with
        # the switch off every query is exactly what it was. Only tag-less songs get them, so a
        # song never has both edges to one tag and tag_hits counts each tag once.
        return query.replace("[:HAS_TAG]", "[:HAS_TAG|HAS_LLM_TAG]") if self._llm_tags else query

    async def plan(self, c: Conditions) -> tuple[str, str] | None:
        """(plan, the smallest anchor's expansion): the second is what a short scan falls
        back to."""
        if c.artists:
            return "expand:artist", "expand:artist"
        if not (c.tags or c.genres):
            return None
        sizes = (
            await self._g.read(_ANCHOR_SIZES, timeout_s=self._timeout, tags=c.tags, genres=c.genres)
        )[0]
        options = []
        if c.tags:
            options.append((sizes["tags"], "expand:tag"))
        if c.genres:
            options.append((sizes["genres"], "expand:genre"))
        size, expand = min(options)
        return ("scan:popularity" if size > ANCHOR_MAX_SONGS else expand), expand

    async def by_conditions(
        self, c: Conditions, k: int = 100
    ) -> tuple[list[tuple[str, int]], str | None]:
        """Songs satisfying the hard conditions, ranked by tag matches then popularity."""
        planned = await self.plan(c)
        if planned is None:
            return [], None
        plan, expand = planned
        rows = await self._run(plan, c, k)
        if plan == "scan:popularity" and len(rows) < k:
            rows = await self._run(expand, c, k)
            plan = f"{plan}>{expand}"
        return [(r["song_id"], r["tag_hits"]) for r in rows], plan

    async def _run(self, plan: str, c: Conditions, k: int) -> list[dict]:
        head = _SCAN if plan == "scan:popularity" else _EXPAND[plan.split(":")[1]]
        query = self._q(head.replace("{filters}", _HARD_FILTERS) + _RANK_BY_TAG_HITS)
        return await self._g.read(
            query,
            timeout_s=self._timeout,
            k=k,
            scan_k=max(3 * k, 300),
            scan_depth=SCAN_DEPTH,
            **c.params(),
        )

    async def cooccurring(
        self,
        seeds: list[str],
        c: Conditions,
        k: int = 100,
        max_playlists: int = 300,
        exclude_playlists: list[str] | None = None,
        weights: str = "count",
        order_by_tags: bool = True,
    ) -> list[tuple[str, float]]:
        """exclude_playlists keeps evaluation honest: a held-out playlist must not be the
        bridge that leads back to its own hidden songs.

        Two scorings (settings.cooccur_weights_feed / _chat):
        - count (Cypher below): each seed's `max_playlists` most-liked playlists; songs by
          tag hits, then by the number of those playlists that hold them.
        - degree (`self.weighted`): every playlist of a seed, each worth 1 / (the seed's
          playlists x the playlist's songs), plus artist edges; then the hard filters, and
          tag hits first when `order_by_tags`. On the home feed (no tag ordering: its tags
          are soft preferences the ranker reads) the source alone rose from Recall@50 0.187
          to 0.262 (docs/results.md, "홈 피드: Personalized PageRank와 동시등장").
        """
        if not seeds:
            return []
        if weights == "auto":
            weights = "degree" if len(seeds) >= 2 else "count"
        if weights == "degree":
            return await self._cooccur_weighted(
                seeds, c, k, exclude_playlists or [], order_by_tags and bool(c.tags)
            )
        if weights != "count":
            raise ValueError(f"cooccur weights: {weights!r}")
        query = self._q(_COOCCUR.replace("{filters}", _HARD_FILTERS))
        rows = await self._g.read(
            query,
            timeout_s=self._timeout,
            seeds=seeds,
            k=k,
            max_playlists=max_playlists,
            exclude_playlists=exclude_playlists or [],
            **c.params(),
        )
        return [(r["song_id"], r["co"]) for r in rows]

    async def _cooccur_weighted(
        self,
        seeds: list[str],
        c: Conditions,
        k: int,
        exclude_playlists: list[str],
        by_tags: bool,
    ) -> list[tuple[str, float]]:
        if self.weighted is None:
            raise RuntimeError("cooccur weights=degree needs the graph file (stunes ppr-build)")
        banned = list(c.exclude) + list(seeds)
        # filters drop some and tag ordering reaches past the top k: fetch a deeper list.
        # The Cypher count orders its whole neighbourhood by tags; this orders the 1,000
        # best-weighted songs, an approximation.
        n = max(k * 3, WEIGHTED_DEPTH) if by_tags else (k * 3 if c.has_hard else k)
        rows = await asyncio.to_thread(self.weighted.cooccur, seeds, exclude_playlists, banned, n)
        if c.has_hard:
            bad = await self.violations([sid for sid, _ in rows], c)
            rows = [r for r in rows if r[0] not in bad]
        if by_tags and rows:
            hits = {
                r["song_id"]: r["tag_hits"]
                for r in await self._g.read(
                    self._q(_TAG_HITS),
                    timeout_s=self._timeout,
                    ids=[sid for sid, _ in rows],
                    tags=c.tags,
                )
            }
            rows.sort(key=lambda r: -hits.get(r[0], 0))  # stable: weight order within a count
        return rows[:k]

    async def popular(self, c: Conditions, k: int = 100) -> list[str]:
        """Most-played songs passing the hard filters (index-ordered scan, stops at k)."""
        query = self._q(
            "MATCH (s:Song) WHERE s.popularity >= 0 WITH s ORDER BY s.popularity DESC "
            "WHERE true" + _HARD_FILTERS + " WITH s LIMIT $k RETURN s.song_id AS song_id"
        )
        rows = await self._g.read(query, timeout_s=self._timeout, k=k, **c.params())
        return [r["song_id"] for r in rows]

    async def violations(self, ids: list[str], c: Conditions) -> dict[str, list[str]]:
        """Which candidates break a hard condition — the deterministic verifier."""
        if not ids or not c.has_hard:
            return {}
        rows = await self._g.read(self._q(_VIOLATIONS), ids=ids, **c.params())
        return {r["song_id"]: r["reasons"] for r in rows}
