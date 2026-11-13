"""Wiring: builds every component once and exposes the strategies behind one interface."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from stunes.agent.graph import AgentRecommender
from stunes.baseline.v1 import V1Baseline
from stunes.config import Settings
from stunes.conversation.delta import DeltaExtractor
from stunes.conversation.extract import SlotExtractor
from stunes.conversation.pipeline import HybridRecommender, TurnResult
from stunes.conversation.state import StateStore
from stunes.db import Stores
from stunes.features.store import FeatureStore
from stunes.llm import ChatModel, Embedder
from stunes.rerank.serve import ChatReranker
from stunes.retrieval import graph as graph_mod
from stunes.retrieval import vector as vector_mod
from stunes.retrieval.bm25 import BM25
from stunes.retrieval.fusion import rrf
from stunes.retrieval.graph import Conditions, GraphRetriever
from stunes.retrieval.linker import EntityLinker
from stunes.retrieval.ppr import PPR
from stunes.retrieval.vector import VectorIndex

logger = logging.getLogger(__name__)

STRATEGIES = {
    "agent": "v2 Phase 5: LangGraph plan → parallel tools with timeouts → feature-aware ranker → verify → relax ≤2",
    "hybrid": "v2: LLM slots → hybrid linking → graph + co-occurrence + BM25 + dense → RRF → verify",
    "hybrid_s2t": "hybrid, and a mood turn also links the whole sentence's nearest tag",
    "hybrid_llmstate": "hybrid without the stored state: the LLM rebuilds the whole state from the history each turn",
    "hybrid_cumstate": "hybrid with the stored state, and the LLM also restates the whole state from the history",
    "hybrid_ctx": "the stored state and the numbered last results go into the LLM's context; its restated slots become the state",
    "hybrid_ctxsplit": "hybrid_ctx, but the stored exclusions are kept by rule and a turn that restates nothing keeps the state",
    "hybrid_ctx_auto": "hybrid_ctx with co-occurrence scored by 1/degree when a turn has two or more seed songs (docs/prereg-chat-cooccur-auto.md)",
    "hybrid_ctx_keys": "hybrid_ctx with words typed in English mode read back as Hangul before the LLM (docs/prereg-graph-plan-link-retype.md)",
    "hybrid_ctx_recent": "hybrid_ctx with only the last exchange as history",
    "hybrid_delta": "the stored state in the context; the LLM outputs only this turn's changes, code applies them",
    "hybrid_delta_recent": "hybrid_delta with only the last exchange as history",
    "hybrid_delta_keys": "the served path: hybrid_delta, words typed in English mode read back as Hangul before the LLM, and signs the message contradicts flipped (docs/prereg-context-long.md, prereg-graph-plan-link-retype.md, prereg-sign-rule.md)",
    "v1_contextual": "v1 port: VirtualQuery state + LLM Text2Cypher (GraphCypherQAChain)",
    "v1_stateless": "v1 port: query rewriting + LLM Text2Cypher",
    "bm25": "BM25 only on the raw message (OpenSearch, Nori)",
    "vector": "dense only on the raw message (Harrier, pgvector)",
}

_NEIGHBORHOOD = """
MATCH (s:Song {song_id: $id})
OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
WITH s, collect(DISTINCT a {.artist_id, .name}) AS artists
OPTIONAL MATCH (s)-[:IN_ALBUM]->(al:Album)
WITH s, artists, head(collect(al {.album_id, .title})) AS album
RETURN s.song_id AS song_id, s.popularity AS popularity, artists, album,
       [(s)-[:HAS_GENRE]->(g:Genre) WHERE g.name IS NOT NULL | g.name] AS genres,
       COUNT { (s)-[:HAS_TAG]->() } AS tag_count
"""

_SAME_ARTIST = """
MATCH (:Song {song_id: $id})-[:PERFORMED_BY]->(a:Artist)<-[:PERFORMED_BY]-(o:Song)
WHERE o.song_id <> $id
WITH DISTINCT o ORDER BY o.popularity DESC LIMIT $k
RETURN o.song_id AS song_id
"""

_STATS = """
CALL () { MATCH (n:Song) RETURN count(n) AS songs }
CALL () { MATCH (n:Artist) RETURN count(n) AS artists }
CALL () { MATCH (n:Album) RETURN count(n) AS albums }
CALL () { MATCH (n:Playlist) RETURN count(n) AS playlists }
CALL () { MATCH (n:Tag) RETURN count(n) AS tags }
CALL () { MATCH ()-[r:INCLUDES]->() RETURN count(r) AS includes }
CALL () { MATCH ()-[r:HAS_TAG]->() RETURN count(r) AS has_tag }
RETURN songs, artists, albums, playlists, tags, includes, has_tag
"""


@dataclass
class Services:
    settings: Settings
    stores: Stores
    llm: ChatModel
    embedder: Embedder
    bm25: BM25
    vectors: VectorIndex
    graph: GraphRetriever
    store: StateStore
    hybrid: HybridRecommender
    genres: list[str]
    stats: dict[str, int]
    features: FeatureStore
    agent: AgentRecommender
    hybrid_s2t: HybridRecommender
    hybrid_llmstate: HybridRecommender
    hybrid_cumstate: HybridRecommender
    hybrid_ctx: HybridRecommender
    hybrid_ctx_auto: HybridRecommender
    hybrid_ctx_keys: HybridRecommender
    hybrid_ctxsplit: HybridRecommender
    hybrid_ctx_recent: HybridRecommender
    hybrid_delta: HybridRecommender
    hybrid_delta_recent: HybridRecommender
    hybrid_delta_keys: HybridRecommender
    _v1: V1Baseline | None = None
    _v1_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    async def build(cls, settings: Settings) -> "Services":
        stores = await Stores.open(settings)
        llm, embedder = ChatModel(settings), Embedder(settings)
        scale, llm_tags = settings.source_budget_scale, settings.llm_tags_in_retrieval
        bm25 = BM25(stores.search, settings.songs_index, settings.entities_index, llm_tags=llm_tags)
        vectors = VectorIndex(stores.pg, timeout_ms=vector_mod.SEARCH_TIMEOUT_MS * scale)
        weights = {settings.cooccur_weights_feed, settings.cooccur_weights_chat}
        if not weights <= graph_mod.COOCCUR_WEIGHTS:
            raise ValueError(f"cooccur weights: {weights}")
        weighted = None
        if weights - {"count"}:
            if settings.ppr_graph_file.exists():
                weighted = await asyncio.to_thread(PPR, settings.ppr_graph_file)
            else:  # serve the count rather than fail every request; health shows it
                logger.error(
                    "cooccur weights %s need %s (run `stunes ppr-build`); serving count",
                    weights,
                    settings.ppr_graph_file,
                )
                settings = settings.model_copy(
                    update={"cooccur_weights_feed": "count", "cooccur_weights_chat": "count"}
                )
        graph = GraphRetriever(
            stores.graph,
            timeout_s=graph_mod.SOURCE_TIMEOUT_S * scale,
            llm_tags=llm_tags,
            weighted=weighted,
        )
        store = StateStore(stores.graph)
        genres = [
            r["name"]
            for r in await stores.graph.read(
                "MATCH (g:Genre) WHERE g.name IS NOT NULL "
                "RETURN DISTINCT g.name AS name ORDER BY name"
            )
        ]
        features = FeatureStore(stores.pg, stores.redis)
        reranker = ChatReranker(settings, embedder, vectors, graph, features)
        parts = dict(
            llm=llm,
            embedder=embedder,
            extractor=SlotExtractor(llm, genres),
            linker=EntityLinker(bm25, vectors, embedder, genres),
            store=store,
            graph=graph,
            bm25=bm25,
            vectors=vectors,
            reranker=reranker,
        )
        hybrid = HybridRecommender(settings, **parts)
        hybrid_s2t = HybridRecommender(settings, **parts, name="hybrid_s2t", s2t=True)
        # docs/prereg-context-eval.md: the same cumulative extractor without and with the state
        cumulative = SlotExtractor(llm, genres, cumulative=True)
        hybrid_llmstate = HybridRecommender(
            settings, **{**parts, "extractor": cumulative}, name="hybrid_llmstate", stateful=False
        )
        hybrid_cumstate = HybridRecommender(
            settings, **{**parts, "extractor": cumulative}, name="hybrid_cumstate"
        )
        hybrid_ctx = HybridRecommender(
            settings,
            **{**parts, "extractor": SlotExtractor(llm, genres, state_context=True)},
            name="hybrid_ctx",
            state_mode="replace",
        )
        hybrid_ctx_auto = HybridRecommender(
            settings.model_copy(update={"cooccur_weights_chat": "auto"}),
            **{**parts, "extractor": SlotExtractor(llm, genres, state_context=True)},
            name="hybrid_ctx_auto",
            state_mode="replace",
        )
        hybrid_ctx_keys = HybridRecommender(
            settings,
            **{**parts, "extractor": SlotExtractor(llm, genres, state_context=True)},
            name="hybrid_ctx_keys",
            state_mode="replace",
            read_back_keys=True,
        )
        hybrid_ctxsplit = HybridRecommender(
            settings,
            **{**parts, "extractor": SlotExtractor(llm, genres, split=True)},
            name="hybrid_ctxsplit",
            state_mode="split",
        )
        # docs/prereg-context-2x2.md: input (whole history or the last exchange) x output (the
        # whole state or the changes)
        hybrid_ctx_recent = HybridRecommender(
            settings,
            **{**parts, "extractor": SlotExtractor(llm, genres, state_context=True, recent=True)},
            name="hybrid_ctx_recent",
            state_mode="replace",
        )
        hybrid_delta = HybridRecommender(
            settings,
            **{**parts, "extractor": DeltaExtractor(llm, genres)},
            name="hybrid_delta",
            state_mode="delta",
        )
        hybrid_delta_recent = HybridRecommender(
            settings,
            **{**parts, "extractor": DeltaExtractor(llm, genres, recent=True)},
            name="hybrid_delta_recent",
            state_mode="delta",
        )
        # served from 2026-11-07: hybrid_delta passed on the 12-turn set, and the read-back
        # passed on hybrid_ctx; the read-back only changes the message before the extractor.
        # The sign rule from 2026-11-08 (replayed on the 12-turn run's slots)
        hybrid_delta_keys = HybridRecommender(
            settings,
            **{**parts, "extractor": DeltaExtractor(llm, genres)},
            name="hybrid_delta_keys",
            state_mode="delta",
            read_back_keys=True,
            fix_signs=True,
        )
        stats = (await stores.graph.read(_STATS))[0]
        agent = AgentRecommender(hybrid, graph, bm25, vectors, features)
        return cls(
            settings,
            stores,
            llm,
            embedder,
            bm25,
            vectors,
            graph,
            store,
            hybrid,
            genres,
            stats,
            features,
            agent,
            hybrid_s2t,
            hybrid_llmstate,
            hybrid_cumstate,
            hybrid_ctx,
            hybrid_ctx_auto,
            hybrid_ctx_keys,
            hybrid_ctxsplit,
            hybrid_ctx_recent,
            hybrid_delta,
            hybrid_delta_recent,
            hybrid_delta_keys,
        )

    async def close(self) -> None:
        await self.stores.close()

    async def v1(self) -> V1Baseline:
        """The v1 port, built on first use. LangChain's Neo4jGraph reads the whole schema at
        construction, and that scan can trip over session nodes deleted mid-read ("Unable to
        load NODE"). At startup that failure killed the worker; here it fails one request."""
        async with self._v1_lock:
            if self._v1 is None:
                for attempt in range(3):
                    try:
                        self._v1 = await asyncio.to_thread(
                            V1Baseline, self.settings, self.stores.graph, self.store, self.bm25
                        )
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        logger.warning("v1 baseline init failed, retrying", exc_info=True)
                        await asyncio.sleep(1)
            return self._v1

    # ------------------------------------------------------------- strategies
    @staticmethod
    def session_key(strategy: str, session_id: str) -> str:
        # each strategy keeps its own state, so the compare view can share one session id
        return f"{strategy}:{session_id}"

    async def chat(
        self,
        strategy: str,
        session_id: str,
        message: str,
        limit: int | None,
        explain: bool,
        debug: bool = False,
    ) -> TurnResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy}")
        sid = self.session_key(strategy, session_id)
        k = max(1, min(50, limit or self.settings.default_limit))
        t0 = time.perf_counter()
        if strategy == "hybrid":
            return await self.hybrid.turn(sid, message, limit=limit, explain=explain, debug=debug)
        if strategy == "hybrid_s2t":
            return await self.hybrid_s2t.turn(
                sid, message, limit=limit, explain=explain, debug=debug
            )
        if strategy in (
            "hybrid_llmstate",
            "hybrid_cumstate",
            "hybrid_ctx",
            "hybrid_ctx_auto",
            "hybrid_ctx_keys",
            "hybrid_ctxsplit",
            "hybrid_ctx_recent",
            "hybrid_delta",
            "hybrid_delta_recent",
            "hybrid_delta_keys",
        ):
            return await getattr(self, strategy).turn(
                sid, message, limit=limit, explain=explain, debug=debug
            )
        if strategy == "agent":
            return await self.agent.turn(sid, message, limit=limit, explain=explain, debug=debug)
        if strategy.startswith("v1_"):
            v1 = await self.v1()
            answer, ids, trace = await v1.turn(strategy.removeprefix("v1_"), sid, message)
            songs = await self._cards(ids[:k])
        else:
            ranked = await self.search(message, strategy, k)
            songs, answer, trace = ranked, f"'{message}' 검색 결과 {len(ranked)}곡", {}
        trace["timings_ms"] = {"total": round((time.perf_counter() - t0) * 1000, 1)}
        return TurnResult(sid, strategy, answer, songs, {"positive": [], "negative": []}, trace)

    async def search(self, q: str, mode: str, k: int) -> list[dict[str, Any]]:
        """Single-shot search without the LLM: bm25 | vector | hybrid (RRF of both)."""
        ranked: dict[str, list[str]] = {}
        if mode in ("bm25", "hybrid"):
            ranked["bm25"] = [sid for sid, _ in await self.bm25.search(q, k=max(k, 50))]
        if mode in ("vector", "hybrid"):
            vec = await self.embedder.embed_query(q)
            ranked["vector"] = [
                sid for sid, _ in await self.vectors.search("song", vec, k=max(k, 50))
            ]
        fused = rrf(ranked)[:k]
        cards = await self.bm25.get_songs([f.song_id for f in fused])
        return [
            {**cards[f.song_id], "score": round(f.score, 5), "sources": f.ranks}
            for f in fused
            if f.song_id in cards
        ]

    async def _cards(self, ids: list[str]) -> list[dict[str, Any]]:
        cards = await self.bm25.get_songs(ids)
        return [
            {**cards[i], "sources": {"text2cypher": n + 1}} for n, i in enumerate(ids) if i in cards
        ]

    async def reset(self, strategy: str, session_id: str) -> None:
        await self.store.clear(self.session_key(strategy, session_id))

    async def session(self, strategy: str, session_id: str) -> dict[str, Any]:
        st = await self.store.load(self.session_key(strategy, session_id))
        return {
            "turn": st.turn,
            "positive": [vars(p) for p in st.prefs if p.positive],
            "negative": [vars(p) for p in st.prefs if not p.positive],
            "history": st.history,
            "last_results": st.last_results,
        }

    # ------------------------------------------------------------- catalog
    async def song(self, song_id: str) -> dict[str, Any] | None:
        rows = await self.stores.graph.read(_NEIGHBORHOOD, id=song_id)
        if not rows:
            return None
        card = (await self.bm25.get_songs([song_id])).get(song_id, {})
        return {**card, **rows[0]}

    async def neighborhood(self, song_id: str, k: int = 8) -> dict[str, Any] | None:
        song = await self.song(song_id)
        if song is None:
            return None
        co, same = await asyncio.gather(
            self.graph.cooccurring([song_id], Conditions(), k=k),
            self.stores.graph.read(_SAME_ARTIST, id=song_id, k=6),
        )
        co_cards, same_cards = await asyncio.gather(
            self.bm25.get_songs([sid for sid, _ in co]),
            self.bm25.get_songs([r["song_id"] for r in same]),
        )
        return {
            "song": song,
            "cooccurring": [
                {**co_cards[sid], "shared_playlists": n} for sid, n in co if sid in co_cards
            ],
            "same_artist": [same_cards[r["song_id"]] for r in same if r["song_id"] in same_cards],
        }

    async def health(self) -> dict[str, Any]:
        async def probe(name, coro):
            t = time.perf_counter()
            try:
                await asyncio.wait_for(coro, timeout=5)
                return name, {"ok": True, "ms": round((time.perf_counter() - t) * 1000, 1)}
            except Exception as e:
                return name, {"ok": False, "error": repr(e)[:200]}

        results = await asyncio.gather(
            probe("neo4j", self.stores.graph.ping()),
            probe("opensearch", self.bm25.ping()),
            probe("postgres", self.stores.pg.fetchval("SELECT 1")),
            probe("llm", self.llm.ping()),
            probe("embedder", self.embedder.ping()),
        )
        deps = dict(results)
        versions = {
            e: (v.version if (v := await self.vectors.active(e)) else None) for e in ("song", "tag")
        }
        return {
            "ok": all(d["ok"] for d in deps.values()),
            "deps": deps,
            "embedding_versions": versions,
            "retrieval": {
                "llm_tags": self.settings.llm_tags_in_retrieval,
                "source_budget_scale": self.settings.source_budget_scale,
                "rerank_variants": self.settings.rerank_variants,
                "rerank_serve": self.settings.rerank_serve,
                "cooccur_weights_feed": self.settings.cooccur_weights_feed,
                "cooccur_weights_chat": self.settings.cooccur_weights_chat,
            },
        }
