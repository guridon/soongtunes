"""Agentic recommender as a LangGraph state machine.

    understand -> plan -> retrieve -> rank -> verify --(short, attempts < 2)--> relax -> retrieve
                                                    \\--------------------------------> explain

Two entries share the graph: a chat turn (utterance + conversation state) and the home feed
(no utterance: the user's features from the Phase 3 feature store). The LLM appears only in
`understand` (slot extraction) and `explain`; plan, rank, verify and relax are code, with a
fixed relaxation order and a hard cap on loops.
"""

import asyncio
import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from stunes import metrics
from stunes.conversation.pipeline import WEIGHTS, HybridRecommender, TurnResult, _clamp, _Linked
from stunes.features.store import FeatureStore
from stunes.retrieval.bm25 import BM25, Exclusions
from stunes.retrieval.fusion import rrf
from stunes.retrieval.graph import Conditions, GraphRetriever
from stunes.retrieval.ppr import PPR
from stunes.retrieval.ppr import VARIANTS as PPR_VARIANTS
from stunes.retrieval.vector import VectorIndex

logger = logging.getLogger(__name__)

MAX_RELAX = 2
RELAX_ORDER = ("genre", "year", "tag")  # what to give up first when results run short
SOURCE_TIMEOUT_S = {
    "graph": 3.0,
    "cooccur": 3.0,
    "bm25": 2.0,
    "vector": 2.0,
    "popular": 2.0,
    "lookup": 2.0,
    "ppr": 3.0,  # every PPR variant
}
# Feed RRF weights for a user with recent plays, tuned on a dev split by coordinate ascent
# (docs/prereg-feed-weights.md, stunes.bench.feed_weights). Tags, vector and popular went to 0:
# with seeds, co-occurrence alone found more of the hidden songs than any blend. A source at
# 0 is not called. Test split, Recall@50: 0.193 -> 0.259 [+0.040, +0.092].
FEED_WEIGHTS = {"cooccur": 1.0, "tags": 0.0, "vector": 0.0, "popular": 0.0}
# no recent plays (cold start), or too few candidates from them: the weights served before
FEED_WEIGHTS_COLD = {"tags": 0.8, "popular": 0.3}

# ranker: linear blend of signals, all scaled to ~[0, 1]
RANK_W = {
    "fusion": 1.0,
    "tag_pref": 0.6,
    "condition_tags": 0.4,
    "popularity": 0.15,
    "artist_repeat": -0.3,
}
# the feed's ranker for a user with recent plays, tuned with FEED_WEIGHTS. The artist-repeat
# penalty was held at -0.3: freed, the tuning dropped it to 0 for +0.014 Recall@50 and a top
# 20 with 31% of its songs by one artist (docs/results.md). Cold start keeps RANK_W.
FEED_RANK_W = {
    "fusion": 1.0,
    "tag_pref": 1.5,
    "condition_tags": 0.6,
    "popularity": 0.15,
    "artist_repeat": -0.3,
}


def select(
    fused: list,
    cards: dict[str, dict],
    prefs: dict[str, float],
    cond_tags: set[str],
    excluded: set[str],
    n: int,
    rank_w: dict[str, float],
) -> list[dict]:
    """The ranker: signals per fused candidate, then a greedy pick of n with an artist-repeat
    penalty. A pure function, so stunes.bench.feed_weights replays it offline exactly."""
    if not fused:
        return []
    top = fused[0].score
    max_pop = math.log1p(max((c.get("playlist_count") or 0) for c in cards.values()) or 1)
    scored = []
    for f in fused:
        card = cards.get(f.song_id)
        # every source honors exclusions here, whatever its own query did
        if not card or f.song_id in excluded:
            continue
        tags = set(card.get("tags") or [])
        signals = {
            "fusion": f.score / top,
            "tag_pref": sum(prefs.get(t, 0.0) for t in tags),
            "condition_tags": len(tags & cond_tags) / max(1, len(cond_tags)),
            "popularity": math.log1p(card.get("playlist_count") or 0) / max_pop,
        }
        scored.append({"song_id": f.song_id, "card": card, "signals": signals, "ranks": f.ranks})
    # greedy selection with an artist-repeat penalty, so one artist can't fill the list
    chosen, seen_artists = [], {}
    pool = list(scored)
    while pool and len(chosen) < n:

        def value(x):
            rep = sum(seen_artists.get(a, 0) for a in x["card"].get("artists") or [])
            return (
                sum(rank_w[k] * v for k, v in x["signals"].items()) + rank_w["artist_repeat"] * rep
            )

        best = max(pool, key=value)
        best["score"] = round(value(best), 4)
        chosen.append(best)
        pool.remove(best)
        for a in best["card"].get("artists") or []:
            seen_artists[a] = seen_artists.get(a, 0) + 1
    return chosen


class AgentState(TypedDict, total=False):
    entry: str  # chat | feed
    session_id: str
    message: str
    user_id: str | None
    limit: int
    explain: bool
    # feed inputs (taken from the feature store unless the caller passes them, e.g. in eval)
    seed_songs: list[str]
    pref_tags: dict[str, float]
    exclude_playlists: list[str]
    disabled_sources: list[str]
    extra_sources: dict[str, float]  # sources the feed does not call by default (eval)
    # working state
    session: Any
    slots: Any
    raw_slots: dict
    linked: Any
    upsert: list
    remove: list
    prefs: list
    cond: Conditions
    seeds: list[str]
    sources: dict[str, float]
    ranked: dict[str, list[str]]
    timed_out: list[str]
    scored: list[dict]
    rrf_ids: list[str]  # plain RRF order the rank node started from (debug traces)
    final: list[str]
    dropped: dict[str, list[str]]
    relaxed: list[str]
    attempts: int
    reply: str
    timings: dict[str, float]


class AgentRecommender:
    name = "agent"

    def __init__(
        self,
        hybrid: HybridRecommender,
        graph: GraphRetriever,
        bm25: BM25,
        vectors: VectorIndex,
        features: FeatureStore,
    ):
        self.h = hybrid
        self.graph = graph
        self.bm25 = bm25
        self.vectors = vectors
        self.features = features
        # Personalized PageRank over the playlist graph; loaded only by `feed-eval --ppr`
        self.ppr: PPR | None = None
        # the feed's co-occurrence scoring (settings.cooccur_weights_feed; feed-eval switches it)
        self.cooccur_weights = hybrid._s.cooccur_weights_feed
        self.app = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("understand", self._timed("understand", self.understand))
        g.add_node("plan", self._timed("plan", self.plan))
        g.add_node("retrieve", self._timed("retrieve", self.retrieve))
        g.add_node("rank", self._timed("rank", self.rank))
        g.add_node("verify", self._timed("verify", self.verify))
        g.add_node("relax", self._timed("relax", self.relax))
        g.add_node("explain", self._timed("explain", self.explain))
        g.set_entry_point("understand")
        g.add_edge("understand", "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "rank")
        g.add_edge("rank", "verify")
        g.add_conditional_edges("verify", self._enough, {"relax": "relax", "explain": "explain"})
        g.add_edge("relax", "retrieve")
        g.add_edge("explain", END)
        return g.compile()

    @staticmethod
    def _timed(name, fn):
        async def wrapper(state: AgentState) -> dict:
            t = time.perf_counter()
            update = await fn(state)
            timings = dict(state.get("timings") or {})
            timings[name] = round(timings.get(name, 0) + (time.perf_counter() - t) * 1000, 1)
            return {**update, "timings": timings}

        return wrapper

    def _enough(self, state: AgentState) -> str:
        short = len(state["final"]) < state["limit"]
        can_relax = (
            state.get("attempts", 0) < MAX_RELAX and self._next_relaxation(state) is not None
        )
        return "relax" if short and can_relax else "explain"

    # ------------------------------------------------------------------ nodes
    async def understand(self, s: AgentState) -> dict:
        if s["entry"] == "feed":
            seeds = s.get("seed_songs")
            prefs = s.get("pref_tags")
            if seeds is None or prefs is None:
                feats = await self.features.get(
                    [s["user_id"]], ["user_recent_plays", "user_tag_pref"]
                )
                f = feats[s["user_id"]]
                seeds = (
                    seeds
                    if seeds is not None
                    else [song for song, _ in (f["user_recent_plays"] or [])]
                )
                prefs = prefs if prefs is not None else (f["user_tag_pref"] or {})
            top_tags = [t for t, _ in sorted(prefs.items(), key=lambda kv: -kv[1])[:3]]
            # recent plays are what the user already heard: never recommend them back
            cond = Conditions(tags=top_tags, exclude=list(seeds))
            return {
                "seeds": list(seeds)[:5],
                "pref_tags": prefs,
                "cond": cond,
                "attempts": 0,
                "relaxed": [],
                "linked": _Linked(),
                "raw_slots": {},
            }

        session = await self.h._store.load(s["session_id"])
        slots, raw = await self.h._extractor.extract(s["message"], session.history)
        linked, _ = await self.h._link(slots, session, s["message"])
        upsert, remove, prefs = self.h._merge(session, linked)
        cond = self.h._conditions(prefs, slots, session)
        seeds = [x.key for x in linked.songs] or (
            ([p.key for p in prefs if p.kind == "song" and p.positive] or session.last_results[:3])
            if slots.similar_to_previous
            else []
        )
        pref_tags: dict[str, float] = {}
        if s.get("user_id"):
            pref_tags = (await self.features.get([s["user_id"]], ["user_tag_pref"]))[s["user_id"]][
                "user_tag_pref"
            ] or {}
        return {
            "session": session,
            "slots": slots,
            "raw_slots": raw,
            "linked": linked,
            "upsert": upsert,
            "remove": remove,
            "prefs": prefs,
            "cond": cond,
            "seeds": seeds,
            "pref_tags": pref_tags,
            "attempts": 0,
            "relaxed": [],
            "limit": _clamp(slots.limit or s.get("limit"), self.h._s.default_limit),
        }

    async def plan(self, s: AgentState) -> dict:
        """Which candidate sources to call, and how much to trust each."""
        off = set(s.get("disabled_sources") or [])
        if s["entry"] == "feed":
            base = FEED_WEIGHTS if s["seeds"] else FEED_WEIGHTS_COLD
            sources = dict(base) | dict(s.get("extra_sources") or {})
            if not s["seeds"]:
                for name in ("cooccur", "vector", *PPR_VARIANTS):
                    sources.pop(name, None)
            if not s["cond"].tags:
                sources.pop("tags", None)
        elif s["slots"].intent == "find_song" and s["linked"].songs:
            sources = dict(WEIGHTS["find_song"])
        elif s["cond"].has_positive:
            sources = dict(WEIGHTS["conditioned"])
        else:
            sources = dict(WEIGHTS["free_text"])
        if s["entry"] == "chat" and not s["seeds"]:
            sources.pop("cooccur", None)
        return {"sources": {k: v for k, v in sources.items() if k not in off and v > 0}}

    async def retrieve(self, s: AgentState) -> dict:
        pool = self.h._s.candidate_pool
        cond: Conditions = s["cond"]

        async def graph_src():
            rows, _ = await self.graph.by_conditions(cond, k=pool)
            return [sid for sid, _ in rows]

        async def cooccur():
            # up to 5 seeds x 100 playlists each: 0.4 s alone; 300 each took 1.3-1.6 s alone and
            # ran past the 3 s budget once two turns overlapped (troubleshooting #18)
            rows = await self.graph.cooccurring(
                s["seeds"],
                cond,
                k=pool,
                max_playlists=100,
                exclude_playlists=s.get("exclude_playlists"),
                weights=self.cooccur_weights if s["entry"] == "feed" else "count",
                order_by_tags=s["entry"] != "feed",  # feed tags are preferences, not conditions
            )
            return [sid for sid, _ in rows]

        async def bm25():
            ex = Exclusions(
                song_ids=cond.exclude,
                artist_ids=cond.neg_artists,
                genres=cond.neg_genres,
                tags=cond.neg_tags,
            )
            return [
                sid for sid, _ in await self.bm25.search(s["slots"].search_text, k=pool, exclude=ex)
            ]

        async def vector():
            if s["entry"] == "feed":
                q = await self._user_vector(s["seeds"])
                if q is None:
                    return []
            else:
                q = await self.h._embedder.embed_query(s["slots"].search_text)
            banned = set(cond.exclude)
            return [
                sid for sid, _ in await self.vectors.search("song", q, k=pool) if sid not in banned
            ]

        async def popular():
            return await self.graph.popular(cond, k=pool)

        def ppr(variant: str):
            async def walk():
                if self.ppr is None:
                    raise RuntimeError("PPR graph not loaded")
                return await asyncio.to_thread(
                    self.ppr.top,
                    s["seeds"],
                    s.get("exclude_playlists") or [],
                    cond.exclude,
                    pool,
                    **PPR_VARIANTS[variant],
                )

            return walk

        async def lookup():
            ids = []
            for m in s["slots"].songs:
                ids += [sid for sid, _ in await self.bm25.find_song(m.title, m.artist, k=5)]
            return ids

        fns = {
            "graph": graph_src,
            "tags": graph_src,
            "cooccur": cooccur,
            "bm25": bm25,
            "vector": vector,
            "popular": popular,
            "lookup": lookup,
            **{name: ppr(name) for name in PPR_VARIANTS},
        }
        names = [n for n in s["sources"] if n in fns]

        async def bounded(name):
            # a slow source is dropped, not waited for (spec: 늦은 소스는 빼고 진행)
            t = time.perf_counter()
            try:
                budget = SOURCE_TIMEOUT_S.get(
                    "ppr" if name in PPR_VARIANTS else name.replace("tags", "graph"), 3.0
                )
                return await asyncio.wait_for(
                    fns[name](), timeout=budget * self.h._s.source_budget_scale
                )
            finally:  # timed-out calls too, so the histogram shows how far over budget they ran
                metrics.STAGE_LATENCY.labels(f"source_{name}").observe(time.perf_counter() - t)

        results = await asyncio.gather(*(bounded(n) for n in names), return_exceptions=True)
        ranked, timed_out = {}, []
        for n, r in zip(names, results, strict=True):
            if isinstance(r, BaseException):
                timed_out.append(f"{n}:{type(r).__name__}")
                logger.warning("source %s dropped: %r", n, r)
                metrics.ERRORS.labels(f"source_{n}").inc()
                continue
            ranked[n] = r
        sources = dict(s["sources"])
        have = {x for xs in ranked.values() for x in xs} - set(cond.exclude)
        if s["entry"] == "feed" and s["seeds"] and len(have) < s["limit"]:
            # rare seeds: too few co-occurring songs for a full feed, so the cold-start sources
            # fill it (1 of 400 dev playlists had 11)
            off = set(s.get("disabled_sources") or [])
            fill = {
                k: v
                for k, v in FEED_WEIGHTS_COLD.items()
                if k not in sources and k not in off and (k != "tags" or cond.tags)
            }
            more = await asyncio.gather(*(bounded(n) for n in fill), return_exceptions=True)
            for n, r in zip(fill, more, strict=True):
                if isinstance(r, BaseException):
                    timed_out.append(f"{n}:{type(r).__name__}")
                    continue
                ranked[n], sources[n] = r, fill[n]
        return {"ranked": ranked, "timed_out": timed_out, "sources": sources}

    async def rank(self, s: AgentState) -> dict:
        fused = rrf(s["ranked"], s["sources"])[:200]
        if not fused:
            return {"scored": [], "rrf_ids": []}
        pinned = [x.key for x in s["linked"].songs] if s["entry"] == "chat" else []
        cards = await self.bm25.get_songs([f.song_id for f in fused] + pinned)
        chosen = select(
            fused,
            cards,
            s.get("pref_tags") or {},
            set(s["cond"].tags),
            set(s["cond"].exclude),
            s["limit"] * 4,
            FEED_RANK_W if s["entry"] == "feed" and s["seeds"] else RANK_W,
        )
        for sid in reversed(pinned):  # a song the user named always comes first
            if sid in cards:
                chosen = [x for x in chosen if x["song_id"] != sid]
                chosen.insert(
                    0,
                    {
                        "song_id": sid,
                        "card": cards[sid],
                        "signals": {},
                        "ranks": {"lookup": 1},
                        "score": None,
                    },
                )
        return {"scored": chosen, "rrf_ids": [f.song_id for f in fused]}

    async def verify(self, s: AgentState) -> dict:
        ids = [x["song_id"] for x in s["scored"]]
        pinned = {x.key for x in s["linked"].songs} if s["entry"] == "chat" else set()
        bad = await self.graph.violations([i for i in ids if i not in pinned], s["cond"])
        already = set(s.get("final") or [])
        final = list(s.get("final") or [])
        for i in ids:
            if i not in bad and i not in already and len(final) < s["limit"]:
                final.append(i)
        return {"final": final, "dropped": {**(s.get("dropped") or {}), **bad}}

    def _next_relaxation(self, s: AgentState) -> str | None:
        c: Conditions = s["cond"]
        done = set(s.get("relaxed") or [])
        for step in RELAX_ORDER:
            if step in done:
                continue
            if (
                (step == "genre" and c.genres)
                or (step == "year" and (c.year_from or c.year_to))
                or (step == "tag" and len(c.tags) > 1)
            ):
                return step
        return None

    async def relax(self, s: AgentState) -> dict:
        step = self._next_relaxation(s)
        c = Conditions(**s["cond"].params())
        if step == "genre":
            c.genres = []
        elif step == "year":
            c.year_from = c.year_to = None
        elif step == "tag":
            c.tags = c.tags[-1:]  # keep the most recent mood only
        c.exclude = list(c.exclude) + list(s.get("final") or [])
        return {
            "cond": c,
            "relaxed": [*s.get("relaxed", []), step],
            "attempts": s.get("attempts", 0) + 1,
        }

    async def explain(self, s: AgentState) -> dict:
        songs = await self._songs(s)
        if s["entry"] == "feed" or not s.get("explain", True):
            reply = self.h._template_reply(songs, s["linked"], s.get("relaxed") or [])
        else:
            reply = await self.h._reply(
                s["message"], s["slots"], s["linked"], songs, s.get("relaxed") or []
            )
        if s["entry"] == "chat":
            await self.h._store.save_turn(
                s["session_id"],
                upsert=s["upsert"],
                remove=s["remove"],
                user_message=s["message"],
                assistant_message=self.h._summary(songs),
                results=[x["song_id"] for x in songs],
            )
        return {"reply": reply}

    # ------------------------------------------------------------------ helpers
    async def _songs(self, s: AgentState) -> list[dict]:
        by_id = {x["song_id"]: x for x in s["scored"]}
        missing = [i for i in s["final"] if i not in by_id]
        cards = await self.bm25.get_songs(missing) if missing else {}
        out = []
        for i in s["final"]:
            x = by_id.get(i)
            card = x["card"] if x else cards.get(i)
            if card:
                out.append(
                    {
                        **card,
                        "score": x.get("score") if x else None,
                        "sources": x["ranks"] if x else {"graph": 1},
                        "signals": {
                            k: round(v, 3) for k, v in (x or {}).get("signals", {}).items()
                        },
                    }
                )
        return out

    async def _user_vector(self, seeds: list[str]) -> np.ndarray | None:
        """User embedding = mean of recent plays' vectors, newest weighted most."""
        version = await self.vectors.active("song")
        if version is None or not seeds:
            return None
        found = await self.vectors.fetch("song", seeds)
        if not found:
            return None
        order = {sid: i for i, sid in enumerate(seeds)}
        weights = np.array([0.8 ** order[sid] for sid in found], dtype=np.float32)
        mat = np.stack(list(found.values()))
        v = (weights[:, None] * mat).sum(axis=0)
        return v / (np.linalg.norm(v) or 1.0)

    # ------------------------------------------------------------------ entries
    async def turn(
        self,
        session_id: str,
        message: str,
        limit: int | None = None,
        explain: bool = True,
        user_id: str | None = None,
        debug: bool = False,
    ) -> TurnResult:
        t = time.perf_counter()
        out = await self.app.ainvoke(
            {
                "entry": "chat",
                "session_id": session_id,
                "message": message,
                "limit": limit or self.h._s.default_limit,
                "explain": explain,
                "user_id": user_id,
                "timings": {},
            }
        )
        return await self._result(out, session_id, t, debug=debug)

    async def feed(
        self,
        user_id: str,
        limit: int = 20,
        *,
        seed_songs: list[str] | None = None,
        pref_tags: dict[str, float] | None = None,
        exclude_playlists: list[str] | None = None,
        disabled_sources: list[str] | None = None,
        extra_sources: dict[str, float] | None = None,
    ) -> TurnResult:
        t = time.perf_counter()
        out = await self.app.ainvoke(
            {
                "entry": "feed",
                "session_id": f"feed:{user_id}",
                "user_id": user_id,
                "limit": limit,
                "seed_songs": seed_songs,
                "pref_tags": pref_tags,
                "exclude_playlists": exclude_playlists or [],
                "disabled_sources": disabled_sources or [],
                "extra_sources": extra_sources or {},
                "timings": {},
            }
        )
        return await self._result(out, f"feed:{user_id}", t)

    async def _result(
        self, out: dict, session_id: str, t0: float, debug: bool = False
    ) -> TurnResult:
        songs = await self._songs(out)
        prefs = out.get("prefs") or []
        state_view = {
            "positive": [
                {"kind": p.kind, "key": p.key, "name": p.name} for p in prefs if p.positive
            ],
            "negative": [
                {"kind": p.kind, "key": p.key, "name": p.name} for p in prefs if not p.positive
            ],
        }
        trace = {
            "slots": out.get("raw_slots"),
            "linked": out["linked"].as_trace() if out.get("linked") else {},
            # accumulated over the session (after any relaxation): what was searched with
            "conditions": out["cond"].params() if out.get("cond") else None,
            "mode": out["entry"],
            "seeds": out.get("seeds"),
            "sources": out.get("sources"),
            "candidates": {k: len(v) for k, v in (out.get("ranked") or {}).items()},
            "timed_out": out.get("timed_out"),
            "dropped_by_verifier": out.get("dropped"),
            "relaxed": out.get("relaxed"),
            # songs the user named are answered even if other conditions would exclude them
            "pinned": [x.key for x in out["linked"].songs] if out.get("linked") else [],
            "timings_ms": {
                **(out.get("timings") or {}),
                "total": round((time.perf_counter() - t0) * 1000, 1),
            },
        }
        if debug:  # the loss map: candidates per source and the ranked order before verifying
            trace["candidate_ids"] = out.get("ranked") or {}
            # the agent's own order before verifying (RRF + signals + artist penalty, top k*4),
            # and the plain RRF order it started from, to separate what the signals changed
            trace["pre_verify"] = [x["song_id"] for x in out.get("scored") or []]
            trace["rrf"] = out.get("rrf_ids") or []
        return TurnResult(session_id, self.name, out.get("reply", ""), songs, state_view, trace)
