"""The v2 conversational recommender ("hybrid" strategy).

    extract (LLM slots) -> link (exact/BM25/dense) -> merge into VirtualQuery state
    -> retrieve in parallel {graph conditions, playlist co-occurrence, BM25, dense}
    -> RRF fuse -> verify hard conditions in Cypher -> relax once if short -> reply

Only `extract` and the optional `reply` call the LLM; everything that decides *which*
songs come back is deterministic and inspectable through `trace`.
"""

import asyncio
import logging
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from stunes import metrics
from stunes.config import Settings
from stunes.conversation import signs
from stunes.conversation.delta import PREF_KIND, DeltaExtractor, to_slots
from stunes.conversation.extract import SlotExtractor, Slots
from stunes.conversation.state import Preference, SessionState, StateStore
from stunes.llm import ChatModel, Embedder
from stunes.retrieval import typo
from stunes.retrieval.bm25 import BM25, Exclusions
from stunes.retrieval.fusion import rrf
from stunes.retrieval.graph import Conditions, GraphRetriever
from stunes.retrieval.linker import EntityLinker, Linked
from stunes.retrieval.vector import VectorIndex

if TYPE_CHECKING:
    from stunes.rerank.serve import ChatReranker, SentenceTags

logger = logging.getLogger(__name__)

MAX_TAGS = 3  # positive tags kept in state; older ones fall off
# how hybrid_ctx names each kind of stored preference in the extractor's context
CONTEXT_NAMES = {"artist": "가수", "genre": "장르", "tag": "분위기", "song": "곡"}

# Source weights for RRF. Graph sources encode the user's explicit conditions, so they lead
# when conditions exist; dense/BM25 carry the free-text meaning and fill the rest.
WEIGHTS = {
    "find_song": {"lookup": 3.0, "bm25": 0.5, "cooccur": 0.5, "vector": 0.3},
    "conditioned": {"graph": 1.0, "cooccur": 0.8, "vector": 0.5, "bm25": 0.3},
    "free_text": {"vector": 1.0, "bm25": 0.6, "cooccur": 0.8},
}


@dataclass
class TurnResult:
    session_id: str
    strategy: str
    reply: str
    songs: list[dict[str, Any]]
    state: dict[str, list[dict[str, Any]]]
    trace: dict[str, Any] = field(default_factory=dict)


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def __call__(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round((time.perf_counter() - t) * 1000, 1)

    def total(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)


@dataclass
class _Linked:
    songs: list[Linked] = field(default_factory=list)
    artists: list[Linked] = field(default_factory=list)
    genres: list[Linked] = field(default_factory=list)
    tags: list[Linked] = field(default_factory=list)
    neg_artists: list[Linked] = field(default_factory=list)
    neg_genres: list[Linked] = field(default_factory=list)
    neg_tags: list[Linked] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)

    @property
    def has_condition(self) -> bool:
        return any(
            (self.artists, self.genres, self.tags, self.neg_artists, self.neg_genres, self.neg_tags)
        )

    def as_trace(self) -> dict[str, list[dict[str, Any]]]:
        return {
            name: [
                {"key": x.key, "name": x.name, "via": x.via, "score": round(x.score, 3)}
                for x in items
            ]
            for name, items in vars(self).items()
            if name != "unlinked" and items
        } | ({"unlinked": self.unlinked} if self.unlinked else {})


def _clamp(n: int | None, default: int) -> int:
    return max(1, min(50, n or default))


class HybridRecommender:
    name = "hybrid"

    def __init__(
        self,
        settings: Settings,
        *,
        llm: ChatModel,
        embedder: Embedder,
        extractor: SlotExtractor | DeltaExtractor,
        linker: EntityLinker,
        store: StateStore,
        graph: GraphRetriever,
        bm25: BM25,
        vectors: VectorIndex,
        reranker: "ChatReranker | None" = None,
        name: str = "hybrid",
        s2t: bool = False,
        stateful: bool = True,
        state_mode: str = "merge",
        read_back_keys: bool = False,
        fix_signs: bool = False,
    ):
        self.name = name
        # hybrid_ctx_keys: words typed in English mode are read back as Hangul before the LLM
        # sees the sentence (retrieval/typo.py read_back)
        self._read_back_keys = read_back_keys
        # hybrid_delta*: an operation whose sign the message contradicts is flipped before
        # linking (conversation/signs.py, docs/prereg-sign-rule.md)
        self._fix_signs = fix_signs
        # "merge": this turn's entities are folded into the stored preferences by rules.
        # "replace" (hybrid_ctx): the stored state is put into the extractor's context and its
        # restated slots become the state
        self._state_mode = state_mode
        # strategy hybrid_llmstate (docs/prereg-context-eval.md): a cumulative extractor rebuilds
        # the state from the message history each turn; the stored preferences, last results
        # and shown songs are unused
        self._stateful = stateful
        self._reranker = reranker
        # strategy hybrid_s2t: a mood turn also links the sentence's nearest tag
        self._s2t = s2t and reranker is not None
        self._s = settings
        self._llm = llm
        self._embedder = embedder
        self._extractor = extractor
        self._linker = linker
        self._store = store
        self._graph = graph
        self._bm25 = bm25
        self._vectors = vectors

    # ---------------------------------------------------------------- turn
    async def turn(
        self,
        session_id: str,
        message: str,
        limit: int | None = None,
        explain: bool = True,
        debug: bool = False,
    ) -> TurnResult:
        timer = _Timer()
        typed, keys = message, []
        if self._read_back_keys:
            message, keys = typo.read_back(message)
        state = await self._store.load(session_id)
        if not self._stateful:
            state = SessionState(session_id, turn=state.turn, history=state.history)
        context: dict[str, str] = {}
        if self._state_mode in ("replace", "split", "delta"):
            context = await self._state_context(state)
        with timer("extract"):
            slots, raw = await self._extractor.extract(message, state.history, **context)
        flips: list[list[str]] = []
        if self._fix_signs and isinstance(raw, dict) and "ops" in raw:
            fixed, flips = signs.fix(raw, message)
            if flips:
                prompt_chars = slots.prompt_chars
                slots = to_slots(fixed)
                slots.search_text = slots.search_text or message
                slots.prompt_chars = prompt_chars

        if slots.intent == "reset":
            await self._store.clear(session_id)
            return TurnResult(
                session_id,
                self.name,
                "대화 조건을 초기화했어요. 새로 찾아볼까요?",
                [],
                {"positive": [], "negative": []},
                {"slots": raw, "conditions": None, "timings_ms": timer.ms},
            )

        k = _clamp(slots.limit or limit, self._s.default_limit)
        with timer("link"):
            linked, st = await self._link(slots, state, message)
        if self._state_mode == "delta":
            update, (upsert, remove, prefs), year = await self._apply_delta(state, linked, slots)
            slots.year_from, slots.year_to = year
        else:
            update, (upsert, remove, prefs) = self._update(self._state_mode, state, linked, slots)
        if update == "kept":
            # nothing restated: the stored year range stands too
            slots.year_from, slots.year_to = state.year_from, state.year_to
        cond = self._conditions(prefs, slots, state)
        seeds = [s.key for s in linked.songs] or (
            ([p.key for p in prefs if p.kind == "song" and p.positive] or state.last_results[:3])
            if slots.similar_to_previous
            else []
        )

        with timer("retrieve"):
            ranked, plans, scored = await self._retrieve(slots, cond, seeds, linked)
        mode = (
            "find_song"
            if slots.intent == "find_song" and linked.songs
            else ("conditioned" if cond.has_positive else "free_text")
        )
        fused = rrf(ranked, WEIGHTS[mode])
        pinned = [s.key for s in linked.songs] if mode == "find_song" else []
        candidates = list(dict.fromkeys(pinned + [f.song_id for f in fused]))[: k * 4]

        with timer("verify"):
            bad = await self._graph.violations([c for c in candidates if c not in pinned], cond)
        final = [c for c in candidates if c not in bad][:k]

        relaxed: list[str] = []
        relax_fill: list[str] = []
        if len(final) < k and cond.genres and mode != "find_song":
            # one deterministic relaxation: genre becomes soft (spec: 정해진 순서로 완화)
            with timer("relax"):
                relaxed_cond = Conditions(
                    **{**cond.params(), "genres": [], "exclude": cond.exclude + final}
                )
                extra, _ = await self._graph.by_conditions(relaxed_cond, k=k * 2)
                # by_conditions already filters in Cypher; verifying again keeps "every returned
                # song passed the verifier" true for this path too
                extra_ids = [sid for sid, _ in extra if sid not in final]
                bad_extra = await self._graph.violations(extra_ids, relaxed_cond)
            relax_fill = [sid for sid in extra_ids if sid not in bad_extra]
            final += relax_fill[: k - len(final)]
            relaxed.append("genre")

        variants: dict[str, list[str]] | None = None
        rerank_info: dict[str, Any] | None = None
        if self._reranker is not None and (self._s.rerank_variants or self._s.rerank_serve):
            with timer("rerank"):
                variants, rerank_info = await self._variants(
                    message,
                    mode,
                    ranked,
                    scored,
                    fused,
                    pinned,
                    cond,
                    k,
                    final,
                    bad,
                    relax_fill,
                    st,
                    f"{session_id}:{state.turn}",
                )
            if self._s.rerank_serve and not rerank_info.get("fallback"):
                final = variants["rerank"]

        with timer("hydrate"):
            cards = await self._bm25.get_songs(final)
        by_id = {f.song_id: f for f in fused}
        songs = []
        for sid in final:
            if sid not in cards:
                continue
            f = by_id.get(sid)
            songs.append(
                {
                    **cards[sid],
                    "score": round(f.score, 5) if f else None,
                    "sources": f.ranks if f else {"lookup": 1},
                }
            )

        if explain:
            with timer("reply"):
                reply = await self._reply(message, slots, linked, songs, relaxed)
        else:
            reply = self._template_reply(songs, linked, relaxed)

        with timer("save"):
            await self._store.save_turn(
                session_id,
                upsert=upsert,
                remove=remove,
                user_message=message,
                assistant_message=self._summary(songs),
                results=[s["song_id"] for s in songs],
                year=(cond.year_from, cond.year_to),
            )

        state_view = {
            "positive": [
                {"kind": p.kind, "key": p.key, "name": p.name} for p in prefs if p.positive
            ],
            "negative": [
                {"kind": p.kind, "key": p.key, "name": p.name} for p in prefs if not p.positive
            ],
        }
        trace = {
            "typed": typed if keys else None,
            "keys_read_back": keys,
            "slots": raw,
            "sign_flips": flips,
            "linked": linked.as_trace(),
            # accumulated over the session: what this turn actually searched with
            "conditions": cond.params(),
            "mode": mode,
            "seeds": seeds,
            "plans": plans,
            "candidates": {src: len(ids) for src, ids in ranked.items()},
            "dropped_by_verifier": bad,
            "relaxed": relaxed,
            "timings_ms": timer.ms | {"total": timer.total()},
        }
        if context:
            trace["state_context"] = context
            trace["state_update"] = update
        if slots.prompt_chars:
            trace["prompt_chars"] = slots.prompt_chars
        if variants is not None:
            trace["variants"] = variants
            trace["rerank"] = rerank_info
        if st is not None and self._s2t:
            trace["s2t"] = {"tags": st.tags, "ms": st.ms}
        if debug:  # the loss map: was a label song among the candidates, and where did it rank?
            trace["candidate_ids"] = ranked
            trace["pre_verify"] = candidates  # pinned + RRF order, top k*4
        return TurnResult(session_id, self.name, reply, songs, state_view, trace)

    # ---------------------------------------------------------------- link
    async def _link(
        self, slots: Slots, state: SessionState, message: str
    ) -> "tuple[_Linked, SentenceTags | None]":
        context_tags: list[str] = []
        if slots.similar_to_previous and state.last_results:
            prev = await self._bm25.get_songs(state.last_results[:5])
            context_tags = list(
                dict.fromkeys(t for c in prev.values() for t in c.get("tags") or [])
            )

        async def songs():
            out = await asyncio.gather(*(self._linker.song(m.title, m.artist) for m in slots.songs))
            return [x for x in out if x]

        async def artists(names):
            out = await asyncio.gather(*(self._linker.artist(n) for n in names))
            return [x for x in out if x], [n for n, x in zip(names, out, strict=True) if not x]

        async def tags(moods, ctx):
            out = await asyncio.gather(
                *(self._linker.tags(m, k=2, context_tags=ctx) for m in moods)
            )
            return [x for xs in out for x in xs], [
                m for m, xs in zip(moods, out, strict=True) if not xs
            ]

        async def retype(artist_names, genre_names, mood_names):
            """Values the LLM filed under the wrong type move by exact name only: a genre among
            the artists ("아이돌"), an artist among the genres or moods ("바닐라 어쿠스틱").
            On test2 these were 12 of the 37 values the served arm said but lost (results.md,
            "Values said but lost in linking")."""
            exact = self._linker.artist_exact
            a_hit, g_hit, m_hit = await asyncio.gather(
                asyncio.gather(*(exact(n) for n in artist_names)),
                asyncio.gather(*(exact(n) for n in genre_names)),
                asyncio.gather(*(exact(n) for n in mood_names)),
            )
            to_genre = [
                n for n, x in zip(artist_names, a_hit, strict=True)
                if x is None and self._linker.genre_exact(n)
            ]  # fmt: skip
            from_genres = [n for n, x in zip(genre_names, g_hit, strict=True) if x]
            mood_artists = [n for n, x in zip(mood_names, m_hit, strict=True) if x]
            return to_genre, from_genres, mood_artists

        # The LLM sometimes files a mood under genres ("'POP_MUSIC' 느낌" -> genres). A "genre"
        # that isn't one of the 25 genre names is retried as an artist (exact name), else as a
        # mood/tag instead of dropped.
        miss = [n for n in slots.genres if not self._linker.genre(n)]
        neg_miss = [n for n in slots.exclude_genres if not self._linker.genre(n)]
        # a mood that isn't exactly a tag (after the generic suffix) may be an artist's name
        is_tag, neg_is_tag = await asyncio.gather(
            asyncio.gather(*(self._linker.tag_exact(m) for m in slots.moods)),
            asyncio.gather(*(self._linker.tag_exact(m) for m in slots.exclude_moods)),
        )
        bare = [m for m, x in zip(slots.moods, is_tag, strict=True) if not x]
        neg_bare = [m for m, x in zip(slots.exclude_moods, neg_is_tag, strict=True) if not x]
        (to_g, g_art, m_art), (neg_to_g, neg_g_art, neg_m_art) = await asyncio.gather(
            retype(slots.artists, miss, bare),
            retype(slots.exclude_artists, neg_miss, neg_bare),
        )
        artist_names = [n for n in slots.artists if n not in to_g] + g_art + m_art
        neg_artist_names = (
            [n for n in slots.exclude_artists if n not in neg_to_g] + neg_g_art + neg_m_art
        )
        genres = [g for g in map(self._linker.genre, slots.genres + to_g) if g]
        neg_genres = [g for g in map(self._linker.genre, slots.exclude_genres + neg_to_g) if g]
        moods = [m for m in slots.moods if m not in m_art] + [n for n in miss if n not in g_art]
        neg_moods = [m for m in slots.exclude_moods if m not in neg_m_art] + [
            n for n in neg_miss if n not in neg_g_art
        ]
        (s, (a, ua), (na, una), (t, ut), (nt, unt)) = await asyncio.gather(
            songs(),
            artists(artist_names),
            artists(neg_artist_names),
            tags(moods, context_tags),
            tags(neg_moods, None),
        )
        unlinked_genres: list[str] = []
        st = None
        if self._s2t and moods:
            # hybrid_s2t: the whole sentence's nearest tag joins the phrase-linked tags
            # (results.md, "LLM 없이 문장 전체를 태그에 연결"); a failure leaves them as is
            try:
                st = await self._reranker.sentence_tags(message)
                tag, sim = st.tags[0]
                # not a tag the turn already has, and never one the user ruled out
                if tag not in {x.key for x in t} | {x.key for x in nt}:
                    t.append(Linked("tag", tag, tag, "sentence", sim))
            except Exception as e:
                logger.warning("sentence -> tag failed: %r", e)
                metrics.ERRORS.labels("s2t").inc()
        return _Linked(
            songs=s,
            artists=a,
            genres=genres,
            tags=t,
            neg_artists=na,
            neg_genres=neg_genres,
            neg_tags=nt,
            unlinked=ua + una + ut + unt + unlinked_genres,
        ), st

    # ---------------------------------------------------------------- state
    async def _state_context(self, state: SessionState) -> dict[str, str]:
        """hybrid_ctx: the stored state as the extractor reads it. The conditions the last turn
        searched with, and the last results numbered as shown, with artists and genres, so
        "두 번째 곡 같은 장르" can be answered from the list."""
        lines = []
        for positive in (True, False):
            for kind, label in CONTEXT_NAMES.items():
                names = [p.name for p in state.prefs if p.kind == kind and p.positive is positive]
                if names:
                    lines.append(f"{'' if positive else '제외 '}{label}: {', '.join(names)}")
            if positive and (state.year_from or state.year_to):
                lines.append(f"연도: {state.year_from or ''}~{state.year_to or ''}")
        ids = state.last_results[:5]
        cards = await self._bm25.get_songs(ids) if ids else {}
        results = [
            f"{i}. {c['title']} - {', '.join(c.get('artists') or [])}"
            f" (장르: {', '.join(c.get('genres') or []) or '없음'})"
            for i, sid in enumerate(ids, 1)
            if (c := cards.get(sid))
        ]
        return {"state": "\n".join(lines), "results": "\n".join(results)}

    @staticmethod
    def _turn_prefs(state: SessionState, linked: _Linked) -> list[Preference]:
        new: list[Preference] = []
        for kind, items, positive in (
            ("song", linked.songs, True),
            ("artist", linked.artists, True),
            ("genre", linked.genres, True),
            ("tag", linked.tags, True),
            ("artist", linked.neg_artists, False),
            ("genre", linked.neg_genres, False),
            ("tag", linked.neg_tags, False),
        ):
            new += [Preference(kind, x.key, x.name, positive, state.turn + 1) for x in items]
        return new

    @classmethod
    def _update(
        cls, mode: str, state: SessionState, linked: _Linked, slots: Slots
    ) -> tuple[str, tuple[list[Preference], list[tuple[str, str]], list[Preference]]]:
        """How this turn changes the stored preferences: (how, (upsert, remove, prefs)).

        - merge (hybrid, hybrid_cumstate): rules fold the turn's entities in;
        - replace (hybrid_ctx): the restated slots are the state;
        - split (hybrid_ctxsplit): the LLM owns what changes, the rules what must hold.
          - The restated positives are the state; the stored exclusions stay unless said
            positively, or the LLM marks a new topic.
          - A turn that restates no condition keeps the stored state: "더 없어?" read as
            `other` and a name read as `find_song` wiped it under replace.
        """
        if mode == "replace":
            return "replace", cls._replace(state, linked)
        if mode == "split":
            if slots.new_topic:
                return "new_topic", cls._replace(state, linked)
            if linked.has_condition or slots.year_from or slots.year_to:
                return "replace", cls._replace(state, linked, keep_negatives=True)
            return "kept", cls._merge(state, linked)
        return "merge", cls._merge(state, linked)

    async def _apply_delta(
        self, state: SessionState, linked: _Linked, slots: Slots
    ) -> tuple[
        str, tuple[list[Preference], list[tuple[str, str]], list[Preference]], tuple[Any, Any]
    ]:
        """hybrid_delta*: apply this turn's operations to the stored state; returns (how,
        (upsert, remove, prefs), year range).

        - how: `delta`, `new_topic`, `no_change`, or `failed_empty` (no operation at all, not
          even no_change: recorded as a failure; the stored state stands).
        - cancel and replace find the stored value by name or by linking it again; cancel takes
          either sign, replace only positives. A replace whose old value matches nothing, or has
          none, replaces every positive of its kind. cancel with no value drops the kind's
          positives and keeps its exclusions.
        - A cancel of a value the state doesn't hold is an exclusion: on dev, "X는 빼줘" came
          back as cancel X (6 of 64 exclude turns).
        - added values come from linking: positives from add and replace, exclusions from
          exclude. A value said with the other sign flips. Tags keep the newest MAX_TAGS.
        - An artist or a genre is one value, as in the served merge: a new one replaces the
          kind's positives even when it came as add (27 of 106 replace turns on dev).
        """
        prefs = list(state.prefs)
        year: tuple[Any, Any] = (state.year_from, state.year_to)
        names = {o.op for o in slots.ops}
        if not slots.ops:
            how = "failed_empty"
        elif names == {"no_change"}:
            how = "no_change"
        else:
            how = "new_topic" if "new_topic" in names else "delta"
        if "new_topic" in names:
            prefs, year = [], (None, None)
        absent: list[Preference] = []  # cancels of values the state doesn't hold

        def without_positives(kind: str) -> list[Preference]:
            return [p for p in prefs if not (p.kind == kind and p.positive)]

        for o in slots.ops:
            if o.kind == "year":
                if o.op == "cancel":
                    year = (None, None)
                elif o.op in ("add", "replace"):
                    year = (o.year_from, o.year_to)
                continue
            kind = PREF_KIND.get(o.kind or "")
            if kind is None or o.op not in ("cancel", "replace"):
                continue
            target = o.value if o.op == "cancel" else o.old
            if target is None:
                prefs = without_positives(kind)
                continue
            hit = await self._matching(kind, target, prefs)
            held = {(p.kind, p.key) for p in prefs} & hit
            if o.op == "cancel" and not held:
                absent += [
                    Preference(kind, x.key, x.name, False, state.turn + 1)
                    for x in await self._link_one(kind, target)
                ]
            elif o.op == "cancel":
                prefs = [p for p in prefs if (p.kind, p.key) not in held]
            elif any(p.positive and (p.kind, p.key) in hit for p in prefs):
                prefs = [p for p in prefs if not (p.positive and (p.kind, p.key) in hit)]
            else:
                prefs = without_positives(kind)

        new = self._turn_prefs(state, linked) + absent
        if linked.songs:
            prefs = [p for p in prefs if p.kind != "song"]
        single = {p.kind for p in new if p.positive and p.kind in ("artist", "genre")}
        prefs = [p for p in prefs if not (p.positive and p.kind in single)]
        keys = {(p.kind, p.key) for p in new}
        prefs = [p for p in prefs if (p.kind, p.key) not in keys] + new
        pos_tags = [p for p in prefs if p.kind == "tag" and p.positive]
        prefs = [p for p in prefs if p not in pos_tags[:-MAX_TAGS]]
        final = {(p.kind, p.key) for p in prefs}
        remove = [(p.kind, p.key) for p in state.prefs if (p.kind, p.key) not in final]
        return how, (prefs, remove, prefs), year

    async def _matching(
        self, kind: str, value: str, prefs: list[Preference]
    ) -> set[tuple[str, str]]:
        """The stored preferences of `kind` a cancel or replace names: by name or key, and by
        linking the value again (a mood links to up to 2 tags)."""

        def norm(x: str | None) -> str:
            return "".join((x or "").split()).lower()

        hit = {
            (p.kind, p.key)
            for p in prefs
            if p.kind == kind and norm(value) in (norm(p.name), norm(p.key))
        }
        return hit | {(kind, x.key) for x in await self._link_one(kind, value)}

    async def _link_one(self, kind: str, value: str) -> list[Linked]:
        if kind == "artist":
            return [x for x in [await self._linker.artist(value)] if x]
        if kind == "genre":
            return [x for x in [self._linker.genre(value)] if x]
        return await self._linker.tags(value, k=2)

    @classmethod
    def _replace(
        cls, state: SessionState, linked: _Linked, keep_negatives: bool = False
    ) -> tuple[list[Preference], list[tuple[str, str]], list[Preference]]:
        """hybrid_ctx: the extractor read the stored state and restated it, so this turn's
        entities are the whole state. What it left out is dropped; a value said both ways is
        kept as positive; tags keep the newest MAX_TAGS. With `keep_negatives` (split), a
        stored exclusion the turn didn't say positively is kept."""
        by_key: dict[tuple[str, str], Preference] = {}
        for p in cls._turn_prefs(state, linked):
            if (p.kind, p.key) not in by_key or p.positive:
                by_key[(p.kind, p.key)] = p
        if keep_negatives:
            for p in state.prefs:
                if not p.positive:
                    by_key.setdefault((p.kind, p.key), p)
        prefs = list(by_key.values())
        pos_tags = [p for p in prefs if p.kind == "tag" and p.positive]
        prefs = [p for p in prefs if p not in pos_tags[:-MAX_TAGS]]
        keys = {(p.kind, p.key) for p in prefs}
        remove = [(p.kind, p.key) for p in state.prefs if (p.kind, p.key) not in keys]
        return prefs, remove, prefs

    @classmethod
    def _merge(
        cls, state: SessionState, linked: _Linked
    ) -> tuple[list[Preference], list[tuple[str, str]], list[Preference]]:
        """Fold this turn's entities into the accumulated preferences.

        - A new positive of a kind replaces earlier positives of that kind ("발라드" then
          "댄스는?" means dance now, not both); tags keep the newest MAX_TAGS.
        - Negatives accumulate, and saying something positively lifts its negative.
        """
        new = cls._turn_prefs(state, linked)

        replaced_kinds = {p.kind for p in new if p.positive}
        new_keys = {(p.kind, p.key) for p in new}
        kept = [
            p
            for p in state.prefs
            if (p.kind, p.key) not in new_keys and not (p.positive and p.kind in replaced_kinds)
        ]
        prefs = kept + new
        pos_tags = [p for p in prefs if p.kind == "tag" and p.positive]
        overflow = (
            {(p.kind, p.key) for p in pos_tags[:-MAX_TAGS]} if len(pos_tags) > MAX_TAGS else set()
        )
        prefs = [p for p in prefs if (p.kind, p.key) not in overflow]

        final_keys = {(p.kind, p.key) for p in prefs}
        remove = [(p.kind, p.key) for p in state.prefs if (p.kind, p.key) not in final_keys]
        upsert = [p for p in new if (p.kind, p.key) in final_keys]
        # an entity flipping sign is an upsert of the same (kind, key), not a remove
        return upsert, remove, prefs

    @staticmethod
    def _conditions(prefs: list[Preference], slots: Slots, state: SessionState) -> Conditions:
        def keys(kind, positive):
            return [p.key for p in prefs if p.kind == kind and p.positive is positive]

        return Conditions(
            tags=keys("tag", True),
            genres=keys("genre", True),
            artists=keys("artist", True),
            neg_tags=keys("tag", False),
            neg_genres=keys("genre", False),
            neg_artists=keys("artist", False),
            # "더 추천해줘" should not repeat what this session already showed
            exclude=state.shown if slots.intent != "find_song" else [],
            year_from=slots.year_from,
            year_to=slots.year_to,
        )

    # ---------------------------------------------------------------- retrieve
    async def _retrieve(
        self, slots: Slots, cond: Conditions, seeds: list[str], linked: _Linked
    ) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, Any]]:
        pool = self._s.candidate_pool
        exclude = Exclusions(
            song_ids=cond.exclude,
            artist_ids=cond.neg_artists,
            genres=cond.neg_genres,
            tags=cond.neg_tags,
        )
        plans: dict[str, Any] = {}
        scored: dict[str, Any] = {}  # bm25/vector hits with scores, and the query vector

        async def graph():
            rows, plan = await self._graph.by_conditions(cond, k=pool)
            plans["graph"] = plan
            return [sid for sid, _ in rows]

        async def cooccur():
            weights = self._s.cooccur_weights_chat
            if weights == "auto":  # the rule GraphRetriever applies, resolved here for the trace
                weights = "degree" if len(seeds) >= 2 else "count"
            plans["cooccur"] = f"{weights}, {len(seeds)} seeds"
            rows = await self._graph.cooccurring(seeds, cond, k=pool, weights=weights)
            return [sid for sid, _ in rows]

        async def bm25():
            hits = await self._bm25.search(slots.search_text, k=pool, exclude=exclude)
            scored["bm25"] = hits
            return [sid for sid, _ in hits]

        async def vector():
            q = await self._embedder.embed_query(slots.search_text)
            scored["q_song"] = q
            hits = await self._vectors.search("song", q, k=pool)
            banned = set(cond.exclude)
            scored["vector"] = [(sid, s) for sid, s in hits if sid not in banned]
            return [sid for sid, _ in scored["vector"]]

        async def lookup():
            ids = []
            for m in slots.songs:
                ids += [sid for sid, _ in await self._bm25.find_song(m.title, m.artist, k=5)]
            return ids

        tasks: dict[str, Any] = {"bm25": bm25(), "vector": vector()}
        if cond.has_positive:
            tasks["graph"] = graph()
        if seeds:
            tasks["cooccur"] = cooccur()
        if slots.songs:
            tasks["lookup"] = lookup()

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        ranked: dict[str, list[str]] = {}
        for name, res in zip(tasks, results, strict=True):
            if isinstance(res, BaseException):
                # one slow/broken source must not fail the turn; it is visible in the trace
                logger.warning("source %s failed: %r", name, res)
                metrics.ERRORS.labels(f"source_{name}").inc()
                plans[f"{name}_error"] = repr(res)[:200]
                continue
            ranked[name] = res
        return ranked, plans, scored

    # ---------------------------------------------------------------- variants
    async def _variants(
        self,
        message: str,
        mode: str,
        ranked: dict[str, list[str]],
        scored: dict[str, Any],
        fused: list,
        pinned: list[str],
        cond: Conditions,
        k: int,
        served: list[str],
        bad: dict[str, list[str]],
        relax_fill: list[str],
        st: "SentenceTags | None",
        turn_key: str,
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        """Other orders of this turn's candidates (docs/prereg-rerank-chat.md, E1).

        `rrf` is what is served. `rrf_explicit` drops a source the mode's weight table doesn't
        list (rrf() would give it weight 1.0). `rerank` and `popularity` order the union of
        the chat candidates and the sentence -> tag source by the model and by popularity;
        if that step fails or overruns its budget, both are `rrf` and `fallback` says why.
        Every order is verified and, if short, filled from the served list's relaxation extras.
        """
        info: dict[str, Any] = {"fallback": None}
        weights = WEIGHTS[mode]
        explicit = rrf({s: ids for s, ids in ranked.items() if s in weights}, weights)
        orders: dict[str, list[str] | None] = {
            "rrf_explicit": list(dict.fromkeys(pinned + [f.song_id for f in explicit])),
            "rerank": None,
            "popularity": None,
        }
        sample = zlib.crc32(turn_key.encode()) % self._s.rerank_sample_every == 0
        stages: dict[str, float] = {}  # how far the step got, also when it overran
        t0 = time.perf_counter()

        def mark(name: str) -> None:
            stages[name] = round((time.perf_counter() - t0) * 1000, 1)

        async def core():
            s2 = st or await self._reranker.sentence_tags(message)
            mark("sentence")
            tag_src = await self._reranker.tag_source([t for t, _ in s2.tags], cond)
            mark("tag_source")
            rest = list(dict.fromkeys([f.song_id for f in fused] + [sid for sid, _ in tag_src]))
            rest = [c for c in rest if c not in pinned]
            sources = {
                "bm25": scored.get("bm25") or [],
                "vector": scored.get("vector") or [],
                "tag": tag_src,
            }
            sc = await self._reranker.score(
                message, sources, s2, scored.get("q_song"), rest, sample=sample
            )
            mark("scored")
            return s2, tag_src, rest, sc

        tag_src: list[tuple[str, None]] = []
        try:
            # the tag matrix and the model load once per worker: not part of the budget
            await self._reranker.ready()
            t0 = time.perf_counter()
            s2, tag_src, rest, sc = await asyncio.wait_for(core(), self._s.rerank_budget_ms / 1000)
            at = {c: i for i, c in enumerate(rest)}  # ties: the served order, then tag source
            orders["rerank"] = (
                pinned
                + sorted([c for c in rest if c in sc.scores], key=lambda c: (-sc.scores[c], at[c]))
                + [c for c in rest if c not in sc.scores]
            )
            orders["popularity"] = pinned + sorted(
                rest, key=lambda c: (-sc.popularity.get(c, -1), c)
            )
            info |= {
                "s2t_tags": s2.tags,
                "tag_source": len(tag_src),
                "union": len(rest),
                "scored": len(sc.scores),
                "ms": sc.ms | {"sentence": s2.ms if st is None else 0.0},
            }
            if sc.sample:
                info["sample"] = sc.sample
        except TimeoutError:
            info["fallback"] = "budget"
            info["stages"] = stages
        except Exception as e:  # the variants must never fail the turn
            logger.warning("re-rank step failed: %r", e)
            metrics.ERRORS.labels("rerank").inc()
            info["fallback"] = repr(e)[:200]
        if info["fallback"]:
            metrics.ERRORS.labels("rerank_fallback").inc()
        info["step_ms"] = round((time.perf_counter() - t0) * 1000, 1)  # budgeted part only

        # the served list's relaxation extras passed only the relaxed conditions (no genre):
        # in a variant's top they are checked against the turn's own conditions
        fill = set(relax_fill)
        checked = {c for c in served if c not in fill} | set(bad)

        # rrf_explicit is cut at k*4 like the served list. rerank and popularity order a union
        # that also holds sources blind to the hard conditions (tag, bm25, vector), so a k*4
        # cut could leave them with few or no songs on a genre turn; they are verified down
        # the whole order instead, and the hard conditions act as a filter, not as a cut.
        def head(name: str, o: list[str]) -> list[str]:
            return o if name in ("rerank", "popularity") else o[: k * 4]

        todo = list(
            dict.fromkeys(
                c
                for name, o in orders.items()
                if o
                for c in head(name, o)
                if c not in pinned and c not in checked
            )
        )
        t1 = time.perf_counter()
        try:
            bad_all = set(bad) | set(await self._graph.violations(todo, cond))
            info["verify"] = {"ids": len(todo), "ms": round((time.perf_counter() - t1) * 1000, 1)}
        except Exception as e:  # the variants must never fail the turn: every order = served
            logger.warning("variant verification failed: %r", e)
            metrics.ERRORS.labels("rerank").inc()
            info["fallback"] = info["fallback"] or f"verify: {e!r}"[:200]
            info["verify_failed"] = True
            return {"rrf": served} | {name: served for name in orders}, info
        out = {"rrf": served}
        for name, o in orders.items():
            if o is None:
                out[name] = served
                continue
            lst = [c for c in head(name, o) if c not in bad_all][:k]
            if len(lst) < k:
                lst += [c for c in relax_fill if c not in lst][: k - len(lst)]
            out[name] = lst
        if orders["rerank"] is not None:
            # how many of rerank's songs no trained source (bm25, vector, tag) found: their
            # source features are all missing, a row shape the model never saw in training
            trained = {sid for name in ("bm25", "vector") for sid, _ in scored.get(name) or []}
            trained |= {sid for sid, _ in tag_src}
            info["rerank_untrained"] = sum(
                1 for c in out["rerank"] if c not in pinned and c not in trained
            )
        return out, info

    # ---------------------------------------------------------------- reply
    @staticmethod
    def _summary(songs: list[dict]) -> str:
        if not songs:
            return "조건에 맞는 곡을 찾지 못했다."
        titles = ", ".join(f"{s['title']} - {', '.join(s.get('artists') or [])}" for s in songs[:3])
        return f"{len(songs)}곡을 추천했다: {titles}"

    @staticmethod
    def _template_reply(songs: list[dict], linked: _Linked, relaxed: list[str]) -> str:
        if not songs:
            return "조건에 맞는 곡을 찾지 못했어요. 조건을 조금 바꿔볼까요?"
        conds = [x.name for x in linked.artists + linked.genres + linked.tags]
        head = f"{' · '.join(conds)} 조건으로 " if conds else ""
        names = {"genre": "장르", "year": "연도", "tag": "분위기 태그 일부"}
        # say exactly what was given up (any relaxation used to read as "genre")
        tail = (
            f" ({', '.join(names.get(r, r) for r in relaxed)} 조건은 완화했어요)" if relaxed else ""
        )
        return f"{head}{len(songs)}곡을 골랐어요.{tail}"

    async def _reply(
        self, message: str, slots: Slots, linked: _Linked, songs: list[dict], relaxed: list[str]
    ) -> str:
        if not songs:
            return self._template_reply(songs, linked, relaxed)
        lines = [f"- {s['title']} / {', '.join(s.get('artists') or [])}" for s in songs[:5]]
        conds = {
            "아티스트": [x.name for x in linked.artists],
            "장르": [x.name for x in linked.genres],
            "태그": [x.name for x in linked.tags],
            "제외": [x.name for x in linked.neg_artists + linked.neg_genres + linked.neg_tags],
        }
        prompt = (
            f"사용자 요청: {message}\n적용한 조건: { {k: v for k, v in conds.items() if v} }\n"
            f"완화한 조건: {relaxed or '없음'}\n추천 결과 상위:\n" + "\n".join(lines)
        )
        try:
            return await self._llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "너는 음악 추천 서비스의 안내 문구를 쓴다. 한국어로 1~2문장, 존댓말. "
                            "적용한 조건과 대표 곡 1~2개를 자연스럽게 언급하고, 완화한 조건이 있으면 알린다. "
                            "목록을 다시 나열하지 말고, 없는 사실을 지어내지 않는다."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            ) or self._template_reply(songs, linked, relaxed)
        except Exception as e:  # the reply is cosmetic; never fail the turn on it
            logger.warning("reply generation failed: %r", e)
            metrics.ERRORS.labels("llm_reply").inc()
            return self._template_reply(songs, linked, relaxed)
