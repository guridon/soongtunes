"""The re-ranker in the chat pipeline: online features, the model, and the sentence → tag link.

Per turn (conversation/pipeline.py calls `score`):
- the turn's text, with the tag instruction, against every tag vector of the active tag
  version: its 3 nearest tags (the training's tag source linked titles the same way);
- the tag source: songs carrying those tags, ranked by tags matched and popularity (the graph
  source's own ranking, as TagIndex did for training);
- per-song stats from the feature store group `song_rerank` (L1 / Redis / Postgres), and the
  candidates' vectors of the active song version;
- rerank/online.py's features, with the online tag counts, scored by the served model.

LightGBM is imported on first use: it needs OpenMP, which only the API image and the rerank
image carry (a macOS host without libomp can still import this module and run its tests).

`stunes rerank serve-stats` loads the `song_rerank` group from the song export.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack
import numpy as np
import orjson
import psycopg

from stunes.config import Settings
from stunes.features.store import FeatureStore
from stunes.llm import Embedder
from stunes.rerank.online import DOC_FIELDS, SongText, feature_block, own_tag_count
from stunes.retrieval.graph import Conditions, GraphRetriever
from stunes.retrieval.vector import VectorIndex

logger = logging.getLogger(__name__)

GROUP = "song_rerank"
TAG_GROUP = "tag_songs"  # each tag's songs by popularity, for the tag source
TAG_SONGS = 5000  # songs kept per tag, as the training's TagIndex (candidates.py)
TAGS_PER_QUERY = 3
POOL = 100
# the instruction the training's tag source and the sentence → tag experiment used
TAG_INSTR = (
    "Instruct: Given a Korean music request, retrieve the playlist tag that names the mood "
    "or situation it asks for\nQuery: "
)


@dataclass
class SentenceTags:
    q_tag: np.ndarray
    tags: list[tuple[str, float]]  # nearest first
    ms: float


@dataclass
class Scored:
    scores: dict[str, float]  # candidate -> model score (only candidates with stats)
    popularity: dict[str, int]
    ms: dict[str, float] = field(default_factory=dict)
    sample: dict[str, Any] | None = None


class ChatReranker:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vectors: VectorIndex,
        graph: GraphRetriever,
        features: FeatureStore,
    ):
        self._s = settings
        self._embedder = embedder
        self._vectors = vectors
        self._graph = graph
        self._features = features
        self._lock = asyncio.Lock()
        self._tag_names: list[str] | None = None
        self._tag_row: dict[str, int] = {}
        self._tag_mat: np.ndarray | None = None
        self._booster = None

    # ------------------------------------------------------------ lazy loads
    async def _tags(self) -> tuple[list[str], dict[str, int], np.ndarray]:
        if self._tag_mat is None:
            async with self._lock:
                if self._tag_mat is None:
                    version = await self._vectors.active("tag")
                    if version is None:
                        raise RuntimeError("no active tag version")
                    rows = await self._vectors._pg.fetch(
                        "SELECT tag, vec FROM tag_embedding WHERE version = $1 ORDER BY tag",
                        version.version,
                    )
                    mat = np.stack(
                        [np.asarray(r["vec"].to_numpy(), dtype=np.float32) for r in rows]
                    )
                    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
                    self._tag_names = [r["tag"] for r in rows]
                    self._tag_row = {t: i for i, t in enumerate(self._tag_names)}
                    self._tag_mat = mat
                    logger.info("re-ranker: %d tag vectors of %s", len(rows), version.version)
        return self._tag_names, self._tag_row, self._tag_mat

    async def ready(self) -> None:
        """One-time loads (the tag matrix, the model), kept out of a request's budget."""
        await self._tags()
        self._model()

    def _model(self):
        if self._booster is None:
            import lightgbm as lgb  # needs OpenMP; see the module docstring

            self._booster = lgb.Booster(model_file=str(self._s.rerank_model_path))
        return self._booster

    # ------------------------------------------------------------ per turn
    async def sentence_tags(self, text: str) -> SentenceTags:
        t = time.perf_counter()
        names, _, mat = await self._tags()
        q = (await self._embedder._embed([TAG_INSTR + text]))[0]
        q = q / np.linalg.norm(q)
        sims = mat @ q
        top = np.argsort(-sims)[:TAGS_PER_QUERY]
        return SentenceTags(
            q.astype(np.float32),
            [(names[j], float(sims[j])) for j in top],
            round((time.perf_counter() - t) * 1000, 1),
        )

    async def tag_source(self, tags: list[str], cond: Conditions) -> list[tuple[str, None]]:
        """Songs of playlists carrying the tags, by tags matched, then popularity, then id: the
        training's TagIndex ranking, over all playlists. The lists are precomputed per tag
        (`tag_songs` group). A Neo4j query for this ran 1.6 s to past its 3 s timeout on a
        smoke test, so the source reads the feature store instead. Songs this session already
        showed are left out; the verifier applies the turn's hard conditions afterwards."""
        lists = await self._features.get(tags, [TAG_GROUP])
        hits: dict[str, int] = {}
        pop: dict[str, int] = {}
        for tag in tags:
            for sid, p in lists.get(tag, {}).get(TAG_GROUP) or []:
                hits[sid] = hits.get(sid, 0) + 1
                pop[sid] = p
        banned = set(cond.exclude)
        ranked = sorted((s for s in hits if s not in banned), key=lambda s: (-hits[s], -pop[s], s))
        return [(sid, None) for sid in ranked[:POOL]]

    async def score(
        self,
        text: str,
        sources: dict[str, list[tuple[str, float | None]]],
        st: SentenceTags,
        q_song: np.ndarray | None,
        cids: list[str],
        sample: bool = False,
    ) -> Scored:
        ms: dict[str, float] = {}
        t = time.perf_counter()
        stats, vecs = await asyncio.gather(
            self._features.get(cids, [GROUP]), self._vectors.fetch("song", cids)
        )
        ms["fetch"] = round((time.perf_counter() - t) * 1000, 1)
        docs = {s: v[GROUP] for s, v in stats.items() if v.get(GROUP)}
        have = [s for s in cids if s in docs]
        t = time.perf_counter()
        _, tag_row, tag_mat = await self._tags()
        tag_sims = tag_mat @ st.q_tag
        song_ids = [s for s in have if s in vecs]
        song_row = {s: i for i, s in enumerate(song_ids)}
        smat = None
        if song_ids:
            smat = np.stack([vecs[s] for s in song_ids]).astype(np.float32)
            smat /= np.linalg.norm(smat, axis=1, keepdims=True)
        q = None if q_song is None else (q_song / np.linalg.norm(q_song)).astype(np.float32)
        own = {s: own_tag_count(docs[s]) for s in have}
        X = feature_block(
            text,
            sources,
            st.tags,
            have,
            docs,
            {s: SongText(docs[s]) for s in have},
            q,
            song_row,
            smat,
            tag_row,
            tag_sims,
            lambda tag, song: own[song](tag),
        )
        ms["features"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        pred = self._model().predict(X) if have else np.zeros(0)
        ms["model"] = round((time.perf_counter() - t) * 1000, 1)
        out = Scored(
            {s: float(p) for s, p in zip(have, pred, strict=True)},
            {s: int(docs[s]["playlist_count"]) for s in have},
            ms,
        )
        if sample:  # for the model-parity check: the rows and the scores this request used
            out.sample = {
                "cids": have,
                "X": [[None if np.isnan(v) else float(v) for v in row] for row in X],
                "scores": [float(p) for p in pred],
            }
        return out


# ------------------------------------------------------------------ the feature group
def load_stats(settings: Settings, export: str | None = None) -> int:
    """Write `song_rerank#<song_id>` rows (the per-song fields the features read) from the
    live song export into the feature table, replacing the group."""
    path = export or str(settings.data_dir / "exports" / "songs.jsonl")
    n = 0
    with psycopg.connect(settings.postgres_dsn) as conn:
        # before the COPY: a connection can't run another statement while a COPY is open
        conn.execute(f"DELETE FROM feature WHERE pk LIKE '{GROUP}#%'")
        with (
            conn.cursor().copy("COPY feature (pk, value) FROM STDIN") as copy,
            open(path, "rb") as f,
        ):
            for line in f:
                d = orjson.loads(line)
                value = {k: d.get(k) for k in DOC_FIELDS}
                value["tags"] = value["tags"] or []
                value["tag_weights"] = value["tag_weights"] or []
                value["playlist_count"] = value["playlist_count"] or 0
                copy.write_row((f"{GROUP}#{d['song_id']}", msgpack.packb(value)))
                n += 1
    logger.info("wrote %d %s rows", n, GROUP)
    return n


def load_tag_songs(settings: Settings, playlists: str | None = None) -> int:
    """Write `tag_songs#<tag>` rows: the songs of all playlists carrying the tag, the TAG_SONGS
    most popular (playlist count, then id), as [[song_id, popularity], ...]."""
    from collections import Counter, defaultdict

    path = playlists or str(settings.data_dir / "rerank" / "playlists.jsonl")
    with open(path, "rb") as f:
        rows = [orjson.loads(line) for line in f]
    pop: Counter = Counter(s for p in rows for s in set(p["songs"]))
    songs_of: dict[str, set[str]] = defaultdict(set)
    for p in rows:
        for tag in p["tags"]:
            songs_of[tag].update(p["songs"])
    with psycopg.connect(settings.postgres_dsn) as conn:
        conn.execute(f"DELETE FROM feature WHERE pk LIKE '{TAG_GROUP}#%'")
        with conn.cursor().copy("COPY feature (pk, value) FROM STDIN") as copy:
            for tag, ss in songs_of.items():
                top = sorted(ss, key=lambda s: (-pop[s], s))[:TAG_SONGS]
                copy.write_row((f"{TAG_GROUP}#{tag}", msgpack.packb([[s, pop[s]] for s in top])))
    logger.info("wrote %d %s rows from %d playlists", len(songs_of), TAG_GROUP, len(rows))
    return len(songs_of)
