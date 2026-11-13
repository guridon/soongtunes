"""Re-ranker data, step 3: candidates for playlist titles, and the gate before any training.

A held-out playlist's title is the query; its songs are the relevant set. Candidates come from
the train-only stores (rerank/stores.py) with no LLM call, so every title can be run:
- bm25: the free-text song query over the `songs_rr_train` alias;
- vector: the title embedded with the chat query instruction, against the train-only song
  vectors (`song-harrier-rr-train-pop5`: songs in >= 5 train playlists);
- tag: the title linked to its 3 nearest tag nodes (whole sentence, tag instruction,
  results/eval/lossmap/sentence_to_tag.py), then songs with those tags in train playlists,
  ranked like the graph source: tags matched, then popularity. Each tag contributes its
  TAG_SONGS most popular songs (the graph source also walks big tags in popularity order).

The gate, measured before training anything:
- the same 10 most popular (train) songs for every query, the floor;
- each source alone, and RRF over them with the chat pipeline's weights;
- candidate recall: the share of a playlist's songs in any source's top POOL;
- the best-order ceiling: the relevant candidates put first. No re-ranker over these
  candidates can beat it.
"""

import asyncio
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import asyncpg
import numpy as np
import orjson

from stunes.config import Settings
from stunes.db import create_pg_pool, create_search
from stunes.embedding.registry import get_version
from stunes.eval.metrics import score_turn
from stunes.llm import Embedder
from stunes.rerank.data import load
from stunes.rerank.stores import SONGS_ALIAS, VECTOR_VERSION
from stunes.retrieval.bm25 import BM25
from stunes.retrieval.fusion import rrf
from stunes.retrieval.vector import IVFFLAT_PROBES, search_sql

logger = logging.getLogger(__name__)

POOL = 100
TAG_VERSION = "tag-harrier-v2"
TAGS_PER_QUERY = 3
TAG_SONGS = 5000
TAG_INSTR = (
    "Instruct: Given a Korean music request, retrieve the playlist tag that names the mood "
    "or situation it asks for\nQuery: "
)
# the chat pipeline's hand-set weights (conversation/pipeline.py); the tag source plays the
# role of the graph source
FUSIONS = {
    "rrf conditioned (tag=graph 1.0, vector 0.5, bm25 0.3)": {
        "tag": 1.0,
        "vector": 0.5,
        "bm25": 0.3,
    },
    "rrf free_text (vector 1.0, bm25 0.6)": {"vector": 1.0, "bm25": 0.6},
    # what the first gate run reported as free_text: rrf() gave the unlisted tag source 1.0
    "rrf all three (tag 1.0, vector 1.0, bm25 0.6)": {"tag": 1.0, "vector": 1.0, "bm25": 0.6},
}
# the chat pipeline's weights when a request has conditions: the comparison fixed in advance
PRESPECIFIED = "rrf conditioned (tag=graph 1.0, vector 0.5, bm25 0.3)"


def query_vectors_path(candidates: Path) -> Path:
    return candidates.with_name(candidates.stem + "-queries.npz")


def sample_queries(playlists: list[dict], split: str, n: int | None, seed: int = 0) -> list[dict]:
    qs = [p for p in playlists if p["split"] == split and (p["title"] or "").strip() and p["songs"]]
    if n is not None and n < len(qs):
        qs = random.Random(seed).sample(qs, n)
    return qs


class TagIndex:
    """Train-playlist tags per song, and each tag's songs by train popularity."""

    def __init__(self, playlists: list[dict]):
        train = [p for p in playlists if p["split"] == "train"]
        # counted once per playlist (the export has no repeated songs; this keeps it that way)
        self.popularity: Counter = Counter(s for p in train for s in set(p["songs"]))
        songs_of: dict[str, set[str]] = defaultdict(set)
        for p in train:
            for t in p["tags"]:
                songs_of[t].update(p["songs"])
        pop = self.popularity
        self.top_songs = {
            t: sorted(ss, key=lambda s: (-pop[s], s))[:TAG_SONGS] for t, ss in songs_of.items()
        }

    def songs(self, tags: list[str], k: int = POOL) -> list[str]:
        hits: Counter = Counter()
        for t in tags:
            hits.update(self.top_songs.get(t, []))
        pop = self.popularity
        return sorted(hits, key=lambda s: (-hits[s], -pop[s], s))[:k]

    def popular(self, k: int = 10) -> list[str]:
        return [s for s, _ in sorted(self.popularity.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


async def _tag_matrix(pg: asyncpg.Pool, allowed: set[str]) -> tuple[list[str], np.ndarray]:
    """Tag vectors, restricted to tags that occur in train playlists: 3,435 of the 29,159 tags
    occur only in val/test, and linking to one would leak that it exists."""
    rows = await pg.fetch(
        "SELECT tag, vec FROM tag_embedding WHERE version = $1 ORDER BY tag", TAG_VERSION
    )
    rows = [r for r in rows if r["tag"] in allowed]
    mat = np.stack([np.asarray(r["vec"].to_numpy(), dtype=np.float32) for r in rows])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return [r["tag"] for r in rows], mat


async def _embed(embedder: Embedder, texts: list[str], prefix: str, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        out.append(await embedder._embed([prefix + t for t in texts[i : i + batch]]))
    q = np.concatenate(out)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


async def generate(settings: Settings, split: str, n: int | None, out: Path) -> Path:
    playlists = load(settings.data_dir / "rerank" / "playlists.jsonl")
    queries = sample_queries(playlists, split, n)
    logger.info("%d %s queries", len(queries), split)
    tags_idx = TagIndex(playlists)
    pg = await create_pg_pool(settings)
    search = create_search(settings)
    bm25 = BM25(search, SONGS_ALIAS, settings.entities_index)
    embedder = Embedder(settings)
    try:
        version = await get_version(pg, VECTOR_VERSION)
        if version is None or version.status == "building":
            raise RuntimeError(f"{VECTOR_VERSION} is not built")
        titles = [q["title"] for q in queries]
        q_song = await _embed(embedder, titles, settings.embed_query_instruction)
        q_tag = await _embed(embedder, titles, TAG_INSTR)
        tag_names, tag_mat = await _tag_matrix(pg, set(tags_idx.top_songs))
        sims = q_tag @ tag_mat.T
        sql = search_sql("song_embedding", "song_id", int(version.dim))
        sem = asyncio.Semaphore(8)

        async def vector(i: int) -> list[tuple[str, float]]:
            async with sem, pg.acquire() as conn, conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {2 * POOL}")
                await conn.execute(f"SET LOCAL ivfflat.probes = {IVFFLAT_PROBES}")
                rows = await conn.fetch(sql, q_song[i], version.version, POOL, 2 * POOL)
                return [(r["song_id"], float(r["sim"])) for r in rows]

        async def lexical(i: int) -> list[tuple[str, float]]:
            async with sem:
                return await bm25.search(titles[i], k=POOL)

        vec = await asyncio.gather(*(vector(i) for i in range(len(queries))))
        lex = await asyncio.gather(*(lexical(i) for i in range(len(queries))))
        out.parent.mkdir(parents=True, exist_ok=True)
        # the query vectors, for the re-ranker's similarity features (rerank/features.py)
        np.savez(
            query_vectors_path(out),
            playlist_id=np.array([q["id"] for q in queries]),
            song=q_song.astype(np.float32),
            tag=q_tag.astype(np.float32),
        )
        tmp = out.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for i, q in enumerate(queries):
                top = np.argsort(-sims[i])[:TAGS_PER_QUERY]
                linked = [(tag_names[j], float(sims[i][j])) for j in top]
                row = {
                    "playlist_id": q["id"],
                    "title": q["title"],
                    "relevant": sorted(set(q["songs"])),
                    "tags_linked": linked,
                    "sources": {
                        "bm25": lex[i],
                        "vector": vec[i],
                        "tag": [[s, None] for s in tags_idx.songs([t for t, _ in linked])],
                    },
                }
                f.write(orjson.dumps(row) + b"\n")
        tmp.replace(out)
        logger.info("wrote candidates for %d queries to %s", len(queries), out)
        return out
    finally:
        await pg.close()
        await search.close()


def gate(settings: Settings, path: Path) -> dict:
    playlists = load(settings.data_dir / "rerank" / "playlists.jsonl")
    popular = TagIndex(playlists).popular(10)
    rows = [orjson.loads(line) for line in path.open("rb")]
    if not rows:
        raise ValueError(f"{path} has no queries")

    def mean(xs):
        return round(sum(xs) / len(xs), 4)

    systems: dict[str, list[list[str]]] = defaultdict(list)
    recall: dict[str, list[float]] = defaultdict(list)
    oracle: list[list[str]] = []
    for r in rows:
        rel = set(r["relevant"])
        ranked = {name: [s for s, _ in hits] for name, hits in r["sources"].items()}
        systems["floor: same popular top 10"].append(popular)
        for name, ids in ranked.items():
            systems[f"{name} alone"].append(ids)
            recall[name].append(len(rel & set(ids)) / len(rel))
        union = list(dict.fromkeys(s for ids in ranked.values() for s in ids))
        recall["union"].append(len(rel & set(union)) / len(rel))
        for label, weights in FUSIONS.items():
            # only the fused sources: rrf() gives a source missing from `weights` weight 1.0
            fused = rrf({k: v for k, v in ranked.items() if k in weights}, weights)
            systems[label].append([f.song_id for f in fused])
        oracle.append([s for s in union if s in rel] + [s for s in union if s not in rel])
    systems["ceiling: union, relevant first"] = oracle

    out: dict = {"queries": len(rows), "pool_per_source": POOL, "systems": {}, "recall_at_pool": {}}
    for name, preds in systems.items():
        ms = [score_turn(p, set(r["relevant"])).as_dict() for p, r in zip(preds, rows, strict=True)]
        out["systems"][name] = {
            k: mean([m[k] for m in ms]) for k in ("hit", "mrr", "ndcg", "precision")
        }
    out["recall_at_pool"] = {k: mean(v) for k, v in recall.items()}
    # paired, per-query bootstrap of the MRR differences that decide the gate
    per = {
        name: np.array(
            [score_turn(p, set(r["relevant"])).mrr for p, r in zip(preds, rows, strict=True)]
        )
        for name, preds in systems.items()
    }
    # the row fixed before seeing results, not the best row on test: selecting on
    # the same queries that give the interval would make it optimistic
    best = PRESPECIFIED
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(rows), (2000, len(rows)))
    out["mrr_differences"] = {}
    for a, b in ((best, "floor: same popular top 10"), ("ceiling: union, relevant first", best)):
        d = per[a] - per[b]
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        out["mrr_differences"][f"{a} - {b}"] = [
            round(float(d.mean()), 4),
            round(float(lo), 4),
            round(float(hi), 4),
        ]
    out["relevant_per_query_median"] = sorted(len(r["relevant"]) for r in rows)[len(rows) // 2]
    return out
