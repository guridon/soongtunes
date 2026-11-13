"""Re-ranker, step 4a: one feature row per (playlist title, candidate song).

The candidates are the union of the three sources' top POOL (rerank/candidates.py). Every feature
comes from the query itself or from the train-only stores (rerank/stores.py: song documents,
song vectors, train-playlist tags), so a held-out playlist never feeds its own features.

Columns, in FEATURES order:
- query: title length in characters and tokens; the similarity of its 3 linked tags; how many
  BM25 hits it had and the top BM25 score;
- sources: rank and score in each source (missing = NaN, which LightGBM routes on its own);
  the query-song cosine for every candidate that has a train-only vector, not only the vector
  source's top POOL; how many sources returned the song; the RRF score with all three sources;
- linked tags, exact: for each of the 3 linked tags, log(1 + the number of train playlists that
  contain the song and carry the tag); the largest share of the song's train playlists that
  carry a linked tag; the sum over linked tags of similarity * log count;
- linked tags, soft: the cosine between the title (tag instruction) and the song's own tags
  (its top 15 train tags), maximum, weighted mean, and against its heaviest tag;
- popularity: log(1 + train playlist count);
- text: the song's title, an artist, its album or a genre written in the title; character
  bigram overlap with the song title, the artist and the song's top 5 train playlist titles;
  the song's year, and whether it falls in a decade the title names ("90년대").
"""

import asyncio
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import orjson

from stunes.config import Settings
from stunes.db import create_pg_pool
from stunes.jobs.index_search import read_jsonl
from stunes.rerank.candidates import TAG_VERSION, query_vectors_path
from stunes.rerank.data import load
from stunes.rerank.stores import VECTOR_VERSION

logger = logging.getLogger(__name__)

# the feature definitions live in rerank/online.py, shared with the chat pipeline
from stunes.rerank.online import (  # noqa: E402,F401  (re-exported for train.py and tests)
    COL,
    FEATURES,
    RRF_ALL,
    RRF_K,
    SOURCES,
    SongText,
    bigrams,
    contains,
    decades,
    feature_block,
    jaccard,
    norm,
    own_tag_count,
)


def _read_candidates(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return [orjson.loads(line) for line in f]


def _train_tag_counts(playlists: list[dict], tags: set[str], songs: set[str]) -> dict[str, Counter]:
    """tag -> song -> number of train playlists that contain the song and carry the tag,
    restricted to the linked tags and the candidate songs (all pairs would be ~20M keys)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for p in playlists:
        if p["split"] != "train":
            continue
        carried = [t for t in p["tags"] if t in tags]
        if not carried:
            continue
        for s in dict.fromkeys(p["songs"]):
            if s in songs:
                for t in carried:
                    out[t][s] += 1
    return out


async def _vectors(settings: Settings, songs: set[str], tags: set[str]):
    pg = await create_pg_pool(settings)
    try:
        rows = await pg.fetch(
            "SELECT song_id, vec FROM song_embedding WHERE version = $1 AND song_id = ANY($2)",
            VECTOR_VERSION,
            sorted(songs),
        )
        song_row = {r["song_id"]: i for i, r in enumerate(rows)}
        smat = np.stack([np.asarray(r["vec"].to_numpy(), dtype=np.float32) for r in rows])
        smat /= np.linalg.norm(smat, axis=1, keepdims=True)
        rows = await pg.fetch(
            "SELECT tag, vec FROM tag_embedding WHERE version = $1 AND tag = ANY($2)",
            TAG_VERSION,
            sorted(tags),
        )
        tag_row = {r["tag"]: i for i, r in enumerate(rows)}
        tmat = np.stack([np.asarray(r["vec"].to_numpy(), dtype=np.float32) for r in rows])
        tmat /= np.linalg.norm(tmat, axis=1, keepdims=True)
    finally:
        await pg.close()
    return song_row, smat, tag_row, tmat


def candidate_ids(row: dict) -> list[str]:
    """The union of the sources, in source order; the row order of the feature matrix."""
    return list(dict.fromkeys(s for name in SOURCES for s, _ in row["sources"][name]))


def features_path(base: Path, split: str, online: bool = False) -> Path:
    return base / f"features-{split}{'-online' if online else ''}.npz"


def build(settings: Settings, split: str, online: bool = False, out: Path | None = None) -> Path:
    """online=False: tag counts over all train playlists (the first model's features).
    online=True: each song's own tag weights (rerank/online.py), what serving can compute."""
    base = settings.data_dir / "rerank"
    cands = base / f"candidates-{split}.jsonl"
    rows = _read_candidates(cands)
    qv = np.load(query_vectors_path(cands))
    if list(qv["playlist_id"]) != [r["playlist_id"] for r in rows]:
        raise ValueError(f"{cands} and its query vectors are out of step; rerun candidates")
    q_song, q_tag = qv["song"], qv["tag"]

    songs = {s for r in rows for s in candidate_ids(r)}
    linked = {t for r in rows for t, _ in r["tags_linked"]}
    docs = {
        d["song_id"]: d
        for d in read_jsonl(base / "export" / "songs.jsonl")
        if d["song_id"] in songs
    }
    missing = songs - docs.keys()
    if missing:
        raise ValueError(f"{len(missing)} candidates have no train-only document")
    playlists = load(base / "playlists.jsonl")
    counts = _train_tag_counts(playlists, linked, songs)
    del playlists
    song_tags = {t for d in docs.values() for t in d["tags"]}
    song_row, smat, tag_row, tmat = asyncio.run(_vectors(settings, songs, song_tags))
    logger.info(
        "%d queries, %d candidate songs (%d with a vector), %d song tags",
        len(rows),
        len(songs),
        len(song_row),
        len(tag_row),
    )
    text = {s: SongText(d) for s, d in docs.items()}

    n_rows = sum(len(candidate_ids(r)) for r in rows)
    X = np.full((n_rows, len(FEATURES)), np.nan, dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.int8)
    ids = np.empty(n_rows, dtype="<U8")
    group = np.zeros(len(rows), dtype=np.int32)
    at = 0
    own = {s: own_tag_count(d) for s, d in docs.items()}

    def exact_count(tag: str, song: str) -> int:
        return counts[tag].get(song, 0)

    def online_count(tag: str, song: str) -> int:
        return own[song](tag)

    tag_count = online_count if online else exact_count
    for qi, r in enumerate(rows):
        cids = candidate_ids(r)
        n = len(cids)
        ids[at : at + n] = cids
        rel = set(r["relevant"])
        y[at : at + n] = [s in rel for s in cids]
        group[qi] = n
        X[at : at + n] = feature_block(
            r["title"],
            r["sources"],
            r["tags_linked"],
            cids,
            docs,
            text,
            q_song[qi],
            song_row,
            smat,
            tag_row,
            tmat @ q_tag[qi],  # the title against every song tag
            tag_count,
        )
        at += n

    out = out or features_path(base, split, online)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(
        tmp,
        X=X,
        y=y,
        group=group,
        song_id=ids,
        playlist_id=np.array([r["playlist_id"] for r in rows]),
        features=np.array(FEATURES),
    )
    tmp.replace(out)
    logger.info(
        "wrote %d rows x %d features for %d queries to %s", n_rows, len(FEATURES), len(rows), out
    )
    return out
