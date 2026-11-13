"""Re-ranker data, step 1: playlists as (query, relevant songs), split, and a leakage audit.

A playlist is a query a person wrote (its title) and the songs they judged to fit it. 115,071
of them are the training signal for a learning-to-rank model over the retrieval candidates.

Splits are assigned by a hash of playlist_id (80/10/10), so every run, and every later feature
rebuild, agrees on which playlists are held out.

The audit counts how much of a held-out playlist the *current* stores already contain. Today
every song feature is built from all playlists, the held-out ones included:
- the search document carries the titles of the song's top 5 playlists by like_cnt
  (jobs/export.py), and the song/v1 embedding text the top 3;
- song tags are the tags of the playlists that contain the song;
- popularity is the playlist count; co-occurrence walks the playlists.
So scoring a held-out title against today's stores would partly look the answer up.
"""

import hashlib
import heapq
import logging
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import orjson
from neo4j import AsyncGraphDatabase, Query

from stunes.config import Settings

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")

PLAYLISTS = """
MATCH (p:Playlist)
RETURN p.playlist_id AS id,
       p.title AS title,
       toInteger(p.like_cnt) AS likes,
       p.updt_date AS updated,
       [(p)-[:HAS_TAG]->(t:Tag) | t.name] AS tags,
       [(p)-[:INCLUDES]->(s:Song) | s.song_id] AS songs
"""


def split_of(playlist_id: str) -> str:
    """80/10/10 by a stable hash (not Python's salted hash())."""
    bucket = int(hashlib.sha1(str(playlist_id).encode()).hexdigest(), 16) % 10
    return "train" if bucket < 8 else ("val" if bucket == 8 else "test")


async def export(settings: Settings, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    seen: set[str] = set()
    tmp = out.with_suffix(".tmp")
    try:
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(Query(PLAYLISTS, timeout=1800))
            with tmp.open("wb") as f:
                async for r in result:
                    row = r.data()
                    # the split is keyed by id: a missing or repeated id would put one playlist
                    # in two splits, or several under "None"
                    if row["id"] is None or row["id"] in seen:
                        raise ValueError(f"playlist id missing or repeated: {row['id']!r}")
                    seen.add(row["id"])
                    row["split"] = split_of(row["id"])
                    f.write(orjson.dumps(row) + b"\n")
    finally:
        await driver.close()
    n = len(seen)
    tmp.replace(out)
    logger.info("exported %d playlists to %s", n, out)
    return n


def load(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return [orjson.loads(line) for line in f]


def top_playlists_per_song(playlists: Iterable[dict], k: int) -> dict[str, list[str]]:
    """Each song's top-k playlists by like_cnt, ties by playlist_id: the order jobs/export.py
    uses to pick the titles that go into the search document and the embedding text."""
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for p in playlists:
        # heap of the k best (likes DESC, id ASC): keep the smallest of (likes, -id) at the top
        key = (p["likes"] or 0, _neg(p["id"]))
        for s in dict.fromkeys(p["songs"]):  # a repeated song must not take two slots
            h = heaps[s]
            if len(h) < k:
                heapq.heappush(h, (key, p["id"]))
            elif key > h[0][0]:
                heapq.heapreplace(h, (key, p["id"]))
    return {s: [pid for _, pid in sorted(h, reverse=True)] for s, h in heaps.items()}


def _neg(playlist_id: str) -> tuple:
    """Sort key that makes a *smaller* playlist_id rank higher inside a max-ordering.

    export.py orders by `p.playlist_id` ascending as a string. Negated code points reverse the
    order character by character. A string that is a prefix of another sorts first ("1" before
    "10"); the trailing length is positive, so it beats any negated code point there.
    """
    return (*(-ord(c) for c in playlist_id), len(playlist_id))


def _median(xs: Iterable[int]) -> float | None:
    xs = list(xs)
    return statistics.median(xs) if xs else None


def audit(playlists: list[dict]) -> dict:
    by_split = defaultdict(list)
    for p in playlists:
        by_split[p["split"]].append(p)
    out: dict = {"playlists": {s: len(by_split[s]) for s in SPLITS}}
    out["songs_per_playlist_median"] = {
        s: _median(len(p["songs"]) for p in by_split[s]) for s in SPLITS
    }
    out["title_chars_median"] = {
        s: _median(len(p["title"] or "") for p in by_split[s]) for s in SPLITS
    }

    held = by_split["test"]
    pairs = sum(len(p["songs"]) for p in held)
    train_songs = {s for p in by_split["train"] for s in p["songs"]}
    cold = sum(1 for p in held for s in p["songs"] if s not in train_songs)

    # empty titles still take a top-5 slot in export.py but are dropped from the document
    top5 = top_playlists_per_song(playlists, 5)
    titled = [p for p in held if p["title"]]
    in_doc = sum(1 for p in titled for s in p["songs"] if p["id"] in top5.get(s, []))
    in_vec = sum(1 for p in titled for s in p["songs"] if p["id"] in top5.get(s, [])[:3])

    # tags: how many of a held-out playlist's own tags reach its songs only through it.
    # Count only the (song, tag) pairs held-out playlists contribute (all pairs: ~20M keys).
    tag_sources: Counter = Counter({(s, t): 0 for p in held for s in p["songs"] for t in p["tags"]})
    for p in playlists:
        for s in p["songs"]:
            for t in p["tags"]:
                if (s, t) in tag_sources:
                    tag_sources[(s, t)] += 1
    own_tag_pairs = sum(len(p["tags"]) * len(p["songs"]) for p in held)
    only_via_self = sum(
        1 for p in held for s in p["songs"] for t in p["tags"] if tag_sources[(s, t)] == 1
    )
    out["test_leakage"] = {
        "song_pairs": pairs,
        "song_in_no_train_playlist": cold,
        "playlist_title_in_search_doc": in_doc,
        "playlist_title_in_embedding_text": in_vec,
        "own_tag_on_song_pairs": own_tag_pairs,
        "own_tag_reaches_song_only_via_this_playlist": only_via_self,
    }
    return out
