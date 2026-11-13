"""On-demand features: computed from L3 on a cache miss.

user_tag_pref is a real join — the user's last 50 plays x song_tag, with a time decay — so
a cache miss costs what a production recompute would, rather than an artificial sleep.
"""

import math
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable

import asyncpg
import msgpack

HALF_LIFE_DAYS = 7.0
TOP_TAGS = 20


async def user_tag_pref(
    pg: asyncpg.Pool, user_ids: list[str], stats: Counter
) -> dict[str, dict[str, float]]:
    stats["l3_sql"] += 1
    rows = await pg.fetch(
        "SELECT pk, value FROM feature WHERE pk = ANY($1)",
        [f"user_recent_plays#{u}" for u in user_ids],
    )
    plays = {r["pk"].split("#", 1)[1]: msgpack.unpackb(r["value"]) for r in rows}
    songs = list({song for ps in plays.values() for song, _ in ps})
    if not songs:
        return {}
    stats["l3_sql"] += 1
    tag_rows = await pg.fetch(
        "SELECT song_id, tag, weight FROM song_tag WHERE song_id = ANY($1)", songs
    )
    tags_by_song: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in tag_rows:
        tags_by_song[r["song_id"]].append((r["tag"], r["weight"]))

    now_ms = time.time() * 1000
    out: dict[str, dict[str, float]] = {}
    for user, ps in plays.items():
        newest = max((ts for _, ts in ps), default=now_ms)
        scores: dict[str, float] = defaultdict(float)
        for song, ts in ps:
            # decay relative to the user's newest play: synthetic timestamps live in the future
            age_days = max(0.0, (newest - ts) / 86_400_000)
            decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
            for tag, weight in tags_by_song.get(song, []):
                scores[tag] += decay * math.log1p(weight)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:TOP_TAGS]
        total = sum(v for _, v in top) or 1.0
        out[user] = {t: round(v / total, 4) for t, v in top}
    return out


async def song_stats(
    pg: asyncpg.Pool, song_ids: list[str], stats: Counter
) -> dict[str, dict[str, int]]:
    stats["l3_sql"] += 1
    rows = await pg.fetch(
        "SELECT song_id, plays, skips, likes FROM song_stats WHERE song_id = ANY($1)", song_ids
    )
    return {
        r["song_id"]: {"plays": r["plays"], "skips": r["skips"], "likes": r["likes"]} for r in rows
    }


COMPUTE: dict[str, Callable[[asyncpg.Pool, list[str], Counter], Awaitable[dict]]] = {
    "user_tag_pref": user_tag_pref,
    "song_stats": song_stats,
}
