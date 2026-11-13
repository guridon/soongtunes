"""Is Redis serving what Postgres holds? And a writer that is not the ingest consumer.

    diff     compare every user_recent_plays row in Postgres with its Redis copy:
             stale   = Redis value differs from the row
             missing = row with no Redis key (a read falls through to Postgres: not wrong)
             orphan  = Redis value for a row Postgres no longer has (serves deleted data)
    correct  change the feature table directly in SQL, the way an operator fixes data:
             drop-song     remove the most frequent song from every recent-plays list
                           (e.g. plays later judged to be bot traffic)
             delete-users  delete N users' recent plays (e.g. a deletion request)

Run: python -m stunes.ingest.cache_check diff | correct --kind drop-song|delete-users [--n 500]
"""

import argparse
import json
from collections import Counter

import msgpack
import psycopg
import redis

from stunes.config import get_settings

PREFIX = "user_recent_plays#"


def diff() -> dict:
    s = get_settings()
    cache = redis.Redis.from_url(s.redis_url)
    with psycopg.connect(s.postgres_dsn) as conn:
        rows = {
            pk: msgpack.unpackb(v)
            for pk, v in conn.execute(
                "SELECT pk, value FROM feature WHERE pk LIKE %s", (PREFIX + "%",)
            )
        }
    out = {"rows": len(rows), "same": 0, "stale": 0, "missing": 0, "orphan": 0}
    for pk, truth in rows.items():
        blob = cache.get(pk)
        if blob is None:
            out["missing"] += 1
        else:
            out["same" if msgpack.unpackb(blob)[0] == truth else "stale"] += 1
    for key in cache.scan_iter(match=PREFIX + "*", count=1000):
        pk = key.decode()
        if (
            pk not in rows
            and not msgpack.unpackb(cache.get(pk) or msgpack.packb([None, 0, 0, True]))[3]
        ):
            out["orphan"] += 1
    return out


def correct(kind: str, n: int) -> dict:
    s = get_settings()
    with psycopg.connect(s.postgres_dsn) as conn, conn.transaction():
        rows = conn.execute(
            "SELECT pk, value FROM feature WHERE pk LIKE %s", (PREFIX + "%",)
        ).fetchall()
        if kind == "delete-users":
            pks = sorted(pk for pk, _ in rows)[:n]
            conn.execute("DELETE FROM feature WHERE pk = ANY(%s)", (pks,))
            return {"kind": kind, "rows": len(pks)}
        plays = {pk: msgpack.unpackb(v) for pk, v in rows}
        song = Counter(p[0] for lst in plays.values() for p in lst).most_common(1)[0][0]
        changed = 0
        for pk, lst in plays.items():
            kept = [p for p in lst if p[0] != song]
            if len(kept) != len(lst):
                conn.execute(
                    "UPDATE feature SET value = %s, updated_at = now() WHERE pk = %s",
                    (msgpack.packb(kept), pk),
                )
                changed += 1
        return {"kind": kind, "song_id": song, "rows": changed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["diff", "correct"])
    ap.add_argument("--kind", choices=["drop-song", "delete-users"], default="drop-song")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    print(json.dumps(diff() if a.action == "diff" else correct(a.kind, a.n)))


if __name__ == "__main__":
    main()
