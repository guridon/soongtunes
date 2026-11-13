"""Phase 3 data setup: song_tag table, synthetic user features, bulk L2 warm-up."""

import logging
import random
import time
from pathlib import Path

import msgpack
import orjson
import psycopg

from stunes.config import Settings

logger = logging.getLogger(__name__)


def load_song_tags(settings: Settings, export_dir: Path) -> int:
    """song_tag(song_id, tag, weight) from the graph export (top tags per song)."""
    sql = (Path(__file__).resolve().parents[4] / "infra/postgres/init/003_song_tag.sql").read_text()
    n = 0
    started = time.monotonic()
    with psycopg.connect(settings.postgres_dsn) as conn:
        conn.execute(sql)
        conn.execute("TRUNCATE song_tag")
        with conn.cursor().copy("COPY song_tag (song_id, tag, weight) FROM STDIN") as copy:
            with (export_dir / "songs.jsonl").open("rb") as f:
                for line in f:
                    doc = orjson.loads(line)
                    for tag, weight in zip(doc["tags"], doc["tag_weights"], strict=True):
                        copy.write_row((doc["song_id"], tag, weight))
                        n += 1
        conn.execute("ANALYZE song_tag")
    logger.info("song_tag: %d rows in %.0fs", n, time.monotonic() - started)
    return n


def seed_users(settings: Settings, n_users: int, seed: int = 11) -> int:
    """user_recent_plays for n synthetic users (one per playlist), written straight to L3.

    The consumer path (Phase 2) produces the same rows from events; seeding directly gives
    the cache experiments a 100k-key population without replaying millions of events.
    """
    from neo4j import GraphDatabase, Query

    rng = random.Random(seed)
    with GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    ) as d:
        # the server default (30 s) protects the API; this one-off read needs longer
        with d.session(database=settings.neo4j_database) as session:
            result = session.run(
                Query(
                    "MATCH (p:Playlist) WITH p ORDER BY p.playlist_id LIMIT $n RETURN p.playlist_id AS id, "
                    "[(p)-[:INCLUDES]->(s:Song) | s.song_id][..50] AS songs",
                    timeout=900,
                ),
                n=n_users,
            )
            records = list(result)
    now_ms = int(time.time() * 1000)
    with psycopg.connect(settings.postgres_dsn) as conn:
        # before the COPY: a connection can't run another statement while a COPY is open
        # (doing this inside the block hung the job — and held a transaction that blocked a
        # concurrent CREATE INDEX CONCURRENTLY)
        conn.execute("DELETE FROM feature WHERE pk LIKE 'user_recent_plays#%'")
        with conn.cursor().copy("COPY feature (pk, value) FROM STDIN") as copy:
            for r in records:
                ts = now_ms - rng.randint(0, 30 * 86_400_000)
                plays = []
                for song in r["songs"]:
                    plays.append([song, ts])
                    ts -= rng.randint(120_000, 3_600_000)
                copy.write_row((f"user_recent_plays#p_{r['id']}", msgpack.packb(plays)))
    logger.info("seeded user_recent_plays for %d users", len(records))
    return len(records)
