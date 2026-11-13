"""Applies one event to the feature store — the unit of work every consumer mode shares.

One Postgres transaction per event:
    1. record the delivery attempt (experiment bookkeeping)
    2. INSERT the event id into processed_event; a conflict means "already applied"
    3. only if new: update song_stats and user_recent_plays
Because 2 and 3 commit together, a crash leaves either nothing or everything applied, and a
redelivery is a no-op. Redis is written after commit (write-through, best effort), unless
write_through is off: then stunes.ingest.cdc_cache applies the commit from the WAL instead.
"""

import time

import msgpack
import psycopg
import redis

from stunes import metrics
from stunes.features.store import envelope, load_groups
from stunes.ingest.events import Event

RECENT_N = 50
# write-through TTL follows the feature spec, not a constant
CACHE_TTL_S = load_groups()["user_recent_plays"].ttl_sec
_COUNTER = {"play": "plays", "skip": "skips", "like": "likes"}


class Processor:
    def __init__(self, dsn: str, redis_url: str, run_id: str, write_through: bool = True):
        self.conn = psycopg.connect(dsn, autocommit=False)
        self.cache = redis.Redis.from_url(redis_url)
        self.run_id = run_id
        self.write_through = write_through

    def close(self) -> None:
        self.conn.close()
        self.cache.close()

    def handle(self, raw: bytes, partition: int, offset: int) -> str:
        """Returns 'ok' or 'duplicate'. Raises pydantic.ValidationError for bad payloads."""
        ev = Event.model_validate_json(raw)
        with self.conn.transaction():
            new = self.conn.execute(
                "INSERT INTO processed_event (event_id) VALUES (%s) ON CONFLICT DO NOTHING RETURNING 1",
                (ev.event_id,),
            ).fetchone()
            self.conn.execute(
                'INSERT INTO ingest_attempt (run_id, event_id, part, "offset", event_ts, duplicate) '
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (self.run_id, ev.event_id, partition, offset, ev.ts, new is None),
            )
            recent = self._apply(ev) if new else None
        if new and self.write_through:
            self._write_through(ev, recent)
            # ingestion lag: how long after the event happened the feature reflects it
            metrics.FRESHNESS.observe(max(0.0, time.time() * 1000 - ev.ts) / 1000)
        metrics.INGEST_EVENTS.labels("ok" if new else "duplicate").inc()
        return "ok" if new else "duplicate"

    def _apply(self, ev: Event) -> list | None:
        col = _COUNTER.get(ev.type)
        if col:
            # col comes from the fixed mapping above, never from the payload
            self.conn.execute(
                f"INSERT INTO song_stats (song_id, {col}) VALUES (%s, 1) "
                f"ON CONFLICT (song_id) DO UPDATE SET {col} = song_stats.{col} + 1, updated_at = now()",
                (ev.song_id,),
            )
        if ev.type != "play":
            return None
        pk = f"user_recent_plays#{ev.user_id}"
        row = self.conn.execute(
            "SELECT value FROM feature WHERE pk = %s FOR UPDATE", (pk,)
        ).fetchone()
        plays = msgpack.unpackb(row[0]) if row else []
        # parallel consumers can apply a user's events out of order: keep the list sorted by
        # event time instead of by arrival
        plays.append([ev.song_id, ev.ts])
        plays.sort(key=lambda p: p[1], reverse=True)
        plays = plays[:RECENT_N]
        self.conn.execute(
            "INSERT INTO feature (pk, value) VALUES (%s, %s) "
            "ON CONFLICT (pk) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (pk, msgpack.packb(plays)),
        )
        return plays

    def _write_through(self, ev: Event, recent: list | None) -> None:
        try:
            pipe = self.cache.pipeline(transaction=False)
            if recent is not None:
                # the envelope FeatureStore reads; this also replaces a cached "no such user"
                pipe.set(
                    f"user_recent_plays#{ev.user_id}", envelope(recent, CACHE_TTL_S), ex=CACHE_TTL_S
                )
                # derived from recent plays: drop it so the next read recomputes
                pipe.delete(f"user_tag_pref#{ev.user_id}")
            if ev.song_id and ev.type in _COUNTER:
                # write-around group: invalidate, the next read refills from Postgres
                pipe.delete(f"song_stats#{ev.song_id}")
            pipe.execute()
        except redis.RedisError:
            pass  # the cache is an optimization; Postgres already has the truth
