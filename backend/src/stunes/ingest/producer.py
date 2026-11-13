"""Synthetic user-event stream.

One playlist = one virtual user who listens to its songs in order. Each step is a play
(75%) or a skip (25%); a play is followed by a like 8% of the time; 3% of steps are a
search for one of the playlist's tags. Users are interleaved by event time, so the stream
looks like many concurrent listeners. Every `poison_every`-th event is deliberately invalid.

Everything is seeded by run_id, so a run can be regenerated to recompute what was sent.
"""

import heapq
import random
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from confluent_kafka import Producer
from neo4j import GraphDatabase

from stunes.config import Settings
from stunes.ingest.events import TOPIC, event_id, is_poison

USERS_QUERY = """
MATCH (p:Playlist)
WITH p, COUNT { (p)-[:INCLUDES]->() } AS n
WHERE n >= 10
WITH p ORDER BY p.playlist_id LIMIT $n
RETURN p.playlist_id AS id,
       [(p)-[:INCLUDES]->(s:Song) | s.song_id] AS songs,
       [(p)-[:HAS_TAG]->(t:Tag) | t.name] AS tags
"""


def load_users(settings: Settings, n: int = 5000) -> list[dict]:
    """Playlists with >= 10 songs, cached under .data/exports."""
    path = settings.data_dir / "exports" / f"users_{n}.jsonl"
    if path.exists():
        return [orjson.loads(line) for line in path.read_bytes().splitlines()]
    with GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    ) as driver:
        records, _, _ = driver.execute_query(USERS_QUERY, n=n, database_=settings.neo4j_database)
    users = [r.data() for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(u) + b"\n" for u in users))
    return users


@dataclass
class Expected:
    """What a run sent, for checking the consumer's effects."""

    total: int = 0
    poison: int = 0
    by_type: Counter = field(default_factory=Counter)

    @property
    def valid(self) -> int:
        return self.total - self.poison


def generate(
    users: list[dict],
    run_id: str,
    n_events: int,
    poison_every: int = 100,
    start_ms: int | None = None,
) -> Iterator[tuple[bytes, bytes, int, bool, str]]:
    """Yields (key, value, seq, poison, type) in event-time order."""
    rng = random.Random(run_id)
    start = start_ms or 1_793_000_000_000
    # heap of (event time, user index, playlist position, action); action "step" picks
    # play/skip/search, "like" is the like that trails a play by one second
    heap = [(start + rng.randint(0, 600_000), i, 0, "step") for i in range(len(users))]
    heapq.heapify(heap)
    for seq in range(n_events):
        ts, ui, pos, action = heapq.heappop(heap)
        user = users[ui]
        uid = f"p_{user['id']}"
        song = user["songs"][pos % len(user["songs"])]
        if action == "like":
            kind, payload = "like", {"song_id": song}
        else:
            r = rng.random()
            if r < 0.03 and user["tags"]:
                kind, payload, gap = "search", {"query": rng.choice(user["tags"])}, 20_000
            elif r < 0.28:
                kind, payload, gap = "skip", {"song_id": song}, rng.randint(5_000, 40_000)
            else:
                kind, payload, gap = "play", {"song_id": song}, rng.randint(120_000, 280_000)
                if rng.random() < 0.08:
                    heapq.heappush(heap, (ts + 1_000, ui, pos, "like"))
            heapq.heappush(heap, (ts + gap, ui, pos + 1, "step"))
        poison = is_poison(seq, poison_every)
        body = {
            "event_id": str(event_id(run_id, seq)),
            "user_id": uid,
            "type": kind,
            "song_id": None,
            "query": None,
            "ts": ts,
            "schema_version": 1,
            **payload,
        }
        if poison:  # break it in one of the ways the schema rejects
            body |= [{"type": "pause"}, {"schema_version": 2}, {"song_id": None, "type": "play"}][
                seq % 3
            ]
        yield uid.encode(), orjson.dumps(body), seq, poison, kind


def produce(
    settings: Settings,
    users: list[dict],
    run_id: str,
    n_events: int,
    rate: float = 0,
    poison_every: int = 100,
    live_ts: bool = False,
) -> Expected:
    """Publish n_events. rate=0 means as fast as possible (backlog for throughput runs).
    live_ts stamps events with the wall clock so end-to-end latency is meaningful."""
    producer = Producer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "acks": "all",
            "enable.idempotence": True,
            "linger.ms": 10,
            "compression.type": "lz4",
            "queue.buffering.max.messages": 500_000,
        }
    )
    expected = Expected()
    started = time.monotonic()
    for key, value, seq, poison, kind in generate(users, run_id, n_events, poison_every):
        if live_ts:
            body = orjson.loads(value)
            body["ts"] = int(time.time() * 1000)
            value = orjson.dumps(body)
        while True:
            try:
                producer.produce(TOPIC, key=key, value=value)
                break
            except BufferError:
                producer.poll(0.1)
        producer.poll(0)
        expected.total += 1
        expected.poison += poison
        if not poison:
            expected.by_type[kind] += 1
        if rate > 0:
            ahead = (seq + 1) / rate - (time.monotonic() - started)
            if ahead > 0:
                time.sleep(ahead)
    producer.flush(60)
    return expected


def write_expected(path: Path, run_id: str, expected: Expected) -> None:
    path.write_bytes(
        orjson.dumps(
            {
                "run_id": run_id,
                "total": expected.total,
                "poison": expected.poison,
                "by_type": dict(expected.by_type),
            }
        )
    )
