"""Feature serving with three tiers.

    L1  per-process memory, 1–5 s TTL — absorbs hot keys (Redis is single-threaded)
    L2  Redis, TTL from the group spec, optional jitter / XFetch / negative caching
    L3  Postgres: the feature table, or an on-demand computation for `compute:` groups

Every L2 value is an envelope [value, expires_at_ms, recompute_ms, negative] so XFetch
(probabilistic early expiration, Vattani et al. 2015) can decide to refresh *before* the key
expires: refresh if  now - recompute_ms * beta * ln(rand) >= expires_at.  A key that took
longer to recompute, or is closer to expiry, is refreshed earlier — and only by the few
requests that win the draw, instead of every request at the moment of expiry.

Cache profiles map 1:1 to the spec's experiments C0..C5 and are chosen per request.
"""

import asyncio
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import msgpack
import redis.asyncio as aioredis
import yaml

from stunes import metrics
from stunes.features import compute

GROUPS_FILE = Path(__file__).with_name("groups.yaml")


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    entity: str
    ttl_sec: int
    source: str = "table"
    write: str = "through"

    @property
    def compute_fn(self) -> str | None:
        return self.source.split(":", 1)[1] if self.source.startswith("compute:") else None


def load_groups(path: Path = GROUPS_FILE) -> dict[str, FeatureGroup]:
    spec = yaml.safe_load(path.read_text())
    groups = {g["name"]: FeatureGroup(**g) for g in spec["feature_groups"]}
    for g in groups.values():
        if g.compute_fn and g.compute_fn not in compute.COMPUTE:
            raise ValueError(f"group {g.name}: unknown compute function {g.compute_fn}")
    return groups


@dataclass(frozen=True)
class CacheProfile:
    name: str
    redis: bool = True
    jitter: float = 0.0  # ± fraction of the TTL
    xfetch_beta: float = 0.0  # 0 = off
    negative_ttl: int = 0  # seconds to remember "no such entity"; 0 = off
    l1_ttl: float = 0.0  # seconds; 0 = off


PROFILES = {
    "C0": CacheProfile("C0", redis=False),
    "C1": CacheProfile("C1"),
    "C2": CacheProfile("C2", jitter=0.15),
    "C3": CacheProfile("C3", jitter=0.15, xfetch_beta=1.0),
    "C4": CacheProfile("C4", jitter=0.15, xfetch_beta=1.0, negative_ttl=30),
    "C5": CacheProfile("C5", jitter=0.15, xfetch_beta=1.0, negative_ttl=30, l1_ttl=2.0),
}
DEFAULT_PROFILE = "C5"


class _L1:
    """Tiny TTL map (no dependency): bounded size, lazy expiry."""

    def __init__(self, ttl: float, maxsize: int = 100_000):
        self.ttl, self.maxsize = ttl, maxsize
        self._d: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> tuple[bool, Any]:
        hit = self._d.get(key)
        if hit is None:
            return False, None
        if hit[0] < time.monotonic():
            self._d.pop(key, None)
            return False, None
        return True, hit[1]

    def put(self, key: str, value: Any) -> None:
        if len(self._d) >= self.maxsize:
            self._d.pop(next(iter(self._d)))  # oldest insertion first
        self._d[key] = (time.monotonic() + self.ttl, value)


def key(group: str, entity_id: str) -> str:
    return f"{group}#{entity_id}"  # same layout as the feature table's pk


class _Ops(Counter):
    """Per-process tallies, mirrored into a Prometheus counter on every increment. The load test
    used to read this dict directly, which with two API workers meant whichever worker answered
    (troubleshooting #20)."""

    def __setitem__(self, key: str, value: int) -> None:
        delta = value - self.get(key, 0)
        super().__setitem__(key, value)
        if delta > 0:
            metrics.FEATURE_OPS.labels(key).inc(delta)


class FeatureStore:
    def __init__(
        self,
        pg: asyncpg.Pool,
        redis_client: aioredis.Redis,
        groups: dict[str, FeatureGroup] | None = None,
    ):
        self.pg = pg
        self.redis = redis_client
        self.groups = groups or load_groups()
        self._l1: dict[str, _L1] = {}
        self._bg: set[asyncio.Task] = set()
        self.stats: Counter = _Ops()  # tier hits/misses, L3 queries — read by the load test

    # ------------------------------------------------------------------ read
    async def get(
        self, entity_ids: list[str], group_names: list[str], profile: str = DEFAULT_PROFILE
    ) -> dict[str, dict[str, Any]]:
        prof = PROFILES[profile]
        out: dict[str, dict[str, Any]] = {e: {} for e in entity_ids}
        for name in group_names:
            values = await self._get_group(self.groups[name], entity_ids, prof)
            for e in entity_ids:
                out[e][name] = values.get(e)
        return out

    async def _get_group(
        self, g: FeatureGroup, ids: list[str], prof: CacheProfile
    ) -> dict[str, Any]:
        found: dict[str, Any] = {}
        todo = list(dict.fromkeys(ids))
        self.stats["lookups"] += len(todo)

        # L1 is per process and only TTL-bounded: a consumer update can take up to l1_ttl to
        # show here, and two workers can disagree for that long (what C5 measures)
        l1 = self._l1.setdefault(prof.name, _L1(prof.l1_ttl)) if prof.l1_ttl else None
        if l1:
            rest = []
            for e in todo:
                hit, v = l1.get(key(g.name, e))
                if hit:
                    found[e] = v
                    self._count(g, "l1_hit")
                else:
                    rest.append(e)
            todo = rest

        refresh: dict[str, bytes] = {}  # entity -> the exact blob we saw, for compare-and-set
        if prof.redis and todo:
            raw = await self.redis.mget([key(g.name, e) for e in todo])
            now_ms = time.time() * 1000
            rest = []
            for e, blob in zip(todo, raw, strict=True):
                if blob is None:
                    self._count(g, "l2_miss")
                    rest.append(e)
                    continue
                value, expires_at, recompute_ms, negative = msgpack.unpackb(blob)
                if prof.xfetch_beta and _xfetch(now_ms, expires_at, recompute_ms, prof.xfetch_beta):
                    # a miss for this request (it recomputes); everyone else keeps the cached value
                    self._count(g, "l2_early_refresh")
                    refresh[e] = blob
                    rest.append(e)
                    continue
                self._count(g, "l2_negative_hit" if negative else "l2_hit")
                found[e] = None if negative else value
            todo = rest

        if todo:
            for _ in todo:
                metrics.FEATURE_LOOKUPS.labels(g.name, "l3_load").inc()
            loaded, recompute_ms = await self._load(g, todo)
            found.update(loaded)
            if prof.redis:
                self._spawn(self._fill_l2(g, prof, todo, loaded, recompute_ms, refresh))
        if l1:
            for e, v in found.items():
                l1.put(key(g.name, e), v)
        return found

    async def _load(self, g: FeatureGroup, ids: list[str]) -> tuple[dict[str, Any], float]:
        """L3 read. Returns values and the per-key share of the recompute time."""
        t = time.perf_counter()
        self.stats["l3_load"] += 1
        self.stats["l3_entities"] += len(ids)
        if g.compute_fn:
            values = await compute.COMPUTE[g.compute_fn](self.pg, ids, self.stats)
        else:
            self.stats["l3_sql"] += 1
            rows = await self.pg.fetch(
                "SELECT pk, value FROM feature WHERE pk = ANY($1)", [key(g.name, e) for e in ids]
            )
            prefix = len(g.name) + 1
            values = {r["pk"][prefix:]: msgpack.unpackb(r["value"]) for r in rows}
        return values, (time.perf_counter() - t) * 1000 / len(ids)

    # ------------------------------------------------------------------ fill
    async def _fill_l2(
        self,
        g: FeatureGroup,
        prof: CacheProfile,
        ids: list[str],
        loaded: dict[str, Any],
        recompute_ms: float,
        refresh: dict[str, bytes],
    ) -> None:
        """Write L3 results back to Redis without clobbering newer data.

        A plain miss uses SET NX: if the consumer wrote a fresher value (write-through) while
        we were reading Postgres, ours is the stale one and is dropped. An XFetch refresh
        replaces the key only if it still holds the exact blob this request saw.
        """
        now_ms = time.time() * 1000
        pipe = self.redis.pipeline(transaction=False)
        for e in ids:
            if e in loaded:
                ttl = (
                    g.ttl_sec * (1 + random.uniform(-prof.jitter, prof.jitter))
                    if prof.jitter
                    else g.ttl_sec
                )
                envelope = [loaded[e], now_ms + ttl * 1000, recompute_ms, False]
            elif prof.negative_ttl:
                ttl = prof.negative_ttl
                envelope = [None, now_ms + ttl * 1000, recompute_ms, True]
            else:
                continue
            blob, ex = msgpack.packb(envelope), max(1, int(ttl))
            if e in refresh:
                pipe.eval(_CAS, 1, key(g.name, e), refresh[e], blob, ex)
            else:
                pipe.set(key(g.name, e), blob, ex=ex, nx=True)
        await pipe.execute()

    def _count(self, g: FeatureGroup, outcome: str) -> None:
        self.stats[outcome] += 1
        metrics.FEATURE_LOOKUPS.labels(g.name, outcome).inc()

    def _spawn(self, coro) -> None:
        # upper tiers are filled in the background: the response doesn't wait for Redis
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def warm(
        self,
        group: str,
        entity_ids: list[str],
        ttl_sec: int,
        jitter: float = 0.0,
        batch: int = 2000,
    ) -> float:
        """Bulk-load L2 so every key expires at one absolute moment (± jitter) — the backfill
        pattern behind a stampede. Returns that moment (epoch ms)."""
        g = self.groups[group]
        if not entity_ids:
            return 0.0
        # XFetch needs what *one request* pays to recompute a key, not batch time / n
        _, single_ms = await self._load(g, entity_ids[:1])
        # one anchor for all batches: later batches must not expire later
        expires_at = time.time() * 1000 + ttl_sec * 1000
        for i in range(0, len(entity_ids), batch):
            loaded, _ = await self._load(g, entity_ids[i : i + batch])
            pipe = self.redis.pipeline(transaction=False)
            for e, v in loaded.items():
                at = expires_at + (
                    random.uniform(-jitter, jitter) * ttl_sec * 1000 if jitter else 0
                )
                pipe.set(key(g.name, e), msgpack.packb([v, at, single_ms, False]), pxat=int(at))
            await pipe.execute()
        return expires_at


# compare-and-set: replace KEYS[1] with ARGV[2] only if it still equals ARGV[1]
_CAS = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
end
return nil
"""


def envelope(value: Any, ttl_sec: int) -> bytes:
    """What a writer (the consumer's write-through) must store: the same layout readers expect."""
    return msgpack.packb([value, time.time() * 1000 + ttl_sec * 1000, 0.0, False])


def _xfetch(now_ms: float, expires_at_ms: float, recompute_ms: float, beta: float) -> bool:
    # -ln(rand) is an Exp(1) draw: usually small, occasionally large -> early refresh
    return now_ms - recompute_ms * beta * math.log(random.random() or 1e-12) >= expires_at_ms
