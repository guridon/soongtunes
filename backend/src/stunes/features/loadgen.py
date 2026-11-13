"""Phase 3 load test against POST /api/features.

Open loop at a fixed request rate (requests don't wait for each other, like real traffic),
user ids drawn from a Zipf distribution over the seeded users, a share of requests for users
that don't exist (cache penetration). Once a second it samples the server's tier counters,
so the output is a time series of latency and *L3 (Postgres) queries per second*.

Stampede setup (`warm_ttl`): before the load starts, every key is written to Redis with the
same TTL — the batch-backfill pattern — so they all expire together mid-run.
"""

import asyncio
import random
import time
from datetime import UTC, datetime

import httpx
import numpy as np
import orjson
import psycopg
import redis.asyncio as aioredis

from stunes.config import Settings
from stunes.features.store import PROFILES, FeatureStore


def seeded_users(settings: Settings, limit: int) -> list[str]:
    with psycopg.connect(settings.postgres_dsn) as conn:
        rows = conn.execute(
            "SELECT substr(pk, length('user_recent_plays#') + 1) FROM feature "
            "WHERE pk LIKE 'user_recent_plays#%%' ORDER BY pk LIMIT %s",
            (limit,),
        ).fetchall()
    return [r[0] for r in rows]


async def warm(
    settings: Settings, users: list[str], groups: list[str], ttl: int, jitter: float
) -> None:
    import asyncpg

    pg = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=4)
    r = aioredis.Redis.from_url(settings.redis_url)
    try:
        store = FeatureStore(pg, r)
        for g in groups:
            # one shared TTL (+ the profile's jitter): what a bulk backfill does
            await store.warm(g, users, ttl, jitter=jitter)
    finally:
        await pg.close()
        await r.aclose()


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    return round(float(np.percentile(xs, q)), 2)


async def run(
    settings: Settings,
    *,
    api: str,
    profile: str,
    rps: float,
    duration_s: int,
    users: list[str],
    groups: list[str],
    zipf_s: float = 1.1,
    missing: float = 0.1,
    missing_pool: int = 1000,
    warm_ttl: int | None = None,
    seed: int = 3,
    label: str = "",
) -> dict:
    rng = random.Random(seed)
    ranks = np.arange(1, len(users) + 1)
    weights = 1.0 / ranks**zipf_s
    cdf = np.cumsum(weights / weights.sum())
    np_rng = np.random.default_rng(seed)

    redis_client = aioredis.Redis.from_url(settings.redis_url)
    await redis_client.flushdb()
    await redis_client.aclose()
    if warm_ttl:
        await warm(settings, users, groups, warm_ttl, PROFILES[profile].jitter)

    latencies: list[tuple[float, float, int]] = []  # (t, ms, status)
    series: list[dict] = []
    async with httpx.AsyncClient(
        base_url=api, timeout=30, limits=httpx.Limits(max_connections=256)
    ) as client:
        # counters are monotonic and shared by the workers: subtract a starting snapshot
        baseline = (await client.get("/api/features/stats")).json()
        sem = asyncio.Semaphore(512)
        t0 = time.monotonic()

        async def one(user: str) -> None:
            async with sem:
                start = time.monotonic()
                try:
                    resp = await client.post(
                        "/api/features",
                        json={"entity_ids": [user], "groups": groups, "profile": profile},
                    )
                    status = resp.status_code
                except httpx.HTTPError:
                    status = 599
                latencies.append((start - t0, (time.monotonic() - start) * 1000, status))

        async def sampler() -> None:
            prev: dict = dict(baseline)
            while time.monotonic() - t0 < duration_s + 2:
                await asyncio.sleep(1.0)
                stats = (await client.get("/api/features/stats")).json()
                delta = {k: v - prev.get(k, 0) for k, v in stats.items()}
                prev = stats
                sec = round(time.monotonic() - t0)
                window = [ms for t, ms, _ in latencies if sec - 1 <= t < sec]
                series.append(
                    {
                        "t": sec,
                        "requests": len(window),
                        "p50": _pct(window, 50),
                        "p99": _pct(window, 99),
                        **{f"d_{k}": v for k, v in delta.items()},
                    }
                )

        sampler_task = asyncio.create_task(sampler())
        tasks = []
        n = int(rps * duration_s)
        for i in range(n):
            delay = t0 + i / rps - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            if rng.random() < missing:
                # never-seeded ids from a fixed pool: the same unknown id comes back (a deleted
                # user, a scraper), which is what negative caching is for. Fresh random ids never
                # repeat, so C4/C5 measured 0 negative hits (troubleshooting #20)
                user = f"p_9{rng.randrange(missing_pool):08d}"
            else:
                user = users[int(np.searchsorted(cdf, np_rng.random()))]
            tasks.append(asyncio.create_task(one(user)))
        await asyncio.gather(*tasks)
        await sampler_task
        end = (await client.get("/api/features/stats")).json()
        final = {k: v - baseline.get(k, 0) for k, v in end.items()}

    ms = [m for _, m, s in latencies if s == 200]
    # (was "d_l3_query", a key the store never wrote: l3_qps came out 0 — troubleshooting #20)
    l3 = [p.get("d_l3_sql", 0) for p in series]
    hits = {
        k: final.get(k, 0)
        for k in ("l1_hit", "l2_hit", "l2_negative_hit", "l2_miss", "l2_early_refresh")
    }
    lookups = sum(hits.values()) - hits["l2_early_refresh"]
    return {
        "label": label,
        "profile": profile,
        "rps": rps,
        "duration_s": duration_s,
        "users": len(users),
        "groups": groups,
        "zipf_s": zipf_s,
        "missing": missing,
        "missing_pool": missing_pool,
        "warm_ttl": warm_ttl,
        "at": datetime.now(UTC).isoformat(),
        "requests": len(latencies),
        "errors": sum(1 for *_, s in latencies if s != 200),
        "latency_ms": {"p50": _pct(ms, 50), "p95": _pct(ms, 95), "p99": _pct(ms, 99)},
        "l3_qps": {
            "mean": round(float(np.mean(l3)), 1) if l3 else None,
            "max": max(l3) if l3 else None,
        },
        "counters": final,
        "hit_ratio": {k: round(v / lookups, 4) for k, v in hits.items()} if lookups else {},
        "series": series,
    }


def save(settings: Settings, result: dict) -> str:
    out = settings.results_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{result['profile']}-{result['label'] or 'run'}-{int(time.time())}.json"
    (out / name).write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    return str(out / name)
