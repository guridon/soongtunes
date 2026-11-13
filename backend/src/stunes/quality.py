"""Data-quality checks (Phase 7).

Each check measures one number and compares it to a threshold:

    embedding_coverage      share of the active version's target songs that have a vector
    index_consistency       OpenSearch song docs / Neo4j Song nodes (derived index drift)
    feature_freshness_s     age of the newest user_recent_plays write (is ingestion alive?)
    dlq_messages            dead-lettered events (schema violations that exhausted retries)
    stale_sessions          VirtualQuery nodes untouched for 7 days (state that never expires)
    eval_hit_at_10          hybrid hit@10 of the latest `make eval` run (offline quality)

`stunes quality check` prints a report and exits non-zero on failure (CI-style);
`stunes quality serve` re-runs every interval and exports gauges for Prometheus alerts.
"""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import orjson
from confluent_kafka import Consumer, KafkaException, TopicPartition
from prometheus_client import Gauge, start_http_server

from stunes.config import SERVED_STRATEGY, Settings
from stunes.db import Stores
from stunes.ingest.events import DLQ_TOPIC

logger = logging.getLogger(__name__)

VALUE = Gauge("stunes_quality_value", "Measured value of a data-quality check", ["check"])
OK = Gauge("stunes_quality_ok", "1 if the check passed, 0 if it failed", ["check"])
LAST_RUN = Gauge("stunes_quality_last_run_timestamp", "When checks last ran (unix seconds)")


@dataclass
class Check:
    name: str
    measure: Callable[[Stores, Settings], Awaitable[float | None]]
    ok: Callable[[float], bool]
    threshold: str


_CORPUS = re.compile(r"corpus: playlist_count>=(\d+)")


async def _embedding_coverage(st: Stores, s: Settings) -> float | None:
    """Vectors in the active version / the songs that version was built for. A popularity-
    filtered version (backfill --min-popularity, recorded in its template) is measured against
    the songs in at least that many playlists, not the whole catalog: against all songs the
    served pop5 version sits at 0.20, so a threshold low enough to pass it (it was 0.10) also
    passed a pop5 version missing half its vectors."""
    row = await st.pg.fetchrow(
        "SELECT version, template FROM embedding_model WHERE entity='song' AND status='active'"
    )
    if row is None:
        return 0.0
    n = await st.pg.fetchval(
        "SELECT count(*) FROM song_embedding WHERE version = $1", row["version"]
    )
    m = _CORPUS.search(row["template"])
    if m:
        body = {"query": {"range": {"playlist_count": {"gte": int(m[1])}}}}
        target = (await st.search.count(index=s.songs_index, body=body))["count"]
    else:
        target = (await st.graph.read("MATCH (s:Song) RETURN count(s) AS n"))[0]["n"]
    return n / target if target else None


async def _index_consistency(st: Stores, s: Settings) -> float | None:
    docs = (await st.search.count(index=s.songs_index))["count"]
    songs = (await st.graph.read("MATCH (s:Song) RETURN count(s) AS n"))[0]["n"]
    return docs / songs if songs else None


async def _feature_freshness(st: Stores, _: Settings) -> float | None:
    age = await st.pg.fetchval(
        "SELECT extract(epoch FROM now() - max(updated_at)) FROM feature WHERE pk LIKE 'user_recent_plays#%'"
    )
    return float(age) if age is not None else None


async def _dlq_messages(_: Stores, s: Settings) -> float | None:
    def size() -> float | None:
        c = Consumer({"bootstrap.servers": s.kafka_bootstrap, "group.id": "stunes-quality"})
        try:
            md = c.list_topics(DLQ_TOPIC, timeout=5).topics.get(DLQ_TOPIC)
            if md is None or md.error:
                return 0.0
            return float(
                sum(
                    c.get_watermark_offsets(TopicPartition(DLQ_TOPIC, p), timeout=5)[1]
                    for p in md.partitions
                )
            )
        except KafkaException:
            return None
        finally:
            c.close()

    return await asyncio.to_thread(size)


async def _stale_sessions(st: Stores, _: Settings) -> float | None:
    rows = await st.graph.read(
        "MATCH (vq:VirtualQuery) WHERE vq.updated_at IS NULL OR vq.updated_at < datetime() - duration('P7D') "
        "RETURN count(vq) AS n"
    )
    return float(rows[0]["n"])


async def _eval_hit(_: Stores, s: Settings) -> float | None:
    runs = sorted((s.results_dir / "eval").glob("*/summary.json"))
    for path in reversed(runs):
        summary = orjson.loads(path.read_bytes())
        # the served strategy once a run holds it; `hybrid` (served before) for older runs
        arms = summary["strategies"]
        arm = arms.get(SERVED_STRATEGY) or arms.get("hybrid")
        if arm and arm["metrics"].get("ALL", {}).get("n", 0) >= 100:  # skip smoke runs
            return arm["metrics"]["ALL"]["hit"]
    return None


CHECKS = [
    Check("embedding_coverage", _embedding_coverage, lambda v: v >= 0.999, ">= 0.999"),
    Check(
        "index_consistency", _index_consistency, lambda v: abs(v - 1.0) <= 0.001, "1.000 ± 0.001"
    ),
    Check("feature_freshness_s", _feature_freshness, lambda v: v <= 3600, "<= 3600 s"),
    Check("dlq_messages", _dlq_messages, lambda v: v <= 1000, "<= 1000"),
    Check("stale_sessions", _stale_sessions, lambda v: v <= 10_000, "<= 10000"),
    Check("eval_hit_at_10", _eval_hit, lambda v: v >= 0.5, ">= 0.50"),
]


async def run_checks(settings: Settings) -> list[dict]:
    stores = await Stores.open(settings)
    results = []
    try:
        for check in CHECKS:
            try:
                value = await check.measure(stores, settings)
            except Exception as e:  # a broken probe is itself a failed check
                logger.warning("check %s failed to run: %r", check.name, e)
                value = None
            ok = value is not None and check.ok(value)
            VALUE.labels(check.name).set(value if value is not None else float("nan"))
            OK.labels(check.name).set(1 if ok else 0)
            results.append(
                {"check": check.name, "value": value, "threshold": check.threshold, "ok": ok}
            )
        LAST_RUN.set(time.time())
    finally:
        await stores.close()
    return results


def check_once(settings: Settings) -> bool:
    results = asyncio.run(run_checks(settings))
    for r in results:
        value = "n/a" if r["value"] is None else f"{r['value']:.4g}"
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['check']:<22} {value:>12}  ({r['threshold']})")
    return all(r["ok"] for r in results)


def serve(settings: Settings, port: int = 9109, interval_s: int = 60) -> None:
    start_http_server(port)
    logger.info("quality checks every %ds, metrics on :%d", interval_s, port)
    while True:
        try:
            asyncio.run(run_checks(settings))
        except Exception as e:  # a store outage must not kill the exporter
            logger.warning("quality run failed: %r", e)
            for check in CHECKS:
                OK.labels(check.name).set(0)
        time.sleep(interval_s)
