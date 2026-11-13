"""Prometheus metrics (Phase 7).

Service health (latency per pipeline stage, candidate counts, verifier drops, LLM errors),
online recommendation quality (coverage, popularity bias, artist diversity), and the
feature-store / ingestion counters. With several uvicorn workers set PROMETHEUS_MULTIPROC_DIR
so /metrics aggregates across processes.
"""

import math
import os
from collections.abc import Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

from stunes.config import SERVED_STRATEGY

STAGE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)

CHAT_LATENCY = Histogram(
    "stunes_chat_seconds", "End-to-end chat turn latency", ["strategy"], buckets=STAGE_BUCKETS
)
STAGE_LATENCY = Histogram(
    "stunes_chat_stage_seconds",
    "Chat stage latency (the served strategy's stages; agent sources as source_<name>)",
    ["stage"],
    buckets=STAGE_BUCKETS,
)
CANDIDATES = Histogram(
    "stunes_candidates",
    "Candidates returned per source per turn",
    ["source"],
    buckets=(0, 1, 5, 10, 25, 50, 100, 200),
)
VERIFIER_DROPS = Counter(
    "stunes_verifier_dropped_total", "Candidates dropped by the Cypher verifier"
)
RELAXED = Counter("stunes_relaxed_total", "Turns where a condition was relaxed", ["condition"])
EMPTY_RESULTS = Counter("stunes_empty_results_total", "Turns that returned no songs", ["strategy"])
ERRORS = Counter("stunes_errors_total", "Failures by component", ["component"])

# online quality: what users are shown, not offline relevance
RECOMMENDED = Counter("stunes_recommended_total", "Songs shown, by popularity bucket", ["bucket"])
POPULARITY = Histogram(
    "stunes_recommended_log_popularity",
    "log1p(playlist count) of shown songs",
    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8),
)
ARTIST_DIVERSITY = Histogram(
    "stunes_artist_diversity",
    "distinct artists / songs in one result list",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# every key the feature store tallies (tier outcomes, L3 loads/queries), summed over all API
# workers: the Phase 3 load test reads this through /api/features/stats
FEATURE_OPS = Counter(
    "stunes_feature_store_ops_total", "Feature store tallies read by the load test", ["op"]
)
FEATURE_LOOKUPS = Counter(
    "stunes_feature_lookups_total", "Feature lookups by tier outcome", ["group", "outcome"]
)

INGEST_EVENTS = Counter(
    "stunes_ingest_events_total",
    "Consumed events by outcome",
    ["outcome"],  # ok|duplicate|retry|dlq
)
FRESHNESS = Histogram(
    "stunes_feature_freshness_seconds",
    "Event time -> feature committed (ingestion lag)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300),
)


def popularity_bucket(playlist_count: int) -> str:
    if playlist_count >= 1000:
        return "head"
    if playlist_count >= 50:
        return "torso"
    return "tail"


def observe_turn(strategy: str, total_ms: float | None, trace: dict, songs: Iterable[dict]) -> None:
    songs = list(songs)
    if total_ms is not None:
        CHAT_LATENCY.labels(strategy).observe(total_ms / 1000)
    if not songs:
        EMPTY_RESULTS.labels(strategy).inc()
    if strategy != SERVED_STRATEGY:
        return
    for stage, ms in (trace.get("timings_ms") or {}).items():
        if stage != "total":
            STAGE_LATENCY.labels(stage).observe(ms / 1000)
    for source, n in (trace.get("candidates") or {}).items():
        CANDIDATES.labels(source).observe(n)
    VERIFIER_DROPS.inc(len(trace.get("dropped_by_verifier") or {}))
    for cond in trace.get("relaxed") or []:
        RELAXED.labels(cond).inc()
    artists = set()
    for s in songs:
        pc = int(s.get("playlist_count") or 0)
        RECOMMENDED.labels(popularity_bucket(pc)).inc()
        POPULARITY.observe(math.log1p(pc))
        artists.update(s.get("artists") or [])
    if songs:  # an empty answer has no diversity (it was a ZeroDivisionError and a 500)
        ARTIST_DIVERSITY.observe(min(1.0, len(artists) / len(songs)))


def _registry():
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    from prometheus_client import REGISTRY

    return REGISTRY


def counter_values(counter: Counter, label: str) -> dict[str, float]:
    """Totals of one labelled counter by label value, summed over every worker process."""
    name = counter._name  # family name, without the _total suffix
    out: dict[str, float] = {}
    for family in _registry().collect():
        if family.name != name:
            continue
        for sample in family.samples:
            if sample.name == f"{name}_total":
                key = sample.labels[label]
                out[key] = out.get(key, 0.0) + sample.value
    return out


def render() -> tuple[bytes, str]:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    from prometheus_client import REGISTRY

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
