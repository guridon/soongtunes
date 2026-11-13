"""Does Redis serve what Postgres committed? The consumer's write-through against CDC.

One run per cache writer, same live stream and the same faults as `stunes ingest chaos`:
    1. chaos: 30,000 live events, the ingest consumer killed -9 five times
       (cdc: the CDC applier is also killed -9 --applier-kills times and restarted)
    2. compare every user_recent_plays row with Redis (stunes.ingest.cache_check.diff)
    3. an operator fixes data in SQL, not through the consumer:
       drop the most frequent song from every list, delete 500 users
    4. compare again (consumer: after 5 s; cdc: until it matches, 120 s at most)

Run (repo root, after `docker compose --profile cdc up -d connect` and registering
infra/cdc/feature-cache.json):
    python -m stunes.ingest.cdc_bench --cache-writer consumer|cdc
"""

import argparse
import json
import logging
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import orjson
import psycopg
from confluent_kafka import Consumer, TopicPartition

from stunes.config import get_settings
from stunes.ingest import bench, cache_check
from stunes.ingest.cdc_cache import TOPICS

logger = logging.getLogger("stunes.ingest.cdc_bench")
GROUP = "stunes-cdc-cache"
STATIC = True  # set from --static


def _applier(lag_log: Path, log: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "stunes.ingest.cdc_cache",
            "--group",
            GROUP,
            "--lag-log",
            str(lag_log),
            *(["--instance-id", "cdc-cache-1"] if STATIC else []),
        ],
        stdout=log.open("ab"),
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )


def _group_lag(settings) -> int:
    """Messages on the CDC topics the applier group has not committed yet."""
    c = Consumer({"bootstrap.servers": settings.kafka_bootstrap, "group.id": GROUP})
    try:
        md = c.list_topics(timeout=5)
        tps = [
            TopicPartition(t, p) for t in TOPICS if t in md.topics for p in md.topics[t].partitions
        ]
        lag = 0
        for tp in c.committed(tps, timeout=5):
            lo, hi = c.get_watermark_offsets(tp, timeout=5)
            lag += hi - (tp.offset if tp.offset >= 0 else lo)
        return lag
    finally:
        c.close()


def _wait(pred, timeout_s: float, every_s: float = 0.5) -> float | None:
    t = time.monotonic()
    while time.monotonic() - t < timeout_s:
        if pred():
            return round(time.monotonic() - t, 2)
        time.sleep(every_s)
    return None


def _clean(d: dict) -> bool:
    return d["stale"] == 0 and d["missing"] == 0 and d["orphan"] == 0


def _slot_wal_bytes(settings) -> int | None:
    with psycopg.connect(settings.postgres_dsn) as conn:
        row = conn.execute(
            "SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn)::bigint "
            "FROM pg_replication_slots WHERE slot_name = 'stunes_feature_cache'"
        ).fetchone()
    return row[0] if row else None


def _pct(xs: list[float]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)
    q = lambda p: round(xs[min(len(xs) - 1, int(p * len(xs)))], 1)  # noqa: E731
    return {
        "n": len(xs),
        "p50": q(0.50),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": round(xs[-1], 1),
        "mean": round(statistics.fmean(xs), 1),
    }


def run(writer: str, events: int, rate: float, kills: int, applier_kills: int, seed: int) -> dict:
    os.environ["CACHE_WRITER"] = writer  # read by the consumers bench.chaos spawns
    settings = get_settings()
    logs = settings.data_dir / "ingest-logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    lag_log, app_log = logs / f"cdc-{stamp}.lag", logs / f"cdc-{stamp}.applier.log"
    out: dict = {"cache_writer": writer}
    applier = None
    stop = threading.Event()
    killed: list[float] = []

    if writer == "cdc":
        applier = _applier(lag_log, app_log)
        # start from a caught-up applier, so the run measures its own changes
        out["warm_s"] = _wait(lambda: _group_lag(settings) == 0, 120, 1.0)
        lag_log.write_text("")
        rng = random.Random(seed + 1)
        duration = events / rate
        at = sorted(rng.uniform(0.15 * duration, 0.85 * duration) for _ in range(applier_kills))

        def killer():
            nonlocal applier
            t0 = time.monotonic()
            for a in at:
                if stop.wait(max(0.0, a - (time.monotonic() - t0))):
                    return
                applier.send_signal(signal.SIGKILL)
                applier.wait()
                killed.append(round(time.monotonic() - t0, 2))
                logger.info("applier kill -9 at %.1fs", killed[-1])
                applier = _applier(lag_log, app_log)

        threading.Thread(target=killer, daemon=True).start()

    chaos = bench.chaos(
        settings,
        mode="parallel_watermark",
        events=events,
        rate=rate,
        kills=kills,
        seed=seed,
        label=f"cache-{writer}",
    )
    stop.set()
    out["chaos_run"] = {k: chaos[k] for k in ("lost", "duplicates", "effects_match", "processed")}
    out["applier_kills_at_s"] = killed

    if writer == "cdc":
        out["after_chaos_converged_s"] = _wait(lambda: _clean(cache_check.diff()), 120)
    out["after_chaos"] = cache_check.diff()

    fixes = [cache_check.correct("drop-song", 0), cache_check.correct("delete-users", 500)]
    out["corrections"] = fixes
    t = time.monotonic()
    if writer == "cdc":
        out["after_fix_converged_s"] = _wait(lambda: _clean(cache_check.diff()), 120)
    else:
        time.sleep(max(0.0, 5 - (time.monotonic() - t)))
    out["after_fix"] = cache_check.diff()

    if applier is not None:
        _wait(lambda: _group_lag(settings) == 0, 60, 1.0)
        applier.send_signal(signal.SIGTERM)
        applier.wait(timeout=30)
        rows = [line.split() for line in lag_log.read_text().splitlines() if line]
        out["commit_to_cache_ms"] = _pct([float(lag) for _, lag in rows])
        out["applier_assigned_at"] = [
            line.split(" INFO")[0]
            for line in app_log.read_text().splitlines()
            if "assigned" in line
        ]
        out["slot_wal_bytes_end"] = _slot_wal_bytes(settings)

    config = {
        "cache_writer": writer,
        "events": events,
        "rate": rate,
        "kills": kills,
        "applier_kills": applier_kills if writer == "cdc" else 0,
        "seed": seed,
        "applier_static_membership": STATIC if writer == "cdc" else None,
    }
    path = settings.results_dir / "ingest" / f"cdc-{writer}-{stamp}.json"
    path.write_bytes(
        orjson.dumps(
            {
                "kind": "cdc",
                "run_id": path.stem,
                "config": config,
                "result": out,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    logger.info("%s", json.dumps(out, ensure_ascii=False))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-writer", choices=["consumer", "cdc"], required=True)
    ap.add_argument("--events", type=int, default=30_000)
    ap.add_argument("--rate", type=float, default=300)
    ap.add_argument("--kills", type=int, default=5)
    ap.add_argument("--applier-kills", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--static", choices=["on", "off"], default="on", help="applier static membership"
    )
    a = ap.parse_args()
    global STATIC
    STATIC = a.static == "on"
    run(a.cache_writer, a.events, a.rate, a.kills, a.applier_kills, a.seed)


if __name__ == "__main__":
    main()
