"""Phase 2 experiments.

    throughput  pre-produce a backlog, start C consumers, measure how fast they drain it
    chaos       produce live at a fixed rate, kill -9 the consumer K times at random moments,
                restart it each time, then count lost / duplicated events

Both reset the topics and feature tables first and use a fresh consumer group per run.
Accounting (all from Postgres, see processor.py):
    lost        = valid events sent - distinct events processed by the main consumers
    duplicates  = processing transactions that found the event already applied
    effects     = song_stats totals vs what the generator sent (exactly-once *effects*)
"""

import logging
import os
import platform
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import orjson
import psycopg
import redis
from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from pydantic import ValidationError

from stunes.config import Settings
from stunes.ingest.events import DLQ_TOPIC, RETRY_TOPIC, TOPIC, Event
from stunes.ingest.producer import Expected, load_users, produce

logger = logging.getLogger("stunes.ingest.bench")


def reset(settings: Settings, partitions: int) -> None:
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    doomed = [t for t in (TOPIC, RETRY_TOPIC, DLQ_TOPIC) if t in existing]
    if doomed:
        for f in admin.delete_topics(doomed).values():
            f.result()
        for _ in range(50):  # deletion is asynchronous
            if not set(doomed) & set(admin.list_topics(timeout=10).topics):
                break
            time.sleep(0.2)
    futures = admin.create_topics(
        [
            NewTopic(TOPIC, num_partitions=partitions, replication_factor=1),
            NewTopic(RETRY_TOPIC, num_partitions=1, replication_factor=1),
            NewTopic(DLQ_TOPIC, num_partitions=1, replication_factor=1),
        ]
    )
    for f in futures.values():
        f.result()
    with psycopg.connect(settings.postgres_dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE processed_event, song_stats, ingest_attempt")
        conn.execute("DELETE FROM feature WHERE pk LIKE 'user_recent_plays#%'")
    redis.Redis.from_url(settings.redis_url).flushdb()


def _spawn(args: list[str], log: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "stunes.ingest.consumer", *args],
        stdout=log.open("ab"),
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )


def _retry_consumer(run_id: str, group: str, logs: Path, idle_s: int) -> tuple:
    """The retry consumer runs next to the main consumers and may only give up on idling once
    `flag` exists, which the caller creates when the main consumers have exited: until then
    they can still send it poison (troubleshooting #25)."""
    flag = logs / f"{run_id}.main-done"
    flag.unlink(missing_ok=True)
    proc = _spawn(
        [
            "--mode", "retry", "--run-id", run_id, "--group", f"{group}-retry",
            "--idle-exit", str(idle_s), "--idle-after", str(flag),
        ],
        logs / f"{run_id}.retry.log",
    )  # fmt: skip
    return proc, flag


def _finish(proc: subprocess.Popen, timeout: float = 900) -> None:
    """Wait for a helper consumer; kill it if it overruns. An orphaned retry consumer from
    one run kept consuming the next run's (recreated) retry topic and double-dead-lettered
    its poison events (troubleshooting #14). 1% poison x 3 backed-off retries ~ 1.4 s each."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise


def _topic_size(settings: Settings, topic: str) -> int:
    c = Consumer({"bootstrap.servers": settings.kafka_bootstrap, "group.id": "stunes-size-probe"})
    try:
        parts = c.list_topics(topic, timeout=10).topics[topic].partitions
        return sum(c.get_watermark_offsets(TopicPartition(topic, p), timeout=10)[1] for p in parts)
    finally:
        c.close()


def collect(settings: Settings, run_id: str, expected: Expected) -> dict:
    with psycopg.connect(settings.postgres_dsn) as conn:
        a = conn.execute(
            """
            SELECT count(*) AS attempts,
                   count(DISTINCT event_id) AS distinct_events,
                   count(*) FILTER (WHERE duplicate) AS duplicates,
                   extract(epoch FROM max(processed_at) - min(processed_at)) AS busy_s,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY lat) AS e2e_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY lat) AS e2e_p95,
                   percentile_cont(0.99) WITHIN GROUP (ORDER BY lat) AS e2e_p99
            FROM (SELECT *, extract(epoch FROM processed_at) * 1000 - event_ts AS lat
                  FROM ingest_attempt WHERE run_id = %s) t
            """,
            (run_id,),
        ).fetchone()
        totals = conn.execute(
            "SELECT coalesce(sum(plays),0), coalesce(sum(skips),0), coalesce(sum(likes),0) FROM song_stats"
        ).fetchone()
    attempts, distinct, dups, busy_s, p50, p95, p99 = a
    sent = expected.by_type
    effects = {"play": int(totals[0]), "skip": int(totals[1]), "like": int(totals[2])}
    return {
        "sent_total": expected.total,
        "sent_valid": expected.valid,
        "sent_poison": expected.poison,
        "attempts": attempts,
        "processed": distinct,
        "lost": expected.valid - distinct,
        "duplicates": dups,
        "busy_s": round(float(busy_s or 0), 2),
        "throughput_msg_s": round(distinct / float(busy_s), 1) if busy_s else None,
        "e2e_ms": {
            k: round(float(v), 1) if v is not None else None
            for k, v in (("p50", p50), ("p95", p95), ("p99", p99))
        },
        "effects_match": all(effects[k] == sent.get(k, 0) for k in effects),
        "effects": effects,
        "sent_by_type": dict(sent),
        "dlq": _topic_size(settings, DLQ_TOPIC),
        "retry_published": _topic_size(settings, RETRY_TOPIC),
    }


def _machine() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "note": "Docker Desktop VM; broker/DB in containers, consumers on host",
    }


def _save(settings: Settings, kind: str, run_id: str, config: dict, result: dict) -> Path:
    out = settings.results_dir / "ingest"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.json"
    path.write_bytes(
        orjson.dumps(
            {
                "kind": kind,
                "run_id": run_id,
                "config": config,
                "result": result,
                "machine": _machine(),
                "at": datetime.now(UTC).isoformat(),
            },
            option=orjson.OPT_INDENT_2,
        )
    )
    write_report(settings)
    return path


def _consumer_args(mode: str, run_id: str, group: str, batch: int, workers: int, idle: float):
    return [
        "--mode",
        mode,
        "--run-id",
        run_id,
        "--group",
        group,
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--idle-exit",
        str(idle),
    ]


def throughput(
    settings: Settings,
    *,
    mode: str,
    events: int,
    partitions: int = 1,
    consumers: int = 1,
    batch: int = 500,
    workers: int = 8,
    label: str = "",
) -> dict:
    run_id = f"tp-{mode}-p{partitions}c{consumers}-{int(time.time())}"
    logs = settings.data_dir / "ingest-logs"
    logs.mkdir(parents=True, exist_ok=True)
    reset(settings, partitions)
    users = load_users(settings)
    t = time.monotonic()
    expected = produce(settings, users, run_id, events)
    logger.info("%s: backlog of %d events produced in %.1fs", run_id, events, time.monotonic() - t)

    group = f"stunes-{run_id}"
    procs = [
        _spawn(_consumer_args(mode, run_id, group, batch, workers, 5), logs / f"{run_id}.log")
        for _ in range(consumers)
    ]
    retry, main_done = _retry_consumer(run_id, group, logs, 8)
    for p in procs:
        p.wait()
    main_done.touch()
    _finish(retry)
    result = collect(settings, run_id, expected)
    config = {
        "mode": mode,
        "events": events,
        "partitions": partitions,
        "consumers": consumers,
        "batch": batch,
        "workers": workers,
        "label": label,
    }
    _save(settings, "throughput", run_id, config, result)
    logger.info("%s: %s", run_id, result)
    return result


def chaos(
    settings: Settings,
    *,
    mode: str,
    events: int,
    rate: float,
    kills: int = 5,
    partitions: int = 1,
    batch: int = 500,
    workers: int = 8,
    seed: int = 7,
    label: str = "",
) -> dict:
    run_id = f"chaos-{mode}-{int(time.time())}"
    logs = settings.data_dir / "ingest-logs"
    logs.mkdir(parents=True, exist_ok=True)
    reset(settings, partitions)
    users = load_users(settings)
    group = f"stunes-{run_id}"
    log = logs / f"{run_id}.log"
    duration = events / rate
    rng = random.Random(seed)
    kill_at = sorted(rng.uniform(0.1 * duration, 0.9 * duration) for _ in range(kills))

    result_box: dict = {}
    producer_thread = threading.Thread(
        target=lambda: result_box.setdefault(
            "expected", produce(settings, users, run_id, events, rate=rate, live_ts=True)
        )
    )
    consumer = _spawn(_consumer_args(mode, run_id, group, batch, workers, 15), log)
    retry, main_done = _retry_consumer(run_id, group, logs, 20)
    started = time.monotonic()
    producer_thread.start()
    killed = []
    for at in kill_at:
        time.sleep(max(0.0, at - (time.monotonic() - started)))
        consumer.send_signal(signal.SIGKILL)
        consumer.wait()
        killed.append(round(time.monotonic() - started, 2))
        logger.info("%s: kill -9 at %.1fs", run_id, killed[-1])
        consumer = _spawn(_consumer_args(mode, run_id, group, batch, workers, 15), log)
    producer_thread.join()
    consumer.wait()
    main_done.touch()
    _finish(retry)
    result = collect(settings, run_id, result_box["expected"])
    result["kills_at_s"] = killed
    config = {
        "mode": mode,
        "events": events,
        "rate": rate,
        "kills": kills,
        "partitions": partitions,
        "batch": batch,
        "workers": workers,
        "seed": seed,
        "label": label,
    }
    _save(settings, "chaos", run_id, config, result)
    logger.info("%s: %s", run_id, result)
    return result


def write_report(settings: Settings) -> None:
    out = settings.results_dir / "ingest"
    runs = [orjson.loads(p.read_bytes()) for p in sorted(out.glob("*.json"))]
    lines = [
        "# Phase 2 이벤트 수집 실행",
        "",
        "results/ingest/*.json에서 생성. 모든 행이 같은 머신 → 절대값보다 행끼리 비교",
        "",
        "## 처리량 (미리 쌓아 둔 백로그 비우기)",
        "",
        "| 실행 | 방식 | 파티션 × 소비자 | 배치 | 워커 | 이벤트 | msg/s | 중복 | 유실 | 반영 정상 | DLQ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs:
        if r["kind"] != "throughput":
            continue
        c, x = r["config"], r["result"]
        lines.append(
            f"| {r['run_id']} | {c['mode']} | {c['partitions']} × {c['consumers']} | {c['batch']} | "
            f"{c['workers']} | {x['sent_total']} | {x['throughput_msg_s']} | {x['duplicates']} | "
            f"{x['lost']} | {x['effects_match']} | {x['dlq']} |"
        )
    # repeats (same label/mode/shape): one run per row above can't separate a strategy's cost
    # from run order or machine noise, so the run script repeats them in shuffled order
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        if r["kind"] == "throughput":
            c = r["config"]
            key = (c.get("label") or "-", c["mode"], c["partitions"], c["consumers"], c["batch"])
            groups.setdefault(key, []).append(r["result"])
    lines += [
        "",
        "## 라벨·방식별 처리량 (반복 실행의 평균 ± 표본 표준편차)",
        "",
        "| 라벨 | 방식 | 파티션 × 소비자 | 배치 | 실행 수 | msg/s 평균 | sd | 최소 | 최대 | 유실 (합계) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (label, mode, parts, cons, batch), xs in sorted(groups.items()):
        v = [x["throughput_msg_s"] for x in xs]
        mean = sum(v) / len(v)
        sd = (sum((a - mean) ** 2 for a in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None
        lines.append(
            f"| {label} | {mode} | {parts} × {cons} | {batch} | {len(v)} | {mean:.1f} | "
            f"{f'{sd:.1f}' if sd is not None else '—'} | {min(v):.1f} | {max(v):.1f} | "
            f"{sum(x['lost'] for x in xs)} |"
        )
    lines += [
        "",
        "## 카오스 테스트 (실시간 스트림, 소비자 kill -9)",
        "",
        "| 실행 | 방식 | 생성 속도 msg/s | 강제 종료 횟수 | 이벤트 | 유실 | 중복 | e2e p95 ms | 반영 정상 | DLQ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs:
        if r["kind"] != "chaos":
            continue
        c, x = r["config"], r["result"]
        lines.append(
            f"| {r['run_id']} | {c['mode']} | {c['rate']} | {len(x.get('kills_at_s', []))} | "
            f"{x['sent_total']} | {x['lost']} | {x['duplicates']} | {x['e2e_ms']['p95']} | "
            f"{x['effects_match']} | {x['dlq']} |"
        )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dlq(settings: Settings, replay: bool = False, force: bool = False, limit: int = 10_000) -> dict:
    """Inspect the dead-letter topic; with replay, re-publish its messages to the main topic.

    Replayed messages carry a `replayed-from-dlq` header. The idempotent processor makes a
    replay of something that was in fact applied a no-op, so replaying is always safe; a
    payload that is still invalid simply comes back to the DLQ after its retries.
    """
    from collections import Counter

    from confluent_kafka import Producer

    c = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": "stunes-dlq-replay",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    c.subscribe([DLQ_TOPIC])
    p = (
        Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap,
                "acks": "all",
                "enable.idempotence": True,
            }
        )
        if replay
        else None
    )
    errors: Counter = Counter()
    seen = idle = replayed = skipped = 0
    while seen < limit and idle < 3:
        m = c.poll(1.0)
        if m is None:
            idle += 1
            continue
        if m.error():
            continue
        idle = 0
        seen += 1
        headers = dict(m.headers() or [])
        errors[(headers.get("error", b"").decode()[:80])] += 1
        if p is None:
            continue
        # replaying something that still fails validation only cycles it back through
        # retry -> DLQ; leave it here unless forced
        try:
            Event.model_validate_json(m.value())
        except ValidationError:
            if not force:
                skipped += 1
                continue
        replays = int((headers.get("replays") or b"0").decode()) + 1
        p.produce(TOPIC, key=m.key(), value=m.value(), headers=[("replays", str(replays))])
        p.flush(10)
        c.commit(message=m, asynchronous=False)  # only after the replay is acked
        replayed += 1
    c.close()
    return {
        "messages": seen,
        "replayed": replayed,
        "skipped_still_invalid": skipped,
        "errors": errors.most_common(10),
    }
