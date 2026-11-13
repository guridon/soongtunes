"""Event consumers, one per commit strategy.

    auto                 enable.auto.commit; offsets stored when a batch is *fetched*   (E1)
    auto_store           auto commit, but offsets stored only after processing          (E1')
    sync                 synchronous commit after every message                         (E2)
    batch                synchronous commit after every fetched batch                   (E3)
    parallel_watermark   per-user worker threads; commit the contiguous done prefix     (E4)
    parallel_max         same workers; commit max(done)+1 (the blog's description)      (E4')

All modes share Processor (one Postgres transaction per event, idempotent), so the
throughput differences come from the commit strategy. "Processed" means the event's
transaction committed; an attempt interrupted by kill -9 rolls back with its bookkeeping
row, so duplicates count *applied* redeliveries, not raw deliveries. Messages that fail validation are
re-published to the retry topic, and only count as done once that publish is confirmed.

Run: python -m stunes.ingest.consumer --mode sync --run-id r1 [--idle-exit 5]
"""

import argparse
import logging
import os
import queue
import signal
import threading
import time
import zlib
from collections import defaultdict
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, Producer, TopicPartition
from pydantic import ValidationError

from stunes import metrics
from stunes.config import Settings, get_settings
from stunes.ingest.events import DLQ_TOPIC, RETRY_TOPIC, TOPIC
from stunes.ingest.processor import Processor

logger = logging.getLogger("stunes.ingest.consumer")


def _processor(settings: Settings, run_id: str) -> Processor:
    return Processor(
        settings.postgres_dsn,
        settings.redis_url,
        run_id,
        write_through=settings.cache_writer == "consumer",
    )


MODES = ("auto", "auto_store", "sync", "batch", "parallel_watermark", "parallel_max")
MAX_ATTEMPTS = 3


def consumer_conf(settings: Settings, group: str, mode: str) -> dict:
    auto = mode in ("auto", "auto_store")
    return {
        "bootstrap.servers": settings.kafka_bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": auto,
        "auto.commit.interval.ms": 5000,
        # E1 stores the offset as soon as a message is handed to the app; E1' stores manually
        "enable.auto.offset.store": mode == "auto",
        "session.timeout.ms": 10000,  # short, so a kill -9'd member is evicted quickly
        "enable.partition.eof": False,
    }


def make_producer(settings: Settings) -> Producer:
    return Producer(
        {"bootstrap.servers": settings.kafka_bootstrap, "acks": "all", "enable.idempotence": True}
    )


class IdleClock:
    """`--idle-exit` that counts only while the consumer holds partitions, and (retry consumer)
    only once the main consumers are done. The clock used to start with the process: in a group
    of 3 the partitions can take longer than the 5 s limit to be handed out (under a 20 ms broker
    delay they did), the members still waiting gave up with 0 messages, and one consumer drained
    all three partitions in a run labelled 3 x 3. The retry consumer likewise gave up between
    two slow poison events and left the rest undelivered (troubleshooting #25)."""

    def __init__(
        self, idle_s: float | None, after: Path | None = None, unassigned_s: float = 120.0
    ) -> None:
        self.idle_s, self.after, self.unassigned_s = idle_s, after, unassigned_s
        self.last = self.assigned_at = time.monotonic()

    def busy(self) -> None:
        self.last = time.monotonic()

    def expired(self, consumer: Consumer) -> bool:
        if not self.idle_s:
            return False
        now = time.monotonic()
        if self.after is not None and not self.after.exists():
            # others can still feed this topic: neither clock runs yet (the
            # unassigned limit kept counting and could expire the moment the flag appeared)
            self.last = self.assigned_at = now
            return False
        if not consumer.assignment():
            self.last = now  # the idle clock runs only while partitions are held
            return now - self.assigned_at > self.unassigned_s
        self.assigned_at = now
        return now - self.last > self.idle_s


class Stopper:
    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGTERM, self._set)
        signal.signal(signal.SIGINT, self._set)

    def _set(self, *_):
        self.stop = True


def _to_retry(producer: Producer, msg: Message, error: str, attempt: int) -> None:
    """Publish to the retry (or DLQ) topic and wait for the ack before calling it done."""
    topic = RETRY_TOPIC if attempt <= MAX_ATTEMPTS else DLQ_TOPIC
    metrics.INGEST_EVENTS.labels("retry" if topic == RETRY_TOPIC else "dlq").inc()
    producer.produce(
        topic,
        key=msg.key(),
        value=msg.value(),
        headers=[("attempt", str(attempt)), ("error", error[:300])],
    )
    if producer.flush(10) > 0:
        raise RuntimeError(f"could not publish to {topic}")


# ---------------------------------------------------------------------- sequential modes
def run_sequential(
    settings: Settings, mode: str, run_id: str, group: str, batch: int, idle_exit: float | None
) -> int:
    consumer = Consumer(consumer_conf(settings, group, mode))
    consumer.subscribe([TOPIC])
    proc = _processor(settings, run_id)
    producer = make_producer(settings)
    stopper, handled, idle = Stopper(), 0, IdleClock(idle_exit)
    try:
        while not stopper.stop:
            msgs = consumer.consume(num_messages=batch, timeout=1.0)
            if not msgs:
                if idle.expired(consumer):
                    break
                continue
            idle.busy()
            batch_next: dict[int, int] = {}
            for m in msgs:
                if m.error():
                    if m.error().code() != KafkaError._PARTITION_EOF:
                        logger.warning("consume error: %s", m.error())
                    continue
                try:
                    proc.handle(m.value(), m.partition(), m.offset())
                except Exception as e:  # invalid payload or failing dependency
                    if not isinstance(e, ValidationError):
                        logger.warning(
                            "processing failed at %s/%s: %r", m.partition(), m.offset(), e
                        )
                        proc.close()
                        proc = _processor(settings, run_id)
                    _to_retry(producer, m, str(e), attempt=1)
                handled += 1
                if mode == "auto_store":
                    consumer.store_offsets(message=m)
                elif mode == "sync":
                    consumer.commit(message=m, asynchronous=False)  # commits offset+1
                batch_next[m.partition()] = m.offset() + 1
            if mode == "batch" and batch_next:
                # explicit offsets: with enable.auto.offset.store=false a bare commit() would
                # commit the (never updated) stored offsets, i.e. nothing
                consumer.commit(
                    offsets=[TopicPartition(TOPIC, p, o) for p, o in batch_next.items()],
                    asynchronous=False,
                )
    finally:
        consumer.close()  # auto modes commit stored offsets here on a graceful stop
        proc.close()
    return handled


# ---------------------------------------------------------------------- parallel modes
class OffsetTracker:
    """Per-partition bookkeeping for out-of-order completion.

    watermark: commit the lowest offset still in flight (everything before it is done)
    max:       commit max(done)+1 — skips anything still in flight if the process dies
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.inflight: dict[int, set[int]] = defaultdict(set)
        self.dispatched_max: dict[int, int] = {}
        self.done_max: dict[int, int] = {}

    def dispatch(self, partition: int, offset: int) -> None:
        self.inflight[partition].add(offset)
        self.dispatched_max[partition] = max(offset, self.dispatched_max.get(partition, -1))

    def complete(self, partition: int, offset: int) -> None:
        self.inflight[partition].discard(offset)
        self.done_max[partition] = max(offset, self.done_max.get(partition, -1))

    def commit_offsets(self, strategy: str | None = None) -> dict[int, int]:
        out = {}
        for p in self.dispatched_max:
            if (strategy or self.strategy) == "watermark":
                out[p] = min(self.inflight[p]) if self.inflight[p] else self.dispatched_max[p] + 1
            elif p in self.done_max:
                out[p] = self.done_max[p] + 1
        return out

    def forget(self, partitions: list[int]) -> None:
        for p in partitions:
            self.inflight.pop(p, None)
            self.dispatched_max.pop(p, None)
            self.done_max.pop(p, None)

    @property
    def total_inflight(self) -> int:
        return sum(len(v) for v in self.inflight.values())


def run_parallel(
    settings: Settings,
    mode: str,
    run_id: str,
    group: str,
    batch: int,
    workers: int,
    idle_exit: float | None,
    max_inflight: int = 2000,
    commit_every_s: float = 0.2,
) -> int:
    tracker = OffsetTracker("watermark" if mode == "parallel_watermark" else "max")
    queues: list[queue.Queue] = [queue.Queue() for _ in range(workers)]
    done: queue.Queue = queue.Queue()
    producer = make_producer(settings)
    lock = threading.Lock()  # serializes produce+flush across workers
    fatal = threading.Event()

    def worker(q: queue.Queue) -> None:
        proc = _processor(settings, run_id)
        try:
            while (m := q.get()) is not None:
                if fatal.is_set():
                    continue  # stop doing work; the message stays uncommitted
                try:
                    try:
                        proc.handle(m.value(), m.partition(), m.offset())
                    except Exception as e:  # bad payload or a failing dependency: retry topic
                        if not isinstance(e, ValidationError):
                            logger.warning(
                                "processing failed at %s/%s: %r", m.partition(), m.offset(), e
                            )
                            proc.close()
                            proc = _processor(settings, run_id)
                        with lock:
                            _to_retry(producer, m, str(e), attempt=1)
                except Exception:
                    # couldn't even hand it to the retry topic: never mark it done, so the
                    # watermark can't pass it; stop the consumer instead
                    logger.exception("giving up on %s/%s", m.partition(), m.offset())
                    fatal.set()
                    continue
                done.put((m.partition(), m.offset()))
        finally:
            proc.close()

    threads = [threading.Thread(target=worker, args=(q,), daemon=True) for q in queues]
    for t in threads:
        t.start()

    consumer: Consumer
    committed: dict[int, int] = {}

    def drain_done(block_s: float = 0.0) -> int:
        n = 0
        deadline = time.monotonic() + block_s
        while True:
            try:
                p, o = (
                    done.get(timeout=max(0.0, deadline - time.monotonic()))
                    if block_s
                    else done.get_nowait()
                )
            except queue.Empty:
                return n
            tracker.complete(p, o)
            n += 1

    def commit(strategy: str | None = None) -> None:
        offsets = tracker.commit_offsets(strategy)
        changed = {p: o for p, o in offsets.items() if committed.get(p) != o}
        if not changed:
            return
        try:
            consumer.commit(
                offsets=[TopicPartition(TOPIC, p, o) for p, o in changed.items()],
                asynchronous=False,
            )
            committed.update(changed)
        except KafkaException as e:
            # e.g. the partition was already reassigned; the next owner redelivers from the
            # last successful commit, and the idempotent processor absorbs the duplicates
            logger.warning("commit failed: %s", e)

    def on_revoke(_consumer, partitions):
        # finish what we dispatched for the revoked partitions, commit, then let them go;
        # otherwise the next owner reprocesses everything since our last commit. Bounded:
        # blocking past the session timeout gets us evicted anyway.
        revoked = {tp.partition for tp in partitions}
        deadline = time.monotonic() + 8.0
        while any(tracker.inflight.get(p) for p in revoked) and time.monotonic() < deadline:
            drain_done(block_s=0.05)
        commit("watermark")  # never skip unfinished offsets on a handover
        tracker.forget(list(revoked))
        for p in revoked:
            committed.pop(p, None)

    def on_lost(_consumer, partitions):
        # already owned by someone else: committing would fail or clobber their progress
        tracker.forget([tp.partition for tp in partitions])
        for tp in partitions:
            committed.pop(tp.partition, None)

    consumer = Consumer(consumer_conf(settings, group, mode))
    consumer.subscribe([TOPIC], on_revoke=on_revoke, on_lost=on_lost)
    stopper, handled, idle, last_commit = Stopper(), 0, IdleClock(idle_exit), 0.0
    try:
        while not stopper.stop and not fatal.is_set():
            room = max_inflight - tracker.total_inflight
            msgs = consumer.consume(num_messages=min(batch, room), timeout=0.2) if room > 0 else []
            for m in msgs:
                if m.error():
                    continue
                tracker.dispatch(m.partition(), m.offset())
                # same user -> same worker: per-user order is preserved
                queues[zlib.crc32(m.key() or b"") % workers].put(m)
                handled += 1
            handled_now = drain_done(block_s=0.0 if msgs else 0.05)
            if msgs or handled_now:
                idle.busy()
            if time.monotonic() - last_commit >= commit_every_s:
                commit()
                last_commit = time.monotonic()
            if not msgs and tracker.total_inflight == 0 and idle.expired(consumer):
                break
        # graceful stop: let workers finish, commit everything that's done
        for q in queues:
            q.put(None)
        for t in threads:
            t.join()  # queues are bounded by max_inflight, so this terminates
        drain_done()
        # the final commit always uses the watermark: in "max" mode a fatal stop could
        # otherwise commit past a message no worker finished
        commit("watermark")
    finally:
        consumer.close()
    return handled


# ---------------------------------------------------------------------- retry consumer
def run_retry(
    settings: Settings,
    run_id: str,
    idle_exit: float | None,
    group: str = "stunes-retry",
    idle_after: Path | None = None,
) -> int:
    """Retry topic: try again with backoff, up to MAX_ATTEMPTS, then dead-letter. With
    `idle_after`, idling counts only once that file exists (the main consumers are done)."""
    consumer = Consumer(consumer_conf(settings, group, "sync"))
    consumer.subscribe([RETRY_TOPIC])
    proc = _processor(settings, run_id)
    producer = make_producer(settings)
    stopper, handled, idle = Stopper(), 0, IdleClock(idle_exit, after=idle_after)
    try:
        while not stopper.stop:
            m = consumer.poll(1.0)
            if m is None:
                if idle.expired(consumer):
                    break
                continue
            if m.error():
                continue
            idle.busy()
            raw_attempt = dict(m.headers() or []).get("attempt") or b"1"
            attempt = int(raw_attempt.decode() if isinstance(raw_attempt, bytes) else raw_attempt)
            try:
                proc.handle(m.value(), m.partition(), m.offset())
            except ValidationError as e:
                time.sleep(min(0.1 * 2**attempt, 2.0))
                _to_retry(producer, m, str(e), attempt=attempt + 1)
            consumer.commit(message=m, asynchronous=False)
            handled += 1
    finally:
        consumer.close()
        proc.close()
    return handled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=[*MODES, "retry"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--group", default=None)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--idle-exit", type=float, default=None, help="exit after N idle seconds")
    ap.add_argument(
        "--idle-after",
        type=Path,
        default=None,
        help="retry mode: start counting idle time only once this file exists",
    )
    ap.add_argument("--metrics-port", type=int, default=None, help="serve Prometheus metrics")
    args = ap.parse_args()
    if args.metrics_port:
        from prometheus_client import start_http_server

        start_http_server(args.metrics_port)
    settings = get_settings()
    group = args.group or f"stunes-{args.mode}"
    started = time.monotonic()
    if args.mode == "retry":
        n = run_retry(
            settings, args.run_id, args.idle_exit, args.group or "stunes-retry", args.idle_after
        )
    elif args.mode.startswith("parallel"):
        n = run_parallel(
            settings, args.mode, args.run_id, group, args.batch, args.workers, args.idle_exit
        )
    else:
        n = run_sequential(settings, args.mode, args.run_id, group, args.batch, args.idle_exit)
    logger.info(
        "pid %d mode=%s handled %d messages in %.1fs",
        os.getpid(),
        args.mode,
        n,
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main()
