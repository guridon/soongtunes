"""Apply the feature tables' committed changes (Debezium, via Redpanda) to Redis.

The consumer's write-through updates Redis right after its own Postgres commit, so the cache
only hears about changes that consumer made. Anything else that writes the feature tables (a
correction in SQL, a seed, a deletion request) leaves Redis serving the old value until the
TTL. Reading the WAL instead, every committed change reaches the cache whoever wrote it, and
the consumer no longer writes two systems (CACHE_WRITER=cdc).

Per change, from the topics cdc.public.feature and cdc.public.song_stats:
    user_recent_plays#u   insert/update -> SET the envelope the readers expect, and DEL
                          user_tag_pref#u (derived from recent plays); delete -> DEL
    other feature keys    any -> DEL (the next read refills from Postgres)
    song_stats            any -> DEL song_stats#id (write-around, as the consumer did)
Every write is guarded by the change's LSN: a key only moves forward, so a replay after a
crash (offsets are committed after Redis has the batch: at-least-once) is a no-op.

Each batch is one Redis pipeline (one round trip). After a kill -9 the restarted applier
waited out the dead member's session (librdkafka's 45 s default): commit-to-cache p50 23-57 s
with three kills. The session is 10 s as in the ingest consumer, and --instance-id (static
membership) gets the partitions back at once. Each change appends "<apply time ms>
<commit-to-cache lag ms>" to --lag-log (source.ts_ms is the transaction's commit time in
Postgres).

Run: python -m stunes.ingest.cdc_cache [--group stunes-cdc-cache] [--idle-exit N]
"""

import argparse
import base64
import json
import logging
import signal
import time
from pathlib import Path

import msgpack
import redis
from confluent_kafka import Consumer, KafkaError, KafkaException

from stunes.config import get_settings
from stunes.features.store import envelope, load_groups

logger = logging.getLogger("stunes.ingest.cdc_cache")

TOPICS = ["cdc.public.feature", "cdc.public.song_stats"]
TTL_S = load_groups()["user_recent_plays"].ttl_sec
LSN_KEY = "cdc_lsn#"

# KEYS[1] value key, KEYS[2] its LSN; ARGV[1] lsn, ARGV[2] value ("" = delete), ARGV[3] ttl
_GUARDED = """
local cur = redis.call('GET', KEYS[2])
if cur and tonumber(cur) >= tonumber(ARGV[1]) then return 0 end
if ARGV[2] == '' then redis.call('DEL', KEYS[1]) else
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3]) end
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[3])
return 1
"""


class Stopper:
    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGTERM, self._set)
        signal.signal(signal.SIGINT, self._set)

    def _set(self, *_):
        self.stop = True


def apply(cache, script, change: dict, topic: str) -> bool:
    """Queues one change's Redis commands on `cache` (a pipeline in run(): one round trip per
    batch). Returns True when the first command queued is the LSN-guarded write, whose result
    (1 applied / 0 skipped) comes back from the pipeline."""
    op, lsn = change.get("op"), change["source"]["lsn"]
    row = change.get("after") or change.get("before") or {}
    if topic.endswith(".song_stats"):
        cache.delete(f"song_stats#{row['song_id']}")
        return False
    pk = row.get("pk")
    if pk is None:
        return False
    if pk.startswith("user_recent_plays#"):
        value = b""
        if op in ("c", "u", "r"):
            plays = msgpack.unpackb(base64.b64decode(change["after"]["value"]))
            value = envelope(plays, TTL_S)
        script(keys=[pk, LSN_KEY + pk], args=[lsn, value, TTL_S], client=cache)
        cache.delete("user_tag_pref#" + pk.split("#", 1)[1])
        return True
    cache.delete(pk)
    return False


def run(
    group: str, idle_exit: float | None, lag_log: Path | None, instance_id: str | None = None
) -> dict:
    settings = get_settings()
    cache = redis.Redis.from_url(settings.redis_url)
    script = cache.register_script(_GUARDED)
    conf = {
        "bootstrap.servers": settings.kafka_bootstrap,
        "group.id": group,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        # as the ingest consumer: a kill -9'd member is evicted after 10 s, not the 45 s default
        "session.timeout.ms": 10000,
    }
    if instance_id:
        # static membership: a restart with the same id gets its partitions back at once
        # instead of waiting for the dead member's session to expire
        conf["group.instance.id"] = instance_id
    consumer = Consumer(conf)
    consumer.subscribe(
        TOPICS, on_assign=lambda c, ps: logger.info("assigned %d partitions", len(ps))
    )
    stopper, applied, skipped, last = Stopper(), 0, 0, time.monotonic()
    lag_f = lag_log.open("a") if lag_log else None
    try:
        while not stopper.stop:
            msgs = consumer.consume(num_messages=500, timeout=0.5)
            if not msgs:
                if idle_exit is not None and time.monotonic() - last > idle_exit:
                    break
                continue
            last = time.monotonic()
            pipe = cache.pipeline(transaction=False)
            guarded, commit_ms = [], []
            for m in msgs:
                if m.error():
                    if m.error().code() != KafkaError._PARTITION_EOF:
                        logger.warning("consume error: %s", m.error())
                    continue
                if m.value() is None:
                    continue
                change = json.loads(m.value())
                if change.get("op") not in ("c", "u", "d", "r"):
                    continue  # truncate / heartbeat-like records
                pos = len(pipe.command_stack)
                if apply(pipe, script, change, m.topic()):
                    guarded.append(pos)
                commit_ms.append(change["source"]["ts_ms"])
            if commit_ms:
                res = pipe.execute()
                now_ms = time.time() * 1000
                n_skip = sum(1 for i in guarded if res[i] == 0)
                skipped += n_skip
                applied += len(commit_ms) - n_skip
                if lag_f:
                    lag_f.write("".join(f"{now_ms:.0f} {now_ms - t:.1f}\n" for t in commit_ms))
                    lag_f.flush()
            # Redis has the batch: now the offsets may move (a crash before this replays it)
            try:
                consumer.commit(asynchronous=False)
            except KafkaException as e:
                if e.args[0].code() != KafkaError._NO_OFFSET:
                    raise
    finally:
        consumer.close()
        if lag_f:
            lag_f.close()
    return {"applied": applied, "skipped_replays": skipped}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="stunes-cdc-cache")
    ap.add_argument("--idle-exit", type=float, default=None, help="exit after N idle seconds")
    ap.add_argument("--lag-log", type=Path, default=None)
    ap.add_argument("--instance-id", default=None, help="group.instance.id (static membership)")
    a = ap.parse_args()
    logger.info("done: %s", run(a.group, a.idle_exit, a.lag_log, a.instance_id))


if __name__ == "__main__":
    main()
