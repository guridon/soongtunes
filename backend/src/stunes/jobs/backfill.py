"""Resumable embedding backfill.

- Input is the export snapshot, processed in key order so a checkpoint is just "last key".
- Batches run concurrently, so they finish out of order. The checkpoint only advances to the
  end of the longest *contiguous* run of finished batches (a watermark): committing the
  highest finished key instead would skip still-running batches after a crash.
- Writes are upserts keyed by (version, key), so re-running after a crash is harmless.
"""

import asyncio
import logging
import re
import time
from pathlib import Path

import asyncpg
import numpy as np
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from stunes.config import Settings
from stunes.db import create_pg_pool
from stunes.embedding.registry import EmbeddingVersion, get_version, register
from stunes.embedding.templates import TEMPLATES
from stunes.jobs.index_search import read_jsonl
from stunes.llm import Embedder
from stunes.retrieval.vector import index_expr

logger = logging.getLogger(__name__)

# <= 40 chars so "<entity>_emb_<method>_<version>" stays within PostgreSQL's 63-byte
# identifier limit (a truncated name would defeat the index validity check)
VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,39}$")
TABLES = {"song": ("song_embedding", "song_id"), "tag": ("tag_embedding", "tag")}
# dropped connections (Docker port forwarding) and a server in crash recovery are both
# transient; the upsert is idempotent, so the batch is simply retried
_TRANSIENT = (
    asyncpg.exceptions.ConnectionDoesNotExistError,
    asyncpg.exceptions.CannotConnectNowError,
    asyncpg.exceptions.InterfaceError,
    ConnectionError,
    OSError,
)


def _sort_key(key: str) -> tuple[int, str]:
    return (int(key), key) if key.isdigit() else (1 << 62, key)


class Watermark:
    """Tracks finished batch indexes and exposes the last contiguous one."""

    def __init__(self) -> None:
        self._done: set[int] = set()
        self.next = 0  # first batch index that is not yet known to be finished

    def finish(self, i: int) -> bool:
        self._done.add(i)
        moved = False
        while self.next in self._done:
            self._done.remove(self.next)
            self.next += 1
            moved = True
        return moved


async def _write_batch(
    conn: asyncpg.Connection, version: str, entity: str, keys: list[str], vecs: np.ndarray
) -> None:
    table, key_col = TABLES[entity]
    async with conn.transaction():
        await conn.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS stage_{entity} ({key_col} text, vec vector) "
            "ON COMMIT DELETE ROWS"
        )
        await conn.copy_records_to_table(
            f"stage_{entity}", records=list(zip(keys, vecs, strict=True)), columns=[key_col, "vec"]
        )
        await conn.execute(
            f"INSERT INTO {table} (version, {key_col}, vec) "
            f"SELECT $1, {key_col}, vec FROM stage_{entity} "
            f"ON CONFLICT (version, {key_col}) DO UPDATE SET vec = EXCLUDED.vec",
            version,
        )


# HNSW builds in memory only while the graph fits in maintenance_work_mem; past that it
# spills to a far slower on-disk build. 707,989 x 1024-d float32 (~3 GB of vectors alone)
# stalled at ~224k tuples on this machine (troubleshooting #16), so big versions get IVFFlat.
HNSW_MAX_ROWS = 200_000


async def _build_index(pg: asyncpg.Pool, v: EmbeddingVersion) -> None:
    table, _ = TABLES[v.entity]
    started = time.monotonic()
    valid_sql = "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = $1"
    async with pg.acquire() as conn:
        # the method is re-decided on every full run, so a version that grew past the
        # threshold since its last build moves to IVFFlat
        rows = await conn.fetchval(f"SELECT count(*) FROM {table} WHERE version = $1", v.version)
        ops = f"(({index_expr(v.dim)}) halfvec_cosine_ops)"
        if rows > HNSW_MAX_ROWS:
            # pgvector guidance: lists ~ rows / 1000 up to 1M rows; queries set ivfflat.probes
            name = v.index_name("ivf")
            method = f"ivfflat {ops} WITH (lists = {max(100, rows // 1000)})"
        else:
            name = v.index_name("hnsw")
            method = f"hnsw {ops} WITH (m = 16, ef_construction = 64)"
        # a failed CREATE INDEX CONCURRENTLY leaves an INVALID index behind, and IF NOT
        # EXISTS would then silently skip the rebuild (troubleshooting #15)
        existing = await conn.fetchval(valid_sql, name)
        if existing is False:
            logger.warning("dropping invalid leftover index %s", name)
            await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
        elif existing:
            logger.info("%s already exists and is valid; not rebuilt", name)
        await conn.execute("SET maintenance_work_mem = '1GB'")
        await conn.execute("SET max_parallel_maintenance_workers = 4")
        # version is validated against VERSION_RE, so inlining it into DDL is safe
        await conn.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table} "
            f"USING {method} WHERE version = '{v.version}'"
        )
        if await conn.fetchval(valid_sql, name) is not True:
            raise RuntimeError(f"index {name} is not valid after build")
        # only now drop this version's other vector indexes (the other method, an older
        # expression), so searches never lose their index; left in place, the planner could
        # keep using one and inserts would keep maintaining it
        # (vector access methods only: any other index on this version survives)
        stale = await conn.fetch(
            "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_am am ON am.oid = c.relam "
            "WHERE i.indrelid = $1::regclass AND am.amname IN ('hnsw', 'ivfflat') "
            "AND c.relname <> $2 "
            "AND pg_get_expr(i.indpred, i.indrelid) = format('(version = %L::text)', $3::text)",
            table,
            name,
            v.version,
        )
        for r in stale:
            logger.info("dropping %s, replaced by %s", r["relname"], name)
            await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {r['relname']}")
    logger.info("built %s (%d rows) in %.0fs", name, rows, time.monotonic() - started)


async def reindex(settings: Settings, version: str) -> None:
    """Bring a finished version's vector index to the current layout (status left alone).

    Builds the index the version should have now (method by row count, halfvec expression)
    and drops the ones it replaces; a no-op when that index already exists and is valid.
    """
    pg = await create_pg_pool(settings)
    try:
        v = await get_version(pg, version)
        if v is None or v.status == "building":
            raise RuntimeError(f"{version} is not a finished version; use backfill")
        await _build_index(pg, v)
    finally:
        await pg.close()


async def backfill(
    settings: Settings,
    version: str,
    entity: str,
    export_dir: Path,
    *,
    batch: int = 128,
    concurrency: int = 8,
    limit: int | None = None,
    min_popularity: int = 0,
) -> None:
    if not VERSION_RE.match(version):
        raise ValueError(f"invalid version name {version!r}")
    template = TEMPLATES[entity]
    embedder = Embedder(settings)
    pg = await create_pg_pool(settings)
    try:
        probe = await embedder.embed_documents(["probe"])
        dim = int(probe.shape[1])
        normalized = bool(abs(float(np.linalg.norm(probe[0])) - 1.0) < 1e-3)
        # the corpus is part of the version: a popularity-filtered subset and the full catalog
        # must never share a version name
        spec = template.spec + (
            f"; corpus: playlist_count>={min_popularity}" if min_popularity else ""
        )
        v = await register(
            pg,
            version=version,
            entity=entity,
            model_name=embedder.model,
            dim=dim,
            normalized=normalized,
            template=spec,
        )
        if v.status != "building":
            raise RuntimeError(f"{version} is {v.status}; backfill only writes 'building' versions")

        source = export_dir / ("songs.jsonl" if entity == "song" else "entities_tag.jsonl")
        docs = sorted(
            (d for d in read_jsonl(source) if d.get("playlist_count", 0) >= min_popularity),
            key=lambda d: _sort_key(template.key(d)),
        )
        await pg.execute(
            "INSERT INTO embedding_backfill (version) VALUES ($1) ON CONFLICT DO NOTHING", version
        )
        state = await pg.fetchrow(
            "SELECT last_key, rows_done FROM embedding_backfill WHERE version = $1", version
        )
        if state["last_key"] is not None:
            last = _sort_key(state["last_key"])
            docs = [d for d in docs if _sort_key(template.key(d)) > last]
            logger.info(
                "resuming %s after key %s (%d rows already done)",
                version,
                state["last_key"],
                state["rows_done"],
            )
        if limit is not None:
            docs = docs[:limit]

        batches = [docs[i : i + batch] for i in range(0, len(docs), batch)]
        logger.info(
            "%s: %d rows in %d batches (dim=%d, normalized=%s)",
            version,
            len(docs),
            len(batches),
            dim,
            normalized,
        )
        sem = asyncio.Semaphore(concurrency)
        mark = Watermark()
        rows_done = state["rows_done"]
        started = time.monotonic()
        lock = asyncio.Lock()

        async def run(i: int, items: list[dict]) -> None:
            nonlocal rows_done
            async with sem:
                # sort by length inside the request so the server pads less
                items = sorted(items, key=lambda d: len(template.render(d)))
                vecs = await embedder.embed_documents([template.render(d) for d in items])
                # Docker Desktop drops forwarded connections when containers start; the pool
                # replaces the dead connection and the upsert is safe to repeat
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(_TRANSIENT),
                    stop=stop_after_attempt(10),
                    wait=wait_exponential(multiplier=0.5, max=15),  # ~2 min: covers crash recovery
                    reraise=True,
                ):
                    with attempt:
                        async with pg.acquire() as conn:
                            await _write_batch(
                                conn, version, entity, [template.key(d) for d in items], vecs
                            )
            async with lock:
                rows_done += len(items)
                if mark.finish(i):
                    # batches[k] is in key order; its last element (pre-sort) is its max key
                    last_key = template.key(batches[mark.next - 1][-1])
                    await pg.execute(
                        "UPDATE embedding_backfill SET last_key = $2, rows_done = $3, "
                        "updated_at = now() WHERE version = $1",
                        version,
                        last_key,
                        rows_done,
                    )
                if (rows_done - state["rows_done"]) % (100 * batch) < len(items):
                    rate = (rows_done - state["rows_done"]) / max(time.monotonic() - started, 1e-6)
                    logger.info(
                        "%s: %d rows done, watermark batch %d/%d, %.0f rows/s",
                        version,
                        rows_done,
                        mark.next,
                        len(batches),
                        rate,
                    )

        # TaskGroup cancels the remaining batches on the first failure; the watermark
        # checkpoint makes the next run resume exactly where the contiguous prefix ended.
        async with asyncio.TaskGroup() as tg:
            for i, b in enumerate(batches):
                tg.create_task(run(i, b))
        elapsed = time.monotonic() - started
        logger.info(
            "%s: embedded %d rows in %.0fs (%.0f rows/s)",
            version,
            len(docs),
            elapsed,
            len(docs) / max(elapsed, 1e-6),
        )

        if limit is None:
            await pg.execute(
                "UPDATE embedding_backfill SET finished_at = now() WHERE version = $1", version
            )
            v = await get_version(pg, version)
            assert v is not None
            await _build_index(pg, v)
            await pg.execute(
                "UPDATE embedding_model SET status = 'ready' WHERE version = $1", version
            )
            logger.info(
                "%s is ready; run `stunes activate --version %s` to serve it", version, version
            )
    finally:
        await pg.close()
