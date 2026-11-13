"""Phase 4 experiment: embedding size vs. storage/latency vs. recall.

On a subset (the N most-played songs of one version) it compares, against exact float32
search (sequential scan, the ground truth):

    float32  HNSW on vector(1024)                    4096 B/vector
    halfvec  HNSW on halfvec(1024)                   2048 B/vector
    binary   HNSW on bit(1024) (hamming), no rerank   128 B/vector
    binary+rerank  top-200 by hamming, re-scored with float32
    dim256   HNSW on the first 256 dims, re-normalized  1024 B/vector
             (only meaningful for Matryoshka-trained models; Harrier isn't one — measured anyway)

pgvector has no int8 vector type, so halfvec and bit stand in for the spec's int8 row.
Queries are the eval set's utterances, embedded with the query instruction.
"""

import asyncio
import csv
import logging
import sys
import time
from pathlib import Path

import asyncpg
import numpy as np
import orjson
import redis.asyncio as aioredis

from stunes.config import Settings
from stunes.db import create_pg_pool
from stunes.llm import Embedder
from stunes.retrieval.vector import search_sql

logger = logging.getLogger(__name__)

VARIANTS = {
    "float32": ("hnsw (vec vector_cosine_ops)", "vec <=> $1::vector(1024)", 4096),
    "halfvec": ("hnsw (h halfvec_cosine_ops)", "h <=> $1::vector(1024)::halfvec(1024)", 2048),
    "binary": (
        "hnsw (b bit_hamming_ops)",
        "b <~> binary_quantize($1::vector(1024))::bit(1024)",
        128,
    ),
    "dim256": (
        "hnsw (s256 vector_cosine_ops)",
        "s256 <=> l2_normalize(subvector($1::vector(1024), 1, 256))::vector(256)",
        1024,
    ),
}


async def _build_table(pg: asyncpg.Pool, version: str, song_ids: list[str]) -> None:
    async with pg.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS emb_exp")
        await conn.execute(
            """
            CREATE TABLE emb_exp AS
            SELECT song_id,
                   vec::vector(1024) AS vec,
                   vec::vector(1024)::halfvec(1024) AS h,
                   binary_quantize(vec::vector(1024))::bit(1024) AS b,
                   l2_normalize(subvector(vec::vector(1024), 1, 256))::vector(256) AS s256
            FROM song_embedding WHERE version = $1 AND song_id = ANY($2)
            """,
            version,
            song_ids,
        )
        await conn.execute("ALTER TABLE emb_exp ADD PRIMARY KEY (song_id)")


async def _exact(conn: asyncpg.Connection, q: np.ndarray, k: int) -> list[str]:
    # ground truth: no index, full float32 scan
    async with conn.transaction():
        await conn.execute("SET LOCAL enable_indexscan = off")
        rows = await conn.fetch(
            "SELECT song_id FROM emb_exp ORDER BY vec <=> $1::vector(1024) LIMIT $2", q, k
        )
    return [r["song_id"] for r in rows]


async def _ann(conn: asyncpg.Connection, expr: str, q: np.ndarray, k: int, ef: int) -> list[str]:
    async with conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {ef}")
        rows = await conn.fetch(f"SELECT song_id FROM emb_exp ORDER BY {expr} LIMIT $2", q, k)
    return [r["song_id"] for r in rows]


async def _binary_rerank(conn: asyncpg.Connection, q: np.ndarray, k: int, pool: int) -> list[str]:
    async with conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {max(pool, 40)}")
        rows = await conn.fetch(
            f"""
            WITH cand AS (
              SELECT song_id, vec FROM emb_exp
              ORDER BY {VARIANTS["binary"][1]} LIMIT {pool}
            )
            SELECT song_id FROM cand ORDER BY vec <=> $1::vector(1024) LIMIT $2
            """,
            q,
            k,
        )
    return [r["song_id"] for r in rows]


async def _exact_whole_version(
    conn: asyncpg.Connection,
    version: str,
    qv: np.ndarray,
    k: int,
    chunk: int = 20_000,
    ordered: bool = False,
) -> list:
    """Exact top-k of every query over the whole version, all queries in one pass: sets, or
    with `ordered` lists best first.

    An exact scan per query would read all 707,989 float32 vectors (2.9 GB) every time.
    """
    q = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    ids: list[str] = []
    top_s = np.full((len(q), k), -np.inf, dtype=np.float32)
    top_i = np.zeros((len(q), k), dtype=np.int64)

    def merge(vecs: list[np.ndarray]) -> None:
        nonlocal top_s, top_i
        m = np.stack(vecs).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        sims = q @ m.T
        base = len(ids) - len(vecs)
        cand_s = np.concatenate([top_s, sims], axis=1)
        cand_i = np.concatenate(
            [top_i, np.broadcast_to(base + np.arange(len(vecs)), sims.shape)], axis=1
        )
        sel = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(cand_s, sel, axis=1)
        top_i = np.take_along_axis(cand_i, sel, axis=1)

    vecs: list[np.ndarray] = []
    async with conn.transaction():
        cur = conn.cursor(
            "SELECT song_id, vec FROM song_embedding WHERE version = $1", version, prefetch=chunk
        )
        async for r in cur:
            ids.append(r["song_id"])
            vecs.append(r["vec"].to_numpy())
            if len(vecs) == chunk:
                merge(vecs)
                vecs = []
    if vecs:
        merge(vecs)
    if ordered:
        order = np.argsort(-top_s, axis=1)
        return [[ids[i] for i in row] for row in np.take_along_axis(top_i, order, axis=1)]
    return [{ids[i] for i in row} for row in top_i]


async def _served_index(
    conn: asyncpg.Connection, version: str, qv: np.ndarray, k: int, probes=(5, 10, 20, 30)
) -> dict:
    """Recall and latency of the index the whole version is actually served from.

    The full catalog is past HNSW_MAX_ROWS and gets IVFFlat (troubleshooting #16, #17), which
    the subset variants above can't show. The query is VectorIndex.search's own SQL.
    """
    sql = search_sql("song_embedding", "song_id", 1024)
    plan = "\n".join(r[0] for r in await conn.fetch(f"EXPLAIN {sql}", qv[0], version, k, 2 * k))
    index = next((w for w in plan.split() if w.startswith("song_emb_")), None)
    t = time.perf_counter()
    truth = await _exact_whole_version(conn, version, qv, k)
    out: dict = {
        "rows": await conn.fetchval(
            "SELECT count(*) FROM song_embedding WHERE version = $1", version
        ),
        "queries": len(qv),
        "plan_index": index,
        "index_mb": round(
            (await conn.fetchval("SELECT pg_relation_size($1::regclass)", index) if index else 0)
            / 2**20
        ),
        "exact_one_pass_s": round(time.perf_counter() - t, 1),
        "probes": {},
    }

    async def search(q: np.ndarray, p: int) -> set[str]:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL ivfflat.probes = {p}")
            return {r["song_id"] for r in await conn.fetch(sql, q, version, k, 2 * k)}

    # one untimed pass at the largest setting, so every row is measured as warm as the page
    # cache allows (the steady state of a running service)
    for q in qv:
        await search(q, max(probes))
    for p in probes:
        recall, ms = [], []
        for q, gold in zip(qv, truth, strict=True):
            t = time.perf_counter()
            got = await search(q, p)
            ms.append((time.perf_counter() - t) * 1000)
            recall.append(len(gold & got) / k)
        out["probes"][str(p)] = {
            "recall_at_k": round(float(np.mean(recall)), 4),
            "p50_ms": round(float(np.percentile(ms, 50)), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2),
        }
    return out


async def attribution(
    settings: Settings, *, versions: list[str], dataset: Path, k: int = 10, pool: int = 50
) -> dict:
    """Where vector-only quality moves between two versions: the catalog or the index.

    Every eval turn's raw message (what the `vector` strategy embeds) is searched four ways,
    exact and through the served index, for each version, and scored against the eval labels
    at k. exact(B) - exact(A) is what the bigger catalog does; served - exact within a version
    is what its ANN index loses. Served uses VectorIndex.search's SQL and settings.
    """
    from stunes.eval.metrics import mean, score_turn
    from stunes.eval.run import load_dataset
    from stunes.retrieval.vector import IVFFLAT_PROBES, SEARCH_TIMEOUT_MS

    turns = [t for conv in load_dataset(dataset) for t in conv]
    embedder = Embedder(settings)
    qv = np.concatenate(
        [
            await embedder.embed_queries([t.text for t in turns[i : i + 64]])
            for i in range(0, len(turns), 64)
        ]
    ).astype(np.float32)
    sql = search_sql("song_embedding", "song_id", qv.shape[1])
    out: dict = {"turns": len(turns), "k": k, "pool": pool, "versions": {}}
    pg = await create_pg_pool(settings)
    try:
        for version in versions:
            async with pg.acquire() as conn:
                t = time.perf_counter()
                exact = await _exact_whole_version(conn, version, qv, pool, ordered=True)
                exact_s = time.perf_counter() - t
                served, ms, timeouts = [], [], 0
                for q in qv:
                    t = time.perf_counter()
                    try:  # production's limits; a timed-out search is a dropped source there
                        async with conn.transaction():
                            ef = max(40, min(pool * 2, 400))
                            await conn.execute(f"SET LOCAL hnsw.ef_search = {ef}")
                            await conn.execute(f"SET LOCAL ivfflat.probes = {IVFFLAT_PROBES}")
                            await conn.execute(f"SET LOCAL statement_timeout = {SEARCH_TIMEOUT_MS}")
                            rows = await conn.fetch(sql, q, version, pool, 2 * pool)
                        served.append([r["song_id"] for r in rows])
                    except asyncpg.QueryCanceledError:
                        served.append([])
                        timeouts += 1
                    ms.append((time.perf_counter() - t) * 1000)
            v: dict = {
                "exact_pass_s": round(exact_s, 1),
                "served_p50_ms": round(float(np.percentile(ms, 50)), 1),
                "served_timeouts": timeouts,
                f"served_recall_at_{pool}": round(
                    float(
                        np.mean(
                            [
                                len(set(a) & set(b)) / pool
                                for a, b in zip(exact, served, strict=True)
                            ]
                        )
                    ),
                    4,
                ),
            }
            for name, ranked in (("exact", exact), ("served", served)):
                by: dict[str, list] = {}
                for r, t in zip(ranked, turns, strict=True):
                    m = score_turn(r[:k], t.relevant, k)
                    by.setdefault("ALL", []).append(m)
                    by.setdefault(t.query_type, []).append(m)
                v[name] = {
                    qt: {"n": len(rows), **{x: round(y, 4) for x, y in mean(rows).items()}}
                    for qt, rows in sorted(by.items())
                }
            out["versions"][version] = v
            logger.info("%s: %s", version, {n: v[n]["ALL"]["mrr"] for n in ("exact", "served")})
    finally:
        await pg.close()
    return out


def _queries(dataset: Path, n: int) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    with dataset.open(encoding="utf-8") as f:
        texts = [row["input"] for row in csv.DictReader(f)]
    rng = np.random.default_rng(0)
    return [texts[i] for i in rng.choice(len(texts), size=min(n, len(texts)), replace=False)]


async def _redis_bytes(settings: Settings, vecs: np.ndarray) -> dict[str, float]:
    """Memory per cached vector in Redis for each format (MEMORY USAGE, includes overhead)."""
    r = aioredis.Redis.from_url(settings.redis_url)
    formats = {
        "float32": lambda v: v.astype(np.float32).tobytes(),
        "halfvec": lambda v: v.astype(np.float16).tobytes(),
        "binary": lambda v: np.packbits(v > 0).tobytes(),
        "dim256": lambda v: (v[:256] / np.linalg.norm(v[:256])).astype(np.float32).tobytes(),
    }
    out = {}
    try:
        for name, enc in formats.items():
            keys = [f"embexp:{name}:{i}" for i in range(len(vecs))]
            pipe = r.pipeline(transaction=False)
            for k, v in zip(keys, vecs, strict=True):
                pipe.set(k, enc(v))
            await pipe.execute()
            usage = [await r.memory_usage(k) for k in keys[:200]]
            out[name] = round(float(np.mean(usage)), 1)
            await r.delete(*keys)
    finally:
        await r.aclose()
    return out


async def run(
    settings: Settings,
    *,
    version: str,
    dataset: Path,
    subset: int = 100_000,
    n_queries: int = 200,
    k: int = 50,
    ef: int = 100,
) -> dict:
    export = settings.data_dir / "exports" / "songs.jsonl"
    docs = [orjson.loads(line) for line in export.read_bytes().splitlines()]
    docs.sort(key=lambda d: -d["playlist_count"])
    song_ids = [d["song_id"] for d in docs[:subset]]

    pg = await create_pg_pool(settings)
    embedder = Embedder(settings)
    try:
        t = time.perf_counter()
        await _build_table(pg, version, song_ids)
        n_rows = await pg.fetchval("SELECT count(*) FROM emb_exp")
        logger.info("emb_exp: %d vectors (%.0fs)", n_rows, time.perf_counter() - t)

        builds = {}
        async with pg.acquire() as conn:
            await conn.execute("SET maintenance_work_mem = '1GB'")
            await conn.execute("SET max_parallel_maintenance_workers = 4")
            for name, (index_def, _, _) in VARIANTS.items():
                t = time.perf_counter()
                await conn.execute(f"CREATE INDEX emb_exp_{name} ON emb_exp USING {index_def}")
                size = await conn.fetchval(f"SELECT pg_relation_size('emb_exp_{name}')")
                builds[name] = {
                    "build_s": round(time.perf_counter() - t, 1),
                    "index_mb": round(size / 2**20, 1),
                }
                logger.info("index %s: %s", name, builds[name])
            await conn.execute("ANALYZE emb_exp")

        queries = _queries(dataset, n_queries)
        qv = await embedder.embed_queries(queries)

        results: dict[str, dict] = {
            v: {"recall": [], "ms": []} for v in [*VARIANTS, "binary+rerank"]
        }
        exact_ms = []
        async with pg.acquire() as conn:
            for q in qv:
                t = time.perf_counter()
                truth = set(await _exact(conn, q, k))
                exact_ms.append((time.perf_counter() - t) * 1000)
                for name, (_, expr, _) in VARIANTS.items():
                    t = time.perf_counter()
                    got = await _ann(conn, expr, q, k, ef)
                    results[name]["ms"].append((time.perf_counter() - t) * 1000)
                    results[name]["recall"].append(len(truth & set(got)) / k)
                t = time.perf_counter()
                got = await _binary_rerank(conn, q, k, pool=200)
                results["binary+rerank"]["ms"].append((time.perf_counter() - t) * 1000)
                results["binary+rerank"]["recall"].append(len(truth & set(got)) / k)

            served = await _served_index(conn, version, qv[:100], k)
            logger.info("served index: %s", served)

        sample = np.stack(
            [
                np.asarray(r["vec"].to_numpy(), dtype=np.float32)
                for r in await pg.fetch("SELECT vec FROM emb_exp LIMIT 2000")
            ]
        )
        redis_bytes = await _redis_bytes(settings, sample)
    finally:
        await pg.close()

    rows = {}
    for name, r in results.items():
        base = name.split("+")[0]
        rows[name] = {
            "recall_at_k": round(float(np.mean(r["recall"])), 4),
            "p50_ms": round(float(np.percentile(r["ms"], 50)), 2),
            "p95_ms": round(float(np.percentile(r["ms"], 95)), 2),
            "bytes_per_vector": VARIANTS[base][2],
            "redis_bytes_per_key": redis_bytes.get(base),
            **builds.get(base, {}),
        }
    return {
        "version": version,
        "subset": int(n_rows),
        "queries": len(queries),
        "k": k,
        "ef_search": ef,
        "exact_p50_ms": round(float(np.percentile(exact_ms, 50)), 2),
        "exact_p95_ms": round(float(np.percentile(exact_ms, 95)), 2),
        "variants": rows,
        "served_index": served,
    }


def save(settings: Settings, result: dict) -> Path:
    out = settings.results_dir / "embedding"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"size-vs-recall-{int(time.time())}.json"
    path.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    lines = [
        f"# 임베딩 크기와 재현율 ({result['version']}, {result['subset']}곡, "
        f"질의 {result['queries']}개, k={result['k']}, ef_search={result['ef_search']})",
        "",
        f"정확한 float32 전체 스캔 (정답 기준): p50 {result['exact_p50_ms']} ms, p95 {result['exact_p95_ms']} ms",
        "",
        "| 변형 | recall@k | p50 ms | p95 ms | 벡터당 bytes | Redis 키당 bytes | 색인 MB | 생성 s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in result["variants"].items():
        lines.append(
            f"| {name} | {r['recall_at_k']:.3f} | {r['p50_ms']} | {r['p95_ms']} | "
            f"{r['bytes_per_vector']} | {r['redis_bytes_per_key']} | {r.get('index_mb', '—')} | "
            f"{r.get('build_s', '—')} |"
        )
    served = result.get("served_index")
    if served:
        lines += [
            "",
            f"## 전체 버전에 건 서비스 경로 색인 ({served['rows']}곡, 색인 "
            f"`{served['plan_index']}` {served['index_mb']:.0f} MB, 질의 {served['queries']}개, "
            "예열 상태)",
            "",
            "| ivfflat.probes | recall@k | p50 ms | p95 ms |",
            "| --- | --- | --- | --- |",
        ]
        for p, r in served["probes"].items():
            lines.append(f"| {p} | {r['recall_at_k']:.3f} | {r['p50_ms']} | {r['p95_ms']} |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(settings: Settings, version: str, dataset: Path, subset: int, queries: int) -> None:
    result = asyncio.run(
        run(settings, version=version, dataset=dataset, subset=subset, n_queries=queries)
    )
    print(save(settings, result))
    print(orjson.dumps(result["variants"], option=orjson.OPT_INDENT_2).decode())
