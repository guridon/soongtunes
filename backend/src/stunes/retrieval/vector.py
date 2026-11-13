"""Dense retrieval over pgvector, always against the *active* embedding version."""

import time

import asyncpg
import numpy as np

from stunes.embedding.registry import EmbeddingVersion, get_active

_TABLES = {"song": ("song_embedding", "song_id"), "tag": ("tag_embedding", "tag")}
# measured on the whole 707,989-song version, 100 eval queries, recall@50 vs exact (warm):
# 30 -> 0.732 (p95 786 ms), 45 -> 0.802 (813 ms), 60 -> 0.845 (1,283 ms). 45 matches what
# float32 HNSW gave on the 100k subset (0.790) and keeps p95 well inside the source timeout.
IVFFLAT_PROBES = 45
# the agent gives the vector source 2 s (agent/graph.py SOURCE_TIMEOUT_S); without a server
# timeout, searches it had given up on kept running, piled up and starved every other
# source (troubleshooting #18)
SEARCH_TIMEOUT_MS = 2000


def index_expr(dim: int) -> str:
    """What every version's partial index is built over (jobs/backfill.py).

    Half precision: a 1024-d float32 vector is 4 KB, so IVFFlat fits one per 8 KB page and the
    full catalog's index was 5.5 GB, more than this machine could keep cached (troubleshooting
    #17). halfvec fits three per page. Similarities are still computed from the float32 column.
    """
    return f"vec::halfvec({dim})"


def search_sql(table: str, key: str, dim: int) -> str:
    """$1 query, $2 version, $3 k, $4 candidates (> k).

    The index orders by rounded halfvec distances, which can swap near-ties at the k-th place,
    so it only picks $4 candidates and the float32 vectors decide the final k.
    The inner ORDER BY must match the partial index exactly (`index_expr` + `version = ...`)
    for the planner to use it.
    """
    return (
        f"SELECT {key}, sim FROM ("
        f"SELECT {key}, 1 - (vec::vector({dim}) <=> $1) AS sim FROM {table} "
        f"WHERE version = $2 ORDER BY {index_expr(dim)} <=> $1::vector({dim})::halfvec({dim}) "
        "LIMIT $4) c ORDER BY sim DESC LIMIT $3"
    )


class VectorIndex:
    def __init__(
        self, pg: asyncpg.Pool, cache_ttl_s: float = 30.0, timeout_ms: float = SEARCH_TIMEOUT_MS
    ):
        self._timeout_ms = int(timeout_ms)
        self._pg = pg
        self._ttl = cache_ttl_s
        self._active: dict[str, tuple[float, EmbeddingVersion | None]] = {}

    async def active(self, entity: str) -> EmbeddingVersion | None:
        """Active version, cached briefly so a pointer flip propagates within the TTL."""
        hit = self._active.get(entity)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        version = await get_active(self._pg, entity)
        self._active[entity] = (time.monotonic(), version)
        return version

    async def search(self, entity: str, query: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        version = await self.active(entity)
        if version is None:
            return []
        table, key = _TABLES[entity]
        dim = int(version.dim)
        if query.shape[-1] != dim:
            raise ValueError(f"query dim {query.shape[-1]} != {version.version} dim {dim}")
        async with self._pg.acquire() as conn, conn.transaction():
            # whichever index the version has (HNSW for small versions, IVFFlat for big ones)
            await conn.execute(f"SET LOCAL hnsw.ef_search = {max(40, min(k * 2, 400))}")
            await conn.execute(f"SET LOCAL ivfflat.probes = {IVFFLAT_PROBES}")
            await conn.execute(f"SET LOCAL statement_timeout = {self._timeout_ms}")
            rows = await conn.fetch(
                search_sql(table, key, dim), query.astype(np.float32), version.version, k, 2 * k
            )
        return [(r[key], float(r["sim"])) for r in rows]

    async def fetch(self, entity: str, keys: list[str]) -> dict[str, np.ndarray]:
        """Stored vectors of the active version for these keys (e.g. a user's recent plays)."""
        version = await self.active(entity)
        if version is None or not keys:
            return {}
        table, key = _TABLES[entity]
        rows = await self._pg.fetch(
            f"SELECT {key}, vec FROM {table} WHERE version = $1 AND {key} = ANY($2)",
            version.version,
            keys,
        )
        return {r[key]: np.asarray(r["vec"].to_numpy(), dtype=np.float32) for r in rows}

    async def rank_among(
        self, entity: str, query: np.ndarray, keys: list[str]
    ) -> list[tuple[str, float]]:
        """Exact cosine ranking restricted to `keys` (e.g. the previous songs' tags)."""
        version = await self.active(entity)
        if version is None or not keys:
            return []
        table, key = _TABLES[entity]
        rows = await self._pg.fetch(
            f"SELECT {key}, vec FROM {table} WHERE version = $1 AND {key} = ANY($2)",
            version.version,
            keys,
        )
        if not rows:
            return []
        # pgvector's asyncpg codec decodes to pgvector.Vector, not ndarray
        mat = np.stack([np.asarray(r["vec"].to_numpy(), dtype=np.float32) for r in rows])
        sims = mat @ query.astype(np.float32)
        if not version.normalized:
            sims /= np.linalg.norm(mat, axis=1) * np.linalg.norm(query) + 1e-9
        order = np.argsort(-sims)
        return [(rows[i][key], float(sims[i])) for i in order]
