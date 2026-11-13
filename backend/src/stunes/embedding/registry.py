"""Embedding version registry.

A version moves building -> ready -> active -> (ready | retired). Exactly one version per
entity is active (enforced by a partial unique index); reads resolve the active version
once and cache it, so switching or rolling back is a single UPDATE, not a redeploy.
"""

import hashlib
import logging
from dataclasses import dataclass

import asyncpg

from stunes.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingVersion:
    version: str
    entity: str
    model_name: str
    dim: int
    normalized: bool
    template_hash: str
    status: str

    def index_name(self, method: str) -> str:
        # "<entity>_emb_<hnsw|ivf>_<version>": with versions capped at 40 chars this stays
        # within PostgreSQL's 63-byte identifier limit
        return f"{self.entity}_emb_{method}_{self.version.replace('-', '_').replace('.', '_')}"


def template_hash(template: str) -> str:
    return hashlib.sha256(template.encode()).hexdigest()[:12]


_COLUMNS = "version, entity, model_name, dim, normalized, template_hash, status"


async def get_version(pg: asyncpg.Pool, version: str) -> EmbeddingVersion | None:
    row = await pg.fetchrow(f"SELECT {_COLUMNS} FROM embedding_model WHERE version = $1", version)
    return EmbeddingVersion(**row) if row else None


async def get_active(pg: asyncpg.Pool, entity: str) -> EmbeddingVersion | None:
    row = await pg.fetchrow(
        f"SELECT {_COLUMNS} FROM embedding_model WHERE entity = $1 AND status = 'active'", entity
    )
    return EmbeddingVersion(**row) if row else None


async def register(
    pg: asyncpg.Pool,
    *,
    version: str,
    entity: str,
    model_name: str,
    dim: int,
    normalized: bool,
    template: str,
) -> EmbeddingVersion:
    """Create the version, or return it if it already exists with the same definition.

    Re-running a backfill must never silently mix two templates/models under one version
    (the spec's '버전 혼입' trap), so a mismatching definition is an error.
    """
    h = template_hash(template)
    await pg.execute(
        """
        INSERT INTO embedding_model (version, entity, model_name, dim, normalized, template, template_hash)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (version) DO NOTHING
        """,
        version,
        entity,
        model_name,
        dim,
        normalized,
        template,
        h,
    )
    existing = await get_version(pg, version)
    assert existing is not None
    if (existing.entity, existing.model_name, existing.dim, existing.template_hash) != (
        entity,
        model_name,
        dim,
        h,
    ):
        raise ValueError(
            f"version {version} already exists with a different definition: {existing}; "
            "pick a new version name"
        )
    return existing


async def activate(pg: asyncpg.Pool, version: str) -> None:
    async with pg.acquire() as conn, conn.transaction():
        target = await conn.fetchrow(
            "SELECT entity, status FROM embedding_model WHERE version = $1 FOR UPDATE", version
        )
        if target is None:
            raise ValueError(f"unknown version {version}")
        if target["status"] not in ("ready", "active"):
            raise ValueError(f"{version} is {target['status']}; only 'ready' versions can serve")
        await conn.execute(
            "UPDATE embedding_model SET status = 'ready' WHERE entity = $1 AND status = 'active'",
            target["entity"],
        )
        await conn.execute(
            "UPDATE embedding_model SET status = 'active', activated_at = now() WHERE version = $1",
            version,
        )
    logger.info("activated %s", version)


async def activate_cli(settings: Settings, version: str) -> None:
    pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=1)
    try:
        await activate(pool, version)
    finally:
        await pool.close()
