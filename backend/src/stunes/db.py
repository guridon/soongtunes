"""Connections to the three stores. Each store has one job (spec: 역할을 섞지 않음):

- Neo4j: relationship-conditioned search and conversation state (VirtualQuery)
- Postgres/pgvector: embedding registry and dense vectors
- OpenSearch: BM25 keyword search (Nori analyzer) over songs and entity names
"""

from dataclasses import dataclass
from typing import Any

import asyncpg
import redis.asyncio as aioredis
from neo4j import AsyncDriver, AsyncGraphDatabase, Query, RoutingControl
from opensearchpy import AsyncOpenSearch
from pgvector.asyncpg import register_vector

from stunes.config import Settings


class Graph:
    def __init__(self, settings: Settings):
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._database = settings.neo4j_database

    async def read(
        self, query: str, timeout_s: float | None = None, **params: Any
    ) -> list[dict[str, Any]]:
        # timeout_s is a server-side transaction timeout: a caller that stops waiting doesn't
        # stop the query, the server has to (troubleshooting #18)
        records, _, _ = await self._driver.execute_query(
            Query(query, timeout=timeout_s) if timeout_s else query,
            params,
            database_=self._database,
            routing_=RoutingControl.READ,
        )
        return [r.data() for r in records]

    async def write(self, query: str, **params: Any) -> list[dict[str, Any]]:
        records, _, _ = await self._driver.execute_query(
            query, params, database_=self._database, routing_=RoutingControl.WRITE
        )
        return [r.data() for r in records]

    async def ping(self) -> bool:
        await self._driver.verify_connectivity()
        return True

    async def close(self) -> None:
        await self._driver.close()


async def _init_pg_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def create_pg_pool(settings: Settings) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        settings.postgres_dsn, min_size=1, max_size=10, init=_init_pg_connection
    )


def create_search(settings: Settings) -> AsyncOpenSearch:
    return AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=10)


@dataclass
class Stores:
    graph: Graph
    pg: asyncpg.Pool
    search: AsyncOpenSearch
    redis: aioredis.Redis

    @classmethod
    async def open(cls, settings: Settings) -> "Stores":
        return cls(
            graph=Graph(settings),
            pg=await create_pg_pool(settings),
            search=create_search(settings),
            redis=aioredis.Redis.from_url(settings.redis_url),
        )

    async def close(self) -> None:
        await self.graph.close()
        await self.pg.close()
        await self.search.close()
        await self.redis.aclose()
