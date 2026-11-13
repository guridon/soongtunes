"""Neo4j indexes and precomputed graph features the API's queries rely on (idempotent)."""

import logging
import time

from neo4j import AsyncGraphDatabase, Query

from stunes.config import Settings

logger = logging.getLogger(__name__)

STATEMENTS = [
    "CREATE RANGE INDEX song_id_index IF NOT EXISTS FOR (s:Song) ON (s.song_id)",
    "CREATE RANGE INDEX artist_id_index IF NOT EXISTS FOR (a:Artist) ON (a.artist_id)",
    "CREATE RANGE INDEX artist_name_index IF NOT EXISTS FOR (a:Artist) ON (a.name)",
    "CREATE RANGE INDEX tag_name_index IF NOT EXISTS FOR (t:Tag) ON (t.name)",
    "CREATE RANGE INDEX genre_name_index IF NOT EXISTS FOR (g:Genre) ON (g.name)",
    "CREATE RANGE INDEX album_id_index IF NOT EXISTS FOR (al:Album) ON (al.album_id)",
    "CREATE RANGE INDEX playlist_id_index IF NOT EXISTS FOR (p:Playlist) ON (p.playlist_id)",
    "CREATE CONSTRAINT virtual_query_session IF NOT EXISTS "
    "FOR (vq:VirtualQuery) REQUIRE vq.sessionId IS UNIQUE",
    "CREATE RANGE INDEX song_popularity_index IF NOT EXISTS FOR (s:Song) ON (s.popularity)",
]

# Song.popularity = number of playlists including the song. Counting it per query made
# supernode anchors (tag '감성' -> 176k songs) take ~3s; as a stored property ranking only
# reads one value per candidate. Recompute after playlist data changes.
POPULARITY = """
MATCH (s:Song)
CALL (s) {
  SET s.popularity = COUNT { (s)<-[:INCLUDES]-() }
} IN TRANSACTIONS OF 10000 ROWS
"""

# Tag/Genre song counts let the retriever pick a plan: expand from a small anchor, or scan
# songs in popularity order and stop early for a big one (see retrieval/graph.py).
SONG_COUNTS = [
    """
    MATCH (t:Tag)
    CALL (t) { SET t.song_count = COUNT { (t)<-[:HAS_TAG]-(:Song) } } IN TRANSACTIONS OF 5000 ROWS
    """,
    "MATCH (g:Genre) SET g.song_count = COUNT { (g)<-[:HAS_GENRE]-(:Song) }",
]


async def compute_graph_features(settings: Settings, only_counts: bool = False) -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        statements = SONG_COUNTS if only_counts else [POPULARITY, *SONG_COUNTS]
        async with driver.session(database=settings.neo4j_database) as session:
            for stmt in statements:
                started = time.monotonic()
                result = await session.run(Query(stmt, timeout=3600))
                summary = await result.consume()
                logger.info(
                    "%s: %d properties in %.0fs",
                    " ".join(stmt.split())[:60],
                    summary.counters.properties_set,
                    time.monotonic() - started,
                )
    finally:
        await driver.close()


async def create_graph_indexes(settings: Settings) -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        for stmt in STATEMENTS:
            await driver.execute_query(stmt, database_=settings.neo4j_database)
            logger.info("ok: %s", stmt)
        await driver.execute_query("CALL db.awaitIndexes(3600)", database_=settings.neo4j_database)
        logger.info("all indexes online")
    finally:
        await driver.close()
