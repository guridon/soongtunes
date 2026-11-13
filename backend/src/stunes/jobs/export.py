"""Export songs and linkable entities from Neo4j into JSONL snapshots.

The search index and the embedding backfill are both *derived* from the graph; exporting
once and building both from the same snapshot keeps them consistent and lets either be
rebuilt without touching Neo4j again.
"""

import asyncio
import logging
import shutil
import time
from pathlib import Path

import orjson
from neo4j import AsyncDriver, AsyncGraphDatabase, Query

from stunes.config import Settings

logger = logging.getLogger(__name__)

# Tags are ranked by how many playlists containing the song carry that tag, which is the
# signal the Song-[:HAS_TAG] edges were derived from but which those edges don't keep.
SONGS = """
MATCH (s:Song) WHERE toInteger(s.song_id) % $parts = $part
CALL (s) {
  MATCH (s)<-[:INCLUDES]-(p:Playlist)-[:HAS_TAG]->(t:Tag)
  WITH t.name AS tag, count(*) AS weight
  ORDER BY weight DESC, tag
  LIMIT $max_tags
  RETURN collect([tag, weight]) AS tags
}
CALL (s) {
  MATCH (s)<-[:INCLUDES]-(p:Playlist)
  WITH p ORDER BY toInteger(p.like_cnt) DESC, p.playlist_id LIMIT $max_playlists
  RETURN collect(p.title) AS playlists
}
RETURN s.song_id AS song_id,
       s.title AS title,
       s.issue_date AS issue_date,
       [(s)-[:PERFORMED_BY]->(a:Artist) | {id: a.artist_id, name: a.name}] AS artists,
       [(s)-[:IN_ALBUM]->(al:Album) | {id: al.album_id, title: al.title}] AS albums,
       [(s)-[:HAS_GENRE]->(g:Genre) WHERE g.name IS NOT NULL | g.name] AS genres,
       COUNT { (s)<-[:INCLUDES]-() } AS playlist_count,
       tags,
       playlists
"""

ENTITIES = {
    "artist": """
        MATCH (a:Artist)
        RETURN a.artist_id AS key, a.name AS name, COUNT { (a)<-[:PERFORMED_BY]-() } AS popularity
    """,
    "genre": """
        MATCH (g:Genre) WHERE g.name IS NOT NULL
        WITH g.name AS name, sum(COUNT { (g)<-[:HAS_GENRE]-() }) AS popularity
        RETURN name AS key, name, popularity
    """,
    # playlist count, not song count: it says how often people actually use the tag
    "tag": """
        MATCH (t:Tag) WHERE t.name IS NOT NULL
        RETURN t.name AS key, t.name AS name, COUNT { (t)<-[:HAS_TAG]-(:Playlist) } AS popularity
    """,
}


def _year(issue_date: str | None) -> int | None:
    if issue_date and len(issue_date) >= 4 and issue_date[:4].isdigit():
        year = int(issue_date[:4])
        return year if 1900 <= year <= 2100 else None
    return None


def normalize_song(row: dict) -> dict:
    """Shape one exported row into the snapshot record used by indexing and embedding."""
    albums = row.get("albums") or []
    return {
        "song_id": row["song_id"],
        "title": row.get("title") or "",
        "issue_date": row.get("issue_date"),
        "year": _year(row.get("issue_date")),
        "artists": [a["name"] for a in row.get("artists") or [] if a.get("name")],
        "artist_ids": [a["id"] for a in row.get("artists") or [] if a.get("id")],
        "album": albums[0]["title"] if albums else None,
        "album_id": albums[0]["id"] if albums else None,
        "genres": sorted(set(row.get("genres") or [])),
        "tags": [tag for tag, _ in row.get("tags") or []],
        "tag_weights": [weight for _, weight in row.get("tags") or []],
        "playlist_count": row.get("playlist_count") or 0,
        "playlists": [p for p in row.get("playlists") or [] if p],
    }


async def _export_song_part(
    driver: AsyncDriver, settings: Settings, path: Path, part: int, parts: int, **params
) -> int:
    n = 0
    async with driver.session(database=settings.neo4j_database, fetch_size=2000) as session:
        with path.open("wb") as f:
            result = await session.run(
                Query(SONGS, timeout=4 * 3600), part=part, parts=parts, **params
            )
            async for record in result:
                f.write(orjson.dumps(normalize_song(record.data())) + b"\n")
                n += 1
                if n % 25_000 == 0:
                    logger.info("songs part %d/%d: %d", part, parts, n)
    return n


async def export_all(
    settings: Settings, out_dir: Path, max_tags: int = 15, max_playlists: int = 5, parts: int = 4
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        async with driver.session(database=settings.neo4j_database, fetch_size=2000) as session:
            for kind, query in ENTITIES.items():
                path = out_dir / f"entities_{kind}.jsonl"
                n = 0
                with path.open("wb") as f:
                    result = await session.run(Query(query, timeout=3600))
                    async for record in result:
                        if record["name"]:
                            f.write(orjson.dumps({"kind": kind, **record.data()}) + b"\n")
                            n += 1
                logger.info("exported %d %s entities -> %s", n, kind, path)

        # Community edition runs each query on one thread; partitioning by song_id gives
        # `parts` concurrent scans whose expensive per-song work doesn't overlap.
        started = time.monotonic()
        part_paths = [out_dir / f"songs.part{i}.tmp" for i in range(parts)]
        counts = await asyncio.gather(
            *(
                _export_song_part(
                    driver, settings, p, i, parts, max_tags=max_tags, max_playlists=max_playlists
                )
                for i, p in enumerate(part_paths)
            )
        )
        path = out_dir / "songs.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("wb") as out:
            for p in part_paths:
                with p.open("rb") as f:
                    shutil.copyfileobj(f, out)
                p.unlink()
        tmp.replace(path)  # never leave a half-written snapshot under the real name
        logger.info(
            "exported %d songs -> %s in %.0fs", sum(counts), path, time.monotonic() - started
        )
    finally:
        await driver.close()
