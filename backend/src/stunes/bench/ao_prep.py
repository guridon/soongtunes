"""Export the served graph as relational tables with names, for the Auto-Ontology baseline
(docs/prereg-ao-baseline.md).

    python -m stunes.bench.ao_prep OUT_DIR   (e.g. .data/ao/music; infra/bench-ao loads it)

Same data the context set's truth uses: Song (title, popularity, issue_date), Artist (name),
Genre (name), Tag (name) and the PERFORMED_BY / HAS_GENRE / HAS_TAG edges from songs.
Playlists, albums and LLM tags are left out: no truth condition reads them.
"""

import asyncio
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

from neo4j import AsyncGraphDatabase, Query

from stunes.config import get_settings

CHUNK = 20_000


def W(out: Path, name: str, header: list[str]):
    f = gzip.open(out / f"{name}.csv.gz", "wt", newline="", encoding="utf-8", compresslevel=3)
    w = csv.writer(f)
    w.writerow(header)
    return f, w


async def read(d, q, **p):
    r, _, _ = await d.execute_query(Query(q, timeout=600), p, database_="neo4j")
    return [x.data() for x in r]


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (offline script)
    s = get_settings()
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    c = Counter()
    try:
        f, w = W(out, "genres", ["genre_id", "genre_name"])
        for r in await read(d, "MATCH (g:Genre) RETURN g.genre_id AS id, g.name AS n"):
            w.writerow([r["id"], r["n"]])
            c["genres"] += 1
        f.close()
        tags = await read(d, "MATCH (t:Tag) RETURN t.tag_id AS id, t.name AS n")
        assert len({t["id"] for t in tags}) == len(tags), "tag_id not unique"
        tid = {t["n"]: t["id"] for t in tags}
        assert len(tid) == len(tags), "tag names not unique"
        f, w = W(out, "tags", ["tag_id", "tag_name"])
        for t in tags:
            w.writerow([t["id"], t["n"]])
            c["tags"] += 1
        f.close()
        f, w = W(out, "artists", ["artist_id", "artist_name"])
        for r in await read(d, "MATCH (a:Artist) RETURN a.artist_id AS id, a.name AS n"):
            w.writerow([r["id"], r["n"]])
            c["artists"] += 1
        f.close()
        ids = [r["id"] for r in await read(d, "MATCH (s:Song) RETURN s.song_id AS id")]
        fs = {
            k: W(out, k, h)
            for k, h in {
                "songs": ["song_id", "title", "popularity", "issue_date", "release_year"],
                "song_artists": ["song_id", "artist_id"],
                "song_genres": ["song_id", "genre_id"],
                "song_tags": ["song_id", "tag_id"],
            }.items()
        }
        q = """MATCH (s:Song) WHERE s.song_id IN $ids
        RETURN s.song_id AS id, s.title AS title, s.popularity AS pop, s.issue_date AS d,
               toInteger(left(s.issue_date, 4)) AS year,
               [(s)-[:HAS_TAG]->(t:Tag) | t.name] AS tags,
               [(s)-[:HAS_GENRE]->(g:Genre) | g.genre_id] AS genres,
               [(s)-[:PERFORMED_BY]->(a:Artist) | a.artist_id] AS artists"""
        for i in range(0, len(ids), CHUNK):
            for r in await read(d, q, ids=ids[i : i + CHUNK]):
                fs["songs"][1].writerow([r["id"], r["title"], r["pop"], r["d"], r["year"]])
                c["songs"] += 1
                for t in r["tags"]:
                    fs["song_tags"][1].writerow([r["id"], tid[t]])
                    c["song_tags"] += 1
                for g in r["genres"]:
                    fs["song_genres"][1].writerow([r["id"], g])
                    c["song_genres"] += 1
                for a in r["artists"]:
                    fs["song_artists"][1].writerow([r["id"], a])
                    c["song_artists"] += 1
            print(f"songs {min(i + CHUNK, len(ids)):,}/{len(ids):,}", flush=True)
        for f, _ in fs.values():
            f.close()
    finally:
        await d.close()
    print(dict(c))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
