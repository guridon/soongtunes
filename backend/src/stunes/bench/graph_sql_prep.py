"""Graph store vs relational store, step 1 (runs where the served Neo4j is).

    python -m stunes.bench.graph_sql_prep export   --out .data/bench/graph
    python -m stunes.bench.graph_sql_prep workload --out .data/bench/graph/workload.jsonl

`export` writes the topology the chat's graph sources read (songs, playlists, artists, genres,
tags and the four edge types) as gzip CSV in `neo4j-admin database import` format. The same
files are loaded into a fresh Neo4j and into Postgres, so both stores hold identical data. The
1536-d v1 embeddings and other unread properties stay out.

`workload` turns recorded eval turns into the graph calls the served pipeline made: the
conditions in each trace (by_conditions) and the co-occurrence seeds, recovered the way
`HybridPipeline.turn` picks them. It also renders the served Cypher templates into
`cypher.json`, so the benchmark runs the served query text without importing this package.
"""

import argparse
import asyncio
import csv
import gzip
import json
import random
from collections import Counter
from pathlib import Path

from neo4j import AsyncGraphDatabase, Query

from stunes.config import get_settings
from stunes.retrieval import graph as served

CHUNK = 20_000
TIMEOUT_S = 600

EXPECTED = {
    "includes": 5_285_871,
    # Song -> Tag only; the other 476,330 HAS_TAG edges start at playlists, which no chat
    # query reads
    "has_tag": 10_984_644,
    "has_genre": 802_859,
    "performed_by": 779_461,
}


async def _read(driver, query: str, **params) -> list[dict]:
    records, _, _ = await driver.execute_query(
        Query(query, timeout=TIMEOUT_S), params, database_="neo4j"
    )
    return [r.data() for r in records]


def _writer(path: Path, header: list[str]):
    f = gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=3)
    w = csv.writer(f)
    w.writerow(header)
    return f, w


async def export(out: Path) -> None:
    s = get_settings()
    out.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (offline script)
    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    counts: Counter = Counter()
    try:
        # Genre names repeat (domestic and foreign codes share 록/메탈, 랩/힙합, ...) and one
        # genre has no name, so genres are keyed by genre_id; a name matches every genre
        # carrying it, in both stores. Tag names are unique and get an integer key.
        genres = await _read(
            driver, "MATCH (g:Genre) RETURN g.genre_id AS id, g.name AS name, g.song_count AS n"
        )
        tags = await _read(driver, "MATCH (t:Tag) RETURN t.name AS name, t.song_count AS n")
        tid = {r["name"]: i for i, r in enumerate(sorted(tags, key=lambda r: r["name"]))}
        f, w = _writer(out / "genres.csv.gz", ["genre_id:ID(Genre)", "name", "song_count:int"])
        for r in genres:
            w.writerow([r["id"], r["name"], r["n"]])
            counts["genres"] += 1
        f.close()
        f, w = _writer(out / "tags.csv.gz", ["tid:ID(Tag)", "name", "song_count:int"])
        for r in tags:
            w.writerow([tid[r["name"]], r["name"], r["n"]])
            counts["tags"] += 1
        f.close()

        artists = await _read(driver, "MATCH (a:Artist) RETURN a.artist_id AS id")
        f, w = _writer(out / "artists.csv.gz", ["artist_id:ID(Artist)"])
        for r in artists:
            w.writerow([r["id"]])
            counts["artists"] += 1
        f.close()

        playlists = await _read(
            driver, "MATCH (p:Playlist) RETURN p.playlist_id AS id, p.like_cnt AS like_cnt"
        )
        f, w = _writer(out / "playlists.csv.gz", ["playlist_id:ID(Playlist)", "like_cnt"])
        for r in playlists:
            w.writerow([r["id"], r["like_cnt"]])
            counts["playlists"] += 1
        f.close()

        song_ids = [r["id"] for r in await _read(driver, "MATCH (s:Song) RETURN s.song_id AS id")]
        files = {
            "songs": _writer(
                out / "songs.csv.gz",
                ["song_id:ID(Song)", "popularity:int", "issue_date", "year:int"],
            ),
            "has_tag": _writer(out / "has_tag.csv.gz", [":START_ID(Song)", ":END_ID(Tag)"]),
            "has_genre": _writer(out / "has_genre.csv.gz", [":START_ID(Song)", ":END_ID(Genre)"]),
            "performed_by": _writer(
                out / "performed_by.csv.gz", [":START_ID(Song)", ":END_ID(Artist)"]
            ),
        }
        song_q = """
        MATCH (s:Song) WHERE s.song_id IN $ids
        RETURN s.song_id AS id, s.popularity AS pop, s.issue_date AS d,
               toInteger(left(s.issue_date, 4)) AS year,
               [(s)-[:HAS_TAG]->(t:Tag) | t.name] AS tags,
               [(s)-[:HAS_GENRE]->(g:Genre) | g.genre_id] AS genres,
               [(s)-[:PERFORMED_BY]->(a:Artist) | a.artist_id] AS artists
        """
        for i in range(0, len(song_ids), CHUNK):
            for r in await _read(driver, song_q, ids=song_ids[i : i + CHUNK]):
                files["songs"][1].writerow([r["id"], r["pop"], r["d"], r["year"]])
                counts["songs"] += 1
                for t in r["tags"]:
                    files["has_tag"][1].writerow([r["id"], tid[t]])
                    counts["has_tag"] += 1
                for g in r["genres"]:
                    files["has_genre"][1].writerow([r["id"], g])
                    counts["has_genre"] += 1
                for a in r["artists"]:
                    files["performed_by"][1].writerow([r["id"], a])
                    counts["performed_by"] += 1
            print(f"songs {min(i + CHUNK, len(song_ids)):,}/{len(song_ids):,}", flush=True)
        for f, _ in files.values():
            f.close()

        f, w = _writer(out / "includes.csv.gz", [":START_ID(Playlist)", ":END_ID(Song)"])
        pl_ids = [r["id"] for r in playlists]
        inc_q = """
        MATCH (p:Playlist) WHERE p.playlist_id IN $ids
        MATCH (p)-[:INCLUDES]->(s:Song) RETURN p.playlist_id AS p, s.song_id AS s
        """
        for i in range(0, len(pl_ids), CHUNK // 4):
            for r in await _read(driver, inc_q, ids=pl_ids[i : i + CHUNK // 4]):
                w.writerow([r["p"], r["s"]])
                counts["includes"] += 1
        f.close()
    finally:
        await driver.close()

    for k, n in EXPECTED.items():
        if counts[k] != n:
            raise SystemExit(f"{k}: exported {counts[k]:,}, the graph has {n:,}")
    (out / "counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


# ------------------------------------------------------------------------------ workload

RUNS = {
    # multi-turn context test run: hybrid / hybrid_cumstate / hybrid_ctx, 2,000 turns each
    "ctx": "results/eval/20261105T025421Z/predictions.jsonl",
    # v1 eval set, warm interleaved run (pop5); the agent's calls go through its own tools
    "v1warm": "results/eval/20261101T130015Z/predictions.jsonl",
}
PIPELINE_STRATEGIES = {"hybrid", "hybrid_ctx", "hybrid_cumstate"}


def _cypher() -> dict[str, str]:
    """The served query text, assembled exactly as GraphRetriever does (llm tags off)."""
    rank = served._RANK_BY_TAG_HITS
    q = {
        "anchor_sizes": served._ANCHOR_SIZES,
        # the bounded walk since 11-07 (its expand fallback is not replayed here); the
        # recorded bench ran the unbounded one
        "scan:popularity": served._SCAN.replace("{filters}", served._HARD_FILTERS).replace(
            "$scan_depth", str(served.SCAN_DEPTH)
        )
        + rank,
        "cooccur": served._COOCCUR.replace("{filters}", served._HARD_FILTERS),
    }
    for kind, head in served._EXPAND.items():
        q[f"expand:{kind}"] = head.replace("{filters}", served._HARD_FILTERS) + rank
    q["ANCHOR_MAX_SONGS"] = str(served.ANCHOR_MAX_SONGS)
    return q


def workload(out: Path, seed: int = 0) -> None:
    root = Path(__file__).resolve().parents[4]
    calls: list[dict] = []
    check: Counter = Counter()
    for run, rel in RUNS.items():
        recs = [json.loads(line) for line in open(root / rel)]
        prev = {(r["strategy"], r["conversation_id"], r["turn"]): r["predicted"] for r in recs}
        for r in recs:
            if r["strategy"] not in PIPELINE_STRATEGIES:
                continue
            t = r.get("trace") or {}
            cond = t.get("conditions")
            if not cond:
                continue
            plans = t.get("plans") or {}
            cands = t.get("candidates") or {}
            has_pos = bool(cond["tags"] or cond["genres"] or cond["artists"])
            base = {
                "run": run,
                "strategy": r["strategy"],
                "conversation_id": r["conversation_id"],
                "turn": r["turn"],
                "conditions": cond,
            }
            ran_graph = "graph" in cands or "graph_error" in plans
            check["graph: recorded / derived / both"] += 0
            check[f"graph recorded={ran_graph} derived={has_pos}"] += 1
            if has_pos:
                calls.append(
                    base
                    | {
                        "kind": "graph",
                        "recorded": {
                            "plan": plans.get("graph"),
                            "timed_out": "graph_error" in plans,
                        },
                    }
                )
            # seeds as HybridPipeline.turn picks them; a stored positive song preference
            # isn't in the trace, so those turns fall back to the last results
            linked_songs = [s["key"] for s in (t.get("linked") or {}).get("songs", [])]
            seeds = linked_songs
            if not seeds and (t.get("slots") or {}).get("similar_to_previous"):
                seeds = (prev.get((r["strategy"], r["conversation_id"], r["turn"] - 1)) or [])[:3]
            ran_co = "cooccur" in cands or "cooccur_error" in plans
            check[f"cooccur recorded={ran_co} derived={bool(seeds)}"] += 1
            if seeds:
                calls.append(
                    base
                    | {
                        "kind": "cooccur",
                        "seeds": seeds,
                        "recorded": {"timed_out": "cooccur_error" in plans, "ran": ran_co},
                    }
                )
    del check["graph: recorded / derived / both"]
    random.Random(seed).shuffle(calls)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c in calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out.parent / "cypher.json").write_text(json.dumps(_cypher(), indent=2))
    kinds = Counter(c["kind"] for c in calls)
    print(json.dumps({"calls": kinds, "check": check}, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("workload")
    w.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "export":
        asyncio.run(export(a.out))
    else:
        workload(a.out)


if __name__ == "__main__":
    main()
