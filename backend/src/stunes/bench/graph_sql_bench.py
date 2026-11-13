"""Graph store vs relational store, step 2: replay the chat's graph calls on both stores.

Self-contained (neo4j + asyncpg only), so it runs next to the bench stores:

    python graph_sql_bench.py --workload workload.jsonl --cypher cypher.json \
        --neo4j bolt://neo4j:7687 --neo4j-password ... --pg postgresql://... \
        --phase c1 --n 1500 --out out/

Neo4j runs the served Cypher text (rendered by graph_sql_prep.py from
stunes.retrieval.graph). Postgres runs SQL written to return the same list:
- the same plan choice (artist anchor first; else the smaller of the tag and genre anchors,
  expanded when it holds <= ANCHOR_MAX_SONGS songs, otherwise songs walked in popularity
  order and stopped at scan_k matches);
- the same hard filters, ranking and tie-break (song_id as a string).
Both pay the anchor-size lookup as its own round trip, as the served path does.
Ties in popularity (the scan's walk) and in like_cnt (co-occurrence's playlist cap) have no
defined order in either store, so lists can differ there; `compare` reports how often.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import asyncpg
from neo4j import AsyncGraphDatabase, Query, RoutingControl

K = 100  # settings.candidate_pool
MAX_PLAYLISTS = 300  # GraphRetriever.cooccurring default
BUDGET_MS = 3000.0  # SOURCE_TIMEOUT_S in serving
TIMEOUT_S = 30.0  # the bench lets a query finish to see its real time


def scan_k(k: int) -> int:
    return max(3 * k, 300)


def choose_plan(c: dict, sizes: dict, anchor_max: int) -> str | None:
    """GraphRetriever.plan."""
    if c["artists"]:
        return "expand:artist"
    if not (c["tags"] or c["genres"]):
        return None
    options = []
    if c["tags"]:
        options.append((sizes["tags"], "expand:tag"))
    if c["genres"]:
        options.append((sizes["genres"], "expand:genre"))
    size, plan = min(options)
    return plan if size <= anchor_max else "scan:popularity"


# ----------------------------------------------------------------------------- Neo4j


class Neo:
    name = "neo4j"

    def __init__(self, uri: str, password: str, cypher: dict):
        self.d = AsyncGraphDatabase.driver(uri, auth=("neo4j", password))
        self.q = cypher
        self.anchor_max = int(cypher["ANCHOR_MAX_SONGS"])

    async def _read(self, query: str, **params) -> list[dict]:
        records, _, _ = await self.d.execute_query(
            Query(query, timeout=TIMEOUT_S),
            params,
            database_="neo4j",
            routing_=RoutingControl.READ,
        )
        return [r.data() for r in records]

    async def graph(self, c: dict) -> tuple[list[str], str | None]:
        if c["artists"]:
            sizes = {}
        else:
            sizes = (await self._read(self.q["anchor_sizes"], tags=c["tags"], genres=c["genres"]))[
                0
            ]
        plan = choose_plan(c, sizes, self.anchor_max)
        if plan is None:
            return [], None
        rows = await self._read(self.q[plan], k=K, scan_k=scan_k(K), **c)
        return [r["song_id"] for r in rows], plan

    async def cooccur(self, seeds: list[str], c: dict) -> list[str]:
        rows = await self._read(
            self.q["cooccur"],
            seeds=seeds,
            k=K,
            max_playlists=MAX_PLAYLISTS,
            exclude_playlists=[],
            **c,
        )
        return [r["song_id"] for r in rows]

    async def close(self) -> None:
        await self.d.close()


# ----------------------------------------------------------------------------- Postgres


class _Args:
    def __init__(self):
        self.values: list = []

    def p(self, value, cast: str) -> str:
        self.values.append(value)
        return f"${len(self.values)}::{cast}"


def _ints(xs: list[str]) -> list[int]:
    return [int(x) for x in xs]


def _tids(a: _Args, names: list[str]) -> str:
    return f"ARRAY(SELECT tid FROM tag WHERE name = ANY({a.p(names, 'text[]')}))"


def _gids(a: _Args, names: list[str]) -> str:
    return f"ARRAY(SELECT genre_id FROM genre WHERE name = ANY({a.p(names, 'text[]')}))"


def _has_tag(a: _Args, alias: str, names: list[str]) -> str:
    return (
        f"EXISTS (SELECT 1 FROM song_tag st WHERE st.sid = {alias}.sid "
        f"AND st.tid = ANY({_tids(a, names)}))"
    )


def _has_genre(a: _Args, alias: str, names: list[str]) -> str:
    return (
        f"EXISTS (SELECT 1 FROM song_genre sg WHERE sg.sid = {alias}.sid "
        f"AND sg.genre_id = ANY({_gids(a, names)}))"
    )


def _has_artist(a: _Args, alias: str, ids: list[str]) -> str:
    return (
        f"EXISTS (SELECT 1 FROM song_artist sa WHERE sa.sid = {alias}.sid "
        f"AND sa.aid = ANY({a.p(_ints(ids), 'int[]')}))"
    )


def _filters(a: _Args, c: dict, alias: str = "s", implied: str = "") -> list[str]:
    """_HARD_FILTERS; a filter the anchor already implies is left out (`implied`)."""
    out = []
    if c["exclude"]:
        out.append(f"NOT ({alias}.sid = ANY({a.p(_ints(c['exclude']), 'int[]')}))")
    if c["genres"] and implied != "genres":
        out.append(_has_genre(a, alias, c["genres"]))
    if c["artists"] and implied != "artists":
        out.append(_has_artist(a, alias, c["artists"]))
    if c["neg_genres"]:
        out.append("NOT " + _has_genre(a, alias, c["neg_genres"]))
    if c["neg_artists"]:
        out.append("NOT " + _has_artist(a, alias, c["neg_artists"]))
    if c["neg_tags"]:
        out.append("NOT " + _has_tag(a, alias, c["neg_tags"]))
    if c["year_from"] is not None:
        out.append(f"{alias}.year >= {a.p(c['year_from'], 'int')}")
    if c["year_to"] is not None:
        out.append(f"{alias}.year <= {a.p(c['year_to'], 'int')}")
    return out


def _tag_hits(a: _Args, alias: str, names: list[str]) -> str:
    if not names:
        return "0"
    return (
        f"(SELECT count(*) FROM song_tag st WHERE st.sid = {alias}.sid "
        f"AND st.tid = ANY({_tids(a, names)}))"
    )


def sql_graph(plan: str, c: dict) -> tuple[str, list]:
    a = _Args()
    where: list[str]
    limit = ""
    if plan == "expand:artist":
        where = [
            f"s.sid IN (SELECT sid FROM song_artist WHERE aid = ANY({a.p(_ints(c['artists']), 'int[]')}))"
        ]
        where += _filters(a, c, implied="artists")
    elif plan == "expand:genre":
        where = [
            f"s.sid IN (SELECT sid FROM song_genre WHERE genre_id = ANY({_gids(a, c['genres'])}))"
        ]
        if c["tags"]:
            where.append(_has_tag(a, "s", c["tags"]))
        where += _filters(a, c, implied="genres")
    elif plan == "expand:tag":
        where = [f"s.sid IN (SELECT sid FROM song_tag WHERE tid = ANY({_tids(a, c['tags'])}))"]
        where += _filters(a, c)
    else:  # scan:popularity
        where = ["s.popularity >= 0"]
        if c["tags"]:
            where.append(_has_tag(a, "s", c["tags"]))
        where += _filters(a, c)
        limit = f"ORDER BY s.popularity DESC LIMIT {a.p(scan_k(K), 'int')}"
    sql = f"""
WITH c AS (
  SELECT s.sid, s.popularity FROM song s
  WHERE {" AND ".join(where)}
  {limit}
)
SELECT c.sid, {_tag_hits(a, "c", c["tags"])} AS tag_hits FROM c
ORDER BY tag_hits DESC, c.popularity DESC, (c.sid::text) COLLATE "C"
LIMIT {a.p(K, "int")}
"""
    return sql, a.values


def sql_cooccur(seeds: list[str], c: dict) -> tuple[str, list]:
    a = _Args()
    seed_p = a.p(_ints(seeds), "int[]")
    seed_p2 = a.p(_ints(seeds), "int[]")
    where = _filters(a, c)
    sql = f"""
WITH sp AS (
  SELECT DISTINCT x.pid
  FROM unnest({seed_p}) AS sd(sid)
  CROSS JOIN LATERAL (
    SELECT i.pid FROM includes i JOIN playlist p ON p.pid = i.pid
    WHERE i.sid = sd.sid
    ORDER BY p.like_cnt DESC
    LIMIT {a.p(MAX_PLAYLISTS, "int")}
  ) x
), co AS (
  SELECT i.sid, count(DISTINCT i.pid) AS co
  FROM sp JOIN includes i ON i.pid = sp.pid
  WHERE NOT (i.sid = ANY({seed_p2}))
  GROUP BY i.sid
)
SELECT s.sid, co.co, {_tag_hits(a, "s", c["tags"])} AS tag_hits
FROM co JOIN song s ON s.sid = co.sid
{("WHERE " + " AND ".join(where)) if where else ""}
ORDER BY tag_hits DESC, co.co DESC, s.popularity DESC, (s.sid::text) COLLATE "C"
LIMIT {a.p(K, "int")}
"""
    return sql, a.values


ANCHOR_SQL = """
SELECT (SELECT coalesce(sum(song_count), 0) FROM tag WHERE name = ANY($1::text[]))::bigint AS tags,
       (SELECT coalesce(sum(song_count), 0) FROM genre WHERE name = ANY($2::text[]))::bigint AS genres
"""


class Pg:
    name = "postgres"

    def __init__(self, dsn: str, anchor_max: int):
        self.dsn = dsn
        self.anchor_max = anchor_max
        self.pool: asyncpg.Pool | None = None

    async def open(self, size: int) -> None:
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=size,
            max_size=size,
            server_settings={"statement_timeout": str(int(TIMEOUT_S * 1000))},
        )

    async def graph(self, c: dict) -> tuple[list[str], str | None]:
        async with self.pool.acquire() as conn:
            sizes = {}
            if not c["artists"]:
                sizes = dict(await conn.fetchrow(ANCHOR_SQL, c["tags"], c["genres"]))
            plan = choose_plan(c, sizes, self.anchor_max)
            if plan is None:
                return [], None
            sql, args = sql_graph(plan, c)
            rows = await conn.fetch(sql, *args)
        return [str(r["sid"]) for r in rows], plan

    async def cooccur(self, seeds: list[str], c: dict) -> list[str]:
        sql, args = sql_cooccur(seeds, c)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [str(r["sid"]) for r in rows]

    async def close(self) -> None:
        await self.pool.close()


# ----------------------------------------------------------------------------- runner


async def one(db, call: dict) -> dict:
    t0 = time.perf_counter()
    plan, ids, err = None, None, None
    try:
        if call["kind"] == "graph":
            ids, plan = await db.graph(call["conditions"])
        else:
            ids = await db.cooccur(call["seeds"], call["conditions"])
            plan = "cooccur"
    except Exception as e:  # recorded, not raised: a failed call is a result
        err = f"{type(e).__name__}: {str(e)[:200]}"
    ms = (time.perf_counter() - t0) * 1000
    return {
        "db": db.name,
        "i": call["i"],
        "kind": call["kind"],
        "plan": plan,
        "ms": ms,
        "ids": ids,
        "error": err,
    }


async def run_interleaved(dbs, calls, seed: int) -> list[dict]:
    """Concurrency 1: each call on both stores back to back, in a random order per call."""
    rng = random.Random(seed)
    out = []
    for n, call in enumerate(calls):
        order = list(dbs)
        rng.shuffle(order)
        for db in order:
            out.append(await one(db, call))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(calls)}", flush=True)
    return out


async def run_concurrent(db, calls, conc: int) -> list[dict]:
    sem = asyncio.Semaphore(conc)

    async def go(call):
        async with sem:
            return await one(db, call)

    return await asyncio.gather(*(go(c) for c in calls))


def load_calls(path: Path) -> list[dict]:
    """Distinct calls, in the workload's (shuffled) order; `weight` counts repeats."""
    seen: dict[str, dict] = {}
    for line in open(path):
        c = json.loads(line)
        key = json.dumps([c["kind"], c["conditions"], c.get("seeds")], sort_keys=True)
        if key in seen:
            seen[key]["weight"] += 1
        else:
            seen[key] = c | {"weight": 1, "i": len(seen)}
    return list(seen.values())


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["db"], r["kind"], "all")].append(r)
        groups[(r["db"], r["kind"], r["plan"] or "-")].append(r)
    out = {}
    for (db, kind, plan), rs in sorted(groups.items()):
        ok = [r["ms"] for r in rs if not r["error"]]
        out[f"{db} {kind} {plan}"] = {
            "n": len(rs),
            "errors": sum(1 for r in rs if r["error"]),
            "p50": round(pct(ok, 0.5), 1) if ok else None,
            "p95": round(pct(ok, 0.95), 1) if ok else None,
            "p99": round(pct(ok, 0.99), 1) if ok else None,
            "mean": round(statistics.fmean(ok), 1) if ok else None,
            "max": round(max(ok), 1) if ok else None,
            "over_budget": sum(1 for r in rs if r["error"] or r["ms"] > BUDGET_MS),
        }
    return out


def compare(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for r in rows:
        by[r["i"]][r["db"]] = r
    stats = defaultdict(lambda: defaultdict(int))
    for d in by.values():
        if len(d) < 2:
            continue
        a, b = d["neo4j"], d["postgres"]
        key = f"{a['kind']} {a['plan']}"
        s = stats[key]
        s["n"] += 1
        if a["error"] or b["error"]:
            s["either_failed"] += 1
            continue
        if a["plan"] != b["plan"]:
            s["plan_differs"] += 1
        if a["ids"] == b["ids"]:
            s["identical"] += 1
        elif set(a["ids"]) == set(b["ids"]):
            s["same_set_other_order"] += 1
        else:
            s["differs"] += 1
    return {k: dict(v) for k, v in stats.items()}


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--cypher", type=Path, required=True)
    p.add_argument("--neo4j", required=True)
    p.add_argument("--neo4j-password", required=True)
    p.add_argument("--pg", required=True)
    p.add_argument("--phase", choices=["warmup", "c1", "conc"], required=True)
    p.add_argument("--offset", type=int, default=0, help="skip the first N distinct calls")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--conc", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    cypher = json.loads(a.cypher.read_text())
    calls = load_calls(a.workload)[a.offset : a.offset + a.n]
    neo = Neo(a.neo4j, a.neo4j_password, cypher)
    pg = Pg(a.pg, int(cypher["ANCHOR_MAX_SONGS"]))
    await pg.open(max(a.conc, 1))
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.phase}" + (f"-c{a.conc}" if a.phase == "conc" else "")
    t0 = time.time()
    try:
        if a.phase in ("warmup", "c1"):
            rows = await run_interleaved([neo, pg], calls, a.seed)
        else:
            rows = []
            # ABBA: each store runs first once, so drift over time hits both
            for db in (neo, pg, pg, neo):
                print(f"  {db.name} conc {a.conc}", flush=True)
                part = await run_concurrent(db, calls, a.conc)
                rep = sum(1 for r in rows if r["db"] == db.name) // max(len(calls), 1)
                rows += [r | {"rep": rep} for r in part]
    finally:
        await neo.close()
        await pg.close()
    meta = {
        "phase": a.phase,
        "conc": a.conc,
        "offset": a.offset,
        "n": len(calls),
        "kinds": dict(
            sorted(
                defaultdict(
                    int, {k: sum(1 for c in calls if c["kind"] == k) for k in ("graph", "cooccur")}
                ).items()
            )
        ),
        "seconds": round(time.time() - t0, 1),
        "budget_ms": BUDGET_MS,
        "timeout_s": TIMEOUT_S,
    }
    with open(a.out / f"{tag}.jsonl", "w") as f:  # noqa: ASYNC230 (offline script, after the run)
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"meta": meta, "latency": summarize(rows), "agreement": compare(rows)}
    (a.out / f"{tag}-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
