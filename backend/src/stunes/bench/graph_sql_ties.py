"""Graph store vs relational store, step 3: are the differing lists ties, or a translation bug?

    python graph_sql_ties.py --rows out/run1/c1.jsonl --workload data/workload.jsonl --pg ...

Two places have no defined order in either store, so lists may differ there and only there:
- scan:popularity walks songs in popularity order and stops at scan_k matches. Songs that tie
  with the last one walked can be in or out, and one let in can displace a more popular song
  from the top k. A difference is a tie when a song in the symmetric difference sits at the
  boundary (the popularity of the scan_k-th match) and none sits below it. With fewer than
  scan_k matches there is no boundary, and any difference is unexplained.
- cooccur keeps each seed's top max_playlists playlists by like_cnt. A seed whose playlist at
  the cap ties with the next one can swap playlists. A difference is a tie when some seed has
  such a tie at the cap.
Anything else that differs is unexplained and counted as such.
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import asyncpg
from graph_sql_bench import MAX_PLAYLISTS, K, _Args, _filters, _has_tag, _ints, load_calls, scan_k


def boundary_sql(c: dict) -> tuple[str, list]:
    """Popularity of the scan_k-th song the scan walk matches (as sql_graph's scan)."""
    a = _Args()
    where = ["s.popularity >= 0"]
    if c["tags"]:
        where.append(_has_tag(a, "s", c["tags"]))
    where += _filters(a, c)
    sql = (
        f"SELECT s.popularity FROM song s WHERE {' AND '.join(where)} "
        f"ORDER BY s.popularity DESC OFFSET {a.p(scan_k(K) - 1, 'int')} LIMIT 1"
    )
    return sql, a.values


CAP_TIE_SQL = """
SELECT sd.sid,
       (SELECT p.like_cnt FROM includes i JOIN playlist p USING (pid) WHERE i.sid = sd.sid
        ORDER BY p.like_cnt DESC OFFSET $2 - 1 LIMIT 1) AS at_cap,
       (SELECT p.like_cnt FROM includes i JOIN playlist p USING (pid) WHERE i.sid = sd.sid
        ORDER BY p.like_cnt DESC OFFSET $2 LIMIT 1) AS after_cap
FROM unnest($1::int[]) AS sd(sid)
"""


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--pg", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    calls = {c["i"]: c for c in load_calls(a.workload)}
    by: dict[int, dict] = {}
    for line in open(a.rows):  # noqa: ASYNC230 (offline script)
        r = json.loads(line)
        by.setdefault(r["i"], {})[r["db"]] = r
    conn = await asyncpg.connect(a.pg)
    verdict: Counter = Counter()
    examples: list[dict] = []
    try:
        for i, d in sorted(by.items()):
            n, g = d.get("neo4j"), d.get("postgres")
            if not n or not g or n["error"] or g["error"]:
                continue
            key = f"{n['kind']} {n['plan']}"
            if n["ids"] == g["ids"]:
                verdict[f"{key}: identical"] += 1
                continue
            call = calls[i]
            if set(n["ids"]) == set(g["ids"]):
                # co-occurrence counts move with a swapped playlist, so the order can too
                tie = None
                if n["plan"] == "cooccur":
                    caps = await conn.fetch(CAP_TIE_SQL, _ints(call["seeds"]), MAX_PLAYLISTS)
                    tie = any(
                        r["after_cap"] is not None and r["at_cap"] == r["after_cap"] for r in caps
                    )
                label = "" if tie is None else (", tie" if tie else ", UNEXPLAINED")
                verdict[f"{key}: same set, other order{label}"] += 1
                continue
            diff = sorted(set(n["ids"]) ^ set(g["ids"]), key=int)
            if n["plan"] == "scan:popularity":
                sql, args = boundary_sql(call["conditions"])
                b = await conn.fetchval(sql, *args)
                pops = await conn.fetch(
                    "SELECT sid, popularity FROM song WHERE sid = ANY($1::int[])", _ints(diff)
                )
                # songs below the boundary are walked by neither store; songs above it are
                # walked by both and can differ only by being displaced by a boundary song
                tie = (
                    b is not None
                    and all(r["popularity"] >= b for r in pops)
                    and any(r["popularity"] == b for r in pops)
                )
                info = {"boundary": b, "diff_pops": sorted({r["popularity"] for r in pops})}
            elif n["plan"] == "cooccur":
                caps = await conn.fetch(CAP_TIE_SQL, _ints(call["seeds"]), MAX_PLAYLISTS)
                tie = any(
                    r["after_cap"] is not None and r["at_cap"] == r["after_cap"] for r in caps
                )
                info = {"caps": [dict(r) for r in caps]}
            else:
                tie, info = False, {}
            verdict[f"{key}: differs, {'tie' if tie else 'UNEXPLAINED'}"] += 1
            if not tie or len(examples) < 5:
                examples.append(
                    {
                        "i": i,
                        "key": key,
                        "tie": tie,
                        "n_diff": len(diff),
                        **info,
                        "conditions": call["conditions"],
                        "seeds": call.get("seeds"),
                    }
                )
    finally:
        await conn.close()
    out = {"verdict": dict(sorted(verdict.items())), "examples": examples[:40]}
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(out["verdict"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
