"""Score auto-ontology records against the context-set truth.

Per turn (reference turns and turns without SQL are kept and labelled):
- top10: the first 10 distinct song ids in the order the agent's own SQL returns its rows.
  The SQL is run as the agent wrote it, with the songs table's song_id added as a column.
- constraint P@10: top10 songs that meet the whole truth state, over 10 (as for SoongTunes).
- EX (execution match, BIRD's rule applied to song sets): the agent's SQL without its outer
  ORDER BY / LIMIT returns exactly the gold song set (gold.py).
- outcome: no_sql | untransformable | sql_error | empty | exact | too_narrow (pred < gold)
  | too_broad (pred > gold) | disjoint | mixed

usage: python score_ao.py RECORDS.jsonl OUT.jsonl
"""
import json, sys
import psycopg
import sqlglot
from sqlglot import exp
from gold import gold_set, music_dsn

TIMEOUT_MS = 30_000  # auto-ontology's own SQL timeout setting
SID = "__sid"


def _songs_alias(select: exp.Select) -> str | None:
    tables = []
    frm = select.args.get("from_") or select.args.get("from")
    if frm is not None:
        tables.append(frm.this)
    tables += [j.this for j in select.args.get("joins") or []]
    for t in tables:
        if isinstance(t, exp.Table) and t.name.lower() == "songs":
            return t.alias_or_name
    return None


def transform(sql: str) -> tuple[str, str] | None:
    """(ranked SQL with __sid, unordered/unlimited SQL with __sid) or None."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001
        return None
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None or select is not tree:
        return None  # set operations etc.; counted as untransformable
    alias = _songs_alias(select)
    if alias is None:
        return None
    col = exp.column("song_id", table=alias)
    ranked = select.copy()
    ranked.set("expressions", [exp.alias_(col.copy(), SID)] + list(ranked.expressions))
    group = ranked.args.get("group")
    if group and not any(isinstance(e, exp.Column) and e.name == "song_id" for e in group.expressions):
        group.set("expressions", list(group.expressions) + [col.copy()])
    unranked = ranked.copy()
    unranked.set("order", None)
    unranked.set("limit", None)
    unranked.set("offset", None)
    return ranked.sql(dialect="postgres"), unranked.sql(dialect="postgres")


def run_top(conn, sql: str, k: int = 10) -> list:
    """The first k distinct song ids in the order the SQL returns its rows.

    Streamed and stopped at k: an agent SQL can return millions of joined rows, and loading
    them all ran the scorer out of memory (test2, 2026-11-09). __sid is the first column.
    """
    top, seen = [], set()
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {TIMEOUT_MS}")
        rows = cur.stream(sql)
        try:
            for r in rows:
                if r[0] not in seen:
                    seen.add(r[0])
                    top.append(r[0])
                    if len(top) == k:
                        break
        finally:
            rows.close()
    return top


def run_set(conn, sql: str) -> set:
    """The set of song ids the SQL returns; Postgres removes the duplicates."""
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {TIMEOUT_MS}")
        cur.execute(f"SELECT DISTINCT q.{SID} FROM ({sql}) AS q")
        return {r[0] for r in cur.fetchall()}


def score(conn, r: dict) -> dict:
    a = r.get("answer") or {}
    sql = a.get("sql_code") or (r["sql_events"][-1] if r.get("sql_events") else None)
    out = {k: r[k] for k in ("conversation_id", "turn", "op", "kind", "text")}
    out.update(sql=sql, seconds=r.get("seconds"), request_error=bool(r.get("error") or r.get("errors")),
               p10=0.0, ex=False, jaccard=0.0, top10=[], n_pred=None, n_gold=None)
    st = r.get("truth")
    gold = gold_set(conn, st) if st else None
    out["n_gold"] = len(gold) if gold is not None else None
    if not sql:
        out["outcome"] = "no_sql"
        return out
    t = transform(sql)
    if t is None:
        out["outcome"] = "untransformable"
        return out
    try:
        top = run_top(conn, t[0])
        conn.rollback()
        pred = run_set(conn, t[1])
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        out.update(outcome="sql_error", sql_error=f"{type(e).__name__}: {str(e)[:200]}")
        return out
    conn.rollback()
    out.update(top10=top, n_pred=len(pred))
    if gold is None:
        out["outcome"] = "reference"
        return out
    out["p10"] = sum(s in gold for s in top) / 10
    out["ex"] = pred == gold
    out["jaccard"] = len(pred & gold) / len(pred | gold) if pred | gold else 1.0
    if not pred:
        out["outcome"] = "empty"
    elif pred == gold:
        out["outcome"] = "exact"
    elif pred < gold:
        out["outcome"] = "too_narrow"
    elif pred > gold:
        out["outcome"] = "too_broad"
    elif not pred & gold:
        out["outcome"] = "disjoint"
    else:
        out["outcome"] = "mixed"
    return out


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    recs = [json.loads(l) for l in open(src)]
    with psycopg.connect(music_dsn()) as conn, open(dst, "w") as f:
        conn.read_only = True
        for r in recs:
            f.write(json.dumps(score(conn, r), ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in open(dst)]
    from collections import Counter
    main_rows = [x for x in rows if x["op"] != "reference" and x["turn"] >= 2]
    print("outcomes (all):", dict(Counter(x["outcome"] for x in rows)))
    for name, sel in (("turn 1", [x for x in rows if x["turn"] == 1]), ("turns 2-5, no reference", main_rows)):
        if sel:
            print(f"{name}: n={len(sel)} P@10={sum(x['p10'] for x in sel)/len(sel):.3f} "
                  f"EX={sum(x['ex'] for x in sel)/len(sel):.3f} "
                  f"no_sql={sum(x['outcome']=='no_sql' for x in sel)}")


if __name__ == "__main__":
    main()
