"""Scorer check: the gold SQL itself, scored as if the agent wrote it, must get EX 1 and P@10 1."""
import json, sys
import psycopg
from gold import gold_sql, music_dsn
from score_ao import score

n = ex = 0
p10 = 0.0
with psycopg.connect(music_dsn()) as conn:
    for line in open(sys.argv[1]):
        t = json.loads(line)
        if t["state"] is None:
            continue
        sql, params = gold_sql(t["state"])
        literal = psycopg.ClientCursor(conn).mogrify(sql + " ORDER BY s.popularity DESC LIMIT 10", params)
        r = {**{k: t.get(k) for k in ("conversation_id", "turn", "op", "kind", "text")},
             "truth": t["state"], "answer": {"sql_code": literal}, "sql_events": []}
        s = score(conn, r)
        n += 1; ex += s["ex"]; p10 += s["p10"]
        if not s["ex"] or s["p10"] < 1:
            print("FAIL", t["conversation_id"], t["turn"], s["outcome"], s.get("sql_error"))
print(f"oracle: {n} turns, EX {ex / n:.3f}, P@10 {p10 / n:.3f}")
