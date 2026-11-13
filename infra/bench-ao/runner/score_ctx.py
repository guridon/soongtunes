"""Score a SoongTunes eval run (predictions.jsonl) the same way as score_ao.py.

- constraint P@10: the arm's first 10 predicted songs that are in the gold set, over 10
- EX: the arm's applied conditions (trace.conditions, conditions_sql) return exactly the gold
  set; a turn searched with no condition is not exact
- outcome: as score_ao.py, plus no_condition

usage: python score_ctx.py PREDICTIONS.jsonl SET.jsonl STRATEGY OUT.jsonl [--conversations N]
"""
import ast, json, sys
from collections import Counter
import psycopg
from gold import conditions_sql, gold_set, music_dsn

TIMEOUT_MS = 30_000


def lit(v):
    return v if not isinstance(v, str) else ast.literal_eval(v)


def main() -> None:
    pred_path, set_path, strategy, dst = sys.argv[1:5]
    n_conv = int(sys.argv[sys.argv.index("--conversations") + 1]) if "--conversations" in sys.argv else None
    truth = {}
    for line in open(set_path):
        t = json.loads(line)
        truth[(t["conversation_id"], int(t["turn"]))] = t
    keep = sorted({c for c, _ in truth})[:n_conv] if n_conv else None
    rows = []
    with psycopg.connect(music_dsn()) as conn:
        conn.read_only = True
        for line in open(pred_path):
            r = json.loads(line)
            if r["strategy"] != strategy:
                continue
            key = (r["conversation_id"], int(r["turn"]))
            if keep is not None and key[0] not in keep:
                continue
            t = truth[key]
            out = {"conversation_id": key[0], "turn": key[1], "op": t["op"], "kind": t.get("kind"),
                   "text": t["text"], "seconds": float(r["latency_ms"]) / 1000 if r.get("latency_ms") else None,
                   "request_error": bool(r.get("error")), "p10": 0.0, "ex": False, "jaccard": 0.0}
            pred = [str(x) for x in lit(r.get("predicted") or "[]")]
            top = list(dict.fromkeys(pred))[:10]
            out["top10"] = top
            st = t["state"]
            if st is None:
                out["outcome"] = "reference"
                rows.append(out)
                continue
            gold = gold_set(conn, st)
            out["n_gold"] = len(gold)
            out["p10"] = sum(int(s) in gold for s in top) / 10
            trace = lit(r.get("trace") or "{}") or {}
            q = conditions_sql(trace.get("conditions") or {})
            if q is None:
                out["outcome"] = "no_condition"
                rows.append(out)
                continue
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {TIMEOUT_MS}")
                cur.execute(*q)
                ps = {x[0] for x in cur.fetchall()}
            conn.rollback()
            out["n_pred"] = len(ps)
            out["ex"] = ps == gold
            out["jaccard"] = len(ps & gold) / len(ps | gold) if ps | gold else 1.0
            out["outcome"] = ("empty" if not ps else "exact" if ps == gold else "too_narrow" if ps < gold
                              else "too_broad" if ps > gold else "disjoint" if not ps & gold else "mixed")
            rows.append(out)
    with open(dst, "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("outcomes:", dict(Counter(x["outcome"] for x in rows)))
    for name, sel in (("turn 1", [x for x in rows if x["turn"] == 1]),
                      ("turns 2-5, no reference", [x for x in rows if x["turn"] >= 2 and x["op"] != "reference"])):
        if sel:
            print(f"{name}: n={len(sel)} P@10={sum(x['p10'] for x in sel)/len(sel):.3f} EX={sum(x['ex'] for x in sel)/len(sel):.3f}")


if __name__ == "__main__":
    main()
