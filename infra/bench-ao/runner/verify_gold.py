"""Scorer check: the gold SQL must give the same answer count as the graph did when the
context set was drawn (its `answers` field, capped at 200)."""
import json, sys, time
import psycopg
from gold import gold_set, music_dsn

paths = sys.argv[1:]
bad = n = 0
t0 = time.time()
with psycopg.connect(music_dsn()) as conn:
    for path in paths:
        for line in open(path):
            t = json.loads(line)
            if t["state"] is None:
                continue
            g = gold_set(conn, t["state"])
            n += 1
            if min(len(g), 200) != t["answers"]:
                bad += 1
                if bad <= 5:
                    print("MISMATCH", t["conversation_id"], t["turn"], len(g), t["answers"], t["state"])
print(f"checked {n} turns, mismatches {bad}, {time.time() - t0:.0f}s")
