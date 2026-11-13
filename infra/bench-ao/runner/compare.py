"""The analysis docs/prereg-ao-baseline.md registered: a SoongTunes arm (the served
hybrid_delta_keys, or hybrid_ctx) against Auto-Ontology on the same context-set turns, both
scored by score_ao.py / score_ctx.py.

- primary: turns 2-5 without reference turns; turn 1 and each operation reported apart
- P@10 and EX per arm, paired differences (arm - Auto-Ontology) with a bootstrap over
  conversations (10,000 resamples, seed 0, percentile bounds)
- floors: the same 10 most popular songs on every turn, and 10 uniformly random songs (5 seeds)
- outcomes and latency per arm; health: request errors, turns without SQL / conditions

usage: python compare.py AO_SCORED.jsonl ARM_SCORED.jsonl SET.jsonl [ARM_NAME]
"""

import json
import random
import statistics
import sys
from collections import Counter, defaultdict

import psycopg

from gold import gold_set, music_dsn

AO, CTX, SET = sys.argv[1:4]
ARM = sys.argv[4] if len(sys.argv) > 4 else "hybrid_ctx"


def load(path):
    return {(r["conversation_id"], r["turn"]): r for r in map(json.loads, open(path))}


ao, ctx = load(AO), load(CTX)
keys = sorted(set(ao) & set(ctx))
only = (set(ao) ^ set(ctx))
truth = {}
for line in open(SET):
    t = json.loads(line)
    truth[(t["conversation_id"], int(t["turn"]))] = t

prim = [k for k in keys if k[1] >= 2 and truth[k]["op"] != "reference"]
first = [k for k in keys if k[1] == 1]


def boot(diffs: dict, n: int = 10_000) -> tuple[float, float, float]:
    by = defaultdict(list)
    for (c, _), d in diffs.items():
        by[c].append(d)
    convs = sorted(by)
    sums = [sum(by[c]) for c in convs]
    cnts = [len(by[c]) for c in convs]
    rng = random.Random(0)
    stats = []
    for _ in range(n):
        idx = [rng.randrange(len(convs)) for _ in convs]
        stats.append(sum(sums[i] for i in idx) / sum(cnts[i] for i in idx))
    stats.sort()
    return sum(sums) / sum(cnts), stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def mean(rows, k, metric):
    return sum(float(rows[x][metric]) for x in k) / len(k)


def floors(k):
    """P@10 of the popularity floor and of random 10 songs on these turns."""
    with psycopg.connect(music_dsn()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT song_id FROM songs ORDER BY popularity DESC, song_id LIMIT 10")
        popular = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT song_id FROM songs")
        allids = [r[0] for r in cur.fetchall()]
        pop, rnd = [], []
        for key in k:
            g = gold_set(conn, truth[key]["state"])
            pop.append(sum(s in g for s in popular) / 10)
            rnd.append(statistics.mean(
                sum(s in g for s in random.Random(f"{seed}-{key}").sample(allids, 10)) / 10
                for seed in range(5)))
    return statistics.mean(pop), statistics.mean(rnd)


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


print(f"turns paired {len(keys)}, in one arm only {len(only)}")
for name, k in (("turns 2-5 without reference (primary)", prim), ("turn 1", first)):
    print(f"\n## {name}, n {len(k)}")
    pf, rf = floors(k)
    print(f"  floors P@10: popular {pf:.3f}, random {rf:.3f}")
    for arm, rows in ((ARM, ctx), ("auto-ontology", ao)):
        print(f"  {arm:18s} P@10 {mean(rows, k, 'p10'):.3f}  EX {mean(rows, k, 'ex'):.3f}")
    for metric in ("p10", "ex"):
        d = {x: float(ctx[x][metric]) - float(ao[x][metric]) for x in k}
        print(f"  {ARM} - auto-ontology {metric}: {fmt(boot(d))}")

print(f"\n## by operation (turns 2-5): P@10 / EX, {ARM} vs auto-ontology")
for op in ("add", "relax", "replace", "exclude", "more", "new_topic"):
    k = [x for x in prim if truth[x]["op"] == op]
    if k:
        print(f"  {op:9s} n {len(k):3d}  arm {mean(ctx, k, 'p10'):.3f} / {mean(ctx, k, 'ex'):.3f}"
              f"   ao {mean(ao, k, 'p10'):.3f} / {mean(ao, k, 'ex'):.3f}")

print("\n## outcomes (all turns)")
for arm, rows in ((ARM, ctx), ("auto-ontology", ao)):
    c = Counter(rows[x]["outcome"] for x in keys)
    print(f"  {arm:14s} " + ", ".join(f"{o} {n}" for o, n in c.most_common()))

print("\n## latency per turn (s) and health")
for arm, rows in ((ARM, ctx), ("auto-ontology", ao)):
    s = sorted(float(rows[x]["seconds"]) for x in keys if rows[x].get("seconds") is not None)
    errs = sum(bool(rows[x].get("request_error")) for x in keys)
    print(f"  {arm:14s} p50 {statistics.median(s):.2f}  p95 {s[int(len(s) * 0.95)]:.2f}  "
          f"request errors {errs}")
