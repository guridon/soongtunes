"""Count vs 1/degree co-occurrence by number of seeds, offline on the PPR dev split.

    python -m stunes.bench.seed_sweep --graph ../.data/ppr/graph.npz --out ../results/feed/seed-sweep.json

On the feed (5 seeds) 1/degree weights beat the count; on one seed ("songs like this one",
stunes.bench.similar_songs) they lost. This finds where the order flips. Same split and hidden
songs as stunes.bench.ppr_dev; the seeds are the n most recent plays. Lists alone, k = 50.
- count: each seed's 100 most-liked playlists, songs by distinct playlists, then popularity
  (the served Cypher without its tag ordering);
- degree: one step of the walk (artist share 0.2), as served on the feed.
"""

import argparse
import json
from pathlib import Path

from stunes.bench.ppr_dev import score, split, two_hop
from stunes.retrieval.ppr import PPR

SEEDS = (1, 2, 3, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()
    g = PPR(a.graph)
    base = split(g, 60_000, 200)
    rows = {}
    for n in SEEDS:
        qs = [{**q, "seeds": q["seen"][::-1][:n]} for q in base]
        count = [two_hop(g, q, a.k, 100) for q in qs]
        degree = [
            [sid for sid, _ in g.cooccur(q["seeds"], [q["playlist"]], q["seen"], a.k)] for q in qs
        ]
        c, d = score(g, qs, count), score(g, qs, degree)
        rows[n] = {"count": c, "degree": d}
        print(
            f"seeds={n}: count R@{a.k} {c['recall_at_k']:.4f} hit {c['hit_at_k']:.3f} | "
            f"degree R@{a.k} {d['recall_at_k']:.4f} hit {d['hit_at_k']:.3f}",
            flush=True,
        )
    a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
