"""Choose the PPR parameters on a dev split, offline (docs/prereg-feed-ppr.md).

    python -m stunes.bench.ppr_dev --graph ../.data/ppr/graph.npz --out ../results/feed/ppr-dev.json

The split mirrors `stunes feed-eval` (stunes.agent.feed_eval): playlists with 15-200 songs in
playlist_id order (as strings), each one's songs sorted then shuffled with one Random(5), the
first 80% are the plays, the 5 most recent of them are the seeds, the last 20% are hidden, and
the playlist is removed from the graph. Dev = SKIP 60,000; the test split (SKIP 50,000) is not
read here. Each candidate list stands alone (no fusion, no ranker), k = 50.

Rows: popularity, 2-hop co-occurrence (every playlist of a seed, and the served cap of each
seed's 100 most-liked playlists, without the served tag-hit ordering), and the PPR grid.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from stunes.retrieval.ppr import PPR

ALPHAS = (0.15, 0.3, 0.5, 0.7)
BETAS = (0.0, 0.25, 0.5)
ARTIST_WS = (0.0, 0.2)
HEAD = 1000  # stunes.metrics.popularity_bucket: head = 1000+ playlists


def split(g: PPR, skip: int, n: int, seed: int = 5) -> list[dict]:
    deg = np.diff(g.ps.indptr)
    ok = [i for i in np.flatnonzero((deg >= 15) & (deg <= 200))]
    ok.sort(key=lambda i: str(g.playlist_ids[i]))
    rng = random.Random(seed)
    out = []
    for p in ok[skip : skip + n]:
        songs = sorted(str(g.song_ids[i]) for i in g.playlist_songs(p))
        rng.shuffle(songs)
        cut = int(len(songs) * 0.8)
        seen = songs[:cut]
        out.append(
            {
                "playlist": str(g.playlist_ids[p]),
                "seen": seen,
                "seeds": list(reversed(seen))[:5],
                "hidden": set(songs[cut:]),
            }
        )
    return out


def two_hop(g: PPR, q: dict, k: int, cap: int | None) -> list[str]:
    """Songs ranked by how many distinct seed playlists hold them, then playlist count."""
    excl = set(g.playlists([q["playlist"]]).tolist())
    y = np.zeros(len(g.playlist_ids))
    for s in g.songs(q["seeds"]):
        pls = [p for p in g.sp.indices[g.sp.indptr[s] : g.sp.indptr[s + 1]] if p not in excl]
        if cap is not None:
            pls = sorted(pls, key=lambda p: -g.like_cnt[p])[:cap]
        y[pls] = 1.0
    co = g.sp @ y
    co[g.songs(q["seen"])] = 0.0
    cand = np.flatnonzero(co > 0)
    cand = cand[np.lexsort((g.song_ids[cand], -g.deg_p[cand], -co[cand]))][:k]
    return [str(g.song_ids[i]) for i in cand]


def popular(g: PPR, q: dict, k: int) -> list[str]:
    score = g.deg_p.copy()
    score[g.songs(q["seen"])] = -1
    top = np.argsort(-score, kind="stable")[:k]
    return [str(g.song_ids[i]) for i in top]


def score(g: PPR, qs: list[dict], lists: list[list[str]]) -> dict:
    rec = [len(q["hidden"] & set(got)) / len(q["hidden"]) for q, got in zip(qs, lists, strict=True)]
    hit = [1.0 if q["hidden"] & set(got) else 0.0 for q, got in zip(qs, lists, strict=True)]
    head = []
    for got in lists:
        d = g.deg_p[g.songs(got)]
        head.append(float((d >= HEAD).mean()) if len(d) else 0.0)
    return {
        "recall_at_k": round(float(np.mean(rec)), 4),
        "hit_at_k": round(float(np.mean(hit)), 4),
        "head_share": round(float(np.mean(head)), 4),
        "per_playlist_recall": [round(r, 4) for r in rec],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip", type=int, default=60_000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()
    g = PPR(a.graph)
    qs = split(g, a.skip, a.n)
    rows: dict[str, dict] = {}
    rows["popular"] = score(g, qs, [popular(g, q, a.k) for q in qs])
    rows["2hop, all playlists"] = score(g, qs, [two_hop(g, q, a.k, None) for q in qs])
    rows["2hop, 100 most-liked per seed"] = score(g, qs, [two_hop(g, q, a.k, 100) for q in qs])
    for w in ARTIST_WS:
        for alpha in ALPHAS:
            t = time.perf_counter()
            lists = {b: [] for b in BETAS}
            for i in range(0, len(qs), a.batch):
                chunk = qs[i : i + a.batch]
                x = g.walk([q["seeds"] for q in chunk], [[q["playlist"]] for q in chunk], alpha, w)
                for j, q in enumerate(chunk):
                    for b in BETAS:
                        lists[b].append(g.rank(x[:, j], q["seen"], a.k, b))
            for b in BETAS:
                rows[f"ppr alpha={alpha} beta={b} artist_w={w}"] = score(g, qs, lists[b])
            print(f"artist_w={w} alpha={alpha}: {time.perf_counter() - t:.0f}s", flush=True)
    # after the choice: one step of the chosen walk, with and without artist edges
    for w in ARTIST_WS:
        lists = []
        for i in range(0, len(qs), a.batch):
            chunk = qs[i : i + a.batch]
            x = g.walk(
                [q["seeds"] for q in chunk], [[q["playlist"]] for q in chunk], 0.5, w, steps=1
            )
            lists += [g.rank(x[:, j], q["seen"], a.k) for j, q in enumerate(chunk)]
        rows[f"walk 1 step, artist_w={w}"] = score(g, qs, lists)
    ppr_rows = {k: v for k, v in rows.items() if k.startswith("ppr")}
    best = max(ppr_rows, key=lambda k: ppr_rows[k]["recall_at_k"])
    out = {
        "split": {"skip": a.skip, "n": len(qs), "k": a.k, "seed": 5},
        "first_playlists": [q["playlist"] for q in qs[:5]],
        "best": best,
        "rows": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{'row':45} {'R@k':>7} {'hit':>6} {'head':>6}")
    for name, r in rows.items():
        mark = " <- best" if name == best else ""
        print(
            f"{name:45} {r['recall_at_k']:7.4f} {r['hit_at_k']:6.3f} {r['head_share']:6.3f}{mark}"
        )


if __name__ == "__main__":
    main()
