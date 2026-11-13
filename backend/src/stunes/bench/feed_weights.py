"""Tune the home feed's fusion and ranker weights offline (docs/prereg-feed-weights.md).

    python -m stunes.bench.feed_weights collect --skip 60000 --n 400 --out ../.data/feed-weights/dev.jsonl.gz
    python -m stunes.bench.feed_weights collect --skip 50000 --n 200 --out ../.data/feed-weights/test.jsonl.gz
    python -m stunes.bench.feed_weights tune --dev ../.data/feed-weights/dev.jsonl.gz --out ../results/feed/weights-tune.json
    python -m stunes.bench.feed_weights score --data ../.data/feed-weights/test.jsonl.gz --weights ../results/feed/weights-tune.json

The feed fuses four candidate lists with weighted RRF (FEED_WEIGHTS), then ranks the fused
top 200 by a linear blend of signals with an artist-repeat penalty (FEED_RANK_W,
agent.graph.select). With every source called, the blend scores below its best source alone
(results.md). `collect` runs the served feed once per playlist (the playlist-continuation split
of `stunes feed-eval`) and keeps each source's list, the preferences and the song cards. `tune`
replays the ranker on them, exactly as served, and searches the weights by coordinate ascent
(Metzler & Croft 2007, "Linear feature-based models for information retrieval") on mean
Recall@50. Scale is fixed by cooccur = 1 in RRF and fusion = 1 in the blend. A source at weight
0 is not called.
"""

import argparse
import asyncio
import gzip
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from neo4j import Query

from stunes.agent.feed_eval import PLAYLISTS, _tag_pref, paired_ci
from stunes.agent.graph import FEED_RANK_W, FEED_WEIGHTS, select
from stunes.retrieval.fusion import rrf

K = 50
GRID = {
    "rrf:tags": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    "rrf:vector": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    "rrf:popular": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    "rank:tag_pref": [0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5],
    "rank:condition_tags": [0.0, 0.1, 0.2, 0.4, 0.6, 1.0],
    "rank:popularity": [0.0, 0.05, 0.1, 0.15, 0.3, 0.5],
    "rank:artist_repeat": [0.0, -0.05, -0.1, -0.2, -0.3, -0.5, -1.0],
}
MIN_GAIN = 0.0005  # a move must raise dev mean Recall@50 by more than this


# ---------------------------------------------------------------------------- collect
async def collect(skip: int, n: int, out: Path, seed: int = 5) -> dict:
    from stunes.config import get_settings
    from stunes.services import Services

    # the served feed: degree-weighted co-occurrence; budgets x3 so no source is dropped
    settings = get_settings().model_copy(
        update={"source_budget_scale": 3.0, "cooccur_weights_feed": "degree"}
    )
    svc = await Services.build(settings)
    try:
        async with svc.stores.graph._driver.session(database=svc.settings.neo4j_database) as ses:
            res = await ses.run(Query(PLAYLISTS, timeout=600), min_songs=15, n=n, skip=skip)
            playlists = [r.data() async for r in res]
        rng = random.Random(seed)  # the same draws as stunes feed-eval on this split
        out.parent.mkdir(parents=True, exist_ok=True)
        dropped = 0
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for i, pl in enumerate(playlists):
                songs = sorted(pl["songs"])
                rng.shuffle(songs)
                cut = int(len(songs) * 0.8)
                seen, hidden = songs[:cut], songs[cut:]
                prefs = await _tag_pref(svc, seen)
                st = await svc.agent.app.ainvoke(
                    {
                        "entry": "feed",
                        "session_id": f"feed:eval_{pl['id']}",
                        "user_id": f"eval_{pl['id']}",
                        "limit": K,
                        "seed_songs": list(reversed(seen)),
                        "pref_tags": prefs,
                        "exclude_playlists": [pl["id"]],
                        "disabled_sources": [],
                        "extra_sources": {},
                        "timings": {},
                    }
                )
                ids = sorted({s for xs in st["ranked"].values() for s in xs})
                cards = await svc.bm25.get_songs(ids)
                dropped += bool(st.get("timed_out"))
                rec = {
                    "playlist": pl["id"],
                    "hidden": hidden,
                    "excluded": list(st["cond"].exclude),
                    "cond_tags": list(st["cond"].tags),
                    "prefs": st.get("pref_tags") or {},
                    "ranked": st["ranked"],
                    "timed_out": st.get("timed_out") or [],
                    "relaxed": st.get("relaxed") or [],
                    "final": st.get("final") or [],
                    "cards": {
                        sid: {
                            "tags": c.get("tags") or [],
                            "playlist_count": c.get("playlist_count") or 0,
                            "artists": c.get("artists") or [],
                        }
                        for sid, c in cards.items()
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"{i + 1}/{len(playlists)}", flush=True)
        return {
            "playlists": len(playlists),
            "with_dropped_source": dropped,
            "cooccur": svc.agent.cooccur_weights,
        }
    finally:
        await svc.close()


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------- replay
def served() -> dict[str, float]:
    w = {f"rrf:{k}": v for k, v in FEED_WEIGHTS.items() if k != "cooccur"}
    w |= {f"rank:{k}": v for k, v in FEED_RANK_W.items() if k != "fusion"}
    return w


def split_w(w: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    rrf_w = {"cooccur": FEED_WEIGHTS["cooccur"]} | {
        k[4:]: v for k, v in w.items() if k.startswith("rrf:")
    }
    rank_w = {"fusion": FEED_RANK_W["fusion"]} | {
        k[5:]: v for k, v in w.items() if k.startswith("rank:")
    }
    return rrf_w, rank_w


def replay(rec: dict, w: dict[str, float]) -> list[str]:
    rrf_w, rank_w = split_w(w)
    ranked = {s: ids for s, ids in rec["ranked"].items() if rrf_w.get(s, 0) > 0}
    fused = rrf(ranked, rrf_w)[:200]
    cards = {f.song_id: rec["cards"][f.song_id] for f in fused if f.song_id in rec["cards"]}
    chosen = select(
        fused, cards, rec["prefs"], set(rec["cond_tags"]), set(rec["excluded"]), K, rank_w
    )
    return [x["song_id"] for x in chosen][:K]


def recalls(data: list[dict], w: dict[str, float]) -> list[float]:
    return [len(set(r["hidden"]) & set(replay(r, w))) / len(r["hidden"]) for r in data]


_DATA: list[dict] = []


def _init(path: str) -> None:
    global _DATA
    _DATA = load(Path(path))


def _mean(w: dict[str, float]) -> float:
    return float(np.mean(recalls(_DATA, w)))


def profile(data: list[dict], w: dict[str, float]) -> dict:
    """Head share of the 50, and artist spread of the first 20 (what the feed shows)."""
    head, distinct, top_share = [], [], []
    for r in data:
        got = replay(r, w)
        if not got:
            continue
        head.append(np.mean([r["cards"][x]["playlist_count"] >= 1000 for x in got]))
        first = got[:20]
        counts: dict[str, int] = {}
        for x in first:
            for art in r["cards"][x]["artists"] or ["?"]:
                counts[art] = counts.get(art, 0) + 1
        distinct.append(len(counts))
        top_share.append(max(counts.values()) / len(first))
    return {
        "head_share": round(float(np.mean(head)), 4),
        "artists_in_top20": round(float(np.mean(distinct)), 2),
        "top_artist_share_top20": round(float(np.mean(top_share)), 3),
    }


def tune(dev_path: Path, max_passes: int = 4, workers: int = 8, fixed: dict | None = None) -> dict:
    fixed = fixed or {}
    w = served() | fixed
    grid_items = [(k, v) for k, v in GRID.items() if k not in fixed]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(dev_path),)) as pool:
        best = pool.submit(_mean, w).result()
        path = [{"step": "served", "weights": dict(w), "recall": round(best, 4)}]
        for p in range(max_passes):
            moved = False
            for name, grid in grid_items:
                t = time.perf_counter()
                scores = dict(
                    zip(grid, pool.map(_mean, [w | {name: v} for v in grid]), strict=True)
                )
                v, s = max(scores.items(), key=lambda kv: (kv[1], -abs(kv[0] - w[name])))
                if s > best + MIN_GAIN:
                    w[name], best, moved = v, s, True
                path.append(
                    {
                        "step": f"pass {p + 1} {name}",
                        "scores": {str(k): round(x, 4) for k, x in scores.items()},
                        "weights": dict(w),
                        "recall": round(best, 4),
                        "seconds": round(time.perf_counter() - t, 1),
                    }
                )
                print(f"pass {p + 1} {name}: {w[name]} -> dev {best:.4f}", flush=True)
            if not moved:
                break
    return {"weights": w, "fixed": fixed, "dev_recall": round(best, 4), "path": path}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--skip", type=int, required=True)
    c.add_argument("--n", type=int, required=True)
    c.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("tune")
    t.add_argument("--dev", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument(
        "--keep-artist-penalty",
        action="store_true",
        help="hold the artist-repeat penalty at its served value (the feed's diversity rule)",
    )
    s = sub.add_parser("score")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--weights", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        print(asyncio.run(collect(a.skip, a.n, a.out)))
    elif a.cmd == "tune":
        dev = load(a.dev)
        same = sum(replay(r, served()) == r["final"][:K] for r in dev)
        print(f"replay with the served weights matches the served list on {same}/{len(dev)}")
        fixed = (
            {"rank:artist_repeat": served()["rank:artist_repeat"]} if a.keep_artist_penalty else {}
        )
        res = tune(a.dev, fixed=fixed)
        res["replay_matches_served"] = f"{same}/{len(dev)}"
        a.out.write_text(json.dumps(res, indent=1))
    else:
        data = load(a.data)
        same = sum(replay(r, served()) == r["final"][:K] for r in data)
        base = recalls(data, served())
        alone = served() | {"rrf:tags": 0.0, "rrf:vector": 0.0, "rrf:popular": 0.0}
        out = {
            "playlists": len(data),
            "replay_matches_served": f"{same}/{len(data)}",
            "with_dropped_source": sum(bool(r["timed_out"]) for r in data),
            "served": {"recall": round(float(np.mean(base)), 4), **profile(data, served())},
            "cooccur alone": {
                "recall": round(float(np.mean(recalls(data, alone))), 4),
                **profile(data, alone),
            },
        }
        for path in a.weights:
            w = json.loads(path.read_text())["weights"]
            tuned = recalls(data, w)
            out[path.stem] = {
                "weights": w,
                "recall": round(float(np.mean(tuned)), 4),
                "minus served": paired_ci(tuned, base),
                **profile(data, w),
            }
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
