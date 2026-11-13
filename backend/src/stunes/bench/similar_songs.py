"""Chat's "songs like this one" source: co-occurrence by count vs by 1/degree.

    python -m stunes.bench.similar_songs --out ../results/feed/similar-songs.json

A chat turn that names a song, or asks for more like the last answer, calls co-occurrence with
that song as the seed (conversation/pipeline.py). Here each query is one song from a playlist:
- the seed is one song drawn with Random(7);
- the relevant songs are the rest of that playlist;
- the playlist is cut from the graph for its own query.
The playlists are the feed test split (SKIP 50,000, 200 playlists of 15-200 songs). Both
scorings go through the served `GraphRetriever.cooccurring` with the chat's arguments (k = 100,
300 playlists per seed for count, no conditions), in random order per query.
Metrics: hit@10, P@10 and MRR@10 (the chat shows 10), Recall@100 (the candidate pool), latency.
"""

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

from neo4j import Query

from stunes.agent.feed_eval import PLAYLISTS, paired_ci
from stunes.config import get_settings
from stunes.db import Graph
from stunes.retrieval.graph import Conditions, GraphRetriever
from stunes.retrieval.ppr import PPR


def metrics(got: list[str], relevant: set[str]) -> dict:
    top = got[:10]
    first = next((i for i, s in enumerate(top) if s in relevant), None)
    return {
        "hit@10": 1.0 if first is not None else 0.0,
        "p@10": sum(s in relevant for s in top) / 10,
        "mrr@10": 0.0 if first is None else 1.0 / (first + 1),
        "recall@100": len(relevant & set(got[:100])) / len(relevant),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--skip", type=int, default=50_000)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    s = get_settings()
    g = Graph(s)
    retriever = GraphRetriever(
        g, timeout_s=30, weighted=await asyncio.to_thread(PPR, s.ppr_graph_file)
    )
    try:
        async with g._driver.session(database=s.neo4j_database) as session:
            res = await session.run(Query(PLAYLISTS, timeout=600), min_songs=15, n=a.n, skip=a.skip)
            playlists = [r.data() async for r in res]
            res = await session.run(
                Query(PLAYLISTS, timeout=600), min_songs=15, n=a.warmup, skip=60_000
            )
            warm = [r.data() async for r in res]
        rng = random.Random(7)
        order_rng = random.Random(8)

        async def one(pl: dict, weights: str) -> tuple[list[str], float]:
            seed = pl["seed"]
            t = time.perf_counter()
            rows = await retriever.cooccurring(
                [seed], Conditions(), k=100, exclude_playlists=[pl["id"]], weights=weights
            )
            return [sid for sid, _ in rows], (time.perf_counter() - t) * 1000

        for pl in warm:  # dev-split playlists, discarded
            pl["seed"] = sorted(pl["songs"])[0]
            for w in ("count", "degree"):
                await one(pl, w)
        out = {w: {"rows": [], "ms": []} for w in ("count", "degree")}
        for pl in playlists:
            songs = sorted(pl["songs"])
            pl["seed"] = rng.choice(songs)
            relevant = set(songs) - {pl["seed"]}
            arms = ["count", "degree"]
            order_rng.shuffle(arms)
            for w in arms:
                got, ms = await one(pl, w)
                out[w]["rows"].append(metrics(got, relevant))
                out[w]["ms"].append(ms)
    finally:
        await g.close()

    def p(xs, q):
        return round(sorted(xs)[max(0, math.ceil(q * len(xs)) - 1)], 1)

    summary = {
        "split": {"skip": a.skip, "n": len(playlists), "seed_rng": 7},
        "arms": {
            w: {
                **{
                    m: round(sum(r[m] for r in v["rows"]) / len(v["rows"]), 4) for m in v["rows"][0]
                },
                "p50_ms": p(v["ms"], 0.5),
                "p95_ms": p(v["ms"], 0.95),
            }
            for w, v in out.items()
        },
        "paired_degree_minus_count": {
            m: paired_ci(
                [r[m] for r in out["degree"]["rows"]], [r[m] for r in out["count"]["rows"]]
            )
            for m in ("hit@10", "p@10", "mrr@10", "recall@100")
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
