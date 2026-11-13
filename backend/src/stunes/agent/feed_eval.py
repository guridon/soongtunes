"""Home-feed evaluation: playlist continuation.

For N playlists, the first 80% of songs play the role of the user's recent plays (seeds and
tag preferences are built from them), the last 20% are hidden, and the feed has to find them.
The playlist itself is excluded from co-occurrence expansion — otherwise it would be the
bridge straight to its own hidden songs (spec: 평가 누수).

Reported: Recall@k and hit@k overall and with each source switched off (ablation), plus the
share of head songs (1000+ playlists) in what was recommended (popularity bias). Every row also
counts the requests in which a source was dropped (past its budget, or failed — the count is
keyed `source:ExceptionType`): a row with drops is partly measured without that source, so
`--budget-scale` exists to measure with none.

`--ppr` runs the Personalized PageRank comparison instead (docs/prereg-feed-ppr.md): PPR added
to the sources, PPR in place of co-occurrence, and each of the two alone, on the same
playlists, with paired 95% intervals over playlists.
"""

import asyncio
import logging
import math
import random
import time
from collections import Counter, defaultdict

import orjson
from neo4j import Query

from stunes.metrics import popularity_bucket
from stunes.retrieval import ppr as ppr_mod
from stunes.services import Services

logger = logging.getLogger(__name__)

PLAYLISTS = """
MATCH (p:Playlist)
WITH p, COUNT { (p)-[:INCLUDES]->() } AS n
WHERE n >= $min_songs AND n <= 200
WITH p ORDER BY p.playlist_id SKIP $skip LIMIT $n
RETURN p.playlist_id AS id, [(p)-[:INCLUDES]->(s:Song) | s.song_id] AS songs
"""


async def _tag_pref(svc: Services, songs: list[str]) -> dict[str, float]:
    rows = await svc.stores.pg.fetch(
        "SELECT song_id, tag, weight FROM song_tag WHERE song_id = ANY($1)", songs
    )
    scores: dict[str, float] = defaultdict(float)
    for r in rows:
        scores[r["tag"]] += math.log1p(r["weight"])
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:20]
    total = sum(v for _, v in top) or 1.0
    return {t: v / total for t, v in top}


PPR = {"ppr": 1.0}  # the RRF weight co-occurrence has (FEED_WEIGHTS), fixed before the run
PPR_CONFIGS = {  # name -> (sources switched off, sources added)
    "all": ([], {}),
    "all + ppr": ([], PPR),
    "-cooccur + ppr": (["cooccur"], PPR),
    "-cooccur": (["cooccur"], {}),
    "only cooccur": (["tags", "vector", "popular"], {}),
    "only ppr": (["cooccur", "tags", "vector", "popular"], PPR),
    "only popular": (["cooccur", "tags", "vector"], {}),
    # where the gain of "only ppr" comes from (secondary in the prereg)
    "only ppr, 1 step": (["cooccur", "tags", "vector", "popular"], {"ppr_1step": 1.0}),
    "only ppr, playlists only": (["cooccur", "tags", "vector", "popular"], {"ppr_playlists": 1.0}),
}
PPR_PAIRS = (  # (a, b): a - b, registered in docs/prereg-feed-ppr.md
    ("all + ppr", "all"),
    ("-cooccur + ppr", "all"),
    ("only ppr", "only cooccur"),
    ("only ppr", "only ppr, 1 step"),  # secondary: multi-hop paths
    ("only ppr", "only ppr, playlists only"),  # secondary: artist edges
    ("only ppr, 1 step", "only cooccur"),  # secondary: 1/degree weights vs distinct counts
)


# `--cooccur`: the served feed with co-occurrence scored by count vs by 1/degree
# (settings.cooccur_weights); name -> (sources off, sources added, co-occurrence scoring)
COOCCUR_CONFIGS = {
    "all, count": ([], {}, "count"),
    "all, degree": ([], {}, "degree"),
    "only cooccur, count": (["tags", "vector", "popular"], {}, "count"),
    "only cooccur, degree": (["tags", "vector", "popular"], {}, "degree"),
}
COOCCUR_PAIRS = (("all, degree", "all, count"), ("only cooccur, degree", "only cooccur, count"))


def paired_ci(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> dict:
    """Mean of a - b per playlist, and its 95% percentile bootstrap interval over playlists."""
    d = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return {
        "diff": round(sum(d) / len(d), 4),
        "ci95": [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)],
        "a_better": sum(x > 0 for x in d),
        "b_better": sum(x < 0 for x in d),
    }


async def run(
    svc: Services,
    n: int = 200,
    k: int = 50,
    seed: int = 5,
    ablations: tuple[str, ...] = ("cooccur", "tags", "vector", "popular"),
    ppr: bool = False,
    cooccur: bool = False,
    skip: int = 50_000,  # the test split; 60,000 is the PPR dev split (stunes.bench.ppr_dev)
) -> dict:
    async with svc.stores.graph._driver.session(database=svc.settings.neo4j_database) as session:
        result = await session.run(Query(PLAYLISTS, timeout=600), min_songs=15, n=n, skip=skip)
        playlists = [r.data() async for r in result]
    rng = random.Random(seed)
    graph_path = svc.settings.ppr_graph_file
    served_weights = svc.agent.cooccur_weights
    if ppr or cooccur:
        if svc.agent.graph.weighted is None:
            svc.agent.graph.weighted = await asyncio.to_thread(ppr_mod.PPR, graph_path)
        svc.agent.ppr = svc.agent.graph.weighted
    if ppr:
        configs = {name: (*v, None) for name, v in PPR_CONFIGS.items()}
        pairs = PPR_PAIRS
    elif cooccur:
        configs, pairs = COOCCUR_CONFIGS, COOCCUR_PAIRS
    else:
        configs = {"all": ([], {}, None), **{f"-{a}": ([a], {}, None) for a in ablations}}
        pairs = ()
    stats = {
        name: {
            "recall": [],
            "hit": [],
            "head": [],
            "ms": [],
            "dropped": Counter(),
            "turns": 0,
            "dropped_at": [],  # per playlist: did a source drop (pair on clean playlists)
        }
        for name in configs
    }
    for i, pl in enumerate(playlists):
        songs = sorted(pl["songs"])  # DB order isn't stable; sort, then a seeded shuffle
        rng.shuffle(songs)  # the graph keeps no play order; a fixed-seed shuffle is the timeline
        cut = int(len(songs) * 0.8)
        seen, hidden = songs[:cut], set(songs[cut:])
        prefs = await _tag_pref(svc, seen)
        for name, (off, extra, weights) in configs.items():
            # one process, both scorings; requests run one at a time
            svc.agent.cooccur_weights = weights or served_weights
            t = time.perf_counter()
            res = await svc.agent.feed(
                f"eval_{pl['id']}",
                limit=k,
                seed_songs=list(reversed(seen)),
                pref_tags=prefs,
                exclude_playlists=[pl["id"]],
                disabled_sources=off,
                extra_sources=extra,
            )
            got = [s["song_id"] for s in res.songs]
            # "cooccur:TimeoutError" = past its budget; another type = the source failed
            dropped = set(res.trace.get("timed_out") or [])
            stats[name]["dropped"].update(dropped)
            stats[name]["turns"] += bool(dropped)
            stats[name]["dropped_at"].append(bool(dropped))
            stats[name]["ms"].append((time.perf_counter() - t) * 1000)
            stats[name]["recall"].append(len(hidden & set(got)) / len(hidden))
            stats[name]["hit"].append(1.0 if hidden & set(got) else 0.0)
            heads = [
                popularity_bucket(int(s.get("playlist_count") or 0)) == "head" for s in res.songs
            ]
            stats[name]["head"].append(sum(heads) / len(heads) if heads else 0.0)
        if (i + 1) % 25 == 0:
            logger.info("%d/%d playlists", i + 1, len(playlists))

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    def p95(xs):  # nearest-rank
        return round(sorted(xs)[max(0, math.ceil(0.95 * len(xs)) - 1)], 1) if xs else None

    svc.agent.cooccur_weights = served_weights
    health = await svc.health()
    extra = {}
    if pairs:
        extra = {
            "paired_recall": {
                f"{a} - {b}": paired_ci(stats[a]["recall"], stats[b]["recall"]) for a, b in pairs
            },
            "playlist_ids": [pl["id"] for pl in playlists],
        }
    if ppr:
        extra |= {
            "ppr": {
                "alpha": ppr_mod.ALPHA,
                "beta": ppr_mod.BETA,
                "artist_w": ppr_mod.ARTIST_W,
                "tol": ppr_mod.TOL,
                "rrf_weight": PPR["ppr"],
                "graph": str(graph_path),
                "includes": int(svc.agent.ppr.sp.nnz),
            },
        }
    if cooccur:
        extra |= {"cooccur_compare": {"graph": str(graph_path)}}
    return {
        "playlists": len(playlists),
        "skip": skip,
        "k": k,
        "seed": seed,
        "embedding_versions": health["embedding_versions"],
        "retrieval": health["retrieval"],
        "results": {
            name: {
                "recall_at_k": mean(v["recall"]),
                "hit_at_k": mean(v["hit"]),
                "head_share": mean(v["head"]),
                "p95_e2e_ms": p95(v["ms"]),
                "requests_with_dropped_source": v["turns"],
                "dropped_by_source": dict(sorted(v["dropped"].items())),
                **(
                    {
                        "per_playlist_recall": [round(x, 4) for x in v["recall"]],
                        "per_playlist_dropped": v["dropped_at"],
                    }
                    if pairs
                    else {}
                ),
            }
            for name, v in stats.items()
        },
        **extra,
    }


def save(svc: Services, result: dict) -> str:
    out = svc.settings.results_dir / "feed"
    out.mkdir(parents=True, exist_ok=True)
    tag = "ppr-" if "ppr" in result else "cooccur-" if "cooccur_compare" in result else ""
    path = out / f"continuation-{tag}{int(time.time())}.json"
    path.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    lines = [
        f"# 피드: 플레이리스트 이어 담기 (플레이리스트 {result['playlists']}개, k={result['k']})",
        "",
        "- 플레이리스트마다 앞 80% = 사용자의 재생, 뒤 20%는 가림",
        "- 해당 플레이리스트는 동시등장 확장에서 제외 (누수 없음)",
        "",
        f"서비스 경로 버전 {result['embedding_versions']}, 검색 설정 {result['retrieval']}",
        "",
        "| 소스 | Recall@k | hit@k | 헤드 곡 비중 | 종단 간 p95 ms | 소스가 빠진 요청 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in result["results"].items():
        drops = ", ".join(f"{s} {n}" for s, n in r["dropped_by_source"].items())
        lines.append(
            f"| {name} | {r['recall_at_k']} | {r['hit_at_k']} | {r['head_share']} | "
            f"{r['p95_e2e_ms']} | {r['requests_with_dropped_source']}{f' ({drops})' if drops else ''} |"
        )
    if "paired_recall" in result:
        lines += [
            "",
            "| Recall@k, a − b | 평균 | 95% 신뢰구간 | a 우세 / b 우세 |",
            "| --- | --- | --- | --- |",
        ]
        for pair, c in result["paired_recall"].items():
            lines.append(
                f"| {pair} | {c['diff']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                f"{c['a_better']} / {c['b_better']} |"
            )
    # README.md keeps the source ablation
    name = "ppr.md" if "ppr" in result else "cooccur.md" if "cooccur_compare" in result else ""
    name = name or "README.md"
    (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


async def main(
    settings, n: int, k: int, ppr: bool = False, warmup: int = 0, cooccur: bool = False
) -> None:
    svc = await Services.build(settings)
    try:
        if warmup:  # dev-split playlists, results discarded: caches warm before the test split
            await run(svc, n=warmup, k=k, ppr=ppr, cooccur=cooccur, skip=60_000)
        result = await run(svc, n=n, k=k, ppr=ppr, cooccur=cooccur)
        result["warmup_playlists"] = warmup
        print(save(svc, result))
        print(orjson.dumps(result["results"], option=orjson.OPT_INDENT_2).decode())
    finally:
        await svc.close()


def cli(
    settings, n: int, k: int, ppr: bool = False, warmup: int = 0, cooccur: bool = False
) -> None:
    asyncio.run(main(settings, n, k, ppr, warmup, cooccur))
