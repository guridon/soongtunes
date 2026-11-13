"""`stunes <command>` — offline jobs that build the derived stores.

stunes export                 Neo4j -> .data/exports/*.jsonl
stunes index-search           JSONL -> OpenSearch (alias swap)
stunes backfill --version V   JSONL -> embeddings in pgvector (resumable)
stunes activate --version V   point reads at an embedding version (rollback = activate old)
stunes reindex --version V    bring a finished version's vector index to the current layout
stunes graph-indexes          create the Neo4j indexes the API relies on
stunes graph-features         precompute Song.popularity (playlist count)
stunes rerank export|audit    re-ranker data: playlists split 80/10/10, leakage audit
stunes rerank stores          train-only song docs + the songs_rr_train search index
stunes rerank candidates      candidates for playlist titles (no LLM); `gate` scores them
stunes rerank features        one feature row per (title, candidate); `train` fits LambdaMART
stunes rerank serve-stats     the per-song stats the re-ranker reads in chat (feature store)
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

import orjson

from stunes.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="stunes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export")
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("index-search")
    p.add_argument("--export-dir", type=Path, default=None)

    p = sub.add_parser("backfill")
    p.add_argument("--version", required=True)
    p.add_argument("--entity", choices=["song", "tag"], default="song")
    p.add_argument("--export-dir", type=Path, default=None)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=None, help="embed only the first N rows (smoke test)"
    )
    p.add_argument("--min-popularity", type=int, default=0, help="songs in >= N playlists only")

    p = sub.add_parser("activate")
    p.add_argument("--version", required=True)

    p = sub.add_parser("reindex")
    p.add_argument("--version", required=True)

    sub.add_parser("graph-indexes")
    sub.add_parser("graph-features")

    p = sub.add_parser("ingest", help="Phase 2 experiments: throughput | chaos | report | dlq")
    p.add_argument("action", choices=["throughput", "chaos", "report", "dlq"])
    p.add_argument("--mode", default="sync")
    p.add_argument("--events", type=int, default=20_000)
    p.add_argument("--partitions", type=int, default=1)
    p.add_argument("--consumers", type=int, default=1)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rate", type=float, default=1000)
    p.add_argument("--kills", type=int, default=5)
    p.add_argument("--label", default="")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--force", action="store_true", help="replay DLQ messages even if still invalid")

    p = sub.add_parser("features", help="Phase 3: load-tags | seed | load")
    p.add_argument("action", choices=["load-tags", "seed", "load"])
    p.add_argument("--users", type=int, default=100_000)
    p.add_argument("--profile", default="C1")
    p.add_argument("--rps", type=float, default=300)
    p.add_argument("--duration", type=int, default=120)
    p.add_argument("--groups", default="user_recent_plays,user_tag_pref")
    p.add_argument("--missing", type=float, default=0.1)
    p.add_argument("--missing-pool", type=int, default=1000, help="distinct never-seeded ids")
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--warm-ttl", type=int, default=None, help="pre-load L2 with this shared TTL")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--label", default="")

    p = sub.add_parser("quality", help="Phase 7 data-quality checks: check | serve")
    p.add_argument("action", choices=["check", "serve"])
    p.add_argument("--port", type=int, default=9109)
    p.add_argument("--interval", type=int, default=60)

    p = sub.add_parser("embed-exp", help="Phase 4: embedding size vs recall vs cost")
    p.add_argument("--version", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--subset", type=int, default=100_000)
    p.add_argument("--queries", type=int, default=200)

    p = sub.add_parser(
        "vector-attribution",
        help="Phase 4: vector-only quality per version, exact vs served index, on the eval turns",
    )
    p.add_argument("--versions", required=True, help="comma-separated, e.g. A,B")
    p.add_argument("--dataset", type=Path, required=True)

    p = sub.add_parser("feed-eval", help="Phase 5: playlist-continuation eval of the home feed")
    p.add_argument("--playlists", type=int, default=200)
    p.add_argument("--k", type=int, default=50)
    p.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="multiply every source budget (client and server side); >1 measures the sources "
        "without dropping slow ones",
    )
    p.add_argument(
        "--ppr",
        action="store_true",
        help="Personalized PageRank vs co-occurrence (docs/prereg-feed-ppr.md); needs "
        "`stunes ppr-build` once",
    )

    p.add_argument(
        "--warmup", type=int, default=0, help="dev-split playlists run first, results discarded"
    )
    p.add_argument(
        "--cooccur",
        action="store_true",
        help="the feed with co-occurrence scored by count vs by 1/degree (cooccur_weights)",
    )

    p = sub.add_parser(
        "ppr-build", help="in-memory playlist graph for degree-weighted co-occurrence and PPR"
    )
    p.add_argument(
        "--src", type=Path, default=None, help="a graph_sql_prep export dir (default: Neo4j)"
    )

    p = sub.add_parser(
        "llm-tags", help="Phase 4: LLM tag features — validate | apply | project | heldout"
    )
    p.add_argument("action", choices=["validate", "apply", "project", "heldout"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=4)

    p = sub.add_parser(
        "rerank", help="re-ranker: export | audit | stores | candidates | gate | features | train"
    )
    p.add_argument(
        "action",
        choices=[
            "export",
            "audit",
            "stores",
            "candidates",
            "gate",
            "features",
            "train",
            "serve-stats",
        ],
    )
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--n", type=int, default=None, help="sample N queries (default: all)")
    p.add_argument("--ablate", action="store_true", help="train: refit without each feature group")
    p.add_argument(
        "--online",
        action="store_true",
        help="features/train: tag counts from each song's own tag weights (servable model)",
    )

    sub.add_parser("alert-drill", help="Phase 7: time from an injected problem to a firing alert")

    p = sub.add_parser("eval-resummarize", help="recompute eval summaries from predictions")
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--why", required=True)

    p = sub.add_parser(
        "eval-compare", help="paired comparison of two eval arms (<run dir>:<strategy>)"
    )
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--metric", default="mrr", choices=["mrr", "hit", "ndcg", "precision"])
    p.add_argument(
        "--all-turns",
        action="store_true",
        help="also compare turns either arm answered without some source or with an error",
    )
    p.add_argument(
        "--mark", type=Path, default=None, help="file of song ids: count turns showing one"
    )
    p.add_argument("--out", type=Path, default=None, help="also write the result as JSON")

    p = sub.add_parser("eval")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--strategies", default="hybrid,v1_contextual,v1_stateless,bm25,vector")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-conversations", type=int, default=None)
    p.add_argument("--note", default="")
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="unrecorded pass over the last N conversations per strategy before measuring",
    )
    p.add_argument(
        "--interleave",
        action="store_true",
        help="all strategies answer each turn, in a seeded random order, instead of one "
        "strategy after another",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue a stopped run in this results dir: finished conversations are kept",
    )
    p.add_argument(
        "--debug-trace",
        action="store_true",
        help="record every source's candidate ids and the pre-verify order per turn "
        "(loss map; predictions.jsonl grows to ~10 MB)",
    )

    args = parser.parse_args()
    settings = get_settings()
    exports = settings.data_dir / "exports"

    match args.cmd:
        case "export":
            from stunes.jobs.export import export_all

            asyncio.run(export_all(settings, args.out or exports))
        case "index-search":
            from stunes.jobs.index_search import build_indices

            asyncio.run(build_indices(settings, args.export_dir or exports))
        case "backfill":
            from stunes.jobs.backfill import backfill

            asyncio.run(
                backfill(
                    settings,
                    args.version,
                    args.entity,
                    args.export_dir or exports,
                    batch=args.batch,
                    concurrency=args.concurrency,
                    limit=args.limit,
                    min_popularity=args.min_popularity,
                )
            )
        case "activate":
            from stunes.embedding.registry import activate_cli

            asyncio.run(activate_cli(settings, args.version))
        case "reindex":
            from stunes.jobs.backfill import reindex

            asyncio.run(reindex(settings, args.version))
        case "graph-indexes":
            from stunes.jobs.graph_indexes import create_graph_indexes

            asyncio.run(create_graph_indexes(settings))
        case "graph-features":
            from stunes.jobs.graph_indexes import compute_graph_features

            asyncio.run(compute_graph_features(settings))
        case "ingest":
            from stunes.ingest import bench

            if args.action == "throughput":
                bench.throughput(
                    settings,
                    mode=args.mode,
                    events=args.events,
                    partitions=args.partitions,
                    consumers=args.consumers,
                    batch=args.batch,
                    workers=args.workers,
                    label=args.label,
                )
            elif args.action == "chaos":
                bench.chaos(
                    settings,
                    mode=args.mode,
                    events=args.events,
                    rate=args.rate,
                    kills=args.kills,
                    partitions=args.partitions,
                    batch=args.batch,
                    workers=args.workers,
                    label=args.label,
                )
            elif args.action == "dlq":
                print(bench.dlq(settings, replay=args.replay, force=args.force))
            else:
                bench.write_report(settings)
        case "features":
            from stunes.features import jobs as fjobs
            from stunes.features import loadgen

            if args.action == "load-tags":
                fjobs.load_song_tags(settings, exports)
            elif args.action == "seed":
                fjobs.seed_users(settings, args.users)
            else:
                users = loadgen.seeded_users(settings, args.users)
                result = asyncio.run(
                    loadgen.run(
                        settings,
                        api=args.api,
                        profile=args.profile,
                        rps=args.rps,
                        duration_s=args.duration,
                        users=users,
                        groups=args.groups.split(","),
                        zipf_s=args.zipf,
                        missing=args.missing,
                        missing_pool=args.missing_pool,
                        warm_ttl=args.warm_ttl,
                        label=args.label,
                    )
                )
                path = loadgen.save(settings, result)
                summary = {
                    k: result[k]
                    for k in ("profile", "requests", "errors", "latency_ms", "l3_qps", "hit_ratio")
                }
                print(path)
                print(summary)
        case "quality":
            from stunes import quality

            if args.action == "check":
                raise SystemExit(0 if quality.check_once(settings) else 1)
            quality.serve(settings, args.port, args.interval)
        case "embed-exp":
            from stunes.embedding import experiment

            experiment.main(settings, args.version, args.dataset, args.subset, args.queries)
        case "vector-attribution":
            from stunes.embedding.experiment import attribution

            result = asyncio.run(
                attribution(settings, versions=args.versions.split(","), dataset=args.dataset)
            )
            out = settings.results_dir / "embedding" / f"attribution-{int(time.time())}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
            print(out)
        case "rerank":
            from stunes.rerank import data as rerank_data

            path = settings.data_dir / "rerank" / "playlists.jsonl"
            if args.action == "export":
                asyncio.run(rerank_data.export(settings, path))
            elif args.action == "stores":
                from stunes.rerank import stores as rerank_stores

                out = settings.data_dir / "rerank" / "export"
                rerank_stores.write_export(
                    rerank_data.load(path), settings.data_dir / "exports" / "songs.jsonl", out
                )
                asyncio.run(rerank_stores.build_search_index(settings, out))
            elif args.action in ("candidates", "gate"):
                from stunes.rerank import candidates as rerank_cands

                cands = settings.data_dir / "rerank" / f"candidates-{args.split}.jsonl"
                if args.action == "candidates":
                    asyncio.run(rerank_cands.generate(settings, args.split, args.n, cands))
                else:
                    result = rerank_cands.gate(settings, cands)
                    print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
            elif args.action == "features":
                from stunes.rerank import features as rerank_features

                rerank_features.build(settings, args.split, online=args.online)
            elif args.action == "serve-stats":
                from stunes.rerank import serve as rerank_serve

                print(rerank_serve.load_stats(settings), rerank_serve.load_tag_songs(settings))
            elif args.action == "train":
                from stunes.rerank import train as rerank_train  # needs the `rerank` group

                result = rerank_train.train(settings, ablate=args.ablate, online=args.online)
                print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
            else:
                result = rerank_data.audit(rerank_data.load(path))
                print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
        case "feed-eval":
            from stunes.agent import feed_eval

            feed_eval.cli(
                settings.model_copy(update={"source_budget_scale": args.budget_scale}),
                args.playlists,
                args.k,
                args.ppr,
                args.warmup,
                args.cooccur,
            )
        case "ppr-build":
            from stunes.retrieval import ppr

            if args.src:
                print(ppr.build(args.src, settings.ppr_graph_file))
            else:
                from stunes.db import Graph

                async def from_graph():
                    g = Graph(settings)
                    try:
                        return await ppr.build_from_graph(g, settings.ppr_graph_file)
                    finally:
                        await g.close()

                print(asyncio.run(from_graph()))
        case "llm-tags":
            from stunes.llm_features import tagger

            if args.action == "validate":
                result = asyncio.run(
                    tagger.validate(settings, n=args.n, concurrency=args.concurrency)
                )
            elif args.action == "project":
                result = asyncio.run(tagger.project(settings))
            elif args.action == "heldout":
                result = asyncio.run(tagger.heldout(settings, n=args.n))
            else:
                result = asyncio.run(
                    tagger.apply(
                        settings, n=args.n, threshold=args.threshold, concurrency=args.concurrency
                    )
                )
            print(tagger.save(settings, args.action, result))
            print(result)
        case "alert-drill":
            from stunes import alert_drill

            alert_drill.main(settings)
        case "eval-resummarize":
            from stunes.eval.run import resummarize

            for run in args.runs:
                s = resummarize(run, args.why)
                print(
                    run,
                    {
                        k: (v["degraded_turns"], v["dropped_by_source"])
                        for k, v in s["strategies"].items()
                    },
                )
        case "eval-compare":
            from stunes.eval import compare as cmp

            result = cmp.compare(
                cmp.load_arm(*cmp.parse_arm(args.a)),
                cmp.load_arm(*cmp.parse_arm(args.b)),
                metric=args.metric,
                clean_only=not args.all_turns,
                marked=set(args.mark.read_text().split()) if args.mark else None,
            )
            print(cmp.markdown(result, args.a, args.b))
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_bytes(
                    orjson.dumps({"a": args.a, "b": args.b, **result}, option=orjson.OPT_INDENT_2)
                )
        case "eval":
            from stunes.eval.run import run_eval

            asyncio.run(
                run_eval(
                    api=args.api,
                    dataset=args.dataset,
                    strategies=[s for s in args.strategies.split(",") if s],
                    out_root=settings.results_dir / "eval",
                    k=args.k,
                    concurrency=args.concurrency,
                    max_conversations=args.max_conversations,
                    note=args.note,
                    warmup=args.warmup,
                    interleave=args.interleave,
                    debug_trace=args.debug_trace,
                    resume=args.resume,
                )
            )
