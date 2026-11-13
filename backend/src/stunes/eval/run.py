"""`stunes eval` — replay the conversational eval set against the running API.

Every strategy sees the same conversations; turns of one conversation run in order on one
session, conversations run concurrently. Results land in results/eval/<run_id>/:
  predictions.jsonl  every turn (ids, latency, server timings, errors) — gitignored
  summary.json       metrics per strategy x query type, latency percentiles, run config
  summary.md         the same as a table

While a run goes, each finished conversation is appended to predictions.partial.jsonl, and
run.json holds the run's config. A run that was stopped is continued with --resume RUN_DIR:
finished conversations are kept, the rest start over from turn 1, their sessions deleted first.
"""

import asyncio
import csv
import hashlib
import logging
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import orjson

from stunes.eval import noise
from stunes.eval.metrics import TurnMetrics, mean, score_turn

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    conversation_id: str
    turn: int
    text: str
    query_type: str
    relevant: set[str]


def load_dataset(path: Path, max_conversations: int | None = None) -> list[list[Turn]]:
    if path.suffix == ".jsonl":
        return _load_context_set(path, max_conversations)
    csv.field_size_limit(sys.maxsize)  # mood/genre labels hold up to ~177k ids per cell
    convs: dict[str, list[Turn]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids = orjson.loads(row["expected_output"].replace("'", '"'))
            convs[row["conversation_id"]].append(
                Turn(
                    row["conversation_id"],
                    int(row["turn"]),
                    row["input"],
                    row["query_type"],
                    {str(i) for i in ids},
                )
            )
    ordered = [
        sorted(t, key=lambda x: x.turn) for _, t in sorted(convs.items(), key=lambda kv: int(kv[0]))
    ]
    return ordered[:max_conversations] if max_conversations else ordered


def _load_context_set(path: Path, max_conversations: int | None) -> list[list[Turn]]:
    """The computed-answer context set (eval/context_set.py). Its answers are graph conditions,
    not id lists, so turns carry no relevant set: the run's own metrics are 0 and the
    registered analysis scores the predictions (docs/prereg-context-eval.md)."""
    convs: dict[str, list[Turn]] = defaultdict(list)
    for line in path.open("rb"):
        r = orjson.loads(line)
        convs[r["conversation_id"]].append(
            Turn(r["conversation_id"], int(r["turn"]), r["text"], r["op"], set())
        )
    ordered = [sorted(t, key=lambda x: x.turn) for _, t in sorted(convs.items())]
    return ordered[:max_conversations] if max_conversations else ordered


_TRACE_KEYS = (
    "mode",
    "linked",
    "plans",
    "candidates",
    "relaxed",
    "error",
    "cypher",
    "entities",
    "rewritten",
    "context_summary",
    "timed_out",
    "slots",
    "conditions",
    # hybrid_ctx / hybrid_ctxsplit: the stored state the extractor read, and how the turn
    # changed it (replace, kept, new_topic)
    "state_context",
    "state_update",
    "prompt_chars",
    # read_back_keys: the text as typed, and the words read back from English mode
    "typed",
    "keys_read_back",
    # fix_signs: operations flipped by the message's own words
    "sign_flips",
    # only with --debug-trace: candidates per source, the order before verifying, and for the
    # agent the plain RRF order its signals started from
    "candidate_ids",
    "pre_verify",
    "rrf",
    # with RERANK_VARIANTS on: the re-rank step's timings, fallback and sampled feature rows,
    # and hybrid_s2t's sentence tags (docs/prereg-rerank-chat.md)
    "rerank",
    "s2t",
)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 1)


async def _post_turn(
    client: httpx.AsyncClient, strategy: str, session: str, t: Turn, k: int, debug: bool = False
) -> list[dict]:
    """The served answer's record, then one record per other order the server returned in
    trace.variants, as strategy "<strategy>@<variant>" (same turn, same candidates).

    A strategy "<server strategy>~<kind>" sends the turn with eval/noise.py's edit of that kind,
    in a session of its own (docs/prereg-typo-robustness.md). "<server strategy>~fresh" sends
    every turn in a new session, with no history or state (docs/prereg-context-eval.md)."""
    server_strategy, _, kind = strategy.partition("~")
    text = t.text
    record: dict = {
        "strategy": strategy,
        "conversation_id": t.conversation_id,
        "turn": t.turn,
        "query_type": t.query_type,
    }
    if kind == "fresh":
        # no context at all: every turn in a session of its own (docs/prereg-context-eval.md)
        session = f"{session}.fresh{t.turn}"
    elif kind:
        key = f"{t.conversation_id}:{t.turn}"
        n = noise.apply(kind, t.text, key, noise.load_targets().get(key))
        text, session = n.text, f"{session}.{kind}"  # the API allows [\w\-:.]
        record |= {"input_clean": t.text, "noise": {"applied": n.applied, "detail": n.detail}}
    record["input"] = text
    started = time.perf_counter()
    variants: dict[str, list[str]] = {}
    try:
        resp = await client.post(
            "/api/chat",
            json={
                "session_id": session,
                "message": text,
                "strategy": server_strategy,
                "limit": k,
                "explain": False,
                "debug": debug,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        predicted = [s["song_id"] for s in body["songs"]]
        record |= {
            "predicted": predicted,
            "server_ms": body.get("trace", {}).get("timings_ms", {}),
            "trace": {key: body.get("trace", {}).get(key) for key in _TRACE_KEYS},
        }
        variants = body.get("trace", {}).get("variants") or {}
    except Exception as e:  # a failed turn scores 0, it is not dropped
        predicted = []
        record |= {"predicted": [], "error": repr(e)[:300]}
    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    if kind == "fresh":
        try:
            await client.delete(f"/api/sessions/{session}", params={"strategy": server_strategy})
        except Exception:
            pass
    record["metrics"] = score_turn(predicted, t.relevant, k).as_dict()
    record["n_relevant"] = len(t.relevant)
    out = [record]
    for name, ids in variants.items():
        if name == "rrf":  # what was served: the record above
            continue
        parent = record["trace"]
        out.append(
            {
                **{key: record[key] for key in ("conversation_id", "turn", "query_type", "input")},
                "strategy": f"{strategy}@{name}",
                "variant_of": strategy,
                "predicted": ids,
                # one request served every order: this is that request's latency, not the
                # re-rank step's (trace.rerank.step_ms of the parent record has that)
                "latency_ms": record["latency_ms"],
                "latency_of": strategy,
                # what compare's filters read (dropped sources), and the step's fallback
                "trace": {
                    "mode": parent.get("mode"),
                    "plans": parent.get("plans"),
                    "timed_out": parent.get("timed_out"),
                    "rerank": {"fallback": (parent.get("rerank") or {}).get("fallback")},
                },
                "metrics": score_turn(ids, t.relevant, k).as_dict(),
                "n_relevant": len(t.relevant),
            }
        )
    return out


async def _end_sessions(client: httpx.AsyncClient, session: str, strategies: list[str]) -> None:
    for strategy in strategies:
        server_strategy, _, kind = strategy.partition("~")
        sid = f"{session}.{kind}" if kind else session
        try:
            await client.delete(f"/api/sessions/{sid}", params={"strategy": server_strategy})
        except Exception:
            pass


async def _run_conversation(
    client: httpx.AsyncClient,
    strategy: str,
    run_id: str,
    conv: list[Turn],
    k: int,
    debug: bool = False,
) -> list[dict]:
    session = f"eval-{run_id}-{conv[0].conversation_id}"
    try:
        out = []
        for t in conv:
            out += await _post_turn(client, strategy, session, t, k, debug)
        return out
    finally:
        await _end_sessions(client, session, [strategy])


async def _run_interleaved(
    client: httpx.AsyncClient,
    strategies: list[str],
    run_id: str,
    conv: list[Turn],
    k: int,
    debug: bool = False,
) -> list[dict]:
    """Every strategy answers each turn before the conversation moves on, in a seeded random
    order per turn: they see the same cache state at nearly the same moment, and neither is
    always the one that warms the stores for the other. Sessions are per strategy server-side
    ("{strategy}:{session}"), so the conversations' states stay separate."""
    session = f"eval-{run_id}-{conv[0].conversation_id}"
    rng = random.Random(f"{run_id}-{conv[0].conversation_id}")
    out = []
    try:
        for t in conv:
            order = list(strategies)
            rng.shuffle(order)
            for strategy in order:
                out += await _post_turn(client, strategy, session, t, k, debug)
    finally:
        await _end_sessions(client, session, strategies)
    return out


PARTIAL = "predictions.partial.jsonl"


def finished(records: list[dict], convs: list[list[Turn]], strategies: list[str]) -> set[str]:
    """Conversations every strategy answered on every turn. A stopped run keeps only these: a
    conversation cut midway starts over, since its stored state may hold a turn with no record."""
    n = Counter(
        (r["strategy"], str(r["conversation_id"])) for r in records if r["strategy"] in strategies
    )
    return {
        str(c[0].conversation_id)
        for c in convs
        if all(n[(s, str(c[0].conversation_id))] == len(c) for s in strategies)
    }


def _resume_state(
    out_dir: Path, dataset: Path, strategies: list[str], convs: list[list[Turn]]
) -> tuple[dict, list[dict], set[str]]:
    run = orjson.loads((out_dir / "run.json").read_bytes())
    if run["dataset_sha256"] != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise SystemExit(f"{out_dir}: run.json is for another dataset")
    if run["strategies"] != strategies:
        raise SystemExit(f"{out_dir}: run.json has strategies {run['strategies']}")
    partial = out_dir / PARTIAL
    records = (
        [orjson.loads(line) for line in partial.open("rb") if line.strip()]
        if partial.exists()
        else []
    )
    done = finished(records, convs, strategies)
    kept = [r for r in records if str(r["conversation_id"]) in done]
    partial.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in kept))
    return run, kept, done


def dropped_sources(trace: dict | None) -> list[str]:
    """Sources a turn was answered without. The agent drops a source that runs past its budget
    (`timed_out`: "cooccur:TimeoutError"); the hybrid pipeline records a source whose query
    failed, in practice the server-side timeout, as `plans["cooccur_error"]`. Counting only the
    first hid the hybrid's drops (180 of 1,190 turns in run 20261031T120654Z)."""
    t = trace or {}
    out = {x.split(":")[0] for x in t.get("timed_out") or []}
    out |= {k.removesuffix("_error") for k in (t.get("plans") or {}) if k.endswith("_error")}
    return sorted(out)


def _summarize(records: list[dict]) -> dict:
    by: dict[tuple[str, str], list[TurnMetrics]] = defaultdict(list)
    lat: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    # turns answered without some source (dropped_sources)
    degraded: dict[str, int] = defaultdict(int)
    by_source: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        m = TurnMetrics(**r["metrics"])
        by[(r["strategy"], r["query_type"])].append(m)
        by[(r["strategy"], "ALL")].append(m)
        lat[r["strategy"]].append(r["latency_ms"])
        errors[r["strategy"]] += "error" in r
        dropped = dropped_sources(r.get("trace"))
        degraded[r["strategy"]] += bool(dropped)
        by_source[r["strategy"]].update(dropped)
    strategies = sorted({s for s, _ in by})
    out = {}
    for s in strategies:
        out[s] = {
            "metrics": {
                qt: {"n": len(rows), **{k: round(v, 4) for k, v in mean(rows).items()}}
                for (st, qt), rows in sorted(by.items())
                if st == s
            },
            "latency_ms": {
                "p50": _pct(lat[s], 0.5),
                "p95": _pct(lat[s], 0.95),
                "mean": round(statistics.fmean(lat[s]), 1) if lat[s] else None,
            },
            "errors": errors[s],
            "degraded_turns": degraded[s],
            "dropped_by_source": dict(sorted(by_source[s].items())),
            "turns": len(lat[s]),
        }
    return out


def _markdown(summary: dict) -> str:
    lines = [
        f"# Eval {summary['run_id']}",
        "",
        f"- dataset: `{summary['dataset']}` "
        f"(sha256 {summary['dataset_sha256'][:12]}), k={summary['k']}",
        f"- commit: `{summary['commit']}`, models: {summary['models']}",
        f"- conversations: {summary['conversations']}, concurrency: {summary.get('concurrency')}, "
        f"warm-up: {summary.get('warmup_conversations', 0)}, "
        f"order: {summary.get('order', 'one strategy at a time')}"
        + (f", note: {summary['note']}" if summary.get("note") else ""),
        "",
    ]
    lines += [
        "| strategy | query type | n | hit@k | MRR | nDCG@k | P@k | R@k |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s, body in summary["strategies"].items():
        for qt, m in body["metrics"].items():
            lines.append(
                f"| {s} | {qt} | {m['n']} | {m['hit']:.3f} | {m['mrr']:.3f} | "
                f"{m['ndcg']:.3f} | {m['precision']:.3f} | {m['recall']:.5f} |"
            )
    lines += [
        "",
        "| strategy | p50 ms | p95 ms | errors / turns | turns with a dropped source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s, body in summary["strategies"].items():
        lat = body["latency_ms"]
        drops = ", ".join(f"{k} {v}" for k, v in (body.get("dropped_by_source") or {}).items())
        lines.append(
            f"| {s} | {lat['p50']} | {lat['p95']} | {body['errors']} / {body['turns']} | "
            f"{body.get('degraded_turns', 0)}{f' ({drops})' if drops else ''} |"
        )
    if summary.get("resummarized"):
        lines += ["", f"Re-summarized from predictions.jsonl: {summary['resummarized']}"]
    return "\n".join(lines) + "\n"


def _head() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def resummarize(run_dir: Path, why: str) -> dict:
    """Recompute a finished run's summary from its predictions (same turns, current
    aggregation). The original run commit stays; the re-summary is recorded next to it."""
    summary = orjson.loads((run_dir / "summary.json").read_bytes())
    with (run_dir / "predictions.jsonl").open("rb") as f:
        records = [orjson.loads(line) for line in f if line.strip()]
    summary["strategies"] = _summarize(records)
    summary["resummarized"] = {
        "commit": _head(),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "why": why,
    }
    (run_dir / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    (run_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


async def run_eval(
    *,
    api: str,
    dataset: Path,
    strategies: list[str],
    out_root: Path,
    k: int = 10,
    concurrency: int = 8,
    max_conversations: int | None = None,
    note: str = "",
    warmup: int = 0,
    interleave: bool = False,
    debug_trace: bool = False,
    resume: Path | None = None,
) -> Path:
    convs = load_dataset(dataset, max_conversations)
    records: list[dict] = []
    resumed: list[str] = []
    if resume is not None:
        out_dir, run_id = resume, resume.name
        run, records, done = _resume_state(out_dir, dataset, strategies, convs)
        commit, resumed = run["commit"], [*run.get("resumed", []), datetime.now(UTC).isoformat()]
        if _head() != commit:
            logger.warning("resuming %s at %s, the run started at %s", run_id, _head(), commit)
        todo = [c for c in convs if str(c[0].conversation_id) not in done]
        logger.info("resume %s: %d conversations kept, %d to run", run_id, len(done), len(todo))
    else:
        # HEAD when the run starts: committing during a long run must not relabel it
        commit = _head()
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = out_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run = {"commit": commit, "dataset": str(dataset), "k": k, "interleave": interleave,
               "strategies": strategies, "max_conversations": max_conversations,
               "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}  # fmt: skip
        todo = convs
    (out_dir / "run.json").write_bytes(
        orjson.dumps({**run, "resumed": resumed}, option=orjson.OPT_INDENT_2)
    )
    partial = (out_dir / PARTIAL).open("ab")

    def keep(rs: list[dict]) -> None:
        records.extend(rs)
        partial.write(b"".join(orjson.dumps(r) + b"\n" for r in rs))
        partial.flush()

    async with httpx.AsyncClient(base_url=api, timeout=180) as client:
        health = (await client.get("/api/health")).json()
        for c in todo if resume is not None else []:
            # a conversation cut midway left its sessions behind: start it with none
            await _end_sessions(client, f"eval-{run_id}-{c[0].conversation_id}", strategies)
        if warmup:
            # Unrecorded pass over the *last* conversations, every strategy: whichever strategy
            # ran first used to pay for cold stores (agent drop rate 48-68% in the first
            # quarter of a run, 0-4% in the last). The last ones, so the measured pass reaches
            # them long after any prompt-prefix cache from the warm-up has turned over.
            t0 = time.monotonic()
            for strategy in strategies:
                sem = asyncio.Semaphore(concurrency)

                async def warm(conv, strategy=strategy, sem=sem):
                    async with sem:
                        await _run_conversation(
                            client, strategy, f"{run_id}-warmup", conv, k, debug_trace
                        )

                await asyncio.gather(*(warm(c) for c in convs[-warmup:]))
            logger.info(
                "warm-up: last %d conversations x %d strategies, %.0fs",
                warmup,
                len(strategies),
                time.monotonic() - t0,
            )
        if interleave:
            sem = asyncio.Semaphore(concurrency)
            started = time.monotonic()

            async def one_interleaved(conv):
                async with sem:
                    return await _run_interleaved(client, strategies, run_id, conv, k, debug_trace)

            n_done = 0
            for fut in asyncio.as_completed([one_interleaved(c) for c in todo]):
                keep(await fut)
                n_done += 1
                if n_done % 50 == 0:
                    logger.info(
                        "interleaved %s: %d/%d conversations (%.0fs)",
                        ",".join(strategies),
                        n_done,
                        len(todo),
                        time.monotonic() - started,
                    )
            logger.info("interleaved run finished in %.0fs", time.monotonic() - started)
        for strategy in [] if interleave else strategies:
            sem = asyncio.Semaphore(concurrency)
            started = time.monotonic()

            async def one(conv, strategy=strategy, sem=sem):
                async with sem:
                    return await _run_conversation(client, strategy, run_id, conv, k, debug_trace)

            n_done = 0
            for fut in asyncio.as_completed([one(c) for c in todo]):
                keep(await fut)
                n_done += 1
                if n_done % 50 == 0:
                    logger.info(
                        "%s: %d/%d conversations (%.0fs)",
                        strategy,
                        n_done,
                        len(todo),
                        time.monotonic() - started,
                    )
            logger.info("%s finished in %.0fs", strategy, time.monotonic() - started)

    partial.close()
    with (out_dir / "predictions.jsonl").open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")
    (out_dir / PARTIAL).unlink()
    summary = {
        "run_id": run_id,
        "note": note,
        "dataset": dataset.name,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "conversations": len(convs),
        "concurrency": concurrency,
        "warmup_conversations": warmup,
        "order": "interleaved per turn, seeded shuffle" if interleave else "one strategy at a time",
        "k": k,
        "commit": commit,
        "resumed": resumed,
        "models": {
            "embedding_versions": health.get("embedding_versions"),
            "retrieval": health.get("retrieval"),
        },
        "strategies": _summarize(records),
    }
    (out_dir / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    (out_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    logger.info("wrote %s", out_dir)
    return out_dir
