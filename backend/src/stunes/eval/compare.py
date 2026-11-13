"""`stunes eval-compare A B` — paired comparison of two eval arms on the same turns.

An arm is `<run dir>:<strategy>`; the two arms can come from different runs (ablation pairs,
A/A repeats) or from one interleaved run (agent vs hybrid). Turns are joined on
(conversation, turn). By default only turns that both arms answered with every source and
without an error are compared ("clean"), so a difference in dropped sources between the runs
doesn't pass for a difference between the arms; `--all-turns` keeps them.

The confidence interval is a bootstrap over conversations, not turns: a follow-up turn
depends on what the same session answered before, so turns of one conversation aren't
independent. Run a second copy of the baseline (A/A) to see how far two runs of the same
system drift apart before reading a delta as an effect.
"""

import random
import statistics
from collections import defaultdict
from pathlib import Path

import orjson

from stunes.eval.run import dropped_sources

METRICS = ("mrr", "hit", "ndcg", "precision")
ORDER = (
    "ALL",
    "specific_song_search",
    "mood_search",
    "mood_follow_up",
    "chained_mood_follow_up",
    "genre_follow_up",
    "chained_genre_follow_up",
)


def parse_arm(spec: str) -> tuple[Path, str]:
    run, sep, strategy = spec.rpartition(":")
    if not sep or not run or not strategy:
        raise ValueError(f"arm must be <run dir>:<strategy>, got {spec!r}")
    return Path(run), strategy


def load_arm(run_dir: Path, strategy: str) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with (run_dir / "predictions.jsonl").open("rb") as f:
        for line in f:
            r = orjson.loads(line)
            if r["strategy"] == strategy:
                out[(str(r["conversation_id"]), int(r["turn"]))] = r
    if not out:
        raise ValueError(f"no {strategy!r} turns in {run_dir}")
    return out


def _clean(r: dict) -> bool:
    return "error" not in r and not dropped_sources(r.get("trace"))


def _bootstrap(
    by_conv: dict[str, list[float]], n: int, seed: int
) -> tuple[float, float] | tuple[None, None]:
    """95% interval of the mean per-turn delta, resampling whole conversations."""
    convs = list(by_conv.values())
    if len(convs) < 2:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        total = count = 0.0
        for _ in range(len(convs)):
            d = convs[rng.randrange(len(convs))]
            total += sum(d)
            count += len(d)
        means.append(total / count)
    # 2.5th and 97.5th percentiles, linear interpolation over the sorted resamples
    # (indexing at int(0.975 * n) sat half a step high on both ends)
    cuts = statistics.quantiles(means, n=40, method="inclusive")
    return cuts[0], cuts[-1]


def compare(
    a: dict[tuple[str, int], dict],
    b: dict[tuple[str, int], dict],
    *,
    metric: str = "mrr",
    clean_only: bool = True,
    marked: set[str] | None = None,
    resamples: int = 2000,
    seed: int = 0,
) -> dict:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    shared = sorted(set(a) & set(b))
    # both arms must have replayed the same labelled turns, or per-type rows would mix types.
    # Checked over every shared turn, degraded ones included, on purpose: one differing label
    # means two different eval sets, whichever turns the clean filter keeps
    mismatched = [k for k in shared if a[k]["query_type"] != b[k]["query_type"]]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} turns have different query types in the two arms "
            f"(first: {mismatched[0]}): not the same eval set"
        )
    keys = [k for k in shared if _clean(a[k]) and _clean(b[k])] if clean_only else shared
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for k in keys:
        groups["ALL"].append(k)
        groups[a[k]["query_type"]].append(k)
    rows = {}
    for qt in [q for q in ORDER if q in groups] + sorted(set(groups) - set(ORDER)):
        ks = groups[qt]
        deltas: dict[str, list[float]] = defaultdict(list)
        for k in ks:
            deltas[k[0]].append(b[k]["metrics"][metric] - a[k]["metrics"][metric])
        flat = [d for ds in deltas.values() for d in ds]
        lo, hi = _bootstrap(deltas, resamples, seed)
        row = {
            "n": len(ks),
            "conversations": len(deltas),
            "a": round(sum(a[k]["metrics"][metric] for k in ks) / len(ks), 4),
            "b": round(sum(b[k]["metrics"][metric] for k in ks) / len(ks), 4),
            "delta": round(sum(flat) / len(flat), 4),
            "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "b_better": sum(d > 0 for d in flat),
            "b_worse": sum(d < 0 for d in flat),
            "top_k_changed": sum(a[k]["predicted"] != b[k]["predicted"] for k in ks),
        }
        if marked is not None:
            row["a_turns_with_marked"] = sum(bool(marked & set(a[k]["predicted"])) for k in ks)
            row["b_turns_with_marked"] = sum(bool(marked & set(b[k]["predicted"])) for k in ks)
        rows[qt] = row
    return {
        "metric": metric,
        "clean_only": clean_only,
        "shared_turns": len(shared),
        "compared_turns": len(keys),
        "excluded_turns": len(shared) - len(keys),
        "only_in_a": len(set(a) - set(b)),
        "only_in_b": len(set(b) - set(a)),
        "resamples": resamples,
        "by_type": rows,
    }


def markdown(result: dict, a_name: str, b_name: str) -> str:
    marked = any("a_turns_with_marked" in r for r in result["by_type"].values())
    head = [
        f"A = `{a_name}`, B = `{b_name}`, {result['metric']}, "
        + (
            f"{result['compared_turns']} turns answered by both with every source "
            f"({result['excluded_turns']} excluded)"
            if result["clean_only"]
            else f"all {result['compared_turns']} shared turns"
        )
        + "; 95% CI: bootstrap over conversations.",
        "",
        "| query type | n | A | B | B − A | 95% CI | B better / worse | top-k changed"
        + (" | turns with a marked song A / B |" if marked else " |"),
        "| --- | --- | --- | --- | --- | --- | --- | ---" + (" | --- |" if marked else " |"),
    ]
    for qt, r in result["by_type"].items():
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]" if r["ci95"] else "—"
        line = (
            f"| {qt} | {r['n']} | {r['a']:.4f} | {r['b']:.4f} | {r['delta']:+.4f} | {ci} "
            f"| {r['b_better']} / {r['b_worse']} | {r['top_k_changed']}"
        )
        if marked:
            line += f" | {r['a_turns_with_marked']} / {r['b_turns_with_marked']}"
        head.append(line + " |")
    return "\n".join(head) + "\n"
