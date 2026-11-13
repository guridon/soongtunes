"""Re-ranker, step 4b: LambdaMART over the candidate features, judged against RRF on test.

Training queries are the val playlists, held out of the stores like test. One fifth of them, by
a hash of the playlist id, is kept apart for early stopping. A query with no relevant candidate
gives LambdaMART no pair to order, so it is left out of training; every test query is scored.

On test, over the same candidates, the model is compared with:
- the floor (the same 10 most popular train songs);
- RRF with the chat pipeline's hand-set weights (the gate's rows);
- RRF with weights tuned on val: a grid over the three sources' weights, the best val MRR.
A gain over the tuned RRF is what the features add beyond re-weighting the sources.
"""

import hashlib
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson

from stunes.config import Settings
from stunes.eval.metrics import score_turn
from stunes.rerank.candidates import FUSIONS, PRESPECIFIED, TagIndex
from stunes.rerank.data import load
from stunes.rerank.features import COL, FEATURES, RRF_K, SOURCES, features_path

logger = logging.getLogger(__name__)

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 20,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 8,
    "verbose": -1,
}
ROUNDS = 3000
PATIENCE = 100
GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
# feature groups for the ablation: each is dropped in turn and the model refitted
GROUPS = {
    "query": [f for f in FEATURES if f.startswith("q_")],
    "source ranks and scores": [
        "bm25_rank",
        "bm25_score",
        "bm25_rel",
        "vector_rank",
        "vector_cos",
        "tag_rank",
        "n_sources",
        "rrf_all",
    ],
    "linked tags, exact": [
        "tag_hits",
        "tag_cnt1",
        "tag_cnt2",
        "tag_cnt3",
        "tag_share_max",
        "tag_wsum",
    ],
    "linked tags, soft": ["soft_max", "soft_wmean", "soft_top1", "n_song_tags"],
    "popularity": ["log_pop"],
    "text and year": [
        "title_in_q",
        "artist_in_q",
        "album_in_q",
        "genre_in_q",
        "title_bigram",
        "artist_bigram",
        "playlist_bigram",
        "year",
        "decade_match",
    ],
}
METRICS = ("hit", "mrr", "ndcg", "precision")


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    song_id: np.ndarray
    playlist_id: np.ndarray
    relevant: list[set[str]]

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.group)])

    def subset(self, keep: np.ndarray) -> "Split":
        """The queries where `keep` is true, rows and all."""
        off = self.offsets
        rows = np.concatenate([np.arange(off[q], off[q + 1]) for q in np.flatnonzero(keep)])
        return Split(
            self.X[rows],
            self.y[rows],
            self.group[keep],
            self.song_id[rows],
            self.playlist_id[keep],
            [r for r, k in zip(self.relevant, keep, strict=True) if k],
        )


def load_split(base: Path, split: str, online: bool = False) -> Split:
    f = np.load(features_path(base, split, online))
    if list(f["features"]) != FEATURES:
        raise ValueError(f"features-{split}.npz has other columns; rebuild it")
    with (base / f"candidates-{split}.jsonl").open("rb") as fh:
        rows = [orjson.loads(line) for line in fh]
    if [r["playlist_id"] for r in rows] != list(f["playlist_id"]):
        raise ValueError(f"features-{split}.npz and its candidates are out of step")
    return Split(
        f["X"],
        f["y"],
        f["group"],
        f["song_id"],
        f["playlist_id"],
        [set(r["relevant"]) for r in rows],
    )


def early_stopping_query(playlist_id: str) -> bool:
    return int(hashlib.sha1(f"es:{playlist_id}".encode()).hexdigest(), 16) % 5 == 0


def order(split: Split, scores: np.ndarray, tiebreak: np.ndarray, k: int = 10) -> list[list[str]]:
    """Each query's top k by score (desc), then tiebreak (asc), then song id; NaN scores dropped."""
    out = []
    off = split.offsets
    for q in range(len(split.group)):
        a, b = off[q], off[q + 1]
        s, t, ids = scores[a:b], tiebreak[a:b], split.song_id[a:b]
        keep = ~np.isnan(s)
        idx = np.flatnonzero(keep)[np.lexsort((ids[keep], t[keep], -s[keep]))]
        out.append(list(ids[idx[:k]]))
    return out


def rrf_scores(split: Split, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """fusion.rrf over the sources with a positive weight, from the rank columns: the score, and
    the best rank among those sources (its tie-break). Songs none of them returned get NaN."""
    score = np.zeros(len(split.y))
    best = np.full(len(split.y), np.inf)
    seen = np.zeros(len(split.y), dtype=bool)
    for name in SOURCES:
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        rank = split.X[:, COL[f"{name}_rank"]].astype(np.float64)
        hit = ~np.isnan(rank)
        score[hit] += w / (RRF_K + rank[hit])
        best[hit] = np.minimum(best[hit], rank[hit])
        seen |= hit
    score[~seen] = np.nan
    return score, best


def per_query(preds: list[list[str]], split: Split) -> dict[str, np.ndarray]:
    ms = [score_turn(p, r).as_dict() for p, r in zip(preds, split.relevant, strict=True)]
    return {k: np.array([m[k] for m in ms]) for k in METRICS}


def tune_rrf(split: Split) -> tuple[dict[str, float], float]:
    best, best_mrr = None, -1.0
    for ws in itertools.product(GRID, repeat=len(SOURCES)):
        if not any(ws):
            continue
        w = dict(zip(SOURCES, ws, strict=True))
        m = per_query(order(split, *rrf_scores(split, w)), split)["mrr"].mean()
        # ties go to the earlier (smaller-weight) grid point, so the result is stable
        if m > best_mrr + 1e-12:
            best, best_mrr = w, float(m)
    return best, best_mrr


def _bootstrap(d: np.ndarray, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)]


def _fit(lgb, fit: Split, stop: Split, names: list[str]):
    cols = [COL[n] for n in names]
    d_fit = lgb.Dataset(fit.X[:, cols], fit.y, group=fit.group, feature_name=names)
    d_stop = lgb.Dataset(stop.X[:, cols], stop.y, group=stop.group, reference=d_fit)
    booster = lgb.train(
        PARAMS,
        d_fit,
        num_boost_round=ROUNDS,
        valid_sets=[d_stop],
        valid_names=["early_stopping"],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(100)],
    )
    return booster, cols


def train(settings: Settings, ablate: bool = False, online: bool = False) -> dict:
    # here, not at the top: the helpers above are tested on hosts without OpenMP (macOS)
    import lightgbm as lgb

    lgb.register_logger(logger)  # its progress lines would otherwise go to stdout, into the JSON
    base = settings.data_dir / "rerank"
    val, test = load_split(base, "val", online), load_split(base, "test", online)
    off = val.offsets
    has_pos = np.array([val.y[off[q] : off[q + 1]].any() for q in range(len(val.group))])
    es = np.array([early_stopping_query(p) for p in val.playlist_id])
    fit, stop = val.subset(has_pos & ~es), val.subset(has_pos & es)
    logger.info(
        "val: %d queries, %d with a relevant candidate; fit %d, early stopping %d",
        len(val.group),
        int(has_pos.sum()),
        len(fit.group),
        len(stop.group),
    )
    booster, _ = _fit(lgb, fit, stop, FEATURES)
    sfx = "-online" if online else ""
    booster.save_model(str(base / f"ltr-model{sfx}.txt"), num_iteration=booster.best_iteration)

    # baselines: the floor, the gate's hand-set rows, RRF tuned on all of val, and the union
    # sorted by popularity alone, ties by song id (a popularity prior without any learning)
    tuned, tuned_val_mrr = tune_rrf(val)
    popular = TagIndex(load(base / "playlists.jsonl")).popular(10)
    systems: dict[str, list[list[str]]] = {
        "floor: same popular top 10": [popular] * len(test.group)
    }
    for label, w in FUSIONS.items():
        systems[label] = order(test, *rrf_scores(test, w))
    tuned_label = "rrf tuned on val (" + ", ".join(f"{k} {v:g}" for k, v in tuned.items()) + ")"
    systems[tuned_label] = order(test, *rrf_scores(test, tuned))
    systems["candidates by popularity alone (ties: song id)"] = order(
        test, test.X[:, COL["log_pop"]].astype(np.float64), np.zeros(len(test.y))
    )
    ltr = booster.predict(test.X, num_iteration=booster.best_iteration)
    systems["lambdamart"] = order(test, ltr, np.zeros_like(ltr))

    ablations = {}
    if ablate:
        for group, drop in GROUPS.items():
            names = [f for f in FEATURES if f not in drop]
            b, cols = _fit(lgb, fit, stop, names)
            pred = b.predict(test.X[:, cols], num_iteration=b.best_iteration)
            label = f"lambdamart without {group}"
            systems[label] = order(test, pred, np.zeros_like(pred))
            ablations[label] = {"dropped": drop, "best_iteration": b.best_iteration}

    per = {name: per_query(p, test) for name, p in systems.items()}
    # per-query scores, for slicing the test set afterwards
    (base / f"test-per-query{sfx}.json").write_bytes(
        orjson.dumps(
            {
                "playlist_id": list(test.playlist_id),
                **{k: {n: m[k].tolist() for n, m in per.items()} for k in ("mrr", "hit")},
            }
        )
    )
    out: dict = {
        "val_queries": len(val.group),
        "fit_queries": len(fit.group),
        "early_stopping_queries": len(stop.group),
        "test_queries": len(test.group),
        "best_iteration": booster.best_iteration,
        "early_stopping_ndcg@10": round(float(booster.best_score["early_stopping"]["ndcg@10"]), 4),
        "tuned_rrf": {"weights": tuned, "val_mrr": round(tuned_val_mrr, 4)},
        "systems": {
            n: {k: round(float(v.mean()), 4) for k, v in m.items()} for n, m in per.items()
        },
    }
    out["differences (mean, 95% paired bootstrap)"] = {
        f"lambdamart - {b}: {k}": _bootstrap(per["lambdamart"][k] - per[b][k])
        for b in (
            tuned_label,
            PRESPECIFIED,
            "candidates by popularity alone (ties: song id)",
            "floor: same popular top 10",
        )
        for k in ("mrr", "hit")
    }
    if ablations:
        out["ablations"] = ablations
        out["ablation differences, mrr (full - without)"] = {
            label: _bootstrap(per["lambdamart"]["mrr"] - per[label]["mrr"]) for label in ablations
        }
    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    total = gain.sum() or 1.0
    out["feature_gain_share"] = {
        FEATURES[i]: round(float(gain[i] / total), 4) for i in np.argsort(-gain) if gain[i] > 0
    }
    return out
