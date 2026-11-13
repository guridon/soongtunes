"""Ranking metrics with binary relevance, all computed at cutoff k.

recall/precision/MRR keep the v1 definitions (scripts_v1/run_evaluation.py) so v1 and v2
numbers are comparable; nDCG uses the standard log2 discount with an ideal list of
min(|relevant|, k) hits.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass
class TurnMetrics:
    hit: float
    mrr: float
    ndcg: float
    recall: float
    precision: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def score_turn(predicted: Sequence[str], relevant: set[str], k: int = 10) -> TurnMetrics:
    top = list(dict.fromkeys(predicted))[:k]  # de-duplicate, keep order
    if not relevant:
        return TurnMetrics(*(5 * [1.0 if not top else 0.0]))

    hits = [1 if song_id in relevant else 0 for song_id in top]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    n_hits = sum(hits)
    return TurnMetrics(
        hit=1.0 if n_hits else 0.0,
        mrr=0.0 if first is None else 1.0 / (first + 1),
        ndcg=dcg / idcg,
        recall=n_hits / len(relevant),
        precision=n_hits / len(top) if top else 0.0,
    )


def mean(rows: Sequence[TurnMetrics]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].as_dict().keys()
    return {key: sum(getattr(r, key) for r in rows) / len(rows) for key in keys}
