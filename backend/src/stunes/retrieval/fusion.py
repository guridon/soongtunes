"""Reciprocal Rank Fusion.

BM25 scores and cosine similarities live on different scales, so we fuse ranks, not scores:
    score(d) = Σ_source  weight_source / (k + rank_source(d))
k=60 is the value from the original RRF paper (Cormack et al., 2009); it damps the
advantage of being #1 in a single list over being consistently high in several.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class Fused:
    song_id: str
    score: float
    # source name -> 1-based rank in that source
    ranks: dict[str, int] = field(default_factory=dict)


def rrf(
    ranked: Mapping[str, Sequence[str]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> list[Fused]:
    fused: dict[str, Fused] = {}
    for source, ids in ranked.items():
        w = 1.0 if weights is None else weights.get(source, 1.0)
        seen: set[str] = set()
        rank = 0
        for song_id in ids:
            if song_id in seen:
                continue
            seen.add(song_id)
            rank += 1
            item = fused.setdefault(song_id, Fused(song_id, 0.0))
            item.score += w / (k + rank)
            item.ranks[source] = rank
    # ties broken by the best single-source rank, then id for determinism
    return sorted(fused.values(), key=lambda f: (-f.score, min(f.ranks.values()), f.song_id))
