import math

import numpy as np

from stunes.rerank.features import (
    COL,
    FEATURES,
    _train_tag_counts,
    bigrams,
    candidate_ids,
    contains,
    decades,
    jaccard,
    norm,
)
from stunes.rerank.train import Split, order, rrf_scores
from stunes.retrieval.fusion import rrf


def test_norm_decodes_entities_and_keeps_letters_and_digits():
    assert norm("Can&#39;t Stand Still") == "cantstandstill"
    assert norm("  비 오는 날, 듣기 좋은 R&B!! ") == "비오는날듣기좋은rb"
    assert norm(None) == ""


def test_contains_needs_two_characters():
    q = norm("아이유 노래 모음")
    assert contains(q, norm("아이유")) == 1.0
    assert contains(q, norm("유")) == 0.0  # one character matches too much
    assert contains(q, "") == 0.0


def test_decades():
    assert decades("90년대 감성 발라드") == [1990]
    assert decades("2000년대 아이돌") == [2000]
    assert decades("00년대 댄스") == [2000]
    assert decades("80's pop") == [1980]
    assert decades("2019 여름") == []  # a year, not a decade
    assert decades("15년대") == []  # not a decade


def test_bigrams_and_jaccard():
    assert bigrams("") == set()
    assert bigrams("a") == {"a"}
    assert jaccard(bigrams("새벽감성"), bigrams("새벽감성")) == 1.0
    assert jaccard(set(), bigrams("x")) == 0.0


def test_candidate_ids_union_in_source_order():
    row = {
        "sources": {
            "bm25": [["a", 3.0], ["b", 2.0]],
            "vector": [["b", 0.9], ["c", 0.8]],
            "tag": [["d", None], ["a", None]],
        }
    }
    assert candidate_ids(row) == ["a", "b", "c", "d"]


def test_train_tag_counts_ignore_held_out_playlists_and_repeats():
    pls = [
        {"split": "train", "tags": ["밤", "비"], "songs": ["a", "a", "b"]},
        {"split": "train", "tags": ["밤"], "songs": ["a", "z"]},
        {"split": "test", "tags": ["밤"], "songs": ["a", "b"]},
    ]
    c = _train_tag_counts(pls, {"밤"}, {"a", "b"})
    assert c["밤"] == {"a": 2, "b": 1}  # z is not a candidate; the test playlist is not counted
    assert "비" not in c


def _split(rows: list[dict]) -> Split:
    """A Split built from candidate rows the way features.build lays them out."""
    X, ids, group = [], [], []
    for r in rows:
        cids = candidate_ids(r)
        group.append(len(cids))
        for s in cids:
            x = np.full(len(FEATURES), np.nan, dtype=np.float32)
            for name in ("bm25", "vector", "tag"):
                ranked = list(dict.fromkeys(t for t, _ in r["sources"][name]))
                if s in ranked:
                    x[COL[f"{name}_rank"]] = ranked.index(s) + 1
            X.append(x)
            ids.append(s)
    return Split(
        np.stack(X),
        np.zeros(len(ids), dtype=np.int8),
        np.array(group),
        np.array(ids),
        np.array([str(i) for i in range(len(rows))]),
        [set() for _ in rows],
    )


def test_rrf_from_rank_columns_matches_fusion_rrf():
    rows = [
        {
            "sources": {
                "bm25": [["a", 3.0], ["b", 2.0], ["a", 1.0], ["e", 0.5]],  # a repeated
                "vector": [["b", 0.9], ["c", 0.8], ["a", 0.7]],
                "tag": [["d", None], ["a", None], ["f", None]],
            }
        },
        {
            "sources": {
                "bm25": [["x", 1.0]],
                "vector": [["y", 0.5]],  # x, y tie in a 1.0/1.0 fusion: id breaks it
                "tag": [],
            }
        },
    ]
    split = _split(rows)
    for weights in (
        {"tag": 1.0, "vector": 0.5, "bm25": 0.3},
        {"vector": 1.0, "bm25": 1.0},
        {"tag": 2.0},
    ):
        got = order(split, *rrf_scores(split, weights), k=10)
        want = [
            [
                f.song_id
                for f in rrf(
                    {n: [s for s, _ in r["sources"][n]] for n in r["sources"] if n in weights},
                    weights,
                )
            ][:10]
            for r in rows
        ]
        assert got == want, weights


def test_order_drops_nan_and_breaks_ties():
    split = _split([{"sources": {"bm25": [["b", 1], ["a", 1], ["c", 1]], "vector": [], "tag": []}}])
    scores = np.array([1.0, 1.0, math.nan])
    assert order(split, scores, np.zeros(3)) == [["a", "b"]]  # tie by id; NaN dropped
    assert order(split, scores, np.array([0.0, 1.0, 0.0])) == [["b", "a"]]  # tiebreak first


def test_ablation_groups_cover_every_feature_once():
    from stunes.rerank.train import GROUPS

    grouped = [f for names in GROUPS.values() for f in names]
    assert sorted(grouped) == sorted(FEATURES)
