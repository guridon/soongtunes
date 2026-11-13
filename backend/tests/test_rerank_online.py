import math

import numpy as np

from stunes.rerank.online import COL, FEATURES, SongText, feature_block, own_tag_count, union


def _doc(title, tags, weights, pop, year=2010):
    return {
        "title": title,
        "artists": ["아이유"],
        "album": title,
        "genres": ["발라드"],
        "tags": tags,
        "tag_weights": weights,
        "playlist_count": pop,
        "playlists": ["비 오는 날 듣는 노래"],
        "year": year,
    }


def test_own_tag_count_reads_the_songs_own_weights():
    count = own_tag_count(_doc("a", ["감성", "새벽"], [5, 2], 10))
    assert (count("감성"), count("새벽"), count("여름")) == (5, 2, 0)


def test_union_keeps_source_order_and_drops_repeats():
    sources = {"bm25": [("1", 3.0), ("2", 2.0)], "vector": [("2", 0.9), ("3", 0.8)], "tag": []}
    assert union(sources) == ["1", "2", "3"]


def test_a_candidate_in_no_source_gets_missing_source_columns():
    docs = {"1": _doc("밤편지", ["감성"], [3], 100), "9": _doc("좋은 날", ["새벽"], [1], 50)}
    text = {s: SongText(d) for s, d in docs.items()}
    sources = {"bm25": [("1", 4.0)], "vector": [], "tag": []}
    tag_row = {"감성": 0, "새벽": 1}
    tag_sims = np.array([0.5, 0.2], dtype=np.float32)
    own = {s: own_tag_count(d) for s, d in docs.items()}
    X = feature_block(
        "비 오는 날 감성",
        sources,
        [("감성", 0.8)],
        ["1", "9"],
        docs,
        text,
        None,
        {},
        None,
        tag_row,
        tag_sims,
        lambda tag, song: own[song](tag),
    )
    assert X.shape == (2, len(FEATURES))
    assert X[0, COL["bm25_rank"]] == 1 and X[0, COL["n_sources"]] == 1
    assert math.isnan(X[1, COL["bm25_rank"]]) and X[1, COL["n_sources"]] == 0
    assert math.isnan(X[0, COL["vector_cos"]])  # no query vector given
    assert X[0, COL["tag_cnt1"]] == np.float32(math.log1p(3))
    assert X[1, COL["tag_cnt1"]] == 0
    assert X[0, COL["soft_top1"]] == np.float32(0.5)
    assert X[0, COL["playlist_bigram"]] > 0  # "비오는날..." shares bigrams with the query


def test_a_source_rank_counts_entries_that_are_not_rows():
    # serving leaves out a pinned song or one without stats; a later song keeps its true rank
    docs = {"2": _doc("밤편지", ["감성"], [3], 100)}
    sources = {"bm25": [("1", 5.0), ("1", 5.0), ("2", 4.0)], "vector": [], "tag": []}
    X = feature_block(
        "밤편지",
        sources,
        [],
        ["2"],
        docs,
        {"2": SongText(docs["2"])},
        None,
        {},
        None,
        {},
        np.zeros(0, dtype=np.float32),
        lambda tag, song: 0,
    )
    assert X[0, COL["bm25_rank"]] == 2  # "1" counted once, though it is not a row
    assert X[0, COL["rrf_all"]] == np.float32(0.6 / (60 + 2))
    assert X[0, COL["bm25_rel"]] == np.float32(4.0 / 5.0)
