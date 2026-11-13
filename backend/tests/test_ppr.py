import gzip

import numpy as np
import pytest

from stunes.retrieval.ppr import PPR, build


def _csv(path, header, rows):
    with gzip.open(path, "wt") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


@pytest.fixture
def graph(tmp_path):
    # playlist 1: songs 0 1 2; playlist 2: songs 2 3; playlist 3 (held out): songs 0 4
    _csv(tmp_path / "songs.csv.gz", "song_id,popularity", [(i, 0) for i in range(5)])
    _csv(tmp_path / "playlists.csv.gz", "playlist_id,like_cnt", [(1, 5), (2, 9), (3, 1)])
    _csv(tmp_path / "artists.csv.gz", "artist_id", [(7,)])
    _csv(
        tmp_path / "includes.csv.gz",
        "playlist,song",
        [(1, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (3, 4)],
    )
    _csv(tmp_path / "performed_by.csv.gz", "song,artist", [(0, 7), (3, 7)])
    out = tmp_path / "g.npz"
    build(tmp_path, out)
    return PPR(out)


def test_mass_and_batch(graph):
    x = graph.walk([["0"], ["2"]], [[], ["3"]], alpha=0.3, tol=1e-6)
    assert np.allclose(x.sum(axis=0), 1.0)
    single = graph.walk([["2"]], [["3"]], alpha=0.3, tol=1e-6)[:, 0]
    assert np.allclose(x[:, 1], single)


def test_held_out_playlist_is_cut(graph):
    # song 4 is reachable only through playlist 3
    kept = graph.walk([["0"]], [[]], alpha=0.3, tol=1e-6)[:, 0]
    cut = graph.walk([["0"]], [["3"]], alpha=0.3, tol=1e-6)[:, 0]
    assert kept[4] > 0
    assert cut[4] == 0
    assert set(graph.top(["0"], ["3"], ["0"], k=10)) == {"1", "2", "3"}


def test_artist_edges_reach_without_playlists(graph):
    # song 3 is two playlists from song 0, one artist away
    no_art = graph.walk([["0"]], [["3"]], alpha=0.5, artist_w=0.0, tol=1e-6)[:, 0]
    art = graph.walk([["0"]], [["3"]], alpha=0.5, artist_w=0.5, tol=1e-6)[:, 0]
    assert art[3] > no_art[3]


def test_unknown_ids_are_dropped(graph):
    assert graph.top(["999", "x"], [], [], k=5) == []


def test_degree_weighted_cooccur(graph):
    # seed 0 without playlist 3: 0.8 to playlist 1 (3 songs), 0.2 to artist 7 (songs 0 and 3)
    rows = dict(graph.cooccur(["0"], ["3"], ["0"], k=10))
    assert set(rows) == {"1", "2", "3"}
    assert rows["1"] == pytest.approx(rows["2"])
    assert rows["1"] / rows["3"] == pytest.approx((0.8 / 3) / (0.2 / 2))


def test_retriever_uses_weighted_and_hard_filters(graph):
    import asyncio

    from stunes.retrieval.graph import Conditions, GraphRetriever

    class FakeGraph:
        async def read(self, query, **params):
            if "AS co" in query:  # the Cypher count path
                return [{"song_id": "from-cypher", "co": 1}]
            if "tag_hits" in query:  # song 3 carries the condition's tag
                return [{"song_id": i, "tag_hits": int(i == "3")} for i in params["ids"]]
            # the verifier: song 2 breaks a condition
            return [{"song_id": "2", "reasons": ["genre"]}] if "2" in params["ids"] else []

    g = GraphRetriever(FakeGraph(), weighted=graph)

    def run(c, **kw):
        rows = asyncio.run(
            g.cooccurring(["0"], c, k=10, exclude_playlists=["3"], weights="degree", **kw)
        )
        return [sid for sid, _ in rows]

    assert run(Conditions(genres=["x"])) == ["1", "3"]  # by weight, song 2 filtered
    assert run(Conditions(genres=["x"], tags=["t"])) == ["3", "1"]  # tag hits first
    assert run(Conditions(genres=["x"], tags=["t"]), order_by_tags=False) == ["1", "3"]
    # auto: count for one seed, degree from two
    one = asyncio.run(g.cooccurring(["0"], Conditions(), weights="auto"))
    two = asyncio.run(g.cooccurring(["0", "1"], Conditions(), weights="auto"))
    assert [sid for sid, _ in one] == ["from-cypher"]
    assert "from-cypher" not in [sid for sid, _ in two] and two
    # an unknown scoring is an error, not a silent count
    with pytest.raises(ValueError):
        asyncio.run(g.cooccurring(["0"], Conditions(), weights="bogus"))
