import math

import pytest

from stunes.eval.metrics import mean, score_turn
from stunes.retrieval.fusion import rrf


def test_rrf_rewards_agreement_over_single_top_rank():
    fused = rrf({"bm25": ["a", "b", "c"], "vector": ["b", "c", "a"]})
    assert [f.song_id for f in fused][:1] == ["b"]
    b = next(f for f in fused if f.song_id == "b")
    assert b.ranks == {"bm25": 2, "vector": 1}
    assert b.score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_weights_and_duplicates():
    fused = rrf({"graph": ["x", "x", "y"], "bm25": ["y"]}, weights={"graph": 2.0, "bm25": 0.5})
    by_id = {f.song_id: f for f in fused}
    # duplicate "x" in one source is counted once and does not push "y" down
    assert by_id["y"].ranks == {"graph": 2, "bm25": 1}
    assert by_id["x"].score == pytest.approx(2.0 / 61)


def test_score_turn_v1_compatible_definitions():
    m = score_turn(["z", "a", "q", "b"], {"a", "b", "c"}, k=10)
    assert m.hit == 1.0
    assert m.mrr == 0.5
    assert m.recall == pytest.approx(2 / 3)
    assert m.precision == pytest.approx(2 / 4)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m.ndcg == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / idcg)


def test_score_turn_cutoff_and_empty():
    assert score_turn([str(i) for i in range(20)], {"15"}, k=10).hit == 0.0
    assert score_turn([], {"a"}).mrr == 0.0
    assert mean([score_turn(["a"], {"a"}), score_turn(["b"], {"a"})])["hit"] == 0.5


def test_watermark_only_advances_over_contiguous_batches():
    from stunes.jobs.backfill import Watermark

    w = Watermark()
    assert not w.finish(1)  # batch 0 still running: nothing is safe to checkpoint
    assert w.next == 0
    assert w.finish(0)
    assert w.next == 2
    assert not w.finish(3)
    assert w.finish(2)
    assert w.next == 4


def test_load_dataset_groups_and_orders_turns(tmp_path):
    from stunes.eval.run import load_dataset

    p = tmp_path / "e.csv"
    p.write_text(
        "conversation_id,turn,input,query_type,expected_output\n"
        "2,2,b,genre_follow_up,\"['5', '6']\"\n"
        "2,1,a,mood_search,\"['4']\"\n"
        "10,1,c,specific_song_search,\"['7']\"\n",
        encoding="utf-8",
    )
    convs = load_dataset(p)
    assert [[t.turn for t in c] for c in convs] == [[1, 2], [1]]
    assert convs[0][1].relevant == {"5", "6"}
