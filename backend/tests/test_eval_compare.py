import orjson
import pytest

from stunes.eval import compare as cmp


def _rec(strategy, conv, turn, qt, mrr, predicted=("1",), trace=None, error=None):
    r = {
        "strategy": strategy,
        "conversation_id": conv,
        "turn": turn,
        "query_type": qt,
        "predicted": list(predicted),
        "trace": trace or {},
        "metrics": {
            "hit": float(mrr > 0),
            "mrr": mrr,
            "ndcg": mrr,
            "recall": 0.0,
            "precision": 0.0,
        },
    }
    if error:
        r["error"] = error
    return r


def _write(tmp_path, name, records):
    d = tmp_path / name
    d.mkdir()
    (d / "predictions.jsonl").write_bytes(b"\n".join(orjson.dumps(r) for r in records) + b"\n")
    return d


def test_pairs_turns_and_drops_degraded_ones(tmp_path):
    a = _write(
        tmp_path,
        "a",
        [
            _rec("hybrid", "1", 1, "mood_search", 0.5),
            _rec("hybrid", "1", 2, "mood_follow_up", 1.0),
            _rec("hybrid", "2", 1, "mood_search", 0.0, trace={"plans": {"vector_error": "x"}}),
            _rec("hybrid", "3", 1, "specific_song_search", 1.0),
            _rec("agent", "3", 1, "specific_song_search", 0.0),  # other strategy: ignored
        ],
    )
    b = _write(
        tmp_path,
        "b",
        [
            _rec("hybrid", "1", 1, "mood_search", 1.0, predicted=("2",)),
            _rec("hybrid", "1", 2, "mood_follow_up", 1.0),
            _rec("hybrid", "2", 1, "mood_search", 1.0),
            _rec("hybrid", "3", 1, "specific_song_search", 1.0, error="boom"),
            _rec("hybrid", "4", 1, "mood_search", 1.0),  # only in b
        ],
    )
    A = cmp.load_arm(*cmp.parse_arm(f"{a}:hybrid"))
    B = cmp.load_arm(*cmp.parse_arm(f"{b}:hybrid"))
    r = cmp.compare(A, B, marked={"2"}, resamples=200)
    assert r["shared_turns"] == 4 and r["compared_turns"] == 2 and r["excluded_turns"] == 2
    assert r["only_in_b"] == 1 and r["only_in_a"] == 0
    all_ = r["by_type"]["ALL"]
    assert all_["n"] == 2 and all_["conversations"] == 1
    assert all_["delta"] == pytest.approx(0.25)
    assert (all_["b_better"], all_["b_worse"], all_["top_k_changed"]) == (1, 0, 1)
    assert (all_["a_turns_with_marked"], all_["b_turns_with_marked"]) == (0, 1)
    assert all_["ci95"] is None  # one conversation: no interval
    assert list(r["by_type"]) == ["ALL", "mood_search", "mood_follow_up"]

    every = cmp.compare(A, B, clean_only=False, resamples=200)
    assert every["compared_turns"] == 4
    lo, hi = every["by_type"]["ALL"]["ci95"]
    assert lo <= every["by_type"]["ALL"]["delta"] <= hi
    assert "| ALL | 4 |" in cmp.markdown(every, "a", "b")


def test_arm_spec_needs_a_strategy():
    with pytest.raises(ValueError):
        cmp.parse_arm("results/eval/x")
    assert cmp.parse_arm("results/eval/x:hybrid")[1] == "hybrid"


def test_refuses_arms_from_different_eval_sets(tmp_path):
    a = _write(tmp_path, "a", [_rec("hybrid", "1", 1, "mood_search", 1.0)])
    b = _write(tmp_path, "b", [_rec("hybrid", "1", 1, "genre_follow_up", 1.0)])
    with pytest.raises(ValueError, match="query types"):
        cmp.compare(cmp.load_arm(a, "hybrid"), cmp.load_arm(b, "hybrid"))


def test_interval_endpoints_are_the_2_5_and_97_5_percentiles():
    # constant per-conversation deltas: every resample mean is one of the conversation values
    lo, hi = cmp._bootstrap({str(i): [float(i)] for i in range(2)}, n=4000, seed=1)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)
    lo, hi = cmp._bootstrap({str(i): [float(i)] for i in range(50)}, n=4000, seed=1)
    assert 24.5 - 5 < lo < 24.5 < hi < 24.5 + 5
