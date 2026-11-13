import hashlib

import orjson
import pytest

from stunes.eval.run import PARTIAL, Turn, _resume_state, finished


def _convs():
    return [[Turn(cid, i, f"t{i}", "start", set()) for i in (1, 2)] for cid in ("a", "b", "c")]


def _rec(strategy, cid, turn):
    return {"strategy": strategy, "conversation_id": cid, "turn": turn}


def test_a_conversation_counts_as_finished_only_when_every_strategy_answered_every_turn():
    recs = [_rec(s, "a", t) for s in ("x", "y") for t in (1, 2)]
    recs += [_rec("x", "b", 1), _rec("x", "b", 2), _rec("y", "b", 1)]  # y cut midway
    recs += [_rec("x@variant", "c", t) for t in (1, 2)]  # variant records don't count
    assert finished(recs, _convs(), ["x", "y"]) == {"a"}


def test_resume_keeps_finished_conversations_and_drops_the_rest(tmp_path):
    data = tmp_path / "set.jsonl"
    data.write_bytes(b"set")
    run = {"dataset_sha256": hashlib.sha256(b"set").hexdigest(), "strategies": ["x"],
           "commit": "c0"}  # fmt: skip
    (tmp_path / "run.json").write_bytes(orjson.dumps(run))
    recs = [_rec("x", "a", 1), _rec("x", "a", 2), _rec("x", "b", 1)]
    (tmp_path / PARTIAL).write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in recs))

    got, kept, done = _resume_state(tmp_path, data, ["x"], _convs())
    assert got["commit"] == "c0" and done == {"a"} and kept == recs[:2]
    # the partial file now holds only the kept conversations, so a second stop loses nothing
    lines = (tmp_path / PARTIAL).read_bytes().splitlines()
    assert [orjson.loads(x) for x in lines] == recs[:2]

    with pytest.raises(SystemExit):
        _resume_state(tmp_path, data, ["x", "y"], _convs())
    data.write_bytes(b"another set")
    with pytest.raises(SystemExit):
        _resume_state(tmp_path, data, ["x"], _convs())
