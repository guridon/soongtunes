from stunes.ingest.consumer import OffsetTracker


def _track(strategy: str) -> OffsetTracker:
    t = OffsetTracker(strategy)
    for o in range(100, 105):
        t.dispatch(0, o)
    for o in (100, 101, 103, 104):  # 102 still in flight
        t.complete(0, o)
    return t


def test_watermark_never_commits_past_in_flight():
    # the spec's example: 100 101 [102 running] 103 104 -> next offset to read is 102
    assert _track("watermark").commit_offsets() == {0: 102}


def test_max_commit_skips_in_flight_message():
    # committing 105 and crashing means 102 is never read again
    assert _track("max").commit_offsets() == {0: 105}


def test_override_to_watermark_for_safe_commits():
    assert _track("max").commit_offsets("watermark") == {0: 102}


def test_all_done_commits_next_offset():
    t = OffsetTracker("watermark")
    t.dispatch(3, 7)
    t.complete(3, 7)
    assert t.commit_offsets() == {3: 8}
