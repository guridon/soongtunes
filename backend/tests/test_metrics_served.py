from prometheus_client import REGISTRY

from stunes import metrics
from stunes.config import SERVED_STRATEGY


def _stage_count(stage: str) -> float:
    return REGISTRY.get_sample_value("stunes_chat_stage_seconds_count", {"stage": stage}) or 0.0


def test_stage_metrics_follow_the_served_strategy():
    trace = {"timings_ms": {"extract": 5.0, "total": 9.0}}
    before = _stage_count("extract")
    metrics.observe_turn(SERVED_STRATEGY, 9.0, trace, [{"playlist_count": 3, "artists": ["a"]}])
    assert _stage_count("extract") == before + 1
    # a comparison arm records end-to-end latency only
    metrics.observe_turn("bm25", 9.0, trace, [])
    assert _stage_count("extract") == before + 1
