import random
import time

import msgpack

from stunes.features.store import _L1, PROFILES, _xfetch, envelope, load_groups


def test_spec_loads_and_profiles_are_cumulative():
    groups = load_groups()
    assert groups["user_tag_pref"].compute_fn == "user_tag_pref"
    assert groups["user_recent_plays"].compute_fn is None
    # each experiment adds one mechanism on top of the previous one
    assert not PROFILES["C0"].redis
    assert PROFILES["C2"].jitter and not PROFILES["C2"].xfetch_beta
    assert PROFILES["C4"].negative_ttl and not PROFILES["C4"].l1_ttl
    assert PROFILES["C5"].l1_ttl


def test_xfetch_refreshes_more_often_near_expiry_and_for_costly_keys():
    random.seed(0)

    def rate(remaining_ms, recompute_ms, n=20_000):
        now = 1_000_000.0
        return sum(_xfetch(now, now + remaining_ms, recompute_ms, 1.0) for _ in range(n)) / n

    assert rate(1, 3) > rate(10, 3) > rate(50, 3)  # closer to expiry -> earlier refresh
    assert rate(50, 30) > rate(50, 3)  # more expensive -> earlier refresh
    assert rate(10_000, 3) == 0  # far from expiry: never


def test_l1_expires_and_is_bounded():
    l1 = _L1(ttl=0.05, maxsize=2)
    l1.put("a", 1)
    l1.put("b", 2)
    l1.put("c", 3)  # evicts the oldest insertion
    assert l1.get("a") == (False, None)
    assert l1.get("c") == (True, 3)
    time.sleep(0.06)
    assert l1.get("c") == (False, None)


def test_writer_envelope_matches_reader_layout():
    value, expires_at, recompute_ms, negative = msgpack.unpackb(envelope([["s1", 1]], 60))
    assert value == [["s1", 1]] and not negative and recompute_ms == 0.0
    assert expires_at > time.time() * 1000
