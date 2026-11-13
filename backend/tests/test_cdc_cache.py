"""stunes.ingest.cdc_cache.apply: what each Debezium change does to Redis (no Redis needed)."""

import base64

import msgpack

from stunes.ingest import cdc_cache


class FakeCache:
    def __init__(self):
        self.deleted = []

    def delete(self, key):
        self.deleted.append(key)


class FakeScript:
    """Stands in for the Lua guard: records calls, applies only a newer LSN."""

    def __init__(self):
        self.calls, self.lsn = [], {}

    def __call__(self, keys, args, client=None):
        self.calls.append((keys, args))
        key, lsn = keys[0], args[0]
        if self.lsn.get(key, -1) >= lsn:
            return 0
        self.lsn[key] = lsn
        return 1


def change(op, lsn, pk="user_recent_plays#u1", plays=None):
    row = {"pk": pk}
    if plays is not None:
        row["value"] = base64.b64encode(msgpack.packb(plays)).decode()
    return {
        "op": op,
        "source": {"lsn": lsn, "ts_ms": 0},
        "after": row if op != "d" else None,
        "before": row if op == "d" else None,
    }


def test_recent_plays_set_through_the_guard_and_drops_the_derived_key():
    cache, script = FakeCache(), FakeScript()
    assert (
        cdc_cache.apply(cache, script, change("u", 10, plays=[[1, 5]]), "cdc.public.feature") == 1
    )
    keys, (lsn, value, _ttl) = script.calls[0]
    assert keys == ["user_recent_plays#u1", "cdc_lsn#user_recent_plays#u1"] and lsn == 10
    assert msgpack.unpackb(value)[0] == [[1, 5]]
    assert cache.deleted == ["user_tag_pref#u1"]


def test_only_the_recent_plays_write_is_guarded():
    # run() reads the guarded command's pipeline result (1 applied / 0 skipped replay)
    cache, script = FakeCache(), FakeScript()
    assert cdc_cache.apply(cache, script, change("u", 20, plays=[[2, 9]]), "cdc.public.feature")
    song = {"op": "u", "source": {"lsn": 21, "ts_ms": 0}, "after": {"song_id": 7}}
    assert not cdc_cache.apply(cache, script, song, "cdc.public.song_stats")


def test_delete_passes_an_empty_value():
    cache, script = FakeCache(), FakeScript()
    cdc_cache.apply(cache, script, change("d", 30), "cdc.public.feature")
    assert script.calls[0][1][1] == b""


def test_other_keys_and_song_stats_are_invalidated():
    cache, script = FakeCache(), FakeScript()
    cdc_cache.apply(cache, script, change("u", 1, pk="user_tag_pref#u1"), "cdc.public.feature")
    song = {"op": "u", "source": {"lsn": 2, "ts_ms": 0}, "after": {"song_id": 7}}
    cdc_cache.apply(cache, script, song, "cdc.public.song_stats")
    assert cache.deleted == ["user_tag_pref#u1", "song_stats#7"] and not script.calls
