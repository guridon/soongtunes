from collections import Counter

from stunes.rerank.data import audit, split_of, top_playlists_per_song


def test_split_is_stable_and_80_10_10():
    assert split_of("61281") == split_of("61281")
    counts = Counter(split_of(str(i)) for i in range(20_000))
    assert abs(counts["train"] / 20_000 - 0.8) < 0.01
    assert abs(counts["val"] / 20_000 - 0.1) < 0.01
    assert abs(counts["test"] / 20_000 - 0.1) < 0.01


def _pl(pid, likes, songs, tags=(), split="train"):
    return {
        "id": pid,
        "title": f"playlist {pid}",
        "likes": likes,
        "songs": list(songs),
        "tags": list(tags),
        "split": split,
    }


def test_top_playlists_follow_export_order():
    # export.py: ORDER BY toInteger(like_cnt) DESC, playlist_id (string ascending)
    pls = [
        _pl("2", 5, ["a"]),
        _pl("10", 5, ["a"]),  # "10" < "2" as strings
        _pl("1", 5, ["a"]),  # "1" < "10"
        _pl("9", 7, ["a"]),
        _pl("3", 1, ["a"]),
    ]
    assert top_playlists_per_song(pls, 3)["a"] == ["9", "1", "10"]
    assert top_playlists_per_song(pls, 10)["a"] == ["9", "1", "10", "2", "3"]


def test_audit_counts_leaks():
    pls = [
        _pl("1", 100, ["a", "b"], ["밤"], split="test"),
        _pl("2", 1, ["a"], ["밤"]),
        _pl("3", 50, ["c"], ["비"], split="val"),
    ]
    leak = audit(pls)["test_leakage"]
    assert leak["song_pairs"] == 2
    assert leak["song_in_no_train_playlist"] == 1  # b is only in the test playlist
    assert leak["playlist_title_in_search_doc"] == 2  # playlist 1 is the top playlist of a and b
    assert leak["own_tag_on_song_pairs"] == 2
    assert leak["own_tag_reaches_song_only_via_this_playlist"] == 1  # (b, 밤); (a, 밤) also via 2


def test_untitled_playlist_is_not_a_title_leak():
    pl = _pl("1", 100, ["a"], split="test")
    pl["title"] = ""
    assert audit([pl, _pl("2", 1, ["a"])])["test_leakage"]["playlist_title_in_search_doc"] == 0


def test_train_docs_drop_everything_from_held_out_playlists(tmp_path):
    import orjson

    from stunes.rerank.stores import train_song_docs

    songs = tmp_path / "songs.jsonl"
    songs.write_bytes(
        b"".join(
            orjson.dumps({"song_id": s, "title": s, "tags": ["old"], "playlists": ["old"]}) + b"\n"
            for s in ("a", "b")
        )
    )
    pls = [
        {**_pl("1", 9, ["a", "b"], ["밤", "비"], split="train"), "title": "밤 비"},
        {**_pl("2", 5, ["a"], ["밤"], split="train"), "title": "새벽"},
        {**_pl("3", 99, ["a", "b"], ["test-tag"], split="test"), "title": "held out"},
    ]
    docs = {d["song_id"]: d for d in train_song_docs(pls, songs)}
    assert docs["a"]["tags"] == ["밤", "비"] and docs["a"]["tag_weights"] == [2, 1]
    assert docs["a"]["playlists"] == ["밤 비", "새벽"]  # by likes; the test playlist is gone
    assert docs["a"]["playlist_count"] == 2
    assert docs["b"]["tags"] == ["밤", "비"] and docs["b"]["playlist_count"] == 1
    assert all(
        "held out" not in d["playlists"] and "test-tag" not in d["tags"] for d in docs.values()
    )
