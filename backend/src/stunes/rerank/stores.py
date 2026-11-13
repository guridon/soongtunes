"""Re-ranker data, step 2: candidate stores built from train playlists only.

The step-1 audit found today's song features carry the held-out playlists: 28% of test
(playlist, song) pairs have the playlist's own title in the song's search document. Every
playlist-derived field is therefore recomputed from the train split, with the same rules
jobs/export.py uses on the whole graph:
- tags: tags of the train playlists containing the song, ranked by how many carry each,
  top 15;
- playlist_count: number of train playlists containing the song (the popularity prior);
- playlists: titles of the song's top 5 train playlists by like_cnt, empty titles dropped.

Song metadata (title, artists, album, genres, year) comes from the export unchanged. LLM tags
are left out: the tagger saw playlist-derived context.

The output goes to a separate export dir, a separate OpenSearch alias and (through
`stunes backfill --export-dir`) a separate embedding version. None of it is activated, so the
service keeps reading its own stores.
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import orjson
from opensearchpy import AsyncOpenSearch

from stunes.config import Settings
from stunes.jobs.index_search import build_index, read_jsonl
from stunes.rerank.data import top_playlists_per_song
from stunes.search_index import SONG_FIELDS, SONGS_BODY

logger = logging.getLogger(__name__)

SONGS_ALIAS = "songs_rr_train"
VECTOR_VERSION = "song-harrier-rr-train-pop5"


def train_song_docs(
    playlists: list[dict], songs_export: Path, max_tags: int = 15, max_playlists: int = 5
) -> Iterator[dict]:
    train = [p for p in playlists if p["split"] == "train"]
    count: Counter = Counter()
    tags: dict[str, Counter] = defaultdict(Counter)
    for p in train:
        for s in dict.fromkeys(p["songs"]):  # once per playlist
            count[s] += 1
            if p["tags"]:
                tags[s].update(p["tags"])
    top = top_playlists_per_song(train, max_playlists)
    title = {p["id"]: p["title"] for p in train}
    for doc in read_jsonl(songs_export):
        s = doc["song_id"]
        # export.py: ORDER BY weight DESC, tag
        ranked = sorted(tags.get(s, Counter()).items(), key=lambda kv: (-kv[1], kv[0]))[:max_tags]
        yield {
            **doc,
            "tags": [t for t, _ in ranked],
            "tag_weights": [w for _, w in ranked],
            "playlist_count": count.get(s, 0),
            "playlists": [title[i] for i in top.get(s, []) if title[i]],
            "llm_tags": [],
        }


def write_export(playlists: list[dict], songs_export: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "songs.jsonl.tmp"
    n = 0
    with tmp.open("wb") as f:
        for doc in train_song_docs(playlists, songs_export):
            f.write(orjson.dumps(doc) + b"\n")
            n += 1
    tmp.replace(out_dir / "songs.jsonl")
    logger.info("wrote %d train-only song docs to %s", n, out_dir)
    return n


async def build_search_index(settings: Settings, export_dir: Path) -> str:
    client = AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=120)
    keys = [*SONG_FIELDS, "album_id", "playlists", "llm_tags"]
    try:
        docs = ({k: d.get(k) for k in keys} for d in read_jsonl(export_dir / "songs.jsonl"))
        return await build_index(client, SONGS_ALIAS, SONGS_BODY, docs, lambda d: d["song_id"])
    finally:
        await client.close()
