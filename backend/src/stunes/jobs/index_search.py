"""Build the OpenSearch BM25 indices from the export snapshot and swap the aliases."""

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import orjson
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_streaming_bulk

from stunes.config import Settings
from stunes.search_index import ENTITIES_BODY, SONG_FIELDS, SONGS_BODY

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("rb") as f:
        for line in f:
            yield orjson.loads(line)


async def _swap_alias(client: AsyncOpenSearch, alias: str, new_index: str) -> list[str]:
    actions: list[dict] = [{"add": {"index": new_index, "alias": alias}}]
    old: list[str] = []
    if await client.indices.exists_alias(name=alias):
        old = list(await client.indices.get_alias(name=alias))
        actions = [{"remove": {"index": i, "alias": alias}} for i in old] + actions
    await client.indices.update_aliases(body={"actions": actions})
    return old


async def build_index(
    client: AsyncOpenSearch,
    alias: str,
    body: dict,
    docs: Iterable[dict],
    doc_id: Callable[[dict], str],
) -> str:
    """Create `<alias>_v<ts>`, bulk-load, verify the count, then point the alias at it.

    Old versions are kept (not deleted) so a bad build can be rolled back by re-pointing
    the alias; `stunes gc-indices` removes them.
    """
    index = f"{alias}_v{int(time.time())}"
    await client.indices.create(index=index, body=body)

    def actions():
        for doc in docs:
            yield {"_index": index, "_id": doc_id(doc), "_source": doc}

    n, started = 0, time.monotonic()
    async for ok, _ in async_streaming_bulk(
        client, actions(), chunk_size=2000, max_retries=3, raise_on_error=True
    ):
        n += ok
        if n % 100_000 == 0:
            logger.info("%s: %d docs (%.0f/s)", index, n, n / (time.monotonic() - started))

    await client.indices.put_settings(index=index, body={"index": {"refresh_interval": "1s"}})
    await client.indices.refresh(index=index)
    count = (await client.count(index=index))["count"]
    if count != n:
        raise RuntimeError(f"{index}: indexed {n} docs but count is {count}")
    old = await _swap_alias(client, alias, index)
    logger.info(
        "%s -> alias %s (%d docs, %.0fs, previous: %s)",
        index,
        alias,
        n,
        time.monotonic() - started,
        old or "none",
    )
    return index


async def build_indices(settings: Settings, export_dir: Path) -> None:
    client = AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=120)
    song_keys = [*SONG_FIELDS, "album_id", "playlists"]
    try:
        songs = (
            {k: doc.get(k) for k in song_keys} for doc in read_jsonl(export_dir / "songs.jsonl")
        )
        await build_index(client, settings.songs_index, SONGS_BODY, songs, lambda d: d["song_id"])

        entities = (
            doc
            for kind in ("artist", "genre", "tag")
            for doc in read_jsonl(export_dir / f"entities_{kind}.jsonl")
        )
        await build_index(
            client,
            settings.entities_index,
            ENTITIES_BODY,
            entities,
            lambda d: f"{d['kind']}:{d['key']}",
        )
    finally:
        await client.close()
