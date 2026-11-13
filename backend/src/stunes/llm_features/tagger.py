"""LLM-predicted mood/situation tags as a feature.

Long-tail songs (13% have no tag at all) are invisible to tag-based retrieval. The LLM reads
only metadata — title, artists, album, genres, year — and picks tags from the *existing*
vocabulary: the JSON schema's enum makes inventing a tag impossible. Several songs share one
prompt, and results are cached by a hash of the input, so a re-run costs nothing.

    validate  hide the tags of well-tagged songs, predict, report precision/recall by
              confidence threshold and how skewed predictions are toward popular tags
    apply     tag long-tail songs and store them as the `song_llm_tags` feature group
    project   copy the stored feature to where retrieval reads tags (Neo4j HAS_LLM_TAG edges,
              the OpenSearch llm_tags field); LLM_TAGS_IN_RETRIEVAL switches their use on
    heldout   does an LLM tag bring a song back to the tag queries it belongs to? The validate
              songs' real tags are hidden, their predictions stand in, and the graph tag source
              is asked for each hidden tag (the eval set can't show this: its mood labels are
              songs that already carry the tag)
"""

import asyncio
import hashlib
import logging
import random
import time
from collections import Counter

import msgpack
import orjson
from opensearchpy.helpers import async_streaming_bulk

from stunes.config import Settings
from stunes.db import Graph, create_pg_pool, create_search
from stunes.llm import ChatModel
from stunes.search_index import LLM_TAGS_FIELD

logger = logging.getLogger(__name__)

# bump whenever the prompt or schema changes: it is part of the cache key
MODEL_VERSION = "tagger-v2"
VOCAB_SIZE = 150
BATCH = 10
MAX_TAGS = 5

SYSTEM = """너는 음악 큐레이터다. 각 곡의 메타데이터만 보고, 사람들이 그 곡을 플레이리스트에 넣을 때 붙일 법한
분위기·상황·감정 태그를 고른다. 반드시 주어진 태그 목록 안에서만 고르고, 곡마다 최대 {max_tags}개.
confidence는 0~1: 확실하면 0.9 이상, 짐작이면 0.5 이하. 모르는 곡이면 장르·연도·제목에서 합리적으로 추정하되
confidence를 낮게 준다. JSON만 출력한다."""


def _schema(vocab: list[str], ids: list[str]) -> dict:
    # exactly one entry per input song, ids limited to this batch: without these the model
    # silently dropped ~30% of the songs from its answer
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["songs"],
        "properties": {
            "songs": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "tags"],
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "tags": {
                            "type": "array",
                            "maxItems": MAX_TAGS,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["tag", "confidence"],
                                "properties": {
                                    "tag": {"type": "string", "enum": vocab},
                                    "confidence": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def _describe(doc: dict) -> str:
    parts = [
        f"id={doc['song_id']}",
        f"제목={doc['title']}",
        f"아티스트={', '.join(doc['artists'][:3])}",
    ]
    if doc.get("album"):
        parts.append(f"앨범={doc['album']}")
    if doc.get("genres"):
        parts.append(f"장르={', '.join(doc['genres'])}")
    if doc.get("year"):
        parts.append(f"발매={doc['year']}")
    return " | ".join(parts)


def load_docs(settings: Settings) -> list[dict]:
    path = settings.data_dir / "exports" / "songs.jsonl"
    return [orjson.loads(line) for line in path.read_bytes().splitlines()]


def vocabulary(settings: Settings, n: int = VOCAB_SIZE) -> list[str]:
    path = settings.data_dir / "exports" / "entities_tag.jsonl"
    tags = [orjson.loads(line) for line in path.read_bytes().splitlines()]
    tags.sort(key=lambda t: -t["popularity"])
    return [t["name"] for t in tags[:n]]


class Tagger:
    def __init__(self, settings: Settings, vocab: list[str]):
        self.llm = ChatModel(settings)
        self.vocab = vocab
        self.system = SYSTEM.format(max_tags=MAX_TAGS) + "\n\n태그 목록: " + ", ".join(vocab)
        self.calls = 0
        self.cache_hits = 0

    async def predict(self, pg, docs: list[dict]) -> dict[str, list[tuple[str, float]]]:
        text = "\n".join(_describe(d) for d in docs)
        key = hashlib.sha256(
            f"{MODEL_VERSION}|{self.llm.model}|{self.system}|{text}".encode()
        ).hexdigest()
        cached = await pg.fetchval("SELECT output FROM llm_cache WHERE key = $1", key)
        if cached is not None:
            self.cache_hits += 1
            out = orjson.loads(cached)
        else:
            self.calls += 1
            out = await self.llm.complete_json(
                [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
                _schema(self.vocab, [d["song_id"] for d in docs]),
                name="song_tags",
                max_tokens=4096,
            )
            await pg.execute(
                "INSERT INTO llm_cache (key, output) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                key,
                orjson.dumps(out).decode(),
            )
        wanted = {d["song_id"] for d in docs}
        return {
            s["id"]: [(t["tag"], max(0.0, min(1.0, float(t["confidence"])))) for t in s["tags"]]
            for s in out.get("songs", [])
            if s["id"] in wanted
        }


async def _ensure_tables(pg) -> None:
    await pg.execute(
        "CREATE TABLE IF NOT EXISTS llm_cache (key text PRIMARY KEY, output text NOT NULL, "
        "created_at timestamptz NOT NULL DEFAULT now())"
    )


async def _run_batches(pg, tagger: Tagger, docs: list[dict], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    preds: dict[str, list[tuple[str, float]]] = {}

    async def one(chunk):
        async with sem:
            try:
                preds.update(await tagger.predict(pg, chunk))
            except Exception as e:  # one bad batch must not sink the run; it's reported
                logger.warning("batch failed: %r", e)

    await asyncio.gather(*(one(docs[i : i + BATCH]) for i in range(0, len(docs), BATCH)))
    return preds


def _validation_docs(settings: Settings, vocab: list[str], n: int, seed: int) -> list[dict]:
    """Well-tagged songs (>= 5 vocabulary tags), a seeded sample: validate and heldout share it,
    so heldout reuses validate's cached predictions."""
    vocab_set = set(vocab)
    docs = [d for d in load_docs(settings) if len(set(d["tags"]) & vocab_set) >= 5]
    random.Random(seed).shuffle(docs)
    return docs[:n]


async def validate(settings: Settings, n: int = 300, concurrency: int = 4, seed: int = 1) -> dict:
    vocab = vocabulary(settings)
    vocab_set = set(vocab)
    docs = _validation_docs(settings, vocab, n, seed)
    pg = await create_pg_pool(settings)
    try:
        await _ensure_tables(pg)
        tagger = Tagger(settings, vocab)
        t = time.perf_counter()
        preds = await _run_batches(pg, tagger, docs, concurrency)
        elapsed = time.perf_counter() - t
    finally:
        await pg.close()

    truth = {d["song_id"]: set(d["tags"][:10]) & vocab_set for d in docs}
    report: dict = {
        "songs": len(docs),
        "predicted_songs": len(preds),
        "llm_calls": tagger.calls,
        "cache_hits": tagger.cache_hits,
        "seconds": round(elapsed, 1),
        "model": tagger.llm.model,
        "thresholds": {},
    }
    for th in (0.0, 0.5, 0.7, 0.9):
        tp = fp = fn = 0
        for sid, gold in truth.items():
            got = {t for t, c in preds.get(sid, []) if c >= th}
            tp += len(got & gold)
            fp += len(got - gold)
            fn += len(gold - got)
        report["thresholds"][str(th)] = {
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "tags_per_song": round((tp + fp) / max(1, len(truth)), 2),
        }
    # skew: how concentrated predictions are vs the ground truth (top-5 tags' share)
    pred_counts = Counter(t for ps in preds.values() for t, _ in ps)
    gold_counts = Counter(t for g in truth.values() for t in g)

    def top5_share(c: Counter) -> float:
        total = sum(c.values()) or 1
        return round(sum(v for _, v in c.most_common(5)) / total, 4)

    report["skew"] = {
        "pred_top5_share": top5_share(pred_counts),
        "truth_top5_share": top5_share(gold_counts),
        "pred_top5": pred_counts.most_common(5),
        "truth_top5": gold_counts.most_common(5),
    }
    per_tag = {}
    for tag, _ in gold_counts.most_common(10):
        tp = sum(
            1
            for sid, g in truth.items()
            if tag in g and any(t == tag for t, _ in preds.get(sid, []))
        )
        pred_n = sum(1 for ps in preds.values() if any(t == tag for t, _ in ps))
        per_tag[tag] = {
            "precision": round(tp / pred_n, 3) if pred_n else None,
            "recall": round(tp / gold_counts[tag], 3),
        }
    report["per_tag"] = per_tag
    return report


async def apply(
    settings: Settings, n: int = 2000, threshold: float = 0.7, concurrency: int = 4
) -> dict:
    """Tag tag-less songs (most played first) and store confident tags as a feature."""
    vocab = vocabulary(settings)
    docs = [d for d in load_docs(settings) if not d["tags"]]
    docs.sort(key=lambda d: -d["playlist_count"])
    docs = docs[:n]
    pg = await create_pg_pool(settings)
    try:
        await _ensure_tables(pg)
        tagger = Tagger(settings, vocab)
        preds = await _run_batches(pg, tagger, docs, concurrency)
        rows = []
        for sid, tags in preds.items():
            keep = [[t, round(c, 3), MODEL_VERSION] for t, c in tags if c >= threshold]
            if keep:
                rows.append((f"song_llm_tags#{sid}", msgpack.packb(keep)))
        await pg.executemany(
            "INSERT INTO feature (pk, value) VALUES ($1, $2) "
            "ON CONFLICT (pk) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            rows,
        )
    finally:
        await pg.close()
    return {
        "candidates": len(docs),
        "predicted": len(preds),
        "stored": len(rows),
        "threshold": threshold,
        "llm_calls": tagger.calls,
        "cache_hits": tagger.cache_hits,
    }


_PROJECT_EDGES = """
UNWIND $items AS it
MATCH (s:Song {song_id: it.song_id})
UNWIND it.tags AS tg
CALL (tg) { MATCH (t:Tag {name: tg.name}) RETURN t LIMIT 1 }
CREATE (s)-[:HAS_LLM_TAG {confidence: tg.confidence, model_version: tg.model_version}]->(t)
RETURN count(*) AS n
"""


async def project(settings: Settings) -> dict:
    """Make the stored `song_llm_tags` feature visible to retrieval, replacing what an earlier
    run projected. Kept apart from the playlist-derived tags (HAS_TAG, `tags`) so retrieval can
    use them or not (LLM_TAGS_IN_RETRIEVAL) and an eval can compare the two. A rebuilt search
    index (`stunes index-search`) starts without the field: run this again after it."""
    pg = await create_pg_pool(settings)
    try:
        rows = await pg.fetch("SELECT pk, value FROM feature WHERE pk LIKE 'song_llm_tags#%'")
    finally:
        await pg.close()
    items = [
        {
            "song_id": r["pk"].split("#", 1)[1],
            "tags": [
                {"name": t, "confidence": c, "model_version": v}
                for t, c, v in msgpack.unpackb(r["value"])
            ],
        }
        for r in rows
    ]
    graph = Graph(settings)
    try:
        removed = (await graph.write("MATCH ()-[r:HAS_LLM_TAG]->() DELETE r RETURN count(r) AS n"))[
            0
        ]["n"]
        edges = 0
        for i in range(0, len(items), 500):
            edges += (await graph.write(_PROJECT_EDGES, items=items[i : i + 500]))[0]["n"]
    finally:
        await graph.close()

    client = create_search(settings)
    index = settings.songs_index
    try:
        await client.indices.put_mapping(
            index=index, body={"properties": {"llm_tags": LLM_TAGS_FIELD}}
        )
        # default conflicts=abort: nothing else writes the index here, so a conflict is a
        # reason to stop, not to leave stale tags behind
        cleared = await client.update_by_query(
            index=index,
            body={
                "query": {"exists": {"field": "llm_tags"}},
                "script": {"source": "ctx._source.remove('llm_tags')"},
            },
            refresh=True,
        )
        if cleared.get("failures"):
            raise RuntimeError(f"clearing llm_tags failed: {cleared['failures'][:3]}")
        actions = (
            {
                "_op_type": "update",
                "_index": index,
                "_id": it["song_id"],
                "doc": {"llm_tags": [t["name"] for t in it["tags"]]},
            }
            for it in items
        )
        docs, missing = 0, 0
        async for ok, info in async_streaming_bulk(
            client, actions, chunk_size=500, raise_on_error=False, max_retries=3
        ):
            if ok:
                docs += 1
            else:
                missing += 1
                logger.warning("llm_tags update failed: %s", info)
        await client.indices.refresh(index=index)
    finally:
        await client.close()
    expected = sum(len(it["tags"]) for it in items)
    if edges != expected:  # a song or tag node the graph doesn't have: say so, don't hide it
        logger.warning("projected %d of %d LLM tag edges", edges, expected)
    return {
        "songs": len(items),
        "tags": expected,
        "graph_edges_removed": removed,
        "graph_edges_created": edges,
        "graph_edges_unmatched": expected - edges,
        "search_docs_cleared": cleared.get("updated", 0),
        "search_docs_updated": docs,
        "search_docs_failed": missing,
    }


# hidden edges carry this run's id, so cleanup only ever touches what this run moved
_HIDE = """
UNWIND $ids AS id
MATCH (s:Song {song_id: id})-[r:HAS_TAG]->(t:Tag)
CREATE (s)-[h:HELDOUT_TAG]->(t) SET h = properties(r), h.heldout_run = $run
DELETE r
RETURN count(*) AS n
"""
_UNHIDE = """
MATCH (s:Song)-[h:HELDOUT_TAG {heldout_run: $run}]->(t:Tag)
CREATE (s)-[r:HAS_TAG]->(t) SET r = properties(h)
REMOVE r.heldout_run
DELETE h
RETURN count(*) AS n
"""
_HELDOUT_VERSION = "heldout-test"


async def heldout(
    settings: Settings,
    n: int = 300,
    seed: int = 1,
    tags_per_song: int = 3,
    k: int = 100,
    concurrency: int = 4,
) -> dict:
    """Hide the validate songs' tags in the graph, let their LLM predictions stand in, and ask
    the graph tag source (top-k, the candidate pool) for each hidden tag and for the song's
    first three tags together. Conditions: real tags; hidden with LLM tags off (must be 0);
    hidden with LLM tags at >= 0.9 and >= 0.7. Only the graph source is measured: the song
    vectors and the BM25 playlist titles were built with the tags and would leak them.

    It rewrites shared graph edges, so run it with nothing else using the graph. The real
    edges are restored in `finally`; if a killed run left hidden edges behind, it refuses to
    start and says how to restore them, instead of guessing whose they are."""
    from stunes.retrieval.graph import Conditions, GraphRetriever

    vocab = vocabulary(settings)
    vocab_set = set(vocab)
    docs = _validation_docs(settings, vocab, n, seed)
    pg = await create_pg_pool(settings)
    try:
        await _ensure_tables(pg)
        tagger = Tagger(settings, vocab)
        preds = await _run_batches(pg, tagger, docs, concurrency)
    finally:
        await pg.close()
    truth = {d["song_id"]: [t for t in d["tags"][:10] if t in vocab_set] for d in docs}
    queries = [(sid, [t]) for sid, ts in truth.items() for t in ts[:tags_per_song]]
    queries += [(sid, ts[:3]) for sid, ts in truth.items() if len(ts) >= 3]
    ids = list(truth)

    graph = Graph(settings)
    sem = asyncio.Semaphore(concurrency)

    async def measure(llm_tags: bool) -> dict:
        r = GraphRetriever(graph, timeout_s=60, llm_tags=llm_tags)

        async def one(sid, tags):
            async with sem:
                rows, _ = await r.by_conditions(Conditions(tags=tags), k=k)
            return len(tags), sid in {x for x, _ in rows}

        got = await asyncio.gather(*(one(sid, tags) for sid, tags in queries))
        out = {}
        for size, label in ((1, "one_tag"), (3, "three_tags")):
            hits = [h for n_, h in got if n_ == size]
            recall = round(sum(hits) / len(hits), 4) if hits else None
            out[label] = {"queries": len(hits), f"recall_at_{k}": recall}
        return out

    async def set_llm_edges(threshold: float | None) -> int:
        await graph.write(
            "MATCH ()-[r:HAS_LLM_TAG {model_version: $v}]->() DELETE r", v=_HELDOUT_VERSION
        )
        if threshold is None:
            return 0
        items = [
            {
                "song_id": sid,
                "tags": [
                    {"name": t, "confidence": c, "model_version": _HELDOUT_VERSION}
                    for t, c in preds.get(sid, [])
                    if c >= threshold
                ],
            }
            for sid in ids
        ]
        return (await graph.write(_PROJECT_EDGES, items=items))[0]["n"]

    has_tag = "MATCH ()-[r:HAS_TAG]->() RETURN count(r) AS n"
    run = f"heldout-{int(time.time())}"
    left = await graph.read(
        "MATCH ()-[h:HELDOUT_TAG]->() RETURN h.heldout_run AS run, count(h) AS n"
    )
    if left:
        await graph.close()
        raise RuntimeError(
            f"hidden tag edges from an earlier run are still in the graph: {left}. Restore with "
            "MATCH (s)-[h:HELDOUT_TAG {heldout_run: $run}]->(t) CREATE (s)-[r:HAS_TAG]->(t) "
            "SET r = properties(h) REMOVE r.heldout_run DELETE h"
        )
    result: dict = {
        "songs": len(ids),
        "predicted_songs": len(preds),
        "llm_calls": tagger.calls,
        "cache_hits": tagger.cache_hits,
        "k": k,
        "tags_per_song": tags_per_song,
        "conditions": {},
    }
    try:
        await set_llm_edges(None)
        before = (await graph.read(has_tag))[0]["n"]
        result["conditions"]["real tags"] = await measure(False)
        result["hidden_edges"] = (await graph.write(_HIDE, ids=ids, run=run))[0]["n"]
        result["conditions"]["hidden, LLM tags off"] = await measure(False)
        for th in (0.9, 0.7):
            edges = await set_llm_edges(th)
            result["conditions"][f"hidden, LLM tags >= {th}"] = {
                "llm_edges": edges,
                **await measure(True),
            }
    finally:
        # real tags first: a failure removing the temporary LLM edges must not leave them hidden
        try:
            result["unhidden_edges"] = (await graph.write(_UNHIDE, run=run))[0]["n"]
        finally:
            try:
                await set_llm_edges(None)
                after = (await graph.read(has_tag))[0]["n"]
            finally:
                await graph.close()
    if after != before:
        raise RuntimeError(f"HAS_TAG count {before} before, {after} after the held-out test")
    result["has_tag_edges"] = after
    return result


def save(settings: Settings, kind: str, result: dict) -> str:
    out = settings.results_dir / "llm_tags"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{kind}-{int(time.time())}.json"
    path.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    return str(path)
