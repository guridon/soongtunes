"""Entity linking: which method per entity type? (docs/prereg-link-methods.md)

    python -m stunes.bench.link_methods --dataset ~/Downloads/evaluation_dataset.csv \
        --run ../results/eval/20261102T104757Z --out ../results/eval/link-methods.json

The linker uses a different method per type (retrieval/linker.py):
- genres: name, then an alias table;
- artists: BM25 over names, accepted when the mention and the name contain each other;
- tags: exact name, else dense and BM25 candidates interleaved.
This replays every method on every type, on mentions with a known answer, to see whether each
type's method is the best one for it.

Mentions are what the LLM extracted in a recorded v1-set run (trace slots), so only linking
varies. Answers come from the v1 labels (docs/eval-set.md):
- artist: specific_song_search, the artists of the one labelled song;
- genre: genre_follow_up, the Genre name whose song count matches the label (chained turns
  are left out: their labels contradict the text);
- tag: mood_search, the one Tag whose songs are the label (the text paraphrases it).
Each mention is also noised as in the robustness eval (stunes.eval.noise): typo, spacing,
English-mode keys. Dev = even conversation ids, test = odd.
"""

import argparse
import ast
import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from stunes.eval import noise
from stunes.retrieval.linker import _norm
from stunes.retrieval.typo import closest, from_keys, looks_hanyoung

csv.field_size_limit(sys.maxsize)
CONDITIONS = ("clean", "typo", "spacing", "hanyoung")
TEMPLATE = {"genre": "음악 장르: ", "artist": "가수: ", "tag": "음악 태그: "}


# typo-robust helpers: the served ones (retrieval/typo.py)


# ------------------------------------------------------------------ data
async def gold_sets(svc, rows: list[dict]) -> tuple[dict, dict]:
    """Artist ids per labelled song, and song count per Genre name."""
    songs = sorted({r["label"][0] for r in rows if r["query_type"] == "specific_song_search"})
    art = await svc.stores.graph.read(
        "UNWIND $ids AS id MATCH (s:Song {song_id: id})-[:PERFORMED_BY]->(a:Artist) "
        "RETURN id, collect(a.artist_id) AS artists",
        ids=songs,
    )
    sizes = await svc.stores.graph.read(
        "MATCH (g:Genre)<-[:HAS_GENRE]-(s:Song) WHERE g.name IS NOT NULL "
        "RETURN g.name AS name, count(DISTINCT s) AS n"
    )
    return {r["id"]: set(r["artists"]) for r in art}, {r["name"]: r["n"] for r in sizes}


def items(
    dataset: Path,
    run: Path,
    label_tags: dict,
    artists: dict,
    genre_n: dict,
    strategy: str = "hybrid",
) -> list[dict]:
    rows = {}
    for r in csv.DictReader(open(dataset, encoding="utf-8-sig")):
        rows[(r["conversation_id"], int(r["turn"]))] = r
    out = []
    for line in open(run / "predictions.jsonl"):
        x = json.loads(line)
        if x["strategy"] != strategy:
            continue
        key = (str(x["conversation_id"]), int(x["turn"]))
        r, slots = rows.get(key), (x.get("trace") or {}).get("slots") or {}
        if r is None:
            continue
        label = [str(s) for s in ast.literal_eval(r["expected_output"])]
        base = {"cid": key[0], "turn": key[1], "input": r["input"]}
        if r["query_type"] == "specific_song_search":
            m = next((s.get("artist") for s in slots.get("songs") or [] if s.get("artist")), None)
            m = m or next(iter(slots.get("artists") or []), None)
            gold = artists.get(label[0])
            if m and gold:
                out.append({**base, "kind": "artist", "mention": m, "gold": sorted(gold)})
        elif r["query_type"] == "genre_follow_up":
            m = next(iter(slots.get("genres") or []), None)
            gold = [g for g, n in genre_n.items() if n in (len(label), len(label) + 1)]
            if m and len(gold) == 1:
                out.append({**base, "kind": "genre", "mention": m, "gold": gold})
        elif r["query_type"] == "mood_search":
            m = next(iter(slots.get("moods") or []), None)
            gold = label_tags.get(f"{key[0]}|{key[1]}")
            if m and gold:
                out.append({**base, "kind": "tag", "mention": m, "gold": gold})
    return out


# ------------------------------------------------------------------ methods
class Methods:
    def __init__(self, svc, genres: list[str], tie: str = "key"):
        self.svc, self.bm25, self.linker = svc, svc.bm25, svc.hybrid._linker
        self.tie = tie  # fuzzy ties: "key" (first measurement) or "order" (BM25 rank)
        self.genres = genres
        self._doc: dict[tuple[str, str], np.ndarray] = {}
        self._query: dict[tuple[str, str], np.ndarray] = {}

    async def _docs(self, kind: str, names: list[str]) -> np.ndarray:
        miss = [n for n in dict.fromkeys(names) if (kind, n) not in self._doc]
        for i in range(0, len(miss), 64):
            chunk = miss[i : i + 64]
            vecs = await self.svc.embedder.embed_documents([TEMPLATE[kind] + n for n in chunk])
            for n, v in zip(chunk, vecs, strict=True):
                self._doc[(kind, n)] = v / np.linalg.norm(v)
        return np.stack([self._doc[(kind, n)] for n in names])

    async def _q(self, kind: str, text: str) -> np.ndarray:
        if (kind, text) not in self._query:
            v = await self.svc.embedder.embed_query(TEMPLATE[kind] + text)
            self._query[(kind, text)] = v / np.linalg.norm(v)
        return self._query[(kind, text)]

    async def exact(self, kind, m):
        if kind == "genre":
            hit = next((g for g in self.genres if _norm(g) == _norm(m)), None)
            return hit
        hits = await self.bm25.link(m, kind, k=5)
        return next((h.key for h in hits if _norm(h.name) == _norm(m)), None)

    async def bm25_top1(self, kind, m):
        hits = await self.bm25.link(m, kind, k=1)
        return hits[0].key if hits else None

    async def bm25_contain(self, kind, m):
        hits = await self.bm25.link(m, kind, k=5)
        if not hits:
            return None
        top = hits[0]
        if top.exact or _norm(m) in _norm(top.name) or _norm(top.name) in _norm(m):
            return top.key
        return None

    async def dense_top1(self, kind, m):
        q = await self._q(kind, m)
        if kind == "tag":
            hits = await self.svc.vectors.search("tag", q, k=1)
            return hits[0][0] if hits else None
        if kind == "genre":
            names = self.genres
        else:  # artists: no name index; rank the 20 BM25 candidates by embedding
            cands = await self.bm25.link(m, kind, k=20)
            if not cands:
                return None
            keys = [c.key for c in cands]
            names = [c.name for c in cands]
            sims = (await self._docs(kind, names)) @ q
            return keys[int(np.argmax(sims))]
        sims = (await self._docs(kind, names)) @ q
        return names[int(np.argmax(sims))]

    async def production(self, kind, m):
        if kind == "genre":
            hit = self.linker.genre(m)
        elif kind == "artist":
            hit = await self.linker.artist(m)
        else:
            got = await self.linker.tags(m, k=2)
            hit = got[0] if got else None
        return hit.key if hit else None

    async def fuzzy(self, kind, m, max_ratio: float):
        """BM25's 20 candidates, the one closest by jamo edit distance, if close enough."""
        cands = await self.bm25.link(m, kind, k=20)
        if kind == "genre":
            cands_kn = [(g, g) for g in self.genres]
        else:
            cands_kn = [(c.key, c.name) for c in cands]
        if not cands_kn:
            return None
        return closest(m, cands_kn, max_ratio, tie=self.tie)

    async def hanyoung_back(self, kind, m):
        """A mention typed with the keyboard in English mode, read back as Hangul and linked."""
        return await self.production(kind, from_keys(m)) if looks_hanyoung(m) else None


METHODS = ("exact", "bm25_top1", "bm25_contain", "dense_top1", "production", "hanyoung_back")
FUZZY = (0.1, 0.2, 0.3)


def fixed(links: dict, ratio: float):
    """Production, then the English-mode read-back, then jamo fuzzy matching at `ratio`."""
    return links["production"] or links["hanyoung_back"] or links[f"fuzzy@{ratio}"]


def fixed2(links: dict, ratio: float = 0.2):
    """Exact name, then the English-mode read-back, then production, then fuzzy matching
    (docs/prereg-link-methods.md, changes)."""
    return (
        links["exact"] or links["hanyoung_back"] or links["production"] or links[f"fuzzy@{ratio}"]
    )


async def run(
    dataset: Path,
    run_dir: Path,
    label_tags_path: Path,
    arms: dict[str, str] | None = None,
    tie: str = "key",
) -> dict:
    """arms: strategy -> condition, to link the mentions an LLM extracted from noised
    utterances (the robustness run) instead of noising clean mentions here."""
    from stunes.config import get_settings
    from stunes.services import Services

    svc = await Services.build(get_settings())
    try:
        rows = []
        for r in csv.DictReader(open(dataset, encoding="utf-8-sig")):  # noqa: ASYNC230
            if r["query_type"] == "specific_song_search":
                r["label"] = [str(s) for s in ast.literal_eval(r["expected_output"])]
                rows.append(r)
        artists, genre_n = await gold_sets(svc, rows)
        tags = json.loads(label_tags_path.read_text())  # noqa: ASYNC240 (offline script)
        if arms:
            data = []
            for strat, cond in arms.items():
                for it in items(dataset, run_dir, tags, artists, genre_n, strat):
                    data.append({**it, "recorded": cond})
        else:
            data = items(dataset, run_dir, tags, artists, genre_n)
        meth = Methods(svc, svc.genres, tie)
        out = []
        for it in data:
            for cond in [it["recorded"]] if arms else CONDITIONS:
                if cond == "clean" or arms:
                    m = it["mention"]
                else:
                    nz = noise.apply(cond, it["mention"], f"{it['cid']}:{it['turn']}")
                    if not nz.applied:
                        continue
                    m = nz.text
                res = {}
                for name in METHODS:
                    try:
                        res[name] = await getattr(meth, name)(it["kind"], m)
                    except Exception as e:  # one failing call must not end the run
                        res[name] = f"ERROR {type(e).__name__}"
                for r in FUZZY:
                    res[f"fuzzy@{r}"] = await meth.fuzzy(it["kind"], m, r)
                out.append({**it, "condition": cond, "used": m, "links": res})
        return {"n_items": len(data), "rows": out}
    finally:
        await svc.close()


def summarize(rows: list[dict]) -> dict:
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        split = "dev" if int(r["cid"]) % 2 == 0 else "test"
        gold = set(map(str, r["gold"]))
        links = dict(r["links"]) | {f"fixed@{x}": fixed(r["links"], x) for x in FUZZY}
        links["fixed2@0.2"] = fixed2(r["links"])
        for name, key in links.items():
            ok = key is not None and str(key) in gold
            wrong = key is not None and not ok
            agg[(split, r["kind"], r["condition"])][name].append((ok, wrong))
    out = {}
    for (split, kind, cond), ms in sorted(agg.items()):
        out[f"{split} {kind} {cond}"] = {
            name: {
                "n": len(v),
                "right": round(sum(a for a, _ in v) / len(v), 3),
                "wrong": round(sum(b for _, b in v) / len(v), 3),
            }
            for name, v in ms.items()
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--label-tags",
        type=Path,
        default=Path(__file__).parents[4] / "results/eval/lossmap/label_tags-20261101T130015Z.json",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--arms", default="", help="strategy=condition,... : link recorded noisy mentions"
    )
    ap.add_argument("--tie", choices=["key", "order"], default="key")
    a = ap.parse_args()
    arms = dict(x.split("=") for x in a.arms.split(",")) if a.arms else None
    res = asyncio.run(run(a.dataset.expanduser(), a.run, a.label_tags, arms, a.tie))
    res["summary"] = summarize(res["rows"])
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res["summary"].items():
        print(
            k, {m: (x["right"], x["wrong"]) for m, x in v.items()}, "n", next(iter(v.values()))["n"]
        )


if __name__ == "__main__":
    main()
