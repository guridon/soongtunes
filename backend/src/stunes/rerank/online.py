"""The re-ranker's features for one query, shared by training (rerank/features.py) and serving.

One function computes a query's feature block, so the matrix a model is trained on and the
matrix it scores in the chat pipeline come from the same code. What differs between the two
is the inputs: training reads the train-only stores, and serving reads the live stores.

`tag_count(tag, song)` is the one input with two definitions:
- exact (the first model): the number of train playlists that contain the song and carry the
  tag, counted over all playlists;
- online (the served model): the song's own tag weight if the tag is among its top 15 tags,
  else 0. A song document carries these weights, so a request needs no playlist scan.
"""

import html
import math
import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np

SOURCES = ("bm25", "vector", "tag")
RRF_K = 60
RRF_ALL = {"tag": 1.0, "vector": 1.0, "bm25": 0.6}

FEATURES = [
    "q_chars",
    "q_tokens",
    "q_link_sim1",
    "q_link_sim2",
    "q_link_sim3",
    "q_bm25_hits",
    "q_bm25_top",
    "bm25_rank",
    "bm25_score",
    "bm25_rel",
    "vector_rank",
    "vector_cos",
    "tag_rank",
    "n_sources",
    "rrf_all",
    "tag_hits",
    "tag_cnt1",
    "tag_cnt2",
    "tag_cnt3",
    "tag_share_max",
    "tag_wsum",
    "soft_max",
    "soft_wmean",
    "soft_top1",
    "n_song_tags",
    "log_pop",
    "title_in_q",
    "artist_in_q",
    "album_in_q",
    "genre_in_q",
    "title_bigram",
    "artist_bigram",
    "playlist_bigram",
    "year",
    "decade_match",
]
COL = {name: i for i, name in enumerate(FEATURES)}

# the per-song fields the features read: a song document of the export, or the feature store's
# `song_rerank` group, which holds exactly these
DOC_FIELDS = (
    "title",
    "artists",
    "album",
    "genres",
    "tags",
    "tag_weights",
    "playlist_count",
    "playlists",
    "year",
)

_DECADE = re.compile(r"(\d{2,4})\s*(?:년대|'?s\b)")


def norm(text: str | None) -> str:
    """Lower case, HTML entities decoded (the export has "Can&#39;t"), letters and digits only."""
    return "".join(c for c in html.unescape(text or "").lower() if c.isalnum())


def bigrams(s: str) -> set[str]:
    return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def decades(title: str) -> list[int]:
    out = []
    for m in _DECADE.finditer(title):
        n = int(m.group(1))
        if n < 100:  # "90년대", "00년대", "80s"
            n += 2000 if n < 30 else 1900
        if n % 10 == 0 and 1900 <= n <= 2020:
            out.append(n)
    return out


def contains(query: str, part: str, min_len: int = 2) -> float:
    return 1.0 if len(part) >= min_len and part in query else 0.0


class SongText:
    """Normalized text of a candidate song, computed once per song."""

    __slots__ = ("album", "artists", "genres", "playlists", "title", "title_bg")

    def __init__(self, doc: Mapping):
        self.title = norm(doc.get("title"))
        self.title_bg = bigrams(self.title)
        self.artists = [norm(a) for a in doc.get("artists") or []]
        self.album = norm(doc.get("album"))
        self.genres = [norm(p) for g in doc.get("genres") or [] for p in g.split("/")]
        self.playlists = [bigrams(norm(t)) for t in doc.get("playlists") or []]


def own_tag_count(doc: Mapping) -> Callable[[str], int]:
    """The online tag_count for one song: its own tag weight, 0 outside its top tags."""
    weights = dict(zip(doc.get("tags") or [], doc.get("tag_weights") or [], strict=False))
    return lambda tag: int(weights.get(tag, 0))


def union(sources: Mapping[str, Sequence[tuple[str, float | None]]]) -> list[str]:
    """The union of the sources, in source order; the row order of the feature matrix."""
    return list(dict.fromkeys(s for name in SOURCES for s, _ in sources.get(name) or []))


def feature_block(
    title: str,
    sources: Mapping[str, Sequence[tuple[str, float | None]]],
    tags_linked: Sequence[tuple[str, float]],
    cids: Sequence[str],
    docs: Mapping[str, Mapping],
    text: Mapping[str, SongText],
    q_song: np.ndarray | None,
    song_row: Mapping[str, int],
    smat: np.ndarray | None,
    tag_row: Mapping[str, int],
    tag_sims: np.ndarray,
    tag_count: Callable[[str, str], int],
) -> np.ndarray:
    """Feature rows for the candidates `cids` of one query, in FEATURES order (NaN = missing).

    - sources: bm25 / vector / tag lists of (song id, score); a candidate may be in none of
      them (serving adds the chat pipeline's other sources), and then its source columns stay
      missing;
    - tags_linked: the query's nearest tags with their similarity, best first;
    - song_row, smat: row of each candidate's normalized vector, if it has one;
    - tag_row, tag_sims: the query (tag instruction) against each tag vector it may need;
    - tag_count(tag, song): see the module docstring.
    """
    n = len(cids)
    block = np.full((n, len(FEATURES)), np.nan, dtype=np.float32)
    qn = norm(title)
    qbg = bigrams(qn)
    q_dec = decades(title)
    sims = [s for _, s in tags_linked]
    block[:, COL["q_chars"]] = len(title)
    block[:, COL["q_tokens"]] = len(title.split())
    for j, s in enumerate(sims[:3]):
        block[:, COL[f"q_link_sim{j + 1}"]] = s
    bm25 = list(sources.get("bm25") or [])
    block[:, COL["q_bm25_hits"]] = len(bm25)
    top = bm25[0][1] if bm25 else math.nan
    block[:, COL["q_bm25_top"]] = top

    pos = {s: i for i, s in enumerate(cids)}
    rank_sum = np.zeros(n, dtype=np.float64)
    n_src = np.zeros(n, dtype=np.float32)
    for name in SOURCES:
        seen: set[str] = set()
        for s, score in sources.get(name) or []:
            if s in seen:
                continue
            seen.add(s)
            rank = len(seen)  # over the whole source list, also entries that aren't rows here
            if s not in pos:
                continue
            i = pos[s]
            block[i, COL[f"{name}_rank"]] = rank
            n_src[i] += 1
            rank_sum[i] += RRF_ALL[name] / (RRF_K + rank)
            if name == "bm25":
                block[i, COL["bm25_score"]] = score
                block[i, COL["bm25_rel"]] = score / top if top else math.nan
    block[:, COL["n_sources"]] = n_src
    block[:, COL["rrf_all"]] = rank_sum

    have = [i for i, s in enumerate(cids) if s in song_row]
    if have and q_song is not None and smat is not None:
        vecs = smat[[song_row[cids[i]] for i in have]]
        block[have, COL["vector_cos"]] = vecs @ q_song

    for i, s in enumerate(cids):
        d, t = docs[s], text[s]
        pop = d["playlist_count"]
        block[i, COL["log_pop"]] = math.log1p(pop)

        cnt = [tag_count(tag, s) for tag, _ in tags_linked]
        block[i, COL["tag_hits"]] = sum(c > 0 for c in cnt)
        for j, c in enumerate(cnt[:3]):
            block[i, COL[f"tag_cnt{j + 1}"]] = math.log1p(c)
        block[i, COL["tag_share_max"]] = max(cnt) / pop if pop and cnt else 0.0
        block[i, COL["tag_wsum"]] = sum(w * math.log1p(c) for w, c in zip(sims, cnt, strict=True))

        own = [
            (tag_row[g], w)
            for g, w in zip(d["tags"], d["tag_weights"], strict=True)
            if g in tag_row
        ]
        block[i, COL["n_song_tags"]] = len(d["tags"])
        if own:
            ts = tag_sims[[k for k, _ in own]]
            ws = np.array([w for _, w in own], dtype=np.float32)
            block[i, COL["soft_max"]] = ts.max()
            block[i, COL["soft_wmean"]] = float(ts @ ws / ws.sum())
            if d["tags"][0] in tag_row:  # tags are ordered by weight
                block[i, COL["soft_top1"]] = ts[0]

        block[i, COL["title_in_q"]] = contains(qn, t.title)
        block[i, COL["artist_in_q"]] = max((contains(qn, a) for a in t.artists), default=0.0)
        block[i, COL["album_in_q"]] = contains(qn, t.album)
        block[i, COL["genre_in_q"]] = max((contains(qn, g) for g in t.genres), default=0.0)
        block[i, COL["title_bigram"]] = jaccard(qbg, t.title_bg)
        block[i, COL["artist_bigram"]] = max(
            (jaccard(qbg, bigrams(a)) for a in t.artists), default=0.0
        )
        block[i, COL["playlist_bigram"]] = max((jaccard(qbg, p) for p in t.playlists), default=0.0)
        year = d.get("year")
        if year:
            block[i, COL["year"]] = year
            if q_dec:
                block[i, COL["decade_match"]] = float(any(dd <= year < dd + 10 for dd in q_dec))
    return block
