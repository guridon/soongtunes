"""Personalized PageRank (PPR) over the playlist graph, a candidate source for the home feed.

A random walk starts at the user's recent plays. Each step goes song -> playlist -> song: a
uniformly random playlist that holds the song, then a uniformly random song in that playlist.
With probability alpha the walk jumps back to a start song. A song's score is the share of time
the walk spends there. Co-occurrence counts only the songs one playlist away from a seed; the
walk also reaches songs several playlists away, and scores them by how many paths lead there.
Pinterest's Pixie serves the same walk on its pin-board graph.

Edges:
- Playlist-INCLUDES-Song.
- Song-PERFORMED_BY-Artist, optional (`artist_w` = the share of each step that goes through
  the song's artists instead of its playlists).
- Song tags are left out. They are counted from playlist tags (infra/postgres/init/
  003_song_tag.sql), so a held-out playlist's tags would lead the walk to its hidden songs.
- Genres are left out: 30 genres hold the whole catalog, so a step through one is a jump to a
  random song.

A held-out playlist (`exclude_playlists`) is removed from the graph for that walk. The walk
leaves it out of its songs' degrees, and no mass passes through it.

    stunes ppr-build          # the served Neo4j -> .data/ppr/graph.npz (a batch job; rerun it
                              # when the graph changes, then restart the API)
    stunes ppr-build --src D  # from a graph_sql_prep export instead (the 11-06 experiments)
"""

import gzip
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Frozen from the dev split (results/feed/ppr-dev.json, docs/prereg-feed-ppr.md): the highest
# dev Recall@50 of the 24 grid points. The test split never set them.
ALPHA = 0.5
BETA = 0.0
ARTIST_W = 0.2
TOL = 0.01  # iterations until (1 - alpha)^n <= TOL

# Feed sources. The two variants take one piece away from `ppr`, to see where its gain comes from:
# - ppr_1step: a single step, so only songs one playlist (or artist) from a seed, weighted by
#   1/degree on both sides; no multi-hop paths.
# - ppr_playlists: no artist edges.
VARIANTS = {
    "ppr": {},
    "ppr_1step": {"steps": 1},
    "ppr_playlists": {"artist_w": 0.0},
}


def _read_ints(path: Path, cols: int) -> np.ndarray:
    with gzip.open(path, "rt") as f:
        return np.loadtxt(
            f, delimiter=",", skiprows=1, usecols=range(cols), dtype=np.int64, ndmin=2
        )


def build(src: Path, out: Path) -> dict:
    """Edges from the graph export (`python -m stunes.bench.graph_sql_prep export`)."""
    t = time.perf_counter()
    songs = _read_ints(src / "songs.csv.gz", 1)[:, 0]
    pl = _read_ints(src / "playlists.csv.gz", 2)
    artists = _read_ints(src / "artists.csv.gz", 1)[:, 0]
    inc = _read_ints(src / "includes.csv.gz", 2)  # playlist, song
    perf = _read_ints(src / "performed_by.csv.gz", 2)  # song, artist
    return _save(out, songs, pl[:, 0], pl[:, 1], artists, inc, perf, t)


async def build_from_graph(graph, out: Path, chunk: int = 5_000) -> dict:
    """Edges read from the served Neo4j (`stunes.db.Graph`)."""
    t = time.perf_counter()

    async def ids(query: str) -> list:
        return [r["id"] for r in await graph.read(query, timeout_s=600)]

    songs = await ids("MATCH (s:Song) RETURN s.song_id AS id")
    pl = await graph.read(
        "MATCH (p:Playlist) RETURN p.playlist_id AS id, toInteger(p.like_cnt) AS like_cnt",
        timeout_s=600,
    )
    artists = await ids("MATCH (a:Artist) RETURN a.artist_id AS id")
    inc, perf = [], []
    pl_ids = [r["id"] for r in pl]
    for i in range(0, len(pl_ids), chunk):
        for r in await graph.read(
            "UNWIND $ids AS id MATCH (p:Playlist {playlist_id: id})-[:INCLUDES]->(s:Song) "
            "RETURN id, collect(s.song_id) AS songs",
            ids=pl_ids[i : i + chunk],
            timeout_s=600,
        ):
            inc += [(int(r["id"]), int(x)) for x in r["songs"]]
    for i in range(0, len(songs), chunk * 4):
        for r in await graph.read(
            "UNWIND $ids AS id MATCH (s:Song {song_id: id})-[:PERFORMED_BY]->(a:Artist) "
            "RETURN id, collect(a.artist_id) AS artists",
            ids=songs[i : i + chunk * 4],
            timeout_s=600,
        ):
            perf += [(int(r["id"]), int(x)) for x in r["artists"]]
    return _save(
        out,
        np.array([int(x) for x in songs]),
        np.array([int(r["id"]) for r in pl]),
        np.array([r["like_cnt"] or 0 for r in pl]),
        np.array([int(x) for x in artists]),
        np.array(inc, dtype=np.int64).reshape(-1, 2),
        np.array(perf, dtype=np.int64).reshape(-1, 2),
        t,
    )


def _save(out, songs, pl_ids, like, artists, inc, perf, t) -> dict:
    songs = np.unique(songs)
    order = np.argsort(pl_ids)
    playlists, like_cnt = pl_ids[order], like[order]
    artists = np.unique(artists)

    def index(ids: np.ndarray, keys: np.ndarray) -> np.ndarray:
        i = np.searchsorted(ids, keys)
        if not (i < len(ids)).all() or not (ids[np.minimum(i, len(ids) - 1)] == keys).all():
            raise ValueError("edge to an unknown node")
        return i

    def csr(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
        m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
        m.sum_duplicates()
        m.data[:] = 1.0
        return m

    sp_m = csr(index(songs, inc[:, 1]), index(playlists, inc[:, 0]), (len(songs), len(playlists)))
    sa_m = csr(index(songs, perf[:, 0]), index(artists, perf[:, 1]), (len(songs), len(artists)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        song_ids=songs,
        playlist_ids=playlists,
        like_cnt=like_cnt,
        artist_ids=artists,
        sp_indptr=sp_m.indptr,
        sp_indices=sp_m.indices,
        sa_indptr=sa_m.indptr,
        sa_indices=sa_m.indices,
    )
    return {
        "songs": len(songs),
        "playlists": len(playlists),
        "artists": len(artists),
        "includes": int(sp_m.nnz),
        "performed_by": int(sa_m.nnz),
        "seconds": round(time.perf_counter() - t, 1),
        "path": str(out),
    }


def _shares(deg_p: np.ndarray, deg_a: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge step probability from a song: its playlists share 1 - w, its artists w; a song
    with only one kind gives that kind everything."""
    has_p, has_a = deg_p > 0, deg_a > 0
    share_p = np.where(has_p, np.where(has_a, 1.0 - w, 1.0), 0.0)
    share_a = np.where(has_a, np.where(has_p, w, 1.0), 0.0)
    return share_p / np.maximum(deg_p, 1), share_a / np.maximum(deg_a, 1)


class PPR:
    def __init__(self, path: Path):
        z = np.load(path)
        self.song_ids = z["song_ids"]
        self.playlist_ids = z["playlist_ids"]
        self.like_cnt = z["like_cnt"]
        n_s, n_p, n_a = len(self.song_ids), len(self.playlist_ids), len(z["artist_ids"])
        ind = z["sp_indices"]
        self.sp = sp.csr_matrix((np.ones(len(ind)), ind, z["sp_indptr"]), shape=(n_s, n_p))
        self.ps = self.sp.T.tocsr()
        ind = z["sa_indices"]
        self.sa = sp.csr_matrix((np.ones(len(ind)), ind, z["sa_indptr"]), shape=(n_s, n_a))
        self.as_ = self.sa.T.tocsr()
        self.deg_p = np.diff(self.sp.indptr).astype(float)  # song -> number of playlists
        self.deg_a = np.diff(self.sa.indptr).astype(float)
        self.inv_pl = 1.0 / np.maximum(np.diff(self.ps.indptr), 1)  # playlist -> 1 / its songs
        self.inv_ar = 1.0 / np.maximum(np.diff(self.as_.indptr), 1)

    def _idx(self, ids: np.ndarray, keys: Sequence[str]) -> np.ndarray:
        """Positions of the known ids among `keys`; unknown ones are dropped."""
        k = np.array([int(x) for x in keys if str(x).isdigit()], dtype=np.int64)
        i = np.searchsorted(ids, k)
        ok = i < len(ids)
        i, k = i[ok], k[ok]
        return i[ids[i] == k]

    def songs(self, keys: Sequence[str]) -> np.ndarray:
        return self._idx(self.song_ids, keys)

    def playlists(self, keys: Sequence[str]) -> np.ndarray:
        return self._idx(self.playlist_ids, keys)

    def playlist_songs(self, p: int) -> np.ndarray:
        return self.ps.indices[self.ps.indptr[p] : self.ps.indptr[p + 1]]

    def walk(
        self,
        seeds: Sequence[Sequence[str]],
        exclude_playlists: Sequence[Sequence[str]],
        alpha: float = ALPHA,
        artist_w: float = ARTIST_W,
        tol: float = TOL,
        steps: int | None = None,
    ) -> np.ndarray:
        """Visit shares, one column per query (songs x queries); each column sums to 1.
        `steps` fixes the number of steps instead of deriving it from `tol`."""
        n, b = len(self.song_ids), len(seeds)
        r = np.zeros((n, b))
        for j, s in enumerate(seeds):
            i = self.songs(s)
            if len(i):
                r[i, j] = 1.0 / len(i)
        deg_a = self.deg_a if artist_w > 0 else np.zeros_like(self.deg_a)
        fp, fa = _shares(self.deg_p, deg_a, artist_w)
        # songs in a held-out playlist: their degree without it, in that query's column only
        fixes, cut = [], []
        for j, pls in enumerate(exclude_playlists):
            pidx = self.playlists(pls)
            cut.append(pidx)
            if len(pidx):
                rows, cnt = np.unique(
                    np.concatenate([self.playlist_songs(p) for p in pidx]), return_counts=True
                )
                fp_r, fa_r = _shares(self.deg_p[rows] - cnt, deg_a[rows], artist_w)
                fixes.append((j, rows, fp_r, fa_r))
        x = r.copy()
        if steps is None:
            steps = max(1, math.ceil(math.log(tol) / math.log(1.0 - alpha)))
        for _ in range(steps):
            xp = x * fp[:, None]
            xa = x * fa[:, None] if artist_w > 0 else None
            for j, rows, fp_r, fa_r in fixes:
                xp[rows, j] = x[rows, j] * fp_r
                if xa is not None:
                    xa[rows, j] = x[rows, j] * fa_r
            yp = self.ps @ xp
            for j, pidx in enumerate(cut):
                yp[pidx, j] = 0.0
            z = self.sp @ (yp * self.inv_pl[:, None])
            if xa is not None:
                z += self.sa @ ((self.as_ @ xa) * self.inv_ar[:, None])
            x = (1.0 - alpha) * z
            x += (1.0 - x.sum(axis=0)) * r  # the restart, and mass that left the graph
        return x

    def rank_scored(
        self, x: np.ndarray, exclude_songs: Sequence[str], k: int, beta: float = BETA
    ) -> list[tuple[str, float]]:
        """Top k of one walk column; beta > 0 divides by playlist count^beta (popularity)."""
        score = x / np.maximum(self.deg_p, 1) ** beta if beta else x.copy()
        score[self.songs(exclude_songs)] = 0.0
        k = min(k, int((score > 0).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.lexsort((self.song_ids[top], -score[top]))]
        return [(str(self.song_ids[i]), float(score[i])) for i in top]

    def rank(
        self, x: np.ndarray, exclude_songs: Sequence[str], k: int, beta: float = BETA
    ) -> list[str]:
        return [sid for sid, _ in self.rank_scored(x, exclude_songs, k, beta)]

    def cooccur(
        self,
        seeds: Sequence[str],
        exclude_playlists: Sequence[str],
        exclude_songs: Sequence[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """Degree-weighted co-occurrence: one step of the walk (docs/results.md, "홈 피드:
        Personalized PageRank와 동시등장").

        score(s) = mean over seeds of
          0.8 * sum over playlists p holding seed and s of 1 / (playlists of seed * songs of p)
        + 0.2 * the same over shared artists.
        Every playlist of a seed counts, so no most-liked cap is needed: a seed in 5,000
        playlists gives each 1/5,000.
        """
        x = self.walk([seeds], [exclude_playlists], steps=1)[:, 0]
        return self.rank_scored(x, exclude_songs, k)

    def top(
        self,
        seeds: Sequence[str],
        exclude_playlists: Sequence[str],
        exclude_songs: Sequence[str],
        k: int,
        **walk: float,
    ) -> list[str]:
        x = self.walk([seeds], [exclude_playlists], **walk)[:, 0]
        return self.rank(x, exclude_songs, k)
