"""BM25 retrieval over OpenSearch (Nori-analyzed Korean + 2-3 gram subfields)."""

from dataclasses import dataclass, field
from typing import Any

from opensearchpy import AsyncOpenSearch

from stunes.search_index import SONG_FIELDS

# Mild popularity prior: x1 for an unseen song, ~x1.7 for one in 1,000 playlists.
# It breaks ties like "좋은날 (by an artist named 좋은날, 8 playlists)" vs
# "좋은 날 (아이유, 919 playlists)" without letting popularity override text match.
_POPULARITY = "_score * (1 + 0.1 * Math.log1p(doc['playlist_count'].value))"


@dataclass
class Exclusions:
    song_ids: list[str] = field(default_factory=list)
    artist_ids: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def clauses(self, llm_tags: bool = False) -> list[dict]:
        out: list[dict] = []
        if self.song_ids:
            out.append({"ids": {"values": self.song_ids}})
        if self.artist_ids:
            out.append({"terms": {"artist_ids": self.artist_ids}})
        if self.genres:
            out.append({"terms": {"genres": self.genres}})
        if self.tags:
            out.append({"terms": {"tags.kw": self.tags}})
            if llm_tags:
                out.append({"terms": {"llm_tags.kw": self.tags}})
        return out


@dataclass
class EntityHit:
    kind: str
    key: str
    name: str
    popularity: int
    score: float
    exact: bool


class BM25:
    def __init__(
        self, client: AsyncOpenSearch, songs_index: str, entities_index: str, llm_tags: bool = False
    ):
        self._os = client
        self._llm_tags = llm_tags  # also match the LLM-predicted tags (separate field)
        self._songs = songs_index
        self._entities = entities_index

    async def _search(self, index: str, body: dict) -> list[dict]:
        resp = await self._os.search(index=index, body=body)
        return resp["hits"]["hits"]

    async def search(
        self, text: str, k: int = 100, exclude: Exclusions | None = None
    ) -> list[tuple[str, float]]:
        """Free-text song search: titles and artists weigh most, tags/playlist titles add mood."""
        if not text.strip():
            return []
        query = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": text,
                            "type": "most_fields",
                            "fields": [
                                "title^3",
                                "artists^3",
                                "album",
                                "tags^1.5",
                                "playlists^0.5",
                                *(["llm_tags^1.5"] if self._llm_tags else []),
                            ],
                        }
                    },
                    {
                        "multi_match": {
                            "query": text,
                            "type": "most_fields",
                            "fields": ["title.gram", "artists.gram"],
                            "boost": 0.5,
                        }
                    },
                ],
                "minimum_should_match": 1,
                "must_not": (exclude or Exclusions()).clauses(self._llm_tags),
            }
        }
        hits = await self._search(
            self._songs,
            {
                "size": k,
                "_source": False,
                "query": {"script_score": {"query": query, "script": {"source": _POPULARITY}}},
            },
        )
        return [(h["_id"], h["_score"]) for h in hits]

    async def find_song(
        self, title: str, artist: str | None, k: int = 5
    ) -> list[tuple[str, float]]:
        """Known-item lookup: the title must match; the artist, when given, boosts strongly."""
        should: list[dict] = [{"term": {"title.exact": {"value": title.lower(), "boost": 5}}}]
        if artist:
            should += [
                {"term": {"artists.exact": {"value": artist.lower(), "boost": 8}}},
                {"match": {"artists": {"query": artist, "boost": 3}}},
                {"match": {"artists.gram": {"query": artist, "boost": 1}}},
            ]
        query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": title,
                            "type": "most_fields",
                            "fields": ["title^2", "title.gram"],
                        }
                    }
                ],
                "should": should,
            }
        }
        hits = await self._search(
            self._songs,
            {
                "size": k,
                "_source": False,
                "query": {"script_score": {"query": query, "script": {"source": _POPULARITY}}},
            },
        )
        return [(h["_id"], h["_score"]) for h in hits]

    async def link(self, name: str, kind: str, k: int = 5) -> list[EntityHit]:
        """Candidate entities of one kind for a surface form, exact-name matches first."""
        name = name.strip()
        if not name:
            return []
        body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [{"term": {"kind": kind}}],
                            "should": [
                                {"term": {"name.exact": {"value": name.lower(), "boost": 10}}},
                                {"match": {"name": {"query": name, "boost": 2}}},
                                {"match": {"name.gram": {"query": name}}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                    "script": {
                        "source": "_score * (1 + 0.05 * Math.log1p(doc['popularity'].value))"
                    },
                }
            },
        }
        hits = await self._search(self._entities, body)
        out = []
        for h in hits:
            src = h["_source"]
            out.append(
                EntityHit(
                    kind=src["kind"],
                    key=src["key"],
                    name=src["name"],
                    popularity=src["popularity"],
                    score=h["_score"],
                    exact=src["name"].strip().lower() == name.lower(),
                )
            )
        out.sort(key=lambda e: (not e.exact, -e.score))
        return out

    async def get_songs(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Song cards keyed by id. The index is the read model for display fields."""
        if not ids:
            return {}
        resp = await self._os.mget(
            index=self._songs, body={"ids": ids}, params={"_source_includes": ",".join(SONG_FIELDS)}
        )
        return {d["_id"]: d["_source"] for d in resp["docs"] if d.get("found")}

    async def ping(self) -> bool:
        return bool(await self._os.ping())
