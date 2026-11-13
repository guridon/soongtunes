"""OpenSearch index definitions (BM25 side of hybrid search).

Indices are versioned (`songs_v<N>`) behind an alias (`songs`), so a rebuild is
build-new → verify → swap alias, the same blue/green move as the embedding registry.
"""

KO_ANALYSIS = {
    "tokenizer": {
        "nori_mixed": {
            "type": "nori_tokenizer",
            "decompound_mode": "mixed",
            "discard_punctuation": True,
        },
        # 2-3 grams catch partial titles/names that morphological analysis splits differently
        # (e.g. "좋은날" vs "좋은 날") and romanized/English names.
        "gram": {"type": "ngram", "min_gram": 2, "max_gram": 3, "token_chars": ["letter", "digit"]},
    },
    "analyzer": {
        "ko": {
            "type": "custom",
            "tokenizer": "nori_mixed",
            "filter": ["lowercase", "nori_part_of_speech", "nori_readingform"],
        },
        "gram": {"type": "custom", "tokenizer": "gram", "filter": ["lowercase"]},
    },
    "normalizer": {"lower": {"type": "custom", "filter": ["lowercase", "trim"]}},
}

_TEXT_KO = {
    "type": "text",
    "analyzer": "ko",
    "fields": {
        "gram": {"type": "text", "analyzer": "gram"},
        "exact": {"type": "keyword", "normalizer": "lower", "ignore_above": 256},
    },
}

LLM_TAGS_FIELD = {"type": "text", "analyzer": "ko", "fields": {"kw": {"type": "keyword"}}}

SONGS_BODY = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "analysis": KO_ANALYSIS,
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "song_id": {"type": "keyword"},
            "title": _TEXT_KO,
            "artists": _TEXT_KO,
            "artist_ids": {"type": "keyword"},
            "album": {"type": "text", "analyzer": "ko"},
            "album_id": {"type": "keyword"},
            "genres": {"type": "keyword"},
            "tags": {"type": "text", "analyzer": "ko", "fields": {"kw": {"type": "keyword"}}},
            # Phase 4 LLM-predicted tags, filled by `stunes llm-tags project`, not by the export
            "llm_tags": LLM_TAGS_FIELD,
            "playlists": {"type": "text", "analyzer": "ko"},
            "playlist_count": {"type": "integer"},
            "year": {"type": "short"},
            "issue_date": {"type": "keyword", "index": False},
        },
    },
}

ENTITIES_BODY = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "analysis": KO_ANALYSIS,
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "kind": {"type": "keyword"},
            "key": {"type": "keyword"},
            "name": _TEXT_KO,
            "popularity": {"type": "integer"},
        },
    },
}

SONG_FIELDS = [
    "song_id",
    "title",
    "artists",
    "artist_ids",
    "album",
    "genres",
    "tags",
    "playlist_count",
    "year",
    "issue_date",
]
