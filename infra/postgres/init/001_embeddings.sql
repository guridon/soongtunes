-- Embedding platform: versioned, re-buildable, and switchable vectors.
-- One row per model version; reads only ever touch the version whose status is 'active'.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

CREATE TABLE IF NOT EXISTS embedding_model (
    version        text PRIMARY KEY,
    entity         text NOT NULL CHECK (entity IN ('song', 'tag')),
    model_name     text NOT NULL,
    dim            int  NOT NULL CHECK (dim > 0),
    normalized     bool NOT NULL,
    template       text NOT NULL,
    template_hash  text NOT NULL,
    status         text NOT NULL DEFAULT 'building'
                   CHECK (status IN ('building', 'ready', 'active', 'retired')),
    created_at     timestamptz NOT NULL DEFAULT now(),
    activated_at   timestamptz
);

-- at most one active version per entity: switching is a pointer flip, rollback flips it back
CREATE UNIQUE INDEX IF NOT EXISTS embedding_model_one_active
    ON embedding_model (entity) WHERE status = 'active';

-- `vec` has no fixed dimension so different versions can coexist; each version gets its own
-- partial index over `vec::halfvec(<dim>)` (HNSW, or IVFFlat past 200k rows), created by the
-- backfill job once it finishes
CREATE TABLE IF NOT EXISTS song_embedding (
    version     text NOT NULL REFERENCES embedding_model (version),
    song_id     text NOT NULL,
    vec         vector NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (version, song_id)
);

CREATE TABLE IF NOT EXISTS tag_embedding (
    version     text NOT NULL REFERENCES embedding_model (version),
    tag         text NOT NULL,
    vec         vector NOT NULL,
    PRIMARY KEY (version, tag)
);

-- resumable backfill: the job restarts after the last committed key
CREATE TABLE IF NOT EXISTS embedding_backfill (
    version      text PRIMARY KEY REFERENCES embedding_model (version),
    last_key     text,
    rows_done    bigint NOT NULL DEFAULT 0,
    finished_at  timestamptz,
    updated_at   timestamptz NOT NULL DEFAULT now()
);
