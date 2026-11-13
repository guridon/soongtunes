-- Phase 2/3: features written by the event consumer, plus experiment bookkeeping.

-- Feature store: key '<feature_group>#<entity_id>', value msgpack.
CREATE TABLE IF NOT EXISTS feature (
    pk          text PRIMARY KEY,
    value       bytea NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS song_stats (
    song_id     text PRIMARY KEY,
    plays       bigint NOT NULL DEFAULT 0,
    skips       bigint NOT NULL DEFAULT 0,
    likes       bigint NOT NULL DEFAULT 0,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- Idempotent consumer: an event's effects are applied in the same transaction that inserts
-- its id here, so a redelivered event is a no-op and a crash can't apply it half-way.
CREATE TABLE IF NOT EXISTS processed_event (
    event_id      uuid PRIMARY KEY,
    processed_at  timestamptz NOT NULL DEFAULT now()
);

-- Every delivery attempt (before dedup), for loss/duplicate/latency accounting in experiments.
-- UNLOGGED: losing it on a Postgres crash only loses experiment data, and it's cheaper to write.
CREATE UNLOGGED TABLE IF NOT EXISTS ingest_attempt (
    run_id        text NOT NULL,
    event_id      uuid NOT NULL,
    part          int NOT NULL,
    "offset"      bigint NOT NULL,
    event_ts      bigint NOT NULL,
    duplicate     bool NOT NULL,
    processed_at  timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS ingest_attempt_run ON ingest_attempt (run_id, event_id);
