-- Phase 3: song -> tag weights (playlist co-occurrence counts from the graph export), used by
-- the on-demand user_tag_pref feature. Loaded by `stunes features load-tags`.
CREATE TABLE IF NOT EXISTS song_tag (
    song_id  text NOT NULL,
    tag      text NOT NULL,
    weight   int  NOT NULL,
    PRIMARY KEY (song_id, tag)
);
