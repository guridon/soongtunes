-- The chat's graph topology as tables. Loaded from the same CSV files as the bench Neo4j
-- (backend/src/stunes/bench/graph_sql_prep.py export). Ids are the graph's string ids, all
-- canonical integers, stored as int; song_id order (the served tie-break) is sid::text "C".

DROP TABLE IF EXISTS song, playlist, artist, genre, tag, includes, song_tag, song_genre, song_artist;

CREATE TABLE song (
  sid        int PRIMARY KEY,
  popularity int,
  issue_date text,
  year       int            -- toInteger(left(issue_date, 4)), computed by Neo4j at export
);
CREATE TABLE playlist (pid int PRIMARY KEY, like_cnt int);
CREATE TABLE artist   (aid int PRIMARY KEY);
CREATE TABLE genre    (genre_id text PRIMARY KEY, name text, song_count int);  -- names repeat
CREATE TABLE tag      (tid int PRIMARY KEY, name text NOT NULL, song_count int);

CREATE TABLE includes    (pid int NOT NULL, sid int NOT NULL);
CREATE TABLE song_tag    (sid int NOT NULL, tid int NOT NULL);
CREATE TABLE song_genre  (sid int NOT NULL, genre_id text NOT NULL);
CREATE TABLE song_artist (sid int NOT NULL, aid int NOT NULL);
