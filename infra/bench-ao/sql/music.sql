-- SoongTunes' music catalog as relational tables, for the auto-ontology baseline.
-- Exported from the served graph (python -m stunes.bench.ao_prep): the same songs, artists,
-- genres, tags and edges the context set's truth conditions read.
\set ON_ERROR_STOP on
BEGIN;
CREATE TABLE genres  (genre_id text PRIMARY KEY, genre_name text);
CREATE TABLE tags    (tag_id bigint PRIMARY KEY, tag_name text NOT NULL UNIQUE);
CREATE TABLE artists (artist_id bigint PRIMARY KEY, artist_name text NOT NULL);
CREATE TABLE songs   (song_id bigint PRIMARY KEY, title text, popularity int,
                      issue_date text, release_year int);
CREATE TABLE song_artists (song_id bigint NOT NULL, artist_id bigint NOT NULL);
CREATE TABLE song_genres  (song_id bigint NOT NULL, genre_id text NOT NULL);
CREATE TABLE song_tags    (song_id bigint NOT NULL, tag_id bigint NOT NULL);

\copy genres  FROM PROGRAM 'gzip -dc /import/genres.csv.gz'  WITH (FORMAT csv, HEADER true)
\copy tags    FROM PROGRAM 'gzip -dc /import/tags.csv.gz'    WITH (FORMAT csv, HEADER true)
\copy artists FROM PROGRAM 'gzip -dc /import/artists.csv.gz' WITH (FORMAT csv, HEADER true)
\copy songs   FROM PROGRAM 'gzip -dc /import/songs.csv.gz'   WITH (FORMAT csv, HEADER true)
\copy song_artists FROM PROGRAM 'gzip -dc /import/song_artists.csv.gz' WITH (FORMAT csv, HEADER true)
\copy song_genres  FROM PROGRAM 'gzip -dc /import/song_genres.csv.gz'  WITH (FORMAT csv, HEADER true)
\copy song_tags    FROM PROGRAM 'gzip -dc /import/song_tags.csv.gz'    WITH (FORMAT csv, HEADER true)

ALTER TABLE song_artists ADD PRIMARY KEY (song_id, artist_id),
  ADD FOREIGN KEY (song_id) REFERENCES songs, ADD FOREIGN KEY (artist_id) REFERENCES artists;
ALTER TABLE song_genres ADD PRIMARY KEY (song_id, genre_id),
  ADD FOREIGN KEY (song_id) REFERENCES songs, ADD FOREIGN KEY (genre_id) REFERENCES genres;
ALTER TABLE song_tags ADD PRIMARY KEY (song_id, tag_id),
  ADD FOREIGN KEY (song_id) REFERENCES songs, ADD FOREIGN KEY (tag_id) REFERENCES tags;
CREATE INDEX song_artists_artist ON song_artists (artist_id, song_id);
CREATE INDEX song_genres_genre   ON song_genres (genre_id, song_id);
CREATE INDEX song_tags_tag       ON song_tags (tag_id, song_id);
CREATE INDEX songs_popularity    ON songs (popularity DESC);
CREATE INDEX songs_year          ON songs (release_year);
CREATE INDEX artists_name        ON artists (artist_name);
CREATE INDEX genres_name         ON genres (genre_name);

COMMENT ON TABLE songs IS 'Songs in the music catalog (Korean music streaming service).';
COMMENT ON COLUMN songs.title IS 'Song title.';
COMMENT ON COLUMN songs.popularity IS 'Popularity score; higher means more popular.';
COMMENT ON COLUMN songs.issue_date IS 'Release date as YYYYMMDD text.';
COMMENT ON COLUMN songs.release_year IS 'Release year, the first four digits of issue_date.';
COMMENT ON TABLE artists IS 'Artists (singers, bands, composers) who perform songs.';
COMMENT ON TABLE song_artists IS 'Which artists perform which songs (a song can have several).';
COMMENT ON TABLE genres IS 'Music genres such as 발라드, 댄스, 랩/힙합. A genre name can appear under two genre_id codes (domestic and foreign).';
COMMENT ON TABLE song_genres IS 'Which genres each song belongs to (a song can have several).';
COMMENT ON TABLE tags IS 'User tags on songs, mostly moods and situations such as 잔잔한, 드라이브, 새벽.';
COMMENT ON TABLE song_tags IS 'Which tags each song has (a song can have many).';
COMMIT;
ANALYZE;
