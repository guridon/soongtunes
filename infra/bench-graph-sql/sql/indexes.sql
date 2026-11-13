-- Both directions of every edge table, as a graph store has adjacency both ways, plus the
-- lookups the served Cypher has indexes for (Song.popularity, Tag.name, Genre.name).
CREATE INDEX song_popularity    ON song (popularity DESC) INCLUDE (sid);
CREATE INDEX tag_name           ON tag (name) INCLUDE (tid, song_count);
CREATE INDEX genre_name         ON genre (name) INCLUDE (genre_id, song_count);
CREATE INDEX includes_sid_pid   ON includes (sid, pid);
CREATE INDEX includes_pid_sid   ON includes (pid, sid);
CREATE INDEX song_tag_tid_sid   ON song_tag (tid, sid);
CREATE INDEX song_tag_sid_tid   ON song_tag (sid, tid);
CREATE INDEX song_genre_g_sid   ON song_genre (genre_id, sid);
CREATE INDEX song_genre_sid_g   ON song_genre (sid, genre_id);
CREATE INDEX song_artist_a_sid  ON song_artist (aid, sid);
CREATE INDEX song_artist_sid_a  ON song_artist (sid, aid);
VACUUM (ANALYZE);
