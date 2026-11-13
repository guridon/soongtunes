// The served indexes (backend/src/stunes/jobs/graph_indexes.py) on the labels the bench holds.
CREATE RANGE INDEX song_id_index IF NOT EXISTS FOR (s:Song) ON (s.song_id);
CREATE RANGE INDEX artist_id_index IF NOT EXISTS FOR (a:Artist) ON (a.artist_id);
CREATE RANGE INDEX tag_name_index IF NOT EXISTS FOR (t:Tag) ON (t.name);
CREATE RANGE INDEX genre_name_index IF NOT EXISTS FOR (g:Genre) ON (g.name);
CREATE RANGE INDEX playlist_id_index IF NOT EXISTS FOR (p:Playlist) ON (p.playlist_id);
CREATE RANGE INDEX song_popularity_index IF NOT EXISTS FOR (s:Song) ON (s.popularity);
CALL db.awaitIndexes(1800);
