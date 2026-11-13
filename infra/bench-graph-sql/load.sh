#!/usr/bin/env bash
# Load the exported topology (./data/*.csv.gz) into both bench stores and record load times and
# on-disk sizes in ./out/load.json. Run in this directory on the bench host.
set -euo pipefail
cd "$(dirname "$0")"
: "${BENCH_PASSWORD:?set BENCH_PASSWORD}"
mkdir -p out
dc() { docker compose "$@"; }
now() { date +%s.%N; }
el() { python3 -c "print(round($2 - $1, 1))"; }

# ---------------------------------------------------------------- Neo4j: offline bulk import
dc stop neo4j >/dev/null 2>&1 || true
t0=$(now)
dc run --rm --no-deps -T neo4j neo4j-admin database import full neo4j \
  --nodes=Song=/import/songs.csv.gz --nodes=Playlist=/import/playlists.csv.gz \
  --nodes=Artist=/import/artists.csv.gz --nodes=Genre=/import/genres.csv.gz \
  --nodes=Tag=/import/tags.csv.gz \
  --relationships=INCLUDES=/import/includes.csv.gz --relationships=HAS_TAG=/import/has_tag.csv.gz \
  --relationships=HAS_GENRE=/import/has_genre.csv.gz \
  --relationships=PERFORMED_BY=/import/performed_by.csv.gz \
  --id-type=string --ignore-empty-strings=true --overwrite-destination=true
t1=$(now)
dc up -d neo4j postgres
until dc exec -T neo4j cypher-shell -u neo4j -p "$BENCH_PASSWORD" "RETURN 1" >/dev/null 2>&1; do sleep 3; done
t2=$(now)
dc exec -T neo4j cypher-shell -u neo4j -p "$BENCH_PASSWORD" < sql/neo4j_indexes.cypher
t3=$(now)
neo_import=$(el "$t0" "$t1"); neo_index=$(el "$t2" "$t3")
neo_store=$(dc exec -T neo4j du -sb /data/databases/neo4j | cut -f1)

# ---------------------------------------------------------------- Postgres: COPY, then indexes
until dc exec -T postgres pg_isready -U bench -d bench >/dev/null 2>&1; do sleep 2; done
psql() { dc exec -T postgres psql -U bench -d bench -v ON_ERROR_STOP=1 -q "$@"; }
psql -f /sql/schema.sql
t4=$(now)
copy() {  # table (columns) file
  dc exec -T postgres sh -c "gzip -dc /import/$3.csv.gz | psql -U bench -d bench -v ON_ERROR_STOP=1 -q -c \"\\copy $1 $2 FROM STDIN WITH (FORMAT csv, HEADER true)\""
}
copy song "(sid, popularity, issue_date, year)" songs
copy playlist "(pid, like_cnt)" playlists
copy artist "(aid)" artists
copy genre "(genre_id, name, song_count)" genres
copy tag "(tid, name, song_count)" tags
copy includes "(pid, sid)" includes
copy song_tag "(sid, tid)" has_tag
copy song_genre "(sid, genre_id)" has_genre
copy song_artist "(sid, aid)" performed_by
t5=$(now)
psql -f /sql/indexes.sql
t6=$(now)
pg_copy=$(el "$t4" "$t5"); pg_index=$(el "$t5" "$t6")
pg_size=$(psql -At -c "SELECT pg_database_size('bench')")
pg_tables=$(psql -At -c "SELECT json_object_agg(relname, pg_total_relation_size(oid)) FROM pg_class WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace")
counts=$(psql -At -c "SELECT json_build_object('song', (SELECT count(*) FROM song), 'includes', (SELECT count(*) FROM includes), 'song_tag', (SELECT count(*) FROM song_tag), 'song_genre', (SELECT count(*) FROM song_genre), 'song_artist', (SELECT count(*) FROM song_artist))")
neo_counts=$(dc exec -T neo4j cypher-shell -u neo4j -p "$BENCH_PASSWORD" --format plain \
  "MATCH (:Song)-[r:HAS_TAG]->() WITH count(r) AS t MATCH ()-[i:INCLUDES]->() RETURN t, count(i)" | tail -1)

cat > out/load.json <<EOF
{"neo4j": {"import_s": $neo_import, "index_s": $neo_index, "store_bytes": $neo_store,
           "counts_has_tag_includes": "$neo_counts"},
 "postgres": {"copy_s": $pg_copy, "index_s": $pg_index, "db_bytes": $pg_size,
              "tables": $pg_tables, "counts": $counts}}
EOF
cat out/load.json
