#!/usr/bin/env bash
# Restore the v1 Neo4j dump into the compose volume.
#
#   scripts/restore_neo4j.sh /path/to/neo4j.dump
#
# The v1 dump was written by an Enterprise instance in the *block* store format, which
# Community cannot open (troubleshooting #1). So: load + migrate to `aligned` once with the
# Enterprise image under its evaluation license, then run Community on the same volume.
set -euo pipefail
DUMP=${1:?usage: $0 /path/to/neo4j.dump}
VERSION=2026.10.1
VOLUME=stunes_neo4j_data
cd "$(dirname "$0")/.."

[ -f "$DUMP" ] || { echo "no such file: $DUMP" >&2; exit 1; }
docker compose stop neo4j 2>/dev/null || true

# neo4j-admin wants <dir>/<database>.dump; a hard link avoids copying 12 GB
STAGE=.data/dumps
mkdir -p "$STAGE"
ln -f "$DUMP" "$STAGE/neo4j.dump" 2>/dev/null || cp "$DUMP" "$STAGE/neo4j.dump"
docker volume create "$VOLUME" >/dev/null

docker run --rm -e NEO4J_ACCEPT_LICENSE_AGREEMENT=eval \
  -v "$VOLUME:/data" -v "$PWD/$STAGE:/dumps:ro" "neo4j:$VERSION-enterprise" \
  bash -c 'set -e
    neo4j-admin database load neo4j --from-path=/dumps --overwrite-destination=true
    neo4j-admin database migrate neo4j --to-format=aligned
    neo4j-admin database info neo4j'

docker compose up -d neo4j
echo "waiting for neo4j..."
until docker compose exec -T neo4j cypher-shell -u neo4j -p "$(grep ^NEO4J_PASSWORD .env | cut -d= -f2)" "RETURN 1" >/dev/null 2>&1; do sleep 3; done

# The OpenAI vector index (1536-d) makes the first index population crawl and cannot serve
# queries without an OpenAI key; the Song.embedding property stays, so it can be recreated:
#   CREATE VECTOR INDEX song_embeddings FOR (s:Song) ON s.embedding OPTIONS {indexConfig:
#     {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
docker compose exec -T neo4j cypher-shell -u neo4j -p "$(grep ^NEO4J_PASSWORD .env | cut -d= -f2)" \
  "DROP INDEX song_embeddings IF EXISTS"

docker compose exec -T neo4j cypher-shell -u neo4j -p "$(grep ^NEO4J_PASSWORD .env | cut -d= -f2)" --format plain \
  "MATCH (s:Song) WITH count(s) AS songs MATCH (p:Playlist) RETURN songs, count(p) AS playlists"
echo "expected: 707989 songs, 115071 playlists"
