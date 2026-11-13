import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from stunes.core.config import settings
from neo4j import GraphDatabase
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

CHUNK_SIZE = 5000
MAX_WORKERS = 8
API_BATCH_SIZE = 256
NEO4J_WRITE_BATCH_SIZE = 1000

def get_total_song_count(driver):
    """🎶DB에 있는 모든 노래 수 가져옴"""
    logging.info("Fetching total song count...")
    with driver.session(database="neo4j") as session:
        result = session.run("MATCH (s:Song) RETURN count(s) AS total").single()
        total = result["total"] if result else 0
        logging.info(f"💿 노래: 총 {total}개")
        return total

def get_song_data_chunk(driver, skip, limit):
    """
    노래 데이터 chunk 단위로 가져옴
    """
    with driver.session(database="neo4j") as session:
        main_query = """
        MATCH (s:Song)
        WITH s
        ORDER BY s.song_id
        SKIP $skip LIMIT $limit
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        OPTIONAL MATCH (s)-[:IN_ALBUM]->(al:Album)
        OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
        OPTIONAL MATCH (p:Playlist)-[:INCLUDES]->(s)
        OPTIONAL MATCH (s)-[:HAS_GENRE]->(g:Genre)
        RETURN s.song_id AS song_id,
               s.title AS title,
               a.name AS artist_name,
               al.title AS album_title,
               COLLECT(DISTINCT g.name) AS genres,
               COLLECT(DISTINCT t.name) AS tags,
               COLLECT(DISTINCT p.title) AS playlists
        """
        song_results = session.run(main_query, skip=skip, limit=limit).data()

        if not song_results:
            return []

        all_genres_in_chunk = list(set(genre for song in song_results for genre in song.get('genres', [])))

        subgenre_lookup = {}
        if all_genres_in_chunk:
            subgenre_query = """
            UNWIND $genres AS genre_name
            MATCH (g:Genre {name: genre_name})-[:CONTAINS]->(sg:SubGenre)
            RETURN g.name AS genre, COLLECT(DISTINCT sg.name) AS subgenres
            """
            subgenre_results = session.run(subgenre_query, genres=all_genres_in_chunk).data()
            subgenre_lookup = {res['genre']: res['subgenres'] for res in subgenre_results}

        for song in song_results:
            song_subgenres = []
            for genre in song.get('genres', []):
                if genre in subgenre_lookup:
                    song_subgenres.extend(subgenre_lookup[genre])
            song['subgenres'] = list(set(song_subgenres)) 

        return song_results

def create_text_for_embedding(song_data):
    parts = []
    if song_data.get("title"): parts.append(f"Title: {song_data.get('title')}")
    if song_data.get("artist_name"): parts.append(f"Artist: {song_data.get('artist_name')}")
    if song_data.get("album_title"): parts.append(f"Album: {song_data.get('album_title')}")
    if song_data.get("genres"): parts.append(f"Genres: {', '.join(song_data.get('genres'))}")
    if song_data.get("subgenres"): parts.append(f"Subgenres: {', '.join(song_data.get('subgenres'))}")
    if song_data.get("tags"): parts.append(f"Tags: {', '.join(song_data.get('tags'))}")
    if song_data.get("playlists"): parts.append(f"Playlists: {', '.join(song_data.get('playlists'))}")
    return "; ".join(parts)

def update_embeddings_in_batch(tx, batch_data):
    query = """
    UNWIND $batch as item
    MATCH (s:Song {song_id: item.song_id})
    SET s.embedding = item.embedding
    """
    tx.run(query, batch=batch_data)

def initialize_vector_index():
    logging.info("🌟 벡터 인덱스를 초기화")
    start_time = time.time()
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
        driver.verify_connectivity()
        logging.info("🌈 Neo4j 연결 완")

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=settings.openai_api_key)
        
        total_songs = get_total_song_count(driver)
        if not total_songs:
            logging.warning("No songs found. Exiting.")
            return

        songs_processed = 0
        with tqdm(total=total_songs, desc="Embedding Songs", unit="song") as pbar:
            for i in range(0, total_songs, CHUNK_SIZE):
                chunk_start_time = time.time()
                num_chunks = (total_songs + CHUNK_SIZE - 1) // CHUNK_SIZE
                logging.info(f"--- 📦 chunk {i // CHUNK_SIZE + 1}/{num_chunks} 처리, (songs {i+1}-{i+CHUNK_SIZE}) ---")

                pbar.set_description(f"Chunk {i // CHUNK_SIZE + 1} - Fetching")
                chunk_songs = get_song_data_chunk(driver, i, CHUNK_SIZE)
                chunk_size = len(chunk_songs)
                if not chunk_songs:
                    logging.warning(f"Expected to fetch songs for chunk starting at {i}, but got none. Stopping.")
                    break

                texts_to_embed = [create_text_for_embedding(song) for song in chunk_songs]
                song_ids = [song['song_id'] for song in chunk_songs]

                pbar.set_description(f"Chunk {i // CHUNK_SIZE + 1} - Embedding")
                embedding_start_time = time.time()
                chunk_embeddings = [None] * chunk_size
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_index = {}
                    for j in range(0, len(texts_to_embed), API_BATCH_SIZE):
                        batch_texts = texts_to_embed[j:j + API_BATCH_SIZE]
                        future = executor.submit(embedding_model.embed_documents, batch_texts)
                        future_to_index[future] = j
                    
                    for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="  - Embeddings", unit="batch", leave=False):
                        start_index = future_to_index[future]
                        try:
                            # Add a 60-second timeout to prevent indefinite hangs
                            result_embeddings = future.result(timeout=60)
                            end_index = start_index + len(result_embeddings)
                            chunk_embeddings[start_index:end_index] = result_embeddings
                        except Exception as exc:
                            # Log timeout errors or other exceptions
                            logging.error(f"A batch starting at index {start_index} failed: {exc}")
                
                embedding_time = time.time() - embedding_start_time
                logging.info(f"Embedding generation for chunk finished in {embedding_time:.2f} seconds.")

                embeddings_to_update = [{"song_id": song_id, "embedding": emb} for song_id, emb in zip(song_ids, chunk_embeddings) if emb is not None]
                
                if not embeddings_to_update:
                    logging.warning("No embeddings were generated for this chunk. Skipping DB update.")
                    pbar.update(chunk_size)
                    continue

                pbar.set_description(f"Chunk {i // CHUNK_SIZE + 1} - Updating DB")
                db_update_start_time = time.time()
                with driver.session(database="neo4j") as session:
                    for j in tqdm(range(0, len(embeddings_to_update), NEO4J_WRITE_BATCH_SIZE), desc="  - DB Update", unit="batch", leave=False):
                        batch_data = embeddings_to_update[j:j + NEO4J_WRITE_BATCH_SIZE]
                        session.execute_write(update_embeddings_in_batch, batch_data)
                        logging.info(f"  ... updated {j + len(batch_data)} / {len(embeddings_to_update)} embeddings in DB for this chunk.")
                
                db_update_time = time.time() - db_update_start_time
                logging.info(f"💾 DB 업데이트 완료-{db_update_time:.2f}초 소요")
                
                chunk_time = time.time() - chunk_start_time
                songs_processed += chunk_size
                if songs_processed > 0:
                    avg_time_per_song = (time.time() - start_time) / songs_processed
                    estimated_remaining = (total_songs - songs_processed) * avg_time_per_song
                    logging.info(f"Chunk finished in {chunk_time:.2f}s. Total processed: {songs_processed}/{total_songs}. Estimated time remaining: {estimated_remaining:.2f}s")

                pbar.update(chunk_size)

        with driver.session(database="neo4j") as session:
            index_query = """
            CREATE VECTOR INDEX `song_embeddings` IF NOT EXISTS
            FOR (s:Song) ON (s.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 1536, 
             `vector.similarity_function`: 'cosine'
            }}
            """
            logging.info("Creating vector index 'song_embeddings' if it doesn't exist...")
            session.run(index_query)
        
        total_time = time.time() - start_time
        logging.info(f"Vector index initialization complete! Total time: {total_time:.2f} seconds.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logging.info("Neo4j driver closed.")

if __name__ == "__main__":
    initialize_vector_index()