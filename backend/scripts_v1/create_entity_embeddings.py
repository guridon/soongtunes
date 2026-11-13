import logging
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from stunes.core.config import settings
from neo4j import GraphDatabase
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- 상수 ---
# 임베딩 모델 및 API 관련
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
API_BATCH_SIZE = 256  
MAX_WORKERS = 8      
NEO4J_WRITE_BATCH_SIZE = 1000

# (Label, Property_Name_To_Embed)
LABELS_AND_PROPERTIES_TO_EMBED = [
    # ("Genre", "name"),  
    # ("Artist", "name"), 
    # ("Tag", "name"),   
    ("Album", "title"),
    ("Playlist", "title")
]

def process_label(driver: GraphDatabase.driver, embedding_model: OpenAIEmbeddings, label: str, property_to_embed: str):
    logging.info(f"--- 🚀 Starting processing for label: [{label}] (property: {property_to_embed}) ---")
    
    with driver.session(database="neo4j") as session:
        # 1. 임베딩할 전체 노드 이름 가져오기
        logging.info(f"[{label}] Fetching all node names from property '{property_to_embed}'...")
        query = f"MATCH (n:{label}) RETURN n.{property_to_embed} AS name" 
        results = session.run(query).data()
        names_to_embed = [record['name'] for record in results if record.get('name')]
        
        if not names_to_embed:
            logging.warning(f"[{label}] No nodes found to embed from property '{property_to_embed}'. Skipping.")
            return

        logging.info(f"[{label}] Found {len(names_to_embed)} nodes to embed.")

        # 2. 텍스트 임베딩 생성
        embeddings = [None] * len(names_to_embed)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {}
            for i in range(0, len(names_to_embed), API_BATCH_SIZE):
                batch_texts = names_to_embed[i:i + API_BATCH_SIZE]
                future = executor.submit(embedding_model.embed_documents, batch_texts)
                future_to_index[future] = i
            
            with tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"  - Embedding {label}s", unit="batch") as pbar:
                for future in pbar:
                    start_index = future_to_index[future]
                    try:
                        result_embeddings = future.result(timeout=60)
                        end_index = start_index + len(result_embeddings)
                        embeddings[start_index:end_index] = result_embeddings
                    except Exception as exc:
                        logging.error(f"[{label}] A batch starting at index {start_index} failed: {exc}")

        embeddings_to_update = [
            {"name": name, "embedding": emb} 
            for name, emb in zip(names_to_embed, embeddings) if emb is not None
        ]

        if not embeddings_to_update:
            logging.warning(f"[{label}] No embeddings were generated. Skipping DB update and index creation.")
            return

        # 3. 생성된 임베딩 -> DB 업데이트
        logging.info(f"[{label}] Updating database with {len(embeddings_to_update)} embeddings...")
        update_query = f"""
        UNWIND $batch AS item
        MATCH (n:{label} {{ {property_to_embed}: item.name}})
        SET n.name_embedding = item.embedding
        """
        for i in tqdm(range(0, len(embeddings_to_update), NEO4J_WRITE_BATCH_SIZE), desc=f"  - Updating DB for {label}", unit="batch"):
            batch_data = embeddings_to_update[i:i + NEO4J_WRITE_BATCH_SIZE]
            session.run(update_query, batch=batch_data)

        logging.info(f"[{label}] Database update complete.")

        # 4. 벡터 인덱스 생성
        index_name = f"{label.lower()}_name_embeddings"
        index_query = f"""
        CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
        FOR (n:{label}) ON (n.name_embedding)
        OPTIONS {{indexConfig: {{
         `vector.dimensions`: {EMBEDDING_DIMENSIONS}, 
         `vector.similarity_function`: 'cosine'
        }}}}
        """
        logging.info(f"[{label}] Creating vector index '{index_name}'...")
        session.run(index_query)
        logging.info(f"[{label}] Index '{index_name}' created successfully.")
        logging.info(f"--- ✅ Finished processing for label: [{label}] ---")


def main():
    logging.info("🌟 Starting entity embedding and index creation process.")
    start_time = time.time()
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
        driver.verify_connectivity()
        logging.info("🌈 Neo4j connection successful.")

        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=settings.openai_api_key)
        
        for label, property_to_embed in LABELS_AND_PROPERTIES_TO_EMBED:
            process_label(driver, embedding_model, label, property_to_embed)
        
        total_time = time.time() - start_time
        logging.info(f"🎉 All entity embeddings and indexes created successfully! Total time: {total_time:.2f} seconds.")

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logging.info("Neo4j driver closed.")

if __name__ == "__main__":
    main()