
import logging
from stunes.core.config import settings
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GDS_GRAPH_NAME = "stunes-songs-playlists"

def run_cypher_query(tx, query, params=None):
    logging.info(f"Executing query: {query}")
    result = tx.run(query, params or {})
    summary = result.consume()
    logging.info(f"Query finished. {summary.counters}")
    return summary

def main():
    driver = None
    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
        driver.verify_connectivity()
        logging.info("✅ Neo4j 연결")

        with driver.session(database="neo4j") as session:
            logging.info(f"Checking for existing GDS graph projection '{GDS_GRAPH_NAME}'...")
            summary = session.run(
                "CALL gds.graph.exists($graph_name) YIELD exists",
                {"graph_name": GDS_GRAPH_NAME}
            ).single()
            if summary and summary['exists']:
                session.execute_write(
                    run_cypher_query,
                    query=f"CALL gds.graph.drop('{GDS_GRAPH_NAME}') YIELD graphName",
                )
                logging.info(f"🗑️ 기존 GDS 그래프 프로젝션 '{GDS_GRAPH_NAME}' 삭제")

            logging.info("Creating new GDS graph projection...")
            session.execute_write(
                run_cypher_query,
                query="""
                CALL gds.graph.project(
                    $graph_name,
                    ['Song', 'Playlist'],
                    {
                        INCLUDES: {
                            type: 'INCLUDES',
                            orientation: 'UNDIRECTED'
                        }
                    }
                )
                """,
                params={"graph_name": GDS_GRAPH_NAME}
            )
            logging.info(f"✨ GDS 그래프 프로젝션 '{GDS_GRAPH_NAME}' 생성완.")
            logging.info("Running Louvain algorithm and writing back community IDs...")
            session.execute_write(
                run_cypher_query,
                query="""
                CALL gds.louvain.write(
                    $graph_name,
                    { writeProperty: 'communityId' }
                )
                """,
                params={"graph_name": GDS_GRAPH_NAME}
            )
            logging.info("🎼 communityId 속성 추가완.")

            # stats_result = session.run("""
            #     MATCH (s:Song)
            #     WHERE s.communityId IS NOT NULL
            #     RETURN s.communityId AS community, count(s) AS size
            #     ORDER BY size DESC
            #     LIMIT 10
            # """).data()
            # logging.info("--- Top 10 Largest Communities ---")
            # for record in stats_result:
            #     logging.info(f"  Community {record['community']}: {record['size']} songs")
            # logging.info("---------------------------------")


    except Exception as e:
        logging.error(f"❌ 오류: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logging.info("🚪 Neo4j 드라이버 연결 닫!")

if __name__ == "__main__":
    main()
