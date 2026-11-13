import logging
import pandas as pd
import random
from stunes.core.config import settings
from neo4j import GraphDatabase
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_SEED_SONGS = 1000
EXCLUDED_GENRES = ['클래식', '재즈', '뉴에이지', 'J-POP', '월드뮤직', 'CCM', '종교음악', '국악']

class DatasetGenerator:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
        self.dataset = []
        self.conversation_id = 0

    def close(self):
        self.driver.close()

    def _run_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def _get_facets_for_song(self, song_id: str) -> Dict[str, Any]:
        """하나의 노래에 대한 모든 관련 속성 정보를 가져옵니다."""
        facet_query = """
        MATCH (s:Song {song_id: $song_id})
        OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        OPTIONAL MATCH (s)-[:HAS_GENRE]->(g:Genre)
        OPTIONAL MATCH (s)-[:IN_ALBUM]->(al:Album)
        OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag) // '분위기'로 활용
        RETURN
            s.issue_date AS issue_date,
            COLLECT(DISTINCT a.name) AS artists,
            COLLECT(DISTINCT g.name) AS genres,
            COLLECT(DISTINCT al.title) AS albums,
            COLLECT(DISTINCT t.name) AS tags
        """
        results = self._run_query(facet_query, {"song_id": song_id})
        return results[0] if results else {}

    def generate_scenarios(self):
        """다양한 속성을 기반으로 다채로운 대화 시나리오를 생성합니다."""
        
        logging.info(f"Fetching {NUM_SEED_SONGS} random, popular seed songs...")
        seed_song_fetch_query = """
        MATCH (s:Song)-[:HAS_GENRE]->(g:Genre)
        WHERE COUNT { (s)<-[:INCLUDES]-() } > 10 AND NOT g.name IN $excluded_genres AND EXISTS((s)-[:PERFORMED_BY]->(:Artist))
        WITH s
        MATCH (s)-[:PERFORMED_BY]->(a:Artist)
        RETURN s.song_id AS song_id, s.title AS title, a.name AS artist_name
        ORDER BY rand() LIMIT $limit
        """
        seed_results = self._run_query(seed_song_fetch_query, {"limit": NUM_SEED_SONGS, "excluded_genres": EXCLUDED_GENRES})
        logging.info(f"Successfully fetched {len(seed_results)} seed songs.")

        for seed_info in seed_results:
            self.conversation_id += 1
            title, seed_song_id, artist_name = seed_info["title"], seed_info["song_id"], seed_info["artist_name"]
            logging.info(f"--- Conv {self.conversation_id}: Starting with '{title}' ---")

            # 턴 1: 노래 제목으로 검색
            self.dataset.append({
                "conversation_id": self.conversation_id, "turn": 1,
                "input": f'{artist_name}의 {title} 같은 노래 찾아줘',
                "expected_output": str([seed_song_id])
            })

            # 턴 2: 다양한 속성 기반의 후속 질문 생성
            facets = self._get_facets_for_song(seed_song_id)
            possible_scenarios = []

            # 1. 아티스트 시나리오
            if facets.get('artists'):
                artist = facets['artists'][0]
                artist_templates = [
                    f"{artist}의 다른 노래도 5곡 알려줘.",
                    f"{artist} 노래 더 듣고 싶어.",
                    f"{artist}의 다른 추천곡 있어? 5개 정도만.",
                    f"{artist} 노래 중에 또 뭐가 좋아?",
                    f"{artist}의 최신곡도 추천해줘.",
                    f"{artist} 명곡으로만 5곡 뽑아줘.",
                    f"{artist} 노래 리스트 한 번 보여줘.",
                    f"{artist} 래퍼토리 중 최근에 뜬 곡은 뭐야?",
                    f"{artist} 비슷한 곡 또 알려줄 수 있어?",
                    f"{artist} 히트곡 위주로 5개만."
                ]
                possible_scenarios.append({
                    'input': random.choice(artist_templates),
                    'ground_truth_query': "MATCH (a:Artist {name:$val})-[:PERFORMED_BY]->(rec:Song) WHERE rec.song_id <> $sid RETURN rec.song_id AS song_id LIMIT 50",
                    'query_params': {"val": artist, "sid": seed_song_id}
                })

            # 2. 장르 시나리오
            if facets.get('genres'):
                genre = facets['genres'][0]
                genre_templates = [
                    f"방금 그 노래랑 비슷한 {genre} 장르로 5곡 더 추천해줄래?",
                    f"다른 {genre} 노래도 좋아할 것 같아. 5개만 알려줘.",
                    f"{genre} 장르 중에서 다른 곡 없어? 5개 정도.",
                    f"좋네. 이런 {genre} 스타일로 계속 추천해줘.",
                    f"오, 좋은데? 이런 {genre} 느낌으로 몇 개 더.",
                    f"이런 분위기 좋아. 비슷한 {genre} 없어?",
                    f"이 노래 말고 다른 {genre} 노래는 뭐 있어?",
                    f"{genre} 다른 추천곡 5개만.",
                    f"계속. 이런 {genre} 스타일로 더 보여줘.",
                    f"맘에 들어. {genre} 장르로 다섯 곡만 더 추천해줘.",
                    f"{genre} 분위기의 곡 또 있을까?",
                    f"{genre} 계열로 추천해줄 만한 거 있으면 알려줘.",
                    f"내 취향에 맞는 {genre} 노래 추가로 들을 수 있을까?",
                    f"{genre} 스타일 노래 더 듣고 싶어.",
                    f"나한테 어울릴 만한 {genre} 노래 뭐 있을까?",
                    f"{genre} 장르 요즘 인기 있는 곡 있어?",
                ]

                possible_scenarios.append({
                    'input': random.choice(genre_templates),
                    'ground_truth_query': "MATCH (rec:Song)-[:HAS_GENRE]->(:Genre {name:$val}) WHERE rec.song_id <> $sid RETURN rec.song_id AS song_id LIMIT 50",
                    'query_params': {"val": genre, "sid": seed_song_id}
                })

            # 3. 발매 시기(연대) 시나리오
            if facets.get('issue_date') and len(facets.get('issue_date', '')) >= 4:
                try:
                    year = int(facets['issue_date'][:4])
                    decade_start = (year // 10) * 10
                    decade_end = decade_start + 9
                    era_templates = [
                        f"비슷한 시기인 {decade_start}년대 다른 노래는 없어? 5곡만.",
                        f"다른 {decade_start}년대 노래도 추천해줄래?",
                        f"그럼 {decade_start}년대 스타일로 몇 곡 더 알려줘.",
                        f"{decade_start}년대 대표곡 좀 추천해줘.",
                        f"{decade_start}년대 분위기로 들을만한 거 있어?",
                    ]

                    ground_truth_query = """
                    MATCH (rec:Song)
                    WHERE toInteger(substring(rec.issue_date, 0, 4)) >= $start_year 
                      AND toInteger(substring(rec.issue_date, 0, 4)) <= $end_year
                      AND rec.song_id <> $sid
                    RETURN rec.song_id AS song_id LIMIT 50
                    """
                    query_params = {"start_year": decade_start, "end_year": decade_end, "sid": seed_song_id}

                    possible_scenarios.append({
                        'input': random.choice(era_templates),
                        'ground_truth_query': ground_truth_query,
                        'query_params': query_params
                    })
                except (ValueError, TypeError):
                    pass # 날짜 형식이 잘못된 경우

            if not possible_scenarios:
                continue

            # 생성된 시나리오 중 하나를 무작위로 선택
            chosen_scenario = random.choice(possible_scenarios)
            expected_results = self._run_query(chosen_scenario['ground_truth_query'], chosen_scenario['query_params'])
            expected_ids = [res['song_id'] for res in expected_results]

            if expected_ids:
                self.dataset.append({
                    "conversation_id": self.conversation_id, "turn": 2,
                    "input": chosen_scenario['input'],
                    "expected_output": str(expected_ids)
                })
                logging.info(f"  Turn 2 generated: {chosen_scenario['input']}")

    def save_to_csv(self, filename="evaluation_dataset.csv"):
        if not self.dataset:
            logging.warning("Dataset is empty. Nothing to save.")
            return
        df = pd.DataFrame(self.dataset)
        df.to_csv(filename, index=False)
        logging.info(f"✅ Successfully saved dataset to {filename} with {len(df)} rows.")

def main():
    generator = DatasetGenerator()
    try:
        generator.generate_scenarios()
        generator.save_to_csv()
    finally:
        generator.close()

if __name__ == "__main__":
    main()