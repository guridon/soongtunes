
import os
import pandas as pd
import requests
from langfuse import get_client
import json
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

langfuse = get_client(request_timeout=30)

API_URL = "http://localhost:8000/api/search"
EVALUATION_DATASET_PATH = "scripts/evaluation_dataset.csv"

def calculate_retrieval_metrics(predicted_ids: list, expected_ids_str: str, index_for_debug: int) -> dict:
    """
    Calculates Recall, Precision, F1-Score, and MRR for retrieval.
    """
    # --- Enhanced Debugging Block ---
    if index_for_debug < 3:
        print("\n    -------------------------------------")
        print(f"    [DEBUG] Inside calculate_retrieval_metrics for case #{index_for_debug + 1}")
        print(f"    [DEBUG]   - Predicted IDs (raw list): {predicted_ids}")
        print(f"    [DEBUG]   - Expected IDs (raw string): {expected_ids_str}")

    try:
        parsed_expected_ids = json.loads(expected_ids_str.replace("'", "\""))
        expected_ids = set(parsed_expected_ids)
        if index_for_debug < 3:
            print(f"    [DEBUG]   - Parsed Expected IDs (set): {expected_ids}")
    except (json.JSONDecodeError, SyntaxError) as e:
        if index_for_debug < 3:
            print(f"    [DEBUG]   - ERROR during parsing expected_ids: {e}")
        return {"recall": 0.0, "precision": 0.0, "f1_score": 0.0, "mrr": 0.0}

    if not expected_ids:
        is_perfect = 1.0 if not predicted_ids else 0.0
        return {"recall": is_perfect, "precision": is_perfect, "f1_score": is_perfect, "mrr": is_perfect}

    predicted_ids_list = predicted_ids
    predicted_ids_set = set(predicted_ids)
    if index_for_debug < 3:
        print(f"    [DEBUG]   - Predicted IDs (set): {predicted_ids_set}")
    
    intersection = predicted_ids_set.intersection(expected_ids)
    if index_for_debug < 3:
        print(f"    [DEBUG]   - Intersection: {intersection}")

    intersection_len = len(intersection)
    predicted_len = len(predicted_ids_list)
    expected_len = len(expected_ids)

    recall = intersection_len / expected_len if expected_len > 0 else 0.0
    precision = intersection_len / predicted_len if predicted_len > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    mrr = 0.0
    for i, p_id in enumerate(predicted_ids_list):
        if p_id in expected_ids:
            mrr = 1 / (i + 1)
            break
    
    if index_for_debug < 3:
        print(f"    [DEBUG]   - Calculated Metrics: R={recall:.2f}, P={precision:.2f}, F1={f1:.2f}, MRR={mrr:.2f}")
        print("    -------------------------------------\n")

    return {"recall": recall, "precision": precision, "f1_score": f1, "mrr": mrr}

def run_evaluation():
    print("Starting evaluation...")

    dataset_name = "evaluation_dataset_v1_1000"
    try:
        langfuse.get_dataset(name=dataset_name)
        print(f"Using existing Langfuse dataset: '{dataset_name}'")
    except Exception:
        print(f"Dataset '{dataset_name}' not found. Creating it.")
        langfuse.create_dataset(name=dataset_name, description="Evaluation data for stunes recommendation v1 (1000 items).")
        print(f"Dataset '{dataset_name}' created.")

    try:
        eval_df = pd.read_csv(EVALUATION_DATASET_PATH)
        print(f"Loaded {len(eval_df)} evaluation cases.")
    except FileNotFoundError:
        print(f"Error: Evaluation dataset not found at '{EVALUATION_DATASET_PATH}'")
        print("Please ensure you are running the script from the project's root directory ('stunes').")
        return

    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Running Evaluation"):
        conversation_id = row["conversation_id"]
        user_input = row["input"]
        expected_output = row["expected_output"]

        with langfuse.start_as_current_span(
            name="chat-recommendation-evaluation",
            metadata={"turn": row["turn"], "dataset": dataset_name, "conversation_id": str(conversation_id)},
        ) as span:
            span.update_trace(user_id="evaluation_user")
            
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="recommendation-api-call",
                input=user_input,
                metadata={"expected_output": expected_output}
            ) as generation:
                try:
                    response = requests.post(
                        API_URL,
                        json={
                            "query": user_input,
                            "user_id": str(conversation_id),
                            "search_type": "rag",
                            "limit": 10,
                        },
                    )
                    response.raise_for_status()
                    api_result = response.json()

                    generation.update(output=api_result)
                    
                    result_types_to_evaluate = [
                        "contextual_results", 
                        "stateless_results", 
                        "vector_results", 
                        "graph_rag_results"
                    ]
    
                    for result_type in result_types_to_evaluate:
                        predicted_songs = api_result.get(result_type, {}).get("songs", [])
                        predicted_song_ids = [song['song_id'] for song in predicted_songs]
                        
                        metrics = calculate_retrieval_metrics(predicted_song_ids, expected_output, index)
    
                        for metric_name, value in metrics.items():
                            generation.score(
                                name=f"{metric_name}_{result_type}",
                                value=value,
                                comment=f"{metric_name} for {result_type}"
                            )
                        print(f"  - Metrics ({result_type}): Recall={metrics['recall']:.2f}, Precision={metrics['precision']:.2f}, F1={metrics['f1_score']:.2f}, MRR={metrics['mrr']:.2f}")

                except requests.RequestException as e:
                    generation.update(level="ERROR", status_message=str(e))
                except (KeyError, json.JSONDecodeError) as e:
                    generation.update(level="ERROR", status_message=f"Error parsing response: {e}")

        if (index + 1) % 50 == 0:
            print(f"\n--- Flushing batch of 50 (up to item #{index + 1}) to Langfuse ---")
            langfuse.flush()
            print(f"--- Successfully flushed batch up to item #{index + 1}. ---")

    langfuse.flush()
    print("\n--- Final flush to Langfuse successful. ---")
    print("\nEvaluation complete. Check the Langfuse dashboard for results.")

if __name__ == "__main__":
    run_evaluation()
