from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO = Path(__file__).resolve().parents[3]

# the chat strategy the API and the web app serve by default, and the one whose stage, candidate
# and diversity metrics are recorded (docs/prereg-context-ctx.md, then prereg-context-long.md and
# prereg-graph-plan-link-retype.md part 3 decided the switches)
SERVED_STRATEGY = "hybrid_delta_keys"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_name: str = "SoongTunes"
    environment: str = "local"
    cors_origins: list[str] = ["http://localhost:3000"]

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(...)
    neo4j_database: str = "neo4j"

    postgres_dsn: str = "postgresql://stunes:stunes@localhost:5432/stunes"
    opensearch_url: str = "http://localhost:9200"
    songs_index: str = "songs"
    entities_index: str = "entities"

    # OpenAI-compatible endpoints served by vLLM on the GPU host (see infra/llm)
    llm_base_url: str = "http://localhost:8200/v1"
    llm_api_key: str = Field(...)
    llm_model: str = "gpt-oss-20b"
    # gpt-oss understands reasoning_effort; other models ignore it (leave empty to omit)
    llm_reasoning_effort: str | None = "low"
    # merged into every chat request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    # to run Qwen3 without its thinking phase
    llm_extra_body: dict = {}
    llm_timeout_s: float = 60.0

    embed_base_url: str = "http://localhost:8201/v1"
    embed_api_key: str = Field(...)
    embed_model: str = "harrier-oss-v1-0.6b"
    # Harrier is instruction-tuned: queries get a one-line task prefix, documents don't.
    embed_query_instruction: str = (
        "Instruct: Given a Korean music request, retrieve songs that match it\nQuery: "
    )

    # Phase 2/3: event ingestion and the feature cache
    kafka_bootstrap: str = "127.0.0.1:19092"
    redis_url: str = "redis://localhost:6379/0"
    # who keeps Redis in step with the feature tables: "consumer" = the ingest consumer's
    # write-through after its commit; "cdc" = the consumer writes Postgres only and
    # stunes.ingest.cdc_cache applies the committed changes from the WAL (compose profile cdc)
    cache_writer: str = "consumer"

    default_limit: int = 10
    candidate_pool: int = 100

    # experiment switches; /api/health reports them and every eval summary records them
    # multiplies every candidate-source budget (client wait and server-side timeout alike):
    # 1.0 online; an offline eval raises it to measure the sources without dropping any
    source_budget_scale: float = 1.0
    # Phase 4 LLM-predicted tags (HAS_LLM_TAG edges, llm_tags field) count as tag evidence
    llm_tags_in_retrieval: bool = False
    # the re-ranker in chat (docs/prereg-rerank-chat.md). rerank_variants: compute the other
    # orders of each turn's candidates and return them in the trace (the served answer stays
    # RRF); rerank_serve: answer with the re-ranked order. Both off by default.
    rerank_variants: bool = False
    rerank_serve: bool = False
    rerank_budget_ms: float = 1000.0  # 300 in the pre-registration; see its changes
    rerank_sample_every: int = 10  # 1 in N turns returns its feature rows (model parity)
    # co-occurrence scoring: "count" = the Cypher count of each seed's most-liked playlists;
    # "degree" = 1/degree weights over an in-memory playlist graph, built by `stunes
    # ppr-build` (docs/results.md, "홈 피드: Personalized PageRank와 동시등장").
    # "auto" = degree from two seeds up. The feed's was measured (degree); the chat's is not.
    cooccur_weights_feed: str = "count"
    cooccur_weights_chat: str = "count"
    ppr_graph_path: Path | None = None  # default: data_dir / "ppr" / "graph.npz"

    @property
    def ppr_graph_file(self) -> Path:
        return self.ppr_graph_path or self.data_dir / "ppr" / "graph.npz"

    # repo-root defaults for host runs; compose mounts them at the same env-configured paths
    data_dir: Path = _REPO / ".data"
    results_dir: Path = _REPO / "results"
    rerank_model_path: Path = _REPO / "results" / "rerank-model-online.txt"


@lru_cache
def get_settings() -> Settings:
    return Settings()
