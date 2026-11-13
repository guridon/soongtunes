"""HTTP API (FastAPI). Run: `uvicorn stunes.api:app`."""

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Literal

import orjson
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from stunes import metrics
from stunes.config import SERVED_STRATEGY, get_settings
from stunes.services import STRATEGIES, Services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("stunes.api")

Strategy = Literal[
    "agent",
    "hybrid",
    "hybrid_s2t",
    "hybrid_llmstate",
    "hybrid_cumstate",
    "hybrid_ctx",
    "hybrid_ctx_auto",
    "hybrid_ctx_keys",
    "hybrid_ctxsplit",
    "hybrid_ctx_recent",
    "hybrid_delta",
    "hybrid_delta_recent",
    "hybrid_delta_keys",
    "v1_contextual",
    "v1_stateless",
    "bm25",
    "vector",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = await Services.build(get_settings())
    logger.info("services ready")
    try:
        yield
    finally:
        await app.state.services.close()


app = FastAPI(
    title="SoongTunes API",
    version="2.0.0",
    description="Knowledge-graph music search with hybrid (BM25 + dense) retrieval",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def svc(request: Request) -> Services:
    return request.app.state.services


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w\-:.]+$")
    message: str = Field(min_length=1, max_length=1000)
    strategy: Strategy = SERVED_STRATEGY
    limit: int | None = Field(default=None, ge=1, le=50)
    explain: bool = True
    # adds every source's candidate ids and the pre-verify order to the trace (eval loss map)
    debug: bool = False


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"ok": True}


@app.get("/api/health")
async def health(request: Request):
    return await svc(request).health()


@app.get("/api/strategies")
async def strategies():
    return [{"name": k, "description": v} for k, v in STRATEGIES.items()]


@app.get("/api/stats")
async def stats(request: Request):
    s = svc(request)
    return {"graph": s.stats, "genres": s.genres}


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    try:
        result = await svc(request).chat(
            body.strategy, body.session_id, body.message, body.limit, body.explain, body.debug
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception:
        metrics.ERRORS.labels("chat").inc()
        raise
    metrics.observe_turn(
        body.strategy,
        (result.trace.get("timings_ms") or {}).get("total"),
        result.trace,
        result.songs,
    )
    return asdict(result)


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    body, content_type = metrics.render()
    return Response(body, media_type=content_type)


@app.get("/api/feed/{user_id}")
async def feed(user_id: str, request: Request, limit: Annotated[int, Query(ge=1, le=50)] = 20):
    """Home feed (Phase 5): no utterance — recent plays and tag preferences from the feature store."""
    result = await svc(request).agent.feed(user_id, limit)
    metrics.observe_turn(
        "feed", (result.trace.get("timings_ms") or {}).get("total"), result.trace, result.songs
    )
    return asdict(result)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request, strategy: Strategy = SERVED_STRATEGY):
    return await svc(request).session(strategy, session_id)


@app.delete("/api/sessions/{session_id}")
async def reset_session(session_id: str, request: Request, strategy: Strategy | None = None):
    for name in [strategy] if strategy else list(STRATEGIES):
        await svc(request).reset(name, session_id)
    return {"ok": True}


@app.get("/api/search")
async def search(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
):
    return await svc(request).search(q, mode, limit)


@app.get("/api/songs")
async def songs(request: Request, ids: Annotated[str, Query(max_length=2000)]):
    """Song cards for a comma-separated id list, in the given order."""
    wanted = [i for i in ids.split(",") if i][:100]
    cards = await svc(request).bm25.get_songs(wanted)
    return [cards[i] for i in wanted if i in cards]


@app.get("/api/users/sample")
async def sample_users(request: Request, n: Annotated[int, Query(ge=1, le=50)] = 8):
    """Users that have features (Phase 3 seed / Phase 2 consumer), for the feed page."""
    rows = await svc(request).stores.pg.fetch(
        "SELECT substr(pk, length('user_recent_plays#') + 1) AS id FROM feature "
        "WHERE pk LIKE 'user_recent_plays#%' ORDER BY random() LIMIT $1",
        n,
    )
    return [r["id"] for r in rows]


@app.get("/api/songs/{song_id}")
async def song(song_id: str, request: Request):
    found = await svc(request).song(song_id)
    if found is None:
        raise HTTPException(404, "song not found")
    return found


@app.get("/api/songs/{song_id}/graph")
async def song_graph(song_id: str, request: Request):
    found = await svc(request).neighborhood(song_id)
    if found is None:
        raise HTTPException(404, "song not found")
    return found


class FeatureRequest(BaseModel):
    entity_ids: list[str] = Field(min_length=1, max_length=500)
    groups: list[str] = Field(min_length=1, max_length=10)
    profile: Literal["C0", "C1", "C2", "C3", "C4", "C5"] = "C5"


@app.post("/api/features")
async def features(body: FeatureRequest, request: Request):
    """Feature serving (Phase 3): L1 memory -> L2 Redis -> L3 Postgres, per cache profile."""
    store = svc(request).features
    unknown = [g for g in body.groups if g not in store.groups]
    if unknown:
        raise HTTPException(400, f"unknown feature groups: {unknown}")
    return await store.get(body.entity_ids, body.groups, body.profile)


@app.get("/api/features/stats")
async def feature_stats():
    """Monotonic feature-store tallies summed over all API workers; readers take deltas."""
    return metrics.counter_values(metrics.FEATURE_OPS, "op")


@app.get("/api/results")
async def experiment_results():
    """Latest numbers from each experiment's results/<kind>/*.json (for the system page)."""
    root = get_settings().results_dir
    out: dict[str, list] = {}
    for kind in ("ingest", "features", "embedding", "llm_tags", "feed"):
        files = sorted((root / kind).glob("*.json"), key=lambda p: p.stat().st_mtime)
        rows = []
        for path in files[-40:]:
            data = orjson.loads(path.read_bytes())
            data.pop("series", None)  # per-second load-test series are large; not needed here
            rows.append(data)
        out[kind] = rows
    return out


@app.get("/api/eval/latest")
async def eval_latest():
    """Summary of the most recent `make eval` run (results/eval/<run>/summary.json)."""
    results = get_settings().results_dir / "eval"
    runs = sorted(results.glob("*/summary.json")) if results.exists() else []
    if not runs:
        return {"runs": []}
    return {"runs": [orjson.loads(p.read_bytes()) for p in runs[-5:]]}
