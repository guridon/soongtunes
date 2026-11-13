"""OpenAI-compatible /v1/embeddings for local development.

The production path serves the same model with vLLM on the GPU host (infra/llm). This
exists because the GPU host's internet link was too slow to download the model; it runs
the identical checkpoint, pooling (last token) and L2 normalization through
sentence-transformers, so vectors are interchangeable up to fp16/bf16 rounding.

    EMBED_API_KEY=... uv run uvicorn server:app --host 127.0.0.1 --port 8201
"""

import asyncio
import base64
import os
import time

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL_ID", "microsoft/harrier-oss-v1-0.6b")
SERVED_NAME = os.environ.get("EMBED_MODEL", "harrier-oss-v1-0.6b")
API_KEY = os.environ.get("EMBED_API_KEY")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model = SentenceTransformer(
    MODEL_ID,
    device=DEVICE,
    model_kwargs={"dtype": torch.float16 if DEVICE == "mps" else torch.float32},
)
model.max_seq_length = int(os.environ.get("EMBED_MAX_TOKENS", "512"))
gpu = asyncio.Lock()  # one accelerator: run batches one at a time
app = FastAPI(title="stunes local embedder")


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"


def _auth(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    _auth(authorization)
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if req.model != SERVED_NAME:
        raise HTTPException(404, f"model {req.model} not served (have {SERVED_NAME})")
    texts = [req.input] if isinstance(req.input, str) else req.input
    started = time.perf_counter()
    async with gpu:
        vecs = await asyncio.to_thread(
            model.encode, texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        )
    vecs = np.asarray(vecs, dtype=np.float32)
    if req.encoding_format == "base64":
        data = [base64.b64encode(v.tobytes()).decode() for v in vecs]
    else:
        data = [v.tolist() for v in vecs]
    return {
        "object": "list",
        "model": SERVED_NAME,
        "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(data)],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
