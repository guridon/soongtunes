"""Clients for the self-hosted models (vLLM, OpenAI-compatible API).

- Chat: gpt-oss-20b. JSON outputs go through vLLM structured outputs (json_schema),
  so the extractor never has to repair malformed JSON.
- Embeddings: harrier-oss-v1-0.6b, L2-normalized by the model's pooling config.
  Vectors travel as base64 float32 — 4 bytes/dim instead of ~10 bytes/dim as JSON text.
"""

import base64
import logging
from typing import Any

import numpy as np
import orjson
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stunes.config import Settings

logger = logging.getLogger(__name__)

_RETRYABLE = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    reraise=True,
)


class ChatModel:
    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
        )
        self.model = settings.llm_model
        self._extra: dict = {
            **(
                {"reasoning_effort": settings.llm_reasoning_effort}
                if settings.llm_reasoning_effort
                else {}
            ),
            **({"extra_body": settings.llm_extra_body} if settings.llm_extra_body else {}),
        }

    @_RETRYABLE
    async def complete(self, messages: list[dict[str, str]], max_tokens: int = 512) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            **self._extra,
        )
        return (resp.choices[0].message.content or "").strip()

    @_RETRYABLE
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        name: str,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            **self._extra,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
        )
        return orjson.loads(resp.choices[0].message.content or "{}")

    async def ping(self) -> bool:
        await self._client.models.list()
        return True


class Embedder:
    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(
            base_url=settings.embed_base_url, api_key=settings.embed_api_key, timeout=60
        )
        self.model = settings.embed_model
        self._query_instruction = settings.embed_query_instruction

    @_RETRYABLE
    async def _embed(self, texts: list[str]) -> np.ndarray:
        resp = await self._client.embeddings.create(
            model=self.model, input=texts, encoding_format="base64"
        )
        rows = sorted(resp.data, key=lambda d: d.index)
        return np.stack(
            [np.frombuffer(base64.b64decode(d.embedding), dtype=np.float32) for d in rows]
        )

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return await self._embed(texts)

    async def embed_queries(self, texts: list[str]) -> np.ndarray:
        return await self._embed([self._query_instruction + t for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self.embed_queries([text]))[0]

    async def ping(self) -> bool:
        await self._client.models.list()
        return True
