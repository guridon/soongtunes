"""v1 strategies, ported for regression comparison (from services/rag_service.py and
services/context_service.py at commit 8a9a62e).

Kept as-is: prompts, the LangChain GraphCypherQAChain (LLM writes Cypher), VirtualQuery
preferences with `r.type`, and how song ids are pulled out of the chain's DB context.
Changed, and why:
- LLM: gpt-4o -> gpt-oss-20b on vLLM (temperature 0). Numbers are a new baseline.
- Entity linking: v1 first tried `<label>_name_embeddings` vector indexes and fell back to
  case-insensitive exact match. Those indexes are not in the dump (troubleshooting #3), so
  v1 effectively ran the fallback; the port runs it directly instead of failing first.
- Chat history lives on the VirtualQuery node instead of process memory, so it survives
  multiple API workers. Content is the same ("I provided a response." for AI turns).
- `Neo4jGraph(sanitize=True)`: LLM-written `RETURN s` returns whole nodes, and the dump's
  nodes still carry 1536-d OpenAI vectors; gpt-4o's 128k window absorbed that, a 16k window
  doesn't (prompts hit 16,385 tokens). sanitize strips lists longer than 128 from results.
- Entity JSON: v1 ran `json.loads` on the raw reply. gpt-oss sometimes wraps JSON in a
  ```json fence, so the fence is stripped first; anything else still fails like v1 did.
"""

import asyncio
import logging
from typing import Any

import orjson
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher import CYPHER_GENERATION_PROMPT
from langchain_openai import ChatOpenAI

from stunes.config import Settings
from stunes.conversation.state import StateStore
from stunes.db import Graph
from stunes.retrieval.bm25 import BM25

logger = logging.getLogger(__name__)

_CYPHER_PROMPT = PromptTemplate(
    input_variables=CYPHER_GENERATION_PROMPT.input_variables,
    template=CYPHER_GENERATION_PROMPT.template
    + "\nIMPORTANT: When returning Song nodes, you must include the `song_id` property.",
)

_ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert music librarian. Your task is to extract key entities and user intents from the query.
                    The possible entities are: 'genres', 'artists', 'moods', 'albums', 'playlists'. All should be lists of strings.
                    Also, detect if the user has a 'negative' intent (e.g., 'not rock', 'except ballads').
                    Respond ONLY with a valid JSON object. Keys should be `added_genres`, `negated_genres`, `added_albums`, etc.""",
        ),
        (
            "human",
            'Query: "show me some exciting rock songs from the album The Wall, but not hair metal"',
        ),
        (
            "ai",
            """{{"added_genres": ["rock"], "negated_genres": ["hair metal"], "added_moods": ["exciting"], "added_albums": ["The Wall"]}}""",
        ),
        ("human", 'Query: "{user_query}"'),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# (label, entity key) as in v1; Album/Playlist have `title`, not `name`, so v1's
# `toLower(e.name)` never matched them — kept, it is part of what v1 measured.
_ENTITY_KEYS = [
    ("Genre", "genres"),
    ("Artist", "artists"),
    ("Tag", "moods"),
    ("Album", "albums"),
    ("Playlist", "playlists"),
]

_UPDATE_PREF = """
MATCH (e:{label}) WHERE toLower(e.name) = toLower($name)
MERGE (vq:VirtualQuery {{sessionId: $session_id}})
MERGE (vq)-[r:HAS_PREFERENCE]->(e)
ON CREATE SET r.type = $pref_type, r.timestamp = timestamp()
RETURN count(r) AS count
"""

_SUMMARY = """
MATCH (vq:VirtualQuery {sessionId: $session_id})-[r:HAS_PREFERENCE]->(e)
WHERE r.type IS NOT NULL
RETURN labels(e)[0] AS entity_type, e.name AS name, r.type AS preference
ORDER BY r.timestamp DESC
"""


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def song_ids_from_context(rag_results: dict[str, Any]) -> list[str]:
    """v1 `_process_rag_results`: first key containing 'song_id' in each context row."""
    steps = rag_results.get("intermediate_steps") or []
    if len(steps) < 2:
        return []
    ids: list[str] = []
    for item in steps[1].get("context") or []:
        for key, value in item.items():
            if "song_id" in key.lower() and value:
                ids.append(str(value))
                break
    return list(dict.fromkeys(ids))


class V1Baseline:
    def __init__(self, settings: Settings, graph: Graph, store: StateStore, bm25: BM25):
        self._graph = graph
        self._store = store
        self._bm25 = bm25
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
            reasoning_effort=settings.llm_reasoning_effort or None,
            extra_body=settings.llm_extra_body or None,
            timeout=settings.llm_timeout_s,
        )
        self._lc_graph = Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            # schema introspection (apoc.meta.data over 1.3M nodes) takes ~30s; this also
            # bounds the LLM-written queries
            timeout=60,
            # drop list properties longer than 128 from query results: `RETURN s` would
            # otherwise paste the dump's 1536-d OpenAI vectors into the QA prompt
            sanitize=True,
        )
        self._chain = GraphCypherQAChain.from_llm(
            cypher_prompt=_CYPHER_PROMPT,
            llm=self._llm,
            graph=self._lc_graph,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )
        self._extract = _ENTITY_PROMPT | self._llm | StrOutputParser()
        self._rewrite = _REWRITE_PROMPT | self._llm | StrOutputParser()

    async def turn(self, mode: str, session_id: str, question: str) -> tuple[str, list[str], dict]:
        if "새로 검색" in question or "리셋" in question:
            await self._clear(session_id)
            return "대화 맥락 초기화", [], {}
        trace: dict[str, Any] = {}
        try:
            if mode == "contextual":
                answer, ids = await self._contextual(session_id, question, trace)
            else:
                answer, ids = await self._stateless(session_id, question, trace)
        except Exception as e:  # v1 returned the error text with no songs
            logger.warning("v1 %s failed: %r", mode, e)
            answer, ids = f"오류 발생: {e}", []
            trace["error"] = repr(e)[:300]
        await self._store.save_turn(
            session_id,
            upsert=[],
            remove=[],
            user_message=question,
            assistant_message="I provided a response.",
            results=ids,
        )
        return answer, ids, trace

    async def _contextual(
        self, session_id: str, question: str, trace: dict
    ) -> tuple[str, list[str]]:
        raw = await self._extract.ainvoke({"user_query": question})
        entities = orjson.loads(_strip_fence(raw))
        trace["entities"] = entities
        linked = 0
        for label, key in _ENTITY_KEYS:
            for pref_type, prefix in (("positive", "added"), ("negative", "negated")):
                for name in entities.get(f"{prefix}_{key}", []) or []:
                    rows = await self._graph.write(
                        _UPDATE_PREF.format(label=label),
                        session_id=session_id,
                        name=name,
                        pref_type=pref_type,
                    )
                    linked += bool(rows and rows[0]["count"])
        trace["linked"] = linked
        summary = await self._context_summary(session_id)
        trace["context_summary"] = summary
        final_prompt = (
            f"{summary}\n\nIMPORTANT: The user's detailed preferences for this conversation are stored in the graph, "
            f"connected to a VirtualQuery node with sessionId '{session_id}'. You should use this structured graph information "
            f'to answer the user\'s latest request: "{question}"'
        )
        result = await asyncio.to_thread(self._chain.invoke, {"query": final_prompt})
        trace["cypher"] = (result.get("intermediate_steps") or [{}])[0].get("query")
        return result.get("result", ""), song_ids_from_context(result)

    async def _stateless(
        self, session_id: str, question: str, trace: dict
    ) -> tuple[str, list[str]]:
        state = await self._store.load(session_id)
        history = [
            {"role": "human" if m["role"] == "user" else "ai", "content": m["content"]}
            for m in state.history
        ]
        standalone = await self._rewrite.ainvoke({"chat_history": history, "input": question})
        trace["rewritten"] = standalone
        result = await asyncio.to_thread(self._chain.invoke, {"query": standalone})
        trace["cypher"] = (result.get("intermediate_steps") or [{}])[0].get("query")
        return result.get("result", ""), song_ids_from_context(result)

    async def _context_summary(self, session_id: str) -> str:
        rows = await self._graph.read(_SUMMARY, session_id=session_id)
        if not rows:
            return ""
        pos = [f"{r['entity_type']}: {r['name']}" for r in rows if r["preference"] == "positive"]
        neg = [f"{r['entity_type']}: {r['name']}" for r in rows if r["preference"] != "positive"]
        summary = "Current conversation context: "
        if pos:
            summary += f"User prefers {', '.join(pos)}. "
        if neg:
            summary += f"User dislikes {', '.join(neg)}. "
        return summary.strip()

    async def _clear(self, session_id: str) -> None:
        await self._graph.write(
            "MATCH (vq:VirtualQuery {sessionId: $sid})-[r:HAS_PREFERENCE]->() DELETE r",
            sid=session_id,
        )
