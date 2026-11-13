"""OpenAI-compatible shim between auto-ontology and the SoongTunes vLLM servers (stdlib only).

- chat: the model "openai/gpt-oss-20b" maps to the served "gpt-oss-20b", so auto-ontology takes
  its ChatOpenAI path (structured output by json_schema). temperature 0 and reasoning_effort
  "low" are added when the request sets neither: SoongTunes' own settings. A model name ending
  in "@medium" or "@high" sets that reasoning effort instead, per model role. The vLLM key is
  added here, so it never enters auto-ontology's config.
- embeddings: Harrier gets its query instruction when input_type is "query" (NIM's convention),
  and vectors are zero-padded to 2048 dimensions, auto-ontology's fixed column size. Padding
  keeps dot products and norms, so cosine similarity is unchanged.
- ranking: NIM's /v1/ranking, answered by Harrier cosine similarity (no reranker is served).
"""
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LLM = os.environ["LLM_BASE_URL"].rstrip("/")
LLM_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-20b")
EMB = os.environ["EMBED_BASE_URL"].rstrip("/")
EMB_KEY = os.environ["EMBED_API_KEY"]
EMB_MODEL = os.environ.get("EMBED_MODEL", "harrier-oss-v1-0.6b")
DIM = int(os.environ.get("PAD_DIM", "2048"))
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a music database, retrieve the tables, columns and "
    "business terms that answer it\nQuery: "
)
TIMEOUT = 600


def _post(url: str, key: str, body: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _embed(texts: list[str]) -> list[list[float]]:
    with _post(f"{EMB}/embeddings", EMB_KEY,
               {"model": EMB_MODEL, "input": texts, "encoding_format": "float"}) as r:
        data = json.load(r)["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def _pad(v: list[float]) -> list[float]:
    if len(v) > DIM:
        raise ValueError(f"embedding has {len(v)} dims, more than {DIM}")
    return v + [0.0] * (DIM - len(v))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code: int, obj) -> None:
        raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")
        if p.endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "openai/gpt-oss-20b", "object": "model"},
                {"id": EMB_MODEL, "object": "model"}]})
        elif p == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        p = self.path.split("?")[0].rstrip("/")
        try:
            if p.endswith("/chat/completions"):
                return self._chat(body)
            if p.endswith("/embeddings"):
                return self._embeddings(body)
            if p.endswith(("/ranking", "/reranking", "/rerank")):
                return self._ranking(body)
            self._send(404, {"error": f"no route {p}"})
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
        except Exception as e:  # noqa: BLE001 (report upstream failures to the caller)
            self._send(502, {"error": f"{type(e).__name__}: {e}"})

    def _chat(self, body: dict) -> None:
        # "openai/gpt-oss-20b@medium" picks the reasoning effort per model role
        _, _, effort = str(body.get("model", "")).partition("@")
        body["model"] = LLM_MODEL
        body.setdefault("temperature", 0)
        body.setdefault("reasoning_effort", effort or "low")
        with _post(f"{LLM}/chat/completions", LLM_KEY, body) as r:
            if not body.get("stream"):
                return self._send(r.status, r.read())
            self.send_response(r.status)
            self.send_header("Content-Type", r.headers.get("Content-Type", "text/event-stream"))
            self.end_headers()
            while chunk := r.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()

    def _embeddings(self, body: dict) -> None:
        inp = body.get("input")
        texts = [inp] if isinstance(inp, str) else list(inp)
        if body.get("input_type") == "query":
            texts = [QUERY_INSTRUCTION + t for t in texts]
        vecs = [_pad(v) for v in _embed(texts)]
        b64 = body.get("encoding_format") == "base64"
        data = [{
            "object": "embedding", "index": i,
            "embedding": base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode() if b64 else v,
        } for i, v in enumerate(vecs)]
        self._send(200, {"object": "list", "data": data, "model": body.get("model", EMB_MODEL),
                         "usage": {"prompt_tokens": 0, "total_tokens": 0}})

    def _ranking(self, body: dict) -> None:
        q = body.get("query")
        query = q.get("text", "") if isinstance(q, dict) else str(q or "")
        docs = [d.get("text", "") if isinstance(d, dict) else str(d)
                for d in (body.get("passages") or body.get("documents") or [])]
        vecs = _embed([QUERY_INSTRUCTION + query] + docs)
        qv = vecs[0]
        scores = [sum(a * b for a, b in zip(qv, dv)) for dv in vecs[1:]]
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
        if self.path.rstrip("/").endswith("/rerank"):
            return self._send(200, {"results": [
                {"index": i, "relevance_score": scores[i]} for i in order]})
        self._send(200, {"rankings": [{"index": i, "logit": scores[i]} for i in order]})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    server.daemon_threads = True
    print("llm-proxy on :8080", flush=True)
    server.serve_forever()
