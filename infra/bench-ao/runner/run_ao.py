"""Replay context-set conversations against auto-ontology's public chat API.

One auto-ontology conversation per context-set conversation (a fresh conversation_id per run),
turns sent in order, each after the previous one finished ([DONE]). Every SSE event is kept,
without the result rows: Auto-Ontology streams every row its SQL returned (sql_response_from_db,
and the charts event's sql_response), hundreds of MB for a broad query. The scorer runs the SQL
itself, so a record keeps the rows' size and their first 2,000 characters.

usage: python run_ao.py SET.jsonl OUT_DIR [--conversations N] [--concurrency C] [--resume RUN_DIR]
  --resume: keep the conversations a stopped run finished; a conversation it left half done, or
            with a turn that failed on the server side (HTTP 5xx, no response), is run again from
            its first turn, in a new Auto-Ontology conversation
A server-side failure stops the run (other conversations finish their current turn): a broken
stack answers every later turn with an error in no time, which would read as the agent's own.
env: AO_URL (default http://auto-ontology-frontend:3000), AO_TOKEN (API token, x-api-key)
"""
import argparse, json, os, threading, time, uuid, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = os.environ.get("AO_URL", "http://auto-ontology-frontend:3000") + "/api/chat/completions"
TOKEN = os.environ["AO_TOKEN"]
TURN_TIMEOUT_S = 900
HEAD_CHARS = 2000
lock = threading.Lock()
stop = threading.Event()


def server_failed(rec: dict) -> bool:
    return rec.get("status") is None or rec["status"] >= 500


def slim_answer(a):
    if not isinstance(a, dict) or "sql_response_from_db" not in a:
        return a
    out = dict(a)
    v = out.pop("sql_response_from_db")
    text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    out["sql_response_chars"] = len(text)
    out["sql_response_head"] = text[:HEAD_CHARS]
    return out


def ask(question: str, conversation_id: str) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(
        {"question": question, "conversation_id": conversation_id}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": TOKEN})
    t0 = time.time()
    events, status, err = [], None, None
    try:
        with urllib.request.urlopen(req, timeout=TURN_TIMEOUT_S) as r:
            status = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if payload.startswith('{"type": "charts"'):
                    events.append({"type": "charts", "chars": len(payload)})
                    continue
                try:
                    e = json.loads(payload)
                except json.JSONDecodeError:
                    events.append({"type": "unparsed", "chars": len(payload), "raw": payload[:HEAD_CHARS]})
                    continue
                if e.get("type") == "result":
                    e = {"type": "result", "answer": slim_answer(e.get("answer"))}
                events.append(e)
                del payload, line, raw
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode("utf-8", "replace")[:2000]
    except Exception as e:  # noqa: BLE001 (recorded as a request error)
        err = f"{type(e).__name__}: {e}"
    result = next((e for e in events if e.get("type") == "result"), None)
    return {"status": status, "error": err, "seconds": round(time.time() - t0, 2),
            "answer": (result or {}).get("answer"),  # already without its rows
            "errors": [e for e in events if e.get("type") == "error"],
            "sql_events": [e.get("sql") for e in events if e.get("type") == "sql"],
            "steps": [e.get("node") for e in events
                      if e.get("type") == "step" and e.get("phase") == "end"],
            "events": [e for e in events if e.get("type") != "result"]}


def conversation(turns: list[dict], run_id: str, out, attempt: int = 0) -> None:
    conv_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL,
                               f"stunes-ao/{run_id}/{turns[0]['conversation_id']}/{attempt}"))
    for t in turns:
        if stop.is_set():
            return
        rec = {"conversation_id": t["conversation_id"], "turn": t["turn"], "op": t["op"],
               "kind": t.get("kind"), "text": t["text"], "truth": t["state"],
               "ao_conversation_id": conv_uuid, **ask(t["text"], conv_uuid)}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            a = rec["answer"] or {}
            print(f"{t['conversation_id']} t{t['turn']} {rec['seconds']:6.1f}s "
                  f"status={rec['status']} sql={'y' if (a.get('sql_code') or rec['sql_events']) else 'n'}"
                  f"{' ERR' if rec['error'] or rec['errors'] else ''}", flush=True)
        if server_failed(rec):
            print(f"STOP: server-side failure at {t['conversation_id']} t{t['turn']}: "
                  f"{str(rec['error'])[:200]}", flush=True)
            stop.set()
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_path"); ap.add_argument("out_dir")
    ap.add_argument("--conversations", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    convs: dict[str, list[dict]] = {}
    for line in open(a.set_path):
        t = json.loads(line)
        convs.setdefault(t["conversation_id"], []).append(t)
    ids = sorted(convs)[: a.conversations] if a.conversations else sorted(convs)
    done: list[dict] = []
    if a.resume:
        out_dir = Path(a.resume)
        run_id = out_dir.name
        prev = [json.loads(l) for l in open(out_dir / "records.jsonl")]
        n_turns = {cid: len(convs[cid]) for cid in ids}
        got: dict[str, int] = {}
        for r in prev:
            got[r["conversation_id"]] = got.get(r["conversation_id"], 0) + 1
        failed = {r["conversation_id"] for r in prev if server_failed(r)}
        complete = {cid for cid, n in got.items() if n == n_turns.get(cid) and cid not in failed}
        done = [r for r in prev if r["conversation_id"] in complete]
        ids = [cid for cid in ids if cid not in complete]
        print(f"resume {run_id}: {len(complete)} conversations kept, {len(ids)} to run", flush=True)
        with open(out_dir / "records.jsonl", "w") as f:
            for r in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_dir / "resumed.txt", "a") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f" kept {len(complete)}\n")
        attempt = 1 + sum(1 for _ in open(out_dir / "resumed.txt"))
        mode = "a"
    else:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_dir = Path(a.out_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        attempt = 0
        mode = "w"
    if not a.resume:
        (out_dir / "run.json").write_text(json.dumps({
            "run_id": run_id, "set": a.set_path, "conversations": len(ids),
            "concurrency": a.concurrency, "note": a.note,
            "auto_ontology_commit": "eb8dd0621d07db3033fe3fddc68f62ff72614610",
            "llm": "gpt-oss-20b via llm-proxy, temperature 0, reasoning effort per model role "
                   "(the model names in the server .env)",
            "embed": "harrier-oss-v1-0.6b, zero-padded to 2048"}, indent=1))
    with open(out_dir / "records.jsonl", mode) as out, ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(lambda cid: conversation(sorted(convs[cid], key=lambda t: t["turn"]), run_id,
                                             out, attempt), ids))
    print("stopped" if stop.is_set() else "done", out_dir)


if __name__ == "__main__":
    main()
