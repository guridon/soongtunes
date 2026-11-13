import asyncio

import httpx

from stunes.eval.run import Turn, _run_interleaved


def test_interleaved_answers_every_turn_with_every_strategy():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            calls.append(("delete", request.url.params["strategy"]))
            return httpx.Response(200, json={})
        body = __import__("json").loads(request.content)
        calls.append(("chat", body["strategy"], body["message"]))
        return httpx.Response(200, json={"songs": [{"song_id": "1"}], "trace": {}})

    conv = [Turn("7", i, f"turn {i}", "mood_search", {"1"}) for i in (1, 2, 3)]

    async def go():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://t"
        ) as c:
            return await _run_interleaved(c, ["agent", "hybrid"], "r", conv, 10)

    out = asyncio.run(go())
    chats = [c for c in calls if c[0] == "chat"]
    # each turn is answered by both strategies before the next turn starts
    assert [m for _, _, m in chats] == ["turn 1", "turn 1", "turn 2", "turn 2", "turn 3", "turn 3"]
    assert all(
        {s for _, s, m in chats if m == f"turn {i}"} == {"agent", "hybrid"} for i in (1, 2, 3)
    )
    assert sorted(c[1] for c in calls if c[0] == "delete") == ["agent", "hybrid"]
    assert len(out) == 6 and all(r["metrics"]["hit"] == 1.0 for r in out)
