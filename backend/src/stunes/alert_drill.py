"""Alert drill (Phase 7): how long from an injected data problem to a firing alert?

Injection: the active song-embedding version is switched off (as if a bad deploy flipped
the pointer), so `embedding_coverage` drops to 0. The pipeline under test is the real one:
quality exporter (interval) -> Prometheus scrape (15 s) -> rule evaluation (15 s) -> `for: 1m`.
Then the version is re-activated and the time to resolution is measured the same way.
"""

import asyncio
import time
from datetime import UTC, datetime

import asyncpg
import httpx
import orjson

from stunes.config import Settings

ALERT = 'ALERTS{alertname="QualityCheckFailing",check="embedding_coverage"}'


async def _alert_state(client: httpx.AsyncClient) -> str:
    resp = await client.get("http://127.0.0.1:9090/api/v1/query", params={"query": ALERT})
    states = {r["metric"].get("alertstate") for r in resp.json()["data"]["result"]}
    return "firing" if "firing" in states else ("pending" if "pending" in states else "inactive")


async def _wait_for(
    client: httpx.AsyncClient, wanted: str, t0: float, timeout_s: int = 600
) -> dict:
    seen: dict[str, float] = {}
    while time.monotonic() - t0 < timeout_s:
        state = await _alert_state(client)
        seen.setdefault(state, round(time.monotonic() - t0, 1))
        if state == wanted:
            return seen
        await asyncio.sleep(5)
    raise TimeoutError(f"alert never became {wanted}: {seen}")


async def run(settings: Settings) -> dict:
    pg = await asyncpg.connect(settings.postgres_dsn)
    out: dict = {"at": datetime.now(UTC).isoformat()}
    try:
        version = await pg.fetchval(
            "SELECT version FROM embedding_model WHERE entity = 'song' AND status = 'active'"
        )
        if version is None:
            raise RuntimeError("no active song embedding version to switch off")
        async with httpx.AsyncClient(timeout=10) as client:
            if await _alert_state(client) != "inactive":
                out["precondition"] = await _wait_for(client, "inactive", time.monotonic())
            t0 = time.monotonic()
            await pg.execute(
                "UPDATE embedding_model SET status = 'ready' WHERE version = $1", version
            )
            try:
                out["inject"] = {"version_switched_off": version}
                out["to_firing_s"] = await _wait_for(client, "firing", t0)
            finally:
                # always put the pointer back, even if the drill times out
                t1 = time.monotonic()
                await pg.execute(
                    "UPDATE embedding_model SET status = 'active', activated_at = now() "
                    "WHERE version = $1",
                    version,
                )
            out["to_resolved_s"] = await _wait_for(client, "inactive", t1)
    finally:
        await pg.close()
    return out


def main(settings: Settings) -> None:
    result = asyncio.run(run(settings))
    path = settings.results_dir / "alerts"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"drill-{int(time.time())}.json").write_bytes(
        orjson.dumps(result, option=orjson.OPT_INDENT_2)
    )
    print(result)
