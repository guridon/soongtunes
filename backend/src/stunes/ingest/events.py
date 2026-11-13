"""User behaviour events.

    {"event_id": "uuid", "user_id": "p_12345", "type": "play|skip|like|search",
     "song_id": "381234", "query": null, "ts": 1793346000123, "schema_version": 1}

Event ids are UUIDv5(run, seq): the experiment can rebuild the full set of published ids
from (run_id, count) alone, so "lost = published - processed" needs no producer-side log.
"""

import uuid
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TOPIC = "user-events"
RETRY_TOPIC = "user-events.retry"
DLQ_TOPIC = "user-events.dlq"

_NAMESPACE = uuid.UUID("6f1d2a9e-5b8c-4c1e-9a77-1f0e2d3c4b5a")


def event_id(run_id: str, seq: int) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, f"{run_id}:{seq}")


def is_poison(seq: int, every: int) -> bool:
    """Which sequence numbers the generator deliberately breaks (1% when every=100)."""
    return every > 0 and seq % every == every - 1


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: uuid.UUID
    user_id: str = Field(pattern=r"^p_\d+$")
    type: Literal["play", "skip", "like", "search"]
    song_id: str | None = None
    query: str | None = None
    ts: int = Field(gt=0)  # epoch milliseconds
    schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.type == "search":
            if not self.query:
                raise ValueError("search events need a query")
        elif not self.song_id:
            raise ValueError(f"{self.type} events need a song_id")
        return self
