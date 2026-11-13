"""Conversation state stored in the graph (kept from v1: the VirtualQuery node).

(:VirtualQuery {sessionId})-[:HAS_PREFERENCE {is_positive, kind, key, turn}]->(entity)

Everything a turn needs lives here, not in process memory, so any API worker can serve
any turn of a conversation.
"""

from dataclasses import dataclass, field

from stunes.db import Graph

HISTORY_MAX = 12  # messages (user + assistant)
SHOWN_MAX = 200  # song ids already recommended in this session


@dataclass(frozen=True)
class Preference:
    kind: str  # artist | genre | tag | song
    key: str
    name: str
    positive: bool
    turn: int = 0


@dataclass
class SessionState:
    session_id: str
    turn: int = 0
    prefs: list[Preference] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    last_results: list[str] = field(default_factory=list)
    shown: list[str] = field(default_factory=list)
    # the year range the last turn searched with; only hybrid_ctx reads it back (the other
    # strategies take the year from this turn's slots alone)
    year_from: int | None = None
    year_to: int | None = None

    def keys(self, kind: str, positive: bool) -> list[str]:
        return [p.key for p in self.prefs if p.kind == kind and p.positive is positive]


_LOAD = """
MATCH (vq:VirtualQuery {sessionId: $sid})
OPTIONAL MATCH (vq)-[r:HAS_PREFERENCE]->(e)
WHERE r.kind IS NOT NULL
RETURN vq.turn AS turn, vq.history AS history, vq.last_results AS last_results,
       vq.shown AS shown, vq.year_from AS year_from, vq.year_to AS year_to,
       collect(CASE WHEN r IS NULL THEN null ELSE
         {kind: r.kind, key: r.key, name: coalesce(e.name, e.title), positive: r.is_positive,
          turn: r.turn} END) AS prefs
"""

# One write per turn: bump the turn, append history, replace preferences that this turn
# overrides, then upsert the new ones. Genre names map to several nodes (domestic/foreign
# codes), hence MATCH ... MERGE per node.
_SAVE = """
MERGE (vq:VirtualQuery {sessionId: $sid})
ON CREATE SET vq.created_at = datetime(), vq.turn = 0
SET vq.turn = vq.turn + 1,
    vq.updated_at = datetime(),
    vq.history = (coalesce(vq.history, []) + $history)[-$history_max..],
    vq.last_results = $last_results,
    vq.shown = (coalesce(vq.shown, []) + $last_results)[-$shown_max..],
    vq.year_from = $year_from,
    vq.year_to = $year_to
WITH vq
CALL (vq) {
  MATCH (vq)-[r:HAS_PREFERENCE]->()
  WHERE any(x IN $remove WHERE x.kind = r.kind AND x.key = r.key)
  DELETE r
}
WITH vq
UNWIND $prefs AS pref
CALL (vq, pref) {
  OPTIONAL MATCH (a:Artist {artist_id: pref.key}) WHERE pref.kind = 'artist'
  OPTIONAL MATCH (g:Genre {name: pref.key}) WHERE pref.kind = 'genre'
  OPTIONAL MATCH (t:Tag {name: pref.key}) WHERE pref.kind = 'tag'
  OPTIONAL MATCH (s:Song {song_id: pref.key}) WHERE pref.kind = 'song'
  WITH [x IN [a, g, t, s] WHERE x IS NOT NULL] AS targets
  UNWIND targets AS e
  MERGE (vq)-[r:HAS_PREFERENCE {kind: pref.kind, key: pref.key}]->(e)
  SET r.is_positive = pref.positive, r.turn = vq.turn, r.updated_at = datetime()
}
RETURN vq.turn AS turn
"""

_CLEAR = """
MATCH (vq:VirtualQuery {sessionId: $sid})
DETACH DELETE vq
"""


class StateStore:
    def __init__(self, graph: Graph):
        self._g = graph

    async def load(self, session_id: str) -> SessionState:
        rows = await self._g.read(_LOAD, sid=session_id)
        if not rows:
            return SessionState(session_id)
        row = rows[0]
        seen: set[tuple[str, str]] = set()
        prefs = []
        for p in row["prefs"]:
            if p is None or (p["kind"], p["key"]) in seen:
                continue  # a genre name links to several nodes -> one preference
            seen.add((p["kind"], p["key"]))
            prefs.append(
                Preference(
                    p["kind"], p["key"], p["name"] or p["key"], p["positive"], p["turn"] or 0
                )
            )
        history = []
        for line in row["history"] or []:
            role, _, content = line.partition("\t")
            history.append({"role": role, "content": content})
        return SessionState(
            session_id=session_id,
            turn=row["turn"] or 0,
            prefs=prefs,
            history=history,
            last_results=list(row["last_results"] or []),
            shown=list(row["shown"] or []),
            year_from=row.get("year_from"),
            year_to=row.get("year_to"),
        )

    async def save_turn(
        self,
        session_id: str,
        *,
        upsert: list[Preference],
        remove: list[tuple[str, str]],
        user_message: str,
        assistant_message: str,
        results: list[str],
        year: tuple[int | None, int | None] = (None, None),
    ) -> None:
        await self._g.write(
            _SAVE,
            sid=session_id,
            history=[f"user\t{user_message}", f"assistant\t{assistant_message}"],
            history_max=HISTORY_MAX,
            last_results=results,
            shown_max=SHOWN_MAX,
            year_from=year[0],
            year_to=year[1],
            remove=[{"kind": k, "key": v} for k, v in remove],
            prefs=[{"kind": p.kind, "key": p.key, "positive": p.positive} for p in upsert],
        )

    async def clear(self, session_id: str) -> None:
        await self._g.write(_CLEAR, sid=session_id)
