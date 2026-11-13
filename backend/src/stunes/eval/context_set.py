"""A multi-turn eval set with computed answers, for context handling
(docs/prereg-context-eval.md).

Each conversation is a program: a start state, then one operation per turn. The truth state of
every turn is known, and a song is relevant when the graph says it meets that state.

- The utterances come from fixed templates, not from an LLM, so the extractor never reads its
  own phrasing.
- Every value in them links to its intended key on its own (checked here, before generation).
  A wrong answer is then a context error, not a linking or paraphrase error.

Operations (search-session reformulation types, plus conversational ones):
- start: one or two constraints;
- add: a constraint of a kind not set (specialization);
- relax: a positive constraint dropped (generalization);
- replace: another value of a kind already set (parallel move);
- exclude: a negative constraint, chosen among values the current answers actually have;
- reference: a value taken from the system's own previous list (the k-th song's artist or
  genre). Its truth is resolved per arm at analysis time, so it is only ever a last turn;
- more: the same state, new songs;
- new_topic: everything before is dropped, and a new start state begins.

usage (repo root): backend/.venv/bin/python -m stunes.eval.context_set OUT_DIR
  --split NAME:N  one more draw of N conversations, seeded by NAME, written as NAME.jsonl. The
                  pools must equal OUT_DIR/pools.json; dev and test are left as they are.
  --turns T       turns per conversation of that draw (default 5). Past the history window
                  (12 messages, 6 turns) the earliest turns are no longer in the LLM's context.
"""

import argparse
import asyncio
import dataclasses
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import orjson

from stunes.config import Settings
from stunes.db import Graph, create_search
from stunes.retrieval.bm25 import BM25
from stunes.retrieval.linker import EntityLinker

logger = logging.getLogger(__name__)

TURNS = 5
MIN_ANSWERS = 20  # every truth state has at least this many songs, so P@10 can reach 1
SPLITS = {"dev": (100, "dev"), "test": (400, "test")}  # conversations, seed

# (surface, Genre.name); each surface is checked against the served linker
GENRES = [
    ("발라드", "발라드"), ("댄스", "댄스"), ("힙합", "랩/힙합"), ("알앤비", "R&B/Soul"),
    ("록", "록/메탈"), ("인디", "인디음악"), ("OST", "OST"), ("재즈", "재즈"),
    ("트로트", "성인가요"), ("EDM", "EDM"), ("포크", "포크/블루스"), ("팝송", "POP"),
    ("일렉트로니카", "일렉트로니카"), ("아이돌", "아이돌"),
]  # fmt: skip
# mood and situation tags with an exact Tag of that name; genre-like tags are left out
TAGS = [
    "잔잔한", "신나는", "감성", "새벽", "드라이브", "카페", "비오는날", "산책", "운동", "힐링",
    "이별", "설렘", "우울", "편안한", "퇴근길", "출근길", "공부", "여름밤", "가을", "겨울", "봄",
    "여름", "기분좋은", "달달한", "몽환", "따뜻한", "조용한", "시원한", "사랑", "추억", "휴식",
    "위로", "행복", "파티", "데이트", "여행", "그리움", "새벽감성",
]  # fmt: skip
DECADES = {1990: "90년대", 2000: "2000년대", 2010: "2010년대"}
# an artist counts as popular music when most of its songs are in one of these genres
POP_GENRES = {"발라드", "댄스", "랩/힙합", "R&B/Soul", "록/메탈", "인디음악", "아이돌", "성인가요",
              "OST", "포크/블루스"}  # fmt: skip
ORDINAL = {1: "첫 번째", 2: "두 번째", 3: "세 번째"}
# artist names that are everyday words (pre-run review, 2026-11-04)
COMMON_WORDS = {"나비", "제이", "지아", "신화", "준수"}
# sampling weights for turns 2..5; an operation that can't apply is drawn again. A last turn
# also draws "reference" (weight 6)
OP_WEIGHTS = {"add": 3, "relax": 2, "replace": 3, "exclude": 3, "more": 1, "new_topic": 1}


# ------------------------------------------------------------------ Korean particles
def _final(word: str) -> tuple[bool, bool]:
    """(ends in a consonant, that consonant is ㄹ), read the way the word is pronounced."""
    w = word.rstrip(" '\")")
    ch = w[-1]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return jong != 0, jong == 8
    if ch.upper() in "LR":
        return True, True
    if ch.upper() in "MN":
        return True, False
    return False, False


def eul(w: str) -> str:
    return w + ("을" if _final(w)[0] else "를")


def eun(w: str) -> str:
    return w + ("은" if _final(w)[0] else "는")


def ro(w: str) -> str:
    fin, rieul = _final(w)
    return w + ("으로" if fin and not rieul else "로")


def i_ga(w: str) -> str:
    return w + ("이" if _final(w)[0] else "가")


# ------------------------------------------------------------------ state
@dataclass
class State:
    artist: dict | None = None  # {"key", "name"}: name is the surface used in the text
    genre: dict | None = None  # {"key": Genre.name, "name": surface}
    tags: list[str] = field(default_factory=list)  # Tag.name, at most 2
    decade: int | None = None
    neg_artists: list[dict] = field(default_factory=list)
    neg_genres: list[dict] = field(default_factory=list)
    neg_tags: list[str] = field(default_factory=list)

    def copy(self) -> "State":
        return State(**orjson.loads(orjson.dumps(dataclasses.asdict(self))))

    def positives(self) -> list[str]:
        return [k for k in ("artist", "genre", "decade") if getattr(self, k)] + (
            ["tags"] if self.tags else []
        )

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["year_from"], d["year_to"] = (
            (self.decade, self.decade + 9) if self.decade else (None, None)
        )
        return d


_COUNT = """
{anchor}
WITH DISTINCT s
WHERE ($genre IS NULL OR EXISTS {{ (s)-[:HAS_GENRE]->(g:Genre) WHERE g.name = $genre }})
  AND ($artist IS NULL OR EXISTS {{ (s)-[:PERFORMED_BY]->(a:Artist) WHERE a.artist_id = $artist }})
  AND ALL(t IN $tags WHERE EXISTS {{ (s)-[:HAS_TAG]->(:Tag {{name: t}}) }})
  AND NOT EXISTS {{ (s)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $neg_genres }}
  AND NOT EXISTS {{ (s)-[:PERFORMED_BY]->(a:Artist) WHERE a.artist_id IN $neg_artists }}
  AND NOT EXISTS {{ (s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $neg_tags }}
  AND ($year_from IS NULL OR toInteger(left(s.issue_date, 4)) >= $year_from)
  AND ($year_to IS NULL OR toInteger(left(s.issue_date, 4)) <= $year_to)
WITH s ORDER BY s.popularity DESC LIMIT $cap
RETURN s.song_id AS id,
       [(s)-[:HAS_GENRE]->(g:Genre) | g.name] AS genres,
       [(s)-[:PERFORMED_BY]->(a:Artist) | a.artist_id] AS artists,
       [(s)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $pool | t.name] AS tags
"""


def _params(s: State) -> dict:
    j = s.to_json()
    return {
        "genre": s.genre["key"] if s.genre else None,
        "artist": s.artist["key"] if s.artist else None,
        "tags": s.tags,
        "neg_genres": [g["key"] for g in s.neg_genres],
        "neg_artists": [a["key"] for a in s.neg_artists],
        "neg_tags": s.neg_tags,
        "year_from": j["year_from"],
        "year_to": j["year_to"],
    }


async def answers(graph: Graph, s: State, cap: int = 200) -> list[dict]:
    """The state's most popular answer songs, with their genres, artists and pool tags."""
    if s.artist:
        anchor = "MATCH (:Artist {artist_id: $artist})<-[:PERFORMED_BY]-(s:Song)"
    elif s.genre:
        anchor = "MATCH (g0:Genre) WHERE g0.name = $genre MATCH (s:Song)-[:HAS_GENRE]->(g0)"
    elif s.tags:
        anchor = "MATCH (:Tag {name: $tags[0]})<-[:HAS_TAG]-(s:Song)"
    else:
        raise ValueError("a state needs a positive constraint")
    q = _COUNT.format(anchor=anchor)
    return await graph.read(q, timeout_s=60, cap=cap, pool=TAGS, **_params(s))


# ------------------------------------------------------------------ value pools
@dataclass
class Pools:
    artists: list[dict]  # {"key", "name", "genre"}
    genres: list[dict]  # {"key", "name"}
    tags: list[str]


async def build_pools(settings: Settings, graph: Graph) -> Pools:
    search = create_search(settings)
    bm25 = BM25(search, settings.songs_index, settings.entities_index)
    names = [r["name"] for r in await graph.read(
        "MATCH (g:Genre) WHERE g.name IS NOT NULL RETURN DISTINCT g.name AS name")]  # fmt: skip
    linker = EntityLinker(bm25, None, None, names)  # type: ignore[arg-type]  # genre/artist only
    try:
        genres = []
        for surface, key in GENRES:
            hit = linker.genre(surface)
            if hit and hit.key == key:
                genres.append({"key": key, "name": surface})
            else:
                logger.warning("genre %r links to %r, not %r: left out", surface, hit, key)
        tags = []
        for t in TAGS:
            hits = await bm25.link(t, "tag", k=1)
            if hits and hits[0].exact and hits[0].key == t:
                tags.append(t)
            else:
                logger.warning("tag %r has no exact Tag: left out", t)
        # nested names (여름 / 여름밤) would let one value read as the other: keep the shorter
        nested = [t for t in tags if any(o != t and o in t for o in tags)]
        if nested:
            logger.warning("nested tags left out: %s", nested)
        tags = [t for t in tags if t not in nested]
        words = tags + [g["name"] for g in genres]
        resp = await search.search(
            index=settings.entities_index,
            body={"size": 1500, "query": {"term": {"kind": "artist"}},
                  "sort": [{"popularity": "desc"}], "_source": ["key", "name"]},
        )  # fmt: skip
        artists = []
        for h in resp["hits"]["hits"]:
            key, name = h["_source"]["key"], h["_source"]["name"]
            m = re.fullmatch(r"(.+?)\s*\((.+)\)", name)
            parts = [m.group(1), m.group(2)] if m else [name]
            surface = next((p for p in parts if re.fullmatch(r"[가-힣 ]{2,}", p.strip())), None)
            if not surface:
                continue
            surface = surface.strip()
            # a name that holds a tag or genre word (여행스케치, 전영록), is a common word, or is a
            # tag name itself could be read as something other than an artist
            if surface in COMMON_WORDS or any(w in surface for w in words):
                continue
            tag_hit = await bm25.link(surface, "tag", k=1)
            if tag_hit and tag_hit[0].exact:
                continue
            row = await graph.read(
                "MATCH (a:Artist {artist_id: $key})<-[:PERFORMED_BY]-(s:Song)-[:HAS_GENRE]->(g:Genre) "
                "WITH g.name AS genre, count(DISTINCT s) AS c ORDER BY c DESC RETURN genre LIMIT 1",
                key=key,
            )
            if not row or row[0]["genre"] not in POP_GENRES:
                continue
            hit = await linker.artist(surface)
            if not hit or hit.key != key:
                continue
            artists.append({"key": key, "name": surface, "genre": row[0]["genre"]})
            if len(artists) >= 150:
                break
        return Pools(artists, genres, tags)
    finally:
        await search.close()


# ------------------------------------------------------------------ templates
def _t(rng: random.Random, options: list[tuple[str, bool]]) -> tuple[str, bool]:
    return rng.choice(options)


def say_start(s: State, rng: random.Random) -> str:
    a = s.artist["name"] if s.artist else None
    g = s.genre["name"] if s.genre else None
    t = s.tags[0] if s.tags else None
    d = DECADES[s.decade] if s.decade else None
    if a and t:
        return rng.choice(
            [f"{a} 노래 중에 {t} 노래 추천해줘", f"{a} 곡 중에 '{t}' 느낌 나는 거 있어?"]
        )
    if g and t:
        return rng.choice([f"{t} {g} 듣고 싶어", f"'{t}' 느낌 나는 {g} 추천해줘"])
    if d and g:
        return rng.choice([f"{d} {g} 추천해줘", f"{d} {g} 노래 틀어줘"])
    if a:
        return rng.choice([f"{a} 노래 추천해줘", f"{a} 노래 듣고 싶어", f"{a} 곡 좀 틀어줘"])
    if g:
        return rng.choice(
            [f"{g} 추천해줘", f"{g} 듣고 싶어", f"요즘 {i_ga(g)} 땡겨", f"{g} 노래 틀어줘"]
        )
    return rng.choice(
        [f"{t} 노래 추천해줘", f"'{t}' 느낌 나는 노래 듣고 싶어", f"{t} 분위기 노래 틀어줘"]
    )


def say(op: str, kind: str, old, new, rng: random.Random, n_tags: int = 0) -> tuple[str, bool]:
    """(utterance, explicit?) for one operation. old/new are surfaces; n_tags is how many tags
    the state had before it, so a tag edit says which reading it means."""
    if op == "add" and kind == "tags" and n_tags:
        return _t(rng, [(f"'{new}' 느낌도 더해서", True), (f"{new} 느낌도 같이", True)])
    if op == "relax" and kind == "tags" and n_tags > 1:
        return f"'{old}' 느낌은 없어도 돼", True
    if op == "replace" and kind == "tags" and n_tags > 1:
        return f"{old} 말고 {new} 느낌으로", True
    if op == "add":
        return _t(rng, {
            "artist": [(f"{new} 노래로만 골라줘", True), (f"그중에서 {new} 걸로", True)],
            "genre": [(f"그중에 {new}만", True), (f"{ro(new)} 골라줘", True)],
            "tags": [(f"좀 더 {new} 느낌으로", True), (f"거기서 '{new}' 분위기 나는 걸로", True),
                     (f"{new} 노래로 추려줘", True)],
            "decade": [(f"{new} 노래로만", True), (f"{new} 걸로 골라줘", True)],
        }[kind])  # fmt: skip
    if op == "relax":
        return _t(rng, {
            "artist": [("가수는 상관없어", True), ("다른 가수 노래도 섞어줘", True)],
            "genre": [("장르는 상관없어", True), ("장르 상관없이 보여줘", True)],
            "tags": [("분위기는 상관없어", True), (f"'{old}' 느낌은 없어도 돼", True)],
            "decade": [("연도는 상관없어", True), ("시대는 상관없어", True)],
        }[kind])  # fmt: skip
    if op == "replace":
        return _t(rng, {
            "artist": [(f"{old} 말고 {new} 노래로 바꿔줘", True), (f"{eun(new)}?", False),
                       (f"{new} 노래는 어때?", False)],
            "genre": [(f"{old} 말고 {ro(new)} 바꿔줘", True), (f"{eun(new)}?", False),
                      (f"{eun(new)} 어때?", False)],
            "tags": [(f"{old} 말고 {new} 느낌으로", True), (f"이번엔 '{new}' 느낌으로", False)],
            "decade": [(f"{new} 걸로 바꿔줘", True), (f"{eun(new)}?", False)],
        }[kind])  # fmt: skip
    if op == "exclude":
        return _t(rng, {
            "artist": [(f"{new} 노래는 빼줘", True), (f"{eun(new)} 빼고", True)],
            "genre": [(f"{eun(new)} 빼줘", True), (f"{new} 말고 다른 걸로", True)],
            "tags": [(f"'{new}' 느낌 나는 건 빼줘", True), (f"{new} 노래는 빼고", True)],
        }[kind])  # fmt: skip
    if op == "reference":
        k = ORDINAL[new]
        return _t(rng, {
            "artist": [(f"방금 {k} 곡 부른 가수 노래 더 들려줘", True), (f"{k} 곡 가수 노래로", True)],
            "genre": [(f"{k} 곡이랑 같은 장르로", True), (f"{k} 곡 같은 장르로 더", True)],
        }[kind])  # fmt: skip
    if op == "more":
        return _t(rng, [("더 보여줘", True), ("다른 곡도 추천해줘", True), ("더 없어?", True)])
    raise ValueError(op)


# ------------------------------------------------------------------ programs
class Gen:
    def __init__(self, graph: Graph, pools: Pools, rng: random.Random, turns: int = TURNS):
        self.g, self.p, self.rng = graph, pools, rng
        self.turns = turns

    async def feasible(self, s: State) -> list[dict] | None:
        ans = await answers(self.g, s)
        return ans if len(ans) >= MIN_ANSWERS else None

    async def start(self) -> tuple[State, list[dict]]:
        for _ in range(50):
            s = State()
            shape = self.rng.choice(
                ["artist", "genre", "tag", "artist+tag", "genre+tag", "decade+genre"]
            )
            if "artist" in shape:
                a = self.rng.choice(self.p.artists)
                s.artist = {"key": a["key"], "name": a["name"]}
            if "genre" in shape:
                s.genre = dict(self.rng.choice(self.p.genres))
            if "tag" in shape:
                s.tags = [self.rng.choice(self.p.tags)]
            if "decade" in shape:
                s.decade = self.rng.choice(list(DECADES))
            ans = await self.feasible(s)
            if ans:
                return s, ans
        raise RuntimeError("no feasible start state")

    async def step(
        self, s: State, ans: list[dict], last: bool, allow_new: bool
    ) -> tuple[dict, State, list[dict]] | None:
        """One operation on s: (turn record without text, new state, its answers), or None."""
        w = {k: v for k, v in OP_WEIGHTS.items() if allow_new or k != "new_topic"}
        if last:
            w["reference"] = 6
        op = self.rng.choices(list(w), weights=list(w.values()))[0]
        r = self.rng
        n = s.copy()
        if op == "more":
            return {"op": op, "kind": None, "old": None, "new": None}, n, ans
        if op == "new_topic":
            n2, a2 = await self.start()
            return {"op": op, "kind": None, "old": None, "new": None}, n2, a2
        if op == "reference":
            kind = r.choice(["artist", "genre"])
            return {"op": op, "kind": kind, "old": None, "new": r.choice([1, 2, 3])}, n, ans
        if op == "add":
            free = [k for k in ("artist", "genre", "decade") if not getattr(s, k)] + (
                ["tags"] if len(s.tags) < 2 else []
            )
            if not free:
                return None
            kind = r.choice(free)
            new = self._value(kind, s, ans)
            if new is None:
                return None
            self._set(n, kind, new)
        elif op == "relax":
            pos = s.positives()
            # what is left must still anchor a search: an artist, a genre or a tag
            pos = [k for k in pos if any(x != k for x in s.positives() if x != "decade")]
            if len(s.positives()) < 2 or not pos:
                return None
            kind = r.choice(pos)
            old = self._surface(s, kind)
            if kind == "tags":
                old = r.choice(s.tags)
                n.tags = [t for t in s.tags if t != old]
            else:
                setattr(n, kind, None)
            rec = {"op": op, "kind": kind, "old": old, "new": None}
            a2 = await self.feasible(n)
            return (rec, n, a2) if a2 else None
        elif op == "replace":
            pos = s.positives()
            if not pos:
                return None
            kind = r.choice(pos)
            old = r.choice(s.tags) if kind == "tags" else self._surface(s, kind)
            new = self._value(kind, s, ans, replacing=True)
            if new is None:
                return None
            if kind == "tags":
                n.tags = [new if t == old else t for t in s.tags]
            else:
                self._set(n, kind, new)
        else:  # exclude: a value the current answers carry, so the exclusion bites
            kind = r.choice(["artist", "genre", "tags"])
            new = self._excludable(kind, s, ans)
            if new is None:
                return None
            if kind == "artist":
                n.neg_artists.append(new)
            elif kind == "genre":
                n.neg_genres.append(new)
            else:
                n.neg_tags.append(new)
            old = None
        a2 = await self.feasible(n)
        if not a2:
            return None
        old = (
            self._surface(s, kind)
            if op == "replace" and kind != "tags"
            else (old if op != "add" else None)
        )
        rec = {"op": op, "kind": kind, "old": old, "new": self._name(kind, new)}
        return rec, n, a2

    def _surface(self, s: State, kind: str):
        v = getattr(s, kind)
        if kind == "decade":
            return DECADES[v] if v else None
        if kind == "tags":
            return s.tags[0] if s.tags else None
        return v["name"] if v else None

    @staticmethod
    def _name(kind: str, v):
        if kind == "decade":
            return DECADES[v]
        return v["name"] if isinstance(v, dict) else v

    def _set(self, n: State, kind: str, v) -> None:
        if kind == "tags":
            n.tags = [*n.tags, v]
        else:
            setattr(n, kind, v)

    def _value(self, kind: str, s: State, ans: list[dict], replacing: bool = False):
        r = self.rng
        if kind == "artist":
            # an artist that has songs in the current answers keeps the state feasible
            have = {a for x in ans for a in x["artists"]}
            opts = [a for a in self.p.artists if (replacing or a["key"] in have)
                    and (not s.artist or a["key"] != s.artist["key"])
                    and a["key"] not in {x["key"] for x in s.neg_artists}]  # fmt: skip
            if replacing and s.genre:
                opts = [a for a in opts if a["genre"] == s.genre["key"]] or opts
            return {"key": (a := r.choice(opts))["key"], "name": a["name"]} if opts else None
        if kind == "genre":
            have = {g for x in ans for g in x["genres"]}
            opts = [g for g in self.p.genres if (replacing or g["key"] in have)
                    and (not s.genre or g["key"] != s.genre["key"])
                    and g["key"] not in {x["key"] for x in s.neg_genres}]  # fmt: skip
            return dict(r.choice(opts)) if opts else None
        if kind == "tags":
            have = [t for x in ans for t in x["tags"]]
            opts = [t for t in self.p.tags if (replacing or t in have)
                    and t not in s.tags and t not in s.neg_tags]  # fmt: skip
            return r.choice(opts) if opts else None
        opts = [d for d in DECADES if d != s.decade]
        return r.choice(opts)

    def _excludable(self, kind: str, s: State, ans: list[dict]):
        """A value carried by at least 10% of the current answers, not one the state asks for."""
        from collections import Counter

        if kind == "artist" and not s.artist:
            c = Counter(a for x in ans for a in set(x["artists"]))
            known = {a["key"]: a for a in self.p.artists}
            opts = [known[k] for k, v in c.items() if k in known and v >= 0.1 * len(ans)]
            return {"key": (a := self.rng.choice(opts))["key"], "name": a["name"]} if opts else None
        if kind == "genre":
            c = Counter(g for x in ans for g in set(x["genres"]))
            known = {g["key"]: g for g in self.p.genres}
            opts = [known[k] for k, v in c.items() if k in known and v >= 0.1 * len(ans)
                    and (not s.genre or k != s.genre["key"])]  # fmt: skip
            return dict(self.rng.choice(opts)) if opts else None
        if kind == "tags":
            c = Counter(t for x in ans for t in set(x["tags"]))
            opts = [t for t, v in c.items() if v >= 0.1 * len(ans) and t not in s.tags]
            return self.rng.choice(opts) if opts else None
        return None

    async def conversation(self, cid: str) -> list[dict]:
        s, ans = await self.start()
        turns = [{"conversation_id": cid, "turn": 1, "op": "start", "kind": None, "explicit": True,
                  "text": say_start(s, self.rng), "state": s.to_json(), "answers": len(ans)}]  # fmt: skip
        new_used = False
        for t in range(2, self.turns + 1):
            for _ in range(30):
                # at most one new topic, never right after the start
                got = await self.step(
                    s, ans, last=t == self.turns, allow_new=t > 2 and not new_used
                )
                if got:
                    break
            else:
                raise RuntimeError(f"{cid}: no feasible operation at turn {t}")
            rec, s2, ans2 = got
            new_used |= rec["op"] == "new_topic"
            if rec["op"] == "new_topic":
                text, explicit = (
                    self.rng.choice(["이제 다른 거 듣고 싶어. ", "됐고, "])
                    + say_start(s2, self.rng),
                    True,
                )
            else:
                text, explicit = say(
                    rec["op"], rec["kind"], rec["old"], rec["new"], self.rng, len(s.tags)
                )
            turn = {"conversation_id": cid, "turn": t, **rec, "explicit": explicit, "text": text,
                    "state": s2.to_json(), "answers": len(ans2)}  # fmt: skip
            if rec["op"] == "reference":
                turn["state"] = None  # resolved per arm from its own previous list
                turn["base"] = s.to_json()
            turns.append(turn)
            s, ans = s2, ans2
        return turns


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


async def main(out: Path, extra: str | None = None, turns: int = TURNS) -> None:
    settings = Settings()
    graph = Graph(settings)
    try:
        pools = await build_pools(settings, graph)
        logger.info(
            "pools: %d artists, %d genres, %d tags",
            len(pools.artists),
            len(pools.genres),
            len(pools.tags),
        )
        dumped = orjson.dumps(dataclasses.asdict(pools), option=orjson.OPT_INDENT_2)
        splits = SPLITS
        if extra:
            # a new draw from the same generator and the same pools, e.g. after a set was used
            name, n = extra.split(":")
            if name in SPLITS:
                raise SystemExit(f"{name} is a fixed split")
            if (out / "pools.json").read_bytes() != dumped:
                raise SystemExit("pools differ from pools.json: the graph changed")
            splits = {name: (int(n), name)}
        else:
            _write(out / "pools.json", dumped)
        for split, (n, seed) in splits.items():
            gen = Gen(graph, pools, random.Random(f"context-set:{seed}"), turns)
            rows = []
            for i in range(n):
                rows += await gen.conversation(f"{split}-{i:04d}")
                if (i + 1) % 25 == 0:
                    logger.info("%s: %d / %d conversations", split, i + 1, n)
            _write(out / f"{split}.jsonl", b"".join(orjson.dumps(r) + b"\n" for r in rows))
    finally:
        await graph.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", help="NAME:N, one more draw (see the module docstring)")
    ap.add_argument(
        "--turns", type=int, default=TURNS, help="turns per conversation (--split only)"
    )
    args = ap.parse_args()
    if args.turns != TURNS and not args.split:
        raise SystemExit("--turns is for a --split draw; dev and test stay at 5 turns")
    asyncio.run(main(args.out, args.split, args.turns))
    sys.exit(0)
