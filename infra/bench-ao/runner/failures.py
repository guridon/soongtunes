"""Failure kinds of Auto-Ontology turns (docs/prereg-ao-baseline.md, "실패 분류").

Reads the run's records (SQL, steps) and the scorer's output, and labels each scored turn
from the SQL's literals against the truth state. Heuristic: the labels say what the SQL
mentions, not the polarity of every predicate; a sample is checked by hand.

Labels (a turn can have several):
- information_route: no SQL, the question went to the information (catalog) agent
- no_sql_other: no SQL for another reason
- missing:<kind>: a truth value of that kind appears nowhere in the SQL
- stale:<kind>: a value from an earlier turn of the conversation that the truth no longer has
- invented: a name literal that is no value or surface word of the conversation so far: a
  Hangul word, or a genre name of the catalog (e.g. an exclusion nobody asked for)
- translated: a tag, genre or artist name literal in Latin letters, while a truth value of that
  kind is in Hangul and missing from the SQL (e.g. 사랑 -> LOVE, 김지수 -> Kim Ji-soo)
- partial_match: LIKE / ILIKE on a name column
- artist_by_name: artist filtered by name, and the truth artist's name has homonyms
- genre_code: genre filtered by genre_id
- kind_confusion: a truth genre (value or surface word) written as a tag literal, or a truth tag
  as a genre literal
- decade: the truth has a decade and the SQL's year bounds differ

usage: python failures.py RECORDS.jsonl SCORED.jsonl SET.jsonl OUT.jsonl
"""

import json
import re
import sys
from collections import Counter

import psycopg
import sqlglot
from sqlglot import exp

from gold import music_dsn

REC, SCORED, SET, OUT = sys.argv[1:5]
HANGUL = re.compile(r"[가-힣]")
NAME_COLS = {"genre_name", "tag_name", "artist_name", "title"}


def literals(sql: str) -> list[tuple[str | None, str, str]]:
    """(column, operator, value) for every string or number literal compared to a column."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for node in tree.walk():
        if isinstance(node, (exp.EQ, exp.NEQ, exp.Like, exp.ILike, exp.GTE, exp.LTE, exp.GT, exp.LT)):
            col = node.find(exp.Column)
            for lit in node.find_all(exp.Literal):
                out.append((col.name if col else None, type(node).__name__, lit.this))
        elif isinstance(node, exp.In):
            col = node.this if isinstance(node.this, exp.Column) else node.find(exp.Column)
            for lit in node.find_all(exp.Literal):
                out.append((col.name if col else None, "In", lit.this))
        elif isinstance(node, exp.Between):
            col = node.this if isinstance(node.this, exp.Column) else None
            for lit in node.find_all(exp.Literal):
                out.append((col.name if col else None, "Between", lit.this))
    return out


def like_hit(pattern: str, value: str) -> bool:
    rx = "^" + re.escape(pattern.lower()).replace("%", ".*").replace("_", ".") + "$"
    return re.match(rx, value.lower()) is not None


def mentions(lits, cols: set[str], value: str) -> bool:
    for col, op, v in lits:
        if col is not None and col not in cols:
            continue
        if op in ("Like", "ILike") and like_hit(v, value):
            return True
        if v == value or v.lower() == value.lower():
            return True
    return False


truth_turns = {}
for line in open(SET):
    t = json.loads(line)
    truth_turns[(t["conversation_id"], int(t["turn"]))] = t
recs = {(r["conversation_id"], r["turn"]): r for r in map(json.loads, open(REC))}
scored = [json.loads(line) for line in open(SCORED)]

with psycopg.connect(music_dsn()) as conn:
    cur = conn.cursor()
    cur.execute("SELECT artist_name FROM artists GROUP BY 1 HAVING count(*) > 1")
    homonyms = {r[0] for r in cur.fetchall()}
    cur.execute("SELECT artist_id, artist_name FROM artists")
    artist_name = {str(a): n for a, n in cur.fetchall()}
    cur.execute("SELECT genre_name, genre_id FROM genres")
    genre_codes: dict[str, set[str]] = {}
    for n, g in cur.fetchall():
        genre_codes.setdefault(n, set()).add(g)


def surfaces(st: dict | None) -> set[str]:
    """The words the utterances used for artists and genres (the state keeps the key)."""
    if not st:
        return set()
    out = set()
    for k in ("artist", "genre"):
        if st.get(k):
            out.add(st[k]["name"])
    for k in ("neg_artists", "neg_genres"):
        out |= {x["name"] for x in st.get(k) or []}
    return out


def values(st: dict | None) -> dict[str, set[str]]:
    v = {k: set() for k in ("tag", "genre", "artist", "neg_tag", "neg_genre", "neg_artist")}
    if not st:
        return v
    v["tag"] |= set(st.get("tags") or [])
    v["neg_tag"] |= set(st.get("neg_tags") or [])
    if st.get("genre"):
        v["genre"].add(st["genre"]["key"])
    v["neg_genre"] |= {g["key"] for g in st.get("neg_genres") or []}
    if st.get("artist"):
        v["artist"].add(str(st["artist"]["key"]))
    v["neg_artist"] |= {str(a["key"]) for a in st.get("neg_artists") or []}
    return v


def present(lits, kind: str, val: str) -> bool:
    base = kind.removeprefix("neg_")
    if base == "tag":
        return mentions(lits, {"tag_name"}, val)
    if base == "genre":
        return mentions(lits, {"genre_name"}, val) or any(
            mentions(lits, {"genre_id"}, c) for c in genre_codes.get(val, ()))
    name = artist_name.get(val, "")
    return mentions(lits, {"artist_id"}, val) or (name and mentions(lits, {"artist_name"}, name))


rows = []
for s in scored:
    key = (s["conversation_id"], s["turn"])
    r, t = recs[key], truth_turns[key]
    labels = []
    if s["outcome"] == "no_sql":
        steps = r.get("steps") or []
        labels.append("information_route" if "information_agent" in steps else "no_sql_other")
    elif s["outcome"] not in ("reference",):
        lits = literals(s["sql"] or "")
        st = t["state"]
        now = values(st)
        for kind, vals in now.items():
            for v in vals:
                if not present(lits, kind, v):
                    labels.append(f"missing:{kind}")
        # values said earlier in the conversation that the truth no longer holds
        earlier = {k: set() for k in now}
        for i in range(1, s["turn"]):
            for k, vs in values(truth_turns[(s["conversation_id"], i)]["state"]).items():
                earlier[k] |= vs
        for kind in now:
            for v in earlier[kind] - now[kind]:
                if present(lits, kind, v) and not any(present(lits, k2, v) for k2 in now if v in now[k2]):
                    labels.append(f"stale:{kind}")
        said = set()
        for i in range(1, s["turn"] + 1):
            said |= surfaces(truth_turns[(s["conversation_id"], i)]["state"])
        known = {w.lower() for w in said}
        for k in now:
            for v in now[k] | earlier[k]:
                known.add(v.lower())
                if k.endswith("artist"):
                    known.add(artist_name.get(v, "").lower())
                if k.endswith("genre"):
                    known |= {c.lower() for c in genre_codes.get(v, ())}
        str_lits = [(c, o, v) for c, o, v in lits if c in NAME_COLS | {"genre_id"}]
        unknown = [(c, v) for c, o, v in str_lits
                   if v.strip("%").lower() not in known and not any(like_hit(v, k) for k in known)]
        if any(HANGUL.search(v) or v in genre_codes for c, v in unknown):
            labels.append("invented")
        hangul_missing = {
            "tag_name": any(HANGUL.search(v) and not present(lits, k, v)
                            for k in ("tag", "neg_tag") for v in now[k]),
            "genre_name": any(HANGUL.search(v) and not present(lits, k, v)
                              for k in ("genre", "neg_genre") for v in now[k]),
            "artist_name": any(HANGUL.search(artist_name.get(v, "")) and not present(lits, k, v)
                               for k in ("artist", "neg_artist") for v in now[k]),
        }
        if any(c in hangul_missing and hangul_missing[c] and not HANGUL.search(v)
               and re.search(r"[A-Za-z]", v) for c, v in unknown):
            labels.append("translated")
        if any(o in ("Like", "ILike") and c in NAME_COLS for c, o, v in lits):
            labels.append("partial_match")
        if any(c == "artist_name" for c, o, v in lits) and any(
               artist_name.get(a) in homonyms for a in now["artist"] | now["neg_artist"]):
            labels.append("artist_by_name")
        if any(c == "genre_id" for c, o, v in lits):
            labels.append("genre_code")
        genre_words = now["genre"] | now["neg_genre"] | {
            x["name"] for x in ([st["genre"]] if st and st.get("genre") else [])
            + (st.get("neg_genres") or [] if st else [])}
        if any(mentions(lits, {"tag_name"}, g) for g in genre_words) or any(
               mentions(lits, {"genre_name"}, x) for x in now["tag"] | now["neg_tag"]):
            labels.append("kind_confusion")
        if st and st.get("year_from") is not None:
            nums = {int(v) for c, o, v in lits if c in ("release_year", "issue_date") and v.isdigit()
                    and len(v) == 4}
            if not ({st["year_from"], st["year_to"]} <= nums or {st["year_from"], st["year_to"] + 1} <= nums):
                labels.append("decade")
    rows.append({**{k: s[k] for k in ("conversation_id", "turn", "op", "kind", "outcome", "p10", "ex")},
                 "labels": sorted(set(labels))})

with open(OUT, "w") as f:
    for x in rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

prim = [x for x in rows if x["turn"] >= 2 and x["op"] != "reference"]
bad = [x for x in prim if not x["ex"]]
print(f"turns 2-5 without reference: {len(prim)}, not exact {len(bad)}")
c = Counter(lab for x in bad for lab in x["labels"])
for lab, n in c.most_common():
    print(f"  {lab:24s} {n:4d}  ({n / len(bad):.0%} of not-exact turns)")
print("turn 1 labels:", dict(Counter(lab for x in rows if x["turn"] == 1 and not x["ex"] for lab in x["labels"])))
