"""Gold SQL for a context-set truth state, on the relational copy of the served graph.

Mirrors context_set._COUNT condition by condition: genre by name (every genre_id carrying it),
artist by id, every tag by exact name, the three exclusions, and the release year range
(release_year = toInteger(left(issue_date, 4)), computed the same way at export).
"""
import os
from urllib.parse import urlparse

import psycopg


def music_dsn() -> str:
    # CONNECTION_STRINGS in the auto-ontology .env points at the music database
    return os.environ["CONNECTION_STRINGS"].split(",")[0].strip()


def gold_sql(st: dict) -> tuple[str, dict]:
    w, p = [], {}
    if st.get("genre"):
        w.append("EXISTS (SELECT 1 FROM song_genres sg JOIN genres g USING (genre_id) "
                 "WHERE sg.song_id = s.song_id AND g.genre_name = %(genre)s)")
        p["genre"] = st["genre"]["key"]
    if st.get("artist"):
        w.append("EXISTS (SELECT 1 FROM song_artists sa "
                 "WHERE sa.song_id = s.song_id AND sa.artist_id = %(artist)s)")
        p["artist"] = int(st["artist"]["key"])
    for i, t in enumerate(st.get("tags") or []):
        w.append(f"EXISTS (SELECT 1 FROM song_tags st JOIN tags t USING (tag_id) "
                 f"WHERE st.song_id = s.song_id AND t.tag_name = %(tag{i})s)")
        p[f"tag{i}"] = t
    if st.get("neg_genres"):
        w.append("NOT EXISTS (SELECT 1 FROM song_genres sg JOIN genres g USING (genre_id) "
                 "WHERE sg.song_id = s.song_id AND g.genre_name = ANY(%(neg_genres)s))")
        p["neg_genres"] = [g["key"] for g in st["neg_genres"]]
    if st.get("neg_artists"):
        w.append("NOT EXISTS (SELECT 1 FROM song_artists sa "
                 "WHERE sa.song_id = s.song_id AND sa.artist_id = ANY(%(neg_artists)s))")
        p["neg_artists"] = [int(a["key"]) for a in st["neg_artists"]]
    if st.get("neg_tags"):
        w.append("NOT EXISTS (SELECT 1 FROM song_tags st JOIN tags t USING (tag_id) "
                 "WHERE st.song_id = s.song_id AND t.tag_name = ANY(%(neg_tags)s))")
        p["neg_tags"] = list(st["neg_tags"])
    if st.get("year_from") is not None:
        w.append("s.release_year BETWEEN %(year_from)s AND %(year_to)s")
        p["year_from"], p["year_to"] = st["year_from"], st["year_to"]
    if not w:
        raise ValueError("a truth state needs a condition")
    return "SELECT s.song_id FROM songs s WHERE " + " AND ".join(w), p


def gold_set(conn: psycopg.Connection, st: dict) -> set[int]:
    sql, p = gold_sql(st)
    with conn.cursor() as cur:
        cur.execute(sql, p)
        return {r[0] for r in cur.fetchall()}


def conditions_sql(c: dict) -> tuple[str, dict] | None:
    """SQL for the conditions a SoongTunes arm applied (trace.conditions), in the gold SQL's
    terms: genres and artists any-of, tags all-of, exclusions, the year range as applied (an
    open end stays open). The already-shown exclusion (`exclude`, song ids) is not a condition."""
    w, p = [], {}
    if c.get("genres"):
        w.append("EXISTS (SELECT 1 FROM song_genres sg JOIN genres g USING (genre_id) "
                 "WHERE sg.song_id = s.song_id AND g.genre_name = ANY(%(genres)s))")
        p["genres"] = list(c["genres"])
    if c.get("artists"):
        w.append("EXISTS (SELECT 1 FROM song_artists sa "
                 "WHERE sa.song_id = s.song_id AND sa.artist_id = ANY(%(artists)s))")
        p["artists"] = [int(a) for a in c["artists"]]
    for i, t in enumerate(c.get("tags") or []):
        w.append(f"EXISTS (SELECT 1 FROM song_tags st JOIN tags t USING (tag_id) "
                 f"WHERE st.song_id = s.song_id AND t.tag_name = %(tag{i})s)")
        p[f"tag{i}"] = t
    for key, sql in (
        ("neg_genres", "NOT EXISTS (SELECT 1 FROM song_genres sg JOIN genres g USING (genre_id) "
                       "WHERE sg.song_id = s.song_id AND g.genre_name = ANY(%(neg_genres)s))"),
        ("neg_artists", "NOT EXISTS (SELECT 1 FROM song_artists sa "
                        "WHERE sa.song_id = s.song_id AND sa.artist_id = ANY(%(neg_artists)s))"),
        ("neg_tags", "NOT EXISTS (SELECT 1 FROM song_tags st JOIN tags t USING (tag_id) "
                     "WHERE st.song_id = s.song_id AND t.tag_name = ANY(%(neg_tags)s))"),
    ):
        if c.get(key):
            w.append(sql)
            p[key] = [int(x) for x in c[key]] if key == "neg_artists" else list(c[key])
    if c.get("year_from") is not None:
        w.append("s.release_year >= %(year_from)s")
        p["year_from"] = c["year_from"]
    if c.get("year_to") is not None:
        w.append("s.release_year <= %(year_to)s")
        p["year_to"] = c["year_to"]
    if not w:
        return None  # searched with no condition
    return "SELECT s.song_id FROM songs s WHERE " + " AND ".join(w), p
