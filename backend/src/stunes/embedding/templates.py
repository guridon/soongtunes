"""Text templates that turn catalog records into embedding inputs.

The template *spec* (format + rendering parameters) is hashed into the version registry, so
changing what goes into the text forces a new embedding version instead of quietly
mixing two input distributions in one index.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    spec: str
    render: Callable[[dict], str]
    key: Callable[[dict], str]


def _join(values, n: int) -> str:
    return ", ".join(v for v in list(values)[:n] if v)


def render_song_v1(doc: dict) -> str:
    lines = [f"{doc['title']} - {_join(doc.get('artists') or [], 3)}".strip(" -")]
    if doc.get("album"):
        lines.append(f"앨범: {doc['album']}")
    if doc.get("year"):
        lines.append(f"발매: {doc['year']}년")
    if doc.get("genres"):
        lines.append(f"장르: {_join(doc['genres'], 3)}")
    if doc.get("tags"):
        lines.append(f"태그: {_join(doc['tags'], 10)}")
    if doc.get("playlists"):
        lines.append(f"플레이리스트: {_join(doc['playlists'], 3)}")
    return "\n".join(lines)


SONG_V1 = Template(
    spec=(
        "song/v1: '{title} - {artists[:3]}' + lines(앨범, 발매 {year}년, 장르[:3], "
        "태그[:10] by playlist co-occurrence, 플레이리스트[:3] by like_cnt); empty lines dropped"
    ),
    render=render_song_v1,
    key=lambda doc: doc["song_id"],
)

TAG_V1 = Template(
    spec="tag/v1: '음악 태그: {name}'",
    render=lambda doc: f"음악 태그: {doc['name']}",
    key=lambda doc: doc["key"],
)

TEMPLATES = {"song": SONG_V1, "tag": TAG_V1}
