"""Synthetic Korean query noise for the robustness eval (docs/prereg-typo-robustness.md).

One edit per turn, in one content token (a whitespace token with two Hangul syllables in a
row), chosen by a seed made of the run's noise kind, conversation and turn:

- `typo`: one jamo is replaced by a neighbouring key of the 2-set (두벌식) keyboard, of the same
  kind (consonant for consonant, vowel for vowel), or the final consonant is dropped;
- `spacing`: either the space between two tokens is removed, or a space is put inside a token
  ('삼성노트북' and '삼성 노트북' are both common queries);
- `hanyoung`: the token is typed with the keyboard left in English mode ('아이유' -> 'dkdldb').

A turn with no content token is left as it is, and `apply` says so.
"""

import functools
import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import orjson

from stunes.config import _REPO

KINDS = ("typo", "spacing", "hanyoung")
TARGETS = _REPO / "results" / "eval" / "noise-targets.json"

_BASE, _LAST = 0xAC00, 0xD7A3
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]

# the 2-set layout: jamo -> the key it is typed with (shift for the doubled ones)
_KEY = {
    "ㅂ": "q", "ㅈ": "w", "ㄷ": "e", "ㄱ": "r", "ㅅ": "t", "ㅛ": "y", "ㅕ": "u", "ㅑ": "i",
    "ㅐ": "o", "ㅔ": "p", "ㅁ": "a", "ㄴ": "s", "ㅇ": "d", "ㄹ": "f", "ㅎ": "g", "ㅗ": "h",
    "ㅓ": "j", "ㅏ": "k", "ㅣ": "l", "ㅋ": "z", "ㅌ": "x", "ㅊ": "c", "ㅍ": "v", "ㅠ": "b",
    "ㅜ": "n", "ㅡ": "m", "ㅃ": "Q", "ㅉ": "W", "ㄸ": "E", "ㄲ": "R", "ㅆ": "T", "ㅒ": "O",
    "ㅖ": "P",
}  # fmt: skip
# compound jamo are typed as two keys
_SPLIT = {
    "ㅘ": "ㅗㅏ", "ㅙ": "ㅗㅐ", "ㅚ": "ㅗㅣ", "ㅝ": "ㅜㅓ", "ㅞ": "ㅜㅔ", "ㅟ": "ㅜㅣ", "ㅢ": "ㅡㅣ",
    "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ",
    "ㄾ": "ㄹㅌ", "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ",
}  # fmt: skip
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_JAMO_OF = {k.lower(): j for j, k in _KEY.items() if k.islower()}


def _neighbours(jamo: str) -> list[str]:
    """Jamo on the keys next to this one's key (left, right, the rows above and below)."""
    key = _KEY.get(jamo, "").lower()
    if not key:
        return []
    r = next(i for i, row in enumerate(_ROWS) if key in row)
    c = _ROWS[r].index(key)
    near = [_ROWS[r][c + d] for d in (-1, 1) if 0 <= c + d < len(_ROWS[r])]
    for rr in (r - 1, r + 1):
        if 0 <= rr < len(_ROWS):
            near += [_ROWS[rr][cc] for cc in (c, c + 1) if 0 <= cc < len(_ROWS[rr])]
    return [_JAMO_OF[k] for k in near if k in _JAMO_OF]


def _is_syllable(ch: str) -> bool:
    return _BASE <= ord(ch) <= _LAST


def decompose(ch: str) -> tuple[int, int, int]:
    n = ord(ch) - _BASE
    return n // 588, (n % 588) // 28, n % 28


def compose(cho: int, jung: int, jong: int) -> str:
    return chr(_BASE + cho * 588 + jung * 28 + jong)


def to_keys(text: str) -> str:
    """What the keys spell with the keyboard in English mode."""
    out = []
    for ch in text:
        if not _is_syllable(ch):
            out.append(ch)
            continue
        cho, jung, jong = decompose(ch)
        for j in (CHO[cho], JUNG[jung], JONG[jong]):
            for part in _SPLIT.get(j, j):
                out.append(_KEY.get(part, part))
    return "".join(out)


@dataclass
class Noised:
    text: str
    applied: bool
    detail: str  # what was changed, for the record


def content_tokens(tokens: list[str]) -> list[int]:
    """Tokens with two Hangul syllables in a row: a word, not a particle or a number."""
    return [
        i
        for i, t in enumerate(tokens)
        if any(_is_syllable(a) and _is_syllable(b) for a, b in itertools.pairwise(t))
    ]


def _typo(token: str, rng: random.Random) -> tuple[str, str] | None:
    spots = [i for i, ch in enumerate(token) if _is_syllable(ch)]
    rng.shuffle(spots)
    for i in spots:
        cho, jung, jong = decompose(token[i])
        options: list[tuple[str, tuple[int, int, int]]] = []
        options += [(f"{CHO[cho]}->{j}", (CHO.index(j), jung, jong))
                    for j in _neighbours(CHO[cho]) if j in CHO]  # fmt: skip
        options += [(f"{JUNG[jung]}->{j}", (cho, JUNG.index(j), jong))
                    for j in _neighbours(JUNG[jung]) if j in JUNG]  # fmt: skip
        if jong:
            options += [(f"{JONG[jong]}->{j}", (cho, jung, JONG.index(j)))
                        for j in _neighbours(JONG[jong]) if j in JONG]  # fmt: skip
            options.append((f"drop final {JONG[jong]}", (cho, jung, 0)))
        if options:
            what, parts = rng.choice(options)
            return token[:i] + compose(*parts) + token[i + 1 :], what
    return None


@functools.cache
def load_targets(path: Path = TARGETS) -> dict[str, list[int]]:
    """Per turn ("conversation:turn"), the tokens that carry the request (noise_targets.py)."""
    if not path.exists():
        return {}
    return orjson.loads(path.read_bytes())["targets"]


def apply(kind: str, text: str, seed: str, targets: list[int] | None = None) -> Noised:
    """The noised turn. `seed` fixes the edit, so every run noises a turn the same way. The
    edit goes to one of `targets` if any is a content token, else to any content token."""
    if kind not in KINDS:
        raise ValueError(f"unknown noise kind {kind!r}")
    rng = random.Random(f"{kind}:{seed}")
    tokens = text.split(" ")
    content = content_tokens(tokens)
    if not content:
        return Noised(text, False, "no content token")
    aimed = [i for i in targets or [] if i in content]
    i = rng.choice(aimed or content)
    tok = tokens[i]
    tag = "" if aimed else " [no target]"
    if kind == "typo":
        got = _typo(tok, rng)
        if got is None:
            return Noised(text, False, "no editable jamo")
        tokens[i], what = got
        return Noised(" ".join(tokens), True, f"{tok} -> {tokens[i]} ({what}){tag}")
    if kind == "hanyoung":
        tokens[i] = to_keys(tok)
        return Noised(" ".join(tokens), True, f"{tok} -> {tokens[i]}{tag}")
    # spacing: join with the next token, or split this one (a coin flip; join needs a neighbour)
    words = [j for j, t in enumerate(tokens) if t]
    if rng.random() < 0.5 and i + 1 < len(tokens) and tokens[i + 1] and len(words) > 1:
        joined = tok + tokens[i + 1]
        detail = f"'{tok} {tokens[i + 1]}' -> '{joined}'"
        tokens[i : i + 2] = [joined]
        return Noised(" ".join(tokens), True, detail + tag)
    cut = [p for p in range(1, len(tok)) if _is_syllable(tok[p - 1]) and _is_syllable(tok[p])]
    p = rng.choice(cut)
    tokens[i] = tok[:p] + " " + tok[p:]
    return Noised(" ".join(tokens), True, f"'{tok}' -> '{tokens[i]}'{tag}")
