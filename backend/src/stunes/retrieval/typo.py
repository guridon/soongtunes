"""Typo helpers for entity linking: English-mode keys read back as Hangul, jamo edit distance.

Used by the linker's fallbacks for genres and artists (docs/prereg-link-methods.md). The noise
these undo is the robustness eval's (stunes.eval.noise): a neighbouring-key or dropped-final
typo, and a word typed with the keyboard left in English mode ('아이유' -> 'dkdldb').
"""

import re
from pathlib import Path

from stunes.eval import noise

_KEY2JAMO = {v: k for k, v in noise._KEY.items()}
_VOWEL_JOIN = {("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ", ("ㅜ", "ㅓ"): "ㅝ",
               ("ㅜ", "ㅔ"): "ㅞ", ("ㅜ", "ㅣ"): "ㅟ", ("ㅡ", "ㅣ"): "ㅢ"}  # fmt: skip
_FINAL_JOIN = {("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ", ("ㄹ", "ㄱ"): "ㄺ",
               ("ㄹ", "ㅁ"): "ㄻ", ("ㄹ", "ㅂ"): "ㄼ", ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ",
               ("ㄹ", "ㅍ"): "ㄿ", ("ㄹ", "ㅎ"): "ㅀ", ("ㅂ", "ㅅ"): "ㅄ"}  # fmt: skip
_NORM = re.compile(r"[\s'\"`’‘“”]+")
_LATIN = re.compile(r"[A-Za-z]{2,}")
_SHIFT = frozenset("QWERTOP")  # the shifted keys: ㅃ ㅉ ㄸ ㄲ ㅆ ㅒ ㅖ
_STOP = frozenset(
    w
    for w in (Path(__file__).with_name("hanyoung_stop.txt")).read_text().split("\n")
    if w and not w.startswith("#")
)


def _is_v(j: str) -> bool:
    return j in noise.JUNG


def _is_c(j: str) -> bool:
    return j in noise.CHO


def from_keys(text: str) -> str:
    """English-mode keys back to Hangul on the 2-set layout ('dkdldb' -> '아이유')."""
    jamo = [_KEY2JAMO.get(ch) or _KEY2JAMO.get(ch.lower()) or ch for ch in text]
    out, i, n = [], 0, len(jamo)
    while i < n:
        cho = jamo[i]
        if not (_is_c(cho) and i + 1 < n and _is_v(jamo[i + 1])):
            out.append(cho)
            i += 1
            continue
        v = jamo[i + 1]
        i += 2
        if i < n and (v, jamo[i]) in _VOWEL_JOIN:
            v = _VOWEL_JOIN[(v, jamo[i])]
            i += 1
        jong = ""
        # a consonant becomes the final unless a vowel follows it (then it starts the next one)
        if i < n and jamo[i] in noise.JONG and not (i + 1 < n and _is_v(jamo[i + 1])):
            jong = jamo[i]
            i += 1
            if i < n and (jong, jamo[i]) in _FINAL_JOIN and not (i + 1 < n and _is_v(jamo[i + 1])):
                jong = _FINAL_JOIN[(jong, jamo[i])]
                i += 1
        out.append(noise.compose(noise.CHO.index(cho), noise.JUNG.index(v), noise.JONG.index(jong)))
    return "".join(out)


def looks_hanyoung(s: str) -> bool:
    """Only Latin letters, and they spell Hangul syllables on the 2-set layout."""
    letters = [c for c in s if c.isalpha()]
    return (
        bool(letters)
        and all(c.isascii() for c in letters)
        and any(noise._is_syllable(c) for c in from_keys(s))
    )


def read_back(text: str) -> tuple[str, list[tuple[str, str]]]:
    """A sentence's words typed with the keyboard in English mode, read back as Hangul before
    the LLM sees them ("dkdldb 노래" -> "아이유 노래"). A word is read back only if:
    - it is not an English word that happens to spell Hangul (hanyoung_stop.txt);
    - its capitals are shifted keys;
    - every letter group becomes a whole syllable.
    On the v1 set's clean sentences 3 of 960 Latin words would change (Ty, Rivals); 1,260 of
    the robustness eval's 1,263 English-mode words come back (docs/results.md)."""
    changed: list[tuple[str, str]] = []

    def sub(m: re.Match) -> str:
        word = m.group(0)
        if word.lower() in _STOP or any(c.isupper() and c not in _SHIFT for c in word):
            return word
        hangul = from_keys(word)
        if not all(noise._is_syllable(c) for c in hangul):
            return word
        changed.append((word, hangul))
        return hangul

    return _LATIN.sub(sub, text), changed


def jamo(s: str) -> str:
    out = []
    for ch in _NORM.sub("", s).lower():
        if noise._is_syllable(ch):
            c, v, f = noise.decompose(ch)
            out += [noise.CHO[c], noise.JUNG[v], noise.JONG[f]]
        else:
            out.append(ch)
    return "".join(out)


def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def closest(
    mention: str, names: list[tuple[str, str]], max_ratio: float, tie: str = "order"
) -> str | None:
    """The key whose name is closest by jamo edit distance / length, if within max_ratio.
    Ties go to the earlier name ("order": BM25's rank, which carries popularity) or to the
    smaller key ("key", the first measurement)."""
    if not names:
        return None
    jm = jamo(mention)
    scored = [
        (edit(jm, jamo(n)) / max(len(jm), len(jamo(n)), 1), i if tie == "order" else k, k)
        for i, (k, n) in enumerate(names)
    ]
    d, _, key = min(scored)
    return key if d <= max_ratio else None
