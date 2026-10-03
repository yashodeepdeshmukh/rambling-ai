"""Deterministic candidate extraction: the option lists the decider chooses from.

Anything the decider may pick must appear here, so coverage of this module bounds
what the whole pipeline can express. Keep it boring and predictable.
"""

from __future__ import annotations

import re
from fractions import Fraction

WORD_NUMBERS = {
    "half": Fraction(1, 2), "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "twenty": 20, "twice": 2, "double": 2, "thrice": 3,
    "triple": 3, "dozen": 12,
}

STOPWORDS = set("""
a an the and or but of to in on at for from by with without into onto over under
is are was were be been being has have had do does did will would can could should
shall may might must this that these those there their they them it its he she him
her his hers we us our you your i me my mine all any each every some no not only
than then as so if when while what which who whom whose how many much more most less
least together total in all equals equal plus minus times number numbers per
cost costs spent sold buy bought make makes made gives give needs need holds hold
same other both its also just one two three four five six seven eight nine ten
twice double half dozen now years year old new
""".split())

_NUM_RE = re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")
_URL_RE = re.compile(r"https?://[^\s,;\"')]+")
_HOST_RE = re.compile(r"\b((?:\d{1,3}\.){3}\d{1,3}|(?:[a-z0-9-]+\.)+[a-z]{2,}|localhost)(?::(\d{2,5}))?\b", re.I)
_FILE_RE = re.compile(r"(?<![\w/])(/[\w.\-/]+\.[a-z0-9]{2,4})\b")
_PATH_RE = re.compile(r"(?<![\w/:.+#-])(/[a-z0-9_\-]+(?:/[a-z0-9_\-{}:]+)*)(?!\w)(?!\.\w)", re.I)
_TOPIC_RE = re.compile(r"(?<![\w/:.])([a-z0-9_\-+#]+(?:/[a-z0-9_\-+#]+)+)(?!\w)(?!\.\w)", re.I)
_DURATION_RE = re.compile(
    r"\b(\d+|" + "|".join(WORD_NUMBERS) + r")\s*(seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h)\b"
    r"|\b(?:every|per|a|each)\s+(second|minute|hour)\b", re.I)
_UNIT_SECONDS = {"s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600}


def fmt(x) -> str:
    x = Fraction(x)
    if x.denominator == 1:
        return str(x.numerator)
    return str(float(x))


def numbers(text: str) -> list[Fraction]:
    """Numbers in order of first appearance, digits and number words, deduplicated."""
    found: list[tuple[int, Fraction]] = []
    scrubbed = _URL_RE.sub(lambda m: " " * len(m.group()), text)
    for m in _NUM_RE.finditer(scrubbed):
        found.append((m.start(), Fraction(m.group(1).replace(",", "") + (m.group(2) or ""))))
    for m in re.finditer(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", text, re.I):
        found.append((m.start(), Fraction(WORD_NUMBERS[m.group(1).lower()])))
    out: list[Fraction] = []
    for _, v in sorted(found, key=lambda t: t[0]):
        if v not in out:
            out.append(v)
    return out


def entities(text: str, limit: int = 20) -> list[str]:
    """Content words in order of first appearance: candidate names for unknowns and fields."""
    out: list[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z_]+", _URL_RE.sub(" ", text)):
        w = w.lower()
        if len(w) >= 3 and w not in STOPWORDS and w not in out:
            out.append(w)
    return out[:limit]


def urls(text: str) -> list[str]:
    return _dedupe(m.group().rstrip(".") for m in _URL_RE.finditer(text))


def hosts(text: str) -> list[str]:
    no_urls = _URL_RE.sub(" ", text)
    no_files = _FILE_RE.sub(" ", no_urls)
    return _dedupe(m.group(0) for m in _HOST_RE.finditer(no_files)
                   if "/" not in m.group(0))


def files(text: str) -> list[str]:
    return _dedupe(m.group(1) for m in _FILE_RE.finditer(_URL_RE.sub(" ", text)))


def http_paths(text: str) -> list[str]:
    scrubbed = _FILE_RE.sub(" ", _URL_RE.sub(" ", text))
    return _dedupe(m.group(1) for m in _PATH_RE.finditer(scrubbed))


def topics(text: str) -> list[str]:
    scrubbed = _FILE_RE.sub(" ", _URL_RE.sub(" ", text))
    return _dedupe(m.group(1) for m in _TOPIC_RE.finditer(scrubbed))


def durations(text: str) -> list[int]:
    """Durations in seconds, e.g. 'every 5 minutes' -> 300."""
    out = []
    for m in _DURATION_RE.finditer(text):
        if m.group(3):
            n, unit = 1, m.group(3).lower()
        else:
            raw = m.group(1).lower()
            n = int(WORD_NUMBERS[raw]) if raw in WORD_NUMBERS else int(raw)
            unit = m.group(2).lower().rstrip("s") or "s"
        out.append(n * _UNIT_SECONDS[unit])
    return _dedupe(out)


def _dedupe(items):
    out = []
    for i in items:
        if i not in out:
            out.append(i)
    return out
