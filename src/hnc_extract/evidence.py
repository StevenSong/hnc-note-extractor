"""Deterministic verification that evidence quotes occur in the source text.

Matching tiers, tried in order (the first that succeeds is recorded as `method`):
  exact       the quote is a literal substring
  whitespace  equal after collapsing whitespace runs to one space
  normalized  additionally case-insensitive, with typographic quotes/dashes/NBSP
              folded to ASCII

Offsets always refer to the ORIGINAL text, so the review export can highlight the
span exactly. Ellipses are never bridged: a quote with "..." is a paraphrase by
construction and fails, which triggers a retry asking for separate spans.
"""

from __future__ import annotations

import dataclasses

_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "‑": "-",
        "−": "-",
        " ": " ",
    }
)


@dataclasses.dataclass(frozen=True)
class SpanMatch:
    quote: str
    found: bool
    method: str | None  # exact | whitespace | normalized | None
    start: int | None  # first occurrence, offsets into the original text
    end: int | None
    n_occurrences: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _index(text: str, fold: bool) -> tuple[str, list[int]]:
    """Collapse whitespace (and optionally fold case/punctuation).

    Returns the normalized string and, for each normalized char, the index of
    the original char it came from.
    """
    out: list[str] = []
    pos: list[int] = []
    prev_space = True  # also strips leading whitespace
    for i, ch in enumerate(text):
        if fold:
            ch = ch.translate(_FOLD)
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                pos.append(i)
            prev_space = True
            continue
        prev_space = False
        for c in ch.lower() if fold else ch:
            out.append(c)
            pos.append(i)
    if out and out[-1] == " ":
        out.pop()
        pos.pop()
    return "".join(out), pos


def _find_all(hay: str, needle: str) -> list[int]:
    hits, i = [], hay.find(needle)
    while i != -1:
        hits.append(i)
        i = hay.find(needle, i + len(needle))
    return hits


class EvidenceIndex:
    """Precomputed normalized views of one document's text."""

    def __init__(self, text: str):
        self.text = text
        self._views = {
            "whitespace": _index(text, fold=False),
            "normalized": _index(text, fold=True),
        }

    def match(self, quote: str, near: int | None = None) -> SpanMatch:
        """Locate `quote`. With several occurrences, report the one closest to
        `near` (e.g. the target specimen's own evidence), else the first."""
        q = quote.strip()
        if not q or "..." in q or "…" in q:
            return SpanMatch(quote, False, None, None, None, 0)
        hits = _find_all(self.text, q)
        if hits:
            s = _closest(hits, near)
            return SpanMatch(quote, True, "exact", s, s + len(q), len(hits))
        for method, (view, pos) in self._views.items():
            nq, _ = _index(q, fold=(method == "normalized"))
            if not nq:
                continue
            hits = _find_all(view, nq)
            if hits:
                spans = {pos[h]: pos[h + len(nq) - 1] + 1 for h in hits}
                s = _closest(list(spans), near)
                return SpanMatch(quote, True, method, s, spans[s], len(hits))
        return SpanMatch(quote, False, None, None, None, 0)


def _closest(starts: list[int], near: int | None) -> int:
    if near is None:
        return starts[0]
    return min(starts, key=lambda s: (abs(s - near), s))


def verify_quotes(
    text: str, quotes: list[str], near: int | None = None
) -> list[SpanMatch]:
    idx = EvidenceIndex(text)
    return [idx.match(q, near) for q in quotes]
