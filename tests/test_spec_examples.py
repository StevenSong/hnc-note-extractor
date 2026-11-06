"""Spec examples must be self-consistent and must not be copied from dev reports.

Examples are shown to the model in every prompt. If one is excerpted from a report
we evaluate on, the model sees part of the answer key and that report's scores are
inflated. The contamination test compares every prompt-visible string in the specs
against the ingested synthetic reports by shared word runs (8-word shingles).
Phrases found in many reports are lab boilerplate (CLIA disclaimers, methodology
paragraphs) and are allowed. The test skips when no ingested data is present (CI).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
INGESTED = ROOT / "data" / "ingested"
SHINGLE = 8
# A word run that appears in at least this many distinct reports is boilerplate.
BOILERPLATE_MIN_DOCS = 5


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _shingles(s: str) -> set[tuple[str, ...]]:
    t = _tokens(s)
    return {tuple(t[i : i + SHINGLE]) for i in range(len(t) - SHINGLE + 1)}


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _spec_files() -> list[Path]:
    return sorted(SPECS.rglob("*.yaml"))


def _examples():
    for path in _spec_files():
        raw = yaml.safe_load(path.read_text())
        for i, ex in enumerate(raw.get("examples") or []):
            yield f"{path.relative_to(SPECS)} example {i}", ex


@pytest.mark.parametrize(
    "where,ex", list(_examples()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_example_evidence_is_verbatim(where, ex):
    excerpt = _norm_ws(ex["excerpt"])
    for e in ex["output"]["extractions"]:
        for q in e["evidence"]:
            assert (
                _norm_ws(q) in excerpt
            ), f"{where}: evidence {q!r} is not in the excerpt"


def test_spec_text_not_copied_from_dev_reports():
    files = sorted(INGESTED.glob("*/*.jsonl")) if INGESTED.exists() else []
    if not files:
        pytest.skip("no ingested data under data/ingested")

    index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                for sh in _shingles(d["text"]):
                    index[sh].add(d["doc_id"])

    problems = []
    for path in _spec_files():
        raw = yaml.safe_load(path.read_text())
        hits: dict[str, list[str]] = defaultdict(list)
        for s in _strings(raw):
            for sh in _shingles(s):
                docs = index.get(sh, set())
                if 0 < len(docs) < BOILERPLATE_MIN_DOCS:
                    for d in docs:
                        hits[d].append(" ".join(sh))
        for doc_id, runs in sorted(hits.items()):
            problems.append(
                f"{path.relative_to(SPECS)} shares {len(runs)} word run(s) with "
                f"{doc_id}, e.g. {runs[0]!r}"
            )
    assert not problems, "spec text copied from dev reports:\n" + "\n".join(problems)
