from hnc_extract.evidence import EvidenceIndex

TEXT = "A.  Left lateral tongue;\n excision:  - Invasive “squamous” cell carcinoma.  B. biopsy  C. biopsy"


def test_exact_match_offsets():
    m = EvidenceIndex(TEXT).match("Invasive")
    assert m.found and m.method == "exact"
    assert TEXT[m.start : m.end] == "Invasive"


def test_whitespace_insensitive_maps_to_original_offsets():
    m = EvidenceIndex(TEXT).match("Left lateral tongue; excision:")
    assert m.found and m.method == "whitespace"
    assert TEXT[m.start : m.end] == "Left lateral tongue;\n excision:"


def test_normalized_case_and_quotes():
    m = EvidenceIndex(TEXT).match('invasive "squamous" cell CARCINOMA')
    assert m.found and m.method == "normalized"
    assert TEXT[m.start : m.end] == "Invasive “squamous” cell carcinoma"


def test_ellipsis_and_paraphrase_fail():
    idx = EvidenceIndex(TEXT)
    assert not idx.match("Invasive ... carcinoma").found
    assert not idx.match("Invasive … carcinoma").found
    assert not idx.match("invasive SCC").found
    assert not idx.match("   ").found


def test_multiple_occurrences_nearest_to_anchor():
    idx = EvidenceIndex(TEXT)
    c_pos = TEXT.index("C. biopsy")
    m = idx.match("biopsy", near=c_pos)
    assert m.n_occurrences == 2
    assert m.start == c_pos + 3
    assert idx.match("biopsy").start == TEXT.index("biopsy")
