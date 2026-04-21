from src.beam_search import beam_search

def _mock_data():
    """Tiny controlled scenario: 2-char sentence, 3 candidates per position."""
    candidates_per_pos = [
        ["我", "找", "戈"],  # position 0
        ["好", "号", "毫"],  # position 1
    ]
    unigram = {"我": 0.04, "找": 0.005, "戈": 0.0001, "好": 0.03, "号": 0.006, "毫": 0.0005}
    bigram = {"我好": 0.002, "我号": 0.0001, "找好": 0.0001, "戈好": 0.00001}
    return candidates_per_pos, unigram, bigram

def test_beam_search_returns_sorted_results():
    cands, uni, bi = _mock_data()
    results = beam_search(cands, bi, uni, beam_width=10, lambda_unigram=0.3)
    assert len(results) > 0
    scores = [s for s, _ in results]
    assert scores == sorted(scores, reverse=True)

def test_beam_search_top_result_is_most_natural():
    cands, uni, bi = _mock_data()
    results = beam_search(cands, bi, uni, beam_width=10, lambda_unigram=0.3)
    _, top_sentence = results[0]
    assert top_sentence == "我好"

def test_beam_search_respects_beam_width():
    cands, uni, bi = _mock_data()
    results = beam_search(cands, bi, uni, beam_width=2, lambda_unigram=0.3)
    assert len(results) <= 2
