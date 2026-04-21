from unittest.mock import patch
from src.decoder import decode, _prepare_candidates

def test_prepare_candidates_uses_freq_sorting():
    reverse_index = {"y": ["你", "伱", "佻"], "t": ["好", "号", "毫"]}
    unigram = {"你": 0.05, "好": 0.03, "号": 0.006}
    result = _prepare_candidates("yt", reverse_index, unigram, candidate_k=2)
    assert len(result) == 2
    assert result[0][0] == "你"
    assert result[1][0] == "好"

@patch("src.decoder.llm_scorer.score_candidates")
def test_decode_returns_verified_results(mock_score):
    mock_score.side_effect = lambda cands, **kw: [(9, s) for _, s in cands]
    unigram = {"你": 0.05, "好": 0.03, "号": 0.006, "伱": 0.0001, "毫": 0.0001}
    bigram = {"你好": 0.002}
    reverse_index = {"y": ["你", "伱"], "t": ["好", "号", "毫"]}
    results = decode(
        "yt",
        reverse_index=reverse_index,
        unigram_freq=unigram,
        bigram_freq=bigram,
        api_key="test",
        llm_model="test",
        beam_width=10,
        candidate_k=3,
        top_n=2,
    )
    for _, sent in results:
        from src.encoder import encode
        assert encode(sent) == "yt"
