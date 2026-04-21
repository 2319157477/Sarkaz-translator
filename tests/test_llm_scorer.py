from unittest.mock import patch, MagicMock
from src.llm_scorer import score_candidates, parse_scores

def test_parse_scores_valid_json():
    raw = "[8, 5, 3, 9, 2]"
    assert parse_scores(raw, count=5) == [8, 5, 3, 9, 2]

def test_parse_scores_extracts_from_text():
    raw = "Here are the scores:\n[7, 4, 6]"
    assert parse_scores(raw, count=3) == [7, 4, 6]

def test_parse_scores_fallback_on_bad_json():
    raw = "not json at all"
    assert parse_scores(raw, count=3) == [5, 5, 5]

@patch("src.llm_scorer._call_llm")
def test_score_candidates_returns_sorted(mock_call):
    mock_call.return_value = "[3, 9, 6]"
    candidates = [(0.0, "你坏了"), (0.0, "我好的"), (0.0, "他来了")]
    result = score_candidates(candidates, api_key="test", model="test")
    assert result[0][1] == "我好的"  # score 9 is highest
    assert result[-1][1] == "你坏了"  # score 3 is lowest
