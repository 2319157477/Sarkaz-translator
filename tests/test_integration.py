from unittest.mock import patch
from src.mapping import build_reverse_index
from src.encoder import encode
from src.decoder import decode

UNIGRAM = {
    "你": 0.04, "好": 0.03, "我": 0.05, "的": 0.08, "是": 0.04,
    "人": 0.02, "了": 0.03, "在": 0.02, "有": 0.02, "这": 0.01,
    "中": 0.01, "大": 0.01, "来": 0.01, "他": 0.02, "不": 0.02,
    "们": 0.01, "到": 0.01, "说": 0.01, "也": 0.01, "和": 0.01,
}
BIGRAM = {
    "你好": 0.002, "我的": 0.003, "是人": 0.0001, "好的": 0.001,
    "我们": 0.002, "他们": 0.001, "不好": 0.0005, "你的": 0.001,
}

@patch("src.decoder.llm_scorer.score_candidates")
def test_roundtrip_simple(mock_score):
    mock_score.side_effect = lambda cands, **kw: [(9, s) for _, s in cands]
    original = "你好"
    encoded = encode(original)
    index = build_reverse_index()
    results = decode(
        encoded,
        reverse_index=index,
        unigram_freq=UNIGRAM,
        bigram_freq=BIGRAM,
        api_key="test",
        llm_model="test",
        beam_width=50,
        candidate_k=20,
        top_n=5,
    )
    sentences = [s for _, s in results]
    assert original in sentences, f"Expected '{original}' in {sentences}"
