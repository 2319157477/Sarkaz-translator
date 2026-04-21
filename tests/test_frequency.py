import json
import tempfile
from pathlib import Path
from src.frequency import load_freq, sort_candidates_by_freq

def test_load_freq_from_json():
    data = {"你": 0.05, "好": 0.03, "的": 0.08}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        path = f.name
    result = load_freq(path)
    assert result["你"] == 0.05
    assert result["的"] == 0.08
    Path(path).unlink()

def test_sort_candidates_by_freq():
    candidates = ["伱", "你", "佻"]
    freq = {"你": 0.05, "佻": 0.001}
    result = sort_candidates_by_freq(candidates, freq, top_k=2)
    assert result[0] == "你"
    assert len(result) == 2

def test_sort_candidates_unknown_chars_last():
    candidates = ["甲", "乙", "丙"]
    freq = {"乙": 0.01}
    result = sort_candidates_by_freq(candidates, freq, top_k=3)
    assert result[0] == "乙"
