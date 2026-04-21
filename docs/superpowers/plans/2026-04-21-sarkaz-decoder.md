# Sarkaz Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python decoder that reverses mod-56 encoded English strings back into Chinese sentences using N-gram beam search + LLM reranking.

**Architecture:** Three-layer pipeline — data layer (reverse index + frequency tables), beam search decoder (bigram-scored path expansion), and LLM scorer (batch reranking + verification). Each layer is a separate module with a clean interface.

**Tech Stack:** Python 3.10+, `openai` or `anthropic` SDK for LLM API, `pytest` for testing, JSON for data storage.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/mapping.py` | Forward/reverse mapping tables, reverse index builder |
| `src/encoder.py` | Encode Chinese → English string (for verification) |
| `src/frequency.py` | Load unigram + bigram frequency tables, candidate filtering |
| `src/beam_search.py` | Beam search decoder with bigram scoring |
| `src/llm_scorer.py` | LLM API batch scoring wrapper |
| `src/decoder.py` | Main pipeline orchestration |
| `src/config.py` | Configuration constants |
| `data/unigram_freq.json` | Character frequency data |
| `data/bigram_freq.json` | Character bigram frequency data |
| `scripts/build_frequency_data.py` | Script to generate frequency JSON files from corpus |
| `tests/test_mapping.py` | Tests for mapping module |
| `tests/test_encoder.py` | Tests for encoder module |
| `tests/test_frequency.py` | Tests for frequency module |
| `tests/test_beam_search.py` | Tests for beam search |
| `tests/test_llm_scorer.py` | Tests for LLM scorer |
| `tests/test_decoder.py` | Integration tests for full pipeline |

---

### Task 1: Project Setup + Config

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create project structure**

```bash
mkdir -p src tests data scripts
```

- [ ] **Step 2: Write config.py**

```python
BEAM_WIDTH = 200
CANDIDATE_K = 50
TOP_N = 5
LAMBDA_UNIGRAM = 0.3
TOP_BEAM_RESULTS = 50

CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
]
PUNCT_RANGES = [
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
]
DIGIT_RANGE = (0x0030, 0x0039)  # ASCII digits 0-9
```

- [ ] **Step 3: Write requirements.txt**

```
openai>=1.0.0
pytest>=7.0.0
```

- [ ] **Step 4: Create __init__.py files**

Empty files for `src/__init__.py` and `tests/__init__.py`.

- [ ] **Step 5: Commit**

```bash
git init
git add src/ tests/ requirements.txt
git commit -m "chore: project scaffold with config"
```

---

### Task 2: Mapping Module (Forward + Reverse)

**Files:**
- Create: `src/mapping.py`
- Create: `tests/test_mapping.py`

- [ ] **Step 1: Write failing tests for mapping**

```python
# tests/test_mapping.py
from src.mapping import FORWARD_MAP, build_reverse_letter_to_remainders, build_reverse_index
from src.config import CJK_RANGES

def test_forward_map_has_56_entries():
    assert len(FORWARD_MAP) == 56
    assert FORWARD_MAP[0] == "g"
    assert FORWARD_MAP[55] == "e"

def test_reverse_letter_to_remainders():
    rev = build_reverse_letter_to_remainders()
    assert 2 in rev["a"] and 38 in rev["a"]
    assert 25 in rev["e"] and 31 in rev["e"] and 55 in rev["e"]
    assert 12 in rev["f"] and len(rev["f"]) == 1

def test_reverse_index_returns_chars_matching_mod56():
    index = build_reverse_index()
    for letter, chars in index.items():
        assert len(chars) > 0
        rev = build_reverse_letter_to_remainders()
        for ch in chars[:5]:
            assert ord(ch) % 56 in rev[letter]

def test_reverse_index_covers_cjk():
    index = build_reverse_index()
    total = sum(len(v) for v in index.values())
    assert total > 5000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mapping.py -v`
Expected: FAIL — `src.mapping` not found

- [ ] **Step 3: Implement mapping.py**

```python
# src/mapping.py
from src.config import CJK_RANGES, PUNCT_RANGES, DIGIT_RANGE

FORWARD_MAP = {
    0: "g", 1: "k", 2: "a", 3: "m", 4: "z", 5: "t", 6: "l", 7: "b",
    8: "d", 9: "q", 10: "i", 11: "y", 12: "f", 13: "u", 14: "c", 15: "x",
    16: "b", 17: "h", 18: "s", 19: "j", 20: "o", 21: "p", 22: "r", 23: "n",
    24: "w", 25: "e", 26: "y", 27: "g", 28: "t", 29: "j", 30: "m", 31: "e",
    32: "v", 33: "c", 34: "h", 35: "d", 36: "x", 37: "s", 38: "a", 39: "n",
    40: "q", 41: "o", 42: "l", 43: "k", 44: "r", 45: "v", 46: "w", 47: "i",
    48: "y", 49: "p", 50: "j", 51: "z", 52: "q", 53: "u", 54: "h", 55: "e",
}

def build_reverse_letter_to_remainders():
    rev = {}
    for remainder, letter in FORWARD_MAP.items():
        rev.setdefault(letter, []).append(remainder)
    return rev

def build_reverse_index():
    letter_to_remainders = build_reverse_letter_to_remainders()
    remainder_to_chars = {}
    all_ranges = CJK_RANGES + PUNCT_RANGES + [(DIGIT_RANGE[0], DIGIT_RANGE[1])]
    for start, end in all_ranges:
        for cp in range(start, end + 1):
            r = cp % 56
            remainder_to_chars.setdefault(r, []).append(chr(cp))
    index = {}
    for letter, remainders in letter_to_remainders.items():
        chars = []
        for r in remainders:
            chars.extend(remainder_to_chars.get(r, []))
        index[letter] = chars
    return index
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mapping.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mapping.py tests/test_mapping.py
git commit -m "feat: forward/reverse mapping tables with reverse index"
```

---

### Task 3: Encoder Module (for verification)

**Files:**
- Create: `src/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_encoder.py
from src.encoder import encode_char, encode, verify

def test_encode_char_ni():
    # 你 U+4F60 = 20320, 20320 % 56 = 48, FORWARD_MAP[48] = 'y'
    assert encode_char("你") == "y"

def test_encode_char_hao():
    # 好 U+597D = 22909, 22909 % 56 = 5, FORWARD_MAP[5] = 't'
    assert encode_char("好") == "t"

def test_encode_char_wo():
    # 我 U+6211 = 25105, 25105 % 56 = 17, FORWARD_MAP[17] = 'h'
    assert encode_char("我") == "h"

def test_encode_sentence():
    assert encode("你好") == "yt"

def test_encode_with_punctuation():
    # 。 U+3002 = 12290, 12290 % 56 = 26, FORWARD_MAP[26] = 'y'
    assert encode("你好。") == "yty"

def test_verify_filters_invalid():
    encoded = encode("你好")
    assert "你好" in verify(["你好", "伱奻"], encoded)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_encoder.py -v`
Expected: FAIL — cannot import `encode_char`

- [ ] **Step 3: Implement encoder.py**

```python
# src/encoder.py
from src.mapping import FORWARD_MAP

def encode_char(ch):
    return FORWARD_MAP[ord(ch) % 56]

def encode(text):
    return "".join(encode_char(ch) for ch in text)

def verify(candidates, encoded_str):
    return [c for c in candidates if encode(c) == encoded_str]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_encoder.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/encoder.py tests/test_encoder.py
git commit -m "feat: encoder with char-level encoding and verification"
```

---

### Task 4: Frequency Module

**Files:**
- Create: `src/frequency.py`
- Create: `tests/test_frequency.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_frequency.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_frequency.py -v`
Expected: FAIL

- [ ] **Step 3: Implement frequency.py**

```python
# src/frequency.py
import json

def load_freq(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sort_candidates_by_freq(candidates, freq, top_k=50):
    scored = sorted(candidates, key=lambda ch: freq.get(ch, 0.0), reverse=True)
    return scored[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_frequency.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/frequency.py tests/test_frequency.py
git commit -m "feat: frequency loading and candidate sorting"
```

---

### Task 5: Beam Search Decoder

**Files:**
- Create: `src/beam_search.py`
- Create: `tests/test_beam_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_beam_search.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_beam_search.py -v`
Expected: FAIL

- [ ] **Step 3: Implement beam_search.py**

```python
# src/beam_search.py
import math

_LOG_FLOOR = math.log(1e-10)

def beam_search(candidates_per_pos, bigram_freq, unigram_freq, beam_width=200, lambda_unigram=0.3):
    BOS = "\x00"
    beams = [(0.0, BOS, "")]

    for candidates in candidates_per_pos:
        new_beams = []
        for score, prev, seq in beams:
            for ch in candidates:
                if prev == BOS:
                    bi_score = math.log(unigram_freq[ch]) if ch in unigram_freq else _LOG_FLOOR
                else:
                    bi_key = prev + ch
                    bi_score = math.log(bigram_freq[bi_key]) if bi_key in bigram_freq else _LOG_FLOOR
                uni_score = math.log(unigram_freq[ch]) if ch in unigram_freq else _LOG_FLOOR
                new_score = score + bi_score + lambda_unigram * uni_score
                new_beams.append((new_score, ch, seq + ch))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    return [(score, seq) for score, _, seq in beams]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_beam_search.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/beam_search.py tests/test_beam_search.py
git commit -m "feat: beam search decoder with bigram scoring"
```

---

### Task 6: LLM Scorer

**Files:**
- Create: `src/llm_scorer.py`
- Create: `tests/test_llm_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_scorer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement llm_scorer.py**

```python
# src/llm_scorer.py
import json
import re
from openai import OpenAI

def _call_llm(prompt, api_key, model, base_url=None):
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

def parse_scores(raw, count):
    match = re.search(r"\[[\d\s,]+\]", raw)
    if match:
        try:
            scores = json.loads(match.group())
            if len(scores) == count:
                return scores
        except (json.JSONDecodeError, ValueError):
            pass
    return [5] * count

def score_candidates(candidates, api_key, model, base_url=None):
    lines = "\n".join(f"{i+1}. {s}" for i, (_, s) in enumerate(candidates))
    prompt = (
        f"对以下{len(candidates)}个中文句子的自然度和通顺度打分（1-10），"
        f"只返回JSON数组格式的整数分数列表，不要其他内容。\n\n{lines}"
    )
    raw = _call_llm(prompt, api_key, model, base_url)
    scores = parse_scores(raw, len(candidates))
    scored = [(llm_s, sent) for (beam_s, sent), llm_s in zip(candidates, scores)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_scorer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_scorer.py tests/test_llm_scorer.py
git commit -m "feat: LLM scorer with batch scoring and JSON parsing"
```

---

### Task 7: Decoder Pipeline

**Files:**
- Create: `src/decoder.py`
- Create: `tests/test_decoder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_decoder.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decoder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement decoder.py**

```python
# src/decoder.py
from src.frequency import sort_candidates_by_freq
from src.beam_search import beam_search
from src.encoder import verify
from src import llm_scorer
from src.config import BEAM_WIDTH, CANDIDATE_K, TOP_N, LAMBDA_UNIGRAM, TOP_BEAM_RESULTS

def _prepare_candidates(encoded_str, reverse_index, unigram_freq, candidate_k=CANDIDATE_K):
    result = []
    for letter in encoded_str:
        chars = reverse_index.get(letter, [])
        filtered = sort_candidates_by_freq(chars, unigram_freq, top_k=candidate_k)
        result.append(filtered)
    return result

def decode(encoded_str, reverse_index, unigram_freq, bigram_freq,
           api_key, llm_model, llm_base_url=None,
           beam_width=BEAM_WIDTH, candidate_k=CANDIDATE_K,
           top_n=TOP_N, lambda_unigram=LAMBDA_UNIGRAM):
    candidates_per_pos = _prepare_candidates(encoded_str, reverse_index, unigram_freq, candidate_k)
    beam_results = beam_search(candidates_per_pos, bigram_freq, unigram_freq, beam_width, lambda_unigram)
    beam_results = beam_results[:TOP_BEAM_RESULTS]
    verified = [(s, sent) for s, sent in beam_results if sent in verify([sent], encoded_str)]
    scored = llm_scorer.score_candidates(verified, api_key=api_key, model=llm_model, base_url=llm_base_url)
    return scored[:top_n]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_decoder.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/decoder.py tests/test_decoder.py
git commit -m "feat: decoder pipeline with beam search + LLM scoring"
```

---

### Task 8: Frequency Data Generation Script

**Files:**
- Create: `scripts/build_frequency_data.py`

- [ ] **Step 1: Write the script**

```python
# scripts/build_frequency_data.py
"""
Generate unigram and bigram frequency JSON files from a Chinese text corpus.

Usage:
    python scripts/build_frequency_data.py <corpus_path> [--out-dir data]

The corpus file should be a plain text file (UTF-8) containing Chinese text.
Any Chinese text works: novels, news articles, Wikipedia dumps, etc.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef0-9]")

def extract_chars(text):
    return CJK_PATTERN.findall(text)

def build_frequencies(corpus_path):
    text = Path(corpus_path).read_text(encoding="utf-8")
    chars = extract_chars(text)
    total = len(chars)
    if total == 0:
        raise ValueError("No CJK characters found in corpus")

    unigram_counts = Counter(chars)
    unigram_freq = {ch: count / total for ch, count in unigram_counts.items()}

    bigram_counts = Counter()
    for i in range(len(chars) - 1):
        bigram_counts[chars[i] + chars[i + 1]] += 1
    bigram_total = sum(bigram_counts.values())
    bigram_freq = {bg: count / bigram_total for bg, count in bigram_counts.items()}

    return unigram_freq, bigram_freq

def main():
    parser = argparse.ArgumentParser(description="Build frequency data from Chinese corpus")
    parser.add_argument("corpus", help="Path to Chinese text corpus file")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    unigram, bigram = build_frequencies(args.corpus)

    with open(out / "unigram_freq.json", "w", encoding="utf-8") as f:
        json.dump(unigram, f, ensure_ascii=False, indent=0)
    with open(out / "bigram_freq.json", "w", encoding="utf-8") as f:
        json.dump(bigram, f, ensure_ascii=False, indent=0)

    print(f"Unigram: {len(unigram)} chars, Bigram: {len(bigram)} pairs")
    print(f"Saved to {out}/")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test manually with a small corpus**

Create a small test corpus and run:

```bash
echo "我是中国人。你好世界，我们的生活很好。" > data/test_corpus.txt
python scripts/build_frequency_data.py data/test_corpus.txt --out-dir data
```

Expected: prints char/pair counts, creates `data/unigram_freq.json` and `data/bigram_freq.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_frequency_data.py
git commit -m "feat: frequency data generation script"
```

---

### Task 9: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test (LLM mocked)**

```python
# tests/test_integration.py
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
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test with roundtrip verification"
```
