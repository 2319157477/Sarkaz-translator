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
