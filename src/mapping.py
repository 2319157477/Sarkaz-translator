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
