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
