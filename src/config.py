BEAM_WIDTH = 2000
CANDIDATE_K = 200
TOP_N = 5
TOP_BEAM_RESULTS = 50

LAMBDA_UNIGRAM = 0.3
LAMBDA_TRIGRAM = 0.5
LAMBDA_FOURGRAM = 0.8

CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
]
PUNCT_RANGES = [
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
]
DIGIT_RANGE = (0x0030, 0x0039)  # ASCII digits 0-9

# LLM
LLM_API_KEY = "sk-88c51ac481f55e1238ca1a4d85b9d06d61998268ea1b26cc44f3fc20e1536d40"
LLM_BASE_URL = "https://timicc.com"
LLM_MODEL = "gpt-5.4"
LLM_TEMPERATURE = 0