from src.mapping import FORWARD_MAP

def encode_char(ch):
    return FORWARD_MAP[ord(ch) % 56]

def encode(text):
    return "".join(encode_char(ch) for ch in text)

def verify(candidates, encoded_str):
    return [c for c in candidates if encode(c) == encoded_str]
