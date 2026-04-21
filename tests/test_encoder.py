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
