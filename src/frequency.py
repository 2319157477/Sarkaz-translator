import json

def load_freq(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sort_candidates_by_freq(candidates, freq, top_k=50):
    scored = sorted(candidates, key=lambda ch: freq.get(ch, 0.0), reverse=True)
    return scored[:top_k]

def sort_candidates_by_transition(candidates, prev_char, transition, unigram_freq, top_k=50):
    """利用转移矩阵对候选字排序：优先选择在前一个字之后高频出现的字"""
    trans = transition.get(prev_char, {})
    if trans:
        scored = sorted(candidates, key=lambda ch: trans.get(ch, 0), reverse=True)
    else:
        scored = sorted(candidates, key=lambda ch: unigram_freq.get(ch, 0.0), reverse=True)
    return scored[:top_k]
