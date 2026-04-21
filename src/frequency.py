import json

def load_freq(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sort_candidates_by_freq(candidates, freq, top_k=50):
    scored = sorted(candidates, key=lambda ch: freq.get(ch, 0.0), reverse=True)
    return scored[:top_k]
