"""从 cn_texts.json 统计中文单字频率、二元频率、三元频率和转移矩阵。

输出:
  data/game_unigram_freq.json   - {"字": prob, ...}
  data/game_bigram_freq.json    - {"字字": prob, ...}
  data/game_trigram_freq.json   - {"字字字": prob, ...}
  data/game_fourgram_freq.json  - {"字字字字": prob, ...}  (top 100)
  data/game_transition.json     - {"字": {"字": count, ...}, ...}
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
INPUT_PATH = os.path.join(DATA_DIR, "cn_texts.json")

# 匹配连续中文字符+标点片段（包含CJK标点和全角符号）
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+")


def extract_cjk_runs(text):
    """提取文本中所有连续中文+标点片段。"""
    return CJK_RE.findall(text)


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        texts = json.load(f)

    uni_counter = Counter()
    bi_counter = Counter()
    tri_counter = Counter()
    four_counter = Counter()
    transition = defaultdict(Counter)

    for text in texts:
        for run in extract_cjk_runs(text):
            for i, ch in enumerate(run):
                uni_counter[ch] += 1
                if i + 1 < len(run):
                    bi_counter[run[i : i + 2]] += 1
                    transition[ch][run[i + 1]] += 1
                if i + 2 < len(run):
                    tri_counter[run[i : i + 3]] += 1
                if i + 3 < len(run):
                    four_counter[run[i : i + 4]] += 1

    uni_total = sum(uni_counter.values())
    bi_total = sum(bi_counter.values())
    tri_total = sum(tri_counter.values())
    four_total = sum(four_counter.values())

    uni_freq = {ch: c / uni_total for ch, c in uni_counter.most_common()}
    bi_freq = {bg: c / bi_total for bg, c in bi_counter.most_common(1000)}
    tri_freq = {tg: c / tri_total for tg, c in tri_counter.most_common(1000)}
    four_freq = {fg: c / four_total for fg, c in four_counter.most_common(100)}
    trans = {ch: dict(nexts.most_common()) for ch, nexts in sorted(transition.items())}

    def save(name, data):
        path = os.path.join(DATA_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  {name}: {len(data)} 条")

    print(f"语料: {len(texts)} 条文本, {uni_total} 字符")
    save("game_unigram_freq.json", uni_freq)
    save("game_bigram_freq.json", bi_freq)
    save("game_trigram_freq.json", tri_freq)
    save("game_fourgram_freq.json", four_freq)
    save("game_transition.json", trans)


if __name__ == "__main__":
    main()
