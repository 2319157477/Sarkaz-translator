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
