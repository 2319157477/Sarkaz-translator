"""
Sarkaz Decoder - 将 mod-56 编码的英文字符串还原为中文句子

Usage:
    python main.py <encoded_string>
    python main.py -i              # 交互模式
"""
import argparse
import sys
from src.config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
                        BEAM_WIDTH, CANDIDATE_K, TOP_N, TOP_BEAM_RESULTS,
                        LAMBDA_UNIGRAM, LAMBDA_TRIGRAM, LAMBDA_FOURGRAM)
from src.mapping import build_reverse_index
from src.frequency import load_freq
from src.decoder import decode


def run(encoded_str, reverse_index, unigram, bigram, transition, trigram, fourgram):
    encoded_str = encoded_str.lower()
    results = decode(
        encoded_str, reverse_index, unigram, bigram,
        api_key=LLM_API_KEY, llm_model=LLM_MODEL, llm_base_url=LLM_BASE_URL,
        transition=transition, trigram_freq=trigram, fourgram_freq=fourgram,
    )
    for i, (score, sent) in enumerate(results, 1):
        print(f"  {i}. [{score}] {sent}")


def main():
    parser = argparse.ArgumentParser(description="Sarkaz Decoder")
    parser.add_argument("encoded", nargs="?", help="编码后的英文字符串")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    reverse_index = build_reverse_index()
    unigram = load_freq("data/game_unigram_freq.json")
    bigram = load_freq("data/game_bigram_freq.json")
    transition = load_freq("data/game_transition.json")
    trigram = load_freq("data/game_trigram_freq.json")
    fourgram = load_freq("data/game_fourgram_freq.json")

    print(f"[CONFIG] beam_width={BEAM_WIDTH}, candidate_k={CANDIDATE_K}, "
          f"top_n={TOP_N}, top_beam_results={TOP_BEAM_RESULTS}")
    print(f"[CONFIG] lambda: unigram={LAMBDA_UNIGRAM}, trigram={LAMBDA_TRIGRAM}, "
          f"fourgram={LAMBDA_FOURGRAM}")
    print(f"[DATA] unigram={len(unigram)}, bigram={len(bigram)}, "
          f"trigram={len(trigram)}, fourgram={len(fourgram)}, "
          f"transition={len(transition)} sources")

    if args.interactive:
        print("Sarkaz Decoder 交互模式 (输入 q 退出)")
        while True:
            try:
                s = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if s.lower() == "q":
                break
            if s:
                run(s, reverse_index, unigram, bigram, transition, trigram, fourgram)
    elif args.encoded:
        run(args.encoded, reverse_index, unigram, bigram, transition, trigram, fourgram)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
