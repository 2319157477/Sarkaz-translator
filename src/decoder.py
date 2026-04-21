import sys
from src.frequency import sort_candidates_by_freq
from src.beam_search import beam_search
from src.encoder import verify
from src import llm_scorer
from src.config import (BEAM_WIDTH, CANDIDATE_K, TOP_N,
                        LAMBDA_UNIGRAM, LAMBDA_TRIGRAM, LAMBDA_FOURGRAM, TOP_BEAM_RESULTS,
                        PUNCT_RANGES)

def _log(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr)

def _is_punct(ch):
    cp = ord(ch)
    return any(start <= cp <= end for start, end in PUNCT_RANGES)

def _prepare_candidates(encoded_str, reverse_index, unigram_freq, candidate_k=CANDIDATE_K):
    result = []
    for i, letter in enumerate(encoded_str):
        chars = reverse_index.get(letter, [])
        punct = [ch for ch in chars if _is_punct(ch)]
        non_punct = [ch for ch in chars if not _is_punct(ch)]
        filtered = sort_candidates_by_freq(non_punct, unigram_freq, top_k=candidate_k)
        for p in punct:
            if p not in filtered:
                filtered.append(p)
        result.append(filtered)
    return result

def decode(encoded_str, reverse_index, unigram_freq, bigram_freq,
           api_key, llm_model, llm_base_url=None,
           beam_width=BEAM_WIDTH, candidate_k=CANDIDATE_K,
           top_n=TOP_N, lambda_unigram=LAMBDA_UNIGRAM,
           transition=None, trigram_freq=None, fourgram_freq=None):
    _log(f"输入: '{encoded_str}' (长度 {len(encoded_str)})")

    candidates_per_pos = _prepare_candidates(encoded_str, reverse_index, unigram_freq, candidate_k)
    empty = [i for i, c in enumerate(candidates_per_pos) if not c]
    if empty:
        _log(f"警告: 位置 {empty} 无候选字符，解码将失败")
        return []


    return _decode_beam(candidates_per_pos, encoded_str, bigram_freq, unigram_freq,
                        beam_width, lambda_unigram, transition, trigram_freq,
                        fourgram_freq, api_key, llm_model, llm_base_url, top_n)

def _decode_beam(candidates_per_pos, encoded_str, bigram_freq, unigram_freq,
                 beam_width, lambda_unigram, transition, trigram_freq,
                 fourgram_freq, api_key, llm_model, llm_base_url, top_n):
    _log("开始 beam search...")
    beam_results = beam_search(
        candidates_per_pos, bigram_freq, unigram_freq, beam_width, lambda_unigram,
        transition=transition, trigram_freq=trigram_freq,
        fourgram_freq=fourgram_freq,
        lambda_trigram=LAMBDA_TRIGRAM, lambda_fourgram=LAMBDA_FOURGRAM,
    )
    _log(f"beam search 完成: {len(beam_results)} 条结果")
    if beam_results:
        _log(f"最高分: {beam_results[0][0]:.2f} '{beam_results[0][1]}'")

    beam_results = beam_results[:TOP_BEAM_RESULTS]
    verified = [(s, sent) for s, sent in beam_results if sent in verify([sent], encoded_str)]
    _log(f"验证通过: {len(verified)}/{len(beam_results)}")
    if not verified:
        return []

    scored = llm_scorer.score_candidates(verified, api_key=api_key, model=llm_model, base_url=llm_base_url)
    return scored[:top_n]
