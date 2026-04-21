from src.frequency import sort_candidates_by_freq
from src.beam_search import beam_search
from src.encoder import verify
from src import llm_scorer
from src.config import BEAM_WIDTH, CANDIDATE_K, TOP_N, LAMBDA_UNIGRAM, TOP_BEAM_RESULTS

def _prepare_candidates(encoded_str, reverse_index, unigram_freq, candidate_k=CANDIDATE_K):
    result = []
    for letter in encoded_str:
        chars = reverse_index.get(letter, [])
        filtered = sort_candidates_by_freq(chars, unigram_freq, top_k=candidate_k)
        result.append(filtered)
    return result

def decode(encoded_str, reverse_index, unigram_freq, bigram_freq,
           api_key, llm_model, llm_base_url=None,
           beam_width=BEAM_WIDTH, candidate_k=CANDIDATE_K,
           top_n=TOP_N, lambda_unigram=LAMBDA_UNIGRAM):
    candidates_per_pos = _prepare_candidates(encoded_str, reverse_index, unigram_freq, candidate_k)
    beam_results = beam_search(candidates_per_pos, bigram_freq, unigram_freq, beam_width, lambda_unigram)
    beam_results = beam_results[:TOP_BEAM_RESULTS]
    verified = [(s, sent) for s, sent in beam_results if sent in verify([sent], encoded_str)]
    scored = llm_scorer.score_candidates(verified, api_key=api_key, model=llm_model, base_url=llm_base_url)
    return scored[:top_n]
