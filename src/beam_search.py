import math

_LOG_FLOOR = math.log(1e-10)
_LOG_SMOOTH = math.log(1e-6)


def _log_prob(freq, key):
    return math.log(freq[key]) if key in freq else _LOG_FLOOR


def precompute_transition_log(transition):
    """预计算转移矩阵的 log 概率，避免 beam search 内层循环重复 sum"""
    trans_log = {}
    for src, targets in transition.items():
        total = sum(targets.values())
        trans_log[src] = {tgt: math.log(cnt / total) for tgt, cnt in targets.items()}
    return trans_log


def beam_search(candidates_per_pos, bigram_freq, unigram_freq, beam_width=200,
                lambda_unigram=0.3, transition=None, trigram_freq=None,
                fourgram_freq=None, lambda_trigram=0.5, lambda_fourgram=0.8):
    BOS = "\x00"
    beams = [(0.0, BOS, "")]

    trans_log = precompute_transition_log(transition) if transition else None

    for pos, candidates in enumerate(candidates_per_pos):
        new_beams = []
        for score, prev, seq in beams:
            prev_trans = trans_log.get(prev) if trans_log else None
            for ch in candidates:
                if prev == BOS:
                    new_score = score + _log_prob(unigram_freq, ch)
                else:
                    bi_score = _get_transition_score(prev, ch, prev_trans, bigram_freq)
                    uni_score = _log_prob(unigram_freq, ch)
                    new_score = score + bi_score + lambda_unigram * uni_score

                    if trigram_freq and len(seq) >= 1:
                        tri_key = seq[-1] + prev + ch
                        if tri_key in trigram_freq:
                            new_score += lambda_trigram * math.log(trigram_freq[tri_key])

                    if fourgram_freq and len(seq) >= 2:
                        four_key = seq[-2:] + prev + ch
                        if four_key in fourgram_freq:
                            new_score += lambda_fourgram * math.log(fourgram_freq[four_key])

                new_beams.append((new_score, ch, seq + ch))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    return [(score, seq) for score, _, seq in beams]


def _get_transition_score(prev, ch, prev_trans, bigram_freq):
    if prev_trans is not None:
        if ch in prev_trans:
            return prev_trans[ch]
        return _LOG_SMOOTH
    bi_key = prev + ch
    if bi_key in bigram_freq:
        return math.log(bigram_freq[bi_key])
    return _LOG_SMOOTH
