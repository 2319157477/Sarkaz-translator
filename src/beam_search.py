import math

_LOG_FLOOR = math.log(1e-10)

def beam_search(candidates_per_pos, bigram_freq, unigram_freq, beam_width=200, lambda_unigram=0.3):
    BOS = "\x00"
    beams = [(0.0, BOS, "")]

    for candidates in candidates_per_pos:
        new_beams = []
        for score, prev, seq in beams:
            for ch in candidates:
                if prev == BOS:
                    bi_score = math.log(unigram_freq[ch]) if ch in unigram_freq else _LOG_FLOOR
                else:
                    bi_key = prev + ch
                    bi_score = math.log(bigram_freq[bi_key]) if bi_key in bigram_freq else _LOG_FLOOR
                uni_score = math.log(unigram_freq[ch]) if ch in unigram_freq else _LOG_FLOOR
                new_score = score + bi_score + lambda_unigram * uni_score
                new_beams.append((new_score, ch, seq + ch))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    return [(score, seq) for score, _, seq in beams]
