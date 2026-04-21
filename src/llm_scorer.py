import json
import re
from openai import OpenAI
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, TOP_N


def _call_llm(prompt, api_key=None, model=None, base_url=None):
    client = OpenAI(
        api_key=api_key or LLM_API_KEY,
        base_url=base_url or LLM_BASE_URL,
    )
    resp = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content


def parse_scores(raw, count):
    match = re.search(r"\[[\d\s,]+\]", raw)
    if match:
        try:
            scores = json.loads(match.group())
            if len(scores) == count:
                return scores
        except (json.JSONDecodeError, ValueError):
            pass
    return [5] * count


def score_candidates(candidates, api_key=None, model=None, base_url=None):
    lines = "\n".join(f"{i+1}. {s}" for i, (_, s) in enumerate(candidates))
    prompt = (
        f"对以下{len(candidates)}个中文句子的自然度和通顺度打分（1-10），"
        f"只返回JSON数组格式的整数分数列表，不要其他内容。\n\n{lines}"
    )
    raw = _call_llm(prompt, api_key, model, base_url)
    scores = parse_scores(raw, len(candidates))
    scored = [(llm_s, sent) for (beam_s, sent), llm_s in zip(candidates, scores)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
