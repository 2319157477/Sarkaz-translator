import os
import numpy as np
import cv2
import gradio as gr

from src.mapping import build_reverse_index
from src.frequency import load_freq
from src.encoder import encode
from src.decoder import decode
from src.ocr import render_templates, preprocess_image_array, match_characters, assemble_text
from src.config import (
    BEAM_WIDTH, CANDIDATE_K, TOP_N, TOP_BEAM_RESULTS,
    LAMBDA_UNIGRAM, LAMBDA_TRIGRAM, LAMBDA_FOURGRAM,
    CJK_RANGES, PUNCT_RANGES, DIGIT_RANGE,
)

# --- Data loading (once at startup) ---
reverse_index = build_reverse_index()
unigram = load_freq("data/game_unigram_freq.json")
bigram = load_freq("data/game_bigram_freq.json")
transition = load_freq("data/game_transition.json")
trigram = load_freq("data/game_trigram_freq.json")
fourgram = load_freq("data/game_fourgram_freq.json")
ocr_templates = render_templates("EndfieldFonts", 50, "templates")

SVG_DIR = "EndfieldFonts"


# --- Encode helpers ---

def render_letter(letter: str, height: int = 64) -> np.ndarray | None:
    png_path = os.path.join(SVG_DIR, f"{letter.upper()}.png")
    if not os.path.exists(png_path):
        return None
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if height != img.shape[0]:
        img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
    return img


def render_encoded_text(encoded: str, height: int = 64, spacing: int = 4) -> np.ndarray | None:
    images = []
    for ch in encoded:
        if ch == " ":
            images.append(np.zeros((height, height // 2, 4), dtype=np.uint8))
            continue
        img = render_letter(ch, height)
        if img is None:
            continue
        images.append(img)
    if not images:
        return None
    max_h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] < max_h:
            pad = np.zeros((max_h - im.shape[0], im.shape[1], im.shape[2]), dtype=np.uint8)
            im = np.vstack([pad, im])
        padded.append(im)
    spacer = np.zeros((max_h, spacing, 4), dtype=np.uint8)
    parts = []
    for i, im in enumerate(padded):
        if i > 0:
            parts.append(spacer)
        parts.append(im)
    return np.hstack(parts)


def _is_encodable(ch: str) -> bool:
    cp = ord(ch)
    for start, end in CJK_RANGES + PUNCT_RANGES:
        if start <= cp <= end:
            return True
    if DIGIT_RANGE[0] <= cp <= DIGIT_RANGE[1]:
        return True
    return False


def do_encode(text: str):
    if not text.strip():
        return "请输入中文语句", None
    bad_chars = [ch for ch in text if not _is_encodable(ch)]
    if bad_chars:
        unique = list(dict.fromkeys(bad_chars))
        return f"以下字符无法编码: {''.join(unique)}", None
    encoded = encode(text)
    img = render_encoded_text(encoded, height=64)
    return encoded, img


# --- Decode helpers ---

def do_decode(text, use_llm, api_key, base_url, model,
              beam_width, candidate_k, top_n, top_beam_results,
              lam_uni, lam_tri, lam_four):
    if not text.strip():
        return "请输入编码字符串"
    if use_llm and not api_key.strip():
        return "已启用 LLM 打分，请填写 API Key"

    lines = [l.strip().lower() for l in text.strip().split("\n") if l.strip()]
    output_parts = []

    for i, line in enumerate(lines):
        invalid = [ch for ch in line if ch not in "abcdefghijklmnopqrstuvwxyz"]
        if invalid:
            output_parts.append(f"**第 {i+1} 行** `{line}`: 包含无效字符 {''.join(set(invalid))}，已跳过\n")
            continue

        results = decode(
            line, reverse_index, unigram, bigram,
            api_key=api_key.strip() if use_llm else None,
            llm_model=model.strip() if use_llm else None,
            llm_base_url=base_url.strip() if use_llm and base_url.strip() else None,
            beam_width=int(beam_width),
            candidate_k=int(candidate_k),
            top_n=int(top_n),
            top_beam_results=int(top_beam_results),
            lambda_unigram=float(lam_uni),
            lambda_trigram=float(lam_tri),
            lambda_fourgram=float(lam_four),
            transition=transition,
            trigram_freq=trigram,
            fourgram_freq=fourgram,
        )

        if not results:
            output_parts.append(f"**第 {i+1} 行** `{line}`: 无候选结果\n")
            continue

        output_parts.append(f"**第 {i+1} 行** `{line}`:\n")
        output_parts.append("| 排名 | 分数 | 候选句子 |")
        output_parts.append("|------|------|----------|")
        for rank, (score, sent) in enumerate(results, 1):
            output_parts.append(f"| {rank} | {score} | {sent} |")
        output_parts.append("")

    return "\n".join(output_parts) if output_parts else "无结果"


# --- UI Layout ---

with gr.Blocks(title="Sarkaz Decoder") as demo:
    gr.Markdown("# Sarkaz Decoder")

    with gr.Tabs():
        # === Decode Tab ===
        with gr.Tab("解码 (Decode)"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs() as input_tabs:
                        with gr.Tab("直接输入"):
                            decode_input = gr.Textbox(
                                label="编码字符串",
                                placeholder="输入编码后的英文字母，支持多行",
                                lines=3,
                            )

                    with gr.Row():
                        use_llm = gr.Checkbox(label="启用 LLM 打分", value=False)
                    with gr.Column(visible=False) as llm_config:
                        llm_api_key = gr.Textbox(label="API Key", type="password")
                        llm_base_url = gr.Textbox(label="Base URL", placeholder="https://api.openai.com")
                        llm_model = gr.Textbox(label="Model", placeholder="gpt-4")

                    decode_btn = gr.Button("解码", variant="primary")

                    with gr.Accordion("高级设置", open=False):
                        adv_beam_width = gr.Number(label="BEAM_WIDTH", value=BEAM_WIDTH)
                        adv_candidate_k = gr.Number(label="CANDIDATE_K", value=CANDIDATE_K)
                        adv_top_n = gr.Number(label="TOP_N", value=TOP_N)
                        adv_top_beam = gr.Number(label="TOP_BEAM_RESULTS", value=TOP_BEAM_RESULTS)
                        adv_lam_uni = gr.Slider(0, 2, value=LAMBDA_UNIGRAM, step=0.1, label="LAMBDA_UNIGRAM")
                        adv_lam_tri = gr.Slider(0, 2, value=LAMBDA_TRIGRAM, step=0.1, label="LAMBDA_TRIGRAM")
                        adv_lam_four = gr.Slider(0, 2, value=LAMBDA_FOURGRAM, step=0.1, label="LAMBDA_FOURGRAM")

                with gr.Column(scale=1):
                    decode_output = gr.Markdown(label="解码结果")

            # Events
            use_llm.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[use_llm],
                outputs=[llm_config],
            )

        # === Encode Tab ===
        with gr.Tab("编码 (Encode)"):
            with gr.Row():
                with gr.Column(scale=1):
                    encode_input = gr.Textbox(
                        label="中文语句",
                        placeholder="输入要编码的中文语句",
                        lines=2,
                    )
                    encode_btn = gr.Button("编码", variant="primary")
                with gr.Column(scale=1):
                    encode_text_output = gr.Textbox(label="编码结果", interactive=False)
                    encode_img_output = gr.Image(label="游戏字符图片", type="numpy")

            encode_btn.click(
                fn=do_encode,
                inputs=[encode_input],
                outputs=[encode_text_output, encode_img_output],
            )


if __name__ == "__main__":
    demo.launch()
