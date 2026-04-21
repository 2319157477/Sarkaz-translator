import os
import sys
import glob
import argparse
import numpy as np
import cv2
import cairosvg


def render_templates(svg_dir: str, height: int, cache_dir: str) -> dict[str, np.ndarray]:
    os.makedirs(cache_dir, exist_ok=True)
    templates = {}

    for svg_path in glob.glob(os.path.join(svg_dir, "*.svg")):
        char_name = os.path.splitext(os.path.basename(svg_path))[0]
        cache_path = os.path.join(cache_dir, f"{char_name}_{height}.png")

        if os.path.exists(cache_path):
            img = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)
        else:
            png_bytes = cairosvg.svg2png(url=svg_path, output_height=height)
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(cache_path, img)

        templates[char_name] = img

    return templates


def preprocess_screenshot(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    inverted = cv2.bitwise_not(binary)
    return binary, inverted


def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def match_characters(
    binary_images: tuple[np.ndarray, np.ndarray],
    templates: dict[str, np.ndarray],
    threshold: float = 0.7,
) -> list[tuple]:
    all_matches = []

    for char_name, tmpl in templates.items():
        th, tw = tmpl.shape
        best_per_pos = {}

        for binary in binary_images:
            if binary.shape[0] < th or binary.shape[1] < tw:
                continue
            result = cv2.matchTemplate(binary, tmpl, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            for y, x in zip(*locations):
                score = result[y, x]
                key = (x // (tw // 2), y // (th // 2))
                if key not in best_per_pos or score > best_per_pos[key][5]:
                    best_per_pos[key] = (x, y, tw, th, char_name, float(score))

        all_matches.extend(best_per_pos.values())

    all_matches.sort(key=lambda m: m[5], reverse=True)
    kept = []
    for match in all_matches:
        box = match[:4]
        if all(_compute_iou(box, k[:4]) < 0.3 for k in kept):
            kept.append(match)

    return kept


def assemble_text(matches: list[tuple], templates: dict[str, np.ndarray]) -> str:
    if not matches:
        return ""

    first_tmpl = next(iter(templates.values()))
    h = first_tmpl.shape[0]
    w = first_tmpl.shape[1]

    sorted_matches = sorted(matches, key=lambda m: (m[1], m[0]))

    lines = []
    current_line = [sorted_matches[0]]

    for match in sorted_matches[1:]:
        if abs(match[1] - current_line[0][1]) > h * 0.5:
            lines.append(current_line)
            current_line = [match]
        else:
            current_line.append(match)
    lines.append(current_line)

    result_lines = []
    for line in lines:
        line.sort(key=lambda m: m[0])
        text = line[0][4]
        for i in range(1, len(line)):
            gap = line[i][0] - (line[i-1][0] + line[i-1][2])
            if gap > w * 1.5:
                text += " "
            text += line[i][4]
        result_lines.append(text)

    return "\n".join(result_lines)


def main():
    parser = argparse.ArgumentParser(description="OCR for custom game font using template matching")
    parser.add_argument("screenshot", help="Path to the game screenshot")
    parser.add_argument("--height", type=int, required=True, help="Template height in pixels (must match text height in screenshot)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Match confidence threshold (0.0-1.0, default: 0.7)")
    parser.add_argument("--svg-dir", default="./output_svgs", help="Path to SVG templates directory")
    parser.add_argument("--cache-dir", default="./templates", help="Path to template cache directory")
    args = parser.parse_args()

    if not os.path.isdir(args.svg_dir):
        print(f"Error: SVG directory not found: {args.svg_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.screenshot):
        print(f"Error: Screenshot not found: {args.screenshot}", file=sys.stderr)
        sys.exit(1)

    templates = render_templates(args.svg_dir, args.height, args.cache_dir)
    if not templates:
        print("Error: No SVG templates found", file=sys.stderr)
        sys.exit(1)

    binary_images = preprocess_screenshot(args.screenshot)
    matches = match_characters(binary_images, templates, args.threshold)
    text = assemble_text(matches, templates)
    print(text)


if __name__ == "__main__":
    main()