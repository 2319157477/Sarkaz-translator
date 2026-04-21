"""从 tableCfg 目录下所有 JSON 文件中提取 "cn" 字段的值，汇总输出。"""

import json
import os
import sys


def find_cn_values(obj):
    """递归查找所有 key 为 'cn' 的字符串值。"""
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "cn" and isinstance(v, str) and v.strip():
                results.append(v)
            else:
                results.extend(find_cn_values(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_cn_values(item))
    return results


def main():
    table_dir = os.path.join(os.path.dirname(__file__), "..", "tableCfg")
    table_dir = os.path.normpath(table_dir)

    if not os.path.isdir(table_dir):
        print(f"目录不存在: {table_dir}", file=sys.stderr)
        sys.exit(1)

    all_texts = []
    file_count = 0

    for fname in sorted(os.listdir(table_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(table_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"跳过 {fname}: {e}", file=sys.stderr)
            continue

        texts = find_cn_values(data)
        if texts:
            all_texts.extend(texts)
            file_count += 1

    unique_texts = sorted(set(all_texts))

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "cn_texts.json")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unique_texts, f, ensure_ascii=False, indent=2)

    print(f"从 {file_count} 个文件中提取了 {len(all_texts)} 条文本（去重后 {len(unique_texts)} 条）")
    print(f"已保存到 {out_path}")


if __name__ == "__main__":
    main()
