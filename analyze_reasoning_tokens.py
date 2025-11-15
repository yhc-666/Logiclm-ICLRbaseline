import argparse
import json
import os
import re
from typing import List, Optional


def try_load_tiktoken(model_name: str):
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        # fall back to a common encoding
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, tokenizer=None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    # simple fallback: whitespace-like token count
    return len(re.findall(r"\S+", text))


def iter_result_files(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.endswith(".json"):
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[WARN] Path not found or unsupported: {p}")
    return files


def analyze_file(path: str, tokenizer=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_tokens = 0
    count = 0

    for sample in data:
        reasoning = sample.get("predicted_reasoning", "") or ""
        total_tokens += count_tokens(reasoning, tokenizer=tokenizer)
        count += 1

    avg = total_tokens / count if count > 0 else 0.0
    return count, total_tokens, avg


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute average token count of 'predicted_reasoning' "
            "for one or more baseline result JSON files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="File or directory paths under baselines/results (JSON files).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4",
        help="Model name for tiktoken encoding (if available).",
    )
    parser.add_argument(
        "--no_tiktoken",
        action="store_true",
        help="Do not use tiktoken even if installed; use simple whitespace tokens.",
    )
    args = parser.parse_args()

    tokenizer = None
    if not args.no_tiktoken:
        tokenizer = try_load_tiktoken(args.model_name)
        if tokenizer is None:
            print("[INFO] tiktoken not available or encoding not found; using whitespace token count.")

    files = iter_result_files(args.paths)
    if not files:
        print("[ERROR] No JSON result files found in the given paths.")
        return

    overall_tokens = 0
    overall_count = 0

    for path in files:
        count, total_tokens, avg = analyze_file(path, tokenizer=tokenizer)
        overall_tokens += total_tokens
        overall_count += count
        print(f"{path}:")
        print(f"  samples           : {count}")
        print(f"  total tokens      : {total_tokens}")
        print(f"  avg tokens/sample : {avg:.2f}")

    if len(files) > 1:
        overall_avg = overall_tokens / overall_count if overall_count > 0 else 0.0
        print("Overall:")
        print(f"  samples           : {overall_count}")
        print(f"  total tokens      : {overall_tokens}")
        print(f"  avg tokens/sample : {overall_avg:.2f}")


if __name__ == "__main__":
    main()

