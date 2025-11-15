import json
import os
from typing import List, Dict, Any

CHOICE_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def canonicalize_dataset_name(name: str) -> str:
    normalized = name.lower().replace("_", "-")
    if normalized in {"chinese-logicqa", "chineselogicqa"}:
        return "chinese-logicqa"
    return name


def _format_options(options: List[str]) -> List[str]:
    formatted = []
    for idx, opt in enumerate(options):
        prefix = CHOICE_SYMBOLS[idx]
        formatted.append(f"{prefix}) {opt.strip()}")
    return formatted


def _load_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _standardize_chinese_logicqa_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    example_id = sample.get("example_id", sample.get("id"))
    idx = sample.get("answer", 0)
    options = sample.get("options", [])
    formatted_options = _format_options(options)
    answer = CHOICE_SYMBOLS[idx]
    return {
        "id": str(example_id),
        "context": sample.get("text", sample.get("context", "")),
        "question": sample.get("question", ""),
        "options": formatted_options,
        "answer": answer,
        "answer_index": idx,
    }


def _load_chinese_logicqa(dataset_dir: str, split: str) -> List[Dict[str, Any]]:
    split_candidates = [split]
    if split == "test":
        split_candidates.append("test_zh")
    if split == "test_zh":
        split_candidates.append("test")
    split_candidates.append("test_zh")

    data_path = None
    loader = None
    for candidate in split_candidates:
        for ext, loader_fn in (
            (".json", _load_json_file),
            (".jsonl", _load_jsonl_file),
            (".txt", _load_jsonl_file),
        ):
            path = os.path.join(dataset_dir, f"{candidate}{ext}")
            if os.path.exists(path):
                data_path = path
                loader = loader_fn
                break
        if data_path:
            break

    if data_path is None or loader is None:
        raise FileNotFoundError(
            f"Cannot locate split '{split}' for chinese-logicqa under {dataset_dir}"
        )

    raw_samples = loader(data_path)
    return [_standardize_chinese_logicqa_sample(sample) for sample in raw_samples]


def load_dataset(dataset_name: str, data_path: str, split: str) -> List[Dict[str, Any]]:
    canonical_name = canonicalize_dataset_name(dataset_name)
    dataset_dir = os.path.join(data_path, canonical_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Cannot find dataset directory for {dataset_name} in {data_path}"
        )

    json_path = os.path.join(dataset_dir, f"{split}.json")
    if os.path.exists(json_path):
        return _load_json_file(json_path)

    if canonical_name.lower() == "chinese-logicqa":
        return _load_chinese_logicqa(dataset_dir, split)

    raise FileNotFoundError(
        f"Cannot find dataset file for {dataset_name} split '{split}' in {dataset_dir}"
    )
