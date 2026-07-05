from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CLEAN_MAIN_CATEGORIES = [
    "blue_and_white_porcelain",
    "artifact_object",
    "artifact_pattern",
    "paper_cutting",
    "window_flower",
    "cultural_clothing",
]


def load_json(path: Path):
    text = path.read_bytes().decode("utf-8", errors="ignore").lstrip("\ufeff")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean-background paper-facing subset from metadata.json")
    parser.add_argument("--dataset-root", default="datasets/starter_cultural_patterns")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    metadata_path = dataset_root / "metadata.json"
    records = load_json(metadata_path)

    clean_records = [record for record in records if record.get("category") in CLEAN_MAIN_CATEGORIES]
    counts = Counter(record.get("category", "") for record in clean_records)

    summary = {
        "total_records": len(records),
        "paper_main_records": len(clean_records),
        "paper_main_categories": CLEAN_MAIN_CATEGORIES,
        "paper_main_category_counts": dict(counts),
        "supplementary_categories": {
            key: sum(1 for record in records if record.get("category") == key)
            for key in sorted({record.get("category", "") for record in records} - set(CLEAN_MAIN_CATEGORIES))
        },
    }

    (dataset_root / "paper_main_metadata.json").write_text(
        json.dumps(clean_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dataset_root / "paper_main_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
