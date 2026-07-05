from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def load_metadata(path: Path) -> list[dict]:
    text = path.read_bytes().decode("utf-8", errors="ignore").lstrip("\ufeff")
    return json.loads(text)


def image_record_stats(record: dict) -> dict:
    image_path = Path(record["image"])
    with Image.open(image_path) as img:
        width, height = img.size
        mode = img.mode
    file_size = image_path.stat().st_size
    min_side = min(width, height)
    flagged = min_side < 256 or file_size < 20 * 1024
    reason = None
    if flagged:
        if min_side < 256 and file_size < 20 * 1024:
            reason = "tiny_ui_asset"
        elif min_side < 256:
            reason = "too_small"
        else:
            reason = "too_lightweight"
    return {
        "id": record["id"],
        "file": str(image_path.resolve()).replace("\\", "/"),
        "category": record["category"],
        "width": width,
        "height": height,
        "mode": mode,
        "file_size_bytes": file_size,
        "flagged": flagged,
        "flag_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit starter cultural pattern dataset quality")
    parser.add_argument(
        "--dataset-root",
        default="datasets/starter_cultural_patterns",
        help="Path to starter dataset root",
    )
    parser.add_argument(
        "--rewrite-metadata",
        action="store_true",
        help="Rewrite metadata.json and manifest.jsonl with only kept samples",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    metadata_path = dataset_root / "metadata.json"
    manifest_path = dataset_root / "manifest.jsonl"
    stats_path = dataset_root / "image_stats.json"
    audit_path = dataset_root / "quality_audit.json"

    records = load_metadata(metadata_path)
    stats = [image_record_stats(record) for record in records]

    flagged_ids = {item["id"] for item in stats if item["flagged"]}
    kept_records = [record for record in records if record["id"] not in flagged_ids]

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "total_records": len(records),
                "kept_records": len(kept_records),
                "flagged_records": len(flagged_ids),
                "flagged_files": [item for item in stats if item["flagged"]],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.rewrite_metadata:
        metadata_path.write_text(json.dumps(kept_records, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_lines = [json.dumps(record, ensure_ascii=False) for record in kept_records]
        manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")

    print(
        json.dumps(
            {
                "total_records": len(records),
                "kept_records": len(kept_records),
                "flagged_records": len(flagged_ids),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
