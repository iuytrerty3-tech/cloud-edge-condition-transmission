from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path


ITEMS = [
    {
        "id": "porcelain_001",
        "category": "blue_and_white_porcelain",
        "filename": "Blue and white vase-like pottery.jpg",
        "prompt": "blue and white porcelain decorative pattern, Chinese ceramic motif, high detail",
    },
    {
        "id": "porcelain_002",
        "category": "blue_and_white_porcelain",
        "filename": "CapitalMuseum4.jpg",
        "prompt": "blue and white porcelain decorative pattern, Chinese ceramic motif, high detail",
    },
    {
        "id": "porcelain_003",
        "category": "blue_and_white_porcelain",
        "filename": "CapitalMuseum6.jpg",
        "prompt": "blue and white porcelain decorative pattern, Chinese ceramic motif, high detail",
    },
    {
        "id": "porcelain_004",
        "category": "blue_and_white_porcelain",
        "filename": "Guangdong Sheng Bowuguan 2012.11.18 10-03-06.jpg",
        "prompt": "blue and white porcelain decorative pattern, Chinese ceramic motif, high detail",
    },
    {
        "id": "porcelain_005",
        "category": "blue_and_white_porcelain",
        "filename": "HKU MAG Fung Ping Shan Museum P01.JPG",
        "prompt": "blue and white porcelain decorative pattern, Chinese ceramic motif, high detail",
    },
    {
        "id": "embroidery_001",
        "category": "gu_embroidery",
        "filename": "Gu embroidery.jpg",
        "prompt": "traditional Chinese embroidery pattern, silk textile motif, ornate and detailed",
    },
    {
        "id": "papercut_001",
        "category": "paper_cutting",
        "filename": "Chinese Paper Cutting.png",
        "prompt": "traditional Chinese paper-cut pattern, folk art motif, symmetric and detailed",
    },
    {
        "id": "papercut_002",
        "category": "paper_cutting",
        "filename": "Chinese paper cutting-Pig.jpg",
        "prompt": "traditional Chinese paper-cut pattern, folk art motif, symmetric and detailed",
    },
]


def main() -> None:
    output_root = Path("datasets/starter_cultural_patterns")
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for item in ITEMS:
        category_dir = images_root / item["category"]
        category_dir.mkdir(parents=True, exist_ok=True)
        encoded_name = urllib.parse.quote(item["filename"], safe="")
        url = f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{encoded_name}"
        suffix = Path(item["filename"]).suffix or ".jpg"
        local_path = category_dir / f"{item['id']}{suffix.lower()}"
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception:
            continue
        records.append(
            {
                "id": item["id"],
                "category": item["category"],
                "image": str(local_path).replace("\\", "/"),
                "prompt": item["prompt"],
                "source_url": url,
            }
        )

    (output_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    (output_root / "metadata.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Downloaded {len(records)} starter images")


if __name__ == "__main__":
    main()
