from __future__ import annotations

import argparse
import json
import re
import traceback
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://commons.wikimedia.org/w/api.php"

CATEGORY_PRESETS = {
    "blue_and_white_porcelain": {
        "category": "Blue and white porcelain of China",
        "prompt": "blue and white porcelain decorative pattern, Chinese ceramic motif, high detail",
    },
    "gu_embroidery": {
        "category": "Gu embroidery",
        "prompt": "traditional Chinese embroidery pattern, silk textile motif, ornate and detailed",
    },
    "paper_cutting": {
        "category": "Chinese paper cutting",
        "prompt": "traditional Chinese paper-cut pattern, folk art motif, symmetric and detailed",
    },
}


def api_get(params: dict) -> dict:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name


def list_category_files(category: str, limit: int) -> list[str]:
    titles: list[str] = []
    cont = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 6,
            "cmlimit": min(500, limit - len(titles)),
        }
        if cont:
            params["cmcontinue"] = cont
        data = api_get(params)
        items = data.get("query", {}).get("categorymembers", [])
        titles.extend(item["title"] for item in items if item["title"].startswith("File:"))
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont or not items:
            break
    return titles[:limit]


def file_info(title: str) -> dict | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": title,
        "iiprop": "url|size|mime|extmetadata",
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info_list = page.get("imageinfo")
        if info_list:
            info = info_list[0]
            info["title"] = title
            return info
    return None


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)


def build_record(category_key: str, info: dict, local_path: Path) -> dict:
    metadata = info.get("extmetadata", {})
    return {
        "id": local_path.stem,
        "category_key": category_key,
        "category": CATEGORY_PRESETS[category_key]["category"],
        "image": str(local_path).replace("\\", "/"),
        "prompt": CATEGORY_PRESETS[category_key]["prompt"],
        "source_url": info.get("descriptionurl"),
        "image_url": info.get("url"),
        "width": info.get("width"),
        "height": info.get("height"),
        "mime": info.get("mime"),
        "license_short_name": metadata.get("LicenseShortName", {}).get("value"),
        "license_url": metadata.get("LicenseUrl", {}).get("value"),
        "artist": metadata.get("Artist", {}).get("value"),
        "image_description": metadata.get("ImageDescription", {}).get("value"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cultural pattern images from Wikimedia Commons")
    parser.add_argument("--output-root", type=str, default="datasets/wikimedia_cultural_patterns")
    parser.add_argument("--max-per-category", type=int, default=20)
    parser.add_argument(
        "--categories",
        nargs="*",
        default=list(CATEGORY_PRESETS.keys()),
        choices=list(CATEGORY_PRESETS.keys()),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    metadata_path = output_root / "metadata.json"
    report_path = output_root / "download_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "started",
                "max_per_category": args.max_per_category,
                "categories": args.categories,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    all_records: list[dict] = []

    try:
        for category_key in args.categories:
            preset = CATEGORY_PRESETS[category_key]
            category_dir = images_root / category_key
            category_dir.mkdir(parents=True, exist_ok=True)
            titles = list_category_files(preset["category"], args.max_per_category)
            downloaded = 0
            for title in titles:
                info = file_info(title)
                if not info or "url" not in info:
                    continue
                if int(info.get("width", 0)) < 512 or int(info.get("height", 0)) < 512:
                    continue

                suffix = Path(urllib.parse.urlparse(info["url"]).path).suffix or ".jpg"
                filename = sanitize_filename(title.removeprefix("File:"))
                local_path = category_dir / f"{Path(filename).stem}{suffix}"
                try:
                    download_file(info["url"], local_path)
                except Exception:
                    continue

                all_records.append(build_record(category_key, info, local_path))
                downloaded += 1
                if downloaded >= args.max_per_category:
                    break

        manifest_lines = [json.dumps(record, ensure_ascii=False) for record in all_records]
        manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
        metadata_path.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
        report_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "downloaded": len(all_records),
                    "manifest": str(manifest_path).replace("\\", "/"),
                    "metadata": str(metadata_path).replace("\\", "/"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Downloaded {len(all_records)} images to {output_root}")
        print(f"Manifest: {manifest_path}")
        print(f"Metadata: {metadata_path}")
    except Exception as exc:
        report_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
