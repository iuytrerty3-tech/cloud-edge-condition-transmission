"""
Download 40 high-density cultural pattern images from Wikimedia Commons.
Run this script directly: python download_more_images.py

Uses hardcoded known Wikimedia Commons file URLs (no API needed).
Falls back to API discovery if network allows it.

Targets:
  - papercut_015 to papercut_029  (15 images)
  - window_009  to window_023     (15 images)
  - cloth_015   to cloth_024      (10 images)
"""

from pathlib import Path
import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
import ssl

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = str(Path(__file__).resolve().parent)
DIRS = {
    "paper_cutting":     os.path.join(BASE, "images", "paper_cutting"),
    "window_flower":     os.path.join(BASE, "images", "window_flower"),
    "cultural_clothing": os.path.join(BASE, "images", "cultural_clothing"),
}
MANIFEST = os.path.join(BASE, "manifest.jsonl")

# ── SSL context ────────────────────────────────────────────────────────────────
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

HEADERS = {"User-Agent": "CulturalPatternDataset/1.0 (research; python-urllib)"}

# ── Known Wikimedia Commons files ─────────────────────────────────────────────
# Format: (commons_file_title, direct_upload_url)
# Direct URLs follow the pattern:
#   https://upload.wikimedia.org/wikipedia/commons/[hash1]/[hash2]/Filename.jpg

PAPER_CUTTING_FILES = [
    ("File:Chinese paper cutting - dragon.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Chinese_paper_cutting_-_dragon.jpg/800px-Chinese_paper_cutting_-_dragon.jpg"),
    ("File:Chinese paper cutting - phoenix.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Chinese_paper_cutting_-_phoenix.jpg/800px-Chinese_paper_cutting_-_phoenix.jpg"),
    ("File:Jianzhi-rooster.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Jianzhi-rooster.jpg/800px-Jianzhi-rooster.jpg"),
    ("File:Chinese New Year paper cutting.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Chinese_New_Year_paper_cutting.jpg/800px-Chinese_New_Year_paper_cutting.jpg"),
    ("File:Paper cutting China.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Paper_cutting_China.jpg/800px-Paper_cutting_China.jpg"),
    ("File:Chinese paper cutting fish.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Chinese_paper_cutting_fish.jpg/800px-Chinese_paper_cutting_fish.jpg"),
    ("File:Papercutting-butterfly.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Papercutting-butterfly.jpg/800px-Papercutting-butterfly.jpg"),
    ("File:Chinese paper cutting flower.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Chinese_paper_cutting_flower.jpg/800px-Chinese_paper_cutting_flower.jpg"),
    ("File:Jianzhi lotus.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Jianzhi_lotus.jpg/800px-Jianzhi_lotus.jpg"),
    ("File:Chinese paper cutting panda.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Chinese_paper_cutting_panda.jpg/800px-Chinese_paper_cutting_panda.jpg"),
    ("File:Paper cutting art China red.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Paper_cutting_art_China_red.jpg/800px-Paper_cutting_art_China_red.jpg"),
    ("File:Chinese paper cutting horse.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Chinese_paper_cutting_horse.jpg/800px-Chinese_paper_cutting_horse.jpg"),
    ("File:Jianzhi-tiger.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Jianzhi-tiger.jpg/800px-Jianzhi-tiger.jpg"),
    ("File:Chinese paper cutting crane.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Chinese_paper_cutting_crane.jpg/800px-Chinese_paper_cutting_crane.jpg"),
    ("File:Paper cutting China 2.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Paper_cutting_China_2.jpg/800px-Paper_cutting_China_2.jpg"),
    ("File:Chinese paper cutting magpie.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Chinese_paper_cutting_magpie.jpg/800px-Chinese_paper_cutting_magpie.jpg"),
    ("File:Jianzhi-rabbit.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Jianzhi-rabbit.jpg/800px-Jianzhi-rabbit.jpg"),
    ("File:Chinese paper cutting peony.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Chinese_paper_cutting_peony.jpg/800px-Chinese_paper_cutting_peony.jpg"),
    ("File:Paper cutting China 3.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Paper_cutting_China_3.jpg/800px-Paper_cutting_China_3.jpg"),
    ("File:Chinese paper cutting zodiac.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Chinese_paper_cutting_zodiac.jpg/800px-Chinese_paper_cutting_zodiac.jpg"),
]

WINDOW_FLOWER_FILES = [
    ("File:Chinese window flower red.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Chinese_window_flower_red.jpg/800px-Chinese_window_flower_red.jpg"),
    ("File:Chuanghua paper cutting.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Chuanghua_paper_cutting.jpg/800px-Chuanghua_paper_cutting.jpg"),
    ("File:Window flower China 1.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Window_flower_China_1.jpg/800px-Window_flower_China_1.jpg"),
    ("File:Chinese window decoration paper.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Chinese_window_decoration_paper.jpg/800px-Chinese_window_decoration_paper.jpg"),
    ("File:Window flower China 2.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Window_flower_China_2.jpg/800px-Window_flower_China_2.jpg"),
    ("File:Chinese New Year window flower.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Chinese_New_Year_window_flower.jpg/800px-Chinese_New_Year_window_flower.jpg"),
    ("File:Chuanghua red pattern.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Chuanghua_red_pattern.jpg/800px-Chuanghua_red_pattern.jpg"),
    ("File:Window flower China 3.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Window_flower_China_3.jpg/800px-Window_flower_China_3.jpg"),
    ("File:Chinese window paper cutting flower.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Chinese_window_paper_cutting_flower.jpg/800px-Chinese_window_paper_cutting_flower.jpg"),
    ("File:Window flower China 4.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Window_flower_China_4.jpg/800px-Window_flower_China_4.jpg"),
    ("File:Chinese window decoration 1.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Chinese_window_decoration_1.jpg/800px-Chinese_window_decoration_1.jpg"),
    ("File:Chuanghua lotus.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Chuanghua_lotus.jpg/800px-Chuanghua_lotus.jpg"),
    ("File:Window flower China 5.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Window_flower_China_5.jpg/800px-Window_flower_China_5.jpg"),
    ("File:Chinese window paper cutting 2.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Chinese_window_paper_cutting_2.jpg/800px-Chinese_window_paper_cutting_2.jpg"),
    ("File:Window flower China 6.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Window_flower_China_6.jpg/800px-Window_flower_China_6.jpg"),
    ("File:Chinese window decoration 2.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Chinese_window_decoration_2.jpg/800px-Chinese_window_decoration_2.jpg"),
    ("File:Chuanghua dragon.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Chuanghua_dragon.jpg/800px-Chuanghua_dragon.jpg"),
    ("File:Window flower China 7.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Window_flower_China_7.jpg/800px-Window_flower_China_7.jpg"),
    ("File:Chinese window paper cutting 3.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Chinese_window_paper_cutting_3.jpg/800px-Chinese_window_paper_cutting_3.jpg"),
    ("File:Window flower China 8.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Window_flower_China_8.jpg/800px-Window_flower_China_8.jpg"),
]

EMBROIDERY_FILES = [
    ("File:Chinese embroidery dragon robe.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Chinese_embroidery_dragon_robe.jpg/800px-Chinese_embroidery_dragon_robe.jpg"),
    ("File:Suzhou embroidery cat.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Suzhou_embroidery_cat.jpg/800px-Suzhou_embroidery_cat.jpg"),
    ("File:Chinese embroidery flowers.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Chinese_embroidery_flowers.jpg/800px-Chinese_embroidery_flowers.jpg"),
    ("File:Hunan embroidery tiger.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Hunan_embroidery_tiger.jpg/800px-Hunan_embroidery_tiger.jpg"),
    ("File:Chinese embroidery phoenix.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Chinese_embroidery_phoenix.jpg/800px-Chinese_embroidery_phoenix.jpg"),
    ("File:Sichuan embroidery panda.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sichuan_embroidery_panda.jpg/800px-Sichuan_embroidery_panda.jpg"),
    ("File:Chinese embroidery landscape.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Chinese_embroidery_landscape.jpg/800px-Chinese_embroidery_landscape.jpg"),
    ("File:Suzhou embroidery fish.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Suzhou_embroidery_fish.jpg/800px-Suzhou_embroidery_fish.jpg"),
    ("File:Chinese embroidery peony.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Chinese_embroidery_peony.jpg/800px-Chinese_embroidery_peony.jpg"),
    ("File:Hunan embroidery lotus.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Hunan_embroidery_lotus.jpg/800px-Hunan_embroidery_lotus.jpg"),
    ("File:Chinese embroidery crane.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Chinese_embroidery_crane.jpg/800px-Chinese_embroidery_crane.jpg"),
    ("File:Sichuan embroidery flowers.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sichuan_embroidery_flowers.jpg/800px-Sichuan_embroidery_flowers.jpg"),
    ("File:Chinese embroidery dragon.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Chinese_embroidery_dragon.jpg/800px-Chinese_embroidery_dragon.jpg"),
    ("File:Suzhou embroidery landscape.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Suzhou_embroidery_landscape.jpg/800px-Suzhou_embroidery_landscape.jpg"),
    ("File:Chinese embroidery bird.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Chinese_embroidery_bird.jpg/800px-Chinese_embroidery_bird.jpg"),
]

# ── Prompts ────────────────────────────────────────────────────────────────────
PROMPTS = {
    "paper_cutting":     "Chinese paper cutting art, intricate pattern, high detail",
    "window_flower":     "Chinese window flower paper cutting, decorative pattern, intricate design",
    "cultural_clothing": "Chinese embroidery pattern, traditional textile design, intricate stitching",
}

PREFIXES = {
    "paper_cutting":     ("papercut", 15, 15, PAPER_CUTTING_FILES),
    "window_flower":     ("window",   9,  15, WINDOW_FLOWER_FILES),
    "cultural_clothing": ("cloth",    15, 10, EMBROIDERY_FILES),
}


# ── helpers ────────────────────────────────────────────────────────────────────
def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


def api_get(params):
    base = "https://commons.wikimedia.org/w/api.php"
    params["format"] = "json"
    url = base + "?" + urllib.parse.urlencode(params)
    return json.loads(fetch(url, timeout=20))


def get_real_url_via_api(file_title):
    """Use Wikimedia API to get the real direct URL for a file."""
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
    }
    data = api_get(params)
    for page in data.get("query", {}).get("pages", {}).values():
        ii = page.get("imageinfo", [{}])[0]
        if ii.get("url"):
            return ii["url"], ii.get("width", 0), ii.get("height", 0)
    return None, 0, 0


def get_category_files_api(category, limit=60):
    """Discover files via API (used when network allows)."""
    files = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": str(limit),
    }
    data = api_get(params)
    members = data.get("query", {}).get("categorymembers", [])
    files.extend(members)
    return files


def load_manifest_ids():
    ids = set()
    if not os.path.exists(MANIFEST):
        return ids
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def append_manifest(entry):
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def existing_ids(cat_key):
    d = DIRS[cat_key]
    if not os.path.isdir(d):
        return set()
    found = set()
    for fn in os.listdir(d):
        name, _ = os.path.splitext(fn)
        found.add(name)
    return found


def next_indices(prefix, start, count, cat_key):
    existing = existing_ids(cat_key)
    needed = []
    i = start
    while len(needed) < count:
        if f"{prefix}_{i:03d}" not in existing:
            needed.append(i)
        i += 1
        if i > start + count + 50:
            break
    return needed


# ── API-based discovery (bonus: runs if network is available) ──────────────────
API_CATEGORIES = {
    "paper_cutting":     ["Chinese_paper_cutting", "Jianzhi"],
    "window_flower":     ["Chinese_paper_cutting", "Window_flowers"],
    "cultural_clothing": ["Chinese_embroidery", "Suzhou_embroidery",
                          "Hunan_embroidery", "Sichuan_embroidery"],
}


def try_api_discovery(cat_key, needed_count):
    """Try to get file list via API. Returns list of (title, url) or []."""
    results = []
    seen = set()
    for cat in API_CATEGORIES.get(cat_key, []):
        try:
            members = get_category_files_api(cat, limit=80)
            titles = [m["title"] for m in members if m["title"] not in seen]
            seen.update(titles)
            # Get image info in batches
            for i in range(0, len(titles), 20):
                batch = titles[i:i+20]
                params = {
                    "action": "query",
                    "titles": "|".join(batch),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                }
                data = api_get(params)
                for page in data.get("query", {}).get("pages", {}).values():
                    t = page.get("title", "")
                    ii = page.get("imageinfo", [{}])[0]
                    mime = ii.get("mime", "")
                    w = ii.get("width", 0)
                    h = ii.get("height", 0)
                    url = ii.get("url", "")
                    if url and mime.startswith("image/") and w >= 256 and h >= 256:
                        results.append((t, url))
                time.sleep(0.3)
            if len(results) >= needed_count * 2:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"    API discovery failed for {cat}: {e}")
    return results


# ── main ───────────────────────────────────────────────────────────────────────
def run():
    manifest_ids = load_manifest_ids()
    total_downloaded = 0

    for cat_key, (prefix, start, count, hardcoded_files) in PREFIXES.items():
        dest_dir = DIRS[cat_key]
        os.makedirs(dest_dir, exist_ok=True)
        prompt = PROMPTS[cat_key]

        indices = next_indices(prefix, start, count, cat_key)
        print(f"\n{'='*60}")
        print(f"Category: {cat_key}  |  Need {count} images, indices: {indices[:count]}")

        # Try API discovery first
        print("  Trying API discovery...", end=" ", flush=True)
        api_files = []
        try:
            api_files = try_api_discovery(cat_key, count)
            print(f"found {len(api_files)} via API")
        except Exception as e:
            print(f"unavailable ({e})")

        # Merge: API results first, then hardcoded fallbacks
        # Deduplicate by title
        all_files = []
        seen_titles = set()
        for title, url in api_files:
            if title not in seen_titles:
                all_files.append((title, url))
                seen_titles.add(title)
        for title, url in hardcoded_files:
            if title not in seen_titles:
                all_files.append((title, url))
                seen_titles.add(title)

        print(f"  Total candidates: {len(all_files)}")

        downloaded_count = 0
        idx_iter = iter(indices)

        for title, url in all_files:
            if downloaded_count >= count:
                break
            try:
                idx = next(idx_iter)
            except StopIteration:
                break

            img_id = f"{prefix}_{idx:03d}"
            if img_id in manifest_ids:
                print(f"  SKIP {img_id} (already in manifest)")
                downloaded_count += 1
                continue

            # Determine extension from URL
            url_lower = url.lower().split("?")[0]
            if url_lower.endswith(".png"):
                ext = ".png"
            elif url_lower.endswith(".gif"):
                ext = ".gif"
            else:
                ext = ".jpg"

            filename = f"{img_id}{ext}"
            dest_path = os.path.join(dest_dir, filename)

            if os.path.exists(dest_path):
                print(f"  SKIP {filename} (file exists)")
                downloaded_count += 1
                continue

            print(f"  [{downloaded_count+1}/{count}] {filename} <- {url[:65]}...", end=" ", flush=True)
            try:
                data = fetch(url, timeout=60)
                if len(data) < 1000:
                    print(f"SKIP (too small: {len(data)} bytes)")
                    continue
                with open(dest_path, "wb") as f:
                    f.write(data)
                print(f"OK ({len(data)//1024}KB)")

                commons_page = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(
                    title.replace(" ", "_"), safe=":/"
                )
                entry = {
                    "category": cat_key,
                    "prompt": prompt,
                    "id": img_id,
                    "source_url": commons_page,
                    "image": dest_path,
                }
                append_manifest(entry)
                manifest_ids.add(img_id)
                downloaded_count += 1
                total_downloaded += 1
                time.sleep(0.5)

            except urllib.error.HTTPError as e:
                print(f"HTTP {e.code}")
            except Exception as e:
                print(f"FAILED: {e}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)

        print(f"  Result: {downloaded_count}/{count} for {cat_key}")

    print(f"\n{'='*60}")
    print(f"TOTAL downloaded this run: {total_downloaded} images")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    run()
