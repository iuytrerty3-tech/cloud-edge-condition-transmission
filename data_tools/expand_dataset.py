#!/usr/bin/env python3
"""Round-3: expand the cultural-pattern dataset from Wikimedia Commons (license-aware),
dedupe, resize to 512, register into an expanded manifest, and regenerate the train/eval split.

Run on a machine with internet access:
  python data_tools/expand_dataset.py \
      --existing datasets/starter_cultural_patterns/paper_main_metadata.json \
      --out-dir datasets/expanded_cultural_patterns \
      --per-category 60 --eval-frac 0.25 --seed 42

Output: <out-dir>/images/<category>/*, paper_main_metadata.json (existing + new),
        train_ids.json, eval_ids.json. Each record keeps its source URL for provenance/licensing.
"""
from __future__ import annotations
import argparse, hashlib, io, json, random, re, urllib.parse, urllib.request
from pathlib import Path
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
CATEGORIES = {
    "blue_and_white_porcelain": ("Blue and white porcelain of China", "blue and white porcelain decorative pattern, Chinese ceramic motif, high detail"),
    "paper_cutting":            ("Chinese paper cutting", "traditional Chinese paper-cut pattern, folk art motif, symmetric and detailed"),
    "window_flower":            ("Chinese window flowers", "traditional Chinese window-flower paper-cut pattern, symmetric folk motif"),
    "cultural_clothing":        ("Hanfu", "traditional Chinese clothing pattern, embroidered textile motif, ornate"),
    "artifact_pattern":         ("Decorative patterns of China", "traditional Chinese decorative pattern, ornamental motif, detailed"),
    "artifact_object":          ("Chinese bronzes", "traditional Chinese artifact, decorative cultural object, detailed"),
}

def api_get(params):
    url = f"{API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def list_files(category, limit):
    titles, cont = [], None
    while len(titles) < limit:
        p = {"action":"query","format":"json","list":"categorymembers","cmtitle":f"Category:{category}",
             "cmnamespace":6,"cmlimit":min(500,limit-len(titles))}
        if cont: p["cmcontinue"]=cont
        d = api_get(p)
        titles += [m["title"] for m in d.get("query",{}).get("categorymembers",[])]
        cont = d.get("continue",{}).get("cmcontinue")
        if not cont: break
    return titles[:limit]

def file_url(title):
    d = api_get({"action":"query","format":"json","prop":"imageinfo","titles":title,"iiprop":"url|extmetadata"})
    for _, page in d.get("query",{}).get("pages",{}).items():
        ii = page.get("imageinfo",[{}])[0]
        return ii.get("url"), ii.get("extmetadata",{}).get("LicenseShortName",{}).get("value","")
    return None, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-category", type=int, default=60)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allowed-licenses", default="CC0,CC BY,CC BY-SA,Public domain")
    args = ap.parse_args()
    out = Path(args.out_dir); (out/"images").mkdir(parents=True, exist_ok=True)
    allowed = [a.strip().lower() for a in args.allowed_licenses.split(",")]

    existing = json.load(open(args.existing, encoding="utf-8"))
    records = list(existing)
    seen_hashes = set()
    for m in existing:  # hash existing to avoid duplicates if files are reachable
        try:
            img_path = Path(m["image"])
            if not img_path.is_absolute():
                img_path = Path(args.existing).resolve().parents[2] / m["image"]
            seen_hashes.add(hashlib.md5(img_path.read_bytes()).hexdigest())
        except Exception: pass

    next_idx = {}
    for title_cat, (commons_cat, prompt) in CATEGORIES.items():
        cat_dir = out/"images"/title_cat; cat_dir.mkdir(parents=True, exist_ok=True)
        got = 0
        for title in list_files(commons_cat, args.per_category*3):
            if got >= args.per_category: break
            try:
                url, lic = file_url(title)
                if not url or not any(a in lic.lower() for a in allowed): continue
                with urllib.request.urlopen(url, timeout=90) as r: raw = r.read()
                h = hashlib.md5(raw).hexdigest()
                if h in seen_hashes: continue
                seen_hashes.add(h)
                img = Image.open(io.BytesIO(raw)).convert("RGB").resize((512,512), Image.Resampling.BILINEAR)
                next_idx[title_cat] = next_idx.get(title_cat, 0)+1
                sid = f"{title_cat}_exp_{next_idx[title_cat]:03d}"
                fp = cat_dir/f"{sid}.jpg"; img.save(fp, quality=95)
                records.append({"category":title_cat,"prompt":prompt,"id":sid,
                                "source_url":f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}",
                                "license":lic,"image":str(fp.resolve())})
                got += 1
            except Exception as e:
                print("skip", title, repr(e)); continue
        print(f"[{title_cat}] added {got}")

    random.seed(args.seed); ids = [r["id"] for r in records]; random.shuffle(ids)
    n_eval = int(len(ids)*args.eval_frac)
    eval_ids, train_ids = ids[:n_eval], ids[n_eval:]
    json.dump(records, open(out/"paper_main_metadata.json","w"), indent=2, ensure_ascii=False)
    json.dump(train_ids, open(out/"train_ids.json","w"), indent=2)
    json.dump(eval_ids, open(out/"eval_ids.json","w"), indent=2)
    print(f"\nTotal records: {len(records)} (was {len(existing)}). Train {len(train_ids)} / Eval {len(eval_ids)}.")
    print(f"Wrote {out}/paper_main_metadata.json, train_ids.json, eval_ids.json")
    print("Next: re-run reproduce_codec_conditions.py and train_hybrid_ddpg_scheduler.py with these files.")

if __name__ == "__main__":
    main()
