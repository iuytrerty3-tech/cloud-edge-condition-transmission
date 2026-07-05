#!/usr/bin/env python3
"""Round-3: qualitative montage (reference | condition | generated under each strategy).
  python experiments/make_qualitative_figure.py --gen-root real_runs/e2e_generation_v1 \
      --metadata datasets/starter_cultural_patterns/paper_main_metadata.json --out figures/figQ_qualitative.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

COLS = [("ref_uncompressed", "Gen (uncompressed)"), ("cond_png_l6", "Gen (PNG-L6)"), ("cond_jpeg_q75", "Gen (JPEG-q75)")]

def _resolve_img(p, PKG):
    from pathlib import Path
    c = Path(p)
    if c.exists():
        return str(c)
    bases = [PKG, PKG.parent, PKG / "cloud_edge_sd_prototype"]
    rels = [p]
    q = p.replace("\\", "/")
    if "cloud_edge_sd_prototype/" in q:
        rels.append(q.split("cloud_edge_sd_prototype/", 1)[1])
    for rel in rels:
        for b in bases:
            if (b / rel).exists():
                return str(b / rel)
    return str(c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="figures/figQ_qualitative.png")
    ap.add_argument("--per-category", type=int, default=1)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1].parent
    meta = json.load(open(args.metadata, encoding="utf-8"))
    gen_root = Path(args.gen_root)
    # pick representative ids that were actually generated
    seen, picks = {}, []
    for m in meta:
        c = m["category"]
        if seen.get(c, 0) >= args.per_category: continue
        if (gen_root / "ref_uncompressed" / f"{m['id']}_generated.png").exists():
            picks.append(m); seen[c] = seen.get(c, 0) + 1
    ncol = 2 + len(COLS); nrow = len(picks)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.1*ncol, 2.1*nrow))
    if nrow == 1: axs = axs.reshape(1, -1)
    heads = ["Reference", "Condition"] + [c[1] for c in COLS]
    for j, h in enumerate(heads): axs[0, j].set_title(h, fontsize=10)
    for i, m in enumerate(picks):
        sid = m["id"]
        ref_img = _resolve_img(m["image"], root)
        try: axs[i,0].imshow(Image.open(ref_img).convert("RGB").resize((256,256)))
        except Exception: pass
        axs[i,1].imshow(Image.open(gen_root/"ref_uncompressed"/f"{sid}_condition.png").convert("RGB").resize((256,256)))
        for j,(strat,_) in enumerate(COLS):
            p = gen_root/strat/f"{sid}_generated.png"
            if p.exists(): axs[i,2+j].imshow(Image.open(p).convert("RGB").resize((256,256)))
        axs[i,0].set_ylabel(m["category"].replace("_","\n"), fontsize=8)
        for j in range(ncol): axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
    plt.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150); print(f"saved {args.out} ({nrow} rows)")

if __name__ == "__main__":
    main()
