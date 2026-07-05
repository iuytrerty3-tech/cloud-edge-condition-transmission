#!/usr/bin/env python3
"""Round-3: generation-quality metrics for the end-to-end experiment.

Two modes:
  * --lite  : ONLY pixel-identity-rate + SSIM vs the uncompressed-condition reference.
              Needs NO extra model downloads (just SSIM math). This already proves the
              key result: lossless PNG branches are pixel-identical to the ideal.
  * (full)  : additionally FID (Inception), CLIP-score (CLIP), LPIPS (AlexNet). Any model
              that fails to load is skipped with a warning rather than crashing.

  python experiments/compute_generation_metrics.py --gen-root real_runs/e2e_generation_v1 \
      --metadata datasets/starter_cultural_patterns/paper_main_metadata.json \
      --eval-ids experiments/fullreal_eval_ids_v1.json --device cuda [--lite]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REF = "ref_uncompressed"

def _resolve_img(p, PKG):
    c = Path(p)
    if c.exists(): return str(c)
    bases = [PKG, PKG.parent, PKG / "cloud_edge_sd_prototype"]; rels = [p]
    q = p.replace("\\", "/")
    if "cloud_edge_sd_prototype/" in q:
        rels.append(q.split("cloud_edge_sd_prototype/", 1)[1])
    for rel in rels:
        for b in bases:
            if (b / rel).exists(): return str(b / rel)
    return str(c)

def load(p, device, size=512):
    im = Image.open(p).convert("RGB").resize((size, size))
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--eval-ids", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lite", action="store_true", help="skip FID/CLIP/LPIPS (no extra model downloads)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    dev = args.device if torch.cuda.is_available() else "cpu"
    meta = {m["id"]: m for m in json.load(open(args.metadata, encoding="utf-8"))}
    eval_ids = json.load(open(args.eval_ids, encoding="utf-8"))
    gen_root = Path(args.gen_root)
    strategies = [d.name for d in sorted(gen_root.iterdir()) if d.is_dir()]

    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)  # no pretrained model

    clip = lp = None; FID = None
    if not args.lite:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance as FID
        except Exception as e: print("[skip FID]", e); FID = None
        try:
            from torchmetrics.multimodal.clip_score import CLIPScore
            clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(dev)
        except Exception as e: print("[skip CLIP]", e); clip = None
        try:
            import lpips; lp = lpips.LPIPS(net="alex").to(dev)
        except Exception as e: print("[skip LPIPS]", e); lp = None

    ref_gen = {sid: load(gen_root / REF / f"{sid}_generated.png", dev)
               for sid in eval_ids if (gen_root / REF / f"{sid}_generated.png").exists()}

    rows = []
    for strat in strategies:
        sdir = gen_root / strat
        fid = FID(feature=2048, normalize=True).to(dev) if FID else None
        clip_vals, lpips_vals, ssim_vals, ident = [], [], [], []
        for sid in eval_ids:
            gp = sdir / f"{sid}_generated.png"
            if not gp.exists(): continue
            fake = load(gp, dev).to(dev)
            if fid is not None:
                real = load(_resolve_img(meta[sid]["image"], root), dev).to(dev)
                fid.update(real, real=True); fid.update(fake, real=False)
            if clip is not None:
                clip_vals.append(clip(fake.squeeze(0), meta[sid]["prompt"]).item())
            if sid in ref_gen:
                r = ref_gen[sid].to(dev)
                ssim_vals.append(ssim(fake, r).item())
                ident.append(1.0 if torch.equal((fake*255).round().byte(), (r*255).round().byte()) else 0.0)
                if lp is not None:
                    lpips_vals.append(lp(fake*2-1, r*2-1).mean().item())
        rows.append({"strategy": strat,
                     "FID_vs_real": round(float(fid.compute().item()), 3) if fid is not None else None,
                     "CLIP": round(float(np.mean(clip_vals)), 3) if clip_vals else None,
                     "LPIPS_vs_ref": round(float(np.mean(lpips_vals)), 5) if lpips_vals else None,
                     "SSIM_vs_ref": round(float(np.mean(ssim_vals)), 5) if ssim_vals else None,
                     "pixel_identity_rate": round(float(np.mean(ident)), 4) if ident else None,
                     "n": len(ssim_vals) or len(clip_vals)})
    out = Path(args.gen_root) / "generation_metrics.json"
    json.dump(rows, open(out, "w"), indent=2)
    print("\n| Strategy | FID(real) | CLIP | LPIPS(vs ref) | SSIM(vs ref) | pixel-identical |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['strategy']} | {r['FID_vs_real']} | {r['CLIP']} | {r['LPIPS_vs_ref']} | {r['SSIM_vs_ref']} | {r['pixel_identity_rate']} |")
    print(f"\nSaved {out}")
    print("Expected (honest): PNG branches -> SSIM=1.0, pixel-identical=1.0 (identical to the ideal);")
    print("only JPEG branches deviate. Report exactly as printed.")

if __name__ == "__main__":
    main()
