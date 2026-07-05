#!/usr/bin/env python3
"""Round-3 experiment: real end-to-end SD1.5+ControlNet generation per transmission strategy.

For each held-out image and each strategy we (i) extract the Canny condition, (ii) apply the
strategy's transport codec, (iii) reconstruct the condition, (iv) generate the final image with a
FIXED seed, and (v) log the real cloud generation time. Lossless PNG branches must reproduce the
reference (uncompressed-condition) image bit-for-bit; JPEG branches may differ slightly.

Outputs: outputs/<strategy>/<id>_generated.png, _condition.png, and end_to_end_summary.json.

Run on a CUDA GPU:
  python experiments/run_end_to_end_generation.py \
      --metadata datasets/starter_cultural_patterns/paper_main_metadata.json \
      --eval-ids experiments/fullreal_eval_ids_v1.json \
      --output-root real_runs/e2e_generation_v1 --device cuda
"""
from __future__ import annotations
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cloud_edge_sd_prototype.cloud import CloudGenerator
from cloud_edge_sd_prototype.edge import EdgeConditionExtractor
from cloud_edge_sd_prototype.utils import simulate_network_time_seconds, measure_cuda_seconds

# (name, kind, fmt, param)  kind: 'ref'|'cond'|'cloud'
STRATEGIES = [
    ("ref_uncompressed", "ref",   None,   None),   # reference for compression-induced change
    ("cond_png_l6",      "cond",  "PNG",  6),
    ("cond_png_l3",      "cond",  "PNG",  3),
    ("cond_jpeg_q75",    "cond",  "JPEG", 75),      # lossy condition transport
    ("cloud_png",        "cloud", "PNG",  6),       # raw image upload, extract on cloud
    ("cloud_jpeg_q75",   "cloud", "JPEG", 75),
]

def cond_to_uint8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(0, 1)
    return (x * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()

def enc_bytes(arr_pil: Image.Image, fmt: str, param) -> bytes:
    buf = io.BytesIO()
    if fmt == "PNG":
        arr_pil.save(buf, format="PNG", compress_level=int(param))
    else:
        arr_pil.save(buf, format="JPEG", quality=int(param))
    return buf.getvalue()

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
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--eval-ids", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bandwidth-mbps", type=float, default=5.0)
    ap.add_argument("--rtt-ms", type=float, default=20.0)
    ap.add_argument("--base-model-id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet-model-id", default="lllyasviel/sd-controlnet-canny")
    ap.add_argument("--limit", type=int, default=0, help="0 = all held-out images")
    args = ap.parse_args()

    meta = {m["id"]: m for m in json.load(open(args.metadata, encoding="utf-8"))}
    eval_ids = json.load(open(args.eval_ids, encoding="utf-8"))
    if args.limit:
        eval_ids = eval_ids[: args.limit]
    out_root = Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)

    extractor = EdgeConditionExtractor(mode="canny", image_size=512,
                                       canny_low_threshold=100, canny_high_threshold=200)
    gen = CloudGenerator(args.base_model_id, args.controlnet_model_id, device=args.device)
    if "cuda" in args.device:
        gen.warmup()

    summary = {s[0]: {"gen_time": [], "payload_kb": [], "total_with_gen": []} for s in STRATEGIES}
    for sid in eval_ids:
        m = meta[sid]; prompt = m["prompt"]; img_path = _resolve_img(m["image"], ROOT)
        base_cond = extractor(img_path)  # original Canny condition tensor
        for name, kind, fmt, param in STRATEGIES:
            sdir = out_root / name; sdir.mkdir(parents=True, exist_ok=True)
            if kind == "ref":
                cond = base_cond; payload = len(enc_bytes(Image.fromarray(cond_to_uint8(base_cond)), "PNG", 0))
            elif kind == "cond":
                pil = Image.fromarray(cond_to_uint8(base_cond))
                pb = enc_bytes(pil, fmt, param); payload = len(pb)
                dec = np.asarray(Image.open(io.BytesIO(pb)).convert("RGB"))
                cond = torch.from_numpy(dec).permute(2, 0, 1).float() / 255.0
            else:  # cloud: upload raw image, extract on cloud
                raw = Image.open(img_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
                pb = enc_bytes(raw, fmt, param); payload = len(pb)
                dec_img = Image.open(io.BytesIO(pb)).convert("RGB")
                cond = extractor(dec_img)
            net = simulate_network_time_seconds(payload, args.bandwidth_mbps, args.rtt_ms)
            out_img, gtime = measure_cuda_seconds(
                lambda: gen.generate(prompt, cond.to(gen.device), num_inference_steps=args.steps,
                                     guidance_scale=args.guidance, seed=args.seed),
                device=args.device)
            out_img.save(sdir / f"{sid}_generated.png")
            Image.fromarray(cond_to_uint8(cond)).save(sdir / f"{sid}_condition.png")
            summary[name]["gen_time"].append(gtime)
            summary[name]["payload_kb"].append(payload / 1024)
            summary[name]["total_with_gen"].append(net + gtime)
        print(f"[done] {sid}")

    agg = {}
    for name in summary:
        g = summary[name]
        agg[name] = {"mean_gen_time_s": float(np.mean(g["gen_time"])),
                     "mean_payload_kb": float(np.mean(g["payload_kb"])),
                     "mean_total_with_gen_s": float(np.mean(g["total_with_gen"])),
                     "n": len(g["gen_time"])}
    json.dump({"strategies": agg, "config": vars(args)},
              open(out_root / "end_to_end_summary.json", "w"), indent=2)
    print("\n=== mean cloud generation time (s) — should be ~equal across strategies ===")
    for name, a in agg.items():
        print(f"  {name:18} gen={a['mean_gen_time_s']:.3f}s  payload={a['mean_payload_kb']:.1f}KB")
    print(f"\nSaved {out_root/'end_to_end_summary.json'}")

if __name__ == "__main__":
    main()
