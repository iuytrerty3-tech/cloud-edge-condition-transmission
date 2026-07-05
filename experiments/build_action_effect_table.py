#!/usr/bin/env python3
"""Measure the REAL action-effect table for the sequential-scheduling experiment.

For every held-out image and every transport action we measure, with the *real*
edge extractor + codecs (no GPU, no Stable Diffusion -- the SD generation time is a
constant that cancels across actions), the two quantities the scheduler trades off:

  * payload_kb      -- real compressed bytes that must be transmitted
  * condition_ssim  -- structural similarity of the *reconstructed* control condition
                       vs the lossless reference condition (the quality the cloud
                       generator actually receives)

Action families (cf. RL_advantage_experiment_plan.md sec.1 / mechanism C):
  * edge-PNG : mode{canny,skeleton} x png_level{1,3,6,9} x downsample{1,2,4}
  * edge-JPEG: mode{canny,skeleton} x quality grid (continuous q is interpolated by the env)
  * cloud    : upload the raw image, extract on the cloud
               - cloud_png  (lossless raw upload  -> ssim 1.0, large payload)
               - cloud_jpeg (lossy raw upload      -> ssim<1, small payload)

The table is consumed by envs/sequential_channel_env.py. We also print a validation
against the 3 branches present in the shipped sample_stats.json so the measurements
can be cross-checked.

Run (CPU, a few minutes):
  python experiments/build_action_effect_table.py \
      --metadata datasets/starter_cultural_patterns/paper_main_metadata.json \
      --out real_runs/seq/action_effect_table.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cloud_edge_sd_prototype.edge import EdgeConditionExtractor  # noqa: E402

IMAGE_SIZE = 512
PNG_LEVELS = [1, 3, 6, 9]
DOWNSAMPLE = [1, 2, 4]
JPEG_Q_GRID = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
MODES = ["canny", "skeleton"]


def _resolve_img(p: str, pkg: Path) -> str:
    c = Path(p)
    if c.exists():
        return str(c)
    bases = [pkg, pkg.parent, pkg / "cloud_edge_sd_prototype"]
    rels = [p]
    q = p.replace("\\", "/")
    if "cloud_edge_sd_prototype/" in q:
        rels.append(q.split("cloud_edge_sd_prototype/", 1)[1])
    for rel in rels:
        for b in bases:
            if (b / rel).exists():
                return str(b / rel)
    return str(c)


def cond_to_uint8(arr: np.ndarray) -> np.ndarray:
    """CHW float[0,1] -> HxW uint8 grayscale (the condition is a replicated 1-channel map)."""
    g = np.clip(arr[0] * 255.0, 0, 255).astype(np.uint8)
    return g


def ssim_gray(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    return float(ssim_fn(a_u8, b_u8, data_range=255))


def png_bytes(cond_u8_rgb: Image.Image, level: int) -> bytes:
    buf = io.BytesIO()
    cond_u8_rgb.save(buf, format="PNG", compress_level=int(level))
    return buf.getvalue()


def jpeg_bytes(img: Image.Image, q: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q))
    return buf.getvalue()


def downsample_up(cond_u8: np.ndarray, factor: int) -> np.ndarray:
    """Downsample the (HxW uint8) condition then upsample back -- the codec's ds path."""
    if factor == 1:
        return cond_u8
    h, w = cond_u8.shape
    small = np.asarray(Image.fromarray(cond_u8).resize((w // factor, h // factor), Image.BILINEAR))
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR))


def measure_image(img_path: str) -> dict:
    out = {}
    raw_pil = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    for mode in MODES:
        extractor = EdgeConditionExtractor(mode=mode, image_size=IMAGE_SIZE,
                                           canny_low_threshold=100, canny_high_threshold=200)
        t0 = time.perf_counter()
        ref_cond = extractor(img_path)  # CHW float [0,1]
        extract_s = time.perf_counter() - t0
        ref_u8 = cond_to_uint8(ref_cond.numpy())
        edge_density = float((ref_u8 > 127).mean())
        ref_rgb = Image.fromarray(np.repeat(ref_u8[:, :, None], 3, axis=2))

        png = {}
        for lvl in PNG_LEVELS:
            for ds in DOWNSAMPLE:
                dec_u8 = downsample_up(ref_u8, ds)
                # payload = PNG of the (downsampled) condition at this compress level
                if ds == 1:
                    pay = png_bytes(ref_rgb, lvl)
                else:
                    h, w = ref_u8.shape
                    small = np.asarray(Image.fromarray(ref_u8).resize((w // ds, h // ds), Image.BILINEAR))
                    small_rgb = Image.fromarray(np.repeat(small[:, :, None], 3, axis=2))
                    pay = png_bytes(small_rgb, lvl)
                s = ssim_gray(ref_u8, dec_u8)
                png[f"L{lvl}_ds{ds}"] = {"payload_kb": len(pay) / 1024.0, "ssim": s}

        jpeg = []
        for q in JPEG_Q_GRID:
            pb = jpeg_bytes(ref_rgb, q)
            dec = np.asarray(Image.open(io.BytesIO(pb)).convert("RGB"))[:, :, 0]
            s = ssim_gray(ref_u8, dec)
            jpeg.append({"q": q, "payload_kb": len(pb) / 1024.0, "ssim": s})

        # cloud: upload the raw image, extract on the cloud side
        cloud_png_pay = png_bytes(raw_pil, 6)
        cloud = {"cloud_png": {"payload_kb": len(cloud_png_pay) / 1024.0, "ssim": 1.0}}
        for q in (50, 75):
            rb = jpeg_bytes(raw_pil, q)
            raw_dec = Image.open(io.BytesIO(rb)).convert("RGB")
            cloud_cond = extractor(raw_dec)
            cloud_u8 = cond_to_uint8(cloud_cond.numpy())
            cloud[f"cloud_jpeg_q{q}"] = {"payload_kb": len(rb) / 1024.0, "ssim": ssim_gray(ref_u8, cloud_u8)}

        out[mode] = {"edge_density": edge_density, "extract_s": extract_s,
                     "png": png, "jpeg": jpeg, "cloud": cloud}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="datasets/starter_cultural_patterns/paper_main_metadata.json")
    ap.add_argument("--out", default="real_runs/seq/action_effect_table.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(args.metadata, encoding="utf-8"))
    if args.limit:
        meta = meta[: args.limit]
    table = {}
    t0 = time.perf_counter()
    for i, m in enumerate(meta):
        path = _resolve_img(m["image"], ROOT)
        table[m["id"]] = {"category": m["category"], "actions": measure_image(path)}
        if (i + 1) % 10 == 0 or i + 1 == len(meta):
            print(f"[{i+1}/{len(meta)}] {m['id']} ({time.perf_counter()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"image_size": IMAGE_SIZE, "png_levels": PNG_LEVELS, "downsample": DOWNSAMPLE,
                 "jpeg_q_grid": JPEG_Q_GRID, "modes": MODES, "n_images": len(table)},
        "images": table,
    }
    json.dump(payload, open(out, "w"), indent=2)
    print(f"Saved {out} ({len(table)} images)")

    # --- validation vs the 3 branches in the shipped sample_stats.json ---
    print("\n=== validation: measured vs shipped sample_stats (canny, mean over images) ===")
    cy = [v["actions"]["canny"] for v in table.values()]
    png_l6 = np.mean([a["png"]["L6_ds1"]["payload_kb"] for a in cy])
    png_l6_ssim = np.mean([a["png"]["L6_ds1"]["ssim"] for a in cy])
    cpng = np.mean([a["cloud"]["cloud_png"]["payload_kb"] for a in cy])
    cjpeg = np.mean([a["cloud"]["cloud_jpeg_q75"]["payload_kb"] for a in cy])
    cjpeg_s = np.mean([a["cloud"]["cloud_jpeg_q75"]["ssim"] for a in cy])
    print(f"  cond_png_l6_ds1 : payload~{png_l6:.1f}KB ssim~{png_l6_ssim:.3f}  (shipped ~14KB / ssim~1.0)")
    print(f"  cloud_png       : payload~{cpng:.1f}KB ssim~1.0   (shipped ~235KB)")
    print(f"  cloud_jpeg_q75  : payload~{cjpeg:.1f}KB ssim~{cjpeg_s:.3f}  (shipped ~29KB / ssim~0.79)")


if __name__ == "__main__":
    main()
