from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_edge_sd_prototype.edge.codec import PNGConditionCodec
from cloud_edge_sd_prototype.edge.extractor import EdgeConditionExtractor
from cloud_edge_sd_prototype.utils.runtime import simulate_network_time_seconds

MANIFEST_PATH = PROJECT_ROOT / "datasets" / "starter_cultural_patterns" / "manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "real_runs" / "parameterized_pilot_benchmark"
DEFAULT_SAMPLE_IDS = [
    "porcelain_001",
    "porcelain_002",
    "porcelain_003",
    "porcelain_004",
    "embroidery_001",
    "papercut_001",
    "papercut_002",
]
DEFAULT_BANDWIDTHS = [1.0, 2.0, 3.0, 5.0, 10.0]
RTT_MS = 20.0
REPEATS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parameterized benchmark on the 7-image paper pilot set")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    parser.add_argument("--sample-ids", nargs="*", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--sample-file", type=str, default="")
    parser.add_argument("--bandwidths", nargs="*", type=float, default=DEFAULT_BANDWIDTHS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    return parser.parse_args()


def load_manifest(path: str | Path) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def timed(fn, repeats: int = REPEATS):
    elapsed = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        elapsed.append(time.perf_counter() - t0)
    return result, float(np.mean(elapsed))


def ssim_scalar(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = a[0].detach().cpu().numpy()
    b_np = b[0].detach().cpu().numpy()
    mu_a, mu_b = a_np.mean(), b_np.mean()
    sig_a, sig_b = a_np.std(), b_np.std()
    sig_ab = ((a_np - mu_a) * (b_np - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a**2 + sig_b**2 + c2)
    return float(num / den)


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)


def encode_rgb_jpeg(image: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_rgb_png(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG", compress_level=9)
    return buf.getvalue()


def maybe_downsample(cond: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return cond
    h, w = cond.shape[-2:]
    return F.interpolate(cond.unsqueeze(0), size=(h // factor, w // factor), mode="bilinear", align_corners=False).squeeze(0)


def encode_cond_jpeg(cond: torch.Tensor, quality: int, downsample_factor: int) -> bytes:
    x = maybe_downsample(cond, downsample_factor).detach().cpu().clamp(0, 1)
    x_u8 = (x * 255).round().to(torch.uint8)
    img = Image.fromarray(np.transpose(x_u8.numpy(), (1, 2, 0)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_cond_jpeg(payload: bytes, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    img = Image.open(BytesIO(payload)).convert("RGB")
    arr = np.asarray(img).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    if tensor.shape[-2:] != size:
        tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return tensor


def encode_edgeaware_png(cond: torch.Tensor, level: int, downsample_factor: int) -> bytes:
    gray = cond[:1].detach().cpu().clamp(0, 1)
    pooled = gray
    if downsample_factor > 1:
        pooled = F.max_pool2d(gray.unsqueeze(0), kernel_size=downsample_factor, stride=downsample_factor).squeeze(0)
    x = pooled.repeat(3, 1, 1)
    arr = (x * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=level)
    return buf.getvalue()


def decode_edgeaware_png(payload: bytes, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    img = Image.open(BytesIO(payload)).convert("RGB")
    arr = np.asarray(img).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    tensor = tensor[:1]
    if tensor.shape[-2:] != size:
        tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    tensor = (tensor > 0.5).float().repeat(3, 1, 1)
    return tensor


def build_action_specs() -> list[dict]:
    specs = []
    for level in [1, 3, 6, 9]:
        specs.append(
            {
                "strategy": f"cond_png_l{level}_ds1",
                "branch": "cond_png",
                "png_level": level,
                "downsample_factor": 1,
            }
        )
    for level in [3, 6, 9]:
        specs.append(
            {
                "strategy": f"cond_png_l{level}_ds2",
                "branch": "cond_png",
                "png_level": level,
                "downsample_factor": 2,
            }
        )
    for level in [3, 6, 9]:
        specs.append(
            {
                "strategy": f"cond_edgepng_l{level}_ds2",
                "branch": "cond_edgepng",
                "png_level": level,
                "downsample_factor": 2,
            }
        )
    for quality in [60, 75, 85, 95]:
        specs.append(
            {
                "strategy": f"cond_jpeg_q{quality}_ds1",
                "branch": "cond_jpeg",
                "jpeg_quality": quality,
                "downsample_factor": 1,
            }
        )
    for quality in [75, 85]:
        specs.append(
            {
                "strategy": f"cond_jpeg_q{quality}_ds2",
                "branch": "cond_jpeg",
                "jpeg_quality": quality,
                "downsample_factor": 2,
            }
        )
    for quality in [60, 75, 85]:
        specs.append(
            {
                "strategy": f"cloud_jpeg_q{quality}",
                "branch": "cloud_jpeg",
                "jpeg_quality": quality,
                "downsample_factor": 1,
            }
        )
    specs.append({"strategy": "cloud_png", "branch": "cloud_png", "downsample_factor": 1})
    return specs


def run_condition_png(cond: torch.Tensor, extract_time: float, bw_mbps: float, action: dict, repeats: int) -> dict:
    codec = PNGConditionCodec(
        downsample_factor=int(action["downsample_factor"]),
        png_compress_level=int(action["png_level"]),
    )
    (payload, meta), enc_time = timed(lambda: codec.encode(cond), repeats=repeats)
    rec, dec_time = timed(lambda: codec.decode(payload, meta), repeats=repeats)
    cond_ssim = ssim_scalar(cond, rec)
    tx_time = simulate_network_time_seconds(len(payload), bw_mbps, RTT_MS)
    return {
        "payload_kb": len(payload) / 1024.0,
        "extract_time_sec": extract_time,
        "encode_time_sec": enc_time,
        "network_time_sec": tx_time,
        "decode_time_sec": dec_time,
        "total_time_sec": extract_time + enc_time + tx_time + dec_time,
        "condition_ssim": cond_ssim,
    }


def run_condition_jpeg(cond: torch.Tensor, extract_time: float, bw_mbps: float, action: dict, repeats: int) -> dict:
    payload, enc_time = timed(
        lambda: encode_cond_jpeg(cond, int(action["jpeg_quality"]), int(action["downsample_factor"])),
        repeats=repeats,
    )
    rec, dec_time = timed(lambda: decode_cond_jpeg(payload), repeats=repeats)
    cond_ssim = ssim_scalar(cond, rec)
    tx_time = simulate_network_time_seconds(len(payload), bw_mbps, RTT_MS)
    return {
        "payload_kb": len(payload) / 1024.0,
        "extract_time_sec": extract_time,
        "encode_time_sec": enc_time,
        "network_time_sec": tx_time,
        "decode_time_sec": dec_time,
        "total_time_sec": extract_time + enc_time + tx_time + dec_time,
        "condition_ssim": cond_ssim,
    }


def run_condition_edgepng(cond: torch.Tensor, extract_time: float, bw_mbps: float, action: dict, repeats: int) -> dict:
    payload, enc_time = timed(
        lambda: encode_edgeaware_png(cond, int(action["png_level"]), int(action["downsample_factor"])),
        repeats=repeats,
    )
    rec, dec_time = timed(lambda: decode_edgeaware_png(payload), repeats=repeats)
    cond_ssim = ssim_scalar(cond, rec)
    tx_time = simulate_network_time_seconds(len(payload), bw_mbps, RTT_MS)
    return {
        "payload_kb": len(payload) / 1024.0,
        "extract_time_sec": extract_time,
        "encode_time_sec": enc_time,
        "network_time_sec": tx_time,
        "decode_time_sec": dec_time,
        "total_time_sec": extract_time + enc_time + tx_time + dec_time,
        "condition_ssim": cond_ssim,
    }


def run_cloud_branch(
    image: Image.Image,
    cond_ref: torch.Tensor,
    extractor: EdgeConditionExtractor,
    bw_mbps: float,
    action: dict,
    repeats: int,
) -> dict:
    if action["branch"] == "cloud_jpeg":
        payload, enc_time = timed(lambda: encode_rgb_jpeg(image, int(action["jpeg_quality"])), repeats=repeats)
    else:
        payload, enc_time = timed(lambda: encode_rgb_png(image), repeats=repeats)

    def extract_from_payload():
        decoded = Image.open(BytesIO(payload)).convert("RGB")
        return extractor(decoded)

    cond_rec, extract_time = timed(extract_from_payload, repeats=repeats)
    cond_ssim = ssim_scalar(cond_ref, cond_rec)
    tx_time = simulate_network_time_seconds(len(payload), bw_mbps, RTT_MS)
    return {
        "payload_kb": len(payload) / 1024.0,
        "extract_time_sec": extract_time,
        "encode_time_sec": enc_time,
        "network_time_sec": tx_time,
        "decode_time_sec": 0.0,
        "total_time_sec": extract_time + enc_time + tx_time,
        "condition_ssim": cond_ssim,
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], float(row["bandwidth_mbps"]))].append(row)

    aggregated = []
    for (strategy, bw), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        aggregated.append(
            {
                "strategy": strategy,
                "mode": "canny",
                "downsample_factor": int(items[0]["downsample_factor"]),
                "bandwidth_mbps": bw,
                "steps": 0,
                "guidance_scale": 0.0,
                "num_samples": len(items),
                "total_time_mean": float(np.mean([item["total_time_sec"] for item in items])),
                "total_time_std": float(np.std([item["total_time_sec"] for item in items])),
                "payload_kb_mean": float(np.mean([item["payload_kb"] for item in items])),
                "condition_ssim_mean": float(np.mean([item["condition_ssim"] for item in items])),
                "extract_time_mean": float(np.mean([item["extract_time_sec"] for item in items])),
                "encode_time_mean": float(np.mean([item["encode_time_sec"] for item in items])),
                "network_time_mean": float(np.mean([item["network_time_sec"] for item in items])),
                "decode_time_mean": float(np.mean([item["decode_time_sec"] for item in items])),
            }
        )
    return aggregated


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    selected_ids = list(args.sample_ids)
    if args.sample_file:
        selected_ids = json.loads(Path(args.sample_file).read_text(encoding="utf-8"))
    keep_ids = set(selected_ids)
    pilot_items = [item for item in manifest if item["id"] in keep_ids]
    pilot_items.sort(key=lambda item: selected_ids.index(item["id"]))

    extractor = EdgeConditionExtractor(mode="canny", image_size=512)
    action_specs = build_action_specs()
    all_rows: list[dict] = []

    for item in pilot_items:
        image_path = Path(item["image"])
        if not image_path.exists():
            print(f"missing image: {image_path}")
            continue
        image = load_rgb_image(image_path)
        cond_ref, extract_time = timed(lambda: extractor(image), repeats=args.repeats)
        cond_density = float((cond_ref > 0.5).float().mean().item()) * 100.0

        for bw in args.bandwidths:
            for action in action_specs:
                if action["branch"] == "cond_png":
                    result = run_condition_png(cond_ref, extract_time, bw, action, repeats=args.repeats)
                elif action["branch"] == "cond_edgepng":
                    result = run_condition_edgepng(cond_ref, extract_time, bw, action, repeats=args.repeats)
                elif action["branch"] == "cond_jpeg":
                    result = run_condition_jpeg(cond_ref, extract_time, bw, action, repeats=args.repeats)
                else:
                    result = run_cloud_branch(image, cond_ref, extractor, bw, action, repeats=args.repeats)

                row = {
                    "sample_id": item["id"],
                    "id": item["id"],
                    "category": item["category"],
                    "strategy": action["strategy"],
                    "branch": action["branch"],
                    "png_level": action.get("png_level"),
                    "jpeg_quality": action.get("jpeg_quality"),
                    "downsample_factor": int(action["downsample_factor"]),
                    "bandwidth_mbps": float(bw),
                    "edge_density": round(cond_density, 6),
                    "mode": "canny",
                    "steps": 0,
                    "guidance_scale": 0.0,
                    **result,
                }
                all_rows.append(row)
                print(
                    f"{item['id']:14s}  bw={bw:>4.1f}  {action['strategy']:18s}  "
                    f"lat={row['total_time_sec']:.4f}s  kb={row['payload_kb']:.2f}  ssim={row['condition_ssim']:.4f}"
                )

    aggregated = aggregate_rows(all_rows)
    (output_dir / "action_catalog.json").write_text(json.dumps(action_specs, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "all_cases.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "sample_stats.json").write_text(json.dumps({"samples": all_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "aggregated_results.json").write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")

    ranking = defaultdict(list)
    for row in aggregated:
        ranking[row["strategy"]].append(float(row["total_time_mean"]))
    avg_ranking = sorted(
        [{"strategy": key, "mean_total_time": float(np.mean(vals))} for key, vals in ranking.items()],
        key=lambda item: item["mean_total_time"],
    )
    (output_dir / "ranking_summary.json").write_text(
        json.dumps({"average_latency_ranking": avg_ranking}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
