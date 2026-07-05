from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_edge_sd_prototype.cloud import CloudGenerator
from cloud_edge_sd_prototype.config import CloudConfig, CodecConfig, EdgeConfig, ExperimentConfig
from cloud_edge_sd_prototype.edge import EdgeConditionExtractor, PNGConditionCodec
from cloud_edge_sd_prototype.utils import format_bytes, measure_cuda_seconds, save_json, simulate_network_time_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud-edge Stable Diffusion prototype")
    parser.add_argument("--input", type=str, required=True, help="Path to reference image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--mode", type=str, default="canny", choices=["canny", "skeleton"])
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--bandwidth-mbps", type=float, default=5.0)
    parser.add_argument("--rtt-ms", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model-id", type=str, default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_cfg = EdgeConfig(mode=args.mode, image_size=args.image_size)
    codec_cfg = CodecConfig(downsample_factor=args.downsample_factor)
    cloud_cfg = CloudConfig(
        base_model_id=args.base_model_id,
        controlnet_model_id=args.controlnet_model_id,
        device=args.device,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    exp_cfg = ExperimentConfig(
        bandwidth_mbps=args.bandwidth_mbps,
        rtt_ms=args.rtt_ms,
        seed=args.seed,
    )

    edge_extractor = EdgeConditionExtractor(
        mode=edge_cfg.mode,
        image_size=edge_cfg.image_size,
        canny_low_threshold=edge_cfg.canny_low_threshold,
        canny_high_threshold=edge_cfg.canny_high_threshold,
    )
    codec = PNGConditionCodec(
        downsample_factor=codec_cfg.downsample_factor,
        png_compress_level=codec_cfg.png_compress_level,
    )

    t0 = time.perf_counter()
    cond_tensor = edge_extractor(args.input)
    edge_extract_time = time.perf_counter() - t0

    raw_tensor_bytes = codec.tensor_nbytes(cond_tensor)

    t1 = time.perf_counter()
    payload, meta = codec.encode(cond_tensor)
    edge_codec_time = time.perf_counter() - t1

    payload_bytes = len(payload)
    network_time = simulate_network_time_seconds(
        num_bytes=payload_bytes,
        bandwidth_mbps=exp_cfg.bandwidth_mbps,
        rtt_ms=exp_cfg.rtt_ms,
    )

    t2 = time.perf_counter()
    decoded_cond = codec.decode(payload, meta, device="cpu")
    cloud_decode_time = time.perf_counter() - t2

    generator = CloudGenerator(
        base_model_id=cloud_cfg.base_model_id,
        controlnet_model_id=cloud_cfg.controlnet_model_id,
        device=cloud_cfg.device,
        torch_dtype=cloud_cfg.torch_dtype,
    )

    if torch.cuda.is_available() and "cuda" in cloud_cfg.device:
        generator.warmup()

    def _generate():
        return generator.generate(
            prompt=args.prompt,
            cond_tensor=decoded_cond.to(cloud_cfg.device),
            negative_prompt=args.negative_prompt,
            num_inference_steps=cloud_cfg.num_inference_steps,
            guidance_scale=cloud_cfg.guidance_scale,
            seed=exp_cfg.seed,
        )

    output_image, cloud_gen_time = measure_cuda_seconds(_generate, device=cloud_cfg.device)

    condition_image = generator.tensor_to_pil(cond_tensor)
    condition_image.save(output_dir / "condition.png")
    output_image.save(output_dir / "generated.png")

    compression_ratio = raw_tensor_bytes / max(payload_bytes, 1)
    bandwidth_saving_ratio = 1.0 - (payload_bytes / max(raw_tensor_bytes, 1))

    stats = {
        "edge_extract_time_sec": edge_extract_time,
        "edge_codec_time_sec": edge_codec_time,
        "network_time_sec": network_time,
        "cloud_decode_time_sec": cloud_decode_time,
        "cloud_gen_time_sec": cloud_gen_time,
        "total_time_sec": edge_extract_time + edge_codec_time + network_time + cloud_decode_time + cloud_gen_time,
        "raw_tensor_size": format_bytes(raw_tensor_bytes),
        "payload_size": format_bytes(payload_bytes),
        "compression_ratio": compression_ratio,
        "bandwidth_saving_ratio": bandwidth_saving_ratio,
        "mode": args.mode,
        "image_size": args.image_size,
        "downsample_factor": args.downsample_factor,
        "bandwidth_mbps": args.bandwidth_mbps,
        "steps": args.steps,
        "prompt": args.prompt,
    }

    save_json(output_dir / "stats.json", stats)
    print(f"Condition saved to: {output_dir / 'condition.png'}")
    print(f"Generated image saved to: {output_dir / 'generated.png'}")
    print(f"Stats saved to: {output_dir / 'stats.json'}")
    print(stats)


if __name__ == "__main__":
    main()
