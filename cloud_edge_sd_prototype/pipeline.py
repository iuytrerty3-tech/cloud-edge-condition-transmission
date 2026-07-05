from __future__ import annotations

import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .cloud import CloudGenerator
from .edge import EdgeConditionExtractor, PNGConditionCodec
from .utils import format_bytes, measure_cuda_seconds, save_json, simulate_network_time_seconds


@dataclass(slots=True)
class RunArtifacts:
    sample_id: str
    strategy: str
    condition_path: Optional[Path]
    generated_path: Path
    stats_path: Path


class CloudEdgeRunner:
    def __init__(
        self,
        edge_extractor: EdgeConditionExtractor,
        codec: PNGConditionCodec,
        generator: CloudGenerator,
        bandwidth_mbps: float,
        rtt_ms: float,
    ) -> None:
        self.edge_extractor = edge_extractor
        self.codec = codec
        self.generator = generator
        self.bandwidth_mbps = bandwidth_mbps
        self.rtt_ms = rtt_ms

    @staticmethod
    def _load_pil(image_path: str | Path, image_size: int) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    @staticmethod
    def _encode_pil(image: Image.Image, image_format: str = "PNG", jpeg_quality: int = 95) -> bytes:
        buffer = BytesIO()
        save_kwargs = {}
        if image_format.upper() == "JPEG":
            save_kwargs["quality"] = jpeg_quality
        image.save(buffer, format=image_format.upper(), **save_kwargs)
        return buffer.getvalue()

    @staticmethod
    def _decode_pil(payload: bytes) -> Image.Image:
        return Image.open(BytesIO(payload)).convert("RGB")

    def _stats_common(self, payload_bytes: int, total_time_sec: float) -> dict:
        return {
            "payload_size": format_bytes(payload_bytes),
            "bandwidth_mbps": self.bandwidth_mbps,
            "rtt_ms": self.rtt_ms,
            "network_time_sec": simulate_network_time_seconds(payload_bytes, self.bandwidth_mbps, self.rtt_ms),
            "total_time_sec": total_time_sec,
        }

    def run_ours(
        self,
        image_path: str | Path,
        prompt: str,
        output_dir: str | Path,
        sample_id: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> RunArtifacts:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        cond_tensor = self.edge_extractor(str(image_path))
        edge_extract_time = time.perf_counter() - t0

        raw_tensor_bytes = self.codec.tensor_nbytes(cond_tensor)

        t1 = time.perf_counter()
        payload, meta = self.codec.encode(cond_tensor)
        edge_codec_time = time.perf_counter() - t1

        payload_bytes = len(payload)
        network_time = simulate_network_time_seconds(payload_bytes, self.bandwidth_mbps, self.rtt_ms)

        t2 = time.perf_counter()
        decoded_cond = self.codec.decode(payload, meta, device="cpu")
        cloud_decode_time = time.perf_counter() - t2

        def _generate():
            return self.generator.generate(
                prompt=prompt,
                cond_tensor=decoded_cond.to(self.generator.device),
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        output_image, cloud_gen_time = measure_cuda_seconds(_generate, device=self.generator.device)
        condition_image = self.generator.tensor_to_pil(cond_tensor)
        condition_path = output_dir / f"{sample_id}_condition.png"
        generated_path = output_dir / f"{sample_id}_generated.png"
        stats_path = output_dir / f"{sample_id}_stats.json"
        condition_image.save(condition_path)
        output_image.save(generated_path)

        total_time_sec = edge_extract_time + edge_codec_time + network_time + cloud_decode_time + cloud_gen_time
        stats = {
            "sample_id": sample_id,
            "strategy": "ours",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "edge_extract_time_sec": edge_extract_time,
            "edge_codec_time_sec": edge_codec_time,
            "cloud_decode_time_sec": cloud_decode_time,
            "cloud_gen_time_sec": cloud_gen_time,
            "raw_tensor_size": format_bytes(raw_tensor_bytes),
            "compression_ratio": raw_tensor_bytes / max(payload_bytes, 1),
            "bandwidth_saving_ratio": 1.0 - (payload_bytes / max(raw_tensor_bytes, 1)),
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        stats.update(self._stats_common(payload_bytes, total_time_sec))
        save_json(stats_path, stats)
        return RunArtifacts(sample_id, "ours", condition_path, generated_path, stats_path)

    def run_cloud_condition(
        self,
        image_path: str | Path,
        prompt: str,
        output_dir: str | Path,
        sample_id: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        image_format: str = "PNG",
        jpeg_quality: int = 95,
    ) -> RunArtifacts:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pil_image = self._load_pil(image_path, self.edge_extractor.image_size)

        t0 = time.perf_counter()
        payload = self._encode_pil(pil_image, image_format=image_format, jpeg_quality=jpeg_quality)
        encode_time = time.perf_counter() - t0

        payload_bytes = len(payload)
        network_time = simulate_network_time_seconds(payload_bytes, self.bandwidth_mbps, self.rtt_ms)

        t1 = time.perf_counter()
        cloud_image = self._decode_pil(payload)
        cond_tensor = self.edge_extractor(cloud_image)
        cloud_extract_time = time.perf_counter() - t1

        def _generate():
            return self.generator.generate(
                prompt=prompt,
                cond_tensor=cond_tensor.to(self.generator.device),
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        output_image, cloud_gen_time = measure_cuda_seconds(_generate, device=self.generator.device)
        condition_image = self.generator.tensor_to_pil(cond_tensor)
        condition_path = output_dir / f"{sample_id}_condition.png"
        generated_path = output_dir / f"{sample_id}_generated.png"
        stats_path = output_dir / f"{sample_id}_stats.json"
        condition_image.save(condition_path)
        output_image.save(generated_path)

        total_time_sec = encode_time + network_time + cloud_extract_time + cloud_gen_time
        raw_upload_bytes = Path(image_path).stat().st_size
        stats = {
            "sample_id": sample_id,
            "strategy": "cloud_condition",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "edge_codec_time_sec": encode_time,
            "cloud_extract_time_sec": cloud_extract_time,
            "cloud_gen_time_sec": cloud_gen_time,
            "image_format": image_format,
            "jpeg_quality": jpeg_quality,
            "raw_file_size": format_bytes(raw_upload_bytes),
            "file_compression_ratio": raw_upload_bytes / max(payload_bytes, 1),
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        stats.update(self._stats_common(payload_bytes, total_time_sec))
        save_json(stats_path, stats)
        return RunArtifacts(sample_id, "cloud_condition", condition_path, generated_path, stats_path)

    def run_experiment(
        self,
        strategy: str,
        image_path: str | Path,
        prompt: str,
        output_dir: str | Path,
        sample_id: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        image_format: str = "PNG",
        jpeg_quality: int = 95,
    ) -> RunArtifacts:
        if strategy == "ours":
            return self.run_ours(
                image_path=image_path,
                prompt=prompt,
                output_dir=output_dir,
                sample_id=sample_id,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        if strategy == "cloud_condition":
            return self.run_cloud_condition(
                image_path=image_path,
                prompt=prompt,
                output_dir=output_dir,
                sample_id=sample_id,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
        raise ValueError(f"Unsupported strategy: {strategy}")

    @staticmethod
    def load_stats(stats_path: str | Path) -> dict:
        return json.loads(Path(stats_path).read_text(encoding="utf-8"))
