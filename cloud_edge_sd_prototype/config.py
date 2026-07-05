from dataclasses import dataclass


@dataclass(slots=True)
class EdgeConfig:
    mode: str = "canny"
    image_size: int = 512
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200


@dataclass(slots=True)
class CodecConfig:
    downsample_factor: int = 1
    png_compress_level: int = 9


@dataclass(slots=True)
class CloudConfig:
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"
    device: str = "cuda"
    torch_dtype: str = "float16"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


@dataclass(slots=True)
class ExperimentConfig:
    bandwidth_mbps: float = 5.0
    rtt_ms: float = 20.0
    seed: int = 42
