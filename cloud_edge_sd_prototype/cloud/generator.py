from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image


def _resolve_dtype(name: str) -> torch.dtype:
    value = name.lower()
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    return torch.float32


class CloudGenerator:
    def __init__(
        self,
        base_model_id: str,
        controlnet_model_id: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
    ) -> None:
        self.device = device
        requested_dtype = _resolve_dtype(torch_dtype)
        self.torch_dtype = requested_dtype if "cuda" in device and torch.cuda.is_available() else torch.float32

        if controlnet_model_id.lower() in {"from_unet", "__from_unet__", "auto_from_unet"}:
            base_pipe = StableDiffusionPipeline.from_pretrained(
                base_model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            )
            controlnet = ControlNetModel.from_unet(base_pipe.unet)
            pipe = StableDiffusionControlNetPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=base_pipe.unet,
                controlnet=controlnet,
                scheduler=base_pipe.scheduler,
                safety_checker=None,
                feature_extractor=base_pipe.feature_extractor,
                image_encoder=getattr(base_pipe, "image_encoder", None),
                requires_safety_checker=False,
            )
        else:
            controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=self.torch_dtype)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(device)

    @staticmethod
    def tensor_to_pil(cond_tensor: torch.Tensor) -> Image.Image:
        x = cond_tensor.detach().cpu()
        if x.ndim == 4:
            x = x[0]
        x = x.clamp(0, 1)
        x = (x * 255.0).round().to(torch.uint8)
        array = x.permute(1, 2, 0).numpy()
        return Image.fromarray(array)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        cond_tensor: torch.Tensor,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        cond_pil = self.tensor_to_pil(cond_tensor)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=cond_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]

    @torch.no_grad()
    def warmup(self, prompt: str = "warmup", seed: int = 0) -> None:
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(dummy)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=1,
            guidance_scale=1.0,
            generator=generator,
        )
