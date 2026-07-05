from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class PNGConditionCodec:
    def __init__(self, downsample_factor: int = 1, png_compress_level: int = 9) -> None:
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1")
        self.downsample_factor = downsample_factor
        self.png_compress_level = png_compress_level

    @staticmethod
    def tensor_nbytes(x: torch.Tensor) -> int:
        return x.numel() * x.element_size()

    @staticmethod
    def _to_uint8_image(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().clamp(0, 1)
        return (x * 255.0).round().to(torch.uint8)

    def encode(self, cond_tensor: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
        x = cond_tensor.detach().cpu()
        if x.ndim == 4:
            x = x[0]
        if x.ndim != 3:
            raise ValueError("Expected CHW or BCHW condition tensor")

        original_shape = tuple(x.shape)
        if self.downsample_factor > 1:
            h, w = x.shape[-2:]
            x = F.interpolate(
                x.unsqueeze(0),
                size=(h // self.downsample_factor, w // self.downsample_factor),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        x_u8 = self._to_uint8_image(x)
        image = Image.fromarray(np.transpose(x_u8.numpy(), (1, 2, 0)))
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=self.png_compress_level)

        payload = buffer.getvalue()
        meta = {
            "original_shape": original_shape,
            "encoded_shape": tuple(x.shape),
        }
        return payload, meta

    def decode(self, payload: bytes, meta: dict[str, Any], device: str = "cpu") -> torch.Tensor:
        image = Image.open(BytesIO(payload)).convert("RGB")
        array = np.asarray(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

        original_shape = meta["original_shape"]
        if tuple(tensor.shape) != tuple(original_shape):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=original_shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor.to(device)
