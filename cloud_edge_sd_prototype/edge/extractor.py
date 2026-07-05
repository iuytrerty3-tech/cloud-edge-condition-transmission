from __future__ import annotations

from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image


ImageLike = Union[Image.Image, np.ndarray, torch.Tensor, str]


class EdgeConditionExtractor(torch.nn.Module):
    def __init__(
        self,
        mode: str = "canny",
        image_size: int = 512,
        canny_low_threshold: int = 100,
        canny_high_threshold: int = 200,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.image_size = image_size
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold

    @torch.no_grad()
    def forward(self, image: ImageLike) -> torch.Tensor:
        rgb = self._load_rgb(image)
        if self.mode == "canny":
            cond = self._extract_canny(rgb)
        elif self.mode == "skeleton":
            cond = self._extract_skeleton(rgb)
        else:
            raise ValueError(f"Unsupported extraction mode: {self.mode}")

        cond = np.repeat(cond[:, :, None], 3, axis=2)
        cond_tensor = torch.from_numpy(cond).permute(2, 0, 1).float() / 255.0
        return cond_tensor.contiguous()

    def _load_rgb(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, torch.Tensor):
            array = image.detach().cpu()
            if array.ndim == 4:
                array = array[0]
            if array.ndim == 3 and array.shape[0] in (1, 3):
                array = array.permute(1, 2, 0)
            array = array.numpy()
            if array.dtype != np.uint8:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array).convert("RGB")
        elif isinstance(image, np.ndarray):
            array = image
            if array.dtype != np.uint8:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                pil_image = Image.fromarray(array).convert("RGB")
            else:
                pil_image = Image.fromarray(array[:, :, :3]).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        pil_image = pil_image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return np.asarray(pil_image)

    def _extract_canny(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.canny_low_threshold, self.canny_high_threshold)
        return edges

    def _extract_skeleton(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        current = binary.copy()

        while True:
            eroded = cv2.erode(current, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(current, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            current = eroded
            if cv2.countNonZero(current) == 0:
                break

        return skeleton
