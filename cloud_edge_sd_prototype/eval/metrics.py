from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision.transforms import ToTensor


class DirectoryMetricEvaluator:
    def __init__(
        self,
        device: str = "cuda",
        clip_model_name_or_path: str = "openai/clip-vit-base-patch32",
        enable_lpips: bool = True,
    ) -> None:
        self.device = device if torch.cuda.is_available() or "cpu" in device else "cpu"
        self.to_tensor = ToTensor()
        self._init_fid()
        self._init_clip(clip_model_name_or_path)
        self._init_ssim()
        self._init_lpips(enable_lpips)

    def _init_fid(self) -> None:
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

    def _init_clip(self, model_name: str) -> None:
        from torchmetrics.multimodal.clip_score import CLIPScore

        self.clip_score = CLIPScore(model_name_or_path=model_name).to(self.device)

    def _init_ssim(self) -> None:
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def _init_lpips(self, enabled: bool) -> None:
        self.lpips_enabled = enabled
        self.lpips = None
        if not enabled:
            return
        import lpips

        self.lpips = lpips.LPIPS(net="alex").to(self.device)
        self.lpips.eval()

    def _load_image(self, path: str | Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.to_tensor(image).unsqueeze(0).to(self.device)

    @staticmethod
    def _match_pairs(reference_dir: str | Path, generated_dir: str | Path) -> list[tuple[Path, Path, str]]:
        reference_dir = Path(reference_dir)
        generated_dir = Path(generated_dir)
        generated_map = {p.name: p for p in generated_dir.glob("*.png")}
        pairs = []
        for reference_path in reference_dir.rglob("*"):
            if not reference_path.is_file():
                continue
            if reference_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if reference_path.name in generated_map:
                pairs.append((reference_path, generated_map[reference_path.name], reference_path.stem))
        return sorted(pairs, key=lambda item: item[0].name)

    @staticmethod
    def _match_pairs_from_manifest(prompt_manifest: str | Path, generated_dir: str | Path) -> tuple[list[tuple[Path, Path, str]], dict[str, str]]:
        generated_dir = Path(generated_dir)
        generated_map = {p.stem: p for p in generated_dir.glob("*.png")}
        manifest = DirectoryMetricEvaluator._load_manifest_entries(prompt_manifest)
        pairs: list[tuple[Path, Path, str]] = []
        prompts: dict[str, str] = {}
        for item in manifest:
            sample_id = str(item["id"])
            generated_path = generated_map.get(f"{sample_id}_generated") or generated_map.get(sample_id)
            reference_path = Path(item["image"])
            if generated_path is None or not reference_path.exists():
                continue
            pairs.append((reference_path, generated_path, sample_id))
            prompts[sample_id] = item["prompt"]
        return pairs, prompts

    def evaluate(
        self,
        reference_dir: str | Path,
        generated_dir: str | Path,
        prompt_manifest: str | Path | None = None,
    ) -> dict:
        prompts = {}
        if prompt_manifest is not None:
            pairs, prompts = self._match_pairs_from_manifest(prompt_manifest, generated_dir)
        else:
            pairs = self._match_pairs(reference_dir, generated_dir)
        if not pairs:
            raise ValueError("No matched image pairs found for evaluation")

        clip_scores: list[float] = []
        ssim_scores: list[float] = []
        lpips_scores: list[float] = []

        self.fid.reset()
        self.ssim.reset()

        for reference_path, generated_path, sample_id in pairs:
            real = self._load_image(reference_path)
            fake = self._load_image(generated_path)
            self.fid.update(real, real=True)
            self.fid.update(fake, real=False)
            ssim_value = self.ssim(fake, real).detach().item()
            ssim_scores.append(ssim_value)

            if self.lpips is not None:
                lpips_value = self.lpips(fake * 2 - 1, real * 2 - 1).mean().detach().item()
                lpips_scores.append(lpips_value)

            prompt = prompts.get(sample_id) or prompts.get(generated_path.stem) or prompts.get(generated_path.name)
            if prompt:
                clip_value = self.clip_score(fake.squeeze(0), prompt).detach().item()
                clip_scores.append(clip_value)

        metrics = {
            "num_pairs": len(pairs),
            "fid": float(self.fid.compute().detach().item()),
            "ssim": float(sum(ssim_scores) / max(len(ssim_scores), 1)),
        }
        if clip_scores:
            metrics["clip_score"] = float(sum(clip_scores) / len(clip_scores))
        if lpips_scores:
            metrics["lpips"] = float(sum(lpips_scores) / len(lpips_scores))
        return metrics

    @staticmethod
    def _load_manifest_entries(prompt_manifest: str | Path) -> list[dict]:
        prompt_manifest = Path(prompt_manifest)
        if prompt_manifest.suffix.lower() == ".json":
            return json.loads(prompt_manifest.read_text(encoding="utf-8"))

        items: list[dict] = []
        for line in prompt_manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items

    @staticmethod
    def _load_prompt_manifest(prompt_manifest: str | Path) -> dict[str, str]:
        return {str(item["id"]): item["prompt"] for item in DirectoryMetricEvaluator._load_manifest_entries(prompt_manifest)}

    @staticmethod
    def save_metrics(path: str | Path, metrics: dict) -> None:
        Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def list_images(directory: str | Path) -> Iterable[Path]:
        return sorted(Path(directory).glob("*.png"))
