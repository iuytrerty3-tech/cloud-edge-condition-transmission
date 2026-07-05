#!/usr/bin/env python3
"""Runtime shim: restore transformers<5 CLIP get_image_features/get_text_features behavior.

transformers 5.x changed CLIPModel.get_image_features/get_text_features to return a
BaseModelOutputWithPooling object instead of the projected-embedding tensor that
torchmetrics 1.9 CLIPScore expects (it calls `.norm(...)` on the result). The canonical
CLIP embedding is visual_projection(vision_model(x).pooler_output) (and the text analog) --
exactly what transformers 4.x returned as a tensor. We re-apply the projection here so the
standard CLIP score is computed unchanged, WITHOUT editing compute_generation_metrics.py.

Usage: python experiments/_clip_shim_runner.py <args passed through to the metrics script>
"""
import runpy
import sys
from pathlib import Path

from transformers import CLIPModel


def _image_features(self, pixel_values=None, *args, **kwargs):
    out = self.vision_model(pixel_values=pixel_values)
    return self.visual_projection(out.pooler_output)


def _text_features(self, input_ids=None, attention_mask=None, *args, **kwargs):
    out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
    return self.text_projection(out.pooler_output)


CLIPModel.get_image_features = _image_features
CLIPModel.get_text_features = _text_features

script = str(Path(__file__).resolve().parent / "compute_generation_metrics.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
