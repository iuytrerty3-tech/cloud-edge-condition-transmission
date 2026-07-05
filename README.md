# Cloud-Edge Cultural Pattern Generation via Lossless Condition Transmission


The edge device extracts a ControlNet structural condition (Canny edge / skeleton map) from a reference image, compresses it losslessly with PNG, and uploads only this compact payload. The cloud reconstructs the condition bit-identically and runs the full Stable Diffusion + ControlNet pipeline, so the generated image is exactly the one that would have been produced from the uncompressed reference. A hybrid-action reinforcement-learning scheduler selects the transmission branch (condition type, PNG level, downsampling) according to bandwidth and image complexity.

## Repository layout

```text
cloud_edge_sd_prototype/     importable package
  edge/                      condition extraction (extractor.py) and codecs (codec.py)
  cloud/                     Stable Diffusion + ControlNet generation
  eval/                      metrics
  pipeline.py                end-to-end pipeline
envs/                        scheduling environments
experiments/                 training, benchmarking, and reproduction scripts
data_tools/                  dataset download / expansion utilities
configs/                     experiment grids and manifests
datasets/starter_cultural_patterns/
  images/                    112 cultural-pattern images (5 categories)
  paper_main_metadata.json   per-image category, prompt, id, source_url, relative path
real_runs/                   measurement records used by the paper (see below)
```

## Installation

```bash
pip install -r requirements.txt        # CPU: scheduling and codec experiments
pip install -r requirements_gpu.txt    # GPU: end-to-end Stable Diffusion generation
```

Scheduling/benchmark experiments run on CPU. End-to-end generation was measured on a single NVIDIA RTX 3070 Laptop GPU (torch 2.11 + cu128, diffusers 0.37.1).

## Dataset

`datasets/starter_cultural_patterns/` contains 112 images in five categories (blue-and-white porcelain, cultural clothing, paper-cutting, window-flower, artifact), collected from Wikimedia Commons; each record in `paper_main_metadata.json` carries its original `source_url`, and the images remain under their original licenses. The train/held-out split used in the paper is fixed by `experiments/fullreal_train_ids_v1.json` (81 ids) and `experiments/fullreal_eval_ids_v1.json` (31 ids).


```bash
# All main-text figures + a numbers.json holding every table entry
python experiments/rebuild_maintext_figures_realonly.py

# Per-branch payload / SSIM / encode-time statistics on the 31 held-out images
python experiments/reproduce_codec_conditions.py .
```

Outputs go to `real_runs/maintext_figures_realonly_v1/`.

To re-run the pipeline from scratch rather than replaying the records:

```bash
# Retrain the scheduler (CPU, ~2 min)
python experiments/train_hybrid_ddpg_scheduler.py \
    --sample-stats real_runs/unified_algorithm_benchmark_fullreal_split_v1/sample_stats.json \
    --output-dir real_runs/my_retrain

# Five-seed stability run
python experiments/run_multiseed_rl.py \
    --sample-stats real_runs/unified_algorithm_benchmark_fullreal_split_v1/sample_stats.json

# End-to-end generation and quality metrics (GPU required)
python experiments/run_end_to_end_generation.py --help
python experiments/compute_generation_metrics.py --help
```


## License

Code is released under the MIT License (see `LICENSE`). Dataset images originate from Wikimedia Commons and remain under their original licenses; see the per-image `source_url` fields.
