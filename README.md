# Cloud-Edge Cultural Pattern Generation via Lossless Condition Transmission

The edge device extracts a ControlNet structural condition (Canny edge / skeleton map) from a reference image, compresses it losslessly with PNG, and uploads only this compact payload. The cloud reconstructs the condition bit-identically and runs the full Stable Diffusion + ControlNet pipeline, so the generated image is exactly the one that would have been produced from the uncompressed reference. A hybrid-action reinforcement-learning scheduler selects the transmission branch (condition type, PNG level, downsampling) according to bandwidth and image complexity.

## Repository layout

```text
cloud_edge_sd_prototype/     importable package
  edge/                      condition extraction (extractor.py) and codecs (codec.py)
  cloud/                     Stable Diffusion + ControlNet generation
  eval/                      metrics
  pipeline.py                end-to-end pipeline
experiments/                 training, benchmarking, and reproduction scripts
data_tools/                  dataset download / expansion utilities
configs/                     experiment grids and manifests
datasets/starter_cultural_patterns/
  images/                    112 cultural-pattern images in 6 category folders
  paper_main_metadata.json   per-image category, prompt, id, source_url, relative path
real_runs/                   measurement records used by the paper (see below)
figures/                     figure assets used by the manuscript
```

## Installation

```bash
pip install -r requirements.txt        # CPU: scheduling and codec experiments
pip install -r requirements_gpu.txt    # GPU: end-to-end Stable Diffusion generation
```

Scheduling/benchmark experiments run on CPU. End-to-end generation was measured on a single NVIDIA RTX 3070 Laptop GPU (torch 2.11 + cu128, diffusers 0.37.1).

## Dataset

`datasets/starter_cultural_patterns/` contains 112 images in six category folders (blue-and-white porcelain 25, artifact-object 31, artifact-pattern 22, cultural clothing 14, paper-cutting 11, window-flower 9), collected from Wikimedia Commons; each record in `paper_main_metadata.json` carries its original `source_url`, and the images remain under their original licenses. The train/held-out split used in the paper is fixed by `experiments/fullreal_train_ids_v1.json` (81 ids) and `experiments/fullreal_eval_ids_v1.json` (31 ids).

## Reproducing the paper

Every number in Tables 2, 3, and 4 and every main-text figure can be regenerated from the shipped measurement records (Table 5, the end-to-end generation quality table, is stored in `real_runs/e2e_generation_v1/generation_metrics.json`):

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

## Measurement records (`real_runs/`)

- `unified_algorithm_benchmark_fullreal_split_v1/` — per-sample benchmark records (all strategies, 0.8–10 Mbps grid); source of Table 2/3.
- `unified_baselines_fullreal_split_v1/` — per-sample records of the rule/threshold/random baselines.
- `hybrid_ddpg_fullreal_split_v1/` — trained policy and policy/action logs (`policy.json`: per-state actor state vector and selected branch; `decision_map.json`), plus the training curve.
- `hybrid_ddpg_multiseed/` — five-seed stability summary.
- `e2e_generation_v1/` — end-to-end Stable Diffusion + ControlNet outputs and quality metrics; `generation_metrics.json` is the source of Table 5.
- `seq/action_effect_table.json` — deterministic per-image, per-branch codec measurements.
- `maintext_figures_realonly_v1/` — regenerated figures and `numbers.json`.

The uplink is modeled analytically as `t_net = 20 ms RTT + payload / bandwidth` over a fixed bandwidth grid (1, 2, 3, 5, 10 Mbps in the main tables; 0.8/1.5/8 Mbps in the supporting sweeps); extraction, encoding, and decoding times are per-sample wall-clock measurements. **No captured network traces are used, and no jitter or packet-loss term appears anywhere in the benchmark or in the scheduler state.** The scheduler state vector is `[bandwidth, payload complexity, extraction complexity, encoding complexity, edge density, category one-hot]` (see `build_state_vector_from_profile` in `experiments/train_hybrid_ddpg_scheduler.py`).

## Reproducibility index

| Paper artifact | Where it lives in this repository |
|---|---|
| Dataset images + per-image category, prompt, source URL | `datasets/starter_cultural_patterns/images/`, `datasets/starter_cultural_patterns/paper_main_metadata.json` |
| Image licenses / provenance | per-image `source_url` in `paper_main_metadata.json` (Wikimedia Commons and other publicly accessible sources) |
| 81/31 train / held-out split identifiers | `experiments/fullreal_train_ids_v1.json`, `experiments/fullreal_eval_ids_v1.json` |
| Per-sample measurement records (all strategies x bandwidths) | `real_runs/unified_algorithm_benchmark_fullreal_split_v1/`, `real_runs/unified_baselines_fullreal_split_v1/`, `real_runs/seq/action_effect_table.json` |
| Policy / action logs of the trained scheduler | `real_runs/hybrid_ddpg_fullreal_split_v1/policy.json`, `decision_map.json` |
| Five-seed stability run | `real_runs/hybrid_ddpg_multiseed/` |
| End-to-end generation outputs and quality metrics (Table 5) | `real_runs/e2e_generation_v1/` |
| Scripts regenerating Tables 2-4 and every main-text figure | `experiments/rebuild_maintext_figures_realonly.py`, `experiments/reproduce_codec_conditions.py` |
| Implementation settings (Table 1) | `experiments/train_hybrid_ddpg_scheduler.py`, `configs/`, and the settings listed above |

## License

Code is released under the MIT License (see `LICENSE`). Dataset images originate from Wikimedia Commons and remain under their original licenses; see the per-image `source_url` fields.
