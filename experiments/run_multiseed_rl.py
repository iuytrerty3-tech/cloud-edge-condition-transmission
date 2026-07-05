#!/usr/bin/env python3
"""Round-3: repeated-seed Hybrid-DDPG training to report mean +/- std (addresses R2 'repeated runs').
Runs the existing trainer for several seeds and aggregates the held-out summary metrics.
  python experiments/run_multiseed_rl.py --sample-stats real_runs/unified_algorithm_benchmark_fullreal_split_v1/sample_stats.json \
      --train-sample-file experiments/fullreal_train_ids_v1.json \
      --eval-sample-file experiments/fullreal_eval_ids_v1.json \
      --seeds 0 1 2 3 4 --out real_runs/hybrid_ddpg_multiseed
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-stats", required=True)
    ap.add_argument("--train-sample-file", required=True)
    ap.add_argument("--eval-sample-file", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--out", default="real_runs/hybrid_ddpg_multiseed")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    trainer = root / "experiments" / "train_hybrid_ddpg_scheduler.py"
    runs = []
    for seed in args.seeds:
        rdir = out / f"seed_{seed}"; rdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(trainer), "--sample-stats", args.sample_stats,
               "--train-sample-file", args.train_sample_file, "--eval-sample-file", args.eval_sample_file,
               "--episodes", str(args.episodes), "--seed", str(seed), "--output-dir", str(rdir)]
        print("RUN:", " ".join(cmd)); subprocess.run(cmd, check=True)
        s = json.load(open(rdir / "summary.json"))
        runs.append(s)
    def agg(k): 
        v = [r[k] for r in runs if k in r]; return (float(np.mean(v)), float(np.std(v)))
    res = {k: {"mean": agg(k)[0], "std": agg(k)[1]} for k in ["mean_latency", "mean_payload", "mean_ssim"]}
    res["seeds"] = args.seeds
    json.dump(res, open(out / "multiseed_summary.json", "w"), indent=2)
    print("\n=== Multi-seed Hybrid-DDPG (mean +/- std over", len(args.seeds), "seeds) ===")
    for k, v in res.items():
        if isinstance(v, dict): print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")
    print(f"Saved {out/'multiseed_summary.json'}")

if __name__ == "__main__":
    main()
