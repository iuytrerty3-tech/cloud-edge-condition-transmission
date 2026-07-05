from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified benchmark from fixed-strategy and learned-policy results")
    parser.add_argument("--comparison-aggregated", type=str, required=True)
    parser.add_argument("--comparison-sample-stats", type=str, required=True)
    parser.add_argument("--spec-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Algorithm result directory in the form key=path. Directory must contain rl_aggregated_results.json and rl_sample_stats.json.",
    )
    return parser.parse_args()


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_samples(payload):
    return payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload


def parse_run_dirs(items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --run-dir value: {item!r}. Expected key=path")
        key, path = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --run-dir value: {item!r}. Empty key")
        mapping[key] = Path(path.strip())
    return mapping


def ranking_summary(rows: list[dict]) -> dict:
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_bw: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(float(row["total_time_mean"]))
        by_bw[float(row["bandwidth_mbps"])].append(row)

    average_latency_ranking = sorted(
        [
            {"strategy": strategy, "mean_total_time": float(np.mean(values))}
            for strategy, values in by_strategy.items()
        ],
        key=lambda item: item["mean_total_time"],
    )
    per_bandwidth_best = []
    for bw in sorted(by_bw):
        best = min(by_bw[bw], key=lambda row: float(row["total_time_mean"]))
        per_bandwidth_best.append(
            {
                "bandwidth_mbps": bw,
                "best_strategy": best["strategy"],
                "best_latency": float(best["total_time_mean"]),
            }
        )
    return {
        "average_latency_ranking": average_latency_ranking,
        "per_bandwidth_best": per_bandwidth_best,
    }


def main() -> None:
    args = parse_args()
    spec = load_json(args.spec_path)
    spec_keys = {item["key"] for item in spec["algorithms"]}
    run_dirs = parse_run_dirs(args.run_dir)

    comparison_aggregated = load_json(args.comparison_aggregated)
    comparison_samples = normalize_samples(load_json(args.comparison_sample_stats))

    learned_keys = set(run_dirs)
    fixed_keys = spec_keys - learned_keys

    combined_aggregated = [row for row in comparison_aggregated if row["strategy"] in fixed_keys]
    combined_samples = [row for row in comparison_samples if row["strategy"] in fixed_keys]

    for key, run_dir in run_dirs.items():
        aggregated_path = run_dir / "rl_aggregated_results.json"
        sample_stats_path = run_dir / "rl_sample_stats.json"
        if not aggregated_path.exists() or not sample_stats_path.exists():
            raise FileNotFoundError(f"Missing outputs for {key}: {run_dir}")
        combined_aggregated.extend(load_json(aggregated_path))
        combined_samples.extend(normalize_samples(load_json(sample_stats_path)))

    metrics = [
        {
            "strategy": item["strategy"],
            "mode": item["mode"],
            "downsample_factor": item["downsample_factor"],
            "bandwidth_mbps": item["bandwidth_mbps"],
            "steps": item["steps"],
            "guidance_scale": item["guidance_scale"],
            "num_pairs": item["num_samples"],
            "ssim": item["condition_ssim_mean"],
        }
        for item in combined_aggregated
    ]
    summary = ranking_summary(combined_aggregated)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregated_results.json").write_text(
        json.dumps(combined_aggregated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "sample_stats.json").write_text(
        json.dumps({"samples": combined_samples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "all_case_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
