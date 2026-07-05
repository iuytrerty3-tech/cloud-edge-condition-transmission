from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_hybrid_ddpg_scheduler import (
    build_action_embedding,
    build_feature_stats,
    build_sample_profiles,
    build_state_vector,
    build_state_vector_from_profile,
    load_sample_ids,
    load_samples,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an action-conditioned latency regressor scheduler")
    parser.add_argument("--sample-stats", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-file", type=str, default="")
    parser.add_argument("--eval-sample-file", type=str, default="")
    parser.add_argument("--quality-threshold", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_env(samples: list[dict]) -> dict:
    env: dict[tuple[str, float], dict[str, dict]] = defaultdict(dict)
    for item in samples:
        item = dict(item)
        item["action_embedding"] = build_action_embedding(item)
        env[(item["sample_id"], float(item["bandwidth_mbps"]))][item["strategy"]] = item
    return env


def aggregate(decision_map: dict[tuple[str, float], str], env: dict) -> tuple[list[dict], list[dict], dict]:
    rows = []
    groups = defaultdict(list)
    action_usage = defaultdict(int)
    for state_key, strategy in decision_map.items():
        item = env[state_key][strategy]
        rows.append(
            {
                "sample_id": state_key[0],
                "strategy": "ours_action_regressor",
                "selected_action": strategy,
                "bandwidth_mbps": state_key[1],
                "payload_kb": item["payload_kb"],
                "extract_time_sec": item["extract_time_sec"],
                "encode_time_sec": item["encode_time_sec"],
                "network_time_sec": item["network_time_sec"],
                "decode_time_sec": item["decode_time_sec"],
                "total_time_sec": item["total_time_sec"],
                "condition_ssim": item["condition_ssim"],
                "mode": "canny",
                "steps": 0,
                "guidance_scale": 0.0,
                "downsample_factor": int(item.get("downsample_factor", 1)),
            }
        )
        groups[state_key[1]].append(item)
        action_usage[strategy] += 1

    aggregated = []
    for bw in sorted(groups):
        items = groups[bw]
        aggregated.append(
            {
                "strategy": "ours_action_regressor",
                "mode": "canny",
                "downsample_factor": 1,
                "bandwidth_mbps": bw,
                "steps": 0,
                "guidance_scale": 0.0,
                "num_samples": len(items),
                "total_time_mean": float(np.mean([item["total_time_sec"] for item in items])),
                "total_time_std": float(np.std([item["total_time_sec"] for item in items])),
                "payload_kb_mean": float(np.mean([item["payload_kb"] for item in items])),
                "condition_ssim_mean": float(np.mean([item["condition_ssim"] for item in items])),
                "extract_time_mean": float(np.mean([item["extract_time_sec"] for item in items])),
                "encode_time_mean": float(np.mean([item["encode_time_sec"] for item in items])),
                "network_time_mean": float(np.mean([item["network_time_sec"] for item in items])),
                "decode_time_mean": float(np.mean([item["decode_time_sec"] for item in items])),
            }
        )

    summary = {
        "mean_latency": float(np.mean([row["total_time_mean"] for row in aggregated])),
        "mean_payload": float(np.mean([row["payload_kb_mean"] for row in aggregated])),
        "mean_ssim": float(np.mean([row["condition_ssim_mean"] for row in aggregated])),
        "action_usage": dict(sorted(action_usage.items(), key=lambda x: (-x[1], x[0]))),
    }
    return rows, aggregated, summary


def make_feature(state_vec: list[float], item: dict) -> list[float]:
    return [
        *state_vec,
        *item["action_embedding"],
        float(item.get("downsample_factor", 1)),
        float(item.get("png_level") or 0.0) / 9.0,
        float(item.get("jpeg_quality") or 0.0) / 100.0,
    ]


def main() -> None:
    args = parse_args()
    samples = load_samples(args.sample_stats)
    train_ids = load_sample_ids(args.train_sample_file)
    eval_ids = load_sample_ids(args.eval_sample_file)
    train_samples, eval_samples = split_samples(samples, train_ids, eval_ids)
    if not train_samples:
        raise ValueError("No training samples selected")
    if not eval_samples:
        raise ValueError("No evaluation samples selected")

    stats = build_feature_stats(train_samples)
    env_train = build_env(train_samples)
    env_eval = build_env(eval_samples)
    train_state_vectors = {
        state_key: build_state_vector(state_key[0], state_key[1], stats)
        for state_key in env_train
    }
    eval_profiles = build_sample_profiles(eval_samples)
    eval_state_vectors = {
        state_key: build_state_vector_from_profile(eval_profiles[state_key[0]], state_key[1], stats)
        for state_key in env_eval
    }

    x_train = []
    y_train = []
    for state_key, action_map in env_train.items():
        state_vec = train_state_vectors[state_key]
        for item in action_map.values():
            x_train.append(make_feature(state_vec, item))
            y_train.append(float(item["total_time_sec"]))
    model = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.06, random_state=args.seed)
    model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32))

    decision_map = {}
    for state_key, action_map in env_eval.items():
        state_vec = eval_state_vectors[state_key]
        valid_items = [item for item in action_map.values() if float(item["condition_ssim"]) >= args.quality_threshold]
        candidates = valid_items if valid_items else list(action_map.values())
        feats = np.asarray([make_feature(state_vec, item) for item in candidates], dtype=np.float32)
        preds = model.predict(feats)
        best_idx = int(np.argmin(preds))
        decision_map[state_key] = candidates[best_idx]["strategy"]

    rows, aggregated, summary = aggregate(decision_map, env_eval)
    summary["train_samples"] = len({item["sample_id"] for item in train_samples})
    summary["eval_samples"] = len({item["sample_id"] for item in eval_samples})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision_map.json").write_text(
        json.dumps({f"{sample_id}@{bw}": action for (sample_id, bw), action in decision_map.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rl_sample_stats.json").write_text(json.dumps({"samples": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rl_aggregated_results.json").write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
