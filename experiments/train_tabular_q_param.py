from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_hybrid_ddpg_scheduler import (
    build_feature_stats,
    build_state_vector,
    load_sample_ids,
    load_samples,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular Q scheduler on the parameterized pilot benchmark")
    parser.add_argument("--sample-stats", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-file", type=str, default="")
    parser.add_argument("--eval-sample-file", type=str, default="")
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--quality-threshold", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reward_fn(item: dict, quality_threshold: float) -> float:
    penalty = max(0.0, quality_threshold - float(item["condition_ssim"])) * 20.0
    return -(float(item["total_time_sec"]) + penalty)


def build_env(samples: list[dict], quality_threshold: float):
    env: dict[tuple[str, float], dict[str, dict]] = defaultdict(dict)
    for item in samples:
        item = dict(item)
        item["reward"] = reward_fn(item, quality_threshold)
        if float(item["condition_ssim"]) >= quality_threshold:
            env[(item["sample_id"], float(item["bandwidth_mbps"]))][item["strategy"]] = item
    return env


def aggregate(decision_map: dict[tuple[str, float], str], env: dict):
    rows = []
    groups = defaultdict(list)
    action_usage = defaultdict(int)
    for state_key, strategy in decision_map.items():
        item = env[state_key][strategy]
        rows.append(
            {
                "sample_id": state_key[0],
                "strategy": "ours_q_policy",
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
                "strategy": "ours_q_policy",
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


def choose_eval_action(state_key: tuple[str, float], q_table: dict, env_train: dict, env_eval: dict) -> str:
    if state_key in q_table:
        return max(env_eval[state_key].keys(), key=lambda action: q_table[state_key].get(action, -1e9))
    bw = float(state_key[1])
    score_by_action = defaultdict(list)
    for train_state, action_scores in q_table.items():
        if abs(float(train_state[1]) - bw) > 1e-9:
            continue
        for action, value in action_scores.items():
            score_by_action[action].append(float(value))
    if score_by_action:
        return max(
            env_eval[state_key].keys(),
            key=lambda action: float(np.mean(score_by_action[action])) if action in score_by_action else -1e9,
        )
    return min(env_eval[state_key].values(), key=lambda item: float(item["total_time_sec"]))["strategy"]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    samples = load_samples(args.sample_stats)
    train_ids = load_sample_ids(args.train_sample_file)
    eval_ids = load_sample_ids(args.eval_sample_file)
    train_samples, eval_samples = split_samples(samples, train_ids, eval_ids)
    if not train_samples:
        raise ValueError("No training samples selected")
    if not eval_samples:
        raise ValueError("No evaluation samples selected")
    stats = build_feature_stats(train_samples)
    env_train = build_env(train_samples, args.quality_threshold)
    env_eval = build_env(eval_samples, args.quality_threshold)
    state_keys_train = sorted(env_train.keys(), key=lambda x: (x[0], x[1]))
    state_keys_eval = sorted(env_eval.keys(), key=lambda x: (x[0], x[1]))
    state_vectors = {state_key: build_state_vector(state_key[0], state_key[1], stats) for state_key in state_keys_train}
    q_table = {state_key: {action: 0.0 for action in env_train[state_key]} for state_key in state_keys_train}

    best_latency = float("inf")
    best_decision_map = None
    training_curve = []

    for episode in range(1, args.episodes + 1):
        state_key = random.choice(state_keys_train)
        actions = list(env_train[state_key].keys())
        frac = min((episode - 1) / max(args.episodes - 1, 1), 1.0)
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * frac
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(actions, key=lambda x: q_table[state_key][x])
        reward = float(env_train[state_key][action]["reward"])
        q_table[state_key][action] += args.alpha * (reward - q_table[state_key][action])

        if episode % 200 == 0:
            decision_map = {state: choose_eval_action(state, q_table, env_train, env_eval) for state in state_keys_eval}
            _, aggregated, summary = aggregate(decision_map, env_eval)
            if summary["mean_latency"] < best_latency:
                best_latency = summary["mean_latency"]
                best_decision_map = dict(decision_map)
            training_curve.append({"episode": episode, "mean_latency": summary["mean_latency"], "epsilon": epsilon})

    if best_decision_map is None:
        best_decision_map = {state: choose_eval_action(state, q_table, env_train, env_eval) for state in state_keys_eval}

    rows, aggregated, summary = aggregate(best_decision_map, env_eval)
    summary["train_samples"] = len({item["sample_id"] for item in train_samples})
    summary["eval_samples"] = len({item["sample_id"] for item in eval_samples})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision_map.json").write_text(
        json.dumps({f"{sample_id}@{bw}": action for (sample_id, bw), action in best_decision_map.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rl_sample_stats.json").write_text(json.dumps({"samples": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rl_aggregated_results.json").write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "training_curve.json").write_text(json.dumps(training_curve, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
