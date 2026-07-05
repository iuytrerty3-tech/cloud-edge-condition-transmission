from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Q-learning scheduler on real benchmark data")
    parser.add_argument("--sample-stats", type=str, required=True)
    parser.add_argument("--extra-sample-stats", type=str, nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--quality-threshold", type=float, default=0.99)
    parser.add_argument("--quality-penalty", type=float, default=5.0)
    parser.add_argument("--payload-weight", type=float, default=0.0005)
    parser.add_argument("--reward-mode", type=str, default="latency", choices=["latency", "latency_payload"])
    parser.add_argument("--payload-bins", type=int, default=3)
    parser.add_argument("--extract-bins", type=int, default=3)
    parser.add_argument("--encode-bins", type=int, default=3)
    parser.add_argument("--include-sample-id", action="store_true")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def make_key(item: dict) -> tuple[str, float, str]:
    return (item["sample_id"], float(item["bandwidth_mbps"]), item["strategy"])


def canonical_strategy(strategy: str) -> str:
    return {
        "ours_adaptive": "ours",
        "cond_jpeg_q95": "condition_jpeg95",
        "cond_pca_ml": "condition_pca_ml",
        "cond_zlib": "condition_zlib",
        "canny_png_L9": "condition_png",
    }.get(strategy, strategy)


def normalize_item(item: dict) -> dict:
    sample_id = item.get("sample_id", item.get("id"))
    return {
        "sample_id": sample_id,
        "strategy": canonical_strategy(item["strategy"]),
        "bandwidth_mbps": float(item["bandwidth_mbps"]),
        "payload_kb": float(item.get("payload_kb", item.get("payload_kb_mean", 0.0))),
        "extract_time_sec": float(item.get("extract_time_sec", item.get("extract_time", 0.0))),
        "encode_time_sec": float(item.get("encode_time_sec", item.get("encode_time", 0.0))),
        "network_time_sec": float(item.get("network_time_sec", item.get("network_time", 0.0))),
        "decode_time_sec": float(item.get("decode_time_sec", item.get("decode_time", 0.0))),
        "total_time_sec": float(item.get("total_time_sec", item.get("total_time", 0.0))),
        "condition_ssim": float(item.get("condition_ssim", item.get("condition_ssim_mean", 0.0))),
    }


def load_samples(path_str: str) -> list[dict]:
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    return [normalize_item(row) for row in rows]


def category_from_sample_id(sample_id: str) -> str:
    if "_" not in sample_id:
        return sample_id
    prefix = sample_id.split("_", 1)[0]
    return {
        "porcelain": "blue_and_white_porcelain",
        "papercut": "paper_cutting",
        "embroidery": "gu_embroidery",
    }.get(prefix, prefix)


def quantile_edges(values: list[float], bins: int) -> list[float]:
    if bins <= 1 or len(values) <= 1:
        return []
    sorted_vals = sorted(values)
    edges = []
    for i in range(1, bins):
        idx = min(len(sorted_vals) - 1, max(0, round(i * len(sorted_vals) / bins) - 1))
        edge = float(sorted_vals[idx])
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def assign_bin(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def build_sample_profiles(
    samples: list[dict],
    payload_bins: int,
    extract_bins: int,
    encode_bins: int,
) -> dict[str, dict]:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for item in samples:
        by_sample[item["sample_id"]].append(item)

    raw_profiles = {}
    payload_refs = []
    extract_refs = []
    encode_refs = []
    for sample_id, rows in by_sample.items():
        cond_rows = [row for row in rows if row["strategy"] == "condition_png"]
        reference_rows = cond_rows if cond_rows else rows
        payload_ref = mean([float(row["payload_kb"]) for row in reference_rows])
        extract_ref = mean([float(row["extract_time_sec"]) for row in reference_rows])
        encode_ref = mean([float(row["encode_time_sec"]) for row in reference_rows])
        raw_profiles[sample_id] = {
            "category": category_from_sample_id(sample_id),
            "payload_ref_kb": payload_ref,
            "extract_ref_sec": extract_ref,
            "encode_ref_sec": encode_ref,
        }
        payload_refs.append(payload_ref)
        extract_refs.append(extract_ref)
        encode_refs.append(encode_ref)

    payload_edges = quantile_edges(payload_refs, payload_bins)
    extract_edges = quantile_edges(extract_refs, extract_bins)
    encode_edges = quantile_edges(encode_refs, encode_bins)

    profiles = {}
    for sample_id, profile in raw_profiles.items():
        profiles[sample_id] = {
            **profile,
            "payload_bin": assign_bin(profile["payload_ref_kb"], payload_edges),
            "extract_bin": assign_bin(profile["extract_ref_sec"], extract_edges),
            "encode_bin": assign_bin(profile["encode_ref_sec"], encode_edges),
        }
    return profiles


def build_state(
    sample_id: str,
    bandwidth: float,
    profiles: dict[str, dict],
    include_sample_id: bool,
) -> tuple[float, str, int, int, int, str]:
    profile = profiles[sample_id]
    return (
        float(bandwidth),
        profile["category"],
        int(profile["payload_bin"]),
        int(profile["extract_bin"]),
        int(profile["encode_bin"]),
        sample_id if include_sample_id else "__shared__",
    )


def build_env(samples: list[dict]) -> tuple[list[float], list[str], dict[tuple[str, float], dict[str, dict]]]:
    bandwidths = sorted({float(item["bandwidth_mbps"]) for item in samples})
    strategies = sorted({item["strategy"] for item in samples})
    env: dict[tuple[str, float], dict[str, dict]] = defaultdict(dict)
    for item in samples:
        env[(item["sample_id"], float(item["bandwidth_mbps"]))][item["strategy"]] = item
    return bandwidths, strategies, env


def reward_fn(
    item: dict,
    quality_threshold: float,
    quality_penalty: float,
    payload_weight: float,
    reward_mode: str,
) -> float:
    latency = float(item["total_time_sec"])
    payload = float(item["payload_kb"])
    ssim = float(item["condition_ssim"])
    penalty = max(0.0, quality_threshold - ssim) * quality_penalty
    payload_term = payload_weight * payload if reward_mode == "latency_payload" else 0.0
    return -(latency + payload_term + penalty)


def aggregate_policy_results(
    env: dict[tuple[str, float], dict[str, dict]],
    decision_map: dict[tuple[str, float], str],
    quality_threshold: float,
    quality_penalty: float,
    payload_weight: float,
    reward_mode: str,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for key, action in decision_map.items():
        sample_id, bandwidth = key
        item = env[(sample_id, bandwidth)][action]
        grouped[float(bandwidth)].append(item)

    rl_sample_stats: list[dict] = []
    aggregated: list[dict] = []
    for bandwidth in sorted(grouped):
        selected = grouped[bandwidth]
        for item in selected:
            rl_item = dict(item)
            rl_item["strategy"] = "ours_rl_policy"
            rl_item["reward"] = reward_fn(item, quality_threshold, quality_penalty, payload_weight, reward_mode)
            rl_sample_stats.append(rl_item)
        aggregated.append(
            {
                "strategy": "ours_rl_policy",
                "mode": "canny",
                "downsample_factor": 1,
                "bandwidth_mbps": bandwidth,
                "steps": 0,
                "guidance_scale": 0.0,
                "num_samples": len(selected),
                "total_time_mean": mean([item["total_time_sec"] for item in selected]),
                "total_time_std": std([item["total_time_sec"] for item in selected]),
                "network_time_mean": mean([item["network_time_sec"] for item in selected]),
                "cloud_gen_time_mean": 0.0,
                "payload_kb_mean": mean([item["payload_kb"] for item in selected]),
                "condition_ssim_mean": mean([item["condition_ssim"] for item in selected]),
                "extract_time_mean": mean([item["extract_time_sec"] for item in selected]),
                "encode_time_mean": mean([item["encode_time_sec"] for item in selected]),
                "decode_time_mean": mean([item["decode_time_sec"] for item in selected]),
            }
        )
    return rl_sample_stats, aggregated


def best_available_action(
    state: tuple[float, str, int, int, int, str],
    strategy_map: dict[str, dict],
    q_table: dict[tuple[float, str, int, int, int, str], dict[str, float]],
) -> str:
    available_actions = list(strategy_map.keys())
    return max(available_actions, key=lambda action: q_table[state][action])


def main() -> None:
    args = parse_args()
    samples = load_samples(args.sample_stats)
    for extra_path in args.extra_sample_stats:
        samples.extend(load_samples(extra_path))

    allowed_strategies = [
        "ours",
        "condition_png",
        "condition_jpeg95",
        "condition_zlib",
        "condition_pca_ml",
        "cloud_condition_png",
        "cloud_condition_jpeg75",
        "canny_png_L3",
        "canny_png_L1",
        "cond_jpeg_q75",
    ]
    samples = [item for item in samples if item["strategy"] in allowed_strategies]
    bandwidths, strategies, env = build_env(samples)
    sample_profiles = build_sample_profiles(samples, args.payload_bins, args.extract_bins, args.encode_bins)

    states = sorted({build_state(sample_id, bandwidth, sample_profiles, args.include_sample_id) for sample_id, bandwidth in env.keys()})
    q_table = {state: {strategy: 0.0 for strategy in strategies} for state in states}
    training_curve = []
    keys = list(env.keys())
    random.seed(42)

    for episode in range(args.episodes):
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * (episode / max(1, args.episodes - 1))
        random.shuffle(keys)
        rewards = []
        for sample_id, bandwidth in keys:
            strategy_map = env[(sample_id, bandwidth)]
            state = build_state(sample_id, bandwidth, sample_profiles, args.include_sample_id)
            available_actions = list(strategy_map.keys())
            if random.random() < epsilon:
                action = random.choice(available_actions)
            else:
                action = max(available_actions, key=lambda a: q_table[state][a])
            item = strategy_map[action]
            reward = reward_fn(item, args.quality_threshold, args.quality_penalty, args.payload_weight, args.reward_mode)
            rewards.append(reward)
            current_q = q_table[state][action]
            target = reward + args.gamma * max(q_table[state][a] for a in available_actions)
            q_table[state][action] = current_q + args.alpha * (target - current_q)
        training_curve.append(
            {
                "episode": episode + 1,
                "epsilon": epsilon,
                "mean_reward": mean(rewards),
            }
        )

    policy = {
        f"bw={state[0]}|cat={state[1]}|payload_bin={state[2]}|extract_bin={state[3]}|encode_bin={state[4]}|sample={state[5]}": best_available_action(
            state,
            env[(state[5], state[0])],
            q_table,
        )
        for state in states
        if state[5] != "__shared__" and (state[5], state[0]) in env
    }
    decision_map = {
        (sample_id, bandwidth): best_available_action(
            build_state(sample_id, bandwidth, sample_profiles, args.include_sample_id),
            env[(sample_id, bandwidth)],
            q_table,
        )
        for sample_id, bandwidth in env.keys()
    }
    rl_sample_stats, rl_aggregated = aggregate_policy_results(
        env,
        decision_map,
        args.quality_threshold,
        args.quality_penalty,
        args.payload_weight,
        args.reward_mode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized_q_table = {
        f"bw={state[0]}|cat={state[1]}|payload_bin={state[2]}|extract_bin={state[3]}|encode_bin={state[4]}|sample={state[5]}": action_values
        for state, action_values in q_table.items()
    }
    (output_dir / "q_table.json").write_text(json.dumps(serialized_q_table, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "policy.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "sample_profiles.json").write_text(
        json.dumps(sample_profiles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "decision_map.json").write_text(
        json.dumps(
            {f"{sample_id}@{bandwidth}": action for (sample_id, bandwidth), action in sorted(decision_map.items())},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "training_curve.json").write_text(
        json.dumps(training_curve, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rl_sample_stats.json").write_text(
        json.dumps({"samples": rl_sample_stats}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rl_aggregated_results.json").write_text(
        json.dumps(rl_aggregated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "num_states": len(states),
                "policy": policy,
                "rl_aggregated_results": str((output_dir / "rl_aggregated_results.json")).replace("\\", "/"),
                "training_curve": str((output_dir / "training_curve.json")).replace("\\", "/"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
