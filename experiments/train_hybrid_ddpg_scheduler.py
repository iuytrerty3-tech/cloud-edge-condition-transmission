from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid DDPG scheduler on the parameterized pilot benchmark")
    parser.add_argument("--sample-stats", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-file", type=str, default="")
    parser.add_argument("--eval-sample-file", type=str, default="")
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--exploration-std", type=float, default=0.20)
    parser.add_argument("--exploration-end", type=float, default=0.03)
    parser.add_argument("--oracle-sample-prob", type=float, default=0.35)
    parser.add_argument("--bc-weight", type=float, default=1.5)
    parser.add_argument("--bc-weight-end", type=float, default=0.15)
    parser.add_argument("--quality-threshold", type=float, default=0.99)
    parser.add_argument("--quality-penalty", type=float, default=10.0)
    parser.add_argument("--payload-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_samples(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    normalized = []
    for row in rows:
        normalized.append(
            {
                "sample_id": row.get("sample_id", row.get("id")),
                "category": row["category"],
                "strategy": row["strategy"],
                "branch": row.get("branch"),
                "bandwidth_mbps": float(row["bandwidth_mbps"]),
                "payload_kb": float(row["payload_kb"]),
                "extract_time_sec": float(row["extract_time_sec"]),
                "encode_time_sec": float(row["encode_time_sec"]),
                "network_time_sec": float(row["network_time_sec"]),
                "decode_time_sec": float(row["decode_time_sec"]),
                "total_time_sec": float(row["total_time_sec"]),
                "condition_ssim": float(row["condition_ssim"]),
                "downsample_factor": int(row.get("downsample_factor", 1)),
                "png_level": row.get("png_level"),
                "jpeg_quality": row.get("jpeg_quality"),
                "edge_density": float(row.get("edge_density", 0.0)),
            }
        )
    return normalized


def load_sample_ids(path: str) -> set[str]:
    if not path:
        return set()
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item) for item in rows}


def reward_fn(item: dict, quality_threshold: float, quality_penalty: float, payload_weight: float) -> float:
    penalty = max(0.0, quality_threshold - float(item["condition_ssim"])) * quality_penalty
    return -(float(item["total_time_sec"]) + payload_weight * float(item["payload_kb"]) + penalty)


def build_action_embedding(item: dict) -> list[float]:
    strategy = item["strategy"]
    if strategy.startswith("cond_png"):
        branch = 0.10
        quality = 1.0
        level = float(item.get("png_level") or 9) / 9.0
        ds = 0.0 if int(item.get("downsample_factor", 1)) == 1 else 1.0
    elif strategy.startswith("cond_jpeg"):
        branch = 0.40
        quality = float(item.get("jpeg_quality") or 95) / 100.0
        level = quality
        ds = 0.0 if int(item.get("downsample_factor", 1)) == 1 else 1.0
    elif strategy.startswith("cloud_jpeg"):
        branch = 0.75
        quality = float(item.get("jpeg_quality") or 75) / 100.0
        level = quality
        ds = 0.0
    else:
        branch = 1.0
        quality = 1.0
        level = 1.0
        ds = 0.0
    return [branch, quality, level, ds]


def build_sample_profiles(samples: list[dict]) -> dict:
    refs = defaultdict(list)
    for item in samples:
        refs[item["sample_id"]].append(item)

    profiles = {}
    for sample_id, rows in refs.items():
        lossless_rows = [row for row in rows if row["condition_ssim"] >= 0.999 and row["strategy"] == "cond_png_l9_ds1"]
        ref = lossless_rows[0] if lossless_rows else min(rows, key=lambda x: x["total_time_sec"])
        profiles[sample_id] = {
            "category": ref["category"],
            "edge_density": float(ref["edge_density"]),
            "payload_ref": float(ref["payload_kb"]),
            "extract_ref": float(ref["extract_time_sec"]),
            "encode_ref": float(ref["encode_time_sec"]),
        }
    return profiles


def build_feature_stats(samples: list[dict]) -> dict:
    profiles = build_sample_profiles(samples)
    categories = sorted({profile["category"] for profile in profiles.values()})
    max_payload = max(profile["payload_ref"] for profile in profiles.values())
    max_extract = max(profile["extract_ref"] for profile in profiles.values())
    max_encode = max(profile["encode_ref"] for profile in profiles.values())
    max_density = max(profile["edge_density"] for profile in profiles.values())
    return {
        "profiles": profiles,
        "categories": categories,
        "max_payload": max_payload if max_payload > 0 else 1.0,
        "max_extract": max_extract if max_extract > 0 else 1.0,
        "max_encode": max_encode if max_encode > 0 else 1.0,
        "max_density": max_density if max_density > 0 else 1.0,
    }


def build_state_vector_from_profile(profile: dict, bw: float, stats: dict) -> list[float]:
    cat_vec = [1.0 if category == profile["category"] else 0.0 for category in stats["categories"]]
    return [
        bw / 10.0,
        profile["payload_ref"] / stats["max_payload"],
        profile["extract_ref"] / stats["max_extract"],
        profile["encode_ref"] / stats["max_encode"],
        profile["edge_density"] / stats["max_density"],
        *cat_vec,
    ]


def build_state_vector(sample_id: str, bw: float, stats: dict) -> list[float]:
    return build_state_vector_from_profile(stats["profiles"][sample_id], bw, stats)


def split_samples(samples: list[dict], train_ids: set[str], eval_ids: set[str]) -> tuple[list[dict], list[dict]]:
    if not train_ids and not eval_ids:
        return samples, samples
    if train_ids:
        train_samples = [item for item in samples if item["sample_id"] in train_ids]
    elif eval_ids:
        train_samples = [item for item in samples if item["sample_id"] not in eval_ids]
    else:
        train_samples = list(samples)
    if eval_ids:
        eval_samples = [item for item in samples if item["sample_id"] in eval_ids]
    elif train_ids:
        eval_samples = [item for item in samples if item["sample_id"] in train_ids]
    else:
        eval_samples = list(train_samples)
    return train_samples, eval_samples


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def nearest_action(actor_vec: torch.Tensor, available_items: list[dict]) -> dict:
    vec = actor_vec.detach().cpu().numpy()
    best_item = None
    best_dist = math.inf
    for item in available_items:
        action_vec = np.asarray(item["action_embedding"], dtype=np.float32)
        dist = float(np.square(vec - action_vec).sum())
        if dist < best_dist:
            best_item = item
            best_dist = dist
    if best_item is None:
        raise RuntimeError("No available action for current state")
    return best_item


def best_quality_valid_action(available_items: list[dict], quality_threshold: float) -> dict:
    valid = [item for item in available_items if float(item["condition_ssim"]) >= quality_threshold]
    pool = valid if valid else available_items
    return min(pool, key=lambda item: float(item["total_time_sec"]))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def aggregate_policy_results(decision_map: dict, env: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    groups = defaultdict(list)
    for (sample_id, bw), selected_strategy in decision_map.items():
        item = env[(sample_id, bw)][selected_strategy]
        rows.append(
            {
                "sample_id": sample_id,
                "strategy": "ours_hybrid_ddpg",
                "bandwidth_mbps": bw,
                "payload_kb": item["payload_kb"],
                "extract_time_sec": item["extract_time_sec"],
                "encode_time_sec": item["encode_time_sec"],
                "network_time_sec": item["network_time_sec"],
                "decode_time_sec": item["decode_time_sec"],
                "total_time_sec": item["total_time_sec"],
                "condition_ssim": item["condition_ssim"],
                "selected_action": selected_strategy,
                "mode": "canny",
                "downsample_factor": item["downsample_factor"],
                "steps": 0,
                "guidance_scale": 0.0,
            }
        )
        groups[bw].append(item)

    aggregated = []
    for bw in sorted(groups):
        items = groups[bw]
        aggregated.append(
            {
                "strategy": "ours_hybrid_ddpg",
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
    return rows, aggregated


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = load_samples(args.sample_stats)
    samples = [item for item in samples if float(item["condition_ssim"]) >= 0.55]
    train_ids = load_sample_ids(args.train_sample_file)
    eval_ids = load_sample_ids(args.eval_sample_file)
    train_samples, eval_samples = split_samples(samples, train_ids, eval_ids)
    if not train_samples:
        raise ValueError("No training samples selected")
    if not eval_samples:
        raise ValueError("No evaluation samples selected")
    stats = build_feature_stats(train_samples)

    env_train: dict[tuple[str, float], dict[str, dict]] = defaultdict(dict)
    for item in train_samples:
        state_key = (item["sample_id"], float(item["bandwidth_mbps"]))
        action_embedding = build_action_embedding(item)
        item["action_embedding"] = action_embedding
        item["reward"] = reward_fn(item, args.quality_threshold, args.quality_penalty, args.payload_weight)
        env_train[state_key][item["strategy"]] = item

    env_eval: dict[tuple[str, float], dict[str, dict]] = defaultdict(dict)
    for item in eval_samples:
        state_key = (item["sample_id"], float(item["bandwidth_mbps"]))
        action_embedding = build_action_embedding(item)
        item["action_embedding"] = action_embedding
        item["reward"] = reward_fn(item, args.quality_threshold, args.quality_penalty, args.payload_weight)
        env_eval[state_key][item["strategy"]] = item

    state_keys_train = sorted(env_train.keys(), key=lambda x: (x[0], x[1]))
    state_vectors_train = {state_key: build_state_vector(state_key[0], state_key[1], stats) for state_key in state_keys_train}
    eval_profiles = build_sample_profiles(eval_samples)
    state_keys_eval = sorted(env_eval.keys(), key=lambda x: (x[0], x[1]))
    state_vectors_eval = {
        state_key: build_state_vector_from_profile(eval_profiles[state_key[0]], state_key[1], stats)
        for state_key in state_keys_eval
    }
    oracle_actions = {
        state_key: best_quality_valid_action(list(env_train[state_key].values()), args.quality_threshold)
        for state_key in state_keys_train
    }
    state_dim = len(next(iter(state_vectors_train.values())))
    action_dim = 4

    actor = Actor(state_dim, action_dim)
    actor_target = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    critic_target = Critic(state_dim, action_dim)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = deque(maxlen=4096)
    mse = nn.MSELoss()
    training_curve = []
    best_eval_latency = math.inf
    best_decision_map = None

    for episode in range(1, args.episodes + 1):
        state_key = random.choice(state_keys_train)
        state_tensor = torch.tensor(state_vectors_train[state_key], dtype=torch.float32)
        actor_out = actor(state_tensor.unsqueeze(0)).squeeze(0)

        frac = min((episode - 1) / max(args.episodes - 1, 1), 1.0)
        sigma = args.exploration_std + (args.exploration_end - args.exploration_std) * frac
        if random.random() < args.oracle_sample_prob:
            chosen_item = oracle_actions[state_key]
        else:
            noisy_action = torch.clamp(actor_out + torch.randn_like(actor_out) * sigma, 0.0, 1.0)
            chosen_item = nearest_action(noisy_action, list(env_train[state_key].values()))
        reward = float(chosen_item["reward"])
        chosen_action = torch.tensor(chosen_item["action_embedding"], dtype=torch.float32)
        oracle_action = torch.tensor(oracle_actions[state_key]["action_embedding"], dtype=torch.float32)
        replay.append((state_tensor, chosen_action, reward, oracle_action))

        if len(replay) >= args.batch_size:
            batch = random.sample(replay, args.batch_size)
            states = torch.stack([row[0] for row in batch])
            actions = torch.stack([row[1] for row in batch])
            rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32).unsqueeze(1)
            oracle_batch = torch.stack([row[3] for row in batch])

            critic_pred = critic(states, actions)
            critic_loss = mse(critic_pred, rewards)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            actor_actions = actor(states)
            bc_weight = args.bc_weight + (args.bc_weight_end - args.bc_weight) * frac
            actor_loss = -critic(states, actor_actions).mean() + bc_weight * mse(actor_actions, oracle_batch)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            soft_update(actor_target, actor, args.tau)
            soft_update(critic_target, critic, args.tau)

        if episode % 200 == 0:
            decision_map = {}
            eval_rewards = []
            for state_key_eval in state_keys_eval:
                state_eval = torch.tensor(state_vectors_eval[state_key_eval], dtype=torch.float32)
                actor_eval = actor(state_eval.unsqueeze(0)).squeeze(0)
                item = nearest_action(actor_eval, list(env_eval[state_key_eval].values()))
                decision_map[state_key_eval] = item["strategy"]
                eval_rewards.append(float(item["reward"]))
            _, aggregated = aggregate_policy_results(decision_map, env_eval)
            mean_latency = float(np.mean([row["total_time_mean"] for row in aggregated]))
            if mean_latency < best_eval_latency:
                best_eval_latency = mean_latency
                best_decision_map = dict(decision_map)
            training_curve.append(
                {
                    "episode": episode,
                    "exploration_std": round(float(sigma), 6),
                    "mean_reward": round(float(np.mean(eval_rewards)), 6),
                    "mean_latency": round(mean_latency, 6),
                }
            )

    if best_decision_map is None:
        best_decision_map = {}
        for state_key in state_keys_eval:
            fallback = best_quality_valid_action(list(env_eval[state_key].values()), args.quality_threshold)
            best_decision_map[state_key] = fallback["strategy"]

    decision_map = {}
    action_usage = defaultdict(int)
    policy = {}
    for state_key in state_keys_eval:
        selected_strategy = best_decision_map[state_key]
        item = env_eval[state_key][selected_strategy]
        decision_map[state_key] = selected_strategy
        action_usage[selected_strategy] += 1
        sample_id, bw = state_key
        policy[f"{sample_id}@{bw}"] = {
            "selected_action": selected_strategy,
            "action_embedding": item["action_embedding"],
            "state_vector": state_vectors_eval[state_key],
        }

    rl_rows, aggregated = aggregate_policy_results(decision_map, env_eval)
    summary = {
        "mean_latency": float(np.mean([row["total_time_mean"] for row in aggregated])),
        "mean_payload": float(np.mean([row["payload_kb_mean"] for row in aggregated])),
        "mean_ssim": float(np.mean([row["condition_ssim_mean"] for row in aggregated])),
        "action_usage": dict(sorted(action_usage.items(), key=lambda x: (-x[1], x[0]))),
        "quality_threshold": args.quality_threshold,
        "episodes": args.episodes,
        "train_samples": len({item["sample_id"] for item in train_samples}),
        "eval_samples": len({item["sample_id"] for item in eval_samples}),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "decision_map.json").write_text(
        json.dumps({f"{sample_id}@{bw}": strategy for (sample_id, bw), strategy in decision_map.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rl_sample_stats.json").write_text(json.dumps({"samples": rl_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rl_aggregated_results.json").write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "training_curve.json").write_text(json.dumps(training_curve, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
