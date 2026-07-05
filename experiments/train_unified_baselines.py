from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_hybrid_ddpg_scheduler import (
    build_feature_stats,
    build_sample_profiles,
    build_state_vector,
    build_state_vector_from_profile,
    load_sample_ids,
    load_samples,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate unified baseline algorithms on the parameterized pilot benchmark")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["rule", "bandwidth_only_rule", "random_selector", "random_forest", "gbdt", "a2c", "ppo", "dueling_ddqn"],
    )
    parser.add_argument("--sample-stats", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-file", type=str, default="")
    parser.add_argument("--eval-sample-file", type=str, default="")
    parser.add_argument("--quality-threshold", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=8000)
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
        env[(item["sample_id"], float(item["bandwidth_mbps"]))][item["strategy"]] = item
    return env


def global_action_list(env: dict) -> list[str]:
    actions = sorted({strategy for action_map in env.values() for strategy in action_map})
    return actions


def choose_best_valid(action_map: dict[str, dict], quality_threshold: float) -> str:
    valid = [item for item in action_map.values() if float(item["condition_ssim"]) >= quality_threshold]
    pool = valid if valid else list(action_map.values())
    return min(pool, key=lambda item: float(item["total_time_sec"]))["strategy"]


def aggregate(decision_map: dict[tuple[str, float], str], env: dict, strategy_name: str) -> tuple[list[dict], list[dict], dict]:
    rows = []
    groups = defaultdict(list)
    action_usage = defaultdict(int)
    for state_key, strategy in decision_map.items():
        item = env[state_key][strategy]
        rows.append(
            {
                "sample_id": state_key[0],
                "strategy": strategy_name,
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
                "strategy": strategy_name,
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


def heuristic_rule(state_key: tuple[str, float], state_vec: list[float], action_map: dict[str, dict], quality_threshold: float) -> str:
    bw = float(state_key[1])
    density = float(state_vec[4])
    prefs = []
    if bw <= 1.0:
        prefs = ["cond_png_l9_ds1", "cond_png_l6_ds1", "cond_png_l3_ds1"]
    elif bw <= 3.0:
        prefs = ["cond_png_l6_ds1", "cond_png_l3_ds1"] if density > 0.6 else ["cond_png_l3_ds1", "cond_png_l6_ds1"]
    elif bw <= 5.0:
        prefs = ["cond_png_l6_ds1", "cloud_jpeg_q75", "cond_png_l3_ds1"]
    else:
        prefs = ["cloud_jpeg_q75", "cond_png_l3_ds1", "cond_png_l6_ds1"]
    for pref in prefs:
        if pref in action_map and float(action_map[pref]["condition_ssim"]) >= quality_threshold:
            return pref
    valid = [item for item in action_map.values() if float(item["condition_ssim"]) >= quality_threshold]
    pool = valid if valid else list(action_map.values())
    return min(pool, key=lambda x: float(x["total_time_sec"]))["strategy"]


def bandwidth_only_rule(state_key: tuple[str, float], action_map: dict[str, dict], quality_threshold: float) -> str:
    bw = float(state_key[1])
    prefs = []
    if bw <= 1.0:
        prefs = ["cond_png_l9_ds1", "cond_png_l6_ds1", "cond_png_l3_ds1"]
    elif bw <= 2.0:
        prefs = ["cond_png_l6_ds1", "cond_png_l3_ds1", "cond_png_l9_ds1"]
    elif bw <= 5.0:
        prefs = ["cond_png_l3_ds1", "cond_png_l6_ds1", "cloud_jpeg_q75"]
    else:
        prefs = ["cond_png_l3_ds1", "cloud_jpeg_q75", "cond_png_l6_ds1"]
    for pref in prefs:
        if pref in action_map and float(action_map[pref]["condition_ssim"]) >= quality_threshold:
            return pref
    return choose_best_valid(action_map, quality_threshold)


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.policy_head(feat), self.value_head(feat)


class DuelingNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(64, 1)
        self.adv_head = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        value = self.value_head(feat)
        adv = self.adv_head(feat)
        return value + adv - adv.mean(dim=-1, keepdim=True)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


def eval_policy(model: nn.Module, algo: str, state_keys, state_vectors, action_list, action_index, valid_masks, env):
    decision_map = {}
    for state_key in state_keys:
        state_tensor = torch.tensor(state_vectors[state_key], dtype=torch.float32).unsqueeze(0)
        mask = valid_masks[state_key]
        if algo == "dueling_ddqn":
            q = model(state_tensor)
            q = q.masked_fill(~mask.unsqueeze(0), -1e9)
            action_idx = int(torch.argmax(q, dim=1).item())
        else:
            logits, _ = model(state_tensor)
            logits = masked_logits(logits, mask.unsqueeze(0))
            action_idx = int(torch.argmax(logits, dim=1).item())
        decision_map[state_key] = action_list[action_idx]
    return decision_map


def run_deep_baseline(algo: str, train_state_keys, eval_state_keys, train_state_vectors, eval_state_vectors, action_list, train_valid_masks, eval_valid_masks, env_train, env_eval, episodes: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_dim = len(next(iter(train_state_vectors.values())))
    action_dim = len(action_list)
    lr = 1e-3
    best_latency = math.inf
    best_decision_map = None
    training_curve = []

    if algo == "dueling_ddqn":
        model = DuelingNet(state_dim, action_dim)
        target = DuelingNet(state_dim, action_dim)
        target.load_state_dict(model.state_dict())
        opt = optim.Adam(model.parameters(), lr=lr)
        gamma = 0.0
    else:
        model = PolicyNet(state_dim, action_dim)
        opt = optim.Adam(model.parameters(), lr=lr)

    for episode in range(1, episodes + 1):
        state_key = random.choice(train_state_keys)
        state = torch.tensor(train_state_vectors[state_key], dtype=torch.float32).unsqueeze(0)
        mask = train_valid_masks[state_key].unsqueeze(0)
        action_map = env_train[state_key]

        if algo == "dueling_ddqn":
            q = model(state)
            q_masked = q.masked_fill(~mask, -1e9)
            epsilon = max(0.05, 0.35 - 0.30 * episode / max(episodes, 1))
            if random.random() < epsilon:
                valid_idx = torch.nonzero(mask[0], as_tuple=False).view(-1).tolist()
                action_idx = random.choice(valid_idx)
            else:
                action_idx = int(torch.argmax(q_masked, dim=1).item())
            action_name = action_list[action_idx]
            reward = float(action_map[action_name]["reward"])
            target_q = q.detach().clone()
            target_q[0, action_idx] = reward
            loss = nn.MSELoss()(q, target_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if episode % 100 == 0:
                target.load_state_dict(model.state_dict())
        else:
            logits, value = model(state)
            logits = masked_logits(logits, mask)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = int(dist.sample().item())
            action_name = action_list[action_idx]
            reward = float(action_map[action_name]["reward"])
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
            advantage = reward_tensor - value

            if algo == "a2c":
                policy_loss = -(dist.log_prob(torch.tensor(action_idx)) * advantage.detach().squeeze(0))
                value_loss = 0.5 * advantage.pow(2).mean()
                entropy_loss = -0.01 * dist.entropy().mean()
                loss = policy_loss + value_loss + entropy_loss
            else:
                old_log_prob = dist.log_prob(torch.tensor(action_idx)).detach()
                ratio = torch.exp(dist.log_prob(torch.tensor(action_idx)) - old_log_prob)
                clipped = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -(torch.min(ratio * advantage.detach().squeeze(0), clipped * advantage.detach().squeeze(0)))
                value_loss = 0.5 * advantage.pow(2).mean()
                entropy_loss = -0.01 * dist.entropy().mean()
                loss = policy_loss + value_loss + entropy_loss

            opt.zero_grad()
            loss.mean().backward()
            opt.step()

        if episode % 200 == 0:
            decision_map = eval_policy(model, algo, eval_state_keys, eval_state_vectors, action_list, None, eval_valid_masks, env_eval)
            _, aggregated, summary = aggregate(decision_map, env_eval, algo)
            if summary["mean_latency"] < best_latency:
                best_latency = summary["mean_latency"]
                best_decision_map = dict(decision_map)
            training_curve.append({"episode": episode, "mean_latency": summary["mean_latency"]})

    if best_decision_map is None:
        best_decision_map = eval_policy(model, algo, eval_state_keys, eval_state_vectors, action_list, None, eval_valid_masks, env_eval)
    return best_decision_map, training_curve


def run_ml_baseline(algo: str, train_state_keys, eval_state_keys, train_state_vectors, eval_state_vectors, env_train, env_eval, quality_threshold: float):
    if train_state_keys != eval_state_keys:
        X_train = np.asarray([train_state_vectors[state_key] for state_key in train_state_keys], dtype=np.float32)
        y_train = [choose_best_valid(env_train[state_key], quality_threshold) for state_key in train_state_keys]
        labels = sorted(set(y_train))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        y_idx = np.asarray([label_to_idx[label] for label in y_train])
        X_eval = np.asarray([eval_state_vectors[state_key] for state_key in eval_state_keys], dtype=np.float32)
        if algo == "random_forest":
            clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        else:
            clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=42)
        clf.fit(X_train, y_idx)
        pred_idx = clf.predict(X_eval)
        return {state_key: labels[int(pred)] for state_key, pred in zip(eval_state_keys, pred_idx)}

    samples_by_group = defaultdict(list)
    X = []
    y = []
    meta = []
    for idx, state_key in enumerate(train_state_keys):
        sample_id = state_key[0]
        X.append(train_state_vectors[state_key])
        y.append(choose_best_valid(env_train[state_key], quality_threshold))
        meta.append(state_key)
        samples_by_group[sample_id].append(idx)

    labels = sorted(set(y))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_idx = np.asarray([label_to_idx[label] for label in y])
    X = np.asarray(X, dtype=np.float32)
    preds = {}

    for group_name, test_indices in samples_by_group.items():
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[test_indices] = False
        if algo == "random_forest":
            clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        else:
            clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=42)
        clf.fit(X[train_mask], y_idx[train_mask])
        pred_idx = clf.predict(X[test_indices])
        for idx_local, pred in zip(test_indices, pred_idx):
            preds[meta[idx_local]] = labels[int(pred)]
    return preds


def run_random_selector(eval_state_keys, env_eval, quality_threshold: float, seed: int):
    random.seed(seed)
    decision_map = {}
    for state_key in eval_state_keys:
        valid = [item["strategy"] for item in env_eval[state_key].values() if float(item["condition_ssim"]) >= quality_threshold]
        pool = valid if valid else list(env_eval[state_key].keys())
        decision_map[state_key] = random.choice(sorted(pool))
    return decision_map


def run_rule_baseline(state_keys, state_vectors, env, quality_threshold: float):
    return {state_key: heuristic_rule(state_key, state_vectors[state_key], env[state_key], quality_threshold) for state_key in state_keys}


def run_bandwidth_only_rule(eval_state_keys, env_eval, quality_threshold: float):
    return {
        state_key: bandwidth_only_rule(state_key, env_eval[state_key], quality_threshold)
        for state_key in eval_state_keys
    }


def save_outputs(output_dir: Path, strategy_name: str, decision_map: dict, rows: list[dict], aggregated: list[dict], summary: dict, training_curve=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision_map.json").write_text(
        json.dumps({f"{sample_id}@{bw}": action for (sample_id, bw), action in decision_map.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rl_sample_stats.json").write_text(json.dumps({"samples": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rl_aggregated_results.json").write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if training_curve is not None:
        (output_dir / "training_curve.json").write_text(json.dumps(training_curve, indent=2, ensure_ascii=False), encoding="utf-8")


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
    env_train = build_env(train_samples, args.quality_threshold)
    env_eval = build_env(eval_samples, args.quality_threshold)
    train_state_keys = sorted(env_train.keys(), key=lambda x: (x[0], x[1]))
    eval_state_keys = sorted(env_eval.keys(), key=lambda x: (x[0], x[1]))
    train_state_vectors = {state_key: build_state_vector(state_key[0], state_key[1], stats) for state_key in train_state_keys}
    eval_profiles = build_sample_profiles(eval_samples)
    eval_state_vectors = {
        state_key: build_state_vector_from_profile(eval_profiles[state_key[0]], state_key[1], stats) for state_key in eval_state_keys
    }
    action_list = global_action_list(env_train if env_train else env_eval)
    train_valid_masks = {
        state_key: torch.tensor([action in env_train[state_key] and float(env_train[state_key][action]["condition_ssim"]) >= args.quality_threshold for action in action_list], dtype=torch.bool)
        for state_key in train_state_keys
    }
    eval_valid_masks = {
        state_key: torch.tensor([action in env_eval[state_key] and float(env_eval[state_key][action]["condition_ssim"]) >= args.quality_threshold for action in action_list], dtype=torch.bool)
        for state_key in eval_state_keys
    }

    if args.algo == "rule":
        decision_map = run_rule_baseline(eval_state_keys, eval_state_vectors, env_eval, args.quality_threshold)
        training_curve = None
    elif args.algo == "bandwidth_only_rule":
        decision_map = run_bandwidth_only_rule(eval_state_keys, env_eval, args.quality_threshold)
        training_curve = None
    elif args.algo == "random_selector":
        decision_map = run_random_selector(eval_state_keys, env_eval, args.quality_threshold, args.seed)
        training_curve = None
    elif args.algo in {"random_forest", "gbdt"}:
        decision_map = run_ml_baseline(args.algo, train_state_keys, eval_state_keys, train_state_vectors, eval_state_vectors, env_train, env_eval, args.quality_threshold)
        training_curve = None
    else:
        decision_map, training_curve = run_deep_baseline(
            args.algo,
            train_state_keys,
            eval_state_keys,
            train_state_vectors,
            eval_state_vectors,
            action_list,
            train_valid_masks,
            eval_valid_masks,
            env_train,
            env_eval,
            args.episodes,
            args.seed,
        )

    strategy_map = {
        "rule": "rule_threshold",
        "bandwidth_only_rule": "bandwidth_only_rule",
        "random_selector": "random_selector",
        "random_forest": "random_forest_selector",
        "gbdt": "gbdt_selector",
        "a2c": "a2c_scheduler",
        "ppo": "ppo_scheduler",
        "dueling_ddqn": "dueling_ddqn",
    }
    strategy_name = strategy_map[args.algo]
    rows, aggregated, summary = aggregate(decision_map, env_eval, strategy_name)
    summary["train_samples"] = len({item["sample_id"] for item in train_samples})
    summary["eval_samples"] = len({item["sample_id"] for item in eval_samples})
    save_outputs(Path(args.output_dir), strategy_name, decision_map, rows, aggregated, summary, training_curve)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
