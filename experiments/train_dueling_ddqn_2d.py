"""
Train Dueling Double DQN with 2D state space (BW + edge density bin).
State: [bw_one_hot(7) | density_bin_one_hot(2)] = 9-dim
Actions: 8 strategies (same as 1D DDQN)
Output: real_runs/extended_benchmark/dueling_ddqn_2d_*.json

Also runs ablation: 4 variants
  A: 1D state + simple DQN
  B: 1D state + Dueling DQN
  C: 2D state + simple DQN
  D: 2D state + Dueling DQN  ← proposed
"""
from __future__ import annotations
import json, random, collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── load benchmark data ──────────────────────────────────────────────────────
OUT_DIR  = Path(__file__).resolve().parent.parent / "real_runs/extended_benchmark"
agg      = json.loads((OUT_DIR / "aggregated_results.json").read_text(encoding="utf-8"))
all_cases = json.loads((OUT_DIR / "all_cases.json").read_text(encoding="utf-8"))

BANDWIDTHS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
BW_TO_IDX  = {bw: i for i, bw in enumerate(BANDWIDTHS)}
N_BW       = len(BANDWIDTHS)

# Realistic bandwidth sampling weights (low BW more common in mobile/edge scenarios)
BW_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02]
BW_WEIGHTS_CUM = []
_cum = 0.0
for w in BW_WEIGHTS:
    _cum += w
    BW_WEIGHTS_CUM.append(_cum)

def sample_bw_idx() -> int:
    """Sample bandwidth index according to realistic distribution."""
    r = random.random()
    for i, c in enumerate(BW_WEIGHTS_CUM):
        if r <= c:
            return i
    return N_BW - 1

ACTIONS = [
    "canny_png_L9",
    "canny_png_L3",
    "canny_png_L1",
    "cond_jpeg_q95",
    "cond_jpeg_q75",
    "cond_zlib",
    "cond_pca_ml",
    "cloud_condition_jpeg75",
]
N_ACTIONS  = len(ACTIONS)
ACT_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Density bins: 0 = low (< 8%), 1 = high (>= 8%)
DENSITY_THRESHOLD = 8.0
N_DENSITY_BINS    = 2

# Build lookup tables: latency[bw_idx][act_idx] and ssim[bw_idx][act_idx]
latency_table = np.full((N_BW, N_ACTIONS), 9.0)
ssim_table    = np.full((N_BW, N_ACTIONS), 0.0)
for row in agg:
    if row["strategy"] not in ACT_TO_IDX: continue
    bi = BW_TO_IDX.get(float(row["bandwidth_mbps"]))
    ai = ACT_TO_IDX[row["strategy"]]
    if bi is None: continue
    latency_table[bi][ai] = row["total_time_mean"]
    ssim_table[bi][ai]    = row["condition_ssim_mean"]

# Build 2D lookup: latency_2d[bw_idx][density_bin][act_idx]
# Use proper per-cell mean via accumulation lists
from collections import defaultdict as _dd
_lat_acc  = _dd(list)
_ssim_acc = _dd(list)
for c in all_cases:
    if c["strategy"] not in ACT_TO_IDX: continue
    bi = BW_TO_IDX.get(float(c["bandwidth_mbps"]))
    if bi is None: continue
    ai = ACT_TO_IDX[c["strategy"]]
    di = 0 if c["edge_density"] < DENSITY_THRESHOLD else 1
    _lat_acc[(bi, di, ai)].append(c["total_time"])
    _ssim_acc[(bi, di, ai)].append(c["condition_ssim"])

latency_2d = np.full((N_BW, N_DENSITY_BINS, N_ACTIONS), 9.0)
ssim_2d    = np.full((N_BW, N_DENSITY_BINS, N_ACTIONS), 0.0)
for (bi, di, ai), vals in _lat_acc.items():
    latency_2d[bi][di][ai] = float(np.mean(vals))
for (bi, di, ai), vals in _ssim_acc.items():
    ssim_2d[bi][di][ai] = float(np.mean(vals))

# Fill missing 2D cells with 1D values
for bi in range(N_BW):
    for di in range(N_DENSITY_BINS):
        for ai in range(N_ACTIONS):
            if latency_2d[bi][di][ai] == 9.0:
                latency_2d[bi][di][ai] = latency_table[bi][ai]
                ssim_2d[bi][di][ai]    = ssim_table[bi][ai]

# ── reward functions ──────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 0.95
QUALITY_PENALTY   = 5.0

def reward_1d(bw_idx: int, act_idx: int) -> float:
    lat  = latency_table[bw_idx][act_idx]
    ssim = ssim_table[bw_idx][act_idx]
    penalty = QUALITY_PENALTY if ssim < QUALITY_THRESHOLD else 0.0
    return -lat - penalty

def reward_2d(bw_idx: int, density_bin: int, act_idx: int) -> float:
    lat  = latency_2d[bw_idx][density_bin][act_idx]
    ssim = ssim_2d[bw_idx][density_bin][act_idx]
    penalty = QUALITY_PENALTY if ssim < QUALITY_THRESHOLD else 0.0
    return -lat - penalty

# ── Network architectures ─────────────────────────────────────────────────────
class SimpleQNet(nn.Module):
    """Standard DQN MLP."""
    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling DQN: shared backbone + separate value and advantage streams."""
    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        # Q = V + (A - mean(A))  — removes identifiability issue
        return V + (A - A.mean(dim=-1, keepdim=True))


# ── State encoders ────────────────────────────────────────────────────────────
def state_1d(bw_idx: int) -> torch.Tensor:
    t = torch.zeros(N_BW)
    t[bw_idx] = 1.0
    return t

def state_2d(bw_idx: int, density_bin: int) -> torch.Tensor:
    t = torch.zeros(N_BW + N_DENSITY_BINS)
    t[bw_idx] = 1.0
    t[N_BW + density_bin] = 1.0
    return t


# ── Replay buffer ─────────────────────────────────────────────────────────────
Transition = collections.namedtuple("Transition", ["s", "a", "r", "s_next", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int = 8192):
        self.buf: collections.deque = collections.deque(maxlen=capacity)
    def push(self, *args): self.buf.append(Transition(*args))
    def sample(self, n): return random.sample(self.buf, n)
    def __len__(self): return len(self.buf)


# ── Training loop ─────────────────────────────────────────────────────────────
EPISODES        = 200_000
LR              = 3e-4
GAMMA           = 0.9
BATCH_SIZE      = 128
TARGET_UPDATE   = 200
EPS_START       = 0.9
EPS_END         = 0.01
EPS_DECAY_STEPS = 160_000
WARMUP          = 256

def get_epsilon(step: int) -> float:
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + (EPS_END - EPS_START) * frac


def train_variant(name: str, use_2d: bool, use_dueling: bool, seed: int = 42):
    """Train one variant and return (policy_dict, mean_latency, training_curve)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    n_states = (N_BW + N_DENSITY_BINS) if use_2d else N_BW
    NetClass = DuelingQNet if use_dueling else SimpleQNet

    online_net = NetClass(n_states, N_ACTIONS)
    target_net = NetClass(n_states, N_ACTIONS)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    buffer    = ReplayBuffer()
    loss_fn   = nn.MSELoss()
    curve     = []

    for ep in range(1, EPISODES + 1):
        bw_idx = sample_bw_idx()
        di     = random.randint(0, N_DENSITY_BINS - 1) if use_2d else 0
        s      = state_2d(bw_idx, di) if use_2d else state_1d(bw_idx)
        eps    = get_epsilon(ep)

        if random.random() < eps:
            a = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad():
                a = int(online_net(s.unsqueeze(0)).argmax().item())

        r = reward_2d(bw_idx, di, a) if use_2d else reward_1d(bw_idx, a)

        next_bw = sample_bw_idx()
        next_di = random.randint(0, N_DENSITY_BINS - 1) if use_2d else 0
        s_next  = state_2d(next_bw, next_di) if use_2d else state_1d(next_bw)
        buffer.push(s, a, r, s_next, False)

        if len(buffer) >= WARMUP:
            batch = buffer.sample(BATCH_SIZE)
            bs  = torch.stack([t.s for t in batch])
            ba  = torch.tensor([t.a for t in batch], dtype=torch.long)
            br  = torch.tensor([t.r for t in batch], dtype=torch.float32)
            bsn = torch.stack([t.s_next for t in batch])

            with torch.no_grad():
                next_actions = online_net(bsn).argmax(dim=1)
                next_q = target_net(bsn).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = br + GAMMA * next_q

            current_q = online_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(online_net.state_dict())

        if ep % 500 == 0:
            total_r = 0.0
            if use_2d:
                for bi in range(N_BW):
                    for di2 in range(N_DENSITY_BINS):
                        with torch.no_grad():
                            ai = int(online_net(state_2d(bi, di2).unsqueeze(0)).argmax().item())
                        total_r += reward_2d(bi, di2, ai) * BW_WEIGHTS[bi]
                mean_r = total_r / N_DENSITY_BINS
            else:
                for bi in range(N_BW):
                    with torch.no_grad():
                        ai = int(online_net(state_1d(bi).unsqueeze(0)).argmax().item())
                    total_r += reward_1d(bi, ai) * BW_WEIGHTS[bi]
                mean_r = total_r
            curve.append({"episode": ep, "epsilon": round(eps, 6),
                          "mean_reward": round(mean_r, 6)})

    # Extract policy
    policy = {}
    lats   = []
    if use_2d:
        for bi, bw in enumerate(BANDWIDTHS):
            for di in range(N_DENSITY_BINS):
                with torch.no_grad():
                    q = online_net(state_2d(bi, di).unsqueeze(0)).squeeze(0)
                ai = int(q.argmax().item())
                key = f"{bw}_{di}"
                policy[key] = ACTIONS[ai]
                lats.append(latency_2d[bi][di][ai] * BW_WEIGHTS[bi])
    else:
        for bi, bw in enumerate(BANDWIDTHS):
            with torch.no_grad():
                q = online_net(state_1d(bi).unsqueeze(0)).squeeze(0)
            ai = int(q.argmax().item())
            policy[str(bw)] = ACTIONS[ai]
            lats.append(latency_table[bi][ai] * BW_WEIGHTS[bi])

    mean_lat = float(np.sum(lats)) if not use_2d else float(np.sum(lats) / N_DENSITY_BINS)
    print(f"  [{name}] mean_latency={mean_lat:.4f}s")
    return policy, mean_lat, curve, online_net


# ── Run all 4 ablation variants ───────────────────────────────────────────────
print("=== Ablation Study: 4 Variants ===")
print("A: 1D state + Simple DQN")
policy_A, lat_A, curve_A, net_A = train_variant("A: 1D+Simple",   use_2d=False, use_dueling=False)
print("B: 1D state + Dueling DQN")
policy_B, lat_B, curve_B, net_B = train_variant("B: 1D+Dueling",  use_2d=False, use_dueling=True)
print("C: 2D state + Simple DQN")
policy_C, lat_C, curve_C, net_C = train_variant("C: 2D+Simple",   use_2d=True,  use_dueling=False)
print("D: 2D state + Dueling DQN (Proposed)")
policy_D, lat_D, curve_D, net_D = train_variant("D: 2D+Dueling",  use_2d=True,  use_dueling=True)

# Baseline: weighted mean latency (using realistic BW distribution)
lat_L3_1d = float(np.sum([latency_table[bi][ACT_TO_IDX["canny_png_L3"]] * BW_WEIGHTS[bi] for bi in range(N_BW)]))
lat_L9_1d = float(np.sum([latency_table[bi][ACT_TO_IDX["canny_png_L9"]] * BW_WEIGHTS[bi] for bi in range(N_BW)]))
lat_L3_2d = float(np.sum([latency_2d[bi][di][ACT_TO_IDX["canny_png_L3"]] * BW_WEIGHTS[bi]
                           for bi in range(N_BW) for di in range(N_DENSITY_BINS)]) / N_DENSITY_BINS)
lat_L9_2d = float(np.sum([latency_2d[bi][di][ACT_TO_IDX["canny_png_L9"]] * BW_WEIGHTS[bi]
                           for bi in range(N_BW) for di in range(N_DENSITY_BINS)]) / N_DENSITY_BINS)

# Use 1D baseline for 1D variants, 2D baseline for 2D variants
lat_L3 = lat_L3_1d  # kept for backward compat in ablation_summary

# Load cloud_condition_png from agg directly
cloud_png_lats = [float(r["total_time_mean"]) for r in agg if r["strategy"] == "cloud_condition_png"]
lat_cloud_png  = float(np.mean(cloud_png_lats)) if cloud_png_lats else 1.929

print(f"\n=== Results ===")
print(f"  Baseline L3 (1D):   {lat_L3_1d:.4f}s")
print(f"  Baseline L3 (2D):   {lat_L3_2d:.4f}s")
print(f"  Baseline L9 (1D):   {lat_L9_1d:.4f}s")
print(f"  Baseline Cloud-PNG: {lat_cloud_png:.4f}s")
print(f"  A (1D+Simple):      {lat_A:.4f}s  gap_vs_L3_1d={(lat_L3_1d-lat_A)/lat_L3_1d*100:.1f}%")
print(f"  B (1D+Dueling):     {lat_B:.4f}s  gap_vs_L3_1d={(lat_L3_1d-lat_B)/lat_L3_1d*100:.1f}%")
print(f"  C (2D+Simple):      {lat_C:.4f}s  gap_vs_L3_2d={(lat_L3_2d-lat_C)/lat_L3_2d*100:.1f}%")
print(f"  D (2D+Dueling):     {lat_D:.4f}s  gap_vs_L3_2d={(lat_L3_2d-lat_D)/lat_L3_2d*100:.1f}%")

# Verify D policy vs oracle
print("\n=== Proposed (D) policy vs 2D oracle ===")
oracle_lats = []
for bi, bw in enumerate(BANDWIDTHS):
    for di in range(N_DENSITY_BINS):
        oracle_ai = int(np.argmin(latency_2d[bi][di]))
        oracle_lat = latency_2d[bi][di][oracle_ai]
        oracle_lats.append(oracle_lat)
        key = f"{bw}_{di}"
        chosen = policy_D.get(key, "?")
        chosen_ai = ACT_TO_IDX.get(chosen, -1)
        chosen_lat = latency_2d[bi][di][chosen_ai] if chosen_ai >= 0 else 9.0
        match = "OK" if chosen == ACTIONS[oracle_ai] else "XX"
        print(f"  {match} BW={bw:4.1f} di={di}: chosen={chosen:30s} {chosen_lat:.4f}s  oracle={ACTIONS[oracle_ai]:30s} {oracle_lat:.4f}s")
print(f"Oracle 2D mean: {np.mean(oracle_lats):.4f}s")

# ── Extract Q-values for proposed model (D) ───────────────────────────────────
q_values_2d = {}
for bi, bw in enumerate(BANDWIDTHS):
    for di in range(N_DENSITY_BINS):
        with torch.no_grad():
            q = net_D(state_2d(bi, di).unsqueeze(0)).squeeze(0)
        key = f"{bw}_{di}"
        q_values_2d[key] = {ACTIONS[ai]: round(float(q[ai]), 6) for ai in range(N_ACTIONS)}

# ── Save all outputs ──────────────────────────────────────────────────────────
ablation_summary = {
    "variants": {
        "A_1D_Simple":   {"mean_latency": round(lat_A, 6), "gap_vs_L3_pct": round((lat_L3_1d-lat_A)/lat_L3_1d*100, 2)},
        "B_1D_Dueling":  {"mean_latency": round(lat_B, 6), "gap_vs_L3_pct": round((lat_L3_1d-lat_B)/lat_L3_1d*100, 2)},
        "C_2D_Simple":   {"mean_latency": round(lat_C, 6), "gap_vs_L3_pct": round((lat_L3_2d-lat_C)/lat_L3_2d*100, 2)},
        "D_2D_Dueling":  {"mean_latency": round(lat_D, 6), "gap_vs_L3_pct": round((lat_L3_2d-lat_D)/lat_L3_2d*100, 2)},
    },
    "baselines": {
        "canny_png_L3":        round(lat_L3_1d, 6),
        "canny_png_L3_2d":     round(lat_L3_2d, 6),
        "canny_png_L9":        round(lat_L9_1d, 6),
        "cloud_condition_png": round(lat_cloud_png, 6),
    },
    "proposed_policy": policy_D,
    "proposed_mean_latency": round(lat_D, 6),
}

(OUT_DIR / "ablation_summary.json").write_text(
    json.dumps(ablation_summary, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT_DIR / "dueling_ddqn_2d_policy.json").write_text(
    json.dumps(policy_D, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT_DIR / "dueling_ddqn_2d_training_curve.json").write_text(
    json.dumps(curve_D, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT_DIR / "dueling_ddqn_2d_q_values.json").write_text(
    json.dumps(q_values_2d, indent=2, ensure_ascii=False), encoding="utf-8")

# Also save all variant curves for ablation figure
all_curves = {
    "A_1D_Simple":  curve_A,
    "B_1D_Dueling": curve_B,
    "C_2D_Simple":  curve_C,
    "D_2D_Dueling": curve_D,
}
(OUT_DIR / "ablation_training_curves.json").write_text(
    json.dumps(all_curves, indent=2, ensure_ascii=False), encoding="utf-8")

# Save training curves separately for convergence plots
training_curves = {
    "episodes": EPISODES,
    "eps_decay_steps": EPS_DECAY_STEPS,
    "checkpoint_interval": 500,
    "curves": all_curves,
}
(OUT_DIR / "training_curves.json").write_text(
    json.dumps(training_curves, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nSaved to {OUT_DIR}")
