"""
Train a Double DQN scheduler on the extended benchmark.
State: discretized bandwidth (5 bins).
Actions: 9 strategies (all except cloud_condition_png kept as baseline reference).
Output: real_runs/extended_benchmark/ddqn_policy.json + ddqn_training_curve.json
"""
from __future__ import annotations
import json, random, collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── load benchmark data ──────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "real_runs/extended_benchmark"
agg = json.loads((OUT_DIR / "aggregated_results.json").read_text(encoding="utf-8"))

BANDWIDTHS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
BW_TO_IDX  = {bw: i for i, bw in enumerate(BANDWIDTHS)}
N_STATES   = len(BANDWIDTHS)

# Strategies available to the DDQN (exclude cloud_condition_png — kept as pure baseline)
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

# Build lookup: latency[bw_idx][act_idx] and ssim[bw_idx][act_idx]
latency_table = np.full((N_STATES, N_ACTIONS), 9.0)
ssim_table    = np.full((N_STATES, N_ACTIONS), 0.0)
for row in agg:
    if row["strategy"] not in ACT_TO_IDX: continue
    bi = BW_TO_IDX.get(float(row["bandwidth_mbps"]))
    ai = ACT_TO_IDX[row["strategy"]]
    if bi is None: continue
    latency_table[bi][ai] = row["total_time_mean"]
    ssim_table[bi][ai]    = row["condition_ssim_mean"]

# ── Double DQN network ────────────────────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_to_tensor(bw_idx: int) -> torch.Tensor:
    """One-hot encode bandwidth index."""
    t = torch.zeros(N_STATES)
    t[bw_idx] = 1.0
    return t


# ── Replay buffer ─────────────────────────────────────────────────────────────
Transition = collections.namedtuple("Transition", ["s", "a", "r", "s_next", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int = 4096):
        self.buf: collections.deque = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# ── reward function ────────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 0.95
QUALITY_PENALTY   = 5.0
MAX_LATENCY       = max(latency_table.max(), 1.0)

def reward(bw_idx: int, act_idx: int) -> float:
    lat  = latency_table[bw_idx][act_idx]
    ssim = ssim_table[bw_idx][act_idx]
    penalty = QUALITY_PENALTY if ssim < QUALITY_THRESHOLD else 0.0
    return -lat - penalty


# ── training ──────────────────────────────────────────────────────────────────
EPISODES        = 100_000
LR              = 3e-4
GAMMA           = 0.9
BATCH_SIZE      = 128
TARGET_UPDATE   = 200
EPS_START       = 0.8
EPS_END         = 0.01
EPS_DECAY_STEPS = 80_000
WARMUP          = 256

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

online_net = QNet(N_STATES, N_ACTIONS)
target_net = QNet(N_STATES, N_ACTIONS)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimizer = optim.Adam(online_net.parameters(), lr=LR)
buffer    = ReplayBuffer(capacity=8192)
loss_fn   = nn.MSELoss()

training_curve = []

def get_epsilon(step: int) -> float:
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + (EPS_END - EPS_START) * frac


for ep in range(1, EPISODES + 1):
    bw_idx = random.randint(0, N_STATES - 1)
    s = state_to_tensor(bw_idx)
    eps = get_epsilon(ep)

    # ε-greedy action
    if random.random() < eps:
        a = random.randint(0, N_ACTIONS - 1)
    else:
        with torch.no_grad():
            a = int(online_net(s.unsqueeze(0)).argmax().item())

    r = reward(bw_idx, a)

    # next state: random bandwidth (episodic, no real transition)
    next_bw = random.randint(0, N_STATES - 1)
    s_next  = state_to_tensor(next_bw)
    buffer.push(s, a, r, s_next, False)

    # learn
    if len(buffer) >= WARMUP:
        batch = buffer.sample(BATCH_SIZE)
        bs  = torch.stack([t.s for t in batch])
        ba  = torch.tensor([t.a for t in batch], dtype=torch.long)
        br  = torch.tensor([t.r for t in batch], dtype=torch.float32)
        bsn = torch.stack([t.s_next for t in batch])

        # Double DQN target: online net selects action, target net evaluates
        with torch.no_grad():
            next_actions = online_net(bsn).argmax(dim=1)
            next_q = target_net(bsn).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = br + GAMMA * next_q

        current_q = online_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
        loss = loss_fn(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())

    if ep % 500 == 0:
        # evaluate greedy policy
        total_r = 0.0
        for bi in range(N_STATES):
            with torch.no_grad():
                ai = int(online_net(state_to_tensor(bi).unsqueeze(0)).argmax().item())
            total_r += reward(bi, ai)
        mean_r = total_r / N_STATES
        training_curve.append({"episode": ep, "epsilon": round(eps, 6),
                                "mean_reward": round(mean_r, 6)})
        if ep % 5000 == 0:
            print(f"  ep {ep:6d}  eps={eps:.3f}  mean_reward={mean_r:.4f}")


# ── extract learned policy ────────────────────────────────────────────────────
policy = {}
ddqn_latency_per_bw = {}
for bi, bw in enumerate(BANDWIDTHS):
    with torch.no_grad():
        q_vals = online_net(state_to_tensor(bi).unsqueeze(0)).squeeze(0)
    ai = int(q_vals.argmax().item())
    best_strategy = ACTIONS[ai]
    policy[str(bw)] = best_strategy
    ddqn_latency_per_bw[bw] = latency_table[bi][ai]
    print(f"  {bw:4.0f} Mbps -> {best_strategy:30s}  latency={latency_table[bi][ai]:.4f}s  q={float(q_vals[ai]):.4f}")

mean_latency = float(np.mean(list(ddqn_latency_per_bw.values())))
print(f"\nDDQN mean latency: {mean_latency:.4f} s")

# compare with best fixed strategy per bandwidth
best_fixed_lats = []
for bi, bw in enumerate(BANDWIDTHS):
    best_lat = min(latency_table[bi])
    best_fixed_lats.append(best_lat)
best_fixed_mean = float(np.mean(best_fixed_lats))
print(f"Best fixed strategy (oracle) mean: {best_fixed_mean:.4f} s")
print(f"DDQN vs best-fixed improvement: {100*(best_fixed_mean-mean_latency)/best_fixed_mean:.2f}%")

# ── save ──────────────────────────────────────────────────────────────────────
(OUT_DIR / "ddqn_policy.json").write_text(
    json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT_DIR / "ddqn_training_curve.json").write_text(
    json.dumps(training_curve, indent=2, ensure_ascii=False), encoding="utf-8")

# Also save Q-table for comparison
qtable = {}
for bi, bw in enumerate(BANDWIDTHS):
    with torch.no_grad():
        q_vals = online_net(state_to_tensor(bi).unsqueeze(0)).squeeze(0)
    qtable[str(bw)] = {ACTIONS[ai]: round(float(q_vals[ai]), 6) for ai in range(N_ACTIONS)}
(OUT_DIR / "ddqn_q_values.json").write_text(
    json.dumps(qtable, indent=2, ensure_ascii=False), encoding="utf-8")

summary = {
    "ddqn_mean_latency": round(mean_latency, 6),
    "ddqn_policy": policy,
    "per_bw_latency": {str(bw): round(ddqn_latency_per_bw[bw], 6) for bw in BANDWIDTHS},
}
(OUT_DIR / "ddqn_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved to {OUT_DIR}")
