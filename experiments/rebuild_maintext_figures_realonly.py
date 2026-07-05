"""Rebuild every main-text benchmark figure and table statistic from the
preserved real measurement records only.

All quantities are computed on the 31-image held-out evaluation split
(experiments/fullreal_eval_ids_v1.json) so that measured policies and
fixed-branch baselines share exactly the same population.

Sources (all real, shipped with the repository):
  * real_runs/unified_algorithm_benchmark_fullreal_split_v1/sample_stats.json
      per-sample measured rows for Ours-Hybrid-DDPG, Dueling-DDQN, PPO, A2C,
      GBDT/RF oracle imitators, Rule-threshold, Cond-PNG-L6, Cloud-JPEG-q75,
      Cloud-PNG.
  * real_runs/unified_baselines_fullreal_split_v1/{bandwidth_only_rule,
      random_selector}/rl_sample_stats.json
      per-sample measured rows for the bandwidth-threshold heuristic and the
      random selector (the latter also provides measured per-branch encode
      times for PNG-L1/L3/L9).
  * real_runs/seq/action_effect_table.json
      deterministic per-image payload/SSIM for every codec branch.

Reconstructed rows (marked "recon"): fixed PNG-L1/L3/L9 latencies are rebuilt
per (image, bandwidth) as
    T = T_extract(img, bw) + T_enc(branch) + 0.02 + payload_kb * 8 / (1024*bw)
        + T_decode(img, bw)
with T_extract/T_decode taken from the measured Cond-PNG-L6 row of the same
(image, bandwidth) state, T_enc(branch) the measured mean encode time of that
branch from the random-selector traces, and the same analytic uplink model
(20 ms RTT + size/bandwidth) used by the benchmark. This mirrors the dagger
convention of Table 2 in the manuscript.

Usage: python experiments/rebuild_maintext_figures_realonly.py
Outputs: real_runs/maintext_figures_realonly_v1/*.png|pdf + numbers.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "real_runs" / "maintext_figures_realonly_v1"
OUT.mkdir(parents=True, exist_ok=True)

GRID = [1.0, 2.0, 3.0, 5.0, 10.0]
SWEEP = [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
RTT_S = 0.02
P_EDGE = 5.0  # W, edge processing power (extract + encode)
P_TX = 2.0    # W, uplink radio power

DISPLAY = {
    "ours_hybrid_ddpg": "Ours-Hybrid-DDPG",
    "dueling_ddqn": "Dueling-DDQN",
    "ppo_scheduler": "PPO",
    "a2c_scheduler": "A2C",
    "gbdt_selector": "GBDT-oracle",
    "random_forest_selector": "RandomForest-oracle",
    "rule_threshold": "Rule-threshold",
    "bandwidth_only_rule": "Bandwidth-threshold",
    "random_selector": "Random-Selector",
    "cond_png_l6_ds1": "Cond-PNG-L6",
    "cond_png_l1_recon": "Cond-PNG-L1",
    "cond_png_l3_recon": "Cond-PNG-L3",
    "cond_png_l9_recon": "Cond-PNG-L9",
    "cloud_jpeg_q75": "Cloud-JPEG-q75",
    "cloud_png": "Cloud-PNG",
}
COLORS = {
    "ours_hybrid_ddpg": "#d62728",
    "dueling_ddqn": "#1f77b4",
    "ppo_scheduler": "#2ca02c",
    "a2c_scheduler": "#9467bd",
    "gbdt_selector": "#8c564b",
    "random_forest_selector": "#7f7f7f",
    "rule_threshold": "#7f7f7f",
    "bandwidth_only_rule": "#e377c2",
    "random_selector": "#e377c2",
    "cond_png_l6_ds1": "#bcbd22",
    "cond_png_l1_recon": "#17becf",
    "cond_png_l3_recon": "#aec7e8",
    "cond_png_l9_recon": "#ff7f0e",
    "cloud_jpeg_q75": "#aec7e8",
    "cloud_png": "#3b3b6d",
}


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def rows_of(payload):
    return payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload


def main() -> None:
    eval_ids = set(load_json(ROOT / "experiments" / "fullreal_eval_ids_v1.json"))
    v1 = rows_of(load_json(ROOT / "real_runs" / "unified_algorithm_benchmark_fullreal_split_v1" / "sample_stats.json"))
    bwr = rows_of(load_json(ROOT / "real_runs" / "unified_baselines_fullreal_split_v1" / "bandwidth_only_rule" / "rl_sample_stats.json"))
    rnd = rows_of(load_json(ROOT / "real_runs" / "unified_baselines_fullreal_split_v1" / "random_selector" / "rl_sample_stats.json"))
    table = load_json(ROOT / "real_runs" / "seq" / "action_effect_table.json")["images"]

    # ---- measured per-sample records, held-out split only -------------------
    # rec[strategy][(sample_id, bw)] = row
    rec: dict[str, dict] = defaultdict(dict)
    for r in v1:
        if r["sample_id"] in eval_ids:
            rec[r["strategy"]][(r["sample_id"], r["bandwidth_mbps"])] = r
    for r in bwr:
        rec["bandwidth_only_rule"][(r["sample_id"], r["bandwidth_mbps"])] = r
    for r in rnd:
        rec["random_selector"][(r["sample_id"], r["bandwidth_mbps"])] = r

    # ---- measured branch encode times from the random-selector traces -------
    enc_branch = defaultdict(list)
    for r in rnd:
        enc_branch[r["selected_action"]].append(r["encode_time_sec"])
    enc_mean = {k: float(np.mean(v)) for k, v in enc_branch.items()}

    # ---- reconstructed fixed-branch rows (PNG-L1/L3/L9) ---------------------
    for lvl in (1, 3, 9):
        strat = f"cond_png_l{lvl}_recon"
        enc = enc_mean[f"cond_png_l{lvl}_ds1"]
        for (sid, bw), base in rec["cond_png_l6_ds1"].items():
            pay = table[sid]["actions"]["canny"]["png"][f"L{lvl}_ds1"]["payload_kb"]
            ssim = table[sid]["actions"]["canny"]["png"][f"L{lvl}_ds1"]["ssim"]
            net = RTT_S + pay * 8.0 / (1024.0 * bw)
            rec[strat][(sid, bw)] = {
                "sample_id": sid,
                "bandwidth_mbps": bw,
                "payload_kb": pay,
                "condition_ssim": ssim,
                "extract_time_sec": base["extract_time_sec"],
                "encode_time_sec": enc,
                "network_time_sec": net,
                "decode_time_sec": base["decode_time_sec"],
                "total_time_sec": base["extract_time_sec"] + enc + net + base["decode_time_sec"],
            }

    strategies = [
        "ours_hybrid_ddpg", "dueling_ddqn", "ppo_scheduler", "a2c_scheduler",
        "gbdt_selector", "random_forest_selector", "rule_threshold",
        "bandwidth_only_rule", "random_selector", "cond_png_l6_ds1",
        "cond_png_l1_recon", "cond_png_l3_recon", "cond_png_l9_recon",
        "cloud_jpeg_q75", "cloud_png",
    ]

    def per_bw(strat: str, field: str, bws=None):
        bws = bws or GRID
        out = []
        for bw in bws:
            vals = [row[field] for (sid, b), row in rec[strat].items() if b == bw]
            out.append(float(np.mean(vals)))
        return out

    def grid_mean(strat: str, field: str) -> float:
        return float(np.mean(per_bw(strat, field)))

    def energy_of(row) -> float:
        return P_EDGE * (row["extract_time_sec"] + row["encode_time_sec"]) + P_TX * row["network_time_sec"]

    numbers = {"grid": GRID, "population": "31 held-out images"}
    numbers["latency_rows"] = {
        s: {"per_bw": [round(x, 4) for x in per_bw(s, "total_time_sec")],
            "mean": round(grid_mean(s, "total_time_sec"), 4)}
        for s in strategies
    }
    numbers["payload_rows"] = {
        s: {"payload_kb": round(grid_mean(s, "payload_kb"), 1),
            "ssim": round(grid_mean(s, "condition_ssim"), 4)}
        for s in strategies
    }
    numbers["encode_ms"] = {k: round(v * 1000, 2) for k, v in sorted(enc_mean.items())}
    l6_rows = [row for (sid, bw), row in rec["cond_png_l6_ds1"].items()]
    numbers["encode_ms"]["cond_png_l6_v1_measured"] = round(float(np.mean([r["encode_time_sec"] for r in l6_rows])) * 1000, 2)

    ours_lat = numbers["latency_rows"]["ours_hybrid_ddpg"]["mean"]
    cp_lat = numbers["latency_rows"]["cloud_png"]["mean"]
    ours_pay = numbers["payload_rows"]["ours_hybrid_ddpg"]["payload_kb"]
    cp_pay = numbers["payload_rows"]["cloud_png"]["payload_kb"]
    numbers["headline"] = {
        "payload_reduction_pct": round(100 * (1 - ours_pay / cp_pay), 1),
        "latency_reduction_pct": round(100 * (1 - ours_lat / cp_lat), 1),
        "ours_payload_kb": ours_pay, "cloudpng_payload_kb": cp_pay,
        "ours_latency_s": ours_lat, "cloudpng_latency_s": cp_lat,
    }

    # ---- tail latency stats over grid samples -------------------------------
    tail_strats = ["ours_hybrid_ddpg", "a2c_scheduler", "random_selector", "cond_png_l1_recon", "cloud_png"]
    tails = {}
    for s in tail_strats:
        vals = np.array([row["total_time_sec"] for (sid, bw), row in rec[s].items() if bw in GRID])
        tails[s] = {"p90": float(np.percentile(vals, 90)), "p95": float(np.percentile(vals, 95)), "max": float(vals.max())}
    numbers["tails"] = {s: {k: round(v, 3) for k, v in d.items()} for s, d in tails.items()}

    # ---- fig7 tail latency ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12.5, 5))
    width = 0.25
    xs = np.arange(len(tail_strats))
    for i, (stat, color) in enumerate((("p90", "#9ecae1"), ("p95", "#4292c6"), ("max", "#2c3e8c"))):
        ax.bar(xs + (i - 1) * width, [tails[s][stat] for s in tail_strats], width, label=stat, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[s] for s in tail_strats], rotation=20, ha="right")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Tail Latency on the Held-Out Benchmark (31 images, 1--10 Mbps)")
    ax.legend(ncol=3, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_tail_latency.png", dpi=200)
    plt.close(fig)

    # ---- fig1 latency vs bandwidth ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in strategies:
        if s in ("ours_q_policy",):
            continue
        style = "-" if s == "ours_hybrid_ddpg" else "--" if "recon" in s or "cloud" in s else "-"
        lw = 2.5 if s == "ours_hybrid_ddpg" else 1.3
        ax.plot(SWEEP if s not in () else GRID, per_bw(s, "total_time_sec", SWEEP), marker="o", ms=3,
                linestyle=style, linewidth=lw, label=DISPLAY[s], color=COLORS[s], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Uplink bandwidth (Mbps)")
    ax.set_ylabel("Pipeline overhead (s, log)")
    ax.set_title("Pipeline overhead vs bandwidth (held-out, real measurements)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_unified_latency_vs_bw.png", dpi=200)
    plt.close(fig)

    # ---- fig4 overall ranking -------------------------------------------------
    ranking = sorted(strategies, key=lambda s: numbers["latency_rows"][s]["mean"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ys = np.arange(len(ranking))[::-1]
    for y, s in zip(ys, ranking):
        m = numbers["latency_rows"][s]["mean"]
        ax.barh(y, m, color=COLORS[s])
        ax.text(m * 1.05, y, f"{m:.4f}", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([DISPLAY[s] for s in ranking], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean pipeline overhead (s, log)")
    ax.set_title("Overall latency ranking (held-out, real measurements)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_unified_overall_ranking.png", dpi=200)
    plt.close(fig)

    # ---- fig9 latency breakdown at 1.5 Mbps ----------------------------------
    bk_strats = ["ours_hybrid_ddpg", "a2c_scheduler", "rule_threshold", "cond_png_l6_ds1",
                 "cond_png_l1_recon", "cond_png_l9_recon", "cloud_jpeg_q75", "cloud_png"]
    comps = ["extract_time_sec", "encode_time_sec", "network_time_sec", "decode_time_sec"]
    comp_labels = ["Extract", "Encode", "Network", "Decode"]
    comp_colors = ["#c6dbef", "#6baed6", "#fd8d3c", "#74c476"]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(bk_strats))
    bottom = np.zeros(len(bk_strats))
    for comp, lab, col in zip(comps, comp_labels, comp_colors):
        vals = []
        for s in bk_strats:
            rows = [row[comp] for (sid, bw), row in rec[s].items() if bw == 1.5]
            vals.append(float(np.mean(rows)))
        ax.bar(xs, vals, bottom=bottom, label=lab, color=col)
        bottom += np.array(vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[s] for s in bk_strats], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Latency component (s)")
    ax.set_title("Pipeline-latency breakdown at 1.5 Mbps (held-out)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_latency_breakdown.png", dpi=200)
    plt.close(fig)

    # ---- fig11 relative reduction vs Cloud-PNG --------------------------------
    red_strats = [s for s in strategies if s != "cloud_png"]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(red_strats))
    reds = [100 * (1 - numbers["latency_rows"][s]["mean"] / cp_lat) for s in red_strats]
    ax.bar(xs, reds, color=[COLORS[s] for s in red_strats])
    for x, v in zip(xs, reds):
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[s] for s in red_strats], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Latency reduction vs Cloud-PNG (%)")
    ax.set_title("Relative latency reduction (held-out, real measurements)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig11_relative_latency_reduction.png", dpi=200)
    plt.close(fig)

    # ---- fig12 throughput ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ths = [1.0 / numbers["latency_rows"][s]["mean"] for s in strategies]
    xs = np.arange(len(strategies))
    ax.bar(xs, ths, color=[COLORS[s] for s in strategies])
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[s] for s in strategies], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Throughput (images / s, overhead only)")
    ax.set_title("Pipeline throughput (held-out, real measurements)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig12_system_throughput.png", dpi=200)
    plt.close(fig)

    # ---- fig17 energy suite (4 panels, real) ----------------------------------
    en_strats = ["ours_hybrid_ddpg", "a2c_scheduler", "rule_threshold", "cond_png_l6_ds1",
                 "cond_png_l1_recon", "cond_png_l9_recon", "cloud_jpeg_q75", "cloud_png"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for s in en_strats:
        es = []
        for bw in GRID:
            rows = [energy_of(row) for (sid, b), row in rec[s].items() if b == bw]
            es.append(float(np.mean(rows)))
        ax.plot(GRID, es, marker="o", label=DISPLAY[s], color=COLORS[s], linewidth=2 if s == "ours_hybrid_ddpg" else 1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Edge energy per image (J, log)")
    ax.set_title("(a) Edge energy vs bandwidth")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    effs = []
    for s in en_strats:
        rows = [energy_of(row) for (sid, b), row in rec[s].items() if b in GRID]
        effs.append(1.0 / float(np.mean(rows)))
    xs = np.arange(len(en_strats))
    ax.bar(xs, effs, color=[COLORS[s] for s in en_strats])
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[s] for s in en_strats], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Images per joule")
    ax.set_title("(b) Energy-conversion efficiency (grid mean)")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    rep_bws = [1.0, 3.0, 10.0]
    width = 0.35
    xs = np.arange(len(rep_bws))
    for off, s in ((-0.5, "ours_hybrid_ddpg"), (0.5, "cloud_png")):
        proc, tx = [], []
        for bw in rep_bws:
            rows = [row for (sid, b), row in rec[s].items() if b == bw]
            proc.append(float(np.mean([P_EDGE * (r["extract_time_sec"] + r["encode_time_sec"]) for r in rows])))
            tx.append(float(np.mean([P_TX * r["network_time_sec"] for r in rows])))
        ax.bar(xs + off * width, proc, width, label=f"{DISPLAY[s]}: processing",
               color=COLORS[s], alpha=0.55)
        ax.bar(xs + off * width, tx, width, bottom=proc, label=f"{DISPLAY[s]}: transmission",
               color=COLORS[s])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b:g} Mbps" for b in rep_bws])
    ax.set_ylabel("Energy per image (J)")
    ax.set_title("(c) Processing vs transmission energy")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    for s in en_strats:
        rows = [row for (sid, b), row in rec[s].items() if b in GRID]
        e = float(np.mean([energy_of(r) for r in rows]))
        th = 1.0 / float(np.mean([r["total_time_sec"] for r in rows]))
        ax.scatter(e, th, s=70, color=COLORS[s], label=DISPLAY[s])
    ax.set_xscale("log")
    ax.set_xlabel("Energy per image (J, log)")
    ax.set_ylabel("Throughput (images / s)")
    ax.set_title("(d) Energy input vs throughput output")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.suptitle("Edge-energy evaluation from per-sample real records ($P_{edge}$=5 W, $P_{tx}$=2 W)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig17_energy_suite.png", dpi=200)
    plt.close(fig)

    # ---- fig18 quality-energy-latency suite -----------------------------------
    te_strats = ["ours_hybrid_ddpg", "a2c_scheduler", "random_selector",
                 "cond_png_l1_recon", "cloud_jpeg_q75", "cloud_png"]
    bw_markers = {1.0: "o", 2.0: "s", 3.0: "^", 5.0: "D", 10.0: "P"}
    pts = []
    for s in te_strats:
        for bw in GRID:
            rows = [row for (sid, b), row in rec[s].items() if b == bw]
            pts.append({
                "s": s, "bw": bw,
                "lat": float(np.mean([r["total_time_sec"] for r in rows])),
                "en": float(np.mean([energy_of(r) for r in rows])),
                "q": float(np.mean([r["condition_ssim"] for r in rows])),
            })
    fig = plt.figure(figsize=(12, 9.5))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    for p in pts:
        ax.scatter(p["lat"], p["en"], p["q"], color=COLORS[p["s"]], marker=bw_markers[p["bw"]], s=45)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_zlabel("Condition SSIM")
    ax.set_title("(a) 3D quality-energy-latency view")
    ax2 = fig.add_subplot(2, 2, 2)
    for p in pts:
        ax2.scatter(p["lat"], p["en"], color=COLORS[p["s"]], marker=bw_markers[p["bw"]], s=45)
    ax2.set_xlabel("Latency (s)")
    ax2.set_ylabel("Energy (J)")
    ax2.set_title("(b) Latency-energy projection")
    ax2.grid(True, alpha=0.25)
    ax3 = fig.add_subplot(2, 2, 3)
    for p in pts:
        ax3.scatter(p["lat"], p["q"], color=COLORS[p["s"]], marker=bw_markers[p["bw"]], s=45)
    ax3.set_xlabel("Latency (s)")
    ax3.set_ylabel("Condition SSIM")
    ax3.set_title("(c) Latency-quality projection")
    ax3.grid(True, alpha=0.25)
    ax4 = fig.add_subplot(2, 2, 4)
    for p in pts:
        ax4.scatter(p["en"], p["q"], color=COLORS[p["s"]], marker=bw_markers[p["bw"]], s=45)
    ax4.set_xlabel("Energy (J)")
    ax4.set_ylabel("Condition SSIM")
    ax4.set_title("(d) Energy-quality projection")
    ax4.grid(True, alpha=0.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle="", marker="o", color=COLORS[s], label=DISPLAY[s]) for s in te_strats]
    handles += [Line2D([], [], linestyle="", marker=m, color="k", label=f"{int(b)} Mbps") for b, m in bw_markers.items()]
    ax4.legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("Quality-energy-latency trade-off from real benchmark records (held-out)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig18_three_objective_tradeoff.pdf")
    fig.savefig(OUT / "fig18_three_objective_tradeoff.png", dpi=200)
    plt.close(fig)

    # ---- fig21 bandwidth-crossover analytic sweep -----------------------------
    sweep = np.linspace(1, 50, 197)
    branch_curves = {}
    per_img = {}
    enc_l6_v1 = {sid: rec["cond_png_l6_ds1"][(sid, 1.0)]["encode_time_sec"] for (sid, bw) in rec["cond_png_l6_ds1"] if bw == 1.0}
    base_ext = {sid: rec["cond_png_l6_ds1"][(sid, 1.0)]["extract_time_sec"] for (sid, bw) in rec["cond_png_l6_ds1"] if bw == 1.0}
    base_dec = {sid: rec["cond_png_l6_ds1"][(sid, 1.0)]["decode_time_sec"] for (sid, bw) in rec["cond_png_l6_ds1"] if bw == 1.0}
    eval_list = sorted(base_ext.keys())
    for lvl in (1, 3, 6, 9):
        curves = []
        for sid in eval_list:
            pay = table[sid]["actions"]["canny"]["png"][f"L{lvl}_ds1"]["payload_kb"]
            enc = enc_l6_v1[sid] if lvl == 6 else enc_mean[f"cond_png_l{lvl}_ds1"]
            curves.append(base_ext[sid] + enc + RTT_S + pay * 8.0 / (1024.0 * sweep) + base_dec[sid])
        per_img[lvl] = np.array(curves)
        branch_curves[lvl] = np.mean(curves, axis=0)
    adaptive = np.mean(np.min(np.stack([per_img[l] for l in (1, 3, 6, 9)], axis=0), axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lvl, col in ((1, "#17becf"), (3, "#2ca02c"), (6, "#bcbd22"), (9, "#ff7f0e")):
        ax.plot(sweep, branch_curves[lvl], label=f"Fixed PNG-L{lvl}", color=col, linewidth=1.4)
    ax.plot(sweep, adaptive, label="Per-state adaptive (lower envelope)", color="#d62728", linewidth=2.4, linestyle="--")
    ax.set_xlabel("Uplink bandwidth (Mbps)")
    ax.set_ylabel("Mean pipeline overhead (s)")
    ax.set_title("Extended bandwidth sweep: fixed PNG branches vs per-state adaptive")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig21_bandwidth_crossover.png", dpi=200)
    plt.close(fig)

    # crossover + adaptive-gain statistics
    l6c, l3c = branch_curves[6], branch_curves[3]
    cross_idx = np.argmax(l3c < l6c)
    crossover_bw = float(sweep[cross_idx]) if (l3c < l6c).any() else None
    best_fixed_lvl = min((1, 3, 6, 9), key=lambda l: float(np.mean(branch_curves[l])))
    best_fixed = branch_curves[best_fixed_lvl]
    gain_avg = float(np.mean(1 - adaptive / best_fixed)) * 100
    gain_max = float(np.max(1 - adaptive / best_fixed)) * 100
    numbers["crossover"] = {
        "l6_to_l3_crossover_mbps": round(crossover_bw, 1) if crossover_bw else None,
        "best_single_fixed_branch": f"PNG-L{best_fixed_lvl}",
        "adaptive_gain_avg_pct_1_50": round(gain_avg, 1),
        "adaptive_gain_max_pct_1_50": round(gain_max, 1),
        "gain_at_50mbps_pct": round(float(1 - adaptive[-1] / best_fixed[-1]) * 100, 1),
    }

    # ---- e2e budget numbers ----------------------------------------------------
    tgen = 4.92
    e2e = {}
    for s in ("ours_hybrid_ddpg", "cloud_png"):
        for bw in (1.0, 10.0):
            rows = [row for (sid, b), row in rec[s].items() if b == bw]
            e2e[f"{s}@{bw:g}"] = {
                "extract": round(float(np.mean([r["extract_time_sec"] for r in rows])), 4),
                "encode": round(float(np.mean([r["encode_time_sec"] for r in rows])), 4),
                "network": round(float(np.mean([r["network_time_sec"] for r in rows])), 4),
                "decode": round(float(np.mean([r["decode_time_sec"] for r in rows])), 4),
                "pipeline_total": round(float(np.mean([r["total_time_sec"] for r in rows])), 4),
                "e2e_with_gen": round(float(np.mean([r["total_time_sec"] for r in rows])) + tgen, 2),
            }
    numbers["e2e"] = e2e

    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
