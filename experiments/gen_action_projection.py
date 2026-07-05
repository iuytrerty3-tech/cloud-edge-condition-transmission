import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - fallback for lean environments
    TSNE = None
    PCA = None


ACTION_META = {
    "cond_png_l6_ds1": ("PNG-L6 / ds1", "#1f77b4"),
    "cond_png_l3_ds1": ("PNG-L3 / ds1", "#d62728"),
}

CATEGORY_META = {
    "artifact": "Artifact",
    "cloth": "Cultural Clothing",
    "papercut": "Paper-Cutting",
    "porcelain": "Blue-and-White Porcelain",
    "window": "Window-Flower",
}

BANDWIDTHS = [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def overleaf_root() -> Path:
    return repo_root().parent / "wenchuang_overleaf"


def parse_category(sample_id: str) -> str:
    prefix = sample_id.split("_", 1)[0]
    return CATEGORY_META.get(prefix, prefix.title())


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def project_features(x: np.ndarray) -> np.ndarray:
    if TSNE is not None and len(x) >= 30:
        return TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(x) // 4)),
            random_state=42,
            init="pca",
            learning_rate="auto",
        ).fit_transform(x)
    if PCA is not None:
        return PCA(n_components=2, random_state=42).fit_transform(x)

    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def load_policy_df() -> pd.DataFrame:
    policy_path = repo_root() / "real_runs" / "hybrid_ddpg_fullreal_split_v1" / "policy.json"
    with policy_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for key, value in data.items():
        sample_id, bw_text = key.split("@")
        bw = float(bw_text)
        state = value["state_vector"]
        action = value["selected_action"]
        category = parse_category(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "bandwidth_mbps": bw,
                "selected_action": action,
                "state_0_bandwidth": state[0],
                "state_1_jitter": state[1],
                "state_2_payload_complexity": state[2],
                "state_3_extract_complexity": state[3],
                "state_4_encode_complexity": state[4],
                "cat_0": state[5],
                "cat_1": state[6],
                "cat_2": state[7],
                "cat_3": state[8],
                "cat_4": state[9],
                "cat_5": state[10],
            }
        )

    df = pd.DataFrame(rows).sort_values(["sample_id", "bandwidth_mbps"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.startswith("state_") or c.startswith("cat_")]
    coords = project_features(df[feature_cols].to_numpy(dtype=float))
    df["proj_x"] = coords[:, 0]
    df["proj_y"] = coords[:, 1]
    df["action_label"] = df["selected_action"].map(lambda x: ACTION_META[x][0])
    return df


def compute_heatmaps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    heatmap = (
        df.assign(is_l3=(df["selected_action"] == "cond_png_l3_ds1").astype(float))
        .pivot_table(
            index="category",
            columns="bandwidth_mbps",
            values="is_l3",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(columns=BANDWIDTHS)
    )
    share = (
        df.groupby(["bandwidth_mbps", "action_label"]).size().reset_index(name="count")
    )
    total = share.groupby("bandwidth_mbps")["count"].transform("sum")
    share["ratio"] = share["count"] / total
    return heatmap, share


def export_artifacts(df: pd.DataFrame, heatmap: pd.DataFrame, share: pd.DataFrame) -> None:
    out_dir = overleaf_root() / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fig19_action_projection_points.csv", index=False)
    heatmap.reset_index().to_csv(out_dir / "fig19_action_heatmap.csv", index=False)
    share.to_csv(out_dir / "fig19_action_bandwidth_share.csv", index=False)


def plot_projection_suite(df: pd.DataFrame, heatmap: pd.DataFrame, share: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    for action, (label, color) in ACTION_META.items():
        sdf = df[df["selected_action"] == action]
        ax1.scatter(
            sdf["proj_x"],
            sdf["proj_y"],
            s=55,
            alpha=0.85,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        ax1.scatter(
            [sdf["proj_x"].mean()],
            [sdf["proj_y"].mean()],
            s=180,
            marker="X",
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
    ax1.set_title("(a) State-Action Projection Colored by Selected Branch")
    ax1.set_xlabel("Projection Dimension 1")
    ax1.set_ylabel("Projection Dimension 2")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best", frameon=True)

    scatter = ax2.scatter(
        df["proj_x"],
        df["proj_y"],
        c=df["bandwidth_mbps"],
        s=55,
        cmap="viridis",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
    )
    ax2.set_title("(b) Same Projection Colored by Bandwidth")
    ax2.set_xlabel("Projection Dimension 1")
    ax2.set_ylabel("Projection Dimension 2")
    ax2.grid(True, linestyle="--", alpha=0.4)
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label("Bandwidth (Mbps)")

    hm = heatmap.to_numpy()
    im = ax3.imshow(hm, cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
    ax3.set_title("(c) L3 Selection Rate by Category and Bandwidth")
    ax3.set_xticks(np.arange(len(heatmap.columns)))
    ax3.set_xticklabels([str(bw) for bw in heatmap.columns])
    ax3.set_yticks(np.arange(len(heatmap.index)))
    ax3.set_yticklabels(heatmap.index)
    ax3.set_xlabel("Bandwidth (Mbps)")
    ax3.set_ylabel("Category")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax3.text(
                j,
                i,
                f"{hm[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if hm[i, j] < 0.6 else "white",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="L3 Ratio")

    pivot = (
        share.pivot(index="bandwidth_mbps", columns="action_label", values="ratio")
        .fillna(0.0)
        .reindex(BANDWIDTHS)
    )
    bottom = np.zeros(len(pivot))
    for label in [ACTION_META["cond_png_l6_ds1"][0], ACTION_META["cond_png_l3_ds1"][0]]:
        color = "#1f77b4" if "L6" in label else "#d62728"
        vals = pivot[label].to_numpy()
        ax4.bar(
            pivot.index.astype(str),
            vals,
            bottom=bottom,
            color=color,
            label=label,
            alpha=0.9,
        )
        bottom += vals
    ax4.set_title("(d) Branch-Migration Pattern Across Bandwidth")
    ax4.set_xlabel("Bandwidth (Mbps)")
    ax4.set_ylabel("Selection Ratio")
    ax4.set_ylim(0.0, 1.05)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Action Projection Interpretability from Real Policy States",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    fig_dir = overleaf_root() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig19_action_projection_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "fig19_action_projection_map.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = load_policy_df()
    heatmap, share = compute_heatmaps(df)
    export_artifacts(df, heatmap, share)
    plot_projection_suite(df, heatmap, share)
    print("Generated fig19_action_projection_map.(png|pdf)")
