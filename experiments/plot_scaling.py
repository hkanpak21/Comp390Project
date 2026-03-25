#!/usr/bin/env python3
"""
plot_scaling.py — Generate scaling plots from bert_inference CSV results.

Usage:
    python3 experiments/plot_scaling.py \\
        --input experiments/results/ \\
        --output experiments/plots/

Reads:
    experiments/results/bert_Ngpu.csv   (one file per GPU count)

Produces:
    experiments/plots/bert_scaling_latency.pdf
    experiments/plots/bert_scaling_speedup.pdf
    experiments/plots/bert_breakdown_stacked.pdf
    experiments/plots/comm_overhead.pdf
"""

import argparse
import glob
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering (no display needed on EC2)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "lines.linewidth":  1.8,
    "lines.markersize": 7,
})

COLORS = {
    "matmul":    "#4472C4",
    "gelu":      "#ED7D31",
    "softmax":   "#A9D18E",
    "layernorm": "#FFC000",
    "keyswitch": "#FF0000",
    "comm":      "#7030A0",
    "other":     "#D9D9D9",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results(result_dir: str) -> pd.DataFrame:
    """Load all bert_*gpu.csv files from result_dir into one DataFrame."""
    frames = []
    pattern = os.path.join(result_dir, "bert_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No CSV files found in {result_dir} matching bert_*.csv")
        print("Run: ./build/benchmarks/bert_inference --n-gpus N --output experiments/results/bert_Ngpu.csv")
        sys.exit(1)

    for f in files:
        df = pd.read_csv(f)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    # Keep only the last row per n_gpus (latest run)
    data = data.sort_values("n_gpus").groupby("n_gpus").last().reset_index()
    return data


# ---------------------------------------------------------------------------
# Plot 1: Latency vs n_gpus
# ---------------------------------------------------------------------------

def plot_latency(data: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(data["n_gpus"], data["total_ms"] / 1000,
            "o-", color="#4472C4", label="Measured")

    # Ideal linear scaling from 1-GPU baseline
    if 1 in data["n_gpus"].values:
        t1 = data.loc[data["n_gpus"] == 1, "total_ms"].values[0]
        n_range = np.array(sorted(data["n_gpus"].unique()))
        ax.plot(n_range, t1 / n_range / 1000,
                "--", color="gray", alpha=0.6, label="Ideal linear")

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("BERT-base Inference Time (s)")
    ax.set_title("BERT-base FHE Inference Latency vs. GPU Count")
    ax.set_xticks(sorted(data["n_gpus"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "bert_scaling_latency.pdf")
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Speedup vs n_gpus
# ---------------------------------------------------------------------------

def plot_speedup(data: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(data["n_gpus"], data["speedup"],
            "o-", color="#ED7D31", label="Measured speedup")

    # Ideal
    n_range = np.array(sorted(data["n_gpus"].unique()))
    ax.plot(n_range, n_range.astype(float),
            "--", color="gray", alpha=0.6, label="Ideal (linear)")

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup vs. 1 GPU")
    ax.set_title("Multi-GPU Speedup (BERT-base FHE Inference)")
    ax.set_xticks(sorted(data["n_gpus"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate efficiency
    for _, row in data.iterrows():
        ax.annotate(f"{row['efficiency']*100:.0f}%",
                    xy=(row["n_gpus"], row["speedup"]),
                    xytext=(3, 5), textcoords="offset points",
                    fontsize=8, color="#4472C4")

    path = os.path.join(outdir, "bert_scaling_speedup.pdf")
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Stacked breakdown bar chart
# ---------------------------------------------------------------------------

def plot_breakdown(data: pd.DataFrame, outdir: str):
    components = ["matmul_ms", "gelu_ms", "softmax_ms",
                  "layernorm_ms", "keyswitch_ms", "comm_ms"]
    labels     = ["MatMul", "GELU", "SoftMax", "LayerNorm", "KeySwitch", "NCCL Comm"]
    colors_list = [COLORS["matmul"], COLORS["gelu"], COLORS["softmax"],
                   COLORS["layernorm"], COLORS["keyswitch"], COLORS["comm"]]

    # Compute "other" category
    accounted = data[components].sum(axis=1)
    data = data.copy()
    data["other_ms"] = (data["total_ms"] - accounted).clip(lower=0)
    components.append("other_ms")
    labels.append("Other")
    colors_list.append(COLORS["other"])

    gpu_counts = sorted(data["n_gpus"].unique())
    x = np.arange(len(gpu_counts))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(gpu_counts))

    for comp, label, color in zip(components, labels, colors_list):
        vals = [data.loc[data["n_gpus"] == g, comp].values[0] / 1000
                for g in gpu_counts]
        ax.bar(x, vals, width, bottom=bottoms, label=label, color=color)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} GPU{'s' if g > 1 else ''}" for g in gpu_counts])
    ax.set_ylabel("Time (s)")
    ax.set_title("BERT-base FHE: Time Breakdown by Component")
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.0))
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(outdir, "bert_breakdown_stacked.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Communication overhead fraction
# ---------------------------------------------------------------------------

def plot_comm_overhead(data: pd.DataFrame, outdir: str):
    if "comm_ms" not in data.columns or data["comm_ms"].sum() == 0:
        print("Skipping comm overhead plot (no comm_ms data yet)")
        return

    comm_frac = data["comm_ms"] / data["total_ms"] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(data["n_gpus"].astype(str), comm_frac, color="#7030A0", alpha=0.8)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("NCCL Communication Overhead (%)")
    ax.set_title("Communication Overhead vs. GPU Count")
    ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(outdir, "comm_overhead.pdf")
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot BERT FHE scaling results")
    parser.add_argument("--input",  default="experiments/results/",
                        help="Directory containing bert_Ngpu.csv files")
    parser.add_argument("--output", default="experiments/plots/",
                        help="Directory to write PDF plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load_results(args.input)

    print(f"Loaded {len(data)} GPU configurations: {sorted(data['n_gpus'].tolist())}")
    print(data[["n_gpus", "total_ms", "speedup", "efficiency"]].to_string(index=False))
    print()

    plot_latency(data, args.output)
    plot_speedup(data, args.output)
    plot_breakdown(data, args.output)
    plot_comm_overhead(data, args.output)

    print(f"\nAll plots saved to {args.output}")


if __name__ == "__main__":
    main()
