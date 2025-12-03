#!/usr/bin/env python3
"""
Plot heatmaps of the final global and last-layer cosine similarity matrices.

Usage:
    python heatmap_cosines.py --run_dir runs/run_<timestamp>
"""

import argparse
import json
import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_final_mats(path: str) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    last_keys: List[str] = []
    last_mat: np.ndarray = np.zeros((0, 0))
    glob_keys: List[str] = []
    glob_mat: np.ndarray = np.zeros((0, 0))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            log = json.loads(line)
            if "keys" in log and "last_cosine_matrix" in log:
                last_keys = log.get("keys", []) or last_keys
                last_mat = np.array(log.get("last_cosine_matrix", last_mat))
            if "global_keys" in log and "global_cosine_matrix" in log:
                glob_keys = log.get("global_keys", []) or glob_keys
                glob_mat = np.array(log.get("global_cosine_matrix", glob_mat))
    return last_keys, last_mat, glob_keys, glob_mat


def plot_heatmap(mat: np.ndarray, labels: List[str], title: str, path: str):
    if mat.size == 0 or len(labels) == 0:
        print(f"[skip] no data for {title}")
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("cosine", rotation=270, labelpad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory containing metrics.jsonl")
    ap.add_argument(
        "--metrics_file",
        default=None,
        help="Path to metrics.jsonl (default: run_dir/metrics.jsonl)",
    )
    args = ap.parse_args()

    metrics_path = args.metrics_file or os.path.join(args.run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    last_keys, last_mat, glob_keys, glob_mat = load_final_mats(metrics_path)

    plot_heatmap(last_mat, last_keys, "Last-layer cosine", os.path.join(args.run_dir, "last_cosine_heatmap.png"))
    plot_heatmap(glob_mat, glob_keys, "Global cosine", os.path.join(args.run_dir, "global_cosine_heatmap.png"))


if __name__ == "__main__":
    main()
