#!/usr/bin/env python3
"""
Plot basic training metrics from runs/run_*/metrics.jsonl.

Usage:
    python plot_metrics.py --run_dir runs/run_20251201_195426

Outputs PNGs in the same run directory:
    conflict.png, weights.png, rewards.png, cosines.png
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


OBJ_NAMES = ("correctness", "format", "length")


def load_metrics(path: str) -> pd.DataFrame:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            log = json.loads(line)
            rec: Dict = {
                "step": log.get("step"),
                "epoch": log.get("epoch"),
                "conflict_frac": log.get("conflict_frac"),
                "global_conflict_frac": log.get("global_conflict_frac"),
            }
            # Weights
            weights = log.get("weights", {})
            for obj in OBJ_NAMES:
                rec[f"weight_{obj}"] = weights.get(obj)
            # Rewards (one dict per objective in rewards_batch)
            rdict: Dict = {}
            for d in log.get("rewards_batch", []):
                rdict.update(d)
            for obj in OBJ_NAMES:
                rec[f"reward_{obj}"] = rdict.get(obj)
            # Cosine diagonal per objective
            keys = log.get("keys", [])
            diag = log.get("cosine_diag", [])
            for k, v in zip(keys, diag):
                rec[f"cos_{k}"] = v
            # Last-layer and global off-diagonal cosines
            last_mat = log.get("last_cosine_matrix")
            if last_mat and keys:
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        rec[f"last_cos_{keys[i]}-{keys[j]}"] = last_mat[i][j]
            g_keys = log.get("global_keys", [])
            g_mat = log.get("global_cosine_matrix")
            if g_mat and g_keys:
                for i in range(len(g_keys)):
                    for j in range(i + 1, len(g_keys)):
                        rec[f"global_cos_{g_keys[i]}-{g_keys[j]}"] = g_mat[i][j]
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    df.sort_values("step", inplace=True)
    return df


def _plot_lines(df: pd.DataFrame, cols: List[str], ylabel: str, title: str, path: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for c in cols:
        if c not in df:
            continue
        ax.plot(df["step"], df[c], label=c.replace("_", " "))
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="run directory containing metrics.jsonl")
    ap.add_argument(
        "--metrics_file",
        default=None,
        help="path to metrics.jsonl (default: run_dir/metrics.jsonl)",
    )
    args = ap.parse_args()

    metrics_path = args.metrics_file or os.path.join(args.run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    df = load_metrics(metrics_path)
    if df.empty:
        raise RuntimeError("No records loaded from metrics.jsonl")

    out_conflict = os.path.join(args.run_dir, "conflict.png")
    out_weights = os.path.join(args.run_dir, "weights.png")
    out_rewards = os.path.join(args.run_dir, "rewards.png")
    out_cos = os.path.join(args.run_dir, "cosines.png")
    out_last = os.path.join(args.run_dir, "last_layer_cosines.png")
    out_global = os.path.join(args.run_dir, "global_cosines.png")

    _plot_lines(df, ["conflict_frac"], "conflict fraction", "Conflict over steps", out_conflict)
    _plot_lines(
        df, [f"weight_{o}" for o in OBJ_NAMES], "weight", "Objective weights", out_weights
    )
    _plot_lines(
        df, [f"reward_{o}" for o in OBJ_NAMES], "reward", "Batch rewards", out_rewards
    )
    _plot_lines(
        df, [c for c in df.columns if c.startswith("cos_")], "cosine", "Cosine diag", out_cos
    )
    _plot_lines(
        df, [c for c in df.columns if c.startswith("last_cos_")], "cosine", "Last-layer cosines", out_last
    )
    _plot_lines(
        df,
        [c for c in df.columns if c.startswith("global_cos_")],
        "cosine",
        "Global cosines",
        out_global,
    )

    # Layer-wise cosines (from CSV)
    layer_csv = os.path.join(args.run_dir, "layer_cos.csv")
    if os.path.exists(layer_csv):
        ldf = pd.read_csv(layer_csv)
        pairs = sorted(ldf["pair"].unique().tolist())
        layers = sorted(ldf["layer"].unique().tolist())
        for layer in layers:
            sub = ldf[ldf["layer"] == layer].sort_values("step")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            for pair in pairs:
                ss = sub[sub["pair"] == pair]
                if ss.empty:
                    continue
                ax.plot(ss["step"], ss["cosine"], label=pair)
            ax.set_xlabel("step")
            ax.set_ylabel("cosine")
            ax.set_title(f"Layer {layer} cosines")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.run_dir, f"layer_cosines_layer{layer}.png"), dpi=200)
            plt.close(fig)
    print(f"Saved plots to {args.run_dir}")


if __name__ == "__main__":
    main()
