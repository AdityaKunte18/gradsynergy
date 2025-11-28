import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def save_cosine_matrix_csv_png(M: np.ndarray, names, csv_path: str, png_path: str):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow([""] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [float(M[i, j]) for j in range(len(names))])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Cosine Similarity", rotation=270, labelpad=12)
    plt.tight_layout(); plt.savefig(png_path, dpi=200); plt.close(fig)

def append_layer_log_rows(csv_writer, step_idx: int, layer_cos):
    for layer_idx, d in sorted(layer_cos.items(), key=lambda kv: kv[0]):
        csv_writer.writerow([step_idx, layer_idx, d.get("C-F", 0.0), d.get("C-L", 0.0), d.get("F-L", 0.0)])

def plot_layer_curves_from_csv(csv_path: str, out_prefix: str):
    df = pd.read_csv(csv_path)
    layers = sorted(df["layer"].unique().tolist())
    for layer in layers:
        sub = df[df["layer"] == layer].sort_values("step")
        if sub.empty: 
            continue
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(sub["step"], sub["C_F"], label="C-F")
        ax.plot(sub["step"], sub["C_L"], label="C-L")
        ax.plot(sub["step"], sub["F_L"], label="F-L")
        ax.set_ylim(-1.05, 1.05); ax.set_xlabel("Update step"); ax.set_ylabel("Cosine")
        ax.set_title(f"Layer {layer}: Objective Cosines over Steps"); ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f"{out_prefix}{layer}.png", dpi=200); plt.close(fig)
