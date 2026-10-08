#!/usr/bin/env python3
"""Plot Hybrid Adaptive vs cuSPARSE GFLOPS on the irregular suite.

Usage (from repo root, with matplotlib installed):
  python3 scripts/plot_irregular_results.py

Reads:  results/irregular_adaptive_vs_cusparse.csv
Writes: results/figures/adaptive_vs_cusparse_gflops.png
        results/figures/adaptive_over_cusparse_ratio.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "irregular_adaptive_vs_cusparse.csv"
OUT_DIR = ROOT / "results" / "figures"


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                (
                    r["matrix"],
                    float(r["adaptive_gflops"]),
                    float(r["cusparse_gflops"]),
                )
            )
    return rows


def plot_gflops(rows, out: Path) -> None:
    rows = sorted(rows, key=lambda r: r[2])  # by cuSPARSE
    labels = [r[0] for r in rows]
    adap = np.array([r[1] for r in rows])
    cusp = np.array([r[2] for r in rows])
    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.barh(y + h / 2, cusp, height=h, label="cuSPARSE", color="#4C78A8")
    ax.barh(y - h / 2, adap, height=h, label="Hybrid Adaptive", color="#F58518")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Performance (GFLOPS)")
    ax.set_title("Irregular SuiteSparse SpMV on NVIDIA A30")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, max(cusp.max(), adap.max()) * 1.12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_ratio(rows, out: Path) -> None:
    ratios = sorted(((m, a / c) for m, a, c in rows), key=lambda r: r[1])
    labels = [r[0] for r in ratios]
    vals = np.array([r[1] for r in ratios])
    colors = ["#54A24B" if v >= 1 else "#E45756" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.barh(np.arange(len(labels)), vals, color=colors)
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--", label="parity")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Adaptive / cuSPARSE (GFLOPS ratio)")
    ax.set_title("Hybrid Adaptive relative to cuSPARSE (A30)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.is_file():
        raise SystemExit(f"missing {CSV_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(CSV_PATH)
    gflops_png = OUT_DIR / "adaptive_vs_cusparse_gflops.png"
    ratio_png = OUT_DIR / "adaptive_over_cusparse_ratio.png"
    plot_gflops(rows, gflops_png)
    plot_ratio(rows, ratio_png)
    wins = sum(1 for _, a, c in rows if a > c)
    print(f"wrote {gflops_png}")
    print(f"wrote {ratio_png}")
    print(f"Adaptive beats cuSPARSE on {wins}/{len(rows)} matrices")


if __name__ == "__main__":
    main()
