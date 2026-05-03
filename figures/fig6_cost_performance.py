"""Generate a cost-performance bubble chart.

Inputs:
  CSV with model size and performance metrics.
Outputs:
  figures/fig6_cost_performance.pdf and .png

Usage:
  python figures/fig6_cost_performance.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import apply_publication_style, get_color_maps, load_config, load_results, save_figure, summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/summary.csv", help="Input CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style()

    config = load_config()
    _, strategy_colors = get_color_maps(config)
    df = summarize_runs(load_results(args.data))
    df["cost_axis"] = df["params_b"]
    best = df.sort_values(["closed_overall", "physgeo_score"], ascending=False).head(3)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for _, row in df.iterrows():
        ax.scatter(
            row["cost_axis"],
            row["closed_overall"],
            s=180 + row["physgeo_score"] * 240,
            color=strategy_colors.get(row["strategy"], "#6B7280"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.text(row["cost_axis"] + 0.2, row["closed_overall"] + 0.003, row["model"], fontsize=8)

    for _, row in best.iterrows():
        ax.annotate(
            "Best trade-off",
            xy=(row["cost_axis"], row["closed_overall"]),
            xytext=(row["cost_axis"] + 2, row["closed_overall"] - 0.03),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=9,
        )

    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel("GeoBench Closed Accuracy")
    ax.set_title("Cost-Performance Trade-off")

    save_figure(fig, Path("figures") / "fig6_cost_performance")


if __name__ == "__main__":
    main()
