"""Generate a radar chart for the best six model configurations.

Inputs:
  CSV with model metrics.
Outputs:
  figures/fig2_radar_chart.pdf and .png

Usage:
  python figures/fig2_radar_chart.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import apply_publication_style, get_color_maps, load_config, load_results, save_figure, summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/summary.csv", help="Input CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style()

    config = load_config()
    model_colors, _ = get_color_maps(config)
    df = summarize_runs(load_results(args.data))
    top_df = df.sort_values("closed_overall", ascending=False).head(6).copy()
    top_df["earthse_accuracy"] = top_df["correctness_norm"]

    categories = [
        "closed_overall",
        "prompt_alignment_norm",
        "answer_relevance_norm",
        "bert_f1",
        "physgeo_score",
        "earthse_accuracy",
    ]
    labels = [
        "GeoBench Closed",
        "Prompt Alignment",
        "Answer Relevance",
        "BERT F1",
        "PhysGeo",
        "EarthSE / Correctness",
    ]
    angles = np.linspace(0, 2 * pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw={"projection": "polar"})
    for _, row in top_df.iterrows():
        values = [row[col] for col in categories]
        values += values[:1]
        label = f'{row["model"]} ({row["strategy"]})'
        color = model_colors.get(row["model"], "#374151")
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Dimensional Performance Comparison", pad=24)
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), frameon=False)

    save_figure(fig, Path("figures") / "fig2_radar_chart")


if __name__ == "__main__":
    main()
