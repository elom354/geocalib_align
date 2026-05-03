"""Generate the GeoCalib-Align trade-off scatter plot.

Inputs:
  CSV with aggregated or repeated experiment rows.
Outputs:
  figures/fig1_tradeoff_scatter.pdf and .png

Usage:
  python figures/fig1_tradeoff_scatter.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from plot_utils import (
    apply_publication_style,
    get_color_maps,
    get_strategy_labels,
    load_config,
    load_results,
    save_figure,
    summarize_runs,
)

LOGGER = logging.getLogger(__name__)


def _pareto_frontier(data):
    frontier = []
    for _, row in data.sort_values(["closed_overall", "prompt_alignment_norm"]).iterrows():
        if not frontier or row["prompt_alignment_norm"] >= frontier[-1]["prompt_alignment_norm"]:
            frontier.append(row)
    return frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/summary.csv", help="Input CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style()
    sns.set_style("white")

    config = load_config()
    _, strategy_colors = get_color_maps(config)
    strategy_labels = get_strategy_labels(config)
    df = summarize_runs(load_results(args.data))

    marker_map = {
        "Llama": "o",
        "Gemma": "s",
        "Phi": "D",
        "Mistral": "^",
        "Qwen": "P",
    }

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for _, row in df.iterrows():
        family = row["model"].split("-")[0]
        marker = marker_map.get(family, "o")
        ax.scatter(
            row["closed_overall"],
            row["prompt_alignment_norm"],
            s=120 + row["physgeo_score"] * 220,
            marker=marker,
            color=strategy_colors.get(row["strategy"], "#6B7280"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )

    frontier = _pareto_frontier(df.sort_values("closed_overall"))
    ax.plot(
        [item["closed_overall"] for item in frontier],
        [item["prompt_alignment_norm"] for item in frontier],
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="Pareto frontier",
    )

    ax.axvline(0.7, color="#D1D5DB", linestyle="--", linewidth=1)
    ax.axhline(0.85, color="#D1D5DB", linestyle="--", linewidth=1)
    ax.text(0.545, 0.93, "Low knowledge,\nHigh alignment", fontsize=10, ha="left")
    ax.text(0.705, 0.93, "Optimal zone", fontsize=10, ha="left")
    ax.text(0.705, 0.76, "High knowledge,\nLow alignment", fontsize=10, ha="left")
    ax.text(0.545, 0.76, "Under-tuned zone", fontsize=10, ha="left")

    ax.set_xlim(0.54, 0.82)
    ax.set_ylim(0.76, 0.96)
    ax.set_xlabel("GeoBench Closed Accuracy")
    ax.set_ylabel("Prompt Alignment (normalized)")
    ax.set_title("Knowledge Gain vs Alignment Preservation")

    label_candidates = pd.concat(
        [
            df.sort_values("closed_overall", ascending=False).head(3),
            df.sort_values("prompt_alignment_norm", ascending=False).head(2),
            df.sort_values("physgeo_score", ascending=False).head(2),
        ]
    ).drop_duplicates(subset=["model", "strategy"])
    label_offsets = [
        (8, 4),
        (8, -10),
        (10, 8),
        (-42, 6),
        (-46, -10),
        (10, -14),
        (10, 10),
    ]
    for idx, (_, row) in enumerate(label_candidates.iterrows()):
        dx, dy = label_offsets[idx % len(label_offsets)]
        ax.annotate(
            f'{row["model"]} ({strategy_labels.get(row["strategy"], row["strategy"])})',
            xy=(row["closed_overall"], row["prompt_alignment_norm"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "linewidth": 0.6},
        )

    handles = []
    for strategy_id, label in strategy_labels.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=strategy_colors.get(strategy_id, "#6B7280"),
                markeredgecolor="black",
                label=label,
            )
        )
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.5, label="Pareto frontier"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    save_figure(fig, Path("figures") / "fig1_tradeoff_scatter")


if __name__ == "__main__":
    main()
