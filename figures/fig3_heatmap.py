"""Generate a heatmap of model-strategy metric performance.

Inputs:
  CSV with result metrics.
Outputs:
  figures/fig3_heatmap.pdf and .png

Usage:
  python figures/fig3_heatmap.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import apply_publication_style, load_results, save_figure, summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/summary.csv", help="Input CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style()

    df = summarize_runs(load_results(args.data))
    df["row_name"] = df["model"] + " | " + df["strategy"]
    df = df.sort_values(["is_proprietary", "closed_overall"], ascending=[True, False])
    matrix = df.set_index("row_name")[
        ["closed_overall", "prompt_alignment_norm", "answer_relevance_norm", "bert_f1", "physgeo_score"]
    ]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, cbar=True, ax=ax)
    open_source_count = int((~df["is_proprietary"]).sum())
    if 0 < open_source_count < len(df):
        ax.hlines(open_source_count, *ax.get_xlim(), colors="black", linewidth=1.5)
    ax.set_title("Model-Metric Performance Matrix")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model | Strategy")

    save_figure(fig, Path("figures") / "fig3_heatmap")


if __name__ == "__main__":
    main()
