"""Generate a stacked PhysGeo breakdown chart.

Inputs:
  CSV with PhysGeo component scores.
Outputs:
  figures/fig5_physgeo_breakdown.pdf and .png

Usage:
  python figures/fig5_physgeo_breakdown.py --data results/summary.csv
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
    criteria = [
        "physgeo_unit",
        "physgeo_spatiotemporal",
        "physgeo_conservation",
        "physgeo_stratigraphic",
        "physgeo_causal",
        "physgeo_hallucination",
    ]
    display_names = [
        "Unit",
        "Spatiotemporal",
        "Conservation",
        "Stratigraphic",
        "Causal",
        "No Hallucination",
    ]
    scaled = df[criteria] / 6.0
    df = df.assign(label=df["model"] + "\n" + df["strategy"], total=df["physgeo_score"])
    order = df.sort_values("total", ascending=False).index
    palette = sns.color_palette("YlGnBu", n_colors=6)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    bottom = None
    for idx, (column, label) in enumerate(zip(criteria, display_names)):
        values = scaled.loc[order, column]
        ax.barh(
            df.loc[order, "label"],
            values,
            bottom=bottom,
            color=palette[idx],
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = values if bottom is None else bottom + values

    ax.set_xlabel("PhysGeo Score")
    ax.set_title("PhysGeo Breakdown by Criterion")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    save_figure(fig, Path("figures") / "fig5_physgeo_breakdown")


if __name__ == "__main__":
    main()
