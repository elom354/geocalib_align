"""Generate grouped bar charts for strategy comparison with error bars.

Inputs:
  CSV with repeated runs across seeds.
Outputs:
  figures/fig4_bar_strategies.pdf and .png

Usage:
  python figures/fig4_bar_strategies.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import apply_publication_style, get_color_maps, get_strategy_labels, load_config, load_results, save_figure, summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/summary.csv", help="Input CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style()

    config = load_config()
    model_colors, _ = get_color_maps(config)
    strategy_labels = get_strategy_labels(config)
    df = summarize_runs(load_results(args.data))
    df = df[~df["is_proprietary"]].copy()

    strategy_order = ["lora_std", "lora_sel", "lora_replay", "mix_cpt"]
    models = list(df["model"].drop_duplicates())
    x = np.arange(len(strategy_order))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
    metrics = [
        ("closed_overall", "GeoBench Closed Accuracy"),
        ("prompt_alignment", "Prompt Alignment"),
    ]

    for axis, (metric, title) in zip(axes, metrics):
        for idx, model in enumerate(models):
            model_df = df[df["model"] == model].set_index("strategy").reindex(strategy_order)
            offset = (idx - 1) * width
            axis.bar(
                x + offset,
                model_df[metric],
                width=width,
                yerr=model_df[f"{metric}_std"],
                capsize=3,
                color=model_colors.get(model, "#6B7280"),
                alpha=0.85,
                label=model,
            )
            baseline = df[(df["model"] == model) & (df["strategy"] == "baseline")][metric].mean()
            axis.axhline(baseline, linestyle="--", linewidth=1, color=model_colors.get(model, "#6B7280"), alpha=0.5)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([strategy_labels[item] for item in strategy_order], rotation=15, ha="right")
        axis.set_ylabel(title)

    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Strategy Comparison Across Open-Source Models")
    fig.tight_layout()

    save_figure(fig, Path("figures") / "fig4_bar_strategies")


if __name__ == "__main__":
    main()
