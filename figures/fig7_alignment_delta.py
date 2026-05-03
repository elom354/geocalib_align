"""Generate alignment degradation delta plot.

Inputs:
  CSV with repeated runs across seeds.
Outputs:
  figures/fig7_alignment_delta.pdf and .png

Usage:
  python figures/fig7_alignment_delta.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import apply_publication_style, get_color_maps, load_config, load_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/summary.csv", help="Input CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style()

    config = load_config()
    model_colors, _ = get_color_maps(config)
    df = load_results(args.data)
    df = df[~df["is_proprietary"]].copy()
    df["alignment_score"] = (df["prompt_alignment"] + df["answer_relevance"]) / 2.0
    strategy_order = ["lora_std", "lora_sel", "lora_replay", "mix_cpt"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(strategy_order))
    for model in df["model"].drop_duplicates():
        baseline = df[(df["model"] == model) & (df["strategy"] == "baseline")]["alignment_score"].mean()
        means = []
        stds = []
        for strategy in strategy_order:
            values = df[(df["model"] == model) & (df["strategy"] == strategy)]["alignment_score"] - baseline
            means.append(values.mean())
            stds.append(values.std(ddof=1))
        color = model_colors.get(model, "#374151")
        means_arr = np.array(means)
        stds_arr = np.nan_to_num(np.array(stds), nan=0.0)
        ax.plot(x, means_arr, marker="o", linewidth=2, color=color, label=model)
        ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15)
        best_idx = int(np.argmax(means_arr))
        ax.annotate(
            "Least degradation",
            xy=(best_idx, means_arr[best_idx]),
            xytext=(best_idx, means_arr[best_idx] + 0.08),
            fontsize=8,
            ha="center",
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )

    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Standard LoRA", "Selective LoRA", "LoRA + Replay", "Mix-CPT"], rotation=15, ha="right")
    ax.set_ylabel("Alignment Delta")
    ax.set_title("Alignment Degradation Relative to Baseline")
    ax.legend(frameon=False)

    from plot_utils import save_figure

    save_figure(fig, Path("figures") / "fig7_alignment_delta")


if __name__ == "__main__":
    main()
