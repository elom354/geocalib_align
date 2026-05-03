"""Compute knowledge/alignment trade-offs from the aggregated summary table.

Inputs:
  results/summary.csv
Outputs:
  results/tradeoff_analysis.csv

Usage:
  python analysis/compute_tradeoff.py --summary results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/summary.csv", help="Summary CSV path.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = pd.read_csv(args.summary)
    df["baseline_alignment"] = None
    rows = []
    for model, model_df in df.groupby("model"):
        baseline = model_df[model_df["strategy"] == "baseline"]
        if baseline.empty:
            logging.warning("Skipping %s because no baseline row exists.", model)
            continue
        baseline_closed = float(baseline["closed_overall"].mean())
        baseline_alignment = float(((baseline["prompt_alignment"] + baseline["answer_relevance"]) / 2).mean())
        for _, row in model_df.iterrows():
            alignment_score = (row["prompt_alignment"] + row["answer_relevance"]) / 2
            knowledge_gain = row["closed_overall"] - baseline_closed
            alignment_delta = alignment_score - baseline_alignment
            tradeoff_ratio = knowledge_gain / max(0.001, -alignment_delta)
            rows.append(
                {
                    "model": model,
                    "strategy": row["strategy"],
                    "knowledge_gain": knowledge_gain,
                    "alignment_delta": alignment_delta,
                    "tradeoff_ratio": tradeoff_ratio,
                }
            )

    result = pd.DataFrame(rows).sort_values("tradeoff_ratio", ascending=False)
    output_path = Path("results") / "tradeoff_analysis.csv"
    result.to_csv(output_path, index=False)
    logging.info("Saved trade-off analysis to %s", output_path)
    logging.info("Trade-off ranking:\n%s", result.to_string(index=False))


if __name__ == "__main__":
    main()
