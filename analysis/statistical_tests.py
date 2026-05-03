"""Run Wilcoxon signed-rank tests across fine-tuning strategies.

Inputs:
  Per-sample evaluation JSON files in results/.
Outputs:
  results/statistical_tests.csv

Usage:
  python analysis/statistical_tests.py
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def _load_scores() -> dict[str, dict[str, list[float]]]:
    by_model: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for path in ROOT.glob("results/*/*/open_results.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        strategy = payload["strategy"]
        if strategy == "baseline":
            continue
        scores = [(sample["prompt_alignment"] + sample["answer_relevance"]) / 2 for sample in payload.get("samples", [])]
        if scores:
            by_model[payload["model"]][strategy] = scores
    return by_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    by_model = _load_scores()
    rows = []
    for model, strategies in by_model.items():
        pairs = list(combinations(sorted(strategies), 2))
        m = max(1, len(pairs))
        for left, right in pairs:
            left_scores = strategies[left]
            right_scores = strategies[right]
            n = min(len(left_scores), len(right_scores))
            statistic, pvalue = wilcoxon(left_scores[:n], right_scores[:n])
            corrected = min(1.0, pvalue * m)
            rows.append(
                {
                    "model": model,
                    "strategy_a": left,
                    "strategy_b": right,
                    "wilcoxon_statistic": statistic,
                    "p_value": pvalue,
                    "bonferroni_p_value": corrected,
                    "significant": corrected < 0.05,
                }
            )

    output = pd.DataFrame(rows)
    output_path = ROOT / "results" / "statistical_tests.csv"
    output.to_csv(output_path, index=False)
    logging.info("Saved p-value matrix to %s", output_path)
    if not output.empty:
        logging.info("Significance table:\n%s", output.to_string(index=False))


if __name__ == "__main__":
    main()
