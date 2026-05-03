"""Aggregate evaluation JSON files into summary CSV and LaTeX outputs.

Inputs:
  JSON files produced by evaluation scripts under results/.
Outputs:
  results/summary.csv and results/summary_latex.tex

Usage:
  python evaluate/aggregate_results.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_meta() -> dict:
    config = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    meta = {}
    for section in ("open_source_models", "proprietary_models"):
        for item in config.get(section, {}).values():
            meta[item["short_name"]] = {"params_b": item["params_b"], "is_proprietary": section == "proprietary_models"}
    return meta


def _to_latex_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\hline"]
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        formatted = []
        for item in row.tolist():
            formatted.append(f"{item:.3f}" if isinstance(item, float) else str(item))
        lines.append(" & ".join(formatted) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    meta = _load_model_meta()
    rows = []
    for strategy_dir in ROOT.glob("results/*/*"):
        if not strategy_dir.is_dir():
            continue
        closed_path = strategy_dir / "closed_results.json"
        open_path = strategy_dir / "open_results.json"
        physgeo_path = strategy_dir / "physgeo_results.json"
        if not closed_path.exists() or not open_path.exists() or not physgeo_path.exists():
            continue
        closed = _load_json(closed_path)
        open_result = _load_json(open_path)
        physgeo = _load_json(physgeo_path)
        model_name = closed["model"]
        info = meta.get(model_name, {"params_b": None, "is_proprietary": False})
        rows.append(
            {
                "model": model_name,
                "strategy": closed["strategy"],
                "true_false_acc": closed["true_false_accuracy"],
                "mc_acc": closed["multiple_choice_accuracy"],
                "closed_overall": closed["overall_accuracy"],
                "prompt_alignment": open_result["prompt_alignment"],
                "correctness": open_result["correctness"],
                "answer_relevance": open_result["answer_relevance"],
                "bert_f1": open_result["bert_f1"],
                "physgeo_score": physgeo["physgeo_score"],
                "physgeo_unit": physgeo["unit_consistency"],
                "physgeo_spatiotemporal": physgeo["spatiotemporal_coherence"],
                "physgeo_conservation": physgeo["conservation_laws"],
                "physgeo_stratigraphic": physgeo["stratigraphic_plausibility"],
                "physgeo_causal": physgeo["causal_correctness"],
                "physgeo_hallucination": physgeo["no_physical_hallucinations"],
                "params_b": info["params_b"],
                "is_proprietary": info["is_proprietary"],
            }
        )

    if not rows:
        raise FileNotFoundError("No complete result triplets found under results/.")

    summary = pd.DataFrame(rows).sort_values("closed_overall", ascending=False)
    summary_path = ROOT / "results" / "summary.csv"
    summary.to_csv(summary_path, index=False)
    latex_path = ROOT / "results" / "summary_latex.tex"
    latex_path.write_text(_to_latex_table(summary), encoding="utf-8")
    logging.info("Saved summary to %s", summary_path)
    logging.info("Saved LaTeX table to %s", latex_path)
    logging.info("Top 5 models by closed_overall:\n%s", summary.head(5)[["model", "strategy", "closed_overall"]].to_string(index=False))


if __name__ == "__main__":
    main()
