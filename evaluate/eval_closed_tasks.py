"""Evaluate closed GeoBench tasks for a base or fine-tuned model.

Inputs:
  Model path and GeoBench dataset.
Outputs:
  results/{model_name}/{strategy}/closed_results.json

Usage:
  python evaluate/eval_closed_tasks.py --model_path meta-llama/Meta-Llama-3.1-8B-Instruct --model_name Llama-3.1-8B --strategy baseline
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from common import (
    build_prompt,
    ensure_output_dir,
    extract_closed_prediction,
    generate_answer,
    load_local_or_hf_geobench,
    load_model_and_tokenizer,
    save_json,
    select_task_subset,
)


def _normalize_label(row: dict) -> str:
    for key in ("label", "answer", "gold", "target"):
        if key in row:
            return str(row[key]).strip().upper()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Model path or HF model ID.")
    parser.add_argument("--model_name", required=True, help="Short model name for results.")
    parser.add_argument("--strategy", required=True, help="Strategy identifier.")
    parser.add_argument("--max_samples", type=int, default=400, help="Maximum evaluation samples.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    geobench = load_local_or_hf_geobench()

    records = []
    metrics = {}
    for task_name in ("true_false", "multiple_choice"):
        subset = select_task_subset(geobench, {task_name})
        if len(subset) > args.max_samples // 2:
            subset = subset.select(range(args.max_samples // 2))
        correct = 0
        for row in subset:
            question = row.get("question") or row.get("prompt") or row.get("instruction") or ""
            context = row.get("context") or ""
            answer = generate_answer(model, tokenizer, build_prompt(question, context), max_new_tokens=16)
            prediction = extract_closed_prediction(answer)
            label = _normalize_label(row)
            is_correct = prediction == label
            correct += int(is_correct)
            records.append(
                {
                    "task_type": task_name,
                    "question": question,
                    "prediction": prediction,
                    "label": label,
                    "correct": is_correct,
                }
            )
        metrics[task_name] = correct / max(1, len(subset))

    result = {
        "model": args.model_name,
        "strategy": args.strategy,
        "true_false_accuracy": metrics.get("true_false", 0.0),
        "multiple_choice_accuracy": metrics.get("multiple_choice", 0.0),
        "overall_accuracy": sum(metrics.values()) / max(1, len(metrics)),
        "n_samples": len(records),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": records,
    }
    output_dir = ensure_output_dir(args.model_name, args.strategy)
    save_json(result, output_dir / "closed_results.json")


if __name__ == "__main__":
    main()
