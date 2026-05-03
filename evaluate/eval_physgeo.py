"""Compute PhysGeo Score using hybrid rule-based and local judge-based evaluation.

Inputs:
  Model path and data/physgeo_eval_template.json.
Outputs:
  results/{model_name}/{strategy}/physgeo_results.json

Usage:
  python evaluate/eval_physgeo.py --model_path meta-llama/Meta-Llama-3.1-8B-Instruct --model_name Llama-3.1-8B --strategy baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from common import build_prompt, ensure_output_dir, extract_json_object, generate_answer, load_model_and_tokenizer, save_json

ROOT = Path(__file__).resolve().parents[1]
JUDGE_CRITERIA = [
    "spatiotemporal_coherence",
    "stratigraphic_plausibility",
    "causal_correctness",
    "no_physical_hallucinations",
]


def _unit_consistency(answer: str) -> float:
    numbers = re.findall(r"\d+(?:\.\d+)?", answer)
    units = re.findall(r"\b(km|m|cm|mm|kg|g|Pa|kPa|MPa|Ma|ka|yr|years|C|K|%)\b", answer)
    return 1.0 if not numbers or units else 0.5 if numbers and not units else 1.0


def _conservation_laws(answer: str) -> float:
    lower = answer.lower()
    red_flags = ["energy created", "mass created", "water appears from nowhere", "violates conservation"]
    return 0.0 if any(flag in lower for flag in red_flags) else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Model path or HF model ID.")
    parser.add_argument("--model_name", required=True, help="Short model name for results.")
    parser.add_argument("--strategy", required=True, help="Strategy identifier.")
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct", help="Local open-source judge model.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    template = json.loads((ROOT / "data" / "physgeo_eval_template.json").read_text(encoding="utf-8"))
    if not template.get("examples"):
        raise ValueError(
            "PhysGeo template contains no annotated evaluation items. "
            "Populate data/physgeo_eval_template.json with expert-reviewed examples before running evaluation."
        )
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    judge_model, judge_tokenizer = load_model_and_tokenizer(args.judge_model)

    samples = []
    for item in template["examples"]:
        answer = generate_answer(model, tokenizer, build_prompt(item["question"]), max_new_tokens=200)
        scores = {
            "unit_consistency": _unit_consistency(answer),
            "conservation_laws": _conservation_laws(answer),
        }
        for criterion in JUDGE_CRITERIA:
            prompt = (
                "You are evaluating geoscience physical plausibility.\n"
                f"Question: {item['question']}\nReference answer: {item['correct_answer']}\n"
                f"Model answer: {answer}\nCriterion: {criterion}\n"
                "Reply only as JSON: {\"score\": 0 or 1, \"rationale\": \"...\"}"
            )
            judged = extract_json_object(generate_answer(judge_model, judge_tokenizer, prompt, max_new_tokens=96))
            scores[criterion] = float(judged["score"])
        samples.append({"id": item["id"], "domain": item["domain"], "answer": answer, **scores})

    criteria = [
        "unit_consistency",
        "spatiotemporal_coherence",
        "conservation_laws",
        "stratigraphic_plausibility",
        "causal_correctness",
        "no_physical_hallucinations",
    ]
    averages = {criterion: sum(sample[criterion] for sample in samples) / len(samples) for criterion in criteria}
    result = {
        "model": args.model_name,
        "strategy": args.strategy,
        "physgeo_score": sum(averages.values()) / len(averages),
        "judge_model": args.judge_model,
        **averages,
        "n_samples": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }
    output_dir = ensure_output_dir(args.model_name, args.strategy)
    save_json(result, output_dir / "physgeo_results.json")


if __name__ == "__main__":
    main()
