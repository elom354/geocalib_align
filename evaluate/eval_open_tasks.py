"""Evaluate open-ended GeoBench tasks with a local open-source judge and BERTScore.

Inputs:
  Model path, GeoBench open tasks, and a local judge model path.
Outputs:
  results/{model_name}/{strategy}/open_results.json

Usage:
  python evaluate/eval_open_tasks.py --model_path meta-llama/Meta-Llama-3.1-8B-Instruct --model_name Llama-3.1-8B --strategy baseline
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from bert_score import score as bert_score

from common import (
    build_prompt,
    ensure_output_dir,
    extract_json_object,
    generate_answer,
    load_local_or_hf_geobench,
    load_model_and_tokenizer,
    save_json,
    select_task_subset,
)

JUDGE_TEMPLATE = """You are an expert geoscience evaluator.
Question: {question}
Model answer: {answer}
Rate the answer on three criteria from 1 (very poor) to 5 (excellent):
- Prompt Alignment: Does the answer address what was asked?
- Correctness: Is the geoscientific content accurate?
- Answer Relevance: Is the answer focused and non-redundant?
Reply ONLY as JSON: {{"prompt_alignment": X, "correctness": X, "answer_relevance": X}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Model path or HF model ID.")
    parser.add_argument("--model_name", required=True, help="Short model name for results.")
    parser.add_argument("--strategy", required=True, help="Strategy identifier.")
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct", help="Local open-source judge model.")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum evaluation samples.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    judge_model, judge_tokenizer = load_model_and_tokenizer(args.judge_model)
    geobench = load_local_or_hf_geobench()
    subset = select_task_subset(geobench, {"factual_qa", "reasoning"})
    if len(subset) > args.max_samples:
        subset = subset.select(range(args.max_samples))

    judged_samples = []
    references = []
    predictions = []
    for row in subset:
        question = row.get("question") or row.get("prompt") or row.get("instruction") or ""
        context = row.get("context") or ""
        reference = row.get("answer") or row.get("reference_answer") or row.get("label") or ""
        answer = generate_answer(model, tokenizer, build_prompt(question, context), max_new_tokens=300)
        prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)
        raw_judge_output = generate_answer(judge_model, judge_tokenizer, prompt, max_new_tokens=128)
        judged = extract_json_object(raw_judge_output)

        # Fallback values if JSON decoding failed
        prompt_align = judged.get("prompt_alignment", 1)  # Default penalty is 1
        correctness = judged.get("correctness", 1)
        relevance = judged.get("answer_relevance", 1)

        judged_samples.append({
            "question": question,
            "answer": answer,
            "reference": reference,
            "prompt_alignment": prompt_align,
            "correctness": correctness,
            "answer_relevance": relevance,
            "raw_judge_output": raw_judge_output if "error" in judged else None
        })
        references.append(reference)
        predictions.append(answer)

    precision, recall, f1 = bert_score(predictions, references, lang="en", verbose=False)
    result = {
        "model": args.model_name,
        "strategy": args.strategy,
        "prompt_alignment": sum(item["prompt_alignment"] for item in judged_samples) / max(1, len(judged_samples)),
        "correctness": sum(item["correctness"] for item in judged_samples) / max(1, len(judged_samples)),
        "answer_relevance": sum(item["answer_relevance"] for item in judged_samples) / max(1, len(judged_samples)),
        "bert_precision": float(precision.mean().item()),
        "bert_recall": float(recall.mean().item()),
        "bert_f1": float(f1.mean().item()),
        "judge_model": args.judge_model,
        "n_samples": len(judged_samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": judged_samples,
    }
    output_dir = ensure_output_dir(args.model_name, args.strategy)
    save_json(result, output_dir / "open_results.json")


if __name__ == "__main__":
    main()
