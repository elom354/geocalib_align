"""Run the full free evaluation stack for any local or Hugging Face model.

This file replaces the previous proprietary API-based workflow. It is kept only
for backward compatibility with earlier repository layouts.

Inputs:
  Model path or model ID, output model name, strategy, and optional local judge.
Outputs:
  results/{model_name}/{strategy}/{closed,open,physgeo}_results.json

Usage:
  python evaluate/eval_proprietary.py --model_path mistralai/Mistral-7B-Instruct-v0.3 --model_name Mistral-7B --strategy baseline
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "evaluate" / script), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Model path or HF model ID.")
    parser.add_argument("--model_name", required=True, help="Short model name for results.")
    parser.add_argument("--strategy", default="baseline", help="Strategy identifier.")
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-3B-Instruct", help="Local open-source judge model.")
    parser.add_argument("--max_samples", type=int, default=120, help="Maximum evaluation samples for each stage.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Running free evaluation stack for %s", args.model_name)

    _run(
        "eval_closed_tasks.py",
        "--model_path",
        args.model_path,
        "--model_name",
        args.model_name,
        "--strategy",
        args.strategy,
        "--max_samples",
        str(args.max_samples),
    )
    _run(
        "eval_open_tasks.py",
        "--model_path",
        args.model_path,
        "--model_name",
        args.model_name,
        "--strategy",
        args.strategy,
        "--judge_model",
        args.judge_model,
        "--max_samples",
        str(args.max_samples),
    )
    _run(
        "eval_physgeo.py",
        "--model_path",
        args.model_path,
        "--model_name",
        args.model_name,
        "--strategy",
        args.strategy,
        "--judge_model",
        args.judge_model,
    )


if __name__ == "__main__":
    main()
