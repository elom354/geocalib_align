"""Prepare GeoSignal and replay data for instruction fine-tuning.

Inputs:
  GeoSignal and Alpaca dataset identifiers.
Outputs:
  data/processed/geosignal_train.json
  data/processed/geosignal_val.json
  data/processed/general_instructions_replay.json

Usage:
  python data/prepare_geosignal.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean

from datasets import Dataset, load_dataset

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def _format_example(example: dict) -> dict:
    instruction = example.get("instruction") or example.get("question") or example.get("prompt") or "Answer the question."
    input_text = example.get("input") or example.get("context") or example.get("metadata") or ""
    output_text = example.get("output") or example.get("answer") or example.get("response") or example.get("label") or ""
    return {"instruction": str(instruction), "input": str(input_text), "output": str(output_text)}


def _length_stats(records: list[dict]) -> dict:
    lengths = [len(f"{item['instruction']} {item['input']} {item['output']}".split()) for item in records]
    lengths_sorted = sorted(lengths)
    return {
        "count": len(lengths),
        "mean_words": round(mean(lengths), 2),
        "p50_words": lengths_sorted[len(lengths_sorted) // 2],
        "p90_words": lengths_sorted[min(len(lengths_sorted) - 1, int(len(lengths_sorted) * 0.9))],
        "max_words": max(lengths_sorted),
    }


def _save_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geosignal-id", default="daven3/geosignal", help="GeoSignal dataset ID.")
    parser.add_argument("--alpaca-id", default="tatsu-lab/alpaca", help="Replay dataset ID.")
    parser.add_argument("--replay-size", type=int, default=5000, help="Number of Alpaca samples to export.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    processed_dir = ROOT / "data" / "processed"

    try:
        geosignal = load_dataset(args.geosignal_id)
    except Exception as exc:
        raise RuntimeError(
            "Unable to download GeoSignal. Verify the dataset ID and access permissions. "
            f"Tried: {args.geosignal_id}"
        ) from exc
    train_split = geosignal["train"] if "train" in geosignal else next(iter(geosignal.values()))
    formatted = train_split.map(_format_example)
    formatted = formatted.remove_columns([col for col in formatted.column_names if col not in {"instruction", "input", "output"}])
    split_data = formatted.train_test_split(test_size=0.1, seed=42)

    train_records = [split_data["train"][idx] for idx in range(len(split_data["train"]))]
    val_records = [split_data["test"][idx] for idx in range(len(split_data["test"]))]
    _save_json(train_records, processed_dir / "geosignal_train.json")
    _save_json(val_records, processed_dir / "geosignal_val.json")

    alpaca = load_dataset(args.alpaca_id)["train"].shuffle(seed=42).select(range(args.replay_size))
    replay_records = [_format_example(alpaca[idx]) for idx in range(len(alpaca))]
    _save_json(replay_records, processed_dir / "general_instructions_replay.json")

    LOGGER.info("GeoSignal train stats: %s", _length_stats(train_records))
    LOGGER.info("GeoSignal validation stats: %s", _length_stats(val_records))
    LOGGER.info("Replay subset stats: %s", _length_stats(replay_records))


if __name__ == "__main__":
    main()
