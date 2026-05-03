"""Train QLoRA with mixed geoscience and general instruction replay.

Inputs:
  Prepared GeoSignal, replay JSON, and experiments.yaml.
Outputs:
  Fine-tuned checkpoints and training_log.json in the output directory.

Usage:
  python finetune/train_lora_replay.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/llama_3_1_8b/lora_replay --config config/experiments.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer

from common import MetricsLoggerCallback, ROOT, apply_lora, build_training_arguments, format_chat_example, load_instruction_records, load_model, load_tokenizer, load_yaml, save_training_history


def _build_replay_dataset(tokenizer, max_length: int, geosignal_records: list[dict], replay_records: list[dict]) -> Dataset:
    del tokenizer, max_length
    mixed_records = [{"record": item, "source": "geosignal"} for item in geosignal_records] + [
        {"record": item, "source": "replay"} for item in replay_records
    ]
    dataset = Dataset.from_list(mixed_records).shuffle(seed=42)
    return dataset.map(lambda row: {"text": format_chat_example(row["record"])}, remove_columns=["record", "source"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Base model ID.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--config", required=True, help="Path to experiments YAML.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_yaml(args.config)
    finetune_cfg = config["finetuning"]
    strategy_cfg = finetune_cfg["lora_replay"]
    tokenizer = load_tokenizer(args.model_id)

    geosignal_train = load_instruction_records(ROOT / "data" / "processed" / "geosignal_train.json")
    geosignal_val = load_instruction_records(ROOT / "data" / "processed" / "geosignal_val.json")
    replay_records = load_instruction_records(ROOT / "data" / "processed" / "general_instructions_replay.json")

    geo_count = len(geosignal_train)
    replay_target = min(int(geo_count * strategy_cfg["replay_ratio"] / max(1e-6, 1 - strategy_cfg["replay_ratio"])), len(replay_records))
    replay_subset = replay_records[:replay_target]
    logging.info("Replay mixing ratio: geoscience=%d, general=%d", geo_count, replay_target)

    train_dataset = _build_replay_dataset(tokenizer, finetune_cfg["max_seq_length"], geosignal_train, replay_subset)
    val_dataset = _build_replay_dataset(tokenizer, finetune_cfg["max_seq_length"], geosignal_val, replay_subset[: len(geosignal_val)])
    model = apply_lora(load_model(args.model_id, finetune_cfg), strategy_cfg)

    metrics_callback = MetricsLoggerCallback()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=build_training_arguments(args.output_dir, finetune_cfg, "lora_replay"),
        dataset_text_field="text",
        callbacks=[metrics_callback],
        max_seq_length=finetune_cfg["max_seq_length"],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    save_training_history(metrics_callback.history, Path(args.output_dir))
    (Path(args.output_dir) / "mixing_summary.json").write_text(
        json.dumps({"geosignal_count": geo_count, "replay_count": replay_target}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
