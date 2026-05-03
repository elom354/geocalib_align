"""Train a standard QLoRA model on GeoSignal with TRL SFTTrainer.

Inputs:
  Prepared GeoSignal train/validation JSON files and experiments.yaml.
Outputs:
  Fine-tuned checkpoints and training_log.json in the output directory.

Usage:
  python finetune/train_lora_standard.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/llama_3_1_8b/lora_standard --config config/experiments.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from trl import SFTTrainer

from common import (
    MetricsLoggerCallback,
    ROOT,
    apply_lora,
    build_datasets,
    build_training_arguments,
    data_collator,
    load_model,
    load_tokenizer,
    load_yaml,
    save_training_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Base model ID.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--config", required=True, help="Path to experiments YAML.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_yaml(args.config)
    finetune_cfg = config["finetuning"]
    strategy_cfg = finetune_cfg["lora_standard"]
    tokenizer = load_tokenizer(args.model_id)
    train_dataset, val_dataset = build_datasets(
        ROOT / "data" / "processed" / "geosignal_train.json",
        ROOT / "data" / "processed" / "geosignal_val.json",
        tokenizer,
        finetune_cfg["max_seq_length"],
    )
    model = apply_lora(load_model(args.model_id, finetune_cfg), strategy_cfg)

    metrics_callback = MetricsLoggerCallback()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=build_training_arguments(args.output_dir, finetune_cfg, "lora_standard"),
        dataset_text_field="text",
        callbacks=[metrics_callback],
        max_seq_length=finetune_cfg["max_seq_length"],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    save_training_history(metrics_callback.history, Path(args.output_dir))


if __name__ == "__main__":
    main()
