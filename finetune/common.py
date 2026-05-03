"""Shared fine-tuning utilities for GeoCalib-Align training scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


class MetricsLoggerCallback(TrainerCallback):
    """Persist trainer logs to JSON after training."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if logs:
            payload = {key: float(value) if isinstance(value, (int, float)) else value for key, value in logs.items()}
            payload["step"] = int(state.global_step)
            self.history.append(payload)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_instruction_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_chat_example(record: dict) -> str:
    input_block = f"\nInput: {record['input']}" if record.get("input") else ""
    return f"Instruction: {record['instruction']}{input_block}\nResponse: {record['output']}"


def build_datasets(train_path: Path, val_path: Path, tokenizer, max_length: int) -> tuple[Dataset, Dataset]:
    def _tokenize(batch):
        texts = [format_chat_example(record) for record in batch["record"]]
        tokens = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        return tokens

    train_records = load_instruction_records(train_path)
    val_records = load_instruction_records(val_path)
    train_dataset = Dataset.from_list([{"record": item} for item in train_records]).map(_tokenize, batched=True, remove_columns=["record"])
    val_dataset = Dataset.from_list([{"record": item} for item in val_records]).map(_tokenize, batched=True, remove_columns=["record"])
    return train_dataset, val_dataset


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_quant_config(settings: dict) -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, settings.get("bnb_4bit_compute_dtype", "float16"))
    return BitsAndBytesConfig(
        load_in_4bit=settings.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model(model_id: str, finetune_cfg: dict):
    quant_config = build_quant_config(finetune_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, strategy_cfg: dict):
    lora_config = LoraConfig(
        r=strategy_cfg["r"],
        lora_alpha=strategy_cfg["lora_alpha"],
        target_modules=strategy_cfg["target_modules"],
        lora_dropout=strategy_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def freeze_bottom_layers(model, freeze_pct: float) -> tuple[list[str], list[str]]:
    layers = None
    for attr in ("model.layers", "model.model.layers", "transformer.h"):
        current = model
        valid = True
        for name in attr.split("."):
            if not hasattr(current, name):
                valid = False
                break
            current = getattr(current, name)
        if valid:
            layers = current
            break
    if layers is None:
        raise ValueError("Unable to locate transformer layers for selective freezing.")

    layer_count = len(layers)
    freeze_count = int(layer_count * freeze_pct)
    frozen = []
    trainable = []
    for idx, layer in enumerate(layers):
        requires_grad = idx >= freeze_count
        for param in layer.parameters():
            param.requires_grad = requires_grad
        name = f"layer_{idx}"
        if requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)
    return frozen, trainable


def build_training_arguments(output_dir: str | Path, finetune_cfg: dict, run_name: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=finetune_cfg["batch_size"],
        per_device_eval_batch_size=finetune_cfg["batch_size"],
        gradient_accumulation_steps=finetune_cfg["gradient_accumulation"],
        learning_rate=float(finetune_cfg["learning_rate"]),
        warmup_steps=finetune_cfg["warmup_steps"],
        max_steps=finetune_cfg["max_steps"],
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 if torch.cuda.is_available() else False,
        fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) if torch.cuda.is_available() else False,
        lr_scheduler_type="cosine",
        report_to=[],
    )


def save_training_history(history: Iterable[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_log.json"
    path.write_text(json.dumps(list(history), indent=2), encoding="utf-8")
    LOGGER.info("Saved training metrics history to %s", path)


def data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
