"""Shared evaluation helpers for GeoCalib-Align."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def ensure_output_dir(model_name: str, strategy: str) -> Path:
    path = ROOT / "results" / model_name / strategy
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_local_or_hf_geobench() -> DatasetDict:
    local_path = ROOT / "data" / "raw" / "geobench"
    if local_path.exists():
        from datasets import load_from_disk

        return load_from_disk(str(local_path))
    return load_dataset("Deng2023/GeoBench")


def select_task_subset(dataset: DatasetDict, task_names: set[str]) -> Dataset:
    if "test" in dataset:
        split = dataset["test"]
    elif "validation" in dataset:
        split = dataset["validation"]
    else:
        split = next(iter(dataset.values()))

    task_column = next((col for col in ("task_type", "task", "category", "type") if col in split.column_names), None)
    if task_column is None:
        return split
    return split.filter(lambda row: str(row.get(task_column, "")).lower() in task_names)


def load_model_and_tokenizer(model_path: str):
    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"

    tokenizer_source = model_path
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        tokenizer_source = adapter_config.get("base_model_name_or_path", model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_config_path.exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def build_prompt(question: str, context: str = "") -> str:
    context_block = f"\nContext: {context}" if context else ""
    return f"Question: {question}{context_block}\nAnswer:"


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:", maxsplit=1)[-1].strip()


def extract_closed_prediction(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"\b(True|False|A|B|C|D)\b", cleaned, flags=re.IGNORECASE)
    return match.group(1).upper() if match else cleaned[:16].upper()


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def infer_model_name(model_path: str) -> str:
    name = Path(model_path).name
    return name.replace("_", "-")


def mean_or_zero(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def extract_json_object(text: str) -> dict[str, Any]:
    # Try to find something that looks like JSON objects
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        # Fallback if no {} block found
        return {"error": f"Model response does not contain JSON: {text}"}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        # Graceful degradation if the JSON is malformed
        return {"error": f"JSON parse error: {exc}", "raw": match.group(0)}
