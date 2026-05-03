"""Train QLoRA with a lightweight Mix-CPT style LSSD objective.

Inputs:
  Prepared GeoSignal train/validation JSON files and experiments.yaml.
Outputs:
  Fine-tuned checkpoints and training_log.json in the output directory.

Usage:
  python finetune/train_mix_cpt.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/llama_3_1_8b/mix_cpt --config config/experiments.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

from common import (
    MetricsLoggerCallback,
    ROOT,
    apply_lora,
    build_datasets,
    build_quant_config,
    build_training_arguments,
    data_collator,
    load_model,
    load_tokenizer,
    load_yaml,
    save_training_history,
)


class LSSDSFTTrainer(SFTTrainer):
    """SFTTrainer with a simple Logit Swap Self-Distillation auxiliary loss."""

    def __init__(self, *args, teacher_model=None, lssd_alpha: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.lssd_alpha = lssd_alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        base_loss = outputs.loss
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        teacher_logits = teacher_outputs.logits.clone()
        student_logits = outputs.logits
        shift_labels = labels[:, 1:].contiguous()
        shift_student = student_logits[:, :-1, :].contiguous()
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        top_tokens = shift_teacher.argmax(dim=-1)
        gt_scores = shift_teacher.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        top_scores = shift_teacher.gather(-1, top_tokens.unsqueeze(-1)).squeeze(-1)
        swapped_teacher = shift_teacher.clone()
        swapped_teacher.scatter_(-1, top_tokens.unsqueeze(-1), gt_scores.unsqueeze(-1))
        swapped_teacher.scatter_(-1, shift_labels.unsqueeze(-1), top_scores.unsqueeze(-1))
        distill_loss = F.kl_div(
            F.log_softmax(shift_student, dim=-1),
            F.softmax(swapped_teacher, dim=-1),
            reduction="batchmean",
        )
        loss = base_loss + self.lssd_alpha * distill_loss
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Base model ID.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--config", required=True, help="Path to experiments YAML.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_yaml(args.config)
    finetune_cfg = config["finetuning"]
    strategy_cfg = finetune_cfg["mix_cpt"]
    tokenizer = load_tokenizer(args.model_id)
    train_dataset, val_dataset = build_datasets(
        ROOT / "data" / "processed" / "geosignal_train.json",
        ROOT / "data" / "processed" / "geosignal_val.json",
        tokenizer,
        finetune_cfg["max_seq_length"],
    )

    student_model = apply_lora(load_model(args.model_id, finetune_cfg), strategy_cfg)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=build_quant_config(finetune_cfg),
        device_map="auto",
        trust_remote_code=True,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    metrics_callback = MetricsLoggerCallback()
    trainer = LSSDSFTTrainer(
        model=student_model,
        teacher_model=teacher_model,
        lssd_alpha=strategy_cfg["lssd_alpha"],
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=build_training_arguments(args.output_dir, finetune_cfg, "mix_cpt"),
        data_collator=data_collator(tokenizer),
        callbacks=[metrics_callback],
        max_seq_length=finetune_cfg["max_seq_length"],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    save_training_history(metrics_callback.history, Path(args.output_dir))


if __name__ == "__main__":
    main()
