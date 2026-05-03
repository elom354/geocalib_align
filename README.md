# GeoCalib-Align

GeoCalib-Align studies a practical failure mode in domain adaptation for geoscience language models: gains in specialized factual knowledge can come with losses in instruction-following behavior and physical plausibility. This repository provides a reproducible research scaffold for comparing lightweight fine-tuning strategies across those competing objectives, using GeoBench and EarthSE for knowledge evaluation, local open-source LLM-as-judge scoring for alignment, and the proposed PhysGeo Score for physically grounded response quality.

The codebase is organized around real data preparation, QLoRA-based fine-tuning, evaluation, aggregation, and downstream statistical analysis for publication-grade experiments.

Only real benchmark data and experiment outputs should be used in any paper submission, appendix, or camera-ready figure. Final tables and figures should be generated exclusively from `results/summary.csv` produced by the evaluation pipeline over GeoBench, EarthSE, GeoSignal, and PhysGeo annotations.

## Requirements

- CUDA 11.8+ recommended for local fine-tuning
- Minimum 16 GB VRAM for lightweight 4-bit LoRA runs
- Python 3.10 via Conda
- No paid API keys required
- Enough disk space and bandwidth to download open-source models from Hugging Face

## Quick Start

```bash
conda env create -f environment.yml
conda activate geocalib_align
python evaluate/aggregate_results.py
python figures/generate_all_figures.py --data results/summary.csv
```

The figure pipeline is intended to run from real aggregated evaluation outputs stored in [`results/summary.csv`](/Users/mac/Desktop/geocalib_align/geocalib_align/results/summary.csv).

## Reproduction Guide

1. Download benchmark data:

```bash
python data/download_data.py
python data/prepare_geosignal.py
python data/build_physgeo_eval.py
```

2. Fine-tune open-source models with each strategy:

```bash
python finetune/train_lora_standard.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/llama_3_1_8b/lora_standard --config config/experiments.yaml
python finetune/train_lora_selective.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/llama_3_1_8b/lora_selective --config config/experiments.yaml
python finetune/train_lora_replay.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/llama_3_1_8b/lora_replay --config config/experiments.yaml
python finetune/train_mix_cpt.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir results/llama_3_1_8b/mix_cpt --config config/experiments.yaml
```

3. Run evaluation:

```bash
python evaluate/eval_closed_tasks.py --model_path results/llama_3_1_8b/lora_standard --model_name Llama-3.1-8B --strategy lora_std
python evaluate/eval_open_tasks.py --model_path results/llama_3_1_8b/lora_standard --model_name Llama-3.1-8B --strategy lora_std --judge_model Qwen/Qwen2.5-3B-Instruct
python evaluate/eval_physgeo.py --model_path results/llama_3_1_8b/lora_standard --model_name Llama-3.1-8B --strategy lora_std --judge_model Qwen/Qwen2.5-3B-Instruct
python evaluate/aggregate_results.py
```

`eval_physgeo.py` requires a manually annotated `data/physgeo_eval_template.json`. The repository intentionally does not fabricate PhysGeo examples.

4. Run analysis and figures:

```bash
python analysis/compute_tradeoff.py --summary results/summary.csv
python analysis/statistical_tests.py
python figures/generate_all_figures.py --data results/summary.csv
```

## Expected Results

| Model / Strategy | Approx. Closed Accuracy | Approx. Prompt Alignment | Approx. PhysGeo |
|---|---:|---:|---:|
| Open-source baseline | 0.55-0.65 | 4.3-4.5 | 0.60-0.66 |
| Standard LoRA | 0.65-0.71 | 3.9-4.1 | 0.64-0.69 |
| Selective LoRA | 0.64-0.68 | 4.2-4.4 | 0.66-0.71 |
| LoRA + Replay | 0.65-0.70 | 4.2-4.4 | 0.68-0.73 |
| Mix-CPT | 0.67-0.72 | 4.3-4.5 | 0.72-0.77 |
| Stronger open-source baselines | 0.68-0.78 | 4.1-4.6 | 0.68-0.78 |

## How To Extend PhysGeo Score

Add expert-reviewed items to [`data/physgeo_eval_template.json`](/Users/mac/Desktop/geocalib_align/geocalib_align/data/physgeo_eval_template.json) with explicit domain coverage, expected physical constraints, and annotation notes. Keep the six core criteria fixed so scores remain comparable, but expand the question pool with adversarial cases, unit traps, temporal reasoning cases, and domain-specific edge conditions. If multiple experts annotate the same sample, store agreement statistics alongside the rubric output before aggregating into the final PhysGeo mean.

## Citation


