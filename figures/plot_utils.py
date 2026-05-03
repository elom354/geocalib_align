"""Shared plotting utilities for GeoCalib-Align figures.

Purpose:
  Centralize styling, config loading, data loading, and save helpers used by
  the publication-quality figure scripts.

Usage:
  from plot_utils import apply_publication_style, load_results, save_figure
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG = ROOT / "config" / "models.yaml"


def apply_publication_style() -> None:
    """Apply a consistent journal-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def load_config(config_path: Path | None = None) -> dict:
    """Load the YAML models configuration."""
    target = config_path or MODELS_CONFIG
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_results(data_path: str | Path) -> pd.DataFrame:
    """Load result CSV and add normalized helper columns."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "model",
        "strategy",
        "closed_overall",
        "prompt_alignment",
        "answer_relevance",
        "bert_f1",
        "physgeo_score",
        "is_proprietary",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["prompt_alignment_norm"] = df["prompt_alignment"] / 5.0
    df["answer_relevance_norm"] = df["answer_relevance"] / 5.0
    df["correctness_norm"] = df["correctness"] / 5.0
    df["params_plot"] = df["params_b"].fillna(20.0)
    return df


def get_color_maps(config: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return color mappings for models and strategies."""
    model_colors: Dict[str, str] = {}
    for section in ("open_source_models", "proprietary_models"):
        for model in config.get(section, {}).values():
            model_colors[model["short_name"]] = model["color"]

    strategy_colors = {
        item["id"]: item["color"] for item in config.get("finetuning_strategies", [])
    }
    return model_colors, strategy_colors


def get_strategy_labels(config: dict) -> Dict[str, str]:
    """Return display labels for strategy IDs."""
    return {item["id"]: item["name"] for item in config.get("finetuning_strategies", [])}


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated rows into mean/std statistics for plotting."""
    numeric_columns = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in {"is_proprietary"}
    ]
    group_cols = ["model", "strategy", "is_proprietary"]
    grouped = df.groupby(group_cols, dropna=False)
    mean_df = grouped[numeric_columns].mean().reset_index()
    std_df = grouped[numeric_columns].std(ddof=1).fillna(0.0).reset_index()
    std_df = std_df.rename(columns={col: f"{col}_std" for col in numeric_columns})
    merged = mean_df.merge(std_df, on=group_cols, how="left")
    merged["prompt_alignment_norm"] = merged["prompt_alignment"] / 5.0
    merged["answer_relevance_norm"] = merged["answer_relevance"] / 5.0
    merged["correctness_norm"] = merged["correctness"] / 5.0
    merged["params_plot"] = merged["params_b"].fillna(20.0)
    return merged


def save_figure(fig: plt.Figure, output_stem: str | Path) -> Tuple[Path, Path]:
    """Save a figure as both PDF and PNG."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    LOGGER.info("Saved figure to %s and %s", pdf_path, png_path)
    return pdf_path, png_path
