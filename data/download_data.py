"""Download GeoBench and EarthSE subsets for GeoCalib-Align.

Inputs:
  Optional Hugging Face dataset identifiers via CLI arguments.
Outputs:
  Serialized datasets under data/raw/geobench/ and data/raw/earthse/

Usage:
  python data/download_data.py
  python data/download_data.py --geobench-id daven3/geobench --earthse-iron-id ai-earth/Earth-Iron --earthse-silver-id ai-earth/Earth-Silver
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def _save_dataset(dataset: DatasetDict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))


def _log_stats(name: str, dataset: DatasetDict) -> None:
    LOGGER.info("Dataset statistics for %s", name)
    for split, split_dataset in dataset.items():
        LOGGER.info("  %s: %d rows", split, len(split_dataset))
        candidate_column = next(
            (column for column in ("task_type", "task", "category", "type") if column in split_dataset.column_names),
            None,
        )
        if candidate_column is None:
            continue
        counts = split_dataset.to_pandas()[candidate_column].value_counts().to_dict()
        LOGGER.info("  %s breakdown: %s", split, counts)


def _download_with_fallback(primary_id: str, fallback_ids: list[str]) -> DatasetDict:
    errors: list[str] = []
    for dataset_id in [primary_id, *fallback_ids]:
        try:
            LOGGER.info("Attempting to load dataset: %s", dataset_id)
            return load_dataset(dataset_id)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{dataset_id}: {exc}")
            LOGGER.warning("Failed to load %s", dataset_id)
    raise RuntimeError("Unable to download dataset. Errors:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geobench-id", default="daven3/geobench", help="GeoBench dataset ID.")
    parser.add_argument("--earthse-iron-id", default="ai-earth/Earth-Iron", help="EarthSE Iron dataset ID.")
    parser.add_argument("--earthse-silver-id", default="ai-earth/Earth-Silver", help="EarthSE Silver dataset ID.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw_dir = ROOT / "data" / "raw"

    geobench = _download_with_fallback(args.geobench_id, ["daven3/geobench", "xp0123/GeoBench"])
    earthse_iron = _download_with_fallback(args.earthse_iron_id, ["ai-earth/Earth-Iron"])
    earthse_silver = _download_with_fallback(args.earthse_silver_id, ["ai-earth/Earth-Silver"])

    _save_dataset(geobench, raw_dir / "geobench")
    _save_dataset(earthse_iron, raw_dir / "earthse" / "iron")
    _save_dataset(earthse_silver, raw_dir / "earthse" / "silver")

    _log_stats("GeoBench", geobench)
    _log_stats("EarthSE Iron", earthse_iron)
    _log_stats("EarthSE Silver", earthse_silver)


if __name__ == "__main__":
    main()
