"""Download GeoBench and EarthSE subsets for GeoCalib-Align.

Inputs:
  Optional Hugging Face dataset identifiers via CLI arguments.
Outputs:
  Serialized datasets under data/raw/geobench/ and data/raw/earthse/

Usage:
  python data/download_data.py
  python data/download_data.py --geobench-id Deng2023/GeoBench --earthse-id ai-earth/EarthSE
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
    parser.add_argument("--geobench-id", default="Deng2023/GeoBench", help="GeoBench dataset ID.")
    parser.add_argument("--earthse-id", default="ai-earth/EarthSE", help="EarthSE dataset ID.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw_dir = ROOT / "data" / "raw"

    geobench = _download_with_fallback(args.geobench_id, ["Deng2023/GeoBench", "GeoX-Lab/GeoBench"])
    earthse = _download_with_fallback(args.earthse_id, ["ai-earth/EarthSE"])

    earthse_filtered = DatasetDict()
    for split, split_dataset in earthse.items():
        if "subset" in split_dataset.column_names:
            subset_values = {str(item).lower() for item in split_dataset["subset"]}
            if {"iron", "silver"} & subset_values:
                earthse_filtered[split] = split_dataset.filter(
                    lambda item: str(item.get("subset", "")).lower() in {"iron", "silver"}
                )
            else:
                earthse_filtered[split] = split_dataset
        else:
            earthse_filtered[split] = split_dataset

    _save_dataset(geobench, raw_dir / "geobench")
    _save_dataset(earthse_filtered, raw_dir / "earthse")

    _log_stats("GeoBench", geobench)
    _log_stats("EarthSE (Iron + Silver)", earthse_filtered)


if __name__ == "__main__":
    main()
