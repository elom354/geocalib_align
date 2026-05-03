"""Create an empty PhysGeo annotation template without fabricated examples.

Inputs:
  None.
Outputs:
  data/physgeo_eval_template.json

Usage:
  python data/build_physgeo_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    payload = {
        "instructions": [
            "Populate this file only with expert-authored or benchmark-sourced geoscience evaluation items.",
            "Do not invent questions, answers, or labels for publication use.",
            "Record provenance for each item, including source benchmark, author, reviewer, and annotation date.",
            "Each item should be reviewed by at least one geoscience domain expert before scoring.",
        ],
        "criteria_reference": [
            "unit_consistency",
            "spatiotemporal_coherence",
            "conservation_laws",
            "stratigraphic_plausibility",
            "causal_correctness",
            "no_physical_hallucinations",
        ],
        "schema": {
            "id": "phys_001",
            "domain": "stratigraphy|seismology|hydrology|geochemistry|climatology|other",
            "question": "Text of the evaluation question",
            "correct_answer": "Reference answer validated by an expert",
            "physically_plausible": True,
            "constraints_to_check": ["unit_consistency", "causal_correctness"],
            "rubric_notes": "Annotation notes, provenance, and reviewer guidance",
            "source": "Benchmark or literature source",
            "reviewer": "Domain expert name or identifier",
            "annotation_date": "YYYY-MM-DD",
        },
        "examples": [],
    }
    output_path = ROOT / "data" / "physgeo_eval_template.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved empty PhysGeo annotation template to %s", output_path)


if __name__ == "__main__":
    main()
