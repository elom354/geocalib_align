"""Run all GeoCalib-Align figure scripts and verify outputs.

Usage:
  python figures/generate_all_figures.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

FIGURE_SCRIPTS = [
    "fig1_tradeoff_scatter.py",
    "fig2_radar_chart.py",
    "fig3_heatmap.py",
    "fig4_bar_strategies.py",
    "fig5_physgeo_breakdown.py",
    "fig6_cost_performance.py",
    "fig7_alignment_delta.py",
    "fig8_leaderboard.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file with real aggregated results.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).resolve().parent
    generated = []

    for script in FIGURE_SCRIPTS:
        script_path = root / script
        logging.info("Running %s", script)
        subprocess.run([sys.executable, str(script_path), "--data", args.data], check=True)
        stem = root / script.replace(".py", "")
        pdf_path = stem.with_suffix(".pdf")
        png_path = stem.with_suffix(".png")
        if not pdf_path.exists() or not png_path.exists():
            raise FileNotFoundError(f"Missing expected outputs for {script}")
        generated.append((pdf_path, png_path))

    logging.info("Generated %d figure pairs successfully.", len(generated))
    for pdf_path, png_path in generated:
        logging.info(
            "%s (%.1f KB) | %s (%.1f KB)",
            pdf_path.name,
            pdf_path.stat().st_size / 1024.0,
            png_path.name,
            png_path.stat().st_size / 1024.0,
        )


if __name__ == "__main__":
    main()
