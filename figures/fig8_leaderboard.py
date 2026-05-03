"""Generate a publication-style leaderboard table figure and LaTeX export.

Inputs:
  CSV with performance metrics.
Outputs:
  figures/fig8_leaderboard.pdf, .png, and .tex

Usage:
  python figures/fig8_leaderboard.py --data results/summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from plot_utils import apply_publication_style, load_results, save_figure, summarize_runs


def _to_latex_table(columns: list[str], rows: list[list[object]]) -> str:
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\hline"]
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(str(item) for item in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/summary.csv", help="Input CSV file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style()

    df = summarize_runs(load_results(args.data))
    df["composite_score"] = 0.4 * df["closed_overall"] + 0.3 * df["prompt_alignment_norm"] + 0.3 * df["physgeo_score"]
    table_df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    table_df["Rank"] = table_df.index + 1
    table_df["Params (B)"] = table_df["params_b"].fillna("API")
    display = table_df[
        [
            "Rank",
            "model",
            "strategy",
            "closed_overall",
            "prompt_alignment",
            "answer_relevance",
            "bert_f1",
            "physgeo_score",
            "Params (B)",
        ]
    ].copy()
    display.columns = ["Rank", "Model", "Strategy", "GeoBench Acc", "Prompt Align.", "Answer Rel.", "BERT F1", "PhysGeo", "Params (B)"]
    for col in ["GeoBench Acc", "Prompt Align.", "Answer Rel.", "BERT F1", "PhysGeo"]:
        display[col] = display[col].map(lambda value: f"{value:.3f}")

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E5E7EB")
            continue
        source_row = table_df.iloc[row - 1]
        if row <= 3:
            cell.set_facecolor("#DCFCE7")
        if bool(source_row["is_proprietary"]):
            cell.set_facecolor("#DBEAFE")

    numeric_cols = ["GeoBench Acc", "Prompt Align.", "Answer Rel.", "BERT F1", "PhysGeo"]
    for col_name in numeric_cols:
        best_idx = display[col_name].astype(float).idxmax() + 1
        col_idx = list(display.columns).index(col_name)
        table[(best_idx, col_idx)].set_text_props(weight="bold")

    tex_path = Path("figures") / "fig8_leaderboard.tex"
    tex_path.write_text(_to_latex_table(display.columns.tolist(), display.values.tolist()), encoding="utf-8")
    logging.info("Saved LaTeX table to %s", tex_path)
    ax.set_title("GeoCalib-Align Leaderboard", pad=14)

    save_figure(fig, Path("figures") / "fig8_leaderboard")


if __name__ == "__main__":
    main()
