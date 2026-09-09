#!/usr/bin/env python3
"""Render the quantization-format table as a Hackster-ready PNG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "quantization-formats-table.png"

HEADERS = ["FORMAT", "WEIGHTS", "ACTIVATIONS", "DESCRIPTION"]
ROWS = [
    ["FP16", "16-bit floating point", "16-bit floating point", "Half precision"],
    ["W8A8", "8-bit", "8-bit", "INT8 quantization"],
    ["W4A16", "4-bit", "16-bit", "4-bit weight quantization"],
    ["W4A4", "4-bit", "4-bit", "Full / pure INT4"],
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(12, 5.5), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#f8fafc")
    axis.axis("off")

    axis.text(
        0.04,
        0.92,
        "Quantization formats used in this project",
        transform=axis.transAxes,
        fontsize=21,
        fontweight="bold",
        color="#0f172a",
        va="center",
    )
    axis.text(
        0.04,
        0.855,
        "W = weight bit width     A = activation bit width",
        transform=axis.transAxes,
        fontsize=11.5,
        color="#475569",
        va="center",
    )

    panel = FancyBboxPatch(
        (0.035, 0.13),
        0.93,
        0.65,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        transform=axis.transAxes,
        linewidth=1,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
        zorder=0,
    )
    axis.add_patch(panel)

    table = axis.table(
        cellText=ROWS,
        colLabels=HEADERS,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.13, 0.25, 0.25, 0.37],
        bbox=[0.055, 0.16, 0.89, 0.57],
        zorder=2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        cell.set_linewidth(0.8)
        cell.PAD = 0.18
        if row == 0:
            cell.set_facecolor("#0f172a")
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#f1f5f9" if row % 2 == 0 else "#ffffff")
            cell.get_text().set_color("#1e293b")
            if column == 0:
                cell.get_text().set_fontweight("bold")
                if ROWS[row - 1][0] == "W4A16":
                    cell.get_text().set_color("#0284c7")

    axis.text(
        0.04,
        0.055,
        "W4A16 reduces weight storage while retaining 16-bit activations for accuracy.",
        transform=axis.transAxes,
        fontsize=11,
        color="#475569",
        va="center",
    )

    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
