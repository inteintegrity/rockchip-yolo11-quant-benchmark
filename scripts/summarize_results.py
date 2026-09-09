#!/usr/bin/env python3
"""Aggregate RK3576 benchmark artifacts and render the article comparison chart."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt


RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

CONFIGS = {
    "FP16": {
        "runs": RESULTS / "formal" / "fp16_r*.json",
        "metrics": RESULTS / "coco1000_fp16_metrics.json",
        "conversion": RESULTS / "convert_fp16.json",
    },
    "INT8": {
        "runs": RESULTS / "formal" / "int8_r*.json",
        "metrics": RESULTS / "coco1000_int8_metrics.json",
        "conversion": RESULTS / "convert_int8.json",
    },
    "W4A16": {
        "runs": RESULTS / "formal_w4_gdq_group128" / "w4a16_gdq_group128_r*.json",
        "metrics": RESULTS / "coco1000_w4a16_gdq_group128_metrics.json",
        "conversion": RESULTS / "convert_w4a16_gdq_group128.json",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> list[dict]:
    rows = []
    for name, paths in CONFIGS.items():
        runs = [read_json(path) for path in sorted(paths["runs"].parent.glob(paths["runs"].name))]
        if len(runs) != 3:
            raise RuntimeError(f"Expected 3 formal runs for {name}, found {len(runs)}")
        metrics = read_json(paths["metrics"])
        conversion = read_json(paths["conversion"])
        rows.append(
            {
                "model": name,
                "model_bytes": conversion["output_bytes"],
                "model_mib": conversion["output_bytes"] / 1024**2,
                "conversion_seconds": conversion["conversion_seconds"],
                "mean_latency_ms": mean(run["mean_ms"] for run in runs),
                "mean_fps": mean(run["fps_from_mean"] for run in runs),
                "mean_peak_rss_mib": mean(run["rss_peak_bytes"] for run in runs) / 1024**2,
                "mean_after_init_rss_mib": mean(run["rss_after_init_bytes"] for run in runs) / 1024**2,
                "map_50_95": metrics["map_50_95"],
                "map_50": metrics["map_50"],
                "map_75": metrics["map_75"],
                "mar_100": metrics["mar_100"],
                "accuracy_images": metrics["images"],
                "sha256": conversion["output_sha256"],
            }
        )
    baseline = rows[0]
    for row in rows:
        row["size_vs_fp16_pct"] = row["model_bytes"] / baseline["model_bytes"] * 100
        row["ap_change_vs_fp16"] = row["map_50_95"] - baseline["map_50_95"]
        row["fps_change_vs_fp16_pct"] = (row["mean_fps"] / baseline["mean_fps"] - 1) * 100
    return rows


def write_outputs(rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_labels(axis, bars, values, fmt: str) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def render_chart(rows: list[dict]) -> None:
    ASSETS.mkdir(exist_ok=True)
    names = [row["model"] for row in rows]
    colors = ["#64748b", "#0ea5e9", "#f97316"]
    panels = [
        ("model_mib", "Model size (MiB) ↓", "{:.2f}"),
        ("mean_latency_ms", "Inference latency (ms) ↓", "{:.2f}"),
        ("mean_peak_rss_mib", "Peak process RSS (MiB) ↓", "{:.2f}"),
        ("map_50_95", "COCO AP50–95 ↑", "{:.4f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for axis, (key, title, fmt) in zip(axes.flat, panels):
        values = [row[key] for row in rows]
        bars = axis.bar(names, values, color=colors, width=0.62)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_ylim(0, max(values) * 1.18)
        add_labels(axis, bars, values, fmt)
    fig.suptitle("YOLO11n on RK3576 — FP16 vs INT8 vs W4A16", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "640×640 · NPU_CORE_0_1 · latency excludes decode, letterbox and NMS · AP on fixed COCO val2017 1,000-image subset",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(ASSETS / "yolo11n-rk3576-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    summary = collect()
    write_outputs(summary)
    render_chart(summary)
    for item in summary:
        print(
            f'{item["model"]}: {item["model_mib"]:.2f} MiB, '
            f'{item["mean_latency_ms"]:.2f} ms, {item["mean_fps"]:.2f} FPS, '
            f'{item["mean_peak_rss_mib"]:.2f} MiB RSS, AP={item["map_50_95"]:.4f}'
        )
