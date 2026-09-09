#!/usr/bin/env python3
"""Render saved COCO-format RKNN predictions as a side-by-side comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COCO_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}


def color_for(category_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(category_id * 3576)
    return tuple(int(value) for value in rng.integers(70, 240, size=3))


def load_predictions(path: Path, image_id: int, confidence: float) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    selected = [
        record for record in records
        if int(record["image_id"]) == image_id and float(record["score"]) >= confidence
    ]
    return sorted(selected, key=lambda record: float(record["score"]), reverse=True)


def draw_panel(source: np.ndarray, label: str, records: list[dict]) -> np.ndarray:
    panel = source.copy()
    scale = max(panel.shape[0], panel.shape[1]) / 900.0
    thickness = max(2, round(2 * scale))
    font_scale = max(0.5, 0.55 * scale)

    for record in records:
        x, y, width, height = record["bbox"]
        p1 = (round(x), round(y))
        p2 = (round(x + width), round(y + height))
        category_id = int(record["category_id"])
        color = color_for(category_id)
        cv2.rectangle(panel, p1, p2, color, thickness, cv2.LINE_AA)
        text = f"{COCO_NAMES.get(category_id, category_id)} {float(record['score']):.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top = max(0, p1[1] - text_height - baseline - 6)
        right = min(panel.shape[1] - 1, p1[0] + text_width + 8)
        cv2.rectangle(panel, (p1[0], top), (right, p1[1]), color, -1)
        cv2.putText(
            panel, text, (p1[0] + 4, p1[1] - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness,
            cv2.LINE_AA,
        )

    header_height = 76
    canvas = np.full((panel.shape[0] + header_height, panel.shape[1], 3), 248, np.uint8)
    canvas[header_height:] = panel
    cv2.putText(canvas, label, (22, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 26, 34), 2, cv2.LINE_AA)
    cv2.putText(
        canvas, f"{len(records)} detections at confidence >= {ARGS.confidence:.2f}",
        (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (75, 85, 99), 1, cv2.LINE_AA,
    )
    return canvas


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--separate-output-dir", type=Path)
    args = parser.parse_args()
    if args.output is None and args.separate_output_dir is None:
        parser.error("provide --output, --separate-output-dir, or both")
    global ARGS
    ARGS = args

    candidates = [
        args.images / f"{args.image_id:012d}{suffix}"
        for suffix in (".jpg", ".jpeg", ".png")
    ]
    image_path = next((path for path in candidates if path.exists()), None)
    if image_path is None:
        raise FileNotFoundError(f"image {args.image_id:012d} not found in {args.images}")
    source = cv2.imread(str(image_path))
    if source is None:
        raise RuntimeError(f"cannot read {image_path}")

    labeled_panels = []
    for label, prediction_path in args.prediction:
        records = load_predictions(prediction_path, args.image_id, args.confidence)
        labeled_panels.append((label, draw_panel(source, label, records)))

    if args.separate_output_dir:
        args.separate_output_dir.mkdir(parents=True, exist_ok=True)
        for label, panel in labeled_panels:
            slug = "-".join(
                part for part in "".join(
                    character.lower() if character.isalnum() else " "
                    for character in label
                ).split() if part
            )
            output_path = args.separate_output_dir / f"{args.image_id:012d}-{slug}.png"
            if not cv2.imwrite(str(output_path), panel):
                raise RuntimeError(f"cannot write {output_path}")
            print(f"wrote {output_path} ({panel.shape[1]}x{panel.shape[0]})")

    if args.output is None:
        return

    panels = [panel for _, panel in labeled_panels]

    target_height = min(720, max(panel.shape[0] for panel in panels))
    resized = []
    for panel in panels:
        ratio = target_height / panel.shape[0]
        resized.append(cv2.resize(panel, (round(panel.shape[1] * ratio), target_height)))
    gutter = np.full((target_height, 12, 3), 230, np.uint8)
    comparison = resized[0]
    for panel in resized[1:]:
        comparison = np.hstack((comparison, gutter, panel))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), comparison):
        raise RuntimeError(f"cannot write {args.output}")
    print(f"wrote {args.output} ({comparison.shape[1]}x{comparison.shape[0]})")


if __name__ == "__main__":
    ARGS = None
    main()
