#!/usr/bin/env python3
"""Run YOLO11 RKNN inference and optionally evaluate COCO AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_LINEAR)
    pad_w = size - resized.shape[1]
    pad_h = size - resized.shape[0]
    left = pad_w // 2
    top = pad_h // 2
    output = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return output, ratio, (left, top)


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def decode_dfl(position: np.ndarray) -> np.ndarray:
    n, channels, height, width = position.shape
    bins = channels // 4
    values = softmax(position.reshape(n, 4, bins, height, width), axis=2)
    weights = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
    return np.sum(values * weights, axis=2)


def decode_branch(position: np.ndarray, image_size: int = 640) -> np.ndarray:
    height, width = position.shape[2:4]
    col, row = np.meshgrid(np.arange(width), np.arange(height))
    grid = np.stack((col, row), axis=0).reshape(1, 2, height, width)
    stride = np.array([image_size / width, image_size / height], dtype=np.float32).reshape(1, 2, 1, 1)
    distances = decode_dfl(position)
    top_left = grid + 0.5 - distances[:, :2]
    bottom_right = grid + 0.5 + distances[:, 2:]
    return np.concatenate((top_left * stride, bottom_right * stride), axis=1)


def flatten_spatial(values: np.ndarray) -> np.ndarray:
    return values.transpose(0, 2, 3, 1).reshape(-1, values.shape[1])


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[index] + areas[order[1:]] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    outputs: list[np.ndarray],
    confidence: float,
    nms_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.concatenate([flatten_spatial(decode_branch(outputs[index])) for index in (0, 3, 6)])
    class_scores = np.concatenate([flatten_spatial(outputs[index]) for index in (1, 4, 7)])
    classes = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)
    selected = scores >= confidence
    boxes, classes, scores = boxes[selected], classes[selected], scores[selected]
    if not len(scores):
        return boxes[:0], classes[:0], scores[:0]

    # OpenCV performs the same class-aware NMS in native code. Keeping at most
    # 300 final detections matches the standard YOLO validation convention;
    # COCO AP itself evaluates at most the top 100 detections per image.
    xywh = boxes.copy()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    kept_array = np.asarray(
        cv2.dnn.NMSBoxesBatched(
            xywh.tolist(),
            scores.tolist(),
            classes.astype(np.int32).tolist(),
            confidence,
            nms_threshold,
        ),
        dtype=np.int64,
    ).reshape(-1)
    if kept_array.size == 0:
        return boxes[:0], classes[:0], scores[:0]
    kept_array = kept_array[np.argsort(scores[kept_array])[::-1]]
    if max_detections > 0:
        kept_array = kept_array[:max_detections]
    return boxes[kept_array], classes[kept_array], scores[kept_array]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--limit", type=int, help="evaluate only the first N sorted images")
    args = parser.parse_args()

    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be greater than zero")
        image_paths = image_paths[:args.limit]
    runtime = RKNNLite(verbose=False)
    ret = runtime.load_rknn(str(args.model.resolve()))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    records: list[dict[str, object]] = []
    evaluated_image_ids: list[int] = []
    try:
        for number, image_path in enumerate(image_paths, start=1):
            source = cv2.imread(str(image_path))
            if source is None:
                continue
            padded, ratio, (pad_x, pad_y) = letterbox(source)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            # RKNNLite on this runtime requires the explicit batch dimension.
            outputs = runtime.inference(inputs=[rgb[np.newaxis, ...]])
            if not outputs:
                raise RuntimeError(f"inference returned no outputs for {image_path}")
            boxes, classes, scores = postprocess(
                outputs,
                args.confidence,
                args.nms,
                args.max_detections,
            )
            if boxes.size:
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, source.shape[1])
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, source.shape[0])
            image_id = int(image_path.stem)
            evaluated_image_ids.append(image_id)
            for box, class_id, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.tolist()
                records.append({
                    "image_id": image_id,
                    "category_id": COCO_CATEGORY_IDS[int(class_id)],
                    "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                    "score": round(float(score), 6),
                })
            if number % 100 == 0 or number == len(image_paths):
                print(f"evaluated {number}/{len(image_paths)} images, predictions={len(records)}", flush=True)
    finally:
        runtime.release()

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    if args.annotations:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        ground_truth = COCO(str(args.annotations))
        detections = ground_truth.loadRes(str(args.predictions))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = evaluated_image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = {
            "model": str(args.model.resolve()),
            "images": len(image_paths),
            "predictions": len(records),
            "confidence": args.confidence,
            "nms": args.nms,
            "max_detections": args.max_detections,
            "map_50_95": float(evaluator.stats[0]),
            "map_50": float(evaluator.stats[1]),
            "map_75": float(evaluator.stats[2]),
            "map_small": float(evaluator.stats[3]),
            "map_medium": float(evaluator.stats[4]),
            "map_large": float(evaluator.stats[5]),
            "mar_100": float(evaluator.stats[8]),
        }
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
