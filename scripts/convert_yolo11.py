#!/usr/bin/env python3
"""Convert one YOLO11 ONNX graph to a reproducible RK3576 RKNN variant."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from rknn.api import RKNN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(ret: int, action: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{action} failed with RKNN return code {ret}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "int8", "w4a16"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--target", default="rk3576")
    parser.add_argument("--algorithm", choices=("normal", "mmse", "kl_divergence", "gdq"), default="normal")
    parser.add_argument("--method", default="channel", help="layer, channel, or group32..group256")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.onnx = args.onnx.resolve()
    args.output = args.output.resolve()
    if not args.onnx.is_file():
        parser.error(f"ONNX model not found: {args.onnx}")

    do_quantization = args.precision != "fp16"
    if do_quantization:
        if args.dataset is None:
            parser.error("--dataset is required for INT8 and W4A16")
        args.dataset = args.dataset.resolve()
        if not args.dataset.is_file():
            parser.error(f"dataset list not found: {args.dataset}")

    quantized_dtype = {"int8": "w8a8", "w4a16": "w4a16"}.get(args.precision)
    config = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": args.target,
        "float_dtype": "float16",
        "optimization_level": 3,
        "custom_string": f"YOLO11n {args.precision} reproducible benchmark",
    }
    if quantized_dtype:
        config.update(
            quantized_dtype=quantized_dtype,
            quantized_algorithm=args.algorithm,
            quantized_method=args.method,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rknn = RKNN(verbose=args.verbose)
    try:
        checked(rknn.config(**config), "config")
        checked(rknn.load_onnx(model=str(args.onnx)), "load_onnx")
        checked(
            rknn.build(
                do_quantization=do_quantization,
                dataset=str(args.dataset) if do_quantization else None,
            ),
            "build",
        )
        checked(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()

    metadata = {
        "precision": args.precision,
        "target": args.target,
        "quantized_dtype": quantized_dtype,
        "quantized_algorithm": args.algorithm if do_quantization else None,
        "quantized_method": args.method if do_quantization else None,
        "do_quantization": do_quantization,
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "onnx": str(args.onnx),
        "onnx_bytes": args.onnx.stat().st_size,
        "onnx_sha256": sha256(args.onnx),
        "dataset": str(args.dataset) if do_quantization else None,
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
        "conversion_seconds": time.time() - started,
        "rknn_toolkit2": importlib.metadata.version("rknn-toolkit2"),
        "python": platform.python_version(),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
