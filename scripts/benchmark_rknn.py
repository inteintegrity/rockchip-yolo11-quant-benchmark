#!/usr/bin/env python3
"""Measure RKNN-only latency and process memory on an RK3576 board."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from rknnlite.api import RKNNLite


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, index))]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def temperatures() -> dict[str, float]:
    result: dict[str, float] = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        name = read_text(str(zone / "type"))
        value = read_text(str(zone / "temp"))
        if name and value:
            try:
                result[name] = int(value) / 1000.0
            except ValueError:
                pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.model = args.model.resolve()

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be >= 0 and iterations must be > 0")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")

    process = psutil.Process(os.getpid())
    rss_initial = process.memory_info().rss
    available_initial = psutil.virtual_memory().available
    temp_start = temperatures()
    npu_frequency_start = read_text("/sys/class/devfreq/27700000.npu/cur_freq")

    rng = np.random.default_rng(3576)
    input_tensor = rng.integers(0, 256, size=(1, 640, 640, 3), dtype=np.uint8)
    runtime = RKNNLite(verbose=False)

    load_started = time.perf_counter()
    ret = runtime.load_rknn(str(args.model))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    load_seconds = time.perf_counter() - load_started
    rss_after_load = process.memory_info().rss

    init_started = time.perf_counter()
    ret = runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    init_seconds = time.perf_counter() - init_started
    rss_after_init = process.memory_info().rss

    rss_samples: list[int] = []
    available_samples: list[int] = []
    stop_sample = threading.Event()

    def sample_memory() -> None:
        while not stop_sample.is_set():
            rss_samples.append(process.memory_info().rss)
            available_samples.append(psutil.virtual_memory().available)
            stop_sample.wait(0.005)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        for _ in range(args.warmup):
            if not runtime.inference(inputs=[input_tensor]):
                raise RuntimeError("warmup inference returned no outputs")

        times_ms: list[float] = []
        outputs = None
        measured_started = time.perf_counter()
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            outputs = runtime.inference(inputs=[input_tensor])
            times_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            if not outputs:
                raise RuntimeError("inference returned no outputs")
        measured_wall_seconds = time.perf_counter() - measured_started
    finally:
        stop_sample.set()
        sampler.join(timeout=1)
        runtime.release()

    assert outputs is not None
    mean_ms = statistics.fmean(times_ms)
    result = {
        "precision": args.precision,
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "model_sha256": sha256(args.model),
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "core_mask": "NPU_CORE_0_1",
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "load_seconds": load_seconds,
        "init_seconds": init_seconds,
        "measured_wall_seconds": measured_wall_seconds,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "p95_ms": percentile(times_ms, 0.95),
        "stddev_ms": statistics.pstdev(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "fps_from_mean": 1000.0 / mean_ms,
        "rss_initial_bytes": rss_initial,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_init_bytes": rss_after_init,
        "rss_peak_bytes": max(rss_samples, default=rss_after_init),
        "system_available_initial_bytes": available_initial,
        "system_available_min_bytes": min(available_samples, default=available_initial),
        "output_shapes": [list(np.asarray(item).shape) for item in outputs],
        "output_dtypes": [str(np.asarray(item).dtype) for item in outputs],
        "output_checksum": float(sum(np.asarray(item, dtype=np.float64).sum() for item in outputs)),
        "temperatures_start_c": temp_start,
        "temperatures_end_c": temperatures(),
        "npu_frequency_start_hz": npu_frequency_start,
        "npu_frequency_end_hz": read_text("/sys/class/devfreq/27700000.npu/cur_freq"),
        "rknn_toolkit_lite2": importlib.metadata.version("rknn-toolkit-lite2"),
        "python": platform.python_version(),
        "kernel": platform.release(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
