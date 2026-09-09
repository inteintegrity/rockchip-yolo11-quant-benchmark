#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${ROOT_DIR}/results/formal_w4_gdq_group128"
MODEL="${ROOT_DIR}/models/yolo11n_w4a16_gdq_group128.rknn"
NPU_DIR="/sys/class/devfreq/27700000.npu"
DMC_DIR="/sys/class/devfreq/dmc"

mkdir -p "${RESULT_DIR}"

read_sysfs() { cat "$1"; }
write_sysfs() { printf '%s' "$2" | sudo tee "$1" >/dev/null; }

CPU0_GOV="$(read_sysfs /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
CPU4_GOV="$(read_sysfs /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"
NPU_GOV="$(read_sysfs "${NPU_DIR}/governor")"
NPU_MIN="$(read_sysfs "${NPU_DIR}/min_freq")"
NPU_MAX="$(read_sysfs "${NPU_DIR}/max_freq")"
DMC_GOV="$(read_sysfs "${DMC_DIR}/governor")"
DMC_MIN="$(read_sysfs "${DMC_DIR}/min_freq")"
DMC_MAX="$(read_sysfs "${DMC_DIR}/max_freq")"

restore_settings() {
  set +e
  write_sysfs /sys/devices/system/cpu/cpufreq/policy0/scaling_governor "${CPU0_GOV}"
  write_sysfs /sys/devices/system/cpu/cpufreq/policy4/scaling_governor "${CPU4_GOV}"
  write_sysfs "${NPU_DIR}/min_freq" "${NPU_MIN}"
  write_sysfs "${NPU_DIR}/max_freq" "${NPU_MAX}"
  write_sysfs "${NPU_DIR}/governor" "${NPU_GOV}"
  write_sysfs "${DMC_DIR}/min_freq" "${DMC_MIN}"
  write_sysfs "${DMC_DIR}/max_freq" "${DMC_MAX}"
  write_sysfs "${DMC_DIR}/governor" "${DMC_GOV}"
}
trap restore_settings EXIT INT TERM

write_sysfs /sys/devices/system/cpu/cpufreq/policy0/scaling_governor performance
write_sysfs /sys/devices/system/cpu/cpufreq/policy4/scaling_governor performance
write_sysfs "${NPU_DIR}/min_freq" 950000000
write_sysfs "${NPU_DIR}/max_freq" 950000000
write_sysfs "${NPU_DIR}/governor" performance
write_sysfs "${DMC_DIR}/min_freq" 2736000000
write_sysfs "${DMC_DIR}/max_freq" 2736000000
write_sysfs "${DMC_DIR}/governor" performance

for repeat in 1 2 3; do
  python "${ROOT_DIR}/scripts/benchmark_rknn.py" \
    --model "${MODEL}" \
    --precision w4a16-gdq-group128 \
    --warmup 50 \
    --iterations 500 \
    --output "${RESULT_DIR}/w4a16_gdq_group128_r${repeat}.json"
  sleep 3
done

python - <<'PY' "${RESULT_DIR}"
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
summary = {
    "runs": len(rows),
    "mean_ms": statistics.fmean(row["mean_ms"] for row in rows),
    "median_ms_mean": statistics.fmean(row["median_ms"] for row in rows),
    "p95_ms_mean": statistics.fmean(row["p95_ms"] for row in rows),
    "fps": statistics.fmean(row["fps_from_mean"] for row in rows),
    "rss_after_init_mib": statistics.fmean(row["rss_after_init_bytes"] for row in rows) / 2**20,
    "rss_peak_mib": statistics.fmean(row["rss_peak_bytes"] for row in rows) / 2**20,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
