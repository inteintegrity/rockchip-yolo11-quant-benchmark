#!/usr/bin/env bash
set -euo pipefail

# Run three repeats in a rotated order to reduce thermal/order bias. The script
# restores all governors and frequency limits even when interrupted.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${ROOT_DIR}/results/formal"
NPU_DIR="/sys/class/devfreq/27700000.npu"
DMC_DIR="/sys/class/devfreq/dmc"

mkdir -p "${RESULT_DIR}"

read_sysfs() {
  cat "$1"
}

write_sysfs() {
  printf '%s' "$2" | sudo tee "$1" >/dev/null
}

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

run_one() {
  local precision="$1"
  local repeat="$2"
  python "${ROOT_DIR}/scripts/benchmark_rknn.py" \
    --model "${ROOT_DIR}/models/yolo11n_${precision}.rknn" \
    --precision "${precision}" \
    --warmup 50 \
    --iterations 500 \
    --output "${RESULT_DIR}/${precision}_r${repeat}.json"
  sleep 3
}

run_one fp16 1
run_one int8 1
run_one w4a16 1

run_one w4a16 2
run_one fp16 2
run_one int8 2

run_one int8 3
run_one w4a16 3
run_one fp16 3

python - <<'PY' "${RESULT_DIR}"
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
for precision in ("fp16", "int8", "w4a16"):
    rows = [json.loads(path.read_text()) for path in sorted(root.glob(f"{precision}_r*.json"))]
    print(
        precision,
        "runs=", len(rows),
        "mean_ms=", round(statistics.fmean(row["mean_ms"] for row in rows), 4),
        "fps=", round(statistics.fmean(row["fps_from_mean"] for row in rows), 4),
        "rss_peak_mib=", round(statistics.fmean(row["rss_peak_bytes"] for row in rows) / 2**20, 2),
    )
PY
