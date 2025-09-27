#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_linear.sh [-noanal] [experiment_name]
# Example: ./run_linear.sh -noanal linear_p2_64
# If experiment_name is not provided, defaults to "linear_p2_64".
# When -noanal is passed, the analyze step is skipped.

# Parse arguments
NO_ANALYZE=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "-na" ] || [ "$arg" = "--noanal" ]; then
    NO_ANALYZE=1
  else
    ARGS+=("$arg")
  fi
done

EXP_NAME="${ARGS[0]:-linear_p2_64}"

PROJECT_ROOT="/home/jianan/mirage"
TEST_DIR="${PROJECT_ROOT}/tests_cu"
OUT_BIN="test_${EXP_NAME}"
OUT_PATH="${TEST_DIR}/${OUT_BIN}"
REPORT_DIR="${PROJECT_ROOT}/report"
REPORT_PATH="${REPORT_DIR}/${EXP_NAME}.ncu-rep"
ANALYZE_SCRIPT="${PROJECT_ROOT}/analyze.sh"
CUDA_VISIBLE_DEVICES=7

mkdir -p "${REPORT_DIR}"

echo "[Build] make -C ${TEST_DIR} test_linear OUT=${OUT_BIN}"
make -C "${TEST_DIR}" test_linear "OUT=${OUT_BIN}"

echo "[Run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${OUT_PATH}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${OUT_PATH}"

if [ "${NO_ANALYZE}" -eq 0 ]; then
echo "[Analyze] ${ANALYZE_SCRIPT} ${REPORT_PATH} linear_kernel_launcher ${OUT_PATH}"
"${ANALYZE_SCRIPT}" "${REPORT_PATH}" linear_kernel_launcher "${OUT_PATH}"
echo "Done. Report: ${REPORT_PATH}"
else
echo "[Analyze] skipped (-noanal)"
echo "Done."
fi
