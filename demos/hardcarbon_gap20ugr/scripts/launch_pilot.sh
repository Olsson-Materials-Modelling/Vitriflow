#!/usr/bin/env bash
set -euo pipefail
NPROCS=${1:-8}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
python "$ROOT/scripts/check_gap20ugr_potential_files.py" "$ROOT/potentials/Carbon_GAP_20U+gr.xml"
python "$ROOT/scripts/verify_vitriflow_custom_schedule.py"
CFG="$ROOT/generated_configs/pilot_N128_rho130_${NPROCS}c.yaml"
"$ROOT/scripts/_make_cfg.sh" "$ROOT/configs/hc_C_GAP20Ugr_hc_custom_pilot_N128.yaml" "$CFG" 1.3 "$NPROCS" 501
vitriflow run-schedule -c "$CFG" -o "$ROOT/runs/pilot_hc_custom_N128_rho130_${NPROCS}c"
