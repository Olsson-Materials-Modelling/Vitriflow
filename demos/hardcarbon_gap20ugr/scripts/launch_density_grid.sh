#!/usr/bin/env bash
set -euo pipefail
NPROCS=${1:-16}
shift || true
if [[ $# -gt 0 ]]; then
  DENSITIES=("$@")
else
  DENSITIES=(0.7 1.0 1.3 1.6 1.9)
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
python "$ROOT/scripts/check_gap20ugr_potential_files.py" "$ROOT/potentials/Carbon_GAP_20U+gr.xml"
python "$ROOT/scripts/verify_vitriflow_custom_schedule.py"
for rho in "${DENSITIES[@]}"; do
  rho_tag=$(python - "$rho" <<'PY'
import sys
rho=float(sys.argv[1])
print(f"rho{rho:.2f}".replace('.', 'p'))
PY
)
  # density-dependent seed, deterministic but distinct
  seed=$(python - "$rho" <<'PY'
import sys
rho=float(sys.argv[1])
print(100000 + int(round(rho*1000)))
PY
)
  CFG="$ROOT/generated_configs/hc_custom_N1024_${rho_tag}_${NPROCS}c.yaml"
  "$ROOT/scripts/_make_cfg.sh" "$ROOT/configs/hc_C_GAP20Ugr_hc_custom_N1024.yaml" "$CFG" "$rho" "$NPROCS" "$seed"
  vitriflow run-schedule -c "$CFG" -o "$ROOT/runs/hc_custom_N1024_${rho_tag}_${NPROCS}c"
done
