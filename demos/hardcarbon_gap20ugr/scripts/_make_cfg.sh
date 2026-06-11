#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "Usage: $0 TEMPLATE OUTCFG DENSITY NPROCS SEED" >&2
  exit 2
fi
TEMPLATE=$1
OUTCFG=$2
DENSITY=$3
NPROCS=$4
SEED=$5
python - "$TEMPLATE" "$OUTCFG" "$DENSITY" "$NPROCS" "$SEED" <<'PY'
from __future__ import annotations
import sys
from pathlib import Path
import yaml

template, outcfg, density, nprocs, seed = sys.argv[1:]
data = yaml.safe_load(Path(template).read_text())
data['lammps']['nprocs'] = int(nprocs)
data['structure']['generate']['packing_density_g_cm3'] = float(density)
data['structure']['generate']['seed'] = int(seed)
data['random_seed'] = int(seed) + 1729
Path(outcfg).parent.mkdir(parents=True, exist_ok=True)
Path(outcfg).write_text(yaml.safe_dump(data, sort_keys=False))
print(outcfg)
PY
