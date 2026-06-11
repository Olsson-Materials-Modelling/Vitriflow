#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XML="${1:-$ROOT/demos/hardcarbon_gap20ugr/potentials/Carbon_GAP_20U+gr.xml}"
LMP_CMD="${LMP_CMD:-lmp_quip}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python "$ROOT/scripts/check_gap20ugr_potential_files.py" "$XML"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$XML" "$TMP/"
for f in "$(dirname "$XML")"/Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_889{1,2,3}; do
  cp "$f" "$TMP/"
done
cat > "$TMP/in.gap20ugr_smoke" <<'LMP'
units metal
atom_style atomic
boundary p p p
lattice diamond 3.567
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 12.011
pair_style quip
pair_coeff * * Carbon_GAP_20U+gr.xml "Potential xml_label=GAP_2022_11_4_0_14_40_15_889" 6
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step temp pe etotal press
thermo 1
run 0
LMP
( cd "$TMP" && "$LMP_CMD" -in in.gap20ugr_smoke )
echo "OK: LAMMPS/QUIP/GAP-20U+gr run-0 smoke test completed."
