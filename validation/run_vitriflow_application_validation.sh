#!/usr/bin/env bash
# Two-pass, application-facing Vitriflow numerical validation.
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
CHECKER="${SCRIPT_DIR}/tools/validate_results.py"

PYTHON_BIN="${PYTHON:-python3}"
VITRIFLOW_BIN="${VITRIFLOW:-vitriflow}"
WORK_ROOT="${PWD}/vitriflow_application_validation_runs"
WITH_CP2K=0
WITH_CP2K_CELL_OPT=0
WITH_SLURM=0
SLURM_TEMPLATE="${SCRIPT_DIR}/slurm/validation_job.slurm"
SLURM_TIMEOUT_SEC=14400
SKIP_INTERFACE_AUDIT=0
PARTIAL_SELECTION=0
ONLY_CASES=()
EXPECTED_VERSION="0.4.37.0"
SEQUENTIAL=0
JOBS=0   # 0 = auto (all selected cases run concurrently within each pass)

usage() {
  sed -n '2,45p' "$0" | sed -n 's/^# \{0,1\}//p'
  cat <<'EOF'

Usage:
  ./run_vitriflow_application_validation.sh [options]

Options:
  --with-cp2k             Include the optional, much heavier 64-atom CP2K Si case.
  --with-cp2k-cell-opt    Include CP2K and exercise CELL_OPT refinement for every
                          accepted Si production box before convergence analysis.
  --with-slurm            Replay each selected non-refined production plan through
                          Slurm task materialisation, submission, collection, and analysis.
  --slurm-template FILE   Site Slurm template (default: bundled site-neutral template).
                          It must contain {{EXECUTE_CMD}}.
  --slurm-timeout SEC     Maximum wait for each submitted ten-box Slurm set
                          (default: 14400 seconds).
  --workdir DIR           Fresh output directory (default: ./vitriflow_application_validation_runs).
  --vitriflow PATH        Vitriflow executable (default: $VITRIFLOW or vitriflow).
  --python PATH           Python from the same environment (default: $PYTHON or python3).
  --only CASE             Run only one named case; repeatable. Cases: minimal_metal,
                          sio2_bks, sio2_kim, si_cp2k. Any --only run is
                          diagnostic-only and cannot issue release sign-off.
  --skip-interface-audit  Developer escape hatch: skip cheap installed-interface
                          contract checks. Never use for release sign-off.
  --jobs N                Run up to N selected cases concurrently within each pass
                          (default: all selected cases in parallel). Cases are fully
                          independent, so parallel is the throughput default and does
                          not affect determinism or release-signoff validity.
  --sequential            Run cases one at a time (equivalent to --jobs 1).
  -h, --help              Show this help.

The output directory must not already contain files. The script runs reference
then comparison passes with identical configs and seeds, exercises every
applicable analysis/plot CLI, and requires exact equality after normalising
the two run-root path prefixes, their derived artifact hashes, and PDF date
metadata. Finite-size configuration is omitted and the default-disabled stage
is not run or plotted. CP2K is excluded unless requested.
EOF
}

while (($#)); do
  case "$1" in
    --with-cp2k)
      WITH_CP2K=1
      shift
      ;;
    --with-cp2k-cell-opt)
      WITH_CP2K=1
      WITH_CP2K_CELL_OPT=1
      shift
      ;;
    --with-slurm)
      WITH_SLURM=1
      shift
      ;;
    --slurm-template)
      [[ $# -ge 2 ]] || { echo "--slurm-template requires a value" >&2; exit 64; }
      SLURM_TEMPLATE="$2"
      shift 2
      ;;
    --slurm-timeout)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] \
        || { echo "--slurm-timeout requires a positive integer number of seconds" >&2; exit 64; }
      SLURM_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --workdir)
      [[ $# -ge 2 ]] || { echo "--workdir requires a value" >&2; exit 64; }
      WORK_ROOT="$2"
      shift 2
      ;;
    --vitriflow)
      [[ $# -ge 2 ]] || { echo "--vitriflow requires a value" >&2; exit 64; }
      VITRIFLOW_BIN="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 64; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --only)
      [[ $# -ge 2 ]] || { echo "--only requires a value" >&2; exit 64; }
      ONLY_CASES+=("$2")
      PARTIAL_SELECTION=1
      [[ "$2" == "si_cp2k" ]] && WITH_CP2K=1
      shift 2
      ;;
    --skip-interface-audit)
      SKIP_INTERFACE_AUDIT=1
      shift
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs requires a value" >&2; exit 64; }
      JOBS="$2"
      [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 64; }
      shift 2
      ;;
    --sequential)
      SEQUENTIAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

case "$WORK_ROOT" in
  /*) ;;
  *) WORK_ROOT="${PWD}/${WORK_ROOT}" ;;
esac
case "$SLURM_TEMPLATE" in
  /*) ;;
  *) SLURM_TEMPLATE="${PWD}/${SLURM_TEMPLATE}" ;;
esac

export LC_ALL=C
export LANG=C
export TZ=UTC
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH=1262304000   # reproducible matplotlib PDF /CreationDate
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vitriflow-validation-mpl-${UID:-0}-$$"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"

# --- MPI core-binding shim for LAMMPS ONLY -------------------------------------
# OpenMPI 5.x binds ranks to cores from core 0 by default, so the many concurrent
# `mpirun -np N` from the parallel fan-out pile onto the same low cores and throttle
# each other (~30-50% CPU) while higher cores idle. LAMMPS `mpi_cmd` must be a single
# token (runner validates it), so we shim `mpirun` on PATH to inject `--bind-to none`
# — but ONLY for LAMMPS. CRITICAL: CP2K's cp2k.psmp links a DIFFERENT OpenMPI (its env
# is 4.1.6; the LAMMPS env is 5.0.10) and conda-run resolves this shim first, so the
# shim must NOT launch cp2k.psmp with the LAMMPS mpirun — that ABI/PMIx mismatch makes
# every rank go singleton and 4x-duplicates CP2K's .ener. For anything that is not
# `lmp` (CP2K, `mpirun --version`, ...) the shim removes itself from PATH and
# re-resolves `mpirun`, yielding each environment's OWN launcher (vitriflow-cp2k 4.1.6
# for CP2K; the LAMMPS 5.0.10 outside conda-run). Verified: LAMMPS ranks spread to
# allowed=[0-23]; CP2K wires to size=4 under its own mpirun.
_MPIRUN_REAL="$(command -v mpirun 2>/dev/null || true)"
if [[ -n "$_MPIRUN_REAL" ]]; then
  _MPI_SHIM_DIR="${MPLCONFIGDIR}/mpi-bind-shim"
  mkdir -p "$_MPI_SHIM_DIR"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'SHIM_DIR=%q\n' "$_MPI_SHIM_DIR"
    printf 'REAL=%q\n' "$_MPIRUN_REAL"
    cat <<'SHIM'
for a in "$@"; do
  case "$a" in
    lmp|lmp_mpi|lmp_serial|*/lmp|*/lmp_mpi|*/lmp_serial)
      exec "$REAL" --bind-to none "$@" ;;
  esac
done
# not LAMMPS (e.g. CP2K, --version): defer to the environment's own mpirun
p=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$SHIM_DIR" | paste -sd: -)
exec env PATH="$p" mpirun "$@"
SHIM
  } > "$_MPI_SHIM_DIR/mpirun"
  chmod +x "$_MPI_SHIM_DIR/mpirun"
  export PATH="${_MPI_SHIM_DIR}:${PATH}"
fi

die() {
  echo "VALIDATION FAILED: $*" >&2
  exit 1
}

on_error() {
  local status=$?
  echo "VALIDATION FAILED at line ${BASH_LINENO[0]} (exit ${status})." >&2
  exit "$status"
}
trap on_error ERR

ALL_CASES=(minimal_metal sio2_bks sio2_kim)
if ((WITH_CP2K)); then
  ALL_CASES+=(si_cp2k)
fi

if ((${#ONLY_CASES[@]})); then
  SELECTED=()
  for requested in "${ONLY_CASES[@]}"; do
    case "$requested" in
      minimal_metal|sio2_bks|sio2_kim|si_cp2k) ;;
      *) die "unknown --only case: $requested" ;;
    esac
    SELECTED+=("$requested")
  done
  ALL_CASES=("${SELECTED[@]}")
fi

for ((i = 0; i < ${#ALL_CASES[@]}; i++)); do
  for ((j = i + 1; j < ${#ALL_CASES[@]}; j++)); do
    [[ "${ALL_CASES[i]}" != "${ALL_CASES[j]}" ]] || die "duplicate --only case: ${ALL_CASES[i]}"
  done
done

SELECTED_LAMMPS=0
SELECTED_CP2K=0
for case_name in "${ALL_CASES[@]}"; do
  if [[ "$case_name" == "si_cp2k" ]]; then
    SELECTED_CP2K=1
  else
    SELECTED_LAMMPS=1
  fi
done

if ((WITH_CP2K_CELL_OPT && SELECTED_CP2K == 0)); then
  die "--with-cp2k-cell-opt requires the si_cp2k case (do not exclude it with --only)"
fi

CP2K_CONFIG="${CONFIG_DIR}/si_cp2k_validation.yaml"

config_for() {
  case "$1" in
    minimal_metal) echo "${CONFIG_DIR}/minimal_metal_validation.yaml" ;;
    sio2_bks) echo "${CONFIG_DIR}/sio2_bks_validation.yaml" ;;
    sio2_kim) echo "${CONFIG_DIR}/sio2_kim_validation.yaml" ;;
    si_cp2k) echo "${CP2K_CONFIG}" ;;
    *) return 2 ;;
  esac
}

engine_for() {
  [[ "$1" == "si_cp2k" ]] && echo cp2k || echo lammps
}

graph_cutoffs_for() {
  case "$1" in
    minimal_metal) echo "3.5 3.3 3.7" ;;
    sio2_bks|sio2_kim) echo "2.2 2.0 2.4" ;;
    si_cp2k) echo "3.0 2.8 3.2" ;;
    *) return 2 ;;
  esac
}

check_config_case() {
  local case_name=$1
  local config=$2
  local extra=()
  if [[ "$case_name" == "si_cp2k" ]] && ((WITH_CP2K_CELL_OPT)); then
    extra+=(--expect-cell-refinement)
  fi
  "$PYTHON_BIN" "$CHECKER" check-config --case "$case_name" --config "$config" "${extra[@]}"
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
command -v "$VITRIFLOW_BIN" >/dev/null 2>&1 || die "Vitriflow executable not found: $VITRIFLOW_BIN"
command -v mpirun >/dev/null 2>&1 || die "MPI launcher 'mpirun' is required"
[[ -f "$CHECKER" ]] || die "result validator is missing: $CHECKER"
if ((WITH_SLURM)); then
  command -v sbatch >/dev/null 2>&1 || die "--with-slurm requires the Slurm 'sbatch' command"
  [[ -f "$SLURM_TEMPLATE" ]] || die "Slurm template is missing: $SLURM_TEMPLATE"
  grep -Fq '{{EXECUTE_CMD}}' "$SLURM_TEMPLATE" \
    || die "Slurm template must contain the literal {{EXECUTE_CMD}} placeholder"
  if ((WITH_CP2K_CELL_OPT && SELECTED_LAMMPS == 0)); then
    die "Slurm replay and CP2K CELL_OPT are separate modes; select a non-refined case for --with-slurm"
  fi
fi

VITRIFLOW_RESOLVED="$(command -v "$VITRIFLOW_BIN")"
"$PYTHON_BIN" - "$VITRIFLOW_RESOLVED" "$EXPECTED_VERSION" <<'PY'
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

launcher = Path(sys.argv[1]).resolve()
expected = sys.argv[2]
first = launcher.open("rb").readline().decode("utf-8", "replace").strip()
if not first.startswith("#!"):
    raise SystemExit(f"Vitriflow launcher has no Python shebang: {launcher}")
parts = shlex.split(first[2:].strip())
if not parts:
    raise SystemExit(f"Vitriflow launcher has an empty shebang: {launcher}")
if Path(parts[0]).name == "env":
    names = [value for value in parts[1:] if not value.startswith("-")]
    if not names:
        raise SystemExit(f"Cannot resolve env-based Vitriflow shebang: {first}")
    executable = shutil.which(names[0])
else:
    executable = parts[0]
if executable is None:
    raise SystemExit(f"Cannot resolve Vitriflow shebang interpreter: {first}")
if os.path.realpath(executable) != os.path.realpath(sys.executable):
    raise SystemExit(
        "--python and --vitriflow use different interpreters: "
        f"{sys.executable!r} != {executable!r}"
    )
# Match the import search-path head used when the launcher itself executes;
# this prevents a same-version source tree in the caller's CWD shadowing the
# package that the launcher would import.
sys.path[0] = str(launcher.parent)
import vitriflow
if str(vitriflow.__version__) != expected:
    raise SystemExit(f"Expected Vitriflow {expected}, imported {vitriflow.__version__}")
print(json.dumps({
    "version": str(vitriflow.__version__),
    "python": os.path.realpath(sys.executable),
    "launcher": str(launcher),
    "package": str(Path(vitriflow.__file__).resolve()),
}, indent=2, sort_keys=True))
PY

VITRIFLOW_PACKAGE_FILE="$("$PYTHON_BIN" - "$VITRIFLOW_RESOLVED" <<'PY'
import sys
from pathlib import Path

launcher = Path(sys.argv[1]).resolve()
sys.path[0] = str(launcher.parent)
import vitriflow
print(Path(vitriflow.__file__).resolve())
PY
)"
[[ -f "$VITRIFLOW_PACKAGE_FILE" ]] || die "cannot identify the package imported by the Vitriflow launcher"
export VITRIFLOW_VALIDATION_EXPECTED_PACKAGE="$VITRIFLOW_PACKAGE_FILE"

[[ "$("$VITRIFLOW_BIN" --version)" == "vitriflow ${EXPECTED_VERSION}" ]] \
  || die "Vitriflow CLI version does not match ${EXPECTED_VERSION}"

if ((WITH_CP2K_CELL_OPT)); then
  EFFECTIVE_CONFIG_DIR="${MPLCONFIGDIR}/effective-configs"
  mkdir -p "$EFFECTIVE_CONFIG_DIR"
  CP2K_CONFIG="${EFFECTIVE_CONFIG_DIR}/si_cp2k_cell_refinement_validation.yaml"
  "$PYTHON_BIN" - "${CONFIG_DIR}/si_cp2k_validation.yaml" "$CP2K_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
data = yaml.safe_load(source.read_text(encoding="utf-8"))
production = data.setdefault("autotune", {}).setdefault("production", {})
production["dft_opt"] = {
    "enabled": True,
    "optimizer": "LBFGS",
    "max_iter": 200,
    "keep_angles": True,
    "external_pressure_bar": 1.0,
    "traj_every": 1,
    "print_level": "LOW",
}
destination.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY
fi

for case_name in "${ALL_CASES[@]}"; do
  config="$(config_for "$case_name")"
  [[ -f "$config" ]] || die "configuration missing: $config"
  check_config_case "$case_name" "$config"
  if [[ "$(engine_for "$case_name")" == "lammps" ]]; then
    command -v lmp >/dev/null 2>&1 || die "LAMMPS executable 'lmp' is required"
  else
    command -v cp2k.psmp >/dev/null 2>&1 || die "CP2K case selected but cp2k.psmp is unavailable"
  fi
done

if [[ -e "$WORK_ROOT" ]] && [[ -n "$(find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  die "work directory is not empty; choose a fresh --workdir: $WORK_ROOT"
fi

if ((SKIP_INTERFACE_AUDIT == 0)); then
  "$PYTHON_BIN" "$CHECKER" audit-code
else
  echo "WARNING: interface source audit skipped; this run is not release-signoff valid." >&2
fi

KIM_MODELS=()
for case_name in "${ALL_CASES[@]}"; do
  case "$case_name" in
    minimal_metal) KIM_MODELS+=("EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005") ;;
    sio2_kim) KIM_MODELS+=("Sim_LAMMPS_Buckingham_CarreHorbachIspas_2008_SiO__SM_886641404623_000") ;;
  esac
done
if ((${#KIM_MODELS[@]})); then
  "$PYTHON_BIN" - "${KIM_MODELS[@]}" <<'PY'
import json
import sys
from vitriflow.kim import ensure_model_installed

rows = []
for model in dict.fromkeys(sys.argv[1:]):
    result = ensure_model_installed(model)
    if not result.success:
        raise SystemExit(f"Unable to prepare required KIM model {model}: {result.stderr or result.stdout}")
    rows.append({"model": model, "ready": True})
print(json.dumps({"kim_models": rows}, indent=2, sort_keys=True))
PY
fi

"$VITRIFLOW_BIN" --help >/dev/null

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/config_contracts" "$WORK_ROOT/comparison_reports" "$WORK_ROOT/comparison_plots"

write_environment_contract() {
  local destination=$1
  "$PYTHON_BIN" - "$destination" "$VITRIFLOW_RESOLVED" "$SELECTED_LAMMPS" "$SELECTED_CP2K" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

destination = Path(sys.argv[1])
vitriflow_launcher = Path(sys.argv[2]).resolve()
with_lammps = bool(int(sys.argv[3]))
with_cp2k = bool(int(sys.argv[4]))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def executable_identity(name: str, version_args: list[str]) -> dict[str, object]:
    selected = shutil.which(name)
    if selected is None:
        raise SystemExit(f"required executable disappeared: {name}")
    path = Path(selected).resolve()
    completed = subprocess.run(
        [str(path), *version_args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"could not identify {name}: exit={completed.returncode}\n{completed.stdout}"
        )
    return {
        "requested_name": name,
        "resolved_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "version_output": completed.stdout.replace("\r\n", "\n"),
    }

sys.path[0] = str(vitriflow_launcher.parent)
from vitriflow.runtime_identity import runtime_identity

executables = {
    "python": {
        "resolved_path": os.path.realpath(sys.executable),
        "size_bytes": Path(sys.executable).resolve().stat().st_size,
        "sha256": sha256(Path(sys.executable).resolve()),
        "version": sys.version,
    },
    "vitriflow_launcher": {
        "resolved_path": str(vitriflow_launcher),
        "size_bytes": vitriflow_launcher.stat().st_size,
        "sha256": sha256(vitriflow_launcher),
    },
    "mpirun": executable_identity("mpirun", ["--version"]),
}
if with_lammps:
    executables["lmp"] = executable_identity("lmp", ["-h"])
if with_cp2k:
    executables["cp2k.psmp"] = executable_identity("cp2k.psmp", ["--version"])

contract = {
    "schema": "vitriflow.application_validation.environment.v1",
    "vitriflow_runtime": runtime_identity(),
    "executables": executables,
    "critical_dependency_versions": {
        name: importlib.metadata.version(name)
        for name in (
            "ase", "matplotlib", "networkx", "numpy", "pydantic",
            "PyYAML", "scipy",
        )
    },
}
destination.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_environment_contract "$WORK_ROOT/environment_contract.json"
for case_name in "${ALL_CASES[@]}"; do
  check_config_case "$case_name" "$(config_for "$case_name")" \
    | tee "$WORK_ROOT/config_contracts/${case_name}.json"
done

run_logged() {
  local log_file=$1
  shift
  echo "+ $*"
  "$@" 2>&1 | tee "$log_file"
}

exercise_case() {
  local pass_name=$1
  local case_name=$2
  local config=$3
  local case_root=$4
  local engine=$5
  local result="${case_root}/autotune_results.json"
  # Derived analysis/plots live OUTSIDE case_root so the reference-vs-reproduction
  # `compare` runs on the PURE autotune output. Vitriflow >=0.4.37.0 embeds
  # absolute run-root paths in the graph-analysis structure_reference sidecars,
  # so their recorded byte sizes differ between the reference/comparison roots
  # (path lengths differ) and cannot be normalised by compare_roots.
  local validation="${WORK_ROOT}/validation/${pass_name}/${case_name}"
  local plots="${validation}/plots"
  local analysis_parity="${validation}/analysis_parity"
  local analysis_graph="${validation}/analysis_graph"
  local log_prefix="${WORK_ROOT}/logs/${pass_name}-${case_name}"
  local stage_dir
  local primary_cutoff lower_cutoff upper_cutoff

  mkdir -p "$validation" "$plots"

  local refinement_args=()
  if [[ "$case_name" == "si_cp2k" ]] && ((WITH_CP2K_CELL_OPT)); then
    refinement_args+=(--expect-cell-refinement)
  fi

  "$PYTHON_BIN" "$CHECKER" check-run --case "$case_name" --result "$result" "${refinement_args[@]}" \
    | tee "${validation}/run_contract.json"

  stage_dir="$("$PYTHON_BIN" "$CHECKER" stage-dir --result "$result")"
  read -r primary_cutoff lower_cutoff upper_cutoff <<<"$(graph_cutoffs_for "$case_name")"

  run_logged "${log_prefix}-analysis-parity.log" \
    "$VITRIFLOW_BIN" analyze-output \
      -c "$config" -i "$result" -o "$analysis_parity" \
      --embed-structures --analysis-workers 1

  "$PYTHON_BIN" "$CHECKER" check-parity \
    --case "$case_name" --source "$result" \
    --analysis "${analysis_parity}/analysis_results.json" "${refinement_args[@]}" \
    | tee "${validation}/analysis_parity_contract.json"

  run_logged "${log_prefix}-analysis-graph.log" \
    "$VITRIFLOW_BIN" analyze-output \
      -c "$config" -i "$result" -o "$analysis_graph" \
      --embed-structures --analysis-workers 1 \
      --graph-cutoff "$primary_cutoff" \
      --graph-cutoff-interval "$lower_cutoff" "$upper_cutoff" \
      --graph-interval-points 3 \
      --soft-logistic "$primary_cutoff" 0.1

  "$PYTHON_BIN" "$CHECKER" check-analysis \
    --case "$case_name" --result "${analysis_graph}/analysis_results.json" --boxes 10 \
    | tee "${validation}/analysis_graph_contract.json"

  run_logged "${log_prefix}-plot-analysis-parity-production.log" \
    "$VITRIFLOW_BIN" plot-production -i "${analysis_parity}/analysis_results.json" \
      -o "${plots}/analysis_parity_production_pages" \
      --title "${case_name} exact replay production validation" --show-boxes --dpi 150

  run_logged "${log_prefix}-plot-analysis-graph-production.log" \
    "$VITRIFLOW_BIN" plot-production -i "${analysis_graph}/analysis_results.json" \
      -o "${plots}/analysis_graph_production_pages" \
      --title "${case_name} graph-rule production validation" --show-boxes --dpi 150

  run_logged "${log_prefix}-plot-autotune.log" \
    "$VITRIFLOW_BIN" plot -i "$result" -o "${plots}/autotune.png" \
      --title "${case_name} autotune validation" --show-replicates

  run_logged "${log_prefix}-plot-tm.log" \
    "$VITRIFLOW_BIN" plot-metric -i "$result" -o "${plots}/metric_tm_density.png" \
      --stage tm_scan --metric density --title "${case_name} Tm density" --show-replicates --dpi 150

  run_logged "${log_prefix}-plot-rate.log" \
    "$VITRIFLOW_BIN" plot-metric -i "$result" -o "${plots}/metric_rate_density.png" \
      --stage rate_scan --metric density --title "${case_name} rate density" --show-replicates --dpi 150

  run_logged "${log_prefix}-plot-production-metric.log" \
    "$VITRIFLOW_BIN" plot-metric -i "$result" -o "${plots}/metric_production_density.png" \
      --stage production --metric density --title "${case_name} production density" --dpi 150

  run_logged "${log_prefix}-plot-production.log" \
    "$VITRIFLOW_BIN" plot-production -i "$result" -o "${plots}/production_pages" \
      --title "${case_name} production validation" --show-boxes --dpi 150

  run_logged "${log_prefix}-plot-stage.log" \
    "$VITRIFLOW_BIN" plot-stage -d "$stage_dir" -o "${plots}/stage.png" \
      --results "$result" --all-thermo --title "${case_name} first production relax" --dpi 150

  run_logged "${log_prefix}-metrics-timeseries.log" \
    "$VITRIFLOW_BIN" metrics-timeseries -c "$config" -d "$stage_dir" \
      -o "${validation}/metrics_timeseries_cli.csv" --results "$result" \
      --stride 1 --max-frames 6 --include-gr-curves

  "$PYTHON_BIN" "$CHECKER" check-metrics-csv \
    --input "${validation}/metrics_timeseries_cli.csv" \
    | tee "${validation}/metrics_timeseries_contract.json"

  run_logged "${log_prefix}-plot-metrics.log" \
    "$VITRIFLOW_BIN" plot-metrics -i "${validation}/metrics_timeseries_cli.csv" \
      -o "${plots}/metrics_pages" --title "${case_name} structural metrics" --dpi 150

  run_logged "${log_prefix}-plot-voids.log" \
    "$VITRIFLOW_BIN" plot-voids -c "$config" -d "$stage_dir" \
      -o "${plots}/voids.png" --results "$result" --n-samples 256 --top-n 128 \
      --write-void-extxyz "${validation}/void_points.extxyz" \
      --write-combined-extxyz "${validation}/atoms_and_voids.extxyz" \
      --title "${case_name} void validation" --dpi 150
  [[ -s "${validation}/void_points.extxyz" ]] || die "void point-cloud export is missing"
  [[ -s "${validation}/atoms_and_voids.extxyz" ]] || die "combined atom/void export is missing"

  if [[ "$engine" == "lammps" ]]; then
    run_logged "${log_prefix}-plot-elastic.log" \
      "$VITRIFLOW_BIN" plot-elastic -d "$stage_dir" -o "${plots}/elastic.png" \
        --title "${case_name} elastic validation" --dpi 150
  fi

  "$PYTHON_BIN" "$CHECKER" check-plots --dir "$plots" --engine "$engine" \
    --result "$result" \
    --parity-analysis "${analysis_parity}/analysis_results.json" \
    --graph-analysis "${analysis_graph}/analysis_results.json" \
    --metrics-csv "${validation}/metrics_timeseries_cli.csv" \
    | tee "${validation}/plot_contract.json"
}

exercise_slurm_case() {
  local pass_name=$1
  local case_name=$2
  local config=$3
  local case_root=$4
  local source_result="${case_root}/autotune_results.json"
  # keep derived slurm-replay analysis outside case_root too (see exercise_case)
  local hpc_root="${WORK_ROOT}/validation/${pass_name}/${case_name}/slurm_replay"
  local hpc_analysis="${hpc_root}/validation/analysis_parity"
  local log_prefix="${WORK_ROOT}/logs/${pass_name}-${case_name}-slurm"
  local deadline count now

  if [[ "$case_name" == "si_cp2k" ]] && ((WITH_CP2K_CELL_OPT)); then
    echo "Skipping Slurm replay for ${pass_name}/${case_name}: external CELL_OPT is explicitly unsupported."
    return 0
  fi

  mkdir -p "${hpc_root}/validation"

  run_logged "${log_prefix}-materialize.log" \
    "$VITRIFLOW_BIN" run -c "$config" -o "$hpc_root" \
      --use-autotune "$source_result" --external-mode dry-run \
      --job-template "$SLURM_TEMPLATE" --no-resume

  "$PYTHON_BIN" - "$hpc_root/run_results.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
production = data.get("production", {})
if data.get("status") != "planned" or data.get("execution_status") != "planned":
    raise SystemExit("external dry-run did not produce a planned top-level state")
if production.get("status") != "planned" or production.get("execution_status") != "planned":
    raise SystemExit("external dry-run did not produce a planned production state")
execution = production.get("execution", {})
if execution.get("mode") != "dry-run" or execution.get("planned_boxes") != 10:
    raise SystemExit("external dry-run did not materialise exactly ten boxes")
if production.get("n_boxes_total") != 0:
    raise SystemExit("dry-run incorrectly reported planned tasks as attempted boxes")
PY

  run_logged "${log_prefix}-submit.log" bash "${hpc_root}/production/submit_all.sh"
  deadline=$(( $(date +%s) + SLURM_TIMEOUT_SEC ))
  while true; do
    count="$(find "${hpc_root}/production" -mindepth 2 -maxdepth 2 -type f -name task_result.json | wc -l)"
    if [[ "$count" -eq 10 ]]; then
      break
    fi
    now="$(date +%s)"
    if [[ "$now" -ge "$deadline" ]]; then
      die "timed out waiting for ten Slurm task results for ${pass_name}/${case_name}; found ${count}"
    fi
    sleep 5
  done

  "$PYTHON_BIN" "$CHECKER" check-task-results-ready --root "$hpc_root" --case "$case_name" \
    | tee "${hpc_root}/validation/slurm_task_contract.json"

  run_logged "${log_prefix}-collect.log" \
    "$VITRIFLOW_BIN" run -c "$config" -o "$hpc_root" \
      --use-autotune "$source_result" --external-mode full-run \
      --job-template "$SLURM_TEMPLATE" --max-parallel-boxes 1 --resume

  "$PYTHON_BIN" "$CHECKER" check-hpc \
    --case "$case_name" --result "${hpc_root}/run_results.json" --source "$source_result" \
    | tee "${hpc_root}/validation/slurm_full_run_contract.json"

  run_logged "${log_prefix}-analysis-parity.log" \
    "$VITRIFLOW_BIN" analyze-output -c "$config" \
      -i "${hpc_root}/run_results.json" -o "$hpc_analysis" \
      --embed-structures --analysis-workers 1
  "$PYTHON_BIN" "$CHECKER" check-parity \
    --case "$case_name" --source "${hpc_root}/run_results.json" \
    --analysis "${hpc_analysis}/analysis_results.json" \
    | tee "${hpc_root}/validation/slurm_analysis_parity_contract.json"
}

# ---- parallel case dispatch ---------------------------------------------------
# Cases are fully independent (per-case output/log/validation paths), so they run
# concurrently within each pass. Only matplotlib's MPLCONFIGDIR is shared state, so
# each worker gets its own. Each worker records its exit code to a status file (via an
# EXIT trap that survives the `set -e`/ERR abort), which is then aggregated.
effective_jobs() {   # $1 = number of units to run (default: number of cases)
  local units=${1:-${#ALL_CASES[@]}}
  if ((SEQUENTIAL)); then
    echo 1
  elif ((JOBS > 0)); then
    echo "$JOBS"
  else
    echo "$units"
  fi
}

throttle_jobs() {   # block until fewer than $1 background jobs are running
  local limit=$1
  while (( $(jobs -rp | wc -l) >= limit )); do
    wait -n 2>/dev/null || true
  done
}

collect_case_status() {   # $1=phase label  $2=status dir  rest=case names; dies on any failure
  local phase=$1 status_dir=$2
  shift 2
  local c rc fail=0
  for c in "$@"; do
    rc=$(cat "${status_dir}/${c}.rc" 2>/dev/null || echo 99)
    if ((rc != 0)); then
      echo "[parallel] ${phase}/${c} FAILED (rc=${rc}); last lines of its console log:" >&2
      tail -25 "${WORK_ROOT}/logs/${phase}-${c}.console.log" 2>/dev/null >&2 || true
      fail=1
    fi
  done
  ((fail == 0)) || die "${phase}: one or more cases failed"
}

# Reference and comparison are independent deterministic replays (identical config +
# seed), so we run BOTH passes' cases concurrently as one fan-out of pass×case units.
# The environment is identical by construction (same process, same moment), which is
# exactly what the old sequential before-comparison drift check verified — so that
# check is unnecessary here. The single environment_contract.json (written earlier)
# still records the runtime identity for the summary.
units=$(( 2 * ${#ALL_CASES[@]} ))
jobs_limit="$(effective_jobs "$units")"
echo "===== REFERENCE + COMPARISON (concurrent) ====="
echo "  (${units} pass×case unit(s), up to ${jobs_limit} concurrent; console -> logs/<pass>-<case>.console.log)"
for pass_name in reference comparison; do
  mkdir -p "${WORK_ROOT}/${pass_name}" "${WORK_ROOT}/logs/status/${pass_name}"
  for case_name in "${ALL_CASES[@]}"; do
    throttle_jobs "$jobs_limit"
    (
      trap 'echo $? > "${WORK_ROOT}/logs/status/${pass_name}/${case_name}.rc"' EXIT
      export MPLCONFIGDIR="${MPLCONFIGDIR}/${pass_name}_${case_name}"
      mkdir -p "$MPLCONFIGDIR"
      config="$(config_for "$case_name")"
      engine="$(engine_for "$case_name")"
      case_root="${WORK_ROOT}/${pass_name}/${case_name}"
      [[ -f "$config" ]] || die "configuration missing: $config"
      echo "--- ${pass_name}/${case_name} ---"
      run_logged "${WORK_ROOT}/logs/${pass_name}-${case_name}-autotune.log" \
        "$VITRIFLOW_BIN" autotune -c "$config" -o "$case_root" --no-resume
      exercise_case "$pass_name" "$case_name" "$config" "$case_root" "$engine"
      if ((WITH_SLURM)); then
        exercise_slurm_case "$pass_name" "$case_name" "$config" "$case_root"
      fi
    ) > "${WORK_ROOT}/logs/${pass_name}-${case_name}.console.log" 2>&1 &
  done
done
wait || true
collect_case_status "reference" "${WORK_ROOT}/logs/status/reference" "${ALL_CASES[@]}"
collect_case_status "comparison" "${WORK_ROOT}/logs/status/comparison" "${ALL_CASES[@]}"

mkdir -p "${WORK_ROOT}/comparison_reports" "${WORK_ROOT}/comparison_plots"
jobs_limit="$(effective_jobs)"
status_dir="${WORK_ROOT}/logs/status/compare"
mkdir -p "$status_dir"
echo "===== COMPARE (${#ALL_CASES[@]} case(s), up to ${jobs_limit} concurrent) ====="
for case_name in "${ALL_CASES[@]}"; do
  throttle_jobs "$jobs_limit"
  (
    trap 'echo $? > "${status_dir}/${case_name}.rc"' EXIT
    export MPLCONFIGDIR="${MPLCONFIGDIR}/compare_${case_name}"
    mkdir -p "$MPLCONFIGDIR"
    ref="${WORK_ROOT}/reference/${case_name}"
    cmp="${WORK_ROOT}/comparison/${case_name}"
    report="${WORK_ROOT}/comparison_reports/${case_name}.json"
    "$PYTHON_BIN" "$CHECKER" compare --reference "$ref" --comparison "$cmp" --report "$report"

    compare_dir="${WORK_ROOT}/comparison_plots/${case_name}"
    run_logged "${WORK_ROOT}/logs/compare-${case_name}-production.log" \
      "$VITRIFLOW_BIN" plot-production-compare \
        -i "${ref}/autotune_results.json" "${cmp}/autotune_results.json" \
        --labels reference deterministic-replay -o "$compare_dir" \
        --title "${case_name}: reference vs deterministic replay" --dpi 150
    "$PYTHON_BIN" "$CHECKER" check-comparison-plots --dir "$compare_dir" \
      --input "${ref}/autotune_results.json" "${cmp}/autotune_results.json"
  ) > "${WORK_ROOT}/logs/compare-${case_name}.console.log" 2>&1 &
done
wait || true
collect_case_status "compare" "$status_dir" "${ALL_CASES[@]}"

"$PYTHON_BIN" - "$WORK_ROOT" "$SKIP_INTERFACE_AUDIT" "$PARTIAL_SELECTION" "$EXPECTED_VERSION" "$VITRIFLOW_RESOLVED" "$WITH_SLURM" "$WITH_CP2K_CELL_OPT" "${ALL_CASES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
audit_skipped = bool(int(sys.argv[2]))
partial_selection = bool(int(sys.argv[3]))
expected_version = sys.argv[4]
launcher = str(Path(sys.argv[5]).resolve())
with_slurm = bool(int(sys.argv[6]))
with_cp2k_cell_opt = bool(int(sys.argv[7]))
cases = sys.argv[8:]
sys.path[0] = str(Path(launcher).parent)
import vitriflow
diagnostic_only = audit_skipped or partial_selection
reports = [json.loads((root / "comparison_reports" / f"{case}.json").read_text()) for case in cases]
config_contracts = [json.loads((root / "config_contracts" / f"{case}.json").read_text()) for case in cases]
summary = {
    "schema": "vitriflow.application_validation.summary.v1",
    "status": "diagnostic_only" if diagnostic_only else "passed",
    "release_signoff_valid": not diagnostic_only,
    "interface_audit_skipped": audit_skipped,
    "partial_case_selection": partial_selection,
    "cases": cases,
    "cp2k_included": "si_cp2k" in cases,
    "cp2k_cell_refinement_included": with_cp2k_cell_opt,
    "slurm_replay_included": with_slurm,
    "size_stage_executed": False,
    "environment_contract": {
        "expected_vitriflow_version": expected_version,
        "actual_vitriflow_version": str(vitriflow.__version__),
        "python": os.path.realpath(sys.executable),
        "launcher": launcher,
        "package": str(Path(vitriflow.__file__).resolve()),
    },
    "config_contracts": config_contracts,
    "comparison_reports": reports,
    "interpretation": (
        "Application execution, analysis, plots, sidecar integrity, convergence plumbing, "
        "and deterministic replay passed. Validation trajectories are intentionally too short "
        "for materials-science interpretation."
    ),
}
name = "VALIDATION_DIAGNOSTIC_ONLY.json" if diagnostic_only else "VALIDATION_PASSED.json"
(root / name).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if ((SKIP_INTERFACE_AUDIT || PARTIAL_SELECTION)); then
  echo "DIAGNOSTIC RUN COMPLETE, BUT RELEASE SIGN-OFF IS INVALID: ${WORK_ROOT}/VALIDATION_DIAGNOSTIC_ONLY.json" >&2
  exit 2
fi
echo "VALIDATION PASSED: ${WORK_ROOT}/VALIDATION_PASSED.json"
