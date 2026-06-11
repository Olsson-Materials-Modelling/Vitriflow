#!/usr/bin/env bash
# Build a LAMMPS executable with ML-QUIP/GAP support inside the active conda env.
# This OpenBLAS version avoids the internal-linalg link failure with QUIP
# (undefined LAPACK symbols such as dsysv_ / dgeev_).
#
# Intended entry point:
#   conda activate vitriflow-quip
#   bash scripts/build_lammps_quip_openblas.sh
#
# Optional overrides:
#   OPENBLAS_LIB=/path/to/libopenblas.so.0 bash scripts/build_lammps_quip_openblas.sh
#   BLAS_LIBRARIES=/path/to/libblas.so LAPACK_LIBRARIES=/path/to/liblapack.so bash scripts/build_lammps_quip_openblas.sh
set -Eeuo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the conda environment first, e.g. 'conda activate vitriflow-quip'." >&2
  exit 2
fi

LAMMPS_TAG="${LAMMPS_TAG:-stable_22Jul2025_update4}"
LAMMPS_REPO="${LAMMPS_REPO:-https://github.com/lammps/lammps.git}"
ROOT="${LAMMPS_QUIP_ROOT:-$CONDA_PREFIX/src/vitriflow-lammps-quip}"
SRC_DIR="${LAMMPS_SRC_DIR:-$ROOT/lammps-$LAMMPS_TAG}"
BUILD_DIR="${LAMMPS_BUILD_DIR:-$ROOT/build-$LAMMPS_TAG-openblas}"
INSTALL_PREFIX="${LAMMPS_INSTALL_PREFIX:-$CONDA_PREFIX/opt/lammps-quip/$LAMMPS_TAG-openblas}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"

mkdir -p "$ROOT" "$CONDA_PREFIX/bin"

echo "==> LAMMPS tag:        $LAMMPS_TAG"
echo "==> Source directory:  $SRC_DIR"
echo "==> Build directory:   $BUILD_DIR"
echo "==> Install prefix:    $INSTALL_PREFIX"
echo "==> Build jobs:        $JOBS"

for exe in git cmake ninja mpicc mpicxx mpifort; do
  if ! command -v "$exe" >/dev/null 2>&1; then
    echo "ERROR: required executable '$exe' not found in PATH." >&2
    exit 3
  fi
done

_find_first_existing() {
  local pattern candidate
  for pattern in "$@"; do
    if [[ "$pattern" != *[\*\?\[]* ]]; then
      if [[ -e "$pattern" ]]; then
        printf '%s\n' "$pattern"
        return 0
      fi
      continue
    fi

    while IFS= read -r candidate; do
      if [[ -e "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done < <(compgen -G "$pattern" | sort || true)
  done
  return 1
}

# Locate BLAS/LAPACK in the active conda environment.
#
# conda-forge OpenBLAS installs can expose the implementation as:
#   libopenblas.so.0 -> libopenblasp-r0.3.xx.so
#   libopenblasp-r0.3.xx.so
# while the selected BLAS/LAPACK ABI packages expose:
#   libblas.so
#   liblapack.so
#
# Prefer the actual OpenBLAS implementation for both BLAS and LAPACK because
# QUIP needs LAPACK symbols from the same OpenBLAS-backed stack.
OPENBLAS_LIB="${OPENBLAS_LIB:-}"
BLAS_LIBRARIES="${BLAS_LIBRARIES:-}"
LAPACK_LIBRARIES="${LAPACK_LIBRARIES:-}"

if [[ -n "$OPENBLAS_LIB" ]]; then
  if [[ ! -e "$OPENBLAS_LIB" ]]; then
    echo "ERROR: OPENBLAS_LIB was set but does not exist: $OPENBLAS_LIB" >&2
    exit 4
  fi
  BLAS_LIBRARIES="${BLAS_LIBRARIES:-$OPENBLAS_LIB}"
  LAPACK_LIBRARIES="${LAPACK_LIBRARIES:-$OPENBLAS_LIB}"
fi

if [[ -z "$BLAS_LIBRARIES" || -z "$LAPACK_LIBRARIES" ]]; then
  OPENBLAS_LIB="$(_find_first_existing \
    "$CONDA_PREFIX/lib/libopenblas.so" \
    "$CONDA_PREFIX/lib/libopenblas.so.*" \
    "$CONDA_PREFIX/lib/libopenblas*.so" \
    "$CONDA_PREFIX/lib/libopenblas*.so.*" \
    "$CONDA_PREFIX/lib/libopenblasp*.so" \
    "$CONDA_PREFIX/lib/libopenblasp*.so.*" \
    "$CONDA_PREFIX/lib/libopenblas.dylib" \
    "$CONDA_PREFIX/lib/libopenblas*.dylib" \
    "$CONDA_PREFIX/lib/libopenblas.a" \
    "$CONDA_PREFIX/lib/libopenblas*.a" \
    || true)"

  if [[ -n "$OPENBLAS_LIB" ]]; then
    BLAS_LIBRARIES="${BLAS_LIBRARIES:-$OPENBLAS_LIB}"
    LAPACK_LIBRARIES="${LAPACK_LIBRARIES:-$OPENBLAS_LIB}"
  fi
fi

if [[ -z "$BLAS_LIBRARIES" ]]; then
  BLAS_LIBRARIES="$(_find_first_existing \
    "$CONDA_PREFIX/lib/libblas.so" \
    "$CONDA_PREFIX/lib/libblas.so.*" \
    "$CONDA_PREFIX/lib/libblas.dylib" \
    "$CONDA_PREFIX/lib/libblas.a" \
    || true)"
fi

if [[ -z "$LAPACK_LIBRARIES" ]]; then
  LAPACK_LIBRARIES="$(_find_first_existing \
    "$CONDA_PREFIX/lib/liblapack.so" \
    "$CONDA_PREFIX/lib/liblapack.so.*" \
    "$CONDA_PREFIX/lib/liblapack.dylib" \
    "$CONDA_PREFIX/lib/liblapack.a" \
    || true)"
fi

if [[ -z "$BLAS_LIBRARIES" || -z "$LAPACK_LIBRARIES" ]]; then
  echo "ERROR: BLAS/LAPACK libraries were not found in $CONDA_PREFIX/lib." >&2
  echo "Looked for libopenblas*, libopenblasp*, libblas*, and liblapack*." >&2
  echo "Install with:" >&2
  echo "  conda install -c conda-forge 'libblas=*=*openblas' 'liblapack=*=*openblas' libopenblas" >&2
  echo "" >&2
  echo "Installed BLAS/LAPACK packages:" >&2
  conda list 2>/dev/null | grep -Ei 'blas|lapack|openblas' >&2 || true
  echo "" >&2
  echo "Library candidates in $CONDA_PREFIX/lib:" >&2
  find "$CONDA_PREFIX/lib" -maxdepth 1 \( \
    -name 'libopenblas*' -o \
    -name 'libopenblasp*' -o \
    -name 'libblas*' -o \
    -name 'liblapack*' \
  \) -printf '  %f -> %l\n' 2>/dev/null | sort >&2 || true
  exit 4
fi

echo "==> OpenBLAS impl:     ${OPENBLAS_LIB:-not directly detected; using BLAS/LAPACK ABI libs}"
echo "==> BLAS library:      $BLAS_LIBRARIES"
echo "==> LAPACK library:    $LAPACK_LIBRARIES"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "==> Cloning LAMMPS $LAMMPS_TAG"
  git clone --depth 1 --branch "$LAMMPS_TAG" "$LAMMPS_REPO" "$SRC_DIR"
else
  echo "==> Reusing existing source tree"
  git -C "$SRC_DIR" fetch --depth 1 origin "$LAMMPS_TAG" || true
  git -C "$SRC_DIR" checkout "$LAMMPS_TAG"
fi

# A previous failed build with USE_INTERNAL_LINALG=yes can leave a cache and
# ExternalProject state that continues to link against liblammps_linalg.a.
# Use a distinct build directory by default; CLEAN_BUILD=1 removes it explicitly.
if [[ "$CLEAN_BUILD" == "1" && -d "$BUILD_DIR" ]]; then
  echo "==> Removing previous build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

cmake -S "$SRC_DIR/cmake" -B "$BUILD_DIR" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -D CMAKE_C_COMPILER=mpicc \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_Fortran_COMPILER=mpifort \
  -D CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -D CMAKE_BUILD_RPATH="$CONDA_PREFIX/lib" \
  -D CMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
  -D BUILD_MPI=yes \
  -D BUILD_SHARED_LIBS=no \
  -D LAMMPS_EXCEPTIONS=no \
  -D PKG_ML-QUIP=yes \
  -D DOWNLOAD_QUIP=yes \
  -D USE_INTERNAL_LINALG=no \
  -D BLA_VENDOR=OpenBLAS \
  -D BLAS_LIBRARIES="$BLAS_LIBRARIES" \
  -D LAPACK_LIBRARIES="$LAPACK_LIBRARIES"

cmake --build "$BUILD_DIR" --parallel "$JOBS"
cmake --install "$BUILD_DIR"

if [[ ! -x "$INSTALL_PREFIX/bin/lmp" ]]; then
  echo "ERROR: expected executable not found: $INSTALL_PREFIX/bin/lmp" >&2
  exit 5
fi

cat > "$CONDA_PREFIX/bin/lmp_quip" <<EOF2
#!/usr/bin/env bash
exec "$INSTALL_PREFIX/bin/lmp" "\$@"
EOF2
chmod +x "$CONDA_PREFIX/bin/lmp_quip"

MANIFEST="$INSTALL_PREFIX/vitriflow_lammps_quip_manifest.txt"
{
  echo "LAMMPS_TAG=$LAMMPS_TAG"
  echo "LAMMPS_REPO=$LAMMPS_REPO"
  echo "LAMMPS_COMMIT=$(git -C "$SRC_DIR" rev-parse HEAD)"
  echo "INSTALL_PREFIX=$INSTALL_PREFIX"
  echo "CONDA_PREFIX=$CONDA_PREFIX"
  echo "BUILD_DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "CMAKE_VERSION=$(cmake --version | head -1)"
  echo "MPICC=$(command -v mpicc)"
  echo "MPICXX=$(command -v mpicxx)"
  echo "MPIFORT=$(command -v mpifort)"
  echo "OPENBLAS_LIB=${OPENBLAS_LIB:-}"
  echo "BLAS_LIBRARIES=$BLAS_LIBRARIES"
  echo "LAPACK_LIBRARIES=$LAPACK_LIBRARIES"
  echo "LMP_EXEC=$INSTALL_PREFIX/bin/lmp"
  echo "--- ldd lmp | grep -Ei 'openblas|lapack|blas|gfortran' ---"
  ldd "$INSTALL_PREFIX/bin/lmp" 2>/dev/null | grep -Ei 'openblas|lapack|blas|gfortran' || true
  echo "--- lmp -h package/style probe ---"
  "$INSTALL_PREFIX/bin/lmp" -h | grep -Ei 'Installed packages|ML-QUIP|pair styles|(^|[[:space:]])quip([[:space:]]|$)' || true
} > "$MANIFEST"

if ! "$INSTALL_PREFIX/bin/lmp" -h | grep -Eq '(^|[[:space:]])quip([[:space:]]|$)|ML-QUIP'; then
  echo "ERROR: lmp -h did not show ML-QUIP / pair_style quip. Build is not usable for GAP." >&2
  echo "See: $MANIFEST" >&2
  exit 6
fi

echo "==> Installed custom LAMMPS/QUIP executable: $CONDA_PREFIX/bin/lmp_quip"
echo "==> Build manifest: $MANIFEST"
echo "==> Use in VitriFlow YAML:"
echo "    lammps:"
echo "      lammps_cmd: lmp_quip"
