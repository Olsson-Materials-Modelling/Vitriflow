# QUIP/GAP LAMMPS setup for the hard-carbon demonstrator

This setup is intentionally separate from the normal Vitriflow conda environment because the stock conda-forge `lammps` executable is not assumed to include `pair_style quip`.

## Build environment

From the package root:

```bash
conda env create -f environment_quip.yml
conda activate vitriflow-quip
pip install -e .
```

`environment_quip.yml` includes OpenMPI, compilers, CMake/Ninja, Packmol, OpenBLAS/LAPACK and the Vitriflow Python dependencies. It deliberately does not install the stock `lammps` package.

## Build LAMMPS with ML-QUIP

```bash
bash scripts/build_lammps_quip.sh
```

The script builds a local LAMMPS executable with:

```text
PKG_ML-QUIP=yes
DOWNLOAD_QUIP=yes
USE_INTERNAL_LINALG=no
BLAS/LAPACK = conda OpenBLAS
```

It installs a wrapper in the active environment:

```text
$CONDA_PREFIX/bin/lmp_quip
```

The build tag defaults to:

```text
stable_22Jul2025_update4
```

Override it only when intentionally changing the reproducibility point:

```bash
LAMMPS_TAG=stable_22Jul2025_update4 JOBS=8 bash scripts/build_lammps_quip.sh
```

Use `CLEAN_BUILD=1` if the CMake cache may contain a previous failed build:

```bash
CLEAN_BUILD=1 bash scripts/build_lammps_quip.sh
```

## Verify

```bash
lmp_quip -h | grep -Ei 'ML-QUIP|(^|[[:space:]])quip([[:space:]]|$)'
```

For GAP-20U+gr, use the sidecar-aware smoke test in:

```text
demos/hardcarbon_gap20ugr/scripts/smoke_test_gap20ugr_lammps.sh
```

The four potential files must be present in:

```text
demos/hardcarbon_gap20ugr/potentials/
```

Potential files are external and are not bundled with Vitriflow.
