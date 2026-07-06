# Vitriflow

Vitriflow automates melt-quench, custom-stage molecular dynamics, and post-run structural analysis for amorphous and disordered materials. It prepares engine inputs, runs ensembles, records provenance, computes structural descriptors, checks convergence, and writes analysis/plotting artifacts.

Current packaged release: `0.4.29.17`  
Python: `>=3.10`  
Primary engine: LAMMPS. CP2K and QUIP/GAP support are available for scoped workflows.

## Features

- Melt-quench autotuning and production runs.
- Continuous custom-stage schedules for fixed temperature/time protocols.
- Standalone analysis of existing runs, final-structure folders, ASE databases, and dataset JSON files.
- RDF, coordination, graph/ring, void, CDF, and ensemble descriptor outputs.
- Provenance sidecars for structure manifests, graph rules, adaptive RDF/shell derivations, descriptor status, and streaming analysis.
- LAMMPS/OpenKIM support, CP2K helpers, and QUIP/GAP build support for the hard-carbon demonstrator.

## Layout

```text
Vitriflow/                    Python package and CLI
Vitriflow/examples/           Example YAML configuration files
demos/hardcarbon_gap20ugr/    GAP-20U+gr custom-schedule demonstrator
docs/                         Descriptor provenance and graph-rule notes
scripts/                      Install, build, validation, and potential helpers
potentials/                   Placeholder for user-supplied potentials
dist/                         Release wheel, when included
```

## Conda setup

Run conda commands from the extracted release or repository root.

Create the standard runtime environment:

```bash
conda env create -f environment.yml
conda activate Vitriflow
```

Update the standard runtime environment:

```bash
conda env update -f environment.yml --prune
# or target the environment explicitly:
conda env update -n Vitriflow -f environment.yml --prune
conda activate Vitriflow
```

Create or update the QUIP/GAP environment:

```bash
conda env create -f environment_quip_openblas.yml
conda activate Vitriflow-quip

# update an existing QUIP/GAP environment
conda env update -f environment_quip_openblas.yml --prune
# or target it explicitly:
conda env update -n Vitriflow-quip -f environment_quip_openblas.yml --prune
```

For CP2K support, create the scoped CP2K environment or apply the CP2K dependency layer to an existing target environment. Do not use `--prune` when using `environment_cp2k.yml` as an overlay.

```bash
# dedicated CP2K environment
conda env create -f environment_cp2k.yml

# CP2K overlay for an existing Vitriflow environment
conda env update -n Vitriflow -f environment_cp2k.yml
```

Environment files included in the release:

| File | Purpose |
| --- | --- |
| `environment.yml` | Standard LAMMPS/OpenKIM runtime. |
| `environment_quip.yml` | QUIP/GAP build and runtime environment. |
| `environment_quip_openblas.yml` | QUIP/GAP environment pinned to OpenBLAS/LAPACK. |
| `environment_cp2k.yml` | CP2K add-on dependencies for scoped DFT workflows. |
| `environment_rhel8_mpi.yml` | Robust build environment of RHEL 8 style systesm. |

## Install

Install from the release wheel after activating the target environment:

```bash
python -m pip uninstall -y Vitriflow
python -m pip install --force-reinstall --no-deps dist/Vitriflow-<version>-py3-none-any.whl
hash -r
Vitriflow --version
```

For this release:

```bash
python -m pip install --force-reinstall --no-deps dist/Vitriflow-0.4.29.17-py3-none-any.whl
```

The bundled helper performs the same install and import check:

```bash
bash install_release.sh
```

Editable development install:

```bash
python -m pip install -e ".[dev]"
```

## Build

Build or refresh the release wheel inside a conda environment:

```bash
conda env create -f environment.yml
conda activate Vitriflow
conda install -c conda-forge setuptools wheel
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

For an existing environment:

```bash
conda activate Vitriflow
conda env update -f environment.yml --prune
conda install -c conda-forge setuptools wheel
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

For QUIP/GAP LAMMPS builds:

```bash
conda env create -f environment_quip_openblas.yml
conda activate Vitriflow-quip
python -m pip install -e .
bash scripts/build_lammps_quip_openblas.sh
```

### Older Linux MPI runtime
 
On RHEL/Rocky/Alma/CentOS 8-style systems, the conda environment can build successfully but fail at the first `lmp` call.
 
Typical runtime error:
 
```text
lmp: /lib64/libm.so.6: version `GLIBC_2.29' not found
lmp: /lib64/libc.so.6: version `GLIBC_2.34' not found
```
 
This is a silent conda solve/scoping issue: the selected LAMMPS binary requires newer glibc symbols than the.
 
For these systems, use the RHEL 8 robust MPI environment:
 
```bash
conda env remove -n vitriflow
conda env create -f environment_rhel8_mpi.yml
conda activate vitriflow
```
 
Verify directly:
 
```bash
mpirun -np 2 lmp -h
```

## Usage

```bash
Vitriflow --help
Vitriflow autotune -c config.yaml -o runs/autotune
Vitriflow run -c config.yaml -o runs/production --use-autotune runs/autotune/autotune_results.json
Vitriflow run-schedule -c custom_schedule.yaml -o runs/custom_schedule
Vitriflow analyze-output -c analysis.yaml -i final_structures_or_dataset -o analysis_out
Vitriflow plot-production -i analysis_out/analysis_results.json -o analysis_out/report.pdf
```

Example YAML files are in `Vitriflow/examples/`. Review engine commands, MPI settings, potential paths, structure sources, and convergence targets before production use.

## Workflows

### Standard melt-quench

Use `Vitriflow autotune` to scan melt temperature, hold time, quench rate, and optional size effects. Use `Vitriflow run` for production ensembles from YAML or a saved autotune/production-plan JSON.

### Custom schedules

Use `Vitriflow run-schedule` when a fixed sequence should run as one continuous LAMMPS trajectory.

```yaml
custom_schedule:
  enforce_temperature_continuity: true
  stages:
    - {name: melt, temperature_K: 3500.0, time_ps: 100.0, role: melt, velocity_mode: create}
    - {name: quench, temperature_start_K: 3500.0, temperature_stop_K: 300.0, time_ps: 20.0, role: quench, velocity_mode: preserve}
    - {name: relax, temperature_K: 300.0, time_ps: 20.0, role: relax, velocity_mode: preserve}
  analysis_roles:
    melt: melt
    quench: quench
    relax: relax
```

Aliases are available for compatibility: `run-custom`, `run-custom-schedule`, `run-hardcarbon`, and `run-hc`.

`run-schedule` is scoped to MD-only, continuous-LAMMPS schedules. Unsupported standard-runner paths such as DFT refinement, elastic production screens, non-LAMMPS custom-stage execution, discontinuous stage execution, and later-stage velocity creation fail early.

### Standalone analysis

Use `Vitriflow analyze-output` for production directories, final-structure folders, ASE databases, `output_dataset.json`, `task_result.json`, or a single ASE-readable structure file.

Recommended large-ensemble settings:

```yaml
analysis:
  analysis_streaming: true
  analysis_workers: 16
  analysis_max_in_flight: 16
  embed_structures: false
```

Equivalent CLI overrides:

```bash
Vitriflow analyze-output -c analysis.yaml -i DATASET -o analysis_out \
  --analysis-workers 16 --analysis-max-in-flight 16
```

Key outputs include `analysis_results.json`, `output_dataset.json`, ensemble CDF sidecars, graph-rule sidecars, descriptor provenance, and compact adaptive RDF/shell derivation sidecars.

## Engines and demonstrator notes

Vitriflow supports LAMMPS with OpenKIM models or explicit LAMMPS potential commands. CP2K support is included for scoped DFT workflows. QUIP/GAP workflows should use `environment_quip_openblas.yml` and `QUIP_GAP_SETUP.md`.

The hard-carbon demonstrator is in `demos/hardcarbon_gap20ugr/`. The GAP XML and `sparseX` sidecars are not bundled; place them in the demonstrator `potentials/` directory before running the demo scripts.

## References and citation guidance

For publications, cite the Vitriflow release/version plus the upstream engines, potentials, datasets, and analysis methods used in the specific workflow. The list below is intended as a practical starting point, not an exhaustive bibliography.

### Vitriflow

Vitriflow:calibrated amorphous structure ensembles from melt--quench simulations. doi: 10.48550/arXiv.2607.01407

### Model building and simulation

- **LAMMPS**: A. P. Thompson et al., "LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales," *Computer Physics Communications* 271, 108171 (2022), doi: [10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).
- **OpenKIM/KIM API**, when using OpenKIM models: R. S. Elliott and E. B. Tadmor, "Knowledgebase of Interatomic Models (KIM) Application Programming Interface (API)," OpenKIM (2011), doi: [10.25950/ff8f563a](https://doi.org/10.25950/ff8f563a). Cite the specific KIM model record as well.
- **Packmol**, when used for initial packing: L. Martínez, R. Andrade, E. G. Birgin, and J. M. Martínez, "PACKMOL: A package for building initial configurations for molecular dynamics simulations," *Journal of Computational Chemistry* 30, 2157-2164 (2009), doi: [10.1002/jcc.21224](https://doi.org/10.1002/jcc.21224).
- **CP2K/Quickstep**, when using DFT workflows: T. D. Kühne et al., "CP2K: An electronic structure and molecular dynamics software package - Quickstep: efficient and accurate electronic structure calculations," *Journal of Chemical Physics* 152, 194103 (2020), doi: [10.1063/5.0007045](https://doi.org/10.1063/5.0007045).
- **GAP/QUIP**, when using Gaussian Approximation Potentials: A. P. Bartók, M. C. Payne, R. Kondor, and G. Csányi, "Gaussian approximation potentials: the accuracy of quantum mechanics, without the electrons," *Physical Review Letters* 104, 136403 (2010), doi: [10.1103/PhysRevLett.104.136403](https://doi.org/10.1103/PhysRevLett.104.136403). For current GAP software details, see S. Klawohn et al., *Journal of Chemical Physics* 159, 174108 (2023), doi: [10.1063/5.0160898](https://doi.org/10.1063/5.0160898).
- **Carbon GAP-20/GAP-20U/GAP-20U+gr demonstrator**, when used: P. Rowe et al., "An accurate and transferable machine learning potential for carbon," *Journal of Chemical Physics* 153, 034702 (2020), doi: [10.1063/5.0005084](https://doi.org/10.1063/5.0005084); and G. A. Marchant et al., "Exploring the configuration space of elemental carbon with empirical and machine learned interatomic potentials," *npj Computational Materials* 9, 131 (2023), doi: [10.1038/s41524-023-01081-w](https://doi.org/10.1038/s41524-023-01081-w).

### Structure handling and analysis

- **ASE**: A. H. Larsen et al., "The Atomic Simulation Environment - a Python library for working with atoms," *Journal of Physics: Condensed Matter* 29, 273002 (2017), doi: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).
- **External structure sources**, when used: A. Jain et al., "The Materials Project: A materials genome approach to accelerating materials innovation," *APL Materials* 1, 011002 (2013), doi: [10.1063/1.4812323](https://doi.org/10.1063/1.4812323); and S. Gražulis et al., "Crystallography Open Database (COD): an open-access collection of crystal structures and platform for world-wide collaboration," *Nucleic Acids Research* 40, D420-D427 (2012), doi: [10.1093/nar/gkr900](https://doi.org/10.1093/nar/gkr900). Cite the specific source record as well.
- **NetworkX**, for graph-based topology: A. A. Hagberg, D. A. Schult, and P. J. Swart, "Exploring network structure, dynamics, and function using NetworkX," *Proceedings of the 7th Python in Science Conference* (2008), pp. 11-15.
- **NumPy/SciPy**, for array, numerical, RDF/CDF, Sobol, and nearest-neighbor routines: C. R. Harris et al., "Array programming with NumPy," *Nature* 585, 357-362 (2020), doi: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2); P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods* 17, 261-272 (2020), doi: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
- **Ring/topology analysis**, for methodological context: S. V. King, "Ring configurations in a random network model of vitreous silica," *Nature* 213, 1112-1113 (1967), doi: [10.1038/2131112a0](https://doi.org/10.1038/2131112a0); and D. S. Franzblau, "Computation of ring statistics for network models of solids," *Physical Review B* 44, 4925-4930 (1991), doi: [10.1103/PhysRevB.44.4925](https://doi.org/10.1103/PhysRevB.44.4925).
- **Void/pore-size analysis**, for porous-carbon context: L. D. Gelb and K. E. Gubbins, "Pore size distributions in porous glasses: a computer simulation study," *Langmuir* 15, 305-308 (1999), doi: [10.1021/la9808418](https://doi.org/10.1021/la9808418); and Y. Wang et al., "Structure and pore size distribution in nanoporous carbon," *Chemistry of Materials* 34, 617-628 (2022), doi: [10.1021/acs.chemmater.1c03279](https://doi.org/10.1021/acs.chemmater.1c03279).

## Validation

Run a basic install check:

```bash
Vitriflow --version
python -c "import Vitriflow; from Vitriflow.cli import main; print(Vitriflow.__version__)"
pytest tests/test_release_entrypoints.py
```

## Version history

| Version | Summary |
| --- | --- |
| `0.4.29.17` | Full-ensemble streaming hotfix: compact adaptive RDF/shell derivations, stable `derivation_ref` pointers, CSV field-limit compatibility, and stream-chunk cleanup. |
| `0.4.29.16` | Large-ensemble analysis scalability: bounded parallel `analyze-output`, streamed sidecars, compact JSON summaries, ensemble graph-rule pass, and `Vitriflow --version`. |
| `0.4.29.15` | Release-readiness install and plotting hotfix: safe wheel artifact, install helper, analysis-only ensemble CDFs, robust plotting, and legacy hard-cutoff labeling. |
| `0.4.29.14` | Analysis-only CDF/plotting hotfix: ensemble CDF sidecars, right-continuous union-support alignment, and explicit descriptor status records. |
| `0.4.29.13` | Analysis-only compatibility patch: compact structure references, advisory filtering, plotting fallback, and sidecar integrity reporting. |
| `0.4.29.12` | Descriptor-provenance and numerical-hygiene update: RepresentationRule/MetricResult schema, JSON-safe serialization, graph families, and void-scaling outputs. |
| `0.4.29.11` | Graph-family and ensemble robustness: network, candidate-contact, soft-ambiguity, and legacy graph families plus ensemble graph-rule outputs. |
| `0.4.29.10` | Adaptive RDF graph-analysis usability hotfix with explicit primary adaptive graph summaries and graph-rule provenance. |
| `0.4.29.9` | RDF-adaptive graph-rule analysis, shell-separability diagnostics, connectivity diagnostics, and fixed-cutoff-free HSE/PBE configs. |
| `0.4.29.8` | Analysis-only convergence robustness with safe undefined-CDF handling and explicit convergence provenance. |
| `0.4.29.7` | Strict CP2K final-restart loading for explicit `*-1.restart` snapshots and source-filter provenance. |
| `0.4.29.6` | Graph-rule robustness and manifest locking with graph-rule objects, structure manifests, uncertainty outputs, and CLI overrides. |
| `0.4.29.5` | Standalone analysis hotfix for final restart discovery, box/sample-id parsing, YAML source filters, and structure/lattice retention. |
| `0.4.29.4` | Custom-schedule regression protection for continuous rendering, validators, guardrails, resume fingerprints, and demo configs. |
| `0.4.29.3` | Custom-schedule resume/provenance fingerprints covering schedules, potentials, metrics, convergence, sources, seeds, and engine context. |
| `0.4.29.2` | Custom-schedule guardrails for unsupported DFT, elastic, non-LAMMPS, discontinuous, missing-potential, and later velocity-creation paths. |
| `0.4.29.1` | Thermostat/barostat hardening with explicit LAMMPS style support and clarified hard-carbon demonstrator scope. |
| `0.4.29.0` | Generalized the hard-carbon-only custom runner into a generic continuous custom-stage schedule workflow. |
| `0.4.27.19` | Analysis database/YAML directory compatibility release. |
| `0.4.27.14` | File-handling audit hotfix release. |

Previous versions were developed outside GitHub and are in the process of being added / integrated to a standalone archive repository.

## License

MIT.
