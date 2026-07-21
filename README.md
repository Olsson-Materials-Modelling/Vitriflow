# Vitriflow

Vitriflow automates melt-quench, custom-stage molecular dynamics, and post-run structural analysis for amorphous and disordered materials. It prepares engine inputs, runs ensembles, records provenance, computes structural descriptors, checks convergence, and writes analysis/plotting artifacts.

Current packaged release: `0.4.37.0`

Python: `>=3.10`

Repository: https://github.com/Olsson-Materials-Modelling/Vitriflow

Primary engine: LAMMPS. CP2K and QUIP/GAP support are available for scoped workflows, subject to installation and validation of their external executables and data on the target host.

## Features

- Melt-quench autotuning and production runs.
- Continuous custom-stage schedules for fixed temperature/time protocols.
- Standalone analysis of existing runs, final-structure folders, ASE databases, and dataset JSON files.
- RDF, coordination, graph/ring, void, CDF, and ensemble descriptor outputs.
- Provenance sidecars for structure manifests, graph rules, adaptive RDF/shell derivations, descriptor status, and streaming analysis.
- LAMMPS/OpenKIM support, CP2K helpers, and QUIP/GAP build support for the hard-carbon demonstrator.

## Layout

```text
vitriflow/                    Python package and CLI
vitriflow/examples/           Example YAML configuration files
demos/hardcarbon_gap20ugr/    GAP-20U+gr custom-schedule demonstrator
docs/                         Descriptor provenance and graph-rule reference
scripts/                      Install, build, and potential helpers
validation/                   Two-pass deterministic application-validation harness
potentials/                   Placeholder for user-supplied potentials
dist/                         Release wheel, when included
```

## Conda setup

Run conda commands from the extracted release or repository root.

Create the standard runtime environment:

```bash
conda env create -f environment.yml
conda activate vitriflow
```

Update the standard runtime environment:

```bash
conda env update -f environment.yml --prune
# or target the environment explicitly:
conda env update -n vitriflow -f environment.yml --prune
conda activate vitriflow
conda update --all
```

Create or update the QUIP/GAP environment:

```bash
conda env create -f environment_quip_openblas.yml
conda activate vitriflow-quip

# update an existing QUIP/GAP environment
conda env update -f environment_quip_openblas.yml --prune
# or target it explicitly:
conda env update -n vitriflow-quip -f environment_quip_openblas.yml --prune
conda activate vitriflow-quip
```

`environment_cp2k.yml` is a dependency overlay, not a complete standalone Vitriflow environment. Apply it to an existing environment and do not use `--prune`, because the file intentionally lists only CP2K/MPI/BLAS additions:

```bash
conda env update -n vitriflow -f environment_cp2k.yml
conda activate vitriflow
```

Environment files included in the release:

| File | Purpose |
| --- | --- |
| `environment.yml` | Standard LAMMPS/OpenKIM runtime. |
| `environment_quip.yml` | QUIP/GAP build and runtime environment. |
| `environment_quip_openblas.yml` | QUIP/GAP environment pinned to OpenBLAS/LAPACK. |
| `environment_cp2k.yml` | CP2K add-on dependencies for scoped DFT workflows. |
| `environment_rhel8_mpi.yml` | Compatibility recipe for RHEL 8-derived systems; validate on the target host. |

## Install

Install from the release wheel after activating the target environment:

```bash
python -m pip uninstall -y vitriflow
python -m pip install --force-reinstall --no-deps dist/vitriflow-<version>-py3-none-any.whl
hash -r
vitriflow --version
```

For the currently documented packaged release, when its wheel is present:

```bash
python -m pip install --force-reinstall --no-deps dist/vitriflow-0.4.37.0-py3-none-any.whl
```

The repository/release-tree helper reads the version from `pyproject.toml`, removes an existing `vitriflow` installation, and installs the exact matching wheel from `dist/` when that wheel exists. If it does not exist, the helper deliberately falls back to an editable install from the current source tree. It then checks the package version and lowercase `vitriflow` entry point from outside the source directory:

```bash
bash install_release.sh
```

`install_release.sh` is a source/release-tree convenience file; it is not installed inside the Python wheel. A wheel installation includes the example data declared in `pyproject.toml`, under the installed `vitriflow` package.

Editable development install:

```bash
python -m pip install -e ".[dev]"
```

## Build

Build or refresh the release wheel inside a conda environment:

```bash
conda env create -f environment.yml
conda activate vitriflow
conda install -c conda-forge setuptools wheel
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

For an existing environment:

```bash
conda activate vitriflow
conda env update -f environment.yml --prune
conda install -c conda-forge setuptools wheel
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

For QUIP/GAP LAMMPS builds:

```bash
conda env create -f environment_quip_openblas.yml
conda activate vitriflow-quip
python -m pip install -e .
bash scripts/build_lammps_quip_openblas.sh
```

### Older Linux MPI runtime
 
On RHEL/Rocky/Alma/CentOS 8-style systems, a conda environment may solve successfully but fail at the first `lmp` call.
 
Typical runtime error:
 
```text
lmp: /lib64/libm.so.6: version `GLIBC_2.29' not found
lmp: /lib64/libc.so.6: version `GLIBC_2.34' not found
```
 
This indicates that the selected LAMMPS binary requires newer glibc symbols than the system libraries provide.
 
`environment_rhel8_mpi.yml` is a compatibility recipe intended to avoid one known incompatible LAMMPS build. It is not a guarantee for every RHEL-derived host, MPI fabric, scheduler, or driver stack. Create it and validate the actual engine on the target system before a production run:
 
```bash
conda env remove -n vitriflow
conda env create -f environment_rhel8_mpi.yml
conda activate vitriflow
```
 
Verify directly:
 
```bash
mpirun -np 2 lmp -h
```

## Quickstart

With Vitriflow installed into an environment that provides LAMMPS and the required KIM model, run a minimal melt-quench → analysis flow on the bundled Al example:

```bash
vitriflow autotune -c vitriflow/examples/minimal_metal.yaml -o runs/autotune
vitriflow run -c vitriflow/examples/minimal_metal.yaml -o runs/production \
  --use-autotune runs/autotune/autotune_results.json
vitriflow analyze-output -c vitriflow/examples/minimal_metal.yaml -i runs/production -o runs/analysis
vitriflow plot-production -i runs/analysis/analysis_results.json -o runs/analysis/report
```

## Usage

```bash
vitriflow --help
vitriflow autotune -c config.yaml -o runs/autotune
vitriflow run -c config.yaml -o runs/production --use-autotune runs/autotune/autotune_results.json
vitriflow run-schedule -c custom_schedule.yaml -o runs/custom_schedule
vitriflow analyze-output -c analysis.yaml -i final_structures_or_dataset -o analysis_out
vitriflow plot-production -i analysis_out/analysis_results.json -o analysis_out/report.pdf
```

In a source or extracted release tree, example YAML files are in `vitriflow/examples/`. The installed release includes the minimal Al, Si/CP2K, and hard-carbon GAP configurations (plus the Al data file), and five SiO2 configurations: the `sio2_bks_zbl_smoke.yaml` CI smoke and four production-ready melt-quench templates, `sio2_{bks,kim}_{packmol,cristobalite}_production.yaml` — the BKS and KIM Buckingham potentials, each started from a packmol amorphous cell or a beta-cristobalite crystal. The smoke is a tiny, ultrashort execution test, not a scientific input (its ensemble is labeled `fixed_count_unassessed`); the production templates are full-scale melt-quench workflows and run every autotune stage at 10 replicas. Review engine commands, MPI settings, potential paths, structure sources, and convergence targets before production use.

### Analysis behavior contract

Cutoff-driven analysis is the default: with `metrics.graph_rules` absent or empty, Vitriflow keeps that path (pooled-ensemble cutoff scope) and does not construct graph-analysis sidecars. Enhanced graph analysis runs only when you supply a non-empty `metrics.graph_rules` list or explicit graph-rule CLI options; structure and provenance manifests are written either way and do not, by themselves, opt a run into graph analysis. See [`docs/graph_rule_robustness.md`](docs/graph_rule_robustness.md) for details.

Autotune convergence uses only scalar scan metrics that define a tolerance; auxiliary metrics without one are reported as diagnostics, and unknown convergence-config keys fail validation.

## Workflows

### Standard melt-quench

Use `vitriflow autotune` to scan melt temperature, hold time, quench rate, and optional size effects. Use `vitriflow run` for production ensembles from YAML or a saved autotune/production-plan JSON.

### Custom schedules

Use `vitriflow run-schedule` when a fixed sequence should run as one continuous LAMMPS trajectory.

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

Aliases are available for compatibility: `run-custom`, `run-custom-schedule`, `run-custom-stages`, `run-cs`, `run-hardcarbon`, and `run-hc`.

`run-schedule` is scoped to MD-only, continuous-LAMMPS schedules. Unsupported standard-runner paths such as DFT refinement, elastic production screens, non-LAMMPS custom-stage execution, discontinuous stage execution, and later-stage velocity creation fail early.

### Standalone analysis

Use `vitriflow analyze-output` for production directories, final-structure folders, ASE databases, `output_dataset.json`, `task_result.json`, or a single ASE-readable structure file.

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
vitriflow analyze-output -c analysis.yaml -i DATASET -o analysis_out \
  --analysis-workers 16 --analysis-max-in-flight 16
```

Key outputs include `analysis_results.json`, `output_dataset.json`, ensemble CDF sidecars, graph-rule sidecars, descriptor provenance, and compact adaptive RDF/shell derivation sidecars.

## Engines and demonstrator notes

Vitriflow can drive LAMMPS with OpenKIM models or explicit LAMMPS potential commands. CP2K integration is included for scoped DFT workflows: generated inputs represent neutral singlets, the MD bridge accepts orthorhombic cells, and post-CELL_OPT analysis can retain a triclinic optimized cell. CP2K version differences are handled automatically, and unconverged SCF cycles stay visible in per-stage diagnostics and result fields. QUIP/GAP workflows should use `environment_quip_openblas.yml` and `QUIP_GAP_SETUP.md`. All engine integrations depend on compatible external executables and data; validate the exact engine command and MPI configuration on the target host.

The hard-carbon demonstrator is in `demos/hardcarbon_gap20ugr/`. The GAP XML and `sparseX` sidecars are not bundled; place them in the demonstrator `potentials/` directory before running the demo scripts.

## Documentation

- [`docs/graph_rule_robustness.md`](docs/graph_rule_robustness.md) — graph-rule descriptors: rule kinds, families, scopes, and outputs.
- [`docs/amorphous_descriptor_provenance.md`](docs/amorphous_descriptor_provenance.md) — descriptor and void provenance, structure embedding, and analysis sidecars.
- [`validation/README.md`](validation/README.md) — application-validation harness: cases, options, and release sign-off.
- [`demos/hardcarbon_gap20ugr/README.md`](demos/hardcarbon_gap20ugr/README.md) — GAP-20U+gr hard-carbon custom-schedule demonstrator.
- [`QUIP_GAP_SETUP.md`](QUIP_GAP_SETUP.md) — building LAMMPS with QUIP/GAP for the demonstrator.

## References and citation guidance

For publications, cite the Vitriflow release/version plus the upstream engines, potentials, datasets, and analysis methods used in the specific workflow. The list below is intended as a practical starting point, not an exhaustive bibliography.

### Vitriflow

"Vitriflow: calibrated amorphous structure ensembles from melt-quench simulations," arXiv:2607.01407 (2026), doi: [10.48550/arXiv.2607.01407](https://doi.org/10.48550/arXiv.2607.01407).

### Model building and simulation

- **LAMMPS**: A. P. Thompson et al., "LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales," *Computer Physics Communications* 271, 108171 (2022), doi: [10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).
- **OpenKIM/KIM API**, when using OpenKIM models: R. S. Elliott and E. B. Tadmor, "Knowledgebase of Interatomic Models (KIM) Application Programming Interface (API)," OpenKIM (2011), doi: [10.25950/ff8f563a](https://doi.org/10.25950/ff8f563a). Cite the specific KIM model record as well.
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

### Quick install check

Run a basic install check:

```bash
vitriflow --version
python -c "import vitriflow; from vitriflow.cli import main; print(vitriflow.__version__)"
pytest tests/test_release_entrypoints.py
```

This checks Python packaging and the CLI entry point. It does not validate live LAMMPS, CP2K, Packmol, QUIP/GAP, MPI, scheduler, or cluster execution; those require target-host execution, which the application-validation harness below exercises.

### Application-validation harness

`validation/` holds a standalone, fail-closed harness that exercises an installed Vitriflow build end to end — execution, numerical parsing, structural analysis, graph/void/elastic sidecars, convergence bookkeeping, plotting, and exact two-pass deterministic replay — and issues release sign-off only when every contract and both replay passes succeed.

From an environment that provides `vitriflow`, `lmp`, `mpirun`, the required KIM models, and the Python dependencies:

```bash
validation/run_vitriflow_application_validation.sh --workdir "$PWD/validation-run"
```

The three LAMMPS cases run by default; `--with-cp2k`, `--with-slurm`, and other paths are optional (see `--help`). Success is recorded in `VALIDATION_PASSED.json`. The harness validates the build on the current host and writes only inside its `--workdir`; it does not modify the source tree.

The validation trajectories are deliberately too short for materials-science interpretation; their only purpose is to exercise the machinery deterministically. See `validation/README.md` for the case matrix and options, and `validation/VALIDATION_CONTRACTS.md` for the exact application-facing behaviours the run asserts.

## Version history

| Version | Summary |
| --- | --- |
| `0.4.37.0` | Scoped KIM Simulator-Model autocore extraction, cross-environment CP2K identity, CP2K preflight/DCD/short-segment handling, disabled-production integrity, and deterministic elastic-screen plotting fixes. |
| `0.4.36.0` | Production-stage diagnostics now preserve and visibly flag scientifically undefined metric values without aborting valid boxes; relative output paths are canonicalized for `autotune --resume` and `run --resume`; a completed but uncommitted production box is validated and post-processed without rerunning its four main engine stages. |
| `0.4.35.1` | Attempt-state and restart-bundle hardening for LAMMPS and CP2K, strict engine namespaces and fresh-output contracts, plus the reviewed 0.4.35 execution-identity, potential-neutral resume, and replay-integrity changes. |
| `0.4.35.0` | Authenticated engine-build and task-artifact replay, potential-neutral local/HPC resume, exact standalone/live convergence parity, and strengthened autocore/CP2K boundary integrity. |
| `0.4.34.0` | Physics and execution hardening: G-invariant Buckingham-only C2 autocore, authenticated CP2K CELL_OPT reuse/restart, exact live/replay density and plotting parity, evidence-bound external convergence, and fail-closed `autotune`/`run` resume semantics. |
| `0.4.33.1` | Hybrid autocore RSQ-grid correction: evaluates generated tables, analytic references, and inflection-warning audits on LAMMPS's exact floating-point RSQ grid; preserves strict numerical gates and improves failed-candidate diagnostics. |
| `0.4.33.0` | Autocore source-audit correction: validates Buckingham/Morse and Coulomb components independently, requires audit-resolution convergence, matches LAMMPS's KSpace unit constants exactly, and keeps all audit-only lookup controls out of production commands. |
| `0.4.32.0` | Autocore safety and hybrid-potential release: constructs and verifies bounded C2 Buckingham-ZBL tables before any integration, supports strict additive Buckingham/Coulomb/Morse hybrids, preserves full configured cutoffs, and fails closed on ambiguous or non-representable command blocks. |
| `0.4.31.0` | Physics and execution hardening release: canonical physical-unit reporting across supported LAMMPS styles and CP2K, corrected stress/elastic and diffusion conversions, fail-closed tabulated-core electrostatics, exact periodic-pair analysis, interpolated ensemble curves, and explicit convergence strength and repeated-look semantics. |
| `0.4.30.1` | Audited correction release: exact package-content/runtime fingerprints, strict replay and resume state, mandatory verified structure manifests, fail-closed CP2K/HPC/config/path handling, and auxiliary scalar diagnostics excluded from convergence unless a tolerance exists. Historical cutoff-driven analysis remains the default; enhanced graph analysis remains explicit opt-in. |
| `0.4.30.0` | Combined reconciliation baseline: preserved cutoff-driven defaults and opt-in graph analysis, consolidated provenance/manifest and packaging work, and corrected pbc-inclusive structure hashing. |
| `0.4.29.19` | Scalar-scan convergence correction: calculated diagnostics without a defined tolerance remain reported but are excluded from rate/size selection; unknown YAML tolerance keys now fail validation. |
| `0.4.29.18` | Reconciled robustness release: strictly opt-in graph analysis, legacy cutoff preservation, validated plans, content-locked resume/HPC caches, charge-safe replication, native CP2K restart continuity, and terminal-only graph finalization. |
| `0.4.29.17` | Full-ensemble streaming hotfix: compact adaptive RDF/shell derivations, stable `derivation_ref` pointers, CSV field-limit compatibility, and stream-chunk cleanup. |
| `0.4.29.16` | Large-ensemble analysis scalability: bounded parallel `analyze-output`, streamed sidecars, compact JSON summaries, ensemble graph-rule pass, and `vitriflow --version`. |
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

Releases before those listed were developed outside GitHub and are archived separately.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the reproducibility and provenance expectations, and the pull-request checklist.

## License

MIT.
