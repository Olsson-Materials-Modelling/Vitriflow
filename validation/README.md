# Vitriflow application-validation harness

A standalone, fail-closed harness that exercises a Vitriflow 0.4.37.0 build end
to end: execution, numerical parsing, structural analysis, graph/void/elastic
sidecars, convergence bookkeeping, plotting, and exact two-pass deterministic
replay. It validates an installed build and writes only inside the run directory
passed with `--workdir`; it does not modify the source tree or the installed
package.

The trajectories are deliberately too short for materials science. Their only
job is to exercise the machinery with fixed seeds and a realistic atom count —
not to produce physical results.

## Cases

| Case | Engine / potential | Atoms |
|---|---|---:|
| `minimal_metal` | LAMMPS + KIM Al EAM | 108 |
| `sio2_bks` | LAMMPS + explicit BKS + C2 ZBL splice | 192 |
| `sio2_kim` | LAMMPS + legacy KIM Buckingham + C2 ZBL splice | 192 |
| `si_cp2k` | CP2K PBE, SZV-MOLOPT-SR-GTH | 64 |

## Run

From an environment that provides `vitriflow`, `lmp`, `mpirun`, the KIM models,
and the Python dependencies:

```bash
./run_vitriflow_application_validation.sh --workdir "$PWD/validation-run"
```

The three LAMMPS cases run by default. Optional heavier paths:

```bash
--with-cp2k            # add the 64-atom CP2K Si case (runs in a separate vitriflow-cp2k conda env)
--with-cp2k-cell-opt   # also exercise CP2K CELL_OPT refinement of every accepted box
--with-slurm --slurm-template FILE   # replay production through real Slurm tasks
```

The output directory must be fresh. Success is recorded in
`VALIDATION_PASSED.json`. `--only CASE` and `--skip-interface-audit` are
developer diagnostics: they write `VALIDATION_DIAGNOSTIC_ONLY.json`, cannot
issue release sign-off, and exit nonzero. Run `--help` for all options.

"Identical" here means exact replay on the *same* build, dependency set, MPI
layout, and hardware — not cross-platform bit-identity, which floating-point
MPI reductions do not provide and which the harness never claims.

`VALIDATION_CONTRACTS.md` lists the application-facing behaviours the run
asserts. See the main Vitriflow README (`../README.md`, "Validation" section)
for how this harness fits the overall validation model.
