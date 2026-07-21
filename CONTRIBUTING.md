# Contributing to Vitriflow

Vitriflow exists to produce amorphous-structure ensembles that are
**reproducible, auditable, and free of hidden behaviour**. Contributions are
welcome when they uphold those properties. This guide covers how to set up, what
a change is expected to preserve, and how to get it reviewed.

## Principles

Three properties govern every change:

- **Reproducible.** The same build, inputs, seeds, and host produce the same
  results. The two-pass validation harness asserts exact replay; a change that
  cannot pass it is not ready.
- **Auditable.** Results carry their provenance — structure manifests, engine and
  runtime identity, descriptor and graph-rule provenance, and SHA-256 artifact
  identity. A new output must record how it was produced.
- **Nothing hidden.** Behaviour is explicit and fails closed. New analysis is
  opt-in, legacy outputs are retained and labelled rather than silently changed,
  and undefined or unconverged results are reported, never fabricated.

The checklist at the end is the practical form of these.

## Development setup

Create the runtime environment and install in editable mode with the dev tools:

```bash
conda env create -f environment.yml
conda activate vitriflow
python -m pip install -e ".[dev]"
vitriflow --version
```

`bin/vitriflow` runs the CLI directly from the source tree. See `README.md` for
the CP2K and QUIP/GAP environment overlays.

## Tests

Run the full suite before opening a pull request:

```bash
pytest
```

Add tests with every behavioural change, and make them deterministic: fix seeds,
avoid wall-clock and network dependencies, and assert exact values where the code
guarantees them. A bug fix should include a test that fails without the fix.

## Style

Format, lint, and type-check with the tools from the `[dev]` extra:

```bash
black .
ruff check .
mypy vitriflow
```

- Match the surrounding code.
- Treat public CLI flags, output keys, and file names as an interface: keep them
  stable, or document a change to them.
- In prose and comments the product name is **Vitriflow** (lowercase `f`). The
  package, CLI, and entry point are lowercase `vitriflow`.

## Reproducibility and the validation harness

Any change that touches execution, parsing, analysis, plotting, convergence,
resume/HPC, or provenance must keep the application-validation harness green:

```bash
validation/run_vitriflow_application_validation.sh --workdir "$PWD/validation-run"
```

It runs each case twice with identical configs and seeds, requires exact
equality, and writes `VALIDATION_PASSED.json` only when every contract and both
passes succeed. "Identical" means exact replay on the same build, dependency set,
MPI layout, and host — not cross-platform bit-identity.

If a change alters results *by design*, say so explicitly in the pull request and
explain why the new output is correct. Do not weaken the harness or its contracts
to make a run pass.

## Provenance and compatibility

- Record provenance (source, rule, parameters, hashes) alongside every new result.
- Do not remove or silently repurpose an existing output field or file. Retain it
  and mark it as legacy or single-rule where a newer path supersedes it.
- New analysis behaviour is opt-in and must not change existing defaults.
- Fail closed: reject incomplete, non-converged, or unverifiable inputs with a
  clear error rather than guessing.

## Versioning and changelog

The version lives in `pyproject.toml` and must equal `vitriflow.__version__`
(a test enforces this). For a release:

- bump `version` in `pyproject.toml`;
- update `Current packaged release` and add a row to the **Version history** table
  in `README.md` (the release-consistency tests check these);
- keep every `dist/vitriflow-<version>-...whl` reference in `README.md` on the
  current version.

The `README.md` version-history table is the project changelog.

## Pull requests

Before requesting review, confirm:

- [ ] `pytest` passes and new behaviour has tests;
- [ ] `black`, `ruff`, and `mypy` are clean;
- [ ] the validation harness passes if execution, analysis, or provenance changed;
- [ ] every new output carries provenance and no existing field or file was
      silently changed;
- [ ] docs are updated (`README.md`, `docs/`, or the relevant sub-README);
- [ ] version and changelog are updated for a release.

Keep pull requests focused, describe what changed and why, and call out any
intended change in results.

## Questions and issues

Open issues and questions on the repository:
<https://github.com/Olsson-Materials-Modelling/Vitriflow>.

## License

By contributing you agree that your contributions are licensed under the
project's MIT License (see `LICENSE`).
