"""Preflight: CP2K stages must reject incomplete type_to_species mappings.

Targeted regression for review finding #1: an input structure carrying
species that do not appear in ``type_to_species`` used to slip through to the
best-effort artifact materializer where they became LAMMPS ``type=0``. The
check has been hoisted to a hard preflight that runs before any CP2K
invocation; this test pins that contract.
"""

from __future__ import annotations

import pytest


def test_validate_cp2k_species_coverage_accepts_subset():
    """type_to_species may legitimately list more species than the cell uses."""
    from vitriflow.workflows.stage_runner import _validate_cp2k_species_coverage

    # No exception expected.
    _validate_cp2k_species_coverage(["Si", "Si", "O"], ["Si", "O"])
    _validate_cp2k_species_coverage(["Si"], ["Si", "O", "N"])


def test_validate_cp2k_species_coverage_rejects_unknown_symbol():
    from vitriflow.workflows.stage_runner import _validate_cp2k_species_coverage

    with pytest.raises(ValueError) as excinfo:
        _validate_cp2k_species_coverage(["Si", "O", "Ge"], ["Si", "O"])

    msg = str(excinfo.value)
    assert "Ge" in msg
    assert "type_to_species" in msg
    # Make sure it points at the offending mapping, not a generic message.
    assert "Si" in msg and "O" in msg


def test_validate_cp2k_species_coverage_rejects_multiple_unknowns_alphabetically():
    from vitriflow.workflows.stage_runner import _validate_cp2k_species_coverage

    with pytest.raises(ValueError) as excinfo:
        _validate_cp2k_species_coverage(["Zn", "Mg", "Mg", "Zn"], ["Si"])

    # Sorted, deterministic listing so the message is stable.
    assert "['Mg', 'Zn']" in str(excinfo.value)


def test_cp2k_preflight_hoisted_outside_artifact_path_is_a_value_error():
    """The check is a ValueError, not an _ARTIFACT_EXPORT_EXCEPTIONS subtype.

    Although ValueError IS in _ARTIFACT_IO_EXCEPTIONS, the contract is that
    the check fires BEFORE the runner is invoked and is not wrapped in the
    best-effort try/except in `_materialize_cp2k_engine_neutral_outputs`.
    The materializer no longer carries a defensive raise that the
    surrounding handler would silently downgrade to a warning.
    """

    import inspect

    from vitriflow.workflows import stage_runner

    src = inspect.getsource(stage_runner._materialize_cp2k_engine_neutral_outputs)
    # Defensive raise must not live inside the materializer's best-effort block;
    # any ValueError that mentions type_to_species inside that function is a
    # regression of finding #1.
    assert "raise ValueError" not in src or "type_to_species" not in src, (
        "_materialize_cp2k_engine_neutral_outputs must not carry a species-coverage "
        "raise; that responsibility belongs to the hard preflight."
    )


def test_run_stage_local_cp2k_preflights_before_runner_invocation(tmp_path, monkeypatch):
    """End-to-end: when atoms carry a species missing from type_to_species,
    the CP2K stage must raise ValueError BEFORE runner.run is invoked.

    In current code paths ASE's own ``specorder`` check usually trips first
    when the data file has more atom types than ``type_to_species`` covers.
    The species-coverage preflight is the last line of defence for any path
    that produces an ``atoms`` object out-of-band (custom readers, future
    refactors). We monkey-patch the ASE reader so atoms come back with a
    symbol that is not in ``type_to_species`` and verify the preflight
    catches it without calling the runner.
    """
    pytest.importorskip("ase")

    from ase import Atoms
    import numpy as np

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig, MDConfig
    from vitriflow.lammps_input import StageSpec
    from vitriflow.runner import Cp2kRunner
    from vitriflow.workflows import stage_runner as sr

    data_path = tmp_path / "input.data"
    # Single-type structure so the ASE specorder check passes on the input.
    data_path.write_text(
        "LAMMPS data\n"
        "\n"
        "2 atoms\n"
        "1 atom types\n"
        "\n"
        "0.0 10.0 xlo xhi\n"
        "0.0 10.0 ylo yhi\n"
        "0.0 10.0 zlo zhi\n"
        "\n"
        "Masses\n"
        "\n"
        "1 28.0855\n"
        "\n"
        "Atoms\n"
        "\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 1.0 1.0 1.0\n"
    )

    stage = StageSpec(
        name="probe",
        input_data=data_path,
        output_data=tmp_path / "out.data",
        sample_ensemble=None,
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=1,
        velocity_mode="create",
        force_isotropic=False,
        replicate=None,
        write_dump=False,
        dump_every=1,
        tail_dump_frames=None,
        tail_dump_stride=None,
        msd_every=1,
        potential_lines=None,
    )

    md = MDConfig(timestep=1.0, ensemble="nvt")

    cfg = Cp2kConfig(
        exec="cp2k",
        kind_settings={
            "Si": Cp2kKindConfig(basis_set="DZVP-MOLOPT-SR-GTH", potential="GTH-PBE")
        },
    )
    runner = Cp2kRunner(cfg)

    runner_calls: list[object] = []

    def _explode(*args, **kwargs):
        runner_calls.append(("run", args, kwargs))
        raise AssertionError("Cp2kRunner.run must not be called when preflight fails")

    monkeypatch.setattr(runner, "run", _explode)
    stage_dir = tmp_path / "stage_dir"
    stage_dir.mkdir()
    (stage_dir / cfg.basis_set_file_name).write_text("")
    (stage_dir / cfg.potential_file_name).write_text("")
    monkeypatch.setattr(runner, "_ensure_data_files_present", lambda *_a, **_k: None)

    # Force the ASE read to return atoms whose chemical symbols are not all
    # covered by type_to_species. This simulates a future code path that
    # constructs atoms from a non-LAMMPS source (e.g. POSCAR, restart) where
    # specorder is no longer the gate. The species-coverage preflight is the
    # contract that protects every such path.
    def _fake_ase_read(*_a, **_k):
        atoms = Atoms(
            symbols=["Si", "Ge"],  # 'Ge' is the alien
            positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            cell=np.eye(3) * 10.0,
            pbc=True,
        )
        return atoms

    import vitriflow.workflows.stage_runner as sr_mod  # noqa: F401  (already imported above)
    # ase.io.read is imported INSIDE _run_stage_local_cp2k via:
    #   from ase.io import read as ase_read
    # so we monkey-patch the source binding in ase.io.
    import ase.io as _ase_io
    monkeypatch.setattr(_ase_io, "read", _fake_ase_read)

    with pytest.raises(ValueError) as excinfo:
        sr.run_stage_local(
            runner,
            None,
            md,
            stage,
            stage_dir,
            type_to_species=["Si"],  # 'Ge' from atoms is not in this list
        )

    msg = str(excinfo.value)
    assert "Ge" in msg
    assert "type_to_species" in msg
    assert runner_calls == [], "preflight must run before runner.run"
