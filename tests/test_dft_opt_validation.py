import json
from pathlib import Path

import pytest


def _cell_opt_identity_fixture(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _build_cp2k_cell_opt_calculation_identity,
    )

    parent = tmp_path / "relax.data"
    basis = tmp_path / "BASIS_SET"
    potential = tmp_path / "POTENTIAL"
    parent.write_text("parent structure v1\n")
    basis.write_text("basis v1\n")
    potential.write_text("potential v1\n")
    calculation = _build_cp2k_cell_opt_calculation_identity(
        parent_relax_data=parent,
        dft_config={"enabled": True, "optimizer": "LBFGS"},
        cp2k_config={"cp2k_cmd": "cp2k", "kind_settings": {"Si": {}}},
        cp2k_version=(2024, 2),
        basis_path=basis,
        potential_path=potential,
        base_input_text="&GLOBAL\n  RUN_TYPE CELL_OPT\n&END GLOBAL\n",
        external_pressure_bar=0.0,
        atom_style="atomic",
        type_to_species=["Si"],
        lammps_units_style="metal",
    )
    dft_dir = tmp_path / "dft_opt"
    dft_dir.mkdir()
    return parent, calculation, dft_dir


def test_cell_opt_input_publication_replaces_alias_without_clobbering_target(
    tmp_path: Path,
):
    from vitriflow.workflows.stage_runner import _atomic_write_cp2k_stage_input

    dft_dir = tmp_path / "dft_opt"
    dft_dir.mkdir()
    victim = tmp_path / "outside.inp"
    victim.write_text("outside must survive\n")
    input_path = dft_dir / "cell_opt.inp"
    try:
        input_path.symlink_to(victim)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    _atomic_write_cp2k_stage_input(input_path, "current input\n")

    assert victim.read_text() == "outside must survive\n"
    assert input_path.read_text() == "current input\n"
    assert not input_path.is_symlink()
    assert input_path.stat().st_nlink == 1


def _write_completed_cell_opt_fixture(dft_dir: Path, calculation) -> None:
    from vitriflow.workflows.autotune import (
        _write_cp2k_cell_opt_identity_manifest,
    )

    artifacts = {
        "input": dft_dir / "cell_opt.inp",
        "output": dft_dir / "cp2k.out",
        "scf_diagnostics": dft_dir / "cp2k_scf_diagnostics.json",
        "trajectory": dft_dir / "traj.dcd",
        "data": dft_dir / "dft_opt.data",
    }
    artifacts["input"].write_text("input\n")
    artifacts["output"].write_text("CELL OPTIMIZATION COMPLETED\n")
    artifacts["scf_diagnostics"].write_text("{}\n")
    artifacts["trajectory"].write_bytes(b"trajectory")
    artifacts["data"].write_text("refined structure\n")
    _write_cp2k_cell_opt_identity_manifest(
        dft_dir / "cell_opt_identity.json",
        calculation=calculation,
        status="completed",
        artifacts=artifacts,
    )


def test_dft_opt_requires_cp2k_block():
    """Dft opt requires."""
    from vitriflow.config import RunConfig

    cfg = {
        "engine": "lammps",
        "structure": {"generate": {"method": "random", "formula": "H2"}},
        "kim": {"model": "DUMMY", "interactions": ["H"]},
        "autotune": {
            "production": {
                "enabled": True,
                "dft_opt": {"enabled": True},
            }
        },
        # rest fill
    }

    with pytest.raises(ValueError, match=r"requires a 'cp2k:' block"):
        RunConfig.model_validate(cfg)


def test_dft_opt_requires_convergence_check():
    """Dft opt requires."""
    from vitriflow.config import RunConfig

    cfg = {
        "engine": "lammps",
        "structure": {"generate": {"method": "random", "formula": "H2"}},
        "kim": {"model": "DUMMY", "interactions": ["H"]},
        "cp2k": {
            "exec": "cp2k",
            "kind_settings": {
                "H": {"basis_set": "DZVP-MOLOPT-SR-GTH", "potential": "GTH-PBE"}
            },
        },
        "autotune": {
            "production": {
                "enabled": True,
                "check_convergence": False,
                "dft_opt": {"enabled": True},
            }
        },
    }

    with pytest.raises(ValueError, match=r"check_convergence=true"):
        RunConfig.model_validate(cfg)


@pytest.mark.parametrize("units", ["electron", "nano", "micro"])
def test_dft_opt_accepts_every_supported_dimensional_lammps_style(units: str):
    from vitriflow.config import RunConfig

    cfg = {
        "engine": "lammps",
        "structure": {"generate": {"method": "random", "formula": "H2"}},
        "kim": {"model": "DUMMY", "interactions": ["H"], "user_units": units},
        "cp2k": {
            "exec": "cp2k",
            "kind_settings": {
                "H": {"basis_set": "DZVP-MOLOPT-SR-GTH", "potential": "GTH-PBE"}
            },
        },
        "autotune": {
            "production": {
                "enabled": True,
                "check_convergence": True,
                "dft_opt": {"enabled": True},
            }
        },
    }
    parsed = RunConfig.model_validate(cfg)
    assert parsed.kim.user_units == units


def test_recovered_cell_opt_writes_unknown_version_scf_diagnostics(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _ensure_recovered_cp2k_cell_opt_scf_diagnostics,
    )

    output = tmp_path / "cp2k.out"
    output.write_text(
        "*** SCF run NOT converged ***\n"
        "*** GEOMETRY OPTIMIZATION COMPLETED ***\n"
    )
    diagnostics = tmp_path / "cp2k_scf_diagnostics.json"

    assert _ensure_recovered_cp2k_cell_opt_scf_diagnostics(output, diagnostics) == 1
    payload = json.loads(diagnostics.read_text())
    assert payload["schema"] == "vitriflow.cp2k_scf_diagnostics.v1"
    assert payload["cp2k_version"] is None
    assert payload["policy"] == "recovered_output_version_unknown"
    assert payload["recovered_from_existing_output"] is True
    assert payload["unconverged_scf_cycles"] == 1
    assert payload["outputs"] == [
        {
            "phase": "cell_optimization",
            "output": "cp2k.out",
            "unconverged_scf_cycles": 1,
        }
    ]


def test_recovered_cell_opt_preserves_matching_original_scf_provenance(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _ensure_recovered_cp2k_cell_opt_scf_diagnostics,
    )

    output = tmp_path / "cp2k.out"
    output.write_text("*** GEOMETRY OPTIMIZATION COMPLETED ***\n")
    diagnostics = tmp_path / "cp2k_scf_diagnostics.json"
    original = {
        "schema": "vitriflow.cp2k_scf_diagnostics.v1",
        "cp2k_version": "2024.1",
        "policy": "explicit_ignore_convergence_failure",
        "recovered_from_existing_output": False,
        "unconverged_scf_cycles": 0,
        "outputs": [
            {
                "phase": "cell_optimization",
                "output": "cp2k.out",
                "unconverged_scf_cycles": 0,
            }
        ],
    }
    diagnostics.write_text(json.dumps(original, sort_keys=True) + "\n")
    before = diagnostics.read_bytes()

    assert _ensure_recovered_cp2k_cell_opt_scf_diagnostics(output, diagnostics) == 0
    assert diagnostics.read_bytes() == before


def test_cell_opt_completed_reuse_requires_exact_calculation_and_artifacts(
    tmp_path: Path,
):
    from vitriflow.workflows.autotune import _resolve_cp2k_cell_opt_resume

    parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    _write_completed_cell_opt_fixture(dft_dir, calculation)

    decision = _resolve_cp2k_cell_opt_resume(
        dft_dir,
        calculation=calculation,
        allow_resume=True,
    )
    assert decision["mode"] == "completed"

    (dft_dir / "cp2k.out").write_text("changed output\n")
    with pytest.raises(RuntimeError, match="output artifact is missing or changed"):
        _resolve_cp2k_cell_opt_resume(
            dft_dir,
            calculation=calculation,
            allow_resume=True,
        )

    _write_completed_cell_opt_fixture(dft_dir, calculation)
    parent.write_text("different parent structure\n")
    # Rebuild against the changed parent while keeping all other calculation
    # inputs identical.  The stored completed result must not be reusable.
    from vitriflow.workflows.autotune import _build_cp2k_cell_opt_calculation_identity

    changed_calculation = _build_cp2k_cell_opt_calculation_identity(
        parent_relax_data=parent,
        dft_config={"enabled": True, "optimizer": "LBFGS"},
        cp2k_config={"cp2k_cmd": "cp2k", "kind_settings": {"Si": {}}},
        cp2k_version=(2024, 2),
        basis_path=tmp_path / "BASIS_SET",
        potential_path=tmp_path / "POTENTIAL",
        base_input_text="&GLOBAL\n  RUN_TYPE CELL_OPT\n&END GLOBAL\n",
        external_pressure_bar=0.0,
        atom_style="atomic",
        type_to_species=["Si"],
        lammps_units_style="metal",
    )
    with pytest.raises(RuntimeError, match="parent structure, configuration"):
        _resolve_cp2k_cell_opt_resume(
            dft_dir,
            calculation=changed_calculation,
            allow_resume=True,
        )


def test_cell_opt_resume_rejects_hardlinked_identity_manifest(tmp_path: Path):
    from vitriflow.workflows.autotune import _resolve_cp2k_cell_opt_resume

    _parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    _write_completed_cell_opt_fixture(dft_dir, calculation)
    manifest = dft_dir / "cell_opt_identity.json"
    victim = tmp_path / "outside-identity.json"
    victim.write_bytes(manifest.read_bytes())
    before = victim.read_bytes()
    manifest.unlink()
    try:
        manifest.hardlink_to(victim)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"hard-link creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="hard-linked"):
        _resolve_cp2k_cell_opt_resume(
            dft_dir,
            calculation=calculation,
            allow_resume=True,
        )
    assert victim.read_bytes() == before


def test_cell_opt_no_resume_never_consumes_and_clears_stale_artifacts(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _clear_cp2k_cell_opt_artifacts,
        _resolve_cp2k_cell_opt_resume,
    )

    _parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    _write_completed_cell_opt_fixture(dft_dir, calculation)
    restart = dft_dir / "dft_opt-1.restart"
    restart.write_text("stale restart\n")
    wfn = dft_dir / "dft_opt-RESTART.wfn"
    wfn.write_text("stale wavefunction\n")
    wfn_backup = dft_dir / "dft_opt-RESTART.wfn.bak-1"
    wfn_backup.write_text("stale backup wavefunction\n")
    outside_wfn = tmp_path / "outside.wfn"
    outside_wfn.write_text("outside target must survive\n")
    wfn_symlink = dft_dir / "dft_opt-RESTART.wfn.bak-2"
    try:
        wfn_symlink.symlink_to(outside_wfn)
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    decision = _resolve_cp2k_cell_opt_resume(
        dft_dir,
        calculation=calculation,
        allow_resume=False,
    )
    assert decision == {"mode": "fresh", "reason": "resume_not_requested"}
    _clear_cp2k_cell_opt_artifacts(dft_dir)
    assert not (dft_dir / "dft_opt.data").exists()
    assert not (dft_dir / "cell_opt_identity.json").exists()
    assert not restart.exists()
    assert not wfn.exists()
    assert not wfn_backup.exists()
    assert not wfn_symlink.exists()
    assert outside_wfn.read_text() == "outside target must survive\n"


def test_cell_opt_restart_requires_exact_failed_manifest_binding(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _resolve_cp2k_cell_opt_resume,
        _write_cp2k_cell_opt_identity_manifest,
    )

    _parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    inp = dft_dir / "cell_opt.inp"
    output = dft_dir / "cp2k.out"
    restart = dft_dir / "dft_opt-1.restart"
    inp.write_text("input\n")
    output.write_text("incomplete output\n")
    restart.write_text("verified restart\n")
    _write_cp2k_cell_opt_identity_manifest(
        dft_dir / "cell_opt_identity.json",
        calculation=calculation,
        status="failed",
        artifacts={"input": inp, "output": output},
        restart_paths=[restart],
    )

    decision = _resolve_cp2k_cell_opt_resume(
        dft_dir,
        calculation=calculation,
        allow_resume=True,
    )
    assert decision["mode"] == "restart"
    assert decision["restart"] == restart

    unbound = dft_dir / "dft_opt-2.restart"
    unbound.write_text("stale unrelated restart\n")
    with pytest.raises(RuntimeError, match="unbound or missing restart"):
        _resolve_cp2k_cell_opt_resume(
            dft_dir,
            calculation=calculation,
            allow_resume=True,
        )
    unbound.unlink()
    restart.write_text("tampered restart\n")
    with pytest.raises(RuntimeError, match="restart artifact .* changed"):
        _resolve_cp2k_cell_opt_resume(
            dft_dir,
            calculation=calculation,
            allow_resume=True,
        )


def test_cell_opt_restart_selection_uses_numeric_cp2k_index(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _resolve_cp2k_cell_opt_resume,
        _write_cp2k_cell_opt_identity_manifest,
    )

    _parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    inp = dft_dir / "cell_opt.inp"
    inp.write_text("input\n")
    restart_2 = dft_dir / "dft_opt-2.restart"
    restart_10 = dft_dir / "dft_opt-10.restart"
    restart_2.write_text("older authenticated restart\n")
    restart_10.write_text("newer authenticated restart\n")
    _write_cp2k_cell_opt_identity_manifest(
        dft_dir / "cell_opt_identity.json",
        calculation=calculation,
        status="failed",
        artifacts={"input": inp},
        restart_paths=[restart_2, restart_10],
    )

    decision = _resolve_cp2k_cell_opt_resume(
        dft_dir,
        calculation=calculation,
        allow_resume=True,
    )
    assert decision["mode"] == "restart"
    assert decision["restart"] == restart_10
    assert decision["restart_index"] == 10


def test_cell_opt_restart_manifest_rejects_symlink_and_hardlink_aliases(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.autotune import (
        _write_cp2k_cell_opt_identity_manifest,
    )

    _parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    inp = dft_dir / "cell_opt.inp"
    inp.write_text("input\n")
    victim = tmp_path / "outside.restart"
    victim.write_text("outside bytes must not be consumed or changed\n")
    before = victim.read_bytes()

    symlink_restart = dft_dir / "dft_opt-1.restart"
    try:
        symlink_restart.symlink_to(victim)
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(RuntimeError, match="non-symlink"):
        _write_cp2k_cell_opt_identity_manifest(
            dft_dir / "cell_opt_identity.json",
            calculation=calculation,
            status="failed",
            artifacts={"input": inp},
            restart_paths=[symlink_restart],
        )
    assert victim.read_bytes() == before
    symlink_restart.unlink()

    hardlink_restart = dft_dir / "dft_opt-1.restart"
    try:
        hardlink_restart.hardlink_to(victim)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"hard-link creation unavailable: {exc}")
    with pytest.raises(RuntimeError, match="hard-linked"):
        _write_cp2k_cell_opt_identity_manifest(
            dft_dir / "cell_opt_identity.json",
            calculation=calculation,
            status="failed",
            artifacts={"input": inp},
            restart_paths=[hardlink_restart],
        )
    assert victim.read_bytes() == before


def test_cell_opt_resume_rejects_artifact_replaced_by_identical_symlink(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.autotune import _resolve_cp2k_cell_opt_resume

    _parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    _write_completed_cell_opt_fixture(dft_dir, calculation)
    output = dft_dir / "cp2k.out"
    outside = tmp_path / "outside.out"
    outside.write_bytes(output.read_bytes())
    output.unlink()
    try:
        output.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="output artifact is missing or changed"):
        _resolve_cp2k_cell_opt_resume(
            dft_dir,
            calculation=calculation,
            allow_resume=True,
        )
    assert outside.read_text() == "CELL OPTIMIZATION COMPLETED\n"


def test_cell_opt_restart_manifest_rejects_noncanonical_name(tmp_path: Path):
    from vitriflow.workflows.autotune import _write_cp2k_cell_opt_identity_manifest

    _parent, calculation, dft_dir = _cell_opt_identity_fixture(tmp_path)
    inp = dft_dir / "cell_opt.inp"
    inp.write_text("input\n")
    malformed = dft_dir / "dft_opt-latest.restart"
    malformed.write_text("ambiguous restart\n")

    with pytest.raises(RuntimeError, match="positive integer"):
        _write_cp2k_cell_opt_identity_manifest(
            dft_dir / "cell_opt_identity.json",
            calculation=calculation,
            status="failed",
            artifacts={"input": inp},
            restart_paths=[malformed],
        )


def test_interrupted_running_cell_opt_is_quarantined_for_fresh_rerun(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _quarantine_interrupted_cp2k_cell_opt,
        _resolve_cp2k_cell_opt_resume,
        _write_cp2k_cell_opt_identity_manifest,
    )

    result_root = tmp_path / "result"
    dft_dir = result_root / "production" / "box_001" / "dft_opt"
    dft_dir.mkdir(parents=True)
    identity_root = tmp_path / "identity"
    identity_root.mkdir()
    _parent, calculation, fixture_dir = _cell_opt_identity_fixture(identity_root)
    fixture_dir.rmdir()
    inp = dft_dir / "cell_opt.inp"
    inp.write_text("input\n")
    _write_cp2k_cell_opt_identity_manifest(
        dft_dir / "cell_opt_identity.json",
        calculation=calculation,
        status="running",
        artifacts={"input": inp},
    )
    uncommitted_restart = dft_dir / "dft_opt-1.restart"
    uncommitted_restart.write_text("created after running manifest\n")

    decision = _resolve_cp2k_cell_opt_resume(
        dft_dir,
        calculation=calculation,
        allow_resume=True,
    )
    assert decision["mode"] == "recover_fresh"
    quarantined = _quarantine_interrupted_cp2k_cell_opt(
        dft_dir,
        result_root=result_root,
    )
    assert dft_dir.is_dir()
    assert list(dft_dir.iterdir()) == []
    assert (quarantined / "cell_opt.inp").read_text() == "input\n"
    assert (quarantined / "dft_opt-1.restart").read_text() == (
        "created after running manifest\n"
    )
    assert _resolve_cp2k_cell_opt_resume(
        dft_dir,
        calculation=calculation,
        allow_resume=True,
    ) == {"mode": "fresh", "reason": "no_prior_artifacts"}


def test_orphan_cell_opt_input_is_quarantined_for_fresh_rerun(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _quarantine_interrupted_cp2k_cell_opt,
        _resolve_cp2k_cell_opt_resume,
    )

    result_root = tmp_path / "result"
    dft_dir = result_root / "production" / "box_001" / "dft_opt"
    dft_dir.mkdir(parents=True)
    orphan = dft_dir / "cell_opt.inp"
    orphan.write_text("uncommitted input\n")
    identity_root = tmp_path / "identity"
    identity_root.mkdir()
    _parent, calculation, fixture_dir = _cell_opt_identity_fixture(identity_root)
    fixture_dir.rmdir()

    decision = _resolve_cp2k_cell_opt_resume(
        dft_dir,
        calculation=calculation,
        allow_resume=True,
    )
    assert decision["mode"] == "recover_fresh"
    quarantined = _quarantine_interrupted_cp2k_cell_opt(
        dft_dir,
        result_root=result_root,
    )
    assert not orphan.exists()
    assert (quarantined / "cell_opt.inp").read_text() == "uncommitted input\n"
    assert dft_dir.is_dir() and list(dft_dir.iterdir()) == []


def test_cell_opt_quarantine_rejects_internal_symlink_without_following_it(
    tmp_path: Path,
):
    from vitriflow.workflows.autotune import _quarantine_interrupted_cp2k_cell_opt

    result_root = tmp_path / "result"
    box_dir = result_root / "production" / "box_001"
    box_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_file = victim / "keep.txt"
    victim_file.write_text("must survive\n")
    dft_dir = box_dir / "dft_opt"
    try:
        dft_dir.symlink_to(victim, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="symbolic links"):
        _quarantine_interrupted_cp2k_cell_opt(
            dft_dir,
            result_root=result_root,
        )
    assert dft_dir.is_symlink()
    assert victim_file.read_text() == "must survive\n"


def test_cell_opt_quarantine_accepts_user_facing_result_root_alias(tmp_path: Path):
    from vitriflow.workflows.autotune import _quarantine_interrupted_cp2k_cell_opt

    real_root = tmp_path / "real_result"
    real_dft = real_root / "production" / "box_001" / "dft_opt"
    real_dft.mkdir(parents=True)
    (real_dft / "cell_opt.inp").write_text("uncommitted input\n")
    alias = tmp_path / "scratch_alias"
    try:
        alias.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    alias_dft = alias / "production" / "box_001" / "dft_opt"
    quarantined = _quarantine_interrupted_cp2k_cell_opt(
        alias_dft,
        result_root=alias,
    )
    assert alias_dft.is_dir() and list(alias_dft.iterdir()) == []
    assert (quarantined / "cell_opt.inp").read_text() == "uncommitted input\n"
    assert quarantined.is_relative_to(real_root)


def test_cell_opt_quarantine_rejects_symlinked_quarantine_parent(tmp_path: Path):
    from vitriflow.workflows.autotune import _quarantine_interrupted_cp2k_cell_opt

    result_root = tmp_path / "result"
    dft_dir = result_root / "production" / "box_001" / "dft_opt"
    dft_dir.mkdir(parents=True)
    (dft_dir / "cell_opt.inp").write_text("uncommitted input\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (result_root / "interrupted_attempts").symlink_to(
            outside,
            target_is_directory=True,
        )
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="quarantine must not contain symbolic links"):
        _quarantine_interrupted_cp2k_cell_opt(
            dft_dir,
            result_root=result_root,
        )
    assert (dft_dir / "cell_opt.inp").read_text() == "uncommitted input\n"
    assert list(outside.iterdir()) == []


def test_dft_coordination_rejection_reconciliation_is_idempotent() -> None:
    from vitriflow.workflows.autotune import _reconcile_dft_coordination_rejections

    rows = [
        {"box": 1, "reason": "coordination_defects_dft"},
        {"box": 1, "reason": "coordination_defects_dft"},
        {"box": 3, "reason": "cp2k_failed", "error": "kept"},
    ]
    boxes = [
        {
            "box": 1,
            "dft_opt": {
                "status": "ok",
                "has_coordination_defects": True,
            },
        },
        {
            "box": 2,
            "dft_opt": {
                "status": "ok",
                "has_coordination_defects": False,
            },
        },
    ]

    once = _reconcile_dft_coordination_rejections(
        rows,
        boxes,
        exclude_defects=True,
    )
    twice = _reconcile_dft_coordination_rejections(
        once,
        boxes,
        exclude_defects=True,
    )
    assert once == twice
    assert once == [
        {"box": 3, "reason": "cp2k_failed", "error": "kept"},
        {"box": 1, "reason": "coordination_defects_dft"},
    ]


def test_refined_density_uses_identical_serialized_source_for_live_and_replay(
    tmp_path: Path,
):
    from ase import Atoms

    from vitriflow.cp2k_driver import density_g_cm3_from_atoms
    from vitriflow.structuregen import write_lammps_data
    from vitriflow.workflows.output_analysis import _estimate_density_from_source

    atoms = Atoms(
        "Si2",
        positions=[[0.123456789012345, 0.2, 0.3], [2.1, 2.2, 2.3]],
        cell=[
            [5.431234567890123, 0.0, 0.0],
            [0.123456789012345, 5.432345678901234, 0.0],
            [0.234567890123456, 0.345678901234567, 5.433456789012345],
        ],
        pbc=True,
    )
    data = tmp_path / "dft_opt.data"
    write_lammps_data(
        data,
        atoms,
        atom_style="atomic",
        specorder=["Si"],
        units_style="metal",
    )
    from vitriflow.io.lammps_data_minimal import read_lammps_data_minimal

    canonical = read_lammps_data_minimal(
        data,
        atom_style="atomic",
        specorder=["Si"],
        units_style="metal",
    )
    live_density = density_g_cm3_from_atoms(canonical)
    replay_density = _estimate_density_from_source(
        data,
        type_to_species=["Si"],
        atom_style="atomic",
        lammps_units_style="metal",
    )
    assert replay_density is not None
    assert live_density == replay_density
