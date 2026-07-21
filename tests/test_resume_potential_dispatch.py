from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _protected_table_lines(filename: str, payload: bytes) -> list[str]:
    metadata = {
        "version": 10,
        "filename": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_relpath": f"preflight/potential_override/{filename}",
        "pairs": [],
    }
    head = {key: value for key, value in metadata.items() if key != "pairs"}
    return [
        "# vitriflow_core_table_begin "
        + json.dumps(head, sort_keys=True, separators=(",", ":")),
        "# vitriflow_core_table_end",
        "pair_style table linear 2000",
        f"pair_coeff 1 1 {filename} P1_1 10",
    ]


def test_potential_kind_is_the_only_kim_install_discriminator() -> None:
    from vitriflow.kim import is_kim_potential

    assert is_kim_potential({"kind": "kim", "model": "MODEL"})
    assert not is_kim_potential({"kind": "lammps", "model": "misleading"})
    assert not is_kim_potential({"kind": "mg2_sin"})
    assert not is_kim_potential(None)


def test_explicit_kim_install_failure_is_fatal_and_diagnostic() -> None:
    from vitriflow.kim import (
        KimInstallResult,
        ensure_potential_model_installed,
    )

    model = "Model__MO_123456789012_001"

    def _failed_installer(requested: str) -> KimInstallResult:
        assert requested == model
        return KimInstallResult(
            attempted=True,
            success=False,
            stdout="collection lookup failed",
            stderr="model unavailable",
        )

    with pytest.raises(RuntimeError, match="KIM model installation failed") as excinfo:
        ensure_potential_model_installed(
            {"kind": "kim", "model": model},
            installer=_failed_installer,
        )

    message = str(excinfo.value)
    assert model in message
    assert "model unavailable" in message
    assert "collection lookup failed" in message


@pytest.mark.parametrize(
    "potential",
    [
        {"kind": "lammps", "model": "misleading"},
        {"kind": "mg2_sin", "model": "misleading"},
        {"kind": "cp2k", "model": "misleading"},
        None,
    ],
)
def test_non_kim_potentials_never_invoke_kim_installer(potential) -> None:
    from vitriflow.kim import ensure_potential_model_installed

    def _must_not_run(_model: str):
        raise AssertionError("non-KIM potential invoked KIM installer")

    assert (
        ensure_potential_model_installed(potential, installer=_must_not_run)
        is None
    )


def test_kim_install_test_double_none_remains_supported() -> None:
    from vitriflow.kim import ensure_potential_model_installed

    assert (
        ensure_potential_model_installed(
            {"kind": "kim", "model": "Model__MO_123456789012_001"},
            installer=lambda _model: None,
        )
        is None
    )


def test_malformed_kim_installer_result_fails_closed() -> None:
    from vitriflow.kim import ensure_potential_model_installed

    with pytest.raises(RuntimeError, match="invalid result"):
        ensure_potential_model_installed(
            {"kind": "kim", "model": "Model__MO_123456789012_001"},
            installer=lambda _model: {"success": True},
        )


def test_all_workflow_kim_install_sites_use_the_checked_gate() -> None:
    """Prevent a future direct installer call from bypassing fail-closed policy."""

    import ast
    import inspect

    from vitriflow.workflows import autotune, custom_schedule, hpc, run

    expected_checked_calls = {
        run: 1,
        autotune: 2,
        custom_schedule: 1,
        hpc: 1,
    }
    for module, expected in expected_checked_calls.items():
        tree = ast.parse(inspect.getsource(module))
        checked_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ensure_potential_model_installed"
        ]
        unchecked_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ensure_model_installed"
        ]
        assert len(checked_calls) == expected
        assert unchecked_calls == []


def test_enabled_zbl_table_is_generated_without_legacy_tabulate_flag(tmp_path: Path) -> None:
    from vitriflow.workflows.resume_integrity import potential_command_file_paths

    core = {
        "enabled": True,
        "style": "zbl",
        "tabulate": False,
        "table_filename": "buck_core.table",
    }
    potential = {
        "kind": "lammps",
        "commands": [
            "pair_style table linear 2000",
            "pair_coeff 1 1 buck_core.table P1_1 10",
        ],
        "core_repulsion": core,
    }
    assert potential_command_file_paths(
        potential=potential,
        plan={"core_repulsion": core},
        declared_values=[],
        base_dir=tmp_path,
    ) == []


def test_hybrid_substyles_are_not_misclassified_as_paths(tmp_path: Path) -> None:
    from vitriflow.workflows.resume_integrity import potential_command_file_paths

    commands = [
        "pair_style hybrid/overlay coul/long 15.0 buck 15.0 morse 15.0",
        "pair_coeff 1 1 coul/long",
        "pair_coeff 1 1 buck 139349.01 0.21 171.08",
        "pair_coeff 1 2 coul/long",
        "pair_coeff 1 2 buck 412.55 0.3 0",
        "pair_coeff 1 2 morse 0.44 2.57 1.91",
        "pair_coeff 2 2 coul/long",
        "pair_coeff 2 2 buck 1388.77 0.36 175.00",
    ]
    potential = {"kind": "lammps", "commands": commands}
    assert potential_command_file_paths(
        potential=potential,
        plan={},
        declared_values=[],
        base_dir=tmp_path,
    ) == []

    with pytest.raises(FileNotFoundError, match="missing.table"):
        potential_command_file_paths(
            potential={
                "kind": "lammps",
                "commands": commands[:1]
                + ["pair_coeff 1 1 table missing.table P1_1 15"],
            },
            plan={},
            declared_values=[],
            base_dir=tmp_path,
        )

    # A later effective non-hybrid override must reset the parser state; its
    # token[3] is a real filename rather than a hybrid sub-style.
    with pytest.raises(FileNotFoundError, match="external.table"):
        potential_command_file_paths(
            potential={"kind": "lammps", "commands": commands},
            plan={
                "potential_lines": [
                    "pair_style table linear 2000",
                    "pair_coeff 1 1 external.table P1_1 15",
                ]
            },
            declared_values=[],
            base_dir=tmp_path,
        )


def test_protected_autocore_table_is_portably_staged_and_authenticated(
    tmp_path: Path,
) -> None:
    from vitriflow.potential import (
        stage_validated_tabulated_core_for_replay,
        validated_tabulated_core_path,
    )

    payload = b"LAMMPS table\n\nP1_1\nN 1\n\n1 0.1 1.0 1.0\n"
    lines = _protected_table_lines("core.table", payload)
    source_root = tmp_path / "source"
    source = source_root / "preflight" / "potential_override" / "core.table"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)

    target_root = tmp_path / "target"
    target = stage_validated_tabulated_core_for_replay(
        lines, source_root=source_root, target_root=target_root
    )
    assert target == target_root / "preflight" / "potential_override" / "core.table"
    assert target.read_bytes() == payload
    assert validated_tabulated_core_path(lines, root=target_root) == target

    source.write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        stage_validated_tabulated_core_for_replay(
            lines, source_root=source_root, target_root=tmp_path / "other"
        )


def test_run_and_hpc_fingerprints_bind_realized_autocore_table(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from vitriflow.config import RunConfig
    from vitriflow.workflows.hpc import _task_input_manifest
    from vitriflow.workflows.production_common import (
        make_production_plan,
        production_plan_to_dict,
    )
    from vitriflow.workflows.run import _build_run_resume_fingerprint

    payload = b"verified table bytes\n"
    lines = _protected_table_lines("core.table", payload)
    table = tmp_path / "preflight" / "potential_override" / "core.table"
    table.parent.mkdir(parents=True)
    table.write_bytes(payload)
    structure = tmp_path / "base.data"
    structure.write_text("LAMMPS data file\n\n0 atoms\n")
    snapshot = tmp_path / "snapshot.data"
    snapshot.write_text("LAMMPS data file\n\n0 atoms\n")
    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si"],
                "commands": ["pair_style buck 10", "pair_coeff 1 1 1 1 1"],
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "table_filename": "core.table",
                },
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "Si",
                    "n_formula_units": 1,
                }
            },
            "autotune": {"metrics": {"enabled": True, "type_to_species": ["Si"]}},
        }
    )
    plan = make_production_plan(
        engine="lammps",
        structure_data=structure,
        T_high=310.0,
        high_total_steps=1,
        t_final=300.0,
        chosen_rate=10.0,
        cooling_rate_ps=10.0,
        replicate=[1, 1, 1],
        pressure=cfg.md.pressure,
        md_use=cfg.md.model_dump(mode="json"),
        potential_config=cfg.kim.model_dump(mode="json"),
        potential_lines=lines,
        core_repulsion=cfg.kim.core_repulsion.model_dump(mode="json"),
        type_to_species=["Si"],
        metrics_cfg=cfg.autotune.metrics.model_dump(mode="json"),
        effective_metrics={"enabled": True},
        production_cfg=cfg.autotune.production.model_dump(mode="json"),
        convergence_cfg=cfg.autotune.convergence.model_dump(mode="json"),
        cutoffs_rate={},
        cutoffs_size={},
        preferred_cutoffs={},
        quench_steps=1,
        relax_steps=cfg.autotune.quench.relax_steps,
        msd_every=cfg.autotune.tm_scan.msd_every,
        seed_base=cfg.random_seed,
        time_unit_ps=1.0,
        sampling_hint=None,
        execution_mode="fixed",
    )
    plan_dict = production_plan_to_dict(plan, relative_to=tmp_path)
    fingerprint = _build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan_dict,
        outdir=tmp_path,
        external_mode="local",
    )
    generated = fingerprint["payload"]["input_identities"][
        "generated_potential_files"
    ]
    assert len(generated) == 1
    assert generated[0]["sha256"] == hashlib.sha256(payload).hexdigest()

    # Public callers commonly supply a relative output directory.  Identity
    # construction must anchor every protected input to its canonical result
    # root rather than raising during relative-to checks or hashing a path from
    # the caller's current directory.
    monkeypatch.chdir(tmp_path.parent)
    relative_fingerprint = _build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan_dict,
        outdir=Path(tmp_path.name),
        external_mode="local",
    )
    assert relative_fingerprint["payload"]["input_identities"][
        "generated_potential_files"
    ][0]["sha256"] == hashlib.sha256(payload).hexdigest()

    task_manifest = _task_input_manifest(
        config=cfg,
        plan=plan_dict,
        input_snapshot=snapshot,
        base_dir=tmp_path,
    )
    assert any(
        entry["sha256"] == hashlib.sha256(payload).hexdigest()
        for entry in task_manifest["dependencies"]
    )

    table.write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        _build_run_resume_fingerprint(
            config=cfg,
            production_plan=plan_dict,
            outdir=tmp_path,
            external_mode="local",
        )


def test_run_and_hpc_use_the_protected_plan_potential_as_authoritative(
    tmp_path: Path,
) -> None:
    from vitriflow.config import RunConfig
    from vitriflow.workflows.hpc import _task_input_manifest
    from vitriflow.workflows.run import _build_run_resume_fingerprint

    plan_file = tmp_path / "plan.table"
    config_file = tmp_path / "config.table"
    plan_file.write_bytes(b"plan-potential-v1\n")
    config_file.write_bytes(b"config-potential-v1\n")
    structure = tmp_path / "base.data"
    snapshot = tmp_path / "snapshot.data"
    structure.write_text("LAMMPS data file\n\n0 atoms\n")
    snapshot.write_text("LAMMPS data file\n\n0 atoms\n")

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "interactions": ["Si"],
                "files": [str(config_file)],
                "commands": [
                    "pair_style table linear 1000",
                    "pair_coeff 1 1 config.table CONFIG 5.0",
                ],
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "Si",
                    "n_formula_units": 1,
                }
            },
        }
    )
    protected_potential = {
        "kind": "lammps",
        "user_units": "metal",
        "interactions": ["Si"],
        "files": [str(plan_file)],
        "commands": [
            "pair_style table linear 1000",
            "pair_coeff 1 1 plan.table PLAN 5.0",
        ],
        "core_repulsion": {"enabled": False},
    }
    plan = {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": str(structure),
        "potential_config": protected_potential,
        "potential_lines": protected_potential["commands"],
        "production_cfg": {},
    }

    fingerprint = _build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    run_hashes = {
        row["sha256"]
        for row in fingerprint["payload"]["input_identities"]["potential_files"]
    }
    assert run_hashes == {hashlib.sha256(plan_file.read_bytes()).hexdigest()}

    task_manifest = _task_input_manifest(
        config=cfg,
        plan=plan,
        input_snapshot=snapshot,
        base_dir=tmp_path,
    )
    task_hashes = {
        row["sha256"] for row in task_manifest["dependencies"]
    }
    assert hashlib.sha256(plan_file.read_bytes()).hexdigest() in task_hashes
    assert hashlib.sha256(config_file.read_bytes()).hexdigest() not in task_hashes

    # An absent protected potential is the only case in which the validated
    # current config is the execution source and therefore must be bound.
    fallback_plan = {key: value for key, value in plan.items() if key != "potential_config"}
    fallback_plan["potential_lines"] = None
    fallback_fingerprint = _build_run_resume_fingerprint(
        config=cfg,
        production_plan=fallback_plan,
        outdir=tmp_path,
        external_mode="local",
    )
    fallback_hashes = {
        row["sha256"]
        for row in fallback_fingerprint["payload"]["input_identities"]["potential_files"]
    }
    assert fallback_hashes == {hashlib.sha256(config_file.read_bytes()).hexdigest()}
