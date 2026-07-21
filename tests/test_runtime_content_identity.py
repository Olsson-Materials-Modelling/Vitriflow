from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _copy_package(tmp_path: Path, name: str) -> Path:
    import vitriflow

    source = Path(vitriflow.__file__).resolve().parent
    target = tmp_path / name / "vitriflow"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return target


def test_package_content_identity_is_path_independent_and_detects_source_mutation(
    tmp_path: Path,
):
    from vitriflow.runtime_identity import package_content_identity

    first_root = _copy_package(tmp_path, "editable")
    second_root = _copy_package(tmp_path, "wheel-layout")
    first = package_content_identity(first_root)
    second = package_content_identity(second_root)

    assert first == second
    assert first["file_count"] > 0

    changed = second_root / "workflows" / "run.py"
    changed.write_text(changed.read_text() + "\n# same-version source mutation fixture\n")
    mutated = package_content_identity(second_root)
    assert mutated["sha256"] != first["sha256"]


def test_package_content_identity_matches_wheel_when_undeclared_source_examples_are_absent(
    tmp_path: Path,
):
    """Editable-only legacy files must not alter the execution identity."""

    from vitriflow.runtime_identity import package_content_identity

    editable_root = _copy_package(tmp_path, "editable-with-legacy-examples")
    wheel_root = _copy_package(tmp_path, "wheel-with-declared-data-only")
    declared_examples = {
        "minimal_metal.yaml",
        "al_fcc_4x4x4.data",
        "si_diamond_cp2k_toy.yaml",
        "sio2_bks_zbl_smoke.yaml",
        "hc_C_GAP20Ugr_hc_custom_demo.yaml",
    }
    for candidate in (wheel_root / "examples").iterdir():
        if candidate.is_file() and candidate.name not in declared_examples:
            candidate.unlink()

    assert package_content_identity(editable_root) == package_content_identity(
        wheel_root
    )

    # A declared runtime input is still identity-bound.
    declared = wheel_root / "examples" / "minimal_metal.yaml"
    declared.write_text(declared.read_text() + "\n# mutation fixture\n")
    assert package_content_identity(editable_root) != package_content_identity(
        wheel_root
    )


def test_run_and_autotune_fingerprints_bind_package_content(monkeypatch, tmp_path: Path):
    from vitriflow.config import RunConfig
    from vitriflow import runtime_identity as identity_module
    from vitriflow.workflows import autotune as autotune_mod
    from vitriflow.workflows import custom_schedule as custom_mod
    from vitriflow.workflows import run as run_mod

    package_root = _copy_package(tmp_path, "runtime")
    monkeypatch.setattr(identity_module, "_DEFAULT_PACKAGE_ROOT", package_root)
    config = RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "MODEL_IDENTIFIER",
                "interactions": ["Al"],
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "Al",
                    "n_formula_units": 1,
                }
            },
            "autotune": {"metrics": {"type_to_species": ["Al"]}},
        }
    )
    structure = tmp_path / "base.data"
    structure.write_text("LAMMPS data file\n\n0 atoms\n")
    plan = {
        "structure_data": str(structure),
        "engine": "lammps",
        "potential_config": config.kim.model_dump(mode="json"),
        "potential_lines": None,
        "production_cfg": config.autotune.production.model_dump(mode="json"),
    }

    run_before = run_mod._build_run_resume_fingerprint(
        config=config,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    autotune_before = autotune_mod._build_autotune_resume_fingerprint(
        config=config,
        outdir=tmp_path,
        selected_structure=structure,
        production_plan=plan,
    )
    schedule = custom_mod.CustomSchedule(
        stages=(
            custom_mod.CustomStageConfig(
                name="melt",
                temperature_start_K=1200.0,
                temperature_stop_K=1200.0,
                steps=10,
                role="melt",
            ),
        ),
        analysis_roles={"melt": "melt", "quench": "melt", "relax": "melt"},
    )
    custom_before = custom_mod._build_resume_fingerprint(
        config=config,
        schedule=schedule,
        analysis_roles=schedule.analysis_roles,
        steps={"melt": 10},
        sched_report={"stages": [{"name": "melt", "steps": 10}]},
        time_unit_ps=1.0,
        md_pressure=0.0,
        lammps_units="metal",
        config_path=None,
    )

    changed = package_root / "workflows" / "production_common.py"
    changed.write_text(changed.read_text() + "\n# runtime identity mutation fixture\n")

    run_after = run_mod._build_run_resume_fingerprint(
        config=config,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    autotune_after = autotune_mod._build_autotune_resume_fingerprint(
        config=config,
        outdir=tmp_path,
        selected_structure=structure,
        production_plan=plan,
    )
    custom_after = custom_mod._build_resume_fingerprint(
        config=config,
        schedule=schedule,
        analysis_roles=schedule.analysis_roles,
        steps={"melt": 10},
        sched_report={"stages": [{"name": "melt", "steps": 10}]},
        time_unit_ps=1.0,
        md_pressure=0.0,
        lammps_units="metal",
        config_path=None,
    )

    assert run_before["sha256"] != run_after["sha256"]
    assert autotune_before["sha256"] != autotune_after["sha256"]
    assert custom_before["sha256"] != custom_after["sha256"]
    assert run_before["payload"]["runtime"]["schema"] == "vitriflow.runtime.v2"
    assert (
        autotune_before["payload"]["runtime"]["package_content"]["sha256"]
        != autotune_after["payload"]["runtime"]["package_content"]["sha256"]
    )


def test_fresh_and_resumed_autotune_share_terminal_input_mutation_gate(
    tmp_path: Path,
):
    import inspect

    from vitriflow.config import RunConfig
    from vitriflow.workflows import autotune as module

    potential = tmp_path / "potential.table"
    structure = tmp_path / "selected.data"
    potential.write_text("potential-v1\n")
    structure.write_text("structure-v1\n")
    config = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "interactions": ["Al"],
                "commands": [
                    "pair_style table linear 1000",
                    f"pair_coeff * * {potential} MODEL",
                ],
                "files": [str(potential)],
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "Al",
                    "n_formula_units": 1,
                }
            },
            "autotune": {"metrics": {"type_to_species": ["Al"]}},
        }
    )
    plan = {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": str(structure),
        "potential_config": config.kim.model_dump(mode="json"),
        "potential_lines": list(config.kim.commands),
    }
    before = module._build_autotune_resume_fingerprint(
        config=config,
        outdir=tmp_path,
        selected_structure=structure,
        production_plan=plan,
    )
    potential.write_text("potential-v2-mutated-during-execution\n")
    after = module._build_autotune_resume_fingerprint(
        config=config,
        outdir=tmp_path,
        selected_structure=structure,
        production_plan=plan,
    )

    with pytest.raises(RuntimeError, match="scientific input bytes changed"):
        module._assert_autotune_terminal_fingerprint_unchanged(
            before,
            after,
            context="execution",
        )
    assert "_assert_autotune_terminal_fingerprint_unchanged" in inspect.getsource(
        module._AutotuneWorkflow.run
    )
    assert "_assert_autotune_terminal_fingerprint_unchanged" in inspect.getsource(
        module._autotune_resume_from_results
    )


def test_hpc_cache_rejects_same_version_package_content_mutation(
    monkeypatch, tmp_path: Path, mock_engine_build_identities
):
    from vitriflow import runtime_identity as identity_module

    package_root = _copy_package(tmp_path, "hpc-runtime")
    monkeypatch.setattr(identity_module, "_DEFAULT_PACKAGE_ROOT", package_root)

    # Import after switching the content root; hpc's runtime helper delegates
    # to the neutral identity module on every cache validation.
    from vitriflow.workflows import hpc

    box_dir = tmp_path / "production" / "box_001"
    input_snapshot = box_dir / "input" / "base.data"
    output = box_dir / "relax" / "relax.data"
    input_snapshot.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    input_snapshot.write_text("input-v1\n")
    output.write_text("output-v1\n")
    outcomes = {"relax": {"output_data": "relax.data", "dump": None}}
    task = {
        "schema": "vitriflow.box_task.v1",
        "runtime": hpc._runtime_identity(),
        "input_manifest": {
            "schema": "vitriflow.task_inputs.v2",
            "structure_snapshot": hpc._file_identity(
                input_snapshot,
                recorded_path=str(input_snapshot.resolve()),
            ),
            "dependencies": [],
            "cp2k_data_files": [],
        },
        "task": {
            "box": 1,
            "box_dir": str(box_dir),
            "input_snapshot": str(input_snapshot),
        },
    }
    result = {
        "schema": hpc.TASK_RESULT_SCHEMA,
        "status": "ok",
        "box": 1,
        "engine_build_identity": mock_engine_build_identities["identities"]["lammps"],
        "engine_build_identity_end_verified": True,
        "task_manifest_sha256": hpc._task_manifest_digest(task),
        "outcomes": outcomes,
        "artifact_manifest": hpc._build_task_artifact_manifest(
            box_dir=box_dir,
            outcomes=outcomes,
        ),
    }
    result = hpc.seal_task_result(result)
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is True

    changed = package_root / "workflows" / "hpc.py"
    changed.write_text(changed.read_text() + "\n# same-version HPC runtime mutation fixture\n")
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False
