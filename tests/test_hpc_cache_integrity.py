from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _valid_task_and_result(tmp_path):
    import vitriflow.engine_identity as engine_identity
    from vitriflow.workflows import hpc

    box_dir = tmp_path / "production" / "box_001"
    input_snapshot = box_dir / "input" / "base.data"
    output = box_dir / "relax" / "relax.data"
    input_snapshot.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    input_snapshot.write_text("input-v1\n")
    output.write_text("output-v1\n")
    task_json = box_dir / "task.json"
    task_result = box_dir / "task_result.json"

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
            "task_json": str(task_json),
            "task_result": str(task_result),
        },
    }
    task_json.write_text(json.dumps(task, sort_keys=True))
    result = {
        "schema": hpc.TASK_RESULT_SCHEMA,
        "status": "ok",
        "box": 1,
        "engine_build_identity_end_verified": True,
        "task_manifest_sha256": hpc._task_manifest_digest(task),
        "engine_build_identity": engine_identity.query_engine_build_identities(
            SimpleNamespace(engine="lammps"),
            primary_engine="lammps",
            workdir=tmp_path,
        )["engines"]["lammps"],
        "outcomes": outcomes,
        "artifact_manifest": hpc._build_task_artifact_manifest(
            box_dir=box_dir,
            outcomes=outcomes,
        ),
    }
    result = hpc.seal_task_result(result)
    return hpc, task, result, input_snapshot, output


def test_hpc_success_cache_requires_current_inputs_and_artifacts(tmp_path):
    hpc, task, result, _input_snapshot, _output = _valid_task_and_result(tmp_path)

    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is True


def test_hpc_success_cache_rejects_modified_or_missing_artifact(tmp_path):
    hpc, task, result, _input_snapshot, output = _valid_task_and_result(tmp_path)

    output.write_text("output-v2\n")
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False

    output.unlink()
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False


def test_hpc_success_cache_rejects_changed_input_even_with_same_task_digest(tmp_path):
    hpc, task, result, input_snapshot, _output = _valid_task_and_result(tmp_path)

    # The serialized task did not change, so its canonical digest still matches;
    # content validation must independently reject the stale success.
    input_snapshot.write_text("input-v2\n")
    assert result["task_manifest_sha256"] == hpc._task_manifest_digest(task)
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False


def test_hpc_success_cache_rejects_truncated_artifact_manifest(tmp_path):
    hpc, task, result, _input_snapshot, _output = _valid_task_and_result(tmp_path)

    result["artifact_manifest"]["files"] = []
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False


def test_hpc_success_cache_binds_declared_diagnostic_artifacts(tmp_path):
    hpc, task, result, _input_snapshot, _output = _valid_task_and_result(tmp_path)
    box_dir = Path(task["task"]["box_dir"])
    summary = box_dir / "relax" / "metrics_timeseries.json"
    summary.write_text('{"status":"ok"}\n')
    diagnostics = {
        "schema": "vitriflow.production_task_diagnostics.v1",
        "status": "ok",
        "path_base": "task_box",
        "plan": {
            "schema": "vitriflow.production_task_diagnostic_plan.v1",
            "stage_metrics": {
                "enabled": True,
                "roles": ["relax"],
                "plot_required": False,
            },
            "elastic_screens": {"roles": {}},
            "elastic_timeseries": {"roles": {}},
        },
        "stage_metrics": {
            "relax": {
                "status": "ok",
                "csv": "relax/metrics_timeseries.csv",
                "summary": "relax/metrics_timeseries.json",
            }
        },
        "elastic_screens": {},
        "elastic_timeseries": {},
    }
    (box_dir / "relax" / "metrics_timeseries.csv").write_text(
        "Step,time\n0,0\n"
    )
    result["diagnostics"] = diagnostics
    result["artifact_manifest"] = hpc._build_task_artifact_manifest(
        box_dir=box_dir,
        outcomes=result["outcomes"],
        diagnostics=diagnostics,
    )
    result = hpc.seal_task_result(result)

    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is True
    summary.unlink()
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False


def test_hpc_success_cache_rejects_legacy_result_without_integrity_manifests(tmp_path):
    hpc, task, result, _input_snapshot, _output = _valid_task_and_result(tmp_path)

    result.pop("artifact_manifest")
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False


def test_hpc_success_cache_rejects_modified_result_metadata(tmp_path):
    hpc, task, result, _input_snapshot, _output = _valid_task_and_result(tmp_path)

    result["density"] = 9.99
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False


def test_delayed_stale_task_cannot_clobber_newer_success(tmp_path):
    import copy
    import json

    import pytest

    hpc, current_task, current_result, _input_snapshot, _output = (
        _valid_task_and_result(tmp_path)
    )
    box_dir = Path(current_task["task"]["box_dir"])
    result_path = box_dir / "task_result.json"
    result_path.write_text(json.dumps(current_result, sort_keys=True))
    before = result_path.read_bytes()

    stale_task = copy.deepcopy(current_task)
    stale_task["production_plan"] = {"seed_base": 12345}
    assert hpc._task_manifest_digest(stale_task) != hpc._task_manifest_digest(
        current_task
    )
    with pytest.raises(RuntimeError, match="disagrees with its canonical task.json"):
        hpc.execute_production_box_task(stale_task)
    assert result_path.read_bytes() == before


def test_output_analysis_rejects_corrupted_current_task_result(tmp_path):
    import json

    from vitriflow.workflows import hpc, output_analysis

    result_path = tmp_path / "task_result.json"
    result = hpc.seal_task_result(
        {
            "schema": hpc.TASK_RESULT_SCHEMA,
            "status": "ok",
            "box": 1,
            "metrics": {"density": 2.0},
            "distributions": {},
        }
    )
    result_path.write_text(json.dumps(result))
    result["metrics"]["density"] = 3.0
    result_path.write_text(json.dumps(result))

    import pytest

    with pytest.raises(ValueError, match="integrity validation failed"):
        output_analysis._load_task_result_entry(result_path)


@pytest.mark.parametrize(
    "invalid_box",
    [None, True, -1, 0.5, float("nan"), float("inf")],
)
def test_failed_task_result_requires_strict_box_identity(tmp_path, invalid_box):
    import json

    from vitriflow.workflows import output_analysis

    result_path = tmp_path / "task_result.json"
    result_path.write_text(
        json.dumps({"status": "failed", "box": invalid_box, "error": "expected"})
    )
    with pytest.raises(ValueError, match="box"):
        output_analysis._load_task_result_entry(result_path)


def test_failed_task_result_preserves_native_zero_box_identity(tmp_path):
    import json

    from vitriflow.workflows import output_analysis

    result_path = tmp_path / "task_result.json"
    result_path.write_text(
        json.dumps({"status": "failed", "box": 0, "error": "expected"})
    )
    entry, rejected = output_analysis._load_task_result_entry(result_path)
    assert entry is None
    assert rejected is not None
    assert rejected["box"] == 0


def test_hpc_input_manifest_rejects_null_or_blank_dependencies(tmp_path):
    import pytest
    from vitriflow.workflows import hpc

    snapshot = tmp_path / "input.data"
    potential = tmp_path / "potential.xml"
    snapshot.write_text("structure\n")
    potential.write_text("potential\n")
    config = SimpleNamespace(kim=SimpleNamespace(files=[]))
    with pytest.raises(ValueError, match="null entry"):
        hpc._task_input_manifest(
            config=config,
            plan={"potential_config": {"files": [None, str(potential)]}},
            input_snapshot=snapshot,
            base_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="blank entry"):
        hpc._task_input_manifest(
            config=config,
            plan={"potential_config": {"files": ["   ", str(potential)]}},
            input_snapshot=snapshot,
            base_dir=tmp_path,
        )


def test_hpc_cache_rejects_task_from_different_runtime(monkeypatch, tmp_path):
    hpc, task, result, _input_snapshot, _output = _valid_task_and_result(tmp_path)
    import vitriflow

    monkeypatch.setattr(vitriflow, "__version__", "999.0")
    assert hpc._cached_task_result_is_reusable(task_data=task, cached=result) is False


def test_hpc_manifest_hashes_command_referenced_potential_file(tmp_path):
    from vitriflow.workflows import hpc

    snapshot = tmp_path / "input.data"
    model = tmp_path / "gap.xml"
    snapshot.write_text("structure\n")
    model.write_text("model-v1\n")
    config = SimpleNamespace(engine="lammps", kim=SimpleNamespace(files=[]))
    plan = {
        "engine": "lammps",
        "potential_config": {"kind": "lammps", "files": [], "commands": []},
        "potential_lines": [f"pair_coeff * * {model.resolve()} C"],
    }

    manifest = hpc._task_input_manifest(
        config=config,
        plan=plan,
        input_snapshot=snapshot,
        base_dir=tmp_path,
    )
    assert [Path(item["path"]).name for item in manifest["dependencies"]] == ["gap.xml"]

    task = {
        "schema": "vitriflow.box_task.v1",
        "runtime": hpc._runtime_identity(),
        "input_manifest": manifest,
        "config": {},
        "production_plan": plan,
        "task": {"box": 1, "box_dir": str(tmp_path), "input_snapshot": str(snapshot)},
    }
    assert hpc._task_inputs_are_current(task) is True
    model.write_text("model-v2\n")
    assert hpc._task_inputs_are_current(task) is False


def test_hpc_manifest_rejects_unmaterialised_command_file(tmp_path):
    import pytest
    from vitriflow.workflows import hpc

    snapshot = tmp_path / "input.data"
    snapshot.write_text("structure\n")
    config = SimpleNamespace(engine="lammps", kim=SimpleNamespace(files=[]))
    plan = {
        "engine": "lammps",
        "potential_config": {"kind": "lammps", "files": [], "commands": []},
        "potential_lines": ["pair_coeff * * missing.xml C"],
    }
    with pytest.raises(FileNotFoundError, match="not materialised"):
        hpc._task_input_manifest(
            config=config,
            plan=plan,
            input_snapshot=snapshot,
            base_dir=tmp_path,
        )


def test_cp2k_hpc_manifest_binds_exact_resolved_data_files(tmp_path):
    from vitriflow.config import RunConfig
    from vitriflow.workflows import hpc

    data_dir = tmp_path / "cp2k-data"
    data_dir.mkdir()
    basis = data_dir / "BASIS_MOLOPT"
    potential = data_dir / "GTH_POTENTIALS"
    basis.write_text("basis-v1\n")
    potential.write_text("potential-v1\n")
    snapshot = tmp_path / "base.xyz"
    snapshot.write_text("structure\n")
    config = RunConfig.model_validate(
        {
            "engine": "cp2k",
            "cp2k": {
                "data_dir": str(data_dir),
                "kind_settings": {"H": {"basis_set": "DZVP-MOLOPT-SR-GTH", "potential": "GTH-PBE"}},
            },
            "structure": {"generate": {"method": "random", "formula": "H2", "n_formula_units": 1}},
            "autotune": {"metrics": {"type_to_species": ["H"], "pairs": [{"pair": ["H", "H"]}]}},
        }
    )
    plan = {"engine": "cp2k"}
    manifest = hpc._task_input_manifest(
        config=config,
        plan=plan,
        input_snapshot=snapshot,
        base_dir=tmp_path,
    )
    by_role = {item["role"]: item for item in manifest["cp2k_data_files"]}
    assert Path(by_role["basis_set"]["path"]) == basis.resolve()
    assert Path(by_role["potential"]["path"]) == potential.resolve()
    assert by_role["basis_set"]["sha256"] == hpc._sha256_file(basis)

    task = {
        "schema": "vitriflow.box_task.v1",
        "runtime": hpc._runtime_identity(),
        "input_manifest": manifest,
        "config": config.model_dump(mode="json"),
        "production_plan": plan,
        "task": {"box": 1, "box_dir": str(tmp_path), "input_snapshot": str(snapshot)},
    }
    assert hpc._task_inputs_are_current(task) is True
    basis.write_text("basis-v2\n")
    assert hpc._task_inputs_are_current(task) is False
