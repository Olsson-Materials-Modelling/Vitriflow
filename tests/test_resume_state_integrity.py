from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _production_state(tmp_path: Path) -> tuple[dict, Path]:
    from vitriflow.workflows.resume_integrity import attach_production_state_integrity

    source = tmp_path / "production" / "box_001" / "relax.data"
    source.parent.mkdir(parents=True)
    source.write_text("structure-v1\n")
    snapshot = source.parent / "structure_snapshot.json"
    manifest = source.parent / "structure_manifest.json"
    snapshot.write_text(json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1}))
    manifest_row = {
        "structure_hash": "structure-1",
        "cell_hash": "cell-1",
        "positions_hash": "positions-1",
        "symbols_hash": "symbols-1",
    }
    manifest.write_text(
        json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest_row]})
    )
    state = {
        "enabled": True,
        "status": "ok",
        "execution_status": "completed",
        "converged": True,
        "check_convergence": True,
        "resumable": True,
        "non_resumable_reason": None,
        "convergence_streak": 1,
        "required_convergence_streak": 1,
        "last_convergence_evaluated_n_boxes_total": 1,
        "last_convergence_evaluated_n_boxes_accepted": 1,
        "min_boxes": 1,
        "n_boxes": 1,
        "n_boxes_accepted": 1,
        "n_boxes_rejected": 0,
        "n_boxes_total": 1,
        "boxes": [
            {
                "box": 1,
                "density": 2.7,
                "metrics": {"coord_Al-Al_mean": 12.0},
                "distributions": {},
                "paths": {
                    "relax_data": "production/box_001/relax.data",
                    "structure_snapshot": "production/box_001/structure_snapshot.json",
                    "structure_manifest": "production/box_001/structure_manifest.json",
                },
                "structure_manifest": manifest_row,
            }
        ],
        "rejected_boxes": [],
    }
    return attach_production_state_integrity(state, outdir=tmp_path), source


def _strict_production_state(tmp_path: Path) -> tuple[dict, Path]:
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        canonical_json_sha256,
        sha256_file,
    )

    state, source = _production_state(tmp_path)
    state.pop("state_integrity")
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    positions = [[1.0, 2.0, 3.0]]
    species = ["Al"]
    pbc = [True, True, True]
    structure = {"cell": cell, "species": species, "positions": positions, "pbc": pbc}
    source_rel = "production/box_001/relax.data"
    source_identity = {
        "path": source_rel,
        "exists": True,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    row = {
        "schema": "vitriflow.structure_manifest.row.v2",
        "box_id": 1,
        "structure_hash": canonical_json_sha256(structure),
        "cell_hash": canonical_json_sha256({"cell": cell}),
        "positions_hash": canonical_json_sha256({"positions": positions}),
        "symbols_hash": canonical_json_sha256({"species": species}),
        "n_atoms": 1,
        "source_path": source_rel,
        "source_role": "final_structure",
        "source_file_identity": source_identity,
    }
    snapshot = {
        "schema": "vitriflow.structure_snapshot.v1",
        "n_atoms": 1,
        "species": species,
        "positions": positions,
        "lattice": {"cell": cell, "pbc": pbc},
    }
    source.parent.joinpath("structure_snapshot.json").write_text(json.dumps(snapshot))
    source.parent.joinpath("structure_manifest.json").write_text(
        json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [row]})
    )
    state["boxes"][0]["structure_manifest"] = row
    return attach_production_state_integrity(state, outdir=tmp_path), source


def test_production_resume_integrity_binds_per_box_diagnostic_artifacts(
    tmp_path: Path,
):
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        validate_production_resume_state,
    )

    state, source = _production_state(tmp_path)
    state.pop("state_integrity")
    diagnostic_dir = source.parent / "diagnostics"
    diagnostic_dir.mkdir()
    csv_path = diagnostic_dir / "stage_metrics.csv"
    plot_path = diagnostic_dir / "stage_metrics.png"
    summary_path = diagnostic_dir / "elastic_summary.json"
    csv_path.write_text("step,value\n0,1.0\n")
    plot_path.write_bytes(b"plot-v1")
    summary_path.write_text('{"status":"ok"}\n')
    state["boxes"][0]["stage_metrics"] = {
        "melt": {
            "status": "ok",
            "csv": str(csv_path.relative_to(tmp_path)),
            "plot": str(plot_path.relative_to(tmp_path)),
        }
    }
    state["boxes"][0]["elastic_melt"] = {
        "status": "ok",
        "summary": str(summary_path.relative_to(tmp_path)),
    }

    protected = attach_production_state_integrity(
        state,
        outdir=tmp_path,
        force_rehash=True,
    )
    validate_production_resume_state(protected, outdir=tmp_path)

    csv_path.write_text("step,value\n0,9.0\n")
    with pytest.raises(RuntimeError, match="contents changed|artifact identities"):
        validate_production_resume_state(protected, outdir=tmp_path)


def test_production_resume_integrity_binds_reused_task_provenance(
    tmp_path: Path,
):
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        validate_production_resume_state,
    )

    state, source = _production_state(tmp_path)
    state.pop("state_integrity")
    task_manifest = source.parent / "task.json"
    task_result = source.parent / "task_result.json"
    task_manifest.write_text('{"schema":"vitriflow.box_task.v1"}\n')
    task_result.write_text('{"schema":"vitriflow.box_task_result.v2"}\n')
    state["boxes"][0]["task_diagnostics_provenance"] = {
        "schema": "vitriflow.reused_task_diagnostics.v1",
        "mode": "validated_read_only_reuse",
        "task_result": str(task_result.relative_to(tmp_path)),
        "task_manifest": str(task_manifest.relative_to(tmp_path)),
    }

    protected = attach_production_state_integrity(
        state,
        outdir=tmp_path,
        force_rehash=True,
    )
    validate_production_resume_state(protected, outdir=tmp_path)

    task_result.write_text('{"schema":"modified"}\n')
    with pytest.raises(RuntimeError, match="contents changed|artifact identities"):
        validate_production_resume_state(protected, outdir=tmp_path)


def test_production_resume_integrity_rejects_incomplete_task_provenance(
    tmp_path: Path,
):
    from vitriflow.workflows.resume_integrity import attach_production_state_integrity

    state, source = _production_state(tmp_path)
    state.pop("state_integrity")
    task_result = source.parent / "task_result.json"
    task_result.write_text('{"schema":"vitriflow.box_task_result.v2"}\n')
    state["boxes"][0]["task_diagnostics_provenance"] = {
        "schema": "vitriflow.reused_task_diagnostics.v1",
        "mode": "validated_read_only_reuse",
        "task_result": str(task_result.relative_to(tmp_path)),
        "task_manifest": None,
    }

    with pytest.raises(RuntimeError, match="lacks task_manifest"):
        attach_production_state_integrity(state, outdir=tmp_path)


@pytest.mark.parametrize(
    "sidecar_kind",
    ["coordination", "amorphous", "graph_chunk", "reject_directory"],
)
def test_production_resume_integrity_binds_public_analysis_sidecars(
    tmp_path: Path,
    sidecar_kind: str,
):
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        validate_production_resume_state,
    )

    state, source = _production_state(tmp_path)
    state.pop("state_integrity")
    box = state["boxes"][0]
    sidecar = source.parent / f"{sidecar_kind}.dat"
    sidecar.write_text(f"{sidecar_kind}-v1\n")
    relative = str(sidecar.relative_to(tmp_path))

    if sidecar_kind == "coordination":
        box["paths"]["coord_defects"] = {
            "status": "ok",
            "detail_json": relative,
        }
    elif sidecar_kind == "amorphous":
        box["paths"]["amorphous"] = {"state_json": relative}
    elif sidecar_kind == "graph_chunk":
        box["graph_analysis"] = {
            "schema": "vitriflow.graph_analysis.streamed_summary.v1",
            "streamed_sidecars": True,
            "chunk_paths": {"metric_results": relative},
        }
    else:
        reject_dir = source.parent / "reject_copy"
        reject_dir.mkdir()
        reject_copy = reject_dir / "relax.data"
        reject_copy.write_text("rejected-copy-v1\n")
        box["reject"] = {
            "reason": "fixture",
            "reject_dir": str(reject_dir.relative_to(tmp_path)),
            "relax_data": str(reject_copy.relative_to(tmp_path)),
            "relax_dump": None,
        }
        sidecar = reject_copy

    protected = attach_production_state_integrity(
        state,
        outdir=tmp_path,
        force_rehash=True,
    )
    validate_production_resume_state(protected, outdir=tmp_path)
    sidecar.write_text(f"{sidecar_kind}-modified\n")
    with pytest.raises(RuntimeError, match="contents changed|artifact identities"):
        validate_production_resume_state(protected, outdir=tmp_path)


def test_current_resume_integrity_cross_links_box_snapshot_manifest_and_source(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        validate_production_resume_state,
    )

    state, source = _strict_production_state(tmp_path)
    validate_production_resume_state(state, outdir=tmp_path)

    wrong_box = json.loads(json.dumps(state))
    wrong_box.pop("state_integrity")
    wrong_box["boxes"][0]["structure_manifest"]["box_id"] = 999
    manifest_path = source.parent / "structure_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text())
    manifest_payload["structures"][0]["box_id"] = 999
    manifest_path.write_text(json.dumps(manifest_payload))
    with pytest.raises(RuntimeError, match="manifest box_id=999 disagrees"):
        attach_production_state_integrity(wrong_box, outdir=tmp_path)


def test_current_resume_integrity_rejects_snapshot_manifest_hash_mismatch(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import attach_production_state_integrity

    state, source = _strict_production_state(tmp_path)
    state.pop("state_integrity")
    snapshot_path = source.parent / "structure_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["positions"][0][0] = 9.0
    snapshot_path.write_text(json.dumps(snapshot))
    with pytest.raises(RuntimeError, match="snapshot structure_hash disagrees"):
        attach_production_state_integrity(state, outdir=tmp_path)


def test_current_resume_integrity_cannot_rebless_changed_manifest_source(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import attach_production_state_integrity

    state, source = _strict_production_state(tmp_path)
    state.pop("state_integrity")
    source.write_text("structure-v2-with-different-content\n")
    with pytest.raises(RuntimeError, match="manifest source identity disagrees"):
        attach_production_state_integrity(state, outdir=tmp_path)


def test_current_resume_integrity_binds_complete_dft_cell_opt_evidence(
    tmp_path: Path,
):
    from vitriflow.workflows.autotune import (
        _build_cp2k_cell_opt_calculation_identity,
        _write_cp2k_cell_opt_identity_manifest,
    )
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        canonical_json_sha256,
        sha256_file,
        validate_production_resume_state,
    )

    state, parent = _strict_production_state(tmp_path)
    state.pop("state_integrity")
    dft_dir = parent.parent / "dft_opt"
    dft_dir.mkdir()
    dft_data = dft_dir / "dft_opt.data"
    dft_input = dft_dir / "cell_opt.inp"
    dft_output = dft_dir / "cp2k.out"
    dft_scf = dft_dir / "cp2k_scf_diagnostics.json"
    dft_traj = dft_dir / "traj.dcd"
    basis = dft_dir / "BASIS_SET"
    potential = dft_dir / "POTENTIAL"
    dft_data.write_text("refined-structure-v1\n")
    dft_input.write_text("&GLOBAL\n  RUN_TYPE CELL_OPT\n&END GLOBAL\n")
    dft_output.write_text("CELL OPTIMIZATION COMPLETED\n")
    dft_scf.write_text(
        json.dumps(
            {
                "schema": "vitriflow.cp2k_scf_diagnostics.v1",
                "cp2k_version": "2024.2",
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
        )
    )
    dft_traj.write_bytes(b"trajectory-v1")
    basis.write_text("basis-v1\n")
    potential.write_text("potential-v1\n")
    calculation = _build_cp2k_cell_opt_calculation_identity(
        parent_relax_data=parent,
        dft_config={"enabled": True, "optimizer": "LBFGS"},
        cp2k_config={"cp2k_cmd": "cp2k"},
        cp2k_version=(2024, 2),
        basis_path=basis,
        potential_path=potential,
        base_input_text=dft_input.read_text(),
        external_pressure_bar=0.0,
        atom_style="atomic",
        type_to_species=["Al"],
        lammps_units_style="metal",
    )
    identity_path = dft_dir / "cell_opt_identity.json"
    _write_cp2k_cell_opt_identity_manifest(
        identity_path,
        calculation=calculation,
        status="completed",
        artifacts={
            "input": dft_input,
            "output": dft_output,
            "scf_diagnostics": dft_scf,
            "trajectory": dft_traj,
            "data": dft_data,
        },
    )

    cell = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]
    positions = [[2.0, 3.0, 4.0]]
    species = ["Al"]
    pbc = [True, True, True]
    structure = {"cell": cell, "species": species, "positions": positions, "pbc": pbc}
    dft_data_rel = str(dft_data.relative_to(tmp_path))
    dft_source_identity = {
        "path": dft_data_rel,
        "exists": True,
        "size_bytes": dft_data.stat().st_size,
        "sha256": sha256_file(dft_data),
    }
    dft_row = {
        "schema": "vitriflow.structure_manifest.row.v2",
        "box_id": 1,
        "structure_hash": canonical_json_sha256(structure),
        "cell_hash": canonical_json_sha256({"cell": cell}),
        "positions_hash": canonical_json_sha256({"positions": positions}),
        "symbols_hash": canonical_json_sha256({"species": species}),
        "n_atoms": 1,
        "source_path": dft_data_rel,
        "source_role": "dft_opt_final",
        "source_file_identity": dft_source_identity,
    }
    dft_snapshot = dft_dir / "structure_snapshot.json"
    dft_structure_manifest = dft_dir / "structure_manifest.json"
    dft_snapshot.write_text(
        json.dumps(
            {
                "schema": "vitriflow.structure_snapshot.v1",
                "n_atoms": 1,
                "species": species,
                "positions": positions,
                "lattice": {"cell": cell, "pbc": pbc},
            }
        )
    )
    dft_structure_manifest.write_text(
        json.dumps(
            {
                "schema": "vitriflow.structure_manifest.v2",
                "structures": [dft_row],
            }
        )
    )
    state["boxes"][0]["dft_opt"] = {
        "status": "ok",
        "density": 2.8,
        "metrics": {},
        "distributions": {},
        "structure_manifest": dft_row,
        "paths": {
            "dft_data": dft_data_rel,
            "dft_input": str(dft_input.relative_to(tmp_path)),
            "dft_output": str(dft_output.relative_to(tmp_path)),
            "dft_scf_diagnostics": str(dft_scf.relative_to(tmp_path)),
            "dft_traj": str(dft_traj.relative_to(tmp_path)),
            "dft_identity": str(identity_path.relative_to(tmp_path)),
            "structure_snapshot": str(dft_snapshot.relative_to(tmp_path)),
            "structure_manifest": str(dft_structure_manifest.relative_to(tmp_path)),
        },
    }
    state = attach_production_state_integrity(
        state, outdir=tmp_path, force_rehash=True
    )
    validate_production_resume_state(state, outdir=tmp_path)

    dft_input.write_text("changed input\n")
    with pytest.raises(RuntimeError, match="dft_input disagrees|contents changed"):
        validate_production_resume_state(state, outdir=tmp_path)


@pytest.mark.parametrize("invalid_size", [None, "not-an-integer", float("inf")])
def test_current_resume_integrity_rejects_malformed_manifest_source_size(
    tmp_path: Path, invalid_size
):
    from vitriflow.workflows.resume_integrity import attach_production_state_integrity

    state, source = _strict_production_state(tmp_path)
    state.pop("state_integrity")
    state["boxes"][0]["structure_manifest"]["source_file_identity"][
        "size_bytes"
    ] = invalid_size
    manifest_path = source.parent / "structure_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text())
    manifest_payload["structures"][0]["source_file_identity"][
        "size_bytes"
    ] = invalid_size
    manifest_path.write_text(json.dumps(manifest_payload))

    with pytest.raises(RuntimeError, match="source identity has an invalid size"):
        attach_production_state_integrity(state, outdir=tmp_path)


def test_production_state_integrity_rejects_metric_tampering(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import validate_production_resume_state

    state, _source = _production_state(tmp_path)
    tampered = json.loads(json.dumps(state))
    tampered["boxes"][0]["metrics"]["coord_Al-Al_mean"] = 6.0
    with pytest.raises(RuntimeError, match="checkpoint state was modified"):
        validate_production_resume_state(tampered, outdir=tmp_path)

    tampered_identity = json.loads(json.dumps(state))
    tampered_identity["state_integrity"]["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="integrity record was modified"):
        validate_production_resume_state(tampered_identity, outdir=tmp_path)


def test_checkpoint_artifact_hashing_is_incremental_but_terminal_revalidates(
    tmp_path: Path, monkeypatch
):
    from vitriflow.workflows import resume_integrity as integrity

    one, _source = _production_state(tmp_path)
    one.pop("state_integrity")
    source2 = tmp_path / "production" / "box_002" / "relax.data"
    source2.parent.mkdir(parents=True)
    source2.write_text("structure-v2\n")
    (source2.parent / "structure_snapshot.json").write_text(
        json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1})
    )
    manifest2 = {
        "structure_hash": "structure-2",
        "cell_hash": "cell-2",
        "positions_hash": "positions-2",
        "symbols_hash": "symbols-2",
    }
    (source2.parent / "structure_manifest.json").write_text(
        json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest2]})
    )
    two = json.loads(json.dumps(one))
    two["boxes"].append(
        {
            "box": 2,
            "metrics": {},
            "distributions": {},
            "paths": {
                "relax_data": "production/box_002/relax.data",
                "structure_snapshot": "production/box_002/structure_snapshot.json",
                "structure_manifest": "production/box_002/structure_manifest.json",
            },
            "structure_manifest": manifest2,
        }
    )
    two.update({"n_boxes": 2, "n_boxes_accepted": 2, "n_boxes_total": 2})

    calls = 0
    real_hash = integrity.sha256_file

    def counted(path):
        nonlocal calls
        calls += 1
        return real_hash(path)

    monkeypatch.setattr(integrity, "sha256_file", counted)
    cache: dict[str, tuple[int, int, str]] = {}
    integrity.attach_production_state_integrity(one, outdir=tmp_path, identity_cache=cache)
    assert calls == 3
    integrity.attach_production_state_integrity(two, outdir=tmp_path, identity_cache=cache)
    assert calls == 6
    integrity.attach_production_state_integrity(
        two, outdir=tmp_path, identity_cache=cache, force_rehash=True
    )
    assert calls == 12


@pytest.mark.parametrize("mode", ["alter", "delete"])
def test_production_state_integrity_rejects_changed_or_missing_source(tmp_path: Path, mode: str):
    from vitriflow.workflows.resume_integrity import validate_production_resume_state

    state, source = _production_state(tmp_path)
    if mode == "alter":
        source.write_text("structure-v2\n")
    else:
        source.unlink()
    with pytest.raises(RuntimeError, match="missing|changed|Required resume"):
        validate_production_resume_state(state, outdir=tmp_path)


@pytest.mark.parametrize("name", ["structure_snapshot.json", "structure_manifest.json"])
def test_production_state_integrity_requires_structure_sidecars(tmp_path: Path, name: str):
    from vitriflow.workflows.resume_integrity import validate_production_resume_state

    state, source = _production_state(tmp_path)
    (source.parent / name).unlink()
    with pytest.raises(RuntimeError, match="sidecar is not valid JSON"):
        validate_production_resume_state(state, outdir=tmp_path)


def test_production_state_integrity_rejects_mutated_manifest_json(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import validate_production_resume_state

    state, source = _production_state(tmp_path)
    manifest = source.parent / "structure_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["structures"][0]["structure_hash"] = "different-structure"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="manifest structure_hash disagrees"):
        validate_production_resume_state(state, outdir=tmp_path)


def test_production_state_integrity_binds_terminal_graph_outputs(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        validate_production_resume_state,
    )

    state, _source = _production_state(tmp_path)
    state.pop("state_integrity")
    graph = tmp_path / "graph_rules.json"
    graph.write_text('{"schema":"vitriflow.graph_rules.v1"}\n')
    state["graph_outputs"] = {"graph_rules": "graph_rules.json"}
    state = attach_production_state_integrity(state, outdir=tmp_path, force_rehash=True)
    validate_production_resume_state(state, outdir=tmp_path)
    graph.write_text('{"schema":"tampered"}\n')
    with pytest.raises(RuntimeError, match="contents changed"):
        validate_production_resume_state(state, outdir=tmp_path)


def test_production_state_integrity_rejects_graph_path_escape(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import attach_production_state_integrity

    state, _source = _production_state(tmp_path)
    state.pop("state_integrity")
    outside = tmp_path.parent / "outside-graph.json"
    outside.write_text("{}\n")
    state["graph_outputs"] = {"graph_rules": str(outside)}
    with pytest.raises(RuntimeError, match="escapes the output directory"):
        attach_production_state_integrity(state, outdir=tmp_path)


def test_production_terminal_status_is_scientifically_truthful():
    from vitriflow.workflows.resume_integrity import production_final_status

    assert production_final_status(
        n_accepted=0,
        min_boxes=2,
        check_convergence=True,
        converged=False,
        max_boxes=2,
        n_total=2,
    )[0] == "incomplete"
    assert production_final_status(
        n_accepted=2,
        min_boxes=2,
        check_convergence=True,
        converged=False,
        max_boxes=2,
        n_total=2,
    )[0] == "not_converged"
    # A final partial batch may raise the attempted-box target above the
    # configured minimum. Scientific completeness is still judged against the
    # configured min_boxes, so a capped six-box ensemble is not "incomplete".
    assert production_final_status(
        n_accepted=6,
        min_boxes=4,
        check_convergence=True,
        converged=False,
        max_boxes=6,
        n_total=6,
    )[0] == "not_converged"
    assert production_final_status(
        n_accepted=2,
        min_boxes=2,
        check_convergence=True,
        converged=True,
        max_boxes=2,
        n_total=2,
    ) == ("ok", None)


def _autotune_config(tmp_path: Path):
    from vitriflow.config import RunConfig

    source = tmp_path / "input.data"
    source.write_text("input-v1\n")
    table = tmp_path / "potential.table"
    table.write_text("potential-v1\n")
    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "commands": [f"pair_style table linear 1000", f"pair_coeff * * {table} TEST"],
                "files": [],
                "interactions": ["Al"],
            },
            "structure": {"lammps_data": str(source)},
            "autotune": {
                "metrics": {"enabled": True, "type_to_species": ["Al"]},
                "production": {"enabled": True, "min_boxes": 1, "max_boxes": 1},
            },
        }
    )
    return cfg, source, table


def test_autotune_fingerprint_binds_full_config_structure_and_command_file(tmp_path: Path):
    from vitriflow.workflows.autotune import _build_autotune_resume_fingerprint

    cfg, source, table = _autotune_config(tmp_path)
    selected = tmp_path / "selected.data"
    selected.write_text("selected-v1\n")
    plan = {"schema": "test-plan.v1", "structure_data": "selected.data"}
    first = _build_autotune_resume_fingerprint(
        config=cfg, outdir=tmp_path, selected_structure=selected, production_plan=plan
    )
    refs = first["payload"]["input_identities"]["potential_command_files"]
    assert [item["filename"] for item in refs] == [table.name]

    table.write_text("potential-v2\n")
    second = _build_autotune_resume_fingerprint(
        config=cfg, outdir=tmp_path, selected_structure=selected, production_plan=plan
    )
    assert second["sha256"] != first["sha256"]

    table.write_text("potential-v1\n")
    source.write_text("input-v2\n")
    third = _build_autotune_resume_fingerprint(
        config=cfg, outdir=tmp_path, selected_structure=selected, production_plan=plan
    )
    assert third["sha256"] != first["sha256"]

    source.write_text("input-v1\n")
    changed = cfg.model_copy(deep=True)
    changed.random_seed += 1
    fourth = _build_autotune_resume_fingerprint(
        config=changed, outdir=tmp_path, selected_structure=selected, production_plan=plan
    )
    assert fourth["sha256"] != first["sha256"]


def test_autotune_resume_fingerprint_canonicalizes_relative_output_structure_path(
    tmp_path: Path, monkeypatch
):
    """Fresh relative paths and resume-resolved paths must identify one file."""

    from vitriflow.workflows.autotune import (
        _build_autotune_resume_fingerprint,
        _validate_autotune_resume_fingerprint,
    )

    cfg, _source, _table = _autotune_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    outdir = Path("relative-output")
    selected = outdir / "structure" / "size_base.data"
    selected.parent.mkdir(parents=True)
    selected.write_text("selected-v1\n")
    plan = {"schema": "test-plan.v1", "structure_data": "structure/size_base.data"}
    previous = {
        "size_scan": {"base_data": "structure/size_base.data"},
        "production_plan": plan,
        "resume_fingerprint": _build_autotune_resume_fingerprint(
            config=cfg,
            outdir=outdir,
            selected_structure=selected,
            production_plan=plan,
        ),
    }

    assert previous["resume_fingerprint"]["payload"]["selected_structure"]["path"] == (
        "structure/size_base.data"
    )
    # Resume resolves the stored result path to absolute before rebuilding the
    # fingerprint.  This was the production failure for a relative ``-o``.
    rebuilt = _validate_autotune_resume_fingerprint(
        previous,
        config=cfg,
        outdir=outdir,
    )
    assert rebuilt["sha256"] == previous["resume_fingerprint"]["sha256"]
    assert rebuilt["payload"]["selected_structure"]["path"] == (
        "structure/size_base.data"
    )

    selected.write_text("selected-v2\n")
    with pytest.raises(RuntimeError, match="Cannot safely resume autotune"):
        _validate_autotune_resume_fingerprint(
            previous,
            config=cfg,
            outdir=outdir,
        )


def test_autotune_fingerprint_rejects_legacy_and_payload_tampering(tmp_path: Path):
    from vitriflow.workflows.autotune import (
        _build_autotune_resume_fingerprint,
        _validate_autotune_resume_fingerprint,
    )

    cfg, _source, _table = _autotune_config(tmp_path)
    selected = tmp_path / "selected.data"
    selected.write_text("selected-v1\n")
    plan = {"schema": "test-plan.v1", "structure_data": "selected.data"}
    previous = {
        "size_scan": {"base_data": "selected.data"},
        "production_plan": plan,
        "resume_fingerprint": _build_autotune_resume_fingerprint(
            config=cfg,
            outdir=tmp_path,
            selected_structure=selected,
            production_plan=plan,
        ),
    }
    _validate_autotune_resume_fingerprint(previous, config=cfg, outdir=tmp_path)

    with pytest.raises(RuntimeError, match="no provenance fingerprint"):
        _validate_autotune_resume_fingerprint(
            {"size_scan": {"base_data": "selected.data"}},
            config=cfg,
            outdir=tmp_path,
        )

    tampered = json.loads(json.dumps(previous))
    tampered["resume_fingerprint"]["payload"]["seed_scheme"] = "altered"
    with pytest.raises(RuntimeError, match="payload was modified"):
        _validate_autotune_resume_fingerprint(tampered, config=cfg, outdir=tmp_path)


def test_autotune_cp2k_identity_resolves_environment_data_without_configured_dir(
    tmp_path: Path, monkeypatch
):
    from vitriflow.config import RunConfig
    from vitriflow.workflows.autotune import _config_input_identities

    data_dir = tmp_path / "cp2k-data"
    data_dir.mkdir()
    (data_dir / "BASIS_MOLOPT").write_text("basis-from-env\n")
    (data_dir / "GTH_POTENTIALS").write_text("potential-from-env\n")
    monkeypatch.setenv("CP2K_DATA_DIR", str(data_dir))
    cfg = RunConfig.model_validate(
        {
            "engine": "cp2k",
            "cp2k": {
                "kind_settings": {
                    "Al": {"basis_set": "DZVP-MOLOPT-SR-GTH", "potential": "GTH-PBE-q3"}
                }
            },
            "structure": {
                "generate": {"method": "random", "formula": "Al", "n_formula_units": 1}
            },
            "autotune": {"metrics": {"enabled": True, "type_to_species": ["Al"]}},
        }
    )
    identities = _config_input_identities(cfg, workdir=tmp_path)
    cp2k = identities["cp2k_data_files"]
    assert {item["role"] for item in cp2k} == {"basis_set", "potential"}
    assert {item["filename"] for item in cp2k} == {"BASIS_MOLOPT", "GTH_POTENTIALS"}
    assert all(len(item["sha256"]) == 64 for item in cp2k)


def test_custom_schedule_rejects_stored_payload_with_unrecomputed_digest(tmp_path: Path):
    from vitriflow.workflows.custom_schedule import (
        _sha256_canonical_json,
        _validate_resume_fingerprint_or_raise,
    )

    payload = {"schema": "demo", "seed_scheme": "v1"}
    current = {"sha256": _sha256_canonical_json(payload), "payload": payload}
    stored = json.loads(json.dumps(current))
    stored["payload"]["seed_scheme"] = "tampered"
    with pytest.raises(RuntimeError, match="modified or corrupted"):
        _validate_resume_fingerprint_or_raise(
            {"resume_fingerprint": stored}, current, outdir=tmp_path
        )


def test_resume_integrity_module_does_not_reverse_runner_ownership():
    root = Path(__file__).resolve().parents[1] / "vitriflow" / "workflows"
    assert "from .autotune import" not in (root / "run.py").read_text()
    assert "from .autotune import" not in (root / "custom_schedule.py").read_text()
    neutral = (root / "resume_integrity.py").read_text()
    assert "from .autotune import" not in neutral
    assert "from .run import" not in neutral
    assert "from .custom_schedule import" not in neutral


def test_production_resume_rejects_gapped_attempt_ids() -> None:
    from vitriflow.workflows.resume_integrity import (
        validate_production_state_semantics,
    )

    state = {
        "enabled": True,
        "status": "running",
        "execution_status": "running",
        "check_convergence": False,
        "n_boxes": 2,
        "n_boxes_accepted": 2,
        "n_boxes_rejected": 0,
        "n_boxes_total": 2,
        "boxes": [{"box": 1}, {"box": 3}],
        "rejected_boxes": [],
    }
    with pytest.raises(RuntimeError, match="contiguous prefix"):
        validate_production_state_semantics(state)


def test_production_resume_accepts_complete_zero_based_custom_prefix() -> None:
    from vitriflow.workflows.resume_integrity import (
        validate_production_state_semantics,
    )

    state = {
        "enabled": True,
        "status": "running",
        "execution_status": "running",
        "check_convergence": False,
        "n_boxes": 2,
        "n_boxes_accepted": 2,
        "n_boxes_rejected": 1,
        "n_boxes_total": 3,
        "boxes": [{"box": 0}, {"box": 2}],
        "rejected_boxes": [{"box": 1}],
    }
    validate_production_state_semantics(state)


def test_production_resume_rejects_gapped_zero_based_custom_prefix() -> None:
    from vitriflow.workflows.resume_integrity import (
        validate_production_state_semantics,
    )

    state = {
        "enabled": True,
        "status": "running",
        "execution_status": "running",
        "check_convergence": False,
        "n_boxes": 2,
        "n_boxes_accepted": 2,
        "n_boxes_rejected": 0,
        "n_boxes_total": 2,
        "boxes": [{"box": 0}, {"box": 2}],
        "rejected_boxes": [],
    }
    with pytest.raises(RuntimeError, match="contiguous prefix"):
        validate_production_state_semantics(state)
