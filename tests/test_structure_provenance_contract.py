from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vitriflow.analysis.common import mic_displacements_and_distances
from vitriflow.analysis.dump import DumpFrame, read_last_dump_frame
from vitriflow.analysis.graph import GraphRule, build_hard_graph, manifest_row_from_frame, structure_hash
from vitriflow.analysis.amorphous import _directed_neighbors
from vitriflow.analysis.voids import sample_void_clearance_points, sample_void_clearance_radii
from vitriflow.analysis.trajectory import read_last_frames_auto
from vitriflow.io.extxyz import write_extxyz_single
from vitriflow.io.cp2k_restart import read_cp2k_restart_frame
from vitriflow.workflows.output_analysis import (
    _sidecar_integrity_record,
    _write_structure_provenance_sidecars,
    analysis_context_from_standalone_config,
    analyze_output_data,
)


def _frame(*, pbc=(True, False, True), y2=9.9) -> DumpFrame:
    return DumpFrame(
        timestep=7,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 1], dtype=int),
        positions=np.asarray([[1.0, 0.1, 1.0], [1.0, y2, 1.0]], dtype=float),
        cell=np.eye(3, dtype=float) * 10.0,
        origin=np.zeros(3, dtype=float),
        pbc=pbc,
    )


def _reference_entry(frame: DumpFrame, source: Path) -> dict:
    row = manifest_row_from_frame(
        frame,
        box_id=1,
        source_path=source,
        source_role="final_structure",
        type_to_species=["Si"],
        density=2.2,
    )
    return {
        "box": 1,
        "density": 2.2,
        "metrics": {},
        "distributions": {},
        # Reproduce the 0.4.30.0 failure mode: hashes exist only in the nested
        # non-embedded structure reference, not at top level.
        "structure": {
            "schema": "vitriflow.structure_snapshot_ref.v1",
            "embedded": False,
            "n_atoms": 2,
            "source_path": str(source),
            "source_role": "final_structure",
            **{name: row[name] for name in ("structure_hash", "cell_hash", "positions_hash", "symbols_hash")},
        },
        "paths": {"analysis_source": str(source)},
        "reject": False,
    }


def test_partial_pbc_roundtrip_hash_and_mic_are_not_silently_periodic(tmp_path: Path):
    partial = _frame()
    periodic = _frame(pbc=(True, True, True))
    source = tmp_path / "partial.extxyz"
    write_extxyz_single(source, partial, type_to_species=["Si"])

    loaded = read_last_frames_auto(source, 1, type_to_species=["Si"])[0]
    assert loaded.pbc == (True, False, True)
    assert structure_hash(loaded, type_to_species=["Si"]) == structure_hash(partial, type_to_species=["Si"])
    assert structure_hash(partial, type_to_species=["Si"]) != structure_hash(periodic, type_to_species=["Si"])

    frac = partial.positions @ np.linalg.inv(partial.cell)
    _vec, dist = mic_displacements_and_distances(
        frac,
        partial.cell,
        np.asarray([0]),
        np.asarray([1]),
        pbc=partial.pbc,
    )
    assert dist[0] == pytest.approx(9.8)

    rule = GraphRule(name="short", kind="hard_cutoff", parameters={"cutoff": 1.0})
    assert build_hard_graph(partial, rule, type_to_species=["Si"]).edges == []
    assert build_hard_graph(periodic, rule, type_to_species=["Si"]).edges == [(0, 1)]

    assert _directed_neighbors(partial, cutoffs={(1, 1): 1.0})[2] == []
    assert _directed_neighbors(periodic, cutoffs={(1, 1): 1.0})[2] == [(0, 1)]


def test_periodic_volume_void_estimator_rejects_partial_pbc_explicitly():
    partial = _frame()
    with pytest.raises(ValueError, match="periodic cell-volume estimator"):
        sample_void_clearance_radii([partial], n_samples=8, sampler="grid")
    with pytest.raises(ValueError, match="open-boundary domain/wall model"):
        sample_void_clearance_points(partial, n_samples=8, sampler="grid")


def test_production_analysis_rejects_source_replacement_after_snapshot(monkeypatch, tmp_path: Path):
    from vitriflow.config import StructureMetricsConfig
    from vitriflow.workflows import production_common as pc

    outdir = tmp_path / "run"
    melt = outdir / "melt"
    quench = outdir / "quench"
    relax = outdir / "relax"
    for path in (melt, quench, relax):
        path.mkdir(parents=True, exist_ok=True)
    source = relax / "final.extxyz"
    source.write_text("source-A\n")
    frame = _frame(pbc=(True, True, True), y2=2.0)

    def _read_snapshot(path, *args, **kwargs):
        assert Path(path) != source
        source.write_text("source-B\n")
        return [frame]

    monkeypatch.setattr(pc, "read_last_frames_auto", _read_snapshot)
    metrics = StructureMetricsConfig(enabled=False)

    with pytest.raises(RuntimeError, match="source changed after its immutable snapshot"):
        pc.analyse_production_box(
            box_id=1,
            outdir=outdir,
            melt_stage_dir=melt,
            quench_stage_dir=quench,
            relax_stage_dir=relax,
            relax_data_path=source,
            density_mean=2.2,
            density_stderr=0.0,
            metrics_cfg=metrics,
            cutoffs={},
            required_pairs=[],
            fixed_cutoffs={},
            type_to_species=["Si"],
            md_timestep=1.0,
            analysis_source_path=source,
        )


def test_lammps_dump_boundary_flags_are_preserved_and_missing_flags_fail(tmp_path: Path):
    body = (
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp ff pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z\n1 1 1 2 3\n"
    )
    source = tmp_path / "partial.dump"
    source.write_text(body)
    assert read_last_dump_frame(source).pbc == (True, False, True)

    missing = tmp_path / "missing.dump"
    missing.write_text(body.replace("ITEM: BOX BOUNDS pp ff pp", "ITEM: BOX BOUNDS"))
    with pytest.raises(ValueError, match="three LAMMPS boundary flags"):
        read_last_dump_frame(missing)


def test_extxyz_without_pbc_fails_instead_of_assuming_periodic(tmp_path: Path):
    source = tmp_path / "unknown.extxyz"
    source.write_text(
        "1\n"
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\n'
        "Si 1 1 1\n"
    )
    with pytest.raises(ValueError, match="missing pbc"):
        read_last_frames_auto(source, 1, type_to_species=["Si"])


def test_cp2k_restart_cell_periodicity_is_preserved(tmp_path: Path):
    source = tmp_path / "calc-1.restart"
    source.write_text(
        "&FORCE_EVAL\n"
        "  &SUBSYS\n"
        "    &CELL\n"
        "      ABC 10 11 12\n"
        "      PERIODIC XZ\n"
        "    &END CELL\n"
        "    &COORD\n"
        "      Si 1 2 3\n"
        "    &END COORD\n"
        "  &END SUBSYS\n"
        "&END FORCE_EVAL\n"
    )
    frame = read_cp2k_restart_frame(source, type_to_species=["Si"])
    assert frame.pbc == (True, False, True)


def test_default_nonembedded_provenance_materializes_verified_json_sidecars(tmp_path: Path):
    outdir = tmp_path / "analysis"
    outdir.mkdir()
    source = tmp_path / "final.extxyz"
    frame = _frame(pbc=(True, True, True), y2=2.0)
    write_extxyz_single(source, frame, type_to_species=["Si"])
    entry = _reference_entry(frame, source)

    result = _write_structure_provenance_sidecars(
        outdir,
        boxes=[entry],
        rejected_boxes=[],
        type_to_species=["Si"],
        atom_style="atomic",
    )

    manifest = json.loads((outdir / "structure_manifest.json").read_text())
    row = manifest["structures"][0]
    assert row["verification"]["status"] == "verified_from_source"
    assert row["verification"]["verified"] is True
    assert row["verification"]["source_artifact_verified"] is True
    assert row["verification"]["structure_hashes_verified"] is True
    assert row["verification"]["pbc_source_verified"] is True
    assert row["verification"]["error"] is None
    assert all(row[name] for name in ("structure_hash", "cell_hash", "positions_hash", "symbols_hash"))
    assert row["pbc"] == [True, True, True]
    assert row["source_file_identity"]["sha256"]
    ref_path = outdir / row["structure_reference"]
    reference = json.loads(ref_path.read_text())
    assert reference["status"] == "verified_from_source"
    assert reference["hashes"]["structure_hash"] == row["structure_hash"]
    assert result["all_hash_locked"] is True
    assert result["all_verified"] is True

    for rel in ("structure_manifest.json", "structure_references.json", row["structure_reference"]):
        integrity = _sidecar_integrity_record(outdir, rel)
        assert integrity["exists"] is True
        assert integrity["size_bytes"] > 0
        assert len(integrity["sha256"]) == 64
        assert integrity["schema"].startswith("vitriflow.")


def test_present_source_with_stale_recorded_hash_is_a_hard_failure(tmp_path: Path):
    outdir = tmp_path / "analysis"
    outdir.mkdir()
    source = tmp_path / "final.extxyz"
    frame = _frame(pbc=(True, True, True), y2=2.0)
    write_extxyz_single(source, frame, type_to_species=["Si"])
    entry = _reference_entry(frame, source)
    entry["structure"]["structure_hash"] = "0" * 64

    with pytest.raises(ValueError, match="stored structure_hash does not match"):
        _write_structure_provenance_sidecars(
            outdir,
            boxes=[entry],
            rejected_boxes=[],
            type_to_species=["Si"],
            atom_style="atomic",
        )


def test_source_file_identity_detects_nonstructural_file_mutation(tmp_path: Path):
    outdir = tmp_path / "analysis"
    outdir.mkdir()
    source = tmp_path / "final.extxyz"
    frame = _frame(pbc=(True, True, True), y2=2.0)
    write_extxyz_single(source, frame, type_to_species=["Si"])
    entry = _reference_entry(frame, source)
    entry["structure_manifest"] = manifest_row_from_frame(
        frame,
        box_id=1,
        source_path=source,
        source_role="final_structure",
        type_to_species=["Si"],
    )
    # A harmless trailing line leaves the parsed final frame unchanged, but it
    # changes the source artifact identity and must invalidate the record.
    source.write_text(source.read_text() + "\n")

    with pytest.raises(ValueError, match="source artifact identity changed"):
        _write_structure_provenance_sidecars(
            outdir,
            boxes=[entry],
            rejected_boxes=[],
            type_to_species=["Si"],
            atom_style="atomic",
        )


def test_lammps_data_pbc_is_disclosed_as_workflow_assumption(tmp_path: Path):
    outdir = tmp_path / "analysis"
    outdir.mkdir()
    source = tmp_path / "final.data"
    source.write_text(
        "LAMMPS data\n\n"
        "1 atoms\n1 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Atoms # atomic\n\n1 1 1 2 3\n"
    )
    frame = read_last_frames_auto(source, 1, type_to_species=["Si"])[0]
    row = manifest_row_from_frame(
        frame,
        box_id=1,
        source_path=source,
        source_role="final_structure",
        type_to_species=["Si"],
    )
    entry = _reference_entry(frame, source)
    entry["structure_manifest"] = row

    result = _write_structure_provenance_sidecars(
        outdir,
        boxes=[entry],
        rejected_boxes=[],
        type_to_species=["Si"],
        atom_style="atomic",
        lammps_units_style="metal",
    )
    written = json.loads((outdir / "structure_manifest.json").read_text())["structures"][0]
    assert written["pbc"] == [True, True, True]
    assert written["pbc_provenance"].startswith("vitriflow_periodic_lammps_cell_contract")
    assert written["verification"]["status"] == "verified_with_declared_pbc_assumption"
    assert written["verification"]["source_artifact_verified"] is True
    assert written["verification"]["pbc_source_verified"] is False
    assert result["all_source_artifacts_verified"] is True
    assert result["all_verified"] is False


def test_unavailable_source_retains_recorded_identity_without_claiming_verification(tmp_path: Path):
    outdir = tmp_path / "analysis"
    outdir.mkdir()
    source = tmp_path / "final.extxyz"
    frame = _frame(pbc=(True, True, True), y2=2.0)
    write_extxyz_single(source, frame, type_to_species=["Si"])
    row = manifest_row_from_frame(
        frame,
        box_id=1,
        source_path=source,
        source_role="final_structure",
        type_to_species=["Si"],
    )
    recorded_sha = row["source_file_identity"]["sha256"]
    entry = _reference_entry(frame, source)
    entry["structure_manifest"] = row
    source.unlink()

    result = _write_structure_provenance_sidecars(
        outdir,
        boxes=[entry],
        rejected_boxes=[],
        type_to_species=["Si"],
        atom_style="atomic",
    )
    written = json.loads((outdir / "structure_manifest.json").read_text())["structures"][0]
    assert written["verification"]["status"] == "hash_locked_source_unavailable"
    assert written["verification"]["verified"] is False
    assert written["source_file_identity"]["sha256"] == recorded_sha
    assert written["source_file_verification"]["exists"] is False
    assert result["all_hash_locked"] is True
    assert result["all_source_artifacts_verified"] is False


def test_analysis_results_exposes_non_null_hashes_without_graph_rules(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.production_common as pc

    source_dir = tmp_path / "ensemble"
    source_dir.mkdir()
    source = source_dir / "final_001.extxyz"
    frame = _frame(pbc=(True, True, True), y2=2.0)
    write_extxyz_single(source, frame, type_to_species=["Si"])

    def _fake_analyse(**kwargs):
        loaded = read_last_frames_auto(Path(kwargs["analysis_source_path"]), 1, type_to_species=["Si"])[0]
        return _reference_entry(loaded, Path(kwargs["analysis_source_path"])), {}

    monkeypatch.setattr(pc, "analyse_production_box", _fake_analyse)
    ctx = analysis_context_from_standalone_config(
        {
            "analysis": {
                "embed_structures": False,
                "type_to_species": ["Si"],
                "metrics": {"enabled": False, "graph_rules": []},
                "production": {"check_convergence": False},
            }
        }
    )
    outdir = tmp_path / "analysis"
    result = analyze_output_data(analysis_context=ctx, input_path=source_dir, outdir=outdir)

    assert result["graph_outputs"] == {}
    embedding = result["boxes"][0]["structure_embedding"]
    assert embedding["status"] == "referenced"
    assert all(embedding[name] for name in ("structure_hash", "cell_hash", "positions_hash", "symbols_hash"))
    assert (outdir / embedding["manifest_sidecar"]).is_file()
    assert (outdir / embedding["structure_reference"]).is_file()
    assert result["diagnostics"]["source_integrity"]["manifest_locked"] is True
    assert result["diagnostics"]["source_integrity"]["all_sources_verified"] is True

    integrity = json.loads((outdir / "sidecar_integrity.json").read_text())
    assert integrity["schema"] == "vitriflow.sidecar_integrity.v2"
    assert integrity["all_present"] is True
    assert integrity["all_content_hashed"] is True
    assert integrity["all_valid"] is True
    assert integrity["sidecars"]["structure_manifest"]["sha256"]
