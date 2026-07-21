"""Regression tests for the 0.4.30.0 combined-tree provenance fixes.

1. Amorphous ``_structure_hash_or_none`` must equal the canonical
   ``graph.structure_hash`` (which hashes ``cell, species, positions, pbc``).
   A prior inline reimplementation dropped ``pbc`` and desynchronised the hash.
2. With ``embed_structures=false`` and no graph rules (the default cutoff-only
   path), the structure-snapshot reference must still carry real content hashes
   and a source path -- not all-None -- so the audit/reload contract holds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vitriflow.analysis import amorphous, graph
from vitriflow.analysis.dump import DumpFrame
from vitriflow.config import StructureMetricsConfig
from vitriflow.workflows import production_common as pc


def _frame(pbc=(True, True, True)) -> DumpFrame:
    fr = DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 1], dtype=int),
        positions=np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        cell=np.eye(3, dtype=float) * 10.0,
        origin=np.zeros(3, dtype=float),
        pbc=pbc,
    )
    return fr


def test_amorphous_structure_hash_matches_canonical_and_includes_pbc():
    import hashlib
    import json

    fr = _frame()
    delegated = amorphous._structure_hash_or_none(fr, type_to_species=["Si"])
    canonical = graph.structure_hash(fr, type_to_species=["Si"])
    assert delegated is not None
    # Delegation -> byte-identical to the graph/manifest provenance hash.
    assert delegated == canonical
    # The canonical hashed object is pbc-inclusive; the old amorphous
    # reimplementation dropped pbc. Confirm pbc is part of the object the hash is
    # computed over, and that a pbc-blind (3-key) hash of the same frame differs
    # -- i.e. the delegated hash is genuinely no longer pbc-blind.
    obj = graph.structure_serialized_object(fr, type_to_species=["Si"])
    assert set(obj.keys()) >= {"cell", "species", "positions", "pbc"}
    pbc_blind = hashlib.sha256(
        json.dumps(
            {k: obj[k] for k in ("cell", "species", "positions")},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert delegated != pbc_blind
    assert graph.structure_hash(_frame((True, False, True)), type_to_species=["Si"]) != delegated


def test_embed_structures_false_default_audit_ref_carries_real_hashes(monkeypatch, tmp_path):
    metrics = StructureMetricsConfig.model_validate(
        {"enabled": True, "collect_during_production_stages": False, "graph_rules": []}
    )
    metrics.voids.enabled = False

    source = tmp_path / "relax.data"
    source.write_text("audit fixture\n")
    monkeypatch.setattr(pc, "read_last_frames_auto", lambda *a, **k: [_frame()])

    entry, _cutoffs = pc.analyse_production_box(
        box_id=1,
        outdir=tmp_path,
        melt_stage_dir=tmp_path / "melt",
        quench_stage_dir=tmp_path / "quench",
        relax_stage_dir=tmp_path,
        relax_data_path=source,
        density_mean=2.0,
        density_stderr=0.0,
        metrics_cfg=metrics,
        cutoffs={(1, 1): 1.5},
        required_pairs=[(1, 1)],
        fixed_cutoffs={(1, 1): 1.5},
        type_to_species=["Si"],
        md_timestep=1.0,
        analysis_source_path=source,
        embed_structures=False,
    )

    # Graph treatment stays strictly opt-in: no graph descriptors by default.
    assert pc.graph_analysis_requested(metrics) is False
    assert "graph_analysis" not in entry

    ref = entry["structure"]
    assert ref["schema"] == "vitriflow.structure_snapshot_ref.v1"
    assert ref["embedded"] is False
    # The audit reference must carry real content hashes + source path.
    assert ref["structure_hash"] is not None
    assert ref["cell_hash"] is not None
    assert ref["positions_hash"] is not None
    assert ref["source_path"] is not None
    # And the structure_hash must equal the canonical manifest hash for the frame.
    assert ref["structure_hash"] == graph.structure_hash(_frame(), type_to_species=["Si"])


def _cutoff_only_metrics() -> StructureMetricsConfig:
    metrics = StructureMetricsConfig.model_validate(
        {"enabled": True, "collect_during_production_stages": False, "graph_rules": []}
    )
    metrics.voids.enabled = False
    return metrics


def test_explicit_missing_analysis_source_is_not_silently_replaced(monkeypatch, tmp_path):
    final_structure = tmp_path / "relax.data"
    final_structure.write_text("fallback must not be selected\n")
    missing_explicit = tmp_path / "requested.extxyz"

    def _unexpected_reader(*args, **kwargs):
        raise AssertionError("reader must not run when the explicit source is missing")

    monkeypatch.setattr(pc, "read_last_frames_auto", _unexpected_reader)
    with pytest.raises(FileNotFoundError, match="Explicit production analysis source"):
        pc.analyse_production_box(
            box_id=1,
            outdir=tmp_path,
            melt_stage_dir=tmp_path / "melt",
            quench_stage_dir=tmp_path / "quench",
            relax_stage_dir=tmp_path,
            relax_data_path=final_structure,
            density_mean=2.0,
            density_stderr=0.0,
            metrics_cfg=_cutoff_only_metrics(),
            cutoffs={(1, 1): 1.5},
            required_pairs=[(1, 1)],
            fixed_cutoffs={(1, 1): 1.5},
            type_to_species=["Si"],
            md_timestep=1.0,
            analysis_source_path=missing_explicit,
            analysis_source_role="final_structure",
        )


def test_automatic_source_fallback_records_the_actual_file_and_role(monkeypatch, tmp_path):
    final_structure = tmp_path / "relax.data"
    final_structure.write_text("selected fallback\n")
    missing_trajectory = tmp_path / "missing.extxyz"
    missing_dump = tmp_path / "missing.lammpstrj"
    seen: list = []

    def _reader(path, *args, **kwargs):
        seen.append(path)
        return [_frame()]

    monkeypatch.setattr(pc, "read_last_frames_auto", _reader)
    entry, _cutoffs = pc.analyse_production_box(
        box_id=1,
        outdir=tmp_path,
        melt_stage_dir=tmp_path / "melt",
        quench_stage_dir=tmp_path / "quench",
        relax_stage_dir=tmp_path,
        relax_data_path=final_structure,
        density_mean=2.0,
        density_stderr=0.0,
        metrics_cfg=_cutoff_only_metrics(),
        cutoffs={(1, 1): 1.5},
        required_pairs=[(1, 1)],
        fixed_cutoffs={(1, 1): 1.5},
        type_to_species=["Si"],
        md_timestep=1.0,
        relax_dump_path=missing_dump,
        relax_traj_path=missing_trajectory,
    )

    assert len(seen) == 1
    # Parsing is deliberately performed from a verified immutable copy; the
    # public provenance below must still identify the selected fallback.
    assert Path(seen[0]).suffix == final_structure.suffix
    assert Path(seen[0]) != final_structure
    assert not Path(seen[0]).exists()
    selection = entry["analysis_source_selection"]
    assert selection == {
        "schema": "vitriflow.analysis_source_selection.v1",
        "explicit_source_requested": False,
        "preferred_path": "missing.extxyz",
        "preferred_role": "relax_trajectory",
        "selected_path": "relax.data",
        "selected_role": "final_structure",
        "fallback_used": True,
        "fallback_reason": "relax_trajectory_missing_or_not_file",
    }
    assert entry["analysis_source_role"] == "final_structure"
    assert entry["paths"]["analysis_source"] == "relax.data"
    assert entry["structure_manifest"]["source_path"] == "relax.data"
    assert entry["structure_manifest"]["source_role"] == "final_structure"
    assert entry["structure_manifest"]["source_selection"] == selection
