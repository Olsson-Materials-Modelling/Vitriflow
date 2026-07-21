from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vitriflow.analysis.dump import DumpFrame
from vitriflow.config import StructureMetricsConfig
from vitriflow.workflows import production_common as pc


def _frame() -> DumpFrame:
    return DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 1], dtype=int),
        positions=np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        cell=np.eye(3, dtype=float) * 10.0,
        origin=np.zeros(3, dtype=float),
    )


def _analyse(
    monkeypatch,
    tmp_path: Path,
    metrics: StructureMetricsConfig,
    *,
    cutoffs=None,
    required_pairs=None,
    fixed_cutoffs=None,
):
    source = tmp_path / "relax.data"
    source.write_text("test fixture\n")
    monkeypatch.setattr(pc, "read_last_frames_auto", lambda *args, **kwargs: [_frame()])
    return pc.analyse_production_box(
        box_id=1,
        outdir=tmp_path,
        melt_stage_dir=tmp_path / "melt",
        quench_stage_dir=tmp_path / "quench",
        relax_stage_dir=tmp_path,
        relax_data_path=source,
        density_mean=2.0,
        density_stderr=0.0,
        metrics_cfg=metrics,
        cutoffs={(1, 1): 1.5} if cutoffs is None else cutoffs,
        required_pairs=[(1, 1)] if required_pairs is None else required_pairs,
        fixed_cutoffs={(1, 1): 1.5} if fixed_cutoffs is None else fixed_cutoffs,
        type_to_species=["Si"],
        md_timestep=1.0,
        analysis_source_path=source,
    )


def test_empty_graph_rules_are_the_cutoff_only_default(monkeypatch, tmp_path):
    from vitriflow.analysis import graph as graph_module
    from vitriflow.analysis import graph_metrics

    metrics = StructureMetricsConfig.model_validate(
        {
            "enabled": True,
            "collect_during_production_stages": False,
            "graph_rules": [],
            "pairs": [{"pair": [1, 1], "cutoff": 1.5}],
            "coordinations": [{"central": 1, "neighbor": 1, "cutoff": 1.5, "expected": 1}],
        }
    )
    # This test targets cutoff-derived bond/coordination data, not void sampling.
    metrics.voids.enabled = False

    def _unexpected(*args, **kwargs):
        raise AssertionError("enhanced graph machinery was called in cutoff-only mode")

    monkeypatch.setattr(graph_metrics, "graph_analysis_for_frame", _unexpected)
    monkeypatch.setattr(graph_module, "build_graph", _unexpected)
    monkeypatch.setattr(graph_module, "verify_manifest_row", _unexpected)

    entry, cutoffs = _analyse(monkeypatch, tmp_path, metrics)

    assert cutoffs == {(1, 1): 1.5}
    assert entry["metrics"]["bondlen_1-1_mean"] == 1.0
    assert entry["metrics"]["coord_1-1_mean"] == 1.0
    assert entry["distributions"]["bondlen"]["bondlen_1-1"]["cdf"]
    assert pc.graph_analysis_requested(metrics) is False
    assert pc.entry_has_graph_analysis(entry) is False
    assert "graph_analysis" not in entry
    # Provenance manifests are always emitted, but their presence does not opt
    # into enhanced graph construction or replace cutoff-driven metrics.
    assert entry["structure_manifest"]["structure_hash"]
    assert entry["structure_manifest"]["source_file_identity"]["sha256"]
    assert (tmp_path / "structure_manifest.json").is_file()
    assert (tmp_path / "structure_snapshot.json").is_file()
    assert "single_rule_output" not in entry


def test_explicit_graph_rule_activates_enhanced_analysis(monkeypatch, tmp_path):
    from vitriflow.analysis import graph as graph_module
    from vitriflow.analysis import graph_metrics

    metrics = StructureMetricsConfig.model_validate(
        {
            "enabled": False,
            "graph_rules": [
                {
                    "name": "requested",
                    "kind": "hard_cutoff",
                    "parameters": {"cutoff": 1.5},
                    "provenance": "unit_test",
                }
            ],
        }
    )
    calls: list[str] = []

    def _fake_analysis(*args, **kwargs):
        calls.append("analysis")
        return {
            "schema": "vitriflow.graph_analysis.v2",
            "structure_manifest": {"structure_hash": "fixture"},
            "graph_rules": [metrics.graph_rules[0].model_dump()],
            "adaptive_graph_rule_records": [],
            "graph_metric_rows": [{"graph_rule_name": "requested"}],
        }

    monkeypatch.setattr(graph_metrics, "graph_analysis_for_frame", _fake_analysis)
    monkeypatch.setattr(graph_module, "verify_manifest_row", lambda *args, **kwargs: calls.append("verify"))
    # The primary-rule selector/build path has its own graph-unit coverage.  This
    # test isolates the production opt-in boundary and payload contract.
    monkeypatch.setattr(pc, "_primary_hard_graph_rule_from_analysis", lambda payload: None)

    entry, _cutoffs = _analyse(monkeypatch, tmp_path, metrics)

    assert calls == ["analysis", "verify"]
    assert pc.graph_analysis_requested(metrics) is True
    assert pc.entry_has_graph_analysis(entry) is True
    assert entry["graph_analysis"]["graph_rules"][0]["name"] == "requested"
    # The mandatory manifest is bound to the immutable source snapshot, not to
    # an independently returned graph payload that could have been assembled
    # from a different file version.
    locked_hash = entry["structure_manifest"]["structure_hash"]
    assert len(locked_hash) == 64
    assert locked_hash != "fixture"
    assert entry["graph_analysis"]["structure_manifest"]["structure_hash"] == locked_hash
    assert entry["single_rule_output"]["present"] is False


def test_graph_finalizer_guards_reject_empty_or_disabled_payloads():
    assert pc.graph_analysis_requested({}) is False
    assert pc.graph_analysis_requested({"graph_rules": []}) is False
    assert pc.graph_analysis_requested({"graph_rules": [{"kind": "hard_cutoff"}]}) is True
    assert pc.entry_has_graph_analysis({}) is False
    assert pc.entry_has_graph_analysis({"graph_analysis": None}) is False
    assert pc.entry_has_graph_analysis({"graph_analysis": {"enabled": False, "graph_rules": [{}]}}) is False
    assert pc.entry_has_graph_analysis(
        {"graph_analysis": {"schema": "vitriflow.graph_analysis.streamed_summary.v1", "streamed_sidecars": True}}
    ) is True


def test_partial_cutoff_map_is_completed_without_overwriting_runtime_values(monkeypatch, tmp_path):
    metrics = StructureMetricsConfig.model_validate({"enabled": False, "graph_rules": []})
    seen: dict = {}

    def _estimate(frames, required_pairs, *, auto, fixed_cutoffs):
        seen["required_pairs"] = list(required_pairs)
        seen["fixed_cutoffs"] = dict(fixed_cutoffs)
        return {**dict(fixed_cutoffs), (1, 2): 2.4}

    monkeypatch.setattr(pc, "estimate_pair_cutoffs", _estimate)
    entry, cutoffs = _analyse(
        monkeypatch,
        tmp_path,
        metrics,
        cutoffs={(1, 1): 1.8},
        required_pairs=[(1, 1), (2, 1)],
        fixed_cutoffs={(1, 1): 9.9},
    )

    assert entry["box"] == 1
    assert seen["required_pairs"] == [(1, 1), (1, 2)]
    assert seen["fixed_cutoffs"] == {(1, 1): 1.8}
    assert cutoffs == {(1, 1): 1.8, (1, 2): 2.4}


def test_disabled_cutoff_scope_rejects_partial_runtime_map(monkeypatch, tmp_path):
    metrics = StructureMetricsConfig.model_validate(
        {"enabled": False, "graph_rules": [], "auto_cutoff": {"scope": "disabled"}}
    )

    with pytest.raises(ValueError, match=r"scope='disabled'.*\(1,2\)"):
        _analyse(
            monkeypatch,
            tmp_path,
            metrics,
            cutoffs={(1, 1): 1.8},
            required_pairs=[(1, 1), (1, 2)],
            fixed_cutoffs={},
        )
