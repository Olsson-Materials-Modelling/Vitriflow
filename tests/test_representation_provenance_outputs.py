from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vitriflow.analysis.dump import DumpFrame
from vitriflow.analysis.graph import (
    GraphRule,
    manifest_row_from_frame,
    structure_serialized_object,
    verify_manifest_row,
)
from vitriflow.analysis.graph_metrics import graph_uncertainty_summary_rows, robust_coordination_partition, write_graph_analysis_outputs
from vitriflow.analysis.provenance import json_dumps_strict, json_sanitize
from vitriflow.config import StructureMetricsConfig


def _frame(types=(1, 2), positions=None, cell=10.0):
    pos = np.asarray(positions if positions is not None else [[1, 1, 1], [2, 1, 1]], dtype=float)
    t = np.asarray(types, dtype=int)
    return DumpFrame(
        timestep=0,
        ids=np.arange(1, len(t) + 1, dtype=int),
        types=t,
        positions=pos,
        cell=np.eye(3) * float(cell),
        origin=np.zeros(3),
    )


def test_strict_json_sanitizes_nonfinite_values():
    payload = {"a": float("nan"), "b": [1.0, float("inf"), -float("inf")]}
    cleaned = json_sanitize(payload)
    assert cleaned == {"a": None, "b": [1.0, None, None]}
    txt = json_dumps_strict(payload)
    assert "NaN" not in txt and "Infinity" not in txt
    json.loads(txt)


def test_manifest_hash_includes_pbc_and_verifies_before_descriptors():
    fr = _frame()
    row = manifest_row_from_frame(fr, box_id=1, source_path=None, source_role="unit", type_to_species=["A", "B"])
    assert row["schema"].endswith("v2")
    assert row["pbc"] == [True, True, True]
    verify_manifest_row(fr, row, type_to_species=["A", "B"])
    bad = dict(row)
    bad["structure_hash"] = "0" * 64
    try:
        verify_manifest_row(fr, bad, type_to_species=["A", "B"])
    except ValueError as exc:
        assert "manifest hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("manifest mismatch did not fail")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"cell": np.ones((3,), dtype=float)}, "cell must have shape"),
        ({"cell": np.zeros((3, 3), dtype=float)}, "strictly positive volume"),
        ({"cell": np.asarray([[10.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 10.0]])}, "cell must contain only finite"),
        ({"positions": np.asarray([1.0, 2.0, 3.0])}, "positions must have shape"),
        ({"positions": np.asarray([[1.0, 1.0, 1.0], [np.inf, 1.0, 1.0]])}, "positions must contain only finite"),
        ({"types": np.asarray([1], dtype=int)}, "atom-type count must match"),
    ],
)
def test_structure_serialization_rejects_malformed_or_nonfinite_frames(replacement, message):
    frame = replace(_frame(), **replacement)
    with pytest.raises(ValueError, match=message):
        structure_serialized_object(frame, type_to_species=["A", "B"])


@pytest.mark.parametrize(
    ("field_name", "tampered_value", "message"),
    [
        ("cell_hash", "0" * 64, "component hash mismatch"),
        ("positions_hash", "1" * 64, "component hash mismatch"),
        ("symbols_hash", "2" * 64, "component hash mismatch"),
        ("pbc", [True, False, True], "pbc mismatch"),
        ("n_atoms", 3, "n_atoms mismatch"),
        ("n_atoms", 2.0, "n_atoms mismatch"),
    ],
)
def test_manifest_verification_rejects_component_and_metadata_tampering(
    field_name,
    tampered_value,
    message,
):
    frame = _frame()
    row = manifest_row_from_frame(
        frame,
        box_id=1,
        source_path=None,
        source_role="unit",
        type_to_species=["A", "B"],
    )
    assert row["n_atoms"] == 2
    row[field_name] = tampered_value
    with pytest.raises(ValueError, match=message):
        verify_manifest_row(frame, row, type_to_species=["A", "B"])


@pytest.mark.parametrize(
    ("identity_field", "tampered_value"),
    [
        ("sha256", "not-a-sha256"),
        ("size_bytes", True),
    ],
)
def test_manifest_verification_rejects_malformed_source_identity(
    tmp_path,
    identity_field,
    tampered_value,
):
    source = tmp_path / "frame.extxyz"
    source.write_text("source identity fixture\n")
    frame = _frame()
    row = manifest_row_from_frame(
        frame,
        box_id=1,
        source_path=source,
        source_role="unit",
        type_to_species=["A", "B"],
    )
    row["source_file_identity"] = {
        **row["source_file_identity"],
        identity_field: tampered_value,
    }
    with pytest.raises(ValueError, match="source artifact identity is malformed"):
        verify_manifest_row(frame, row, type_to_species=["A", "B"])


def test_missing_expected_coordination_is_not_applicable_not_failure():
    fr = _frame(types=[1, 2, 2], positions=[[1, 1, 1], [2, 1, 1], [4, 1, 1]])
    metrics = StructureMetricsConfig.model_validate(
        {
            "enabled": True,
            "coordinations": [{"central": 1, "neighbor": 2}],
            "voids": {"enabled": True, "n_samples": 128, "sampler": "random"},
        }
    )
    interval = GraphRule(name="interval", kind="hard_cutoff", parameters={"r_min": 1.1, "r_max": 2.0}, provenance="unit")
    rows, shell = robust_coordination_partition(fr, metrics, box_id=1, interval_rule=interval, type_to_species=["A", "B"])
    assert rows and shell
    assert rows[0]["status"] == "not_applicable"
    assert rows[0]["reason"] == "expected_integer_coordination_not_configured"
    assert shell[0]["numerical_status"] == "not_applicable"


def test_uncertainty_summary_uses_null_for_undefined_width_over_se():
    rows = [
        {
            "graph_rule_scope": "per_structure",
            "graph_family": "network_graph",
            "metric_family": "coordination",
            "metric_name": "coord_A-B_mean",
            "metric_value": 4.0,
            "graph_rule_name": "r1",
        }
    ]
    out = graph_uncertainty_summary_rows(rows)
    assert out[0]["width_over_se"] is None
    assert out[0]["uncertainty_status"] in {"bootstrap_not_applicable", "zero_variance"}


def test_sidecar_outputs_include_representation_columns(tmp_path: Path):
    write_graph_analysis_outputs(tmp_path, boxes=[], rejected_boxes=[], metrics=StructureMetricsConfig.model_validate({"enabled": False}))
    paths = {p.name for p in tmp_path.iterdir()}
    assert "representation_rules.json" in paths
    assert "metric_results.csv" in paths
    assert "void_scaling_summary.json" in paths
    with (tmp_path / "metric_results.csv").open(newline="") as f:
        header = next(csv.reader(f))
    assert "representation_rule_name" in header
    assert "metric_status" in header
    assert "numerical_status" in header
