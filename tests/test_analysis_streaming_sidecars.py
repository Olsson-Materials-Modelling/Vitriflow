import csv
import json
from pathlib import Path

from vitriflow.analysis.graph_metrics import (
    finalize_streamed_graph_analysis_outputs,
    strip_graph_analysis_payload,
    write_graph_analysis_entry_chunks,
)
from vitriflow.config import ProductionEnsembleConfig
from vitriflow.workflows.output_analysis import analysis_context_from_standalone_config


def _metric_row(box_id, rule, value):
    return {
        "box_id": box_id,
        "structure_hash": f"hash-{box_id}",
        "material_id": "test",
        "source_space": "structure",
        "representation_map": "graph_induction",
        "descriptor_map": "metric",
        "graph_rule_scope": "per_structure",
        "graph_family": "network_graph",
        "graph_rule_name": rule,
        "graph_rule_kind": "hard_cutoff",
        "graph_rule_parameters": "{}",
        "graph_rule_provenance": "{}",
        "representation_rule_name": rule,
        "representation_rule_kind": "hard_graph",
        "representation_rule_parameters": "{}",
        "representation_rule_provenance": "{}",
        "metric_family": "coordination",
        "metric_name": "coord_Si-N_mean",
        "metric_value": value,
        "metric_units": "count",
        "metric_status": "ok",
        "metric_status_reason": "",
        "numerical_status": "ok",
        "uncertainty_status": "ok",
    }


def test_streamed_graph_sidecars_strip_and_finalize_without_embedding_rows(tmp_path):
    chunk_dir = tmp_path / ".analysis_stream_chunks"
    entries = []
    for box_id, base in [(1, 3.9), (2, 4.1)]:
        entry = {
            "box": box_id,
            "structure_manifest": {
                "box_id": box_id,
                "structure_hash": f"hash-{box_id}",
                "source_path": f"box_{box_id:03d}/final.restart",
                "source_role": "final_structure",
            },
            "graph_analysis": {
                "structure_manifest": {
                    "box_id": box_id,
                    "structure_hash": f"hash-{box_id}",
                    "source_path": f"box_{box_id:03d}/final.restart",
                    "source_role": "final_structure",
                },
                "graph_rules": [
                    {"name": "r1", "kind": "hard_cutoff", "parameters": {"cutoff": 1.9}, "provenance": {}},
                    {"name": "r2", "kind": "hard_cutoff", "parameters": {"cutoff": 2.0}, "provenance": {}},
                ],
                "adaptive_graph_rule_records": [],
                "graph_metric_rows": [
                    _metric_row(box_id, "r1", base),
                    _metric_row(box_id, "r2", base + 0.2),
                ],
                "coordination_stability_rows": [],
                "shell_separability_rows": [],
            },
        }
        summary = write_graph_analysis_entry_chunks(entry, chunk_dir)
        stripped = strip_graph_analysis_payload(entry, summary)
        assert stripped["graph_analysis"]["streamed_sidecars"] is True
        assert "graph_metric_rows" not in stripped["graph_analysis"]
        entries.append(stripped)

    paths = finalize_streamed_graph_analysis_outputs(
        tmp_path,
        chunk_dir=chunk_dir,
        boxes=entries,
        rejected_boxes=[],
        metrics=None,
        type_to_species=None,
        legacy_cutoffs=None,
    )
    metric_path = tmp_path / paths["graph_metric_by_rule"]
    with metric_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {r["graph_rule_name"] for r in rows} == {"r1", "r2"}

    manifest = json.loads((tmp_path / paths["structure_manifest"]).read_text())
    assert [r["box_id"] for r in manifest["structures"]] == [1, 2]

    uncertainty_path = tmp_path / paths["graph_uncertainty_summary"]
    with uncertainty_path.open(newline="") as f:
        uncertainty_rows = list(csv.DictReader(f))
    assert uncertainty_rows
    assert uncertainty_rows[0]["metric_name"] == "coord_Si-N_mean"
    assert float(uncertainty_rows[0]["width"]) > 0.0


def test_standalone_config_accepts_analysis_streaming_parallel_controls():
    ctx = analysis_context_from_standalone_config(
        {
            "analysis": {
                "analysis_workers": 8,
                "analysis_streaming": True,
                "analysis_max_in_flight": 12,
                "type_to_species": ["Si", "N"],
                "metrics": {
                    "enabled": False,
                    "type_to_species": ["Si", "N"],
                    "pairs": [],
                    "rings": {"enabled": False},
                    "gr": [],
                    "sq": [],
                },
            }
        }
    )
    assert ctx.analysis_workers == 8
    assert ctx.analysis_streaming is True
    assert ctx.analysis_max_in_flight == 12


def test_production_ensemble_config_validates_analysis_workers():
    cfg = ProductionEnsembleConfig(analysis_workers=16, analysis_streaming=True, analysis_max_in_flight=16)
    assert cfg.analysis_workers == 16
    assert cfg.analysis_max_in_flight == 16


def test_streamed_graph_sidecars_compact_repeated_derivation_payloads(tmp_path):
    huge_derivation = [
        {
            "pair": [1, 2],
            "pair_species": ["A", "B"],
            "selected_cutoff": 1.9,
            "rdf_first_minimum": 1.8,
            "shell_separability": {"available": True, "shell_objective_cutoff": 1.9},
            "connectivity": {"per_structure": [{"box": i, "component_count": 1} for i in range(2000)]},
        }
    ]
    params = {
        "cutoffs": [{"pair": [1, 2], "cutoff": 1.9}],
        "pair_cutoffs": [{"pair": [1, 2], "cutoff": 1.9}],
        "pair_intervals": [{"pair": [1, 2], "r_min": 1.8, "r_max": 2.0}],
        "rdf_adaptive": True,
        "structure_hash": "hash-1",
        "derivation_method": "unit_test_huge_payload",
        "derivation": huge_derivation,
    }
    row = _metric_row(1, "adaptive", 4.0)
    row["graph_rule_parameters"] = params
    row["representation_rule_parameters"] = params
    entry = {
        "box": 1,
        "structure_manifest": {"box_id": 1, "structure_hash": "hash-1", "source_path": "final.restart", "source_role": "final_structure"},
        "graph_analysis": {
            "structure_manifest": {"box_id": 1, "structure_hash": "hash-1", "source_path": "final.restart", "source_role": "final_structure"},
            "graph_rules": [{"name": "adaptive", "kind": "hard_cutoff", "parameters": params, "provenance": {}}],
            "adaptive_graph_rule_records": [{"name": "adaptive", "kind": "hard_cutoff", "parameters": params, "provenance": {}}],
            "graph_metric_rows": [row],
            "coordination_stability_rows": [],
            "shell_separability_rows": [],
        },
    }
    chunk_dir = tmp_path / ".analysis_stream_chunks"
    write_graph_analysis_entry_chunks(entry, chunk_dir)
    paths = finalize_streamed_graph_analysis_outputs(tmp_path, chunk_dir=chunk_dir, boxes=[entry], rejected_boxes=[], metrics=None)

    # Successful finalization removes transient chunk files by default.
    assert not chunk_dir.exists()

    graph_rules = json.loads((tmp_path / paths["graph_rules"]).read_text())
    compact_params = graph_rules["graph_rules"][0]["parameters"]
    assert "derivation" not in compact_params
    assert compact_params["derivation_stored_in_sidecar"] is True
    assert compact_params["derivation_ref"].startswith("deriv:")

    deriv = json.loads((tmp_path / paths["adaptive_graph_rule_derivations"]).read_text())
    assert deriv["adaptive_graph_rule_derivation_records"]
    assert deriv["adaptive_graph_rule_derivation_records"][0]["derivation"] == huge_derivation

    with (tmp_path / paths["graph_metric_by_rule"]).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert len(rows[0]["graph_rule_parameters"]) < 32768
    assert "\"derivation\":" not in rows[0]["graph_rule_parameters"]
    assert "derivation_ref" in rows[0]["graph_rule_parameters"]
    assert "derivation_pair_summary" not in rows[0]["graph_rule_parameters"]
