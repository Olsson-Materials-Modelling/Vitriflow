from __future__ import annotations

import json

from vitriflow.workflows.output_analysis import (
    analysis_context_from_standalone_config,
    analyze_output_data,
)


def _context(*, graph_rules, streaming: bool):
    return analysis_context_from_standalone_config(
        {
            "metrics": {"enabled": False, "graph_rules": list(graph_rules)},
            "production": {
                "check_convergence": False,
                "analysis_streaming": bool(streaming),
            },
        }
    )


def test_default_output_analysis_writes_only_general_sidecars(monkeypatch, tmp_path):
    import vitriflow.workflows.production_common as pc

    def _unexpected_writer(*args, **kwargs):
        raise AssertionError("graph writer called without explicit graph_rules")

    monkeypatch.setattr(pc, "write_graph_analysis_outputs", _unexpected_writer)
    input_dir = tmp_path / "empty_ensemble"
    input_dir.mkdir()
    outdir = tmp_path / "analysis"

    result = analyze_output_data(
        analysis_context=_context(graph_rules=[], streaming=True),
        input_path=input_dir,
        outdir=outdir,
    )

    assert result["graph_outputs"] == {}
    assert result["paths"]["ensemble_cdfs"] == "ensemble_cdfs.json"
    assert result["paths"]["sidecar_integrity"] == "sidecar_integrity.json"
    assert set(result["sidecars"]) == {
        "ensemble_cdfs",
        "sidecar_integrity",
        "structure_manifest",
        "structure_references",
    }
    assert "graph_rules" not in result
    assert "graph_metric_by_rule" not in result
    assert result["structure_manifest"]["exists"] is True
    assert result["structure_manifest"]["sha256"]
    assert (outdir / "ensemble_cdfs.json").is_file()
    assert (outdir / "sidecar_integrity.json").is_file()
    assert (outdir / "structure_manifest.json").is_file()
    assert (outdir / "structure_references.json").is_file()
    assert json.loads((outdir / "ensemble_cdfs.json").read_text())["schema"] == "vitriflow.ensemble_cdfs.v1"
    assert not (outdir / ".analysis_stream_chunks").exists()


def test_explicit_in_memory_graph_analysis_calls_writer_once(monkeypatch, tmp_path):
    import vitriflow.workflows.production_common as pc

    calls: list[dict] = []

    def _writer(outdir, **kwargs):
        calls.append(dict(kwargs))
        return {"graph_rules": "graph_rules.json"}

    monkeypatch.setattr(pc, "write_graph_analysis_outputs", _writer)
    input_dir = tmp_path / "empty_ensemble"
    input_dir.mkdir()

    result = analyze_output_data(
        analysis_context=_context(
            graph_rules=[
                {
                    "name": "requested",
                    "kind": "hard_cutoff",
                    "parameters": {"cutoff": 1.5},
                }
            ],
            streaming=False,
        ),
        input_path=input_dir,
        outdir=tmp_path / "analysis",
    )

    assert len(calls) == 1
    assert calls[0]["metrics"].graph_rules[0].name == "requested"
    assert result["graph_outputs"] == {"graph_rules": "graph_rules.json"}
    assert result["graph_rules"]["path"] == "graph_rules.json"
    assert result["graph_rules"]["exists"] is False
    assert result["graph_rules"]["status"] == "missing"
    assert result["paths"]["ensemble_cdfs"] == "ensemble_cdfs.json"
    assert "ensemble_cdfs" not in result["graph_outputs"]
    assert "sidecar_integrity" not in result["graph_outputs"]
