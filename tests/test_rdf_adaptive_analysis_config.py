from __future__ import annotations

from vitriflow.workflows.metric_requirements import fixed_cutoffs_from_metrics, required_pairs_from_metrics
from vitriflow.workflows.output_analysis import (
    _metrics_have_adaptive_graph_rules,
    _resolve_output_analysis_cutoffs,
    analysis_context_from_standalone_config,
)


def test_rdf_adaptive_analyze_output_does_not_synthesize_hidden_legacy_cutoff_map():
    ctx = analysis_context_from_standalone_config(
        {
            "analysis": {
                "type_to_species": ["Si", "N"],
                "check_convergence": False,
                "metrics": {
                    "enabled": True,
                    "type_to_species": ["Si", "N"],
                    "coordinations": [
                        {"central": "Si", "neighbor": "N", "expected": 4},
                        {"central": "N", "neighbor": "Si", "expected": 3},
                    ],
                    "rings": {
                        "enabled": True,
                        "mode": "bond_graph",
                        "nodes": ["Si", "N"],
                        "bond_pairs": [{"pair": ["Si", "N"]}],
                    },
                    "graph_rules": [
                        {
                            "name": "si3n4_rdf_shell_network",
                            "kind": "rdf_adaptive",
                            "parameters": {
                                "pairs": [["Si", "N"]],
                                "mode": "all",
                                "search_radius": "auto",
                                "connectivity_fraction": 1.0,
                            },
                        }
                    ],
                },
            }
        }
    )
    required = required_pairs_from_metrics(ctx.metrics_cfg, type_to_species=ctx.type_to_species)
    fixed = fixed_cutoffs_from_metrics(ctx.metrics_cfg, type_to_species=ctx.type_to_species)
    cutoffs, provenance = _resolve_output_analysis_cutoffs(
        raw_boxes=[],
        ctx=ctx,
        required_pairs=required,
        fixed_cutoffs=fixed,
    )

    assert _metrics_have_adaptive_graph_rules(ctx.metrics_cfg) is True
    assert required == [(1, 2)]
    assert fixed == {}
    assert cutoffs == {}
    assert provenance["mode"] == "adaptive_graph_rules_only"
    assert "No legacy single-cutoff map" in provenance["notes"][0]
