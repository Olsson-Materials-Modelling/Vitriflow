from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


def _analysis_context(*, scope: str, cutoffs=None, graph_rules=None):
    from vitriflow.workflows.output_analysis import analysis_context_from_standalone_config

    analysis = {
        "type_to_species": ["X"],
        "metrics": {
            "enabled": True,
            "type_to_species": ["X"],
            "pairs": [{"pair": ["X", "X"]}],
            "auto_cutoff": {"scope": scope},
            "graph_rules": list(graph_rules or []),
        },
        "production": {"min_boxes": 1, "batch_boxes": 1},
    }
    if cutoffs is not None:
        analysis["cutoffs"] = cutoffs
    return analysis_context_from_standalone_config({"analysis": analysis})


def test_auto_cutoff_configuration_validates_scope_and_numerics():
    from pydantic import ValidationError

    from vitriflow.config import AutoCutoffConfig

    cfg = AutoCutoffConfig(scope="per_box", smooth=6)
    assert cfg.scope == "per_box"
    assert cfg.smooth == 7

    with pytest.raises(ValidationError, match="scope"):
        AutoCutoffConfig(scope="unknown")
    with pytest.raises(ValidationError, match="nbins"):
        AutoCutoffConfig(nbins=9)
    with pytest.raises(ValidationError, match="peak_search"):
        AutoCutoffConfig(peak_search=(4.0, 4.0))


def test_disabled_estimator_rejects_missing_but_accepts_complete_fixed_map():
    pytest.importorskip("ase")

    from vitriflow.analysis.structure import estimate_pair_cutoffs
    from vitriflow.config import AutoCutoffConfig

    auto = AutoCutoffConfig(scope="disabled")
    with pytest.raises(ValueError, match=r"scope='disabled'.*\(1,2\)"):
        estimate_pair_cutoffs(
            [],
            required_pairs=[(2, 1)],
            auto=auto,
            fixed_cutoffs={},
        )
    assert estimate_pair_cutoffs(
        [],
        required_pairs=[(2, 1)],
        auto=auto,
        fixed_cutoffs={(1, 2): 2.25},
    ) == {(1, 2): 2.25}


def test_output_cutoff_scope_per_box_never_pools_or_reuses_plan(monkeypatch):
    import vitriflow.workflows.output_analysis as oa

    ctx = _analysis_context(
        scope="per_box",
        cutoffs=[{"pair": [1, 1], "cutoff": 9.9}],
    )
    monkeypatch.setattr(
        oa,
        "_analysis_frames_for_box",
        lambda *args, **kwargs: pytest.fail("per_box scope must not pool frames"),
    )

    cutoffs, provenance = oa._resolve_output_analysis_cutoffs(
        raw_boxes=[],
        ctx=ctx,
        required_pairs=[(1, 1)],
        fixed_cutoffs={},
    )

    assert cutoffs == {}
    assert provenance["scope"] == "per_box"
    assert provenance["mode"] == "per_box_auto"
    assert provenance["plan_cutoffs_reused"] is False


def test_output_cutoff_scope_disabled_requires_complete_explicit_coverage():
    import vitriflow.workflows.output_analysis as oa

    missing_ctx = _analysis_context(scope="disabled")
    with pytest.raises(ValueError, match=r"complete explicit cutoff map.*\(1,1\)"):
        oa._resolve_output_analysis_cutoffs(
            raw_boxes=[],
            ctx=missing_ctx,
            required_pairs=[(1, 1)],
            fixed_cutoffs={},
        )

    explicit_ctx = _analysis_context(
        scope="disabled",
        cutoffs=[{"pair": [1, 1], "cutoff": 2.4}],
    )
    cutoffs, provenance = oa._resolve_output_analysis_cutoffs(
        raw_boxes=[],
        ctx=explicit_ctx,
        required_pairs=[(1, 1)],
        fixed_cutoffs={},
    )
    assert cutoffs == {(1, 1): 2.4}
    assert provenance["mode"] == "disabled_explicit"
    assert provenance["plan_cutoffs_reused"] is True


def test_output_cutoff_scope_pooled_preserves_explicit_legacy_shared_fallback():
    import vitriflow.workflows.output_analysis as oa

    ctx = _analysis_context(scope="pooled_ensemble")
    cutoffs, provenance = oa._resolve_output_analysis_cutoffs(
        raw_boxes=[],
        ctx=ctx,
        required_pairs=[(1, 1)],
        fixed_cutoffs={},
    )
    assert cutoffs == {}
    assert provenance["scope"] == "pooled_ensemble"
    assert provenance["mode"] == "per_box_fallback"
    assert provenance["fallback_policy"] == "first_successful_box_then_shared"
    assert "Set scope='per_box'" in provenance["notes"][-1]


def test_cli_graph_rules_are_additive_and_names_are_unique():
    from vitriflow.cli import _graph_rules_from_cli_args
    from vitriflow.workflows.output_analysis import _merge_analysis_graph_rules

    ctx = _analysis_context(
        scope="per_box",
        graph_rules=[
            {
                "name": "yaml_rule",
                "kind": "hard_cutoff",
                "parameters": {"cutoff": 1.8},
            }
        ],
    )
    cli_rules = _graph_rules_from_cli_args(
        Namespace(
            graph_cutoff=[2.0],
            graph_cutoff_sweep=None,
            graph_cutoff_interval=None,
            graph_interval_points=9,
            soft_logistic=None,
        )
    )
    merged = _merge_analysis_graph_rules(ctx.metrics_cfg, cli_rules)
    assert [rule.name for rule in merged.graph_rules] == [
        "yaml_rule",
        "cli_hard_cutoff_1",
    ]

    collision = _analysis_context(
        scope="per_box",
        graph_rules=[
            {
                "name": "cli_hard_cutoff_1",
                "kind": "hard_cutoff",
                "parameters": {"cutoff": 1.8},
            }
        ],
    )
    with pytest.raises(ValueError, match="Duplicate graph-rule name 'cli_hard_cutoff_1'"):
        _merge_analysis_graph_rules(collision.metrics_cfg, cli_rules)


def test_analyze_output_installs_merged_rules_in_context_before_cutoff_resolution(
    monkeypatch,
    tmp_path: Path,
):
    import vitriflow.workflows.output_analysis as oa

    ctx = _analysis_context(
        scope="per_box",
        graph_rules=[
            {
                "name": "yaml_rule",
                "kind": "hard_cutoff",
                "parameters": {"cutoff": 1.8},
            }
        ],
    )
    monkeypatch.setattr(oa, "discover_output_dataset", lambda *args, **kwargs: ([], [], [], {}))
    captured = {}

    class _StopAfterResolution(Exception):
        pass

    def _capture(**kwargs):
        captured["ctx"] = kwargs["ctx"]
        raise _StopAfterResolution

    monkeypatch.setattr(oa, "_resolve_output_analysis_cutoffs", _capture)
    with pytest.raises(_StopAfterResolution):
        oa.analyze_output_data(
            analysis_context=ctx,
            input_path=tmp_path / "input",
            outdir=tmp_path / "output",
            graph_rules_override=[
                {
                    "name": "cli_rule",
                    "kind": "hard_cutoff",
                    "parameters": {"cutoff": 2.0},
                    "provenance": "unit-test CLI",
                }
            ],
        )

    assert [rule.name for rule in captured["ctx"].metrics_cfg.graph_rules] == [
        "yaml_rule",
        "cli_rule",
    ]


def test_custom_resume_restores_persisted_cutoffs_and_rejects_conflicts():
    from vitriflow.workflows.custom_schedule import _restore_custom_resume_cutoffs

    previous = {
        "production": {
            "cutoffs": [
                {"pair": [1, 1], "cutoff": 1.85},
                {"pair": [1, 2], "cutoff": 2.15},
            ]
        }
    }
    restored = _restore_custom_resume_cutoffs(
        previous,
        fixed_cutoffs={(1, 1): 1.85},
    )
    assert restored == {(1, 1): 1.85, (1, 2): 2.15}

    with pytest.raises(RuntimeError, match="inconsistent explicit cutoff"):
        _restore_custom_resume_cutoffs(
            previous,
            fixed_cutoffs={(1, 1): 1.9},
        )


def test_custom_resume_cutoff_parser_rejects_malformed_persisted_state():
    from vitriflow.workflows.custom_schedule import _restore_custom_resume_cutoffs

    with pytest.raises(RuntimeError, match=r"production.cutoffs\[0\].*both pair and cutoff"):
        _restore_custom_resume_cutoffs(
            {"production": {"cutoffs": [{"pair": [1, 2]}]}},
            fixed_cutoffs={},
        )
