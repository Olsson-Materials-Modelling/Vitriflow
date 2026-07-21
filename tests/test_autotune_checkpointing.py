from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

pytest.importorskip("ase")

from vitriflow.config import ConvergenceConfig
from vitriflow.workflows.autotune import (
    _attach_production_state_integrity,
    _initial_production_checkpoint_status,
    _run_production_ensemble,
)
from vitriflow.workflows.progress import CondensedProgressLog


def test_preproduction_checkpoint_status_matches_enabled_state(tmp_path: Path):
    from vitriflow.workflows.resume_integrity import (
        validate_production_state_semantics,
    )

    assert _initial_production_checkpoint_status(enabled=True) == "starting"
    disabled_status = _initial_production_checkpoint_status(enabled=False)
    assert disabled_status == "not_requested"

    disabled_state = {
        "enabled": False,
        "status": disabled_status,
        "execution_status": disabled_status,
        "n_boxes": 0,
        "n_boxes_accepted": 0,
        "n_boxes_rejected": 0,
        "n_boxes_total": 0,
        "boxes": [],
        "rejected_boxes": [],
    }
    validate_production_state_semantics(disabled_state)
    protected = _attach_production_state_integrity(disabled_state, outdir=tmp_path)
    assert protected["status"] == "not_requested"
    assert protected["execution_status"] == "not_requested"


def test_production_ensemble_emits_start_checkpoint_before_first_box(monkeypatch, tmp_path: Path):
    states: list[dict] = []

    monkeypatch.setattr(
        "vitriflow.workflows.autotune.plan_production_stage_diagnostics",
        lambda **kwargs: {
            "dump_traj": True,
            "dump_every": 500,
            "collect_stage_metric_series": False,
            "collect_elastic_series": {"melt": False, "quench": False, "relax": False},
            "need_stage_dump": {"melt": True, "quench": True, "relax": True},
            "quench_dump_every": 250,
            "quench_window_steps_range": (0.0, 10.0),
        },
    )
    monkeypatch.setattr("vitriflow.workflows.autotune.required_pairs_from_metrics", lambda *args, **kwargs: [])
    monkeypatch.setattr("vitriflow.workflows.autotune.fixed_cutoffs_from_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr("vitriflow.workflows.autotune.should_run_elastic_screen", lambda *args, **kwargs: (False, False, None))
    monkeypatch.setattr(
        "vitriflow.workflows.autotune._stage_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    config = SimpleNamespace(
        random_seed=7,
        md=SimpleNamespace(pressure=0.0),
        autotune=SimpleNamespace(
            production=SimpleNamespace(
                enabled=True,
                min_boxes=1,
                max_boxes=1,
                batch_boxes=1,
                check_convergence=True,
                dump_trajectory=True,
                dump_every_steps=500,
                dft_opt=None,
                exclude_coordination_defects=False,
                rejects_subdir="rejects",
                store_distributions=True,
                consecutive_converged_checks=1,
                bondlen_cdf_points=200,
                angle_cdf_points=180,
            ),
            convergence=ConvergenceConfig(),
        ),
    )
    metrics_cfg = SimpleNamespace(enabled=True, time_average_frames=1, time_average_stride=1, elastic=SimpleNamespace())
    md_use = SimpleNamespace(timestep=0.001, stage_continuity="discontinuous", force_isotropic=False)
    q_cfg = SimpleNamespace(t_final=300.0, relax_steps=1000)
    tm_cfg = SimpleNamespace(msd_every=100)
    base_data = tmp_path / "base.data"
    base_data.write_text("# placeholder\n")
    progress = CondensedProgressLog(tmp_path / "condensed.log")

    with pytest.raises(RuntimeError, match="boom"):
        _run_production_ensemble(
            config=config,
            outdir=tmp_path,
            runner=object(),
            pot_cfg=SimpleNamespace(),
            md_use=md_use,
            potential_lines=None,
            type_to_species=["Al"],
            metrics_cfg=metrics_cfg,
            tm_cfg=tm_cfg,
            q_cfg=q_cfg,
            size_base_data=base_data,
            chosen_replicate=[1, 1, 1],
            chosen_rate=10.0,
            dt_ref=0.001,
            dt_mq=0.001,
            cooling_rate_ps=10.0,
            cutoffs_rate={},
            cutoffs_size={},
            T_high=1200.0,
            high_total_steps=5000,
            resume_state=None,
            progress=progress,
            checkpoint_cb=lambda state: states.append(state),
        )

    assert states, "checkpoint callback should be invoked before stage execution"
    first = states[0]
    assert first["status"] == "starting"
    assert first["enabled"] is True
    assert first["n_boxes_total"] == 0
    assert first["replicate"] == [1, 1, 1]
    assert first["T_high"] == pytest.approx(1200.0)


def _finish_from_resumed_box(
    monkeypatch,
    tmp_path: Path,
    *,
    graph_rules,
    graph_writer,
    check_convergence: bool = False,
    required_streak: int = 1,
    store_distributions: bool = True,
    resume_overrides: dict | None = None,
):
    states: list[dict] = []
    monkeypatch.setattr(
        "vitriflow.workflows.autotune.plan_production_stage_diagnostics",
        lambda **kwargs: {
            "dump_traj": True,
            "dump_every": 500,
            "collect_stage_metric_series": False,
            "collect_elastic_series": {"melt": False, "quench": False, "relax": False},
            "need_stage_dump": {"melt": True, "quench": True, "relax": True},
            "quench_dump_every": 250,
            "quench_window_steps_range": (0.0, 10.0),
        },
    )
    monkeypatch.setattr("vitriflow.workflows.autotune.required_pairs_from_metrics", lambda *args, **kwargs: [])
    monkeypatch.setattr("vitriflow.workflows.autotune.fixed_cutoffs_from_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr("vitriflow.workflows.autotune.summarize_production_crystal_motifs", lambda *args, **kwargs: {})
    monkeypatch.setattr("vitriflow.workflows.autotune.write_graph_analysis_outputs", graph_writer)

    config = SimpleNamespace(
        random_seed=7,
        md=SimpleNamespace(pressure=0.0),
        autotune=SimpleNamespace(
            production=SimpleNamespace(
                enabled=True,
                min_boxes=1,
                max_boxes=1,
                batch_boxes=1,
                check_convergence=check_convergence,
                dump_trajectory=True,
                dump_every_steps=500,
                dft_opt=None,
                exclude_coordination_defects=False,
                rejects_subdir="rejects",
                store_distributions=store_distributions,
                consecutive_converged_checks=required_streak,
                bondlen_cdf_points=200,
                angle_cdf_points=180,
            ),
            convergence=ConvergenceConfig(),
        ),
    )
    metrics_cfg = SimpleNamespace(
        enabled=True,
        graph_rules=list(graph_rules),
        time_average_frames=1,
        time_average_stride=1,
        elastic=SimpleNamespace(),
    )
    base_data = tmp_path / "base.data"
    base_data.write_text("# placeholder\n")
    (tmp_path / "structure_snapshot.json").write_text(
        json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1})
    )
    manifest_row = {"structure_hash": "h1"}
    (tmp_path / "structure_manifest.json").write_text(
        json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest_row]})
    )
    resume_state = {
        "status": "running",
        "execution_status": "running",
        "boxes": [{
            "box": 1,
            "density": 2.35,
            "metrics": {},
            "distributions": {},
            "seeds": {"warmup": 11, "melt": 12, "quench": 13, "relax": 14},
            "paths": {
                "relax_data": "base.data",
                "structure_snapshot": "structure_snapshot.json",
                "structure_manifest": "structure_manifest.json",
            },
            "structure_manifest": manifest_row,
        }],
        "rejected_boxes": [],
        "convergence_spec": {"bondlen_names": []},
        "converged_md": False,
        "convergence_md": {},
        "convergence_streak": 0,
        "required_convergence_streak": required_streak,
        "last_convergence_evaluated_n_boxes_total": None,
        "last_convergence_evaluated_n_boxes_accepted": None,
        "resumable": True,
    }
    resume_state.update(dict(resume_overrides or {}))
    resume_state = _attach_production_state_integrity(resume_state, outdir=tmp_path)

    result = _run_production_ensemble(
        config=config,
        outdir=tmp_path,
        runner=object(),
        pot_cfg=SimpleNamespace(),
        md_use=SimpleNamespace(timestep=0.001, stage_continuity="discontinuous", force_isotropic=False),
        potential_lines=None,
        type_to_species=["Al"],
        metrics_cfg=metrics_cfg,
        tm_cfg=SimpleNamespace(msd_every=100),
        q_cfg=SimpleNamespace(t_final=300.0, relax_steps=1000),
        size_base_data=base_data,
        chosen_replicate=[1, 1, 1],
        chosen_rate=10.0,
        dt_ref=0.001,
        dt_mq=0.001,
        cooling_rate_ps=10.0,
        cutoffs_rate={},
        cutoffs_size={},
        T_high=1200.0,
        high_total_steps=5000,
        resume_state=resume_state,
        progress=CondensedProgressLog(tmp_path / "condensed.log"),
        checkpoint_cb=lambda state: states.append(state),
    )
    return result, states


def test_production_checkpoint_and_terminal_skip_graph_writer_by_default(monkeypatch, tmp_path: Path):
    def _unexpected_writer(*args, **kwargs):
        raise AssertionError("graph writer called without explicit graph_rules")

    result, states = _finish_from_resumed_box(
        monkeypatch,
        tmp_path,
        graph_rules=[],
        graph_writer=_unexpected_writer,
    )

    assert states
    assert all(state["graph_outputs"] == {} for state in states)
    assert all(state["paths"] == {} for state in states)
    assert result["status"] == "ok"
    assert result["execution_status"] == "completed"
    assert result["converged"] is None
    assert result["converged_md"] is None
    assert result["convergence_status"] == "fixed_count_unassessed"
    assert result["convergence_inference_status"] == (
        "fixed_n_terminal_posthoc_not_sequentially_valid"
    )
    assert result["achieved_convergence_degree"]["n_boxes"] == 1
    assert result["posthoc_convergence_criterion_met"] is False
    assert result["convergence"]["status"] == "fixed_n_terminal_posthoc_assessed"
    assert result["convergence"]["sampling_design"] == "fixed_n"
    assert result["convergence"]["used_for_stopping"] is False
    assert result["convergence"]["stopping_status"] == "fixed_count_unassessed"
    assert result["convergence"]["posthoc_failed_items"] == [
        {
            "section": "ci",
            "name": "scalar:density",
            "reason": "tolerance_not_met",
        },
        {
            "section": "stability",
            "name": "stability",
            "reason": "active_section_unassessed",
        },
    ]
    assert result["graph_outputs"] == {}
    assert result["paths"] == {}


def test_explicit_graph_rules_finalize_once_after_empty_checkpoints(monkeypatch, tmp_path: Path):
    calls: list[dict] = []

    def _writer(*args, **kwargs):
        calls.append(dict(kwargs))
        (Path(args[0]) / "graph_metric_by_rule.csv").write_text("metric\n")
        return {"graph_metric_by_rule": "graph_metric_by_rule.csv"}

    result, states = _finish_from_resumed_box(
        monkeypatch,
        tmp_path,
        graph_rules=[{"name": "requested", "kind": "hard_cutoff", "parameters": {"cutoff": 1.5}}],
        graph_writer=_writer,
    )

    assert states
    assert all(state["graph_outputs"] == {} for state in states)
    assert all(state["paths"] == {} for state in states)
    assert len(calls) == 1
    assert calls[0]["metrics"].graph_rules[0]["name"] == "requested"
    assert result["graph_outputs"] == {"graph_metric_by_rule": "graph_metric_by_rule.csv"}
    assert result["paths"] == result["graph_outputs"]


def test_resume_does_not_double_count_unchanged_convergence_check(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "vitriflow.workflows.autotune.check_production_convergence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged ensemble must not be re-evaluated")
        ),
    )

    result, states = _finish_from_resumed_box(
        monkeypatch,
        tmp_path,
        graph_rules=[],
        graph_writer=lambda *args, **kwargs: {},
        check_convergence=True,
        required_streak=2,
        resume_overrides={
            "converged_md": True,
            "convergence_md": {"density": {"converged": True}},
            "convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": 1,
            "last_convergence_evaluated_n_boxes_accepted": 1,
        },
    )

    assert result["status"] == "not_converged"
    assert result["converged"] is False
    assert result["convergence_streak"] == 1
    assert states[-1]["convergence_streak"] == 1


def test_autotune_terminal_state_surfaces_inference_qualified_degree(monkeypatch, tmp_path: Path):
    convergence_report = {
        "inference_contract": {"sequentially_valid": False},
        "achieved_convergence_degree": {
            "n_boxes": 1,
            "overall_active": {"worst_tolerance_utilization_ratio": 0.4},
        },
        "convergence_degree": {
            "ci": {"n_checked": 1, "n_passed": 1, "pass_fraction": 1.0},
            "overall": {"n_checked": 1, "n_passed": 1, "pass_fraction": 1.0},
        },
    }
    result, _states = _finish_from_resumed_box(
        monkeypatch,
        tmp_path,
        graph_rules=[],
        graph_writer=lambda *args, **kwargs: {},
        check_convergence=True,
        required_streak=1,
        resume_overrides={
            "converged_md": True,
            "convergence_md": convergence_report,
            "convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": 1,
            "last_convergence_evaluated_n_boxes_accepted": 1,
        },
    )

    assert result["convergence_status"] == "converged"
    assert result["convergence_inference_status"] == (
        "criterion_met_repeated_looks_not_sequentially_valid"
    )
    assert result["achieved_convergence_degree"] == convergence_report[
        "achieved_convergence_degree"
    ]
    assert result["convergence_criterion_coverage"]["overall"]["pass_fraction"] == 1.0


def test_terminal_adaptive_state_without_distributions_is_explicitly_non_resumable(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "vitriflow.workflows.autotune.check_production_convergence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("already evaluated ensemble must not be re-evaluated")
        ),
    )

    result, states = _finish_from_resumed_box(
        monkeypatch,
        tmp_path,
        graph_rules=[],
        graph_writer=lambda *args, **kwargs: {},
        check_convergence=True,
        required_streak=2,
        store_distributions=False,
        resume_overrides={
            "converged_md": True,
            "convergence_md": {"density": {"converged": True}},
            "convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": 1,
            "last_convergence_evaluated_n_boxes_accepted": 1,
        },
    )

    # Running checkpoints retain the full convergence evidence even when the
    # requested public terminal report is compact.
    assert all("distributions" in state["boxes"][0] for state in states)
    assert result["status"] == "not_converged"
    assert result["resumable"] is False
    assert "distributions" not in result["boxes"][0]
    assert "omitted" in result["non_resumable_reason"]
