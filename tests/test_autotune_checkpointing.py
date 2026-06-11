from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")

from vitriflow.workflows.autotune import _run_production_ensemble
from vitriflow.workflows.progress import CondensedProgressLog


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
            convergence=SimpleNamespace(),
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
