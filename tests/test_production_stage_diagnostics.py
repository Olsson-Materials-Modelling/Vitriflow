from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("ase")

from vitriflow.workflows.production_common import plan_production_stage_diagnostics, resolve_production_relax_dump_settings


def test_plan_production_stage_diagnostics_uses_shared_defaults(monkeypatch):
    monkeypatch.setattr(
        "vitriflow.workflows.production_common.should_collect_stage_metrics_timeseries",
        lambda metrics_cfg: True,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.production_common.should_collect_elastic_stage_timeseries",
        lambda metrics_cfg, *, runner, stage_role, force_isotropic: (stage_role == "quench", False, None),
    )
    monkeypatch.setattr(
        "vitriflow.workflows.production_common.recommended_quench_dump_every",
        lambda **kwargs: 123,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.production_common.quench_window_steps",
        lambda **kwargs: (11.0, 22.0),
    )

    prod_cfg = SimpleNamespace(dump_trajectory=False, dump_every_steps=500)
    metrics_cfg = SimpleNamespace(quench_tail_min_frames=24, elastic=SimpleNamespace(quench_tail_min_frames=12))

    plan = plan_production_stage_diagnostics(
        prod_cfg=prod_cfg,
        metrics_cfg=metrics_cfg,
        runner=object(),
        force_isotropic=False,
        total_quench_steps=1000,
        temperature_start=1200.0,
        temperature_stop=300.0,
        sampling_hint={"Tm": 900.0, "freeze_temperature": 500.0},
    )

    assert plan["dump_traj"] is False
    assert plan["dump_every"] == 500
    assert plan["collect_stage_metric_series"] is True
    assert plan["collect_elastic_series"]["quench"] is True
    assert plan["need_stage_dump"] == {"melt": True, "quench": True, "relax": True}
    assert plan["quench_min_window_frames"] == 24
    assert plan["quench_dump_every"] == 123
    assert plan["quench_window_steps_range"] == (11.0, 22.0)


def test_resolve_production_relax_dump_settings_uses_tail_only_dump_when_metrics_need_frames():
    stage_diag = {
        "need_stage_dump": {"melt": False, "quench": False, "relax": False},
        "dump_every": 500,
    }
    metrics_cfg = SimpleNamespace(enabled=True, time_average_frames=4, time_average_stride=250)

    policy = resolve_production_relax_dump_settings(stage_diag=stage_diag, metrics_cfg=metrics_cfg)

    assert policy == {
        "write_dump": True,
        "dump_every": None,
        "tail_dump_frames": 4,
        "tail_dump_stride": 250,
        "mode": "tail_only",
    }


def test_resolve_production_relax_dump_settings_reuses_full_stage_dump_when_already_required():
    stage_diag = {
        "need_stage_dump": {"melt": False, "quench": False, "relax": True},
        "dump_every": 400,
    }
    metrics_cfg = SimpleNamespace(enabled=True, time_average_frames=4, time_average_stride=250)

    policy = resolve_production_relax_dump_settings(stage_diag=stage_diag, metrics_cfg=metrics_cfg)

    assert policy == {
        "write_dump": True,
        "dump_every": 400,
        "tail_dump_frames": None,
        "tail_dump_stride": None,
        "mode": "full",
    }
