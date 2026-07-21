from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _FakeMetricsTimeSeries:
    columns = ["Step", "time", "density"]
    data = np.asarray([[0.0, 0.0, 2.2]], dtype=float)
    metadata = {"selection": "test"}

    def to_csv(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("Step,time,density\n0,0,2.2\n")


def _stage_metrics_cfg(*, make_plot: bool = True):
    return SimpleNamespace(
        enabled=True,
        stage_timeseries_frame_stride=1,
        stage_timeseries_max_frames=1,
        stage_timeseries_make_plot=bool(make_plot),
        quench_tail_focus_fraction=0.67,
        quench_tail_min_frames=1,
        quench_tail_fallback_fraction=0.40,
    )


def test_requested_stage_metrics_plot_failure_is_terminal(monkeypatch, tmp_path: Path):
    from vitriflow.analysis import timeseries
    from vitriflow import plotting
    from vitriflow.workflows.stage_metrics import collect_stage_metrics_timeseries

    monkeypatch.setattr(
        timeseries,
        "compute_metrics_timeseries",
        lambda **kwargs: _FakeMetricsTimeSeries(),
    )
    monkeypatch.setattr(
        plotting,
        "plot_metrics_timeseries",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("renderer failed")),
    )

    with pytest.raises(RuntimeError, match="Requested stage metrics plot failed"):
        collect_stage_metrics_timeseries(
            stage_dir=tmp_path / "relax",
            metrics_cfg=_stage_metrics_cfg(),
            cutoffs={},
            md_timestep=0.001,
            outdir=tmp_path,
            stage_role="relax",
        )

    manifest = json.loads(
        (tmp_path / "relax" / "metrics_timeseries.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["plot_status"] == "failed"
    assert "renderer failed" in manifest["plot_error"]


def test_requested_stage_metrics_plot_must_be_nonempty(monkeypatch, tmp_path: Path):
    from vitriflow.analysis import timeseries
    from vitriflow import plotting
    from vitriflow.workflows.stage_metrics import collect_stage_metrics_timeseries

    monkeypatch.setattr(
        timeseries,
        "compute_metrics_timeseries",
        lambda **kwargs: _FakeMetricsTimeSeries(),
    )
    monkeypatch.setattr(plotting, "plot_metrics_timeseries", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="non-empty PDF"):
        collect_stage_metrics_timeseries(
            stage_dir=tmp_path / "melt",
            metrics_cfg=_stage_metrics_cfg(),
            cutoffs={},
            md_timestep=0.001,
            outdir=tmp_path,
            stage_role="melt",
        )
