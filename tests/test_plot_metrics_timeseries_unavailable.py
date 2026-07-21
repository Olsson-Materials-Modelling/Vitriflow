from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write_csv(path: Path, payload: str) -> Path:
    path.write_text(payload)
    return path


def test_metrics_timeseries_plot_preserves_gaps_and_annotates_unavailable_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from matplotlib.axes import Axes

    from vitriflow.plotting import plot_metrics_timeseries

    csv_path = _write_csv(
        tmp_path / "metrics.csv",
        "Step,time,density,sq_Ga_Ga_peak_fwhm,partially_available\n"
        "0,0,5.1,nan,1.0\n"
        "1,1,5.0,nan,nan\n"
        "2,2,4.9,nan,3.0\n",
    )
    output = tmp_path / "metrics.pdf"

    annotations: list[str] = []
    plotted_values: list[np.ndarray] = []
    original_text = Axes.text
    original_plot = Axes.plot

    def capture_text(self, *args, **kwargs):
        if len(args) >= 3:
            annotations.append(str(args[2]))
        return original_text(self, *args, **kwargs)

    def capture_plot(self, *args, **kwargs):
        if len(args) >= 2:
            plotted_values.append(np.asarray(args[1], dtype=float))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture_text)
    monkeypatch.setattr(Axes, "plot", capture_plot)

    plot_metrics_timeseries(csv_path, output)

    assert output.is_file()
    assert output.stat().st_size > 0
    assert annotations.count(
        "Metric unavailable: no finite values in selected frames"
    ) == 1
    assert any(np.allclose(values, [5.1, 5.0, 4.9]) for values in plotted_values)
    assert any(
        values.shape == (3,)
        and values[0] == pytest.approx(1.0)
        and np.isnan(values[1])
        and values[2] == pytest.approx(3.0)
        for values in plotted_values
    )


class _UnavailableMetricSeries:
    columns = ["Step", "time", "sq_Ga_Ga_peak_fwhm"]
    data = np.asarray(
        [[0.0, 0.0, np.nan], [1.0, 0.001, np.nan]], dtype=float
    )
    metadata = {"selection": "test"}

    def to_csv(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            "Step,time,sq_Ga_Ga_peak_fwhm\n"
            "0,0,nan\n"
            "1,0.001,nan\n"
        )


def test_stage_metrics_all_unavailable_metric_is_audited_without_aborting_box(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vitriflow.analysis import timeseries
    from vitriflow.workflows.stage_metrics import collect_stage_metrics_timeseries

    monkeypatch.setattr(
        timeseries,
        "compute_metrics_timeseries",
        lambda **kwargs: _UnavailableMetricSeries(),
    )
    cfg = SimpleNamespace(
        enabled=True,
        stage_timeseries_frame_stride=1,
        stage_timeseries_max_frames=2,
        stage_timeseries_make_plot=True,
        quench_tail_focus_fraction=0.67,
        quench_tail_min_frames=1,
        quench_tail_fallback_fraction=0.40,
    )
    stage_dir = tmp_path / "melt"

    result = collect_stage_metrics_timeseries(
        stage_dir=stage_dir,
        metrics_cfg=cfg,
        cutoffs={},
        md_timestep=0.001,
        outdir=tmp_path,
        stage_role="melt",
    )

    manifest = json.loads((stage_dir / "metrics_timeseries.json").read_text())
    assert result["status"] == "ok"
    assert result["plot"] == "melt/metrics_timeseries.pdf"
    assert result["unavailable_metric_columns"] == ["sq_Ga_Ga_peak_fwhm"]
    assert result["plot_warning"] == (
        "1 metric column had no finite selected-frame values and is "
        "annotated as unavailable in the plot"
    )
    assert manifest["status"] == "ok"
    assert manifest["plot_status"] == "ok"
    assert manifest["unavailable_metric_columns"] == ["sq_Ga_Ga_peak_fwhm"]
    assert manifest["plot_warning"] == result["plot_warning"]
    assert (stage_dir / "metrics_timeseries.pdf").stat().st_size > 0


@pytest.mark.parametrize(
    ("payload", "metrics", "error"),
    [
        (
            "Step,density\n0,5.1\n",
            None,
            "Missing required column for xaxis='time'",
        ),
        (
            "Step,time,density\n0,nan,5.1\n",
            None,
            "x column 'time' contains no finite values",
        ),
        (
            "Step,time,density\n0,0,5.1\n",
            ["sq_Ga_Ga_peak_fwhm"],
            "Requested metric column\\(s\\) are absent",
        ),
        (
            "Step,time,density\nnot,numeric,data\n",
            None,
            "No numeric rows parsed",
        ),
    ],
)
def test_metrics_timeseries_plot_still_rejects_invalid_or_missing_evidence(
    tmp_path: Path,
    payload: str,
    metrics: list[str] | None,
    error: str,
) -> None:
    from vitriflow.plotting import plot_metrics_timeseries

    csv_path = _write_csv(tmp_path / "invalid.csv", payload)
    with pytest.raises(ValueError, match=error):
        plot_metrics_timeseries(
            csv_path,
            tmp_path / "invalid.pdf",
            metrics=metrics,
        )
