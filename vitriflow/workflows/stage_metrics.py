from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..lammps_units import time_to_ps_factor



def _relpath_or_str(path: Path, base: Path) -> str:
    p = Path(path)
    b = Path(base)
    try:
        return str(p.relative_to(b))
    except Exception:
        return str(p)


def should_collect_stage_metrics_timeseries(metrics_cfg) -> bool:
    """Collect stage metrics."""
    try:
        return bool(getattr(metrics_cfg, "enabled", False)) and bool(
            getattr(metrics_cfg, "collect_during_production_stages", True)
        )
    except Exception:
        return False


def collect_stage_metrics_timeseries(
    *,
    stage_dir: Path,
    metrics_cfg,
    cutoffs: Mapping[Tuple[int, int], float],
    md_timestep: float,
    type_to_species: Optional[Sequence[str]] = None,
    outdir: Optional[Path] = None,
    stage_role: Optional[str] = None,
    quench_window_steps_range: Optional[Tuple[float, float]] = None,
    sampling_hint: Optional[Mapping[str, float]] = None,
    lammps_units_style: Optional[str] = "metal",
    engine: str = "lammps",
) -> dict[str, Any]:
    """Stage metrics timeseries."""

    stage_dir = Path(stage_dir)
    frame_stride = int(getattr(metrics_cfg, "stage_timeseries_frame_stride", 1) or 1)
    max_frames_requested = int(getattr(metrics_cfg, "stage_timeseries_max_frames", 64) or 64)
    make_plot = bool(getattr(metrics_cfg, "stage_timeseries_make_plot", False))
    quench_tail_fraction = float(getattr(metrics_cfg, "quench_tail_focus_fraction", 0.67) or 0.67)
    quench_tail_min_frames = int(getattr(metrics_cfg, "quench_tail_min_frames", 8) or 8)
    quench_tail_fallback_fraction = float(getattr(metrics_cfg, "quench_tail_fallback_fraction", 0.40) or 0.40)

    role = str(stage_role or "").strip().lower()
    max_frames_effective = int(max_frames_requested)

    from ..analysis.timeseries import compute_metrics_timeseries

    tm_temperature = None
    freeze_temperature = None
    if sampling_hint is not None:
        try:
            tm_temperature = sampling_hint.get("Tm", sampling_hint.get("tm_temperature"))
        except Exception:
            tm_temperature = None
        try:
            freeze_temperature = sampling_hint.get(
                "freeze_temperature",
                sampling_hint.get("diffusion_freeze_temperature"),
            )
        except Exception:
            freeze_temperature = None

    engine_name = str(engine or "lammps").strip().lower()
    timestep_ps = (
        float(md_timestep) * 1.0e-3
        if engine_name == "cp2k"
        else float(md_timestep) * float(time_to_ps_factor(lammps_units_style))
    )
    mts = compute_metrics_timeseries(
        stage_dir=stage_dir,
        metrics=metrics_cfg,
        cutoffs=cutoffs,
        md_timestep=timestep_ps,
        type_to_species=type_to_species,
        frame_stride=int(frame_stride),
        max_frames=int(max_frames_effective),
        include_gr_curves=False,
        include_coord_defects=True,
        stage_role=role or None,
        quench_window_steps_range=quench_window_steps_range,
        tm_temperature=(float(tm_temperature) if tm_temperature is not None else None),
        diffusion_freeze_temperature=(float(freeze_temperature) if freeze_temperature is not None else None),
        quench_tail_fraction=float(quench_tail_fraction),
        quench_tail_min_frames=int(quench_tail_min_frames),
        quench_tail_fallback_fraction=float(quench_tail_fallback_fraction),
        trajectory_lammps_units_style=lammps_units_style,
    )

    csv_path = stage_dir / "metrics_timeseries.csv"
    mts.to_csv(csv_path)

    metric_columns = [
        str(name) for name in mts.columns if str(name) not in {"Step", "time"}
    ]
    column_index = {str(name): idx for idx, name in enumerate(mts.columns)}
    metric_data = np.asarray(mts.data, dtype=float)
    plot_coordinate = (
        metric_data[:, column_index["time"]]
        if "time" in column_index
        else np.full(metric_data.shape[0], np.nan, dtype=float)
    )
    unavailable_metric_columns = [
        name
        for name in metric_columns
        if not np.any(
            np.isfinite(plot_coordinate)
            & np.isfinite(metric_data[:, column_index[name]])
        )
    ]
    plot_warning = (
        f"{len(unavailable_metric_columns)} metric "
        f"column{'s' if len(unavailable_metric_columns) != 1 else ''} had no finite "
        "selected-frame values and "
        f"{'are' if len(unavailable_metric_columns) != 1 else 'is'} annotated "
        "as unavailable in the plot"
        if make_plot and unavailable_metric_columns
        else None
    )

    manifest = {
        "status": "ok",
        "plot_status": "pending" if make_plot else "not_requested",
        "engine": engine_name,
        "reporting_contract": "vitriflow.canonical_physical_units.v1",
        "time_unit": "ps",
        "n_rows": int(mts.data.shape[0]),
        "n_columns": int(len(mts.columns)),
        "columns": list(mts.columns),
        "unavailable_metric_columns": list(unavailable_metric_columns),
        "plot_warning": plot_warning,
        "frame_stride": int(frame_stride),
        "max_frames_requested": int(max_frames_requested),
        "max_frames_effective": int(max_frames_effective),
        "max_frames_hard_cap": True,
        "stage_role": role or None,
        "quench_window_steps": [float(quench_window_steps_range[0]), float(quench_window_steps_range[1])] if quench_window_steps_range is not None else None,
        "sampling_hint": dict(sampling_hint or {}),
        "tm_temperature": float(tm_temperature) if tm_temperature is not None else None,
        "diffusion_freeze_temperature": float(freeze_temperature) if freeze_temperature is not None else None,
        "quench_tail_fraction": float(quench_tail_fraction),
        "quench_tail_min_frames": int(quench_tail_min_frames),
        "quench_tail_fallback_fraction": float(quench_tail_fallback_fraction),
        "selection": dict(getattr(mts, "metadata", {}) or {}),
    }
    manifest_path = stage_dir / "metrics_timeseries.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    plot_path = None
    if make_plot:
        try:
            from ..plotting import plot_metrics_timeseries

            plot_path = stage_dir / "metrics_timeseries.pdf"
            plot_metrics_timeseries(
                csv_path,
                plot_path,
                xaxis="time",
                title=f"Stage metrics: {stage_dir.name}",
            )
            if not plot_path.is_file() or int(plot_path.stat().st_size) < 1:
                raise RuntimeError(
                    "plot_metrics_timeseries returned without producing a non-empty PDF"
                )
        except Exception as exc:
            # A requested plot is part of the public production-artifact
            # contract.  Silently returning status='ok' with plot=None masks
            # plotting regressions and makes local/external validation appear
            # complete when it is not.
            manifest["status"] = "failed"
            manifest["plot_status"] = "failed"
            manifest["plot_error"] = str(exc)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            raise RuntimeError(
                "Requested stage metrics plot failed for "
                f"stage_role={role or 'unspecified'} directory={stage_dir}: {exc}"
            ) from exc
        else:
            manifest["plot_status"] = "ok"
            manifest["plot"] = str(plot_path.name)
            manifest_path.write_text(json.dumps(manifest, indent=2))

    base = Path(outdir) if outdir is not None else stage_dir.parent
    return {
        "status": "ok",
        "engine": engine_name,
        "reporting_contract": "vitriflow.canonical_physical_units.v1",
        "time_unit": "ps",
        "csv": _relpath_or_str(csv_path, base),
        "summary": _relpath_or_str(manifest_path, base),
        "plot": _relpath_or_str(plot_path, base) if plot_path is not None and plot_path.exists() else None,
        "n_rows": int(mts.data.shape[0]),
        "n_columns": int(len(mts.columns)),
        "unavailable_metric_columns": list(unavailable_metric_columns),
        "plot_warning": plot_warning,
    }
