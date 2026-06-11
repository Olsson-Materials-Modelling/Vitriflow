from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple



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
    if role == "quench" and (
        quench_window_steps_range is not None or (sampling_hint is not None and len(dict(sampling_hint)) > 0)
    ):
        max_frames_effective = max(
            int(max_frames_requested),
            int(quench_tail_min_frames) + 4,
            2 * int(max_frames_requested),
        )

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

    mts = compute_metrics_timeseries(
        stage_dir=stage_dir,
        metrics=metrics_cfg,
        cutoffs=cutoffs,
        md_timestep=float(md_timestep),
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
    )

    csv_path = stage_dir / "metrics_timeseries.csv"
    mts.to_csv(csv_path)

    manifest = {
        "status": "ok",
        "n_rows": int(mts.data.shape[0]),
        "n_columns": int(len(mts.columns)),
        "columns": list(mts.columns),
        "frame_stride": int(frame_stride),
        "max_frames_requested": int(max_frames_requested),
        "max_frames_effective": int(max_frames_effective),
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
        except Exception:
            plot_path = None

    base = Path(outdir) if outdir is not None else stage_dir.parent
    return {
        "status": "ok",
        "csv": _relpath_or_str(csv_path, base),
        "summary": _relpath_or_str(manifest_path, base),
        "plot": _relpath_or_str(plot_path, base) if plot_path is not None and plot_path.exists() else None,
        "n_rows": int(mts.data.shape[0]),
        "n_columns": int(len(mts.columns)),
    }
