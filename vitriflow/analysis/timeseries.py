from __future__ import annotations

"""Time-resolved structural metric evaluation."""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import StructureMetricsConfig
from ..io.thermo import parse_thermo_csv
from .trajectory import select_stage_frames, stage_trajectory_path, read_frames_auto

if TYPE_CHECKING:  # pragma: no cover
    from .structure import StructureMetrics


@dataclass(frozen=True)
class MetricsTimeSeries:
    """Metrics time series."""

    columns: list[str]
    data: np.ndarray  # shape n m
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_csv(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(self.columns))
            for row in self.data.tolist():
                w.writerow([float(x) if x is not None else "" for x in row])


def _interp_thermo(table_cols: Sequence[str], table_data: np.ndarray, steps: np.ndarray) -> Dict[str, np.ndarray]:
    cols = list(table_cols)
    data = np.asarray(table_data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("thermo table must be 2D with >=2 rows")
    if "Step" not in cols:
        raise ValueError("thermo table missing required column 'Step'")
    step_col = cols.index("Step")
    x = np.asarray(data[:, step_col], dtype=float)
    if not np.isfinite(x).any():
        raise ValueError("thermo Step column is non-finite")
    order = np.argsort(x)
    x = x[order]
    out: Dict[str, np.ndarray] = {}
    for j, nm in enumerate(cols):
        if j == step_col:
            continue
        y = np.asarray(data[:, j], dtype=float)[order]
        m = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(m)) < 2:
            out[nm] = np.full_like(steps, np.nan, dtype=float)
            continue
        out[nm] = np.interp(steps, x[m], y[m], left=np.nan, right=np.nan)
    return out


def compute_metrics_timeseries(
    *,
    stage_dir: Path,
    metrics: StructureMetricsConfig,
    cutoffs: Mapping[Tuple[int, int], float],
    md_timestep: float,
    type_to_species: Optional[Sequence[str]] = None,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    include_gr_curves: bool = False,
    include_coord_defects: bool = True,
    stage_role: Optional[str] = None,
    quench_window_steps_range: Optional[Tuple[float, float]] = None,
    tm_temperature: Optional[float] = None,
    diffusion_freeze_temperature: Optional[float] = None,
    quench_tail_fraction: float = 0.67,
    quench_tail_min_frames: int = 8,
    quench_tail_fallback_fraction: float = 0.40,
    trajectory_lammps_units_style: Optional[str] = "metal",
) -> MetricsTimeSeries:
    """Metrics timeseries with ``md_timestep`` expressed in canonical ps."""

    if int(frame_stride) < 1:
        raise ValueError("frame_stride must be >= 1")
    frame_stride = int(frame_stride)
    if not (np.isfinite(float(md_timestep)) and float(md_timestep) > 0.0):
        raise ValueError("md_timestep must be > 0")

    stage_dir = Path(stage_dir)
    traj = stage_trajectory_path(stage_dir)
    if traj is None or not Path(traj).exists():
        raise FileNotFoundError(f"No trajectory found in stage directory: {stage_dir}")

    frames_all = list(
        read_frames_auto(Path(traj), units_style=trajectory_lammps_units_style)
    )
    if not frames_all:
        raise ValueError(f"No frames parsed from trajectory: {traj}")

    thermo_path = stage_dir / "thermo.csv"
    thermo_table = None
    thermo_all: Dict[str, np.ndarray] = {}
    steps_all = np.asarray([float(fr.timestep) for fr in frames_all], dtype=float)
    if thermo_path.exists():
        thermo_table = parse_thermo_csv(thermo_path)
        thermo_all = _interp_thermo(thermo_table.columns, thermo_table.data, steps_all)
    temps_strided = None
    if "Temp" in thermo_all:
        temps_strided = np.asarray(thermo_all["Temp"], dtype=float)[::frame_stride]

    frames, selection_meta = select_stage_frames(
        frames_all,
        frame_stride=int(frame_stride),
        max_frames=max_frames,
        stage_role=stage_role,
        quench_window_steps_range=quench_window_steps_range,
        temperatures=temps_strided,
        tm_temperature=tm_temperature,
        diffusion_freeze_temperature=diffusion_freeze_temperature,
        quench_tail_fraction=float(quench_tail_fraction),
        quench_tail_min_frames=int(quench_tail_min_frames),
        quench_tail_fallback_fraction=float(quench_tail_fallback_fraction),
    )
    if not frames:
        raise ValueError(f"No frames selected from trajectory: {traj}")

    steps = np.asarray([float(fr.timestep) for fr in frames], dtype=float)
    time = steps * float(md_timestep)

    thermo_vals: Dict[str, np.ndarray] = {}
    if thermo_table is not None:
        thermo_vals = _interp_thermo(thermo_table.columns, thermo_table.data, steps)

    from .structure import StructureMetrics as _StructureMetrics, compute_coordination_defects, compute_structure_metrics

    rows: list[dict[str, Any]] = []
    for i_row, (fr, step, t) in enumerate(zip(frames, steps.tolist(), time.tolist())):
        vals_obj = compute_structure_metrics(fr, metrics, cutoffs=cutoffs, type_to_species=type_to_species)
        if isinstance(vals_obj, _StructureMetrics):
            vals: Mapping[str, Any] = vals_obj.values
        elif isinstance(vals_obj, Mapping):
            vals = vals_obj
        else:
            raise TypeError(f"compute_structure_metrics returned unsupported type: {type(vals_obj)}")

        out_row: dict[str, Any] = {"Step": float(step), "time": float(t)}
        for k, v in thermo_vals.items():
            out_row[f"thermo_{k}"] = float(v[i_row]) if i_row < len(v) else float("nan")
        for k, v in vals.items():
            out_row[str(k)] = float(v) if v is not None else float("nan")

        if getattr(metrics, "gr", None):
            from .gr import compute_first_peak_gr

            def _slug(s: str) -> str:
                return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

            for gm in list(metrics.gr):
                pair = getattr(gm, "pair", None)
                label = "all" if pair is None else f"{pair[0]}-{pair[1]}"
                feat = compute_first_peak_gr(
                    [fr],
                    r_max=float(getattr(gm, "r_max", 8.0)),
                    nbins=int(getattr(gm, "nbins", 400)),
                    smooth=int(getattr(gm, "smooth", 7)),
                    pair=pair,
                    type_to_species=type_to_species,
                )
                pref = f"gr_{_slug(label)}"
                out_row[f"{pref}_peak_r"] = float(feat.peak_r)
                out_row[f"{pref}_peak_height"] = float(feat.peak_height)
                out_row[f"{pref}_peak_fwhm"] = float(feat.peak_fwhm)

        if getattr(metrics, "sq", None):
            from .sq import compute_first_peak_sq

            def _slug(s: str) -> str:
                return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

            for sm in list(metrics.sq):
                pair = getattr(sm, "pair", None)
                label = "all" if pair is None else f"{pair[0]}-{pair[1]}"
                peak_search = tuple(getattr(sm, "peak_search", (0.5, 3.0)))
                feat = compute_first_peak_sq(
                    [fr],
                    q_max=float(getattr(sm, "q_max", 20.0)),
                    nq=int(getattr(sm, "nq", 400)),
                    r_max=float(getattr(sm, "r_max", 10.0)),
                    nbins=int(getattr(sm, "nbins", 800)),
                    smooth=int(getattr(sm, "smooth", 7)),
                    peak_search=(float(peak_search[0]), float(peak_search[1])),
                    pair=pair,
                    type_to_species=type_to_species,
                    window=str(getattr(sm, "window", "lorch")),
                )
                pref = f"sq_{_slug(label)}"
                out_row[f"{pref}_peak_q"] = float(feat.peak_q)
                out_row[f"{pref}_peak_height"] = float(feat.peak_height)
                out_row[f"{pref}_peak_fwhm"] = float(feat.peak_fwhm)

        void_cfg = getattr(metrics, "voids", None)
        if void_cfg is not None and bool(getattr(void_cfg, "enabled", False)):
            n_vs = int(getattr(void_cfg, "n_samples_timeseries", 0) or 0)
            if n_vs > 0:
                from .voids import sample_void_clearance_radii, clearance_scalar_metrics

                radii = sample_void_clearance_radii(
                    [fr],
                    n_samples=n_vs,
                    sampler=str(getattr(void_cfg, "sampler", "sobol")),
                    seed=int(getattr(void_cfg, "seed", 0) or 0),
                    k_nearest=int(getattr(void_cfg, "k_nearest", 16) or 16),
                    type_to_species=type_to_species,
                    radii_by_species=dict(getattr(void_cfg, "radii", {}) or {}),
                    default_radius=float(getattr(void_cfg, "default_radius", 0.0) or 0.0),
                )
                sm_void = clearance_scalar_metrics(radii, probe_radii=list(getattr(void_cfg, "probe_radii", []) or []))

                def _slug_float(x: float) -> str:
                    return f"{float(x):.6g}".replace("-", "m").replace(".", "p")

                out_row["void_clearance_mean"] = float(sm_void.get("mean", float("nan")))
                out_row["void_clearance_median"] = float(sm_void.get("median", float("nan")))
                out_row["void_clearance_p95"] = float(sm_void.get("p95", float("nan")))
                out_row["void_clearance_max"] = float(sm_void.get("max", float("nan")))
                out_row["void_clearance_n_samples"] = float(len(radii))
                for rp in list(getattr(void_cfg, "probe_radii", []) or []):
                    key = f"void_clearance_frac_ge_r{_slug_float(float(rp))}"
                    out_row[key] = float(sm_void.get(f"frac_ge_{float(rp)}", float("nan")))

        if include_coord_defects:
            defects = compute_coordination_defects(fr, metrics, cutoffs=cutoffs, type_to_species=type_to_species)
            for nm, d in (defects or {}).items():
                if not isinstance(d, dict):
                    continue
                frac = d.get("defect_fraction", None)
                out_row[f"{nm}_defect_fraction"] = float(frac) if frac is not None else float("nan")

        if include_gr_curves:
            from .gr import compute_gr
            for gm in list(metrics.gr):
                label = "all" if gm.pair is None else f"{gm.pair[0]}-{gm.pair[1]}"
                key = f"gr_curve_{label}"
                _r, g, _ = compute_gr(
                    [fr],
                    r_max=float(gm.r_max),
                    nbins=int(gm.nbins),
                    pair=gm.pair,
                    type_to_species=type_to_species,
                )
                out_row[f"{key}_g_mean"] = float(np.nanmean(g))
                out_row[f"{key}_g_max"] = float(np.nanmax(g))

        rows.append(out_row)

    base_cols = ["Step", "time"]
    thermo_cols = sorted([c for c in rows[0].keys() if c.startswith("thermo_")])
    metric_cols = sorted([c for c in rows[0].keys() if c not in set(base_cols + thermo_cols)])
    cols = base_cols + thermo_cols + metric_cols
    data = np.asarray([[float(r.get(c, float("nan"))) for c in cols] for r in rows], dtype=float)
    metadata = dict(selection_meta)
    metadata.update(
        {
            "reporting_contract": "vitriflow.canonical_physical_units.v1",
            "time_unit": "ps",
            "length_unit": "angstrom",
            "trajectory_lammps_units_style": str(trajectory_lammps_units_style),
        }
    )
    return MetricsTimeSeries(columns=cols, data=data, metadata=metadata)
