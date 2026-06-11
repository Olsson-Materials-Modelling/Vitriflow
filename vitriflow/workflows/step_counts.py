from __future__ import annotations

"""Utilities for scaling step counts across timestep changes.

LAMMPS input scripts are expressed in *steps*; when the timestep is modified (e.g.
by preflight), step counts should be rescaled to preserve physical time.

This module centralizes the reference-timestep logic for recommendations vs.
configuration defaults.
"""

import math
from typing import Any, Mapping, Optional

from ..utils import scale_steps_for_timestep


def recommendation_md_timestep(rec: Mapping[str, Any], *, fallback: float) -> float:
    """Recommendation md timestep."""

    # preferred format autotune
    md = rec.get("md") if isinstance(rec, Mapping) else None
    if isinstance(md, Mapping):
        try:
            dt = float(md.get("timestep"))
            if math.isfinite(dt) and dt > 0.0:
                return dt
        except Exception:
            pass

    # compatibility keys
    for k in ("md_timestep", "timestep"):
        if k in rec:
            try:
                dt = float(rec.get(k))
                if math.isfinite(dt) and dt > 0.0:
                    return dt
            except Exception:
                pass

    return float(fallback)


def scale_recommended_highT_steps(
    high_steps_raw: int,
    rec: Mapping[str, Any],
    *,
    dt_cfg: float,
    dt_use: float,
) -> int:
    """Scale recommended high."""

    dt_rec = recommendation_md_timestep(rec, fallback=float(dt_cfg))
    dt_ref = float(dt_rec) if "highT_steps" in rec else float(dt_cfg)
    return int(scale_steps_for_timestep(int(high_steps_raw), dt_ref, float(dt_use), min_steps=1))


def scale_config_steps(steps_cfg: int, *, dt_cfg: float, dt_use: float, min_steps: int = 1) -> int:
    """Scale config steps."""

    return int(scale_steps_for_timestep(int(steps_cfg), float(dt_cfg), float(dt_use), min_steps=min_steps))


def extend_highT_steps_for_force_isotropic(
    steps: int,
    *,
    force_isotropic: bool,
    factor: float = 1.5,
) -> int:
    """Extend high t."""

    s = int(steps)
    if s < 1:
        return 1
    if not bool(force_isotropic):
        return s
    fac = float(factor)
    if not (math.isfinite(fac) and fac >= 1.0):
        fac = 1.5
    return max(1, int(math.ceil(float(s) * fac)))



def resolve_lammps_units_style(
    config: Any | None = None,
    *,
    pot_cfg: Any | None = None,
    default: str = "metal",
) -> str:
    """Lammps units style."""

    candidates = [
        pot_cfg,
        getattr(config, "kim", None) if config is not None else None,
        getattr(config, "potential", None) if config is not None else None,
    ]
    for obj in candidates:
        try:
            units = str(getattr(obj, "user_units", "") or "").strip().lower()
        except Exception:
            units = ""
        if units:
            return units
    return str(default or "metal").strip().lower() or "metal"



def resolve_md_pressure(
    config: Any | None = None,
    *,
    md_use: Any | None = None,
    override: Any | None = None,
    default: float = 0.0,
) -> float:
    """Md pressure."""

    def _coerce(obj: Any | None) -> float | None:
        if obj is None:
            return None
        try:
            val = float(obj)
        except Exception:
            return None
        if math.isfinite(val):
            return float(val)
        return None

    val = _coerce(override)
    if val is not None:
        return val

    for obj in (md_use, getattr(config, "md", None) if config is not None else None):
        try:
            if isinstance(obj, Mapping):
                val = _coerce(obj.get("pressure", None))
            else:
                val = _coerce(getattr(obj, "pressure", None))
        except Exception:
            val = None
        if val is not None:
            return val

    return float(default)


def recommended_quench_dump_every(
    *,
    total_steps: int,
    temperature_start: float,
    temperature_stop: float,
    base_dump_every: int,
    sampling_hint: Optional[Mapping[str, float]],
    min_window_frames: int = 12,
) -> int:
    """Recommended quench dump."""

    from ..analysis.trajectory import quench_window_steps

    base = max(int(base_dump_every), 1)
    if sampling_hint is None:
        return base
    window = quench_window_steps(
        T_start=float(temperature_start),
        T_stop=float(temperature_stop),
        total_steps=int(total_steps),
        T_upper=sampling_hint.get("Tm"),
        T_lower=sampling_hint.get("freeze_temperature"),
    )
    if window is None:
        return base
    dense_steps = max(float(window[1] - window[0]), 0.0)
    n_target = max(int(min_window_frames), 2)
    stride_target = max(1, int(math.floor(dense_steps / max(n_target - 1, 1))))
    return int(max(1, min(base, stride_target)))
