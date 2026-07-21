from __future__ import annotations

"""Shared utilities for quench-rate handling.

This module exists to avoid importing implementation details from the autotune
workflow in other workflows.
"""

import math
from typing import Optional

from ..config import RunConfig


def lammps_timeunit_ps(units_style: str) -> float | None:
    """Lammps timeunit ps."""

    u = str(units_style).strip().lower()
    if u == "metal":
        return 1.0
    if u == "real":
        return 0.001
    if u == "nano":
        return 1.0e3
    if u == "micro":
        return 1.0e6
    if u in ("si", "cgs"):
        return 1.0e12
    if u == "electron":
        # lammps electron conversion
        return 0.001
    if u == "lj":
        return None
    return None


def geomspace_desc(high: float, low: float, n: int) -> list[float]:
    """Geomspace desc."""

    if n < 2:
        raise ValueError("n must be >= 2")
    high = float(high)
    low = float(low)
    if high <= 0 or low <= 0:
        raise ValueError("high and low must be > 0")
    if high < low:
        high, low = low, high
    if n == 2:
        return [high, low]
    ratio = (low / high) ** (1.0 / float(n - 1))
    return [high * (ratio**i) for i in range(n)]


def resolve_quench_rates_K_per_time(config: RunConfig) -> tuple[list[float], float | None, list[float] | None]:
    """Quench rates k."""

    q = config.autotune.quench

    # classical depends lammps
    # interprets timestep femtoseconds
    if getattr(config, "engine", "lammps") == "cp2k":
        time_unit_ps = 0.001
    else:
        if config.kim is None:
            raise ValueError("resolve_quench_rates_K_per_time requires config.kim when engine='lammps'")
        time_unit_ps = lammps_timeunit_ps(config.kim.user_units)

    # expert direct time
    if q.rates_K_per_time is not None:
        rates_time = [float(r) for r in q.rates_K_per_time]
        rates_time = sorted(rates_time, reverse=True)
        if time_unit_ps is not None:
            rates_ps = [float(r) / float(time_unit_ps) for r in rates_time]
        else:
            rates_ps = None
        return rates_time, time_unit_ps, rates_ps

    # facing k ps
    if time_unit_ps is None:
        raise ValueError(
            f"Cannot interpret quench rates in K/ps for LAMMPS units style '{config.kim.user_units}'. "
            "Please specify autotune.quench.rates_K_per_time (K per LAMMPS time unit)."
        )

    if q.rates_K_per_ps is not None:
        rates_ps = [float(r) for r in q.rates_K_per_ps]
    else:
        rates_ps = geomspace_desc(float(q.rate_max_K_per_ps), float(q.rate_min_K_per_ps), int(q.n_rates))

    rates_ps = sorted(rates_ps, reverse=True)
    rates_time = [float(r) * float(time_unit_ps) for r in rates_ps]
    return rates_time, time_unit_ps, rates_ps


def convert_rate_Kps_to_Ktime(rate_K_per_ps: float, *, time_unit_ps: float | None) -> float:
    """Rate kps to."""

    if time_unit_ps is None or not (math.isfinite(time_unit_ps) and time_unit_ps > 0.0):
        raise ValueError("time_unit_ps must be finite and > 0 to convert K/ps to K/time")
    return float(rate_K_per_ps) * float(time_unit_ps)

def quench_steps_for_rate(
    delta_T: float,
    rate_K_per_time: float,
    timestep: float,
    *,
    min_steps: int = 1,
) -> int:
    """Quench steps for."""

    dT = float(delta_T)
    r = float(rate_K_per_time)
    dt = float(timestep)

    if not (math.isfinite(r) and r > 0.0):
        raise ValueError('rate_K_per_time must be finite and > 0')
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError('timestep must be finite and > 0')

    if not (math.isfinite(dT) and dT > 0.0):
        raise ValueError(
            'delta_T must be finite and > 0 (quench start temperature must exceed final temperature)'
        )

    n = int(math.ceil(dT / (r * dt)))
    return int(max(int(min_steps), n))
