from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .stats import integrated_autocorr_time


@dataclass(frozen=True)
class DiffusionEstimate:
    D: float
    D_stderr: float
    slope: float
    slope_stderr: float
    intercept: float
    nfit: int
    tmin: float
    tmax: float
    # The unconstrained regression is retained for diagnosis.  Diffusion is a
    # nonnegative transport coefficient, so a finite negative slope caused by
    # noise in an immobile trajectory is reported at the physical boundary
    # D=0 rather than being passed downstream as negative diffusion.
    D_unconstrained: float = float("nan")
    boundary_constrained: bool = False


def _newey_west_slope_stderr(t: np.ndarray, resid: np.ndarray) -> float:
    """Newey west slope."""

    t = np.asarray(t, dtype=float)
    u = np.asarray(resid, dtype=float)
    if t.shape != u.shape:
        raise ValueError("t and resid must have the same shape")
    n = int(t.size)
    if n < 3:
        return float("nan")

    # design matrix x
    X = np.column_stack((np.ones(n, dtype=float), t))
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("nan")

    # selection residual correlation
    # samples correlation lengths
    tau = float(integrated_autocorr_time(u))
    L_raw = int(math.ceil(5.0 * tau))

    # control cost degeneracy
    L_cap = min(n - 1, 2000, max(1, int(math.floor(0.25 * n))))
    L = int(max(1, min(L_raw, L_cap)))

    S = np.zeros((2, 2), dtype=float)

    # k term
    w0 = u * u
    S += X.T @ (X * w0[:, None])

    # k terms
    for k in range(1, L + 1):
        w = 1.0 - float(k) / float(L + 1)
        uk = u[k:] * u[:-k]
        # gamma k sum
        Gamma = X[k:].T @ (X[:-k] * uk[:, None])
        S += w * (Gamma + Gamma.T)

    V = XtX_inv @ S @ XtX_inv
    vbb = float(V[1, 1])
    if not np.isfinite(vbb):
        return float("nan")
    return float(math.sqrt(max(0.0, vbb)))


def estimate_diffusion_from_msd(
    step: np.ndarray,
    msd: np.ndarray,
    timestep: float,
    fit_start_fraction: float = 0.5,
    *,
    stderr_method: Literal["nw", "ols"] = "nw",
) -> DiffusionEstimate:
    """Diffusion from msd."""

    step = np.asarray(step, dtype=float)
    msd = np.asarray(msd, dtype=float)

    if step.ndim != 1 or msd.ndim != 1:
        raise ValueError("step and msd must be one-dimensional")
    if step.shape != msd.shape:
        raise ValueError("step and msd must have the same shape")
    n = int(step.size)
    if n < 5:
        raise ValueError("Need >= 5 MSD points for a stable estimate")
    if not np.all(np.isfinite(step)):
        raise ValueError("step must contain only finite values")
    if not np.all(np.diff(step) > 0.0):
        raise ValueError("step must be strictly increasing with no duplicates")
    if not np.all(np.isfinite(msd)):
        raise ValueError("msd must contain only finite values")
    if np.any(msd < 0.0):
        raise ValueError("msd must contain only nonnegative values")

    timestep_value = float(timestep)
    if not math.isfinite(timestep_value) or timestep_value <= 0.0:
        raise ValueError("timestep must be finite and > 0")
    fit_fraction = float(fit_start_fraction)
    if not math.isfinite(fit_fraction) or not (0.0 <= fit_fraction < 1.0):
        raise ValueError("fit_start_fraction must be in [0,1)")
    method = str(stderr_method).strip().lower()
    if method not in {"nw", "ols"}:
        raise ValueError("stderr_method must be one of {'nw','ols'}")

    i0 = int(np.floor(n * fit_fraction))
    if n - i0 < 3:
        i0 = max(0, n - 3)

    t = step[i0:] * timestep_value
    y = msd[i0:]

    # ols closed form
    tmean = float(np.mean(t))
    ymean = float(np.mean(y))
    dt = t - tmean
    Sxx = float(np.sum(dt * dt))
    if not np.isfinite(Sxx) or Sxx <= 0.0:
        raise ValueError("Degenerate time axis in MSD data")

    b = float(np.sum(dt * (y - ymean)) / Sxx)
    a = float(ymean - b * tmean)

    resid = y - (a + b * t)

    if method == "ols":
        dof = max(1, int(t.size) - 2)
        s2 = float(np.sum(resid * resid) / float(dof))
        b_stderr = float(math.sqrt(max(0.0, s2 / Sxx)))
    else:
        b_stderr = float(_newey_west_slope_stderr(t, resid))

    D_unconstrained = b / 6.0
    boundary_constrained = bool(D_unconstrained < 0.0)
    D = max(0.0, D_unconstrained)
    D_stderr = b_stderr / 6.0

    return DiffusionEstimate(
        D=float(D),
        D_stderr=float(D_stderr),
        slope=float(b),
        slope_stderr=float(b_stderr),
        intercept=float(a),
        nfit=int(t.size),
        tmin=float(t[0]),
        tmax=float(t[-1]),
        D_unconstrained=float(D_unconstrained),
        boundary_constrained=boundary_constrained,
    )
