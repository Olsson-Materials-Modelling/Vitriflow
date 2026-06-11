from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class WindowStats:
    mean: float
    stderr: float
    n: int

    # integrated autocorrelation estimation
    # i tau int
    tau_int: float = float("nan")

    # effective sample estimation
    n_eff: float = float("nan")


def integrated_autocorr_time(x: np.ndarray, *, max_lag: Optional[int] = None) -> float:
    """Integrated autocorr time."""

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 4:
        return 0.5

    x = x - float(np.mean(x))
    var0 = float(np.mean(x * x))
    if not np.isfinite(var0) or var0 <= 0.0:
        return 0.5

    # based autocovariance length
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * np.conj(f), n=nfft)[:n].real

    # unbiased normalization divide
    acov = acov / np.arange(n, 0, -1, dtype=float)

    c0 = float(acov[0])
    if not np.isfinite(c0) or c0 <= 0.0:
        return 0.5

    rho = acov / c0

    if max_lag is None:
        max_lag = n - 1
    max_lag = int(max(1, min(int(max_lag), n - 1)))

    # autocorrelations pair positive
    tau = 0.5
    k = 1
    while k + 1 <= max_lag:
        pair = float(rho[k] + rho[k + 1])
        if not np.isfinite(pair) or pair <= 0.0:
            break
        tau += pair
        k += 2

    tau = float(max(0.5, tau))
    tau = float(min(tau, 0.5 * n))
    return tau


def window_mean_stderr(
    x: np.ndarray,
    start_fraction: float = 0.5,
    *,
    autocorr: bool = True,
    max_lag: Optional[int] = None,
) -> WindowStats:
    """Window mean stderr."""

    x = np.asarray(x, dtype=float)
    n_total = int(x.size)
    if n_total == 0:
        return WindowStats(mean=float("nan"), stderr=float("nan"), n=0)

    if n_total < 2:
        m = float(np.nanmean(x))
        return WindowStats(mean=m, stderr=float("nan"), n=n_total)

    if not (0.0 <= float(start_fraction) < 1.0):
        raise ValueError("start_fraction must be in [0,1)")

    i0 = int(np.floor(n_total * float(start_fraction)))
    if n_total - i0 < 2:
        i0 = max(0, n_total - 2)

    w = x[i0:]
    w = w[np.isfinite(w)]
    n = int(w.size)
    if n == 0:
        return WindowStats(mean=float("nan"), stderr=float("nan"), n=0)

    mean = float(np.mean(w))
    if n < 2:
        return WindowStats(mean=mean, stderr=float("nan"), n=n)

    s2 = float(np.var(w, ddof=1))
    if not np.isfinite(s2) or s2 <= 0.0:
        return WindowStats(mean=mean, stderr=0.0, n=n, tau_int=0.5, n_eff=float(n))

    if autocorr and n >= 4:
        tau = float(integrated_autocorr_time(w, max_lag=max_lag))
        n_eff = float(n) / float(2.0 * tau) if tau > 0 else float(n)
        n_eff = float(max(1.0, min(float(n), n_eff)))
        se = math.sqrt(s2 / n_eff)
        return WindowStats(mean=mean, stderr=float(se), n=n, tau_int=tau, n_eff=n_eff)

    se = math.sqrt(s2 / float(n))
    return WindowStats(mean=mean, stderr=float(se), n=n, tau_int=0.5, n_eff=float(n))


@dataclass(frozen=True)
class TwoWindowChange:
    """Two window change."""

    early_mean: float
    late_mean: float
    abs_change: float
    rel_change: float
    n_early: int
    n_late: int


def early_late_change(
    x: np.ndarray,
    *,
    split_fraction: float = 0.5,
    eps: float = 1.0e-12,
    denom: str = "late",
) -> TwoWindowChange:
    """Early late change."""

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return TwoWindowChange(
            early_mean=float("nan"),
            late_mean=float("nan"),
            abs_change=float("nan"),
            rel_change=float("nan"),
            n_early=0,
            n_late=0,
        )

    sf = float(split_fraction)
    if not (0.0 < sf < 1.0):
        raise ValueError("split_fraction must be in (0,1)")

    i = int(np.floor(n * sf))
    i = int(max(1, min(n - 1, i)))

    early = x[:i]
    late = x[i:]
    n_early = int(early.size)
    n_late = int(late.size)

    mu_early = float(np.mean(early)) if n_early > 0 else float("nan")
    mu_late = float(np.mean(late)) if n_late > 0 else float("nan")

    if not (math.isfinite(mu_early) and math.isfinite(mu_late)):
        return TwoWindowChange(
            early_mean=mu_early,
            late_mean=mu_late,
            abs_change=float("nan"),
            rel_change=float("nan"),
            n_early=n_early,
            n_late=n_late,
        )

    abs_change = float(abs(mu_late - mu_early))

    denom_key = str(denom).strip().lower()
    if denom_key == "late":
        d = abs(mu_late)
    elif denom_key == "early":
        d = abs(mu_early)
    elif denom_key == "max":
        d = max(abs(mu_early), abs(mu_late))
    else:
        raise ValueError("denom must be one of {'late','early','max'}")

    rel_change = float(abs_change / max(float(eps), float(d)))
    return TwoWindowChange(
        early_mean=mu_early,
        late_mean=mu_late,
        abs_change=abs_change,
        rel_change=rel_change,
        n_early=n_early,
        n_late=n_late,
    )
