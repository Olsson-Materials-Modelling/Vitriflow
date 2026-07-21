from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TmEstimate:
    Tm: float
    idx: int
    method: str
    score: float

    dlogD_dT: np.ndarray
    # structural derivatives structure
    dlogH_dT: np.ndarray | None = None
    dlogW_dT: np.ndarray | None = None
    combined_score: np.ndarray | None = None

    # conservative temperature choosing
    T_liquid: float = float('nan')
    D_liquid_target: float = float('nan')


def _validated_temperature_diffusion(
    T: np.ndarray,
    D: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Validate and sort a complete temperature/diffusion scan."""

    T_arr = np.asarray(T, dtype=float)
    D_arr = np.asarray(D, dtype=float)
    if T_arr.ndim != 1 or D_arr.ndim != 1:
        raise ValueError("T and D must be one-dimensional")
    if T_arr.shape != D_arr.shape:
        raise ValueError("T and D must have same length")
    if T_arr.size < 3:
        raise ValueError("Need >= 3 temperatures to estimate Tm")
    if not np.all(np.isfinite(T_arr)):
        raise ValueError("T must contain only finite temperatures")
    if np.any(T_arr < 0.0):
        raise ValueError("T must contain only nonnegative absolute temperatures")
    if not np.all(np.isfinite(D_arr)):
        raise ValueError("D must contain complete finite diffusion evidence")
    if np.any(D_arr < 0.0):
        raise ValueError("D must contain only nonnegative diffusion estimates")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps must be finite and > 0")

    order = np.argsort(T_arr, kind="mergesort")
    T_sorted = T_arr[order]
    D_sorted = D_arr[order]
    if not np.all(np.diff(T_sorted) > 0.0):
        raise ValueError("T must contain distinct temperatures")
    return T_sorted, D_sorted, order, eps_value


def _validated_scan_array(
    name: str,
    values: np.ndarray,
    *,
    n: int,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size != int(n):
        raise ValueError(f"{name} must be one-dimensional with the same length as T")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain complete finite evidence")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain only values > 0")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must contain only values >= 0")
    return arr


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= 1")
    numeric = float(value)
    integer = int(numeric)
    if not math.isfinite(numeric) or numeric != float(integer) or integer < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return integer


def _finite_parameter(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if strict_minimum and not result > minimum:
            raise ValueError(f"{name} must be > {minimum:g}")
        if not strict_minimum and result < minimum:
            raise ValueError(f"{name} must be >= {minimum:g}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}")
    return result



def estimate_tm_from_diffusion(T: np.ndarray, D: np.ndarray, eps: float = 1e-30) -> TmEstimate:
    """Tm from diffusion."""
    T, D, _order, eps = _validated_temperature_diffusion(T, D, eps=eps)

    y = np.log(D + eps)

    # simple smoothing window
    y_s = y.copy()
    if len(y) >= 3:
        for i in range(1, len(y) - 1):
            y_s[i] = (y[i - 1] + y[i] + y[i + 1]) / 3.0

    d = np.zeros_like(y_s)
    # central differences interior
    for i in range(1, len(T) - 1):
        d[i] = (y_s[i + 1] - y_s[i - 1]) / (T[i + 1] - T[i - 1])
    # sided ends
    d[0] = (y_s[1] - y_s[0]) / (T[1] - T[0])
    d[-1] = (y_s[-1] - y_s[-2]) / (T[-1] - T[-2])

    idx = int(np.argmax(d))
    return TmEstimate(Tm=float(T[idx]), idx=idx, method="max_dlogD_dT", score=float(d[idx]), dlogD_dT=d)


def estimate_tm(
    T: np.ndarray,
    D: np.ndarray,
    *,
    gr_peak_height: np.ndarray | None = None,
    gr_peak_fwhm: np.ndarray | None = None,
    # volume melt confirmation
    msd_rms_last: np.ndarray | None = None,
    vol_last: np.ndarray | None = None,
    natoms: int | None = None,
    melt_confirm: int = 2,
    mobility_ratio_threshold: float = 0.5,
    height_frac_threshold: float = 0.65,
    width_factor_threshold: float = 1.5,
    baseline_n: int = 3,
    w_diffusion: float = 1.0,
    w_peak_height: float = 1.0,
    w_peak_fwhm: float = 0.5,
    # conservative selection parameters
    liquid_D_frac: float = 0.2,
    liquid_top_k: int = 3,
    liquid_min_consecutive: int = 2,
    liquid_mobility_threshold: float = 1.0,
    eps: float = 1e-30,
) -> TmEstimate:
    """Tm."""

    T, D, order, eps = _validated_temperature_diffusion(T, D, eps=eps)
    n_scan = int(T.size)

    structural_any = gr_peak_height is not None or gr_peak_fwhm is not None
    structural_complete = gr_peak_height is not None and gr_peak_fwhm is not None
    if structural_any and not structural_complete:
        raise ValueError("gr_peak_height and gr_peak_fwhm must be provided together")

    mobility_items = (msd_rms_last, vol_last, natoms)
    mobility_any = any(value is not None for value in mobility_items)
    mobility_complete = all(value is not None for value in mobility_items)
    if mobility_any and not mobility_complete:
        raise ValueError("msd_rms_last, vol_last, and natoms must be provided together")
    if mobility_complete and not structural_complete:
        raise ValueError("mobility evidence requires structural peak evidence")

    melt_confirm = _positive_integer("melt_confirm", melt_confirm)
    baseline_n = _positive_integer("baseline_n", baseline_n)
    liquid_top_k = _positive_integer("liquid_top_k", liquid_top_k)
    liquid_min_consecutive = _positive_integer(
        "liquid_min_consecutive", liquid_min_consecutive
    )
    mobility_ratio_threshold = _finite_parameter(
        "mobility_ratio_threshold", mobility_ratio_threshold, minimum=0.0
    )
    height_frac_threshold = _finite_parameter(
        "height_frac_threshold",
        height_frac_threshold,
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
    )
    width_factor_threshold = _finite_parameter(
        "width_factor_threshold", width_factor_threshold, minimum=0.0, strict_minimum=True
    )
    w_diffusion = _finite_parameter("w_diffusion", w_diffusion, minimum=0.0)
    w_peak_height = _finite_parameter("w_peak_height", w_peak_height, minimum=0.0)
    w_peak_fwhm = _finite_parameter("w_peak_fwhm", w_peak_fwhm, minimum=0.0)
    if max(w_diffusion, w_peak_height, w_peak_fwhm) <= 0.0:
        raise ValueError("at least one Tm indicator weight must be > 0")
    liquid_D_frac = _finite_parameter(
        "liquid_D_frac", liquid_D_frac, minimum=0.0, maximum=1.0, strict_minimum=True
    )
    liquid_mobility_threshold = _finite_parameter(
        "liquid_mobility_threshold", liquid_mobility_threshold, minimum=0.0
    )

    H: np.ndarray | None = None
    W: np.ndarray | None = None
    if structural_complete:
        H = _validated_scan_array(
            "gr_peak_height", gr_peak_height, n=n_scan, positive=True
        )[order]
        W = _validated_scan_array(
            "gr_peak_fwhm", gr_peak_fwhm, n=n_scan, positive=True
        )[order]

    msd_values: np.ndarray | None = None
    volume_values: np.ndarray | None = None
    natoms_value: int | None = None
    if mobility_complete:
        msd_values = _validated_scan_array(
            "msd_rms_last", msd_rms_last, n=n_scan, nonnegative=True
        )[order]
        volume_values = _validated_scan_array(
            "vol_last", vol_last, n=n_scan, positive=True
        )[order]
        natoms_value = _positive_integer("natoms", natoms)

    def _smooth3(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        y_s = y.copy()
        if len(y) >= 3:
            for i in range(1, len(y) - 1):
                y_s[i] = (y[i - 1] + y[i] + y[i + 1]) / 3.0
        return y_s

    def _deriv(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        d = np.zeros_like(y)
        for i in range(1, len(T) - 1):
            d[i] = (y[i + 1] - y[i - 1]) / (T[i + 1] - T[i - 1])
        d[0] = (y[1] - y[0]) / (T[1] - T[0])
        d[-1] = (y[-1] - y[-2]) / (T[-1] - T[-2])
        return d

    def _zscore(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != n_scan or not np.all(np.isfinite(x)):
            raise ValueError("Tm indicator derivatives must be complete and finite")
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        if sd <= 0:
            return np.zeros_like(x)
        return (x - mu) / sd

    yD = np.log(D + eps)
    dD = _deriv(_smooth3(yD))

    if not structural_complete:
        idx = int(np.argmax(dD))
        return TmEstimate(
            Tm=float(T[idx]),
            idx=idx,
            method="max_dlogD_dT",
            score=float(dD[idx]),
            dlogD_dT=dD,
            dlogH_dT=None,
            dlogW_dT=None,
            combined_score=None,
        )

    assert H is not None and W is not None

    # peak height negative
    Hp = H
    Wp = W
    yH = -np.log(Hp + eps)
    yW = np.log(Wp + eps)
    dH = _deriv(_smooth3(yH))
    dW = _deriv(_smooth3(yW))

    score = (
        float(w_diffusion) * _zscore(dD)
        + float(w_peak_height) * _zscore(dH)
        + float(w_peak_fwhm) * _zscore(dW)
    )
    # selection requiring sustained
    # selection requiring sustained
    # liquid regime adjacent
    # idx
    idx = int(np.argmax(score))
    method = "combined_dynamics_structure"
    best_score = float(score[idx])

    # mobility classifier displacement
    mobility_ok = None
    ratio = None
    if mobility_complete:
        assert msd_values is not None and volume_values is not None and natoms_value is not None
        l = (volume_values / float(natoms_value)) ** (1.0 / 3.0)
        ratio = msd_values / (l + eps)
        mobility_ok = ratio >= mobility_ratio_threshold

    if mobility_ok is None:
        # fallback
        # lowest temperatures baseline
        b = int(max(1, min(baseline_n, len(T))))
        base = yD[:b]
        base_med = float(np.nanmedian(base))
        # order magnitude increase
        mobility_ok = np.isfinite(yD) & (yD >= base_med + np.log(10.0))

    # structure classifier peak
    struct_ok = np.ones_like(mobility_ok, dtype=bool)
    b = int(max(1, min(baseline_n, len(T))))
    H0 = float(np.median(Hp[:b]))
    W0 = float(np.median(Wp[:b]))
    struct_ok = (
        (Hp <= height_frac_threshold * H0)
        & (Wp >= width_factor_threshold * W0)
    )

    melted = mobility_ok & struct_ok


    # conservative temperature selection
    # diffusion highest temperatures
    # mobility diffusion temperatures
    D_valid = [(float(T[i]), float(D[i])) for i in range(len(T)) if D[i] > 0.0]
    D_valid_sorted = [d for _, d in sorted(D_valid, key=lambda x: x[0], reverse=True)]
    if len(D_valid_sorted) == 0:
        D_scale = float('nan')
    else:
        k = int(max(1, min(int(liquid_top_k), len(D_valid_sorted))))
        D_scale = float(np.median(D_valid_sorted[:k]))
    D_liquid_target = float(liquid_D_frac * D_scale) if np.isfinite(D_scale) else float('nan')

    # mobility ratio otherwise
    chi = ratio if ratio is not None else np.full_like(T, np.nan, dtype=float)

    liquid_like = np.zeros_like(melted, dtype=bool)
    if np.isfinite(D_liquid_target):
        liquid_like = D >= D_liquid_target
        if np.any(np.isfinite(chi)):
            liquid_like = liquid_like & np.isfinite(chi) & (chi >= liquid_mobility_threshold)

    mliq = liquid_min_consecutive
    run2 = 0
    onset_liq = None
    for i, ok in enumerate(liquid_like.tolist()):
        run2 = run2 + 1 if ok else 0
        if run2 >= mliq:
            onset_liq = i - mliq + 1
            break
    T_liquid = float(T[onset_liq]) if onset_liq is not None else float("nan")

    m = melt_confirm
    onset = None
    if len(melted) >= m:
        for i in range(0, len(melted) - m + 1):
            if bool(np.all(melted[i : i + m])):
                onset = i
                break

    if onset is not None:
        # choose temperature sustained
        idx2 = int(max(onset - 1, 0))
        idx = idx2
        method = f"confirmed_melt_onset(m={m})"
        best_score = float(score[idx])

    return TmEstimate(
        Tm=float(T[idx]),
        idx=idx,
        T_liquid=float(T_liquid),
        D_liquid_target=float(D_liquid_target) if np.isfinite(D_liquid_target) else float('nan'),
        method=method,
        score=best_score,
        dlogD_dT=dD,
        dlogH_dT=dH,
        dlogW_dT=dW,
        combined_score=score,
    )
