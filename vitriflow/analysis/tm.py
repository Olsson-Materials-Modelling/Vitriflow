from __future__ import annotations

from dataclasses import dataclass
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



def estimate_tm_from_diffusion(T: np.ndarray, D: np.ndarray, eps: float = 1e-30) -> TmEstimate:
    """Tm from diffusion."""
    if len(T) != len(D):
        raise ValueError("T and D must have same length")
    if len(T) < 3:
        raise ValueError("Need >= 3 temperatures to estimate Tm")

    # sort temperature
    order = np.argsort(T)
    T = np.asarray(T, dtype=float)[order]
    D = np.asarray(D, dtype=float)[order]

    # diffusion physically msd
    # negative slopes number
    Dp = np.where(np.isfinite(D) & (D > 0.0), D, 0.0)
    y = np.log(Dp + eps)

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

    if len(T) != len(D):
        raise ValueError("T and D must have same length")
    if len(T) < 3:
        raise ValueError("Need >= 3 temperatures to estimate Tm")

    order = np.argsort(T)
    T = np.asarray(T, dtype=float)[order]
    D = np.asarray(D, dtype=float)[order]

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
        m = np.isfinite(x)
        if int(np.sum(m)) < 2:
            return np.zeros_like(x)
        mu = float(np.mean(x[m]))
        sd = float(np.std(x[m], ddof=1))
        if sd <= 0:
            return np.zeros_like(x)
        z = np.zeros_like(x)
        z[m] = (x[m] - mu) / sd
        z[~m] = 0.0
        return z

    # clamp diffusion transform
    D = np.asarray(D, dtype=float)
    Dp = np.where(np.isfinite(D) & (D > 0.0), D, 0.0)
    yD = np.log(Dp + eps)
    dD = _deriv(_smooth3(yD))

    if gr_peak_height is None or gr_peak_fwhm is None:
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

    H = np.asarray(gr_peak_height, dtype=float)[order]
    W = np.asarray(gr_peak_fwhm, dtype=float)[order]

    # peak height negative
    Hp = np.where(np.isfinite(H) & (H > 0.0), H, 0.0)
    Wp = np.where(np.isfinite(W) & (W > 0.0), W, 0.0)
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
    if (
        msd_rms_last is not None
        and vol_last is not None
        and natoms is not None
        and int(natoms) > 0
    ):
        msd_rms_last = np.asarray(msd_rms_last, dtype=float)[order]
        vol_last = np.asarray(vol_last, dtype=float)[order]
        l = np.where(
            np.isfinite(vol_last) & (vol_last > 0.0),
            (vol_last / float(natoms)) ** (1.0 / 3.0),
            np.nan,
        )
        ratio = msd_rms_last / (l + eps)
        mobility_ok = np.isfinite(ratio) & (ratio >= float(mobility_ratio_threshold))

    if mobility_ok is None:
        # fallback
        # lowest temperatures baseline
        b = int(max(1, min(int(baseline_n), len(T))))
        base = yD[:b]
        base_med = float(np.nanmedian(base))
        # order magnitude increase
        mobility_ok = np.isfinite(yD) & (yD >= base_med + np.log(10.0))

    # structure classifier peak
    struct_ok = np.ones_like(mobility_ok, dtype=bool)
    if gr_peak_height is not None and gr_peak_fwhm is not None:
        b = int(max(1, min(int(baseline_n), len(T))))
        H0 = float(np.nanmedian(Hp[:b]))
        W0 = float(np.nanmedian(Wp[:b]))
        if not (np.isfinite(H0) and H0 > 0.0 and np.isfinite(W0) and W0 > 0.0):
            struct_ok = np.zeros_like(mobility_ok, dtype=bool)
        else:
            struct_ok = (
                np.isfinite(Hp)
                & np.isfinite(Wp)
                & (Hp <= float(height_frac_threshold) * H0)
                & (Wp >= float(width_factor_threshold) * W0)
            )

    melted = mobility_ok & struct_ok


    # conservative temperature selection
    # diffusion highest temperatures
    # mobility diffusion temperatures
    D_valid = [(float(T[i]), float(Dp[i])) for i in range(len(T)) if np.isfinite(Dp[i]) and Dp[i] > 0.0]
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
        liquid_like = np.isfinite(Dp) & (Dp >= D_liquid_target)
        if np.any(np.isfinite(chi)):
            liquid_like = liquid_like & np.isfinite(chi) & (chi >= float(liquid_mobility_threshold))

    mliq = int(max(1, int(liquid_min_consecutive)))
    run2 = 0
    onset_liq = None
    for i, ok in enumerate(liquid_like.tolist()):
        run2 = run2 + 1 if ok else 0
        if run2 >= mliq:
            onset_liq = i - mliq + 1
            break
    T_liquid = float(T[onset_liq]) if onset_liq is not None else float(T[-1])

    m = int(max(1, int(melt_confirm)))
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
