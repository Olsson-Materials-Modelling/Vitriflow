from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Literal

import numpy as np

from ..config import AtomSelector
from .dump import DumpFrame
from .gr import compute_gr
from .common import resolve_selector as _resolve_selector


@dataclass(frozen=True)
class SqResult:
    q: np.ndarray
    s: np.ndarray
    peak_q: float
    peak_height: float
    peak_fwhm: float


def _window_weights(r: np.ndarray, r_max: float, kind: str) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be 1D")
    if not (math.isfinite(float(r_max)) and float(r_max) > 0.0):
        raise ValueError("r_max must be > 0")

    k = str(kind).strip().lower()
    if k in ("none", "off", "false"):
        return np.ones_like(r, dtype=float)

    x = (math.pi * r) / float(r_max)

    if k == "lorch":
        # lorch window sin
        w = np.ones_like(r, dtype=float)
        m = np.abs(x) > 0
        w[m] = np.sin(x[m]) / x[m]
        return w

    if k in ("hann", "hanning"):
        # hann window max
        w = 0.5 * (1.0 + np.cos(x))
        # exactly floating error
        w = np.clip(w, 0.0, 1.0)
        return w

    raise ValueError(f"Unknown window kind: {kind}")


def _effective_density_and_base(
    frames: Sequence[DumpFrame],
    pair: Optional[Tuple[AtomSelector, AtomSelector]],
    type_to_species: Optional[Sequence[str]],
) -> tuple[float, float]:
    if not frames:
        raise ValueError("frames must be non-empty")

    rhos: list[float] = []

    if pair is None:
        for fr in frames:
            V = abs(float(np.linalg.det(fr.cell)))
            if fr.n_atoms > 0 and V > 0:
                rhos.append(float(fr.n_atoms) / V)
        if not rhos:
            return float("nan"), 1.0
        return float(np.mean(rhos)), 1.0

    a_sel, b_sel = pair
    A_types = set(_resolve_selector(a_sel, type_to_species))
    B_types = set(_resolve_selector(b_sel, type_to_species))
    if A_types != B_types and (A_types & B_types):
        raise ValueError(
            f"Overlapping type selections for S(q) pair {pair} are not supported: {sorted(A_types & B_types)}"
        )

    rhoA: list[float] = []
    rhoB: list[float] = []
    for fr in frames:
        V = abs(float(np.linalg.det(fr.cell)))
        if V <= 0:
            continue
        t = fr.types
        Na = float(np.sum(np.isin(t, list(A_types))))
        Nb = float(np.sum(np.isin(t, list(B_types))))
        if Na > 0:
            rhoA.append(Na / V)
        if Nb > 0:
            rhoB.append(Nb / V)

    if A_types == B_types:
        if not rhoA:
            return float("nan"), 1.0
        return float(np.mean(rhoA)), 1.0

    if not rhoA or not rhoB:
        return float("nan"), 0.0
    return float(math.sqrt(float(np.mean(rhoA)) * float(np.mean(rhoB)))), 0.0


def compute_sq(
    frames: Sequence[DumpFrame],
    *,
    q_max: float,
    nq: int,
    r_max: float,
    nbins: int,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    window: Literal["lorch", "hann", "none"] = "lorch",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sq."""
    if not frames:
        raise ValueError("compute_sq requires at least one frame")
    if not (math.isfinite(float(q_max)) and float(q_max) > 0.0):
        raise ValueError("q_max must be > 0")
    if int(nq) < 10:
        raise ValueError("nq must be >= 10")
    if not (math.isfinite(float(r_max)) and float(r_max) > 0.0):
        raise ValueError("r_max must be > 0")
    if int(nbins) < 50:
        raise ValueError("nbins must be >= 50")

    r, g, _l = compute_gr(frames, r_max=float(r_max), nbins=int(nbins), pair=pair, type_to_species=type_to_species)
    if r.size < 2:
        raise ValueError("g(r) grid too small")

    dr = float(r[1] - r[0])
    if not (math.isfinite(dr) and dr > 0.0):
        raise ValueError("invalid g(r) grid spacing")

    # effective cutoff normalization
    r_max_eff = float(r[-1] + 0.5 * dr)

    w = _window_weights(r, r_max_eff, str(window))
    h = np.asarray(g, dtype=float) - 1.0

    rho_eff, base = _effective_density_and_base(frames, pair, type_to_species)
    if not (math.isfinite(rho_eff) and rho_eff > 0.0):
        # meaningful s density
        q = np.linspace(0.0, float(q_max), int(nq), dtype=float)
        return q, np.full_like(q, float("nan"), dtype=float)

    A = (r**2) * h * w * dr

    q = np.linspace(0.0, float(q_max), int(nq), dtype=float)

    # vectorised sinc limit
    qr = np.outer(q, r)
    sinc = np.ones_like(qr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.abs(qr) > 0.0
        sinc[m] = np.sin(qr[m]) / qr[m]

    S = float(base) + 4.0 * math.pi * float(rho_eff) * (sinc @ A)
    return q, np.asarray(S, dtype=float)


def first_peak_features(
    q: np.ndarray,
    s: np.ndarray,
    *,
    smooth: int = 7,
    q_min: float = 0.5,
    q_max: Optional[float] = 3.0,
) -> tuple[float, float, float]:
    """First peak features."""
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)
    if q.ndim != 1 or s.ndim != 1 or q.size != s.size:
        raise ValueError("q and s must be 1D arrays of equal length")
    if q.size < 10:
        return float("nan"), float("nan"), float("nan")

    # enforce monotone grid
    if not np.all(np.diff(q) > 0):
        raise ValueError("q grid must be strictly increasing")

    w = int(smooth)
    if w < 1:
        w = 1
    if w % 2 == 0:
        w += 1
    if w > 1:
        ker = np.ones(w, dtype=float) / float(w)
        s_s = np.convolve(np.nan_to_num(s, nan=0.0), ker, mode="same")
    else:
        s_s = np.array(s, dtype=float)

    q0 = max(float(q_min), float(q[0]))
    q1 = float(q[-1]) if q_max is None else min(float(q_max), float(q[-1]))
    if not (q1 > q0):
        return float("nan"), float("nan"), float("nan")

    m = (q >= q0) & (q <= q1) & np.isfinite(s_s)
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")

    idxs = np.where(m)[0]
    idx_peak = int(idxs[int(np.argmax(s_s[m]))])

    q_peak = float(q[idx_peak])
    h = float(s_s[idx_peak])

    # baseline peak window
    m_pre = (q >= q0) & (q <= q_peak) & np.isfinite(s_s)
    baseline = float(np.min(s_s[m_pre])) if np.any(m_pre) else float(s_s[idx_peak])
    half = baseline + 0.5 * (h - baseline)

    # crossing
    left = None
    for i in range(idx_peak, 0, -1):
        if not (math.isfinite(float(s_s[i - 1])) and math.isfinite(float(s_s[i]))):
            continue
        if s_s[i - 1] <= half <= s_s[i] or s_s[i] <= half <= s_s[i - 1]:
            x0, x1 = float(q[i - 1]), float(q[i])
            y0, y1 = float(s_s[i - 1]), float(s_s[i])
            if y1 == y0:
                left = x0
            else:
                left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    # crossing
    right = None
    for i in range(idx_peak, q.size - 1):
        if not (math.isfinite(float(s_s[i])) and math.isfinite(float(s_s[i + 1]))):
            continue
        if s_s[i] >= half >= s_s[i + 1] or s_s[i] <= half <= s_s[i + 1]:
            x0, x1 = float(q[i]), float(q[i + 1])
            y0, y1 = float(s_s[i]), float(s_s[i + 1])
            if y1 == y0:
                right = x1
            else:
                right = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    if left is None or right is None:
        fwhm = float("nan")
    else:
        fwhm = float(max(0.0, right - left))

    return q_peak, h, fwhm


def compute_first_peak_sq(
    frames: Sequence[DumpFrame],
    *,
    q_max: float,
    nq: int,
    r_max: float,
    nbins: int,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    window: Literal["lorch", "hann", "none"] = "lorch",
    smooth: int = 7,
    peak_search: Tuple[float, float] = (0.5, 3.0),
) -> SqResult:
    q, s = compute_sq(
        frames,
        q_max=float(q_max),
        nq=int(nq),
        r_max=float(r_max),
        nbins=int(nbins),
        pair=pair,
        type_to_species=type_to_species,
        window=window,
    )
    q0, q1 = float(peak_search[0]), float(peak_search[1])
    q_peak, h, fwhm = first_peak_features(q, s, smooth=int(smooth), q_min=q0, q_max=q1)
    return SqResult(q=q, s=s, peak_q=q_peak, peak_height=h, peak_fwhm=fwhm)
