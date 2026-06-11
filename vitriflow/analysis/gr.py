from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from ase import Atoms
    from ase.neighborlist import neighbor_list
except Exception as e:  # pragma: no cover
    raise ImportError(
        "vitriflow.analysis.gr requires 'ase'. Install via pip/conda: pip install ase"
    ) from e

from ..config import AtomSelector
from .dump import DumpFrame
from .common import resolve_selector as _resolve_selector, wrap_frac as _wrap_frac, mic_distances as _mic_distances



@dataclass(frozen=True)
class GrResult:
    r: np.ndarray
    g: np.ndarray
    peak_r: float
    peak_height: float
    peak_fwhm: float


def compute_gr(
    frames: Sequence[DumpFrame],
    *,
    r_max: float,
    nbins: int,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Gr."""
    if not frames:
        raise ValueError("compute_gr requires at least one frame")
    if r_max <= 0:
        raise ValueError("r_max must be > 0")
    if nbins < 10:
        raise ValueError("nbins must be >= 10")

    # pair based requested
    A_types: Optional[set[int]] = None
    B_types: Optional[set[int]] = None
    if pair is not None:
        a_sel, b_sel = pair
        A_types = set(_resolve_selector(a_sel, type_to_species))
        B_types = set(_resolve_selector(b_sel, type_to_species))
        if A_types != B_types and (A_types & B_types):
            raise ValueError(
                f"Overlapping type selections for g(r) pair {pair} are not supported: {sorted(A_types & B_types)}"
            )

    # smallest box consistent
    half_lengths = []
    mean_spacings = []
    for fr in frames:
        lens = [float(np.linalg.norm(fr.cell[i])) for i in range(3)]
        half_lengths.append(0.5 * min(lens))
        V = abs(float(np.linalg.det(fr.cell)))
        if fr.n_atoms > 0 and V > 0:
            rho = fr.n_atoms / V
            mean_spacings.append(rho ** (-1.0 / 3.0))
    if not half_lengths:
        raise ValueError("No frames available")
    r_max_eff = min(float(r_max), float(min(half_lengths)))
    if r_max_eff <= 0:
        raise ValueError("Effective r_max <= 0; check cell")

    edges = np.linspace(0.0, r_max_eff, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = float(edges[1] - edges[0])

    g_frames: list[np.ndarray] = []

    for fr in frames:
        invH = np.linalg.inv(fr.cell)
        frac = _wrap_frac((fr.positions - fr.origin) @ invH)
        posw = fr.origin + frac @ fr.cell

        # update shortest box
        lens = [float(np.linalg.norm(fr.cell[i])) for i in range(3)]
        rcut = min(r_max_eff, 0.5 * min(lens))

        atoms = Atoms(numbers=np.ones(fr.n_atoms, dtype=int), positions=posw, cell=fr.cell, pbc=True)
        ii, jj = neighbor_list("ij", atoms, rcut)
        m = ii < jj
        ii = ii[m]
        jj = jj[m]

        if pair is not None and A_types is not None and B_types is not None:
            ti = fr.types[ii]
            tj = fr.types[jj]
            if A_types == B_types:
                want = np.isin(ti, list(A_types)) & np.isin(tj, list(A_types))
            else:
                want = (np.isin(ti, list(A_types)) & np.isin(tj, list(B_types))) | (
                    np.isin(ti, list(B_types)) & np.isin(tj, list(A_types))
                )
            ii = ii[want]
            jj = jj[want]

        if ii.size == 0:
            g_frames.append(np.full((centers.size,), np.nan, dtype=float))
            continue

        dist = _mic_distances(frac, fr.cell, ii, jj)
        dist = dist[np.isfinite(dist)]
        dist = dist[(dist > 1e-8) & (dist <= r_max_eff)]

        counts, _ = np.histogram(dist, bins=edges)
        counts = counts.astype(float)

        V = abs(float(np.linalg.det(fr.cell)))
        if V <= 0 or fr.n_atoms < 2:
            g_frames.append(np.full((centers.size,), np.nan, dtype=float))
            continue

        if pair is None:
            n_pairs = fr.n_atoms * (fr.n_atoms - 1) / 2.0
        else:
            # selections
            t = fr.types
            if A_types == B_types:
                Na = float(np.sum(np.isin(t, list(A_types))))
                n_pairs = Na * (Na - 1.0) / 2.0
            else:
                Na = float(np.sum(np.isin(t, list(A_types))))
                Nb = float(np.sum(np.isin(t, list(B_types))))
                n_pairs = Na * Nb

        if n_pairs <= 0:
            g_frames.append(np.full((centers.size,), np.nan, dtype=float))
            continue

        shell_vol = 4.0 * math.pi * (centers**2) * dr
        ideal = (n_pairs / V) * shell_vol

        g = np.full_like(centers, np.nan, dtype=float)
        m2 = ideal > 0
        g[m2] = counts[m2] / ideal[m2]
        g_frames.append(g)

    g_stack = np.vstack([g for g in g_frames if g.size == centers.size])
    if g_stack.size == 0:
        raise ValueError("Failed to compute g(r)")

    # warn
    # contains finite samples
    finite = np.isfinite(g_stack)
    counts = np.sum(finite, axis=0)
    g_mean = np.full((centers.size,), np.nan, dtype=float)
    if np.any(counts > 0):
        sums = np.nansum(g_stack, axis=0)
        m = counts > 0
        g_mean[m] = sums[m] / counts[m]

    # spacing peak heuristics
    l = float(np.mean(mean_spacings)) if mean_spacings else float("nan")
    return centers, g_mean, l


def first_peak_features(
    r: np.ndarray,
    g: np.ndarray,
    *,
    smooth: int = 7,
    r_ignore: Optional[float] = None,
    r_search_max: Optional[float] = None,
    r_ignore_factor: float = 0.3,
    r_search_factor: float = 2.5,
    mean_spacing: Optional[float] = None,
) -> Tuple[float, float, float]:
    """First peak features."""
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    if r.ndim != 1 or g.ndim != 1 or r.size != g.size:
        raise ValueError("r and g must be 1D arrays of equal length")
    if r.size < 10:
        return float("nan"), float("nan"), float("nan")

    # smooth
    w = int(smooth)
    if w < 1:
        w = 1
    if w % 2 == 0:
        w += 1
    if w > 1:
        ker = np.ones(w, dtype=float) / float(w)
        g_s = np.convolve(np.nan_to_num(g, nan=0.0), ker, mode="same")
    else:
        g_s = np.array(g, dtype=float)

    l = float(mean_spacing) if mean_spacing is not None and math.isfinite(float(mean_spacing)) else float("nan")

    if r_ignore is None:
        r_ignore = float(r_ignore_factor * l) if math.isfinite(l) else float(r[1])
    if r_search_max is None:
        r_search_max = float(r_search_factor * l) if math.isfinite(l) else float(r[-1])

    r_ignore = max(float(r_ignore), float(r[0]))
    r_search_max = min(float(r_search_max), float(r[-1]))

    m = (r >= r_ignore) & (r <= r_search_max) & np.isfinite(g_s)
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")

    idxs = np.where(m)[0]
    idx_peak = int(idxs[int(np.argmax(g_s[m]))])

    r_peak = float(r[idx_peak])
    h = float(g_s[idx_peak])

    # baseline minimum peak
    m_pre = (r >= r_ignore) & (r <= r_peak) & np.isfinite(g_s)
    baseline = float(np.min(g_s[m_pre])) if np.any(m_pre) else 0.0
    half = baseline + 0.5 * (h - baseline)

    # crossing
    left = None
    for i in range(idx_peak, 0, -1):
        if not math.isfinite(float(g_s[i - 1])) or not math.isfinite(float(g_s[i])):
            continue
        if g_s[i - 1] <= half <= g_s[i] or g_s[i] <= half <= g_s[i - 1]:
            # interpolate between i
            x0, x1 = float(r[i - 1]), float(r[i])
            y0, y1 = float(g_s[i - 1]), float(g_s[i])
            if y1 == y0:
                left = x0
            else:
                left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    right = None
    for i in range(idx_peak, r.size - 1):
        if not math.isfinite(float(g_s[i])) or not math.isfinite(float(g_s[i + 1])):
            continue
        if g_s[i] >= half >= g_s[i + 1] or g_s[i] <= half <= g_s[i + 1]:
            x0, x1 = float(r[i]), float(r[i + 1])
            y0, y1 = float(g_s[i]), float(g_s[i + 1])
            if y1 == y0:
                right = x1
            else:
                right = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    if left is None or right is None or right <= left:
        fwhm = float("nan")
    else:
        fwhm = float(right - left)

    return r_peak, h, fwhm


def compute_first_peak_gr(
    frames: Sequence[DumpFrame],
    *,
    r_max: float,
    nbins: int,
    smooth: int = 7,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    r_ignore_factor: float = 0.3,
    r_search_factor: float = 2.5,
) -> GrResult:
    """First peak gr."""
    r, g, l = compute_gr(frames, r_max=r_max, nbins=nbins, pair=pair, type_to_species=type_to_species)
    rp, h, w = first_peak_features(
        r,
        g,
        smooth=smooth,
        r_ignore_factor=r_ignore_factor,
        r_search_factor=r_search_factor,
        mean_spacing=l,
    )
    return GrResult(r=r, g=g, peak_r=rp, peak_height=h, peak_fwhm=w)
