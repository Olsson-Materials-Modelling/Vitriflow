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
from .dump import DumpFrame, frame_pbc
from .common import (
    canonical_unique_mic_pairs as _canonical_unique_mic_pairs,
    resolve_selector as _resolve_selector,
    wrap_frac as _wrap_frac,
)



@dataclass(frozen=True)
class GrResult:
    r: np.ndarray
    g: np.ndarray
    peak_r: float
    peak_height: float
    peak_fwhm: float


def _shortest_lattice_translation(
    cell: np.ndarray,
    *,
    pbc: Sequence[bool] = (True, True, True),
) -> float:
    """Return the shortest non-zero lattice-vector length.

    The smallest basis-vector norm is not invariant under a change of lattice
    basis and can overestimate the unique-image RDF radius in a skew cell.  A
    singular-value bound makes the finite integer search exhaustive.
    """
    h = np.asarray(cell, dtype=float)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ValueError("cell must be a finite 3x3 lattice matrix")
    periodic = tuple(bool(x) for x in pbc)
    if len(periodic) != 3:
        raise ValueError("pbc must contain exactly three flags")
    periodic_axes = [i for i, enabled in enumerate(periodic) if enabled]
    if not periodic_axes:
        return float("inf")
    periodic_basis = h[np.asarray(periodic_axes, dtype=int), :]
    singular = np.linalg.svd(periodic_basis, compute_uv=False)
    sigma_min = float(np.min(singular))
    basis_upper = float(min(np.linalg.norm(h[i]) for i in periodic_axes))
    if not (sigma_min > 0.0 and math.isfinite(basis_upper) and basis_upper > 0.0):
        raise ValueError("cell lattice is singular")
    # ||n H|| >= sigma_min ||n||.  No vector with ||n|| larger than this
    # bound can improve on the shortest basis vector already known.
    bound = int(math.ceil(basis_upper / sigma_min))
    bound = max(1, bound)
    if bound > 64:
        raise ValueError(
            "cell is too ill-conditioned for an exhaustive unique-image RDF cutoff; "
            f"integer search bound={bound}"
        )
    best = basis_upper
    ranges = [
        range(-bound, bound + 1) if periodic[axis] else (0,)
        for axis in range(3)
    ]
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                if i == 0 and j == 0 and k == 0:
                    continue
                n = np.asarray([i, j, k], dtype=float)
                # A candidate whose integer norm cannot beat the current best
                # is eliminated by the same singular-value lower bound.
                if sigma_min * float(np.linalg.norm(n)) > best + 1.0e-12:
                    continue
                length = float(np.linalg.norm(n @ h))
                if length < best:
                    best = length
    return float(best)


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
    if not (math.isfinite(float(r_max)) and float(r_max) > 0.0):
        raise ValueError("r_max must be finite and > 0")
    if isinstance(nbins, bool) or int(nbins) != nbins or int(nbins) < 10:
        raise ValueError("nbins must be >= 10")
    for index, frame in enumerate(frames):
        if not all(frame_pbc(frame)):
            raise ValueError(
                "normalized periodic g(r) currently requires PBC in all three directions; "
                f"frame {index} has pbc={list(frame_pbc(frame))}"
            )

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
    for frame_index, fr in enumerate(frames):
        shortest = _shortest_lattice_translation(fr.cell, pbc=frame_pbc(fr))
        # Exclude the Wigner--Seitz boundary itself: at exactly half a lattice
        # translation, two equidistant periodic images can both be emitted.
        half_lengths.append(float(np.nextafter(0.5 * shortest, 0.0)))
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
    shell_vol = (4.0 * math.pi / 3.0) * (
        edges[1:] ** 3 - edges[:-1] ** 3
    )

    g_frames: list[np.ndarray] = []

    for frame_index, fr in enumerate(frames):
        pbc = frame_pbc(fr)
        invH = np.linalg.inv(fr.cell)
        frac = _wrap_frac((fr.positions - fr.origin) @ invH, pbc=pbc)
        posw = fr.origin + frac @ fr.cell

        # The half-shortest-translation sphere contains at most one periodic
        # image of each unordered atom pair.
        rcut = min(r_max_eff, float(half_lengths[frame_index]))

        atoms = Atoms(numbers=np.ones(fr.n_atoms, dtype=int), positions=posw, cell=fr.cell, pbc=pbc)
        ii, jj = neighbor_list("ij", atoms, rcut)
        ii, jj, _vec, dist = _canonical_unique_mic_pairs(
            frac, fr.cell, ii, jj, cutoff=float(rcut), pbc=pbc
        )

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
            dist = dist[want]

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

        if ii.size:
            dist = dist[np.isfinite(dist)]
            dist = dist[(dist > 1e-8) & (dist <= r_max_eff)]
            radial_counts, _ = np.histogram(dist, bins=edges)
            radial_counts = radial_counts.astype(float)
        else:
            # No observed events is a measured zero RDF for a pair population,
            # not missing data.  Only n_pairs<=0 is mathematically undefined.
            radial_counts = np.zeros_like(centers, dtype=float)

        ideal = (n_pairs / V) * shell_vol

        g = np.full_like(centers, np.nan, dtype=float)
        m2 = ideal > 0
        g[m2] = radial_counts[m2] / ideal[m2]
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
    if not np.all(np.isfinite(r)) or not np.all(np.diff(r) > 0.0):
        raise ValueError("r grid must be finite and strictly increasing")

    # smooth
    w = int(smooth)
    if w < 1:
        w = 1
    # ``numpy.convolve(..., mode='same')`` returns the length of the longer
    # operand.  Cap the window before convolution so an oversized user window
    # cannot produce an array longer than the RDF grid.
    w = min(w, int(r.size))
    if w % 2 == 0:
        w = max(1, w - 1)
    if w > 1:
        ker = np.ones(w, dtype=float)
        finite = np.isfinite(g)
        numerator = np.convolve(np.where(finite, g, 0.0), ker, mode="same")
        denominator = np.convolve(finite.astype(float), ker, mode="same")
        g_s = np.full_like(g, np.nan, dtype=float)
        np.divide(numerator, denominator, out=g_s, where=denominator > 0.0)
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
    prominence = h - baseline
    scale = max(1.0, abs(h), abs(baseline))
    if not (math.isfinite(prominence) and prominence > 32.0 * np.finfo(float).eps * scale):
        return float("nan"), float("nan"), float("nan")
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
