from __future__ import annotations

"""Shared analysis helpers.

These helpers are used by multiple analysis modules (e.g. g(r) and structure
metrics) to avoid drift in selector resolution and periodic-boundary handling.
"""

from typing import Optional, Sequence, Tuple, List

import numpy as np

from ..config import AtomSelector
from .dump import normalize_pbc


def resolve_selector(selector: AtomSelector, type_to_species: Optional[Sequence[str]]) -> List[int]:
    """Selector."""

    if isinstance(selector, int):
        if selector < 1:
            raise ValueError(f"Type indices are 1-based; got {selector}")
        return [int(selector)]

    if type_to_species is None:
        raise ValueError(
            f"Selector '{selector}' is a string but no type_to_species mapping is available. "
            "Provide autotune.metrics.type_to_species or use integer type selectors."
        )

    out = [i + 1 for i, sp in enumerate(type_to_species) if sp == selector]
    if not out:
        raise ValueError(f"Selector '{selector}' not found in type_to_species mapping: {type_to_species}")
    return out


def wrap_frac(
    frac: np.ndarray,
    pbc: Sequence[bool] | bool = (True, True, True),
) -> np.ndarray:
    """Wrap fractional coordinates only along periodic lattice directions."""

    out = np.asarray(frac, dtype=float).copy()
    periodic = np.asarray(normalize_pbc(pbc), dtype=bool)
    out[..., periodic] -= np.floor(out[..., periodic])
    return out


def _cell_is_orthogonal(cell: np.ndarray, *, rtol: float = 1.0e-10, atol: float = 1.0e-12) -> bool:
    """Cell is orthogonal."""

    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        return False
    a = cell[0]
    b = cell[1]
    c = cell[2]
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    lc = float(np.linalg.norm(c))
    if not (la > 0.0 and lb > 0.0 and lc > 0.0):
        return False

    # relative orthogonality tests
    ab = float(np.dot(a, b))
    ac = float(np.dot(a, c))
    bc = float(np.dot(b, c))
    if abs(ab) > float(atol) + float(rtol) * la * lb:
        return False
    if abs(ac) > float(atol) + float(rtol) * la * lc:
        return False
    if abs(bc) > float(atol) + float(rtol) * lb * lc:
        return False
    return True


def mic_displacements(
    frac: np.ndarray,
    cell: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    *,
    pbc: Sequence[bool] | bool = (True, True, True),
) -> np.ndarray:
    """Mic displacements."""

    frac = np.asarray(frac, dtype=float)
    cell = np.asarray(cell, dtype=float)
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    periodic = np.asarray(normalize_pbc(pbc), dtype=bool)

    df = frac[j] - frac[i]

    # fast orthogonal cells
    if _cell_is_orthogonal(cell):
        df[:, periodic] -= np.round(df[:, periodic])
        return df @ cell

    # triclinic orthogonal ase
    try:
        try:
            from ase.geometry import find_mic  # type: ignore
        except Exception:  # pragma: no cover
            from ase.geometry.geometry import find_mic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Non-orthogonal periodic cells require ASE for MIC handling. "
            "Install 'ase' or restrict simulations/analysis to orthogonal cells."
        ) from e

    dr0 = df @ cell
    out = find_mic(dr0, cell, pbc=periodic.tolist())
    # ase defensively signature
    if isinstance(out, tuple) and len(out) == 2:
        dr_mic = np.asarray(out[0], dtype=float)
        return dr_mic
    return np.asarray(out, dtype=float)


def mic_distances(
    frac: np.ndarray,
    cell: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    *,
    pbc: Sequence[bool] | bool = (True, True, True),
) -> np.ndarray:
    """Mic distances."""

    dr, dist = mic_displacements_and_distances(frac, cell, i, j, pbc=pbc)
    return dist


def mic_displacements_and_distances(
    frac: np.ndarray,
    cell: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    *,
    pbc: Sequence[bool] | bool = (True, True, True),
) -> Tuple[np.ndarray, np.ndarray]:
    """Mic displacements and."""

    frac = np.asarray(frac, dtype=float)
    cell = np.asarray(cell, dtype=float)
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    periodic = np.asarray(normalize_pbc(pbc), dtype=bool)

    df = frac[j] - frac[i]

    if _cell_is_orthogonal(cell):
        df[:, periodic] -= np.round(df[:, periodic])
        dr = df @ cell
        return dr, np.linalg.norm(dr, axis=1)

    try:
        try:
            from ase.geometry import find_mic  # type: ignore
        except Exception:  # pragma: no cover
            from ase.geometry.geometry import find_mic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Non-orthogonal periodic cells require ASE for MIC handling. "
            "Install 'ase' or restrict simulations/analysis to orthogonal cells."
        ) from e

    dr0 = df @ cell
    out = find_mic(dr0, cell, pbc=periodic.tolist())
    if isinstance(out, tuple) and len(out) == 2:
        dr_mic = np.asarray(out[0], dtype=float)
        dist = np.asarray(out[1], dtype=float)
        # distances downstream code
        dist = dist.reshape((-1,))
        return dr_mic, dist

    dr_mic = np.asarray(out, dtype=float)
    return dr_mic, np.linalg.norm(dr_mic, axis=1)


def canonical_unique_mic_pairs(
    frac: np.ndarray,
    cell: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    *,
    cutoff: float,
    pbc: Sequence[bool] | bool = (True, True, True),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalise a periodic neighbour-list result by atom identity.

    ASE enumerates periodic *images*.  Once a search radius reaches beyond a
    cell's unique-image sphere, the same unordered atom pair (and even images
    of an atom with itself) can therefore occur multiple times.  Analysis
    descriptors in Vitriflow count distinct atoms, not image entries.  This
    helper first reduces candidates to one lexicographically ordered ``(i,j)``
    with ``i < j``, then recomputes one deterministic minimum-image vector and
    applies the physical cutoff to that vector.

    Canonical atom ordering also fixes the orientation used in exact
    half-lattice ties.  Their distance is well defined even though two image
    vectors are equivalent; no result depends on ASE's image enumeration.
    """

    radius = float(cutoff)
    if not (np.isfinite(radius) and radius > 0.0):
        raise ValueError("cutoff must be finite and > 0")
    ii = np.asarray(i, dtype=int).reshape(-1)
    jj = np.asarray(j, dtype=int).reshape(-1)
    if ii.size != jj.size:
        raise ValueError("neighbour-list i/j arrays must have equal length")
    if ii.size == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.zeros((0, 3), dtype=float),
            np.asarray([], dtype=float),
        )

    lo = np.minimum(ii, jj)
    hi = np.maximum(ii, jj)
    distinct = lo < hi
    lo = lo[distinct]
    hi = hi[distinct]
    if lo.size == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.zeros((0, 3), dtype=float),
            np.asarray([], dtype=float),
        )

    # np.unique over a two-column array provides a deterministic
    # lexicographic order and removes both reverse-direction and periodic-image
    # duplicates without relying on a potentially overflowing scalar key.
    pairs = np.unique(np.column_stack((lo, hi)), axis=0)
    ii_u = np.asarray(pairs[:, 0], dtype=int)
    jj_u = np.asarray(pairs[:, 1], dtype=int)
    vec, dist = mic_displacements_and_distances(
        frac,
        cell,
        ii_u,
        jj_u,
        pbc=pbc,
    )
    vec = np.asarray(vec, dtype=float).reshape((-1, 3))
    dist = np.asarray(dist, dtype=float).reshape(-1)
    keep = np.isfinite(dist) & (dist <= radius)
    return ii_u[keep], jj_u[keep], vec[keep], dist[keep]
