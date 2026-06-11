from __future__ import annotations

"""Shared analysis helpers.

These helpers are used by multiple analysis modules (e.g. g(r) and structure
metrics) to avoid drift in selector resolution and periodic-boundary handling.
"""

from typing import Optional, Sequence, Tuple, List

import numpy as np

from ..config import AtomSelector


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


def wrap_frac(frac: np.ndarray) -> np.ndarray:
    """Wrap frac."""

    frac = np.asarray(frac, dtype=float)
    return frac - np.floor(frac)


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


def mic_displacements(frac: np.ndarray, cell: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Mic displacements."""

    frac = np.asarray(frac, dtype=float)
    cell = np.asarray(cell, dtype=float)
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)

    df = frac[j] - frac[i]

    # fast orthogonal cells
    if _cell_is_orthogonal(cell):
        df -= np.round(df)
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
    out = find_mic(dr0, cell, pbc=True)
    # ase defensively signature
    if isinstance(out, tuple) and len(out) == 2:
        dr_mic = np.asarray(out[0], dtype=float)
        return dr_mic
    return np.asarray(out, dtype=float)


def mic_distances(frac: np.ndarray, cell: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Mic distances."""

    dr, dist = mic_displacements_and_distances(frac, cell, i, j)
    return dist


def mic_displacements_and_distances(
    frac: np.ndarray, cell: np.ndarray, i: np.ndarray, j: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Mic displacements and."""

    frac = np.asarray(frac, dtype=float)
    cell = np.asarray(cell, dtype=float)
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)

    df = frac[j] - frac[i]

    if _cell_is_orthogonal(cell):
        df -= np.round(df)
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
    out = find_mic(dr0, cell, pbc=True)
    if isinstance(out, tuple) and len(out) == 2:
        dr_mic = np.asarray(out[0], dtype=float)
        dist = np.asarray(out[1], dtype=float)
        # distances downstream code
        dist = dist.reshape((-1,))
        return dr_mic, dist

    dr_mic = np.asarray(out, dtype=float)
    return dr_mic, np.linalg.norm(dr_mic, axis=1)
