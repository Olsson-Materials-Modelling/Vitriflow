from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Literal

import numpy as np

from .dump import DumpFrame, frame_pbc
from .common import wrap_frac as _wrap_frac


def _replicated_positions_and_radii(
    fr: DumpFrame,
    *,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicated positions and."""
    if fr.n_atoms < 1:
        raise ValueError("frame has no atoms")
    if radii.shape != (fr.n_atoms,):
        raise ValueError("radii must have shape (n_atoms,)")
    _require_fully_periodic_void_frame(fr)

    # In a highly skew unreduced basis, the minimum image can require lattice
    # coefficients outside {-1,0,1}.  Reduce the basis first; in three
    # dimensions the Voronoi-relevant neighbours of a Minkowski-reduced basis
    # are contained in the adjacent 3x3x3 images.
    try:
        from ase.geometry import minkowski_reduce
    except Exception as exc:  # pragma: no cover - ASE is a package dependency
        raise ImportError("periodic void analysis requires ase.geometry.minkowski_reduce") from exc
    reduced_cell, _op = minkowski_reduce(
        np.asarray(fr.cell, dtype=float), pbc=np.asarray(frame_pbc(fr), dtype=bool)
    )
    reduced_cell = np.asarray(reduced_cell, dtype=float)
    invH = np.linalg.inv(reduced_cell)
    frac = _wrap_frac(
        (fr.positions - fr.origin) @ invH, pbc=frame_pbc(fr)
    )
    posw = fr.origin + frac @ reduced_cell

    shifts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    # shift vectors cartesian
    shift_cart = shifts @ reduced_cell
    # broadcast add n
    imgs = (shift_cart[:, None, :] + posw[None, :, :]).reshape((-1, 3))
    rad = np.tile(np.asarray(radii, dtype=float), int(shifts.shape[0]))
    return imgs, rad, reduced_cell


def _points_in_reduced_cell(fr: DumpFrame, points: np.ndarray, reduced_cell: np.ndarray) -> np.ndarray:
    frac = _wrap_frac(
        (np.asarray(points, dtype=float) - fr.origin) @ np.linalg.inv(reduced_cell),
        pbc=frame_pbc(fr),
    )
    return fr.origin + frac @ reduced_cell


def _require_fully_periodic_void_frame(frame: DumpFrame) -> None:
    pbc = frame_pbc(frame)
    if not all(pbc):
        raise ValueError(
            "void clearance sampling currently uses a periodic cell-volume estimator and requires PBC "
            f"in all three directions; received pbc={list(pbc)}. An open-boundary domain/wall model "
            "must be specified before partial-PBC void values are scientifically defined."
        )


def _sample_frac_points(
    n: int,
    *,
    sampler: Literal["sobol", "random", "grid"],
    seed: int,
) -> np.ndarray:
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")

    s = str(sampler).strip().lower()
    if s == "random":
        rng = np.random.default_rng(int(seed))
        return rng.random((n, 3), dtype=float)

    if s == "sobol":
        try:
            from scipy.stats import qmc
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Void analysis with sampler='sobol' requires scipy (scipy.stats.qmc). "
                "Install scipy or set voids.sampler='random'."
            ) from e

        # seed
        # warn
        eng = qmc.Sobol(d=3, scramble=True, seed=int(seed))
        # random returns points
        m = int(math.ceil(math.log2(float(n)))) if int(n) > 1 else 0
        pts = eng.random_base2(m)
        if pts.shape[0] > int(n):
            pts = pts[: int(n), :]
        # strictly artifacts wrapping
        eps = np.finfo(float).eps
        pts = np.clip(pts, 0.0, 1.0 - eps)
        return np.asarray(pts, dtype=float)

    if s == "grid":
        # Construct a containing cubic grid and choose exactly n sites spread
        # across its full flattened extent.  Rounding the cube root downward
        # returned fewer samples than requested (e.g. 10 -> 8).
        ngrid = int(math.ceil(n ** (1.0 / 3.0)))
        ngrid = max(1, ngrid)
        # cell centered boundaries
        ax = (np.arange(ngrid, dtype=float) + 0.5) / float(ngrid)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        if pts.shape[0] > n:
            # Mid-stratum ranks avoid concentrating a truncated grid at one
            # end of the deterministic mesh ordering.
            ranks = np.floor(
                (np.arange(n, dtype=float) + 0.5)
                * float(pts.shape[0])
                / float(n)
            ).astype(int)
            pts = pts[ranks, :]
        return np.asarray(pts, dtype=float)

    raise ValueError(f"Unknown sampler: {sampler}")


def _clearance_from_tree_exact(
    tree: Any,
    points: np.ndarray,
    image_radii: np.ndarray,
    *,
    initial_k: int,
) -> np.ndarray:
    """Exact ``min(distance - radius)`` over the replicated candidate set.

    Nearest centres are not necessarily nearest surfaces when radii differ.
    The search expands until the best current clearance is no larger than the
    lower bound ``next_center_distance - max_radius`` for every unvisited
    centre, which proves that omitted centres cannot improve the result.
    """
    pts = np.asarray(points, dtype=float)
    radii = np.asarray(image_radii, dtype=float).reshape(-1)
    n_centres = int(radii.size)
    if n_centres < 1:
        return np.full((pts.shape[0],), float("nan"), dtype=float)
    k = min(max(1, int(initial_k)), n_centres)
    max_radius = float(np.max(radii))
    while True:
        # One additional neighbour supplies a lower bound for everything not
        # included in the candidate minimum.
        query_k = min(n_centres, k + 1)
        dist, idx = tree.query(pts, k=query_k)
        dist_arr = np.asarray(dist, dtype=float)
        idx_arr = np.asarray(idx, dtype=int)
        if query_k == 1:
            dist_arr = dist_arr.reshape((-1, 1))
            idx_arr = idx_arr.reshape((-1, 1))
        candidate = np.min(
            dist_arr[:, :k] - radii[idx_arr[:, :k]], axis=1
        )
        if k >= n_centres:
            return np.asarray(candidate, dtype=float)
        lower_unvisited = dist_arr[:, k] - max_radius
        if np.all(candidate <= lower_unvisited + 1.0e-14):
            return np.asarray(candidate, dtype=float)
        k = min(n_centres, max(k + 1, 2 * k))


def _radii_from_species(
    fr: DumpFrame,
    *,
    type_to_species: Optional[Sequence[str]],
    radii_by_species: Mapping[str, float],
    default_radius: float,
) -> np.ndarray:
    r0 = float(default_radius)
    if not (math.isfinite(r0) and r0 >= 0.0):
        raise ValueError("default_radius must be >= 0")

    if type_to_species is None:
        return np.full((fr.n_atoms,), r0, dtype=float)

    # lammps based
    t2s = list(type_to_species)
    out = np.full((fr.n_atoms,), r0, dtype=float)
    for i, t in enumerate(fr.types.tolist()):
        ti = int(t)
        if ti < 1 or ti > len(t2s):
            continue
        sp = str(t2s[ti - 1])
        v = radii_by_species.get(sp, r0)
        try:
            vf = float(v)
        except Exception:
            vf = r0
        if math.isfinite(vf) and vf >= 0.0:
            out[i] = vf
    return out


def sample_void_clearance_radii(
    frames: Sequence[DumpFrame],
    *,
    n_samples: int,
    sampler: Literal["sobol", "random", "grid"] = "sobol",
    seed: int = 0,
    k_nearest: int = 16,
    type_to_species: Optional[Sequence[str]] = None,
    radii_by_species: Optional[Mapping[str, float]] = None,
    default_radius: float = 0.0,
) -> np.ndarray:
    """Sample void clearance."""
    if not frames:
        raise ValueError("frames must be non-empty")
    n_total = int(n_samples)
    if n_total < 1:
        raise ValueError("n_samples must be >= 1")
    k = int(k_nearest)
    if k < 1:
        raise ValueError("k_nearest must be >= 1")

    rb = dict(radii_by_species or {})

    n_frames = int(len(frames))
    # Exact balanced allocation.  The previous ceil-and-truncate scheme
    # overweighted early frames and could leave the final frame with a much
    # smaller contribution.
    base, remainder = divmod(n_total, n_frames)
    extra_frames: set[int] = set()
    if remainder:
        extra_frames = {
            min(
                n_frames - 1,
                int(math.floor((rank + 0.5) * n_frames / remainder)),
            )
            for rank in range(remainder)
        }
    samples_per_frame = [
        int(base + (1 if fi in extra_frames else 0)) for fi in range(n_frames)
    ]

    try:
        from scipy.spatial import cKDTree
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Void analysis requires scipy (scipy.spatial.cKDTree). Install scipy to enable void metrics."
        ) from e

    out: list[np.ndarray] = []

    # seed
    base_seed = int(seed)
    for fi, fr in enumerate(frames):
        n_frame = int(samples_per_frame[fi])
        if n_frame == 0:
            continue
        _require_fully_periodic_void_frame(fr)
        if fr.n_atoms < 1:
            continue

        rad0 = _radii_from_species(
            fr,
            type_to_species=type_to_species,
            radii_by_species=rb,
            default_radius=float(default_radius),
        )

        pos_img, rad_img, reduced_cell = _replicated_positions_and_radii(fr, radii=rad0)
        tree = cKDTree(pos_img)

        pts_frac = _sample_frac_points(n_frame, sampler=sampler, seed=base_seed + 104729 * fi)
        pts = fr.origin + pts_frac @ fr.cell
        pts_query = _points_in_reduced_cell(fr, pts, reduced_cell)

        clear = _clearance_from_tree_exact(
            tree, pts_query, rad_img, initial_k=int(k)
        )

        rad = np.maximum(clear, 0.0)
        out.append(rad)

    if not out:
        return np.asarray([], dtype=float)

    arr = np.concatenate(out, axis=0)
    if arr.size != n_total:
        raise RuntimeError(
            f"void sampler produced {arr.size} values for requested n_samples={n_total}"
        )
    return np.asarray(arr, dtype=float)


def clearance_cdf_on_grid(
    radii: Sequence[float],
    *,
    r_max: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Clearance cdf on."""
    if int(n_points) < 2:
        raise ValueError("n_points must be >= 2")
    if not (math.isfinite(float(r_max)) and float(r_max) > 0.0):
        raise ValueError("r_max must be > 0")

    x = np.linspace(0.0, float(r_max), int(n_points), dtype=float)
    arr = np.asarray([float(v) for v in radii], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return x, np.full_like(x, float("nan"), dtype=float)

    arr = np.sort(arr)
    idx = np.searchsorted(arr, x, side="right")
    cdf = idx.astype(float) / float(arr.size)
    return x, np.asarray(cdf, dtype=float)


def clearance_scalar_metrics(
    radii: Sequence[float],
    *,
    probe_radii: Optional[Sequence[float]] = None,
) -> dict[str, float]:
    """Clearance scalar metrics."""
    arr = np.asarray([float(v) for v in radii], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    out: dict[str, float] = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }

    if probe_radii is not None:
        for rp in probe_radii:
            rpf = float(rp)
            if not (math.isfinite(rpf) and rpf >= 0.0):
                continue
            out[f"frac_ge_{rpf}"] = float(np.mean(arr >= rpf))

    return out


def sample_void_clearance_points(
    frame: DumpFrame,
    *,
    n_samples: int,
    sampler: Literal["sobol", "random", "grid"] = "sobol",
    seed: int = 0,
    k_nearest: int = 16,
    type_to_species: Optional[Sequence[str]] = None,
    radii_by_species: Optional[Mapping[str, float]] = None,
    default_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample void clearance."""

    if not isinstance(frame, DumpFrame):
        raise TypeError("frame must be a DumpFrame")
    _require_fully_periodic_void_frame(frame)

    n = int(n_samples)
    if n < 1:
        raise ValueError("n_samples must be >= 1")

    k = int(k_nearest)
    if k < 1:
        raise ValueError("k_nearest must be >= 1")

    rb = dict(radii_by_species or {})

    try:
        from scipy.spatial import cKDTree
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Void visualisation requires scipy (scipy.spatial.cKDTree). Install scipy to enable void plotting."
        ) from e

    if frame.n_atoms < 1:
        pts_frac = _sample_frac_points(n, sampler=sampler, seed=int(seed))
        pts = frame.origin + pts_frac @ frame.cell
        return np.asarray(pts, dtype=float), np.full((n,), float("nan"), dtype=float)

    rad0 = _radii_from_species(
        frame,
        type_to_species=type_to_species,
        radii_by_species=rb,
        default_radius=float(default_radius),
    )

    pos_img, rad_img, reduced_cell = _replicated_positions_and_radii(frame, radii=rad0)
    tree = cKDTree(pos_img)

    pts_frac = _sample_frac_points(n, sampler=sampler, seed=int(seed))
    pts = frame.origin + pts_frac @ frame.cell
    pts_query = _points_in_reduced_cell(frame, pts, reduced_cell)

    clear = _clearance_from_tree_exact(
        tree, pts_query, rad_img, initial_k=int(k)
    )

    clearance = np.maximum(np.asarray(clear, dtype=float), 0.0)
    return np.asarray(pts, dtype=float), np.asarray(clearance, dtype=float)
