from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Literal

import numpy as np

from .dump import DumpFrame
from .common import wrap_frac as _wrap_frac


def _replicated_positions_and_radii(
    fr: DumpFrame,
    *,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicated positions and."""
    if fr.n_atoms < 1:
        raise ValueError("frame has no atoms")
    if radii.shape != (fr.n_atoms,):
        raise ValueError("radii must have shape (n_atoms,)")

    invH = np.linalg.inv(fr.cell)
    frac = _wrap_frac((fr.positions - fr.origin) @ invH)
    posw = fr.origin + frac @ fr.cell

    shifts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    # shift vectors cartesian
    shift_cart = shifts @ fr.cell
    # broadcast add n
    imgs = (shift_cart[:, None, :] + posw[None, :, :]).reshape((-1, 3))
    rad = np.tile(np.asarray(radii, dtype=float), int(shifts.shape[0]))
    return imgs, rad


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
        # approximate cubic deterministic
        ngrid = int(round(n ** (1.0 / 3.0)))
        ngrid = max(1, ngrid)
        # cell centered boundaries
        ax = (np.arange(ngrid, dtype=float) + 0.5) / float(ngrid)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        if pts.shape[0] > n:
            pts = pts[:n, :]
        return np.asarray(pts, dtype=float)

    raise ValueError(f"Unknown sampler: {sampler}")


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
    # distribute samples frames
    n_per = int(math.ceil(float(n_total) / float(n_frames)))

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
        if fr.n_atoms < 1:
            continue

        rad0 = _radii_from_species(
            fr,
            type_to_species=type_to_species,
            radii_by_species=rb,
            default_radius=float(default_radius),
        )

        pos_img, rad_img = _replicated_positions_and_radii(fr, radii=rad0)
        tree = cKDTree(pos_img)

        pts_frac = _sample_frac_points(n_per, sampler=sampler, seed=base_seed + 104729 * fi)
        pts = fr.origin + pts_frac @ fr.cell

        k_use = min(int(k), int(pos_img.shape[0]))
        dist, idx = tree.query(pts, k=k_use)

        if k_use == 1:
            # dist n idx
            clear = np.asarray(dist, dtype=float) - rad_img[np.asarray(idx, dtype=int)]
        else:
            dist = np.asarray(dist, dtype=float)
            idx = np.asarray(idx, dtype=int)
            # dist idx n
            rad_sel = rad_img[idx]
            clear = np.min(dist - rad_sel, axis=1)

        rad = np.maximum(clear, 0.0)
        out.append(rad)

    if not out:
        return np.asarray([], dtype=float)

    arr = np.concatenate(out, axis=0)
    if arr.size > n_total:
        arr = arr[:n_total]
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

    pos_img, rad_img = _replicated_positions_and_radii(frame, radii=rad0)
    tree = cKDTree(pos_img)

    pts_frac = _sample_frac_points(n, sampler=sampler, seed=int(seed))
    pts = frame.origin + pts_frac @ frame.cell

    k_use = min(int(k), int(pos_img.shape[0]))
    dist, idx = tree.query(pts, k=k_use)

    if k_use == 1:
        clear = np.asarray(dist, dtype=float) - rad_img[np.asarray(idx, dtype=int)]
    else:
        dist = np.asarray(dist, dtype=float)
        idx = np.asarray(idx, dtype=int)
        clear = np.min(dist - rad_img[idx], axis=1)

    clearance = np.maximum(np.asarray(clear, dtype=float), 0.0)
    return np.asarray(pts, dtype=float), np.asarray(clearance, dtype=float)
