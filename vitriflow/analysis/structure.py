from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import re

try:
    from ase import Atoms
    from ase.neighborlist import neighbor_list
except Exception as e:  # pragma: no cover
    raise ImportError(
        "vitriflow.analysis.structure requires 'ase'. Install via pip/conda: pip install ase"
    ) from e

try:
    import networkx as nx
except Exception as e:  # pragma: no cover
    raise ImportError(
        "vitriflow.analysis.structure requires 'networkx'. Install via pip/conda: pip install networkx"
    ) from e

from ..config import (
    AngleMetricConfig,
    AtomSelector,
    AutoCutoffConfig,
    CoordinationMetricConfig,
    PairMetricConfig,
    RingMetricsConfig,
    StructureMetricsConfig,
)
from .dump import DumpFrame
from .common import resolve_selector as _resolve_selector, wrap_frac as _wrap_frac, mic_displacements_and_distances as _mic_displacements
from .gr import compute_first_peak_gr


def _pair_key(t1: int, t2: int) -> Tuple[int, int]:
    return (t1, t2) if t1 <= t2 else (t2, t1)



def estimate_pair_cutoffs(
    frames: Sequence[DumpFrame],
    required_pairs: Sequence[Tuple[int, int]],
    *,
    auto: AutoCutoffConfig,
    fixed_cutoffs: Mapping[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """Pair cutoffs."""
    cutoffs: Dict[Tuple[int, int], float] = dict(fixed_cutoffs)

    # precompute fractional neighbor
    for pair in required_pairs:
        key = _pair_key(pair[0], pair[1])
        if key in cutoffs:
            continue

        # gather distances max
        dists_all: List[float] = []
        for fr in frames:
            # wrap coords cell
            invH = np.linalg.inv(fr.cell)
            frac = _wrap_frac((fr.positions - fr.origin) @ invH)
            posw = fr.origin + frac @ fr.cell

            atoms = Atoms(numbers=np.ones(fr.n_atoms, dtype=int), positions=posw, cell=fr.cell, pbc=True)
            ii, jj = neighbor_list("ij", atoms, auto.r_max)
            # unique pairs
            m = ii < jj
            ii = ii[m]
            jj = jj[m]
            # filter
            t_i = fr.types[ii]
            t_j = fr.types[jj]
            want = (t_i == pair[0]) & (t_j == pair[1]) | (t_i == pair[1]) & (t_j == pair[0])
            if not np.any(want):
                continue
            ii = ii[want]
            jj = jj[want]
            _, dist = _mic_displacements(frac, fr.cell, ii, jj)
            dists_all.extend(dist.tolist())

        if len(dists_all) < 10:
            raise ValueError(
                f"Not enough neighbor distances to estimate cutoff for pair {pair} "
                f"(found {len(dists_all)} distances <= {auto.r_max}). Provide an explicit cutoff."
            )

        d = np.asarray(dists_all, dtype=float)
        d = d[np.isfinite(d)]
        d = d[(d > 1e-8) & (d <= auto.r_max)]
        if d.size < 10:
            raise ValueError(f"Insufficient finite distances for cutoff estimation for pair {pair}.")

        # histogram
        edges = np.linspace(0.0, auto.r_max, int(auto.nbins) + 1)
        counts, _ = np.histogram(d, bins=edges)
        counts = counts.astype(float)

        # smooth moving average
        w = int(auto.smooth)
        if w < 1:
            w = 1
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w, dtype=float) / float(w)
        smooth = np.convolve(counts, kernel, mode="same")

        centers = 0.5 * (edges[:-1] + edges[1:])

        # peak search
        p0, p1 = auto.peak_search
        m_peak = (centers >= p0) & (centers <= p1)
        if not np.any(m_peak):
            raise ValueError(f"Invalid peak_search window {auto.peak_search} for r_max={auto.r_max}")
        peak_idx = int(np.argmax(smooth[m_peak]))
        peak_idx = int(np.where(m_peak)[0][0] + peak_idx)

        # minimum search peak
        m0, m1 = auto.min_search
        m_min = (centers >= m0) & (centers <= m1) & (np.arange(centers.size) > peak_idx)
        if np.any(m_min):
            min_rel = int(np.argmin(smooth[m_min]))
            min_idx = int(np.where(m_min)[0][0] + min_rel)
            cutoff = float(centers[min_idx])
            # guard
            cutoff = max(cutoff, float(centers[peak_idx]) * 1.05)
        else:
            cutoff = float(centers[peak_idx]) * float(auto.fallback_factor)

        cutoffs[key] = cutoff

    return cutoffs


@dataclass(frozen=True)
class StructureMetrics:
    """Structure metrics."""

    values: Dict[str, float]


def _cdf_on_grid(samples: Sequence[float], grid: np.ndarray) -> np.ndarray:
    """Cdf on grid."""
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1 or g.size < 2:
        raise ValueError("grid must be a 1D array with >=2 points")
    if not np.all(np.diff(g) >= 0):
        raise ValueError("grid must be non-decreasing")
    arr = np.asarray([float(x) for x in samples], dtype=float)
    m = np.isfinite(arr)
    if int(np.sum(m)) == 0:
        return np.full_like(g, np.nan, dtype=float)
    x = np.sort(arr[m])
    idx = np.searchsorted(x, g, side="right")
    return idx.astype(float) / float(x.size)


def compute_structure_distributions_timeavg(
    frames: Sequence[DumpFrame],
    metrics: StructureMetricsConfig,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]] = None,
    bondlen_cdf_points: int = 200,
    angle_cdf_points: int = 180,
) -> Dict[str, Dict[str, Dict[str, list[float]]]]:
    """Structure distributions timeavg."""
    if not frames:
        raise ValueError("compute_structure_distributions_timeavg requires at least one frame")

    if not metrics.enabled:
        return {"bondlen": {}, "angle": {}, "coord": {}, "void": {}}

    if int(bondlen_cdf_points) < 10:
        raise ValueError("bondlen_cdf_points must be >= 10")
    if int(angle_cdf_points) < 10:
        raise ValueError("angle_cdf_points must be >= 10")

    out: Dict[str, Dict[str, Dict[str, list[float]]]] = {"bondlen": {}, "angle": {}, "coord": {}, "void": {}}

    # volume distribution cutoffs
    void_cfg = getattr(metrics, "voids", None)
    if void_cfg is not None and bool(getattr(void_cfg, "enabled", False)):
        # analysis requested unavailable
        from .voids import sample_void_clearance_radii, clearance_cdf_on_grid

        radii = sample_void_clearance_radii(
            frames,
            n_samples=int(getattr(void_cfg, "n_samples", 0) or 0),
            sampler=str(getattr(void_cfg, "sampler", "sobol")),
            seed=int(getattr(void_cfg, "seed", 0) or 0),
            k_nearest=int(getattr(void_cfg, "k_nearest", 16) or 16),
            type_to_species=type_to_species,
            radii_by_species=dict(getattr(void_cfg, "radii", {}) or {}),
            default_radius=float(getattr(void_cfg, "default_radius", 0.0) or 0.0),
        )
        x, cdf = clearance_cdf_on_grid(
            radii,
            r_max=float(getattr(void_cfg, "r_max", 5.0) or 5.0),
            n_points=int(getattr(void_cfg, "cdf_points", 200) or 200),
        )
        out["void"]["void_clearance"] = {
            "x": [float(v) for v in x.tolist()],
            "cdf": [float(v) for v in cdf.tolist()],
        }

    # remaining distributions cutoffs
    if not cutoffs:
        return out

    # pre selectors grids
    pair_defs: list[tuple[str, list[int], list[int], float]] = []
    for pm in metrics.pairs:
        a_sel, b_sel = pm.pair
        a_types = _resolve_selector(a_sel, type_to_species)
        b_types = _resolve_selector(b_sel, type_to_species)
        # cutoff included pairs
        rmax = float("nan")
        cmax = -float("inf")
        for ta in a_types:
            for tb in b_types:
                key = _pair_key(int(ta), int(tb))
                c = cutoffs.get(key, None)
                if c is None:
                    continue
                cmax = max(cmax, float(c))
        if math.isfinite(cmax) and cmax > 0:
            rmax = float(cmax)
        name = f"bondlen_{a_sel}-{b_sel}"
        pair_defs.append((name, a_types, b_types, rmax))

    coord_defs: list[tuple[str, list[int], list[int]]] = []
    for cm in metrics.coordinations:
        c_types = _resolve_selector(cm.central, type_to_species)
        n_types = _resolve_selector(cm.neighbor, type_to_species)
        name = f"coord_{cm.central}-{cm.neighbor}"
        coord_defs.append((name, c_types, n_types))

    angle_defs: list[tuple[str, str, str, list[int], list[int], list[int]]] = []
    for am in metrics.angles:
        a_sel, b_sel, c_sel = am.triplet
        a_types = _resolve_selector(a_sel, type_to_species)
        b_types = _resolve_selector(b_sel, type_to_species)
        c_types = _resolve_selector(c_sel, type_to_species)
        name = f"angle_{a_sel}-{b_sel}-{c_sel}"
        angle_defs.append((name, a_sel, c_sel, a_types, b_types, c_types))

    # accumulate pooled samples
    bond_samples: Dict[str, List[float]] = {name: [] for (name, *_rest) in pair_defs}
    coord_samples: Dict[str, List[int]] = {name: [] for (name, *_rest) in coord_defs}
    angle_samples: Dict[str, List[float]] = {name: [] for (name, *_rest) in angle_defs}

    max_cut = float(max(cutoffs.values()))

    for fr in frames:
        invH = np.linalg.inv(fr.cell)
        frac = _wrap_frac((fr.positions - fr.origin) @ invH)
        posw = fr.origin + frac @ fr.cell

        atoms = Atoms(numbers=np.ones(fr.n_atoms, dtype=int), positions=posw, cell=fr.cell, pbc=True)
        ii, jj = neighbor_list("ij", atoms, max_cut)
        m = ii < jj
        ii = ii[m]
        jj = jj[m]
        if ii.size == 0:
            continue

        dr, dist = _mic_displacements(frac, fr.cell, ii, jj)
        t_i = fr.types[ii]
        t_j = fr.types[jj]

        # neighbor pair cutoffs
        nbr_ids: List[List[int]] = [[] for _ in range(fr.n_atoms)]
        nbr_vecs: List[List[np.ndarray]] = [[] for _ in range(fr.n_atoms)]
        nbr_dists: List[List[float]] = [[] for _ in range(fr.n_atoms)]
        dist_by_pair: Dict[Tuple[int, int], List[float]] = {}

        for a, b, vec, d, ta, tb in zip(ii.tolist(), jj.tolist(), dr.tolist(), dist.tolist(), t_i.tolist(), t_j.tolist()):
            key = _pair_key(int(ta), int(tb))
            c = cutoffs.get(key, None)
            if c is None or float(d) > float(c):
                continue
            # add directed
            nbr_ids[a].append(b)
            nbr_vecs[a].append(np.asarray(vec, dtype=float))
            nbr_dists[a].append(float(d))
            nbr_ids[b].append(a)
            nbr_vecs[b].append(-np.asarray(vec, dtype=float))
            nbr_dists[b].append(float(d))
            dist_by_pair.setdefault(key, []).append(float(d))

        # bond length samples
        for (name, a_types, b_types, _rmax) in pair_defs:
            ds: List[float] = []
            for ta in a_types:
                for tb in b_types:
                    ds.extend(dist_by_pair.get(_pair_key(int(ta), int(tb)), []))
            if ds:
                bond_samples[name].extend(ds)

        # coordination samples
        for (name, c_types, n_types) in coord_defs:
            counts: List[int] = []
            nset = set(int(x) for x in n_types)
            cset = set(int(x) for x in c_types)
            for idx in range(fr.n_atoms):
                if int(fr.types[idx]) not in cset:
                    continue
                k = 0
                for nb in nbr_ids[idx]:
                    if int(fr.types[nb]) in nset:
                        k += 1
                counts.append(int(k))
            if counts:
                coord_samples[name].extend(counts)

        # angle samples
        for (name, a_sel, c_sel, a_types, b_types, c_types) in angle_defs:
            aset = set(int(x) for x in a_types)
            bset = set(int(x) for x in b_types)
            cset = set(int(x) for x in c_types)
            angles: List[float] = []
            same_ac = (aset == cset) and (len(aset) > 0) and (len(cset) > 0) and (a_sel == c_sel)
            for b_idx in range(fr.n_atoms):
                if int(fr.types[b_idx]) not in bset:
                    continue
                neighA: List[np.ndarray] = []
                neighA_idx: List[int] = []
                neighC: List[np.ndarray] = []
                neighC_idx: List[int] = []
                for nb, vec in zip(nbr_ids[b_idx], nbr_vecs[b_idx]):
                    tnb = int(fr.types[nb])
                    if tnb in aset:
                        neighA.append(np.asarray(vec, dtype=float))
                        neighA_idx.append(int(nb))
                    if tnb in cset:
                        neighC.append(np.asarray(vec, dtype=float))
                        neighC_idx.append(int(nb))
                if not neighA or not neighC:
                    continue
                if same_ac:
                    # unique unordered pairs
                    for p, q in combinations(range(len(neighA)), 2):
                        v = neighA[p]
                        w2 = neighA[q]
                        denom = float(np.linalg.norm(v) * np.linalg.norm(w2))
                        if denom <= 0:
                            continue
                        cos = float(np.dot(v, w2) / denom)
                        cos = max(-1.0, min(1.0, cos))
                        angles.append(float(math.degrees(math.acos(cos))))
                else:
                    for v, idxv in zip(neighA, neighA_idx):
                        for w2, idxw in zip(neighC, neighC_idx):
                            if idxv == idxw:
                                continue
                            denom = float(np.linalg.norm(v) * np.linalg.norm(w2))
                            if denom <= 0:
                                continue
                            cos = float(np.dot(v, w2) / denom)
                            cos = max(-1.0, min(1.0, cos))
                            angles.append(float(math.degrees(math.acos(cos))))
            if angles:
                angle_samples[name].extend(angles)

    # finalise bond length
    for (name, _a_types, _b_types, rmax) in pair_defs:
        if not math.isfinite(float(rmax)) or float(rmax) <= 0:
            # cutoff selector pair
            out["bondlen"][name] = {"x": [], "cdf": []}
            continue
        x = np.linspace(0.0, float(rmax), int(bondlen_cdf_points), dtype=float)
        cdf = _cdf_on_grid(bond_samples.get(name, []), x)
        out["bondlen"][name] = {"x": [float(v) for v in x.tolist()], "cdf": [float(v) for v in cdf.tolist()]}

    # angle cdfs
    if angle_defs:
        xang = np.linspace(0.0, 180.0, int(angle_cdf_points), dtype=float)
        for (name, *_rest) in angle_defs:
            cdf = _cdf_on_grid(angle_samples.get(name, []), xang)
            out["angle"][name] = {"x": [float(v) for v in xang.tolist()], "cdf": [float(v) for v in cdf.tolist()]}

    # coordination observed metric
    for (name, *_rest) in coord_defs:
        counts = np.asarray([int(v) for v in coord_samples.get(name, [])], dtype=int)
        if counts.size == 0:
            out["coord"][name] = {"x": [], "cdf": []}
            continue
        kmax = int(np.max(counts))
        xk = np.arange(0, kmax + 1, dtype=float)
        # cdf integer count
        cdf = np.zeros_like(xk, dtype=float)
        for i, k in enumerate(xk.astype(int).tolist()):
            cdf[i] = float(np.mean(counts <= int(k)))
        out["coord"][name] = {"x": [float(v) for v in xk.tolist()], "cdf": [float(v) for v in cdf.tolist()]}

    return out


def compute_structure_metrics_timeavg(
    frames: Sequence[DumpFrame],
    metrics: StructureMetricsConfig,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]] = None,
) -> StructureMetrics:
    """Structure metrics timeavg."""
    if not frames:
        raise ValueError("compute_structure_metrics_timeavg requires at least one frame")
    if not metrics.enabled:
        return StructureMetrics(values={})

    per: list[dict[str, float]] = []
    for fr in frames:
        sm = compute_structure_metrics(fr, metrics, cutoffs=cutoffs, type_to_species=type_to_species)
        per.append(dict(sm.values))

    keys: set[str] = set()
    for d in per:
        keys.update(d.keys())

    out: dict[str, float] = {}
    for k in sorted(keys):
        arr = np.asarray([float(d.get(k, float("nan"))) for d in per], dtype=float)
        m = np.isfinite(arr)
        if int(np.sum(m)) == 0:
            out[k] = float("nan")
        else:
            out[k] = float(np.mean(arr[m]))

    # peak descriptors averaged
    gr_cfgs = getattr(metrics, 'gr', [])
    if gr_cfgs:
        def _slug(s: str) -> str:
            return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

        for gm in gr_cfgs:
            pair = getattr(gm, 'pair', None)
            label = 'all' if pair is None else f"{pair[0]}-{pair[1]}"
            feat = compute_first_peak_gr(
                frames,
                r_max=float(getattr(gm, 'r_max', 8.0)),
                nbins=int(getattr(gm, 'nbins', 400)),
                smooth=int(getattr(gm, 'smooth', 7)),
                pair=pair,
                type_to_species=type_to_species,
            )
            pref = f"gr_{_slug(label)}"
            out[f"{pref}_peak_r"] = float(feat.peak_r)
            out[f"{pref}_peak_height"] = float(feat.peak_height)
            out[f"{pref}_peak_fwhm"] = float(feat.peak_fwhm)



    # peak descriptors averaged
    sq_cfgs = getattr(metrics, 'sq', [])
    if sq_cfgs:
        from .sq import compute_first_peak_sq

        def _slug(s: str) -> str:
            return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

        for sm in sq_cfgs:
            pair = getattr(sm, 'pair', None)
            label = 'all' if pair is None else f"{pair[0]}-{pair[1]}"
            feat = compute_first_peak_sq(
                frames,
                q_max=float(getattr(sm, 'q_max', 20.0)),
                nq=int(getattr(sm, 'nq', 400)),
                r_max=float(getattr(sm, 'r_max', 10.0)),
                nbins=int(getattr(sm, 'nbins', 800)),
                smooth=int(getattr(sm, 'smooth', 7)),
                peak_search=tuple(getattr(sm, 'peak_search', (0.5, 3.0))),
                pair=pair,
                type_to_species=type_to_species,
                window=str(getattr(sm, 'window', 'lorch')),
            )
            pref = f"sq_{_slug(label)}"
            out[f"{pref}_peak_q"] = float(feat.peak_q)
            out[f"{pref}_peak_height"] = float(feat.peak_height)
            out[f"{pref}_peak_fwhm"] = float(feat.peak_fwhm)

    # volume scalar metrics
    void_cfg = getattr(metrics, 'voids', None)
    if void_cfg is not None and bool(getattr(void_cfg, 'enabled', False)):
        from .voids import sample_void_clearance_radii, clearance_scalar_metrics

        n_samples = int(getattr(void_cfg, 'n_samples', 0) or 0)
        if n_samples > 0:
            radii = sample_void_clearance_radii(
                frames,
                n_samples=n_samples,
                sampler=str(getattr(void_cfg, 'sampler', 'sobol')),
                seed=int(getattr(void_cfg, 'seed', 0) or 0),
                k_nearest=int(getattr(void_cfg, 'k_nearest', 16) or 16),
                type_to_species=type_to_species,
                radii_by_species=dict(getattr(void_cfg, 'radii', {}) or {}),
                default_radius=float(getattr(void_cfg, 'default_radius', 0.0) or 0.0),
            )
            sm_void = clearance_scalar_metrics(radii, probe_radii=list(getattr(void_cfg, 'probe_radii', []) or []))

            def _slug_float(x: float) -> str:
                s = f"{float(x):.6g}".replace('-', 'm').replace('.', 'p')
                return s

            out['void_clearance_mean'] = float(sm_void.get('mean', float('nan')))
            out['void_clearance_median'] = float(sm_void.get('median', float('nan')))
            out['void_clearance_p95'] = float(sm_void.get('p95', float('nan')))
            out['void_clearance_max'] = float(sm_void.get('max', float('nan')))
            out['void_clearance_n_samples'] = float(len(radii))
            for rp in list(getattr(void_cfg, 'probe_radii', []) or []):
                key = f"void_clearance_frac_ge_r{_slug_float(float(rp))}"
                out[key] = float(sm_void.get(f"frac_ge_{float(rp)}", float('nan')))
    return StructureMetrics(values=out)


def compute_structure_metrics(
    frame: DumpFrame,
    metrics: StructureMetricsConfig,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]] = None,
) -> StructureMetrics:
    """Structure metrics."""

    if not metrics.enabled:
        return StructureMetrics(values={})

    invH = np.linalg.inv(frame.cell)
    frac = _wrap_frac((frame.positions - frame.origin) @ invH)
    posw = frame.origin + frac @ frame.cell

    # neighbor maximum cutoff
    if not cutoffs:
        return StructureMetrics(values={})
    max_cut = float(max(cutoffs.values()))
    atoms = Atoms(numbers=np.ones(frame.n_atoms, dtype=int), positions=posw, cell=frame.cell, pbc=True)
    ii, jj = neighbor_list("ij", atoms, max_cut)
    m = ii < jj
    ii = ii[m]
    jj = jj[m]

    # coordination analysis diagnostics
    sweep_cfg = getattr(metrics, "coordination_sweep", None)
    do_sweep = bool(getattr(sweep_cfg, "enabled", False))
    sweep_dr = float(getattr(sweep_cfg, "dr", 0.0)) if do_sweep else 0.0
    sweep_n_below = int(getattr(sweep_cfg, "n_below", 0)) if do_sweep else 0
    sweep_n_above = int(getattr(sweep_cfg, "n_above", 0)) if do_sweep else 0
    sweep_strained_delta = float(getattr(sweep_cfg, "strained_delta", 0.0)) if do_sweep else 0.0

    dr, dist = _mic_displacements(frac, frame.cell, ii, jj)
    t_i = frame.types[ii]
    t_j = frame.types[jj]

    # neighbor pair cutoffs
    nbr_ids: List[List[int]] = [[] for _ in range(frame.n_atoms)]
    nbr_vecs: List[List[np.ndarray]] = [[] for _ in range(frame.n_atoms)]
    nbr_dists: List[List[float]] = [[] for _ in range(frame.n_atoms)]

    # pair bond metrics
    dist_by_pair: Dict[Tuple[int, int], List[float]] = {}

    for a, b, vec, d, ta, tb in zip(ii.tolist(), jj.tolist(), dr.tolist(), dist.tolist(), t_i.tolist(), t_j.tolist()):
        key = _pair_key(int(ta), int(tb))
        c = cutoffs.get(key, None)
        if c is None:
            continue
        if d > c:
            continue
        # add directed
        nbr_ids[a].append(b)
        nbr_vecs[a].append(np.asarray(vec, dtype=float))
        nbr_dists[a].append(float(d))
        nbr_ids[b].append(a)
        nbr_vecs[b].append(-np.asarray(vec, dtype=float))
        nbr_dists[b].append(float(d))

        dist_by_pair.setdefault(key, []).append(float(d))

    vals: Dict[str, float] = {}

    # bond length metrics
    for pm in metrics.pairs:
        a_sel, b_sel = pm.pair
        a_types = _resolve_selector(a_sel, type_to_species)
        b_types = _resolve_selector(b_sel, type_to_species)
        ds: List[float] = []
        for ta in a_types:
            for tb in b_types:
                key = _pair_key(ta, tb)
                ds.extend(dist_by_pair.get(key, []))
        name = f"bondlen_{a_sel}-{b_sel}"
        if len(ds) == 0:
            vals[f"{name}_mean"] = float("nan")
            vals[f"{name}_std"] = float("nan")
        else:
            arr = np.asarray(ds, dtype=float)
            vals[f"{name}_mean"] = float(np.mean(arr))
            vals[f"{name}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    # coordination metrics
    for cm in metrics.coordinations:
        c_types = _resolve_selector(cm.central, type_to_species)
        n_types = _resolve_selector(cm.neighbor, type_to_species)
        # count central atom
        counts: List[int] = []
        for idx in range(frame.n_atoms):
            t = int(frame.types[idx])
            if t not in c_types:
                continue
            # count neighbors desired
            k = 0
            for nb, d in zip(nbr_ids[idx], nbr_dists[idx]):
                if int(frame.types[nb]) in n_types:
                    k += 1
            counts.append(k)
        name = f"coord_{cm.central}-{cm.neighbor}"
        if len(counts) == 0:
            vals[f"{name}_mean"] = float("nan")
            vals[f"{name}_std"] = float("nan")
        else:
            arr = np.asarray(counts, dtype=float)
            vals[f"{name}_mean"] = float(np.mean(arr))
            vals[f"{name}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    # angle metrics
    for am in metrics.angles:
        a_sel, b_sel, c_sel = am.triplet
        a_types = _resolve_selector(a_sel, type_to_species)
        b_types = _resolve_selector(b_sel, type_to_species)
        c_types = _resolve_selector(c_sel, type_to_species)

        angles: List[float] = []
        for b_idx in range(frame.n_atoms):
            if int(frame.types[b_idx]) not in b_types:
                continue
            # neighbor vectors desired
            neighA: List[np.ndarray] = []
            neighA_idx: List[int] = []
            neighC: List[np.ndarray] = []
            neighC_idx: List[int] = []
            for nb, vec in zip(nbr_ids[b_idx], nbr_vecs[b_idx]):
                tnb = int(frame.types[nb])
                if tnb in a_types:
                    neighA.append(np.asarray(vec, dtype=float))
                    neighA_idx.append(int(nb))
                if tnb in c_types:
                    neighC.append(np.asarray(vec, dtype=float))
                    neighC_idx.append(int(nb))

            if not neighA or not neighC:
                continue

            if set(a_types) == set(c_types) and a_sel == c_sel:
                # unique pairs
                for p, q in combinations(range(len(neighA)), 2):
                    v = neighA[p]
                    w2 = neighA[q]
                    denom = float(np.linalg.norm(v) * np.linalg.norm(w2))
                    if denom <= 0:
                        continue
                    cos = float(np.dot(v, w2) / denom)
                    cos = max(-1.0, min(1.0, cos))
                    angles.append(float(math.degrees(math.acos(cos))))
            else:
                for v, idxv in zip(neighA, neighA_idx):
                    for w2, idxw in zip(neighC, neighC_idx):
                        if idxv == idxw:
                            continue
                        denom = float(np.linalg.norm(v) * np.linalg.norm(w2))
                        if denom <= 0:
                            continue
                        cos = float(np.dot(v, w2) / denom)
                        cos = max(-1.0, min(1.0, cos))
                        angles.append(float(math.degrees(math.acos(cos))))

        name = f"angle_{a_sel}-{b_sel}-{c_sel}"
        if len(angles) == 0:
            vals[f"{name}_mean"] = float("nan")
            vals[f"{name}_std"] = float("nan")
        else:
            arr = np.asarray(angles, dtype=float)
            vals[f"{name}_mean"] = float(np.mean(arr))
            vals[f"{name}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    # ring metrics
    if metrics.rings is not None and metrics.rings.enabled:
        nbr_ids_for_rings: Sequence[Sequence[int]] = nbr_ids
        if metrics.rings.bond_pairs:
            ring_keys: set[Tuple[int, int]] = set()
            for bp in metrics.rings.bond_pairs:
                a_sel, b_sel = bp.pair
                for ta in _resolve_selector(a_sel, type_to_species):
                    for tb in _resolve_selector(b_sel, type_to_species):
                        ring_keys.add(_pair_key(ta, tb))
            filtered: List[List[int]] = [[] for _ in range(frame.n_atoms)]
            for i in range(frame.n_atoms):
                ti = int(frame.types[i])
                for j in nbr_ids[i]:
                    tj = int(frame.types[j])
                    if _pair_key(ti, tj) in ring_keys:
                        filtered[i].append(int(j))
            nbr_ids_for_rings = filtered

        vals.update(_compute_ring_metrics(frame, metrics.rings, nbr_ids_for_rings, type_to_species))

    return StructureMetrics(values=vals)



def compute_coordination_defects(
    frame: DumpFrame,
    metrics: StructureMetricsConfig,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Coordination defects."""
    if not metrics.enabled:
        return {}

    # coordination metrics expectation
    cms: list[CoordinationMetricConfig] = []
    for cm in metrics.coordinations:
        if getattr(cm, "expected", None) is not None or getattr(cm, "allowed", None) is not None:
            cms.append(cm)
    if not cms:
        return {}

    if not cutoffs:
        return {}

    invH = np.linalg.inv(frame.cell)
    frac = _wrap_frac((frame.positions - frame.origin) @ invH)
    posw = frame.origin + frac @ frame.cell

    max_cut = float(max(cutoffs.values()))
    atoms = Atoms(numbers=np.ones(frame.n_atoms, dtype=int), positions=posw, cell=frame.cell, pbc=True)
    ii, jj = neighbor_list("ij", atoms, max_cut)
    m = ii < jj
    ii = ii[m]
    jj = jj[m]
    if ii.size == 0:
        # neighbours
        out0: Dict[str, Dict[str, object]] = {}
        for cm in cms:
            name = f"coord_{cm.central}-{cm.neighbor}"
            allowed = getattr(cm, "allowed", None)
            expected = getattr(cm, "expected", None)
            if allowed is None and expected is not None:
                allowed = [int(expected)]
            tol = float(getattr(cm, "defect_frac_tol", 0.0))
            out0[name] = {
                "expected": int(expected) if expected is not None else None,
                "allowed": [int(x) for x in allowed] if allowed is not None else None,
                "n_central": 0,
                "n_defective": 0,
                "defect_fraction": float("nan"),
                "defect_frac_tol": float(tol),
                "has_defect": False,
            }
        return out0

    dr, dist = _mic_displacements(frac, frame.cell, ii, jj)
    t_i = frame.types[ii]
    t_j = frame.types[jj]

    nbr_ids: List[List[int]] = [[] for _ in range(frame.n_atoms)]
    nbr_dists: List[List[float]] = [[] for _ in range(frame.n_atoms)]

    for a, b, d, ta, tb in zip(ii.tolist(), jj.tolist(), dist.tolist(), t_i.tolist(), t_j.tolist()):
        key = _pair_key(int(ta), int(tb))
        c = cutoffs.get(key, None)
        if c is None or float(d) > float(c):
            continue
        nbr_ids[a].append(int(b))
        nbr_dists[a].append(float(d))
        nbr_ids[b].append(int(a))
        nbr_dists[b].append(float(d))

    out: Dict[str, Dict[str, object]] = {}
    for cm in cms:
        c_types = _resolve_selector(cm.central, type_to_species)
        n_types = _resolve_selector(cm.neighbor, type_to_species)
        cset = set(int(x) for x in c_types)
        nset = set(int(x) for x in n_types)

        allowed = getattr(cm, "allowed", None)
        expected = getattr(cm, "expected", None)
        if allowed is None and expected is not None:
            allowed = [int(expected)]
        allowed_set = set(int(x) for x in allowed) if allowed is not None else set()

        counts: List[int] = []
        for idx in range(frame.n_atoms):
            if int(frame.types[idx]) not in cset:
                continue
            k = 0
            for nb in nbr_ids[idx]:
                if int(frame.types[nb]) in nset:
                    k += 1
            counts.append(int(k))

        name = f"coord_{cm.central}-{cm.neighbor}"
        tol = float(getattr(cm, "defect_frac_tol", 0.0))
        if len(counts) == 0 or allowed is None:
            out[name] = {
                "expected": int(expected) if expected is not None else None,
                "allowed": [int(x) for x in allowed] if allowed is not None else None,
                "n_central": int(len(counts)),
                "n_defective": 0,
                "defect_fraction": float("nan") if len(counts) == 0 else 0.0,
                "defect_frac_tol": float(tol),
                "has_defect": False,
            }
            continue

        arr = np.asarray(counts, dtype=int)
        bad = ~np.isin(arr, list(allowed_set))
        n_def = int(np.sum(bad))
        frac_def = float(n_def) / float(arr.size) if arr.size > 0 else float("nan")
        out[name] = {
            "expected": int(expected) if expected is not None else None,
            "allowed": [int(x) for x in allowed] if allowed is not None else None,
            "n_central": int(arr.size),
            "n_defective": int(n_def),
            "defect_fraction": float(frac_def),
            "defect_frac_tol": float(tol),
            "has_defect": bool(math.isfinite(frac_def) and frac_def > float(tol)),
        }

    return out


def compute_coordination_defect_details(
    frame: DumpFrame,
    metrics: StructureMetricsConfig,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Coordination defect details."""

    if not metrics.enabled:
        return {}

    # coordination cutoff diagnostics
    # diagnostic payloads affect
    sweep_cfg = getattr(metrics, "coordination_sweep", None)
    do_sweep = bool(getattr(sweep_cfg, "enabled", False))
    sweep_dr = float(getattr(sweep_cfg, "dr", 0.0)) if do_sweep else 0.0
    sweep_n_below = int(getattr(sweep_cfg, "n_below", 0)) if do_sweep else 0
    sweep_n_above = int(getattr(sweep_cfg, "n_above", 0)) if do_sweep else 0
    sweep_strained_delta = float(getattr(sweep_cfg, "strained_delta", 0.0)) if do_sweep else 0.0

    # coordination metrics expectation
    cms: list[CoordinationMetricConfig] = []
    for cm in metrics.coordinations:
        if getattr(cm, "expected", None) is not None or getattr(cm, "allowed", None) is not None:
            cms.append(cm)
    if not cms:
        return {}
    if not cutoffs:
        return {}

    invH = np.linalg.inv(frame.cell)
    frac = _wrap_frac((frame.positions - frame.origin) @ invH)
    posw = frame.origin + frac @ frame.cell

    max_cut = float(max(cutoffs.values()))
    atoms = Atoms(numbers=np.ones(frame.n_atoms, dtype=int), positions=posw, cell=frame.cell, pbc=True)
    ii, jj = neighbor_list("ij", atoms, max_cut)
    m = ii < jj
    ii = ii[m]
    jj = jj[m]

    # neighbour pair cutoffs
    nbr_ids: List[List[int]] = [[] for _ in range(frame.n_atoms)]
    # pair cutoff diagnostics
    ii_all = np.asarray(ii, dtype=int)
    jj_all = np.asarray(jj, dtype=int)
    dist_all = np.asarray([], dtype=float)
    t_i_all = np.asarray([], dtype=int)
    t_j_all = np.asarray([], dtype=int)
    if ii.size > 0:
        _, dist = _mic_displacements(frac, frame.cell, ii, jj)
        t_i = frame.types[ii]
        t_j = frame.types[jj]
        dist_all = np.asarray(dist, dtype=float)
        t_i_all = np.asarray(t_i, dtype=int)
        t_j_all = np.asarray(t_j, dtype=int)
        for a, b, d, ta, tb in zip(ii.tolist(), jj.tolist(), dist.tolist(), t_i.tolist(), t_j.tolist()):
            key = _pair_key(int(ta), int(tb))
            c = cutoffs.get(key, None)
            if c is None or float(d) > float(c):
                continue
            nbr_ids[a].append(int(b))
            nbr_ids[b].append(int(a))

    out: Dict[str, Dict[str, object]] = {}

    for cm in cms:
        name = f"coord_{cm.central}-{cm.neighbor}"

        c_types = _resolve_selector(cm.central, type_to_species)
        n_types = _resolve_selector(cm.neighbor, type_to_species)
        cset = set(int(x) for x in c_types)
        nset = set(int(x) for x in n_types)

        allowed = getattr(cm, "allowed", None)
        expected = getattr(cm, "expected", None)
        if allowed is None and expected is not None:
            allowed = [int(expected)]
        allowed_set = set(int(x) for x in allowed) if allowed is not None else set()

        # pair cutoffs neighbor
        pair_cutoffs: list[dict[str, object]] = []
        seen: set[tuple[int, int]] = set()
        for ta in sorted(cset):
            for tb in sorted(nset):
                key = _pair_key(int(ta), int(tb))
                if key in seen:
                    continue
                seen.add(key)
                if key in cutoffs:
                    pair_cutoffs.append({"pair": [int(key[0]), int(key[1])], "cutoff": float(cutoffs[key])})

        # enumerate central coordination
        central_idx = [i for i in range(frame.n_atoms) if int(frame.types[i]) in cset]
        counts: list[int] = []
        defective_ids: list[int] = []
        defective_idx: list[int] = []
        defective_coord: list[int] = []
        shell_idx_set: set[int] = set()
        shell_id_set: set[int] = set()
        under_ids: list[int] = []
        under_idx: list[int] = []
        under_coord: list[int] = []
        over_ids: list[int] = []
        over_idx: list[int] = []
        over_coord: list[int] = []

        for idx in central_idx:
            nb = [j for j in nbr_ids[idx] if int(frame.types[j]) in nset]
            k = int(len(nb))
            counts.append(k)
            if allowed_set and (k not in allowed_set):
                defective_ids.append(int(frame.ids[idx]))
                defective_idx.append(int(idx))
                defective_coord.append(int(k))
                # coordination neighbours requested
                for j in nb:
                    shell_idx_set.add(int(j))
                    shell_id_set.add(int(frame.ids[int(j)]))
                if expected is not None:
                    if k < int(expected):
                        under_ids.append(int(frame.ids[idx]))
                        under_idx.append(int(idx))
                        under_coord.append(int(k))
                    elif k > int(expected):
                        over_ids.append(int(frame.ids[idx]))
                        over_idx.append(int(idx))
                        over_coord.append(int(k))

        n_central = int(len(counts))
        n_def = int(len(defective_ids))
        defect_fraction = float(n_def) / float(n_central) if n_central > 0 else float("nan")
        tol = float(getattr(cm, "defect_frac_tol", 0.0))
        has_defect = bool(math.isfinite(defect_fraction) and allowed_set and defect_fraction > float(tol))

        # histogram quick inspection
        hist: dict[int, int] = {}
        for k in counts:
            hist[int(k)] = int(hist.get(int(k), 0) + 1)

        # cutoff sensitivity diagnostics
        sweep_payload: dict[str, object] | None = None
        under_delta_to_fix: list[float] = []
        under_kind: list[str] = []
        over_delta_to_fix: list[float] = []
        over_kind: list[str] = []

        if (
            do_sweep
            and (sweep_n_above > 0 or sweep_n_below > 0)
            and sweep_dr > 0.0
            and ii_all.size > 0
            and n_central > 0
            and bool(allowed_set)
        ):
            max_shift = float(sweep_n_above) * float(sweep_dr)

            # pair neighbour pairs
            # restricted search radius
            delta_lists: list[list[float]] = [[] for _ in range(frame.n_atoms)]

            for a, b, d, ta, tb in zip(
                ii_all.tolist(),
                jj_all.tolist(),
                dist_all.tolist(),
                t_i_all.tolist(),
                t_j_all.tolist(),
            ):
                ta_i = int(ta)
                tb_i = int(tb)
                dd = float(d)

                if ta_i in cset and tb_i in nset:
                    base = cutoffs.get(_pair_key(ta_i, tb_i), None)
                    if base is not None and dd <= float(base) + max_shift:
                        delta_lists[int(a)].append(dd - float(base))
                if tb_i in cset and ta_i in nset:
                    base = cutoffs.get(_pair_key(ta_i, tb_i), None)
                    if base is not None and dd <= float(base) + max_shift:
                        delta_lists[int(b)].append(dd - float(base))

            delta_sorted: list[list[float]] = [sorted(v) for v in delta_lists]

            k_grid = list(range(-int(sweep_n_below), int(sweep_n_above) + 1))
            shift_grid = [float(k) * float(sweep_dr) for k in k_grid]
            ndef_grid: list[int] = []
            frac_grid: list[float] = []
            for s in shift_grid:
                ndef = 0
                for idx in central_idx:
                    k = int(bisect_right(delta_sorted[int(idx)], float(s)))
                    if k not in allowed_set:
                        ndef += 1
                ndef_grid.append(int(ndef))
                frac_grid.append(float(ndef) / float(n_central) if n_central > 0 else float("nan"))

            sweep_payload = {
                "dr": float(sweep_dr),
                "n_below": int(sweep_n_below),
                "n_above": int(sweep_n_above),
                "k": [int(k) for k in k_grid],
                "delta_r": [float(x) for x in shift_grid],
                "n_defective": [int(x) for x in ndef_grid],
                "defect_fraction": [float(x) for x in frac_grid],
                "strained_delta": float(sweep_strained_delta),
            }

            # distances relative cutoff
            # coordination smallest neighbour
            # coordination reduction neighbour
            if expected is not None:
                kexp = int(expected)
                if kexp >= 0:
                    for idx in under_idx:
                        ds = delta_sorted[int(idx)]
                        if len(ds) >= kexp:
                            dfix = float(max(0.0, float(ds[kexp - 1]))) if kexp > 0 else 0.0
                        else:
                            dfix = float("inf")
                        under_delta_to_fix.append(float(dfix))
                        under_kind.append(
                            "strained" if (math.isfinite(dfix) and dfix <= float(sweep_strained_delta)) else "missing"
                        )

                    for idx in over_idx:
                        ds = delta_sorted[int(idx)]
                        if len(ds) >= kexp + 1:
                            # cutoff neighbour order
                            dfix = float(max(0.0, -float(ds[kexp])))
                        else:
                            dfix = float("inf")
                        over_delta_to_fix.append(float(dfix))
                        over_kind.append(
                            "strained" if (math.isfinite(dfix) and dfix <= float(sweep_strained_delta)) else "extra"
                        )

        out[name] = {
            "expected": int(expected) if expected is not None else None,
            "allowed": [int(x) for x in allowed] if allowed is not None else None,
            "pair_cutoffs": pair_cutoffs,
            "n_central": int(n_central),
            "n_defective": int(n_def),
            "defect_fraction": float(defect_fraction),
            "defect_frac_tol": float(tol),
            "has_defect": bool(has_defect),
            "coord_hist": {str(int(k)): int(v) for k, v in sorted(hist.items())},
            "defective_ids": defective_ids,
            "defective_idx": defective_idx,
            "defective_coord": defective_coord,
            "shell_ids": sorted([int(x) for x in (shell_id_set | set(defective_ids))]),
            "shell_idx": sorted([int(x) for x in (shell_idx_set | set(defective_idx))]),
            "under_ids": under_ids,
            "under_idx": under_idx,
            "under_coord": under_coord,
            "over_ids": over_ids,
            "over_idx": over_idx,
            "over_coord": over_coord,
            "coordination_sweep": sweep_payload,
            "under_delta_to_fix": under_delta_to_fix,
            "under_kind": under_kind,
            "over_delta_to_fix": over_delta_to_fix,
            "over_kind": over_kind,
        }

    return out



def _compute_ring_metrics(
    frame: DumpFrame,
    ring: RingMetricsConfig,
    nbr_ids: Sequence[Sequence[int]],
    type_to_species: Optional[Sequence[str]],
) -> Dict[str, float]:
    """Ring metrics."""

    if not ring.enabled:
        return {}
    if ring.mode not in ("bond_graph", "projected"):
        raise ValueError(f"Unsupported ring mode: {ring.mode}")

    node_types: List[int] = []
    for sel in ring.nodes:
        node_types.extend(_resolve_selector(sel, type_to_species))
    node_types = sorted(set(node_types))
    bridge_types: List[int] = []
    if ring.mode == "projected":
        if ring.bridge is None:
            raise ValueError("rings.mode='projected' requires rings.bridge")
        bridge_types = sorted(set(_resolve_selector(ring.bridge, type_to_species)))

    allowed_types = set(node_types) | set(bridge_types) if ring.mode == "projected" else set(node_types)
    allowed_nodes = [i for i in range(frame.n_atoms) if int(frame.types[i]) in allowed_types]
    allowed_set = set(allowed_nodes)

    # bond graph allowed
    G = nx.Graph()
    G.add_nodes_from(allowed_nodes)
    for i in allowed_nodes:
        for j in nbr_ids[i]:
            if j in allowed_set and i < j:
                G.add_edge(i, j)

    if ring.mode == "bond_graph":
        Gb = G
    else:
        # project bridge nodes
        nodes = [i for i in allowed_nodes if int(frame.types[i]) in set(node_types)]
        bridges = [i for i in allowed_nodes if int(frame.types[i]) in set(bridge_types)]
        Gp = nx.Graph()
        Gp.add_nodes_from(nodes)
        nodes_set = set(nodes)
        for b in bridges:
            nbrs = [n for n in G.neighbors(b) if n in nodes_set]
            if len(nbrs) < 2:
                continue
            for u, v in combinations(sorted(nbrs), 2):
                Gp.add_edge(u, v)
        Gb = Gp

    # ring enumeration algorithm
    if getattr(ring, "algorithm", "cycle_basis") == "cycle_basis":
        cycles = nx.cycle_basis(Gb)
    else:
        cycles = _primitive_rings(Gb, max_cycle_size=int(ring.max_cycle_size), max_paths_per_edge=int(getattr(ring, "max_paths_per_edge", 16)))

    sizes = [len(c) for c in cycles if len(c) >= 3 and len(c) <= int(ring.max_cycle_size)]
    out: Dict[str, float] = {}
    if not sizes:
        out["ring_mean_size"] = float("nan")
        for k in range(3, int(ring.max_cycle_size) + 1):
            out[f"ring_frac_{k}"] = 0.0
        return out

    sizes_arr = np.asarray(sizes, dtype=int)
    out["ring_mean_size"] = float(np.mean(sizes_arr))
    # histogram
    counts = {k: int(np.sum(sizes_arr == k)) for k in range(3, int(ring.max_cycle_size) + 1)}
    total = float(sum(counts.values()))
    for k in range(3, int(ring.max_cycle_size) + 1):
        out[f"ring_frac_{k}"] = float(counts.get(k, 0) / total) if total > 0 else 0.0
    return out


def _canonical_cycle(nodes: Sequence[int]) -> Tuple[int, ...]:
    """Canonical cycle."""
    cyc = list(map(int, nodes))
    k = len(cyc)
    if k == 0:
        return tuple()
    rots = [tuple(cyc[i:] + cyc[:i]) for i in range(k)]
    rev = list(reversed(cyc))
    rots_r = [tuple(rev[i:] + rev[:i]) for i in range(k)]
    return min(rots + rots_r)


def _is_chordless_cycle(G: "nx.Graph", cycle: Sequence[int]) -> bool:
    """Is chordless cycle."""
    nodes = list(map(int, cycle))
    k = len(nodes)
    if k < 3:
        return False
    cycle_edges: set[tuple[int, int]] = set()
    for i in range(k):
        a = nodes[i]
        b = nodes[(i + 1) % k]
        cycle_edges.add((a, b) if a <= b else (b, a))
    sub_edges: set[tuple[int, int]] = set()
    for a, b in G.subgraph(nodes).edges():
        sub_edges.add((a, b) if a <= b else (b, a))
    return sub_edges == cycle_edges


def _primitive_rings(G: "nx.Graph", *, max_cycle_size: int, max_paths_per_edge: int = 16) -> List[List[int]]:
    """Primitive rings."""
    if max_cycle_size < 3:
        return []
    if max_paths_per_edge < 1:
        max_paths_per_edge = 1

    G_orig = G.copy()
    rings: set[Tuple[int, ...]] = set()

    edges = list(G.edges())
    for u, v in edges:
        if not G.has_edge(u, v):
            continue
        G.remove_edge(u, v)
        try:
            try:
                d = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                continue
            k = int(d + 1)
            if k < 3 or k > int(max_cycle_size):
                continue

            npaths = 0
            for path in nx.all_shortest_paths(G, u, v):
                npaths += 1
                if npaths > int(max_paths_per_edge):
                    break
                if len(path) != k:
                    # defensive occur shortest
                    continue
                if _is_chordless_cycle(G_orig, path):
                    rings.add(_canonical_cycle(path))
        finally:
            G.add_edge(u, v)

    return [list(t) for t in sorted(rings)]


def required_pairs_from_metrics(metrics: StructureMetricsConfig, *, type_to_species: Optional[Sequence[str]]) -> List[Tuple[int, int]]:
    """Required pairs from."""

    pairs: List[Tuple[int, int]] = []

    for pm in metrics.pairs:
        a, b = pm.pair
        for ta in _resolve_selector(a, type_to_species):
            for tb in _resolve_selector(b, type_to_species):
                pairs.append(_pair_key(ta, tb))

    for cm in metrics.coordinations:
        for ta in _resolve_selector(cm.central, type_to_species):
            for tb in _resolve_selector(cm.neighbor, type_to_species):
                pairs.append(_pair_key(ta, tb))

    for am in metrics.angles:
        a, b, c = am.triplet
        for tb in _resolve_selector(b, type_to_species):
            for ta in _resolve_selector(a, type_to_species):
                pairs.append(_pair_key(tb, ta))
            for tc in _resolve_selector(c, type_to_species):
                pairs.append(_pair_key(tb, tc))

    if metrics.rings is not None and metrics.rings.enabled and metrics.rings.bond_pairs:
        for bp in metrics.rings.bond_pairs:
            a, b = bp.pair
            for ta in _resolve_selector(a, type_to_species):
                for tb in _resolve_selector(b, type_to_species):
                    pairs.append(_pair_key(ta, tb))

    # unique
    out = sorted(set(pairs))
    return out


def fixed_cutoffs_from_metrics(metrics: StructureMetricsConfig, *, type_to_species: Optional[Sequence[str]]) -> Dict[Tuple[int, int], float]:
    """Fixed cutoffs from."""
    out: Dict[Tuple[int, int], float] = {}
    for pm in metrics.pairs:
        if pm.cutoff is None:
            continue
        a, b = pm.pair
        for ta in _resolve_selector(a, type_to_species):
            for tb in _resolve_selector(b, type_to_species):
                out[_pair_key(ta, tb)] = float(pm.cutoff)
    for cm in metrics.coordinations:
        if cm.cutoff is None:
            continue
        for ta in _resolve_selector(cm.central, type_to_species):
            for tb in _resolve_selector(cm.neighbor, type_to_species):
                out[_pair_key(ta, tb)] = float(cm.cutoff)
    if metrics.rings is not None and metrics.rings.enabled:
        for bp in metrics.rings.bond_pairs:
            if bp.cutoff is None:
                continue
            a, b = bp.pair
            for ta in _resolve_selector(a, type_to_species):
                for tb in _resolve_selector(b, type_to_species):
                    out[_pair_key(ta, tb)] = float(bp.cutoff)
    return out
