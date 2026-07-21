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
from .dump import DumpFrame, frame_pbc
from .common import (
    canonical_unique_mic_pairs as _canonical_unique_mic_pairs,
    resolve_selector as _resolve_selector,
    wrap_frac as _wrap_frac,
)
from .gr import compute_first_peak_gr, _shortest_lattice_translation
from .graph import GraphRule, StructureGraph, build_hard_graph, directed_neighbor_lists, legacy_graph_rule_from_cutoffs, pair_cutoffs_from_parameters


def _pair_key(t1: int, t2: int) -> Tuple[int, int]:
    return (t1, t2) if t1 <= t2 else (t2, t1)



def estimate_pair_cutoffs(
    frames: Sequence[DumpFrame],
    required_pairs: Sequence[Tuple[int, int]],
    *,
    auto: AutoCutoffConfig,
    fixed_cutoffs: Mapping[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """Estimate missing pair cutoffs under the configured scope policy.

    The caller determines whether ``frames`` represent one box or a pooled
    ensemble.  This routine enforces the important negative contract:
    ``scope='disabled'`` never estimates a missing value.
    """
    cutoffs: Dict[Tuple[int, int], float] = dict(fixed_cutoffs)
    missing = sorted(
        {
            _pair_key(int(pair[0]), int(pair[1]))
            for pair in required_pairs
            if _pair_key(int(pair[0]), int(pair[1])) not in cutoffs
        }
    )
    if str(getattr(auto, "scope", "pooled_ensemble")) == "disabled" and missing:
        missing_text = ", ".join(f"({a},{b})" for a, b in missing)
        raise ValueError(
            "auto_cutoff.scope='disabled' forbids cutoff estimation, but "
            f"explicit cutoffs are missing for required pair(s): {missing_text}. "
            "Set cutoff on the relevant pairs/coordinations/ring bond pairs, "
            "or provide a complete explicit analysis cutoff map."
        )
    if not missing:
        return cutoffs

    unique_image_limits: list[float] = []
    for frame_index, fr in enumerate(frames):
        pbc = frame_pbc(fr)
        if not all(pbc):
            raise ValueError(
                "shell-normalized automatic cutoff estimation requires PBC in all three "
                f"directions; frame {frame_index} has pbc={list(pbc)}. Provide explicit cutoffs."
            )
        shortest = _shortest_lattice_translation(fr.cell, pbc=pbc)
        unique_image_limits.append(float(np.nextafter(0.5 * shortest, 0.0)))
    if not unique_image_limits:
        raise ValueError("automatic cutoff estimation requires at least one frame")
    r_max_eff = min(float(auto.r_max), float(min(unique_image_limits)))
    if not (math.isfinite(r_max_eff) and r_max_eff > 0.0):
        raise ValueError("automatic cutoff estimation has no positive unique-image radius")

    # Derive each canonical pair once.  Histogram extrema must be taken from a
    # shell-normalised pair distribution, not raw counts (whose expectation
    # grows as r^2 even when g(r) is flat).
    for pair in missing:
        key = _pair_key(int(pair[0]), int(pair[1]))
        edges = np.linspace(0.0, float(r_max_eff), int(auto.nbins) + 1)
        shell_vol = (4.0 * math.pi / 3.0) * (
            edges[1:] ** 3 - edges[:-1] ** 3
        )
        observed = np.zeros((int(auto.nbins),), dtype=float)
        ideal = np.zeros((int(auto.nbins),), dtype=float)
        n_observed = 0
        for fr in frames:
            pbc = frame_pbc(fr)
            # wrap coords cell
            invH = np.linalg.inv(fr.cell)
            frac = _wrap_frac((fr.positions - fr.origin) @ invH, pbc=pbc)
            posw = fr.origin + frac @ fr.cell

            atoms = Atoms(numbers=np.ones(fr.n_atoms, dtype=int), positions=posw, cell=fr.cell, pbc=pbc)
            ii, jj = neighbor_list("ij", atoms, r_max_eff)
            ii, jj, _vec, dist = _canonical_unique_mic_pairs(
                frac, fr.cell, ii, jj, cutoff=float(r_max_eff), pbc=pbc
            )
            # filter
            t_i = fr.types[ii]
            t_j = fr.types[jj]
            want = (t_i == pair[0]) & (t_j == pair[1]) | (t_i == pair[1]) & (t_j == pair[0])
            if np.any(want):
                dist = np.asarray(dist, dtype=float)[want]
                dist = dist[
                    np.isfinite(dist)
                    & (dist > 1.0e-8)
                    & (dist <= float(r_max_eff))
                ]
                if dist.size:
                    frame_counts, _ = np.histogram(dist, bins=edges)
                    observed += frame_counts.astype(float)
                    n_observed += int(dist.size)

            volume = abs(float(np.linalg.det(np.asarray(fr.cell, dtype=float))))
            if not (math.isfinite(volume) and volume > 0.0):
                raise ValueError("Cannot estimate a cutoff from a frame with non-positive volume")
            types = np.asarray(fr.types, dtype=int)
            na = int(np.sum(types == int(key[0])))
            nb = int(np.sum(types == int(key[1])))
            if key[0] == key[1]:
                possible_pairs = float(na * (na - 1)) / 2.0
            else:
                possible_pairs = float(na * nb)
            if possible_pairs > 0.0:
                ideal += (possible_pairs / volume) * shell_vol

        if n_observed < 10:
            raise ValueError(
                f"Not enough neighbor distances to estimate cutoff for pair {pair} "
                f"(found {n_observed} distances <= {r_max_eff}). Provide an explicit cutoff."
            )

        radial_g = np.full_like(observed, np.nan, dtype=float)
        valid_ideal = ideal > 0.0
        radial_g[valid_ideal] = observed[valid_ideal] / ideal[valid_ideal]
        if not np.all(valid_ideal):
            raise ValueError(f"Invalid ideal shell population while estimating cutoff for pair {pair}")

        # smooth moving average
        w = int(auto.smooth)
        if w < 1:
            w = 1
        w = min(w, int(radial_g.size))
        if w % 2 == 0:
            w = max(1, w - 1)
        kernel = np.ones(w, dtype=float) / float(w)
        smooth = np.convolve(radial_g, kernel, mode="same")

        centers = 0.5 * (edges[:-1] + edges[1:])

        # peak search
        p0, p1 = auto.peak_search
        m_peak = (centers >= p0) & (centers <= p1)
        if not np.any(m_peak):
            raise ValueError(
                f"Invalid peak_search window {auto.peak_search} for effective unique-image "
                f"r_max={r_max_eff}"
            )
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


def _angles_for_center(
    neighbor_ids: Sequence[int],
    neighbor_vectors: Sequence[np.ndarray],
    atom_types: np.ndarray,
    a_types: set[int],
    c_types: set[int],
) -> list[float]:
    """Return each geometrically unordered A-centre-C angle exactly once.

    Endpoint labels describe membership constraints, not an orientation of the
    angle.  Enumerating unordered neighbour pairs avoids double counting when
    selectors are equivalent, overlap, or are expressed through different
    but semantically identical tokens.
    """
    out: list[float] = []
    for p, q in combinations(range(len(neighbor_ids)), 2):
        ip = int(neighbor_ids[p])
        iq = int(neighbor_ids[q])
        if ip == iq:
            continue
        tp = int(atom_types[ip])
        tq = int(atom_types[iq])
        matches = (tp in a_types and tq in c_types) or (
            tq in a_types and tp in c_types
        )
        if not matches:
            continue
        v = np.asarray(neighbor_vectors[p], dtype=float)
        w = np.asarray(neighbor_vectors[q], dtype=float)
        denom = float(np.linalg.norm(v) * np.linalg.norm(w))
        if denom <= 0.0:
            continue
        cosine = max(-1.0, min(1.0, float(np.dot(v, w) / denom)))
        out.append(float(math.degrees(math.acos(cosine))))
    return out


def _length_scale_v_over_n(frames: Sequence[DumpFrame]) -> Optional[float]:
    vals: list[float] = []
    for fr in frames:
        try:
            n = int(fr.n_atoms)
            vol = abs(float(np.linalg.det(np.asarray(fr.cell, dtype=float))))
            if n > 0 and math.isfinite(vol) and vol > 0.0:
                vals.append(float((vol / float(n)) ** (1.0 / 3.0)))
        except Exception:
            pass
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=float)))


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
            "units": "angstrom",
            "normalization": "per_box_sample_cdf",
            "representation_rule_name": "void_field_raw_absolute",
            "void_metric_mode": "raw_absolute",
        }
        length_scale = _length_scale_v_over_n(frames)
        if length_scale is not None and length_scale > 0.0:
            out["void"]["void_clearance_scaled"] = {
                "x": [float(v) / float(length_scale) for v in x.tolist()],
                "cdf": [float(v) for v in cdf.tolist()],
                "units": "reduced_length",
                "normalization": "per_box_sample_cdf",
                "length_scale": float(length_scale),
                "length_scale_definition": "(V/N)^(1/3)",
                "representation_rule_name": "void_field_density_scaled",
                "void_metric_mode": "density_scaled",
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
        pair_keys = {
            _pair_key(int(ta), int(tb)) for ta in a_types for tb in b_types
        }
        for key in pair_keys:
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
        pbc = frame_pbc(fr)
        invH = np.linalg.inv(fr.cell)
        frac = _wrap_frac((fr.positions - fr.origin) @ invH, pbc=pbc)
        posw = fr.origin + frac @ fr.cell

        atoms = Atoms(numbers=np.ones(fr.n_atoms, dtype=int), positions=posw, cell=fr.cell, pbc=pbc)
        ii, jj = neighbor_list("ij", atoms, max_cut)
        ii, jj, dr, dist = _canonical_unique_mic_pairs(
            frac, fr.cell, ii, jj, cutoff=float(max_cut), pbc=pbc
        )
        if ii.size == 0:
            continue

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
            for pair_key in {
                _pair_key(int(ta), int(tb)) for ta in a_types for tb in b_types
            }:
                ds.extend(dist_by_pair.get(pair_key, []))
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
        for (name, _a_sel, _c_sel, a_types, b_types, c_types) in angle_defs:
            aset = set(int(x) for x in a_types)
            bset = set(int(x) for x in b_types)
            cset = set(int(x) for x in c_types)
            angles: List[float] = []
            for b_idx in range(fr.n_atoms):
                if int(fr.types[b_idx]) not in bset:
                    continue
                angles.extend(
                    _angles_for_center(
                        nbr_ids[b_idx],
                        nbr_vecs[b_idx],
                        np.asarray(fr.types, dtype=int),
                        aset,
                        cset,
                    )
                )
            if angles:
                angle_samples[name].extend(angles)

    # finalise bond length
    for (name, _a_types, _b_types, rmax) in pair_defs:
        samples = [float(v) for v in bond_samples.get(name, []) if math.isfinite(float(v))]
        if not math.isfinite(float(rmax)) or float(rmax) <= 0:
            # cutoff selector pair
            out["bondlen"][name] = {
                "x": [],
                "cdf": [],
                "sample_count": 0,
                "available": False,
                "skip_reason": "no finite cutoff for selector pair",
            }
            continue
        if len(samples) == 0:
            out["bondlen"][name] = {
                "x": [],
                "cdf": [],
                "sample_count": 0,
                "available": False,
                "skip_reason": "no finite bond-length samples",
            }
            continue
        x = np.linspace(0.0, float(rmax), int(bondlen_cdf_points), dtype=float)
        cdf = _cdf_on_grid(samples, x)
        out["bondlen"][name] = {
            "x": [float(v) for v in x.tolist()],
            "cdf": [float(v) for v in cdf.tolist()],
            "sample_count": int(len(samples)),
            "available": True,
        }

    # angle cdfs
    if angle_defs:
        xang = np.linspace(0.0, 180.0, int(angle_cdf_points), dtype=float)
        for (name, *_rest) in angle_defs:
            samples = [float(v) for v in angle_samples.get(name, []) if math.isfinite(float(v))]
            if len(samples) == 0:
                out["angle"][name] = {
                    "x": [],
                    "cdf": [],
                    "sample_count": 0,
                    "available": False,
                    "skip_reason": "no finite angle samples",
                }
                continue
            cdf = _cdf_on_grid(samples, xang)
            out["angle"][name] = {
                "x": [float(v) for v in xang.tolist()],
                "cdf": [float(v) for v in cdf.tolist()],
                "sample_count": int(len(samples)),
                "available": True,
            }

    # coordination observed metric
    for (name, *_rest) in coord_defs:
        counts = np.asarray([int(v) for v in coord_samples.get(name, [])], dtype=int)
        if counts.size == 0:
            out["coord"][name] = {
                "x": [],
                "cdf": [],
                "sample_count": 0,
                "available": False,
                "skip_reason": "no coordination samples",
            }
            continue
        kmax = int(np.max(counts))
        xk = np.arange(0, kmax + 1, dtype=float)
        # cdf integer count
        cdf = np.zeros_like(xk, dtype=float)
        for i, k in enumerate(xk.astype(int).tolist()):
            cdf[i] = float(np.mean(counts <= int(k)))
        out["coord"][name] = {
            "x": [float(v) for v in xk.tolist()],
            "cdf": [float(v) for v in cdf.tolist()],
            "sample_count": int(counts.size),
            "available": True,
        }

    return out




def _graph_from_legacy_cutoffs(frame: DumpFrame, cutoffs: Mapping[Tuple[int, int], float], *, type_to_species: Optional[Sequence[str]]) -> StructureGraph:
    """Build the backward-compatible single hard-cutoff graph."""

    if not cutoffs:
        raise ValueError("legacy graph construction requires at least one cutoff")
    return build_hard_graph(frame, legacy_graph_rule_from_cutoffs(cutoffs), type_to_species=type_to_species)


def _graph_neighbor_state(frame: DumpFrame, graph: StructureGraph) -> tuple[List[List[int]], List[List[np.ndarray]], List[List[float]], List[List[float]], Dict[Tuple[int, int], List[float]]]:
    nbr_ids, nbr_vecs, nbr_dists, nbr_weights = directed_neighbor_lists(graph, int(frame.n_atoms))
    dist_by_pair: Dict[Tuple[int, int], List[float]] = {}
    for (a, b), d in zip(graph.edges, graph.edge_distances):
        key = _pair_key(int(frame.types[int(a)]), int(frame.types[int(b)]))
        dist_by_pair.setdefault(key, []).append(float(d))
    return nbr_ids, nbr_vecs, nbr_dists, nbr_weights, dist_by_pair

def compute_structure_distributions_for_graph(
    frame: DumpFrame,
    metrics: StructureMetricsConfig,
    *,
    graph: StructureGraph,
    type_to_species: Optional[Sequence[str]] = None,
    bondlen_cdf_points: int = 200,
    angle_cdf_points: int = 180,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Structure distributions induced by an explicit graph.

    This is the graph-rule-aware counterpart of the legacy cutoff distribution
    path.  Bond lengths, angle distributions and coordination CDFs use exactly
    the supplied graph and each payload carries graph provenance.  Coordinate-only
    void-clearance distributions remain coordinate based.
    """

    if not metrics.enabled:
        return {"bondlen": {}, "angle": {}, "coord": {}, "void": {}}
    if int(bondlen_cdf_points) < 10:
        raise ValueError("bondlen_cdf_points must be >= 10")
    if int(angle_cdf_points) < 10:
        raise ValueError("angle_cdf_points must be >= 10")

    out: Dict[str, Dict[str, Dict[str, object]]] = {"bondlen": {}, "angle": {}, "coord": {}, "void": {}}

    # Coordinate-only void distribution does not induce a graph.
    void_cfg = getattr(metrics, "voids", None)
    if void_cfg is not None and bool(getattr(void_cfg, "enabled", False)):
        try:
            out_void = compute_structure_distributions_timeavg(
                [frame],
                metrics,
                cutoffs={},
                type_to_species=type_to_species,
                bondlen_cdf_points=int(bondlen_cdf_points),
                angle_cdf_points=int(angle_cdf_points),
            ).get("void", {})
            out["void"] = dict(out_void)
        except Exception:
            out["void"] = {}

    nbr_ids, nbr_vecs, _nbr_dists, nbr_weights, dist_by_pair = _graph_neighbor_state(frame, graph)
    rule_payload = graph.graph_rule.to_json()
    provenance = {"graph_rule": rule_payload, "structure_hash": str(graph.structure_hash), "single_rule_output": True}

    for pm in metrics.pairs:
        a_sel, b_sel = pm.pair
        a_types = _resolve_selector(a_sel, type_to_species)
        b_types = _resolve_selector(b_sel, type_to_species)
        ds: List[float] = []
        selector_pair_keys = {
            _pair_key(int(ta), int(tb)) for ta in a_types for tb in b_types
        }
        for pair_key in selector_pair_keys:
            ds.extend(dist_by_pair.get(pair_key, []))
        name = f"bondlen_{a_sel}-{b_sel}"
        samples = [float(v) for v in ds if math.isfinite(float(v))]
        if samples:
            rmax = max(samples) if samples else 0.0
            # Prefer the explicit graph-rule cutoff for the CDF grid when present.
            pc = pair_cutoffs_from_parameters(graph.graph_rule.parameters)
            cands = [pc.get(pair_key) for pair_key in selector_pair_keys]
            cands = [float(x) for x in cands if x is not None and math.isfinite(float(x)) and float(x) > 0.0]
            if cands:
                rmax = max(float(rmax), max(cands))
            x = np.linspace(0.0, float(rmax), int(bondlen_cdf_points), dtype=float)
            cdf = _cdf_on_grid(samples, x)
            out["bondlen"][name] = {
                "x": [float(v) for v in x.tolist()],
                "cdf": [float(v) for v in cdf.tolist()],
                "sample_count": int(len(samples)),
                "available": True,
                **provenance,
            }
        else:
            out["bondlen"][name] = {"x": [], "cdf": [], "sample_count": 0, "available": False, "skip_reason": "no graph-induced bond-length samples", **provenance}

    for cm in metrics.coordinations:
        c_types = _resolve_selector(cm.central, type_to_species)
        n_types = _resolve_selector(cm.neighbor, type_to_species)
        cset = set(int(x) for x in c_types)
        nset = set(int(x) for x in n_types)
        counts: List[float] = []
        for idx in range(frame.n_atoms):
            if int(frame.types[idx]) not in cset:
                continue
            k = 0.0
            for nb, w in zip(nbr_ids[idx], nbr_weights[idx]):
                if int(frame.types[int(nb)]) in nset:
                    k += float(w)
            counts.append(float(k))
        name = f"coord_{cm.central}-{cm.neighbor}"
        if counts:
            arr = np.asarray(counts, dtype=float)
            # Hard graphs give integer CDF support; soft graphs keep a numeric grid.
            if graph.is_hard:
                imax = int(math.ceil(float(np.max(arr))))
                xk = np.arange(0, imax + 1, dtype=float)
            else:
                xk = np.linspace(0.0, max(1.0, float(np.max(arr))), int(max(10, min(200, len(arr) * 4))), dtype=float)
            cdf = np.asarray([float(np.mean(arr <= float(x))) for x in xk], dtype=float)
            out["coord"][name] = {
                "x": [float(v) for v in xk.tolist()],
                "cdf": [float(v) for v in cdf.tolist()],
                "sample_count": int(arr.size),
                "available": True,
                **provenance,
            }
        else:
            out["coord"][name] = {"x": [], "cdf": [], "sample_count": 0, "available": False, "skip_reason": "no coordination samples", **provenance}

    if graph.is_hard:
        xang = np.linspace(0.0, 180.0, int(angle_cdf_points), dtype=float)
        for am in metrics.angles:
            a_sel, b_sel, c_sel = am.triplet
            aset = set(int(x) for x in _resolve_selector(a_sel, type_to_species))
            bset = set(int(x) for x in _resolve_selector(b_sel, type_to_species))
            cset = set(int(x) for x in _resolve_selector(c_sel, type_to_species))
            samples: List[float] = []
            for b_idx in range(frame.n_atoms):
                if int(frame.types[b_idx]) not in bset:
                    continue
                samples.extend(
                    _angles_for_center(
                        nbr_ids[b_idx],
                        nbr_vecs[b_idx],
                        np.asarray(frame.types, dtype=int),
                        aset,
                        cset,
                    )
                )
            name = f"angle_{a_sel}-{b_sel}-{c_sel}"
            if samples:
                cdf = _cdf_on_grid(samples, xang)
                out["angle"][name] = {
                    "x": [float(v) for v in xang.tolist()],
                    "cdf": [float(v) for v in cdf.tolist()],
                    "sample_count": int(len(samples)),
                    "available": True,
                    **provenance,
                }
            else:
                out["angle"][name] = {"x": [], "cdf": [], "sample_count": 0, "available": False, "skip_reason": "no graph-induced angle samples", **provenance}
    else:
        for am in metrics.angles:
            a_sel, b_sel, c_sel = am.triplet
            name = f"angle_{a_sel}-{b_sel}-{c_sel}"
            out["angle"][name] = {"x": [], "cdf": [], "sample_count": 0, "available": False, "skip_reason": "exact angle distributions are hard-graph-only", **provenance}

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
            length_scale = _length_scale_v_over_n(frames)
            if length_scale is not None and length_scale > 0.0:
                radii_scaled = np.asarray(radii, dtype=float) / float(length_scale)
                sm_void_scaled = clearance_scalar_metrics(radii_scaled, probe_radii=None)
                out['void_clearance_scaled_mean'] = float(sm_void_scaled.get('mean', float('nan')))
                out['void_clearance_scaled_median'] = float(sm_void_scaled.get('median', float('nan')))
                out['void_clearance_scaled_p95'] = float(sm_void_scaled.get('p95', float('nan')))
                out['void_clearance_scaled_max'] = float(sm_void_scaled.get('max', float('nan')))
                out['void_clearance_length_scale'] = float(length_scale)
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
    graph: Optional[StructureGraph] = None,
) -> StructureMetrics:
    """Structure metrics.

    When ``graph`` is supplied, all graph-derived descriptors consume exactly
    that graph and do not infer a cutoff internally.  The ``cutoffs`` path is
    retained only as the backward-compatible legacy single-rule mode.
    """

    if not metrics.enabled:
        return StructureMetrics(values={})

    if graph is None:
        if not cutoffs:
            return StructureMetrics(values={})
        graph = _graph_from_legacy_cutoffs(frame, cutoffs, type_to_species=type_to_species)

    nbr_ids, nbr_vecs, nbr_dists, nbr_weights, dist_by_pair = _graph_neighbor_state(frame, graph)
    vals: Dict[str, float] = {}

    # bond length metrics induced by G_lambda
    for pm in metrics.pairs:
        a_sel, b_sel = pm.pair
        a_types = _resolve_selector(a_sel, type_to_species)
        b_types = _resolve_selector(b_sel, type_to_species)
        ds: List[float] = []
        for key in {
            _pair_key(int(ta), int(tb)) for ta in a_types for tb in b_types
        }:
            ds.extend(dist_by_pair.get(key, []))
        name = f"bondlen_{a_sel}-{b_sel}"
        vals[f"bond_incidence_{a_sel}-{b_sel}_count"] = float(len(ds))
        if len(ds) == 0:
            vals[f"{name}_mean"] = float("nan")
            vals[f"{name}_std"] = float("nan")
        else:
            arr = np.asarray(ds, dtype=float)
            vals[f"{name}_mean"] = float(np.mean(arr))
            vals[f"{name}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    # coordination metrics induced by G_lambda; soft graphs use edge weights.
    for cm in metrics.coordinations:
        c_types = _resolve_selector(cm.central, type_to_species)
        n_types = _resolve_selector(cm.neighbor, type_to_species)
        counts: List[float] = []
        for idx in range(frame.n_atoms):
            t = int(frame.types[idx])
            if t not in c_types:
                continue
            k = 0.0
            for nb, w in zip(nbr_ids[idx], nbr_weights[idx]):
                if int(frame.types[nb]) in n_types:
                    k += float(w)
            counts.append(float(k))
        name = f"coord_{cm.central}-{cm.neighbor}"
        if len(counts) == 0:
            vals[f"{name}_mean"] = float("nan")
            vals[f"{name}_std"] = float("nan")
        else:
            arr = np.asarray(counts, dtype=float)
            vals[f"{name}_mean"] = float(np.mean(arr))
            vals[f"{name}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    # Hard-only angular descriptors use neighbour lists from G_lambda.
    if graph.is_hard:
        for am in metrics.angles:
            a_sel, b_sel, c_sel = am.triplet
            a_types = _resolve_selector(a_sel, type_to_species)
            b_types = _resolve_selector(b_sel, type_to_species)
            c_types = _resolve_selector(c_sel, type_to_species)
            aset = set(int(x) for x in a_types)
            bset = set(int(x) for x in b_types)
            cset = set(int(x) for x in c_types)

            angles: List[float] = []
            for b_idx in range(frame.n_atoms):
                if int(frame.types[b_idx]) not in bset:
                    continue
                angles.extend(
                    _angles_for_center(
                        nbr_ids[b_idx],
                        nbr_vecs[b_idx],
                        np.asarray(frame.types, dtype=int),
                        aset,
                        cset,
                    )
                )

            name = f"angle_{a_sel}-{b_sel}-{c_sel}"
            if len(angles) == 0:
                vals[f"{name}_mean"] = float("nan")
                vals[f"{name}_std"] = float("nan")
            else:
                arr = np.asarray(angles, dtype=float)
                vals[f"{name}_mean"] = float(np.mean(arr))
                vals[f"{name}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

        # Ring metrics are exact hard-graph descriptors.
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
    graph: Optional[StructureGraph] = None,
) -> Dict[str, Dict[str, object]]:
    """Coordination defects induced by an explicit graph.

    The legacy ``cutoffs`` argument is converted into a labelled
    ``legacy_single_cutoff`` graph.  If ``graph`` is supplied, no cutoff is
    constructed inside this function.
    """
    if not metrics.enabled:
        return {}

    cms: list[CoordinationMetricConfig] = []
    for cm in metrics.coordinations:
        if getattr(cm, "expected", None) is not None or getattr(cm, "allowed", None) is not None:
            cms.append(cm)
    if not cms:
        return {}

    if graph is None:
        if not cutoffs:
            return {}
        graph = _graph_from_legacy_cutoffs(frame, cutoffs, type_to_species=type_to_species)

    if graph.is_soft:
        # Soft coordination is reported through soft ambiguity metrics; sharp
        # integer defect labels are intentionally not inferred from a soft graph.
        return {}

    nbr_ids, _nbr_vecs, _nbr_dists, _nbr_weights, _dist_by_pair = _graph_neighbor_state(frame, graph)
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
        allowed_set = set(int(x) for x in (allowed or []))
        counts: list[int] = []
        central_idx: list[int] = []
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

        for idx in range(frame.n_atoms):
            if int(frame.types[idx]) not in cset:
                continue
            central_idx.append(int(idx))
            k = 0
            local_shell: list[int] = []
            for nb in nbr_ids[idx]:
                if int(frame.types[int(nb)]) in nset:
                    k += 1
                    local_shell.append(int(nb))
            counts.append(int(k))
            if allowed_set and int(k) not in allowed_set:
                defective_idx.append(int(idx))
                defective_ids.append(int(frame.ids[idx]))
                defective_coord.append(int(k))
                shell_idx_set.add(int(idx))
                shell_id_set.add(int(frame.ids[idx]))
                for nb in local_shell:
                    shell_idx_set.add(int(nb))
                    shell_id_set.add(int(frame.ids[int(nb)]))
                if expected is not None and int(k) < int(expected):
                    under_idx.append(int(idx))
                    under_ids.append(int(frame.ids[idx]))
                    under_coord.append(int(k))
                if expected is not None and int(k) > int(expected):
                    over_idx.append(int(idx))
                    over_ids.append(int(frame.ids[idx]))
                    over_coord.append(int(k))

        n_central = int(len(counts))
        n_def = int(len(defective_ids))
        frac_def = float(n_def) / float(n_central) if n_central > 0 else float("nan")
        tol = float(getattr(cm, "defect_frac_tol", 0.0))
        hist: dict[int, int] = {}
        for k in counts:
            hist[int(k)] = int(hist.get(int(k), 0) + 1)
        name = f"coord_{cm.central}-{cm.neighbor}"
        out[name] = {
            "expected": int(expected) if expected is not None else None,
            "allowed": [int(x) for x in allowed] if allowed is not None else None,
            "graph_rule": graph.graph_rule.to_json(),
            "structure_hash": str(graph.structure_hash),
            "n_central": int(n_central),
            "n_defective": int(n_def),
            "defect_fraction": float(frac_def),
            "defect_frac_tol": float(tol),
            "has_defect": bool(math.isfinite(frac_def) and allowed_set and frac_def > float(tol)),
            "coord_hist": {str(int(k)): int(v) for k, v in sorted(hist.items())},
            "defective_ids": defective_ids,
            "defective_idx": defective_idx,
            "defective_coord": defective_coord,
            "shell_ids": sorted([int(x) for x in shell_id_set]),
            "shell_idx": sorted([int(x) for x in shell_idx_set]),
            "under_ids": under_ids,
            "under_idx": under_idx,
            "under_coord": under_coord,
            "over_ids": over_ids,
            "over_idx": over_idx,
            "over_coord": over_coord,
        }

    return out


def compute_coordination_defect_details(
    frame: DumpFrame,
    metrics: StructureMetricsConfig,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]] = None,
    graph: Optional[StructureGraph] = None,
) -> Dict[str, Dict[str, object]]:
    """Coordination defect details.

    When an explicit hard graph is supplied, its graph-rule parameters provide
    the cutoffs used for the detailed shell/artefact report and its provenance is
    attached to every detail payload.
    """

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
    graph_rule_payload = None
    graph_structure_hash = None
    if graph is not None:
        if graph.is_soft:
            return {}
        graph_rule_payload = graph.graph_rule.to_json()
        graph_structure_hash = str(graph.structure_hash)
        graph_cutoffs = pair_cutoffs_from_parameters(graph.graph_rule.parameters)
        if graph_cutoffs:
            cutoffs = graph_cutoffs
    if not cutoffs:
        return {}

    pbc = frame_pbc(frame)
    invH = np.linalg.inv(frame.cell)
    frac = _wrap_frac((frame.positions - frame.origin) @ invH, pbc=pbc)
    posw = frame.origin + frac @ frame.cell

    max_cut = float(max(cutoffs.values()))
    atoms = Atoms(numbers=np.ones(frame.n_atoms, dtype=int), positions=posw, cell=frame.cell, pbc=pbc)
    ii, jj = neighbor_list("ij", atoms, max_cut)
    ii, jj, _vec, dist = _canonical_unique_mic_pairs(
        frac, frame.cell, ii, jj, cutoff=float(max_cut), pbc=pbc
    )

    # neighbour pair cutoffs
    nbr_ids: List[List[int]] = [[] for _ in range(frame.n_atoms)]
    # pair cutoff diagnostics
    ii_all = np.asarray(ii, dtype=int)
    jj_all = np.asarray(jj, dtype=int)
    dist_all = np.asarray([], dtype=float)
    t_i_all = np.asarray([], dtype=int)
    t_j_all = np.asarray([], dtype=int)
    if ii.size > 0:
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
            "graph_rule": graph_rule_payload,
            "structure_hash": graph_structure_hash,
            "legacy_single_cutoff": bool(graph_rule_payload is None),
            "single_rule_output": bool(graph_rule_payload is not None),
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
        overlap = sorted(set(node_types) & set(bridge_types))
        if overlap:
            raise ValueError(
                "projected ring node and bridge selectors must resolve to disjoint atom types; "
                f"overlap={overlap}"
            )

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
        cycle_size_divisor = 1
        minimum_reported_size = 3
    else:
        # Preserve bridge identity and enumerate the original alternating
        # bipartite cycles.  A one-mode clique projection turns a single
        # multicoordinate bridge into false node-only triangles and collapses
        # distinct bridges between the same node pair.
        nodes = [i for i in allowed_nodes if int(frame.types[i]) in set(node_types)]
        bridges = [i for i in allowed_nodes if int(frame.types[i]) in set(bridge_types)]
        Gp = nx.Graph()
        Gp.add_nodes_from(nodes, bipartite=0)
        Gp.add_nodes_from(bridges, bipartite=1)
        nodes_set = set(nodes)
        bridges_set = set(bridges)
        for b in bridges:
            for n in G.neighbors(b):
                if n in nodes_set and b in bridges_set:
                    Gp.add_edge(int(n), int(b))
        Gb = Gp
        cycle_size_divisor = 2
        # Two distinct bridges joining the same two network nodes form a
        # legitimate two-member alternating ring and must not be collapsed.
        minimum_reported_size = 2

    # ring enumeration algorithm
    if getattr(ring, "algorithm", "cycle_basis") == "cycle_basis":
        cycles = nx.cycle_basis(Gb)
    else:
        cycles = _primitive_rings(
            Gb,
            max_cycle_size=int(ring.max_cycle_size) * int(cycle_size_divisor),
            max_paths_per_edge=int(getattr(ring, "max_paths_per_edge", 16)),
        )

    sizes: list[int] = []
    for cycle in cycles:
        raw_size = int(len(cycle))
        if raw_size % int(cycle_size_divisor) != 0:
            continue
        size = raw_size // int(cycle_size_divisor)
        if minimum_reported_size <= size <= int(ring.max_cycle_size):
            sizes.append(int(size))
    out: Dict[str, float] = {}
    if not sizes:
        out["ring_count"] = 0.0
        out["ring_mean_size"] = float("nan")
        out["ring_entropy"] = 0.0
        for k in range(minimum_reported_size, int(ring.max_cycle_size) + 1):
            out[f"ring_frac_{k}"] = 0.0
        return out

    sizes_arr = np.asarray(sizes, dtype=int)
    out["ring_count"] = float(len(sizes))
    out["ring_mean_size"] = float(np.mean(sizes_arr))
    # histogram
    counts = {
        k: int(np.sum(sizes_arr == k))
        for k in range(minimum_reported_size, int(ring.max_cycle_size) + 1)
    }
    total = float(sum(counts.values()))
    entropy = 0.0
    for k in range(minimum_reported_size, int(ring.max_cycle_size) + 1):
        frac = float(counts.get(k, 0) / total) if total > 0 else 0.0
        out[f"ring_frac_{k}"] = frac
        if frac > 0.0:
            entropy -= frac * math.log(frac)
    out["ring_entropy"] = float(entropy)
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
