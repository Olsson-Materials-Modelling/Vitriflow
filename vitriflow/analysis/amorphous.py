from __future__ import annotations

"""Amorphous / crystallinity diagnostics for quenched boxes.

The production-convergence machinery checks reproducibility across boxes. This module
adds an orthogonal state-classification layer to detect crystal-like ordering after
quenching. The implementation combines:

- total S(q) peak sharpness,
- averaged local bond-order metrics / ordered-cluster fractions,
- optional crystal-reference peak fingerprints from Materials Project structures.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_prominences, peak_widths

try:
    from scipy.special import sph_harm  # sci py
except Exception:  # pragma: no cover - SciPy >= 1.15 compatibility
    from scipy.special import sph_harm_y as _sph_harm_y

    def sph_harm(m, n, theta, phi):
        # backwards compatible wrapper
        # signature theta azimuth
        return _sph_harm_y(n, m, phi, theta)

from .dump import DumpFrame
from .sq import compute_sq
from .trajectory import _atoms_to_dumpframe
from .motif_summary import summarize_production_crystal_motifs as _summarize_production_crystal_motifs

_REFERENCE_LIBRARY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula(formula: str) -> dict[str, int]:
    if formula is None or str(formula).strip() == "":
        raise ValueError("Empty formula")
    tokens = _FORMULA_RE.findall(str(formula).strip())
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula}")
    counts: dict[str, int] = {}
    for el, num in tokens:
        n = int(num) if num else 1
        counts[str(el)] = counts.get(str(el), 0) + int(n)
    return counts



if TYPE_CHECKING:  # pragma: no cover
    from ase import Atoms


def _require_ase():
    try:
        from ase import Atoms as ASEAtoms
        from ase.io import write as ase_write
        from ase.neighborlist import neighbor_list
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Amorphous/crystallinity analysis requires ASE") from exc
    return ASEAtoms, ase_write, neighbor_list


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if int(a) <= int(b) else (b, a)


def _wrap_frac(frac: np.ndarray) -> np.ndarray:
    x = np.asarray(frac, dtype=float)
    return x - np.floor(x)


def _reduce_counts(counts: Mapping[str, int]) -> dict[str, int]:
    vals = [int(v) for v in counts.values() if int(v) > 0]
    if not vals:
        return {}
    g = int(vals[0])
    for v in vals[1:]:
        g = math.gcd(g, int(v))
    g = max(1, int(g))
    out: dict[str, int] = {}
    for k, v in counts.items():
        vi = int(v)
        if vi > 0:
            out[str(k)] = int(vi // g)
    return out


def _formula_from_counts(counts: Mapping[str, int]) -> str:
    # order conventional deterministic
    parts: list[str] = []
    keys = list(counts.keys())
    ordered: list[str]
    if "C" in keys:
        ordered = ["C"]
        if "H" in keys:
            ordered.append("H")
        ordered.extend(sorted(k for k in keys if k not in {"C", "H"}))
    else:
        ordered = sorted(keys)
    for k in ordered:
        n = int(counts[k])
        if n <= 0:
            continue
        parts.append(str(k))
        if n != 1:
            parts.append(str(n))
    return "".join(parts)


def reduced_formula_from_frame(frame: DumpFrame, *, type_to_species: Optional[Sequence[str]]) -> Optional[str]:
    if type_to_species is None:
        return None
    counts: dict[str, int] = {}
    for t in frame.types.tolist():
        ti = int(t)
        if ti < 1 or ti > len(type_to_species):
            return None
        s = str(type_to_species[ti - 1])
        counts[s] = counts.get(s, 0) + 1
    rc = _reduce_counts(counts)
    return _formula_from_counts(rc) if rc else None


def _normalized_composition_from_formula(formula: str) -> dict[str, int]:
    return _reduce_counts(_parse_formula(str(formula)))


def _normalized_composition_from_obj(obj: Any) -> Optional[dict[str, int]]:
    if obj is None:
        return None
    if hasattr(obj, "as_dict"):
        try:
            data = obj.as_dict()
            if isinstance(data, Mapping):
                out = {str(k): int(round(float(v))) for k, v in data.items() if float(v) > 0}
                return _reduce_counts(out)
        except Exception:
            pass
    if hasattr(obj, "get_el_amt_dict"):
        try:
            data = obj.get_el_amt_dict()
            if isinstance(data, Mapping):
                out = {str(k): int(round(float(v))) for k, v in data.items() if float(v) > 0}
                return _reduce_counts(out)
        except Exception:
            pass
    if isinstance(obj, Mapping):
        try:
            out = {str(k): int(round(float(v))) for k, v in obj.items() if float(v) > 0}
            return _reduce_counts(out)
        except Exception:
            pass
    s = str(getattr(obj, "reduced_formula", obj)).strip()
    if s != "":
        try:
            return _normalized_composition_from_formula(s)
        except Exception:
            return None
    return None


def _resolve_mp_api_key(ref_cfg) -> Optional[str]:
    if ref_cfg is None or not bool(getattr(ref_cfg, "enabled", False)):
        return None
    val = getattr(ref_cfg, "mp_api_key", None)
    if val is not None and str(val).strip() != "":
        return str(val).strip()
    env_name = str(getattr(ref_cfg, "mp_api_key_env", "") or "").strip()
    if env_name != "":
        env_val = os.environ.get(env_name)
        if env_val is not None and str(env_val).strip() != "":
            return str(env_val).strip()
    return None


def _message(progress: Any, level: str, message: str) -> None:
    if progress is None:
        return
    meth = getattr(progress, str(level), None)
    if callable(meth):
        try:
            meth("amorphous", str(message))
            return
        except TypeError:
            pass
    try:
        progress.info("amorphous", str(message))
    except Exception:
        pass


def _scalar_or_nan(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _smooth_signal(y: np.ndarray, w: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    ww = int(max(1, w))
    if ww % 2 == 0:
        ww += 1
    if ww <= 1 or arr.size < ww:
        return arr.copy()
    ker = np.ones(ww, dtype=float) / float(ww)
    return np.convolve(np.nan_to_num(arr, nan=0.0), ker, mode="same")


def _sq_peak_features(
    q: np.ndarray,
    s: np.ndarray,
    *,
    smooth: int,
    peak_search: tuple[float, float],
    prominence_min: float,
    height_min: float,
) -> list[dict[str, float]]:
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)
    if q.ndim != 1 or s.ndim != 1 or q.size != s.size or q.size < 8:
        return []
    if not np.all(np.diff(q) > 0):
        return []
    ss = _smooth_signal(s, int(smooth))
    q0, q1 = float(peak_search[0]), float(peak_search[1])
    mask = (q >= q0) & (q <= q1) & np.isfinite(ss)
    if not np.any(mask):
        return []
    idxs = np.where(mask)[0]
    work = np.asarray(ss[idxs], dtype=float)
    peaks, props = find_peaks(work, prominence=float(prominence_min), height=float(height_min))
    if peaks.size == 0:
        return []
    prom = peak_prominences(work, peaks)[0]
    widths_idx = peak_widths(work, peaks, rel_height=0.5)[0]
    dq = float(np.mean(np.diff(q[idxs]))) if idxs.size > 1 else 1.0
    widths_q = np.asarray(widths_idx, dtype=float) * float(dq)
    out: list[dict[str, float]] = []
    for p, pr, wq in zip(peaks.tolist(), prom.tolist(), widths_q.tolist()):
        gi = int(idxs[int(p)])
        qq = float(q[gi])
        hh = float(ss[gi])
        wf = float(max(float(wq), 1.0e-12))
        sharp = float(pr) / wf
        out.append(
            {
                "q": qq,
                "height": hh,
                "prominence": float(pr),
                "fwhm": wf,
                "sharpness": sharp,
            }
        )
    out.sort(key=lambda x: (x["q"], -x["prominence"]))
    return out


def _normalised_peak_weights(peaks: Sequence[Mapping[str, float]]) -> np.ndarray:
    w = np.asarray([max(0.0, float(p.get("prominence", 0.0))) for p in peaks], dtype=float)
    if w.size == 0:
        return w
    if not np.any(w > 0.0):
        w = np.ones((w.size,), dtype=float)
    s = float(np.sum(w))
    return (w / s) if s > 0.0 else np.full((w.size,), 1.0 / float(w.size), dtype=float)


def _peak_match_tolerance(box_peak: Mapping[str, float], ref_peak: Mapping[str, float], *, q_tol: float) -> float:
    tol = max(float(q_tol), 1.0e-8)
    try:
        bw = float(box_peak.get("fwhm", float("nan")))
    except Exception:
        bw = float("nan")
    try:
        rw = float(ref_peak.get("fwhm", float("nan")))
    except Exception:
        rw = float("nan")
    if math.isfinite(bw):
        tol = max(tol, 0.5 * abs(float(bw)))
    if math.isfinite(rw):
        tol = max(tol, 0.5 * abs(float(rw)))
    return max(tol, 1.0e-8)


def _peak_width_similarity(box_peak: Mapping[str, float], ref_peak: Mapping[str, float]) -> float:
    """Peak width similarity."""

    try:
        bw = float(box_peak.get("fwhm", float("nan")))
    except Exception:
        bw = float("nan")
    try:
        rw = float(ref_peak.get("fwhm", float("nan")))
    except Exception:
        rw = float("nan")
    if not (math.isfinite(bw) and math.isfinite(rw) and bw > 1.0e-12 and rw > 1.0e-12):
        return 1.0
    ratio = min(float(bw), float(rw)) / max(float(bw), float(rw))
    # penalise mismatches strongly
    # penalising modest broadening
    return float(min(1.0, max(0.0, math.sqrt(max(ratio, 0.0)))))


def _matched_peak_support_threshold(n_box: int, n_ref: int) -> float:
    """Matched peak support."""

    denom = max(1, int(max(n_box, n_ref)))
    # historical fingerprints scaling
    # peaks fingerprints accumulate
    # significant individually peaks
    return float(min(0.10, 0.75 / float(denom)))


def _discriminative_support_threshold(min_support: float) -> float:
    """Discriminative support threshold."""

    return float(max(0.15, 2.0 * float(min_support)))


def _reference_peak_overlap_details(
    box_peaks: Sequence[Mapping[str, float]],
    ref_peaks: Sequence[Mapping[str, float]],
    *,
    q_tol: float,
) -> dict[str, Any]:
    """Reference peak overlap."""

    n_box = int(len(box_peaks))
    n_ref = int(len(ref_peaks))
    out: dict[str, Any] = {
        "overlap": float("nan") if n_ref == 0 else 0.0,
        "box_peak_count": int(n_box),
        "ref_peak_count": int(n_ref),
        "matched_pairs": 0,
        "matched_significant_pairs": 0,
        "matched_support_weight": 0.0,
        "matched_significant_support_weight": 0.0,
        "matched_pairs_info": [],
        "significant_support_threshold": float("nan"),
        "significant_quality_threshold": float("nan"),
        "discriminative_min_significant_pairs": 2,
        "discriminative_min_significant_support_weight": float("nan"),
        "discriminative": False,
        "reason": None,
    }
    if n_ref == 0:
        out["reason"] = "no_reference_peaks"
        return out
    if n_box == 0:
        out["reason"] = "no_box_peaks"
        return out

    box_w = _normalised_peak_weights(box_peaks)
    ref_w = _normalised_peak_weights(ref_peaks)
    if box_w.size == 0 or ref_w.size == 0:
        out["reason"] = "empty_peak_weights"
        return out

    support = np.zeros((len(box_peaks), len(ref_peaks)), dtype=float)
    quality = np.zeros((len(box_peaks), len(ref_peaks)), dtype=float)
    score = np.zeros((len(box_peaks), len(ref_peaks)), dtype=float)
    for i, bp in enumerate(box_peaks):
        try:
            bq = float(bp.get("q", float("nan")))
        except Exception:
            bq = float("nan")
        if not math.isfinite(bq):
            continue
        for j, rp in enumerate(ref_peaks):
            try:
                rq = float(rp.get("q", float("nan")))
            except Exception:
                rq = float("nan")
            if not math.isfinite(rq):
                continue
            tol_ij = _peak_match_tolerance(bp, rp, q_tol=q_tol)
            g = math.exp(-0.5 * ((float(bq) - float(rq)) / float(tol_ij)) ** 2)
            shape = _peak_width_similarity(bp, rp)
            s_ij = float(min(float(box_w[i]), float(ref_w[j])))
            q_ij = float(g) * float(shape)
            support[i, j] = float(s_ij)
            quality[i, j] = float(q_ij)
            score[i, j] = float(s_ij) * float(q_ij)

    if score.size == 0 or float(np.max(score)) <= 0.0:
        out["reason"] = "no_positive_matches"
        return out

    row_ind, col_ind = linear_sum_assignment(-score)
    matched_info: list[dict[str, Any]] = []
    matched = 0.0
    matched_support = 0.0
    matched_support_sig = 0.0
    matched_pairs = 0
    matched_sig = 0
    # significance thresholds fraction
    # peak distributions geometrically
    min_support = _matched_peak_support_threshold(n_box=n_box, n_ref=n_ref)
    min_quality = 0.50
    min_support_total = _discriminative_support_threshold(min_support)
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        sc = float(score[int(i), int(j)])
        sp = float(support[int(i), int(j)])
        qv = float(quality[int(i), int(j)])
        if sc <= 0.0:
            continue
        matched += float(sc)
        matched_support += float(sp)
        matched_pairs += 1
        significant = bool(sp >= float(min_support) and qv >= float(min_quality))
        if significant:
            matched_sig += 1
            matched_support_sig += float(sp)
        matched_info.append(
            {
                "box_index": int(i),
                "ref_index": int(j),
                "score": float(sc),
                "support_weight": float(sp),
                "quality": float(qv),
                "significant": bool(significant),
            }
        )

    out.update(
        {
            "overlap": float(min(1.0, max(0.0, matched))),
            "matched_pairs": int(matched_pairs),
            "matched_significant_pairs": int(matched_sig),
            "matched_support_weight": float(matched_support),
            "matched_significant_support_weight": float(matched_support_sig),
            "matched_pairs_info": matched_info,
            "significant_support_threshold": float(min_support),
            "significant_quality_threshold": float(min_quality),
            "discriminative_min_significant_pairs": 2,
            "discriminative_min_significant_support_weight": float(min_support_total),
        }
    )
    # crystalline fingerprint independently
    # matched peaks cumulative
    # otherwise match advisory
    discriminative = bool(
        n_box >= 2
        and n_ref >= 2
        and matched_sig >= 2
        and float(matched_support_sig) >= float(min_support_total)
    )
    out["discriminative"] = bool(discriminative)
    if discriminative:
        out["reason"] = None
    elif n_box < 2 or n_ref < 2 or matched_sig < 2:
        out["reason"] = "insufficient_matched_peak_support"
    else:
        out["reason"] = "insufficient_cumulative_peak_support"
    return out


def _reference_peak_overlap(

    box_peaks: Sequence[Mapping[str, float]],
    ref_peaks: Sequence[Mapping[str, float]],
    *,
    q_tol: float,
) -> float:
    """Reference peak overlap."""

    info = _reference_peak_overlap_details(box_peaks=box_peaks, ref_peaks=ref_peaks, q_tol=q_tol)
    return float(info.get("overlap", float("nan")))


def _qbar_similarity(box_qbar: Mapping[str, float], ref_qbar: Mapping[str, float]) -> float:
    """Qbar similarity."""

    keys = sorted(set(str(k) for k in box_qbar.keys()) & set(str(k) for k in ref_qbar.keys()))
    if len(keys) == 0:
        return float("nan")
    rel_err2: list[float] = []
    for key in keys:
        try:
            xb = float(box_qbar.get(key, float("nan")))
            xr = float(ref_qbar.get(key, float("nan")))
        except Exception:
            continue
        if not (math.isfinite(xb) and math.isfinite(xr)):
            continue
        scale = max(abs(xb), abs(xr), 1.0e-8)
        rel_err2.append(((xb - xr) / scale) ** 2)
    if len(rel_err2) == 0:
        return float("nan")
    d = math.sqrt(float(np.mean(np.asarray(rel_err2, dtype=float))))
    return float(1.0 / (1.0 + d))


def _motif_overlap_thresholds(amorph_cfg) -> tuple[float, float]:
    """Motif overlap thresholds."""

    detected = float(getattr(amorph_cfg, "max_reference_peak_overlap", 0.65))
    if not math.isfinite(detected):
        detected = 0.65
    detected = float(min(1.0, max(0.0, detected)))
    candidate = float(max(0.10, min(detected, 0.5 * detected)))
    return candidate, detected


def _rank_reference_motifs(
    *,
    peaks: Sequence[Mapping[str, float]],
    local_order: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    amorph_cfg,
) -> list[dict[str, Any]]:
    """Rank reference motifs."""

    q_tol = float(getattr(amorph_cfg, "reference_peak_match_tol", 0.20))
    cand_thr, det_thr = _motif_overlap_thresholds(amorph_cfg)
    box_qbar = dict(local_order.get("qbar", {}) or {})
    ranked: list[dict[str, Any]] = []
    for ref in list(refs or []):
        ref_peaks = list((((ref.get("sq", {}) or {}).get("peaks", [])) or []))
        ov_info = _reference_peak_overlap_details(box_peaks=peaks, ref_peaks=ref_peaks, q_tol=q_tol)
        ov = float(ov_info.get("overlap", float("nan")))
        ref_qbar = dict((((ref.get("local_order", {}) or {}).get("qbar", {})) or {}))
        qsim = _qbar_similarity(box_qbar, ref_qbar)
        # peak fingerprint qbar
        # similarity acts breaker
        score = float(ov) if not math.isfinite(qsim) else float(0.80 * float(ov) + 0.20 * float(qsim))
        candidate = bool(math.isfinite(ov) and float(ov) >= cand_thr)
        detected = bool(math.isfinite(ov) and float(ov) >= det_thr and bool(ov_info.get("discriminative", False)))
        ranked.append(
            {
                "material_id": str(ref.get("material_id", "") or ""),
                "formula_pretty": str(ref.get("formula_pretty", "") or ""),
                "energy_above_hull": ref.get("energy_above_hull", None),
                "peak_overlap": float(ov),
                "qbar_similarity": (float(qsim) if math.isfinite(qsim) else None),
                "motif_score": float(score),
                "candidate": bool(candidate),
                "detected": bool(detected),
                "reference_discriminative": bool(ov_info.get("discriminative", False)),
                "matched_pairs": int(ov_info.get("matched_pairs", 0)),
                "matched_significant_pairs": int(ov_info.get("matched_significant_pairs", 0)),
                "matched_significant_support_weight": float(ov_info.get("matched_significant_support_weight", 0.0)),
                "box_peak_count": int(ov_info.get("box_peak_count", len(peaks))),
                "ref_peak_count": int(ov_info.get("ref_peak_count", len(ref_peaks))),
                "significant_support_threshold": float(ov_info.get("significant_support_threshold", float("nan"))),
                "significant_quality_threshold": float(ov_info.get("significant_quality_threshold", float("nan"))),
                "discriminative_min_significant_support_weight": float(
                    ov_info.get("discriminative_min_significant_support_weight", float("nan"))
                ),
                "match_reason": ov_info.get("reason", None),
            }
        )
    ranked.sort(
        key=lambda x: (
            float(x.get("detected", False)),
            float(x.get("candidate", False)),
            float(x.get("motif_score", float("-inf"))),
            float(x.get("peak_overlap", float("-inf"))),
            -float(x.get("energy_above_hull", 0.0) if x.get("energy_above_hull") is not None else 0.0),
            str(x.get("material_id", "")),
        ),
        reverse=True,
    )
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = int(idx)
    return ranked


def summarize_production_crystal_motifs(
    accepted_boxes: Sequence[Mapping[str, Any]],
    *,
    rejected_boxes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Production crystal motifs."""

    return _summarize_production_crystal_motifs(accepted_boxes, rejected_boxes=rejected_boxes)


def _directed_neighbors(

    frame: DumpFrame,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
) -> tuple[list[list[int]], list[list[np.ndarray]], list[tuple[int, int]]]:
    if not cutoffs:
        return [[] for _ in range(frame.n_atoms)], [[] for _ in range(frame.n_atoms)], []
    invH = np.linalg.inv(frame.cell)
    frac = _wrap_frac((frame.positions - frame.origin) @ invH)
    posw = frame.origin + frac @ frame.cell
    max_cut = float(max(float(v) for v in cutoffs.values()))
    ASEAtoms, _ase_write, neighbor_list = _require_ase()
    atoms = ASEAtoms(numbers=np.ones(frame.n_atoms, dtype=int), positions=posw, cell=frame.cell, pbc=True)
    ii, jj = neighbor_list("ij", atoms, max_cut)
    if ii.size == 0:
        return [[] for _ in range(frame.n_atoms)], [[] for _ in range(frame.n_atoms)], []
    m = ii < jj
    ii = ii[m]
    jj = jj[m]
    if ii.size == 0:
        return [[] for _ in range(frame.n_atoms)], [[] for _ in range(frame.n_atoms)], []
    dfrac = frac[jj] - frac[ii]
    dfrac -= np.round(dfrac)
    dvec = dfrac @ frame.cell
    dist = np.linalg.norm(dvec, axis=1)
    nbr_ids: list[list[int]] = [[] for _ in range(frame.n_atoms)]
    nbr_vecs: list[list[np.ndarray]] = [[] for _ in range(frame.n_atoms)]
    edges: list[tuple[int, int]] = []
    for a, b, vec, d in zip(ii.tolist(), jj.tolist(), dvec.tolist(), dist.tolist()):
        key = _pair_key(int(frame.types[a]), int(frame.types[b]))
        cut = cutoffs.get(key, None)
        if cut is None or float(d) > float(cut):
            continue
        vv = np.asarray(vec, dtype=float)
        nbr_ids[int(a)].append(int(b))
        nbr_vecs[int(a)].append(vv)
        nbr_ids[int(b)].append(int(a))
        nbr_vecs[int(b)].append(-vv)
        edges.append((int(a), int(b)))
    return nbr_ids, nbr_vecs, edges


def _qlm_from_nbr_vecs(nbr_vecs: Sequence[Sequence[np.ndarray]], l: int) -> np.ndarray:
    n_atoms = int(len(nbr_vecs))
    out = np.zeros((n_atoms, 2 * int(l) + 1), dtype=complex)
    for i, vecs in enumerate(nbr_vecs):
        if len(vecs) == 0:
            continue
        arr = np.asarray(vecs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        norms = np.linalg.norm(arr, axis=1)
        m = norms > 1.0e-12
        if not np.any(m):
            continue
        u = arr[m] / norms[m, None]
        theta = np.arccos(np.clip(u[:, 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2.0 * math.pi)
        vals = np.zeros((u.shape[0], 2 * int(l) + 1), dtype=complex)
        for idx, mm in enumerate(range(-int(l), int(l) + 1)):
            vals[:, idx] = sph_harm(int(mm), int(l), phi, theta)
        out[i] = np.mean(vals, axis=0)
    return out


def _average_qlm(qlm: np.ndarray, nbr_ids: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros_like(qlm, dtype=complex)
    for i, nbrs in enumerate(nbr_ids):
        if len(nbrs) == 0:
            out[i] = qlm[i]
            continue
        acc = np.array(qlm[i], dtype=complex)
        for j in nbrs:
            acc += qlm[int(j)]
        out[i] = acc / float(len(nbrs) + 1)
    return out


def _ql_scalar(qlm: np.ndarray, l: int) -> np.ndarray:
    fac = (4.0 * math.pi) / float(2 * int(l) + 1)
    return np.sqrt(np.maximum(0.0, fac * np.sum(np.abs(qlm) ** 2, axis=1).real))



def _extract_qbar_stat(qbar: Mapping[str, Any], *, l: int, names: Sequence[str]) -> Optional[float]:
    for name in list(names or []):
        key = f"qbar_{int(l)}_{str(name)}"
        try:
            val = float(qbar.get(key, float("nan")))
        except Exception:
            val = float("nan")
        if math.isfinite(val) and val > 0.0:
            return float(val)
    return None


def _resolve_local_order_calibration(
    *,
    reference_refs: Optional[Sequence[Mapping[str, Any]]],
    amorph_cfg,
    l_solid: int,
) -> dict[str, Any]:
    """Local order calibration."""

    default_scale = 0.18 if int(l_solid) == 6 else 0.12
    scale_vals: list[float] = []
    floor_vals: list[float] = []
    for ref in list(reference_refs or []):
        if not isinstance(ref, Mapping):
            continue
        qbar = dict((((ref.get("local_order", {}) or {}).get("qbar", {})) or {}))
        scale = _extract_qbar_stat(qbar, l=int(l_solid), names=("p75", "p90", "p50", "mean"))
        floor = _extract_qbar_stat(qbar, l=int(l_solid), names=("p25", "p10", "p50", "mean"))
        if scale is not None:
            scale_vals.append(float(scale))
        if floor is not None:
            floor_vals.append(float(floor))
    if len(scale_vals) > 0:
        scale = float(np.median(np.asarray(scale_vals, dtype=float)))
        source = "reference"
    else:
        scale = float(default_scale)
        source = "default"
    if len(floor_vals) > 0:
        floor_ref = float(np.median(np.asarray(floor_vals, dtype=float)))
        floor = float(min(scale, max(1.0e-6, 0.80 * floor_ref)))
    else:
        floor = float(max(1.0e-6, 0.60 * scale))
    return {
        "qbar_scale": float(scale),
        "qbar_floor": float(floor),
        "source": str(source),
    }


def _local_order_analysis(
    frame: DumpFrame,
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    amorph_cfg,
    reference_refs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    nbr_ids, nbr_vecs, edges = _directed_neighbors(frame, cutoffs=cutoffs)
    n_atoms = int(frame.n_atoms)
    degrees = np.asarray([len(v) for v in nbr_ids], dtype=int)
    l_values = [int(x) for x in list(getattr(amorph_cfg, "l_values", [4, 6]))]
    qbar_by_l: dict[int, np.ndarray] = {}
    qbarlm_by_l: dict[int, np.ndarray] = {}
    for l in l_values:
        qlm = _qlm_from_nbr_vecs(nbr_vecs, int(l))
        qbarlm = _average_qlm(qlm, nbr_ids)
        qbarlm_by_l[int(l)] = qbarlm
        qbar_by_l[int(l)] = _ql_scalar(qbarlm, int(l))

    l_solid = int(getattr(amorph_cfg, "solid_like_l", 6))
    qbarlm_s = qbarlm_by_l.get(l_solid)
    if qbarlm_s is None:
        qbarlm_s = _average_qlm(_qlm_from_nbr_vecs(nbr_vecs, l_solid), nbr_ids)
        qbar_by_l[int(l_solid)] = _ql_scalar(qbarlm_s, l_solid)
    qbar_s = np.asarray(qbar_by_l.get(int(l_solid), _ql_scalar(qbarlm_s, l_solid)), dtype=float)

    norms = np.linalg.norm(qbarlm_s, axis=1)
    qhat = np.zeros_like(qbarlm_s, dtype=complex)
    m = norms > 1.0e-18
    qhat[m] = qbarlm_s[m] / norms[m, None]

    calib = _resolve_local_order_calibration(reference_refs=reference_refs, amorph_cfg=amorph_cfg, l_solid=l_solid)
    solid_qbar_scale = float(calib.get("qbar_scale", float("nan")))
    solid_qbar_floor = float(calib.get("qbar_floor", float("nan")))

    solid_thr = float(getattr(amorph_cfg, "solid_like_bond_threshold", 0.5))
    solid_counts = np.zeros((n_atoms,), dtype=int)
    solid_edges: list[tuple[int, int, float]] = []
    for a, b in edges:
        qa = float(qbar_s[int(a)]) if int(a) < qbar_s.size else float("nan")
        qb = float(qbar_s[int(b)]) if int(b) < qbar_s.size else float("nan")
        amp = min(qa, qb)
        if not math.isfinite(amp) or amp < float(solid_qbar_floor):
            continue
        va = qhat[int(a)]
        vb = qhat[int(b)]
        if not (np.any(np.abs(va) > 0.0) and np.any(np.abs(vb) > 0.0)):
            continue
        corr_raw = float(np.real(np.vdot(va, vb)))
        if math.isfinite(solid_qbar_scale) and float(solid_qbar_scale) > 1.0e-12:
            amp_factor = min(1.0, max(0.0, float(amp) / float(solid_qbar_scale)))
        else:
            amp_factor = 1.0
        corr = float(corr_raw) * float(amp_factor)
        if corr >= solid_thr:
            solid_counts[int(a)] += 1
            solid_counts[int(b)] += 1
            solid_edges.append((int(a), int(b), corr))

    ordered_min_neighbors = int(getattr(amorph_cfg, "ordered_min_neighbors", 3))
    ordered_min_fraction = float(getattr(amorph_cfg, "ordered_min_fraction", 0.6))
    ordered = np.zeros((n_atoms,), dtype=bool)
    for i in range(n_atoms):
        deg = int(degrees[i])
        if deg <= 0:
            continue
        qi = float(qbar_s[int(i)]) if int(i) < qbar_s.size else float("nan")
        if not math.isfinite(qi) or qi < float(solid_qbar_floor):
            continue
        req = max(int(ordered_min_neighbors), int(math.ceil(float(ordered_min_fraction) * float(deg))))
        ordered[i] = bool(int(solid_counts[i]) >= int(req))

    G = nx.Graph()
    G.add_nodes_from([int(i) for i in range(n_atoms) if bool(ordered[i])])
    for a, b, _c in solid_edges:
        if bool(ordered[int(a)]) and bool(ordered[int(b)]):
            G.add_edge(int(a), int(b))
    largest_cluster = 0
    if G.number_of_nodes() > 0:
        try:
            largest_cluster = max(len(c) for c in nx.connected_components(G))
        except Exception:
            largest_cluster = 0

    qbar_stats: dict[str, float] = {}
    for l in sorted(qbar_by_l):
        arr = np.asarray(qbar_by_l[int(l)], dtype=float)
        if arr.size == 0:
            qbar_stats[f"qbar_{int(l)}_mean"] = float("nan")
            qbar_stats[f"qbar_{int(l)}_std"] = float("nan")
            qbar_stats[f"qbar_{int(l)}_p10"] = float("nan")
            qbar_stats[f"qbar_{int(l)}_p25"] = float("nan")
            qbar_stats[f"qbar_{int(l)}_p50"] = float("nan")
            qbar_stats[f"qbar_{int(l)}_p75"] = float("nan")
            qbar_stats[f"qbar_{int(l)}_p90"] = float("nan")
        else:
            qbar_stats[f"qbar_{int(l)}_mean"] = float(np.mean(arr))
            qbar_stats[f"qbar_{int(l)}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            qbar_stats[f"qbar_{int(l)}_p10"] = float(np.quantile(arr, 0.10))
            qbar_stats[f"qbar_{int(l)}_p25"] = float(np.quantile(arr, 0.25))
            qbar_stats[f"qbar_{int(l)}_p50"] = float(np.quantile(arr, 0.50))
            qbar_stats[f"qbar_{int(l)}_p75"] = float(np.quantile(arr, 0.75))
            qbar_stats[f"qbar_{int(l)}_p90"] = float(np.quantile(arr, 0.90))

    crystalline_fraction = float(np.sum(ordered)) / float(n_atoms) if n_atoms > 0 else float("nan")
    largest_cluster_fraction = float(largest_cluster) / float(n_atoms) if n_atoms > 0 else float("nan")
    return {
        "n_atoms": int(n_atoms),
        "degrees_mean": float(np.mean(degrees)) if degrees.size > 0 else float("nan"),
        "degrees_median": float(np.median(degrees)) if degrees.size > 0 else float("nan"),
        "solid_like_bonds": int(len(solid_edges)),
        "solid_like_threshold": float(solid_thr),
        "solid_like_qbar_scale": float(solid_qbar_scale),
        "solid_like_qbar_floor": float(solid_qbar_floor),
        "solid_like_calibration_source": str(calib.get("source", "default")),
        "ordered_atoms": int(np.sum(ordered)),
        "crystalline_fraction": float(crystalline_fraction),
        "largest_cluster": int(largest_cluster),
        "largest_cluster_fraction": float(largest_cluster_fraction),
        "ordered_atom_indices": [int(i) for i in np.where(ordered)[0].tolist()],
        "solid_like_counts": [int(x) for x in solid_counts.tolist()],
        "qbar": qbar_stats,
    }


def _finite_mean(values: Sequence[Any]) -> float:
    arr = np.asarray([
        float(x) for x in list(values or []) if x is not None and math.isfinite(_scalar_or_nan(x))
    ], dtype=float)
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def _aggregate_qbar_stats(local_reports: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys: set[str] = set()
    for rep in list(local_reports or []):
        keys.update(str(k) for k in dict(rep.get("qbar", {}) or {}).keys())
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals: list[float] = []
        for rep in list(local_reports or []):
            try:
                val = float(dict(rep.get("qbar", {}) or {}).get(key, float("nan")))
            except Exception:
                val = float("nan")
            if math.isfinite(val):
                vals.append(float(val))
        out[str(key)] = float(np.mean(np.asarray(vals, dtype=float))) if len(vals) > 0 else float("nan")
    return out


def _aggregate_local_order_reports(local_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reports = [dict(rep) for rep in list(local_reports or []) if isinstance(rep, Mapping)]
    if len(reports) == 0:
        raise ValueError("_aggregate_local_order_reports requires at least one local-order report")

    out = dict(reports[-1])
    cf_vals = [rep.get("crystalline_fraction", None) for rep in reports]
    lcf_vals = [rep.get("largest_cluster_fraction", None) for rep in reports]
    ordered_vals = [rep.get("ordered_atoms", None) for rep in reports]
    largest_vals = [rep.get("largest_cluster", None) for rep in reports]
    solid_bond_vals = [rep.get("solid_like_bonds", None) for rep in reports]

    out["crystalline_fraction"] = _finite_mean(cf_vals)
    out["largest_cluster_fraction"] = _finite_mean(lcf_vals)

    ordered_mean = _finite_mean(ordered_vals)
    largest_mean = _finite_mean(largest_vals)
    solid_bonds_mean = _finite_mean(solid_bond_vals)
    out["ordered_atoms"] = int(round(ordered_mean)) if math.isfinite(ordered_mean) else 0
    out["largest_cluster"] = int(round(largest_mean)) if math.isfinite(largest_mean) else 0
    out["solid_like_bonds"] = int(round(solid_bonds_mean)) if math.isfinite(solid_bonds_mean) else 0
    out["ordered_atoms_mean"] = float(ordered_mean)
    out["largest_cluster_mean"] = float(largest_mean)
    out["solid_like_bonds_mean"] = float(solid_bonds_mean)

    for key in (
        "degrees_mean",
        "degrees_median",
        "solid_like_threshold",
        "solid_like_qbar_scale",
        "solid_like_qbar_floor",
    ):
        out[key] = _finite_mean([rep.get(key, None) for rep in reports])

    sources = [str(rep.get("solid_like_calibration_source", "")) for rep in reports if str(rep.get("solid_like_calibration_source", "")) != ""]
    out["solid_like_calibration_source"] = sources[-1] if len(set(sources)) > 1 else (sources[0] if sources else str(out.get("solid_like_calibration_source", "default")))
    out["qbar"] = _aggregate_qbar_stats(reports)
    out["aggregation"] = {
        "policy": "mean_over_tail_frames",
        "n_frames": int(len(reports)),
        "representative_frame": "last",
        "ordered_atom_indices_frame_policy": "last",
        "solid_like_counts_frame_policy": "last",
        "crystalline_fraction_per_frame": [
            (float(v) if math.isfinite(_scalar_or_nan(v)) else None) for v in cf_vals
        ],
        "largest_cluster_fraction_per_frame": [
            (float(v) if math.isfinite(_scalar_or_nan(v)) else None) for v in lcf_vals
        ],
        "ordered_atoms_per_frame": [
            (int(round(float(v))) if math.isfinite(_scalar_or_nan(v)) else None) for v in ordered_vals
        ],
        "largest_cluster_per_frame": [
            (int(round(float(v))) if math.isfinite(_scalar_or_nan(v)) else None) for v in largest_vals
        ],
        "solid_like_bonds_per_frame": [
            (int(round(float(v))) if math.isfinite(_scalar_or_nan(v)) else None) for v in solid_bond_vals
        ],
    }
    return out


def _local_order_analysis_timeavg(
    frames: Sequence[DumpFrame],
    *,
    cutoffs: Mapping[Tuple[int, int], float],
    amorph_cfg,
    reference_refs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    reports = [
        _local_order_analysis(fr, cutoffs=cutoffs, amorph_cfg=amorph_cfg, reference_refs=reference_refs)
        for fr in list(frames or [])
    ]
    return _aggregate_local_order_reports(reports)


def _best_reference_match_by_overlap(matches: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    best: Optional[dict[str, Any]] = None
    best_ov = float("nan")
    for item in list(matches or []):
        if not isinstance(item, Mapping):
            continue
        try:
            ov = float(item.get("peak_overlap", float("nan")))
        except Exception:
            ov = float("nan")
        if not math.isfinite(ov):
            continue
        if best is None or float(ov) > float(best_ov):
            best = dict(item)
            best_ov = float(ov)
    return best


def _best_discriminative_reference_match(matches: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    discr = [item for item in list(matches or []) if isinstance(item, Mapping) and bool(item.get("reference_discriminative", False))]
    return _best_reference_match_by_overlap(discr)


def _resolve_sq_params(metrics_cfg, amorph_cfg) -> dict[str, Any]:
    for sm in list(getattr(metrics_cfg, "sq", []) or []):
        if getattr(sm, "pair", None) is None:
            return {
                "q_max": float(getattr(sm, "q_max", 20.0)),
                "nq": int(getattr(sm, "nq", 400)),
                "r_max": float(getattr(sm, "r_max", 10.0)),
                "nbins": int(getattr(sm, "nbins", 800)),
                "window": str(getattr(sm, "window", "lorch")),
                "smooth": int(getattr(sm, "smooth", getattr(amorph_cfg, "smooth", 7))),
                "peak_search": tuple(getattr(sm, "peak_search", getattr(amorph_cfg, "peak_search", (0.5, 12.0)))),
            }
    return {
        "q_max": float(getattr(amorph_cfg, "q_max", 20.0)),
        "nq": int(getattr(amorph_cfg, "nq", 400)),
        "r_max": float(getattr(amorph_cfg, "r_max", 10.0)),
        "nbins": int(getattr(amorph_cfg, "nbins", 800)),
        "window": str(getattr(amorph_cfg, "window", "lorch")),
        "smooth": int(getattr(amorph_cfg, "smooth", 7)),
        "peak_search": tuple(getattr(amorph_cfg, "peak_search", (0.5, 12.0))),
    }


def _atoms_to_reference_frame(atoms: "Atoms", *, type_to_species: Optional[Sequence[str]]) -> DumpFrame:
    return _atoms_to_dumpframe(atoms, type_to_species=type_to_species, timestep=0)


def _repeat_reference_atoms(atoms: "Atoms", *, min_length_A: float, min_atoms: int) -> tuple["Atoms", tuple[int, int, int]]:
    ats = atoms.copy()
    ats.set_pbc(True)
    lens = np.asarray(ats.get_cell().lengths(), dtype=float)
    reps = [max(1, int(math.ceil(float(min_length_A) / max(float(L), 1.0e-6)))) for L in lens.tolist()]
    nat = int(len(ats)) * int(np.prod(np.asarray(reps, dtype=int)))
    if nat < int(min_atoms):
        scale = int(math.ceil((float(min_atoms) / max(float(nat), 1.0)) ** (1.0 / 3.0)))
        scale = max(1, int(scale))
        reps = [int(r) * int(scale) for r in reps]
    rep_t = (int(reps[0]), int(reps[1]), int(reps[2]))
    if rep_t != (1, 1, 1):
        ats = ats.repeat(rep_t)
    return ats, rep_t


def _structure_to_ase_atoms(structure: Any) -> "Atoms":
    ASEAtoms, _ase_write, _neighbor_list = _require_ase()
    if isinstance(structure, ASEAtoms):
        atoms = structure.copy()
        atoms.set_pbc(True)
        return atoms
    if hasattr(structure, "to_ase_atoms"):
        atoms = structure.to_ase_atoms()
        if isinstance(atoms, ASEAtoms):
            atoms.set_pbc(True)
            return atoms
    try:
        from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore

        atoms = AseAtomsAdaptor.get_atoms(structure)
        if isinstance(atoms, ASEAtoms):
            atoms.set_pbc(True)
            return atoms
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Failed to convert crystal reference to ASE Atoms") from exc
    raise TypeError("Unsupported reference structure type")


def _maybe_conventional_structure(structure: Any) -> Any:
    if hasattr(structure, "to_conventional"):
        try:
            return structure.to_conventional()
        except Exception:
            return structure
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore

        return SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    except Exception:
        return structure


def _reference_cache_key(cache_dir: Path, spec: Mapping[str, Any]) -> tuple[str, str]:
    return (str(cache_dir.resolve()), json.dumps(spec, sort_keys=True))


def _load_library_from_manifest(manifest_path: Path, spec: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if not isinstance(data, Mapping):
        return None
    if dict(data.get("spec", {})) != dict(spec):
        return None
    refs = list(data.get("references", []) or [])
    if len(refs) == 0:
        return dict(data)
    for ref in refs:
        if not isinstance(ref, Mapping):
            return None
        p = Path(manifest_path.parent) / Path(str(ref.get("structure_file", "")))
        if not p.exists():
            return None
    return dict(data)


def build_materials_project_reference_library(
    *,
    cache_dir: Path,
    formula: str,
    type_to_species: Optional[Sequence[str]],
    cutoffs: Mapping[Tuple[int, int], float],
    metrics_cfg,
    progress: Any = None,
) -> dict[str, Any]:
    amorph_cfg = getattr(metrics_cfg, "amorphous", None)
    ref_cfg = getattr(amorph_cfg, "reference", None)
    if not bool(getattr(amorph_cfg, "enabled", False)):
        return {"enabled": False, "used": False, "references": []}
    if ref_cfg is None or not bool(getattr(ref_cfg, "enabled", False)):
        return {"enabled": True, "used": False, "references": [], "reason": "reference fingerprints disabled"}

    target_comp = _normalized_composition_from_formula(str(formula))
    if not target_comp:
        raise ValueError(f"Could not parse reduced formula for crystal references: {formula!r}")

    sq_params = _resolve_sq_params(metrics_cfg, amorph_cfg)
    spec = {
        "formula": str(_formula_from_counts(target_comp)),
        "composition": {str(k): int(v) for k, v in sorted(target_comp.items())},
        "type_to_species": (None if type_to_species is None else [str(x) for x in type_to_species]),
        "cutoffs": {f"{int(k[0])}-{int(k[1])}": float(v) for k, v in sorted(cutoffs.items())},
        "sq": dict(sq_params),
        "l_values": [int(x) for x in list(getattr(amorph_cfg, "l_values", [4, 6]))],
        "solid_like_l": int(getattr(amorph_cfg, "solid_like_l", 6)),
        "reference": {
            "source": str(getattr(ref_cfg, "source", "materials_project")),
            "material_ids": [str(x) for x in list(getattr(ref_cfg, "material_ids", []) or [])],
            "stable_only": bool(getattr(ref_cfg, "stable_only", True)),
            "energy_above_hull_max": float(getattr(ref_cfg, "energy_above_hull_max", 0.05)),
            "max_candidates": int(getattr(ref_cfg, "max_candidates", 6)),
            "use_conventional_cell": bool(getattr(ref_cfg, "use_conventional_cell", True)),
            "min_supercell_length_A": float(getattr(ref_cfg, "min_supercell_length_A", 15.0)),
            "min_supercell_atoms": int(getattr(ref_cfg, "min_supercell_atoms", 256)),
        },
    }
    cache_key = _reference_cache_key(Path(cache_dir), spec)
    if cache_key in _REFERENCE_LIBRARY_CACHE:
        return dict(_REFERENCE_LIBRARY_CACHE[cache_key])

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "reference_manifest.json"
    cached = _load_library_from_manifest(manifest_path, spec)
    if cached is not None:
        _REFERENCE_LIBRARY_CACHE[cache_key] = dict(cached)
        return dict(cached)

    api_key = _resolve_mp_api_key(ref_cfg)
    if api_key is None:
        msg = "Materials Project API key not available for amorphous crystal references"
        if bool(getattr(ref_cfg, "required", False)):
            raise ValueError(msg)
        lib = {"enabled": True, "used": False, "references": [], "reason": msg, "spec": dict(spec)}
        manifest_path.write_text(json.dumps(lib, indent=2))
        _REFERENCE_LIBRARY_CACHE[cache_key] = dict(lib)
        return lib

    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        if bool(getattr(ref_cfg, "required", False)):
            raise ImportError("amorphous reference detection requires the optional 'mp-api' package") from exc
        msg = "mp-api package not available for Materials Project crystal references"
        lib = {"enabled": True, "used": False, "references": [], "reason": msg, "spec": dict(spec)}
        manifest_path.write_text(json.dumps(lib, indent=2))
        _REFERENCE_LIBRARY_CACHE[cache_key] = dict(lib)
        return lib

    mpids_req = [str(x) for x in list(getattr(ref_cfg, "material_ids", []) or [])]
    chemsys = "-".join(sorted(target_comp.keys()))
    fields = ["material_id", "formula_pretty", "energy_above_hull", "is_stable", "structure", "composition_reduced"]

    with MPRester(api_key) as mpr:  # type: ignore[misc]
        if len(mpids_req) > 0:
            docs = mpr.materials.summary.search(material_ids=mpids_req, fields=fields)
        else:
            docs = mpr.materials.summary.search(chemsys=chemsys, fields=fields)

    candidates: list[dict[str, Any]] = []
    for doc in list(docs or []):
        mid = str(getattr(doc, "material_id", getattr(doc, "get", lambda *a, **k: None)("material_id", None)))
        if mid == "None" or mid == "":
            continue
        comp = _normalized_composition_from_obj(getattr(doc, "composition_reduced", None))
        if comp is None and hasattr(doc, "get"):
            try:
                comp = _normalized_composition_from_obj(doc.get("composition_reduced", None))
            except Exception:
                comp = None
        if comp != target_comp:
            # fall formula pretty
            fp = str(getattr(doc, "formula_pretty", "") or "")
            if fp == "" and hasattr(doc, "get"):
                fp = str(doc.get("formula_pretty", "") or "")
            if not fp or _normalized_composition_from_formula(fp) != target_comp:
                continue
        eah = _scalar_or_nan(getattr(doc, "energy_above_hull", None))
        stable = bool(getattr(doc, "is_stable", False))
        if hasattr(doc, "get"):
            try:
                if not math.isfinite(eah):
                    eah = _scalar_or_nan(doc.get("energy_above_hull", None))
                stable = bool(stable or doc.get("is_stable", False))
            except Exception:
                pass
        if bool(getattr(ref_cfg, "stable_only", True)) and not stable:
            continue
        if math.isfinite(eah) and float(eah) > float(getattr(ref_cfg, "energy_above_hull_max", 0.05)):
            continue
        struct = getattr(doc, "structure", None)
        if struct is None and hasattr(doc, "get"):
            try:
                struct = doc.get("structure", None)
            except Exception:
                struct = None
        if struct is None:
            continue
        if hasattr(struct, "is_ordered") and not bool(getattr(struct, "is_ordered")):
            continue
        candidates.append(
            {
                "material_id": str(mid),
                "formula_pretty": str(getattr(doc, "formula_pretty", "") or ""),
                "energy_above_hull": (None if not math.isfinite(eah) else float(eah)),
                "is_stable": bool(stable),
                "structure": struct,
            }
        )

    candidates.sort(key=lambda x: (float(x.get("energy_above_hull", 0.0) if x.get("energy_above_hull") is not None else 0.0), str(x.get("material_id", ""))))
    max_candidates = int(getattr(ref_cfg, "max_candidates", 6))
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    if len(candidates) == 0:
        msg = f"No crystalline Materials Project references found for reduced formula {_formula_from_counts(target_comp)}"
        if bool(getattr(ref_cfg, "required", False)):
            raise ValueError(msg)
        lib = {"enabled": True, "used": False, "references": [], "reason": msg, "spec": dict(spec)}
        manifest_path.write_text(json.dumps(lib, indent=2))
        _REFERENCE_LIBRARY_CACHE[cache_key] = dict(lib)
        return lib

    refs: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        struct = cand["structure"]
        if bool(getattr(ref_cfg, "use_conventional_cell", True)):
            struct = _maybe_conventional_structure(struct)
        atoms = _structure_to_ase_atoms(struct)
        atoms_rep, reps = _repeat_reference_atoms(
            atoms,
            min_length_A=float(getattr(ref_cfg, "min_supercell_length_A", 15.0)),
            min_atoms=int(getattr(ref_cfg, "min_supercell_atoms", 256)),
        )
        fr = _atoms_to_reference_frame(atoms_rep, type_to_species=type_to_species)
        q, s = compute_sq(
            [fr],
            q_max=float(sq_params["q_max"]),
            nq=int(sq_params["nq"]),
            r_max=float(sq_params["r_max"]),
            nbins=int(sq_params["nbins"]),
            pair=None,
            type_to_species=type_to_species,
            window=str(sq_params["window"]),
        )
        peaks = _sq_peak_features(
            q,
            s,
            smooth=int(sq_params["smooth"]),
            peak_search=tuple(sq_params["peak_search"]),
            prominence_min=float(getattr(amorph_cfg, "peak_prominence_min", 0.15)),
            height_min=float(getattr(amorph_cfg, "peak_height_min", 1.05)),
        )
        local = _local_order_analysis(fr, cutoffs=cutoffs, amorph_cfg=amorph_cfg)
        fn = f"reference_{idx:02d}_{str(cand['material_id']).replace('/', '_')}.extxyz"
        _ASEAtoms, ase_write, _neighbor_list = _require_ase()
        ase_write(str(cache_dir / fn), atoms_rep)
        refs.append(
            {
                "material_id": str(cand["material_id"]),
                "formula_pretty": str(cand.get("formula_pretty", "") or ""),
                "energy_above_hull": cand.get("energy_above_hull", None),
                "is_stable": bool(cand.get("is_stable", False)),
                "structure_file": str(fn),
                "supercell_repeat": [int(x) for x in reps],
                "n_atoms": int(len(atoms_rep)),
                "sq": {
                    "q": [float(x) for x in np.asarray(q, dtype=float).tolist()],
                    "s": [float(x) for x in np.asarray(s, dtype=float).tolist()],
                    "peaks": [{str(k): float(v) for k, v in pk.items()} for pk in peaks],
                },
                "local_order": {
                    "crystalline_fraction": float(local.get("crystalline_fraction", float("nan"))),
                    "largest_cluster_fraction": float(local.get("largest_cluster_fraction", float("nan"))),
                    "qbar": {str(k): float(v) for k, v in dict(local.get("qbar", {})).items()},
                },
            }
        )

    lib = {
        "enabled": True,
        "used": True,
        "source": "materials_project",
        "formula": str(_formula_from_counts(target_comp)),
        "references": refs,
        "spec": dict(spec),
    }
    manifest_path.write_text(json.dumps(lib, indent=2))
    _REFERENCE_LIBRARY_CACHE[cache_key] = dict(lib)
    _message(progress, "info", f"prepared {len(refs)} crystal reference fingerprint(s) for {lib['formula']}")
    return lib


def analyse_amorphous_state(
    frames: Sequence[DumpFrame],
    *,
    metrics_cfg,
    cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]],
    cache_dir: Optional[Path] = None,
    formula_override: Optional[str] = None,
    progress: Any = None,
) -> dict[str, Any]:
    amorph_cfg = getattr(metrics_cfg, "amorphous", None)
    if amorph_cfg is None or not bool(getattr(amorph_cfg, "enabled", False)):
        return {"enabled": False, "passed": True, "criteria": {}, "scalar_metrics": {}, "reference": {"used": False}}
    if not frames:
        raise ValueError("analyse_amorphous_state requires at least one frame")

    sq_params = _resolve_sq_params(metrics_cfg, amorph_cfg)
    q, s = compute_sq(
        frames,
        q_max=float(sq_params["q_max"]),
        nq=int(sq_params["nq"]),
        r_max=float(sq_params["r_max"]),
        nbins=int(sq_params["nbins"]),
        pair=None,
        type_to_species=type_to_species,
        window=str(sq_params["window"]),
    )
    peaks = _sq_peak_features(
        q,
        s,
        smooth=int(sq_params["smooth"]),
        peak_search=tuple(sq_params["peak_search"]),
        prominence_min=float(getattr(amorph_cfg, "peak_prominence_min", 0.15)),
        height_min=float(getattr(amorph_cfg, "peak_height_min", 1.05)),
    )
    bragg_sharpness = float(np.sum([float(p.get("sharpness", 0.0)) for p in peaks], dtype=float)) if len(peaks) > 0 else 0.0

    formula = str(formula_override).strip() if formula_override is not None and str(formula_override).strip() != "" else None
    if formula is None:
        formula = reduced_formula_from_frame(frames[0], type_to_species=type_to_species)
    ref_cfg = getattr(amorph_cfg, "reference", None)

    lib: dict[str, Any] = {"enabled": bool(ref_cfg is not None and bool(getattr(ref_cfg, "enabled", False))), "used": False, "references": []}
    refs: list[dict[str, Any]] = []
    if formula and ref_cfg is not None and bool(getattr(ref_cfg, "enabled", False)) and cache_dir is not None:
        try:
            lib = build_materials_project_reference_library(
                cache_dir=Path(cache_dir),
                formula=str(formula),
                type_to_species=type_to_species,
                cutoffs=cutoffs,
                metrics_cfg=metrics_cfg,
                progress=progress,
            )
        except Exception as exc:
            if bool(getattr(ref_cfg, "required", False)):
                raise
            lib = {"enabled": True, "used": False, "references": [], "reason": str(exc)}
            _message(progress, "warn", f"amorphous reference fingerprints unavailable: {exc}")
        refs = [dict(x) for x in list(lib.get("references", []) or []) if isinstance(x, Mapping)]

    local = _local_order_analysis_timeavg(frames, cutoffs=cutoffs, amorph_cfg=amorph_cfg, reference_refs=refs)
    crystalline_fraction = float(local.get("crystalline_fraction", float("nan")))
    largest_cluster_fraction = float(local.get("largest_cluster_fraction", float("nan")))

    reference_report: dict[str, Any] = {
        "used": bool(lib.get("used", False) and len(refs) > 0),
        "n_candidates": int(len(refs)),
        "formula": str(lib.get("formula", formula)) if formula is not None else None,
        "reason": lib.get("reason", None),
        "overlaps": [],
    }
    motifs_report: dict[str, Any] = {
        "enabled": bool(ref_cfg is not None and bool(getattr(ref_cfg, "enabled", False))),
        "used": bool(reference_report.get("used", False)),
        "formula": str(lib.get("formula", formula)) if formula is not None else None,
        "reason": lib.get("reason", None),
        "thresholds": {},
        "top_matches": [],
        "candidate_matches": [],
        "detected": [],
    }
    best_overlap = float("nan")
    best_ref: Optional[dict[str, Any]] = None
    best_gate_overlap = float("nan")
    best_gate_ref: Optional[dict[str, Any]] = None
    if reference_report.get("used", False):
        ranked = _rank_reference_motifs(peaks=peaks, local_order=local, refs=refs, amorph_cfg=amorph_cfg)
        cand_thr, det_thr = _motif_overlap_thresholds(amorph_cfg)
        motifs_report.update(
            {
                "thresholds": {
                    "candidate_peak_overlap": float(cand_thr),
                    "detected_peak_overlap": float(det_thr),
                },
                "top_matches": [dict(item) for item in list(ranked[: min(3, len(ranked))])],
                "candidate_matches": [dict(item) for item in ranked if bool(item.get("candidate", False))],
                "detected": [dict(item) for item in ranked if bool(item.get("detected", False))],
            }
        )
        for item in ranked:
            reference_report["overlaps"].append(item)
        best_ref = _best_reference_match_by_overlap(reference_report["overlaps"])
        if best_ref is not None:
            try:
                best_overlap = float(best_ref.get("peak_overlap", float("nan")))
            except Exception:
                best_overlap = float("nan")
        best_gate_ref = _best_discriminative_reference_match(reference_report["overlaps"])
        if best_gate_ref is not None:
            try:
                best_gate_overlap = float(best_gate_ref.get("peak_overlap", float("nan")))
            except Exception:
                best_gate_overlap = float("nan")
        reference_report["best_material_id"] = None if best_ref is None else best_ref.get("material_id")
        reference_report["best_formula_pretty"] = None if best_ref is None else best_ref.get("formula_pretty")
        reference_report["best_peak_overlap"] = None if not math.isfinite(best_overlap) else float(best_overlap)
        reference_report["best_discriminative_material_id"] = None if best_gate_ref is None else best_gate_ref.get("material_id")
        reference_report["best_discriminative_formula_pretty"] = None if best_gate_ref is None else best_gate_ref.get("formula_pretty")
        reference_report["best_discriminative_peak_overlap"] = None if not math.isfinite(best_gate_overlap) else float(best_gate_overlap)
        reference_report["gating_material_id"] = None if best_gate_ref is None else best_gate_ref.get("material_id")
        reference_report["gating_formula_pretty"] = None if best_gate_ref is None else best_gate_ref.get("formula_pretty")
        reference_report["gating_peak_overlap"] = None if not math.isfinite(best_gate_overlap) else float(best_gate_overlap)
        reference_report["best_motif_material_id"] = None if len(reference_report["overlaps"]) == 0 else reference_report["overlaps"][0].get("material_id")
        reference_report["best_motif_formula_pretty"] = None if len(reference_report["overlaps"]) == 0 else reference_report["overlaps"][0].get("formula_pretty")
        reference_report["best_motif_score"] = None if len(reference_report["overlaps"]) == 0 else float(reference_report["overlaps"][0].get("motif_score", float("nan")))

    criteria: dict[str, Any] = {}
    checks: list[bool] = []
    crit = {
        "bragg_sharpness": (float(bragg_sharpness), float(getattr(amorph_cfg, "max_bragg_sharpness", 25.0))),
        "crystalline_fraction": (float(crystalline_fraction), float(getattr(amorph_cfg, "max_crystalline_fraction", 0.15))),
        "largest_cluster_fraction": (float(largest_cluster_fraction), float(getattr(amorph_cfg, "max_largest_cluster_fraction", 0.10))),
    }
    for name, (val, thr) in crit.items():
        passed = bool(math.isfinite(val) and val <= thr)
        criteria[name] = {"value": float(val), "threshold": float(thr), "passed": passed}
        checks.append(passed)
    if reference_report.get("used", False):
        thr = float(getattr(amorph_cfg, "max_reference_peak_overlap", 0.65))
        discr = bool(best_gate_ref is not None and bool(best_gate_ref.get("reference_discriminative", False)))
        if discr:
            passed_ref = bool(math.isfinite(best_gate_overlap) and float(best_gate_overlap) <= thr)
            criteria["reference_peak_overlap"] = {
                "value": float(best_gate_overlap),
                "threshold": float(thr),
                "passed": passed_ref,
                "best_material_id": None if best_gate_ref is None else best_gate_ref.get("material_id"),
                "reference_discriminative": True,
                "matched_pairs": (None if best_gate_ref is None else int(best_gate_ref.get("matched_pairs", 0))),
                "matched_significant_pairs": (None if best_gate_ref is None else int(best_gate_ref.get("matched_significant_pairs", 0))),
                "box_peak_count": (None if best_gate_ref is None else int(best_gate_ref.get("box_peak_count", 0))),
                "ref_peak_count": (None if best_gate_ref is None else int(best_gate_ref.get("ref_peak_count", 0))),
                "significant_support_threshold": (None if best_gate_ref is None else float(best_gate_ref.get("significant_support_threshold", float("nan")))),
                "discriminative_min_significant_support_weight": (
                    None if best_gate_ref is None else float(best_gate_ref.get("discriminative_min_significant_support_weight", float("nan")))
                ),
                "advisory_best_material_id": None if best_ref is None else best_ref.get("material_id"),
                "advisory_best_peak_overlap": (None if not math.isfinite(best_overlap) else float(best_overlap)),
            }
            checks.append(passed_ref)
        else:
            criteria["reference_peak_overlap"] = {
                "value": (None if not math.isfinite(best_overlap) else float(best_overlap)),
                "threshold": float(thr),
                "passed": True,
                "skipped": True,
                "reason": (
                    "reference fingerprint not discriminative enough for a hard gate"
                    if best_ref is not None
                    else reference_report.get("reason", "reference fingerprints unavailable")
                ),
                "best_material_id": None if best_ref is None else best_ref.get("material_id"),
                "best_discriminative_material_id": None if best_gate_ref is None else best_gate_ref.get("material_id"),
                "reference_discriminative": False,
                "matched_pairs": (None if best_ref is None else int(best_ref.get("matched_pairs", 0))),
                "matched_significant_pairs": (None if best_ref is None else int(best_ref.get("matched_significant_pairs", 0))),
                "box_peak_count": (None if best_ref is None else int(best_ref.get("box_peak_count", 0))),
                "ref_peak_count": (None if best_ref is None else int(best_ref.get("ref_peak_count", 0))),
                "significant_support_threshold": (None if best_ref is None else float(best_ref.get("significant_support_threshold", float("nan")))),
                "discriminative_min_significant_support_weight": (
                    None if best_ref is None else float(best_ref.get("discriminative_min_significant_support_weight", float("nan")))
                ),
                "advisory_only": bool(math.isfinite(best_overlap) and float(best_overlap) > float(thr)),
            }
    else:
        criteria["reference_peak_overlap"] = {
            "value": None,
            "threshold": float(getattr(amorph_cfg, "max_reference_peak_overlap", 0.65)),
            "passed": True,
            "skipped": True,
            "reason": reference_report.get("reason", "reference fingerprints unavailable"),
        }

    scalar_metrics = {
        "amorphous_bragg_sharpness": float(bragg_sharpness),
        "amorphous_crystalline_fraction": float(crystalline_fraction),
        "amorphous_largest_cluster_fraction": float(largest_cluster_fraction),
        "amorphous_reference_peak_overlap": (float(best_gate_overlap) if math.isfinite(best_gate_overlap) else float("nan")),
        "amorphous_reference_peak_overlap_advisory": (float(best_overlap) if math.isfinite(best_overlap) else float("nan")),
    }
    for k, v in dict(local.get("qbar", {})).items():
        scalar_metrics[f"amorphous_{str(k)}"] = float(v)

    return {
        "enabled": True,
        "passed": bool(all(checks)),
        "criteria": criteria,
        "scalar_metrics": scalar_metrics,
        "sq": {
            "q": [float(x) for x in np.asarray(q, dtype=float).tolist()],
            "s": [float(x) for x in np.asarray(s, dtype=float).tolist()],
            "peaks": [{str(k): float(v) for k, v in pk.items()} for pk in peaks],
        },
        "local_order": local,
        "reference": reference_report,
        "motifs": motifs_report,
    }

def summarize_rate_amorphous_acceptance(
rate_results: Sequence[Mapping[str, Any]], *, amorph_cfg) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    req = float(getattr(amorph_cfg, "min_pass_fraction", 1.0))
    for rr in list(rate_results or []):
        reps = [rep for rep in list(rr.get("replicates", []) or []) if isinstance(rep, Mapping)]
        flags = [bool(((rep.get("amorphous", {}) or {}).get("passed", False))) for rep in reps]
        n = int(len(flags))
        n_pass = int(sum(1 for x in flags if x))
        frac = (float(n_pass) / float(n)) if n > 0 else float("nan")

        crit_keys: set[str] = set()
        for rep in reps:
            crit_keys.update(str(k) for k in dict(((rep.get("amorphous", {}) or {}).get("criteria", {}) or {})).keys())
        criteria_summary: dict[str, Any] = {}
        failed_criteria: list[str] = []
        for name in sorted(crit_keys):
            vals: list[float] = []
            thresholds: list[float] = []
            n_eval = 0
            n_failed = 0
            n_skipped = 0
            for rep in reps:
                crit = dict(((rep.get("amorphous", {}) or {}).get("criteria", {}) or {}).get(name, {}) or {})
                if not crit:
                    continue
                n_eval += 1
                if bool(crit.get("skipped", False)):
                    n_skipped += 1
                passed = crit.get("passed", None)
                if passed is False:
                    n_failed += 1
                try:
                    val = crit.get("value", None)
                    if val is not None:
                        fv = float(val)
                        if math.isfinite(fv):
                            vals.append(fv)
                except Exception:
                    pass
                try:
                    thr = crit.get("threshold", None)
                    if thr is not None:
                        ft = float(thr)
                        if math.isfinite(ft):
                            thresholds.append(ft)
                except Exception:
                    pass
            if n_eval == 0:
                continue
            threshold_val = None
            if thresholds:
                if max(thresholds) - min(thresholds) <= 1.0e-12:
                    threshold_val = float(thresholds[0])
            if n_failed > 0:
                failed_criteria.append(str(name))
            criteria_summary[str(name)] = {
                "n_evaluated": int(n_eval),
                "n_failed": int(n_failed),
                "n_skipped": int(n_skipped),
                "threshold": threshold_val,
                "mean": (float(np.mean(np.asarray(vals, dtype=float))) if vals else None),
                "max": (float(np.max(np.asarray(vals, dtype=float))) if vals else None),
                "min": (float(np.min(np.asarray(vals, dtype=float))) if vals else None),
            }

        out.append(
            {
                "rate": float(rr.get("rate", float("nan"))),
                "n": int(n),
                "n_pass": int(n_pass),
                "pass_fraction": float(frac),
                "required_pass_fraction": float(req),
                "accepted": bool(n > 0 and math.isfinite(frac) and frac >= req),
                "criteria_summary": criteria_summary,
                "failed_criteria": failed_criteria,
            }
        )
    return out
