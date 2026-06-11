from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def summarize_production_crystal_motifs(
    accepted_boxes: Sequence[Mapping[str, Any]],
    *,
    rejected_boxes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Production crystal motifs."""

    accepted = [dict(b) for b in list(accepted_boxes or []) if isinstance(b, Mapping)]
    rejected = [dict(b) for b in list(rejected_boxes or []) if isinstance(b, Mapping)]

    def _box_id(box: Mapping[str, Any]) -> Optional[int]:
        try:
            return int(box.get("box", 0) or 0)
        except Exception:
            return None

    thresholds: dict[str, Any] = {}
    any_used = False
    any_enabled = False
    per_mid: dict[str, dict[str, Any]] = {}

    def _touch(mid: str, *, formula_pretty: str, energy_above_hull: Any) -> dict[str, Any]:
        st = per_mid.get(mid)
        if st is None:
            st = {
                "material_id": str(mid),
                "formula_pretty": str(formula_pretty or ""),
                "energy_above_hull": energy_above_hull,
                "candidate_boxes": set(),
                "detected_boxes": set(),
                "top_match_boxes": set(),
                "accepted_candidate_boxes": set(),
                "accepted_detected_boxes": set(),
                "rejected_candidate_boxes": set(),
                "rejected_detected_boxes": set(),
                "peak_overlaps": [],
                "motif_scores": [],
            }
            per_mid[str(mid)] = st
        return st

    def _consume(boxes: Sequence[Mapping[str, Any]], *, accepted_flag: bool) -> None:
        nonlocal any_used, any_enabled, thresholds
        for box in boxes:
            box_id = _box_id(box)
            amorph = dict(box.get("amorphous", {}) or {})
            motifs = dict(amorph.get("motifs", {}) or {})
            if motifs:
                any_enabled = any_enabled or bool(motifs.get("enabled", False))
                any_used = any_used or bool(motifs.get("used", False))
                if not thresholds and isinstance(motifs.get("thresholds", None), Mapping):
                    thresholds = {str(k): v for k, v in dict(motifs.get("thresholds", {})).items()}
            top_matches = list(motifs.get("top_matches", []) or [])
            extra_matches = list(motifs.get("candidate_matches", []) or []) + list(motifs.get("detected", []) or [])
            merged_items: dict[str, dict[str, Any]] = {}
            for item in top_matches + extra_matches:
                if not isinstance(item, Mapping):
                    continue
                mid = str(item.get("material_id", "") or "")
                if mid == "":
                    continue
                merged_items[mid] = dict(item)
            for item in merged_items.values():
                mid = str(item.get("material_id", "") or "")
                if mid == "":
                    continue
                st = _touch(mid, formula_pretty=str(item.get("formula_pretty", "") or ""), energy_above_hull=item.get("energy_above_hull", None))
                if box_id is not None and int(item.get("rank", 0) or 0) == 1:
                    st["top_match_boxes"].add(int(box_id))
                try:
                    ov = float(item.get("peak_overlap", float("nan")))
                except Exception:
                    ov = float("nan")
                try:
                    sc = float(item.get("motif_score", float("nan")))
                except Exception:
                    sc = float("nan")
                if math.isfinite(ov):
                    st["peak_overlaps"].append(float(ov))
                if math.isfinite(sc):
                    st["motif_scores"].append(float(sc))
                if box_id is not None and bool(item.get("candidate", False)):
                    st["candidate_boxes"].add(int(box_id))
                    if accepted_flag:
                        st["accepted_candidate_boxes"].add(int(box_id))
                    else:
                        st["rejected_candidate_boxes"].add(int(box_id))
                if box_id is not None and bool(item.get("detected", False)):
                    st["detected_boxes"].add(int(box_id))
                    if accepted_flag:
                        st["accepted_detected_boxes"].add(int(box_id))
                    else:
                        st["rejected_detected_boxes"].add(int(box_id))

    _consume(accepted, accepted_flag=True)
    _consume(rejected, accepted_flag=False)

    n_total = int(len(accepted) + len(rejected))
    motifs: list[dict[str, Any]] = []
    for mid, st in per_mid.items():
        peak_arr = np.asarray(list(st.pop("peak_overlaps", [])), dtype=float)
        score_arr = np.asarray(list(st.pop("motif_scores", [])), dtype=float)
        cand_boxes = sorted(int(x) for x in st.pop("candidate_boxes"))
        det_boxes = sorted(int(x) for x in st.pop("detected_boxes"))
        top_boxes = sorted(int(x) for x in st.pop("top_match_boxes"))
        acc_cand = sorted(int(x) for x in st.pop("accepted_candidate_boxes"))
        acc_det = sorted(int(x) for x in st.pop("accepted_detected_boxes"))
        rej_cand = sorted(int(x) for x in st.pop("rejected_candidate_boxes"))
        rej_det = sorted(int(x) for x in st.pop("rejected_detected_boxes"))
        motifs.append(
            {
                **st,
                "candidate_boxes": cand_boxes,
                "detected_boxes": det_boxes,
                "top_match_boxes": top_boxes,
                "accepted_candidate_boxes": acc_cand,
                "accepted_detected_boxes": acc_det,
                "rejected_candidate_boxes": rej_cand,
                "rejected_detected_boxes": rej_det,
                "n_boxes_candidate": int(len(cand_boxes)),
                "n_boxes_detected": int(len(det_boxes)),
                "n_boxes_top_match": int(len(top_boxes)),
                "fraction_boxes_candidate": (float(len(cand_boxes)) / float(n_total) if n_total > 0 else float("nan")),
                "fraction_boxes_detected": (float(len(det_boxes)) / float(n_total) if n_total > 0 else float("nan")),
                "mean_peak_overlap": (float(np.mean(peak_arr)) if peak_arr.size > 0 else float("nan")),
                "max_peak_overlap": (float(np.max(peak_arr)) if peak_arr.size > 0 else float("nan")),
                "mean_motif_score": (float(np.mean(score_arr)) if score_arr.size > 0 else float("nan")),
                "max_motif_score": (float(np.max(score_arr)) if score_arr.size > 0 else float("nan")),
            }
        )
    motifs.sort(
        key=lambda x: (
            int(x.get("n_boxes_detected", 0)),
            int(x.get("n_boxes_candidate", 0)),
            int(x.get("n_boxes_top_match", 0)),
            float(x.get("max_motif_score", float("-inf"))),
            str(x.get("material_id", "")),
        ),
        reverse=True,
    )

    return {
        "enabled": bool(any_enabled),
        "used": bool(any_used),
        "n_boxes_accepted": int(len(accepted)),
        "n_boxes_rejected": int(len(rejected)),
        "n_boxes_total": int(n_total),
        "thresholds": dict(thresholds),
        "motifs": motifs,
    }
