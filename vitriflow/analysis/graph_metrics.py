from __future__ import annotations

"""Graph-rule aware descriptor evaluation and reporting helpers."""

import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx
except Exception as e:  # pragma: no cover
    raise ImportError("vitriflow.analysis.graph_metrics requires networkx") from e

from ..config import StructureMetricsConfig
from .common import (
    resolve_selector as _resolve_selector,
    wrap_frac as _wrap_frac,
    mic_displacements_and_distances as _mic_displacements,
)
from .dump import frame_pbc
from .provenance import (
    csv_scalar,
    direct_coordinate_rule_fields,
    finite_or_none,
    graph_rule_to_representation_fields,
    json_sanitize,
    metric_result_row,
    metric_units_for_name,
    void_rule_fields,
    write_json_strict,
)
from .graph import (
    GraphRule,
    StructureGraph,
    build_graph,
    directed_neighbor_lists,
    expand_graph_rules,
    expand_graph_rules_for_frame,
    expand_graph_rules_for_frames,
    interval_graph_rules,
    manifest_row_from_frame,
    pair_cutoffs_from_parameters,
    pair_intervals_from_parameters,
    graph_family_from_rule,
    structure_hash,
    verify_manifest_row,
)


def _json_safe(value: Any) -> Any:
    return json_sanitize(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(json_sanitize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


# Descriptor-map provenance can be much larger than the descriptor values when
# adaptive RDF rules are swept over hundreds of structures.  Public rows still
# carry the graph rule needed to reconstruct G_lambda(x), but heavy derivation
# diagnostics are stored once in JSON sidecars and referenced by a stable id.
_MAX_CSV_FIELD_CHARS = 32768
_HEAVY_RULE_KEYS = {
    "derivation",
    "per_structure",
    "rdf_curve",
    "curve",
    "histogram",
    "distances",
    "edge_vectors",
    "edges",
    "positions",
}


def _set_csv_field_limit() -> None:
    """Allow old large chunks to be read, while new chunks remain compact."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:  # pragma: no cover - platform dependent
            limit = int(limit // 10)
            if limit <= 131072:
                try:
                    csv.field_size_limit(10_000_000)
                except Exception:
                    pass
                return


def _short_hash(value: Any, *, n: int = 16) -> str:
    import hashlib

    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()[: int(n)]


def _rule_params_mapping(rule_or_record: Any) -> Mapping[str, Any]:
    raw: Any
    if isinstance(rule_or_record, Mapping):
        raw = rule_or_record.get("parameters", rule_or_record.get("graph_rule_parameters", {}))
    else:
        raw = getattr(rule_or_record, "parameters", {})
    if isinstance(raw, Mapping):
        return dict(raw or {})
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return dict(parsed or {}) if isinstance(parsed, Mapping) else {}
        except Exception:
            return {}
    return {}


def _rule_name(rule_or_record: Any) -> str:
    if isinstance(rule_or_record, Mapping):
        return str(rule_or_record.get("name", rule_or_record.get("graph_rule_name", "graph_rule")))
    return str(getattr(rule_or_record, "name", "graph_rule"))


def _rule_kind(rule_or_record: Any) -> str:
    if isinstance(rule_or_record, Mapping):
        return str(rule_or_record.get("kind", rule_or_record.get("graph_rule_kind", "hard_cutoff")))
    return str(getattr(rule_or_record, "kind", "hard_cutoff"))


def _rule_provenance(rule_or_record: Any) -> Any:
    if isinstance(rule_or_record, Mapping):
        return rule_or_record.get("provenance", rule_or_record.get("graph_rule_provenance", "runtime"))
    return getattr(rule_or_record, "provenance", "runtime")


def _derivation_ref(rule_or_record: Any, params: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    p = dict(params or _rule_params_mapping(rule_or_record))
    deriv = p.get("derivation", None)
    if deriv is None:
        return None
    scope = str(p.get("graph_rule_scope", p.get("rule_scope", "per_structure")))
    sh = str(p.get("structure_hash", p.get("ensemble_label", p.get("ensemble_size", ""))))
    payload = {
        # Do not include the concrete sweep/family rule name here.  All sweep
        # points produced from the same RDF/shell derivation must reference one
        # derivation sidecar record, not duplicate it once per lambda.
        "parent_rule_name": p.get("parent_rule_name"),
        "parent_rule_kind": p.get("parent_rule_kind"),
        "scope": scope,
        "structure_hash_or_ensemble": sh,
        "derivation_method": p.get("derivation_method"),
        "derivation": deriv,
    }
    return "deriv:" + _short_hash(payload, n=24)


def _derivation_pair_summary(derivation: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(derivation or []):
        if not isinstance(item, Mapping):
            continue
        shell = item.get("shell_separability", {}) if isinstance(item.get("shell_separability", {}), Mapping) else {}
        conn = item.get("connectivity", {}) if isinstance(item.get("connectivity", {}), Mapping) else {}
        conn_summary: dict[str, Any] = {}
        for key in (
            "available",
            "connectivity_floor_skipped",
            "connectivity_lower_bound",
            "component_count_at_lower_bound",
            "largest_component_fraction_at_lower_bound",
            "requested_largest_component_fraction",
            "target_component_count",
            "target_largest_component_fraction",
            "skip_reason",
        ):
            if key in conn and key != "per_structure":
                conn_summary[key] = conn.get(key)
        if isinstance(conn.get("per_structure"), Sequence) and not isinstance(conn.get("per_structure"), (str, bytes, bytearray)):
            conn_summary["per_structure_count"] = len(list(conn.get("per_structure") or []))
        out.append(
            json_sanitize(
                {
                    "pair": item.get("pair"),
                    "pair_species": item.get("pair_species"),
                    "selected_cutoff": item.get("selected_cutoff"),
                    "rdf_first_minimum": item.get("rdf_first_minimum"),
                    "rdf_first_minimum_height": item.get("rdf_first_minimum_height"),
                    "first_peak_r": item.get("first_peak_r"),
                    "second_shell_onset": item.get("second_shell_onset"),
                    "search_radius": item.get("search_radius"),
                    "bin_width": item.get("bin_width"),
                    "smooth_width": item.get("smooth_width"),
                    "nbins": item.get("nbins"),
                    "interval": item.get("interval"),
                    "shell_available": shell.get("available"),
                    "shell_objective_cutoff": shell.get("shell_objective_cutoff"),
                    "shell_objective_loss": shell.get("shell_objective_loss"),
                    "shell_objective_under_fraction": shell.get("shell_objective_under_fraction"),
                    "shell_objective_accidental_fraction": shell.get("shell_objective_accidental_fraction"),
                    "shell_separable": shell.get("shell_separable"),
                    "connectivity_summary": conn_summary,
                }
            )
        )
    return out


def _compact_nested_for_rule(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return {"omitted": True, "reason": "maximum_compaction_depth_exceeded"}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            kk = str(k)
            if kk in _HEAVY_RULE_KEYS:
                if kk == "derivation":
                    out["derivation_pair_summary"] = _derivation_pair_summary(v)
                    out["derivation_ref"] = "deriv:" + _short_hash(v, n=24)
                    out["derivation_stored_in_sidecar"] = True
                elif kk == "per_structure" and isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
                    out[f"{kk}_count"] = len(list(v or []))
                    out[f"{kk}_stored_in_sidecar"] = True
                else:
                    out[f"{kk}_omitted"] = True
                continue
            out[kk] = _compact_nested_for_rule(v, depth=depth + 1)
        return json_sanitize(out)
    if isinstance(value, (list, tuple)):
        # Lists of scalars / small mappings are safe.  Very long lists are audit
        # payloads and should not be repeated in every descriptor row.
        if len(value) > 256:
            return {"omitted": True, "length": len(value), "sha256": _short_hash(value, n=24), "reason": "large_sequence_stored_in_sidecar"}
        return [_compact_nested_for_rule(v, depth=depth + 1) for v in value]
    return json_sanitize(value)


def _compact_rule_parameters(params: Mapping[str, Any], *, rule_or_record: Any = None, include_derivation_summary: bool = True) -> dict[str, Any]:
    p = dict(params or {})
    deriv = p.get("derivation", None)
    ref = _derivation_ref(rule_or_record if rule_or_record is not None else {"parameters": p}, p) if deriv is not None else None
    out: dict[str, Any] = {}
    for k, v in p.items():
        kk = str(k)
        if kk == "derivation":
            out["derivation_ref"] = ref
            out["derivation_stored_in_sidecar"] = True
            if bool(include_derivation_summary):
                out["derivation_pair_summary"] = _derivation_pair_summary(v)
            continue
        out[kk] = _compact_nested_for_rule(v)
    if ref is not None:
        out.setdefault("derivation_ref", ref)
        out.setdefault("derivation_stored_in_sidecar", True)
    return json_sanitize(out)


def _compact_rule_provenance(provenance: Any) -> Any:
    if isinstance(provenance, Mapping):
        out: dict[str, Any] = {}
        for k, v in provenance.items():
            kk = str(k)
            if kk == "parent_rule" and isinstance(v, Mapping):
                pv = dict(v)
                if isinstance(pv.get("parameters"), Mapping):
                    pv["parameters"] = _compact_rule_parameters(pv.get("parameters", {}), rule_or_record=pv)
                if isinstance(pv.get("provenance"), Mapping):
                    pv["provenance"] = _compact_rule_provenance(pv.get("provenance"))
                out[kk] = json_sanitize(pv)
            elif kk in _HEAVY_RULE_KEYS:
                out[f"{kk}_omitted"] = True
            else:
                out[kk] = _compact_nested_for_rule(v)
        return json_sanitize(out)
    return _compact_nested_for_rule(provenance)


def _compact_rule_record(rule_or_record: Any) -> dict[str, Any]:
    p = _rule_params_mapping(rule_or_record)
    return json_sanitize(
        {
            "name": _rule_name(rule_or_record),
            "kind": _rule_kind(rule_or_record),
            "parameters": _compact_rule_parameters(p, rule_or_record=rule_or_record),
            "provenance": _compact_rule_provenance(_rule_provenance(rule_or_record)),
        }
    )


def _adaptive_derivation_records_from_rules(rules: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in list(rules or []):
        params = _rule_params_mapping(rule)
        deriv = params.get("derivation", None)
        if deriv is None:
            continue
        ref = _derivation_ref(rule, params)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        rec = {
            "schema": "vitriflow.adaptive_graph_rule_derivation.v1",
            "derivation_ref": ref,
            "graph_rule_name": _rule_name(rule),
            "graph_rule_kind": _rule_kind(rule),
            "graph_rule_scope": str(params.get("graph_rule_scope", params.get("rule_scope", "per_structure"))),
            "graph_family": graph_family_from_rule(rule) if not isinstance(rule, Mapping) else str(params.get("graph_family", "unclassified_graph")),
            "structure_hash": params.get("structure_hash"),
            "ensemble_size": params.get("ensemble_size"),
            "parent_rule_name": params.get("parent_rule_name"),
            "parent_rule_kind": params.get("parent_rule_kind"),
            "derivation_method": params.get("derivation_method"),
            "pair_cutoffs": params.get("pair_cutoffs", params.get("cutoffs", [])),
            "pair_intervals": params.get("pair_intervals", []),
            "derivation_pair_summary": _derivation_pair_summary(deriv),
            "derivation": deriv,
            "provenance": _compact_rule_provenance(_rule_provenance(rule)),
        }
        out.append(json_sanitize(rec))
    return out


def _compact_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row or {})
    for key in ("graph_rule_parameters", "representation_rule_parameters"):
        val = out.get(key)
        if isinstance(val, Mapping):
            out[key] = _compact_rule_parameters(val, rule_or_record={"name": out.get("graph_rule_name", out.get("representation_rule_name", "rule")), "kind": out.get("graph_rule_kind", "hard_cutoff"), "parameters": val}, include_derivation_summary=False)
        elif isinstance(val, str) and len(val) > _MAX_CSV_FIELD_CHARS:
            try:
                parsed = json.loads(val)
                if isinstance(parsed, Mapping):
                    out[key] = _compact_rule_parameters(parsed, rule_or_record={"name": out.get("graph_rule_name", out.get("representation_rule_name", "rule")), "kind": out.get("graph_rule_kind", "hard_cutoff"), "parameters": parsed}, include_derivation_summary=False)
                else:
                    out[key] = {"payload_ref": "csv:" + _short_hash(val, n=24), "payload_length": len(val), "payload_omitted": True}
            except Exception:
                out[key] = {"payload_ref": "csv:" + _short_hash(val, n=24), "payload_length": len(val), "payload_omitted": True}
    for key in ("graph_rule_provenance", "representation_rule_provenance"):
        val = out.get(key)
        if isinstance(val, Mapping):
            out[key] = _compact_rule_provenance(val)
        elif isinstance(val, str) and len(val) > _MAX_CSV_FIELD_CHARS:
            try:
                parsed = json.loads(val)
                out[key] = _compact_rule_provenance(parsed)
            except Exception:
                out[key] = {"payload_ref": "csv:" + _short_hash(val, n=24), "payload_length": len(val), "payload_omitted": True}
    return json_sanitize(out)


def _compact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_metric_row(r) if isinstance(r, Mapping) else {} for r in list(rows or [])]


def _finite_or_none(value: Any) -> Optional[float]:
    return finite_or_none(value)


def _pair_key(a: int, b: int) -> tuple[int, int]:
    ai = int(a)
    bi = int(b)
    return (ai, bi) if ai <= bi else (bi, ai)


def metric_family_for_name(name: str) -> str:
    s = str(name)
    if s.startswith("coord_"):
        return "coordination"
    if s.startswith("angle_"):
        return "angle"
    if s.startswith("ring_"):
        return "ring"
    if s.startswith("bondlen_") or s.startswith("bond_incidence_"):
        return "bond"
    if s.startswith("defect_") or "_defect_" in s:
        return "defect"
    if s.startswith("component_") or s.startswith("graph_") or s.startswith("path_") or s.startswith("local_"):
        return "graph_topology"
    if s.startswith("homopolar_"):
        return "homopolar_bond"
    if s.startswith("motif_"):
        return "motif"
    if s.startswith("edge_sharing_"):
        return "edge_sharing"
    if s.startswith("soft_"):
        return "soft_graph"
    return "graph_metric"


def _material_id_from_graph(graph: StructureGraph) -> str:
    counts: dict[str, int] = {}
    for sp in list(graph.species or []):
        key = str(sp)
        counts[key] = int(counts.get(key, 0)) + 1
    return "-".join(f"{k}{counts[k]}" for k in sorted(counts)) if counts else "unknown"


def _row_base(*, box_id: int, graph: StructureGraph, metric_name: str, metric_value: Any, metric_family: Optional[str] = None) -> dict[str, Any]:
    rule = graph.graph_rule
    full_params = dict(rule.parameters or {})
    params = _compact_rule_parameters(full_params, rule_or_record=rule, include_derivation_summary=False)
    family = graph_family_from_rule(rule)
    scope = str(full_params.get("graph_rule_scope", full_params.get("rule_scope", "per_structure")))
    compact_rule = GraphRule(
        name=str(rule.name),
        kind=str(rule.kind),
        parameters=params,
        provenance=_compact_rule_provenance(rule.provenance),
    )
    rep = graph_rule_to_representation_fields(compact_rule, structure_hash=str(graph.structure_hash), graph_family=family)
    row = metric_result_row(
        box_id=int(box_id),
        structure_hash=str(graph.structure_hash),
        material_id=str(params.get("material_id", _material_id_from_graph(graph))),
        metric_family=str(metric_family or metric_family_for_name(metric_name)),
        metric_name=str(metric_name),
        metric_value=metric_value,
        metric_units=metric_units_for_name(str(metric_name)),
        representation_fields=rep,
    )
    # Backward-compatible graph_* aliases are retained for existing notebooks and
    # CSV parsers, but representation_rule_* is now the canonical map schema.
    row.update(
        {
            "graph_rule_scope": scope,
            "graph_family": family,
            "graph_rule_name": str(rule.name),
            "graph_rule_kind": str(rule.kind),
            "graph_rule_parameters": params,
            "graph_rule_provenance": _compact_rule_provenance(rule.provenance),
        }
    )
    return json_sanitize(row)


def _graph_nx(graph: StructureGraph, *, weighted: bool = False) -> "nx.Graph":
    G = nx.Graph()
    G.add_nodes_from([int(x) for x in graph.nodes])
    for (a, b), d, w in zip(graph.edges, graph.edge_distances, graph.edge_weights):
        attrs = {"distance": float(d), "weight": float(w)} if weighted else {}
        G.add_edge(int(a), int(b), **attrs)
    return G


def _graph_topology_values(graph: StructureGraph) -> dict[str, float]:
    n = max(1, len(graph.nodes))
    G = _graph_nx(graph)
    degrees = np.asarray([float(G.degree(i)) for i in graph.nodes], dtype=float)
    weighted_degree = np.zeros(len(graph.nodes), dtype=float)
    for (a, b), w in zip(graph.edges, graph.edge_weights):
        weighted_degree[int(a)] += float(w)
        weighted_degree[int(b)] += float(w)

    comps = list(nx.connected_components(G)) if len(graph.nodes) > 0 else []
    comp_sizes = [len(c) for c in comps]
    largest = max(comp_sizes) if comp_sizes else 0
    vals: dict[str, float] = {
        "graph_edge_count": float(len(graph.edges)),
        "graph_edge_weight_sum": float(np.sum(np.asarray(graph.edge_weights, dtype=float))) if graph.edge_weights else 0.0,
        "local_degree_mean": float(np.mean(degrees)) if degrees.size else float("nan"),
        "local_degree_std": float(np.std(degrees, ddof=1)) if degrees.size > 1 else 0.0,
        "local_weighted_degree_mean": float(np.mean(weighted_degree)) if weighted_degree.size else float("nan"),
        "component_count": float(len(comps)),
        "component_largest_fraction": float(largest) / float(n),
    }

    # Homopolar edges are graph-derived because the edge set comes from G_lambda.
    hom = 0
    for a, b in graph.edges:
        if graph.species[int(a)] == graph.species[int(b)]:
            hom += 1
    vals["homopolar_bond_count"] = float(hom)
    vals["homopolar_bond_fraction"] = float(hom) / float(len(graph.edges)) if graph.edges else 0.0

    if largest > 1:
        largest_nodes = max(comps, key=len)
        sub = G.subgraph(largest_nodes)
        try:
            vals["path_length_largest_component_mean"] = float(nx.average_shortest_path_length(sub))
        except Exception:
            vals["path_length_largest_component_mean"] = float("nan")
    else:
        vals["path_length_largest_component_mean"] = float("nan")

    try:
        tri = sum(nx.triangles(G).values()) / 3.0
    except Exception:
        tri = 0.0
    vals["motif_triangle_count"] = float(tri)

    adj = {int(i): set(G.neighbors(int(i))) for i in graph.nodes}
    edge_sharing = 0
    nodes = [int(i) for i in graph.nodes]
    for u_i, u in enumerate(nodes):
        au = adj.get(u, set())
        if len(au) < 2:
            continue
        for v in nodes[u_i + 1 :]:
            if len(au & adj.get(v, set())) >= 2:
                edge_sharing += 1
    vals["edge_sharing_unit_count"] = float(edge_sharing)
    return vals


def _angle_counts(frame: Any, metrics: StructureMetricsConfig, graph: StructureGraph, *, type_to_species: Optional[Sequence[str]]) -> dict[str, float]:
    if graph.is_soft:
        return {}
    nbr_ids, nbr_vecs, _nbr_dists, _nbr_w = directed_neighbor_lists(graph, int(getattr(frame, "n_atoms")))
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    vals: dict[str, float] = {}
    for am in metrics.angles:
        a_sel, b_sel, c_sel = am.triplet
        aset = set(int(x) for x in _resolve_selector(a_sel, type_to_species))
        bset = set(int(x) for x in _resolve_selector(b_sel, type_to_species))
        cset = set(int(x) for x in _resolve_selector(c_sel, type_to_species))
        count = 0
        for b_idx in range(int(getattr(frame, "n_atoms"))):
            if int(types[b_idx]) not in bset:
                continue
            neighbors = [int(nb) for nb in nbr_ids[b_idx]]
            for ia, ic in combinations(neighbors, 2):
                ta = int(types[int(ia)])
                tc = int(types[int(ic)])
                if (ta in aset and tc in cset) or (tc in aset and ta in cset):
                    count += 1
        vals[f"angle_{a_sel}-{b_sel}-{c_sel}_count"] = float(count)
    return vals


def _soft_coordination_values(frame: Any, metrics: StructureMetricsConfig, graph: StructureGraph, *, type_to_species: Optional[Sequence[str]]) -> dict[str, float]:
    if not graph.is_soft:
        return {}
    nbr_ids, _nbr_vecs, _nbr_dists, nbr_w = directed_neighbor_lists(graph, int(getattr(frame, "n_atoms")))
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    vals: dict[str, float] = {}
    for cm in metrics.coordinations:
        cset = set(int(x) for x in _resolve_selector(cm.central, type_to_species))
        nset = set(int(x) for x in _resolve_selector(cm.neighbor, type_to_species))
        coords: list[float] = []
        amb: list[float] = []
        for i in range(int(getattr(frame, "n_atoms"))):
            if int(types[i]) not in cset:
                continue
            c = 0.0
            a = 0.0
            for nb, w in zip(nbr_ids[i], nbr_w[i]):
                if int(types[int(nb)]) not in nset:
                    continue
                ww = float(w)
                c += ww
                a += ww * (1.0 - ww)
            coords.append(float(c))
            amb.append(float(a))
        name = f"soft_coord_{cm.central}-{cm.neighbor}"
        if coords:
            vals[f"{name}_mean"] = float(np.mean(np.asarray(coords, dtype=float)))
            vals[f"{name}_ambiguity_mean"] = float(np.mean(np.asarray(amb, dtype=float)))
            vals[f"{name}_ambiguity_max"] = float(np.max(np.asarray(amb, dtype=float)))
        else:
            vals[f"{name}_mean"] = float("nan")
            vals[f"{name}_ambiguity_mean"] = float("nan")
            vals[f"{name}_ambiguity_max"] = float("nan")
    return vals


def _defect_graph_values(frame: Any, metrics: StructureMetricsConfig, graph: StructureGraph, *, type_to_species: Optional[Sequence[str]]) -> dict[str, float]:
    if graph.is_soft:
        return {}
    from .structure import compute_coordination_defects

    defects = compute_coordination_defects(frame, metrics, cutoffs={}, type_to_species=type_to_species, graph=graph)
    if not defects:
        return {}
    G = _graph_nx(graph)
    pos = np.asarray(getattr(frame, "positions"), dtype=float)
    cell = np.asarray(getattr(frame, "cell"), dtype=float)
    origin = np.asarray(getattr(frame, "origin"), dtype=float)
    pbc = frame_pbc(frame)
    frac = _wrap_frac((pos - origin) @ np.linalg.inv(cell), pbc=pbc)
    vals: dict[str, float] = {}
    for name, detail in defects.items():
        if not isinstance(detail, Mapping):
            continue
        prefix = f"defect_{name}"
        vals[f"{prefix}_n_defective"] = float(detail.get("n_defective", 0) or 0)
        vals[f"{prefix}_fraction"] = float(detail.get("defect_fraction", float("nan")))
        vals[f"{prefix}_has_defect"] = 1.0 if bool(detail.get("has_defect", False)) else 0.0
        idxs = [int(i) for i in detail.get("defective_idx", []) or []]
        if idxs:
            sub = G.subgraph(idxs)
            comps = list(nx.connected_components(sub))
            vals[f"{prefix}_cluster_count"] = float(len(comps))
            vals[f"{prefix}_largest_cluster_size"] = float(max((len(c) for c in comps), default=0))
        else:
            vals[f"{prefix}_cluster_count"] = 0.0
            vals[f"{prefix}_largest_cluster_size"] = 0.0
        if len(idxs) >= 2:
            defect_pairs = list(combinations(idxs, 2))
            pair_i = np.asarray([int(a) for a, _b in defect_pairs], dtype=int)
            pair_j = np.asarray([int(b) for _a, b in defect_pairs], dtype=int)
            _dr, mic_dist = _mic_displacements(
                frac, cell, pair_i, pair_j, pbc=pbc
            )
            ed: list[float] = [float(x) for x in np.asarray(mic_dist, dtype=float)]
            gd: list[float] = []
            for a, b in defect_pairs:
                try:
                    gd.append(float(nx.shortest_path_length(G, int(a), int(b))))
                except Exception:
                    pass
            vals[f"{prefix}_defect_distance_min"] = float(np.min(ed)) if ed else float("nan")
            vals[f"{prefix}_graph_path_mean"] = float(np.mean(gd)) if gd else float("nan")
        else:
            vals[f"{prefix}_defect_distance_min"] = float("nan")
            vals[f"{prefix}_graph_path_mean"] = float("nan")
    return vals




def _metric_pair_tokens(metric_name: str) -> Optional[tuple[str, str]]:
    name = str(metric_name)
    for prefix in ("bond_incidence_", "bondlen_", "coord_"):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            # Strip common suffixes.
            for suffix in ("_count", "_mean", "_std", "_fraction"):
                if rest.endswith(suffix):
                    rest = rest[: -len(suffix)]
            parts = rest.split("-")
            if len(parts) >= 2:
                return str(parts[0]), str(parts[1])
    return None


def _selector_token_to_types(token: str, type_to_species: Optional[Sequence[str]]) -> list[int]:
    try:
        return [int(x) for x in _resolve_selector(token, type_to_species)]
    except Exception:
        return []


def _rule_pair_keys(rule: GraphRule) -> set[tuple[int, int]]:
    return set(pair_cutoffs_from_parameters(dict(rule.parameters or {})).keys())


def _metric_pair_in_rule(metric_name: str, rule: GraphRule, type_to_species: Optional[Sequence[str]]) -> bool:
    tokens = _metric_pair_tokens(metric_name)
    if tokens is None:
        return True
    keys = _rule_pair_keys(rule)
    if not keys:
        return True
    a_types = _selector_token_to_types(tokens[0], type_to_species)
    b_types = _selector_token_to_types(tokens[1], type_to_species)
    if not a_types or not b_types:
        return True
    wanted = {_pair_key(a, b) for a in a_types for b in b_types}
    return bool(keys & wanted)


def _network_metric_allowed(metric_name: str, rule: GraphRule, type_to_species: Optional[Sequence[str]]) -> bool:
    name = str(metric_name)
    # Do not report trivial same-species/homopolar results from the backbone graph;
    # those belong to the candidate-contact graph.
    if name.startswith("homopolar_") or name.startswith("candidate_contact_"):
        return False
    if name.startswith("bondlen_") or name.startswith("bond_incidence_"):
        return _metric_pair_in_rule(name, rule, type_to_species)
    return True


def _soft_edge_weight_values(graph: StructureGraph) -> dict[str, float]:
    w = np.asarray(graph.edge_weights, dtype=float) if graph.edge_weights else np.asarray([], dtype=float)
    vals = {
        "soft_edge_count": float(len(graph.edges)),
        "soft_edge_weight_sum": float(np.sum(w)) if w.size else 0.0,
        "soft_edge_weight_mean": float(np.mean(w)) if w.size else float("nan"),
        "soft_edge_weight_std": float(np.std(w, ddof=1)) if w.size > 1 else 0.0,
    }
    return vals


def _candidate_contact_values(frame: Any, metrics: StructureMetricsConfig, graph: StructureGraph, *, type_to_species: Optional[Sequence[str]]) -> dict[str, float]:
    """Candidate close-contact/homopolar descriptors from a non-backbone graph."""

    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    dist_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (a, b), d in zip(graph.edges, graph.edge_distances):
        key = _pair_key(int(types[int(a)]), int(types[int(b)]))
        dist_by_pair[key].append(float(d))

    vals: dict[str, float] = {
        "candidate_contact_edge_count": float(len(graph.edges)),
    }
    same_count = 0
    for (a, b), ds in sorted(dist_by_pair.items()):
        lab_a = str(type_to_species[a - 1]) if type_to_species is not None and 1 <= a <= len(type_to_species) else f"type{a}"
        lab_b = str(type_to_species[b - 1]) if type_to_species is not None and 1 <= b <= len(type_to_species) else f"type{b}"
        arr = np.asarray(ds, dtype=float)
        vals[f"candidate_contact_bond_incidence_{lab_a}-{lab_b}_count"] = float(arr.size)
        vals[f"candidate_contact_bondlen_{lab_a}-{lab_b}_mean"] = float(np.mean(arr)) if arr.size else float("nan")
        vals[f"candidate_contact_bondlen_{lab_a}-{lab_b}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        if int(a) == int(b):
            same_count += int(arr.size)
            vals[f"homopolar_{lab_a}-{lab_b}_count"] = float(arr.size)
            vals[f"homopolar_{lab_a}-{lab_b}_fraction"] = float(arr.size) / float(len(graph.edges)) if graph.edges else 0.0
    vals["homopolar_bond_count"] = float(same_count)
    vals["homopolar_bond_fraction"] = float(same_count) / float(len(graph.edges)) if graph.edges else 0.0

    # Motif/edge-sharing perturbation descriptors are explicitly prefixed so they
    # cannot be mistaken for the primary network topology.
    topo = _graph_topology_values(graph)
    for key in ("motif_triangle_count", "edge_sharing_unit_count", "graph_edge_count", "local_degree_mean"):
        if key in topo:
            vals[f"candidate_contact_{key}"] = float(topo[key])
    return vals


def _family_filtered_values(
    frame: Any,
    metrics: StructureMetricsConfig,
    graph: StructureGraph,
    *,
    type_to_species: Optional[Sequence[str]],
) -> dict[str, Any]:
    from .structure import StructureMetrics, compute_structure_metrics

    family = graph_family_from_rule(graph.graph_rule)
    if family == "soft_ambiguity_graph":
        combined: dict[str, Any] = {}
        combined.update(_soft_coordination_values(frame, metrics, graph, type_to_species=type_to_species))
        combined.update(_soft_edge_weight_values(graph))
        return combined
    if family == "candidate_contact_graph":
        return _candidate_contact_values(frame, metrics, graph, type_to_species=type_to_species)

    sm = compute_structure_metrics(frame, metrics, cutoffs={}, type_to_species=type_to_species, graph=graph)
    vals: Mapping[str, Any]
    if isinstance(sm, StructureMetrics):
        vals = sm.values
    elif isinstance(sm, Mapping):
        vals = sm
    else:
        vals = {}
    combined = dict(vals)
    combined.update(_angle_counts(frame, metrics, graph, type_to_species=type_to_species))
    topo = _graph_topology_values(graph)
    if family == "network_graph":
        topo = {k: v for k, v in topo.items() if not str(k).startswith("homopolar_")}
    combined.update(topo)
    if family in {"network_graph", "legacy_single_cutoff_graph", "unclassified_graph"}:
        combined.update(_defect_graph_values(frame, metrics, graph, type_to_species=type_to_species))
    if family == "network_graph":
        combined = {k: v for k, v in combined.items() if _network_metric_allowed(str(k), graph.graph_rule, type_to_species)}
    return combined

def compute_graph_metric_rows(
    frame: Any,
    metrics: StructureMetricsConfig,
    *,
    box_id: int,
    graph_rules: Sequence[GraphRule],
    type_to_species: Optional[Sequence[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute graph-derived descriptors for every explicit graph rule."""

    rows: list[dict[str, Any]] = []
    graph_rule_records: list[dict[str, Any]] = []
    for rule in graph_rules:
        graph = build_graph(frame, rule, type_to_species=type_to_species)
        graph_rule_records.append(_compact_rule_record(graph.graph_rule))
        combined = _family_filtered_values(frame, metrics, graph, type_to_species=type_to_species)
        for name, value in sorted(combined.items()):
            rows.append(_row_base(box_id=int(box_id), graph=graph, metric_name=str(name), metric_value=value))
    return rows, graph_rule_records


def _distances_for_coordination(frame: Any, cm: Any, *, r_max: float, type_to_species: Optional[Sequence[str]]) -> tuple[list[int], list[list[float]]]:
    from .graph import GraphRule, build_hard_graph, directed_neighbor_lists

    # A temporary all-pair graph over the conservative upper cutoff lets the
    # robust partition use the same PBC/MIC machinery as descriptor graphs.
    tmp = build_hard_graph(
        frame,
        GraphRule(name="robust_interval_search", kind="hard_cutoff", parameters={"cutoff": float(r_max)}, provenance="robust_coordination_partition"),
        type_to_species=type_to_species,
    )
    nbr_ids, _nbr_vecs, nbr_dists, _nbr_w = directed_neighbor_lists(tmp, int(getattr(frame, "n_atoms")))
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    cset = set(int(x) for x in _resolve_selector(cm.central, type_to_species))
    nset = set(int(x) for x in _resolve_selector(cm.neighbor, type_to_species))
    central_idx: list[int] = []
    ds_by_central: list[list[float]] = []
    for i in range(int(getattr(frame, "n_atoms"))):
        if int(types[i]) not in cset:
            continue
        vals: list[float] = []
        for nb, d in zip(nbr_ids[i], nbr_dists[i]):
            if int(types[int(nb)]) in nset:
                vals.append(float(d))
        central_idx.append(int(i))
        ds_by_central.append(sorted(vals))
    return central_idx, ds_by_central


def robust_coordination_partition(
    frame: Any,
    metrics: StructureMetricsConfig,
    *,
    box_id: int,
    interval_rule: GraphRule,
    type_to_species: Optional[Sequence[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Interval robust coordination and shell separability.

    This module is material-profile driven: if no integer expected
    coordination is configured for a coordination metric, it returns an
    explicit not_applicable row rather than failing or inventing a valence.
    """

    params = dict(interval_rule.parameters or {})
    pair_intervals = pair_intervals_from_parameters(params)
    global_r_min = params.get("r_min", params.get("min", None))
    global_r_max = params.get("r_max", params.get("max", None))
    if global_r_min is not None and global_r_max is not None:
        try:
            gr_min = float(global_r_min)
            gr_max = float(global_r_max)
        except Exception:
            gr_min = float("nan")
            gr_max = float("nan")
    else:
        gr_min = float("nan")
        gr_max = float("nan")
    sh = structure_hash(frame, type_to_species=type_to_species)
    species = []
    try:
        from .graph import species_from_frame
        species = species_from_frame(frame, type_to_species=type_to_species)
    except Exception:
        species = []
    counts_by_sp: dict[str, int] = {}
    for sp in species:
        counts_by_sp[str(sp)] = int(counts_by_sp.get(str(sp), 0)) + 1
    material_id = "-".join(f"{k}{counts_by_sp[k]}" for k in sorted(counts_by_sp)) if counts_by_sp else "unknown"
    rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []

    def _interval_for_coordination(cm: Any) -> tuple[float, float, list[tuple[int, int]]]:
        c_types = [int(x) for x in _resolve_selector(cm.central, type_to_species)]
        n_types = [int(x) for x in _resolve_selector(cm.neighbor, type_to_species)]
        keys = sorted({_pair_key(a, b) for a in c_types for b in n_types})
        vals = [pair_intervals[k] for k in keys if k in pair_intervals]
        if vals:
            rlo = min(float(v[0]) for v in vals)
            rhi = max(float(v[1]) for v in vals)
            return rlo, rhi, keys
        return float(gr_min), float(gr_max), keys

    def _base(cm: Any, metric: str, expected: Any, r_min: Any, r_max: Any, interval_pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
        rule_params_full = dict(interval_rule.parameters or {})
        rule_params_full.setdefault("coordination_interval_pairs", [[int(a), int(b)] for a, b in interval_pairs])
        if r_min is not None and r_max is not None and finite_or_none(r_min) is not None and finite_or_none(r_max) is not None:
            rule_params_full.setdefault("coordination_interval", {"r_min": float(r_min), "r_max": float(r_max)})
        rule_params = _compact_rule_parameters(rule_params_full, rule_or_record=interval_rule, include_derivation_summary=False)
        compact_interval_rule = GraphRule(
            name=str(interval_rule.name),
            kind=str(interval_rule.kind),
            parameters=rule_params,
            provenance=_compact_rule_provenance(interval_rule.provenance),
        )
        rep = graph_rule_to_representation_fields(compact_interval_rule, structure_hash=str(sh), graph_family=graph_family_from_rule(interval_rule))
        out = {
            "box_id": int(box_id),
            "structure_hash": sh,
            "material_id": material_id,
            "graph_rule_scope": str(rule_params.get("graph_rule_scope", rule_params.get("rule_scope", "per_structure"))),
            "graph_family": graph_family_from_rule(interval_rule),
            "graph_rule_name": str(interval_rule.name),
            "graph_rule_kind": str(interval_rule.kind),
            "graph_rule_parameters": rule_params,
            "graph_rule_provenance": _compact_rule_provenance(interval_rule.provenance),
            "coordination_metric": metric,
            "expected_coordination": (None if expected is None else int(expected)),
            "r_min": finite_or_none(r_min),
            "r_max": finite_or_none(r_max),
            "pair_intervals_used": [[int(a), int(b)] for a, b in interval_pairs],
            "metric_family": "coordination",
            "representation_family": graph_family_from_rule(interval_rule),
        }
        out.update(rep)
        return out

    def _quant(vals: Sequence[float], q: float) -> Optional[float]:
        arr = np.asarray([float(x) for x in vals if math.isfinite(float(x))], dtype=float)
        if arr.size == 0:
            return None
        return float(np.percentile(arr, float(q)))

    for cm in metrics.coordinations:
        expected = getattr(cm, "expected", None)
        metric = f"coord_{cm.central}-{cm.neighbor}"
        if expected is None:
            base = _base(cm, metric, None, None, None, [])
            rows.append(
                json_sanitize(
                    {
                        **base,
                        "n_central": 0,
                        "robust_ideal_fraction": None,
                        "robust_undercoordinated_fraction": None,
                        "robust_overcoordinated_fraction": None,
                        "ambiguous_fraction": None,
                        "robust_ideal_count": None,
                        "robust_undercoordinated_count": None,
                        "robust_overcoordinated_count": None,
                        "ambiguous_count": None,
                        "status": "not_applicable",
                        "reason": "expected_integer_coordination_not_configured",
                        "metric_status": "not_applicable",
                        "metric_status_reason": "expected_integer_coordination_not_configured",
                        "numerical_status": "not_applicable",
                        "uncertainty_status": "not_applicable",
                    }
                )
            )
            shell_rows.append(
                json_sanitize(
                    {
                        **base,
                        "n_central": 0,
                        "q05_d_z": None,
                        "q50_d_z": None,
                        "q95_d_z": None,
                        "q05_d_z_plus_1": None,
                        "q50_d_z_plus_1": None,
                        "q95_d_z_plus_1": None,
                        "d_z_mean": None,
                        "d_z_plus_1_mean": None,
                        "shell_gap_mean": None,
                        "shell_gap_min": None,
                        "shell_gap_p05": None,
                        "shell_gap_median": None,
                        "shell_overlap_score": None,
                        "n_missing_z_plus_1": None,
                        "status": "not_applicable",
                        "reason": "expected_integer_coordination_not_configured",
                        "metric_status": "not_applicable",
                        "metric_status_reason": "expected_integer_coordination_not_configured",
                        "numerical_status": "not_applicable",
                        "uncertainty_status": "not_applicable",
                    }
                )
            )
            continue
        z = int(expected)
        r_min, r_max, interval_pairs = _interval_for_coordination(cm)
        base = _base(cm, metric, z, r_min, r_max, interval_pairs)
        if not (math.isfinite(r_min) and math.isfinite(r_max) and r_max >= r_min > 0):
            reason = "valid_graph_interval_not_available"
            rows.append(json_sanitize({**base, "n_central": 0, "robust_ideal_fraction": None, "robust_undercoordinated_fraction": None, "robust_overcoordinated_fraction": None, "ambiguous_fraction": None, "robust_ideal_count": None, "robust_undercoordinated_count": None, "robust_overcoordinated_count": None, "ambiguous_count": None, "status": "not_applicable", "reason": reason, "metric_status": "not_applicable", "metric_status_reason": reason, "numerical_status": "not_applicable", "uncertainty_status": "not_applicable"}))
            shell_rows.append(json_sanitize({**base, "n_central": 0, "q05_d_z": None, "q50_d_z": None, "q95_d_z": None, "q05_d_z_plus_1": None, "q50_d_z_plus_1": None, "q95_d_z_plus_1": None, "d_z_mean": None, "d_z_plus_1_mean": None, "shell_gap_mean": None, "shell_gap_min": None, "shell_gap_p05": None, "shell_gap_median": None, "shell_overlap_score": None, "n_missing_z_plus_1": None, "status": "not_applicable", "reason": reason, "metric_status": "not_applicable", "metric_status_reason": reason, "numerical_status": "not_applicable", "uncertainty_status": "not_applicable"}))
            continue
        central_idx, ds_by_central = _distances_for_coordination(frame, cm, r_max=float(r_max), type_to_species=type_to_species)
        counts = {"robust_ideal": 0, "robust_undercoordinated": 0, "robust_overcoordinated": 0, "ambiguous": 0}
        dz_vals: list[float] = []
        dz1_vals: list[float] = []
        gap_vals: list[float] = []
        n_missing_zp1 = 0
        for _idx, ds in zip(central_idx, ds_by_central):
            inf = float("inf")
            d_z = ds[z - 1] if z > 0 and len(ds) >= z else inf
            d_z1 = ds[z] if len(ds) >= z + 1 else inf
            if math.isfinite(d_z):
                dz_vals.append(float(d_z))
            if math.isfinite(d_z1):
                dz1_vals.append(float(d_z1))
            else:
                n_missing_zp1 += 1
            if math.isfinite(d_z) and math.isfinite(d_z1):
                gap_vals.append(float(d_z1 - d_z))
            if z == 0:
                lab = "robust_ideal" if d_z1 > r_max else ("robust_overcoordinated" if d_z1 <= r_min else "ambiguous")
            elif d_z <= r_min and d_z1 > r_max:
                lab = "robust_ideal"
            elif d_z > r_max:
                lab = "robust_undercoordinated"
            elif d_z1 <= r_min:
                lab = "robust_overcoordinated"
            else:
                lab = "ambiguous"
            counts[lab] += 1
        n = max(1, len(ds_by_central))
        rows.append(
            json_sanitize(
                {
                    **base,
                    "n_central": int(len(ds_by_central)),
                    "robust_ideal_fraction": float(counts["robust_ideal"]) / float(n),
                    "robust_undercoordinated_fraction": float(counts["robust_undercoordinated"]) / float(n),
                    "robust_overcoordinated_fraction": float(counts["robust_overcoordinated"]) / float(n),
                    "ambiguous_fraction": float(counts["ambiguous"]) / float(n),
                    "robust_ideal_count": int(counts["robust_ideal"]),
                    "robust_undercoordinated_count": int(counts["robust_undercoordinated"]),
                    "robust_overcoordinated_count": int(counts["robust_overcoordinated"]),
                    "ambiguous_count": int(counts["ambiguous"]),
                    "status": "ok",
                    "reason": "",
                    "metric_status": "ok",
                    "metric_status_reason": "",
                    "numerical_status": "ok",
                    "uncertainty_status": "not_applicable",
                    "uncertainty_status_reason": "per_structure_partition",
                }
            )
        )
        dz_arr = np.asarray(dz_vals, dtype=float)
        dz1_arr = np.asarray(dz1_vals, dtype=float)
        gap_arr = np.asarray(gap_vals, dtype=float)
        if dz_arr.size and dz1_arr.size:
            q95_dz = float(np.percentile(dz_arr, 95.0))
            q05_dz1 = float(np.percentile(dz1_arr, 5.0))
            overlap = 0.5 * (float(np.mean(dz_arr >= q05_dz1)) + float(np.mean(dz1_arr <= q95_dz)))
        else:
            overlap = None
        shell_status = "ok" if int(n_missing_zp1) == 0 else "incomplete"
        shell_reason = "" if shell_status == "ok" else "z_plus_1_neighbour_not_found_within_search_radius"
        shell_rows.append(
            json_sanitize(
                {
                    **base,
                    "n_central": int(len(ds_by_central)),
                    "q05_d_z": _quant(dz_vals, 5.0),
                    "q50_d_z": _quant(dz_vals, 50.0),
                    "q95_d_z": _quant(dz_vals, 95.0),
                    "q05_d_z_plus_1": _quant(dz1_vals, 5.0),
                    "q50_d_z_plus_1": _quant(dz1_vals, 50.0),
                    "q95_d_z_plus_1": _quant(dz1_vals, 95.0),
                    "d_z_mean": float(np.mean(dz_arr)) if dz_arr.size else None,
                    "d_z_plus_1_mean": float(np.mean(dz1_arr)) if dz1_arr.size else None,
                    "shell_gap_mean": float(np.mean(gap_arr)) if gap_arr.size else None,
                    "shell_gap_min": float(np.min(gap_arr)) if gap_arr.size else None,
                    "shell_gap_p05": float(np.percentile(gap_arr, 5.0)) if gap_arr.size else None,
                    "shell_gap_median": float(np.median(gap_arr)) if gap_arr.size else None,
                    "shell_overlap_score": overlap,
                    "n_missing_z_plus_1": int(n_missing_zp1),
                    "status": shell_status,
                    "reason": shell_reason,
                    "metric_status": shell_status,
                    "metric_status_reason": shell_reason,
                    "numerical_status": "ok" if shell_status == "ok" else "insufficient_samples",
                    "uncertainty_status": "not_applicable",
                    "uncertainty_status_reason": "per_structure_shell_summary",
                }
            )
        )
    return rows, shell_rows


def graph_analysis_for_frame(
    frame: Any,
    metrics: StructureMetricsConfig,
    *,
    box_id: int,
    type_to_species: Optional[Sequence[str]],
    legacy_cutoffs: Mapping[Tuple[int, int], float],
    source_path: Optional[Path],
    source_role: Optional[str],
    density: Optional[float],
) -> dict[str, Any]:
    raw_rules = list(getattr(metrics, "graph_rules", []) or [])
    manifest = manifest_row_from_frame(
        frame,
        box_id=int(box_id),
        source_path=source_path,
        source_role=source_role,
        type_to_species=type_to_species,
        density=density,
    )
    # Manifest lock is verified before descriptor evaluation. This protects the
    # graph pass from stale first-frame/final-frame or unsanitised restart input.
    verify_manifest_row(frame, manifest, type_to_species=type_to_species)
    rules, interval_rules = expand_graph_rules_for_frame(
        raw_rules,
        frame=frame,
        metrics=metrics,
        box_id=int(box_id),
        legacy_cutoffs=legacy_cutoffs,
        type_to_species=type_to_species,
    )
    metric_rows, _computed_rule_records = compute_graph_metric_rows(frame, metrics, box_id=int(box_id), graph_rules=rules, type_to_species=type_to_species)
    all_rule_objects = list(rules) + list(interval_rules)
    # Keep the direct Python API fully inspectable for a single structure; the
    # streaming/file writers compact this payload before it reaches public CSV or
    # full-ensemble JSON sidecars.
    rule_records = [rr.to_json() for rr in list(rules)]
    compact_rule_records = [_compact_rule_record(rr) for rr in list(rules)]
    adaptive_records: list[dict[str, Any]] = []
    seen_adaptive: set[str] = set()
    for rr in all_rule_objects:
        params = dict(getattr(rr, "parameters", {}) or {})
        if bool(params.get("rdf_adaptive", False)):
            rec = _compact_rule_record(rr)
            key = _json_dumps(rec)
            if key not in seen_adaptive:
                seen_adaptive.add(key)
                adaptive_records.append(rec)
    adaptive_derivation_records = _adaptive_derivation_records_from_rules(all_rule_objects)
    stability_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    for interval in list(interval_rules):
        rr, ss = robust_coordination_partition(frame, metrics, box_id=int(box_id), interval_rule=interval, type_to_species=type_to_species)
        stability_rows.extend(rr)
        shell_rows.extend(ss)
    return {
        "schema": "vitriflow.graph_analysis.v2",
        "structure_manifest": manifest,
        "representation_rules": _representation_records_from_graph_rules(compact_rule_records),
        "graph_rules": rule_records,
        "adaptive_graph_rule_records": adaptive_records,
        "adaptive_graph_rule_derivation_records": adaptive_derivation_records,
        "graph_metric_rows": metric_rows,
        "coordination_stability_rows": stability_rows,
        "shell_separability_rows": shell_rows,
        "legacy_single_cutoff": {
            "present": (not bool(raw_rules)) and bool(legacy_cutoffs),
            "label": "legacy_single_cutoff" if ((not bool(raw_rules)) and bool(legacy_cutoffs)) else None,
            "note": "Existing metrics/distributions fields are retained for backward compatibility; graph_metric_by_rule.csv is the graph-provenanced descriptor table. Adaptive RDF graph rules resolve per structure and do not use YAML-fixed Angstrom cutoffs.",
        },
    }


def _csv_value(value: Any) -> Any:
    out = csv_scalar(value)
    if isinstance(out, str) and len(out) > _MAX_CSV_FIELD_CHARS:
        return json.dumps(
            {
                "payload_omitted": True,
                "payload_length": int(len(out)),
                "payload_ref": "csv:" + _short_hash(out, n=24),
                "reason": "field_exceeded_streaming_csv_limit_use_json_sidecar",
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_rows = _compact_rows([r for r in list(rows or []) if isinstance(r, Mapping)])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in sorted(compact_rows, key=lambda r: _json_dumps(dict(r)) if isinstance(r, Mapping) else str(r)):
            w.writerow({c: _csv_value(row.get(c, "")) for c in columns})




# Canonical sidecar columns used by both the in-memory writer and the streaming
# writer.  Keep these stable for downstream tooling.
GRAPH_METRIC_COLUMNS = [
    "box_id",
    "structure_hash",
    "material_id",
    "graph_rule_scope",
    "graph_family",
    "graph_rule_name",
    "graph_rule_kind",
    "graph_rule_parameters",
    "graph_rule_provenance",
    "representation_rule_name",
    "representation_rule_kind",
    "representation_rule_parameters",
    "representation_rule_provenance",
    "representation_rule_version",
    "source_space",
    "representation_map",
    "descriptor_map",
    "target_distribution",
    "metric_family",
    "metric_name",
    "metric_value",
    "metric_units",
    "metric_status",
    "metric_status_reason",
    "numerical_status",
    "numerical_status_reason",
    "uncertainty_status",
    "uncertainty_status_reason",
    "n_samples",
    "normalization",
]

GRAPH_STABILITY_COLUMNS = [
    "box_id",
    "structure_hash",
    "material_id",
    "graph_rule_scope",
    "graph_family",
    "graph_rule_name",
    "graph_rule_kind",
    "graph_rule_parameters",
    "graph_rule_provenance",
    "representation_rule_name",
    "representation_rule_kind",
    "representation_rule_parameters",
    "representation_rule_provenance",
    "representation_rule_version",
    "source_space",
    "representation_map",
    "coordination_metric",
    "expected_coordination",
    "r_min",
    "r_max",
    "n_central",
    "robust_ideal_fraction",
    "robust_undercoordinated_fraction",
    "robust_overcoordinated_fraction",
    "ambiguous_fraction",
    "robust_ideal_count",
    "robust_undercoordinated_count",
    "robust_overcoordinated_count",
    "ambiguous_count",
    "status",
    "reason",
    "metric_status",
    "metric_status_reason",
    "numerical_status",
    "uncertainty_status",
]

GRAPH_SHELL_COLUMNS = [
    "box_id",
    "structure_hash",
    "material_id",
    "graph_rule_scope",
    "graph_family",
    "graph_rule_name",
    "graph_rule_kind",
    "graph_rule_parameters",
    "graph_rule_provenance",
    "representation_rule_name",
    "representation_rule_kind",
    "representation_rule_parameters",
    "representation_rule_provenance",
    "representation_rule_version",
    "source_space",
    "representation_map",
    "coordination_metric",
    "expected_coordination",
    "r_min",
    "r_max",
    "n_central",
    "q05_d_z",
    "q50_d_z",
    "q95_d_z",
    "q05_d_z_plus_1",
    "q50_d_z_plus_1",
    "q95_d_z_plus_1",
    "d_z_mean",
    "d_z_plus_1_mean",
    "shell_gap_mean",
    "shell_gap_min",
    "shell_gap_p05",
    "shell_gap_median",
    "shell_overlap_score",
    "n_missing_z_plus_1",
    "status",
    "reason",
    "metric_status",
    "metric_status_reason",
    "numerical_status",
    "uncertainty_status",
]

GRAPH_UNCERTAINTY_COLUMNS = [
    "graph_rule_scope",
    "graph_family",
    "representation_family",
    "metric_family",
    "metric_name",
    "metric_units",
    "n_graph_rules",
    "n_values",
    "lambda_min_rule",
    "lambda_max_rule",
    "mean_min",
    "mean_max",
    "width",
    "bootstrap_se",
    "width_over_se",
    "dominant_uncertainty",
    "uncertainty_status",
    "uncertainty_status_reason",
    "status",
    "reason",
    "rule_means",
]


def _write_jsonl_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows or []:
            if isinstance(row, Mapping):
                f.write(_json_dumps(dict(row)))
                f.write("\n")


def _write_csv_chunk(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    # Chunk files are sorted locally and then concatenated by box id.  This is
    # deterministic while avoiding a full-ensemble in-memory sort.
    _write_csv(Path(path), list(rows or []), list(columns))


def _chunk_box_id_from_path(path: Path) -> int:
    m = re.search(r"box_(\d+)", Path(path).name)
    if not m:
        return 10**12
    try:
        return int(m.group(1))
    except Exception:
        return 10**12


def _concat_csv_chunks(path: Path, chunks: Sequence[Path], columns: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _set_csv_field_limit()
    ordered = sorted([Path(c) for c in chunks if Path(c).exists()], key=lambda p: (_chunk_box_id_from_path(p), str(p)))
    with path.open("w", newline="") as out_f:
        w = csv.DictWriter(out_f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for chunk in ordered:
            with Path(chunk).open("r", newline="") as in_f:
                r = csv.DictReader(in_f)
                for row in r:
                    # Current chunks are already compact.  This extra compaction
                    # lets interrupted 0.4.29.16-style runs with oversized JSON
                    # fields be finalized instead of raising csv.field_size_limit.
                    crow = _compact_metric_row(row) if isinstance(row, Mapping) else {}
                    w.writerow({c: _csv_value(crow.get(c, "")) for c in columns})


def _jsonl_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted([Path(p) for p in paths if Path(p).exists()], key=lambda p: (_chunk_box_id_from_path(p), str(p))):
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, Mapping):
                    continue
                key = _json_dumps(obj)
                if key in seen:
                    continue
                seen.add(key)
                out.append(json_sanitize(dict(obj)))
    return out


def _write_json_array_payload(path: Path, *, schema: str, key: str, records: Sequence[Mapping[str, Any]]) -> None:
    write_json_strict(Path(path), {"schema": str(schema), str(key): [json_sanitize(dict(r)) for r in records if isinstance(r, Mapping)]})


def graph_uncertainty_summary_rows_from_metric_csv(path: Path) -> list[dict[str, Any]]:
    """Compute graph uncertainty summaries by streaming a metric CSV file."""
    path = Path(path)
    if not path.exists():
        return []
    by_metric: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    row_meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    _set_csv_field_limit()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = _finite_or_none(row.get("metric_value", None))
            if val is None:
                continue
            key = (
                str(row.get("graph_rule_scope", row.get("representation_rule_scope", "per_structure"))),
                str(row.get("graph_family", row.get("representation_family", "unclassified_graph"))),
                str(row.get("metric_family", "")),
                str(row.get("metric_name", "")),
            )
            rule = str(row.get("graph_rule_name", row.get("representation_rule_name", "")))
            by_metric[key][rule].append(float(val))
            row_meta.setdefault(key, dict(row))
    rows: list[dict[str, Any]] = []
    for (scope, graph_family, family, name), by_rule in sorted(by_metric.items()):
        means: dict[str, float] = {}
        ses: list[float] = []
        n_values = 0
        for rule, vals in sorted(by_rule.items()):
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            means[rule] = float(np.mean(arr))
            n_values += int(arr.size)
            se = _bootstrap_se(arr.tolist())
            if se is not None and math.isfinite(float(se)):
                ses.append(float(se))
        if not means:
            continue
        sorted_means = sorted(means.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
        mean_vals = [float(v) for _k, v in sorted_means]
        width = float(max(mean_vals) - min(mean_vals)) if len(mean_vals) > 1 else 0.0
        se_box = float(max(ses)) if ses else None
        dominant, width_over_se, uncertainty_status, uncertainty_reason = _dominant_uncertainty(width, se_box)
        status = "ok" if int(len(means)) > 1 else "not_applicable"
        reason = "" if status == "ok" else "single_representation_rule"
        meta = row_meta.get((scope, graph_family, family, name), {})
        rows.append(json_sanitize({
            "graph_rule_scope": scope,
            "graph_family": graph_family,
            "representation_family": graph_family,
            "metric_family": family,
            "metric_name": name,
            "metric_units": str(meta.get("metric_units", metric_units_for_name(name))),
            "n_graph_rules": int(len(means)),
            "n_values": int(n_values),
            "lambda_min_rule": str(sorted_means[0][0]),
            "lambda_max_rule": str(sorted_means[-1][0]),
            "mean_min": float(min(mean_vals)),
            "mean_max": float(max(mean_vals)),
            "width": float(width),
            "bootstrap_se": se_box,
            "width_over_se": width_over_se,
            "dominant_uncertainty": dominant,
            "uncertainty_status": uncertainty_status,
            "uncertainty_status_reason": uncertainty_reason,
            "status": status,
            "reason": reason,
            "rule_means": dict(means),
        }))
    return rows


def write_graph_analysis_entry_chunks(
    entry: Mapping[str, Any],
    chunk_dir: Path,
    *,
    metrics: Optional[StructureMetricsConfig] = None,
) -> dict[str, Any]:
    """Write one box's heavy graph-analysis payload to deterministic chunk files.

    The returned summary is intentionally small and safe to keep in memory or
    send back from a worker process.  Full descriptor rows remain on disk until
    ``finalize_streamed_graph_analysis_outputs`` concatenates them.
    """
    if not isinstance(entry, Mapping):
        return {"status": "skipped", "reason": "entry_not_mapping"}
    box_id = int(entry.get("box", entry.get("box_id", 0)) or 0)
    prefix = f"box_{box_id:06d}"
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ga = entry.get("graph_analysis", {}) if isinstance(entry.get("graph_analysis", {}), Mapping) else {}
    manifest = ga.get("structure_manifest", None)
    raw_graph_rules = [dict(r) for r in list(ga.get("graph_rules", []) or []) if isinstance(r, Mapping)]
    raw_adaptive_rules = [dict(r) for r in list(ga.get("adaptive_graph_rule_records", []) or []) if isinstance(r, Mapping)]
    raw_derivation_records = [dict(r) for r in list(ga.get("adaptive_graph_rule_derivation_records", []) or []) if isinstance(r, Mapping)]
    graph_rules = [_compact_rule_record(r) for r in raw_graph_rules]
    adaptive_rules = [_compact_rule_record(r) for r in raw_adaptive_rules]
    derivation_records = list(raw_derivation_records)
    if not derivation_records:
        derivation_records = _adaptive_derivation_records_from_rules(raw_graph_rules + raw_adaptive_rules)
    graph_rows = _compact_rows([dict(r) for r in list(ga.get("graph_metric_rows", []) or []) if isinstance(r, Mapping)])
    stability_rows = _compact_rows([dict(r) for r in list(ga.get("coordination_stability_rows", []) or []) if isinstance(r, Mapping)])
    shell_rows = _compact_rows([dict(r) for r in list(ga.get("shell_separability_rows", []) or []) if isinstance(r, Mapping)])
    coord_rows = _coordinate_metric_rows_from_entries([entry], metrics)
    metric_result_rows = list(graph_rows) + list(coord_rows)
    representation_records = _representation_records_from_graph_rules(graph_rules) + _representation_records_from_metric_rows(metric_result_rows)
    paths = {
        "manifest": chunk_dir / f"{prefix}_structure_manifest.jsonl",
        "graph_rules": chunk_dir / f"{prefix}_graph_rules.jsonl",
        "adaptive_graph_rules": chunk_dir / f"{prefix}_adaptive_graph_rules.jsonl",
        "representation_rules": chunk_dir / f"{prefix}_representation_rules.jsonl",
        "adaptive_graph_rule_derivations": chunk_dir / f"{prefix}_adaptive_graph_rule_derivations.jsonl",
        "graph_metric_by_rule": chunk_dir / f"{prefix}_graph_metric_by_rule.csv",
        "metric_results": chunk_dir / f"{prefix}_metric_results.csv",
        "coordination_stability": chunk_dir / f"{prefix}_coordination_stability.csv",
        "shell_separability": chunk_dir / f"{prefix}_shell_separability.csv",
    }
    _write_jsonl_rows(paths["manifest"], [dict(manifest)] if isinstance(manifest, Mapping) else [])
    _write_jsonl_rows(paths["graph_rules"], graph_rules)
    _write_jsonl_rows(paths["adaptive_graph_rules"], adaptive_rules)
    _write_jsonl_rows(paths["adaptive_graph_rule_derivations"], derivation_records)
    _write_jsonl_rows(paths["representation_rules"], representation_records)
    _write_csv_chunk(paths["graph_metric_by_rule"], graph_rows, GRAPH_METRIC_COLUMNS)
    _write_csv_chunk(paths["metric_results"], metric_result_rows, GRAPH_METRIC_COLUMNS)
    _write_csv_chunk(paths["coordination_stability"], stability_rows, GRAPH_STABILITY_COLUMNS)
    _write_csv_chunk(paths["shell_separability"], shell_rows, GRAPH_SHELL_COLUMNS)
    return json_sanitize({
        "schema": "vitriflow.graph_analysis_stream_chunk.v1",
        "status": "ok",
        "box_id": int(box_id),
        "paths": {k: str(v) for k, v in paths.items()},
        "counts": {
            "graph_rules": int(len(graph_rules)),
            "adaptive_graph_rules": int(len(adaptive_rules)),
            "adaptive_graph_rule_derivations": int(len(derivation_records)),
            "graph_metric_rows": int(len(graph_rows)),
            "metric_result_rows": int(len(metric_result_rows)),
            "coordination_stability_rows": int(len(stability_rows)),
            "shell_separability_rows": int(len(shell_rows)),
        },
    })


def strip_graph_analysis_payload(entry: Mapping[str, Any], chunk_summary: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Remove heavy per-rule graph rows from a box entry after streaming."""
    out = dict(entry)
    ga = out.get("graph_analysis", {}) if isinstance(out.get("graph_analysis", {}), Mapping) else {}
    manifest = ga.get("structure_manifest", out.get("structure_manifest", {})) if isinstance(ga, Mapping) else out.get("structure_manifest", {})
    out["graph_analysis"] = json_sanitize({
        "schema": "vitriflow.graph_analysis.streamed_summary.v1",
        "streamed_sidecars": True,
        "structure_manifest": manifest if isinstance(manifest, Mapping) else {},
        "counts": dict((chunk_summary or {}).get("counts", {})) if isinstance(chunk_summary, Mapping) else {},
        "chunk_paths": dict((chunk_summary or {}).get("paths", {})) if isinstance(chunk_summary, Mapping) else {},
        "note": "Heavy graph metric rows and adaptive graph rule records were streamed to sidecar chunks and are not embedded in the per-box JSON object.",
    })
    return json_sanitize(out)


def _chunk_paths(chunk_dir: Path, suffix: str) -> list[Path]:
    return sorted(Path(chunk_dir).glob(f"box_*_{suffix}"), key=lambda p: (_chunk_box_id_from_path(p), str(p)))


def _write_ensemble_graph_outputs_from_frames(
    outdir: Path,
    *,
    frames: Sequence[Any],
    box_ids: Sequence[int],
    metrics: Optional[StructureMetricsConfig],
    type_to_species: Optional[Sequence[str]],
    legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]],
) -> dict[str, Path]:
    outdir = Path(outdir)
    paths = {
        "ensemble_graph_rules": outdir / "ensemble_graph_rules.json",
        "ensemble_adaptive_graph_rules": outdir / "ensemble_adaptive_graph_rules.json",
        "ensemble_adaptive_graph_rule_derivations": outdir / "ensemble_adaptive_graph_rule_derivations.json",
        "ensemble_graph_metric_by_rule": outdir / "ensemble_graph_metric_by_rule.csv",
        "ensemble_coordination_stability": outdir / "ensemble_coordination_stability.csv",
        "ensemble_shell_separability": outdir / "ensemble_shell_separability.csv",
        "ensemble_graph_uncertainty_summary": outdir / "ensemble_graph_uncertainty_summary.csv",
    }
    if metrics is None or not frames:
        write_json_strict(paths["ensemble_graph_rules"], {"schema": "vitriflow.ensemble_graph_rules.v1", "graph_rules": []})
        write_json_strict(paths["ensemble_adaptive_graph_rules"], {"schema": "vitriflow.ensemble_adaptive_graph_rules.v1", "adaptive_graph_rule_records": [], "derivations_sidecar": str(paths["ensemble_adaptive_graph_rule_derivations"].name)})
        write_json_strict(paths["ensemble_adaptive_graph_rule_derivations"], {"schema": "vitriflow.ensemble_adaptive_graph_rule_derivations.v1", "adaptive_graph_rule_derivation_records": []})
        _write_csv(paths["ensemble_graph_metric_by_rule"], [], GRAPH_METRIC_COLUMNS)
        _write_csv(paths["ensemble_coordination_stability"], [], GRAPH_STABILITY_COLUMNS)
        _write_csv(paths["ensemble_shell_separability"], [], GRAPH_SHELL_COLUMNS)
        _write_csv(paths["ensemble_graph_uncertainty_summary"], [], GRAPH_UNCERTAINTY_COLUMNS)
        return paths
    raw_rules = list(getattr(metrics, "graph_rules", []) or [])
    rules, interval_rules = expand_graph_rules_for_frames(
        raw_rules,
        frames=list(frames),
        metrics=metrics,
        type_to_species=type_to_species,
        legacy_cutoffs=legacy_cutoffs,
        label="ensemble",
    )
    all_ensemble_rules = list(rules) + list(interval_rules)
    adaptive_records: list[dict[str, Any]] = []
    seen_adaptive: set[str] = set()
    for rr in all_ensemble_rules:
        params = dict(getattr(rr, "parameters", {}) or {})
        if bool(params.get("rdf_adaptive", False)):
            rec = _compact_rule_record(rr)
            key = _json_dumps(rec)
            if key not in seen_adaptive:
                seen_adaptive.add(key)
                adaptive_records.append(rec)
    adaptive_derivation_records = _adaptive_derivation_records_from_rules(all_ensemble_rules)
    seen_rules: set[str] = set()
    rule_records: list[dict[str, Any]] = []
    chunk_tmp = outdir / ".analysis_stream_chunks" / "ensemble"
    chunk_tmp.mkdir(parents=True, exist_ok=True)
    metric_chunks: list[Path] = []
    stability_chunks: list[Path] = []
    shell_chunks: list[Path] = []
    for fr, box_id in zip(frames, box_ids):
        rows, recs = compute_graph_metric_rows(fr, metrics, box_id=int(box_id), graph_rules=rules, type_to_species=type_to_species)
        for rec in recs:
            compact_rec = _compact_rule_record(rec)
            key = _json_dumps(compact_rec)
            if key not in seen_rules:
                seen_rules.add(key)
                rule_records.append(dict(compact_rec))
        stab_rows: list[dict[str, Any]] = []
        sh_rows: list[dict[str, Any]] = []
        for interval in list(interval_rules):
            rr, ss = robust_coordination_partition(fr, metrics, box_id=int(box_id), interval_rule=interval, type_to_species=type_to_species)
            stab_rows.extend(rr)
            sh_rows.extend(ss)
        prefix = f"box_{int(box_id):06d}"
        mp = chunk_tmp / f"{prefix}_ensemble_graph_metric_by_rule.csv"
        sp = chunk_tmp / f"{prefix}_ensemble_coordination_stability.csv"
        hp = chunk_tmp / f"{prefix}_ensemble_shell_separability.csv"
        _write_csv_chunk(mp, rows, GRAPH_METRIC_COLUMNS)
        _write_csv_chunk(sp, stab_rows, GRAPH_STABILITY_COLUMNS)
        _write_csv_chunk(hp, sh_rows, GRAPH_SHELL_COLUMNS)
        metric_chunks.append(mp)
        stability_chunks.append(sp)
        shell_chunks.append(hp)
    write_json_strict(paths["ensemble_graph_rules"], {"schema": "vitriflow.ensemble_graph_rules.v1", "graph_rules": rule_records})
    write_json_strict(paths["ensemble_adaptive_graph_rules"], {"schema": "vitriflow.ensemble_adaptive_graph_rules.v1", "adaptive_graph_rule_records": adaptive_records, "derivations_sidecar": str(paths["ensemble_adaptive_graph_rule_derivations"].name)})
    write_json_strict(paths["ensemble_adaptive_graph_rule_derivations"], {"schema": "vitriflow.ensemble_adaptive_graph_rule_derivations.v1", "adaptive_graph_rule_derivation_records": adaptive_derivation_records})
    _concat_csv_chunks(paths["ensemble_graph_metric_by_rule"], metric_chunks, GRAPH_METRIC_COLUMNS)
    _concat_csv_chunks(paths["ensemble_coordination_stability"], stability_chunks, GRAPH_STABILITY_COLUMNS)
    _concat_csv_chunks(paths["ensemble_shell_separability"], shell_chunks, GRAPH_SHELL_COLUMNS)
    ens_unc = graph_uncertainty_summary_rows_from_metric_csv(paths["ensemble_graph_metric_by_rule"])
    _write_csv(paths["ensemble_graph_uncertainty_summary"], ens_unc, GRAPH_UNCERTAINTY_COLUMNS)
    return paths


def finalize_streamed_graph_analysis_outputs(
    outdir: Path,
    *,
    chunk_dir: Path,
    boxes: Sequence[Mapping[str, Any]],
    rejected_boxes: Optional[Sequence[Mapping[str, Any]]] = None,
    metrics: Optional[StructureMetricsConfig] = None,
    type_to_species: Optional[Sequence[str]] = None,
    legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]] = None,
    ensemble_frames: Optional[Sequence[Any]] = None,
    ensemble_box_ids: Optional[Sequence[int]] = None,
    keep_chunks: bool = False,
) -> dict[str, str]:
    """Finalize sidecars from per-box streaming chunks.

    This avoids materialising full-ensemble graph metric rows in memory and is
    the default path for large analysis-only ensembles.
    """
    outdir = Path(outdir)
    chunk_dir = Path(chunk_dir)
    manifest_path = outdir / "structure_manifest.json"
    rules_path = outdir / "graph_rules.json"
    adaptive_path = outdir / "adaptive_graph_rules.json"
    representation_rules_path = outdir / "representation_rules.json"
    derivations_path = outdir / "adaptive_graph_rule_derivations.json"
    metric_path = outdir / "graph_metric_by_rule.csv"
    metric_results_path = outdir / "metric_results.csv"
    stability_path = outdir / "coordination_stability.csv"
    shell_path = outdir / "shell_separability.csv"
    uncertainty_path = outdir / "graph_uncertainty_summary.csv"
    representation_uncertainty_path = outdir / "representation_uncertainty_summary.csv"
    void_scaling_path = outdir / "void_scaling_summary.json"
    legacy_path = outdir / "legacy_single_cutoff_summary.json"
    graph_family_path = outdir / "graph_family_summary.json"

    manifest_records = _jsonl_records(_chunk_paths(chunk_dir, "structure_manifest.jsonl"))
    graph_rule_records = _jsonl_records(_chunk_paths(chunk_dir, "graph_rules.jsonl"))
    adaptive_records = _jsonl_records(_chunk_paths(chunk_dir, "adaptive_graph_rules.jsonl"))
    derivation_records = _jsonl_records(_chunk_paths(chunk_dir, "adaptive_graph_rule_derivations.jsonl"))
    representation_records = _jsonl_records(_chunk_paths(chunk_dir, "representation_rules.jsonl"))

    write_json_strict(manifest_path, {"schema": "vitriflow.structure_manifest.v2", "structures": manifest_records})
    write_json_strict(rules_path, {"schema": "vitriflow.graph_rules.v1", "graph_rules": [_compact_rule_record(r) for r in graph_rule_records]})
    write_json_strict(adaptive_path, {"schema": "vitriflow.adaptive_graph_rules.v1", "adaptive_graph_rule_records": [_compact_rule_record(r) for r in adaptive_records], "derivations_sidecar": str(derivations_path.name)})
    write_json_strict(derivations_path, {"schema": "vitriflow.adaptive_graph_rule_derivations.v1", "adaptive_graph_rule_derivation_records": derivation_records})

    _concat_csv_chunks(metric_path, _chunk_paths(chunk_dir, "graph_metric_by_rule.csv"), GRAPH_METRIC_COLUMNS)
    _concat_csv_chunks(metric_results_path, _chunk_paths(chunk_dir, "metric_results.csv"), GRAPH_METRIC_COLUMNS)
    _concat_csv_chunks(stability_path, _chunk_paths(chunk_dir, "coordination_stability.csv"), GRAPH_STABILITY_COLUMNS)
    _concat_csv_chunks(shell_path, _chunk_paths(chunk_dir, "shell_separability.csv"), GRAPH_SHELL_COLUMNS)

    uncertainty_rows = graph_uncertainty_summary_rows_from_metric_csv(metric_path)
    _write_csv(uncertainty_path, uncertainty_rows, GRAPH_UNCERTAINTY_COLUMNS)
    _write_csv(representation_uncertainty_path, uncertainty_rows, GRAPH_UNCERTAINTY_COLUMNS)

    void_scaling_payload = _void_scaling_summary_from_entries(list(boxes or []), metrics)
    write_json_strict(void_scaling_path, void_scaling_payload)
    void_reps = list(void_scaling_payload.get("representation_rules", []) or []) if isinstance(void_scaling_payload, Mapping) else []
    write_json_strict(
        representation_rules_path,
        {
            "schema": "vitriflow.representation_rules.v1",
            "representation_rules": representation_records + void_reps,
            "note": "RepresentationRule is the canonical descriptor-map schema. Streaming analysis wrote graph/coordinate rules per box and finalized this file without retaining all metric rows in memory.",
        },
    )

    ensemble_paths = _write_ensemble_graph_outputs_from_frames(
        outdir,
        frames=list(ensemble_frames or []),
        box_ids=list(ensemble_box_ids or []),
        metrics=metrics,
        type_to_species=type_to_species,
        legacy_cutoffs=legacy_cutoffs,
    )

    graph_family_payload = {
        "schema": "vitriflow.graph_family_summary.v1",
        "families": {
            "network_graph": {
                "role": "primary backbone topology",
                "used_for": ["coordination", "coordination defects", "angles", "rings", "components", "path lengths", "local graph topology"],
                "note": "Contains expected-shell/ring/angle network pairs only; same-species diagnostic pairs are not allowed to define backbone rings or topology.",
            },
            "candidate_contact_graph": {
                "role": "homopolar and close-contact candidate evidence",
                "used_for": ["homopolar bonds", "same-species candidate edges", "candidate-contact motifs", "candidate-contact edge-sharing perturbations"],
                "note": "Retains RDF-derived candidate pairs but is not used as the primary covalent topology graph unless explicitly configured.",
            },
            "soft_ambiguity_graph": {
                "role": "soft-neighbour ambiguity",
                "used_for": ["soft coordination", "local transition-shell ambiguity", "edge-weight summaries"],
                "note": "Not used for exact rings or hard topology unless explicitly projected by a separate hard rule.",
            },
            "auxiliary_graph": {
                "role": "specialized user-defined or non-bond topology",
                "used_for": ["voronoi adjacency", "persistent topology", "user-defined graph construction"],
                "note": "Auxiliary graphs are never merged into the backbone unless explicitly requested by configuration.",
            },
            "legacy_single_cutoff_graph": {
                "role": "backward compatibility",
                "used_for": ["legacy metrics/distributions/coordination-defect outputs"],
                "note": "Retained for existing hard-cutoff workflows; graph provenance and legacy marker remain explicit.",
            },
        },
        "per_structure_rules_present": bool(graph_rule_records),
        "ensemble_rules_present": bool(ensemble_paths.get("ensemble_graph_rules") and Path(ensemble_paths["ensemble_graph_rules"]).exists()),
        "streamed": True,
    }
    write_json_strict(graph_family_path, graph_family_payload)

    legacy_payload = {
        "schema": "vitriflow.legacy_single_rule_summary.v1",
        "legacy_single_cutoff_fields_retained": True,
        "adaptive_rdf_graph_rules_available": bool(adaptive_records),
        "legacy_marker": "legacy_single_rule_output",
        "legacy_single_cutoff_marker": "legacy_single_cutoff_output",
        "streamed_sidecars": True,
        "note": "Backward-compatible single-cutoff fields are retained when a legacy cutoff map is available, while graph_metric_by_rule.csv is the graph-provenanced descriptor table.",
    }
    write_json_strict(legacy_path, legacy_payload)

    out = {
        "structure_manifest": str(manifest_path.name),
        "graph_rules": str(rules_path.name),
        "adaptive_graph_rules": str(adaptive_path.name),
        "adaptive_graph_rule_derivations": str(derivations_path.name),
        "representation_rules": str(representation_rules_path.name),
        "metric_results": str(metric_results_path.name),
        "graph_metric_by_rule": str(metric_path.name),
        "coordination_stability": str(stability_path.name),
        "shell_separability": str(shell_path.name),
        "graph_uncertainty_summary": str(uncertainty_path.name),
        "representation_uncertainty_summary": str(representation_uncertainty_path.name),
        "void_scaling_summary": str(void_scaling_path.name),
        "legacy_single_cutoff_summary": str(legacy_path.name),
        "graph_family_summary": str(graph_family_path.name),
    }
    out.update({k: str(Path(v).name) for k, v in ensemble_paths.items()})
    if not bool(keep_chunks):
        try:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        except Exception:
            pass
    return out


def _bootstrap_se(values: Sequence[float], *, n_boot: int = 200) -> Optional[float]:
    vals = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    n = int(vals.size)
    if n < 2:
        return None
    if float(np.std(vals)) == 0.0:
        return 0.0
    rng = np.random.default_rng(246813579)
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = rng.choice(vals, size=n, replace=True)
        means[i] = float(np.mean(sample))
    se = float(np.std(means, ddof=1))
    return se if math.isfinite(se) else None


def _dominant_uncertainty(width: float, se: Optional[float]) -> tuple[str, Optional[float], str, str]:
    if se is None:
        return "not_applicable", None, "bootstrap_not_applicable", "single_structure_or_missing_bootstrap_variance"
    if not math.isfinite(float(se)):
        return "not_applicable", None, "bootstrap_not_applicable", "nonfinite_bootstrap_se"
    if float(se) == 0.0:
        if float(width) == 0.0:
            return "negligible", None, "zero_variance", "width_and_bootstrap_se_are_zero"
        return "representation_rule", None, "zero_variance", "bootstrap_se_is_zero_width_over_se_undefined"
    ratio = float(width) / float(se)
    if ratio < 0.5:
        dom = "sampling"
    elif ratio > 2.0:
        dom = "representation_rule"
    else:
        dom = "both"
    return dom, ratio, "ok", ""


def graph_uncertainty_summary_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_metric: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    row_meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in metric_rows:
        val = _finite_or_none(row.get("metric_value", None))
        if val is None:
            continue
        key = (
            str(row.get("graph_rule_scope", row.get("representation_rule_scope", "per_structure"))),
            str(row.get("graph_family", row.get("representation_family", "unclassified_graph"))),
            str(row.get("metric_family", "")),
            str(row.get("metric_name", "")),
        )
        rule = str(row.get("graph_rule_name", row.get("representation_rule_name", "")))
        by_metric[key][rule].append(float(val))
        row_meta.setdefault(key, dict(row))
    out: list[dict[str, Any]] = []
    for (scope, graph_family, family, name), by_rule in sorted(by_metric.items()):
        means: dict[str, float] = {}
        ses: list[float] = []
        n_values = 0
        for rule, vals in sorted(by_rule.items()):
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            means[rule] = float(np.mean(arr))
            n_values += int(arr.size)
            se = _bootstrap_se(arr.tolist())
            if se is not None and math.isfinite(float(se)):
                ses.append(float(se))
        if not means:
            continue
        sorted_means = sorted(means.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
        mean_vals = [float(v) for _k, v in sorted_means]
        width = float(max(mean_vals) - min(mean_vals)) if len(mean_vals) > 1 else 0.0
        se_box: Optional[float]
        if ses:
            se_box = float(max(ses))
        else:
            se_box = None
        dominant, width_over_se, uncertainty_status, uncertainty_reason = _dominant_uncertainty(width, se_box)
        status = "ok" if int(len(means)) > 1 else "not_applicable"
        reason = "" if status == "ok" else "single_representation_rule"
        meta = row_meta.get((scope, graph_family, family, name), {})
        out.append(
            json_sanitize(
                {
                    "graph_rule_scope": scope,
                    "graph_family": graph_family,
                    "representation_family": graph_family,
                    "metric_family": family,
                    "metric_name": name,
                    "metric_units": str(meta.get("metric_units", metric_units_for_name(name))),
                    "n_graph_rules": int(len(means)),
                    "n_values": int(n_values),
                    "lambda_min_rule": str(sorted_means[0][0]),
                    "lambda_max_rule": str(sorted_means[-1][0]),
                    "mean_min": float(min(mean_vals)),
                    "mean_max": float(max(mean_vals)),
                    "width": float(width),
                    "bootstrap_se": se_box,
                    "width_over_se": width_over_se,
                    "dominant_uncertainty": dominant,
                    "uncertainty_status": uncertainty_status,
                    "uncertainty_status_reason": uncertainty_reason,
                    "status": status,
                    "reason": reason,
                    "rule_means": dict(means),
                }
            )
        )
    return out


def collect_graph_rows_from_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    rule_records: list[dict[str, Any]] = []
    adaptive_records: list[dict[str, Any]] = []
    adaptive_derivation_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    seen_rules: set[str] = set()
    for ent in entries:
        ga = ent.get("graph_analysis", {}) if isinstance(ent, Mapping) else {}
        if not isinstance(ga, Mapping):
            continue
        m = ga.get("structure_manifest", None)
        if isinstance(m, Mapping):
            manifests.append(dict(m))
        for rr in list(ga.get("graph_rules", []) or []):
            if not isinstance(rr, Mapping):
                continue
            rec = _compact_rule_record(rr)
            key = _json_dumps(rec)
            if key in seen_rules:
                continue
            seen_rules.add(key)
            rule_records.append(dict(rec))
        for rr in list(ga.get("adaptive_graph_rule_records", []) or []):
            if isinstance(rr, Mapping):
                adaptive_records.append(_compact_rule_record(rr))
        for rr in list(ga.get("adaptive_graph_rule_derivation_records", []) or []):
            if isinstance(rr, Mapping):
                adaptive_derivation_records.append(json_sanitize(dict(rr)))
        if not ga.get("adaptive_graph_rule_derivation_records"):
            adaptive_derivation_records.extend(_adaptive_derivation_records_from_rules(list(ga.get("graph_rules", []) or []) + list(ga.get("adaptive_graph_rule_records", []) or [])))
        for rr in list(ga.get("graph_metric_rows", []) or []):
            if isinstance(rr, Mapping):
                metric_rows.append(_compact_metric_row(dict(rr)))
        for rr in list(ga.get("coordination_stability_rows", []) or []):
            if isinstance(rr, Mapping):
                stability_rows.append(_compact_metric_row(dict(rr)))
        for rr in list(ga.get("shell_separability_rows", []) or []):
            if isinstance(rr, Mapping):
                shell_rows.append(_compact_metric_row(dict(rr)))
    return {
        "structure_manifest": manifests,
        "graph_rules": rule_records,
        "adaptive_graph_rule_records": adaptive_records,
        "adaptive_graph_rule_derivation_records": adaptive_derivation_records,
        "graph_metric_by_rule": metric_rows,
        "coordination_stability": stability_rows,
        "shell_separability": shell_rows,
        "graph_uncertainty_summary": graph_uncertainty_summary_rows(metric_rows),
    }




def _frame_from_entry_structure(entry: Mapping[str, Any]) -> Optional[Any]:
    st = entry.get("structure", None) if isinstance(entry, Mapping) else None
    if not isinstance(st, Mapping):
        return None
    try:
        from .dump import DumpFrame

        lattice = dict(st.get("lattice", {}) or {})
        cell = np.asarray(lattice.get("cell", []), dtype=float).reshape(3, 3)
        origin = np.asarray(lattice.get("origin", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        positions = np.asarray(st.get("positions", []), dtype=float).reshape((-1, 3))
        n = int(positions.shape[0])
        ids = np.asarray(st.get("ids", list(range(1, n + 1))), dtype=int).reshape(-1)
        types = np.asarray(st.get("types", []), dtype=int).reshape(-1)
        if ids.size != n or types.size != n:
            return None
        return DumpFrame(
            timestep=int(st.get("timestep", 0) or 0),
            ids=ids,
            types=types,
            positions=positions,
            cell=cell,
            origin=origin,
        )
    except Exception:
        return None


def collect_ensemble_graph_rows_from_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    metrics: Optional[StructureMetricsConfig],
    type_to_species: Optional[Sequence[str]],
    legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Compute graph-rule outputs using one ensemble-derived rule set."""

    if metrics is None:
        return {
            "graph_rules": [],
            "adaptive_graph_rule_records": [],
            "adaptive_graph_rule_derivation_records": [],
            "graph_metric_by_rule": [],
            "coordination_stability": [],
            "shell_separability": [],
            "graph_uncertainty_summary": [],
        }
    frames: list[Any] = []
    box_ids: list[int] = []
    for ent in entries:
        if bool((ent or {}).get("reject", False)):
            continue
        fr = _frame_from_entry_structure(ent)
        if fr is None:
            continue
        frames.append(fr)
        box_ids.append(int((ent or {}).get("box", len(box_ids) + 1) or len(box_ids) + 1))
    if not frames:
        return {
            "graph_rules": [],
            "adaptive_graph_rule_records": [],
            "adaptive_graph_rule_derivation_records": [],
            "graph_metric_by_rule": [],
            "coordination_stability": [],
            "shell_separability": [],
            "graph_uncertainty_summary": [],
        }
    raw_rules = list(getattr(metrics, "graph_rules", []) or [])
    rules, interval_rules = expand_graph_rules_for_frames(
        raw_rules,
        frames=frames,
        metrics=metrics,
        type_to_species=type_to_species,
        legacy_cutoffs=legacy_cutoffs,
        label="ensemble",
    )
    metric_rows: list[dict[str, Any]] = []
    rule_records: list[dict[str, Any]] = []
    seen_rules: set[str] = set()
    for fr, box_id in zip(frames, box_ids):
        rows, recs = compute_graph_metric_rows(fr, metrics, box_id=int(box_id), graph_rules=rules, type_to_species=type_to_species)
        metric_rows.extend(rows)
        for rec in recs:
            compact_rec = _compact_rule_record(rec)
            key = _json_dumps(compact_rec)
            if key not in seen_rules:
                seen_rules.add(key)
                rule_records.append(dict(compact_rec))
    all_rules = list(rules) + list(interval_rules)
    adaptive_records: list[dict[str, Any]] = []
    seen_adaptive: set[str] = set()
    for rr in all_rules:
        params = dict(getattr(rr, "parameters", {}) or {})
        if bool(params.get("rdf_adaptive", False)):
            rec = _compact_rule_record(rr)
            key = _json_dumps(rec)
            if key not in seen_adaptive:
                seen_adaptive.add(key)
                adaptive_records.append(rec)
    adaptive_derivation_records = _adaptive_derivation_records_from_rules(all_rules)
    stability_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    for fr, box_id in zip(frames, box_ids):
        for interval in interval_rules:
            rr, ss = robust_coordination_partition(fr, metrics, box_id=int(box_id), interval_rule=interval, type_to_species=type_to_species)
            stability_rows.extend(rr)
            shell_rows.extend(ss)
    return {
        "graph_rules": rule_records,
        "adaptive_graph_rule_records": adaptive_records,
        # Keep the (potentially large) derivation payload separate from the
        # compact rule records, but do not drop it here: this collector is the
        # authoritative bridge between ensemble rule construction and the
        # provenance sidecar written below.  Omitting the key made a genuinely
        # adaptive analysis indistinguishable from one with no derivation.
        "adaptive_graph_rule_derivation_records": adaptive_derivation_records,
        "graph_metric_by_rule": metric_rows,
        "coordination_stability": stability_rows,
        "shell_separability": shell_rows,
        "graph_uncertainty_summary": graph_uncertainty_summary_rows(metric_rows),
    }

def _representation_records_from_graph_rules(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records or []:
        if not isinstance(rec, Mapping):
            continue
        try:
            rule = GraphRule.from_any(rec)
        except Exception:
            continue
        params = dict(getattr(rule, "parameters", {}) or {})
        sh = str(params.get("structure_hash", rec.get("structure_hash", "")) or "")
        fam = str(params.get("graph_family", "unclassified_graph"))
        rr = graph_rule_to_representation_fields(rule, structure_hash=sh, graph_family=fam)
        rr["graph_rule_name"] = str(rule.name)
        rr["graph_rule_kind"] = str(rule.kind)
        rr["descriptor_target"] = "graph-conditioned descriptor distribution"
        key = _json_dumps(rr)
        if key not in seen:
            seen.add(key)
            out.append(json_sanitize(rr))
    return out


def _representation_records_from_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate RepresentationRule records already attached to metric rows."""

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("representation_rule_name", "") or "")
        kind = str(row.get("representation_rule_kind", "") or "")
        if not name or not kind:
            continue
        rr = {
            "representation_rule_name": name,
            "representation_rule_kind": kind,
            "representation_rule_parameters": row.get("representation_rule_parameters", {}),
            "representation_rule_provenance": row.get("representation_rule_provenance", {}),
            "representation_rule_version": row.get("representation_rule_version", "v1"),
            "structure_hash": row.get("structure_hash", ""),
            "material_id": row.get("material_id", "unknown"),
            "source_space": row.get("source_space", "structure"),
            "representation_map": row.get("representation_map", "unknown"),
            "target_distribution": row.get("target_distribution", "descriptor distribution"),
        }
        if row.get("graph_rule_name"):
            rr["graph_rule_name"] = row.get("graph_rule_name")
            rr["graph_rule_kind"] = row.get("graph_rule_kind")
            rr["graph_family"] = row.get("graph_family")
        key = _json_dumps(rr)
        if key not in seen:
            seen.add(key)
            out.append(json_sanitize(rr))
    return out


def _coordinate_metric_rows_from_entries(entries: Sequence[Mapping[str, Any]], metrics: Optional[StructureMetricsConfig]) -> list[dict[str, Any]]:
    """Coordinate/void descriptor rows for the generic MetricResult table.

    Graph-derived convenience values in boxes[].metrics are intentionally not
    copied here; their qualified rows come from graph_metric_by_rule.  This keeps
    the main metric table from silently mixing descriptor maps.
    """

    graph_prefixes = (
        "coord_",
        "angle_",
        "ring_",
        "bondlen_",
        "bond_incidence_",
        "homopolar_",
        "defect_",
        "component_",
        "graph_",
        "path_",
        "local_",
        "motif_",
        "edge_sharing_",
        "soft_",
        "candidate_contact_",
    )
    rows: list[dict[str, Any]] = []
    for ent in entries or []:
        if not isinstance(ent, Mapping) or bool(ent.get("reject", False)):
            continue
        manifest = ent.get("structure_manifest", {}) if isinstance(ent.get("structure_manifest", {}), Mapping) else {}
        box_id = int(ent.get("box", len(rows) + 1) or len(rows) + 1)
        sh = str(manifest.get("structure_hash", ""))
        material_id = str(manifest.get("material_id", "unknown"))
        if finite_or_none(ent.get("density", None)) is not None:
            rows.append(
                metric_result_row(
                    box_id=box_id,
                    structure_hash=sh,
                    material_id=material_id,
                    metric_family="density",
                    metric_name="density",
                    metric_value=ent.get("density"),
                    metric_units="g/cm^3",
                    representation_fields=direct_coordinate_rule_fields(
                        structure_hash=sh,
                        name="density_from_structure_volume_and_composition",
                        parameters={"normalization": "mass_per_cell_volume", "units": "g/cm^3"},
                        provenance="structure_manifest.volume + species masses",
                    ),
                    normalization="per_box",
                )
            )
        vals = ent.get("metrics", {}) if isinstance(ent.get("metrics", {}), Mapping) else {}
        for name, value in sorted(vals.items()):
            nm = str(name)
            if nm.startswith(graph_prefixes):
                continue
            if nm.startswith("void_"):
                mode = "density_scaled" if "scaled" in nm else "raw_absolute"
                rep = void_rule_fields(
                    structure_hash=sh,
                    name="void_field_density_scaled" if mode == "density_scaled" else "void_field_raw_absolute",
                    kind="density_normalized_void_field" if mode == "density_scaled" else "void_field",
                    parameters={"void_metric_mode": mode, "source_metric": nm},
                    provenance="metrics.voids",
                )
                family = "void"
            elif nm.startswith("gr_") or nm.startswith("sq_"):
                rep = direct_coordinate_rule_fields(
                    structure_hash=sh,
                    name="coordinate_pair_distribution" if nm.startswith("gr_") else "coordinate_structure_factor",
                    parameters={"source_metric": nm},
                    provenance="metrics.gr/sq",
                )
                family = "coordinate_distribution"
            else:
                # Avoid copying unknown scalar convenience values into the generic
                # descriptor table without a declared representation rule.
                continue
            rows.append(
                metric_result_row(
                    box_id=box_id,
                    structure_hash=sh,
                    material_id=material_id,
                    metric_family=family,
                    metric_name=nm,
                    metric_value=value,
                    representation_fields=rep,
                    normalization="per_box",
                )
            )
        void_dist = (ent.get("distributions", {}) or {}).get("void", {}) if isinstance(ent.get("distributions", {}), Mapping) else {}
        for nm, payload in sorted(dict(void_dist or {}).items()):
            if not isinstance(payload, Mapping):
                continue
            mode = str(payload.get("void_metric_mode", "density_scaled" if "scaled" in str(nm) else "raw_absolute"))
            rep = void_rule_fields(
                structure_hash=sh,
                name=str(payload.get("representation_rule_name", "void_field_density_scaled" if mode == "density_scaled" else "void_field_raw_absolute")),
                kind="density_normalized_void_field" if mode == "density_scaled" else "void_field",
                parameters={
                    "void_metric_mode": mode,
                    "grid_points": {"length": len(list(payload.get("x", []) or [])), "units": payload.get("units")},
                    "normalization": payload.get("normalization", "per_box_sample_cdf"),
                    "length_scale": payload.get("length_scale"),
                },
                provenance="metrics.voids",
            )
            rows.append(
                metric_result_row(
                    box_id=box_id,
                    structure_hash=sh,
                    material_id=material_id,
                    metric_family="void_distribution",
                    metric_name=str(nm),
                    metric_value=None,
                    metric_units=str(payload.get("units", "angstrom")),
                    representation_fields=rep,
                    status="ok",
                    status_reason="array_payload_recorded_in_boxes_distributions_and_void_scaling_summary",
                    n_samples=len(list(payload.get("x", []) or [])),
                    normalization=str(payload.get("normalization", "per_box_sample_cdf")),
                )
            )
    return [json_sanitize(r) for r in rows]


def _void_scaling_summary_from_entries(entries: Sequence[Mapping[str, Any]], metrics: Optional[StructureMetricsConfig]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    void_cfg = getattr(metrics, "voids", None) if metrics is not None else None
    params = {
        "sampler": str(getattr(void_cfg, "sampler", "unknown")) if void_cfg is not None else "unknown",
        "n_samples": int(getattr(void_cfg, "n_samples", 0) or 0) if void_cfg is not None else None,
        "k_nearest": int(getattr(void_cfg, "k_nearest", 0) or 0) if void_cfg is not None else None,
        "default_radius": float(getattr(void_cfg, "default_radius", 0.0) or 0.0) if void_cfg is not None else None,
        "radii": dict(getattr(void_cfg, "radii", {}) or {}) if void_cfg is not None else {},
        "r_max": float(getattr(void_cfg, "r_max", 0.0) or 0.0) if void_cfg is not None else None,
        "probe_radii": list(getattr(void_cfg, "probe_radii", []) or []) if void_cfg is not None else [],
        "periodic_boundary_treatment": "minimum_image_periodic",
        "clearance_definition": "distance_to_nearest_atom_surface_using_configured_radii",
        "normalization": "per_box_distribution",
        "units": {"clearance": "angstrom", "density_scaled_clearance": "reduced_length"},
    }
    for ent in entries or []:
        if not isinstance(ent, Mapping) or bool(ent.get("reject", False)):
            continue
        st = ent.get("structure", {}) if isinstance(ent.get("structure", {}), Mapping) else {}
        manifest = ent.get("structure_manifest", {}) if isinstance(ent.get("structure_manifest", {}), Mapping) else {}
        n_atoms = None
        try:
            if isinstance(st, Mapping) and st.get("n_atoms", None) is not None:
                n_atoms = int(st.get("n_atoms"))
            else:
                n_atoms = len(list(st.get("positions", []) or []))
        except Exception:
            n_atoms = None
        volume = finite_or_none(manifest.get("volume", None))
        length_scale = None
        if volume is not None and n_atoms is not None and int(n_atoms) > 0:
            length_scale = float((float(volume) / float(n_atoms)) ** (1.0 / 3.0))
        void_dist = (ent.get("distributions", {}) or {}).get("void", {}) if isinstance(ent.get("distributions", {}), Mapping) else {}
        rows.append(
            json_sanitize(
                {
                    "box_id": int(ent.get("box", len(rows) + 1) or len(rows) + 1),
                    "structure_hash": manifest.get("structure_hash"),
                    "material_id": manifest.get("material_id", "unknown"),
                    "volume": volume,
                    "n_atoms": n_atoms,
                    "length_scale": length_scale,
                    "length_scale_definition": "(V/N)^(1/3)",
                    "raw_void_metrics_present": bool(void_dist),
                    "density_scaled_metrics_available": length_scale is not None,
                    "raw_absolute_status": "ok" if bool(void_dist) else "unavailable",
                    "density_scaled_status": "ok" if length_scale is not None else "not_applicable",
                    "density_scaled_reason": "" if length_scale is not None else "missing_volume_or_atom_count",
                }
            )
        )
    return {
        "schema": "vitriflow.void_scaling_summary.v1",
        "representation_rules": [
            {
                "representation_rule_name": "void_field_raw_absolute",
                "representation_rule_kind": "void_field",
                "representation_rule_parameters": params,
                "representation_rule_provenance": "metrics.voids",
                "representation_rule_version": "v1",
                "representation_map": "void_field",
                "void_metric_mode": "raw_absolute",
            },
            {
                "representation_rule_name": "void_field_density_scaled",
                "representation_rule_kind": "density_normalized_void_field",
                "representation_rule_parameters": {**params, "length_scale": "(V/N)^(1/3)", "clearance_scaled": "clearance / (V/N)^(1/3)"},
                "representation_rule_provenance": "metrics.voids + structure_manifest.volume/n_atoms",
                "representation_rule_version": "v1",
                "representation_map": "void_field",
                "void_metric_mode": "density_scaled",
            },
        ],
        "rows": rows,
        "note": "Raw void descriptors are not overwritten by density-scaled descriptors; scale-aware metrics are separate representation rules.",
    }


def write_graph_analysis_outputs(
    outdir: Path,
    *,
    boxes: Sequence[Mapping[str, Any]],
    rejected_boxes: Optional[Sequence[Mapping[str, Any]]] = None,
    metrics: Optional[StructureMetricsConfig] = None,
    type_to_species: Optional[Sequence[str]] = None,
    legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]] = None,
) -> dict[str, str]:
    """Persist graph-rule outputs assembled from production/analysis entries."""

    outdir = Path(outdir)
    entries = list(boxes or []) + list(rejected_boxes or [])
    collected = collect_graph_rows_from_entries(entries)
    ensemble_collected = collect_ensemble_graph_rows_from_entries(
        list(boxes or []),
        metrics=metrics,
        type_to_species=type_to_species,
        legacy_cutoffs=legacy_cutoffs,
    ) if metrics is not None else {
        "graph_rules": [],
        "adaptive_graph_rule_records": [],
        "adaptive_graph_rule_derivation_records": [],
        "graph_metric_by_rule": [],
        "coordination_stability": [],
        "shell_separability": [],
        "graph_uncertainty_summary": [],
    }

    manifest_path = outdir / "structure_manifest.json"
    rules_path = outdir / "graph_rules.json"
    adaptive_path = outdir / "adaptive_graph_rules.json"
    adaptive_derivations_path = outdir / "adaptive_graph_rule_derivations.json"
    metric_path = outdir / "graph_metric_by_rule.csv"
    stability_path = outdir / "coordination_stability.csv"
    shell_path = outdir / "shell_separability.csv"
    uncertainty_path = outdir / "graph_uncertainty_summary.csv"
    legacy_path = outdir / "legacy_single_cutoff_summary.json"
    ensemble_rules_path = outdir / "ensemble_graph_rules.json"
    ensemble_adaptive_path = outdir / "ensemble_adaptive_graph_rules.json"
    ensemble_adaptive_derivations_path = outdir / "ensemble_adaptive_graph_rule_derivations.json"
    ensemble_metric_path = outdir / "ensemble_graph_metric_by_rule.csv"
    ensemble_stability_path = outdir / "ensemble_coordination_stability.csv"
    ensemble_shell_path = outdir / "ensemble_shell_separability.csv"
    ensemble_uncertainty_path = outdir / "ensemble_graph_uncertainty_summary.csv"
    graph_family_path = outdir / "graph_family_summary.json"
    representation_rules_path = outdir / "representation_rules.json"
    metric_results_path = outdir / "metric_results.csv"
    representation_uncertainty_path = outdir / "representation_uncertainty_summary.csv"
    void_scaling_path = outdir / "void_scaling_summary.json"

    manifest_payload = {
        "schema": "vitriflow.structure_manifest.v2",
        "structures": collected["structure_manifest"],
    }
    rules_payload = {
        "schema": "vitriflow.graph_rules.v1",
        "graph_rules": [_compact_rule_record(r) for r in collected["graph_rules"]],
    }
    adaptive_payload = {
        "schema": "vitriflow.adaptive_graph_rules.v1",
        "adaptive_graph_rule_records": [_compact_rule_record(r) for r in collected.get("adaptive_graph_rule_records", [])],
        "derivations_sidecar": str(adaptive_derivations_path.name),
    }
    adaptive_derivations_payload = {
        "schema": "vitriflow.adaptive_graph_rule_derivations.v1",
        "adaptive_graph_rule_derivation_records": list(collected.get("adaptive_graph_rule_derivation_records", [])),
    }
    coordinate_metric_rows = _coordinate_metric_rows_from_entries(list(boxes or []), metrics)
    metric_result_rows = list(collected.get("graph_metric_by_rule", [])) + coordinate_metric_rows
    ensemble_metric_result_rows = list(ensemble_collected.get("graph_metric_by_rule", []))
    graph_representation_records = _representation_records_from_graph_rules(
        list(collected.get("graph_rules", [])) + list(ensemble_collected.get("graph_rules", []))
    )
    row_representation_records = _representation_records_from_metric_rows(
        metric_result_rows + ensemble_metric_result_rows
    )
    void_scaling_payload = _void_scaling_summary_from_entries(list(boxes or []), metrics)
    representation_payload = {
        "schema": "vitriflow.representation_rules.v1",
        "representation_rules": graph_representation_records + row_representation_records + list(void_scaling_payload.get("representation_rules", [])),
        "note": "RepresentationRule is the canonical descriptor-map schema. GraphRule is one specialization used for graph-derived metrics; coordinate and void descriptors have their own representation rules.",
    }
    write_json_strict(manifest_path, manifest_payload)
    write_json_strict(rules_path, rules_payload)
    write_json_strict(adaptive_path, adaptive_payload)
    write_json_strict(adaptive_derivations_path, adaptive_derivations_payload)
    write_json_strict(representation_rules_path, representation_payload)
    write_json_strict(void_scaling_path, void_scaling_payload)

    metric_cols = [
        "box_id",
        "structure_hash",
        "material_id",
        "graph_rule_scope",
        "graph_family",
        "graph_rule_name",
        "graph_rule_kind",
        "graph_rule_parameters",
        "graph_rule_provenance",
        "representation_rule_name",
        "representation_rule_kind",
        "representation_rule_parameters",
        "representation_rule_provenance",
        "representation_rule_version",
        "source_space",
        "representation_map",
        "descriptor_map",
        "target_distribution",
        "metric_family",
        "metric_name",
        "metric_value",
        "metric_units",
        "metric_status",
        "metric_status_reason",
        "numerical_status",
        "numerical_status_reason",
        "uncertainty_status",
        "uncertainty_status_reason",
        "n_samples",
        "normalization",
    ]
    _write_csv(metric_path, collected["graph_metric_by_rule"], metric_cols)
    _write_csv(metric_results_path, metric_result_rows, metric_cols)

    stability_cols = [
        "box_id",
        "structure_hash",
        "material_id",
        "graph_rule_scope",
        "graph_family",
        "graph_rule_name",
        "graph_rule_kind",
        "graph_rule_parameters",
        "graph_rule_provenance",
        "representation_rule_name",
        "representation_rule_kind",
        "representation_rule_parameters",
        "representation_rule_provenance",
        "representation_rule_version",
        "source_space",
        "representation_map",
        "coordination_metric",
        "expected_coordination",
        "r_min",
        "r_max",
        "n_central",
        "robust_ideal_fraction",
        "robust_undercoordinated_fraction",
        "robust_overcoordinated_fraction",
        "ambiguous_fraction",
        "robust_ideal_count",
        "robust_undercoordinated_count",
        "robust_overcoordinated_count",
        "ambiguous_count",
        "status",
        "reason",
        "metric_status",
        "metric_status_reason",
        "numerical_status",
        "uncertainty_status",
    ]
    _write_csv(stability_path, collected["coordination_stability"], stability_cols)

    shell_cols = [
        "box_id",
        "structure_hash",
        "material_id",
        "graph_rule_scope",
        "graph_family",
        "graph_rule_name",
        "graph_rule_kind",
        "graph_rule_parameters",
        "graph_rule_provenance",
        "representation_rule_name",
        "representation_rule_kind",
        "representation_rule_parameters",
        "representation_rule_provenance",
        "representation_rule_version",
        "source_space",
        "representation_map",
        "coordination_metric",
        "expected_coordination",
        "r_min",
        "r_max",
        "n_central",
        "q05_d_z",
        "q50_d_z",
        "q95_d_z",
        "q05_d_z_plus_1",
        "q50_d_z_plus_1",
        "q95_d_z_plus_1",
        "d_z_mean",
        "d_z_plus_1_mean",
        "shell_gap_mean",
        "shell_gap_min",
        "shell_gap_p05",
        "shell_gap_median",
        "shell_overlap_score",
        "n_missing_z_plus_1",
        "status",
        "reason",
        "metric_status",
        "metric_status_reason",
        "numerical_status",
        "uncertainty_status",
    ]
    _write_csv(shell_path, collected["shell_separability"], shell_cols)

    uncertainty_cols = [
        "graph_rule_scope",
        "graph_family",
        "representation_family",
        "metric_family",
        "metric_name",
        "metric_units",
        "n_graph_rules",
        "n_values",
        "lambda_min_rule",
        "lambda_max_rule",
        "mean_min",
        "mean_max",
        "width",
        "bootstrap_se",
        "width_over_se",
        "dominant_uncertainty",
        "uncertainty_status",
        "uncertainty_status_reason",
        "status",
        "reason",
        "rule_means",
    ]
    _write_csv(uncertainty_path, collected["graph_uncertainty_summary"], uncertainty_cols)
    _write_csv(representation_uncertainty_path, collected["graph_uncertainty_summary"], uncertainty_cols)

    ensemble_rules_payload = {
        "schema": "vitriflow.ensemble_graph_rules.v1",
        "graph_rules": [_compact_rule_record(r) for r in ensemble_collected.get("graph_rules", [])],
    }
    ensemble_adaptive_payload = {
        "schema": "vitriflow.ensemble_adaptive_graph_rules.v1",
        "adaptive_graph_rule_records": [_compact_rule_record(r) for r in ensemble_collected.get("adaptive_graph_rule_records", [])],
        "derivations_sidecar": str(ensemble_adaptive_derivations_path.name),
    }
    ensemble_adaptive_derivations_payload = {
        "schema": "vitriflow.ensemble_adaptive_graph_rule_derivations.v1",
        "adaptive_graph_rule_derivation_records": list(ensemble_collected.get("adaptive_graph_rule_derivation_records", [])),
    }
    write_json_strict(ensemble_rules_path, ensemble_rules_payload)
    write_json_strict(ensemble_adaptive_path, ensemble_adaptive_payload)
    write_json_strict(ensemble_adaptive_derivations_path, ensemble_adaptive_derivations_payload)
    _write_csv(ensemble_metric_path, ensemble_collected.get("graph_metric_by_rule", []), metric_cols)
    _write_csv(ensemble_stability_path, ensemble_collected.get("coordination_stability", []), stability_cols)
    _write_csv(ensemble_shell_path, ensemble_collected.get("shell_separability", []), shell_cols)
    _write_csv(ensemble_uncertainty_path, ensemble_collected.get("graph_uncertainty_summary", []), uncertainty_cols)

    graph_family_payload = {
        "schema": "vitriflow.graph_family_summary.v1",
        "families": {
            "network_graph": {
                "role": "primary backbone topology",
                "used_for": ["coordination", "coordination defects", "angles", "rings", "components", "path lengths", "local graph topology"],
                "note": "Contains expected-shell/ring/angle network pairs only; same-species diagnostic pairs are not allowed to define backbone rings or topology.",
            },
            "candidate_contact_graph": {
                "role": "homopolar and close-contact candidate evidence",
                "used_for": ["homopolar bonds", "same-species candidate edges", "candidate-contact motifs", "candidate-contact edge-sharing perturbations"],
                "note": "Retains all RDF-derived candidate pairs but is not used as the primary covalent topology graph.",
            },
            "soft_ambiguity_graph": {
                "role": "soft-neighbour ambiguity",
                "used_for": ["soft coordination", "local transition-shell ambiguity", "edge-weight summaries"],
                "note": "Not used for exact rings or hard topology unless explicitly projected by a separate hard rule.",
            },
            "auxiliary_graph": {
                "role": "specialized user-defined or non-bond topology",
                "used_for": ["voronoi adjacency", "persistent topology", "user-defined graph construction"],
                "note": "Auxiliary graphs are never merged into the backbone unless explicitly requested by configuration.",
            },
            "legacy_single_cutoff_graph": {
                "role": "backward compatibility",
                "used_for": ["legacy metrics/distributions/coordination-defect outputs"],
                "note": "Retained for existing hard-cutoff workflows; graph provenance and legacy marker remain explicit.",
            },
        },
        "per_structure_rules_present": bool(collected.get("graph_rules", [])),
        "ensemble_rules_present": bool(ensemble_collected.get("graph_rules", [])),
    }
    write_json_strict(graph_family_path, graph_family_payload)

    legacy_payload = {
        "schema": "vitriflow.legacy_single_rule_summary.v1",
        "legacy_single_cutoff_fields_retained": True,
        "adaptive_rdf_graph_rules_available": bool(collected.get("adaptive_graph_rule_records", [])),
        "legacy_fields": [
            "boxes[].metrics",
            "boxes[].distributions.bondlen",
            "boxes[].distributions.angle",
            "boxes[].distributions.coord",
            "boxes[].coordination_defects",
            "boxes[].has_coordination_defects",
            "production/analysis exclude_coordination_defects decisions",
        ],
        "legacy_marker": "legacy_single_rule_output",
        "legacy_single_cutoff_marker": "legacy_single_cutoff_output",
        "note": "Backward-compatible single-cutoff fields are retained when a legacy cutoff map is available, but graph_metric_by_rule.csv is the graph-provenanced descriptor table. For RDF-adaptive analysis, concrete cutoffs are resolved per structure and recorded in adaptive_graph_rules.json and graph_rule_provenance.",
    }
    write_json_strict(legacy_path, legacy_payload)

    return {
        "structure_manifest": str(manifest_path.name),
        "graph_rules": str(rules_path.name),
        "adaptive_graph_rules": str(adaptive_path.name),
        "adaptive_graph_rule_derivations": str(adaptive_derivations_path.name),
        "representation_rules": str(representation_rules_path.name),
        "metric_results": str(metric_results_path.name),
        "graph_metric_by_rule": str(metric_path.name),
        "coordination_stability": str(stability_path.name),
        "shell_separability": str(shell_path.name),
        "graph_uncertainty_summary": str(uncertainty_path.name),
        "representation_uncertainty_summary": str(representation_uncertainty_path.name),
        "void_scaling_summary": str(void_scaling_path.name),
        "legacy_single_cutoff_summary": str(legacy_path.name),
        "ensemble_graph_rules": str(ensemble_rules_path.name),
        "ensemble_adaptive_graph_rules": str(ensemble_adaptive_path.name),
        "ensemble_adaptive_graph_rule_derivations": str(ensemble_adaptive_derivations_path.name),
        "ensemble_graph_metric_by_rule": str(ensemble_metric_path.name),
        "ensemble_coordination_stability": str(ensemble_stability_path.name),
        "ensemble_shell_separability": str(ensemble_shell_path.name),
        "ensemble_graph_uncertainty_summary": str(ensemble_uncertainty_path.name),
        "graph_family_summary": str(graph_family_path.name),
    }
