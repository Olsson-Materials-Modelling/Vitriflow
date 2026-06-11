from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..analysis.datafile import read_datafile_frame
from ..config import (
    AngleMetricConfig,
    CoordinationMetricConfig,
    GrMetricConfig,
    PairMetricConfig,
    RingMetricsConfig,
    SqMetricConfig,
)

WarnFn = Callable[[str], None]


def _unique_type_selectors(*, structure_data: Optional[Path], type_to_species: Optional[Sequence[str]]) -> list[Any]:
    if type_to_species is not None and len(type_to_species) > 0:
        return [str(s) for s in type_to_species]
    if structure_data is not None and Path(structure_data).exists():
        try:
            fr = read_datafile_frame(Path(structure_data))
            return [int(t) for t in sorted({int(x) for x in fr.types.tolist()})]
        except Exception:
            pass
    return []


def _ordered_pairs(selectors: Sequence[Any]) -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    for a in selectors:
        for b in selectors:
            out.append((a, b))
    return out


def _unordered_pairs(selectors: Sequence[Any]) -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    for i, a in enumerate(selectors):
        for b in selectors[i:]:
            out.append((a, b))
    return out


def resolve_effective_metrics_config(
    metrics_cfg,
    *,
    structure_data: Optional[Path],
    type_to_species: Optional[Sequence[str]],
    warn_fn: Optional[WarnFn] = None,
    context: str = "workflow",
):
    """Effective metrics config."""

    def _warn(msg: str) -> None:
        if warn_fn is not None:
            warn_fn(msg)

    cfg = metrics_cfg.model_copy(deep=True)
    warnings: list[str] = []

    selectors = _unique_type_selectors(structure_data=Path(structure_data) if structure_data is not None else None, type_to_species=type_to_species)
    if len(selectors) == 0:
        selectors = [1]
        msg = f"{context}: could not infer species/type selectors; defaulting to selector [1] for automatic metric setup"
        warnings.append(msg)
        _warn(msg)

    if not bool(getattr(cfg, "enabled", False)):
        cfg.enabled = True
        msg = f"{context}: autotune.metrics.enabled was false or omitted; enabling automatic default structural metrics"
        warnings.append(msg)
        _warn(msg)

    if not bool(getattr(cfg.voids, "enabled", False)):
        cfg.voids.enabled = True
        msg = f"{context}: void analysis was not configured; enabling default void metrics"
        warnings.append(msg)
        _warn(msg)

    if not list(getattr(cfg, "pairs", []) or []):
        cfg.pairs = [PairMetricConfig(pair=(a, b)) for (a, b) in _unordered_pairs(selectors)]
        msg = f"{context}: pair metrics not configured; using all unordered pairs {[(a, b) for (a, b) in _unordered_pairs(selectors)]}"
        warnings.append(msg)
        _warn(msg)

    if not list(getattr(cfg, "coordinations", []) or []):
        cfg.coordinations = [CoordinationMetricConfig(central=a, neighbor=b) for (a, b) in _ordered_pairs(selectors)]
        msg = f"{context}: coordination metrics not configured; using all ordered central-neighbour pairs"
        warnings.append(msg)
        _warn(msg)

    if not list(getattr(cfg, "angles", []) or []):
        angles: list[AngleMetricConfig] = []
        for central in selectors:
            for a in selectors:
                for c in selectors:
                    angles.append(AngleMetricConfig(triplet=(a, central, c)))
        cfg.angles = angles
        msg = f"{context}: angle metrics not configured; using all triplets with each species/type as the angle centre"
        warnings.append(msg)
        _warn(msg)

    rings = getattr(cfg, "rings", None)
    rings_src = getattr(metrics_cfg, "rings", None)
    rings_enabled_explicitly_set = bool(getattr(rings_src, "model_fields_set", set()) and "enabled" in getattr(rings_src, "model_fields_set", set()))
    rings_explicitly_disabled = bool(rings_src is not None and rings_enabled_explicitly_set and not bool(getattr(rings_src, "enabled", False)))
    if rings is not None and not bool(getattr(rings, "enabled", False)) and not rings_explicitly_disabled:
        if len(selectors) == 1:
            cfg.rings = RingMetricsConfig(enabled=True, mode="bond_graph", nodes=[selectors[0]])
            msg = f"{context}: ring metrics not configured; enabling single-species bond-graph rings"
        else:
            cfg.rings = RingMetricsConfig(enabled=True, mode="projected", nodes=[selectors[0]], bridge=selectors[1])
            msg = f"{context}: ring metrics not configured; enabling projected rings with nodes={selectors[0]!r}, bridge={selectors[1]!r}"
        warnings.append(msg)
        _warn(msg)

    if not list(getattr(cfg, "gr", []) or []):
        gr = [GrMetricConfig(pair=None)]
        gr.extend(GrMetricConfig(pair=(a, b)) for (a, b) in _unordered_pairs(selectors))
        cfg.gr = gr
        msg = f"{context}: g(r) metrics not configured; using total and all unordered partial pairs"
        warnings.append(msg)
        _warn(msg)

    if not list(getattr(cfg, "sq", []) or []):
        sq = [SqMetricConfig(pair=None)]
        cfg.sq = sq
        msg = f"{context}: S(q) metrics not configured; using total structure factor"
        warnings.append(msg)
        _warn(msg)

    amorph = getattr(cfg, "amorphous", None)
    if amorph is not None and bool(getattr(amorph, "enabled", False)):
        has_total_sq = any(getattr(sm, "pair", None) is None for sm in list(getattr(cfg, "sq", []) or []))
        if not has_total_sq:
            cfg.sq = [SqMetricConfig(pair=None), *list(getattr(cfg, "sq", []) or [])]
            msg = f"{context}: amorphous detection requires total S(q); prepending a total-structure-factor metric"
            warnings.append(msg)
            _warn(msg)

    summary = {
        "auto_enabled": bool(getattr(metrics_cfg, "enabled", False) is False or not bool(getattr(metrics_cfg, "enabled", False))),
        "selectors": [str(s) for s in selectors],
        "n_pairs": int(len(getattr(cfg, "pairs", []) or [])),
        "n_coordinations": int(len(getattr(cfg, "coordinations", []) or [])),
        "n_angles": int(len(getattr(cfg, "angles", []) or [])),
        "rings_enabled": bool(getattr(getattr(cfg, "rings", None), "enabled", False)),
        "n_gr": int(len(getattr(cfg, "gr", []) or [])),
        "n_sq": int(len(getattr(cfg, "sq", []) or [])),
        "voids_enabled": bool(getattr(getattr(cfg, "voids", None), "enabled", False)),
        "amorphous_enabled": bool(getattr(getattr(cfg, "amorphous", None), "enabled", False)),
        "warnings": list(warnings),
    }
    return cfg, warnings, summary
