from __future__ import annotations

"""Pure helpers for metric-driven neighbour-cutoff requirements.

These utilities intentionally avoid importing :mod:`vitriflow.analysis.structure`
so that lightweight planning / dataset-discovery paths do not require ASE at
module import time.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from ..analysis.common import resolve_selector
from ..config import StructureMetricsConfig


def _pair_key(t1: int, t2: int) -> Tuple[int, int]:
    return (int(t1), int(t2)) if int(t1) <= int(t2) else (int(t2), int(t1))


def required_pairs_from_metrics(
    metrics: StructureMetricsConfig,
    *,
    type_to_species: Optional[Sequence[str]],
) -> List[Tuple[int, int]]:
    """Required pairs from."""

    pairs: List[Tuple[int, int]] = []

    for pm in metrics.pairs:
        a, b = pm.pair
        for ta in resolve_selector(a, type_to_species):
            for tb in resolve_selector(b, type_to_species):
                pairs.append(_pair_key(ta, tb))

    for cm in metrics.coordinations:
        for ta in resolve_selector(cm.central, type_to_species):
            for tb in resolve_selector(cm.neighbor, type_to_species):
                pairs.append(_pair_key(ta, tb))

    for am in metrics.angles:
        a, b, c = am.triplet
        for tb in resolve_selector(b, type_to_species):
            for ta in resolve_selector(a, type_to_species):
                pairs.append(_pair_key(tb, ta))
            for tc in resolve_selector(c, type_to_species):
                pairs.append(_pair_key(tb, tc))

    if metrics.rings is not None and bool(metrics.rings.enabled) and metrics.rings.bond_pairs:
        for bp in metrics.rings.bond_pairs:
            a, b = bp.pair
            for ta in resolve_selector(a, type_to_species):
                for tb in resolve_selector(b, type_to_species):
                    pairs.append(_pair_key(ta, tb))

    return sorted(set(pairs))


def fixed_cutoffs_from_metrics(
    metrics: StructureMetricsConfig,
    *,
    type_to_species: Optional[Sequence[str]],
) -> Dict[Tuple[int, int], float]:
    """Fixed cutoffs from."""

    out: Dict[Tuple[int, int], float] = {}

    for pm in metrics.pairs:
        if pm.cutoff is None:
            continue
        a, b = pm.pair
        for ta in resolve_selector(a, type_to_species):
            for tb in resolve_selector(b, type_to_species):
                out[_pair_key(ta, tb)] = float(pm.cutoff)

    for cm in metrics.coordinations:
        if cm.cutoff is None:
            continue
        for ta in resolve_selector(cm.central, type_to_species):
            for tb in resolve_selector(cm.neighbor, type_to_species):
                out[_pair_key(ta, tb)] = float(cm.cutoff)

    if metrics.rings is not None and bool(metrics.rings.enabled):
        for bp in metrics.rings.bond_pairs:
            if bp.cutoff is None:
                continue
            a, b = bp.pair
            for ta in resolve_selector(a, type_to_species):
                for tb in resolve_selector(b, type_to_species):
                    out[_pair_key(ta, tb)] = float(bp.cutoff)

    return out
