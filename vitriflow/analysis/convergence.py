from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ConvergenceDecision:
    chosen_index: int
    chosen_value: float
    reference_value: float
    deltas: list[float]
    allowed: list[float]
    passed: list[bool]
    kind: str  # rate size


def allowed_delta(mu_ref: float, se: float, se_ref: float, rel_tol: float, abs_tol: float, z: float) -> float:
    """Allowed delta."""
    tol = max(float(abs_tol), float(rel_tol) * abs(float(mu_ref)))
    sigma = float(np.sqrt(float(se) ** 2 + float(se_ref) ** 2))
    return float(tol - float(z) * sigma)


def choose_fastest_converged(
    x: Sequence[float],
    mu: Sequence[float],
    se: Sequence[float],
    *,
    rel_tol: float,
    abs_tol: float,
    z: float,
    kind: str,
) -> ConvergenceDecision:
    """Fastest converged."""
    if not (len(x) == len(mu) == len(se)):
        raise ValueError("x, mu, se must have same length")
    if len(x) < 2:
        raise ValueError("Need >= 2 points for convergence decision")

    mu_ref = float(mu[-1])
    se_ref = float(se[-1])

    deltas: list[float] = []
    allowed: list[float] = []
    passed: list[bool] = []

    chosen = None
    for i in range(len(x)):
        d = abs(float(mu[i]) - mu_ref)
        a = allowed_delta(mu_ref, float(se[i]), se_ref, rel_tol, abs_tol, z)
        ok = d <= a
        deltas.append(d)
        allowed.append(a)
        passed.append(ok)
        if chosen is None and ok:
            chosen = i

    if chosen is None:
        chosen = len(x) - 1  # fall reference itself

    return ConvergenceDecision(
        chosen_index=int(chosen),
        chosen_value=float(x[chosen]),
        reference_value=mu_ref,
        deltas=deltas,
        allowed=allowed,
        passed=passed,
        kind=kind,
    )
