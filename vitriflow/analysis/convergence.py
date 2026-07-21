from __future__ import annotations

from dataclasses import dataclass, field
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
    # Pointwise ``passed`` is retained for backwards compatibility.  A scan
    # point is selectable only when it *and every higher-fidelity point through
    # the reference* passes; otherwise a non-monotone excursion could be hidden
    # by comparing only the candidate and reference endpoints.
    tail_passed: list[bool] = field(default_factory=list)
    # ``chosen_*`` is always populated so an execution plan can still be
    # produced when a bounded scan has no passing point.  These fields make
    # that fallback impossible to mistake for a converged selection.
    selection_converged: bool = False
    fallback_used: bool = True
    selection_status: str = "unassessed"
    point_assessed: list[bool] = field(default_factory=list)
    tail_assessed: list[bool] = field(default_factory=list)
    blocking_points: list[dict[str, object]] = field(default_factory=list)
    selection_reason: str | None = None


def allowed_delta(mu_ref: float, se: float, se_ref: float, rel_tol: float, abs_tol: float, z: float) -> float:
    """Return the tolerance remaining after an uncertainty allowance."""

    values = {
        "mu_ref": float(mu_ref),
        "se": float(se),
        "se_ref": float(se_ref),
        "rel_tol": float(rel_tol),
        "abs_tol": float(abs_tol),
        "z": float(z),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("convergence tolerance inputs must be finite")
    for name in ("se", "se_ref", "rel_tol", "abs_tol", "z"):
        if values[name] < 0.0:
            raise ValueError(f"{name} must be >= 0")
    tol = max(values["abs_tol"], values["rel_tol"] * abs(values["mu_ref"]))
    sigma = float(np.hypot(values["se"], values["se_ref"]))
    return float(tol - values["z"] * sigma)


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
    point_assessed: list[bool] = []
    blocking_points: list[dict[str, object]] = []

    for i in range(len(x)):
        x_i = float(x[i])
        mu_i = float(mu[i])
        se_i = float(se[i])
        assessed = bool(
            np.isfinite(x_i)
            and np.isfinite(mu_i)
            and np.isfinite(se_i)
            and se_i >= 0.0
            and np.isfinite(mu_ref)
            and np.isfinite(se_ref)
            and se_ref >= 0.0
        )
        if assessed:
            d = abs(mu_i - mu_ref)
            a = allowed_delta(mu_ref, se_i, se_ref, rel_tol, abs_tol, z)
            ok = bool(np.isfinite(a) and d <= a)
        else:
            d = float("nan")
            a = float("nan")
            ok = False
            fields: list[str] = []
            if not np.isfinite(x_i):
                fields.append("x")
            if not np.isfinite(mu_i):
                fields.append("mean")
            if not np.isfinite(se_i) or se_i < 0.0:
                fields.append("stderr")
            if not np.isfinite(mu_ref):
                fields.append("reference_mean")
            if not np.isfinite(se_ref) or se_ref < 0.0:
                fields.append("reference_stderr")
            blocking_points.append(
                {
                    "index": int(i),
                    "x": (float(x_i) if np.isfinite(x_i) else None),
                    "fields": fields,
                    "reason": "scan point has missing, non-finite, or invalid uncertainty evidence",
                }
            )
        deltas.append(d)
        allowed.append(a)
        passed.append(ok)
        point_assessed.append(assessed)

    tail_passed = [bool(all(passed[i:])) for i in range(len(passed))]
    tail_assessed = [
        bool(all(point_assessed[i:])) for i in range(len(point_assessed))
    ]
    chosen = next(
        (
            i
            for i, ok in enumerate(tail_passed)
            if ok and tail_assessed[i]
        ),
        None,
    )

    selection_converged = chosen is not None
    if chosen is None:
        # Retain the highest-fidelity reference as an explicitly labelled
        # fallback.  The reference is not automatically converged: its own
        # uncertainty term can be larger than the requested tolerance.
        chosen = len(x) - 1

    return ConvergenceDecision(
        chosen_index=int(chosen),
        chosen_value=float(x[chosen]),
        reference_value=mu_ref,
        deltas=deltas,
        allowed=allowed,
        passed=passed,
        kind=kind,
        tail_passed=tail_passed,
        selection_converged=bool(selection_converged),
        fallback_used=bool(not selection_converged),
        selection_status=(
            "converged"
            if selection_converged
            else (
                "incomplete_evidence_unassessed"
                if not tail_assessed[chosen]
                else "fallback_unconverged"
            )
        ),
        point_assessed=point_assessed,
        tail_assessed=tail_assessed,
        blocking_points=blocking_points,
        selection_reason=(
            "the selected fallback tail contains missing, non-finite, or invalid uncertainty evidence"
            if not tail_assessed[chosen]
            else None
        ),
    )
