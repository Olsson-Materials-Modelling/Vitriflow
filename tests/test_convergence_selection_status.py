from __future__ import annotations

import math

import pytest

from vitriflow.analysis.convergence import choose_fastest_converged


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("se", -0.1),
        ("se_ref", -0.1),
        ("rel_tol", -0.1),
        ("abs_tol", -0.1),
        ("z", -0.1),
        ("mu_ref", float("nan")),
    ],
)
def test_allowed_delta_rejects_invalid_physical_inputs(field: str, value: float) -> None:
    from vitriflow.analysis.convergence import allowed_delta

    kwargs = {
        "mu_ref": 1.0,
        "se": 0.1,
        "se_ref": 0.1,
        "rel_tol": 0.1,
        "abs_tol": 0.01,
        "z": 1.96,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        allowed_delta(**kwargs)


def test_no_passing_scan_candidate_is_explicit_unconverged_fallback():
    decision = choose_fastest_converged(
        [10.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        rel_tol=0.0,
        abs_tol=0.01,
        z=1.96,
        kind="rate",
    )

    assert decision.passed == [False, False]
    assert decision.chosen_index == 1
    assert decision.selection_converged is False
    assert decision.fallback_used is True
    assert decision.selection_status == "fallback_unconverged"


def test_scan_selection_requires_every_point_through_reference_to_pass():
    decision = choose_fastest_converged(
        [3.0, 2.0, 1.0],
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        rel_tol=0.0,
        abs_tol=0.1,
        z=1.96,
        kind="rate",
    )

    assert decision.passed == [True, False, True]
    assert decision.tail_passed == [False, False, True]
    assert decision.chosen_index == 2
    assert decision.chosen_value == 1.0


def test_single_realization_uncertainty_is_explicitly_unassessed():
    decision = choose_fastest_converged(
        [10.0, 1.0],
        [1.0, 1.0],
        [float("nan"), float("nan")],
        rel_tol=0.0,
        abs_tol=0.1,
        z=1.96,
        kind="rate",
    )

    assert decision.selection_converged is False
    assert decision.selection_status == "incomplete_evidence_unassessed"
    assert decision.point_assessed == [False, False]
    assert decision.tail_assessed == [False, False]
    assert decision.selection_reason is not None
    assert all("stderr" in row["fields"] for row in decision.blocking_points)
    assert all(math.isnan(value) for value in decision.allowed)
