from __future__ import annotations

import json

import numpy as np
import pytest

from vitriflow.workflows.preflight import (
    _compare_pair_table_sections,
    _format_tabulation_candidate_summary,
    _record_tabulation_candidate_error,
    _table_force_energy_consistency_report,
    _tabulation_candidate_sort_key,
    _tabulation_candidate_verification_status,
    _write_tabulation_refinement_report,
)


def _candidate(**updates):
    candidate = {
        "force_mode": "fd_consistent",
        "table_points": 128000,
        "verify_passed": False,
        "verification_status": "not_run",
        "stability_ok": None,
        "stability_status": "not_run",
        "passed": False,
        "warnings": [],
    }
    candidate.update(updates)
    return candidate


def test_candidate_execution_error_reports_stage_headline_and_unavailable_metrics(
    tmp_path,
) -> None:
    candidate = _candidate()
    error = RuntimeError(
        "Command failed (code 1): lmp -in in.lammps\n"
        "--- screen.out (tail) ---\n"
        "OMP_NUM_THREADS environment is not set.\n"
        "ERROR: Set command on system without atoms (src/set.cpp:87)\n"
        "Last input line: set type 1 charge 1.910"
    )
    _record_tabulation_candidate_error(
        candidate,
        error,
        failure_stage="verification",
        stability_attempted=False,
    )

    assert candidate["verification_status"] == "error"
    assert candidate["failure_stage"] == "verification"
    assert candidate["error_type"] == "RuntimeError"
    assert candidate["stability_status"] == "not_run"
    formatted = _format_tabulation_candidate_summary(candidate)
    assert "verify=error" in formatted
    assert "failure_stage=verification" in formatted
    assert "error_type=RuntimeError" in formatted
    assert "max_energy_ratio=n/a" in formatted
    assert "max_force_ratio=n/a" in formatted
    assert "ERROR: Set command on system without atoms" in formatted
    assert "OMP_NUM_THREADS" not in formatted
    assert "nan" not in formatted.lower()

    json_path, text_path = _write_tabulation_refinement_report(
        outdir=tmp_path,
        report={
            "status": "rejected_fail_closed",
            "fallback_to_analytic": False,
            "reason": "all candidates errored before energy/force comparison",
            "accepted_candidate": None,
            "best_candidate": candidate,
            "candidates": [candidate],
        },
    )
    summary = text_path.read_text()
    assert "ERROR: Set command on system without atoms" in summary
    assert "OMP_NUM_THREADS" not in summary
    assert "nan" not in summary.lower()
    persisted = json.loads(json_path.read_text())
    assert "OMP_NUM_THREADS" in persisted["candidates"][0]["error"]


def test_candidate_stability_exception_preserves_successful_verification() -> None:
    candidate = _candidate(
        verify_passed=True,
        verification_status="pass",
        comparison={"overall": {"max_energy_ratio": 0.2, "max_force_ratio": 0.3}},
    )
    _record_tabulation_candidate_error(
        candidate,
        RuntimeError("stability runner failed"),
        failure_stage="stability",
        stability_attempted=True,
    )

    assert _tabulation_candidate_verification_status(candidate) == "pass"
    assert candidate["failure_stage"] == "stability"
    assert candidate["stability_ok"] is False
    assert candidate["stability_status"] == "fail"


def test_candidate_metrics_distinguish_nonfinite_and_sort_fail_closed() -> None:
    nonfinite = _candidate(
        verification_status="fail",
        comparison={
            "overall": {
                "max_energy_ratio": float("nan"),
                "max_force_ratio": float("inf"),
            }
        },
        self_consistency={"overall": {"max_force_ratio": float("nan")}},
    )
    finite = _candidate(
        verification_status="fail",
        comparison={"overall": {"max_energy_ratio": 2.0, "max_force_ratio": 3.0}},
        self_consistency={"overall": {"max_force_ratio": 4.0}},
    )

    formatted = _format_tabulation_candidate_summary(nonfinite)
    assert "max_energy_ratio=nonfinite" in formatted
    assert "max_force_ratio=nonfinite" in formatted
    assert not any(np.isnan(value) for value in _tabulation_candidate_sort_key(nonfinite))
    assert _tabulation_candidate_sort_key(finite) < _tabulation_candidate_sort_key(
        nonfinite
    )


def test_candidate_sort_prefers_numerically_valid_warning_audit_failure() -> None:
    numerically_valid = _candidate(
        force_mode="analytic",
        table_style="spline",
        verification_status="fail",
        comparison={
            "passed": True,
            "overall": {"max_energy_ratio": 0.03, "max_force_ratio": 0.14},
        },
        self_consistency={
            "passed": True,
            "overall": {"max_force_ratio": 0.02},
        },
        blocking_warnings=["warning classifier mismatch"],
    )
    coarse_fallback = _candidate(
        force_mode="fd_consistent",
        table_style="linear",
        verification_status="fail",
        comparison={
            "passed": False,
            "overall": {"max_energy_ratio": 25.2, "max_force_ratio": 87.0},
        },
        self_consistency={
            "passed": False,
            "overall": {"max_force_ratio": 1.6},
        },
        blocking_warnings=[],
    )

    assert _tabulation_candidate_sort_key(numerically_valid) < (
        _tabulation_candidate_sort_key(coarse_fallback)
    )
    summary = _format_tabulation_candidate_summary(numerically_valid)
    assert "curve=pass" in summary
    assert "work=pass" in summary
    assert "warning_audit=fail" in summary


@pytest.mark.parametrize("field", ["radius", "energy", "force"])
def test_pair_table_comparison_explicitly_rejects_nonfinite_realized_curve(
    field: str,
) -> None:
    radius = np.asarray([0.5, 1.0, 1.5])
    energy = np.asarray([2.0, 1.0, 0.0])
    force = np.asarray([1.0, 1.0, 0.0])
    reference = {"P1_1": {"r": radius, "energy": energy, "force": force}}
    realized = {
        "P1_1": {
            "r": radius.copy(),
            "energy": energy.copy(),
            "force": force.copy(),
        }
    }
    key = "r" if field == "radius" else field
    realized["P1_1"][key][1] = float("nan")

    with pytest.raises(
        ValueError,
        match=rf"realized pair table section 'P1_1' contains non-finite {field}",
    ):
        _compare_pair_table_sections(
            reference,
            realized,
            rel_tol=5.0e-5,
            abs_tol_frac=1.0e-7,
        )


def test_force_energy_consistency_explicitly_rejects_nonfinite_energy() -> None:
    sections = {
        "P1_1": {
            "r": np.asarray([0.5, 1.0, 1.5]),
            "energy": np.asarray([2.0, float("nan"), 0.0]),
            "force": np.asarray([1.0, 1.0, 0.0]),
        }
    }

    with pytest.raises(
        ValueError,
        match="realized pair table section 'P1_1' contains non-finite energy",
    ):
        _table_force_energy_consistency_report(
            sections,
            rel_tol=5.0e-5,
            abs_tol_frac=1.0e-7,
        )
