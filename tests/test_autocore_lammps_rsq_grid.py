from __future__ import annotations

from pathlib import Path

import numpy as np

from vitriflow.potential import (
    _buckingham_energy_derivatives,
    _lammps_rsq_grid,
    _repulsive_regularized_pair_energy_derivatives,
    _tabulated_buckingham_section_arrays,
)
from vitriflow.workflows import preflight


_POINTS = 288_000
_RLO = 0.1
_RHI = 15.0


def _target_hybrid_oo_pair() -> dict[str, object]:
    """Reproduce the O--O pair from the reported Ga2O3 hybrid failure."""

    pair: dict[str, object] = {
        "section": "P2_2",
        "pair": [2, 2],
        "A": 1388.77,
        "rho": 0.36,
        "C": 175.0,
        "buck_cutoff": 15.0,
        "pair_cutoff": 15.0,
        "z_i": 8,
        "z_j": 8,
        "r_in": 1.4479934691025773,
        "r_out": 1.7069519041351013,
        "q_i": -1.2,
        "q_j": -1.2,
        "morse_terms": [],
        "shift_buck": False,
        "shift_morse": False,
        "coul_mode": "long",
        "coul_cutoff": 15.0,
        "gewald": 0.24022096,
    }
    join_energy, _du, _d2u = _buckingham_energy_derivatives(
        np.asarray([pair["r_out"]], dtype=float),
        A=float(pair["A"]),
        rho=float(pair["rho"]),
        C=float(pair["C"]),
        cutoff=float(pair["buck_cutoff"]),
        shift=False,
    )
    pair["join_energy"] = float(join_energy[0])
    pair["join_energy_component"] = "buckingham"
    return pair


def _lammps_force_secant_indices(
    radius: np.ndarray,
    energy: np.ndarray,
    force: np.ndarray,
) -> np.ndarray:
    left = -(energy[1:-1] - energy[:-2]) / (radius[1:-1] - radius[:-2])
    right = -(energy[2:] - energy[1:-1]) / (radius[2:] - radius[1:-1])
    interior = force[1:-1]
    return np.flatnonzero(
        (interior < np.minimum(left, right))
        | (interior > np.maximum(left, right))
    ) + 1


def test_lammps_rsq_grid_matches_pair_table_cpp_operation_order() -> None:
    indices = np.arange(_POINTS, dtype=np.float64)
    expected = np.sqrt(
        _RLO * _RLO
        + ((_RHI * _RHI - _RLO * _RLO) * indices) / (_POINTS - 1)
    )
    actual = _lammps_rsq_grid(_RLO, _RHI, _POINTS)

    assert np.array_equal(actual, expected)

    # The former linspace construction is mathematically identical but not
    # operation-order identical.  This exact target grid differs at 53,186
    # knots, including the roundoff-sensitive tail inflection from the report.
    legacy = np.sqrt(
        np.linspace(_RLO * _RLO, _RHI * _RHI, _POINTS, dtype=float)
    )
    assert np.count_nonzero(actual != legacy) == 53_186
    assert np.max(np.abs(actual - legacy)) == np.float64(
        1.7763568394002505e-15
    )


def test_historical_nonruntime_rsq_grid_is_rejected_fail_closed(
    monkeypatch,
) -> None:
    pair = _target_hybrid_oo_pair()
    legacy_radius = np.sqrt(
        np.linspace(_RLO * _RLO, _RHI * _RHI, _POINTS, dtype=float)
    )
    evaluation_radius = legacy_radius.copy()
    evaluation_radius[-1] = np.nextafter(_RHI, _RLO)
    energy, derivative, _curvature = _repulsive_regularized_pair_energy_derivatives(
        evaluation_radius,
        pair=pair,
        units_style="metal",
    )
    force = -np.asarray(derivative, dtype=float)

    monkeypatch.setattr(
        preflight,
        "_parse_pair_table_file",
        lambda _path: {
            "P2_2": {
                "r": legacy_radius,
                "energy": np.asarray(energy, dtype=float),
                "force": force,
            }
        },
    )
    report = preflight._audit_lammps_inflection_warnings(
        spec={
            "points": _POINTS,
            "r_min": _RLO,
            "units": "metal",
            "pairs": [pair],
        },
        table_path=Path("unused.table"),
        observed_warnings=[],
    )

    # Transcendental rounding can create an adjacent secant flag on NumPy
    # 1.26 but not 2.x.  That count is deliberately irrelevant: a table on the
    # wrong runtime grid must be rejected before warnings are classified.
    assert report["passed"] is False
    assert any("exact LAMMPS RSQ" in row for row in report["blocking_warnings"])
    assert report["advisory_warnings"] == []


def test_new_hybrid_table_warnings_on_lammps_grid_are_all_proven_inflections(
    monkeypatch,
) -> None:
    pair = _target_hybrid_oo_pair()
    spec = {
        "version": 10,
        "kind": "additive_hybrid_buckingham_zbl_table",
        "points": _POINTS,
        "r_min": _RLO,
        "units": "metal",
        "force_mode": "analytic",
        "pairs": [pair],
    }
    section = _tabulated_buckingham_section_arrays(pair, spec=spec)
    radius = np.asarray(section["r"], dtype=float)
    energy = np.asarray(section["energy"], dtype=float)
    force = np.asarray(section["force"], dtype=float)

    assert np.array_equal(radius, _lammps_rsq_grid(_RLO, _RHI, _POINTS))
    flagged_indices = _lammps_force_secant_indices(radius, energy, force)
    monkeypatch.setattr(
        preflight,
        "_parse_pair_table_file",
        lambda _path: {
            "P2_2": {"r": radius, "energy": energy, "force": force}
        },
    )
    warnings = []
    if flagged_indices.size:
        warnings = [
            f"WARNING: {len(flagged_indices)} of 288000 force values in table P2_2 "
            "are inconsistent with -dE/dr."
        ]
    report = preflight._audit_lammps_inflection_warnings(
        spec=spec,
        table_path=Path("unused.table"),
        observed_warnings=warnings,
    )
    audited = report["pairs"]["P2_2"]["flagged_knots"]
    assert [row["index"] for row in audited] == flagged_indices.tolist()
    assert all(row["brackets_analytic_inflection"] for row in audited)
    assert report["passed"] is True
    assert report["blocking_warnings"] == []
    assert report["advisory_warnings"] == warnings
