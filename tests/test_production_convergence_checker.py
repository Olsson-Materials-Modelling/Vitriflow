from __future__ import annotations

import pytest

pytest.importorskip("ase")

from vitriflow.config import ConvergenceConfig
from vitriflow.workflows.production_common import (
    _is_explicit_zero_incidence_curve,
    _prepare_cdf_curve_payload,
    _stack_sampled_curves_for_boxes,
    assess_fixed_count_convergence_posthoc,
    build_production_convergence_spec,
    compare_convergence_assessments,
    check_production_convergence,
)


def _box(density: float) -> dict:
    return {"density": float(density), "metrics": {}, "distributions": {}}


def _cdf_payload(**count_metadata) -> dict:
    return {
        "x": [0.0, 1.0, 2.0],
        "cdf": [0.0, 0.5, 1.0],
        **count_metadata,
    }


def _sq_representation(
    *,
    r_requested: float = 5.0,
    r_effective: float = 5.0,
    n_frames: int = 2,
    window: str = "lorch",
) -> dict:
    window_definition = {
        "lorch": "sinc(pi*r/r_support_effective)",
        "hann": "0.5*(1+cos(pi*r/r_support_effective))",
        "none": "1",
    }[window]
    scale = max(1.0, abs(float(r_requested)), abs(float(r_effective)))
    return {
        "schema": "vitriflow.sq_representation.v1",
        "observable": "static_structure_factor",
        "estimator": "isotropic_rdf_fourier_transform",
        "normalization": "unweighted_number_number_total",
        "normalization_family": "number_number",
        "normalization_formula": (
            "S_NN(q) = 1 + 4*pi*rho*integral[r^2*(g_NN(r)-1)*sinc(q*r) dr]"
        ),
        "self_term": 1.0,
        "pair": None,
        "rdf_normalization": "finite_population_unordered_pair_shell_volume",
        "scattering_weights": "none",
        "scattering_weighted": False,
        "dimensionless": True,
        "q_unit": "angstrom^-1",
        "r_unit": "angstrom",
        "termination_window": window,
        "termination_window_definition": window_definition,
        "radial_transform_kernel": "4*pi*r^2*sinc(q*r)",
        "radial_quadrature": "uniform_bin_midpoint",
        "r_support_requested_A": float(r_requested),
        "r_support_effective_A": float(r_effective),
        "r_support_clipped_to_unique_image_radius": bool(
            float(r_effective)
            < float(r_requested) - 64.0 * 2.220446049250313e-16 * scale
        ),
        "r_support_policy": (
            "minimum_half_shortest_lattice_translation_across_frames"
        ),
        "n_r_bins": 100,
        "q_min_A^-1": 0.0,
        "q_max_A^-1": 9.0,
        "n_q_points": 10,
        "q_zero_semantics": (
            "finite_r_windowed_rdf_transform_extrapolation_not_thermodynamic_compressibility"
        ),
        "frame_aggregation": (
            "equal_frame_mean_after_per_frame_density_transform"
        ),
        "density_handling": "per_frame_number_density_prefactor",
        "density_prefactor_unit": "angstrom^-3",
        "density_prefactors_A^-3": [0.05] * int(n_frames),
        "n_frames_requested": int(n_frames),
        "n_frames_used": int(n_frames),
    }


def _sq_box(box_id: int, *, representation=...) -> dict:
    payload = {"q": [float(i) for i in range(10)], "s": [1.0] * 10}
    if representation is not ...:
        payload["representation"] = representation
    return {
        "box": int(box_id),
        "density": 2.35,
        "metrics": {},
        "distributions": {"sq": {"sq_all": payload}},
    }


@pytest.mark.parametrize(
    "invalid_count",
    [1.5, 0, -1, float("nan"), float("inf"), True, False, ""],
)
def test_cdf_sample_count_aliases_require_exact_positive_integers(invalid_count):
    with pytest.raises(ValueError, match="finite exact positive integer"):
        _prepare_cdf_curve_payload(_cdf_payload(sample_count=invalid_count))


def test_cdf_sample_count_aliases_must_all_agree():
    with pytest.raises(ValueError, match="CDF count aliases disagree"):
        _prepare_cdf_curve_payload(
            _cdf_payload(sample_count=5, n_samples=6, count=5)
        )


def test_cdf_sample_count_aliases_accept_equal_exact_integer_encodings():
    _x, _cdf, sample_count = _prepare_cdf_curve_payload(
        _cdf_payload(sample_count=5, n_samples=5.0, count="5.0")
    )

    assert sample_count == 5


@pytest.mark.parametrize("boolean_count", [False, True])
def test_boolean_count_never_encodes_explicit_zero_incidence(boolean_count):
    assert not _is_explicit_zero_incidence_curve(
        {
            "available": False,
            "sample_count": boolean_count,
            "x": [],
            "cdf": [],
        },
        "cdf",
        grid_key="x",
    )


def test_conflicting_count_aliases_never_encode_explicit_zero_incidence():
    assert not _is_explicit_zero_incidence_curve(
        {
            "available": False,
            "sample_count": 0,
            "n_samples": 2,
            "x": [],
            "cdf": [],
        },
        "cdf",
        grid_key="x",
    )


def test_check_production_convergence_density_only_ci_passes_for_identical_boxes():
    boxes = [_box(2.35), _box(2.35)]
    ok, report = check_production_convergence(boxes, {}, ConvergenceConfig(mode="ci"))

    assert ok is True
    assert report["scalars"]["density"]["passed"] is True
    assert report["groups"]["long"]["passed"] is True
    assert report["ci_converged"] is True
    achieved = report["achieved_convergence_degree"]
    assert achieved["n_boxes"] == 2
    assert achieved["ci"]["worst_tolerance_utilization_ratio"] == pytest.approx(0.0)
    assert achieved["overall_active"]["worst_check"]["name"] == "scalar:density"


def test_fixed_count_terminal_posthoc_reports_degree_without_claiming_a_stop():
    report = assess_fixed_count_convergence_posthoc(
        [_box(2.35), _box(2.35)],
        {},
        ConvergenceConfig(mode="ci"),
        execution_target_met=True,
        min_boxes=2,
    )

    assert report["status"] == "fixed_n_terminal_posthoc_assessed"
    assert report["sampling_design"] == "fixed_n"
    assert report["assessment_role"] == "terminal_posthoc_diagnostic"
    assert report["assessment_performed"] is True
    assert report["posthoc_criterion_met"] is True
    assert report["posthoc_failed_items"] == []
    assert report["achieved_convergence_degree"]["n_boxes"] == 2
    assert report["achieved_convergence_degree"]["overall_active"][
        "worst_tolerance_utilization_ratio"
    ] == pytest.approx(0.0)
    assert report["check_convergence"] is False
    assert report["used_for_stopping"] is False
    assert report["stopping_assessment_performed"] is False
    assert report["stopping_status"] == "fixed_count_unassessed"
    assert report["inference_contract"]["sequentially_valid"] is False
    assert report["inference_contract"]["assessment_design"] == (
        "fixed_n_terminal_posthoc"
    )


def test_fixed_count_terminal_posthoc_names_failed_final_ensemble_items():
    report = assess_fixed_count_convergence_posthoc(
        [_box(1.0), _box(3.0)],
        {},
        ConvergenceConfig(mode="ci"),
        execution_target_met=True,
        min_boxes=2,
    )

    assert report["posthoc_criterion_met"] is False
    assert report["posthoc_failed_items"] == [
        {
            "section": "ci",
            "name": "scalar:density",
            "reason": "tolerance_not_met",
        }
    ]
    assert report["achieved_convergence_degree"]["overall_active"][
        "worst_tolerance_utilization_ratio"
    ] > 1.0


def test_fixed_count_terminal_posthoc_zero_accepted_is_explicitly_unassessed():
    report = assess_fixed_count_convergence_posthoc(
        [],
        None,
        ConvergenceConfig(mode="ci"),
        execution_target_met=False,
        min_boxes=2,
    )

    assert report["status"] == "fixed_n_terminal_posthoc_unassessed"
    assert report["assessment_performed"] is False
    assert report["posthoc_criterion_met"] is None
    assert report["achieved_convergence_degree"]["n_boxes"] == 0
    assert report["convergence_degree"]["overall"]["pass_fraction"] is None
    assert report["posthoc_failed_items"][-1] == {
        "section": "ensemble",
        "name": "accepted_boxes",
        "reason": "no_accepted_boxes",
    }


def test_check_production_convergence_density_only_stability_detects_shift():
    boxes = [_box(1.0), _box(1.0), _box(2.0), _box(2.0)]
    cfg = ConvergenceConfig(mode="stability", stability_bootstrap=0, stability_distance="wasserstein")
    ok, report = check_production_convergence(boxes, {}, cfg)

    assert ok is False
    assert report["stability"]["checks"]["density"]["passed"] is False
    assert report["stability_converged"] is False


def test_active_stability_mode_without_enough_boxes_is_unassessed_not_vacuous():
    ok, report = check_production_convergence(
        [_box(2.35), _box(2.35)],
        {},
        ConvergenceConfig(mode="stability", stability_bootstrap=0),
    )
    overall = report["convergence_degree"]["overall"]
    assert ok is False
    assert overall["converged"] is False
    assert overall["pass_fraction"] == 0.0
    assert overall["unassessed_active_sections"] == ["stability"]



def test_check_production_convergence_fails_closed_on_nonfinite_curve_payloads():
    boxes = []
    for i in range(4):
        boxes.append(
            {
                "box": i + 1,
                "density": 2.35,
                "metrics": {},
                "distributions": {
                    "bondlen": {
                        "bondlen_Si-Si": {
                            "x": [0.0, 1.0, 2.0],
                            "cdf": [float("nan"), float("nan"), float("nan")],
                        }
                    }
                },
            }
        )
    spec = {"bondlen_names": ["bondlen_Si-Si"]}

    ok, report = check_production_convergence(boxes, spec, ConvergenceConfig(mode="ci"))

    assert ok is False
    assert report["scalars"]["density"]["passed"] is True
    assert report["convergence_spec_effective"]["bondlen_names"] == []
    assert report["skipped_metrics"]
    assert report["skipped_metrics"][0]["name"] == "bondlen_Si-Si"
    assert report["criteria_integrity"]["passed"] is False
    assert report["converged"] is False
    assert report["convergence_degree"]["overall"]["converged"] is False
    assert report["convergence_degree"]["overall"]["pass_fraction"] < 1.0


def test_check_production_convergence_fails_on_mixed_zero_incidence():
    boxes = []
    for i, cdf in enumerate(([0.0, 0.5, 1.0], [])):
        boxes.append(
            {
                "box": i + 1,
                "density": 2.35,
                "metrics": {},
                "distributions": {
                    "angle": {
                        "angle_N-Si-N": {
                            "x": [0.0, 90.0, 180.0] if cdf else [],
                            "cdf": list(cdf),
                            "sample_count": 3 if cdf else 0,
                            "available": bool(cdf),
                        }
                    }
                },
            }
        )
    spec = {"angle_names": ["angle_N-Si-N"]}

    ok, report = check_production_convergence(boxes, spec, ConvergenceConfig(mode="ci"))

    assert ok is False
    assert report["convergence_spec_effective"]["angle_names"] == []
    assert any(item["name"] == "angle_N-Si-N" for item in report["skipped_metrics"])
    assert any(
        item["status"] == "mixed_zero_and_nonzero_incidence"
        for item in report["criteria_integrity"]["blocking_issues"]
    )


def test_all_box_zero_incidence_is_valid_but_does_not_invent_a_cdf():
    boxes = []
    for i in range(4):
        boxes.append(
            {
                "box": i + 1,
                "density": 2.35,
                "metrics": {"bond_incidence_Si-Si_count": 0.0},
                "distributions": {
                    "bondlen": {
                        "bondlen_Si-Si": {
                            "x": [],
                            "cdf": [],
                            "sample_count": 0,
                            "available": False,
                            "skip_reason": "no finite bond-length samples",
                        }
                    }
                },
            }
        )

    ok, report = check_production_convergence(
        boxes, {"bondlen_names": ["bondlen_Si-Si"]}, ConvergenceConfig(mode="ci")
    )

    assert ok is True
    assert report["criteria_integrity"]["passed"] is True
    assert report["criteria_integrity"]["valid_zero_incidence_count"] >= 1
    assert report["convergence_spec_effective"]["bondlen_names"] == []
    assert report["ensemble_cdfs"]["families"]["bondlen"] == {}


def test_ks_stability_uses_dimensionless_tolerance_not_density_units():
    boxes = [_box(1000.0), _box(1000.0), _box(1001.0), _box(1001.0)]
    cfg = ConvergenceConfig(
        mode="stability",
        stability_bootstrap=0,
        stability_distance="ks",
        stability_ks_tol=0.2,
        density_abs_tol=100.0,
    )
    ok, report = check_production_convergence(boxes, {}, cfg)

    check = report["stability"]["checks"]["density"]
    assert ok is False
    assert check["distance"] == pytest.approx(1.0)
    assert check["tol"] == pytest.approx(0.2)
    assert check["tolerance_basis"] == "dimensionless_stability_ks_tol"
    assert report["inference_contract"]["sequentially_valid"] is False


@pytest.mark.parametrize(
    ("family", "label", "grid_key", "value_key", "spec_key"),
    [
        ("gr", "gr_all", "r", "g", "gr_labels"),
        ("sq", "sq_all", "q", "s", "sq_labels"),
    ],
)
def test_sampled_curve_stability_aligns_physical_grids(
    family, label, grid_key, value_key, spec_key
):
    boxes = []
    for i in range(4):
        grid = [0.0, 1.0, 2.0] if i < 2 else [0.0, 2.0, 4.0]
        boxes.append(
            {
                "box": i + 1,
                "density": 2.35,
                "metrics": {},
                "distributions": {
                    family: {
                        label: {grid_key: grid, value_key: list(grid)}
                    }
                },
            }
        )

    ok, report = check_production_convergence(
        boxes,
        {spec_key: [label]},
        ConvergenceConfig(
            mode="stability",
            stability_bootstrap=0,
            stability_distance="wasserstein",
        ),
    )

    check = report["stability"]["checks"][f"{family}_curve:{label}"]
    assert ok is True
    assert check["passed"] is True
    assert check["grid"]["p"] == 3
    assert check["grid"]["grid_source"] == "ensemble_common_support_grid"
    curve = report["distributions"][label]
    assert curve["grid_alignment_method"] == "linear_interpolation_on_common_support"
    assert curve["source_grids_same"] is False


def test_sampled_curve_without_common_support_fails_closed():
    boxes = []
    for i, grid in enumerate(([0.0, 1.0], [2.0, 3.0])):
        boxes.append(
            {
                "box": i + 1,
                "density": 2.35,
                "metrics": {},
                "distributions": {
                    "gr": {"gr_all": {"r": list(grid), "g": [1.0, 1.0]}}
                },
            }
        )

    ok, report = check_production_convergence(
        boxes, {"gr_labels": ["gr_all"]}, ConvergenceConfig(mode="ci")
    )

    assert ok is False
    assert report["convergence_spec_effective"]["gr_labels"] == []
    issue = next(
        row
        for row in report["criteria_integrity"]["blocking_issues"]
        if row["name"] == "gr_all"
    )
    assert issue["status"] == "incompatible_curve_grids"


def test_legacy_sq_curves_remain_compatible_but_report_unavailable_representation():
    _q, _matrix, meta = _stack_sampled_curves_for_boxes(
        [_sq_box(1), _sq_box(2)],
        "sq",
        "sq_all",
        xkey="q",
        ykey="s",
    )

    assert meta["representation_validation_status"] == "legacy_unavailable"
    assert meta["representation_effective_r_support_A"] is None
    assert meta["representation_validation"]["reason"] == (
        "all S(q) payloads omit representation metadata"
    )


def test_validated_sq_representation_and_effective_support_are_exposed():
    representation = _sq_representation(r_requested=6.0, r_effective=5.0)
    _q, _matrix, meta = _stack_sampled_curves_for_boxes(
        [_sq_box(1, representation=representation), _sq_box(2, representation=dict(representation))],
        "sq",
        "sq_all",
        xkey="q",
        ykey="s",
    )

    validation = meta["representation_validation"]
    assert meta["representation_validation_status"] == "validated"
    assert meta["representation_effective_r_support_A"] == pytest.approx(5.0)
    assert validation["schema"] == "vitriflow.sq_representation.v1"
    assert validation["r_support_requested_A"] == pytest.approx(6.0)
    assert validation["r_support_effective_by_box_A"] == pytest.approx([5.0, 5.0])
    assert validation["n_frames_requested"] == 2
    assert validation["n_frames_used"] == 2


def test_mixed_legacy_and_represented_sq_curves_fail_closed():
    boxes = [_sq_box(1, representation=_sq_representation()), _sq_box(2)]

    ok, report = check_production_convergence(
        boxes,
        {"sq_labels": ["sq_all"]},
        ConvergenceConfig(mode="ci"),
    )

    assert ok is False
    assert report["convergence_spec_effective"]["sq_labels"] == []
    issue = next(
        row
        for row in report["criteria_integrity"]["blocking_issues"]
        if row["name"] == "sq_all"
    )
    assert issue["status"] == "incompatible_curve_grids"
    assert "mixed S(q) representation metadata availability" in issue["reason"]


def test_sq_representation_invariant_mismatch_blocks_convergence():
    lorch = _sq_representation(window="lorch")
    hann = _sq_representation(window="hann")

    ok, report = check_production_convergence(
        [_sq_box(1, representation=lorch), _sq_box(2, representation=hann)],
        {"sq_labels": ["sq_all"]},
        ConvergenceConfig(mode="ci"),
    )

    assert ok is False
    assert report["convergence_spec_effective"]["sq_labels"] == []
    issue = next(
        row
        for row in report["criteria_integrity"]["blocking_issues"]
        if row["name"] == "sq_all"
    )
    assert "invariant mismatch" in issue["reason"]
    assert "termination_window" in issue["reason"]


def test_sq_effective_real_space_support_mismatch_cannot_be_regridded_away():
    with pytest.raises(
        ValueError,
        match="q-grid interpolation cannot repair a different real-space estimator support",
    ):
        _stack_sampled_curves_for_boxes(
            [
                _sq_box(
                    1,
                    representation=_sq_representation(
                        r_requested=6.0, r_effective=5.0
                    ),
                ),
                _sq_box(
                    2,
                    representation=_sq_representation(
                        r_requested=6.0, r_effective=4.9
                    ),
                ),
            ],
            "sq",
            "sq_all",
            xkey="q",
            ykey="s",
        )


def test_sq_effective_support_equality_uses_scale_aware_roundoff_tolerance():
    eps = 2.220446049250313e-16
    support = 5.0
    within_tolerance = support + 64.0 * eps * support

    _q, _matrix, meta = _stack_sampled_curves_for_boxes(
        [
            _sq_box(
                1,
                representation=_sq_representation(
                    r_requested=6.0, r_effective=support
                ),
            ),
            _sq_box(
                2,
                representation=_sq_representation(
                    r_requested=6.0, r_effective=within_tolerance
                ),
            ),
        ],
        "sq",
        "sq_all",
        xkey="q",
        ykey="s",
    )

    assert meta["representation_validation_status"] == "validated"
    assert meta["representation_validation"]["r_support_match_tolerance_A"] == pytest.approx(
        128.0 * eps * within_tolerance
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda rep: rep["density_prefactors_A^-3"].__setitem__(0, 0.0),
            "density_prefactors_A\\^-3\\[0\\] must be finite and > 0",
        ),
        (
            lambda rep: rep.__setitem__("n_frames_used", 1),
            "n_frames_requested must equal n_frames_used",
        ),
        (
            lambda rep: rep.__setitem__("schema", "vitriflow.sq_representation.v2"),
            "schema must be",
        ),
    ],
)
def test_invalid_sq_representation_payload_is_rejected(mutation, reason):
    representation = _sq_representation()
    mutation(representation)

    with pytest.raises(ValueError, match=reason):
        _stack_sampled_curves_for_boxes(
            [_sq_box(1, representation=representation)],
            "sq",
            "sq_all",
            xkey="q",
            ykey="s",
        )


def test_sq_frame_count_must_match_across_boxes():
    with pytest.raises(ValueError, match="must be equal across boxes"):
        _stack_sampled_curves_for_boxes(
            [
                _sq_box(1, representation=_sq_representation(n_frames=2)),
                _sq_box(2, representation=_sq_representation(n_frames=3)),
            ],
            "sq",
            "sq_all",
            xkey="q",
            ykey="s",
        )


@pytest.mark.parametrize(
    ("x", "cdf", "reason_fragment"),
    [
        ([0.0, 2.0, 1.0], [0.0, 0.5, 1.0], "strictly increasing"),
        ([0.0, 1.0, 2.0], [0.0, 0.8, 0.7], "nondecreasing"),
        ([0.0, 1.0, 2.0], [0.0, 0.5, 1.1], "outside [0,1]"),
    ],
)
def test_corrupt_finite_cdf_payload_fails_closed(x, cdf, reason_fragment):
    boxes = [
        {
            "box": i + 1,
            "density": 2.35,
            "metrics": {},
            "distributions": {
                "bondlen": {
                    "bondlen_Si-Si": {
                        "x": list(x),
                        "cdf": list(cdf),
                        "sample_count": 10,
                        "available": True,
                    }
                }
            },
        }
        for i in range(2)
    ]

    ok, report = check_production_convergence(
        boxes,
        {"bondlen_names": ["bondlen_Si-Si"]},
        ConvergenceConfig(mode="ci"),
    )

    assert ok is False
    assert report["convergence_spec_effective"]["bondlen_names"] == []
    issue = next(
        row
        for row in report["criteria_integrity"]["blocking_issues"]
        if row["name"] == "bondlen_Si-Si"
    )
    assert reason_fragment in issue["reason"]


def test_fractional_coordination_cdf_uses_explicit_grid_not_array_index():
    import numpy as np

    from vitriflow.workflows.production_common import _stack_coord_cdfs_for_boxes

    boxes = [
        {
            "box": 1,
            "distributions": {
                "coord": {
                    "coord_soft": {
                        "x": [0.0, 0.5, 1.0],
                        "cdf": [0.2, 0.7, 1.0],
                        "sample_count": 5,
                    }
                }
            },
        },
        {
            "box": 2,
            "distributions": {
                "coord": {
                    "coord_soft": {
                        "x": [0.0, 1.0, 2.0],
                        "cdf": [0.2, 0.8, 1.0],
                        "sample_count": 5,
                    }
                }
            },
        },
    ]

    x, matrix, meta = _stack_coord_cdfs_for_boxes(boxes, "coord_soft")

    assert x.tolist() == [0.0, 0.5, 1.0, 2.0]
    assert matrix[0].tolist() == pytest.approx([0.2, 0.7, 1.0, 1.0])
    assert matrix[1].tolist() == pytest.approx([0.2, 0.2, 0.8, 1.0])
    assert meta["grid_source"] == "ensemble_common_explicit_coordination_grid"
    assert np.all(np.diff(x) > 0.0)


def test_singleton_zero_coordination_cdf_is_valid():
    from vitriflow.workflows.production_common import _stack_coord_cdfs_for_boxes

    boxes = [
        {
            "box": i + 1,
            "distributions": {
                "coord": {
                    "coord_zero": {
                        "x": [0.0],
                        "cdf": [1.0],
                        "sample_count": 4,
                    }
                }
            },
        }
        for i in range(2)
    ]

    x, matrix, meta = _stack_coord_cdfs_for_boxes(boxes, "coord_zero")

    assert x.tolist() == [0.0]
    assert matrix.tolist() == [[1.0], [1.0]]
    assert meta["grid_source"] == "ensemble_common_integer_grid"


def _exhaustive_convergence_box(box_id: int) -> dict:
    return {
        "box": int(box_id),
        "density": 2.35,
        "metrics": {
            "bond_incidence_Al-Al_count": 96.0,
            "bondlen_Al-Al_mean": 2.50,
            "bondlen_Al-Al_std": 0.10,
            "coord_Al-Al_mean": 12.0,
            "coord_Al-Al_std": 0.20,
            "angle_Al-Al-Al_mean": 60.0,
            "angle_Al-Al-Al_std": 5.0,
            "ring_frac_3": 0.25,
            "ring_mean_size": 3.0,
            "ring_count": 32.0,
            "gr_all_peak_r": 2.50,
            "gr_all_peak_height": 3.0,
            "gr_all_peak_fwhm": 0.25,
            "sq_all_peak_q": 2.0,
        },
        "distributions": {
            "bondlen": {
                "bondlen_Al-Al": {
                    "x": [2.0, 2.5, 3.0],
                    "cdf": [0.0, 0.5, 1.0],
                    "sample_count": 96,
                }
            },
            "angle": {
                "angle_Al-Al-Al": {
                    "x": [0.0, 60.0, 180.0],
                    "cdf": [0.0, 0.5, 1.0],
                    "sample_count": 192,
                }
            },
            "coord": {
                "coord_Al-Al": {
                    "x": [0.0, 6.0, 12.0],
                    "cdf": [0.0, 0.25, 1.0],
                    "sample_count": 64,
                }
            },
            "gr": {
                "gr_all": {"r": [0.5, 1.5, 2.5], "g": [0.0, 1.0, 3.0]}
            },
            "sq": {
                "sq_all": {"q": [0.0, 1.0, 2.0], "s": [1.0, 1.1, 1.2]}
            },
            "void": {
                "void_all": {
                    "x": [0.0, 0.5, 1.0],
                    "cdf": [0.0, 0.5, 1.0],
                    "sample_count": 256,
                }
            },
        },
    }


def _exhaustive_metrics_cfg() -> dict:
    return {
        "pairs": [{"pair": ["Al", "Al"]}],
        "angles": [{"triplet": ["Al", "Al", "Al"]}],
        "coordinations": [{"central": "Al", "neighbor": "Al"}],
        "rings": {"enabled": True},
        "gr": [{"pair": None}],
        "sq": [{"pair": None}],
        "voids": {"enabled": True},
    }


def test_all_configured_short_medium_long_metrics_have_strict_majority_evidence():
    boxes = [_exhaustive_convergence_box(i + 1) for i in range(10)]
    spec = build_production_convergence_spec(boxes[0], _exhaustive_metrics_cfg())

    ok, report = check_production_convergence(
        boxes,
        spec,
        ConvergenceConfig(mode="both", stability_bootstrap=0),
    )

    assert ok is True
    coverage = report["evidence_coverage"]
    assert coverage["minimum_boxes_required"] == 6
    assert coverage["passed"] is True
    assert all(coverage["groups"][group]["configured"] for group in ("short", "medium", "long"))
    assert all(coverage["groups"][group]["covered"] for group in ("short", "medium", "long"))
    active = set(coverage["active_sections"])
    assessed_items = [
        payload
        for payload in coverage["items"].values()
        if payload["section"] in active
    ]
    assert assessed_items
    assert all(payload["n_contributing_boxes"] >= 6 for payload in assessed_items)
    assert all(payload["strict_majority_supported"] for payload in assessed_items)
    assert all(payload["covered"] for payload in coverage["families"].values())
    assert report["metric_plumbing_coverage"]["strict_majority_enters_convergence"] is True

    classifications = report["convergence_spec_effective"]["scalar_metric_classification"]
    assert classifications["bondlen_Al-Al_mean"]["role"] == "convergence"
    assert classifications["sq_all_peak_q"]["role"] == "diagnostic_only"
    assert "complete S(q) curve" in classifications["sq_all_peak_q"]["reason"]
    assert classifications["ring_count"]["role"] == "diagnostic_only"


def test_configured_family_cannot_silently_disappear_from_convergence():
    boxes = [_box(2.35) for _ in range(10)]
    spec = build_production_convergence_spec(
        boxes[0],
        {"gr": [{"pair": None}]},
    )

    ok, report = check_production_convergence(
        boxes,
        spec,
        ConvergenceConfig(mode="ci"),
    )

    assert ok is False
    assert report["evidence_coverage"]["families"]["gr_curve"]["covered"] is False
    assert report["evidence_coverage"]["families"]["gr_peak"]["covered"] is False
    assert {
        issue["name"]
        for issue in report["criteria_integrity"]["blocking_issues"]
        if issue["kind"] == "configured_metric_family"
    } == {"gr_curve", "gr_peak"}


def test_canonical_convergence_parity_ignores_context_labels_but_not_numbers():
    import copy

    boxes = [_exhaustive_convergence_box(i + 1) for i in range(10)]
    spec = build_production_convergence_spec(boxes[0], _exhaustive_metrics_cfg())
    _ok, reference = check_production_convergence(
        boxes,
        spec,
        ConvergenceConfig(mode="ci"),
    )
    replay = copy.deepcopy(reference)
    replay.update(
        {
            "status": "analysis_posthoc",
            "assessment_role": "terminal_posthoc_diagnostic",
            "used_for_stopping": False,
        }
    )
    replay["groups"]["short"]["status"] = "analysis_only"

    equal = compare_convergence_assessments(reference, replay)
    assert equal["equivalent"] is True
    assert equal["n_differences"] == 0

    replay["scalars"]["density"]["mean"] = 2.36
    unequal = compare_convergence_assessments(reference, replay)
    assert unequal["equivalent"] is False
    assert any(row["path"] == "scalars.density.mean" for row in unequal["differences"])
