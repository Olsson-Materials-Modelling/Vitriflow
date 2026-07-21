from __future__ import annotations

import math

import pytest

from vitriflow.config import ConvergenceConfig, ProductionEnsembleConfig
from vitriflow.workflows.output_analysis import _analysis_convergence_report_from_boxes


def test_analysis_ensemble_cdf_aligns_adaptive_box_specific_grids() -> None:
    boxes = [
        {
            "box": 1,
            "density": 2.0,
            "metrics": {},
            "distributions": {
                "bondlen": {
                    "bondlen_A-B": {
                        "x": [0.0, 1.0, 2.0],
                        "cdf": [0.0, 0.5, 1.0],
                        "sample_count": 10,
                    }
                }
            },
        },
        {
            "box": 2,
            "density": 2.1,
            "metrics": {},
            "distributions": {
                "bondlen": {
                    "bondlen_A-B": {
                        "x": [0.0, 0.5, 1.5, 2.5],
                        "cdf": [0.0, 0.2, 0.8, 1.0],
                        "sample_count": 12,
                    }
                }
            },
        },
    ]
    spec = {
        "bondlen_names": ["bondlen_A-B"],
        "angle_names": [],
        "coord_names": [],
        "ring_keys": [],
        "ring_has_mean_size": False,
        "gr_labels": [],
        "sq_labels": [],
        "void_names": [],
    }

    report = _analysis_convergence_report_from_boxes(
        boxes=boxes,
        conv_spec=spec,
        conv_cfg=ConvergenceConfig(),
        prod_cfg=ProductionEnsembleConfig(),
        status="ok",
        reason="test",
    )

    entry = report["distributions"]["bondlen_A-B"]
    assert entry["alignment"]["method"] == "union_support_right_continuous_cdf"
    assert entry["x"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert entry["n_effective"] == 2
    assert report["skipped_metrics"] == []
    assert all(math.isfinite(float(v)) for v in entry["mean"])
    assert all(0.0 <= float(v) <= 1.0 for v in entry["mean"])
    assert entry["mean"] == sorted(entry["mean"])


def _analysis_spec(**overrides):
    spec = {
        "bondlen_names": [],
        "angle_names": [],
        "coord_names": [],
        "ring_keys": [],
        "ring_has_mean_size": False,
        "gr_labels": [],
        "sq_labels": [],
        "void_names": [],
    }
    spec.update(overrides)
    return spec


def _analysis_report(boxes, spec):
    return _analysis_convergence_report_from_boxes(
        boxes=boxes,
        conv_spec=spec,
        conv_cfg=ConvergenceConfig(),
        prod_cfg=ProductionEnsembleConfig(),
        status="ok",
        reason="test",
    )


@pytest.mark.parametrize(
    ("bad_x", "bad_cdf", "reason_fragment"),
    [
        ([0.0, 2.0, 1.0], [0.0, 0.5, 1.0], "strictly increasing"),
        ([0.0, 1.0, 1.0], [0.0, 0.5, 1.0], "strictly increasing"),
        ([0.0, 1.0, 2.0], [0.0, 0.8, 0.7], "nondecreasing"),
        ([0.0, 1.0, 2.0], [0.0, 0.5, 1.1], "[0, 1]"),
    ],
)
def test_analysis_ensemble_cdf_rejects_malformed_payload_and_fails_closed(
    bad_x, bad_cdf, reason_fragment
) -> None:
    valid = {
        "x": [0.0, 1.0, 2.0],
        "cdf": [0.0, 0.5, 1.0],
        "sample_count": 10,
    }
    boxes = [
        {"box": 1, "metrics": {}, "distributions": {"bondlen": {"bondlen_A-B": dict(valid)}}},
        {"box": 2, "metrics": {}, "distributions": {"bondlen": {"bondlen_A-B": dict(valid)}}},
        {
            "box": 3,
            "metrics": {},
            "distributions": {
                "bondlen": {
                    "bondlen_A-B": {
                        "x": bad_x,
                        "cdf": bad_cdf,
                        "sample_count": 10,
                    }
                }
            },
        },
    ]

    report = _analysis_report(boxes, _analysis_spec(bondlen_names=["bondlen_A-B"]))
    entry = report["distributions"]["bondlen_A-B"]

    # The valid subset remains useful descriptively, but incomplete evidence
    # cannot become a positive convergence result.
    assert entry["status"] == "incomplete"
    assert entry["n_available"] == 2
    assert entry["mean"] == pytest.approx([0.0, 0.5, 1.0])
    assert entry["available_subset_ci_within_tolerance"] is True
    assert entry["passed"] is None
    assert entry["convergence_assessed"] is False
    assert entry["convergence_status"] == "unassessed_incomplete_evidence"
    assert entry["invalid_boxes"] == [3]
    assert reason_fragment in entry["invalid_payloads"][0]["reason"]
    assert report["ensemble_cdfs"]["status"] == "incomplete"
    assert report["ensemble_cdfs"]["blocking_distributions"] == ["bondlen_A-B"]


def test_analysis_ensemble_curve_missing_declared_box_fails_closed() -> None:
    payload = {"r": [0.0, 1.0, 2.0], "g": [0.0, 1.0, 0.5]}
    boxes = [
        {"box": 1, "metrics": {}, "distributions": {"gr": {"gr_A-B": dict(payload)}}},
        {"box": 2, "metrics": {}, "distributions": {"gr": {"gr_A-B": dict(payload)}}},
        {"box": 3, "metrics": {}, "distributions": {"gr": {}}},
    ]

    report = _analysis_report(boxes, _analysis_spec(gr_labels=["gr_A-B"]))
    entry = report["distributions"]["gr_A-B"]

    assert entry["mean"] == pytest.approx(payload["g"])
    assert entry["available_subset_ci_within_tolerance"] is True
    assert entry["passed"] is None
    assert entry["missing_boxes"] == [3]
    assert entry["blocking_boxes"] == [
        {"box": 3, "kind": "missing_payload", "reason": "curve payload is absent"}
    ]


@pytest.mark.parametrize("bad_count", [1.5, -1, float("nan"), float("inf"), True, "invalid"])
def test_analysis_ensemble_curve_rejects_invalid_declared_sample_count(bad_count) -> None:
    valid = {
        "x": [0.0, 1.0, 2.0],
        "cdf": [0.0, 0.5, 1.0],
        "sample_count": 10,
    }
    malformed = dict(valid, sample_count=bad_count)
    boxes = [
        {"box": 1, "metrics": {}, "distributions": {"bondlen": {"bondlen_A-B": valid}}},
        {"box": 2, "metrics": {}, "distributions": {"bondlen": {"bondlen_A-B": malformed}}},
    ]

    entry = _analysis_report(
        boxes,
        _analysis_spec(bondlen_names=["bondlen_A-B"]),
    )["distributions"]["bondlen_A-B"]

    assert entry["status"] == "incomplete"
    assert entry["n_available"] == 1
    assert entry["sample_count_total"] == 10
    assert entry["invalid_boxes"] == [2]
    assert "finite exact nonnegative integer" in entry["invalid_payloads"][0]["reason"]
    assert entry["passed"] is None


def test_analysis_ensemble_curve_rejects_conflicting_sample_count_aliases() -> None:
    payload = {
        "x": [0.0, 1.0, 2.0],
        "cdf": [0.0, 0.5, 1.0],
        "sample_count": 10,
        "n_samples": 9,
    }
    boxes = [
        {"box": 1, "metrics": {}, "distributions": {"bondlen": {"bondlen_A-B": payload}}}
    ]

    entry = _analysis_report(
        boxes,
        _analysis_spec(bondlen_names=["bondlen_A-B"]),
    )["distributions"]["bondlen_A-B"]

    assert entry["status"] == "unavailable"
    assert entry["n_available"] == 0
    assert "conflicting declared sample counts" in entry["invalid_payloads"][0]["reason"]


def test_analysis_fractional_coordination_cdf_preserves_physical_axis() -> None:
    boxes = [
        {
            "box": 1,
            "metrics": {},
            "distributions": {
                "coord": {
                    "coord_soft": {
                        "x": [0.0, 0.5, 1.0],
                        "cdf": [0.2, 0.7, 1.0],
                    }
                }
            },
        },
        {
            "box": 2,
            "metrics": {},
            "distributions": {
                "coord": {
                    "coord_soft": {
                        "x": [0.0, 1.0, 2.0],
                        "cdf": [0.2, 0.8, 1.0],
                    }
                }
            },
        },
    ]

    entry = _analysis_report(
        boxes, _analysis_spec(coord_names=["coord_soft"])
    )["distributions"]["coord_soft"]

    assert entry["x"] == [0.0, 0.5, 1.0, 2.0]
    assert entry["mean"] == pytest.approx([0.2, 0.45, 0.9, 1.0])
    assert entry["alignment"]["grid_source"] == "ensemble_common_explicit_coordination_grid"
    assert entry["alignment"]["coordination_support"] == "fractional_or_nonuniform"


def test_analysis_coordination_singleton_and_implicit_integer_legacy_are_valid() -> None:
    singleton_boxes = [
        {
            "box": idx + 1,
            "metrics": {},
            "distributions": {"coord": {"coord_zero": {"x": [0.0], "cdf": [1.0]}}},
        }
        for idx in range(2)
    ]
    singleton = _analysis_report(
        singleton_boxes, _analysis_spec(coord_names=["coord_zero"])
    )["distributions"]["coord_zero"]
    assert singleton["x"] == [0.0]
    assert singleton["mean"] == [1.0]
    assert singleton["alignment"]["grid_source"] == "ensemble_common_integer_grid"

    legacy_boxes = [
        {
            "box": idx + 1,
            "metrics": {},
            # Historical coordination payloads may omit x entirely; only this
            # absent-axis shape is interpreted as integer array-index support.
            "distributions": {"coord": {"coord_legacy": {"cdf": [0.25, 1.0]}}},
        }
        for idx in range(2)
    ]
    legacy = _analysis_report(
        legacy_boxes, _analysis_spec(coord_names=["coord_legacy"])
    )["distributions"]["coord_legacy"]
    assert legacy["x"] == [0.0, 1.0]
    assert legacy["axis"]["source"] == "implicit_integer_index_legacy"
    assert legacy["alignment"]["grid_source"] == "ensemble_common_integer_grid"


def test_analysis_ring_summary_is_descriptive_but_unassessed_when_box_missing() -> None:
    boxes = [
        {"box": 1, "metrics": {"ring_frac_3": 0.4}, "distributions": {}},
        {"box": 2, "metrics": {"ring_frac_3": 0.4}, "distributions": {}},
        {"box": 3, "metrics": {}, "distributions": {}},
    ]

    report = _analysis_report(boxes, _analysis_spec(ring_keys=["ring_frac_3"]))
    entry = report["distributions"]["ring"]

    assert entry["mean"] == [0.4]
    assert entry["available_subset_ci_within_tolerance"] is True
    assert entry["passed"] is None
    assert entry["convergence_assessed"] is False
    assert entry["convergence_status"] == "unassessed_incomplete_evidence"
    assert entry["blocking_boxes"][0]["box"] == 3
    assert entry["blocking_boxes"][0]["metrics"] == [
        {"key": "ring_frac_3", "reason": "missing_metric"}
    ]
    assert report["ensemble_cdfs"]["status"] == "incomplete"
