import json
from pathlib import Path

import numpy as np
import pytest

from vitriflow.plotting import _align_plot_cdf_payloads, plot_production_results


@pytest.mark.parametrize(
    "payload",
    [
        {"x": [0.0, 2.0, 1.0], "cdf": [0.0, 0.5, 1.0]},
        {"x": [0.0, 1.0, 1.0], "cdf": [0.0, 0.5, 1.0]},
        {"x": [0.0, 1.0, 2.0], "cdf": [0.0, 0.8, 0.7]},
        {"x": [0.0, 1.0, 2.0], "cdf": [0.0, 0.5, 1.1]},
        {"x": [0.0, float("nan"), 2.0], "cdf": [0.0, 0.5, 1.0]},
    ],
)
def test_plot_cdf_alignment_rejects_malformed_scientific_input(payload) -> None:
    with pytest.raises(ValueError):
        _align_plot_cdf_payloads([payload])


def test_plot_cdf_alignment_uses_right_continuous_physical_union_support() -> None:
    x, matrix, meta = _align_plot_cdf_payloads(
        [
            {"x": [0.0, 0.5, 1.0], "cdf": [0.2, 0.7, 1.0]},
            {"x": [0.0, 1.0, 2.0], "cdf": [0.2, 0.8, 1.0]},
        ]
    )

    assert x.tolist() == [0.0, 0.5, 1.0, 2.0]
    assert matrix[0].tolist() == pytest.approx([0.2, 0.7, 1.0, 1.0])
    assert matrix[1].tolist() == pytest.approx([0.2, 0.2, 0.8, 1.0])
    assert meta["grid_alignment_method"] == "right_continuous_cdf_evaluation"
    assert np.all(np.diff(x) > 0.0)


def test_plot_coord_cdf_legacy_axis_inference_is_absent_key_only() -> None:
    x, matrix, meta = _align_plot_cdf_payloads(
        [{"cdf": [0.25, 1.0]}, {"cdf": [0.2, 1.0]}],
        allow_implicit_integer_axis=True,
    )
    assert x.tolist() == [0.0, 1.0]
    assert np.allclose(matrix, np.asarray([[0.25, 1.0], [0.2, 1.0]], dtype=float))
    assert meta["axis_source"] == "implicit_integer_index_legacy"

    # A present-but-empty x field is malformed current-format input and must
    # not be reinterpreted as the legacy integer index convention.
    with pytest.raises(ValueError):
        _align_plot_cdf_payloads(
            [{"x": [], "cdf": [0.25, 1.0]}],
            allow_implicit_integer_axis=True,
        )


def test_plot_cdf_alignment_accepts_valid_singleton_distribution() -> None:
    x, matrix, _meta = _align_plot_cdf_payloads(
        [{"x": [0.0], "cdf": [1.0]}, {"x": [0.0], "cdf": [1.0]}]
    )
    assert x.tolist() == [0.0]
    assert matrix.tolist() == [[1.0], [1.0]]


def test_plot_production_regrids_box_specific_cdf_grids(tmp_path: Path) -> None:
    payload = {
        "schema": "vitriflow.analysis_results.v2",
        "units": {},
        "converged": False,
        "n_boxes": 2,
        "n_boxes_accepted": 2,
        "n_boxes_rejected": 0,
        "n_boxes_total": 2,
        "convergence_spec": {
            "bondlen_names": ["bondlen_Si-N"],
            "angle_names": [],
            "coord_names": [],
            "ring_keys": [],
            "gr_labels": [],
            "sq_labels": [],
            "void_names": [],
            "ring_has_mean_size": False,
        },
        "convergence": {
            "schema": "vitriflow.analysis_descriptor_convergence.v1",
            "status": "ok",
            "familywise": {"alpha_per_test": 0.05, "bounded_ci_method": "t"},
            "groups": {"short": {"passed": False}, "medium": {}, "long": {}},
            "scalars": {
                "density": {
                    "group": "long",
                    "mean": 1.05,
                    "std": 0.07,
                    "stderr": 0.05,
                    "ci_halfwidth": 0.1,
                    "abs_tol": 0.1,
                    "rel_tol": 0.0,
                    "passed": False,
                }
            },
            "distributions": {
                "bondlen_Si-N": {
                    "group": "short",
                    "kind": "bondlen_cdf",
                    "abs_tol": 0.02,
                    "rel_tol": 0.0,
                    "x": [0.0, 1.0, 2.0, 3.0, 4.0],
                    "mean": [0.0, 0.2, 0.55, 0.95, 1.0],
                    "ci_halfwidth": [0.0, 0.1, 0.1, 0.05, 0.0],
                    "passed": False,
                }
            },
            "stability": {"enabled": False},
        },
        "boxes": [
            {
                "box": 1,
                "box_id": 1,
                "density": 1.0,
                "metrics": {},
                "distributions": {
                    "bondlen": {"bondlen_Si-N": {"x": [0.0, 1.0, 2.0], "cdf": [0.0, 0.5, 1.0]}},
                    "angle": {},
                    "coord": {},
                    "void": {},
                },
            },
            {
                "box": 2,
                "box_id": 2,
                "density": 1.1,
                "metrics": {},
                "distributions": {
                    "bondlen": {"bondlen_Si-N": {"x": [0.0, 1.5, 3.0, 4.0], "cdf": [0.0, 0.4, 0.9, 1.0]}},
                    "angle": {},
                    "coord": {},
                    "void": {},
                },
            },
        ],
    }
    src = tmp_path / "analysis_results.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "plots"
    plot_production_results(src, out, dpi=80, max_pages=3)
    assert (out / "03_Bond_length:_Si-N.png").exists()
