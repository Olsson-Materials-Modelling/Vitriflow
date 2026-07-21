from __future__ import annotations

import json
from pathlib import Path

from vitriflow.plotting import plot_production_results


def test_plot_production_hydrates_missing_convergence_distribution_from_ensemble_cdf_sidecar(tmp_path: Path) -> None:
    payload = {
        "schema": "vitriflow.analysis_results.v2",
        "status": "ok",
        "n_boxes": 2,
        "n_boxes_accepted": 2,
        "n_boxes_rejected": 0,
        "n_boxes_total": 2,
        "converged": False,
        "convergence_spec": {
            "bondlen_names": ["bondlen_Si-N"],
            "angle_names": [],
            "coord_names": [],
            "ring_keys": [],
            "ring_has_mean_size": False,
            "gr_labels": [],
            "sq_labels": [],
            "void_names": [],
        },
        "convergence": {
            "schema": "vitriflow.analysis_descriptor_convergence.v1",
            "status": "advisory",
            "familywise": {"alpha_per_test": 0.05, "bounded_ci_method": "t"},
            "scalars": {
                "density": {"group": "long", "mean": 3.0, "abs_tol": 0.1, "rel_tol": 0.0, "passed": True}
            },
            "distributions": {},
            "groups": {"short": {}, "medium": {}, "long": {}},
        },
        "ensemble_cdfs": {"path": "ensemble_cdfs.json", "exists": True},
        # Deliberately no per-box distributions. The sidecar is the authoritative
        # ensemble-level CDF object for this plotting test.
        "boxes": [
            {"box": 1, "box_id": 1, "density": 3.0, "metrics": {}},
            {"box": 2, "box_id": 2, "density": 3.1, "metrics": {}},
        ],
    }
    sidecar = {
        "schema": "vitriflow.analysis_ensemble_cdfs.v1",
        "status": "ok",
        "n_boxes": 2,
        "normalization": "per_box_unweighted_mean_of_box_curves",
        "distributions": {
            "bondlen_Si-N": {
                "name": "bondlen_Si-N",
                "group": "short",
                "kind": "bondlen_cdf",
                "status": "ok",
                "x": [1.5, 1.7, 1.9, 2.1],
                "mean": [0.0, 0.3, 0.9, 1.0],
                "cdf": [0.0, 0.3, 0.9, 1.0],
                "stderr": [0.0, 0.05, 0.05, 0.0],
                "ci_halfwidth": [0.0, 0.1, 0.1, 0.0],
                "abs_tol": 0.1,
                "rel_tol": 0.0,
                "passed": True,
            }
        },
    }
    src = tmp_path / "analysis_results.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "ensemble_cdfs.json").write_text(json.dumps(sidecar), encoding="utf-8")

    out = tmp_path / "plots"
    plot_production_results(src, out, dpi=80, max_pages=3)

    assert (out / "03_Bond_length:_Si-N.png").exists()


def test_python_module_entrypoint_imports() -> None:
    import vitriflow.__main__ as main_module

    assert hasattr(main_module, "main")
