from __future__ import annotations

import pytest

pytest.importorskip("ase")

from vitriflow.config import ConvergenceConfig
from vitriflow.workflows.production_common import check_production_convergence


def _box(density: float) -> dict:
    return {"density": float(density), "metrics": {}, "distributions": {}}


def test_check_production_convergence_density_only_ci_passes_for_identical_boxes():
    boxes = [_box(2.35), _box(2.35)]
    ok, report = check_production_convergence(boxes, {}, ConvergenceConfig(mode="ci"))

    assert ok is True
    assert report["scalars"]["density"]["passed"] is True
    assert report["groups"]["long"]["passed"] is True
    assert report["ci_converged"] is True


def test_check_production_convergence_density_only_stability_detects_shift():
    boxes = [_box(1.0), _box(1.0), _box(2.0), _box(2.0)]
    cfg = ConvergenceConfig(mode="stability", stability_bootstrap=0, stability_distance="wasserstein")
    ok, report = check_production_convergence(boxes, {}, cfg)

    assert ok is False
    assert report["stability"]["checks"]["density"]["passed"] is False
    assert report["stability_converged"] is False
