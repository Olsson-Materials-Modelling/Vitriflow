import numpy as np
import pytest


pytest.importorskip("ase")


def test_compute_coordination_defect_details_returns_under_over_lists():
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.analysis.structure import compute_coordination_defect_details
    from vitriflow.config import CoordinationMetricConfig, StructureMetricsConfig

    # neighbours cutoff undercoordinated
    cell = np.eye(3) * 20.0
    origin = np.zeros(3)
    types = np.array([1, 2, 2, 2], dtype=int)
    positions = np.array(
        [
            [10.0, 10.0, 10.0],  # si id
            [11.0, 10.0, 10.0],  # n id
            [10.0, 11.0, 10.0],  # n id
            [10.0, 10.0, 11.0],  # n id
        ],
        dtype=float,
    )
    ids = np.arange(1, 5, dtype=int)
    frame = DumpFrame(timestep=0, cell=cell, origin=origin, types=types, positions=positions, ids=ids)

    metrics_cfg = StructureMetricsConfig(
        enabled=True,
        coordinations=[
            CoordinationMetricConfig(
                central="Si",
                neighbor="N",
                expected=4,
                defect_frac_tol=0.0,
            )
        ],
    )

    cutoffs = {(1, 2): 2.0}
    type_to_species = ["Si", "N"]

    det = compute_coordination_defect_details(frame, metrics_cfg, cutoffs=cutoffs, type_to_species=type_to_species)
    key = "coord_Si-N"
    assert key in det
    assert det[key]["n_central"] == 1
    assert det[key]["n_defective"] == 1
    assert det[key]["under_ids"] == [1]
    assert det[key]["over_ids"] == []
    # contain defective neighbours
    assert set(det[key]["shell_ids"]) == {1, 2, 3, 4}
