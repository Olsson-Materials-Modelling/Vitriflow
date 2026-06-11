import numpy as np
import pytest


pytest.importorskip("ase")


def test_compute_coordination_defects_flags_undercoordinated_sites():
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.analysis.structure import compute_coordination_defects
    from vitriflow.config import CoordinationMetricConfig, StructureMetricsConfig

    # neighbours cutoff expected
    cell = np.eye(3) * 20.0
    origin = np.zeros(3)
    types = np.array([1, 2, 2, 2], dtype=int)
    positions = np.array(
        [
            [10.0, 10.0, 10.0],  # si
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [10.0, 10.0, 11.0],
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

    defects = compute_coordination_defects(frame, metrics_cfg, cutoffs=cutoffs, type_to_species=type_to_species)
    assert "coord_Si-N" in defects
    assert defects["coord_Si-N"]["has_defect"] is True
    assert defects["coord_Si-N"]["defect_fraction"] == 1.0


def test_compute_coordination_defects_passes_perfect_coordination():
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.analysis.structure import compute_coordination_defects
    from vitriflow.config import CoordinationMetricConfig, StructureMetricsConfig

    # neighbours cutoff expected
    cell = np.eye(3) * 20.0
    origin = np.zeros(3)
    types = np.array([1, 2, 2, 2, 2], dtype=int)
    positions = np.array(
        [
            [10.0, 10.0, 10.0],  # si
            [11.0, 10.0, 10.0],
            [9.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [10.0, 10.0, 11.0],
        ],
        dtype=float,
    )
    ids = np.arange(1, 6, dtype=int)
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

    defects = compute_coordination_defects(frame, metrics_cfg, cutoffs=cutoffs, type_to_species=type_to_species)
    assert defects["coord_Si-N"]["has_defect"] is False
    assert defects["coord_Si-N"]["defect_fraction"] == 0.0
