import numpy as np
import pytest


def test_mic_orthogonal_cell_rounding():
    from vitriflow.analysis.common import mic_displacements_and_distances

    cell = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]], dtype=float)
    # separated boundary displacement
    frac = np.array([[0.99, 0.5, 0.5], [0.01, 0.5, 0.5]], dtype=float)
    i = np.array([0], dtype=int)
    j = np.array([1], dtype=int)

    dr, dist = mic_displacements_and_distances(frac, cell, i, j)
    assert dr.shape == (1, 3)
    assert dist.shape == (1,)
    assert dist[0] == pytest.approx(0.2)


def test_mic_rotated_orthogonal_cell_rounding():
    from vitriflow.analysis.common import mic_displacements_and_distances

    # orthogonal rotated rotation
    s2 = float(np.sqrt(2.0))
    a = np.array([10.0 / s2, 10.0 / s2, 0.0])
    b = np.array([-20.0 / s2, 20.0 / s2, 0.0])
    c = np.array([0.0, 0.0, 30.0])
    cell = np.vstack([a, b, c]).astype(float)

    frac = np.array([[0.99, 0.5, 0.5], [0.01, 0.5, 0.5]], dtype=float)
    i = np.array([0], dtype=int)
    j = np.array([1], dtype=int)

    dr, dist = mic_displacements_and_distances(frac, cell, i, j)
    assert dist[0] == pytest.approx(0.2)


def test_mic_triclinic_uses_ase_if_available():
    pytest.importorskip("ase")
    from vitriflow.analysis.common import mic_displacements_and_distances

    # simple triclinic cell
    cell = np.array([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [0.0, 0.0, 8.0]], dtype=float)
    frac = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.1]], dtype=float)
    i = np.array([0], dtype=int)
    j = np.array([1], dtype=int)

    dr, dist = mic_displacements_and_distances(frac, cell, i, j)

    # brute force translation
    df = frac[j] - frac[i]
    dmin = float("inf")
    for nx in range(-2, 3):
        for ny in range(-2, 3):
            for nz in range(-2, 3):
                dr0 = (df - np.array([nx, ny, nz], dtype=float)) @ cell
                d0 = float(np.linalg.norm(dr0))
                dmin = min(dmin, d0)

    assert dist[0] == pytest.approx(dmin, rel=1e-12, abs=1e-10)
