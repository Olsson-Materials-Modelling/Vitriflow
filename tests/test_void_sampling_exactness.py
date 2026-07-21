from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.spatial import cKDTree

from vitriflow.analysis.voids import _clearance_from_tree_exact, _sample_frac_points


def test_grid_sampler_returns_exact_requested_count():
    points = _sample_frac_points(10, sampler="grid", seed=0)
    assert points.shape == (10, 3)
    assert np.all(points >= 0.0)
    assert np.all(points < 1.0)


def test_radii_aware_clearance_expands_beyond_nearest_center():
    centers = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.asarray([0.0, 1.9])
    value = _clearance_from_tree_exact(
        cKDTree(centers),
        np.asarray([[0.1, 0.0, 0.0]]),
        radii,
        initial_k=1,
    )
    # The nearest centre gives 0.1; the farther, larger sphere gives 0.0.
    assert value[0] == pytest.approx(0.0)


def test_skew_cell_clearance_matches_ase_minimum_image():
    pytest.importorskip("ase")
    from ase.geometry import find_mic

    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.analysis.voids import sample_void_clearance_points

    cell = np.asarray(
        [[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=float,
    )
    frame = DumpFrame(
        timestep=0,
        ids=np.asarray([1], dtype=int),
        types=np.asarray([1], dtype=int),
        positions=np.asarray([[0.2, 0.2, 0.2]], dtype=float),
        cell=cell,
        origin=np.zeros(3),
    )
    points, clearance = sample_void_clearance_points(
        frame,
        n_samples=10,
        sampler="grid",
        k_nearest=1,
        default_radius=0.3,
    )
    displacement = frame.positions[0][None, :] - points
    _mic, distances = find_mic(displacement, cell, pbc=True)
    expected = np.maximum(np.asarray(distances, dtype=float) - 0.3, 0.0)
    assert clearance == pytest.approx(expected)
