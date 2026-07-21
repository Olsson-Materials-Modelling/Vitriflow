import numpy as np
import pytest

pytest.importorskip("ase")

from vitriflow.analysis.dump import DumpFrame
from vitriflow.analysis.gr import compute_gr, _shortest_lattice_translation


def _random_frame(seed: int, *, n_atoms: int = 600, L: float = 20.0) -> DumpFrame:
    rng = np.random.default_rng(seed)
    pos = rng.random((n_atoms, 3)) * float(L)
    ids = np.arange(1, n_atoms + 1, dtype=int)
    types = np.ones(n_atoms, dtype=int)
    cell = np.eye(3, dtype=float) * float(L)
    origin = np.zeros(3, dtype=float)
    return DumpFrame(timestep=int(seed), ids=ids, types=types, positions=pos, cell=cell, origin=origin)


def test_compute_gr_ideal_gas_normalization():
    frames = [_random_frame(i) for i in range(8)]
    r, g, _ = compute_gr(frames, r_max=5.0, nbins=120)

    m = (r > 1.0) & (r < 4.5)
    assert np.all(np.isfinite(g[m]))

    mean_g = float(np.mean(g[m]))
    assert abs(mean_g - 1.0) < 0.25
    assert float(np.min(g[m])) >= 0.0


def test_pair_population_with_no_events_is_zero_not_missing():
    frame = DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 2], dtype=int),
        positions=np.asarray([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]], dtype=float),
        cell=np.eye(3, dtype=float) * 10.0,
        origin=np.zeros(3, dtype=float),
    )
    _r, g, _ = compute_gr(
        [frame], r_max=1.0, nbins=10, pair=(1, 2)
    )
    assert np.all(np.isfinite(g))
    assert np.all(g == 0.0)


def test_triclinic_rdf_radius_uses_shortest_lattice_translation():
    cell = np.asarray(
        [[10.0, 0.0, 0.0], [9.5, 1.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=float,
    )
    frame = DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 1], dtype=int),
        positions=np.asarray([[0.1, 0.1, 0.1], [4.0, 4.0, 4.0]], dtype=float),
        cell=cell,
        origin=np.zeros(3, dtype=float),
    )
    r, _g, _ = compute_gr([frame], r_max=3.0, nbins=10)
    shortest = np.linalg.norm(cell[0] - cell[1])
    assert float(r[-1]) < 0.5 * float(shortest)


def test_shortest_translation_ignores_nonperiodic_lattice_axes():
    cell = np.diag([10.0, 1.0e-6, 12.0])
    assert _shortest_lattice_translation(
        cell, pbc=(True, False, True)
    ) == pytest.approx(10.0)
    assert np.isinf(
        _shortest_lattice_translation(cell, pbc=(False, False, False))
    )


def test_rdf_excludes_duplicate_image_boundary_at_half_box():
    frame = DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 1], dtype=int),
        positions=np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
        cell=np.eye(3) * 10.0,
        origin=np.zeros(3),
    )
    r, g, _ = compute_gr([frame], r_max=5.0, nbins=10)
    assert float(r[-1]) < 5.0
    assert np.all(g == 0.0)
