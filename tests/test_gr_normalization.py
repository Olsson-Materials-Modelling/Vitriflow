import numpy as np
import pytest

pytest.importorskip("ase")

from vitriflow.analysis.dump import DumpFrame
from vitriflow.analysis.gr import compute_gr


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
