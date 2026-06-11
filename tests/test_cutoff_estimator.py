import numpy as np
import pytest

pytest.importorskip("ase")

from vitriflow.analysis.dump import DumpFrame
from vitriflow.analysis.structure import estimate_pair_cutoffs
from vitriflow.config import AutoCutoffConfig


def _simple_cubic_frame(n: int = 3, a: float = 1.0) -> DumpFrame:
    # atoms simple box
    coords = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], dtype=float) * float(a)
    N = coords.shape[0]
    ids = np.arange(1, N + 1, dtype=int)
    types = np.ones(N, dtype=int)
    L = float(n) * float(a)
    cell = np.eye(3, dtype=float) * L
    origin = np.zeros(3, dtype=float)
    return DumpFrame(timestep=0, ids=ids, types=types, positions=coords, cell=cell, origin=origin)


def test_estimate_pair_cutoffs_simple_cubic():
    fr = _simple_cubic_frame(n=4, a=1.0)
    auto = AutoCutoffConfig(r_max=2.0, nbins=200, smooth=7, peak_search=(0.8, 1.2), min_search=(1.05, 1.6))
    cut = estimate_pair_cutoffs([fr], required_pairs=[(1, 1)], auto=auto, fixed_cutoffs={})
    c = float(cut[(1, 1)])
    assert 1.05 < c < 1.6
