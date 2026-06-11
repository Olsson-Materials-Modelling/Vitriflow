from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")


def _simple_frame(seed: int = 0, *, n_atoms: int = 64, L: float = 20.0):
    from vitriflow.analysis.dump import DumpFrame

    rng = np.random.default_rng(int(seed))
    pos = rng.random((int(n_atoms), 3)) * float(L)
    ids = np.arange(1, int(n_atoms) + 1, dtype=int)
    types = np.ones(int(n_atoms), dtype=int)
    cell = np.eye(3, dtype=float) * float(L)
    origin = np.zeros(3, dtype=float)
    return DumpFrame(timestep=0, ids=ids, types=types, positions=pos, cell=cell, origin=origin)


def test_plot_voids_map_writes_outputs(tmp_path: Path):
    from vitriflow.io.extxyz import write_extxyz_frames
    from vitriflow.plotting import plot_voids_map

    stage = tmp_path / "box" / "relax"
    stage.mkdir(parents=True)

    fr = _simple_frame(seed=1, n_atoms=80, L=25.0)
    write_extxyz_frames(stage / "traj.extxyz", [fr], type_to_species=["X"])

    out = tmp_path / "voids.pdf"
    out_void = tmp_path / "voids_points.extxyz"
    out_comb = tmp_path / "atoms_plus_voids.extxyz"

    plot_voids_map(
        stage,
        out,
        n_samples=2048,
        sampler="sobol",
        seed=0,
        k_nearest=8,
        type_to_species=["X"],
        radii_by_species={},
        default_radius=0.0,
        min_clearance=None,
        top_n=250,
        show_atoms=True,
        units_style="metal",
        write_void_extxyz=out_void,
        write_combined_extxyz=out_comb,
    )

    assert out.exists()
    assert out.stat().st_size > 1_000

    assert out_void.exists()
    txt = out_void.read_text()
    assert "Properties" in txt and "clearance" in txt

    assert out_comb.exists()
    txt2 = out_comb.read_text()
    assert "is_void" in txt2 and "clearance" in txt2
