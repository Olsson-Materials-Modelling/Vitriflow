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


def test_plot_voids_map_canonicalizes_raw_electron_dump(tmp_path: Path):
    from vitriflow.io.extxyz import read_extxyz_frames
    from vitriflow.lammps_units import length_from_angstrom_factor
    from vitriflow.plotting import plot_voids_map

    stage = tmp_path / "electron" / "relax"
    stage.mkdir(parents=True)
    native = float(length_from_angstrom_factor("electron"))
    atoms_a = np.asarray(
        [[1.0, 1.0, 1.0], [7.0, 1.0, 1.0], [1.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
        dtype=float,
    )
    rows = "".join(
        f"{idx} 1 {xyz[0] * native:.17g} {xyz[1] * native:.17g} {xyz[2] * native:.17g}\n"
        for idx, xyz in enumerate(atoms_a, start=1)
    )
    (stage / "relax.lammpstrj").write_text(
        "ITEM: TIMESTEP\n11\n"
        "ITEM: NUMBER OF ATOMS\n4\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        f"0 {8.0 * native:.17g}\n0 {8.0 * native:.17g}\n0 {8.0 * native:.17g}\n"
        "ITEM: ATOMS id type x y z\n"
        + rows
    )
    out = tmp_path / "electron_voids.png"
    combined = tmp_path / "electron_combined.extxyz"

    plot_voids_map(
        stage,
        out,
        n_samples=27,
        sampler="grid",
        seed=0,
        k_nearest=4,
        type_to_species=["Si"],
        top_n=8,
        units_style="electron",
        write_combined_extxyz=combined,
        dpi=80,
    )

    loaded = read_extxyz_frames(
        combined,
        last_n=1,
        type_to_species=["Si", "V"],
    )[0]
    np.testing.assert_allclose(loaded.cell, np.eye(3) * 8.0, rtol=1e-12)
    np.testing.assert_allclose(loaded.positions[:4], atoms_a, rtol=1e-12)
    assert out.exists() and out.stat().st_size > 0
