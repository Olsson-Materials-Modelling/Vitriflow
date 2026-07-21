from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vitriflow.analysis.dump import DumpFrame
from vitriflow.analysis.trajectory import read_frames_auto, read_last_frames_auto
from vitriflow.io.extxyz import write_extxyz_single
from vitriflow.lammps_units import length_from_angstrom_factor
from vitriflow.workflows.output_analysis import analysis_context_from_standalone_config


def _write_native_dump(path: Path, *, units_style: str) -> None:
    native = float(length_from_angstrom_factor(units_style))
    path.write_text(
        "ITEM: TIMESTEP\n7\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        f"0 {10.0 * native:.17g}\n"
        f"0 {12.0 * native:.17g}\n"
        f"0 {14.0 * native:.17g}\n"
        "ITEM: ATOMS id type x y z\n"
        f"2 2 {4.0 * native:.17g} {5.0 * native:.17g} {6.0 * native:.17g}\n"
        f"1 1 {1.0 * native:.17g} {2.0 * native:.17g} {3.0 * native:.17g}\n"
    )


@pytest.mark.parametrize("units_style", ["real", "electron"])
def test_raw_dump_source_is_canonicalized_from_declared_units(
    tmp_path: Path,
    units_style: str,
):
    source = tmp_path / f"traj-{units_style}.lammpstrj"
    _write_native_dump(source, units_style=units_style)

    frame = read_last_frames_auto(source, 1, units_style=units_style)[0]

    assert frame.timestep == 7
    assert frame.ids.tolist() == [1, 2]
    assert frame.types.tolist() == [1, 2]
    np.testing.assert_allclose(frame.cell, np.diag([10.0, 12.0, 14.0]), rtol=1e-13)
    np.testing.assert_allclose(frame.positions, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], rtol=1e-13)


def test_raw_lammps_source_requires_units_but_canonical_extxyz_does_not(tmp_path: Path):
    raw = tmp_path / "traj.lammpstrj"
    _write_native_dump(raw, units_style="electron")
    with pytest.raises(ValueError, match="units_style is required"):
        read_last_frames_auto(raw, 1, units_style=None)

    canonical = tmp_path / "final.extxyz"
    frame = DumpFrame(
        timestep=3,
        ids=np.asarray([1], dtype=int),
        types=np.asarray([1], dtype=int),
        positions=np.asarray([[1.25, 2.5, 3.75]], dtype=float),
        cell=np.diag([8.0, 9.0, 10.0]),
        origin=np.zeros(3),
    )
    write_extxyz_single(canonical, frame, type_to_species=["Si"])
    loaded = read_last_frames_auto(
        canonical,
        1,
        type_to_species=["Si"],
        units_style=None,
    )[0]
    np.testing.assert_allclose(loaded.cell, frame.cell)
    np.testing.assert_allclose(loaded.positions, frame.positions)


def test_strict_raw_data_reader_preserves_charge_layout_and_electron_geometry(tmp_path: Path):
    native = float(length_from_angstrom_factor("electron"))
    source = tmp_path / "charged.data"
    source.write_text(
        "LAMMPS data\n\n"
        "2 atoms\n2 atom types\n\n"
        f"{5.0 * native:.17g} {15.0 * native:.17g} xlo xhi\n"
        f"{-2.0 * native:.17g} {10.0 * native:.17g} ylo yhi\n"
        f"{3.0 * native:.17g} {17.0 * native:.17g} zlo zhi\n\n"
        "Atoms # charge\n\n"
        f"2 2 -0.5 {9.0 * native:.17g} {3.0 * native:.17g} {9.0 * native:.17g}\n"
        f"1 1 0.5 {6.0 * native:.17g} {-1.0 * native:.17g} {4.0 * native:.17g}\n"
    )

    frame = read_last_frames_auto(
        source,
        1,
        atom_style="charge",
        units_style="electron",
    )[0]

    assert frame.ids.tolist() == [1, 2]
    assert frame.types.tolist() == [1, 2]
    np.testing.assert_allclose(frame.origin, [5.0, -2.0, 3.0], rtol=1e-13)
    np.testing.assert_allclose(frame.cell, np.diag([10.0, 12.0, 14.0]), rtol=1e-13)
    np.testing.assert_allclose(frame.positions, [[6.0, -1.0, 4.0], [9.0, 3.0, 9.0]], rtol=1e-13)


@pytest.mark.parametrize(
    "atom_rows, message",
    [
        ("1.5 1 1 2 3\n2 1 4 5 6\n", "positive integer"),
        ("1 1 1 2 3\n1 1 4 5 6\n", "unique positive integers"),
        ("1 1 1 2 3\n", "Atom-count mismatch"),
        ("1 1 nan 2 3\n2 1 4 5 6\n", "Non-finite atom position"),
    ],
)
def test_raw_data_malformed_atom_evidence_is_rejected(
    tmp_path: Path,
    atom_rows: str,
    message: str,
):
    source = tmp_path / "malformed.data"
    source.write_text(
        "LAMMPS data\n\n"
        "2 atoms\n1 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Atoms # atomic\n\n"
        + atom_rows
    )

    with pytest.raises(ValueError, match=message):
        read_frames_auto(source, units_style="real")


def test_standalone_analysis_source_units_are_explicit_not_implicitly_metal():
    base = {
        "type_to_species": ["Si"],
        "metrics": {"enabled": False},
        "production": {"min_boxes": 1, "batch_boxes": 1},
    }
    unresolved = analysis_context_from_standalone_config({"analysis": dict(base)})
    electron = analysis_context_from_standalone_config(
        {"analysis": {**base, "units_style": "electron"}}
    )

    assert unresolved.lammps_units_style is None
    assert electron.lammps_units_style == "electron"


class _AseFrameWithStep:
    def __init__(self, step):
        self.info = {"Step": step}

    def get_chemical_symbols(self):
        return ["Si"]

    def get_positions(self):
        return np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    def get_cell(self):
        return np.eye(3) * 8.0

    def get_pbc(self):
        return np.asarray([True, True, True], dtype=bool)


@pytest.mark.parametrize(
    "bad_step",
    [
        1.25,
        float("nan"),
        float("inf"),
        True,
        "1.5",
        -1,
        str(int(np.iinfo(np.intp).max) + 1),
    ],
)
def test_generic_ase_step_metadata_must_be_a_finite_exact_integer(
    monkeypatch,
    tmp_path: Path,
    bad_step,
):
    ase_mod = types.ModuleType("ase")
    io_mod = types.ModuleType("ase.io")
    io_mod.read = lambda *args, **kwargs: _AseFrameWithStep(bad_step)
    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.io", io_mod)
    source = tmp_path / "CONTCAR"
    source.write_text("mock structure\n")

    with pytest.raises(ValueError, match="finite exact integer"):
        read_last_frames_auto(source, 1, type_to_species=["Si"], units_style=None)
