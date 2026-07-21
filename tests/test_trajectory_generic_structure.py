from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


class _FakeAtoms:
    def __init__(self):
        self.info = {"step": 17}

    def get_chemical_symbols(self):
        return ["Si", "Si"]

    def get_positions(self):
        return np.asarray([[0.0, 0.0, 0.0], [1.25, 1.25, 1.25]], dtype=float)

    def get_cell(self):
        return np.diag([5.0, 5.0, 5.0])

    def get_pbc(self):
        return np.asarray([True, False, True], dtype=bool)

    def get_volume(self):
        return 125.0

    def get_masses(self):
        return np.asarray([28.085, 28.085], dtype=float)



def test_read_last_frames_auto_reads_generic_structure_via_ase(monkeypatch, tmp_path: Path):
    from vitriflow.analysis.trajectory import read_last_frames_auto

    def _fake_read(path, index=None, format=None, style=None, specorder=None):
        assert Path(path).name == "CONTCAR"
        return _FakeAtoms()

    ase_mod = types.ModuleType("ase")
    io_mod = types.ModuleType("ase.io")
    io_mod.read = _fake_read
    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.io", io_mod)

    source = tmp_path / "CONTCAR"
    source.write_text("vasp snapshot\n")

    frames = read_last_frames_auto(source, 1, type_to_species=["Si"])

    assert len(frames) == 1
    fr = frames[0]
    assert fr.timestep == 17
    assert fr.n_atoms == 2
    assert np.allclose(fr.cell, np.diag([5.0, 5.0, 5.0]))
    assert fr.types.tolist() == [1, 1]
    assert fr.pbc == (True, False, True)
