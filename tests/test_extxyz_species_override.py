from pathlib import Path

import numpy as np


def test_write_extxyz_single_with_species_overrides_species_column(tmp_path: Path):
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.io.extxyz import write_extxyz_single_with_species

    cell = np.eye(3) * 10.0
    origin = np.zeros(3)
    ids = np.array([1, 2], dtype=int)
    types = np.array([1, 2], dtype=int)
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    fr = DumpFrame(timestep=7, ids=ids, types=types, positions=positions, cell=cell, origin=origin)

    out = tmp_path / "marked.extxyz"
    write_extxyz_single_with_species(out, fr, ["Sm", "O"], wrap=True)
    txt = out.read_text().splitlines()

    assert txt[0].strip() == "2"
    assert "Properties=species:S:1:pos:R:3:type:I:1:id:I:1" in txt[1]
    assert txt[2].split()[0] == "Sm"
    assert txt[3].split()[0] == "O"



def test_write_extxyz_frames_accepts_legacy_species_alias(tmp_path: Path):
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.io.extxyz import write_extxyz_frames

    cell = np.eye(3) * 10.0
    origin = np.zeros(3)
    ids = np.array([1, 2], dtype=int)
    types = np.array([1, 2], dtype=int)
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    fr = DumpFrame(timestep=7, ids=ids, types=types, positions=positions, cell=cell, origin=origin)

    out = tmp_path / "legacy.extxyz"
    write_extxyz_frames(out, [fr], species=["Si", "N"], wrap=True)
    txt = out.read_text().splitlines()

    assert txt[2].split()[0] == "Si"
    assert txt[3].split()[0] == "N"
