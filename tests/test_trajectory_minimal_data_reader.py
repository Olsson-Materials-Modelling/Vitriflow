from __future__ import annotations

from pathlib import Path

import numpy as np


def test_minimal_data_reader_preserves_absolute_origin_and_triclinic_cell(tmp_path: Path):
    from vitriflow.analysis.trajectory import _read_lammps_data_dumpframe_minimal

    source = tmp_path / "triclinic.data"
    source.write_text(
        """LAMMPS data

2 atoms
1 atom types

5 15 xlo xhi
-2 8 ylo yhi
3 13 zlo zhi
1.5 -0.5 0.75 xy xz yz

Atoms # atomic

2 1 7.0 0.0 5.0
1 1 6.0 -1.0 4.0
"""
    )

    frame = _read_lammps_data_dumpframe_minimal(source)

    np.testing.assert_allclose(frame.origin, [5.0, -2.0, 3.0])
    np.testing.assert_allclose(
        frame.cell,
        [[10.0, 0.0, 0.0], [1.5, 10.0, 0.0], [-0.5, 0.75, 10.0]],
    )
    np.testing.assert_allclose(frame.positions, [[6.0, -1.0, 4.0], [7.0, 0.0, 5.0]])
    frac = (frame.positions - frame.origin) @ np.linalg.inv(frame.cell)
    np.testing.assert_allclose(frac[0], [0.091125, 0.0925, 0.1])
