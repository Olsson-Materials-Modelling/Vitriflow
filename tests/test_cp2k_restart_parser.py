from __future__ import annotations

import numpy as np
import pytest

from vitriflow.analysis.trajectory import read_last_frames_auto


def test_read_strict_cp2k_text_restart_vectors(tmp_path):
    restart = tmp_path / "box_001_hse06-1.restart"
    restart.write_text(
        """
&GLOBAL
  PROJECT box_001_hse06
&END GLOBAL
&FORCE_EVAL
  &SUBSYS
    &CELL
      A 10.0 0.0 0.0
      B 0.0 11.0 0.0
      C 0.0 0.0 12.0
      PERIODIC XYZ
    &END CELL
    &COORD
      Si 0.0 0.0 0.0
      N  1.8 0.0 0.0
      N  0.0 1.8 0.0
    &END COORD
  &END SUBSYS
&END FORCE_EVAL
""".strip()
    )

    frames = read_last_frames_auto(restart, 1, type_to_species=["Si", "N"])

    assert len(frames) == 1
    frame = frames[0]
    assert frame.n_atoms == 3
    assert frame.types.tolist() == [1, 2, 2]
    np.testing.assert_allclose(frame.cell, np.diag([10.0, 11.0, 12.0]))
    np.testing.assert_allclose(frame.positions[1], [1.8, 0.0, 0.0])


def test_read_cp2k_text_restart_scaled_positions(tmp_path):
    restart = tmp_path / "box_002_pbe-1.restart"
    restart.write_text(
        """
&FORCE_EVAL
  &SUBSYS
    &CELL
      ABC 8.0 10.0 12.0
      ALPHA_BETA_GAMMA 90.0 90.0 90.0
    &END CELL
    &COORD SCALED
      Si 0.25 0.50 0.75
      N  0.50 0.25 0.00
    &END COORD
  &END SUBSYS
&END FORCE_EVAL
""".strip()
    )

    frame = read_last_frames_auto(restart, 1, type_to_species=["Si", "N"])[0]

    np.testing.assert_allclose(frame.cell, np.diag([8.0, 10.0, 12.0]), atol=1e-12)
    np.testing.assert_allclose(frame.positions[0], [2.0, 5.0, 9.0], atol=1e-12)
    np.testing.assert_allclose(frame.positions[1], [4.0, 2.5, 0.0], atol=1e-12)


def test_lammps_data_with_restart_suffix_uses_lammps_reader(tmp_path):
    restart = tmp_path / "box_010-1.restart"
    restart.write_text(
        """LAMMPS data file via write_data, version test

2 atoms
2 atom types

0.0 5.0 xlo xhi
0.0 5.0 ylo yhi
0.0 5.0 zlo zhi

Masses

1 28.0855
2 14.0067

Atoms # atomic

1 1 0.0 0.0 0.0
2 2 1.8 0.0 0.0
"""
    )

    frame = read_last_frames_auto(restart, 1, type_to_species=["Si", "N"])[0]

    assert frame.n_atoms == 2
    assert frame.types.tolist() == [1, 2]
    np.testing.assert_allclose(frame.cell, np.diag([5.0, 5.0, 5.0]), atol=1e-12)
    np.testing.assert_allclose(frame.positions[1], [1.8, 0.0, 0.0], atol=1e-12)


def test_non_final_restart_name_is_blocked_before_parsing(tmp_path):
    restart = tmp_path / "box_001_hse061.restart"
    restart.write_text("this is not a final CP2K restart and should not be ASE-probed\n")

    with pytest.raises(ValueError, match=r"automatic frame loading from \.restart files is disabled"):
        read_last_frames_auto(restart, 1, type_to_species=["Si", "N"])


def test_strict_named_non_cp2k_restart_is_not_sent_to_ase_generic_reader(tmp_path):
    restart = tmp_path / "box_001_hse06-1.restart"
    restart.write_text("this is not a cp2k restart and should not be ASE-probed\n")

    with pytest.raises(ValueError, match="strict final restart could not be parsed"):
        read_last_frames_auto(restart, 1, type_to_species=["Si", "N"])
