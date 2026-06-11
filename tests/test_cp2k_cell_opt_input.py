import numpy as np
import pytest


pytest.importorskip("ase")


def test_render_cp2k_cell_opt_includes_keep_angles_and_stress_tensor():
    from ase import Atoms

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig
    from vitriflow.cp2k_driver import render_cp2k_cell_opt_input

    atoms = Atoms(
        symbols=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )

    cfg = Cp2kConfig(
        exec="cp2k",
        kind_settings={
            "H": Cp2kKindConfig(basis_set="DZVP-MOLOPT-SR-GTH", potential="GTH-PBE")
        },
    )

    txt = render_cp2k_cell_opt_input(
        atoms=atoms,
        cfg=cfg,
        project="test_cell_opt",
        optimizer="LBFGS",
        max_iter=25,
        keep_angles=True,
        external_pressure_bar=0.0,
        traj_every=1,
        traj_file="traj.dcd",
        print_level="LOW",
    )

    assert "RUN_TYPE CELL_OPT" in txt
    assert "&CELL_OPT" in txt
    assert "TYPE DIRECT_CELL_OPT" in txt
    assert "KEEP_ANGLES TRUE" in txt
    assert "EXTERNAL_PRESSURE [bar]" in txt
    assert "STRESS_TENSOR ANALYTICAL" in txt
    assert "&EACH" in txt and "CELL_OPT 1" in txt
