import numpy as np
import pytest


pytest.importorskip("ase")


@pytest.mark.parametrize(
    ("cp2k_version", "expects_keyword"),
    [((2023, 2, 0), False), ((2024, 1, 0), True)],
)
def test_render_cp2k_cell_opt_includes_keep_angles_stress_and_version_policy(
    cp2k_version, expects_keyword
):
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
        cp2k_version=cp2k_version,
    )

    assert "RUN_TYPE CELL_OPT" in txt
    assert "&CELL_OPT" in txt
    assert "TYPE DIRECT_CELL_OPT" in txt
    assert "KEEP_ANGLES TRUE" in txt
    assert "EXTERNAL_PRESSURE [bar]" in txt
    assert "STRESS_TENSOR ANALYTICAL" in txt
    assert "IGNORE_CONVERGENCE_FAILURE F" not in txt
    assert ("IGNORE_CONVERGENCE_FAILURE T" in txt) is expects_keyword
    assert "&EACH" in txt and "CELL_OPT 1" in txt
    assert "SCF_GUESS ATOMIC" in txt
    assert "SCF_GUESS RESTART" not in txt


def test_cell_opt_geometry_restart_never_enables_implicit_wfn_restart():
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
        scf_guess="RESTART",
        kind_settings={
            "H": Cp2kKindConfig(
                basis_set="DZVP-MOLOPT-SR-GTH",
                potential="GTH-PBE",
            )
        },
    )

    text = render_cp2k_cell_opt_input(
        atoms=atoms,
        cfg=cfg,
        project="dft_opt",
        restart_file="dft_opt-10.restart",
        cp2k_version=(2024, 2),
    )

    assert "RESTART_FILE_NAME dft_opt-10.restart" in text
    assert "SCF_GUESS ATOMIC" in text
    assert "SCF_GUESS RESTART" not in text
