import pytest


def test_dft_opt_requires_cp2k_block():
    """Dft opt requires."""
    from vitriflow.config import RunConfig

    cfg = {
        "engine": "lammps",
        "structure": {"generate": {"method": "random", "formula": "H2"}},
        "kim": {"model": "DUMMY", "interactions": ["H"]},
        "autotune": {
            "production": {
                "enabled": True,
                "dft_opt": {"enabled": True},
            }
        },
        # rest fill
    }

    with pytest.raises(ValueError, match=r"requires a 'cp2k:' block"):
        RunConfig.model_validate(cfg)


def test_dft_opt_requires_convergence_check():
    """Dft opt requires."""
    from vitriflow.config import RunConfig

    cfg = {
        "engine": "lammps",
        "structure": {"generate": {"method": "random", "formula": "H2"}},
        "kim": {"model": "DUMMY", "interactions": ["H"]},
        "cp2k": {
            "exec": "cp2k",
            "kind_settings": {
                "H": {"basis_set": "DZVP-MOLOPT-SR-GTH", "potential": "GTH-PBE"}
            },
        },
        "autotune": {
            "production": {
                "enabled": True,
                "check_convergence": False,
                "dft_opt": {"enabled": True},
            }
        },
    }

    with pytest.raises(ValueError, match=r"check_convergence=true"):
        RunConfig.model_validate(cfg)
