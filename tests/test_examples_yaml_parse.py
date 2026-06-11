from __future__ import annotations

from pathlib import Path


def test_selected_example_yamls_parse():
    from vitriflow.config import RunConfig

    root = Path(__file__).resolve().parents[1]
    exdir = root / "vitriflow" / "examples"
    assert exdir.is_dir()

    # contained external placeholders
    allow = {
        "minimal_metal.yaml",
        "minimal_metal_mpi.yaml",
        "minimal_metal_mpi8.yaml",
        "minimal_metal_packmol.yaml",
        "si_diamond_cp2k_toy.yaml",
        "si_materials_project_cp2k_toy.yaml",
        "sio2_beta_cristobalite_cp2k_smoketest_192.yaml",
        "sio2_beta_cristobalite_buckingham_192.yaml",
        "sio2_beta_cristobalite_buckingham_648.yaml",
        "sio2_cristobalite_vashishta_192.yaml",
        "sulfate_CaSO4_cod.yaml",
        "sulfate_MgSO4_cod.yaml",
        "sulfate_Na2SO4_cod.yaml",
    }

    for name in sorted(allow):
        p = exdir / name
        assert p.exists(), f"Missing example: {name}"
        cfg = RunConfig.from_yaml(p)
        # sanity include autotune
        assert cfg.autotune is not None
