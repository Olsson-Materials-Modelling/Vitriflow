from __future__ import annotations

from pathlib import Path


def test_selected_example_yamls_parse():
    from vitriflow.config import RunConfig

    root = Path(__file__).resolve().parents[1]
    exdir = root / "vitriflow" / "examples"
    assert exdir.is_dir()

    # Every self-contained YAML intentionally included in the release package
    # is mandatory.  The packaged QUIP/GAP demonstration deliberately names
    # user-supplied external potential files and is covered by its dedicated
    # custom-schedule/package-content tests instead.
    release_examples = {
        "minimal_metal.yaml",
        "si_diamond_cp2k_toy.yaml",
        "sio2_bks_zbl_smoke.yaml",
        "sio2_bks_packmol_production.yaml",
        "sio2_bks_cristobalite_production.yaml",
        "sio2_kim_packmol_production.yaml",
        "sio2_kim_cristobalite_production.yaml",
    }

    # These selected examples are developer fixtures, not release package
    # data.  A source checkout must contain the complete set if it contains any
    # of it; an sdist containing none still runs all release-example checks.
    developer_examples = {
        "minimal_metal_mpi.yaml",
        "minimal_metal_mpi8.yaml",
        "minimal_metal_packmol.yaml",
        "si_materials_project_cp2k_toy.yaml",
        "sio2_beta_cristobalite_cp2k_smoketest_192.yaml",
        "sio2_beta_cristobalite_buckingham_192.yaml",
        "sio2_beta_cristobalite_buckingham_648.yaml",
        "sio2_cristobalite_vashishta_192.yaml",
        "sulfate_CaSO4_cod.yaml",
        "sulfate_MgSO4_cod.yaml",
        "sulfate_Na2SO4_cod.yaml",
    }

    for name in sorted(release_examples):
        p = exdir / name
        assert p.exists(), f"Missing release example: {name}"
        cfg = RunConfig.from_yaml(p)
        assert cfg.autotune is not None

    present_developer_examples = {
        name for name in developer_examples if (exdir / name).is_file()
    }
    if present_developer_examples:
        assert present_developer_examples == developer_examples, (
            "Partial developer example fixture set: missing "
            f"{sorted(developer_examples - present_developer_examples)}"
        )
        for name in sorted(developer_examples):
            cfg = RunConfig.from_yaml(exdir / name)
            assert cfg.autotune is not None
