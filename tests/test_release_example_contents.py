from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_installed_example_payload_is_the_declared_release_subset() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    package_data = project["tool"]["setuptools"]["package-data"]["vitriflow"]
    example_entries = {entry for entry in package_data if entry.startswith("examples/")}
    assert example_entries == {
        "examples/minimal_metal.yaml",
        "examples/al_fcc_4x4x4.data",
        "examples/si_diamond_cp2k_toy.yaml",
        "examples/sio2_bks_zbl_smoke.yaml",
        "examples/hc_C_GAP20Ugr_hc_custom_demo.yaml",
        "examples/sio2_bks_packmol_production.yaml",
        "examples/sio2_bks_cristobalite_production.yaml",
        "examples/sio2_kim_packmol_production.yaml",
        "examples/sio2_kim_cristobalite_production.yaml",
    }
    assert not any("*" in entry for entry in example_entries)

    manifest_lines = {
        line.strip()
        for line in (ROOT / "MANIFEST.in").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        f"include vitriflow/{entry}" for entry in example_entries
    }.issubset(manifest_lines)
    assert not any(
        line.startswith("recursive-include vitriflow/examples")
        for line in manifest_lines
    )


def test_release_smoke_examples_preserve_fast_and_physical_input_contracts() -> None:
    from vitriflow.config import LammpsPotentialConfig, RunConfig

    examples = ROOT / "vitriflow" / "examples"

    metal = RunConfig.from_yaml(examples / "minimal_metal.yaml")
    assert metal.autotune.tm_scan.replicates_per_temp == 1
    assert metal.autotune.highT.replicates == 1
    assert metal.autotune.quench.replicates_per_rate == 1
    assert metal.autotune.production.min_boxes == 2
    assert metal.autotune.production.max_boxes == 2

    cp2k = RunConfig.from_yaml(examples / "si_diamond_cp2k_toy.yaml")
    assert cp2k.structure.generate is not None
    assert cp2k.structure.generate.n_formula_units == 16
    assert cp2k.cp2k is not None
    assert cp2k.cp2k.ramp_max_segments >= 24
    assert cp2k.cp2k.timeout_sec is not None

    bks = RunConfig.from_yaml(examples / "sio2_bks_zbl_smoke.yaml")
    assert isinstance(bks.kim, LammpsPotentialConfig)
    assert bks.kim.interactions == ["Si", "O"]
    assert bks.structure.charges == {"Si": 2.4, "O": -1.2}
    assert bks.kim.commands[:4] == [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381",
        "pair_coeff 2 2 1388.7730 0.362319 175.0",
    ]
    assert bks.kim.core_repulsion.enabled is True
    assert bks.kim.core_repulsion.tabulate is True
    assert bks.kim.core_repulsion.style == "zbl"
    assert bks.kim.core_repulsion.table_points == 128000
    assert bks.kim.core_repulsion.table_points_max == 128000
    assert bks.kim.core_repulsion.table_verify_points == 50001
    assert bks.autotune.metrics.time_average_frames == 1
    assert bks.autotune.production.min_boxes == 1
    assert bks.autotune.production.max_boxes == 1
    assert bks.autotune.production.check_convergence is False
