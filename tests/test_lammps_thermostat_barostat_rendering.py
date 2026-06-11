from __future__ import annotations

from pathlib import Path


def _stage(tmp_path: Path):
    from vitriflow.lammps_input import StageSpec

    data = tmp_path / "input.data"
    data.write_text("\n")
    return StageSpec(
        name="sample",
        input_data=data,
        output_data=tmp_path / "out.data",
        temperature_start=300.0,
        temperature_stop=500.0,
        pressure=1.0,
        equil_steps=0,
        run_steps=10,
        seed=12345,
        velocity_mode="create",
        write_dump=False,
    )


def test_lammps_default_nose_hoover_rendering(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_stage

    script = render_stage(
        KimConfig(model="TEST_MODEL", interactions=["Si"]),
        MDConfig(ensemble="nvt", thermostat={"style": "nose-hoover", "tdamp": 0.1}),
        _stage(tmp_path),
    )
    assert "fix int all nvt temp" in script
    assert "fix th all temp/csvr" not in script
    assert "fix int all nve" not in script


def test_lammps_csvr_rendering_uses_nve_plus_csvr(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_stage

    script = render_stage(
        KimConfig(model="TEST_MODEL", interactions=["Si"]),
        MDConfig(ensemble="nvt", thermostat={"style": "csvr", "tdamp": 0.1}),
        _stage(tmp_path),
    )
    assert "fix int all nve" in script
    assert "fix th all temp/csvr 300.0 500.0 0.1 12345" in script
    assert "unfix th" in script


def test_lammps_langevin_with_nose_hoover_barostat_uses_nph(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_stage

    script = render_stage(
        KimConfig(model="TEST_MODEL", interactions=["Si"]),
        MDConfig(
            ensemble="npt",
            thermostat={"style": "langevin", "tdamp": 0.1},
            barostat={"style": "nose-hoover", "pdamp": 1.0, "mode": "iso"},
        ),
        _stage(tmp_path),
    )
    assert "fix int all nph iso 1.0 1.0 1.0" in script
    assert "fix th all langevin 300.0 500.0 0.1 12345" in script


def test_lammps_berendsen_barostat_rendering(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_stage

    script = render_stage(
        KimConfig(model="TEST_MODEL", interactions=["Si"]),
        MDConfig(
            ensemble="npt",
            thermostat={"style": "berendsen", "tdamp": 0.1},
            barostat={"style": "berendsen", "pdamp": 1.0, "mode": "iso"},
        ),
        _stage(tmp_path),
    )
    assert "fix int all nve" in script
    assert "fix th all temp/berendsen 300.0 500.0 0.1" in script
    assert "fix bar all press/berendsen iso 1.0 1.0 1.0" in script
