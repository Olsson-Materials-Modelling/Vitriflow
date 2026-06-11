from __future__ import annotations

from pathlib import Path


def test_render_continuous_stages_single_read_and_velocity_once(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import StageSpec, render_continuous_stages

    # containing section content
    data = tmp_path / "input.data"
    data.write_text(
        """
LAMMPS data file via vitriflow test

1 atoms
1 atom types

0.0 1.0 xlo xhi
0.0 1.0 ylo yhi
0.0 1.0 zlo zhi

Masses

1 28.0855

Atoms # atomic

1 1 0.0 0.0 0.0
""".lstrip()
    )

    pot = KimConfig(model="TEST_MODEL", interactions=["Si"])
    md = MDConfig(ensemble="nvt")

    melt = StageSpec(
        name="melt",
        input_data=data,
        output_data=tmp_path / "melt.data",
        temperature_start=2000.0,
        temperature_stop=2000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=1234,
        velocity_mode="create",
        force_isotropic=True,
        replicate=(2, 2, 2),
        write_dump=True,
    )
    quench = StageSpec(
        name="quench",
        input_data=tmp_path / "melt.data",  # ignored
        output_data=tmp_path / "quench.data",
        temperature_start=2000.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=20,
        seed=5678,
        velocity_mode="preserve",
        replicate=None,
        write_dump=True,
    )

    script = render_continuous_stages(
        pot,
        md,
        [melt, quench],
        stage_dir_prefixes={"melt": "../melt", "quench": "../quench"},
    )

    # read velocity
    assert script.count("read_data") == 1
    assert script.count("velocity all create") == 1

    # melt isotropic box
    assert script.count("change_box all triclinic") == 1

    # snapshot output materialization
    # unfixes integrator
    assert "write_dump all custom ../melt/melt.final.lammpstrj id type xu yu zu modify sort id" in script
    assert "write_dump all custom ../quench/quench.final.lammpstrj id type xu yu zu modify sort id" in script
    assert "write_data" not in script
    # occurrences between sample
    assert script.count("unfix int") >= 3
