from __future__ import annotations

from pathlib import Path


def test_render_stage_velocity_mode_preserve_omits_velocity_create(tmp_path: Path):
    from vitriflow.config import LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import StageSpec, render_stage

    pot = LammpsPotentialConfig(
        interactions=["X"],
        commands=[
            "pair_style lj/cut 2.5",
            "pair_coeff * * 1.0 1.0 2.5",
        ],
    )
    md = MDConfig(atom_style="atomic")

    stage_create = StageSpec(
        name="s",
        input_data=tmp_path / "in.data",
        output_data=tmp_path / "out.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
        velocity_mode="create",
    )
    txt_create = render_stage(pot, md, stage_create)
    assert "velocity all create" in txt_create

    stage_preserve = StageSpec(
        name="s",
        input_data=tmp_path / "in.data",
        output_data=tmp_path / "out.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
        velocity_mode="preserve",
    )
    txt_preserve = render_stage(pot, md, stage_preserve)
    assert "velocity all create" not in txt_preserve
    assert "initial velocities: preserved" in txt_preserve


def test_datafile_velocity_detection_and_frame_parsing(tmp_path: Path):
    from vitriflow.analysis.datafile import datafile_has_velocities, read_datafile_frame

    p = tmp_path / "a.data"
    p.write_text(
        """
LAMMPS data file via write_data

4 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # atomic

1 1 1.0 2.0 3.0
2 2 4.0 5.0 6.0
3 1 7.0 8.0 9.0
4 2 1.5 2.5 3.5

Velocities

1 0.1 0.2 0.3
2 0.0 0.0 0.0
3 -0.1 -0.2 -0.3
4 0.5 0.6 0.7
""".lstrip()
    )

    assert datafile_has_velocities(p) is True

    fr = read_datafile_frame(p, atom_style="atomic")
    assert fr.n_atoms == 4
    assert fr.ids.tolist() == [1, 2, 3, 4]
    assert fr.types.tolist() == [1, 2, 1, 2]
    assert fr.positions.shape == (4, 3)
    assert fr.positions[1].tolist() == [4.0, 5.0, 6.0]
    assert fr.origin.tolist() == [0.0, 0.0, 0.0]
    assert fr.cell[0, 0] == 10.0
    assert fr.cell[1, 1] == 10.0
    assert fr.cell[2, 2] == 10.0

    p2 = tmp_path / "b.data"
    p2.write_text(p.read_text().split("Velocities", 1)[0])
    assert datafile_has_velocities(p2) is False


def test_final_extxyz_prefers_output_data_over_last_dump(tmp_path: Path):
    from vitriflow.config import MDConfig
    from vitriflow.workflows.stage_runner import _materialize_lammps_engine_neutral_outputs

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # output snapshot extxyz
    out = stage_dir / "output.data"
    out.write_text(
        """
LAMMPS data file

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # atomic

1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
""".lstrip()
    )

    # dump different extxyz
    dump_path = stage_dir / "stage.lammpstrj"
    dump_path.write_text(
        """
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type xu yu zu
1 1 7.0 8.0 9.0
2 1 10.0 11.0 12.0
""".lstrip()
    )

    md = MDConfig(atom_style="atomic")

    traj_extxyz, final_extxyz = _materialize_lammps_engine_neutral_outputs(
        stage_dir=stage_dir,
        output_data=out,
        dump_path=dump_path,
        md_cfg=md,
        type_to_species=None,
    )

    assert traj_extxyz is not None
    assert final_extxyz.exists()

    # parse extxyz dump
    txt = final_extxyz.read_text().splitlines()
    assert int(txt[0].strip()) == 2
    atom_lines = txt[2:]
    assert len(atom_lines) == 2
    p1 = [float(x) for x in atom_lines[0].split()[1:4]]
    p2 = [float(x) for x in atom_lines[1].split()[1:4]]
    assert p1 == [1.0, 2.0, 3.0]
    assert p2 == [4.0, 5.0, 6.0]


def test_parse_all_thermo_tables(tmp_path: Path):
    from vitriflow.parse import parse_all_thermo_tables

    log = tmp_path / "log.lammps"
    log.write_text(
        """
Step Temp PotEng Density
0 300 -10 2.0
1 300 -11 2.1
Loop time of 0.1 on 1 procs
Step Temp PotEng Density
0 300 -12 2.2
1 300 -13 2.3
""".lstrip()
    )

    tbls = parse_all_thermo_tables(log)
    assert len(tbls) == 2
    assert tbls[0].columns[0] == "Step"
    assert tbls[0].data.shape == (2, 4)
    assert tbls[1].data.shape == (2, 4)

def test_render_stage_force_isotropic_emits_change_box(tmp_path: Path):
    from vitriflow.config import LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import StageSpec, render_stage

    pot = LammpsPotentialConfig(
        interactions=["X"],
        commands=[
            "pair_style lj/cut 2.5",
            "pair_coeff * * 1.0 1.0 2.5",
        ],
    )
    md = MDConfig(atom_style="atomic")

    stage = StageSpec(
        name="melt",
        input_data=tmp_path / "in.data",
        output_data=tmp_path / "out.data",
        temperature_start=2000.0,
        temperature_stop=2000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
        force_isotropic=True,
    )
    txt = render_stage(pot, md, stage)
    assert "change_box all triclinic" in txt
    assert "$(vol^(1.0/3.0))" in txt


def test_render_stage_write_data_uses_nocoeff(tmp_path: Path):
    from vitriflow.config import LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import StageSpec, render_stage

    pot = LammpsPotentialConfig(
        interactions=["X"],
        commands=[
            "pair_style lj/cut 2.5",
            "pair_coeff * * 1.0 1.0 2.5",
        ],
    )
    md = MDConfig(atom_style="atomic")

    stage = StageSpec(
        name="restartable",
        input_data=tmp_path / "input.data",
        output_data=tmp_path / "output.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
    )

    txt = render_stage(pot, md, stage)
    assert "write_data output.data nocoeff" in txt
