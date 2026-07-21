from __future__ import annotations

from pathlib import Path


def _write_minimal_datafile(path: Path, *, x_offset: float = 0.0) -> None:
    path.write_text(
        f"""
LAMMPS data file via vitriflow test

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # atomic

1 1 {1.0 + x_offset:.6f} 2.0 3.0
2 1 {4.0 + x_offset:.6f} 5.0 6.0
""".lstrip()
    )


def _write_minimal_datafile_with_pair_coeffs(path: Path) -> None:
    path.write_text(
        """
LAMMPS data file via write_data

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Pair Coeffs # lj/cut

1 1.0 1.0

Atoms # atomic

1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
""".lstrip()
    )


def _write_thermo_log(path: Path) -> None:
    path.write_text(
        """
Step Temp Press PotEng Volume Density
0 1000.0 0.0 -10.0 1000.0 2.5000
100 1000.0 0.0 -11.0 1001.0 2.4975
200 1000.0 0.0 -12.0 1002.0 2.4950
""".lstrip()
    )


def _write_msd(path: Path, *, valid: bool = True) -> None:
    if valid:
        text = """
0 0.0
100 1.0
200 2.0
""".lstrip()
    else:
        text = """
0 0.0
100 1.0
""".lstrip()
    path.write_text(text)


def _write_final_dump(path: Path, *, x_offset: float = 0.0) -> None:
    path.write_text(
        f"""
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type xu yu zu
1 1 {1.0 + x_offset:.6f} 2.0 3.0
2 1 {4.0 + x_offset:.6f} 5.0 6.0
""".lstrip()
    )


def test_cp2k_authoritative_thermo_serialization_fails_closed_without_placeholder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A completed CP2K stage must not authenticate invented thermo evidence."""

    import numpy as np
    import pytest

    from vitriflow.parse import ThermoTable
    from vitriflow.workflows import stage_runner

    thermo_csv = tmp_path / "thermo.csv"

    def fail_write(*_args, **_kwargs) -> None:
        raise OSError("simulated canonical thermo write failure")

    monkeypatch.setattr(stage_runner, "write_thermo_csv", fail_write)

    with pytest.raises(
        stage_runner.ThermoArtifactError,
        match="Failed to write engine-neutral thermo CSV",
    ) as excinfo:
        stage_runner._materialize_thermo_csv_from_table(
            table=ThermoTable(
                columns=["Step", "Temp", "Press", "PotEng", "Volume", "Density"],
                data=np.asarray(
                    [[0.0, 300.0, 0.0, -10.0, 100.0, 2.0]],
                    dtype=float,
                ),
            ),
            thermo_csv=thermo_csv,
        )

    assert isinstance(excinfo.value.__cause__, OSError)
    assert not thermo_csv.exists()


def test_run_stage_local_logs_msd_placeholder_on_parse_failure(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    from vitriflow.config import LammpsConfig, LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import StageSpec
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stage_local

    input_data = tmp_path / "input.data"
    _write_minimal_datafile(input_data)

    stage_dir = tmp_path / "stage"
    stage = StageSpec(
        name="melt",
        input_data=input_data,
        output_data=tmp_path / "requested" / "melt.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=200,
        seed=7,
        write_dump=False,
    )

    def fake_autoskin(runner, script_builder, workdir, md_cfg, *, log_name, timeout_sec=None, cleanup_paths=None):
        script_builder(md_cfg)
        _write_thermo_log(Path(workdir) / log_name)
        _write_msd(Path(workdir) / "melt.msd.dat", valid=False)
        _write_minimal_datafile(Path(workdir) / "melt.data")
        return 2.5, 0

    monkeypatch.setattr(
        "vitriflow.workflows.stage_runner.run_with_neighbor_skin_autotune",
        fake_autoskin,
    )

    caplog.set_level("WARNING")
    arts = run_stage_local(
        LammpsRunner(LammpsConfig()),
        LammpsPotentialConfig(
            interactions=["X"],
            commands=["pair_style lj/cut 2.5", "pair_coeff * * 1.0 1.0 2.5"],
        ),
        MDConfig(atom_style="atomic"),
        stage,
        stage_dir,
    )

    assert arts.msd_csv.exists()
    assert arts.msd_csv.read_text() == ""
    assert arts.thermo_csv.exists()
    assert arts.manifest_path == stage_dir / "stage_artifacts.json"
    manifest = __import__("json").loads(arts.manifest_path.read_text())
    assert manifest["schema"] == "vitriflow.stage_artifacts.v1"
    assert manifest["engine"] == "lammps"
    assert manifest["timestep_ps"] == 1.0
    assert manifest["artifacts"]["thermo_csv"]["available"] is True
    assert manifest["artifacts"]["msd_csv"]["available"] is False
    assert arts.output_local.exists()
    assert (tmp_path / "requested" / "melt.data").exists()
    assert "Failed to parse MSD series" in caplog.text


def test_run_stage_local_removes_stale_unit_manifest_before_engine_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    from vitriflow.config import LammpsConfig, LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import StageSpec
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stage_local

    input_data = tmp_path / "input.data"
    _write_minimal_datafile(input_data)
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    stale = stage_dir / "stage_artifacts.json"
    stale.write_text('{"stale": true}\n')
    stage = StageSpec(
        name="melt",
        input_data=input_data,
        output_data=Path("melt.data"),
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=200,
        seed=7,
        write_dump=False,
    )

    def fail_before_outputs(*args, **kwargs):
        assert not stale.exists()
        raise RuntimeError("engine failed")

    monkeypatch.setattr(
        "vitriflow.workflows.stage_runner.run_with_neighbor_skin_autotune",
        fail_before_outputs,
    )
    with pytest.raises(RuntimeError, match="engine failed"):
        run_stage_local(
            LammpsRunner(LammpsConfig()),
            LammpsPotentialConfig(
                interactions=["X"],
                commands=["pair_style lj/cut 2.5", "pair_coeff * * 1.0 1.0 2.5"],
            ),
            MDConfig(atom_style="atomic"),
            stage,
            stage_dir,
        )
    assert not stale.exists()


def test_run_stage_local_strips_pair_coeff_sections_from_localized_input(tmp_path: Path, monkeypatch) -> None:
    from vitriflow.config import LammpsConfig, LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import StageSpec
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stage_local

    input_data = tmp_path / "input_with_coeffs.data"
    _write_minimal_datafile_with_pair_coeffs(input_data)

    stage_dir = tmp_path / "stage"
    stage = StageSpec(
        name="melt",
        input_data=input_data,
        output_data=tmp_path / "requested" / "melt.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=200,
        seed=7,
        write_dump=False,
    )

    def fake_autoskin(runner, script_builder, workdir, md_cfg, *, log_name, timeout_sec=None, cleanup_paths=None):
        localized = Path(workdir) / "input.data"
        txt = localized.read_text()
        assert "Pair Coeffs" not in txt
        assert "Atoms # atomic" in txt
        script = script_builder(md_cfg)
        assert "write_data melt.data nocoeff" in script
        _write_thermo_log(Path(workdir) / log_name)
        _write_msd(Path(workdir) / "melt.msd.dat", valid=True)
        _write_minimal_datafile(Path(workdir) / "melt.data")
        return 2.5, 0

    monkeypatch.setattr(
        "vitriflow.workflows.stage_runner.run_with_neighbor_skin_autotune",
        fake_autoskin,
    )

    arts = run_stage_local(
        LammpsRunner(LammpsConfig()),
        LammpsPotentialConfig(
            interactions=["X"],
            commands=["pair_style lj/cut 2.5", "pair_coeff * * 1.0 1.0 2.5"],
        ),
        MDConfig(atom_style="atomic"),
        stage,
        stage_dir,
    )

    assert "Pair Coeffs" not in arts.input_local.read_text()
    assert arts.output_local.exists()



def test_run_stages_continuous_lammps_writes_engine_neutral_msd_csv(tmp_path: Path, monkeypatch) -> None:
    from vitriflow.config import LammpsConfig, LammpsPotentialConfig, MDConfig
    from vitriflow.io.thermo import parse_msd_csv
    from vitriflow.lammps_input import StageSpec
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stages_continuous_lammps

    input_data = tmp_path / "input.data"
    _write_minimal_datafile(input_data)

    stage_dirs = [tmp_path / "melt", tmp_path / "quench"]
    workdir = tmp_path / "work"

    stages = [
        StageSpec(
            name="melt",
            input_data=input_data,
            output_data=tmp_path / "requested" / "melt.data",
            temperature_start=1800.0,
            temperature_stop=1800.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=200,
            seed=11,
            velocity_mode="create",
            write_dump=False,
        ),
        StageSpec(
            name="quench",
            input_data=tmp_path / "requested" / "melt.data",
            output_data=tmp_path / "requested" / "quench.data",
            temperature_start=1800.0,
            temperature_stop=300.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=200,
            seed=12,
            velocity_mode="preserve",
            write_dump=False,
        ),
    ]

    def fake_autoskin(runner, script_builder, workdir, md_cfg, *, log_name, timeout_sec=None, cleanup_paths=None):
        script_builder(md_cfg)
        for idx, st in enumerate(stages):
            sd = stage_dirs[idx]
            sd.mkdir(parents=True, exist_ok=True)
            _write_thermo_log(sd / log_name)
            _write_msd(sd / f"{st.name}.msd.dat", valid=True)
            _write_minimal_datafile(sd / Path(st.output_data).name, x_offset=float(idx))
        return 3.5, 1

    monkeypatch.setattr(
        "vitriflow.workflows.stage_runner.run_with_neighbor_skin_autotune",
        fake_autoskin,
    )

    arts = run_stages_continuous_lammps(
        LammpsRunner(LammpsConfig()),
        LammpsPotentialConfig(
            interactions=["X"],
            commands=["pair_style lj/cut 2.5", "pair_coeff * * 1.0 1.0 2.5"],
        ),
        MDConfig(atom_style="atomic", stage_continuity="continuous"),
        stages,
        stage_dirs,
        workdir,
    )

    assert len(arts) == 2
    for art in arts:
        step, msd = parse_msd_csv(art.msd_csv)
        assert step.tolist() == [0.0, 100.0, 200.0]
        assert msd.tolist() == [0.0, 1.0, 2.0]
        assert art.thermo_csv.exists()
        assert art.output_local.exists()

    assert arts[0].neighbor_skin == 3.5
    assert arts[0].neighbor_skin_retries == 1
    assert arts[1].input_local.read_text() == arts[0].output_local.read_text()
    assert (tmp_path / "requested" / "melt.data").exists()
    assert (tmp_path / "requested" / "quench.data").exists()


def test_run_stages_continuous_lammps_materializes_output_from_final_dump(tmp_path: Path, monkeypatch) -> None:
    from vitriflow.config import LammpsConfig, LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import StageSpec
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stages_continuous_lammps

    input_data = tmp_path / "input.data"
    _write_minimal_datafile(input_data)

    stage_dirs = [tmp_path / "warmup", tmp_path / "melt"]
    workdir = tmp_path / "work"

    stages = [
        StageSpec(
            name="warmup",
            input_data=input_data,
            output_data=tmp_path / "requested" / "warmup.data",
            temperature_start=300.0,
            temperature_stop=1800.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=200,
            seed=11,
            velocity_mode="create",
            write_dump=False,
        ),
        StageSpec(
            name="melt",
            input_data=tmp_path / "requested" / "warmup.data",
            output_data=tmp_path / "requested" / "melt.data",
            temperature_start=1800.0,
            temperature_stop=1800.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=200,
            seed=12,
            velocity_mode="preserve",
            write_dump=False,
        ),
    ]

    def fake_autoskin(runner, script_builder, workdir, md_cfg, *, log_name, timeout_sec=None, cleanup_paths=None):
        script = script_builder(md_cfg)
        assert "write_data" not in script
        for idx, st in enumerate(stages):
            sd = stage_dirs[idx]
            sd.mkdir(parents=True, exist_ok=True)
            _write_thermo_log(sd / log_name)
            _write_msd(sd / f"{st.name}.msd.dat", valid=True)
            _write_final_dump(sd / f"{st.name}.final.lammpstrj", x_offset=float(idx))
        return 3.5, 0

    monkeypatch.setattr(
        "vitriflow.workflows.stage_runner.run_with_neighbor_skin_autotune",
        fake_autoskin,
    )

    arts = run_stages_continuous_lammps(
        LammpsRunner(LammpsConfig()),
        LammpsPotentialConfig(
            interactions=["X"],
            commands=["pair_style lj/cut 2.5", "pair_coeff * * 1.0 1.0 2.5"],
        ),
        MDConfig(atom_style="atomic", stage_continuity="continuous"),
        stages,
        stage_dirs,
        workdir,
    )

    assert len(arts) == 2
    assert arts[0].output_local.exists()
    assert arts[1].output_local.exists()
    txt0 = arts[0].output_local.read_text()
    txt1 = arts[1].output_local.read_text()
    assert "1 1 1 2 3" in txt0.replace(".0", "")
    assert "2 1 4 5 6" in txt0.replace(".0", "")
    assert "1 1 2 2 3" in txt1.replace(".0", "")
    assert arts[1].input_local.read_text() == arts[0].output_local.read_text()
