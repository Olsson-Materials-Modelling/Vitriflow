from __future__ import annotations

import os
from pathlib import Path

import pytest


def _minimal_data(path: Path) -> None:
    path.write_text(
        """LAMMPS data

1 atoms
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # atomic

1 1 1 1 1
"""
    )


def _stage(input_data: Path, *, output_data: object = "output.data", name: str = "melt"):
    from vitriflow.lammps_input import StageSpec

    return StageSpec(
        name=name,
        input_data=input_data,
        output_data=output_data,
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=1,
        seed=7,
        write_dump=False,
    )


def _potential(*, files=()):
    from vitriflow.config import LammpsPotentialConfig

    return LammpsPotentialConfig(
        interactions=["X"],
        commands=["pair_style lj/cut 2.5", "pair_coeff * * 1 1 2.5"],
        files=list(files),
    )


def test_published_gap_plus_basename_validates_and_materializes_exact_bytes(
    tmp_path: Path,
):
    """The shipped GAP-20U+gr spelling is safe and must remain executable."""

    from vitriflow.config import (
        LammpsPotentialConfig,
        validated_lammps_localized_filename,
    )
    from vitriflow.potential import prepare_potential_files

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sources = [
        source_dir / "Carbon_GAP_20U+gr.xml",
        source_dir
        / "Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_8891",
    ]
    for index, source in enumerate(sources):
        source.write_bytes(f"protected GAP bytes {index}\n".encode("ascii"))

    assert (
        validated_lammps_localized_filename(
            sources[0].name,
            field_name="potential filename",
        )
        == "Carbon_GAP_20U+gr.xml"
    )

    potential = LammpsPotentialConfig(
        interactions=["C"],
        commands=[
            "pair_style quip",
            'pair_coeff * * Carbon_GAP_20U+gr.xml "Potential xml_label=GAP" 6',
        ],
        files=sources,
    )
    stage = tmp_path / "stage"
    prepare_potential_files(potential, stage)

    assert sorted(path.name for path in stage.iterdir()) == sorted(
        source.name for source in sources
    )
    for source in sources:
        localized = stage / source.name
        assert localized.read_bytes() == source.read_bytes()
        assert not localized.is_symlink()
        assert localized.stat().st_nlink == 1


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.xml",
        "subdir/file.xml",
        r"subdir\file.xml",
        ".",
        "..",
        " leading.xml",
        "bad\nname.xml",
    ],
)
def test_localized_lammps_filename_allowlist_still_rejects_unsafe_names(
    unsafe_name: str,
):
    from vitriflow.config import validated_lammps_localized_filename

    with pytest.raises(ValueError, match="path-safe basename"):
        validated_lammps_localized_filename(
            unsafe_name,
            field_name="potential filename",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra_args", ["-skiprun"]),
        ("extra_args", ["-in=other.in"]),
        ("extra_args", [" -sf"]),
        ("lammps_cmd", ["lmp", "-skiprun"]),
        ("lammps_cmd", ["lmp", "-log", "other.log"]),
        ("lammps_cmd", ["lmp", "-restart2data"]),
    ],
)
def test_lammps_config_rejects_runner_control_overrides(field, value):
    from vitriflow.config import LammpsConfig

    with pytest.raises(ValueError):
        LammpsConfig(**{field: value})


def test_lammps_runner_revalidates_mutated_command_before_creating_workdir(
    tmp_path: Path,
):
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    cfg = LammpsConfig.model_construct(
        lammps_cmd=["lmp", "-skiprun"],
        mpi_cmd=None,
        nprocs=1,
        extra_args=[],
        timeout_sec=None,
        kill_grace_sec=1.0,
    )
    workdir = tmp_path / "not-created"
    with pytest.raises(ValueError, match="execution controls"):
        LammpsRunner(cfg).run("run 0\n", workdir, "log.lammps")
    assert not workdir.exists()


def test_lammps_runner_replaces_alias_entries_without_touching_victims(
    tmp_path: Path, monkeypatch
):
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    workdir = tmp_path / "run"
    workdir.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("must survive\n")
    aliases = {
        "in.lammps": "symlink",
        "log.lammps": "hardlink",
        "screen.out": "symlink",
        "stdout.txt": "hardlink",
        "stderr.txt": "symlink",
    }
    for name, kind in aliases.items():
        target = workdir / name
        try:
            if kind == "symlink":
                target.symlink_to(victim)
            else:
                os.link(victim, target)
        except OSError as exc:
            pytest.skip(f"filesystem link creation unavailable: {exc}")

    def successful_run(cmd, *, cwd, **_kwargs):
        log_name = cmd[cmd.index("-log") + 1]
        (Path(cwd) / log_name).write_text("fresh LAMMPS log\n")
        return 0, "captured stdout\n", "captured stderr\n"

    monkeypatch.setattr("vitriflow.runner.run_cmd", successful_run)
    LammpsRunner(LammpsConfig()).run("run 0\n", workdir, "log.lammps")

    assert victim.read_text() == "must survive\n"
    for name in aliases:
        assert not (workdir / name).is_symlink()
    assert (workdir / "in.lammps").read_text() == "run 0\n"
    assert (workdir / "stdout.txt").read_text() == "captured stdout\n"


@pytest.mark.parametrize(
    "field,value",
    [
        ("mpi_cmd", " mpirun"),
        ("mpi_cmd", 7),
        ("nprocs", 0),
        ("nprocs", True),
        ("timeout_sec", 0.0),
        ("timeout_sec", float("nan")),
        ("kill_grace_sec", -1.0),
        ("kill_grace_sec", float("inf")),
    ],
)
def test_lammps_runner_revalidates_mutated_runtime_fields_before_mutation(
    tmp_path: Path, field: str, value: object
):
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    cfg = LammpsConfig().model_copy(update={field: value})
    workdir = tmp_path / "not-created"
    with pytest.raises(ValueError):
        LammpsRunner(cfg).run("run 0\n", workdir, "log.lammps")
    assert not workdir.exists()


def test_lammps_runner_rejects_invalid_explicit_timeout_before_mutation(
    tmp_path: Path,
):
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    workdir = tmp_path / "not-created"
    with pytest.raises(ValueError, match="timeout_sec override"):
        LammpsRunner(LammpsConfig()).run(
            "run 0\n", workdir, "log.lammps", timeout_sec=-1.0
        )
    assert not workdir.exists()


@pytest.mark.parametrize("mode", ["missing", "empty", "symlink", "hardlink"])
def test_zero_exit_requires_fresh_nonempty_direct_single_link_log(
    tmp_path: Path, monkeypatch, mode: str
):
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    victim = tmp_path / "victim.log"
    victim.write_text("outside log\n")

    def fake_run(cmd, *, cwd, **_kwargs):
        log = Path(cwd) / cmd[cmd.index("-log") + 1]
        if mode == "empty":
            log.write_text("")
        elif mode == "symlink":
            log.symlink_to(victim)
        elif mode == "hardlink":
            os.link(victim, log)
        return 0, "", ""

    monkeypatch.setattr("vitriflow.runner.run_cmd", fake_run)
    with pytest.raises(RuntimeError, match="LAMMPS returned success"):
        LammpsRunner(LammpsConfig()).run(
            "run 0\n", tmp_path / "run", "log.lammps"
        )
    assert victim.read_text() == "outside log\n"


def test_lammps_runner_fails_before_execution_if_stale_artifact_is_directory(
    tmp_path: Path, monkeypatch
):
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    workdir = tmp_path / "run"
    (workdir / "log.lammps").mkdir(parents=True)
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return 0, "", ""

    monkeypatch.setattr("vitriflow.runner.run_cmd", fake_run)
    with pytest.raises(RuntimeError, match="Cannot remove stale LAMMPS attempt artifact"):
        LammpsRunner(LammpsConfig()).run("run 0\n", workdir, "log.lammps")
    assert called is False


def test_autoskin_clears_stage_outputs_before_initial_and_retry_attempts(
    tmp_path: Path,
):
    from vitriflow.config import MDConfig
    from vitriflow.utils import CommandFailureContext, ExternalCommandError
    from vitriflow.workflows.autoskin import run_with_neighbor_skin_autotune

    stale = tmp_path / "output.data"
    stale.write_text("old\n")

    class RetryRunner:
        calls = 0

        def run(self, *_args, **_kwargs):
            self.calls += 1
            assert not stale.exists()
            if self.calls == 1:
                stale.write_text("failed-attempt\n")
                raise ExternalCommandError(
                    ["lmp"],
                    1,
                    "",
                    "",
                    context=CommandFailureContext(
                        screen_tail="Out of range atoms - cannot compute PPPM"
                    ),
                )
            return None

    runner = RetryRunner()
    md = MDConfig(
        atom_style="atomic",
        neighbor_skin=2.0,
        neighbor_skin_autotune=True,
        neighbor_skin_step=0.5,
        neighbor_skin_max=3.0,
    )
    skin, retries = run_with_neighbor_skin_autotune(
        runner,
        lambda _md: "kspace_style pppm 1e-6\nrun 1\n",
        tmp_path,
        md,
        cleanup_paths=[stale],
    )
    assert (skin, retries, runner.calls) == (2.5, 1, 2)


@pytest.mark.parametrize("reserved", ["input.data", "log.lammps", "thermo.csv"])
def test_single_stage_namespace_collision_fails_before_directory_mutation(
    tmp_path: Path, reserved: str
):
    from vitriflow.config import LammpsConfig, MDConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stage_local

    source = tmp_path / "source.data"
    _minimal_data(source)
    stage_dir = tmp_path / "stage"
    with pytest.raises(ValueError, match="collision"):
        run_stage_local(
            LammpsRunner(LammpsConfig()),
            _potential(),
            MDConfig(atom_style="atomic"),
            _stage(source, output_data=reserved),
            stage_dir,
        )
    assert not stage_dir.exists()


def test_potential_asset_collision_fails_before_stage_directory_mutation(
    tmp_path: Path,
):
    from vitriflow.config import LammpsConfig, MDConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stage_local

    source = tmp_path / "source.data"
    _minimal_data(source)
    auxiliary_dir = tmp_path / "aux"
    auxiliary_dir.mkdir()
    auxiliary = auxiliary_dir / "input.data"
    auxiliary.write_text("potential bytes\n")
    stage_dir = tmp_path / "stage"

    with pytest.raises(ValueError, match="collision"):
        run_stage_local(
            LammpsRunner(LammpsConfig()),
            _potential(files=[auxiliary]),
            MDConfig(atom_style="atomic"),
            _stage(source),
            stage_dir,
        )
    assert not stage_dir.exists()


def test_runtime_mg2_filename_escape_fails_before_stage_directory_mutation(
    tmp_path: Path,
):
    from vitriflow.config import LammpsConfig, MDConfig, MG2SiNPotentialConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stage_local

    source = tmp_path / "source.data"
    _minimal_data(source)
    pot = MG2SiNPotentialConfig.model_construct(
        kind="mg2_sin", table_filename="../escape.table"
    )
    stage_dir = tmp_path / "stage"
    with pytest.raises(ValueError, match="MG2 table_filename"):
        run_stage_local(
            LammpsRunner(LammpsConfig()),
            pot,
            MDConfig(atom_style="atomic"),
            _stage(source),
            stage_dir,
        )
    assert not stage_dir.exists()


def test_continuous_directories_are_bounded_and_validated_before_mutation(
    tmp_path: Path,
):
    from vitriflow.config import LammpsConfig, MDConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stages_continuous_lammps

    source = tmp_path / "source.data"
    _minimal_data(source)
    root = tmp_path / "box"
    outside = tmp_path / "outside"
    stages = [_stage(source, name="melt"), _stage(source, name="quench")]
    with pytest.raises(ValueError, match="same canonical parent"):
        run_stages_continuous_lammps(
            LammpsRunner(LammpsConfig()),
            _potential(),
            MDConfig(atom_style="atomic"),
            stages,
            [root / "melt", outside / "quench"],
            root / "continuous",
        )
    assert not root.exists()
    assert not outside.exists()


def test_continuous_rejects_symlink_stage_directory_without_following_it(
    tmp_path: Path,
):
    from vitriflow.config import LammpsConfig, MDConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.stage_runner import run_stages_continuous_lammps

    source = tmp_path / "source.data"
    _minimal_data(source)
    root = tmp_path / "box"
    root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    alias = root / "quench"
    try:
        alias.symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    stages = [_stage(source, name="melt"), _stage(source, name="quench")]
    with pytest.raises(ValueError, match="symbolic link"):
        run_stages_continuous_lammps(
            LammpsRunner(LammpsConfig()),
            _potential(),
            MDConfig(atom_style="atomic"),
            stages,
            [root / "melt", alias],
            root / "continuous",
        )
    assert alias.is_symlink()
    assert list(victim.iterdir()) == []
    assert not (root / "continuous").exists()


def test_generated_mg2_table_atomically_replaces_symlink_without_victim_write(
    tmp_path: Path,
):
    from vitriflow.config import MG2SiNPotentialConfig
    from vitriflow.potential import prepare_potential_files

    stage = tmp_path / "stage"
    stage.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("must survive\n")
    table = stage / "mg2_sin.table"
    try:
        table.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    prepare_potential_files(MG2SiNPotentialConfig(kind="mg2_sin"), stage)
    assert victim.read_text() == "must survive\n"
    assert table.is_file() and not table.is_symlink()
    assert table.stat().st_size > 0
