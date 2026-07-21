from __future__ import annotations

import os
from pathlib import Path

import pytest


def _stage(tmp_path: Path, *, output_name: str = "output.data"):
    from vitriflow.lammps_input import StageSpec

    source = tmp_path / "source.data"
    source.write_text("structure\n")
    return StageSpec(
        name="probe",
        input_data=source,
        output_data=tmp_path / output_name,
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=1,
        seed=12345,
        write_dump=True,
        msd_every=1,
    )


def test_cp2k_preflight_high_temperature_outputs_avoid_reserved_stage_prefix():
    import inspect

    from vitriflow.workflows import preflight

    source = inspect.getsource(preflight._run_preflight_cp2k)
    assert "preflight_highT_out.data" not in source
    # NPT and NVT candidate construction must both use the non-colliding name.
    assert source.count('"highT.out.data"') == 2


@pytest.mark.parametrize(
    "token",
    [
        "-i",
        "--input-file=other.inp",
        "-o",
        "--output-file=other.out",
        "--batch",
        "--check",
        "--dry-run",
        "--help",
        "--html-manual",
        "--keep-alive",
        "--memory",
        "--mpi-mapping",
        "--run",
        "--shell",
        "--shell-posix",
        "--version",
        "--xml",
        "-iother.inp",
        "-oother.out",
    ],
)
def test_cp2k_extra_args_reject_runner_override_and_utility_modes(token):
    from vitriflow.config import Cp2kConfig

    with pytest.raises(ValueError, match="non-production execution mode"):
        Cp2kConfig(extra_args=[token])


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra_args", [" --echo"]),
        ("cp2k_cmd", ["cp2k.psmp "]),
        ("basis_set_file_name", " BASIS_MOLOPT"),
        ("potential_file_name", "GTH_POTENTIALS "),
    ],
)
def test_cp2k_tokens_and_filenames_reject_whitespace_aliases(field, value):
    from vitriflow.config import Cp2kConfig

    with pytest.raises(ValueError):
        Cp2kConfig(**{field: value})


def test_cp2k_runner_revalidates_mutated_extra_args_before_execution(tmp_path, monkeypatch):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    workdir = tmp_path / "work"
    workdir.mkdir()
    input_file = workdir / "current.inp"
    input_file.write_text("&GLOBAL\n&END GLOBAL\n")
    cfg = Cp2kConfig(cp2k_cmd=["cp2k.psmp"])
    cfg.extra_args = ["--dry-run"]
    runner = Cp2kRunner(cfg)
    monkeypatch.setattr(
        "vitriflow.runner.run_cmd",
        lambda *_args, **_kwargs: pytest.fail("reserved mode reached CP2K"),
    )

    with pytest.raises(ValueError, match="non-production execution mode"):
        runner.run(input_file, workdir)


def test_cp2k_runner_revalidates_mutated_command_before_execution(tmp_path, monkeypatch):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    workdir = tmp_path / "work"
    workdir.mkdir()
    input_file = workdir / "current.inp"
    input_file.write_text("&GLOBAL\n&END GLOBAL\n")
    cfg = Cp2kConfig(cp2k_cmd=["cp2k.psmp"])
    cfg.cp2k_cmd = ["cp2k.psmp", "--shell-posix"]
    runner = Cp2kRunner(cfg)
    monkeypatch.setattr(
        "vitriflow.runner.run_cmd",
        lambda *_args, **_kwargs: pytest.fail("reserved command reached CP2K"),
    )

    with pytest.raises(ValueError, match="must not preselect"):
        runner.run(input_file, workdir)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("exec_prefix", ["env", " "]),
        ("mpi_cmd", " mpiexec"),
        ("nprocs", 0),
        ("omp_num_threads", 0),
        ("timeout_sec", 0.0),
        ("kill_grace_sec", -1.0),
    ],
)
def test_cp2k_runner_revalidates_model_construct_execution_fields_before_mutation(
    tmp_path, field, bad_value, monkeypatch
):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    payload = Cp2kConfig(cp2k_cmd=["cp2k.psmp"]).model_dump(mode="python")
    payload[field] = bad_value
    cfg = Cp2kConfig.model_construct(**payload)
    runner = Cp2kRunner(cfg)
    workdir = tmp_path / f"must_not_exist_{field}"
    monkeypatch.setattr(
        "vitriflow.runner.run_cmd",
        lambda *_args, **_kwargs: pytest.fail("invalid runtime config reached CP2K"),
    )

    with pytest.raises(ValueError):
        runner.run(workdir / "missing.inp", workdir)
    assert not workdir.exists()


@pytest.mark.parametrize("bad_timeout", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_cp2k_runner_rejects_invalid_timeout_override_before_mutation(
    tmp_path, bad_timeout
):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    runner = Cp2kRunner(Cp2kConfig(cp2k_cmd=["cp2k.psmp"]))
    workdir = tmp_path / "must_not_exist_timeout_override"
    with pytest.raises(ValueError, match="timeout_sec override"):
        runner.run(workdir / "missing.inp", workdir, timeout_sec=bad_timeout)
    assert not workdir.exists()


def test_cp2k_version_probe_revalidates_runtime_config_before_mutation(tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    cfg = Cp2kConfig(cp2k_cmd=["cp2k.psmp"])
    cfg.nprocs = 0
    runner = Cp2kRunner(cfg)
    workdir = tmp_path / "must_not_exist_version"
    with pytest.raises(ValueError):
        runner.query_version(workdir)
    assert not workdir.exists()


def test_cp2k_runner_requires_direct_unaliased_workdir_input(tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    workdir = tmp_path / "work"
    workdir.mkdir()
    outside = tmp_path / "outside.inp"
    outside.write_text("&GLOBAL\n&END GLOBAL\n")
    runner = Cp2kRunner(Cp2kConfig(cp2k_cmd=["cp2k.psmp"]))

    with pytest.raises(ValueError, match="direct child"):
        runner.run(outside, workdir)

    direct = workdir / "direct.inp"
    os.link(outside, direct)
    with pytest.raises(ValueError, match="hard-linked"):
        runner.run(direct, workdir)


def test_cp2k_runner_rejects_output_traversal_and_data_collision(tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    workdir = tmp_path / "work"
    workdir.mkdir()
    input_file = workdir / "current.inp"
    input_file.write_text("&GLOBAL\n&END GLOBAL\n")

    runner = Cp2kRunner(Cp2kConfig(cp2k_cmd=["cp2k.psmp"]))
    with pytest.raises(ValueError, match="path-safe basename"):
        runner.run(input_file, workdir, output_name="../outside.out")

    collision = Cp2kRunner(
        Cp2kConfig(
            cp2k_cmd=["cp2k.psmp"],
            basis_set_file_name="current.inp",
        )
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        collision.run(input_file, workdir)


def test_cp2k_runner_clears_stale_output_and_requires_current_output(tmp_path, monkeypatch):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    workdir = tmp_path / "work"
    workdir.mkdir()
    input_file = workdir / "current.inp"
    input_file.write_text("&GLOBAL\n&END GLOBAL\n")
    stale = workdir / "cp2k.out"
    stale.write_text("old scientific output\n")

    runner = Cp2kRunner(Cp2kConfig(cp2k_cmd=["cp2k.psmp"]))
    monkeypatch.setattr(
        runner, "_ensure_data_files_present", lambda _workdir, **_kwargs: None
    )
    monkeypatch.setattr("vitriflow.runner.run_cmd", lambda *_args, **_kwargs: (0, "", ""))

    with pytest.raises(RuntimeError, match="without creating current output"):
        runner.run(input_file, workdir)
    assert not stale.exists()


def test_cp2k_runner_accepts_only_fresh_single_link_output(tmp_path, monkeypatch):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    workdir = tmp_path / "work"
    workdir.mkdir()
    input_file = workdir / "current.inp"
    input_file.write_text("&GLOBAL\n&END GLOBAL\n")
    runner = Cp2kRunner(Cp2kConfig(cp2k_cmd=["cp2k.psmp"], extra_args=["--echo"]))
    monkeypatch.setattr(
        runner, "_ensure_data_files_present", lambda _workdir, **_kwargs: None
    )

    def fake_run(cmd, **kwargs):
        output_name = cmd[cmd.index("-o") + 1]
        (Path(kwargs["cwd"]) / output_name).write_text("PROGRAM ENDED AT\n")
        return 0, "banner", ""

    monkeypatch.setattr("vitriflow.runner.run_cmd", fake_run)
    result = runner.run(input_file, workdir)
    assert result.log_file.read_text() == "PROGRAM ENDED AT\n"
    assert result.cmd[-1] == "--echo"


@pytest.mark.parametrize(
    "output_name,basis_name,potential_name,match",
    [
        ("screen.out", "BASIS_MOLOPT", "GTH_POTENTIALS", "'.data' basename"),
        ("input.data", "BASIS_MOLOPT", "GTH_POTENTIALS", "collision"),
        ("probe_seg000.data", "BASIS_MOLOPT", "GTH_POTENTIALS", "reserved project prefix"),
        ("output.data", "input.data", "GTH_POTENTIALS", "collision"),
        ("output.data", "SHARED", "SHARED", "collision"),
        ("output.data", "probe_equil.inp", "GTH_POTENTIALS", "reserved project prefix"),
    ],
)
def test_cp2k_stage_namespace_rejects_every_direct_child_collision(
    tmp_path, output_name, basis_name, potential_name, match
):
    from vitriflow.config import Cp2kConfig
    from vitriflow.workflows.stage_runner import _validate_cp2k_stage_artifact_namespace

    stage = _stage(tmp_path, output_name=output_name)
    cfg = Cp2kConfig(
        cp2k_cmd=["cp2k.psmp"],
        basis_set_file_name=basis_name,
        potential_file_name=potential_name,
    )
    with pytest.raises(ValueError, match=match):
        _validate_cp2k_stage_artifact_namespace(stage, cfg=cfg, log_name="log.lammps")


def test_cp2k_invalid_namespace_fails_before_stage_directory_mutation(tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner
    from vitriflow.workflows.stage_runner import _run_stage_local_cp2k

    stage = _stage(tmp_path, output_name="probe_seg000.data")
    stage_dir = tmp_path / "must_not_be_created"
    runner = Cp2kRunner(Cp2kConfig(cp2k_cmd=["cp2k.psmp"]))

    with pytest.raises(ValueError, match="reserved project prefix"):
        _run_stage_local_cp2k(
            runner,
            object(),
            stage,
            stage_dir,
            type_to_species=["Si"],
            log_name="log.lammps",
        )
    assert not stage_dir.exists()


def test_cp2k_absolute_data_file_inside_stage_namespace_cannot_alias_input(tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.workflows.stage_runner import _validate_cp2k_stage_artifact_namespace

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    colliding_basis = stage_dir / "input.data"
    colliding_basis.write_text("basis\n")
    stage = _stage(tmp_path)
    cfg = Cp2kConfig(
        cp2k_cmd=["cp2k.psmp"],
        basis_set_file_name=str(colliding_basis),
    )
    with pytest.raises(ValueError, match="collision"):
        _validate_cp2k_stage_artifact_namespace(
            stage,
            cfg=cfg,
            log_name="log.lammps",
            stage_dir=stage_dir,
        )


def test_cp2k_stage_dir_final_symlink_is_rejected(tmp_path):
    from vitriflow.workflows.stage_runner import _ensure_real_cp2k_stage_dir

    target = tmp_path / "real"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="non-symlink directory"):
        _ensure_real_cp2k_stage_dir(alias)
