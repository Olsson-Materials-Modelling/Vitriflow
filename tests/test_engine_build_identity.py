from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


def _run_config():
    from vitriflow.config import RunConfig

    return RunConfig.model_validate(
        {
            "lammps": {
                "lammps_cmd": sys.executable,
                "mpi_cmd": sys.executable,
                "nprocs": 8,
            },
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Al"],
                "commands": ["pair_style zero 5.0", "pair_coeff * *"],
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "Al",
                    "n_formula_units": 1,
                }
            },
            "autotune": {
                "metrics": {
                    "enabled": True,
                    "type_to_species": ["Al"],
                    "pairs": [{"pair": ["Al", "Al"]}],
                },
                "production": {
                    "enabled": True,
                    "min_boxes": 1,
                    "max_boxes": 2,
                    "batch_boxes": 1,
                },
            },
        }
    )


def _identity(monkeypatch, tmp_path: Path, release: str):
    import vitriflow.engine_identity as module

    monkeypatch.setattr(
        module,
        "run_cmd",
        lambda cmd, **kwargs: (
            0,
            "Large-scale Atomic/Molecular Massively Parallel Simulator - "
            f"{release}\nInstalled packages: KSPACE MANYBODY\n",
            "",
        ),
    )
    return module.query_lammps_build_identity(
        _run_config().lammps,
        workdir=tmp_path,
    )


def test_lammps_identity_uses_direct_help_and_binds_full_build_banner(
    monkeypatch, tmp_path: Path
):
    import vitriflow.engine_identity as module

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return (
            0,
            "Large-scale Atomic/Molecular Massively Parallel Simulator - "
            "22 Jul 2025 - Update 4\nInstalled packages: KSPACE MANYBODY\n",
            "",
        )

    monkeypatch.setattr(module, "run_cmd", fake_run)
    identity = module.query_lammps_build_identity(
        _run_config().lammps,
        workdir=tmp_path,
    )
    module.validate_engine_build_identity(identity, expected_engine="lammps")
    assert calls == [identity["probe"]["resolved_command"]]
    assert identity["configured_execution_command"][:3] == [
        sys.executable,
        "-np",
        "8",
    ]
    assert identity["probe"]["release_banner"] == "22 Jul 2025 - Update 4"

    changed = dict(identity)
    monkeypatch.setattr(
        module,
        "run_cmd",
        lambda cmd, **kwargs: (
            0,
            "Large-scale Atomic/Molecular Massively Parallel Simulator - "
            "22 Jul 2025 - Update 4\nInstalled packages: KSPACE MANYBODY ML-IAP\n",
            "",
        ),
    )
    changed = module.query_lammps_build_identity(
        _run_config().lammps,
        workdir=tmp_path,
    )
    assert changed["probe"]["release_banner"] == identity["probe"]["release_banner"]
    assert changed["identity_sha256"] != identity["identity_sha256"]


@pytest.mark.parametrize(
    "returncode,stdout,match",
    [
        (1, "", "return code"),
        (0, "not a LAMMPS banner\n", "unambiguous"),
        (
            0,
            "Large-scale Atomic/Molecular Massively Parallel Simulator - A\n"
            "Large-scale Atomic/Molecular Massively Parallel Simulator - B\n",
            "unambiguous",
        ),
    ],
)
def test_lammps_identity_fails_closed_on_query_or_banner_ambiguity(
    monkeypatch, tmp_path: Path, returncode: int, stdout: str, match: str
):
    import vitriflow.engine_identity as module

    monkeypatch.setattr(
        module, "run_cmd", lambda cmd, **kwargs: (returncode, stdout, "")
    )
    with pytest.raises(RuntimeError, match=match):
        module.query_lammps_build_identity(_run_config().lammps, workdir=tmp_path)


def test_cp2k_identity_uses_exec_prefix_without_mpi_and_rejects_conflicts(
    monkeypatch, tmp_path: Path
):
    import vitriflow.engine_identity as module
    from vitriflow.config import Cp2kConfig

    cfg = Cp2kConfig.model_validate(
        {
            "exec_prefix": ["env", "OMP_NUM_THREADS=1"],
            "cp2k_cmd": sys.executable,
            "mpi_cmd": sys.executable,
            "nprocs": 4,
        }
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return 0, "CP2K version 2024.2\n revision git:abc123\n", ""

    monkeypatch.setattr(module, "run_cmd", fake_run)
    identity = module.query_cp2k_build_identity(cfg, workdir=tmp_path)
    module.validate_engine_build_identity(identity, expected_engine="cp2k")
    assert calls == [identity["probe"]["resolved_command"]]
    assert identity["probe"]["version"] == "2024.2"
    assert identity["configured_execution_command"] == [
        "env",
        "OMP_NUM_THREADS=1",
        sys.executable,
        "-np",
        "4",
        sys.executable,
    ]

    # A valid self-digest cannot compensate for deleting the file identity of
    # an explicitly path-qualified delegated executable.  CP2K must be bound
    # in all three command roles and MPI in both launch roles.
    required_path_evidence = (
        ("engine_command", 0, "delegated engine executable"),
        ("execution_command", 5, "delegated engine executable"),
        ("probe_command", 5, "delegated engine executable"),
        ("execution_command", 2, "delegated MPI executable"),
        ("probe_command", 2, "delegated MPI executable"),
    )
    for role, token_index, match in required_path_evidence:
        tampered = json.loads(json.dumps(identity))
        tampered["command_file_identities"] = [
            row
            for row in tampered["command_file_identities"]
            if not (row["role"] == role and row["token_index"] == token_index)
        ]
        payload = {
            key: value
            for key, value in tampered.items()
            if key not in {"algorithm", "identity_sha256"}
        }
        tampered["identity_sha256"] = module._canonical_sha256(payload)
        with pytest.raises(RuntimeError, match=match):
            module.validate_engine_build_identity(tampered, expected_engine="cp2k")

    monkeypatch.setattr(
        module,
        "run_cmd",
        lambda cmd, **kwargs: (
            0,
            "CP2K version 2024.2\nCP2K version 2025.1\n",
            "",
        ),
    )
    with pytest.raises(RuntimeError, match="unambiguous"):
        module.query_cp2k_build_identity(cfg, workdir=tmp_path)


def test_cp2k_identity_resolves_prefixed_environment_not_local_path(
    monkeypatch, tmp_path: Path
):
    import shutil

    import vitriflow.engine_identity as module
    from vitriflow.config import Cp2kConfig

    # Model the packaged cross-environment configuration exactly: neither
    # bare executable exists in this process' PATH, while the prefix selects
    # an environment where both are available.
    cp2k_name = "cp2k.psmp"
    mpi_name = "mpiexec"
    real_which = shutil.which

    def caller_environment_which(command):
        if command in {cp2k_name, mpi_name}:
            return None
        return real_which(command)

    monkeypatch.setattr(module.shutil, "which", caller_environment_which)

    prefix_script = tmp_path / "select_cp2k_environment.py"
    prefix_script.write_text("# prefix environment revision one\n")
    cfg = Cp2kConfig.model_validate(
        {
            "exec_prefix": [sys.executable, str(prefix_script)],
            "cp2k_cmd": cp2k_name,
            "mpi_cmd": mpi_name,
            "nprocs": 6,
            "extra_args": ["--machine-readable"],
        }
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return 0, "CP2K version 2024.2\n revision git:prefixed-build\n", ""

    monkeypatch.setattr(module, "run_cmd", fake_run)
    first = module.query_cp2k_build_identity(cfg, workdir=tmp_path)
    module.validate_engine_build_identity(first, expected_engine="cp2k")

    expected_probe = [
        str(Path(sys.executable).resolve(strict=True)),
        str(prefix_script.resolve(strict=True)),
        mpi_name,
        "-np",
        "1",
        cp2k_name,
        "--version",
    ]
    assert calls == [expected_probe]
    assert first["resolved_engine_command"] == [cp2k_name]
    assert first["configured_execution_command"] == [
        sys.executable,
        str(prefix_script),
        mpi_name,
        "-np",
        "6",
        cp2k_name,
        "--machine-readable",
    ]
    assert first["exec_prefix_binding"] == {
        "mode": "runtime_environment_delegation_v1",
        "configured_prefix": [sys.executable, str(prefix_script)],
        "prefix_executable_index": 0,
        "execution_cp2k_index": 5,
        "probe_cp2k_index": 5,
        "execution_mpi_index": 2,
        "probe_mpi_index": 2,
        "probe_mpi_nprocs": 1,
    }
    assert not any(
        row["role"] == "engine_command" and row["token_index"] == 0
        for row in first["command_file_identities"]
    )
    assert {
        (row["role"], row["token_index"]) for row in first["command_file_identities"]
    }.issuperset({("execution_command", 0), ("probe_command", 0)})

    # The self-digest is not the structural validator.  Even after a malformed
    # record is re-digested, delegated token positions and the prefix trust
    # anchor must still be rejected fail closed.
    tampered = json.loads(json.dumps(first))
    tampered["exec_prefix_binding"]["execution_cp2k_index"] = 4
    payload = {
        key: value
        for key, value in tampered.items()
        if key not in {"algorithm", "identity_sha256"}
    }
    tampered["identity_sha256"] = module._canonical_sha256(payload)
    with pytest.raises(RuntimeError, match="delegated engine command"):
        module.validate_engine_build_identity(tampered, expected_engine="cp2k")

    unbound = json.loads(json.dumps(first))
    unbound["command_file_identities"] = [
        row
        for row in unbound["command_file_identities"]
        if not (row["role"] == "probe_command" and row["token_index"] == 0)
    ]
    payload = {
        key: value
        for key, value in unbound.items()
        if key not in {"algorithm", "identity_sha256"}
    }
    unbound["identity_sha256"] = module._canonical_sha256(payload)
    with pytest.raises(RuntimeError, match="exec-prefix executable"):
        module.validate_engine_build_identity(unbound, expected_engine="cp2k")

    # Prefix contents are provenance, not decoration: changing the local
    # environment selector changes the sealed engine identity even when the
    # configured downstream tokens and reported CP2K version are unchanged.
    prefix_script.write_text("# prefix environment revision two\n")
    calls.clear()
    second = module.query_cp2k_build_identity(cfg, workdir=tmp_path)
    module.validate_engine_build_identity(second, expected_engine="cp2k")
    assert calls == [expected_probe]
    assert first["identity_sha256"] != second["identity_sha256"]


def test_cp2k_identity_without_prefix_still_requires_local_executable(
    monkeypatch, tmp_path: Path
):
    import shutil

    import vitriflow.engine_identity as module
    from vitriflow.config import Cp2kConfig

    cp2k_name = "cp2k.psmp"
    real_which = shutil.which
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda command: None if command == cp2k_name else real_which(command),
    )
    called = False

    def fake_run(cmd, **kwargs):
        nonlocal called
        called = True
        return 0, "CP2K version 2024.2\n", ""

    monkeypatch.setattr(module, "run_cmd", fake_run)
    with pytest.raises(RuntimeError, match="executable cannot be resolved"):
        module.query_cp2k_build_identity(
            Cp2kConfig(cp2k_cmd=cp2k_name), workdir=tmp_path
        )
    assert called is False


def test_wrapper_script_content_is_bound_for_lammps_and_cp2k(
    monkeypatch, tmp_path: Path
):
    import vitriflow.engine_identity as module
    from vitriflow.config import Cp2kConfig, LammpsConfig

    wrapper = tmp_path / "engine_wrapper.py"
    wrapper.write_text("# wrapper revision one\n")
    lammps_cfg = LammpsConfig.model_validate(
        {"lammps_cmd": [sys.executable, str(wrapper)]}
    )
    monkeypatch.setattr(
        module,
        "run_cmd",
        lambda cmd, **kwargs: (
            0,
            "Large-scale Atomic/Molecular Massively Parallel Simulator - "
            "22 Jul 2025 - Update 4\n",
            "",
        ),
    )
    lammps_first = module.query_lammps_build_identity(
        lammps_cfg, workdir=tmp_path
    )
    wrapper.write_text("# wrapper revision two\n")
    lammps_second = module.query_lammps_build_identity(
        lammps_cfg, workdir=tmp_path
    )
    assert lammps_first["identity_sha256"] != lammps_second["identity_sha256"]
    assert any(
        row["token_index"] == 1
        and row["configured_token"] == str(wrapper)
        and row["sha256"] != next(
            other["sha256"]
            for other in lammps_second["command_file_identities"]
            if other["role"] == row["role"] and other["token_index"] == 1
        )
        for row in lammps_first["command_file_identities"]
        if row["role"] == "engine_command"
    )

    cp2k_cfg = Cp2kConfig.model_validate(
        {"cp2k_cmd": [sys.executable, str(wrapper)]}
    )
    monkeypatch.setattr(
        module,
        "run_cmd",
        lambda cmd, **kwargs: (0, "CP2K version 2024.2\n", ""),
    )
    cp2k_first = module.query_cp2k_build_identity(cp2k_cfg, workdir=tmp_path)
    wrapper.write_text("# wrapper revision three\n")
    cp2k_second = module.query_cp2k_build_identity(cp2k_cfg, workdir=tmp_path)
    assert cp2k_first["identity_sha256"] != cp2k_second["identity_sha256"]


def test_engine_identity_rejects_unbound_module_symlink_and_probe_race(
    monkeypatch, tmp_path: Path
):
    import vitriflow.engine_identity as module
    from vitriflow.config import LammpsConfig

    module_cfg = LammpsConfig.model_validate(
        {"lammps_cmd": [sys.executable, "-m", "unbound_engine_module"]}
    )
    with pytest.raises(RuntimeError, match="unbound Python module"):
        module.query_lammps_build_identity(module_cfg, workdir=tmp_path)

    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("# stable\n")
    wrapper_link = tmp_path / "wrapper_link.py"
    wrapper_link.symlink_to(wrapper)
    symlink_cfg = LammpsConfig.model_validate(
        {"lammps_cmd": [sys.executable, str(wrapper_link)]}
    )
    with pytest.raises(RuntimeError, match="symbolic link"):
        module.query_lammps_build_identity(symlink_cfg, workdir=tmp_path)

    race_cfg = LammpsConfig.model_validate(
        {"lammps_cmd": [sys.executable, str(wrapper)]}
    )

    def mutate_during_probe(cmd, **kwargs):
        wrapper.write_text("# replaced during probe\n")
        return (
            0,
            "Large-scale Atomic/Molecular Massively Parallel Simulator - "
            "22 Jul 2025 - Update 4\n",
            "",
        )

    monkeypatch.setattr(module, "run_cmd", mutate_during_probe)
    with pytest.raises(RuntimeError, match="changed during build-identity query"):
        module.query_lammps_build_identity(race_cfg, workdir=tmp_path)


def test_end_of_execution_identity_guards_reject_wrapper_switch(
    monkeypatch, tmp_path: Path
):
    from vitriflow.engine_identity import (
        assert_engine_build_identity_bundle_unchanged,
        engine_build_identity_bundle,
    )
    from vitriflow.workflows import hpc

    first = _identity(monkeypatch, tmp_path, "22 Jul 2025 - Update 4")
    second = _identity(monkeypatch, tmp_path, "10 Sep 2025")
    first_bundle = engine_build_identity_bundle(
        primary_engine="lammps", identities={"lammps": first}
    )
    second_bundle = engine_build_identity_bundle(
        primary_engine="lammps", identities={"lammps": second}
    )
    with pytest.raises(RuntimeError, match="mixed engine builds"):
        assert_engine_build_identity_bundle_unchanged(
            first_bundle,
            second_bundle,
            context="during test execution",
        )
    with pytest.raises(RuntimeError, match="changed during production-box"):
        hpc._assert_task_engine_build_identity_unchanged(first, second)


def test_run_and_autotune_resume_fingerprints_reject_engine_build_switch(
    monkeypatch, tmp_path: Path
):
    from vitriflow.engine_identity import engine_build_identity_bundle
    from vitriflow.workflows import autotune as autotune_module
    from vitriflow.workflows import run as run_module

    cfg = _run_config()
    structure = tmp_path / "structure.data"
    structure.write_text("structure\n")
    plan = {
        "engine": "lammps",
        "structure_data": str(structure),
        "potential_config": cfg.kim.model_dump(mode="json"),
        "potential_lines": None,
        "production_cfg": cfg.autotune.production.model_dump(mode="json"),
    }
    first_identity = _identity(monkeypatch, tmp_path, "22 Jul 2025 - Update 4")
    first_bundle = engine_build_identity_bundle(
        primary_engine="lammps", identities={"lammps": first_identity}
    )
    second_identity = _identity(monkeypatch, tmp_path, "10 Sep 2025")
    second_bundle = engine_build_identity_bundle(
        primary_engine="lammps", identities={"lammps": second_identity}
    )

    first_run = run_module._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
        engine_build_identities=first_bundle,
    )
    second_run = run_module._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
        engine_build_identities=second_bundle,
    )
    with pytest.raises(RuntimeError, match="differ"):
        run_module._validate_run_resume_fingerprint(
            {"resume_fingerprint": first_run}, second_run
        )

    first_autotune = autotune_module._build_autotune_resume_fingerprint(
        config=cfg,
        outdir=tmp_path,
        selected_structure=structure,
        production_plan=plan,
        engine_build_identities=first_bundle,
    )
    previous = {
        "resume_fingerprint": first_autotune,
        "production_plan": plan,
        "size_scan": {"base_data": str(structure)},
    }
    with pytest.raises(RuntimeError, match="differ"):
        autotune_module._validate_autotune_resume_fingerprint(
            previous,
            config=cfg,
            outdir=tmp_path,
            engine_build_identities=second_bundle,
        )


def test_custom_schedule_resume_fingerprint_rejects_engine_build_switch(
    monkeypatch, tmp_path: Path
):
    from vitriflow.engine_identity import engine_build_identity_bundle
    from vitriflow.workflows import custom_schedule as custom

    cfg = _run_config()
    schedule = custom.CustomSchedule(
        stages=(
            custom.CustomStageConfig(
                name="sample",
                temperature_start_K=300.0,
                temperature_stop_K=300.0,
                steps=1,
                role="melt",
            ),
        ),
        analysis_roles={"melt": "sample", "quench": "sample", "relax": "sample"},
    )
    first = _identity(monkeypatch, tmp_path, "22 Jul 2025 - Update 4")
    second = _identity(monkeypatch, tmp_path, "10 Sep 2025")
    first_bundle = engine_build_identity_bundle(
        primary_engine="lammps", identities={"lammps": first}
    )
    second_bundle = engine_build_identity_bundle(
        primary_engine="lammps", identities={"lammps": second}
    )

    def fingerprint(bundle):
        return custom._build_resume_fingerprint(
            config=cfg,
            schedule=schedule,
            analysis_roles=schedule.analysis_roles,
            steps={"sample": 1},
            sched_report={"stages": [{"name": "sample", "steps": 1}]},
            time_unit_ps=1.0,
            md_pressure=0.0,
            lammps_units="metal",
            config_path=None,
            engine_build_identities=bundle,
        )

    first_fingerprint = fingerprint(first_bundle)
    second_fingerprint = fingerprint(second_bundle)
    assert first_fingerprint["schema"].endswith(".v3")
    assert (
        first_fingerprint["payload"]["runner"]["engine_build_identities"]
        == first_bundle
    )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        custom._validate_resume_fingerprint_or_raise(
            {"resume_fingerprint": first_fingerprint},
            second_fingerprint,
            outdir=tmp_path,
        )


def test_external_worker_identities_are_homogeneous_and_resume_bound(
    monkeypatch, tmp_path: Path
):
    from vitriflow.engine_identity import homogeneous_successful_task_engine_identity

    first = _identity(monkeypatch, tmp_path, "22 Jul 2025 - Update 4")
    second = _identity(monkeypatch, tmp_path, "10 Sep 2025")
    assert homogeneous_successful_task_engine_identity(
        [
            {"status": "ok", "engine_build_identity": first},
            {"status": "failed"},
            {"status": "success", "engine_build_identity": first},
        ],
        expected=first,
    ) == first
    with pytest.raises(RuntimeError, match="heterogeneous"):
        homogeneous_successful_task_engine_identity(
            [
                {"status": "ok", "engine_build_identity": first},
                {"status": "ok", "engine_build_identity": second},
            ]
        )
    with pytest.raises(RuntimeError, match="resumed"):
        homogeneous_successful_task_engine_identity(
            [{"status": "ok", "engine_build_identity": second}],
            expected=first,
        )
    with pytest.raises(RuntimeError, match="no verified engine identity"):
        homogeneous_successful_task_engine_identity([{"status": "ok"}])


def test_hpc_collector_rejects_heterogeneous_and_resume_switched_workers(
    monkeypatch, tmp_path: Path
):
    from vitriflow.workflows import hpc

    first = _identity(monkeypatch, tmp_path, "22 Jul 2025 - Update 4")
    second = _identity(monkeypatch, tmp_path, "10 Sep 2025")
    prod = tmp_path / "production"
    prod.mkdir()

    monkeypatch.setattr(hpc, "validate_task_result_integrity", lambda *args, **kwargs: None)
    monkeypatch.setattr(hpc, "_cached_task_result_is_reusable", lambda **kwargs: True)
    for box_id, identity in ((1, first), (2, second)):
        box = prod / f"box_{box_id:03d}"
        box.mkdir()
        task = {"schema": "vitriflow.box_task.v1", "task": {"box": box_id}}
        (box / "task.json").write_text(json.dumps(task))
        result = {
            "status": "ok",
            "task_manifest_sha256": hpc._task_manifest_digest(task),
            "engine_build_identity": identity,
        }
        (box / "task_result.json").write_text(json.dumps(result))

    with pytest.raises(RuntimeError, match="heterogeneous"):
        hpc.validate_external_task_engine_identities(
            production_dir=prod,
            box_ids=[1, 2],
        )

    (prod / "box_002" / "task_result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "task_manifest_sha256": hpc._task_manifest_digest(
                    json.loads((prod / "box_002" / "task.json").read_text())
                ),
                "engine_build_identity": first,
            }
        )
    )
    assert hpc.validate_external_task_engine_identities(
        production_dir=prod,
        box_ids=[1, 2],
        resume_state={"n_boxes_accepted": 1, "engine_build_identity": first},
    ) == first
    with pytest.raises(RuntimeError, match="resumed"):
        hpc.validate_external_task_engine_identities(
            production_dir=prod,
            box_ids=[1, 2],
            resume_state={"n_boxes_accepted": 1, "engine_build_identity": second},
        )
    with pytest.raises(RuntimeError, match="no protected worker engine identity"):
        hpc.validate_external_task_engine_identities(
            production_dir=prod,
            box_ids=[1, 2],
            resume_state={"n_boxes_accepted": 1},
        )


def test_standalone_output_analysis_rejects_missing_or_heterogeneous_current_engines(
    monkeypatch, tmp_path: Path
):
    from vitriflow.workflows import hpc, output_analysis

    first = _identity(monkeypatch, tmp_path, "22 Jul 2025 - Update 4")
    second = _identity(monkeypatch, tmp_path, "10 Sep 2025")

    def write(name: str, identity):
        path = tmp_path / name
        payload = {
            "schema": hpc.TASK_RESULT_SCHEMA,
            "runtime": hpc._runtime_identity(),
            "status": "ok",
            "box": 1,
            "engine_build_identity_end_verified": True,
        }
        if identity is not None:
            payload["engine_build_identity"] = identity
        path.write_text(json.dumps(hpc.seal_task_result(payload)))
        return path

    first_path = write("first.json", first)
    same_path = write("same.json", first)
    second_path = write("second.json", second)
    missing_path = write("missing.json", None)

    assert output_analysis._validated_current_task_engine_identity(
        [first_path, same_path]
    ) == first
    with pytest.raises(ValueError, match="heterogeneous"):
        output_analysis._validated_current_task_engine_identity(
            [first_path, second_path]
        )
    with pytest.raises(ValueError, match="no verified engine identity"):
        output_analysis._validated_current_task_engine_identity([missing_path])


def test_task_result_v2_is_integrity_checked_but_never_current():
    from vitriflow.workflows.resume_integrity import (
        TASK_RESULT_INTEGRITY_SCHEMA,
        TASK_RESULT_SCHEMA,
        canonical_json_sha256,
        seal_task_result,
        validate_task_result_integrity,
    )

    legacy = {
        "schema": "vitriflow.box_task_result.v2",
        "status": "ok",
        "box": 1,
    }
    legacy["result_integrity"] = {
        "schema": TASK_RESULT_INTEGRITY_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "payload_sha256": canonical_json_sha256(legacy),
    }
    assert validate_task_result_integrity(legacy, require_current=False) is False
    with pytest.raises(RuntimeError, match="legacy"):
        validate_task_result_integrity(legacy, require_current=True)

    tampered = dict(legacy)
    tampered["box"] = 2
    with pytest.raises(RuntimeError, match="modified or corrupted"):
        validate_task_result_integrity(tampered, require_current=False)

    current_without_end_probe = seal_task_result(
        {"schema": TASK_RESULT_SCHEMA, "status": "ok", "box": 1}
    )
    with pytest.raises(RuntimeError, match="end-of-task engine verification"):
        validate_task_result_integrity(
            current_without_end_probe,
            require_current=False,
        )
