from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")

_RELEASED_0_4_35_1_RUNTIME = {
    "schema": "vitriflow.runtime.v2",
    "vitriflow_version": "0.4.35.1",
    "package_content": {
        "schema": "vitriflow.package_content.v1",
        "algorithm": "sha256:length-prefixed-relative-path-and-content:v1",
        "sha256": "48f804d12295796a928a1257e56b6d91a7c250291cc4a67b092fc1346ca5f445",
        "file_count": 72,
    },
}

_RELEASED_0_4_36_0_RUNTIME = {
    "schema": "vitriflow.runtime.v2",
    "vitriflow_version": "0.4.36.0",
    "package_content": {
        "schema": "vitriflow.package_content.v1",
        "algorithm": "sha256:length-prefixed-relative-path-and-content:v1",
        "sha256": "6c098bcf84e2d01e36bca5d393e1cd0df570c3804a15f3911159f0426c75f186",
        "file_count": 72,
    },
}


def _pin_historical_0_4_36_runtime(monkeypatch, module) -> None:
    """Exercise the closed historical migration independently of this release.

    The production policy intentionally remains 0.4.35.1 -> 0.4.36.0 only.
    Pinning the fingerprint producer here prevents a later package version from
    accidentally broadening that allow-list merely to keep a historical test
    green.
    """

    monkeypatch.setattr(
        module,
        "runtime_identity",
        lambda: json.loads(json.dumps(_RELEASED_0_4_36_0_RUNTIME)),
    )


def _config(tmp_path: Path):
    from vitriflow.config import RunConfig

    source = tmp_path / "configured-input.data"
    source.write_text("configured-input-v1\n")
    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "structure": {"lammps_data": str(source)},
            "autotune": {
                "metrics": {"enabled": True, "type_to_species": ["Al"]},
                "production": {
                    "enabled": True,
                    "min_boxes": 1,
                    "max_boxes": 1,
                    "batch_boxes": 1,
                },
            },
        }
    )


def _active_zero_box_state(*, status: str = "starting") -> dict:
    return {
        "enabled": True,
        "status": status,
        "execution_status": status,
        "resumable": True,
        "n_boxes": 0,
        "n_boxes_total": 0,
        "n_boxes_accepted": 0,
        "n_boxes_rejected": 0,
        "boxes": [],
        "rejected_boxes": [],
    }


def _as_released_0_4_35_1(fingerprint: dict, *, hash_fn) -> dict:
    old = json.loads(json.dumps(fingerprint))
    old["payload"]["vitriflow_version"] = "0.4.35.1"
    old["payload"]["runtime"] = json.loads(
        json.dumps(_RELEASED_0_4_35_1_RUNTIME)
    )
    old["sha256"] = hash_fn(old["payload"])
    return old


def _autotune_case(tmp_path: Path, monkeypatch):
    from vitriflow.workflows import autotune as module

    _pin_historical_0_4_36_runtime(monkeypatch, module)

    cfg = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    outdir = Path("relative-autotune")
    selected = outdir / "structure" / "size_base.data"
    selected.parent.mkdir(parents=True)
    selected.write_text("selected-v1\n")
    plan = {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": "structure/size_base.data",
        "seed_base": 25924,
    }
    current = module._build_autotune_resume_fingerprint(
        config=cfg,
        outdir=outdir,
        selected_structure=selected,
        production_plan=plan,
    )
    stored = _as_released_0_4_35_1(
        current,
        hash_fn=module._canonical_json_sha256,
    )
    # Reproduce the pre-fix absolute/relative spelling difference while
    # preserving the exact same content-authenticated file.
    stored["payload"]["selected_structure"]["path"] = str(selected.resolve())
    stored["sha256"] = module._canonical_json_sha256(stored["payload"])
    previous = {
        "status": "running",
        "execution_status": "running",
        "production": _active_zero_box_state(),
        "size_scan": {"base_data": "structure/size_base.data"},
        "production_plan": plan,
        "resume_fingerprint": stored,
    }
    return module, cfg, outdir, selected, current, previous


def test_autotune_exact_0_4_35_1_first_box_checkpoint_migrates(
    tmp_path: Path,
    monkeypatch,
):
    module, cfg, outdir, selected, current, previous = _autotune_case(
        tmp_path,
        monkeypatch,
    )
    migrated = module._validate_autotune_resume_fingerprint(
        previous,
        config=cfg,
        outdir=outdir,
    )
    record = migrated.pop("release_resume_migration")
    assert migrated == current
    assert record["from_version"] == "0.4.35.1"
    assert record["to_version"] == "0.4.36.0"
    assert record["canonicalized_path_fields"] == ["selected_structure.path"]

    from vitriflow.workflows.resume_integrity import (
        validate_release_resume_migration,
    )

    validate_release_resume_migration(record)
    module._assert_autotune_terminal_fingerprint_unchanged(
        migrated,
        current,
        context="migrated test",
    )
    # A second resume is an ordinary same-release equality check.
    previous["resume_fingerprint"] = migrated
    rebuilt = module._validate_autotune_resume_fingerprint(
        previous,
        config=cfg,
        outdir=outdir,
    )
    assert rebuilt == current

    selected.write_text("selected-v2\n")
    with pytest.raises(RuntimeError, match="Cannot safely resume autotune"):
        module._validate_autotune_resume_fingerprint(
            previous,
            config=cfg,
            outdir=outdir,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "config",
        "plan",
        "engine",
        "seed_scheme",
        "selected_sha",
        "old_package",
        "committed_box",
    ],
)
def test_autotune_release_migration_rejects_every_non_allowlisted_difference(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
):
    module, cfg, outdir, _selected, _current, previous = _autotune_case(
        tmp_path,
        monkeypatch,
    )
    if mutation == "committed_box":
        previous["production"]["n_boxes"] = 1
        previous["production"]["n_boxes_total"] = 1
        previous["production"]["n_boxes_accepted"] = 1
        previous["production"]["boxes"] = [{"box": 1}]
    else:
        payload = previous["resume_fingerprint"]["payload"]
        if mutation == "config":
            payload["effective_config"]["random_seed"] += 1
        elif mutation == "plan":
            payload["production_plan"]["seed_base"] += 1
        elif mutation == "engine":
            payload["engine_build_identities"]["primary_engine"] = "cp2k"
        elif mutation == "seed_scheme":
            payload["seed_scheme"] = "different"
        elif mutation == "selected_sha":
            payload["selected_structure"]["sha256"] = "f" * 64
        elif mutation == "old_package":
            payload["runtime"]["package_content"]["sha256"] = "e" * 64
        previous["resume_fingerprint"]["sha256"] = module._canonical_json_sha256(
            payload
        )
    with pytest.raises(RuntimeError, match="Cannot safely resume autotune"):
        module._validate_autotune_resume_fingerprint(
            previous,
            config=cfg,
            outdir=outdir,
        )


def test_run_exact_0_4_35_1_first_box_checkpoint_migrates(
    tmp_path: Path,
    monkeypatch,
):
    from vitriflow.workflows import run as module

    _pin_historical_0_4_36_runtime(monkeypatch, module)

    cfg = _config(tmp_path)
    outdir = tmp_path / "run-output"
    structure = outdir / "structure" / "base.data"
    structure.parent.mkdir(parents=True)
    structure.write_text("selected-v1\n")
    plan = {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": "structure/base.data",
        "potential_config": cfg.kim.model_dump(mode="json"),
        "potential_lines": None,
        "production_cfg": cfg.autotune.production.model_dump(mode="json"),
        "seed_base": 25924,
    }
    current = module._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=outdir,
        external_mode="local",
    )
    stored = _as_released_0_4_35_1(
        current,
        hash_fn=module._canonical_json_sha256,
    )
    stored["payload"]["production_plan"]["structure_data"] = str(structure)
    stored["payload"]["input_identities"]["structure_data"][
        "configured_path"
    ] = str(structure)
    stored["sha256"] = module._canonical_json_sha256(stored["payload"])
    previous = {
        "status": "starting",
        "execution_status": "starting",
        "production": _active_zero_box_state(),
        "resume_fingerprint": stored,
    }
    record = module._validate_run_resume_fingerprint(
        previous,
        current,
        outdir=outdir,
    )
    assert record is not None
    assert set(record["canonicalized_path_fields"]) == {
        "production_plan.structure_data",
        "input_identities.structure_data.configured_path",
    }

    from vitriflow.workflows.resume_integrity import (
        validate_release_resume_migration,
    )

    validate_release_resume_migration(record)
    previous["resume_fingerprint"] = current
    assert (
        module._validate_run_resume_fingerprint(
            previous,
            current,
            outdir=outdir,
        )
        is None
    )

    changed = json.loads(json.dumps(current))
    changed["payload"]["production_plan"]["seed_base"] += 1
    changed["sha256"] = module._canonical_json_sha256(changed["payload"])
    with pytest.raises(RuntimeError, match="differ from the existing run"):
        module._validate_run_resume_fingerprint(
            previous,
            changed,
            outdir=outdir,
        )


def test_release_resume_migration_record_detects_modification(tmp_path: Path, monkeypatch):
    module, cfg, outdir, _selected, _current, previous = _autotune_case(
        tmp_path,
        monkeypatch,
    )
    migrated = module._validate_autotune_resume_fingerprint(
        previous,
        config=cfg,
        outdir=outdir,
    )
    record = migrated["release_resume_migration"]
    record["from_version"] = "forged"

    from vitriflow.workflows.resume_integrity import (
        validate_release_resume_migration,
    )

    with pytest.raises(RuntimeError, match="was modified"):
        validate_release_resume_migration(record)


def test_autotune_resume_installs_migrated_fingerprint_before_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    from vitriflow.kim import KimInstallResult
    from vitriflow.workflows import autotune as module
    from vitriflow.workflows.production_common import (
        make_production_plan,
        production_plan_to_dict,
    )

    _pin_historical_0_4_36_runtime(monkeypatch, module)

    cfg = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    outdir = Path("resume-output")
    selected = outdir / "structure" / "size_base.data"
    selected.parent.mkdir(parents=True)
    selected.write_text("selected-v1\n")
    plan = production_plan_to_dict(
        make_production_plan(
            engine="lammps",
            structure_data=selected,
            T_high=2000.0,
            high_total_steps=100,
            t_final=300.0,
            chosen_rate=10.0,
            cooling_rate_ps=10.0,
            replicate=[1, 1, 1],
            pressure=0.0,
            md_use=cfg.md.model_dump(mode="json"),
            potential_config=cfg.kim.model_dump(mode="json"),
            potential_lines=None,
            core_repulsion=cfg.kim.core_repulsion.model_dump(mode="json"),
            type_to_species=["Al"],
            metrics_cfg=cfg.autotune.metrics.model_dump(mode="json"),
            effective_metrics={"enabled": True},
            production_cfg=cfg.autotune.production.model_dump(mode="json"),
            convergence_cfg=cfg.autotune.convergence.model_dump(mode="json"),
            cutoffs_rate={},
            cutoffs_size={},
            preferred_cutoffs={},
            quench_steps=170,
            relax_steps=cfg.autotune.quench.relax_steps,
            msd_every=cfg.autotune.tm_scan.msd_every,
            seed_base=cfg.random_seed + 13579,
            time_unit_ps=1.0,
            sampling_hint=None,
            execution_mode="adaptive",
            source_kind="autotune",
        ),
        relative_to=outdir,
    )
    production = {
        **_active_zero_box_state(),
        "converged": False,
        "converged_md": False,
        "check_convergence": True,
        "convergence_streak": 0,
        "required_convergence_streak": 1,
        "last_convergence_evaluated_n_boxes_total": None,
        "last_convergence_evaluated_n_boxes_accepted": None,
        "min_boxes": 1,
        "max_boxes": 1,
        "batch_boxes": 1,
        "rate_K_per_time": 10.0,
        "replicate": [1, 1, 1],
        "T_high": 2000.0,
        "highT_steps": 100,
        "structure_data": "structure/size_base.data",
    }
    production = module._attach_production_state_integrity(
        production,
        outdir=outdir,
    )
    current = module._build_autotune_resume_fingerprint(
        config=cfg,
        outdir=outdir,
        selected_structure=selected,
        production_plan=plan,
    )
    stored = _as_released_0_4_35_1(
        current,
        hash_fn=module._canonical_json_sha256,
    )
    stored["payload"]["selected_structure"]["path"] = str(selected.resolve())
    stored["sha256"] = module._canonical_json_sha256(stored["payload"])
    previous = {
        "status": "running",
        "execution_status": "running",
        "production": production,
        "production_plan": plan,
        "size_scan": {"base_data": "structure/size_base.data"},
        "resume_fingerprint": stored,
    }

    snapshots: list[dict] = []

    def capture_write(_outdir, results):
        snapshots.append(json.loads(json.dumps(results)))

    monkeypatch.setattr(module, "write_autotune_outputs", capture_write)
    monkeypatch.setattr(
        module,
        "ensure_model_installed",
        lambda _model: KimInstallResult(attempted=False, success=True),
    )
    production_calls: list[dict] = []

    def fake_production(**kwargs):
        production_calls.append(dict(kwargs))
        kwargs["checkpoint_cb"](
            module._attach_production_state_integrity(
                {
                    **production,
                    "status": "running",
                    "execution_status": "running",
                    "state_integrity": production["state_integrity"],
                },
                outdir=outdir,
            )
        )
        return {
            **production,
            "status": "ok",
            "execution_status": "completed",
            "graph_outputs": {},
        }

    monkeypatch.setattr(module, "_run_production_ensemble", fake_production)
    result = module._autotune_resume_from_results(
        config=cfg,
        outdir=outdir,
        prev=previous,
    )

    assert production_calls
    assert snapshots
    for snapshot in snapshots:
        assert snapshot["resume_fingerprint"]["sha256"] == current["sha256"]
        assert snapshot["resume_fingerprint"]["payload"]["vitriflow_version"] == (
            "0.4.36.0"
        )
        assert len(snapshot["release_resume_migrations"]) == 1
    assert result["resume_fingerprint"]["sha256"] == current["sha256"]
