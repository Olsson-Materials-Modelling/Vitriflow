from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")
pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _write_box_artifacts(outdir: Path, box: int, *, text: str = "structure\n"):
    bdir = outdir / "production" / f"box_{box:03d}"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "relax.data").write_text(text)
    (bdir / "structure_snapshot.json").write_text(
        json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1})
    )
    manifest = {"structure_hash": f"structure-{box}"}
    (bdir / "structure_manifest.json").write_text(
        json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest]})
    )
    return {
        "relax_data": str((bdir / "relax.data").relative_to(outdir)),
        "structure_snapshot": str((bdir / "structure_snapshot.json").relative_to(outdir)),
        "structure_manifest": str((bdir / "structure_manifest.json").relative_to(outdir)),
    }, manifest


def _make_run_config():
    from vitriflow.config import RunConfig

    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "structure": {"generate": {"method": "random", "formula": "Al", "n_formula_units": 1}},
            "autotune": {
                "metrics": {"enabled": True, "pairs": [{"pair": ["Al", "Al"]}]},
                "production": {"enabled": True, "min_boxes": 2, "max_boxes": 2, "batch_boxes": 1},
            },
        }
    )


def _make_plan(cfg, base: Path, *, seed_base: int):
    from vitriflow.workflows.production_common import make_production_plan

    return make_production_plan(
        engine="lammps",
        structure_data=base,
        T_high=1200.0,
        high_total_steps=5000,
        t_final=300.0,
        chosen_rate=10.0,
        cooling_rate_ps=10.0,
        replicate=[2, 2, 2],
        pressure=0.0,
        md_use=cfg.md.model_dump(mode="json"),
        potential_config=cfg.kim.model_dump(mode="json"),
        potential_lines=["pair_style kim ..."],
        core_repulsion={"enabled": False},
        type_to_species=["Al"],
        metrics_cfg=cfg.autotune.metrics.model_dump(mode="json"),
        effective_metrics={"source": "plan"},
        production_cfg={
            **cfg.autotune.production.model_dump(mode="json"),
            "enabled": True,
            "min_boxes": 2,
            "max_boxes": 2,
            "batch_boxes": 1,
        },
        convergence_cfg=cfg.autotune.convergence.model_dump(mode="json"),
        cutoffs_rate={(1, 1): 3.0},
        cutoffs_size={(1, 1): 3.2},
        preferred_cutoffs={(1, 1): 3.2},
        quench_steps=90,
        relax_steps=222,
        msd_every=77,
        seed_base=seed_base,
        time_unit_ps=1.0,
        sampling_hint={"Tm": 900.0, "freeze_temperature": 700.0},
        execution_mode="adaptive",
        source_kind="autotune",
    )


def test_run_resume_mode_is_explicit_and_fail_closed(tmp_path: Path):
    from vitriflow.workflows.run import _resolve_run_resume_mode

    outdir = tmp_path / "out"
    outdir.mkdir()
    results = outdir / "run_results.json"

    with pytest.raises(RuntimeError, match="--resume was requested"):
        _resolve_run_resume_mode(outdir=outdir, results_path=results, resume=True)
    assert _resolve_run_resume_mode(
        outdir=outdir, results_path=results, resume=False
    ) is False

    orphan = outdir / "production"
    orphan.mkdir()
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        _resolve_run_resume_mode(outdir=outdir, results_path=results, resume=None)
    orphan.rmdir()

    results.write_text("{}")
    assert _resolve_run_resume_mode(
        outdir=outdir, results_path=results, resume=True
    ) is True
    assert _resolve_run_resume_mode(
        outdir=outdir, results_path=results, resume=None
    ) is True
    with pytest.raises(RuntimeError, match="--no-resume"):
        _resolve_run_resume_mode(outdir=outdir, results_path=results, resume=False)

    results.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    results.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symbolic link"):
        _resolve_run_resume_mode(outdir=outdir, results_path=results, resume=True)


def test_run_meltquench_auto_resume_uses_existing_run_results_plan_and_state(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")

    stored_plan = _make_plan(cfg, base, seed_base=24680)

    outdir = tmp_path / "run_out"
    outdir.mkdir()
    box1_paths, box1_manifest = _write_box_artifacts(outdir, 1, text="box-1\n")
    previous = {
        "status": "running",
        "execution_status": "running",
        "production": {
            "enabled": True,
            "status": "running",
            "execution_status": "running",
            "converged": False,
            "converged_md": False,
            "check_convergence": True,
            "resumable": True,
            "convergence_streak": 0,
            "required_convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": 1,
            "last_convergence_evaluated_n_boxes_accepted": 1,
            "min_boxes": 2,
            "n_boxes": 1,
            "n_boxes_accepted": 1,
            "n_boxes_rejected": 0,
            "n_boxes_total": 1,
            "boxes": [{"box": 1, "metrics": {}, "distributions": {}, "paths": box1_paths, "structure_manifest": box1_manifest, "seed_warmup": 1, "seed_melt": 2, "seed_quench": 3, "seed_relax": 4}],
            "rejected_boxes": [],
        },
        "metric_warnings": ["old metric warning"],
        "run_warnings": ["old run warning"],
        "production_plan": production_plan_to_dict(stored_plan, relative_to=outdir),
    }
    previous["resume_fingerprint"] = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=previous["production_plan"],
        outdir=outdir,
        external_mode="local",
    )
    previous["production"] = run_mod._attach_production_state_integrity(
        previous["production"], outdir=outdir
    )
    (outdir / "run_results.json").write_text(json.dumps(previous, indent=2))

    called: dict[str, object] = {}

    def _fake_exec(**kwargs):
        called.update(kwargs)
        box2_paths, box2_manifest = _write_box_artifacts(outdir, 2, text="box-2\n")
        return {
            "status": "not_converged",
            "converged": False,
            "converged_md": False,
            "check_convergence": True,
            "resumable": True,
            "convergence_streak": 0,
            "required_convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": 1,
            "last_convergence_evaluated_n_boxes_accepted": 1,
            "min_boxes": 2,
            "n_boxes": 2,
            "n_boxes_accepted": 2,
            "n_boxes_rejected": 0,
            "n_boxes_total": 2,
            "boxes": [
                dict(previous["production"]["boxes"][0]),
                {"box": 2, "density": 2.7, "metrics": {}, "distributions": {}, "paths": box2_paths, "structure_manifest": box2_manifest},
            ],
            "rejected_boxes": [],
        }

    monkeypatch.setattr(run_mod, "_run_production_executor", _fake_exec)
    monkeypatch.setattr(run_mod, "ensure_model_installed", lambda model: None)
    monkeypatch.setattr(
        run_mod,
        "prepare_initial_structure",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not prepare structure when resuming")),
    )
    monkeypatch.setattr(
        run_mod,
        "run_preflight",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run preflight when resuming")),
    )
    identity_queries: list[dict] = []
    original_identity_query = run_mod.query_engine_build_identities

    def _tracked_identity_query(*args, **kwargs):
        result = original_identity_query(*args, **kwargs)
        identity_queries.append(result)
        return result

    monkeypatch.setattr(
        run_mod, "query_engine_build_identities", _tracked_identity_query
    )

    summary = run_mod.run_meltquench(
        cfg,
        outdir,
        production_source={"production_plan": production_plan_to_dict(stored_plan, relative_to=tmp_path)},
        recommendation_base_dir=tmp_path,
    )

    assert called["seed_base"] == 24680
    assert called["resume_state"] == previous["production"]
    assert called["chosen_replicate"] == [2, 2, 2]
    assert called["quench_steps_override"] == 90
    assert summary["metric_warnings"] == ["old metric warning"]
    assert summary["run_warnings"] == ["old run warning"]
    assert summary["production_plan"]["seed_base"] == 24680
    assert len(identity_queries) == 2


def test_run_resume_rejects_a_conflicting_explicit_source(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")
    protected_plan = _make_plan(cfg, base, seed_base=24680)
    conflicting_plan = _make_plan(cfg, base, seed_base=13579)
    outdir = tmp_path / "run_out"
    outdir.mkdir()
    previous = {
        "status": "running",
        "execution_status": "running",
        "production": {
            "enabled": True,
            "status": "running",
            "execution_status": "running",
            "converged": False,
            "check_convergence": True,
            "resumable": True,
            "convergence_streak": 0,
            "required_convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": None,
            "last_convergence_evaluated_n_boxes_accepted": None,
            "min_boxes": 2,
            "n_boxes": 0,
            "n_boxes_accepted": 0,
            "n_boxes_rejected": 0,
            "n_boxes_total": 0,
            "boxes": [],
            "rejected_boxes": [],
        },
        "production_plan": production_plan_to_dict(protected_plan, relative_to=outdir),
    }
    previous["resume_fingerprint"] = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=previous["production_plan"],
        outdir=outdir,
        external_mode="local",
    )
    previous["production"] = run_mod._attach_production_state_integrity(
        previous["production"], outdir=outdir
    )
    (outdir / "run_results.json").write_text(json.dumps(previous))

    monkeypatch.setattr(
        run_mod,
        "_run_production_executor",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("conflicting plan must fail first")),
    )
    with pytest.raises(RuntimeError, match="differs from the plan protected"):
        run_mod.run_meltquench(
            cfg,
            outdir,
            production_source={
                "production_plan": production_plan_to_dict(
                    conflicting_plan, relative_to=tmp_path
                )
            },
            recommendation_base_dir=tmp_path,
            resume=True,
        )


def test_run_meltquench_resume_returns_cached_summary_when_already_complete(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    outdir = tmp_path / "run_out"
    outdir.mkdir()
    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")
    plan = _make_plan(cfg, base, seed_base=24680)
    box_paths, box_manifest = _write_box_artifacts(outdir, 1, text="box-1\n")
    source = outdir / box_paths["relax_data"]
    existing = {
        "status": "ok",
        "execution_status": "completed",
        "production": {
            "enabled": True,
            "status": "ok",
            "execution_status": "completed",
                "converged": True,
                "converged_md": True,
                "check_convergence": True,
                "resumable": True,
                "convergence_streak": 1,
                "required_convergence_streak": 1,
                "last_convergence_evaluated_n_boxes_total": 1,
                "last_convergence_evaluated_n_boxes_accepted": 1,
            "min_boxes": 1,
            "n_boxes": 1,
            "n_boxes_accepted": 1,
            "n_boxes_rejected": 0,
            "n_boxes_total": 1,
            "boxes": [{"box": 1, "distributions": {}, "paths": box_paths, "structure_manifest": box_manifest}],
            "rejected_boxes": [],
        },
        "metric_warnings": [],
        "run_warnings": [],
        "production_plan": production_plan_to_dict(plan, relative_to=outdir),
    }
    existing["resume_fingerprint"] = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=existing["production_plan"],
        outdir=outdir,
        external_mode="local",
    )
    existing["production"] = run_mod._attach_production_state_integrity(
        existing["production"], outdir=outdir
    )
    (outdir / "run_results.json").write_text(json.dumps(existing, indent=2))

    monkeypatch.setattr(
        run_mod,
        "_run_production_executor",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("completed run should not be re-executed")),
    )

    summary = run_mod.run_meltquench(cfg, outdir, resume=True)

    assert summary == existing

    source.unlink()
    with pytest.raises(RuntimeError, match="Required resume/provenance input is missing"):
        run_mod.run_meltquench(cfg, outdir, resume=True)


def test_run_resume_fingerprint_tracks_structure_and_potential_contents(tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    potential = tmp_path / "potential.xml"
    base.write_text("structure-v1\n")
    potential.write_text("potential-v1\n")
    plan = production_plan_to_dict(_make_plan(cfg, base, seed_base=24680))
    plan["potential_config"] = dict(plan["potential_config"] or {})
    plan["potential_config"]["files"] = [str(potential)]

    first = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    base.write_text("structure-v2\n")
    second = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    assert first["sha256"] != second["sha256"]

    base.write_text("structure-v1\n")
    potential.write_text("potential-v2\n")
    third = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    assert first["sha256"] != third["sha256"]

    line_only = tmp_path / "line_only.table"
    line_only.write_text("line-potential-v1\n")
    plan["potential_lines"] = [f"pair_coeff * * {line_only} TEST"]
    fourth = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    refs = fourth["payload"]["input_identities"]["potential_command_files"]
    assert [item["filename"] for item in refs] == [line_only.name]
    line_only.write_text("line-potential-v2\n")
    fifth = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    assert fourth["sha256"] != fifth["sha256"]


def test_run_resume_fingerprint_canonicalizes_relative_output_structure_path(
    monkeypatch,
    tmp_path: Path,
):
    """Fresh and resumed plans must name the same output structure identically."""

    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import (
        production_plan_from_dict,
        production_plan_to_dict,
    )

    monkeypatch.chdir(tmp_path)
    cfg = _make_run_config()
    outdir = Path("relative_run")
    structure = outdir / "structure" / "base.data"
    structure.parent.mkdir(parents=True)
    structure.write_text("structure-v1\n")

    fresh_plan = production_plan_to_dict(
        _make_plan(cfg, structure, seed_base=24680),
        relative_to=outdir,
    )
    assert fresh_plan["structure_data"] == "structure/base.data"

    # Public resume parses the stored relative plan beneath outdir, producing
    # an absolute runtime path, and then serializes the protected plan again.
    resumed_plan = production_plan_to_dict(
        production_plan_from_dict(fresh_plan, base_dir=outdir),
        relative_to=outdir,
    )
    assert resumed_plan == fresh_plan

    engine_identity = {
        "schema": "test.engine_build_identities.v1",
        "identity_sha256": "a" * 64,
    }
    fresh = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=fresh_plan,
        outdir=outdir,
        external_mode="local",
        engine_build_identities=engine_identity,
    )
    resumed = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=resumed_plan,
        outdir=outdir,
        external_mode="local",
        engine_build_identities=engine_identity,
    )
    assert resumed["sha256"] == fresh["sha256"]
    assert resumed["payload"] == fresh["payload"]

    structure.write_text("structure-v2\n")
    mutated = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=resumed_plan,
        outdir=outdir,
        external_mode="local",
        engine_build_identities=engine_identity,
    )
    assert mutated["sha256"] != fresh["sha256"]


def test_run_rejects_potential_mutated_during_local_execution(
    monkeypatch,
    tmp_path: Path,
):
    from vitriflow.config import RunConfig
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    potential = tmp_path / "model.table"
    potential.write_text("protected-potential-v1\n")
    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Al"],
                "commands": [
                    "pair_style table linear 1000",
                    f"pair_coeff * * {potential} MODEL",
                ],
                "files": [str(potential)],
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
                    "max_boxes": 1,
                    "batch_boxes": 1,
                },
            },
        }
    )
    structure = tmp_path / "base.data"
    structure.write_text("LAMMPS data file\n\n0 atoms\n")
    plan = production_plan_to_dict(_make_plan(cfg, structure, seed_base=24680))
    plan["potential_config"] = cfg.kim.model_dump(mode="json")
    plan["potential_lines"] = list(cfg.kim.commands)

    def mutate_then_return(**_kwargs):
        potential.write_text("protected-potential-v2\n")
        return {"status": "not_converged"}

    monkeypatch.setattr(run_mod, "_run_production_executor", mutate_then_return)
    outdir = tmp_path / "run_out"
    with pytest.raises(
        RuntimeError,
        match="scientific input bytes changed during execution",
    ):
        run_mod.run_meltquench(
            cfg,
            outdir,
            production_source={"production_plan": plan},
            recommendation_base_dir=tmp_path,
            resume=False,
        )
    assert not (outdir / "run_results.json").exists()


def test_run_resume_fingerprint_binds_external_job_template(tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    template = tmp_path / "job.slurm"
    base.write_text("structure\n")
    template.write_text("#!/bin/sh\n{{VITRIFLOW_COMMAND}}\n")
    plan = production_plan_to_dict(_make_plan(cfg, base, seed_base=24680))

    first = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="dry-run",
        job_template=template,
    )
    template.write_text("#!/bin/sh\n#SBATCH -N 1\n{{VITRIFLOW_COMMAND}}\n")
    second = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="dry-run",
        job_template=template,
    )
    assert first["sha256"] != second["sha256"]
    assert first["payload"]["job_template"]["sha256"] != second["payload"]["job_template"]["sha256"]

    with pytest.raises(ValueError, match="meaningful only for external"):
        run_mod._build_run_resume_fingerprint(
            config=cfg,
            production_plan=plan,
            outdir=tmp_path,
            external_mode="local",
            job_template=template,
        )


def test_run_resume_fingerprint_binds_cp2k_refinement_configuration(
    monkeypatch, tmp_path: Path
):
    from vitriflow.config import Cp2kConfig
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    cp2k_a = Cp2kConfig.model_validate(
        {
            "cp2k_cmd": "cp2k.psmp",
            "eps_scf": 1.0e-6,
            "kind_settings": {
                "Al": {"basis_set": "SZV-MOLOPT-SR-GTH", "potential": "GTH-PBE"}
            },
        }
    )
    cp2k_b = cp2k_a.model_copy(update={"eps_scf": 1.0e-7})
    cfg_a = cfg.model_copy(update={"cp2k": cp2k_a})
    cfg_b = cfg.model_copy(update={"cp2k": cp2k_b})
    base = tmp_path / "base.data"
    base.write_text("structure\n")
    plan = production_plan_to_dict(_make_plan(cfg, base, seed_base=24680))
    plan["production_cfg"] = dict(plan["production_cfg"])
    plan["production_cfg"]["dft_opt"] = {
        **dict(plan["production_cfg"].get("dft_opt", {}) or {}),
        "enabled": True,
    }
    monkeypatch.setattr(
        run_mod.Cp2kRunner,
        "resolved_data_files",
        lambda self, workdir, require=True: {},
    )

    first = run_mod._build_run_resume_fingerprint(
        config=cfg_a,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    second = run_mod._build_run_resume_fingerprint(
        config=cfg_b,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    assert first["sha256"] != second["sha256"]
    assert first["payload"]["cp2k_execution_config"]["eps_scf"] == pytest.approx(1.0e-6)
    assert second["payload"]["cp2k_execution_config"]["eps_scf"] == pytest.approx(1.0e-7)


def test_run_resume_fingerprint_allows_only_validated_dry_to_full_transition(
    tmp_path: Path,
):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")
    plan = production_plan_to_dict(_make_plan(cfg, base, seed_base=24680))
    dry = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="dry-run",
    )
    full = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="full-run",
    )
    previous = {
        "status": "planned",
        "execution_status": "planned",
        "resume_fingerprint": dry,
        "production": {
            "status": "planned",
            "execution_status": "planned",
            "resumable": True,
            "n_boxes_total": 0,
            "n_boxes_accepted": 0,
            "n_boxes_rejected": 0,
            "execution": {"mode": "dry-run"},
        },
    }

    with pytest.raises(RuntimeError, match="differ from the existing run"):
        run_mod._validate_run_resume_fingerprint(previous, full)
    run_mod._validate_run_resume_fingerprint(
        previous,
        full,
        allow_dry_run_to_full_run=True,
    )

    changed = json.loads(json.dumps(full))
    changed["payload"]["production_plan"]["seed_base"] += 1
    changed["sha256"] = run_mod._canonical_json_sha256(changed["payload"])
    with pytest.raises(RuntimeError, match="differ from the existing run"):
        run_mod._validate_run_resume_fingerprint(
            previous,
            changed,
            allow_dry_run_to_full_run=True,
        )


def test_run_resume_fingerprint_rejects_unmaterialised_potential_line_file(tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    base.write_text("structure\n")
    plan = production_plan_to_dict(_make_plan(cfg, base, seed_base=24680))
    plan["potential_lines"] = ["pair_coeff * * missing_model.table TEST"]
    with pytest.raises(FileNotFoundError, match="not materialised"):
        run_mod._build_run_resume_fingerprint(
            config=cfg,
            production_plan=plan,
            outdir=tmp_path,
            external_mode="local",
        )


def test_run_resume_fingerprint_resolves_or_rejects_potential_variables(tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    model = tmp_path / "variable_model.xml"
    base.write_text("structure\n")
    model.write_text("model-v1\n")
    plan = production_plan_to_dict(_make_plan(cfg, base, seed_base=24680))
    plan["potential_lines"] = [
        f"variable pot string {model}",
        "pair_coeff * * ${pot} TEST",
    ]

    first = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    assert [
        item["filename"]
        for item in first["payload"]["input_identities"]["potential_command_files"]
    ] == [model.name]
    model.write_text("model-v2\n")
    second = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )
    assert first["sha256"] != second["sha256"]

    plan["potential_lines"] = ["pair_coeff * * ${undefined_model} TEST"]
    with pytest.raises(ValueError, match="unresolved LAMMPS variable"):
        run_mod._build_run_resume_fingerprint(
            config=cfg,
            production_plan=plan,
            outdir=tmp_path,
            external_mode="local",
        )


def test_run_resume_fingerprint_rejects_missing_or_tampered_state(tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    base.write_text("structure\n")
    plan = production_plan_to_dict(_make_plan(cfg, base, seed_base=24680))
    current = run_mod._build_run_resume_fingerprint(
        config=cfg,
        production_plan=plan,
        outdir=tmp_path,
        external_mode="local",
    )

    with pytest.raises(RuntimeError, match="no run resume fingerprint"):
        run_mod._validate_run_resume_fingerprint({}, current)

    tampered = json.loads(json.dumps(current))
    tampered["payload"]["seed_scheme"] = "altered"
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        run_mod._validate_run_resume_fingerprint(
            {"resume_fingerprint": tampered},
            current,
        )


def test_production_resume_advances_rng_using_all_recorded_stage_seeds(monkeypatch, tmp_path: Path):
    from vitriflow.workflows.autotune import _run_production_ensemble
    from vitriflow.workflows.progress import CondensedProgressLog

    monkeypatch.setattr(
        "vitriflow.workflows.autotune.plan_production_stage_diagnostics",
        lambda **kwargs: {
            "dump_traj": False,
            "dump_every": 500,
            "collect_stage_metric_series": False,
            "collect_elastic_series": {"melt": False, "quench": False, "relax": False},
            "need_stage_dump": {"melt": False, "quench": False, "relax": False},
            "quench_dump_every": 250,
            "quench_window_steps_range": (0.0, 10.0),
        },
    )
    monkeypatch.setattr("vitriflow.workflows.autotune.required_pairs_from_metrics", lambda *args, **kwargs: [])
    monkeypatch.setattr("vitriflow.workflows.autotune.fixed_cutoffs_from_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr("vitriflow.workflows.autotune.should_run_elastic_screen", lambda *args, **kwargs: (False, False, None))
    monkeypatch.setattr(
        "vitriflow.workflows.autotune.should_collect_elastic_stage_timeseries",
        lambda *args, **kwargs: (False, False, None),
    )
    monkeypatch.setattr("vitriflow.workflows.autotune.validate_production_entry_against_spec", lambda *args, **kwargs: None)
    monkeypatch.setattr("vitriflow.workflows.autotune.check_production_convergence", lambda *args, **kwargs: (True, {"ok": True}))
    monkeypatch.setattr("vitriflow.workflows.autotune.summarize_production_crystal_motifs", lambda *args, **kwargs: {})

    def _fake_stage_run(*args, **kwargs):
        return SimpleNamespace(output_data="stage.data", density_mean=1.0, density_stderr=0.0)

    captured: dict[str, int] = {}

    def _fake_analyse_production_box(*, box_id: int, seeds, **kwargs):
        captured.update({str(k): int(v) for k, v in dict(seeds).items()})
        paths, manifest = _write_box_artifacts(tmp_path, box_id, text=f"box-{box_id}\n")
        return (
            {
                "box": int(box_id),
                "density": 1.0,
                "metrics": {},
                "distributions": {},
                "paths": paths,
                "structure_manifest": manifest,
            },
            {},
        )

    monkeypatch.setattr("vitriflow.workflows.autotune._stage_run", _fake_stage_run)
    monkeypatch.setattr("vitriflow.workflows.autotune.analyse_production_box", _fake_analyse_production_box)

    config = SimpleNamespace(
        random_seed=7,
        engine="lammps",
        md=SimpleNamespace(pressure=0.0),
        autotune=SimpleNamespace(
            production=SimpleNamespace(
                enabled=True,
                min_boxes=2,
                max_boxes=2,
                batch_boxes=1,
                check_convergence=True,
                dump_trajectory=False,
                dump_every_steps=500,
                dft_opt=None,
                exclude_coordination_defects=False,
                rejects_subdir="rejects",
                store_distributions=True,
                consecutive_converged_checks=1,
                bondlen_cdf_points=200,
                angle_cdf_points=180,
                warmup_start_temperature=300.0,
                warmup_duration_ps=5.0,
            ),
            convergence=SimpleNamespace(),
        ),
    )
    metrics_cfg = SimpleNamespace(enabled=True, time_average_frames=1, time_average_stride=1, elastic=SimpleNamespace())
    md_use = SimpleNamespace(timestep=0.001, stage_continuity="discontinuous", force_isotropic=False, pressure=0.0)
    q_cfg = SimpleNamespace(t_final=300.0, relax_steps=100)
    tm_cfg = SimpleNamespace(msd_every=100)
    base_data = tmp_path / "base.data"
    base_data.write_text("# placeholder\n")
    progress = CondensedProgressLog(tmp_path / "condensed.log")

    seed_base = 1234
    previous_paths, previous_manifest = _write_box_artifacts(tmp_path, 1, text="box-1\n")
    prev_prod = {
        "boxes": [
            {
                "box": 1,
                "metrics": {},
                "distributions": {},
                "paths": previous_paths,
                "structure_manifest": previous_manifest,
                "seed_warmup": 11,
                "seed_melt": 22,
                "seed_quench": 33,
                "seed_relax": 44,
            }
        ],
        "rejected_boxes": [],
    }
    from vitriflow.workflows.autotune import _attach_production_state_integrity

    prev_prod = _attach_production_state_integrity(prev_prod, outdir=tmp_path)

    _run_production_ensemble(
        config=config,
        outdir=tmp_path,
        runner=object(),
        pot_cfg=SimpleNamespace(),
        md_use=md_use,
        potential_lines=None,
        type_to_species=["Al"],
        metrics_cfg=metrics_cfg,
        tm_cfg=tm_cfg,
        q_cfg=q_cfg,
        size_base_data=base_data,
        chosen_replicate=[1, 1, 1],
        chosen_rate=10.0,
        dt_ref=0.001,
        dt_mq=0.001,
        cooling_rate_ps=10.0,
        cutoffs_rate={},
        cutoffs_size={},
        T_high=1200.0,
        high_total_steps=5000,
        resume_state=prev_prod,
        progress=progress,
        seed_base=seed_base,
    )

    rng = random.Random(seed_base)
    for _ in range(4):
        rng.randrange(1, 2**31 - 1)
    expected = {
        "warmup": rng.randrange(1, 2**31 - 1),
        "melt": rng.randrange(1, 2**31 - 1),
        "quench": rng.randrange(1, 2**31 - 1),
        "relax": rng.randrange(1, 2**31 - 1),
    }

    assert captured == expected
