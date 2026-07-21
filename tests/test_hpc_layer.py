from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _lammps_config():
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
                "metrics": {
                    "enabled": True,
                    "type_to_species": ["Al"],
                    "pairs": [{"pair": ["Al", "Al"]}],
                    "collect_during_production_stages": False,
                    "elastic": {"enabled": False},
                },
                "production": {"enabled": True, "min_boxes": 2, "max_boxes": 3, "batch_boxes": 1},
            },
        }
    )


def _cp2k_config():
    from vitriflow.config import RunConfig

    return RunConfig.model_validate(
        {
            "engine": "cp2k",
            "cp2k": {
                "exec": "cp2k.psmp",
                "kind_settings": {"H": {"basis_set": "DZVP-MOLOPT-SR-GTH", "potential": "GTH-PBE"}},
            },
            "structure": {"generate": {"method": "random", "formula": "H2", "n_formula_units": 1}},
            "autotune": {
                "metrics": {
                    "enabled": True,
                    "type_to_species": ["H"],
                    "pairs": [{"pair": ["H", "H"]}],
                    "collect_during_production_stages": False,
                    "elastic": {"enabled": False},
                },
                "production": {"enabled": True, "min_boxes": 1, "max_boxes": 1, "batch_boxes": 1},
            },
        }
    )


def _plan_dict(tmp_path: Path, cfg, *, engine: str, basename: str) -> dict[str, object]:
    base = tmp_path / basename
    base.write_text("placeholder\n")
    pot_cfg = cfg.kim.model_dump(mode="json") if getattr(cfg, "kim", None) is not None else None
    return {
        "schema": "vitriflow.production_plan.v1",
        "engine": engine,
        "structure_data": str(base),
        "T_high": 1200.0,
        "high_total_steps": 5000,
        "t_final": 300.0,
        "chosen_rate": 10.0,
        "cooling_rate_ps": 10.0,
        "replicate": [2, 2, 2],
        "pressure": 0.0,
        "md_use": cfg.md.model_dump(mode="json"),
        "potential_config": pot_cfg,
        "potential_lines": ["pair_style kim ..."] if pot_cfg is not None else None,
        "core_repulsion": {"enabled": False} if pot_cfg is not None else None,
        "type_to_species": list(cfg.autotune.metrics.type_to_species or []),
        "metrics_cfg": cfg.autotune.metrics.model_dump(mode="json"),
        "effective_metrics": {"source": "plan"},
        "production_cfg": cfg.autotune.production.model_dump(mode="json"),
        "convergence_cfg": cfg.autotune.convergence.model_dump(mode="json"),
        "cutoffs_rate": [{"pair": [1, 1], "cutoff": 3.0}],
        "cutoffs_size": [{"pair": [1, 1], "cutoff": 3.2}],
        "preferred_cutoffs": [{"pair": [1, 1], "cutoff": 3.2}],
        "quench_steps": 111,
        "relax_steps": 222,
        "msd_every": 77,
        "seed_base": 24680,
        "time_unit_ps": 1.0,
        "sampling_hint": {"Tm": 900.0, "freeze_temperature": 700.0},
        "execution_mode": "adaptive",
        "source_kind": "autotune",
    }


def _import_hpc_with_stubs(monkeypatch):
    prod_mod = types.ModuleType("vitriflow.workflows.production_common")
    prod_mod.plan_production_stage_diagnostics = lambda **kwargs: {
        "need_stage_dump": {"melt": True, "quench": True, "relax": True},
        "dump_every": 100,
        "quench_dump_every": 50,
        "quench_window_steps_range": None,
    }
    prod_mod.resolve_production_relax_dump_settings = lambda **kwargs: {
        "write_dump": True,
        "dump_every": 100,
        "tail_dump_frames": None,
        "tail_dump_stride": None,
        "mode": "full",
    }
    prod_mod.resolve_production_time_unit_ps = lambda **kwargs: 1.0
    prod_mod.resolve_production_warmup_duration_ps = lambda *, prod_cfg: float(getattr(prod_cfg, "warmup_duration_ps", 5.0))
    prod_mod.resolve_production_warmup_start_temperature = lambda *, prod_cfg, T_high=None: float(getattr(prod_cfg, "warmup_start_temperature", 300.0))
    prod_mod.resolve_production_warmup_steps = lambda *, prod_cfg, md_timestep, time_unit_ps: int(round(float(getattr(prod_cfg, "warmup_duration_ps", 5.0)) / (float(md_timestep) * float(time_unit_ps))))
    prod_mod.production_plan_to_dict = lambda plan, relative_to=None: dict(plan)
    prod_mod.graph_analysis_requested = lambda metrics: False
    prod_mod.build_ensemble_cdf_sidecar = lambda boxes, spec=None: {}
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", prod_mod)
    sys.modules.pop("vitriflow.workflows.hpc", None)
    module = importlib.import_module("vitriflow.workflows.hpc")
    # These orchestration tests synthesize minimal task results and do not
    # exercise result authentication.  Dedicated engine-identity tests cover
    # the real homogeneous-worker collector contract.
    monkeypatch.setattr(
        module,
        "validate_external_task_engine_identities",
        lambda *, expected_current=None, **kwargs: expected_current,
    )
    return module


def test_materialize_external_production_writes_task_manifests_and_submission_scripts(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    template = tmp_path / "job.slurm.in"
    template.write_text("#!/bin/bash\n#SBATCH -J {{JOB_NAME}}\ncd {{TASK_DIR}}\n{{EXECUTE_CMD}}\n")

    res = hpc_mod.materialize_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
        job_template=template,
    )

    prod_dir = tmp_path / "production"
    assert res["planned_boxes"] == 3
    assert (prod_dir / "box_001" / "task.json").exists()
    assert (prod_dir / "box_001" / "preview" / "melt.in.lammps").exists()
    assert (prod_dir / "box_001" / "submit.slurm").exists()
    assert "sbatch" in (prod_dir / "submit_all.sh").read_text()

    task = json.loads((prod_dir / "box_001" / "task.json").read_text())
    import vitriflow

    assert task["runtime"]["vitriflow_version"] == vitriflow.__version__
    assert task["input_manifest"]["schema"] == "vitriflow.task_inputs.v2"
    assert task["task"]["task_json"].endswith("box_001/task.json")
    assert task["task"]["preview_inputs"]
    dataset = json.loads((prod_dir / "output_dataset.json").read_text())
    assert dataset["planned_boxes"] == 3
    assert len(dataset["boxes"]) == 3
    assert dataset["source_root"] == str(prod_dir)
    first = dataset["boxes"][0]
    assert first["box_dir"] == "box_001"
    assert first["task_result"] == "box_001/task_result.json"
    assert (prod_dir / first["box_dir"]).resolve() == (prod_dir / "box_001").resolve()
    assert (prod_dir / first["task_result"]).resolve() == (
        prod_dir / "box_001" / "task_result.json"
    ).resolve()
    (prod_dir / first["task_result"]).write_text(
        json.dumps({"status": "ok", "box": 1})
    )
    import vitriflow.workflows.output_analysis as output_analysis

    discovered, preanalysed, rejected, _meta = output_analysis.discover_output_dataset(
        prod_dir / "output_dataset.json"
    )
    assert preanalysed == [] and rejected == []
    assert len(discovered) == 3
    assert discovered[0].box_dir == prod_dir / "box_001"
    assert discovered[0].task_result == prod_dir / "box_001" / "task_result.json"


def test_materialize_external_production_rejects_redirected_production_root(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (tmp_path / "production").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="must not contain a symbolic link"):
        hpc_mod.materialize_external_production(
            config=cfg,
            outdir=tmp_path,
            plan=plan,
        )
    assert not any(outside.iterdir())


def test_materialize_external_production_rejects_redirected_box(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    production = tmp_path / "production"
    production.mkdir()
    outside = tmp_path / "outside_box"
    outside.mkdir()
    try:
        (production / "box_001").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="must not contain a symbolic link"):
        hpc_mod.materialize_external_production(
            config=cfg,
            outdir=tmp_path,
            plan=plan,
        )
    assert not any(outside.iterdir())


@pytest.mark.parametrize("redirected_name", ["input", "preview", "warmup"])
def test_materialize_external_production_rejects_redirected_task_subdirectory(
    monkeypatch, tmp_path: Path, redirected_name: str
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    box = tmp_path / "production" / "box_001"
    box.mkdir(parents=True)
    outside = tmp_path / f"outside_{redirected_name}"
    outside.mkdir()
    try:
        (box / redirected_name).symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="must not contain a symbolic link"):
        hpc_mod.materialize_external_production(
            config=cfg,
            outdir=tmp_path,
            plan=plan,
        )
    assert not any(outside.iterdir())


@pytest.mark.parametrize(
    "field",
    ["box_dir", "task_json", "task_result", "input_snapshot"],
)
def test_execute_task_rejects_manifest_path_escape_before_lock_or_execution(
    monkeypatch, tmp_path: Path, field: str
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    hpc_mod.materialize_external_production(
        config=cfg, outdir=tmp_path, plan=plan
    )
    task_json = tmp_path / "production" / "box_001" / "task.json"
    task = json.loads(task_json.read_text())
    outside = tmp_path / "outside_task_escape"
    outside.mkdir()
    outside_input = outside / "input.data"
    outside_input.write_text("outside\n")
    replacements = {
        "box_dir": str(outside),
        "task_json": str(outside / "task.json"),
        "task_result": str(outside / "task_result.json"),
        "input_snapshot": str(outside_input),
    }
    task["task"][field] = replacements[field]
    task_json.write_text(json.dumps(task, indent=2))

    monkeypatch.setattr(
        hpc_mod,
        "_task_engine_build_identity",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("engine execution was reached before path validation")
        ),
    )
    with pytest.raises(RuntimeError, match="External task"):
        hpc_mod.execute_production_box_task(task_json)
    assert not (outside / ".vitriflow.lock").exists()
    assert sorted(path.name for path in outside.iterdir()) == ["input.data"]


def test_execute_task_rejects_symlink_manifest_before_lock(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    hpc_mod.materialize_external_production(
        config=cfg, outdir=tmp_path, plan=plan
    )
    task_json = tmp_path / "production" / "box_001" / "task.json"
    alias = tmp_path / "task_alias.json"
    try:
        alias.symlink_to(task_json)
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="regular non-symlink"):
        hpc_mod.execute_production_box_task(alias)
    assert not (tmp_path / ".vitriflow.lock").exists()


def test_materialize_external_production_cp2k_preview_is_json_serializable(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _cp2k_config()
    plan = _plan_dict(tmp_path, cfg, engine="cp2k", basename="base.xyz")

    hpc_mod.materialize_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
        job_template=None,
    )

    preview_path = tmp_path / "production" / "box_001" / "preview" / "stage_specs.json"
    assert preview_path.exists()
    preview = json.loads(preview_path.read_text())
    assert preview["engine"] == "cp2k"
    assert isinstance(preview["stages"], list) and len(preview["stages"]) == 4
    assert all(isinstance(stage["input_data"], str) for stage in preview["stages"])


def test_external_fixed_count_summary_preserves_terminal_posthoc_degree(
    monkeypatch, tmp_path: Path
):
    from vitriflow.config import ProductionEnsembleConfig

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "min_boxes": 2,
        "max_boxes": 2,
        "check_convergence": False,
    }
    prod_cfg = ProductionEnsembleConfig.model_validate(plan["production_cfg"])
    report = {
        "status": "fixed_n_terminal_posthoc_assessed",
        "sampling_design": "fixed_n",
        "assessment_role": "terminal_posthoc_diagnostic",
        "assessment_performed": True,
        "posthoc_criterion_met": False,
        "posthoc_failed_items": [
            {
                "section": "ci",
                "name": "scalar:density",
                "reason": "tolerance_not_met",
            }
        ],
        "achieved_convergence_degree": {
            "n_boxes": 2,
            "overall_active": {
                "worst_tolerance_utilization_ratio": 1.5,
            },
        },
        "convergence_degree": {
            "overall": {
                "n_checked": 1,
                "n_passed": 0,
                "pass_fraction": 0.0,
            }
        },
        "used_for_stopping": False,
        "stopping_status": "fixed_count_unassessed",
    }

    state = hpc_mod._summarise_analysis_as_production_state(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
        prod_cfg=prod_cfg,
        analysis={
            "status": "ok",
            "n_boxes": 2,
            "n_boxes_accepted": 2,
            "n_boxes_rejected": 0,
            "n_boxes_total": 2,
            "boxes": [{"box": 1}, {"box": 2}],
            "rejected_boxes": [],
            "convergence": report,
        },
        mode="full-run",
        max_parallel_boxes=1,
        job_template=None,
        planned_boxes=2,
        convergence_streak=0,
        required_convergence_streak=1,
        last_convergence_evaluated_n_boxes_total=2,
        last_convergence_evaluated_n_boxes_accepted=2,
        convergence_look_history=[],
        converged_md=False,
        converged=True,
    )

    assert state["status"] == "ok"
    assert state["converged"] is None
    assert state["convergence_status"] == "fixed_count_unassessed"
    assert state["convergence_inference_status"] == (
        "fixed_n_terminal_posthoc_not_sequentially_valid"
    )
    assert state["achieved_convergence_degree"]["n_boxes"] == 2
    assert state["posthoc_convergence_criterion_met"] is False
    assert state["posthoc_convergence_failed_items"] == report[
        "posthoc_failed_items"
    ]
    assert state["convergence"]["status"] == "fixed_n_terminal_posthoc_assessed"
    assert state["convergence"]["stopping_status"] == "fixed_count_unassessed"


def test_external_submission_scripts_shell_quote_paths(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    outdir = tmp_path / "run with spaces;touch SHOULD_NOT_EXIST"
    outdir.mkdir()
    cfg = _lammps_config()
    plan = _plan_dict(outdir, cfg, engine="lammps", basename="base data.data")
    template = tmp_path / "job.slurm.in"
    template.write_text("#!/bin/bash\ncd {{TASK_DIR}}\n{{EXECUTE_CMD}}\n")

    hpc_mod.materialize_external_production(
        config=cfg,
        outdir=outdir,
        plan=plan,
        job_template=template,
        n_boxes=1,
    )

    submit = (outdir / "production" / "box_001" / "submit.slurm").read_text()
    submit_all = (outdir / "production" / "submit_all.sh").read_text()
    assert "cd '" in submit
    assert "--task '" in submit
    assert 'cd -- "$(dirname -- "$0")"' in submit_all
    assert "sbatch box_001/submit.slurm" in submit_all


def test_external_materialization_preserves_requested_stage_diagnostics(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["metrics_cfg"] = {
        **dict(plan["metrics_cfg"]),
        "collect_during_production_stages": True,
        "stage_timeseries_make_plot": True,
    }

    outdir = tmp_path / "external"
    result = hpc_mod.materialize_external_production(
        config=cfg,
        outdir=outdir,
        plan=plan,
        n_boxes=1,
    )
    task = json.loads(
        (outdir / "production" / "box_001" / "task.json").read_text()
    )
    diagnostic_plan = task["diagnostic_plan"]
    assert diagnostic_plan["stage_metrics"] == {
        "enabled": True,
        "roles": ["melt", "quench", "relax"],
        "plot_required": True,
    }
    assert result["diagnostic_plan"] == diagnostic_plan


def test_external_materialization_preserves_requested_elastic_diagnostics(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["metrics_cfg"] = {
        **dict(plan["metrics_cfg"]),
        "collect_during_production_stages": False,
        "elastic": {
            "enabled": True,
            "run_on_relax": True,
            "collect_during_production_stages": False,
        },
    }

    outdir = tmp_path / "external"
    hpc_mod.materialize_external_production(
        config=cfg,
        outdir=outdir,
        plan=plan,
        n_boxes=1,
    )
    task = json.loads(
        (outdir / "production" / "box_001" / "task.json").read_text()
    )
    roles = task["diagnostic_plan"]["elastic_screens"]["roles"]
    assert roles["melt"]["enabled"] is False
    assert roles["relax"] == {
        "enabled": True,
        "strict": True,
        "plot_required": True,
    }
    series = task["diagnostic_plan"]["elastic_timeseries"]["roles"]
    assert all(value["enabled"] is False for value in series.values())


def test_external_task_executes_and_manifests_requested_diagnostic_families(
    monkeypatch, tmp_path: Path
):
    from vitriflow.runner import LammpsRunner

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["metrics_cfg"] = {
        **dict(plan["metrics_cfg"]),
        "collect_during_production_stages": True,
        "stage_timeseries_make_plot": True,
        "elastic": {
            "enabled": True,
            "run_on_relax": True,
            "collect_during_production_stages": True,
            "stage_timeseries_make_plot": True,
            "make_plot": True,
        },
    }
    box_dir = tmp_path / "production" / "box_001"
    outcomes = {}
    for role in ("warmup", "melt", "quench", "relax"):
        stage_dir = box_dir / role
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / f"{role}.data").write_text(f"{role}\n")
        outcomes[role] = {"output_data": f"{role}.data", "dump": None}

    def _relative(path: Path) -> str:
        return str(Path(path).relative_to(box_dir))

    def _fake_stage_metrics(**kwargs):
        stage_dir = Path(kwargs["stage_dir"])
        csv_path = stage_dir / "metrics_timeseries.csv"
        summary_path = stage_dir / "metrics_timeseries.json"
        plot_path = stage_dir / "metrics_timeseries.pdf"
        csv_path.write_text("Step,time\n0,0\n")
        summary_path.write_text('{"status":"ok"}\n')
        plot_path.write_bytes(b"pdf")
        return {
            "status": "ok",
            "csv": _relative(csv_path),
            "summary": _relative(summary_path),
            "plot": _relative(plot_path),
        }

    def _fake_elastic_screen(*_args, **kwargs):
        directory = Path(kwargs["stage_dir"]) / "elastic"
        directory.mkdir(parents=True, exist_ok=True)
        summary = directory / "elastic_screen.json"
        plot = directory / "elastic_screen.png"
        summary.write_text('{"status":"ok"}\n')
        plot.write_bytes(b"png")
        return {
            "status": "ok",
            "dir": _relative(directory),
            "summary": _relative(summary),
            "plot": _relative(plot),
        }

    def _fake_elastic_timeseries(*_args, **kwargs):
        directory = Path(kwargs["stage_dir"]) / "elastic_timeseries"
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "elastic_timeseries.csv"
        summary = directory / "elastic_timeseries.json"
        plot = directory / "elastic_timeseries.png"
        csv_path.write_text("Step,time\n0,0\n")
        summary.write_text('{"status":"ok"}\n')
        plot.write_bytes(b"png")
        return {
            "status": "ok",
            "dir": _relative(directory),
            "csv": _relative(csv_path),
            "summary": _relative(summary),
            "plot": _relative(plot),
        }

    monkeypatch.setattr(
        hpc_mod, "collect_stage_metrics_timeseries", _fake_stage_metrics
    )
    monkeypatch.setattr(hpc_mod, "run_elastic_screen_lammps", _fake_elastic_screen)
    monkeypatch.setattr(
        hpc_mod,
        "run_elastic_screen_timeseries_lammps",
        _fake_elastic_timeseries,
    )

    diagnostics = hpc_mod._run_task_production_diagnostics(
        config=cfg,
        plan=plan,
        box_dir=box_dir,
        stage_diag={
            "collect_elastic_series": {
                "melt": True,
                "quench": True,
                "relax": True,
            },
            "quench_window_steps_range": [10.0, 20.0],
        },
        runner=LammpsRunner(cfg.lammps),
        pot_cfg=cfg.kim,
        md_use=cfg.md,
        type_to_species=["Al"],
        potential_lines=["pair_style kim ..."],
        force_isotropic=False,
        outcomes=outcomes,
    )

    assert diagnostics["schema"] == "vitriflow.production_task_diagnostics.v1"
    assert diagnostics["status"] == "ok"
    assert set(diagnostics["stage_metrics"]) == {"melt", "quench", "relax"}
    assert diagnostics["elastic_screens"]["melt"] is None
    assert diagnostics["elastic_screens"]["relax"]["status"] == "ok"
    assert all(
        diagnostics["elastic_timeseries"][role]["status"] == "ok"
        for role in ("melt", "quench", "relax")
    )

    manifest = hpc_mod._build_task_artifact_manifest(
        box_dir=box_dir,
        outcomes=outcomes,
        diagnostics=diagnostics,
    )
    manifested = {entry["path"] for entry in manifest["files"]}
    assert "relax/metrics_timeseries.pdf" in manifested
    assert "relax/elastic/elastic_screen.json" in manifested
    assert "quench/elastic_timeseries/elastic_timeseries.csv" in manifested


def test_completed_external_diagnostics_reject_failed_strict_role(
    monkeypatch,
):
    import pytest

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    diagnostics = {
        "schema": "vitriflow.production_task_diagnostics.v1",
        "status": "degraded",
        "path_base": "task_box",
        "plan": {
            "schema": "vitriflow.production_task_diagnostic_plan.v1",
            "stage_metrics": {"enabled": False, "roles": [], "plot_required": False},
            "elastic_screens": {
                "roles": {
                    "relax": {
                        "enabled": True,
                        "strict": True,
                        "plot_required": True,
                    }
                }
            },
            "elastic_timeseries": {
                "roles": {
                    "melt": {"enabled": False},
                    "quench": {"enabled": False},
                    "relax": {"enabled": False},
                }
            },
        },
        "stage_metrics": None,
        "elastic_screens": {"relax": {"status": "failed", "error": "boom"}},
        "elastic_timeseries": {"melt": None, "quench": None, "relax": None},
    }

    with pytest.raises(RuntimeError, match="Strict elastic_screens role=relax"):
        hpc_mod._validate_completed_task_diagnostics(diagnostics)


def test_external_stage_diagnostics_reject_duplicate_protected_cutoffs_before_writes(
    monkeypatch, tmp_path: Path
):
    import pytest

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["metrics_cfg"] = {
        **dict(plan["metrics_cfg"]),
        "collect_during_production_stages": True,
    }
    plan["preferred_cutoffs"] = [
        {"pair": [1, 1], "cutoff": 3.0},
        {"pair": [1, 1], "cutoff": 3.0},
    ]
    outdir = tmp_path / "external"

    with pytest.raises(ValueError, match="Duplicate production-plan cutoff"):
        hpc_mod.materialize_external_production(
            config=cfg,
            outdir=outdir,
            plan=plan,
            n_boxes=1,
        )
    assert not (outdir / "production").exists()


def test_external_materialization_rejects_dft_refinement_before_writing_tasks(
    monkeypatch, tmp_path: Path
):
    import pytest

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "dft_opt": {"enabled": True},
    }

    with pytest.raises(ValueError, match="does not support production.dft_opt cell refinement"):
        hpc_mod.materialize_external_production(
            config=cfg,
            outdir=tmp_path / "external",
            plan=plan,
        )
    assert not (tmp_path / "external" / "production").exists()


def test_slurm_template_requires_execute_command_before_materialization(
    monkeypatch, tmp_path: Path
):
    import pytest

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    template = tmp_path / "broken.slurm.in"
    template.write_text("#!/bin/bash\ncd {{TASK_DIR}}\necho no-task\n")

    outdir = tmp_path / "external"
    with pytest.raises(ValueError, match="must contain .*EXECUTE_CMD"):
        hpc_mod.materialize_external_production(
            config=cfg,
            outdir=outdir,
            plan=plan,
            job_template=template,
        )
    assert not (outdir / "production").exists()


def test_slurm_template_rejects_unresolved_placeholders_before_materialization(
    monkeypatch, tmp_path: Path
):
    import pytest

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    template = tmp_path / "broken.slurm.in"
    template.write_text(
        "#!/bin/bash\n{{EXECUTE_CMD}}\n#SBATCH --partition={{UNKNOWN_PARTITION}}\n"
    )

    outdir = tmp_path / "external"
    with pytest.raises(ValueError, match="unresolved placeholder.*UNKNOWN_PARTITION"):
        hpc_mod.materialize_external_production(
            config=cfg,
            outdir=outdir,
            plan=plan,
            job_template=template,
        )
    assert not (outdir / "production").exists()


def test_full_run_external_production_batches_boxes_incrementally(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")

    materialized_counts: list[int] = []

    def _fake_materialize(*, config, outdir, plan, job_template=None, n_boxes=None, progress=None):
        prod_dir = Path(outdir) / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        count = int(n_boxes or 0)
        materialized_counts.append(count)
        for box_id in range(1, count + 1):
            box_dir = prod_dir / f"box_{box_id:03d}"
            box_dir.mkdir(parents=True, exist_ok=True)
            task_json = box_dir / "task.json"
            if not task_json.exists():
                task_json.write_text(json.dumps({"task": {"box": box_id}}, indent=2))
        return {"planned_boxes": count}

    def _fake_execute(task_json: Path, **_kwargs):
        box_dir = Path(task_json).parent
        task_result = box_dir / "task_result.json"
        task_result.write_text(json.dumps({"status": "ok", "box": int(box_dir.name.split("_")[-1])}, indent=2))
        return json.loads(task_result.read_text())

    analyses = iter(
        [
            {
                "status": "ok",
                "converged": False,
                "n_boxes": 2,
                "n_boxes_accepted": 2,
                "n_boxes_rejected": 0,
                "n_boxes_total": 2,
                "boxes": [{"box": 1}, {"box": 2}],
                "rejected_boxes": [],
                "convergence": {"ok": False},
                "cutoffs": [{"pair": [1, 1], "cutoff": 3.2}],
            },
            {
                "status": "ok",
                "converged": True,
                "n_boxes": 3,
                "n_boxes_accepted": 3,
                "n_boxes_rejected": 0,
                "n_boxes_total": 3,
                "boxes": [{"box": 1}, {"box": 2}, {"box": 3}],
                "rejected_boxes": [],
                "convergence": {
                    "ok": True,
                    "achieved_convergence_degree": {
                        "n_boxes": 3,
                        "overall_active": {
                            "worst_tolerance_utilization_ratio": 0.5
                        },
                    },
                    "convergence_degree": {
                        "overall": {
                            "n_checked": 2,
                            "n_passed": 2,
                            "pass_fraction": 1.0,
                        }
                    },
                },
                "cutoffs": [{"pair": [1, 1], "cutoff": 3.2}],
            },
        ]
    )

    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
    monkeypatch.setattr(hpc_mod, "_cached_task_result_is_reusable", lambda **kwargs: True)
    monkeypatch.setattr(hpc_mod, "execute_production_box_task", _fake_execute)
    monkeypatch.setattr(hpc_mod, "analyze_output_data", lambda **kwargs: next(analyses))

    res = hpc_mod.full_run_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
        job_template=None,
        max_parallel_boxes=1,
    )

    assert materialized_counts == [2, 3]
    assert res["converged"] is True
    assert res["n_boxes"] == 3
    assert res["execution"]["planned_boxes"] == 3


def test_full_run_collects_homogeneous_slurm_workers_without_login_engine_probe(
    monkeypatch,
    tmp_path: Path,
    mock_engine_build_identities,
):
    """Worker-only modules must not be required on the Slurm login node."""

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "min_boxes": 1,
        "max_boxes": 1,
        "batch_boxes": 1,
        "check_convergence": False,
    }
    worker_identity = mock_engine_build_identities["identities"]["lammps"]

    def _fake_materialize(
        *, config, outdir, plan, job_template=None, n_boxes=None, progress=None
    ):
        box_dir = Path(outdir) / "production" / "box_001"
        box_dir.mkdir(parents=True, exist_ok=True)
        (box_dir / "task.json").write_text(
            json.dumps({"task": {"box": 1}})
        )
        (box_dir / "task_result.json").write_text(
            json.dumps({"status": "ok", "box": 1})
        )
        return {"planned_boxes": 1}

    def _collect_workers(*, expected_current=None, resume_state=None, **_kwargs):
        assert expected_current is None
        assert resume_state is None
        return worker_identity

    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
    monkeypatch.setattr(
        hpc_mod,
        "_task_engine_build_identity",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("login-node engine probe was attempted")
        ),
    )
    monkeypatch.setattr(
        hpc_mod, "_cached_task_result_is_reusable", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        hpc_mod, "validate_external_task_engine_identities", _collect_workers
    )
    monkeypatch.setattr(
        hpc_mod,
        "analyze_output_data",
        lambda **_kwargs: {
            "status": "ok",
            "converged": False,
            "n_boxes": 1,
            "n_boxes_accepted": 1,
            "n_boxes_rejected": 0,
            "n_boxes_total": 1,
            "boxes": [{"box": 1}],
            "rejected_boxes": [],
            "convergence": {},
        },
    )

    result = hpc_mod.full_run_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
    )

    assert result["status"] == "ok"
    assert result["engine_build_identity"] == worker_identity
    assert result["engine_build_identity_status"] == "verified_homogeneous_workers"


def test_external_precompleted_maximum_is_replayed_as_genuine_prefix_looks(
    monkeypatch, tmp_path: Path
):
    """A Slurm batch may finish all tasks before the collector starts.

    The collector must still assess the configured 2-box then 3-box prefixes;
    seeing the same 3-box ensemble twice would be pseudo-replication.
    """

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "consecutive_converged_checks": 2,
    }
    prod_dir = tmp_path / "production"
    for box_id in (1, 2, 3):
        box_dir = prod_dir / f"box_{box_id:03d}"
        box_dir.mkdir(parents=True, exist_ok=True)
        (box_dir / "task_result.json").write_text(
            json.dumps({"status": "ok", "box": box_id})
        )

    active_prefix = 0
    analysed_prefixes: list[int] = []

    def _fake_materialize(
        *, config, outdir, plan, job_template=None, n_boxes=None, progress=None
    ):
        nonlocal active_prefix
        active_prefix = int(n_boxes or 0)
        for box_id in range(1, active_prefix + 1):
            box_dir = Path(outdir) / "production" / f"box_{box_id:03d}"
            (box_dir / "task.json").write_text(
                json.dumps({"task": {"box": box_id}})
            )
        return {"planned_boxes": active_prefix}

    def _fake_analysis(**kwargs):
        analysed_prefixes.append(active_prefix)
        boxes = [{"box": box_id} for box_id in range(1, active_prefix + 1)]
        return {
            "status": "ok",
            "converged": True,
            "n_boxes": active_prefix,
            "n_boxes_accepted": active_prefix,
            "n_boxes_rejected": 0,
            "n_boxes_total": active_prefix,
            "boxes": boxes,
            "rejected_boxes": [],
            "convergence": {"ok": True},
        }

    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
    monkeypatch.setattr(hpc_mod, "_cached_task_result_is_reusable", lambda **kwargs: True)
    monkeypatch.setattr(
        hpc_mod,
        "execute_production_box_task",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("precompleted Slurm task was unexpectedly rerun")
        ),
    )
    monkeypatch.setattr(hpc_mod, "analyze_output_data", _fake_analysis)

    result = hpc_mod.full_run_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
    )

    assert analysed_prefixes == [2, 3]
    assert result["converged"] is True
    assert result["convergence_streak"] == 2
    history = result["convergence_look_history"]
    assert [row["n_boxes_total"] for row in history] == [2, 3]
    assert [row["accepted_box_ids"] for row in history] == [[1, 2], [1, 2, 3]]
    assert all(row["advanced_streak"] for row in history)


def test_external_resume_rejects_changed_evidence_at_an_evaluated_prefix(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "consecutive_converged_checks": 2,
    }

    def _fake_materialize(
        *, config, outdir, plan, job_template=None, n_boxes=None, progress=None
    ):
        prod_dir = Path(outdir) / "production"
        for box_id in range(1, int(n_boxes or 0) + 1):
            box_dir = prod_dir / f"box_{box_id:03d}"
            box_dir.mkdir(parents=True, exist_ok=True)
            (box_dir / "task.json").write_text(
                json.dumps({"task": {"box": box_id}})
            )
            (box_dir / "task_result.json").write_text(
                json.dumps({"status": "ok", "box": box_id})
            )
        return {"planned_boxes": int(n_boxes or 0)}

    original_analysis = {
        "boxes": [{"box": 1, "metrics": {"density": 2.0}}, {"box": 2}],
    }
    ids, digest = hpc_mod._accepted_evidence_identity(original_analysis)
    resumed = {
        "required_convergence_streak": 2,
        "convergence_streak": 1,
        "converged_md": True,
        "last_convergence_evaluated_n_boxes_total": 2,
        "last_convergence_evaluated_n_boxes_accepted": 2,
        "convergence_look_history": [
            {
                "n_boxes_total": 2,
                "n_boxes_accepted": 2,
                "accepted_box_ids": ids,
                "accepted_evidence_sha256": digest,
                "criterion_met": True,
                "advanced_streak": True,
                "convergence_streak_after": 1,
            }
        ],
    }
    changed = {
        "status": "ok",
        "converged": True,
        "n_boxes": 2,
        "n_boxes_accepted": 2,
        "n_boxes_rejected": 0,
        "n_boxes_total": 2,
        "boxes": [{"box": 1, "metrics": {"density": 2.1}}, {"box": 2}],
        "rejected_boxes": [],
        "convergence": {"ok": True},
    }
    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
    monkeypatch.setattr(
        hpc_mod, "_cached_task_result_is_reusable", lambda **kwargs: True
    )
    monkeypatch.setattr(hpc_mod, "analyze_output_data", lambda **kwargs: changed)

    with pytest.raises(RuntimeError, match="different accepted evidence"):
        hpc_mod.full_run_external_production(
            config=cfg,
            outdir=tmp_path,
            plan=plan,
            resume_state=resumed,
        )


def test_external_adaptive_resume_preserves_streak_and_requires_new_batch(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "consecutive_converged_checks": 2,
    }

    materialized_counts: list[int] = []

    def _fake_materialize(*, config, outdir, plan, job_template=None, n_boxes=None, progress=None):
        prod_dir = Path(outdir) / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        count = int(n_boxes or 0)
        materialized_counts.append(count)
        for box_id in range(1, count + 1):
            box_dir = prod_dir / f"box_{box_id:03d}"
            box_dir.mkdir(parents=True, exist_ok=True)
            (box_dir / "task.json").write_text(
                json.dumps({"task": {"box": box_id}}, indent=2)
            )
        return {"planned_boxes": count}

    def _fake_execute(task_json: Path, **_kwargs):
        box_dir = Path(task_json).parent
        result = {"status": "ok", "box": int(box_dir.name.split("_")[-1])}
        (box_dir / "task_result.json").write_text(json.dumps(result))
        return result

    analyses = iter(
        [
            {
                "status": "ok",
                "converged": True,
                "n_boxes": 2,
                "n_boxes_accepted": 2,
                "n_boxes_rejected": 0,
                "n_boxes_total": 2,
                "boxes": [{"box": 1}, {"box": 2}],
                "rejected_boxes": [],
                "convergence": {"ok": True},
            },
            {
                "status": "ok",
                "converged": True,
                "n_boxes": 3,
                "n_boxes_accepted": 3,
                "n_boxes_rejected": 0,
                "n_boxes_total": 3,
                "boxes": [{"box": 1}, {"box": 2}, {"box": 3}],
                "rejected_boxes": [],
                "convergence": {
                    "ok": True,
                    "achieved_convergence_degree": {
                        "n_boxes": 3,
                        "overall_active": {
                            "worst_tolerance_utilization_ratio": 0.5
                        },
                    },
                    "convergence_degree": {
                        "overall": {
                            "n_checked": 2,
                            "n_passed": 2,
                            "pass_fraction": 1.0,
                        }
                    },
                },
            },
        ]
    )
    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
    monkeypatch.setattr(hpc_mod, "execute_production_box_task", _fake_execute)
    monkeypatch.setattr(hpc_mod, "analyze_output_data", lambda **kwargs: next(analyses))

    checkpoints: list[dict] = []
    resumed = {
        "required_convergence_streak": 2,
        "convergence_streak": 1,
        "converged_md": True,
        "last_convergence_evaluated_n_boxes_total": 2,
        "last_convergence_evaluated_n_boxes_accepted": 2,
    }
    result = hpc_mod.full_run_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
        resume_state=resumed,
        checkpoint_cb=lambda state: checkpoints.append(state),
    )

    assert materialized_counts == [2, 3]
    assert checkpoints[0]["convergence_streak"] == 1
    assert checkpoints[0]["converged"] is False
    assert result["status"] == "ok"
    assert result["convergence_streak"] == 2
    assert result["last_convergence_evaluated_n_boxes_total"] == 3
    assert result["convergence_status"] == "converged"
    assert result["convergence_inference_status"] == (
        "criterion_met_repeated_looks_not_sequentially_valid"
    )
    assert result["achieved_convergence_degree"]["n_boxes"] == 3
    assert result["convergence_criterion_coverage"]["overall"]["pass_fraction"] == 1.0


def test_external_resume_replays_checkpoint_prefix_before_stale_higher_prefix(
    monkeypatch, tmp_path: Path
):
    """A crash after expanding tasks must not skip a sequential look.

    Prefix two was checkpointed, then prefix three was materialised/completed
    before the controller died.  Resume must first rewrite the authoritative
    output_dataset.json to two, validate that prior look, and only then expose
    the already completed third box as the next genuine look.
    """

    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "min_boxes": 2,
        "max_boxes": 3,
        "batch_boxes": 1,
        "check_convergence": True,
        "consecutive_converged_checks": 1,
    }

    hpc_mod.materialize_external_production(
        config=cfg, outdir=tmp_path, plan=plan, n_boxes=3
    )
    for box_id in range(1, 4):
        result_path = (
            tmp_path / "production" / f"box_{box_id:03d}" / "task_result.json"
        )
        result_path.write_text(json.dumps({"status": "ok", "box": box_id}))

    real_materialize = hpc_mod.materialize_external_production
    materialized_prefixes: list[int] = []
    analysed_prefixes: list[int] = []

    def tracked_materialize(**kwargs):
        materialized_prefixes.append(int(kwargs.get("n_boxes") or 0))
        return real_materialize(**kwargs)

    def analyse_authoritative_prefix(**_kwargs):
        dataset = json.loads(
            (tmp_path / "production" / "output_dataset.json").read_text()
        )
        count = int(dataset["planned_boxes"])
        analysed_prefixes.append(count)
        return {
            "status": "ok",
            "converged": count >= 3,
            "n_boxes": count,
            "n_boxes_accepted": count,
            "n_boxes_rejected": 0,
            "n_boxes_total": count,
            "boxes": [{"box": box_id} for box_id in range(1, count + 1)],
            "rejected_boxes": [],
            "convergence": {"ok": count >= 3},
        }

    monkeypatch.setattr(
        hpc_mod, "materialize_external_production", tracked_materialize
    )
    monkeypatch.setattr(
        hpc_mod, "_cached_task_result_is_reusable", lambda **_kwargs: True
    )
    monkeypatch.setattr(hpc_mod, "analyze_output_data", analyse_authoritative_prefix)

    result = hpc_mod.full_run_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
        resume_state={
            "required_convergence_streak": 1,
            "convergence_streak": 0,
            "converged_md": False,
            "last_convergence_evaluated_n_boxes_total": 2,
            "last_convergence_evaluated_n_boxes_accepted": 2,
        },
    )

    assert materialized_prefixes == [2, 3]
    assert analysed_prefixes == [2, 3]
    assert result["n_boxes_total"] == 3
    assert result["converged"] is True
    assert len(list((tmp_path / "production").glob("box_*"))) == 3


def test_external_resume_rejects_stale_converged_flag_against_look_history(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "consecutive_converged_checks": 2,
    }
    ids, digest = hpc_mod._accepted_evidence_identity(
        {"boxes": [{"box": 1}, {"box": 2}]}
    )

    with pytest.raises(RuntimeError, match="final look-history record"):
        hpc_mod.full_run_external_production(
            config=cfg,
            outdir=tmp_path,
            plan=plan,
            resume_state={
                "required_convergence_streak": 2,
                "convergence_streak": 0,
                "converged_md": True,
                "last_convergence_evaluated_n_boxes_total": 2,
                "last_convergence_evaluated_n_boxes_accepted": 2,
                "convergence_look_history": [
                    {
                        "n_boxes_total": 2,
                        "n_boxes_accepted": 2,
                        "accepted_box_ids": ids,
                        "accepted_evidence_sha256": digest,
                        "criterion_met": False,
                        "advanced_streak": False,
                        "convergence_streak_after": 0,
                    }
                ],
            },
        )


@pytest.mark.parametrize("bad_box_id", [True, 1.0, 1.5, "1", float("nan")])
def test_external_accepted_evidence_rejects_non_integral_box_ids(
    monkeypatch, bad_box_id
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)

    with pytest.raises(RuntimeError, match="box id"):
        hpc_mod._accepted_evidence_identity(
            {"boxes": [{"box": bad_box_id}]}
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("n_boxes_total", None),
        ("n_boxes_total", True),
        ("n_boxes_total", 1.0),
        ("n_boxes_accepted", "1"),
        ("convergence_streak_after", False),
        ("criterion_met", 1),
        ("criterion_met", "false"),
        ("advanced_streak", 0),
        ("advanced_streak", "true"),
    ],
)
def test_external_convergence_history_rejects_coerced_scalar_types(
    monkeypatch, field, bad_value
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    record = {
        "n_boxes_total": 1,
        "n_boxes_accepted": 1,
        "accepted_box_ids": [1],
        "accepted_evidence_sha256": "a" * 64,
        "criterion_met": True,
        "advanced_streak": True,
        "convergence_streak_after": 1,
    }
    record[field] = bad_value

    with pytest.raises(RuntimeError, match="look history record 0"):
        hpc_mod._validated_convergence_look_history([record])


def test_external_rejected_only_batch_does_not_advance_convergence_streak(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")
    plan["production_cfg"] = {
        **dict(plan["production_cfg"]),
        "consecutive_converged_checks": 2,
    }

    def _fake_materialize(*, config, outdir, plan, job_template=None, n_boxes=None, progress=None):
        prod_dir = Path(outdir) / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        for box_id in range(1, int(n_boxes or 0) + 1):
            box_dir = prod_dir / f"box_{box_id:03d}"
            box_dir.mkdir(parents=True, exist_ok=True)
            (box_dir / "task.json").write_text(
                json.dumps({"task": {"box": box_id}}, indent=2)
            )
        return {"planned_boxes": int(n_boxes or 0)}

    def _fake_execute(task_json: Path, **_kwargs):
        box_dir = Path(task_json).parent
        result = {"status": "ok", "box": int(box_dir.name.split("_")[-1])}
        (box_dir / "task_result.json").write_text(json.dumps(result))
        return result

    analyses = iter(
        [
            {
                "status": "ok",
                "converged": True,
                "n_boxes": 2,
                "n_boxes_accepted": 2,
                "n_boxes_rejected": 0,
                "n_boxes_total": 2,
                "boxes": [{"box": 1}, {"box": 2}],
                "rejected_boxes": [],
                "convergence": {"ok": True},
            },
            {
                "status": "ok",
                "converged": True,
                "n_boxes": 2,
                "n_boxes_accepted": 2,
                "n_boxes_rejected": 1,
                "n_boxes_total": 3,
                "boxes": [{"box": 1}, {"box": 2}],
                "rejected_boxes": [{"box": 3}],
                "convergence": {"ok": True},
            },
        ]
    )
    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
    monkeypatch.setattr(hpc_mod, "execute_production_box_task", _fake_execute)
    monkeypatch.setattr(hpc_mod, "analyze_output_data", lambda **kwargs: next(analyses))

    result = hpc_mod.full_run_external_production(
        config=cfg,
        outdir=tmp_path,
        plan=plan,
        resume_state={
            "required_convergence_streak": 2,
            "convergence_streak": 1,
            "converged_md": True,
            "last_convergence_evaluated_n_boxes_total": 2,
            "last_convergence_evaluated_n_boxes_accepted": 2,
        },
    )

    assert result["converged"] is False
    assert result["convergence_streak"] == 1
    assert result["last_convergence_evaluated_n_boxes_total"] == 3
    assert result["last_convergence_evaluated_n_boxes_accepted"] == 2


def test_external_analysis_path_normalisation_preserves_nested_diagnostics(
    monkeypatch, tmp_path: Path
):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    source = tmp_path / "production" / "box_001" / "relax" / "relax.data"
    source.parent.mkdir(parents=True)
    source.write_text("identity-locked source\n")
    identity = hpc_mod._file_identity(
        source,
        recorded_path="box_001/relax/relax.data",
    )
    entry = hpc_mod._normalise_analysis_entry_paths(
        {
            "box": 1,
            "paths": {
                "relax_data": "box_001/relax/relax.data",
                "coord_defects": {
                    "detail_json": "box_artifacts/box_001/coordination.json",
                    "error": "coordination analysis failed: no neighbours",
                },
            },
            "structure_manifest": {
                "structure_hash": "unchanged-hash",
                "source_path": "box_001/relax/relax.data",
                "source_file_identity": identity,
            },
            "structure": {
                "source_path": "box_001/relax/relax.data",
                "source_file_identity": identity,
            },
            "structure_embedding": {
                "source_path": "box_001/relax/relax.data",
                "manifest_sidecar": "structure_manifest.json",
                "structure_reference": "structure_references/box_000001.json",
                "verification": {"error": "diagnostic text is not a path"},
            },
        }
    )

    assert entry["paths"]["relax_data"] == "production/box_001/relax/relax.data"
    assert (
        entry["paths"]["coord_defects"]["detail_json"]
        == "production/box_artifacts/box_001/coordination.json"
    )
    assert (
        entry["paths"]["coord_defects"]["error"]
        == "coordination analysis failed: no neighbours"
    )
    assert entry["structure_manifest"]["structure_hash"] == "unchanged-hash"
    for key in ("structure_manifest", "structure", "structure_embedding"):
        assert entry[key]["source_path"] == "production/box_001/relax/relax.data"
        assert (tmp_path / entry[key]["source_path"]).is_file()
    assert (
        entry["structure_manifest"]["source_file_identity"]["path"]
        == "production/box_001/relax/relax.data"
    )
    assert hpc_mod._identity_matches(
        tmp_path / entry["structure_manifest"]["source_file_identity"]["path"],
        entry["structure_manifest"]["source_file_identity"],
    )
    assert (
        entry["structure_embedding"]["verification"]["error"]
        == "diagnostic text is not a path"
    )


def test_external_analysis_path_normalisation_rejects_parent_escape(monkeypatch):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)

    import pytest

    with pytest.raises(ValueError, match="escapes its production root"):
        hpc_mod._normalise_analysis_entry_paths(
            {"box": 1, "paths": {"relax_data": "../outside.data"}}
        )


def test_external_analysis_fails_if_planned_task_is_omitted(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")

    def _fake_materialize(*, outdir, n_boxes=None, **kwargs):
        prod_dir = Path(outdir) / "production"
        for box_id in range(1, int(n_boxes or 0) + 1):
            box_dir = prod_dir / f"box_{box_id:03d}"
            box_dir.mkdir(parents=True, exist_ok=True)
            (box_dir / "task.json").write_text(json.dumps({"task": {"box": box_id}}))
            (box_dir / "task_result.json").write_text(
                json.dumps({"status": "ok", "box": box_id})
            )
        return {"planned_boxes": int(n_boxes or 0)}

    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
    monkeypatch.setattr(hpc_mod, "_cached_task_result_is_reusable", lambda **kwargs: True)
    monkeypatch.setattr(
        hpc_mod,
        "analyze_output_data",
        lambda **kwargs: {
            "status": "ok",
            "converged": False,
            "n_boxes": 1,
            "n_boxes_accepted": 1,
            "n_boxes_rejected": 0,
            "n_boxes_total": 1,
            "boxes": [{"box": 1}],
            "rejected_boxes": [],
            "convergence": {"ok": False},
        },
    )

    with pytest.raises(RuntimeError, match="exact configured prefix"):
        hpc_mod.full_run_external_production(
            config=cfg,
            outdir=tmp_path,
            plan=plan,
        )


def test_materialize_external_production_refreshes_input_snapshot_when_source_changes(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")

    base = Path(plan["structure_data"])
    base.write_text("first\n")
    hpc_mod.materialize_external_production(config=cfg, outdir=tmp_path, plan=plan, job_template=None)

    snap = tmp_path / "production" / "box_001" / "input" / base.name
    assert snap.read_text() == "first\n"

    base.write_text("second\n")
    hpc_mod.materialize_external_production(config=cfg, outdir=tmp_path, plan=plan, job_template=None)

    assert snap.read_text() == "second\n"


def test_execute_production_box_task_refuses_unauthorized_stale_success_cache(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")

    hpc_mod.materialize_external_production(config=cfg, outdir=tmp_path, plan=plan, job_template=None)
    task_json = tmp_path / "production" / "box_001" / "task.json"
    task_result = tmp_path / "production" / "box_001" / "task_result.json"
    task_result.write_text(json.dumps({"status": "ok", "box": 1}, indent=2))

    import pytest

    before = task_result.read_bytes()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        hpc_mod.execute_production_box_task(task_json)
    assert task_result.read_bytes() == before


def test_explicit_full_run_retry_archives_exact_stale_result(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")

    hpc_mod.materialize_external_production(
        config=cfg, outdir=tmp_path, plan=plan, job_template=None
    )
    task_json = tmp_path / "production" / "box_001" / "task.json"
    task_result = tmp_path / "production" / "box_001" / "task_result.json"
    stale = b'{"status":"ok","box":1}\n'
    task_result.write_bytes(stale)
    task_data = json.loads(task_json.read_text())
    authorization = hpc_mod._task_result_retry_authorization(
        task_data=task_data,
        task_result=task_result,
    )

    def _boom(**kwargs):
        raise RuntimeError("authenticated re-run reached execution")

    monkeypatch.setattr(hpc_mod, "_stage_specs_for_box", _boom)
    import pytest

    with pytest.raises(RuntimeError, match="authenticated re-run reached execution"):
        hpc_mod.execute_production_box_task(
            task_json,
            retry_existing_result=authorization,
        )

    archive = task_result.with_name(
        "task_result.json.superseded-" + authorization["existing_result_sha256"][:16]
    )
    assert archive.read_bytes() == stale
    failed = json.loads(task_result.read_text())
    assert failed["status"] == "failed"
    assert failed["result_integrity"]["payload_sha256"]
