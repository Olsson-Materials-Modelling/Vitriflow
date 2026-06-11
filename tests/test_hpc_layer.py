from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


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
                "metrics": {"enabled": True, "type_to_species": ["Al"], "pairs": [{"pair": ["Al", "Al"]}]},
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
                "metrics": {"enabled": True, "type_to_species": ["H"], "pairs": [{"pair": ["H", "H"]}]},
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
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", prod_mod)
    sys.modules.pop("vitriflow.workflows.hpc", None)
    return importlib.import_module("vitriflow.workflows.hpc")


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
    assert task["task"]["task_json"].endswith("box_001/task.json")
    assert task["task"]["preview_inputs"]
    dataset = json.loads((prod_dir / "output_dataset.json").read_text())
    assert dataset["planned_boxes"] == 3
    assert len(dataset["boxes"]) == 3


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

    def _fake_execute(task_json: Path):
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
                "convergence": {"ok": True},
                "cutoffs": [{"pair": [1, 1], "cutoff": 3.2}],
            },
        ]
    )

    monkeypatch.setattr(hpc_mod, "materialize_external_production", _fake_materialize)
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


def test_execute_production_box_task_does_not_reuse_stale_success_cache(monkeypatch, tmp_path: Path):
    hpc_mod = _import_hpc_with_stubs(monkeypatch)
    cfg = _lammps_config()
    plan = _plan_dict(tmp_path, cfg, engine="lammps", basename="base.data")

    hpc_mod.materialize_external_production(config=cfg, outdir=tmp_path, plan=plan, job_template=None)
    task_json = tmp_path / "production" / "box_001" / "task.json"
    task_result = tmp_path / "production" / "box_001" / "task_result.json"
    task_result.write_text(json.dumps({"status": "ok", "box": 1}, indent=2))

    def _boom(**kwargs):
        raise RuntimeError("re-run required")

    monkeypatch.setattr(hpc_mod, "_stage_specs_for_box", _boom)

    import pytest

    with pytest.raises(RuntimeError, match="re-run required"):
        hpc_mod.execute_production_box_task(task_json)

    failed = json.loads(task_result.read_text())
    assert failed["status"] == "failed"
    assert failed["task_manifest_sha256"]
