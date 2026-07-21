from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
import json

import pytest

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _config():
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
                "production": {"enabled": True, "min_boxes": 2, "max_boxes": 4, "batch_boxes": 2},
            },
        }
    )


def _plan_dict(tmp_path: Path, cfg) -> dict[str, object]:
    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")
    return {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": str(base),
        "T_high": 1200.0,
        "high_total_steps": 5000,
        "t_final": 300.0,
        "chosen_rate": 10.0,
        "cooling_rate_ps": 10.0,
        "replicate": [2, 2, 2],
        "pressure": 0.0,
        "md_use": cfg.md.model_dump(mode="json"),
        "potential_config": cfg.kim.model_dump(mode="json"),
        "potential_lines": ["pair_style kim ..."],
        "core_repulsion": {"enabled": False},
        "type_to_species": ["Al"],
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


def _install_run_stubs(monkeypatch, plan_dict: dict[str, object], *, dry_result=None, full_result=None):
    prod_mod = types.ModuleType("vitriflow.workflows.production_common")

    def _cutoffs_dict_from_any(obj):
        out = {}
        if isinstance(obj, list):
            for entry in obj:
                pair = entry.get("pair")
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = int(pair[0]), int(pair[1])
                    out[(min(a, b), max(a, b))] = float(entry["cutoff"])
        return out

    def _production_plan_from_source(source, base_dir=None):
        data = dict(source.get("production_plan", source))
        ns = types.SimpleNamespace(**data)
        ns.structure_data = Path(str(data["structure_data"]))
        ns.replicate = tuple(int(x) for x in data["replicate"])
        return ns

    def _production_plan_to_dict(plan, relative_to=None):
        return dict(plan_dict)

    prod_mod.cutoffs_dict_from_any = _cutoffs_dict_from_any
    prod_mod.make_production_plan = lambda **kwargs: types.SimpleNamespace(**kwargs)
    prod_mod.production_plan_from_source = _production_plan_from_source
    prod_mod.production_plan_to_dict = _production_plan_to_dict
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", prod_mod)

    structuregen_mod = types.ModuleType("vitriflow.structuregen")
    structuregen_mod.prepare_initial_structure = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not prepare structure when plan is supplied"))
    monkeypatch.setitem(sys.modules, "vitriflow.structuregen", structuregen_mod)

    hpc_mod = types.ModuleType("vitriflow.workflows.hpc")
    hpc_mod.dry_run_external_production = dry_result or (lambda **kwargs: (_ for _ in ()).throw(AssertionError("dry-run dispatcher was not expected")))
    hpc_mod.full_run_external_production = full_result or (lambda **kwargs: (_ for _ in ()).throw(AssertionError("full-run dispatcher was not expected")))
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.hpc", hpc_mod)

    sys.modules.pop("vitriflow.workflows.run", None)
    return importlib.import_module("vitriflow.workflows.run")


def test_run_dry_run_external_dispatches_to_hpc(monkeypatch, tmp_path: Path):
    cfg = _config()
    plan_dict = _plan_dict(tmp_path, cfg)
    called: dict[str, object] = {}

    def _fake_dry(**kwargs):
        called.update(kwargs)
        return {
            "enabled": True,
            "status": "planned",
            "converged": False,
            "n_boxes": 0,
            "boxes": [],
            "rejected_boxes": [],
            "execution": {"mode": "dry-run", "planned_boxes": 4},
        }

    run_mod = _install_run_stubs(monkeypatch, plan_dict, dry_result=_fake_dry)
    monkeypatch.setattr(run_mod, "_run_production_executor", lambda **kwargs: (_ for _ in ()).throw(AssertionError("local executor should not be used in dry-run mode")))
    monkeypatch.setattr(run_mod, "ensure_model_installed", lambda model: (_ for _ in ()).throw(AssertionError("dry-run should not install/check models")))
    monkeypatch.setattr(run_mod, "run_preflight", lambda *a, **k: (_ for _ in ()).throw(AssertionError("plan replay should not run preflight")))

    outdir = tmp_path / "run_out"
    summary = run_mod.run_meltquench(
        cfg,
        outdir,
        production_source={"production_plan": plan_dict},
        recommendation_base_dir=tmp_path,
        external_mode="dry-run",
    )

    assert summary["status"] == "planned"
    assert summary["production"]["execution"]["mode"] == "dry-run"
    assert called["outdir"] == outdir
    assert called["job_template"] is None
    assert called["plan"]["seed_base"] == 24680


def test_run_full_run_external_dispatches_parallel_limit(monkeypatch, tmp_path: Path):
    cfg = _config()
    plan_dict = _plan_dict(tmp_path, cfg)
    called: dict[str, object] = {}

    def _fake_full(**kwargs):
        called.update(kwargs)
        entries = []
        for box in (1, 2):
            source = Path(kwargs["outdir"]) / "production" / f"box_{box:03d}" / "relax.data"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"box-{box}\n")
            snapshot = source.parent / "structure_snapshot.json"
            manifest_path = source.parent / "structure_manifest.json"
            snapshot.write_text(json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1}))
            manifest = {"structure_hash": f"structure-{box}"}
            manifest_path.write_text(
                json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest]})
            )
            entries.append(
                {
                    "box": box,
                    "metrics": {},
                    "distributions": {},
                    "paths": {
                        "relax_data": str(source.relative_to(kwargs["outdir"])),
                        "structure_snapshot": str(snapshot.relative_to(kwargs["outdir"])),
                        "structure_manifest": str(manifest_path.relative_to(kwargs["outdir"])),
                    },
                    "structure_manifest": manifest,
                }
            )
        return {
            "enabled": True,
            "status": "ok",
            "converged": True,
            "converged_md": True,
            "check_convergence": True,
            "resumable": True,
            "convergence_streak": 1,
            "required_convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": 2,
            "last_convergence_evaluated_n_boxes_accepted": 2,
            "min_boxes": 2,
            "n_boxes": 2,
            "n_boxes_accepted": 2,
            "n_boxes_rejected": 0,
            "n_boxes_total": 2,
            "boxes": entries,
            "rejected_boxes": [],
            "execution": {"mode": "full-run", "planned_boxes": 2, "max_parallel_boxes": kwargs["max_parallel_boxes"]},
        }

    run_mod = _install_run_stubs(monkeypatch, plan_dict, full_result=_fake_full)
    monkeypatch.setattr(run_mod, "_run_production_executor", lambda **kwargs: (_ for _ in ()).throw(AssertionError("local executor should not be used in full-run mode")))
    monkeypatch.setattr(run_mod, "ensure_model_installed", lambda model: (_ for _ in ()).throw(AssertionError("full-run planning should not install/check models before dispatch")))
    monkeypatch.setattr(run_mod, "run_preflight", lambda *a, **k: (_ for _ in ()).throw(AssertionError("plan replay should not run preflight")))

    outdir = tmp_path / "run_out"
    summary = run_mod.run_meltquench(
        cfg,
        outdir,
        production_source={"production_plan": plan_dict},
        recommendation_base_dir=tmp_path,
        external_mode="full-run",
        max_parallel_boxes=3,
    )

    assert summary["status"] == "ok"
    assert summary["production"]["execution"]["mode"] == "full-run"
    assert called["outdir"] == outdir
    assert called["max_parallel_boxes"] == 3
    assert called["plan"]["seed_base"] == 24680
