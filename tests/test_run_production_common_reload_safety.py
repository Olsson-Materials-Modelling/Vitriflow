from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def test_run_meltquench_uses_current_production_common_after_stubbed_import(monkeypatch, tmp_path: Path):
    from vitriflow.config import RunConfig

    cfg = RunConfig.model_validate(
        {
            "potential": {"kind": "mg2_sin", "user_units": "metal", "interactions": ["Si", "N"]},
            "structure": {"generate": {"method": "random", "formula": "Si3N4", "n_formula_units": 1}},
            "autotune": {
                "metrics": {"enabled": True, "pairs": [{"pair": ["Si", "N"]}]},
                "production": {"enabled": True, "min_boxes": 2, "consecutive_converged_checks": 1},
            },
        }
    )

    stale = types.ModuleType("vitriflow.workflows.production_common")
    stale.cutoffs_dict_from_any = lambda obj: (_ for _ in ()).throw(AssertionError("stale cutoffs helper used"))
    stale.make_production_plan = lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale make_production_plan used"))
    stale.production_plan_from_source = lambda source, base_dir=None: (_ for _ in ()).throw(AssertionError("stale production_plan_from_source used"))
    stale.production_plan_to_dict = lambda plan, relative_to=None: (_ for _ in ()).throw(AssertionError("stale production_plan_to_dict used"))

    structuregen_stub = types.ModuleType("vitriflow.structuregen")
    structuregen_stub.prepare_initial_structure = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("placeholder structure stub should be monkeypatched before use"))

    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", stale)
    monkeypatch.setitem(sys.modules, "vitriflow.structuregen", structuregen_stub)
    sys.modules.pop("vitriflow.workflows.run", None)
    run_mod = importlib.import_module("vitriflow.workflows.run")

    fresh = types.ModuleType("vitriflow.workflows.production_common")

    def _cutoffs_dict_from_any(obj):
        out = {}
        if isinstance(obj, list):
            for entry in obj:
                pair = entry.get("pair")
                if isinstance(pair, list) and len(pair) == 2:
                    a, b = int(pair[0]), int(pair[1])
                    out[(min(a, b), max(a, b))] = float(entry["cutoff"])
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, tuple) and len(key) == 2:
                    a, b = int(key[0]), int(key[1])
                    out[(min(a, b), max(a, b))] = float(value)
        return out

    fresh.cutoffs_dict_from_any = _cutoffs_dict_from_any
    fresh.make_production_plan = lambda **kwargs: SimpleNamespace(**kwargs)
    fresh.production_plan_from_source = lambda source, base_dir=None: None

    def _production_plan_to_dict(plan, relative_to=None):
        data = dict(plan.__dict__)
        if isinstance(data.get("structure_data"), Path):
            data["structure_data"] = str(data["structure_data"])
        if isinstance(data.get("replicate"), tuple):
            data["replicate"] = list(data["replicate"])
        return data

    fresh.production_plan_to_dict = _production_plan_to_dict

    # replace module importing
    # helpers keeping bindings
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", fresh)

    dummy_data = tmp_path / "initial.data"
    dummy_data.write_text("LAMMPS data file\n\n0 atoms\n")

    monkeypatch.setattr(run_mod, "prepare_initial_structure", lambda config, outdir: str(dummy_data))
    monkeypatch.setattr(
        run_mod,
        "resolve_effective_metrics_config",
        lambda metrics_cfg, structure_data, type_to_species, warn_fn, context: (metrics_cfg, {}, {"source": "test"}),
    )
    monkeypatch.setattr(run_mod, "ensure_model_installed", lambda model: None)
    monkeypatch.setattr(
        run_mod,
        "run_preflight",
        lambda runner, config, initial_data, outdir: SimpleNamespace(
            selected_timestep=1.0,
            selected_ensemble="npt",
            selected_tdamp=100.0,
            selected_pdamp=1000.0,
            potential_lines=None,
        ),
    )
    monkeypatch.setattr(
        run_mod,
        "_run_production_executor",
        lambda **kwargs: {
            "status": "ok",
            "converged": True,
            "n_boxes_total": 2,
            "boxes": [{"box": 1, "metrics": {}, "distributions": {}}, {"box": 2, "metrics": {}, "distributions": {}}],
            "rejected_boxes": [],
            "convergence": {"passed": True},
            "cutoffs": [{"pair": [1, 2], "cutoff": 2.5}],
        },
    )

    summary = run_mod.run_meltquench(cfg, tmp_path / "run_out", n_replicates=2)

    assert summary["production"]["n_boxes_total"] == 2
    assert summary["production_plan"]["execution_mode"] == "fixed"
    assert summary["production_plan"]["seed_base"] == cfg.random_seed + 13579
