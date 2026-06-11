from __future__ import annotations

import pytest

pytest.importorskip("ase")

from pathlib import Path


def test_run_replays_explicit_production_plan(monkeypatch, tmp_path: Path):
    from vitriflow.config import RunConfig
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import make_production_plan, production_plan_to_dict

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "structure": {"generate": {"method": "random", "formula": "Al", "n_formula_units": 1}},
            "autotune": {
                "metrics": {"enabled": True, "voids": {"enabled": True}, "pairs": [{"pair": ["Al", "Al"]}]},
                "production": {"enabled": True, "min_boxes": 5, "max_boxes": 10, "batch_boxes": 5},
            },
        }
    )

    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")

    plan = make_production_plan(
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
            "min_boxes": 3,
            "max_boxes": 3,
            "batch_boxes": 3,
        },
        convergence_cfg=cfg.autotune.convergence.model_dump(mode="json"),
        cutoffs_rate={(1, 1): 3.0},
        cutoffs_size={(1, 1): 3.2},
        preferred_cutoffs={(1, 1): 3.2},
        quench_steps=111,
        relax_steps=222,
        msd_every=77,
        seed_base=24680,
        time_unit_ps=1.0,
        sampling_hint={"Tm": 900.0, "freeze_temperature": 700.0},
        execution_mode="adaptive",
        source_kind="autotune",
    )

    called: dict[str, object] = {}

    def _fake_exec(**kwargs):
        called.update(kwargs)
        return {
            "status": "ok",
            "boxes": [{"box": 1, "density": 2.7, "distributions": {}, "metrics": {}}],
            "rejected_boxes": [],
            "cutoffs": [{"pair": [1, 1], "cutoff": 3.2}],
        }

    monkeypatch.setattr(run_mod, "_run_production_executor", _fake_exec)
    monkeypatch.setattr(run_mod, "ensure_model_installed", lambda model: None)
    monkeypatch.setattr(run_mod, "prepare_initial_structure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not prepare structure when plan is supplied")))
    monkeypatch.setattr(run_mod, "run_preflight", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run preflight when plan is supplied")))

    outdir = tmp_path / "run_out"
    summary = run_mod.run_meltquench(
        cfg,
        outdir,
        production_source={"production_plan": production_plan_to_dict(plan, relative_to=tmp_path)},
        recommendation_base_dir=tmp_path,
        n_replicates=99,
    )

    assert called["seed_base"] == 24680
    assert called["pressure_override"] == 0.0
    assert called["chosen_replicate"] == [2, 2, 2]
    assert called["quench_steps_override"] == 111
    assert called["relax_steps_override"] == 222
    assert called["potential_lines"] == ["pair_style kim ..."]
    assert summary["production_plan"]["seed_base"] == 24680
    assert summary["parameters"]["execution_mode"] == "adaptive"
