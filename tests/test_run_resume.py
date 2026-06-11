from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")


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
        quench_steps=111,
        relax_steps=222,
        msd_every=77,
        seed_base=seed_base,
        time_unit_ps=1.0,
        sampling_hint={"Tm": 900.0, "freeze_temperature": 700.0},
        execution_mode="adaptive",
        source_kind="autotune",
    )


def test_run_meltquench_auto_resume_uses_existing_run_results_plan_and_state(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import production_plan_to_dict

    cfg = _make_run_config()
    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")

    stored_plan = _make_plan(cfg, base, seed_base=24680)
    fresh_plan = _make_plan(cfg, base, seed_base=13579)

    outdir = tmp_path / "run_out"
    outdir.mkdir()
    previous = {
        "status": "running",
        "production": {
            "status": "running",
            "n_boxes_total": 1,
            "boxes": [{"box": 1, "metrics": {}, "distributions": {}, "seed_warmup": 1, "seed_melt": 2, "seed_quench": 3, "seed_relax": 4}],
            "rejected_boxes": [],
        },
        "metric_warnings": ["old metric warning"],
        "run_warnings": ["old run warning"],
        "production_plan": production_plan_to_dict(stored_plan, relative_to=outdir),
    }
    (outdir / "run_results.json").write_text(json.dumps(previous, indent=2))

    called: dict[str, object] = {}

    def _fake_exec(**kwargs):
        called.update(kwargs)
        return {
            "status": "ok",
            "boxes": [{"box": 2, "density": 2.7, "metrics": {}, "distributions": {}}],
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

    summary = run_mod.run_meltquench(
        cfg,
        outdir,
        production_source={"production_plan": production_plan_to_dict(fresh_plan, relative_to=tmp_path)},
        recommendation_base_dir=tmp_path,
    )

    assert called["seed_base"] == 24680
    assert called["resume_state"] == previous["production"]
    assert called["chosen_replicate"] == [2, 2, 2]
    assert called["quench_steps_override"] == 111
    assert summary["metric_warnings"] == ["old metric warning"]
    assert summary["run_warnings"] == ["old run warning"]
    assert summary["production_plan"]["seed_base"] == 24680


def test_run_meltquench_resume_returns_cached_summary_when_already_complete(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import run as run_mod

    cfg = _make_run_config()
    outdir = tmp_path / "run_out"
    outdir.mkdir()
    existing = {
        "status": "ok",
        "production": {"status": "ok", "boxes": [], "rejected_boxes": []},
        "metric_warnings": [],
        "run_warnings": [],
    }
    (outdir / "run_results.json").write_text(json.dumps(existing, indent=2))

    monkeypatch.setattr(
        run_mod,
        "_run_production_executor",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("completed run should not be re-executed")),
    )

    summary = run_mod.run_meltquench(cfg, outdir, resume=True)

    assert summary == existing


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
        return ({"box": int(box_id), "density": 1.0, "metrics": {}, "distributions": {}}, {})

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
    prev_prod = {
        "boxes": [
            {
                "box": 1,
                "metrics": {},
                "distributions": {},
                "seed_warmup": 11,
                "seed_melt": 22,
                "seed_quench": 33,
                "seed_relax": 44,
            }
        ],
        "rejected_boxes": [],
    }

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
