from __future__ import annotations

import pytest

pytest.importorskip("ase")

from pathlib import Path


def _config(*, warmup_start_temperature: float = 450.0, warmup_duration_ps: float = 5.0):
    from vitriflow.config import RunConfig

    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "md": {"timestep": 0.001},
            "structure": {"generate": {"method": "random", "formula": "Al", "n_formula_units": 1}},
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
                    "warmup_start_temperature": warmup_start_temperature,
                    "warmup_duration_ps": warmup_duration_ps,
                },
            },
        }
    )


def test_resolve_production_warmup_start_temperature_defaults_and_bounds():
    from vitriflow.config import ProductionEnsembleConfig
    from vitriflow.workflows.production_common import resolve_production_warmup_start_temperature

    assert resolve_production_warmup_start_temperature(prod_cfg=ProductionEnsembleConfig()) == 300.0
    assert (
        resolve_production_warmup_start_temperature(
            prod_cfg=ProductionEnsembleConfig(warmup_start_temperature=450.0),
            T_high=1200.0,
        )
        == 450.0
    )

    with pytest.raises(ValueError, match=r"<= T_high"):
        resolve_production_warmup_start_temperature(
            prod_cfg=ProductionEnsembleConfig(warmup_start_temperature=1300.0),
            T_high=1200.0,
        )


def test_resolve_production_warmup_steps_defaults_and_scaling():
    from vitriflow.config import ProductionEnsembleConfig
    from vitriflow.workflows.production_common import resolve_production_warmup_steps

    assert (
        resolve_production_warmup_steps(
            prod_cfg=ProductionEnsembleConfig(),
            md_timestep=0.001,
            time_unit_ps=1.0,
        )
        == 5000
    )
    assert (
        resolve_production_warmup_steps(
            prod_cfg=ProductionEnsembleConfig(warmup_duration_ps=7.5),
            md_timestep=1.0,
            time_unit_ps=0.001,
        )
        == 7500
    )


def test_run_summary_cfg_import_preserves_warmup_fields():
    from vitriflow.workflows.run import _production_cfg_from_summary

    base_cfg = _config(warmup_start_temperature=300.0, warmup_duration_ps=5.0).autotune.production
    merged = _production_cfg_from_summary(
        base_cfg,
        {"warmup_start_temperature": 425.0, "warmup_duration_ps": 8.0},
    )

    assert merged.warmup_start_temperature == 425.0
    assert merged.warmup_duration_ps == 8.0


def test_hpc_stage_specs_use_production_warmup_start_temperature(tmp_path: Path):
    from vitriflow.workflows import hpc as hpc_mod

    cfg = _config(warmup_start_temperature=450.0, warmup_duration_ps=5.0)
    base = tmp_path / "base.data"
    base.write_text("LAMMPS data file\n\n0 atoms\n")

    plan = {
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

    box_dir = tmp_path / "production" / "box_001"
    box_dir.mkdir(parents=True)
    input_snapshot = box_dir / "input" / base.name
    input_snapshot.parent.mkdir(parents=True)
    input_snapshot.write_text(base.read_text())

    stages, *_rest = hpc_mod._stage_specs_for_box(
        config=cfg,
        plan=plan,
        box_dir=box_dir,
        input_snapshot=input_snapshot,
    )

    assert [stage.name for stage in stages] == ["warmup", "melt", "quench", "relax"]
    assert stages[0].temperature_start == 450.0
    assert stages[0].temperature_stop == 1200.0
    assert stages[0].run_steps == 5000
    assert stages[1].temperature_start == 1200.0
    assert stages[1].temperature_stop == 1200.0
    assert stages[1].run_steps == 5000
    assert stages[2].temperature_start == 1200.0
    assert stages[2].temperature_stop == 300.0
