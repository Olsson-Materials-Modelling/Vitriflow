from __future__ import annotations

import pytest

pytest.importorskip("ase")
pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")

from pathlib import Path
import json


@pytest.mark.parametrize(
    ("potential_payload", "potential_lines", "expected_install_calls"),
    [
        pytest.param(
            {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            ["pair_style kim ..."],
            ["EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005"],
            id="kim",
        ),
        pytest.param(
            {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Al"],
                "commands": ["pair_style zero 5.0", "pair_coeff * *"],
            },
            ["pair_style zero 5.0", "pair_coeff * *"],
            [],
            id="analytic-lammps",
        ),
    ],
)
def test_run_replays_explicit_production_plan_without_hidden_kim_dispatch(
    monkeypatch,
    tmp_path: Path,
    potential_payload,
    potential_lines,
    expected_install_calls,
):
    from vitriflow.config import RunConfig
    from vitriflow.workflows import run as run_mod
    from vitriflow.workflows.production_common import make_production_plan, production_plan_to_dict

    cfg = RunConfig.model_validate(
        {
            "potential": potential_payload,
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
        potential_lines=potential_lines,
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
        quench_steps=90,
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
        artifact = Path(kwargs["outdir"]) / "fake_relax.data"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("verified production artifact\n")
        snapshot = artifact.parent / "structure_snapshot.json"
        manifest_path = artifact.parent / "structure_manifest.json"
        snapshot.write_text(json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1}))
        manifest = {"structure_hash": "fake-structure"}
        manifest_path.write_text(
            json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest]})
        )
        return {
            "status": "incomplete",
            "error": "accepted 1 boxes, below min_boxes=3",
            "converged": False,
            "converged_md": False,
            "check_convergence": True,
            "resumable": True,
            "convergence_streak": 0,
            "required_convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": 1,
            "last_convergence_evaluated_n_boxes_accepted": 1,
            "min_boxes": 3,
            "n_boxes": 1,
            "n_boxes_accepted": 1,
            "n_boxes_rejected": 0,
            "n_boxes_total": 1,
            "boxes": [{
                "box": 1,
                "density": 2.7,
                "distributions": {},
                "metrics": {},
                "paths": {
                    "relax_data": "fake_relax.data",
                    "analysis_source": "fake_relax.data",
                    "structure_snapshot": "structure_snapshot.json",
                    "structure_manifest": "structure_manifest.json",
                },
                "structure_manifest": manifest,
            }],
            "rejected_boxes": [],
            "cutoffs": [{"pair": [1, 1], "cutoff": 3.2}],
        }

    monkeypatch.setattr(run_mod, "_run_production_executor", _fake_exec)
    install_calls: list[str] = []
    monkeypatch.setattr(
        run_mod,
        "ensure_model_installed",
        lambda model: install_calls.append(str(model)),
    )
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
    assert called["quench_steps_override"] == 90
    assert called["relax_steps_override"] == 222
    assert called["potential_lines"] == potential_lines
    assert install_calls == expected_install_calls
    assert summary["production_plan"]["seed_base"] == 24680
    assert summary["parameters"]["execution_mode"] == "adaptive"
