from __future__ import annotations

import pytest

pytest.importorskip("ase")

from pathlib import Path

from vitriflow.workflows.production_common import (
    cutoffs_dict_from_any,
    make_production_plan,
    production_plan_from_dict,
    production_plan_to_dict,
)


def test_production_plan_roundtrip_relative_paths(tmp_path: Path):
    outdir = tmp_path / "results"
    outdir.mkdir()
    base = outdir / "base.data"
    base.write_text("# placeholder\n")

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
        md_use={"ensemble": "npt", "timestep": 0.001, "pressure": 0.0},
        potential_config={"kind": "kim", "model": "dummy", "user_units": "metal", "interactions": ["Al"]},
        potential_lines=["pair_style kim ..."],
        core_repulsion={"enabled": False},
        type_to_species=["Al"],
        metrics_cfg={"enabled": True},
        effective_metrics={"pairs": 1},
        production_cfg={"enabled": True, "min_boxes": 5, "max_boxes": 10, "batch_boxes": 5},
        convergence_cfg={"mode": "both"},
        cutoffs_rate={(1, 1): 3.0},
        cutoffs_size={(1, 1): 3.1},
        preferred_cutoffs={(1, 1): 3.1},
        quench_steps=100,
        relax_steps=200,
        msd_every=100,
        seed_base=13586,
        time_unit_ps=1.0,
        sampling_hint={"Tm": 900.0, "freeze_temperature": 700.0},
        execution_mode="adaptive",
        source_kind="autotune",
    )

    payload = production_plan_to_dict(plan, relative_to=outdir)
    assert payload["structure_data"] == "base.data"

    roundtrip = production_plan_from_dict(payload, base_dir=outdir)
    assert roundtrip.structure_data == base.resolve(strict=False)
    assert roundtrip.replicate == (2, 2, 2)
    assert roundtrip.seed_base == 13586
    assert cutoffs_dict_from_any(roundtrip.preferred_cutoffs) == {(1, 1): 3.1}
    assert roundtrip.execution_mode == "adaptive"
