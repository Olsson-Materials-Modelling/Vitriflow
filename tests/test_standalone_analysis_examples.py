from __future__ import annotations

from pathlib import Path

import yaml


def test_sin_ase_database_example_builds_standalone_analysis_context():
    from vitriflow.workflows.output_analysis import analysis_context_from_standalone_config

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "vitriflow" / "examples" / "analysis_sin_ase_database.yaml"
    assert cfg_path.exists()

    data = yaml.safe_load(cfg_path.read_text())
    ctx = analysis_context_from_standalone_config(data)

    assert ctx.type_to_species == ["Si", "N"]
    assert ctx.atom_style == "atomic"
    assert ctx.md_timestep == 1.0
    assert ctx.metrics_cfg.enabled is True
    assert ctx.metrics_cfg.time_average_frames == 1
    assert ctx.prod_cfg.exclude_coordination_defects is False
    assert ctx.prod_cfg.store_distributions is True

    pairs = [tuple(entry.pair) if entry.pair is not None else None for entry in ctx.metrics_cfg.pairs]
    assert ("Si", "N") in pairs
    assert ("Si", "Si") in pairs
    assert ("N", "N") in pairs

    coord = {(c.central, c.neighbor): c.expected for c in ctx.metrics_cfg.coordinations}
    assert coord[("Si", "N")] == 4
    assert coord[("N", "Si")] == 3

    rings = ctx.metrics_cfg.rings
    assert rings.enabled is True
    assert rings.mode == "bond_graph"
    assert rings.nodes == ["Si", "N"]
    assert [list(bp.pair) for bp in rings.bond_pairs] == [["Si", "N"]]
