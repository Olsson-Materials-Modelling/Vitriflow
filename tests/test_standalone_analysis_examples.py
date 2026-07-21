from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_sin_ase_database_example_builds_standalone_analysis_context():
    from vitriflow.workflows.output_analysis import analysis_context_from_standalone_config

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "vitriflow" / "examples" / "analysis_sin_ase_database.yaml"
    if not cfg_path.is_file():
        pytest.skip(
            "development-only standalone-analysis example is intentionally absent "
            "from the release source distribution"
        )

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


def _minimal_standalone(**updates):
    analysis = {
        "type_to_species": ["Si"],
        "metrics": {"enabled": False, "type_to_species": ["Si"]},
    }
    analysis.update(updates)
    return {"analysis": analysis}


@pytest.mark.parametrize(
    "payload,match",
    [
        (_minimal_standalone(embed_structuers=False), "embed_structuers"),
        (_minimal_standalone(sources={"include_glob": ["*.extxyz"]}), "include_glob"),
        (_minimal_standalone(embed_structures="flase"), "analysis.embed_structures must be a boolean"),
        (_minimal_standalone(embed_structures=2), r"boolean \(or 0/1\)"),
        (_minimal_standalone(embed_structures=None), "explicit null is not allowed"),
    ],
)
def test_standalone_analysis_config_fails_closed_on_typos(payload, match):
    from vitriflow.workflows.output_analysis import analysis_context_from_standalone_config

    with pytest.raises(ValueError, match=match):
        analysis_context_from_standalone_config(payload)


def test_standalone_production_embed_setting_is_honored_and_conflicts_fail():
    from vitriflow.workflows.output_analysis import analysis_context_from_standalone_config

    ctx = analysis_context_from_standalone_config(
        _minimal_standalone(production={"embed_structures": True})
    )
    assert ctx.embed_structures is True
    assert ctx.prod_cfg.embed_structures is True

    with pytest.raises(ValueError, match="Conflicting standalone embed_structures"):
        analysis_context_from_standalone_config(
            _minimal_standalone(
                embed_structures=False,
                production={"embed_structures": True},
            )
        )
