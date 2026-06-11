from pathlib import Path

from vitriflow.config import RunConfig
from vitriflow.workflows.metrics_policy import resolve_effective_metrics_config


def _minimal_cfg(metrics_overrides: dict):
    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Sm", "O"],
                "commands": [
                    "pair_style lj/cut 2.5",
                    "pair_coeff 1 1 1.0 1.0 2.5",
                    "pair_coeff 2 2 1.0 1.0 2.5",
                    "pair_coeff 1 2 1.0 1.0 2.5",
                ],
            },
            "structure": {"generate": {"method": "packmol", "formula": "Sm2O3", "n_formula_units": 1}},
            "autotune": {"metrics": metrics_overrides},
        }
    )


def test_explicit_ring_disable_is_respected():
    cfg = _minimal_cfg({"enabled": True, "rings": {"enabled": False}})
    metrics_eff, warnings, summary = resolve_effective_metrics_config(
        cfg.autotune.metrics,
        structure_data=None,
        type_to_species=["Sm", "O"],
        context="autotune production",
    )
    assert metrics_eff.rings.enabled is False
    assert summary["rings_enabled"] is False
    assert not any("ring metrics not configured" in w for w in warnings)


def test_omitted_ring_config_still_auto_enables_defaults():
    cfg = _minimal_cfg({"enabled": True})
    metrics_eff, warnings, summary = resolve_effective_metrics_config(
        cfg.autotune.metrics,
        structure_data=None,
        type_to_species=["Sm", "O"],
        context="autotune production",
    )
    assert metrics_eff.rings.enabled is True
    assert summary["rings_enabled"] is True
    assert any("ring metrics not configured" in w for w in warnings)
