from __future__ import annotations

from types import SimpleNamespace

from vitriflow.workflows.step_counts import resolve_lammps_units_style


def test_resolve_lammps_units_style_prefers_explicit_potential_config():
    cfg = SimpleNamespace(kim=SimpleNamespace(user_units="real"))
    pot = SimpleNamespace(user_units="metal")
    assert resolve_lammps_units_style(cfg, pot_cfg=pot, default="si") == "metal"


def test_resolve_lammps_units_style_falls_back_to_config_or_default():
    cfg = SimpleNamespace()
    assert resolve_lammps_units_style(cfg, pot_cfg=None, default="metal") == "metal"

    cfg2 = SimpleNamespace(kim=SimpleNamespace(user_units="  REAL  "))
    assert resolve_lammps_units_style(cfg2, pot_cfg=None, default="metal") == "real"
