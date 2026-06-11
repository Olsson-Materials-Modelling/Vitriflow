from __future__ import annotations

from types import SimpleNamespace

from vitriflow.workflows.step_counts import resolve_md_pressure


def test_resolve_md_pressure_prefers_override_then_md_use_then_config():
    cfg = SimpleNamespace(md=SimpleNamespace(pressure=1.5))
    md_use = SimpleNamespace(pressure=2.5)
    assert resolve_md_pressure(cfg, md_use=md_use, override=3.5, default=0.0) == 3.5
    assert resolve_md_pressure(cfg, md_use=md_use, default=0.0) == 2.5
    assert resolve_md_pressure(cfg, md_use=SimpleNamespace(), default=0.0) == 1.5


def test_resolve_md_pressure_accepts_mapping_md_use():
    cfg = SimpleNamespace(md=SimpleNamespace(pressure=1.5))
    assert resolve_md_pressure(cfg, md_use={"pressure": 4.25}, default=0.0) == 4.25
