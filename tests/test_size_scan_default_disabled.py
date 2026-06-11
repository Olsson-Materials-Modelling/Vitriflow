from __future__ import annotations


def test_size_scan_disabled_by_default():
    from vitriflow.config import SizeConfig

    cfg = SizeConfig()
    assert cfg.enabled is False


def test_size_scan_can_be_enabled():
    from vitriflow.config import SizeConfig

    cfg = SizeConfig(enabled=True)
    assert cfg.enabled is True
