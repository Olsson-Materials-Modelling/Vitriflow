import pytest


def test_rings_enabled_requires_nodes():
    from vitriflow.config import RingMetricsConfig

    with pytest.raises(ValueError, match=r"rings\.nodes"):
        RingMetricsConfig(enabled=True)


def test_rings_projected_requires_bridge():
    from vitriflow.config import RingMetricsConfig

    with pytest.raises(ValueError, match=r"rings\.bridge"):
        RingMetricsConfig(enabled=True, nodes=["Si"])  # projected


def test_rings_bond_graph_does_not_require_bridge():
    from vitriflow.config import RingMetricsConfig

    cfg = RingMetricsConfig(enabled=True, mode="bond_graph", nodes=["Si", "O"], max_cycle_size=12)
    assert cfg.enabled is True
    assert cfg.mode == "bond_graph"
