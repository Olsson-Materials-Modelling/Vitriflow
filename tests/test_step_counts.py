import math

import pytest

from vitriflow.workflows.step_counts import (
    extend_highT_steps_for_force_isotropic,
    recommendation_md_timestep,
    scale_config_steps,
    scale_recommended_highT_steps,
)


def test_recommendation_md_timestep_prefers_md_block():
    rec = {"md": {"timestep": 0.5}, "highT_steps": 1000}
    assert recommendation_md_timestep(rec, fallback=2.0) == pytest.approx(0.5)


def test_recommendation_md_timestep_legacy_keys():
    rec = {"md_timestep": 0.25}
    assert recommendation_md_timestep(rec, fallback=1.0) == pytest.approx(0.25)


def test_scale_recommended_highT_steps_uses_recommendation_timestep():
    # regression yaml timestep
    rec = {"md": {"timestep": 0.5}, "highT_steps": 1000}
    dt_cfg = 2.0
    dt_use = 1.0
    scaled = scale_recommended_highT_steps(1000, rec, dt_cfg=dt_cfg, dt_use=dt_use)
    # preserve physical steps
    assert scaled == 500


def test_scale_recommended_highT_steps_falls_back_safely():
    rec = {"md": {"timestep": "not-a-number"}, "highT_steps": 1000}
    dt_cfg = 2.0
    dt_use = 1.0
    scaled = scale_recommended_highT_steps(1000, rec, dt_cfg=dt_cfg, dt_use=dt_use)
    assert scaled == 2000


def test_scale_recommended_highT_steps_ignores_md_timestep_if_not_recommended():
    # present interpreted defined
    rec = {"md": {"timestep": 0.5}}
    dt_cfg = 2.0
    dt_use = 1.0
    scaled = scale_recommended_highT_steps(1000, rec, dt_cfg=dt_cfg, dt_use=dt_use)
    assert scaled == 2000


def test_scale_config_steps():
    assert scale_config_steps(100, dt_cfg=2.0, dt_use=1.0) == 200


def test_extend_highT_steps_for_force_isotropic_only_when_enabled():
    assert extend_highT_steps_for_force_isotropic(1000, force_isotropic=False) == 1000
    assert extend_highT_steps_for_force_isotropic(1000, force_isotropic=True) == 1500


def test_extend_highT_steps_for_force_isotropic_uses_ceil():
    assert extend_highT_steps_for_force_isotropic(3, force_isotropic=True) == int(math.ceil(4.5))
