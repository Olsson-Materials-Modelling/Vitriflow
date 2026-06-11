import math

import pytest

from vitriflow.workflows.quench_rates import quench_steps_for_rate


def test_quench_steps_for_rate_realized_rate_not_faster_than_target():
    dT = 1200.0
    rate = 10.0  # k time
    dt = 0.5
    n = quench_steps_for_rate(dT, rate, dt, min_steps=1)
    assert n >= 1
    realized = dT / (n * dt)
    assert realized <= rate + 1e-12


def test_quench_steps_for_rate_exact_division():
    dT = 100.0
    rate = 10.0
    dt = 1.0
    n = quench_steps_for_rate(dT, rate, dt, min_steps=1)
    assert n == 10


def test_quench_steps_for_rate_invalid_inputs():
    with pytest.raises(ValueError):
        quench_steps_for_rate(100.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        quench_steps_for_rate(100.0, 10.0, 0.0)
    with pytest.raises(ValueError):
        quench_steps_for_rate(float('nan'), 10.0, 1.0)
