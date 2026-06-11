import numpy as np
import pytest

from vitriflow.analysis.stats import early_late_change


def test_early_late_change_basic():
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    chg = early_late_change(x, split_fraction=0.5, denom='late')
    assert chg.early_mean == pytest.approx(0.0)
    assert chg.late_mean == pytest.approx(1.0)
    assert chg.abs_change == pytest.approx(1.0)
    assert chg.rel_change == pytest.approx(1.0)


def test_early_late_change_filters_nonfinite():
    x = np.array([0.0, np.nan, 0.0, 2.0, 2.0, np.inf])
    chg = early_late_change(x, split_fraction=0.5, denom='late')
    assert np.isfinite(chg.rel_change)


def test_early_late_change_requires_two_points():
    chg = early_late_change(np.array([1.0]))
    assert np.isnan(chg.rel_change)
