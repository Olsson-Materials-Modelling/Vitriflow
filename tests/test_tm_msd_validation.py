from __future__ import annotations

import math

import numpy as np
import pytest


def _valid_msd_inputs() -> tuple[np.ndarray, np.ndarray]:
    step = np.arange(6, dtype=float)
    msd = 2.0 * step
    return step, msd


@pytest.mark.parametrize(
    ("step", "msd", "timestep", "match"),
    [
        (np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3), 1.0, "one-dimensional"),
        (np.asarray([0.0, 1.0, 1.0, 2.0, 3.0]), np.arange(5.0), 1.0, "strictly increasing"),
        (np.asarray([0.0, 2.0, 1.0, 3.0, 4.0]), np.arange(5.0), 1.0, "strictly increasing"),
        (np.arange(5.0), np.asarray([0.0, 1.0, np.nan, 3.0, 4.0]), 1.0, "finite"),
        (np.arange(5.0), np.asarray([0.0, 1.0, -0.1, 3.0, 4.0]), 1.0, "nonnegative"),
        (np.arange(5.0), np.arange(5.0), 0.0, "timestep"),
        (np.arange(5.0), np.arange(5.0), float("nan"), "timestep"),
    ],
)
def test_diffusion_estimator_rejects_malformed_evidence(step, msd, timestep, match):
    from vitriflow.analysis.msd import estimate_diffusion_from_msd

    with pytest.raises(ValueError, match=match):
        estimate_diffusion_from_msd(step, msd, timestep)


def test_diffusion_estimator_rejects_unknown_uncertainty_method():
    from vitriflow.analysis.msd import estimate_diffusion_from_msd

    step, msd = _valid_msd_inputs()
    with pytest.raises(ValueError, match="stderr_method"):
        estimate_diffusion_from_msd(step, msd, 1.0, stderr_method="silent-fallback")


def test_negative_noisy_msd_slope_is_explicitly_constrained_at_D_zero():
    from vitriflow.analysis.msd import estimate_diffusion_from_msd

    step = np.arange(6, dtype=float)
    msd = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    estimate = estimate_diffusion_from_msd(
        step,
        msd,
        1.0,
        fit_start_fraction=0.0,
        stderr_method="ols",
    )

    assert estimate.slope == pytest.approx(-1.0)
    assert estimate.D_unconstrained == pytest.approx(-1.0 / 6.0)
    assert estimate.D == 0.0
    assert estimate.boundary_constrained is True


@pytest.mark.parametrize(
    ("T", "D", "eps", "match"),
    [
        (np.asarray([[300.0], [400.0], [500.0]]), np.ones((3, 1)), 1.0e-30, "one-dimensional"),
        (np.asarray([300.0, 300.0, 500.0]), np.asarray([0.0, 1.0, 2.0]), 1.0e-30, "distinct"),
        (np.asarray([300.0, np.nan, 500.0]), np.asarray([0.0, 1.0, 2.0]), 1.0e-30, "finite"),
        (np.asarray([300.0, 400.0, 500.0]), np.asarray([0.0, np.nan, 2.0]), 1.0e-30, "complete finite"),
        (np.asarray([300.0, 400.0, 500.0]), np.asarray([0.0, -1.0, 2.0]), 1.0e-30, "nonnegative"),
        (np.asarray([300.0, 400.0, 500.0]), np.asarray([0.0, 1.0, 2.0]), 0.0, "eps"),
    ],
)
def test_tm_diffusion_api_rejects_malformed_or_missing_evidence(T, D, eps, match):
    from vitriflow.analysis.tm import estimate_tm_from_diffusion

    with pytest.raises(ValueError, match=match):
        estimate_tm_from_diffusion(T, D, eps=eps)


def test_tm_api_rejects_partial_or_corrupt_structural_evidence():
    from vitriflow.analysis.tm import estimate_tm

    T = np.asarray([300.0, 400.0, 500.0])
    D = np.asarray([0.0, 0.1, 1.0])
    with pytest.raises(ValueError, match="provided together"):
        estimate_tm(T, D, gr_peak_height=np.asarray([3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="complete finite"):
        estimate_tm(
            T,
            D,
            gr_peak_height=np.asarray([3.0, np.nan, 1.0]),
            gr_peak_fwhm=np.asarray([0.2, 0.3, 0.4]),
        )
    with pytest.raises(ValueError, match="values > 0"):
        estimate_tm(
            T,
            D,
            gr_peak_height=np.asarray([3.0, 2.0, 0.0]),
            gr_peak_fwhm=np.asarray([0.2, 0.3, 0.4]),
        )


def test_tm_api_rejects_partial_or_ignored_mobility_evidence():
    from vitriflow.analysis.tm import estimate_tm

    T = np.asarray([300.0, 400.0, 500.0])
    D = np.asarray([0.0, 0.1, 1.0])
    H = np.asarray([3.0, 2.0, 1.0])
    W = np.asarray([0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="must be provided together"):
        estimate_tm(T, D, gr_peak_height=H, gr_peak_fwhm=W, msd_rms_last=np.ones(3))
    with pytest.raises(ValueError, match="requires structural"):
        estimate_tm(T, D, msd_rms_last=np.ones(3), vol_last=np.ones(3), natoms=1)
    with pytest.raises(ValueError, match="same length as T"):
        estimate_tm(
            T,
            D,
            gr_peak_height=H,
            gr_peak_fwhm=W,
            msd_rms_last=np.ones(2),
            vol_last=np.ones(3),
            natoms=1,
        )


def test_tm_unconfirmed_liquid_temperature_remains_unassessed():
    from vitriflow.analysis.tm import estimate_tm

    estimate = estimate_tm(
        np.asarray([300.0, 400.0, 500.0]),
        np.zeros(3),
        gr_peak_height=np.ones(3),
        gr_peak_fwhm=np.ones(3),
    )
    assert math.isnan(estimate.T_liquid)
    assert math.isnan(estimate.D_liquid_target)


def test_tm_replicate_summary_never_drops_missing_values_or_imputes_se_zero():
    from vitriflow.workflows.autotune import _complete_tm_replicate_summary

    mean, stderr, median = _complete_tm_replicate_summary([2.0])
    assert mean == 2.0
    assert median == 2.0
    assert math.isnan(stderr)

    incomplete = _complete_tm_replicate_summary([1.0, float("nan")])
    assert all(math.isnan(value) for value in incomplete)

    unconstrained = _complete_tm_replicate_summary(
        [0.0, -0.1],
        require_nonnegative=True,
    )
    assert all(math.isnan(value) for value in unconstrained)


def test_stage_outcome_preserves_diffusion_boundary_diagnostic_defaults():
    from vitriflow.workflows.stage_runner import StageOutcome

    outcome = StageOutcome(
        name="stage",
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=1,
        n_atoms=1,
        vol_last=10.0,
        density_mean=1.0,
        density_stderr=float("nan"),
        pe_mean=-1.0,
        pe_stderr=float("nan"),
        D=0.0,
        D_stderr=0.1,
        msd_rms_last=0.0,
        output_data="output.data",
    )
    assert math.isnan(outcome.D_unconstrained)
    assert outcome.D_boundary_constrained is False
