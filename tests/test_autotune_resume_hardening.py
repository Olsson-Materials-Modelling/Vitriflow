from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")

SUPPORTED_SCAN_METRICS = {
    "density": 2.70,
    "coord_Al-Al_mean": 12.0,
    "bondlen_Al-Al_mean": 2.86,
    "angle_Al-Al-Al_mean": 60.0,
    "ring_frac_3": 0.25,
    "ring_mean_size": 6.0,
    "gr_all_peak_r": 2.86,
    "gr_all_peak_height": 2.5,
    "gr_all_peak_fwhm": 0.30,
}

# These are finite scalar outputs produced by the metrics enabled in the
# development-only minimal_metal_mpi8 example, but the example defines no
# scalar-scan tolerance for any of them.  They remain analysis outputs, not
# implicit convergence criteria.
UNTOLERANCED_AUXILIARY_METRICS = (
    "bond_incidence_Al-Al_count",
    "ring_count",
    "ring_entropy",
    "sq_all_peak_q",
    "sq_all_peak_height",
    "sq_all_peak_fwhm",
    "sq_Al_Al_peak_q",
    "sq_Al_Al_peak_height",
    "sq_Al_Al_peak_fwhm",
    "void_clearance_mean",
    "void_clearance_median",
    "void_clearance_p95",
    "void_clearance_max",
    "void_clearance_n_samples",
    "void_clearance_scaled_mean",
    "void_clearance_scaled_median",
    "void_clearance_scaled_p95",
    "void_clearance_scaled_max",
    "void_clearance_length_scale",
    "void_clearance_frac_ge_r0p5",
    "void_clearance_frac_ge_r1",
    "void_clearance_frac_ge_r1p5",
)


def _minimal_example_convergence_config():
    from vitriflow.config import ConvergenceConfig

    # Keep this unit-test fixture independent of the development-only example:
    # release source distributions intentionally ship only the public smoke
    # examples.  test_examples_yaml_parse.py separately loads and validates the
    # developer example whenever that complete fixture set is present.
    return ConvergenceConfig(
        density_rel_tol=0.01,
        density_abs_tol=0.001,
        coord_rel_tol=0.0,
        coord_abs_tol=0.1,
        bondlen_rel_tol=0.0,
        bondlen_abs_tol=0.03,
        angle_rel_tol=0.0,
        angle_abs_tol=3.0,
        ring_rel_tol=0.0,
        ring_abs_tol=0.1,
        ring_size_rel_tol=0.0,
        ring_size_abs_tol=0.5,
        gr_peak_r_rel_tol=0.0,
        gr_peak_r_abs_tol=0.03,
        gr_peak_height_rel_tol=0.05,
        gr_peak_height_abs_tol=0.2,
        gr_peak_fwhm_rel_tol=0.05,
        gr_peak_fwhm_abs_tol=0.03,
    )


def _scan_maps(*, include_auxiliary: bool, perturb_supported: bool = False):
    reference = dict(SUPPORTED_SCAN_METRICS)
    candidate = dict(reference)
    if perturb_supported:
        candidate["bondlen_Al-Al_mean"] = reference["bondlen_Al-Al_mean"] + 1.0

    if include_auxiliary:
        for index, name in enumerate(UNTOLERANCED_AUXILIARY_METRICS, start=1):
            reference[name] = float(index)
            # Deliberately far apart: these values would fail any plausible
            # scalar convergence check if they were incorrectly included.
            candidate[name] = float(index + 10_000)

    stderr = {name: 0.0 for name in reference}
    return [candidate, reference], [dict(stderr), dict(stderr)]


def test_kim_install_state_is_strict_json_compatible():
    from vitriflow.kim import KimInstallResult
    from vitriflow.workflows.autotune import _kim_install_jsonable

    state = KimInstallResult(attempted=True, success=False, stdout="out", stderr="err")
    payload = _kim_install_jsonable(state)

    assert payload == {
        "attempted": True,
        "success": False,
        "stdout": "out",
        "stderr": "err",
    }
    json.dumps(payload, allow_nan=False)
    assert _kim_install_jsonable(payload) == payload
    assert _kim_install_jsonable(None) is None


def test_resume_terminalizes_authenticated_disabled_preproduction_checkpoint(
    monkeypatch, tmp_path
):
    from vitriflow.config import RunConfig
    from vitriflow.workflows import autotune as module

    config = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Al"],
                "commands": ["pair_style zero 5.0", "pair_coeff * *"],
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "Al",
                    "n_formula_units": 1,
                }
            },
            "autotune": {"production": {"enabled": False}},
        }
    )
    selected = tmp_path / "selected.data"
    selected.write_text("authenticated selected structure\n")
    plan = {
        "engine": "lammps",
        "structure_data": "selected.data",
        "potential_config": config.kim.model_dump(mode="json"),
        "production_cfg": config.autotune.production.model_dump(mode="json"),
    }
    disabled = module._attach_production_state_integrity(
        {
            "enabled": False,
            "status": "not_requested",
            "execution_status": "not_requested",
            "resumable": True,
            "n_boxes": 0,
            "n_boxes_accepted": 0,
            "n_boxes_rejected": 0,
            "n_boxes_total": 0,
            "boxes": [],
            "rejected_boxes": [],
        },
        outdir=tmp_path,
    )
    previous = {
        "status": "running",
        "execution_status": "running",
        "production": disabled,
        "production_plan": plan,
        "size_scan": {"base_data": "selected.data"},
    }
    previous["resume_fingerprint"] = module._build_autotune_resume_fingerprint(
        config=config,
        outdir=tmp_path,
        selected_structure=selected,
        production_plan=plan,
    )

    snapshots: list[dict] = []
    monkeypatch.setattr(
        module,
        "write_autotune_outputs",
        lambda _outdir, result: snapshots.append(json.loads(json.dumps(result))),
    )
    result = module._autotune_resume_from_results(
        config=config,
        outdir=tmp_path,
        prev=previous,
    )

    assert result["status"] == "ok"
    assert result["execution_status"] == "completed"
    assert result["production"] == disabled
    assert snapshots == [result]


def test_unknown_autotune_metric_requires_an_explicit_tolerance():
    from vitriflow.config import ConvergenceConfig
    from vitriflow.workflows.autotune import _tol_for_metric

    with pytest.raises(ValueError, match="novel_descriptor"):
        _tol_for_metric("novel_descriptor", ConvergenceConfig())


@pytest.mark.parametrize("kind", ["rate", "size"])
def test_minimal_example_scan_skips_untoleranced_auxiliary_metrics(kind):
    from vitriflow.workflows.autotune import _multimetric_decision

    conv = _minimal_example_convergence_config()
    control_mu, control_se = _scan_maps(include_auxiliary=False)
    enriched_mu, enriched_se = _scan_maps(include_auxiliary=True)

    control = _multimetric_decision(
        [1.0, 2.0], control_mu, control_se, conv=conv, kind=kind
    )
    enriched = _multimetric_decision(
        [1.0, 2.0], enriched_mu, enriched_se, conv=conv, kind=kind
    )

    assert enriched["chosen_index"] == control["chosen_index"] == 0
    assert enriched["chosen_value"] == control["chosen_value"] == 1.0
    assert enriched["combined_passed"] == control["combined_passed"] == [True, True]
    assert enriched["metrics"] == control["metrics"]
    assert set(enriched["metrics"]) == set(SUPPORTED_SCAN_METRICS)

    skipped = enriched["skipped_metrics"]
    assert {entry["name"] for entry in skipped} == set(UNTOLERANCED_AUXILIARY_METRICS)
    assert {entry["kind"] for entry in skipped} == {"scalar_scan"}
    assert {entry["reason"] for entry in skipped} == {
        "auxiliary metric has no defined scalar-scan tolerance and was excluded from selection"
    }


@pytest.mark.parametrize("kind", ["rate", "size"])
def test_minimal_example_supported_metric_still_controls_scan_selection(kind):
    from vitriflow.workflows.autotune import _multimetric_decision

    mu_maps, se_maps = _scan_maps(include_auxiliary=True, perturb_supported=True)
    decision = _multimetric_decision(
        [1.0, 2.0],
        mu_maps,
        se_maps,
        conv=_minimal_example_convergence_config(),
        kind=kind,
    )

    assert decision["metrics"]["bondlen_Al-Al_mean"]["passed"] == [False, True]
    assert decision["combined_passed"] == [False, True]
    assert decision["chosen_index"] == 1
    assert decision["chosen_value"] == 2.0


@pytest.mark.parametrize("kind", ["rate", "size"])
def test_multimetric_no_pass_reference_is_labelled_unconverged_fallback(kind):
    from vitriflow.workflows.autotune import _multimetric_decision

    decision = _multimetric_decision(
        [1.0, 2.0],
        [{"density": 1.0}, {"density": 1.0}],
        [{"density": 1.0}, {"density": 1.0}],
        conv=_minimal_example_convergence_config(),
        kind=kind,
    )

    assert decision["combined_passed"] == [False, False]
    assert decision["chosen_index"] == 1
    assert decision["selection_converged"] is False
    assert decision["fallback_used"] is True
    assert decision["selection_status"] == "fallback_unconverged"


def test_multimetric_empty_eligible_set_is_unassessed_not_vacuously_converged():
    from vitriflow.workflows.autotune import _multimetric_decision

    decision = _multimetric_decision(
        [1.0, 2.0],
        [{"auxiliary_only": 1.0}, {"auxiliary_only": 1.0}],
        [{"auxiliary_only": 0.0}, {"auxiliary_only": 0.0}],
        conv=_minimal_example_convergence_config(),
        kind="size",
    )

    assert decision["metrics"] == {}
    assert decision["combined_passed"] == [False, False]
    assert decision["selection_converged"] is False
    assert decision["selection_status"] == "no_eligible_metrics_unassessed"


def test_multimetric_selection_requires_a_contiguous_passing_tail():
    from vitriflow.workflows.autotune import _multimetric_decision

    decision = _multimetric_decision(
        [1.0, 2.0, 3.0],
        [{"density": 1.0}, {"density": 2.0}, {"density": 1.0}],
        [{"density": 0.0}, {"density": 0.0}, {"density": 0.0}],
        conv=_minimal_example_convergence_config(),
        kind="size",
    )

    assert decision["combined_passed"] == [True, False, True]
    assert decision["combined_tail_passed"] == [False, False, True]
    assert decision["metrics"]["density"]["tail_passed"] == [False, False, True]
    assert decision["chosen_index"] == 2


def test_multimetric_union_metric_missing_at_reference_blocks_assessment():
    from vitriflow.workflows.autotune import _multimetric_decision

    decision = _multimetric_decision(
        [1.0, 2.0, 3.0],
        [
            {"density": 1.0, "bondlen_Al-Al_mean": 2.86},
            {"density": 1.0, "bondlen_Al-Al_mean": 2.86},
            {"density": 1.0},
        ],
        [
            {"density": 0.0, "bondlen_Al-Al_mean": 0.0},
            {"density": 0.0, "bondlen_Al-Al_mean": 0.0},
            {"density": 0.0},
        ],
        conv=_minimal_example_convergence_config(),
        kind="size",
    )

    assert "bondlen_Al-Al_mean" in decision["metrics"]
    assert decision["criteria_complete"] is False
    assert decision["selection_converged"] is False
    assert decision["chosen_index"] == 2
    assert decision["selection_status"] == "incomplete_eligible_metrics_unassessed"
    blocker = next(
        row
        for row in decision["blocking_metrics"]
        if row["name"] == "bondlen_Al-Al_mean" and row["index"] == 2
    )
    assert set(blocker["fields"]) == {"mean", "stderr"}
    # Missing numerics are represented as strict-JSON nulls, not NaN tokens.
    json.dumps(decision, allow_nan=False)


def test_multimetric_incomplete_early_point_does_not_poison_later_clean_tail():
    from vitriflow.workflows.autotune import _multimetric_decision

    decision = _multimetric_decision(
        [1.0, 2.0, 3.0],
        [
            {"density": 1.0},
            {"density": 1.0, "bondlen_Al-Al_mean": 2.86},
            {"density": 1.0, "bondlen_Al-Al_mean": 2.86},
        ],
        [
            {"density": 0.0},
            {"density": 0.0, "bondlen_Al-Al_mean": 0.0},
            {"density": 0.0, "bondlen_Al-Al_mean": 0.0},
        ],
        conv=_minimal_example_convergence_config(),
        kind="size",
    )

    assert decision["criteria_complete"] is False
    assert decision["point_criteria_complete"] == [False, True, True]
    assert decision["tail_criteria_complete"] == [False, True, True]
    assert decision["combined_tail_passed"] == [False, True, True]
    assert decision["chosen_index"] == 1
    assert decision["selection_converged"] is True
    assert decision["selection_criteria_complete"] is True
    assert decision["selection_status"] == "converged"


def test_scan_metric_aggregation_requires_every_replicate_and_two_for_stderr():
    from vitriflow.workflows.autotune import _aggregate_scalar_metrics

    mixed_mu, mixed_se = _aggregate_scalar_metrics(
        [{"density": 2.3}, {"density": float("nan")}, {"other": 1.0}]
    )
    assert math.isnan(mixed_mu["density"])
    assert math.isnan(mixed_se["density"])

    one_mu, one_se = _aggregate_scalar_metrics([{"density": 2.3}])
    assert one_mu["density"] == pytest.approx(2.3)
    assert math.isnan(one_se["density"])

    complete_mu, complete_se = _aggregate_scalar_metrics(
        [{"density": 2.2}, {"density": 2.4}]
    )
    assert complete_mu["density"] == pytest.approx(2.3)
    assert complete_se["density"] == pytest.approx(0.1)


def test_single_replicate_scan_metric_is_reported_as_incomplete_evidence():
    from vitriflow.workflows.autotune import (
        _aggregate_scalar_metrics,
        _multimetric_decision,
    )

    mu_maps = []
    se_maps = []
    for density in (2.3, 2.3):
        mu, se = _aggregate_scalar_metrics([{"density": density}])
        mu_maps.append(mu)
        se_maps.append(se)

    decision = _multimetric_decision(
        [2.0, 1.0],
        mu_maps,
        se_maps,
        conv=_minimal_example_convergence_config(),
        kind="rate",
    )

    assert decision["selection_converged"] is False
    assert decision["selection_status"] == "incomplete_eligible_metrics_unassessed"
    assert decision["selection_reason"] is not None
    assert all(
        "stderr" in row["fields"] for row in decision["blocking_metrics"]
    )


def test_negative_scan_standard_error_is_invalid_evidence() -> None:
    from vitriflow.workflows.autotune import _multimetric_decision

    decision = _multimetric_decision(
        [1.0, 2.0],
        [{"density": 2.3}, {"density": 2.3}],
        [{"density": -0.1}, {"density": 0.1}],
        conv=_minimal_example_convergence_config(),
        kind="size",
    )

    assert decision["point_criteria_complete"] == [False, True]
    assert decision["selection_converged"] is False
    assert decision["selection_status"] == "fallback_unconverged"
    assert any(
        row["index"] == 0 and "stderr" in row["fields"]
        for row in decision["blocking_metrics"]
    )


def test_misspelled_convergence_tolerance_is_rejected():
    from pydantic import ValidationError

    from vitriflow.config import ConvergenceConfig

    payload = _minimal_example_convergence_config().model_dump()
    payload["bondlen_abs_toll"] = payload["bondlen_abs_tol"]

    with pytest.raises(ValidationError, match="bondlen_abs_toll"):
        ConvergenceConfig.model_validate(payload)


@pytest.mark.parametrize(
    (
        "config_potential_payload",
        "plan_potential_payload",
        "plan_potential_lines",
        "expected_install_calls",
        "expected_install_state",
    ),
    [
        pytest.param(
            {
                "kind": "lammps",
                "interactions": ["C"],
                "commands": ["pair_style zero 5.0", "pair_coeff * *"],
            },
            {
                "kind": "kim",
                "model": "MODEL_IDENTIFIER",
                "interactions": ["C"],
            },
            [
                "kim init MODEL_IDENTIFIER metal",
                "kim interactions C",
            ],
            ["MODEL_IDENTIFIER"],
            {
                "attempted": False,
                "success": True,
                "stdout": "",
                "stderr": "",
            },
            id="kim",
        ),
        pytest.param(
            {
                "kind": "kim",
                "model": "CURRENT_CONFIG_KIM_MUST_NOT_INSTALL",
                "interactions": ["C"],
            },
            {
                "kind": "lammps",
                "interactions": ["C"],
                "commands": [
                    "pair_style hybrid/overlay buck 10.0 morse 10.0",
                    "pair_coeff 1 1 buck 1000.0 0.3 20.0",
                    "pair_coeff 1 1 morse 0.2 2.0 1.5",
                ],
            },
            [
                "pair_style hybrid/overlay buck 10.0 morse 10.0",
                "pair_coeff 1 1 buck 1000.0 0.3 20.0",
                "pair_coeff 1 1 morse 0.2 2.0 1.5",
            ],
            [],
            None,
            id="protected-analytic-hybrid-over-current-kim",
        ),
    ],
)
def test_autotune_resume_dispatches_only_explicit_kim_potentials(
    monkeypatch,
    tmp_path,
    config_potential_payload,
    plan_potential_payload,
    plan_potential_lines,
    expected_install_calls,
    expected_install_state,
):
    from pydantic import TypeAdapter

    from vitriflow.config import PotentialConfig, RunConfig
    from vitriflow.kim import KimInstallResult
    from vitriflow.workflows import autotune as module

    base_data = tmp_path / "base.data"
    base_data.write_text("placeholder\n")

    config = RunConfig.model_validate(
        {
            "potential": config_potential_payload,
            "structure": {
                "generate": {"method": "random", "formula": "C", "n_formula_units": 1}
            },
            "autotune": {
                "metrics": {
                    "enabled": True,
                    "type_to_species": ["CURRENT_CONFIG_SPECIES_MUST_NOT_LEAK"],
                },
                "production": {"enabled": True, "min_boxes": 1, "max_boxes": 1},
            },
        }
    )
    protected_potential = TypeAdapter(PotentialConfig).validate_python(
        plan_potential_payload
    )
    previous = {
        "status": "running",
        "execution_status": "running",
        "production": {
            "enabled": True,
            "status": "running",
            "execution_status": "running",
            "converged": False,
            "converged_md": False,
            "check_convergence": True,
            "resumable": True,
            "convergence_streak": 0,
            "required_convergence_streak": 1,
            "last_convergence_evaluated_n_boxes_total": None,
            "last_convergence_evaluated_n_boxes_accepted": None,
            "min_boxes": 1,
            "max_boxes": 1,
            "batch_boxes": 5,
            "n_boxes": 0,
            "n_boxes_accepted": 0,
            "n_boxes_rejected": 0,
            "n_boxes_total": 0,
            "rate_K_per_time": 10.0,
            "replicate": [1, 1, 1],
            "T_high": 2000.0,
            "highT_steps": 100,
            "structure_data": str(base_data),
            "boxes": [],
            "rejected_boxes": [],
        },
        "recommendation": {},
        "preflight": {},
        "size_scan": {"base_data": str(base_data)},
        "rate_scan": {},
        "units": {},
    }
    previous["production_plan"] = {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "md_use": config.md.model_dump(mode="json"),
        "potential_config": protected_potential.model_dump(mode="json"),
        "potential_lines": list(plan_potential_lines),
        "core_repulsion": protected_potential.core_repulsion.model_dump(mode="json"),
        "type_to_species": ["C"],
        "chosen_rate": 10.0,
        "replicate": [1, 1, 1],
        "T_high": 2000.0,
        "high_total_steps": 100,
        "time_unit_ps": None,
        "cooling_rate_ps": None,
        "structure_data": str(base_data),
        "metrics_cfg": {
            **config.autotune.metrics.model_dump(mode="json"),
            "type_to_species": ["C"],
        },
        "effective_metrics": {"enabled": True},
        "cutoffs_rate": [],
        "cutoffs_size": [],
        "preferred_cutoffs": [],
        "t_final": config.autotune.quench.t_final,
        "relax_steps": config.autotune.quench.relax_steps,
        "msd_every": config.autotune.tm_scan.msd_every,
        "production_cfg": config.autotune.production.model_dump(mode="json"),
        "convergence_cfg": config.autotune.convergence.model_dump(mode="json"),
        "pressure": config.md.pressure,
        "seed_base": config.random_seed + 13579,
        "quench_steps": 170,
        "sampling_hint": None,
        "execution_mode": "adaptive",
        "source_kind": "autotune",
    }
    previous["production"] = module._attach_production_state_integrity(
        previous["production"], outdir=tmp_path
    )
    previous["resume_fingerprint"] = module._build_autotune_resume_fingerprint(
        config=config,
        outdir=tmp_path,
        selected_structure=base_data,
        production_plan=previous["production_plan"],
    )
    compact_terminal = json.loads(json.dumps(previous))
    compact_terminal["status"] = "incomplete"
    compact_terminal["execution_status"] = "completed"
    compact_terminal["production"] = module._attach_production_state_integrity(
        {
            **{
                key: value
                for key, value in compact_terminal["production"].items()
                if key != "state_integrity"
            },
            "status": "incomplete",
            "execution_status": "completed",
            "resumable": False,
            "non_resumable_reason": "adaptive convergence distributions were omitted",
        },
        outdir=tmp_path,
    )

    install_calls: list[str] = []

    def fake_install(model):
        install_calls.append(model)
        return KimInstallResult(attempted=False, success=True)

    serialized_snapshots: list[dict] = []

    def strict_write(_outdir, results):
        # This is the contract enforced by the real strict JSON writer.
        json.dumps(results, allow_nan=False)
        serialized_snapshots.append(dict(results))

    monkeypatch.setattr(module, "ensure_model_installed", fake_install)
    monkeypatch.setattr(module, "_get_type_to_species", lambda _cfg: ["C"])
    monkeypatch.setattr(
        module,
        "resolve_effective_metrics_config",
        lambda *_a, **_k: (SimpleNamespace(), {}, {"enabled": True}),
    )
    monkeypatch.setattr(module, "write_autotune_outputs", strict_write)
    production_calls: list[dict] = []

    def capture_production(**kwargs):
        production_calls.append(dict(kwargs))
        return {"status": "ok", "boxes": [], "rejected_boxes": []}

    monkeypatch.setattr(module, "_run_production_ensemble", capture_production)
    identity_queries: list[dict] = []
    original_identity_query = module.query_engine_build_identities

    def tracked_identity_query(*args, **kwargs):
        result = original_identity_query(*args, **kwargs)
        identity_queries.append(result)
        return result

    monkeypatch.setattr(
        module, "query_engine_build_identities", tracked_identity_query
    )

    mutation_previous = json.loads(json.dumps(previous))
    result = module._autotune_resume_from_results(
        config=config,
        outdir=tmp_path,
        prev=previous,
    )

    assert install_calls == expected_install_calls
    assert result["kim_install"] == expected_install_state
    assert len(serialized_snapshots) == 2
    assert len(identity_queries) == 2
    assert len(production_calls) == 1
    assert production_calls[0]["pot_cfg"].model_dump(mode="json") == (
        protected_potential.model_dump(mode="json")
    )
    assert production_calls[0]["potential_lines"] == list(plan_potential_lines)
    assert production_calls[0]["type_to_species"] == ["C"]

    def mutate_structure_during_resumed_production(**_kwargs):
        base_data.write_text("mutated while resumed production was running\n")
        return {"status": "ok", "boxes": [], "rejected_boxes": []}

    monkeypatch.setattr(
        module,
        "_run_production_ensemble",
        mutate_structure_during_resumed_production,
    )
    with pytest.raises(
        RuntimeError,
        match="scientific input bytes changed during resumed production",
    ):
        module._autotune_resume_from_results(
            config=config,
            outdir=tmp_path,
            prev=mutation_previous,
        )
    base_data.write_text("placeholder\n")

    install_calls.clear()
    cached = module._autotune_resume_from_results(
        config=config,
        outdir=tmp_path,
        prev=compact_terminal,
    )
    assert cached == compact_terminal
    assert install_calls == []
    assert len(identity_queries) == 5
