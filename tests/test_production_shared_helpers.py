from __future__ import annotations

import pytest

pytest.importorskip("ase")


def test_autotune_binds_shared_production_helpers():
    from vitriflow.workflows import autotune as autotune_mod
    from vitriflow.workflows import production_common as common

    names = [
        "analyse_production_box",
        "build_production_convergence_spec",
        "check_production_convergence",
        "metrics_checked_from_conv_spec",
        "plan_production_stage_diagnostics",
        "validate_production_entry_against_spec",
    ]

    for name in names:
        assert getattr(autotune_mod, name) is getattr(common, name)


def test_run_has_shared_production_executor_entrypoint():
    from vitriflow.workflows import run as run_mod

    assert callable(run_mod._run_production_executor)


def test_recommended_quench_dump_every_has_single_source_of_truth():
    from vitriflow.workflows import elastic_screen, step_counts

    assert elastic_screen.recommended_quench_dump_every is step_counts.recommended_quench_dump_every
