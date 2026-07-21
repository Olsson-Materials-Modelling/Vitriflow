from __future__ import annotations


def test_nested_convergence_group_uses_explicit_passed_value():
    from vitriflow.workflows.progress import summarise_convergence_report

    summary = summarise_convergence_report(
        {
            "passed": False,
            "groups": {
                "short": {"passed": False, "items": ["angle"]},
                "medium": {"passed": True, "items": ["ring"]},
            },
        }
    )

    assert summary["groups"] == {"short": False, "medium": True}


def test_convergence_group_preserves_legacy_boolean_schema():
    from vitriflow.workflows.progress import summarise_convergence_report

    summary = summarise_convergence_report(
        {"passed": False, "groups": {"short": False, "long": True}}
    )

    assert summary["groups"] == {"short": False, "long": True}


def test_group_mapping_without_truth_field_is_not_reported_as_passed():
    from vitriflow.workflows.progress import summarise_convergence_report

    summary = summarise_convergence_report(
        {"passed": False, "groups": {"unknown": {"items": ["density"]}}}
    )

    assert summary["groups"]["unknown"] is False
