#!/usr/bin/env python3
"""Fail-closed validation and exact two-pass comparison for Vitriflow outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


class ValidationError(RuntimeError):
    pass


CASE_SPEC = {
    "minimal_metal": {
        "engine": "lammps", "atoms": 108, "core": False, "nprocs": 2,
        "thermostat_tuple": (0.001, 0.1, 1.0),
        "temperature_grid": (400.0, 2000.0, 400.0),
        "rates": (200.0, 100.0, 50.0),
        "metric_counts": {"bondlen": 1, "coord": 1, "angle": 1, "gr": 2, "sq": 2, "void": 2},
    },
    "sio2_bks": {
        "engine": "lammps", "atoms": 192, "core": True, "nprocs": 2,
        "thermostat_tuple": (0.0005, 0.05, 0.5),
        # melt-bracketing window (onset ~6000-6500 K): 11 regular points 5000..10000
        "temperature_grid": (5000.0, 10000.0, 500.0),
        "rates": (200.0, 100.0, 50.0),
        "metric_counts": {"bondlen": 1, "coord": 2, "angle": 2, "gr": 4, "sq": 4, "void": 2},
    },
    "sio2_kim": {
        "engine": "lammps", "atoms": 192, "core": True, "nprocs": 2,
        "thermostat_tuple": (0.0005, 0.05, 0.5),
        # melt-bracketing window (onset ~5500 K): 11 regular points 3000..8000
        "temperature_grid": (3000.0, 8000.0, 500.0),
        "rates": (200.0, 100.0, 50.0),
        "metric_counts": {"bondlen": 1, "coord": 2, "angle": 2, "gr": 4, "sq": 4, "void": 2},
    },
    "si_cp2k": {
        "engine": "cp2k", "atoms": 64, "core": False,
        "thermostat_tuple": (0.5, 50.0, 5000.0),
        "temperature_grid": (500.0, 2500.0, 500.0),
        "rates": (5_000_000.0, 2_500_000.0, 1_250_000.0),
        "metric_counts": {"bondlen": 1, "coord": 1, "angle": 1, "gr": 2, "sq": 2, "void": 2},
    },
}


def fail(message: str) -> None:
    raise ValidationError(message)


def require(condition: Any, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path) -> Any:
    def bad_constant(value: str) -> None:
        fail(f"{path}: non-standard JSON numeric token {value!r}")

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=bad_constant)
    except ValidationError:
        raise
    except Exception as exc:
        fail(f"cannot read strict JSON {path}: {exc}")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_map(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def as_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def check_count(value: Any, expected: int, label: str) -> None:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    require(value == expected, f"{label}: expected {expected}, got {value}")


def _positive_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{label} must be a positive integer")
    return int(value)


def _require_float_sequence(actual: Sequence[Any], expected: Sequence[float], label: str) -> list[float]:
    check_count(len(actual), len(expected), f"{label} length")
    require(all(finite(value) for value in actual), f"{label} contains a non-finite value")
    values = [float(value) for value in actual]
    require(len(set(values)) == len(values), f"{label} contains duplicate values")
    require(
        all(math.isclose(got, want, rel_tol=0.0, abs_tol=1.0e-12) for got, want in zip(values, expected)),
        f"{label} drifted: expected {list(expected)!r}, got {values!r}",
    )
    return values


def _lammps_data_atom_count(path: Path, label: str) -> int:
    require(path.is_file() and path.stat().st_size > 0, f"{label} is missing/empty: {path}")
    match = re.search(r"(?m)^\s*(\d+)\s+atoms\s*$", path.read_text(encoding="utf-8"))
    require(match is not None, f"{label} does not declare a LAMMPS atom count: {path}")
    return int(match.group(1))


def _finite_numeric_tree(value: Any, label: str) -> None:
    if isinstance(value, list):
        require(value, f"{label} is empty")
        for index, item in enumerate(value):
            _finite_numeric_tree(item, f"{label}[{index}]")
        return
    require(finite(value), f"{label} contains a non-finite/non-numeric value")


def _assert_fixed_look_contract(value: Any, label: str) -> Mapping[str, Any]:
    contract = as_map(value, label)
    require(contract.get("interval_method") == "fixed_n_per_look",
            f"{label}: interval method is not fixed_n_per_look")
    require(contract.get("sequentially_valid") is False,
            f"{label}: repeated-look intervals are incorrectly presented as sequentially valid")
    require(contract.get("optional_stopping_coverage_guaranteed") is False,
            f"{label}: optional-stopping coverage qualification is missing")
    return contract


def _assert_convergence_degree(report: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    require(report.get("mode") == "both", f"{label}: CI and stability modes were not both exercised")
    require(report.get("passed") is True and report.get("converged") is True,
            f"{label}: convergence report did not pass")
    _assert_fixed_look_contract(report.get("inference_contract"), f"{label}.inference_contract")
    integrity = as_map(report.get("criteria_integrity"), f"{label}.criteria_integrity")
    require(integrity.get("passed") is True, f"{label}: convergence criteria integrity failed")
    check_count(integrity.get("blocking_issue_count"), 0, f"{label}.criteria_integrity.blocking_issue_count")
    require(as_list(integrity.get("blocking_issues"), f"{label}.criteria_integrity.blocking_issues") == [],
            f"{label}: convergence has blocking criteria-integrity issues")

    degree = as_map(report.get("convergence_degree"), f"{label}.convergence_degree")
    for section in ("ci", "stability"):
        item = as_map(degree.get(section), f"{label}.convergence_degree.{section}")
        n_checked = _positive_int(item.get("n_checked"), f"{label}.{section}.n_checked")
        check_count(item.get("n_passed"), n_checked, f"{label}.{section}.n_passed")
        require(math.isclose(float(item.get("pass_fraction", float("nan"))), 1.0,
                             rel_tol=0.0, abs_tol=1.0e-15),
                f"{label}: {section} convergence pass fraction is not one")
        require(as_list(item.get("failed_items"), f"{label}.{section}.failed_items") == [],
                f"{label}: {section} convergence contains failed items")
    criteria_degree = as_map(degree.get("criteria_integrity"), f"{label}.convergence_degree.criteria_integrity")
    require(criteria_degree.get("passed") is True,
            f"{label}: criteria-integrity convergence degree did not pass")
    overall = as_map(degree.get("overall"), f"{label}.convergence_degree.overall")
    require(overall.get("assessed") is True and overall.get("converged") is True,
            f"{label}: overall convergence degree is not assessed/converged")
    n_overall = _positive_int(overall.get("n_checked"), f"{label}.overall.n_checked")
    check_count(overall.get("n_passed"), n_overall, f"{label}.overall.n_passed")
    require(overall.get("status") == "converged" and
            as_list(overall.get("unassessed_active_sections"), f"{label}.overall.unassessed_active_sections") == [],
            f"{label}: overall convergence has an unassessed active section")
    require(isinstance(report.get("achieved_convergence_degree"), Mapping),
            f"{label}: achieved convergence-degree diagnostic is missing")
    return degree


def _assert_criterion_coverage(value: Any, label: str) -> None:
    coverage = as_map(value, label)
    require(set(("ci", "stability", "criteria_integrity", "overall")).issubset(coverage),
            f"{label}: criterion coverage is incomplete")
    for section in ("ci", "stability"):
        item = as_map(coverage.get(section), f"{label}.{section}")
        n_checked = _positive_int(item.get("n_checked"), f"{label}.{section}.n_checked")
        check_count(item.get("n_passed"), n_checked, f"{label}.{section}.n_passed")
    require(as_map(coverage.get("criteria_integrity"), f"{label}.criteria_integrity").get("passed") is True,
            f"{label}: criteria-integrity coverage did not pass")
    overall = as_map(coverage.get("overall"), f"{label}.overall")
    require(overall.get("assessed") is True and overall.get("converged") is True,
            f"{label}: overall criterion coverage is not assessed/converged")


METRIC_PREFIXES = ("bondlen_", "coord_", "angle_", "ring_", "gr_", "sq_", "void_")


def _metric_may_be_undefined(metric_name: str) -> bool:
    """Structural descriptors that can be legitimately undefined for a valid
    structure, where Vitriflow returns None instead of fabricating a value.

    - ``sq_*_peak_fwhm``: the half-maximum width is undefined when the principal
      S(q) peak sits on a background that never falls back to half its height
      (e.g. the total and Si-Si structure factors, which asymptote toward 1).
      Partials that do fall below half-max (Si-O, O-O) still carry a finite
      width, so this is scoped narrowly and only tolerates a genuinely
      unresolvable width -- it does not tolerate a missing peak position/height.
    """
    return metric_name.startswith("sq_") and metric_name.endswith("_peak_fwhm")


def _assert_metric_families(value: Any, label: str) -> Mapping[str, Any]:
    metrics = as_map(value, label)
    for prefix in METRIC_PREFIXES:
        names = [str(name) for name in metrics if str(name).startswith(prefix)]
        require(names, f"{label}: analytical metric family {prefix!r} is absent")
        require(any(finite(metrics[name]) for name in names),
                f"{label}: analytical metric family {prefix!r} has no finite value")
    return metrics


def _assert_distribution_families(value: Any, spec: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    distributions = as_map(value, label)
    counts = as_map(spec.get("metric_counts"), f"{label}.expected_metric_counts")
    curve_keys = {
        "bondlen": ("x", "cdf"),
        "coord": ("x", "cdf"),
        "angle": ("x", "cdf"),
        "gr": ("r", "g"),
        "sq": ("q", "s"),
        "void": ("x", "cdf"),
    }
    for family, (grid_key, value_key) in curve_keys.items():
        payloads = as_map(distributions.get(family), f"{label}.{family}")
        check_count(len(payloads), int(counts[family]), f"{label}.{family} configured curve count")
        for name, raw_payload in payloads.items():
            payload = as_map(raw_payload, f"{label}.{family}.{name}")
            grid = as_list(payload.get(grid_key), f"{label}.{family}.{name}.{grid_key}")
            values = as_list(payload.get(value_key), f"{label}.{family}.{name}.{value_key}")
            require(len(grid) >= 2 and len(grid) == len(values),
                    f"{label}.{family}.{name}: curve grid/value lengths are invalid")
            _finite_numeric_tree(grid, f"{label}.{family}.{name}.{grid_key}")
            _finite_numeric_tree(values, f"{label}.{family}.{name}.{value_key}")
    return distributions


def _assert_effective_convergence_families(report: Mapping[str, Any], spec: Mapping[str, Any], label: str) -> None:
    effective = as_map(report.get("convergence_spec_effective"), f"{label}.convergence_spec_effective")
    counts = as_map(spec.get("metric_counts"), f"{label}.expected_metric_counts")
    fields = {
        "bondlen": "bondlen_names", "coord": "coord_names", "angle": "angle_names",
        "gr": "gr_labels", "sq": "sq_labels", "void": "void_names",
    }
    for family, field in fields.items():
        names = as_list(effective.get(field), f"{label}.convergence_spec_effective.{field}")
        check_count(len(names), int(counts[family]), f"{label} active {family} convergence descriptors")
        require(len(set(str(name) for name in names)) == len(names),
                f"{label}: duplicate active {family} convergence descriptors")
    require(bool(as_list(effective.get("ring_keys"), f"{label}.convergence_spec_effective.ring_keys")),
            f"{label}: ring convergence descriptors are absent")


def _assert_convergence_evidence_coverage(
    report: Mapping[str, Any],
    *,
    n_boxes: int,
    label: str,
) -> Mapping[str, Any]:
    """Require every active criterion to use a strict majority of the ensemble."""

    coverage = as_map(report.get("evidence_coverage"), f"{label}.evidence_coverage")
    require(
        coverage.get("schema") == "vitriflow.convergence_evidence_coverage.v1",
        f"{label}: convergence evidence-coverage schema is absent or wrong",
    )
    require(
        coverage.get("policy")
        == "n_contributing_boxes_strictly_greater_than_fraction_of_accepted_ensemble",
        f"{label}: convergence evidence policy is not the strict-majority contract",
    )
    require(
        finite(coverage.get("minimum_evidence_fraction_exclusive"))
        and math.isclose(
            float(coverage["minimum_evidence_fraction_exclusive"]),
            0.5,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        f"{label}: convergence evidence fraction must be exactly one half, exclusive",
    )
    check_count(coverage.get("n_boxes_total"), int(n_boxes), f"{label}.evidence_coverage.n_boxes_total")
    minimum_required = int(math.floor(0.5 * float(n_boxes)) + 1)
    check_count(
        coverage.get("minimum_boxes_required"),
        minimum_required,
        f"{label}.evidence_coverage.minimum_boxes_required",
    )
    require(coverage.get("passed") is True, f"{label}: convergence evidence coverage did not pass")
    require(
        as_list(coverage.get("insufficient_items"), f"{label}.evidence_coverage.insufficient_items") == [],
        f"{label}: at least one convergence criterion lacks strict-majority evidence",
    )
    require(
        as_list(coverage.get("missing_families"), f"{label}.evidence_coverage.missing_families") == [],
        f"{label}: at least one configured convergence family is missing",
    )

    active_sections = set(
        str(value)
        for value in as_list(
            coverage.get("active_sections"),
            f"{label}.evidence_coverage.active_sections",
        )
    )
    require(active_sections == {"ci", "stability"}, f"{label}: CI and stability evidence are not both active")
    items = as_map(coverage.get("items"), f"{label}.evidence_coverage.items")
    require(items, f"{label}: convergence evidence item inventory is empty")
    for name, raw in items.items():
        item = as_map(raw, f"{label}.evidence_coverage.items.{name}")
        require(str(item.get("section")) in active_sections,
                f"{label}: evidence item {name!r} belongs to an inactive section")
        check_count(
            item.get("minimum_boxes_required"),
            minimum_required,
            f"{label}.evidence_coverage.items.{name}.minimum_boxes_required",
        )
        n_contributing = _positive_int(
            item.get("n_contributing_boxes"),
            f"{label}.evidence_coverage.items.{name}.n_contributing_boxes",
        )
        require(
            n_contributing >= minimum_required and item.get("strict_majority_supported") is True,
            f"{label}: evidence item {name!r} is not supported by a strict majority of boxes",
        )
        require(
            str(item.get("group")) in {"short", "medium", "long"},
            f"{label}: evidence item {name!r} has no short/medium/long classification",
        )

    families = as_map(coverage.get("families"), f"{label}.evidence_coverage.families")
    configured = {
        str(value)
        for value in as_list(
            coverage.get("configured_metric_families"),
            f"{label}.evidence_coverage.configured_metric_families",
        )
    }
    require(configured and configured == set(str(name) for name in families),
            f"{label}: configured-family inventory and family coverage disagree")
    for family, raw in families.items():
        item = as_map(raw, f"{label}.evidence_coverage.families.{family}")
        require(item.get("covered") is True,
                f"{label}: configured convergence family {family!r} is not covered")
        require(str(item.get("group")) in {"short", "medium", "long"},
                f"{label}: convergence family {family!r} has no physical-length group")

    groups = as_map(coverage.get("groups"), f"{label}.evidence_coverage.groups")
    require(set(groups) == {"short", "medium", "long"},
            f"{label}: convergence does not report exactly short/medium/long groups")
    for group in ("short", "medium", "long"):
        item = as_map(groups.get(group), f"{label}.evidence_coverage.groups.{group}")
        require(
            item.get("configured") is True
            and item.get("covered") is True
            and item.get("status") == "covered",
            f"{label}: {group}-range convergence evidence is not configured and covered",
        )
        require(
            bool(as_list(item.get("required_families"), f"{label}.evidence_coverage.groups.{group}.required_families")),
            f"{label}: {group}-range group has no required metric family",
        )
    return coverage


def _metric_descriptor_inventory(boxes: Sequence[Mapping[str, Any]], label: str) -> set[str]:
    """Descriptors actually emitted with finite majority-box evidence."""

    require(bool(boxes), f"{label}: no boxes are available for metric inventory")
    minimum_required = int(math.floor(0.5 * float(len(boxes))) + 1)
    scalar_names: set[str] = set()
    for box in boxes:
        scalar_names.update(str(name) for name in as_map(box.get("metrics"), f"{label}.metrics"))
    finite_scalars = {
        name
        for name in scalar_names
        if sum(
            finite(as_map(box.get("metrics"), f"{label}.metrics").get(name))
            for box in boxes
        )
        >= minimum_required
    }
    distribution_names: set[str] = set()
    for family in ("bondlen", "coord", "angle", "gr", "sq", "void"):
        family_sets = [
            set(
                str(name)
                for name in as_map(
                    as_map(box.get("distributions"), f"{label}.distributions").get(family),
                    f"{label}.distributions.{family}",
                )
            )
            for box in boxes
        ]
        require(family_sets and all(names == family_sets[0] for names in family_sets[1:]),
                f"{label}: emitted {family} descriptor identities differ between boxes")
        distribution_names.update(family_sets[0])
    return {"density", *finite_scalars, *distribution_names}


def _is_structurally_diagnostic(name: str) -> bool:
    """True for a metric descriptor whose family has no ConvergenceConfig
    tolerance and therefore can never enter convergence (amorphous order
    parameters, S(q) peaks, void clearances, ring topology counts, bond
    incidence). These are excluded from strict-majority coverage denominators so
    that simple systems with few convergence-eligible families are not
    penalised."""
    s = str(name)
    if s.startswith(("amorphous_", "void_clearance_", "bond_incidence_")):
        return True
    if s in ("ring_count", "ring_entropy"):
        return True
    # S(q) PEAK descriptors have no tolerance family (unlike g(r) peaks or the
    # S(q) curve, both of which do).
    if s.startswith("sq_") and "_peak" in s:
        return True
    return False


def _assert_majority_of_emitted_metrics_enter_convergence(
    report: Mapping[str, Any],
    boxes: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    emitted = _metric_descriptor_inventory(boxes, label)
    checked = {str(value) for value in as_list(report.get("metrics_checked"), f"{label}.metrics_checked")}
    # Some callers store metrics_checked alongside rather than inside the
    # convergence report.  The caller may inject it into a shallow copy.
    require(checked, f"{label}: convergence metric inventory is empty")
    unknown = checked - emitted
    require(not unknown, f"{label}: convergence names descriptors that were not emitted: {sorted(unknown)!r}")
    # Strict majority measured over convergence-ELIGIBLE descriptors only
    # (structurally-diagnostic families excluded; see _is_structurally_diagnostic).
    eligible = {name for name in emitted if not _is_structurally_diagnostic(name)}
    require(eligible, f"{label}: no convergence-eligible metric descriptors were emitted")
    ratio = float(len(checked & eligible)) / float(len(eligible))
    require(
        ratio > 0.5,
        f"{label}: only {len(checked & eligible)}/{len(eligible)} convergence-ELIGIBLE "
        f"metric descriptors enter convergence "
        f"({len(emitted) - len(eligible)} structurally-diagnostic descriptors excluded)",
    )
    return {
        "emitted_descriptor_count": len(emitted),
        "convergence_descriptor_count": len(checked & emitted),
        "convergence_descriptor_fraction": ratio,
        "diagnostic_only_descriptors": sorted(emitted - checked),
    }


def _assert_metric_plumbing_coverage(
    report: Mapping[str, Any],
    metrics_checked: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Validate the producer-owned emitted-to-convergence inventory.

    The independently derived ratio above catches data-dependent omissions.
    This assertion separately consumes the canonical coverage object emitted
    by the production convergence implementation, so the application test
    cannot pass merely because the checker reconstructed a favourable subset.
    """

    coverage = as_map(
        report.get("metric_plumbing_coverage"),
        f"{label}.metric_plumbing_coverage",
    )
    require(
        coverage.get("schema") == "vitriflow.metric_plumbing_coverage.v1",
        f"{label}: metric-plumbing coverage schema is absent or wrong",
    )
    n_emitted = _positive_int(
        coverage.get("n_emitted_descriptors"),
        f"{label}.metric_plumbing_coverage.n_emitted_descriptors",
    )
    n_convergence = _positive_int(
        coverage.get("n_convergence_descriptors"),
        f"{label}.metric_plumbing_coverage.n_convergence_descriptors",
    )
    n_diagnostic = coverage.get("n_diagnostic_only_descriptors")
    require(
        isinstance(n_diagnostic, int)
        and not isinstance(n_diagnostic, bool)
        and n_diagnostic >= 0,
        f"{label}: diagnostic-only descriptor count is invalid",
    )
    check_count(
        n_convergence + int(n_diagnostic),
        n_emitted,
        f"{label}.metric_plumbing_coverage partition size",
    )
    fraction = coverage.get("fraction_entering_convergence")
    expected_fraction = float(n_convergence) / float(n_emitted)
    require(
        finite(fraction)
        and math.isclose(
            float(fraction),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        f"{label}: metric-plumbing fraction disagrees with its descriptor counts",
    )
    # Strict-majority coverage is measured over convergence-ELIGIBLE descriptors
    # only. The producer assigns a convergence "family" solely to descriptors it
    # can map to a ConvergenceConfig tolerance; structurally-diagnostic
    # descriptors (amorphous order parameters, S(q) peaks, void clearances, ring
    # topology counts, bond incidence) carry no family and can never enter
    # convergence. Excluding them from the denominator stops simple systems
    # (e.g. monatomic Al: one Al-Al pair, hence few convergence-eligible
    # families, but the same system-independent diagnostic set) from being
    # penalised, while still requiring a strict majority of *eligible*
    # descriptors to actually enter convergence (catching an eligible descriptor
    # silently dropped for non-finiteness).
    diagnostic_rows = coverage.get("diagnostic_only_descriptors") or []
    n_eligible_diagnostic = sum(
        1
        for row in diagnostic_rows
        if isinstance(row, dict) and row.get("family") is not None
    )
    n_eligible = int(n_convergence) + int(n_eligible_diagnostic)
    require(
        n_eligible > 0,
        f"{label}: no convergence-eligible descriptors were emitted",
    )
    eligible_fraction = float(n_convergence) / float(n_eligible)
    require(
        eligible_fraction > 0.5,
        f"{label}: fewer than a strict majority of convergence-ELIGIBLE "
        f"descriptors enter convergence "
        f"({n_convergence}/{n_eligible}={eligible_fraction:.3f}; "
        f"{int(n_emitted) - n_eligible} structurally-diagnostic descriptors excluded)",
    )

    convergence_rows = [
        as_map(row, f"{label}.metric_plumbing_coverage.convergence_descriptors")
        for row in as_list(
            coverage.get("convergence_descriptors"),
            f"{label}.metric_plumbing_coverage.convergence_descriptors",
        )
    ]
    diagnostic_rows = [
        as_map(row, f"{label}.metric_plumbing_coverage.diagnostic_only_descriptors")
        for row in as_list(
            coverage.get("diagnostic_only_descriptors"),
            f"{label}.metric_plumbing_coverage.diagnostic_only_descriptors",
        )
    ]
    check_count(
        len(convergence_rows),
        n_convergence,
        f"{label}.metric_plumbing_coverage convergence inventory length",
    )
    check_count(
        len(diagnostic_rows),
        int(n_diagnostic),
        f"{label}.metric_plumbing_coverage diagnostic inventory length",
    )

    def descriptor_identity(row: Mapping[str, Any], row_label: str) -> tuple[str, str]:
        kind = str(row.get("kind", ""))
        name = str(row.get("name", ""))
        require(kind in {"scalar", "distribution"} and bool(name),
                f"{row_label}: descriptor kind/name is invalid")
        return kind, name

    convergence_ids = {
        descriptor_identity(row, f"{label}.convergence_descriptor")
        for row in convergence_rows
    }
    diagnostic_ids = {
        descriptor_identity(row, f"{label}.diagnostic_descriptor")
        for row in diagnostic_rows
    }
    check_count(len(convergence_ids), n_convergence,
                f"{label}.metric_plumbing_coverage unique convergence descriptors")
    check_count(len(diagnostic_ids), int(n_diagnostic),
                f"{label}.metric_plumbing_coverage unique diagnostic descriptors")
    require(not (convergence_ids & diagnostic_ids),
            f"{label}: a descriptor is simultaneously convergence and diagnostic-only")

    groups: set[str] = set()
    for row in convergence_rows:
        require(row.get("role") == "convergence",
                f"{label}: convergence inventory contains a non-convergence role")
        require(bool(str(row.get("family", "")).strip()),
                f"{label}: convergence descriptor has no metric family")
        group = str(row.get("group", ""))
        require(group in {"short", "medium", "long"},
                f"{label}: convergence descriptor has no short/medium/long classification")
        groups.add(group)
    require(groups == {"short", "medium", "long"},
            f"{label}: metric-plumbing convergence inventory does not span short, medium, and long range")
    for row in diagnostic_rows:
        require(row.get("role") == "diagnostic_only" and bool(str(row.get("reason", "")).strip()),
                f"{label}: diagnostic-only metric lacks an explicit reason")

    checked = [
        str(value)
        for value in as_list(metrics_checked, f"{label}.metrics_checked")
    ]
    require(checked and len(checked) == len(set(checked)),
            f"{label}: metrics_checked is empty or contains duplicate descriptors")
    reported_checked = {name for _kind, name in convergence_ids}
    require(
        set(checked) == reported_checked,
        f"{label}: metrics_checked and producer metric-plumbing inventory disagree; "
        f"only_checked={sorted(set(checked) - reported_checked)!r}, "
        f"only_coverage={sorted(reported_checked - set(checked))!r}",
    )
    require(("scalar", "density") in convergence_ids,
            f"{label}: density is absent from metric-plumbing convergence coverage")
    return coverage


def _strictly_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _result_artifact(root: Path, raw_path: Any, label: str) -> Path:
    require(isinstance(raw_path, str) and raw_path.strip(), f"{label} has no artifact path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    require(_strictly_under(path, root), f"{label} escapes the result root: {path}")
    require(path.is_file() and path.stat().st_size > 0, f"{label} is missing/empty: {path}")
    return path


def _pdf_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 100 and path.read_bytes()[:5] == b"%PDF-"


def _csv_has_data(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file() and path.stat().st_size > 0, f"{label} CSV is missing/empty: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    require(fields, f"{label} CSV has no header: {path}")
    require(rows, f"{label} CSV has no data rows: {path}")
    return fields, rows


def stage_dir_from_result(result_path: Path) -> Path:
    data = as_map(load_json(result_path), str(result_path))
    production = as_map(data.get("production"), "production")
    boxes = as_list(production.get("boxes"), "production.boxes")
    require(boxes, "production.boxes is empty")
    box = as_map(boxes[0], "production.boxes[0]")
    paths = as_map(box.get("paths"), "production.boxes[0].paths")
    rel = paths.get("relax_dir")
    require(isinstance(rel, str) and rel.strip(), "first production box has no paths.relax_dir")
    candidate = Path(rel)
    if not candidate.is_absolute():
        candidate = result_path.parent / candidate
    candidate = candidate.resolve(strict=False)
    require(_strictly_under(candidate, result_path.parent), f"resolved stage escapes result root: {candidate}")
    require(candidate.is_dir(), f"resolved relax stage directory is missing: {candidate}")
    return candidate


def check_config(
    path: Path,
    case: str,
    *,
    expect_cell_refinement: bool = False,
) -> dict[str, Any]:
    spec = CASE_SPEC.get(case)
    require(spec is not None, f"unknown case {case!r}")
    try:
        import yaml
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot parse raw YAML configuration {path}: {exc}")
    raw_config = as_map(raw_config, f"{path} raw configuration")
    raw_autotune = as_map(raw_config.get("autotune"), f"{path}.autotune")
    require(
        "size" not in raw_autotune,
        f"{case}: validation YAML must omit autotune.size entirely; the default-disabled stage is not a test target",
    )
    try:
        from vitriflow.config import RunConfig
    except Exception as exc:
        fail(f"cannot import Vitriflow configuration model: {exc}")
    try:
        cfg = RunConfig.from_yaml(path)
    except Exception as exc:
        fail(f"configuration does not validate: {path}: {exc}")

    require(str(cfg.engine) == spec["engine"], f"{case}: wrong configured engine")
    if spec["engine"] == "lammps":
        check_count(cfg.lammps.nprocs, int(spec.get("nprocs", 2)), f"{case}.lammps.nprocs")
        require(str(cfg.lammps.mpi_cmd or "").strip() == "mpirun",
                f"{case}: LAMMPS MPI launcher drifted")
    else:
        require(cfg.cp2k is not None, f"{case}: CP2K configuration is absent")
        check_count(cfg.cp2k.nprocs, 4, f"{case}.cp2k.nprocs")
        require(str(cfg.cp2k.mpi_cmd or "").strip() == "mpirun",
                f"{case}: CP2K MPI launcher drifted")
    auto = cfg.autotune
    require(auto is not None, f"{case}: autotune block is absent")
    require(auto.size.enabled is False, f"{case}: finite-size stage must remain disabled")
    check_count(auto.tm_scan.replicates_per_temp, 3, f"{case}.tm_scan.replicates_per_temp")
    check_count(auto.highT.replicates, 3, f"{case}.highT.replicates")
    check_count(auto.quench.replicates_per_rate, 3, f"{case}.quench.replicates_per_rate")
    rates = list(auto.quench.rates_K_per_ps or [])
    check_count(len(rates), 3, f"{case} configured rate count")
    require(tuple(float(x) for x in rates) == tuple(spec["rates"]), f"{case}: cooling-rate grid drifted")
    require(all(float(rates[i]) > float(rates[i + 1]) > 0.0 for i in range(len(rates) - 1)),
            f"{case}: cooling rates must be unique, positive, and strictly descending")
    tm = auto.tm_scan
    expected_tmin, expected_tmax, expected_dt = spec["temperature_grid"]
    expected_span = (expected_tmax - expected_tmin) / expected_dt
    require(math.isclose(expected_span, round(expected_span), rel_tol=0.0, abs_tol=1.0e-9),
            f"{case}: declared temperature_grid is not a regular grid")
    n_points = int(round(expected_span)) + 1
    span = (float(tm.t_max) - float(tm.t_min)) / float(tm.dT)
    require(math.isclose(span, expected_span, rel_tol=0.0, abs_tol=1.0e-12),
            f"{case}: T grid is not the expected {n_points} regular points")
    require(
        math.isclose(float(tm.t_min), expected_tmin)
        and math.isclose(float(tm.t_max), expected_tmax)
        and math.isclose(float(tm.dT), expected_dt),
        f"{case}: broad {n_points}-temperature window drifted",
    )
    require(tm.gr.enabled is True, f"{case}: Tm g(r) indicator is disabled")
    check_count(tm.gr.frames, 6, f"{case}.tm_scan.gr.frames")
    preflight = auto.preflight
    require([str(value) for value in list(preflight.ensembles or [])] == ["npt"],
            f"{case}: preflight must expose only the NPT ensemble")
    require(preflight.allow_nvt_fallback is False and preflight.allow_implicit_dt_fallback is False,
            f"{case}: preflight fallback would introduce an undeclared setting")
    check_count(len(list(preflight.dt_candidates or [])), 1, f"{case} preflight timestep candidates")
    check_count(len(list(preflight.tdamp_factors)), 1, f"{case} preflight TDAMP candidates")
    check_count(len(list(preflight.pdamp_factors)), 1, f"{case} preflight PDAMP candidates")
    expected_dt_md, expected_tdamp, expected_pdamp = spec["thermostat_tuple"]
    require(math.isclose(float(cfg.md.timestep), expected_dt_md, rel_tol=0.0, abs_tol=1.0e-15),
            f"{case}: configured timestep drifted")
    require(math.isclose(float(cfg.md.thermostat.tdamp), expected_tdamp, rel_tol=0.0, abs_tol=1.0e-15),
            f"{case}: configured TDAMP/TIMECON drifted")
    require(math.isclose(float(cfg.md.barostat.pdamp), expected_pdamp, rel_tol=0.0, abs_tol=1.0e-12),
            f"{case}: configured PDAMP/TIMECON drifted")
    require(math.isclose(float(preflight.dt_candidates[0]), expected_dt_md, rel_tol=0.0, abs_tol=1.0e-15),
            f"{case}: sole preflight timestep differs from md.timestep")
    require(math.isclose(float(preflight.tdamp_factors[0]) * expected_dt_md, expected_tdamp,
                         rel_tol=0.0, abs_tol=1.0e-12), f"{case}: sole TDAMP factor does not resolve to the configured value")
    require(math.isclose(float(preflight.pdamp_factors[0]) * expected_dt_md, expected_pdamp,
                         rel_tol=0.0, abs_tol=1.0e-9), f"{case}: sole PDAMP factor does not resolve to the configured value")
    prod = auto.production
    require(prod.enabled is True and prod.check_convergence is True, f"{case}: adaptive production convergence is not enabled")
    check_count(prod.min_boxes, 6, f"{case}.production.min_boxes")
    check_count(prod.batch_boxes, 4, f"{case}.production.batch_boxes")
    check_count(prod.max_boxes, 10, f"{case}.production.max_boxes")
    check_count(prod.consecutive_converged_checks, 2, f"{case}.production.consecutive_converged_checks")
    require(prod.store_distributions is True and prod.embed_structures is True, f"{case}: production evidence storage is incomplete")
    require(auto.convergence.mode == "both" and auto.convergence.stability_split == "half",
            f"{case}: convergence must exercise CI and a half-ensemble stability split")
    metrics = auto.metrics
    require(metrics.enabled is True and metrics.voids.enabled is True, f"{case}: structural/void metrics are not enabled")
    check_count(metrics.time_average_frames, 6, f"{case}.metrics.time_average_frames")
    metric_counts = spec["metric_counts"]
    check_count(len(list(metrics.pairs)), int(metric_counts["bondlen"]), f"{case}.metrics.pairs")
    check_count(len(list(metrics.coordinations)), int(metric_counts["coord"]), f"{case}.metrics.coordinations")
    check_count(len(list(metrics.angles)), int(metric_counts["angle"]), f"{case}.metrics.angles")
    check_count(len(list(metrics.gr)), int(metric_counts["gr"]), f"{case}.metrics.gr")
    check_count(len(list(metrics.sq)), int(metric_counts["sq"]), f"{case}.metrics.sq")
    require(metrics.rings.enabled is True and metrics.coordination_sweep.enabled is True,
            f"{case}: ring or coordination-sweep analytical plumbing is disabled")
    require(metrics.collect_during_production_stages is True and metrics.stage_timeseries_make_plot is True,
            f"{case}: automatic stage metric/plot collection is disabled")
    require(metrics.amorphous.enabled is True, f"{case}: amorphous diagnostic plumbing is disabled")
    require(prod.dump_trajectory is True and int(prod.dump_every_steps) == 1,
            f"{case}: production trajectories are not available at validation cadence")
    if spec["engine"] == "lammps":
        elastic = metrics.elastic
        require(elastic.enabled is True and elastic.run_on_relax is True and elastic.make_plot is True,
                f"{case}: direct elastic screening/plotting is disabled")
        require(elastic.collect_during_production_stages is True and elastic.stage_timeseries_make_plot is True,
                f"{case}: elastic stage-timeseries plumbing is disabled")
    else:
        require(metrics.elastic.enabled is False, f"{case}: unsupported CP2K elastic screen must be disabled")

    structure = cfg.structure
    if case == "minimal_metal":
        require(cfg.kim.model == "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "minimal_metal KIM model drifted")
        require(list(cfg.kim.interactions or []) == ["Al"], "minimal_metal species mapping drifted")
        require(structure.lammps_data is not None, "minimal_metal must use the packaged 108-atom data file")
        data_path = Path(structure.lammps_data)
        require(data_path.is_file(), f"minimal_metal packaged data file is missing: {data_path}")
        atom_match = re.search(r"(?m)^\s*(\d+)\s+atoms\s*$", data_path.read_text(encoding="utf-8"))
        require(atom_match is not None and int(atom_match.group(1)) == 108,
                "minimal_metal packaged LAMMPS data does not declare 108 atoms")
    elif case in {"sio2_bks", "sio2_kim"}:
        generated = structure.generate
        require(generated is not None and generated.builtin_name == "beta_cristobalite", f"{case}: wrong builtin structure")
        require(tuple(generated.repeat or ()) == (2, 2, 2), f"{case}: SiO2 repeat must be 2x2x2 (192 atoms)")
        core = cfg.kim.core_repulsion
        require(core.enabled is True and core.tabulate is True, f"{case}: C2 tabulated core is disabled")
        check_count(len(list(core.dt_candidates or [])), 1, f"{case} core timestep candidates")
        require(math.isclose(float(core.dt_candidates[0]), expected_dt_md, rel_tol=0.0, abs_tol=1.0e-15),
                f"{case}: core stability timestep differs from the sole MD timestep")
        if case == "sio2_kim":
            require(cfg.kim.model == "Sim_LAMMPS_Buckingham_CarreHorbachIspas_2008_SiO__SM_886641404623_000",
                    "sio2_kim legacy KIM model drifted")
            require(list(cfg.kim.interactions or []) == ["Si", "O"], "sio2_kim species/type mapping drifted")
            require(structure.charges is None, "sio2_kim must obtain fixed charges from KIM at runtime")
        else:
            require(structure.charges == {"Si": 2.4, "O": -1.2}, "sio2_bks must declare standard BKS charges")
            commands = "\n".join(str(line) for line in list(cfg.kim.commands or []))
            require("pair_style buck/coul/long 10.0" in commands, "sio2_bks must retain the 10 Å BKS cutoff")
            for expected_command in (
                "pair_coeff 1 1 0.0 1.0 0.0",
                "pair_coeff 1 2 18003.7572 0.205205 133.5381",
                "pair_coeff 2 2 1388.7730 0.362319 175.0",
                "kspace_style pppm 1.0e-4",
            ):
                require(expected_command in commands, f"sio2_bks standard parameter missing: {expected_command}")
            require("hybrid/overlay" not in commands.lower(), "sio2_bks must not use the old additive core overlay")
    elif case == "si_cp2k":
        generated = structure.generate
        require(generated is not None and generated.builtin_name == "si_diamond", "si_cp2k: wrong builtin structure")
        check_count(generated.n_formula_units, 64, "si_cp2k formula-unit/atom count")
        kind = cfg.cp2k.kind_settings.get("Si") if cfg.cp2k is not None else None
        require(kind is not None and kind.basis_set == "SZV-MOLOPT-SR-GTH", "si_cp2k must use the minimal SZV basis")
        require(kind.potential == "GTH-PBE-q4" and cfg.cp2k.xc_functional == "PBE",
                "si_cp2k potential/functional drifted")
        require(math.isclose(float(cfg.md.thermostat.tdamp), 50.0) and math.isclose(float(cfg.md.barostat.pdamp), 5000.0),
                "si_cp2k must retain the established 50/5000 fs TIMECON values")

    dft_opt = prod.dft_opt
    if bool(expect_cell_refinement):
        require(case == "si_cp2k", "CELL_OPT validation is defined only for the CP2K Si case")
        require(dft_opt.enabled is True, "si_cp2k: requested CELL_OPT validation is not enabled")
        require(
            dft_opt.optimizer == "LBFGS"
            and int(dft_opt.max_iter) == 200
            and dft_opt.keep_angles is True
            and int(dft_opt.traj_every) == 1
            and dft_opt.print_level == "LOW",
            "si_cp2k: CELL_OPT validation contract drifted",
        )
        require(
            finite(dft_opt.external_pressure_bar)
            and math.isclose(float(dft_opt.external_pressure_bar), 1.0),
            "si_cp2k: CELL_OPT external pressure must remain 1 bar",
        )
    else:
        require(dft_opt.enabled is False, f"{case}: CELL_OPT unexpectedly enabled")

    return {
        "case": case,
        "engine": str(cfg.engine),
        "size_stage_enabled": False,
        "temperature_points": n_points,
        "temperature_replicas": 3,
        "highT_replicas": 3,
        "cooling_rates": 3,
        "rate_replicas": 3,
        "production_boxes_required": 10,
        "convergence_first_look_boxes": 6,
        "time_average_frames": 6,
        "cell_refinement_enabled": bool(dft_opt.enabled),
    }


def _check_tabulated_core_evidence(
    root: Path,
    case: str,
    *,
    joins: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report_path = root / "preflight" / "table_verify" / "refinement_report.json"
    report = as_map(load_json(report_path), str(report_path))
    require(report.get("status") == "tabulated_verified", f"{case}: tabulated core was not verified")
    require(report.get("fallback_to_analytic") is False, f"{case}: analytic/overlay core fallback was used")
    accepted = as_map(report.get("accepted_candidate"), f"{case}.accepted_candidate")
    require(accepted.get("passed") is True and accepted.get("verify_passed") is True,
            f"{case}: accepted C2 table did not pass strict verification")
    require(accepted.get("stability_ok") is True, f"{case}: accepted C2 table did not pass calibrated stability")
    require(accepted.get("verification_status") == "pass" and accepted.get("stability_status") == "pass",
            f"{case}: C2 tri-state verification/stability status is not pass/pass")
    require(list(accepted.get("blocking_warnings", []) or []) == [], f"{case}: C2 verification has blocking warnings")
    comparison = as_map(accepted.get("comparison"), f"{case}.accepted_candidate.comparison")
    overall = as_map(comparison.get("overall"), f"{case}.accepted_candidate.comparison.overall")
    for key in ("max_energy_ratio", "max_force_ratio", "max_r_error"):
        require(finite(overall.get(key)), f"{case}: accepted C2 comparison lacks finite {key}")
    require(float(overall["max_energy_ratio"]) <= 1.0 and float(overall["max_force_ratio"]) <= 1.0,
            f"{case}: accepted C2 energy/force errors exceed their verification tolerances")

    override = root / "preflight" / "potential_override"
    potential_lines = override / "potential_lines.lmp"
    require(potential_lines.is_file() and potential_lines.stat().st_size > 0,
            f"{case}: accepted potential override is missing")
    tables = sorted(path for path in override.glob("*.table") if path.is_file() and path.stat().st_size > 0)
    check_count(len(tables), 1, f"{case} accepted core table count")
    lines = potential_lines.read_text(encoding="utf-8")
    require("hybrid/overlay" not in lines.lower(), f"{case}: unbounded hybrid/overlay fallback appears in execution lines")
    require("pair_style table" in lines, f"{case}: execution lines do not select the replacement table")
    metadata_line = next(
        (line for line in lines.splitlines() if line.startswith("# vitriflow_core_table_begin ")),
        None,
    )
    require(metadata_line is not None, f"{case}: accepted potential lacks core-table metadata")
    try:
        metadata = json.loads(metadata_line.split("# vitriflow_core_table_begin ", 1)[1])
    except Exception as exc:
        fail(f"{case}: cannot parse accepted core-table metadata: {exc}")
    metadata = as_map(metadata, f"{case}.core_table_metadata")
    # Core-table schema v10: required for the analytic Buckingham autocore, whose
    # join_energy semantics differ from earlier schema revisions. The short-range
    # regularization descriptor is c2_force_blended_buckingham_to_zbl.
    require(metadata.get("version") == 10 and metadata.get("kind") == "buckingham_zbl_table",
            f"{case}: core-table metadata kind/version is not the hardened schema")
    require(metadata.get("short_range_regularization") == "c2_force_blended_buckingham_to_zbl",
            f"{case}: core metadata does not identify the C2 ZBL replacement")
    require(metadata.get("generated_by") == "vitriflow_generated", f"{case}: table provenance is not Vitriflow-generated")
    require(metadata.get("filename") == tables[0].name, f"{case}: metadata names a different table file")
    require(isinstance(metadata.get("points"), int) and int(metadata["points"]) >= 2,
            f"{case}: metadata table point count is invalid")
    check_count(metadata.get("points"), int(accepted.get("table_points")), f"{case} accepted table points")
    require(metadata.get("force_mode") == accepted.get("force_mode") and
            metadata.get("table_style") == accepted.get("table_style") and
            metadata.get("include_fprime") is True,
            f"{case}: executed table realization differs from the accepted candidate")
    require(isinstance(metadata.get("sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]) is not None,
            f"{case}: table content hash is absent")
    require(hashlib.sha256(tables[0].read_bytes()).hexdigest() == metadata["sha256"],
            f"{case}: accepted table bytes do not match their execution metadata hash")
    audit = as_map(metadata.get("runtime_charge_audit"), f"{case}.runtime_charge_audit")
    require(audit.get("passed") is True and audit.get("variable_charge_commands") is False,
            f"{case}: fixed runtime charge audit did not pass")
    require(audit.get("comparison_units") == "elementary_charge",
            f"{case}: runtime charge audit does not use elementary-charge units")
    check_count(audit.get("n_atoms"), int(CASE_SPEC[case]["atoms"]), f"{case}.runtime_charge_audit.n_atoms")
    effective_charges = as_map(audit.get("effective_charges_e"), f"{case}.runtime_charge_audit.effective_charges_e")
    require(set(effective_charges) == {"Si", "O"} and all(finite(value) for value in effective_charges.values()),
            f"{case}: runtime charge audit does not resolve finite Si/O charges")
    if case == "sio2_kim":
        require("kim" in str(audit.get("source", "")).lower(),
                f"{case}: runtime charges are not attributed to the KIM model")
        require(audit.get("explicit_set_required_for_present_types") is True,
                f"{case}: generated KIM structure charges were not proven by explicit type assignments")
    else:
        require(math.isclose(float(effective_charges["Si"]), 2.4, rel_tol=0.0, abs_tol=1.0e-12) and
                math.isclose(float(effective_charges["O"]), -1.2, rel_tol=0.0, abs_tol=1.0e-12),
                f"{case}: executed charges differ from the standard BKS charges")

    pair_rows: list[Mapping[str, Any]] = []
    for line in lines.splitlines():
        if not line.startswith("# vitriflow_core_table_pair "):
            continue
        try:
            pair_rows.append(as_map(json.loads(line.split("# vitriflow_core_table_pair ", 1)[1]),
                                    f"{case}.core_table_pair"))
        except ValidationError:
            raise
        except Exception as exc:
            fail(f"{case}: cannot parse pair-resolved core-table metadata: {exc}")
    check_count(len(pair_rows), 3, f"{case} pair metadata count")
    expected_pair_types = {(1, 1), (1, 2), (2, 2)}
    expected_species = {("Si", "Si"), ("Si", "O"), ("O", "O")}
    observed_pair_types: set[tuple[int, int]] = set()
    observed_species: set[tuple[str, str]] = set()
    join_by_species = {
        tuple(str(value) for value in as_list(row.get("species"), f"{case}.resolved join species")): row
        for row in joins
    }
    bks_parameters = {
        ("Si", "Si"): (0.0, 1.0, 0.0),
        ("Si", "O"): (18003.7572, 0.205205, 133.5381),
        ("O", "O"): (1388.7730, 0.362319, 175.0),
    }
    for index, row in enumerate(pair_rows, start=1):
        pair = tuple(_positive_int(value, f"{case}.pair[{index}].pair")
                     for value in as_list(row.get("pair"), f"{case}.pair[{index}].pair"))
        species = tuple(str(value) for value in as_list(row.get("species"), f"{case}.pair[{index}].species"))
        require(len(pair) == 2 and len(species) == 2, f"{case}: malformed pair metadata row {index}")
        observed_pair_types.add(pair)
        observed_species.add(species)
        for key in ("A", "rho", "C", "buck_cutoff", "pair_cutoff", "requested_r_in",
                    "requested_r_out", "r_in", "r_out", "join_energy", "q_i", "q_j"):
            require(finite(row.get(key)), f"{case}: pair {species!r} has non-finite {key}")
        require(float(row["rho"]) > 0.0 and float(row["buck_cutoff"]) > 0.0 and
                float(row["pair_cutoff"]) >= float(row["buck_cutoff"]),
                f"{case}: pair {species!r} has invalid Buckingham/cutoff metadata")
        require(0.0 < float(row["r_in"]) < float(row["r_out"]) < float(row["pair_cutoff"]),
                f"{case}: pair {species!r} has invalid resolved C2 join radii")
        require(math.isclose(float(row["q_i"]), float(effective_charges[species[0]]), rel_tol=0.0, abs_tol=1.0e-12) and
                math.isclose(float(row["q_j"]), float(effective_charges[species[1]]), rel_tol=0.0, abs_tol=1.0e-12),
                f"{case}: pair {species!r} table charges differ from the audited runtime charges")
        validation = as_map(row.get("validation"), f"{case}.pair[{index}].validation")
        require(validation.get("c2_join_validated") is True and
                validation.get("repulsive_through_r_out") is True and
                finite(validation.get("minimum_core_force")) and
                _positive_int(validation.get("probe_points"), f"{case}.pair[{index}].validation.probe_points") > 1,
                f"{case}: pair {species!r} lacks validated repulsive C2-core evidence")
        require(bool(as_map(row.get("join_resolution"), f"{case}.pair[{index}].join_resolution")),
                f"{case}: pair {species!r} join-resolution provenance is empty")
        join = as_map(join_by_species.get(species), f"{case}.resolved_join[{species!r}]")
        for metadata_key, result_key in (
            ("requested_r_in", "requested_r_in"), ("requested_r_out", "requested_r_out"),
            ("r_in", "resolved_r_in"), ("r_out", "resolved_r_out"),
        ):
            require(math.isclose(float(row[metadata_key]), float(join[result_key]), rel_tol=0.0, abs_tol=1.0e-12),
                    f"{case}: pair {species!r} execution metadata disagrees with preflight {result_key}")
        if case == "sio2_bks":
            expected = bks_parameters[species]
            require(all(math.isclose(float(row[key]), value, rel_tol=0.0, abs_tol=1.0e-12)
                        for key, value in zip(("A", "rho", "C"), expected)),
                    f"{case}: pair {species!r} does not carry the standard BKS parameters")
            require(math.isclose(float(row["buck_cutoff"]), 10.0, rel_tol=0.0, abs_tol=1.0e-12) and
                    math.isclose(float(row["pair_cutoff"]), 10.0, rel_tol=0.0, abs_tol=1.0e-12),
                    f"{case}: pair {species!r} truncated the original 10 Å BKS cutoff")
    require(observed_pair_types == expected_pair_types and observed_species == expected_species,
            f"{case}: core table metadata does not cover each Si/O pair exactly once")
    return {
        "status": report.get("status"),
        "table": str(tables[0].relative_to(root)),
        "max_energy_ratio": overall.get("max_energy_ratio"),
        "max_force_ratio": overall.get("max_force_ratio"),
    }


def _check_stage_metrics_artifacts(root: Path, payload: Any, *, case: str, box_index: int, engine: str) -> None:
    stages = as_map(payload, f"{case}.box[{box_index}].stage_metrics")
    require(set(stages) == {"melt", "quench", "relax"},
            f"{case}: box {box_index} stage-metric coverage is incomplete")
    for role in ("melt", "quench", "relax"):
        item = as_map(stages.get(role), f"{case}.box[{box_index}].stage_metrics.{role}")
        require(item.get("status") == "ok" and item.get("engine") == engine,
                f"{case}: box {box_index} {role} stage metrics did not complete")
        require(item.get("reporting_contract") == "vitriflow.canonical_physical_units.v1",
                f"{case}: box {box_index} {role} stage metrics lost the units contract")
        require(isinstance(item.get("n_rows"), int) and int(item["n_rows"]) >= 1,
                f"{case}: box {box_index} {role} stage metrics have no rows")
        csv_path = _result_artifact(root, item.get("csv"), f"{case} box {box_index} {role} metrics CSV")
        csv_summary = check_metrics_csv(csv_path)
        check_count(csv_summary["rows"], int(item["n_rows"]), f"{case} box {box_index} {role} metrics row count")
        summary_path = _result_artifact(root, item.get("summary"), f"{case} box {box_index} {role} metrics summary")
        summary = as_map(load_json(summary_path), str(summary_path))
        require(summary.get("status") == "ok" and summary.get("stage_role") == role,
                f"{case}: box {box_index} {role} metrics manifest is inconsistent")
        check_count(summary.get("n_rows"), int(item["n_rows"]), f"{case} box {box_index} {role} manifest rows")
        plot_path = _result_artifact(root, item.get("plot"), f"{case} box {box_index} {role} automatic metrics plot")
        require(_pdf_ok(plot_path), f"{case}: box {box_index} {role} automatic metrics PDF is invalid")


def _check_elastic_artifacts(root: Path, box: Mapping[str, Any], *, case: str, box_index: int) -> None:
    direct = as_map(box.get("elastic_relax"), f"{case}.box[{box_index}].elastic_relax")
    require(direct.get("status") == "ok", f"{case}: box {box_index} direct elastic screen failed")
    elastic_summary_path = _result_artifact(root, direct.get("summary"), f"{case} box {box_index} elastic summary")
    elastic_summary = as_map(load_json(elastic_summary_path), str(elastic_summary_path))
    require(elastic_summary.get("status") == "ok", f"{case}: box {box_index} elastic summary status is not ok")
    require(elastic_summary.get("kind") == "static_affine_born_snapshot_diagnostic",
            f"{case}: box {box_index} elastic diagnostic kind is wrong")
    method = as_map(elastic_summary.get("method"), f"{case}.box[{box_index}].elastic.method")
    require(method.get("thermodynamic_elastic_tensor") is False and
            method.get("relaxed_elastic_modulus") is False and
            method.get("per_atom_virial_is_unique_local_cauchy_stress") is False,
            f"{case}: box {box_index} elastic diagnostic overclaims its physical interpretation")
    check_count(elastic_summary.get("n_atoms"), int(CASE_SPEC[case]["atoms"]),
                f"{case}.box[{box_index}].elastic.n_atoms")
    for key in (
        "volume_native", "voigt_born_bulk_response_native", "voigt_born_shear_response_native",
        "isotropy_residual", "normal_shear_coupling_norm", "diag_spread_rel",
        "offdiag_spread_rel", "shear_spread_rel",
    ):
        require(finite(elastic_summary.get(key)),
                f"{case}: box {box_index} elastic summary lacks finite {key}")
    born_matrix = as_list(elastic_summary.get("born_matrix_native"),
                          f"{case}.box[{box_index}].elastic.born_matrix_native")
    check_count(len(born_matrix), 6, f"{case}.box[{box_index}] Born matrix rows")
    for row_index, row in enumerate(born_matrix):
        values = as_list(row, f"{case}.box[{box_index}].elastic.born_matrix_native[{row_index}]")
        check_count(len(values), 6, f"{case}.box[{box_index}] Born matrix columns")
        _finite_numeric_tree(values, f"{case}.box[{box_index}].elastic.born_matrix_native[{row_index}]")
    stress = as_list(elastic_summary.get("global_stress_voigt_native"),
                     f"{case}.box[{box_index}].elastic.global_stress_voigt_native")
    check_count(len(stress), 6, f"{case}.box[{box_index}] global stress components")
    _finite_numeric_tree(stress, f"{case}.box[{box_index}].elastic.global_stress_voigt_native")
    eigenvalues = as_list(elastic_summary.get("born_eigenvalues_native"),
                          f"{case}.box[{box_index}].elastic.born_eigenvalues_native")
    check_count(len(eigenvalues), 6, f"{case}.box[{box_index}] Born eigenvalues")
    _finite_numeric_tree(eigenvalues, f"{case}.box[{box_index}].elastic.born_eigenvalues_native")
    proxy = as_map(elastic_summary.get("average_volume_normalized_virial_proxy_summary"),
                   f"{case}.box[{box_index}].elastic.virial_proxy")
    require(proxy.get("atomic_volume_partition") is False and proxy.get("unique_local_cauchy_stress") is False,
            f"{case}: box {box_index} virial proxy provenance is physically overstated")
    vm_proxy = as_map(proxy.get("von_mises_proxy_native"), f"{case}.box[{box_index}].elastic.von_mises_proxy")
    for key in ("p05", "p50", "p95", "max"):
        require(finite(vm_proxy.get(key)),
                f"{case}: box {box_index} virial-proxy diagnostic {key} is non-finite")
    elastic_plot = _result_artifact(root, direct.get("plot"), f"{case} box {box_index} automatic elastic plot")
    require(_png_ok(elastic_plot), f"{case}: box {box_index} automatic elastic plot is invalid")

    series = as_map(box.get("elastic_timeseries"), f"{case}.box[{box_index}].elastic_timeseries")
    require(set(series) == {"melt", "quench", "relax"}, f"{case}: box {box_index} elastic timeseries coverage is incomplete")
    for role in ("melt", "quench", "relax"):
        item = as_map(series.get(role), f"{case}.box[{box_index}].elastic_timeseries.{role}")
        require(item.get("status") == "ok", f"{case}: box {box_index} {role} elastic timeseries degraded/failed")
        require(isinstance(item.get("n_frames"), int) and int(item["n_frames"]) >= 1,
                f"{case}: box {box_index} {role} elastic timeseries has no frames")
        csv_path = _result_artifact(root, item.get("csv"), f"{case} box {box_index} {role} elastic CSV")
        fields, rows = _csv_has_data(csv_path, f"{case} box {box_index} {role} elastic")
        diagnostic_fields = {
            "Step", "time", "status_ok", "isotropy_residual", "normal_shear_coupling_norm",
            "voigt_born_bulk_response_native", "voigt_born_shear_response_native", "virial_proxy_vm_p95",
        }
        require(diagnostic_fields.issubset(fields),
                f"{case}: box {box_index} {role} elastic CSV lacks required fields")
        check_count(len(rows), int(item["n_frames"]), f"{case} box {box_index} {role} elastic frame rows")
        require(all(float(row["status_ok"]) == 1.0 for row in rows),
                f"{case}: box {box_index} {role} elastic frame failure is recorded")
        for row_index, row in enumerate(rows, start=2):
            for field in diagnostic_fields:
                try:
                    value = float(row[field])
                except Exception as exc:
                    fail(f"{case}: box {box_index} {role} elastic CSV row {row_index} invalid {field}: {exc}")
                require(math.isfinite(value),
                        f"{case}: box {box_index} {role} elastic CSV row {row_index} non-finite {field}")
        summary_path = _result_artifact(root, item.get("summary"), f"{case} box {box_index} {role} elastic summary")
        summary = as_map(load_json(summary_path), str(summary_path))
        require(summary.get("status") == "ok" and summary.get("stage_role") == role,
                f"{case}: box {box_index} {role} elastic summary is inconsistent")
        plot_path = _result_artifact(root, item.get("plot"), f"{case} box {box_index} {role} elastic plot")
        require(_png_ok(plot_path), f"{case}: box {box_index} {role} automatic elastic plot is invalid")


def _check_coordination_and_amorphous_artifacts(
    root: Path,
    box: Mapping[str, Any],
    *,
    case: str,
    box_index: int,
) -> None:
    expected_coord = int(as_map(CASE_SPEC[case].get("metric_counts"), "metric counts")["coord"])
    details = as_map(
        box.get("coordination_defect_details"),
        f"{case}.box[{box_index}].coordination_defect_details",
    )
    # Vitriflow (>=0.4.36.0) reports the per-box coordination sweep through the
    # verified `coordination_stability.csv` sidecar and `paths.coord_defects`
    # (validated below), and leaves the inline `coordination_defect_details`
    # empty when no per-descriptor defect entries are flagged.  Requiring a fixed
    # inline count therefore penalises a legitimate, fully-analysed output.
    # Validate any inline entries that ARE present, but treat the sidecar/paths
    # plumbing as the coordination-sweep contract rather than a mandatory inline
    # count.
    require(
        len(details) <= expected_coord,
        f"{case}.box[{box_index}] coordination-detail count {len(details)} "
        f"exceeds the {expected_coord} configured coordination descriptor(s)",
    )
    for name, raw in details.items():
        detail = as_map(raw, f"{case}.box[{box_index}].coordination_defect_details.{name}")
        sweep = as_map(
            detail.get("coordination_sweep"),
            f"{case}.box[{box_index}].coordination_defect_details.{name}.coordination_sweep",
        )
        require(
            finite(sweep.get("dr"))
            and float(sweep["dr"]) > 0.0
            and sweep.get("n_below") == 2
            and sweep.get("n_above") == 2,
            f"{case}: box {box_index} coordination sweep {name!r} configuration drifted",
        )
        delta = as_list(sweep.get("delta_r"), f"{case}.box[{box_index}].{name}.delta_r")
        fractions = as_list(
            sweep.get("defect_fraction"),
            f"{case}.box[{box_index}].{name}.defect_fraction",
        )
        ndef = as_list(sweep.get("n_defective"), f"{case}.box[{box_index}].{name}.n_defective")
        check_count(len(delta), 5, f"{case}.box[{box_index}].{name} sweep points")
        check_count(len(fractions), 5, f"{case}.box[{box_index}].{name} sweep fractions")
        check_count(len(ndef), 5, f"{case}.box[{box_index}].{name} sweep defect counts")
        _finite_numeric_tree(delta, f"{case}.box[{box_index}].{name}.delta_r")
        _finite_numeric_tree(fractions, f"{case}.box[{box_index}].{name}.defect_fraction")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in ndef),
                f"{case}: box {box_index} coordination sweep {name!r} has invalid defect counts")
        require(
            all(float(delta[i]) < float(delta[i + 1]) for i in range(4))
            and math.isclose(float(delta[2]), 0.0, rel_tol=0.0, abs_tol=1.0e-15),
            f"{case}: box {box_index} coordination sweep {name!r} grid is invalid",
        )

    paths = as_map(box.get("paths"), f"{case}.box[{box_index}].paths")
    coord_paths = as_map(paths.get("coord_defects"), f"{case}.box[{box_index}].paths.coord_defects")
    require(
        coord_paths.get("status") == "ok"
        and coord_paths.get("coordination_sweep_enabled") is True,
        f"{case}: box {box_index} coordination-sweep artifact status is not ok/enabled",
    )
    _result_artifact(root, coord_paths.get("detail_json"), f"{case} box {box_index} coordination detail JSON")
    _result_artifact(root, coord_paths.get("marked_extxyz"), f"{case} box {box_index} marked coordination EXTXYZ")

    amorphous = as_map(box.get("amorphous"), f"{case}.box[{box_index}].amorphous")
    require(amorphous.get("enabled") is True, f"{case}: box {box_index} amorphous analysis was not enabled")
    scalar_metrics = as_map(
        amorphous.get("scalar_metrics"),
        f"{case}.box[{box_index}].amorphous.scalar_metrics",
    )
    require(scalar_metrics, f"{case}: box {box_index} amorphous scalar metrics are empty")
    metrics = as_map(box.get("metrics"), f"{case}.box[{box_index}].metrics")
    for name, value in scalar_metrics.items():
        require(str(name).startswith("amorphous_"),
                f"{case}: box {box_index} malformed amorphous metric name {name!r}")
        require(name in metrics and metrics.get(name) == value,
                f"{case}: box {box_index} amorphous scalar {name!r} was not merged into production metrics")
        if not finite(value):
            # Reference-derived values are legitimately unavailable when the
            # validation config explicitly disables external references, but
            # their criterion must say so rather than silently disappearing.
            require("reference" in str(name),
                    f"{case}: box {box_index} non-reference amorphous metric {name!r} is non-finite")
            reference = as_map(
                as_map(amorphous.get("criteria"), f"{case}.box[{box_index}].amorphous.criteria").get("reference_peak_overlap"),
                f"{case}.box[{box_index}].amorphous.criteria.reference_peak_overlap",
            )
            require(reference.get("skipped") is True and bool(str(reference.get("reason", "")).strip()),
                    f"{case}: box {box_index} unavailable reference metric lacks an explicit not-applicable reason")
    amorphous_paths = as_map(paths.get("amorphous"), f"{case}.box[{box_index}].paths.amorphous")
    _result_artifact(root, amorphous_paths.get("state_json"), f"{case} box {box_index} amorphous state JSON")


def check_run(
    result_path: Path,
    case: str,
    *,
    expect_cell_refinement: bool = False,
) -> dict[str, Any]:
    spec = CASE_SPEC.get(case)
    require(spec is not None, f"unknown case {case!r}")
    root = result_path.parent
    data = as_map(load_json(result_path), str(result_path))

    require(data.get("execution_status") == "completed", f"{case}: execution_status is not completed")
    require(data.get("status") == "ok", f"{case}: workflow status is {data.get('status')!r}, expected 'ok'")
    units = as_map(data.get("units"), f"{case}.units")
    require(units.get("engine") == spec["engine"], f"{case}: wrong engine in result units")

    preflight = as_map(data.get("preflight"), f"{case}.preflight")
    require(finite(preflight.get("selected_timestep")), f"{case}: selected timestep missing/non-finite")
    require(finite(preflight.get("selected_tdamp")), f"{case}: selected TDAMP missing/non-finite")
    require(finite(preflight.get("selected_pdamp")), f"{case}: selected PDAMP missing/non-finite")
    expected_tuple = tuple(float(value) for value in spec["thermostat_tuple"])
    actual_tuple = (
        float(preflight["selected_timestep"]),
        float(preflight["selected_tdamp"]),
        float(preflight["selected_pdamp"]),
    )
    require(all(math.isclose(got, expected, rel_tol=0.0, abs_tol=1.0e-12)
                for got, expected in zip(actual_tuple, expected_tuple)),
            f"{case}: runtime selected timestep/TDAMP/PDAMP drifted: {actual_tuple!r}")
    candidates = as_list(preflight.get("candidates"), f"{case}.preflight.candidates")
    check_count(len(candidates), 1, f"{case} runtime preflight candidate count")
    def candidate_succeeded(raw: Any) -> bool:
        candidate = as_map(raw, "preflight candidate")
        if bool(candidate.get("ok")):
            return True
        details = candidate.get("details")
        return bool(spec["engine"] == "cp2k" and isinstance(details, Mapping) and details.get("hard_ok") is True)

    require(any(candidate_succeeded(item) for item in candidates), f"{case}: no successful preflight candidate")
    for item in candidates:
        candidate = as_map(item, f"{case}.preflight candidate")
        require(candidate.get("ensemble") == "npt", f"{case}: undeclared preflight ensemble was executed")
        require(all(finite(candidate.get(key)) for key in ("timestep", "tdamp", "pdamp")),
                f"{case}: preflight candidate has incomplete/non-finite settings")
        candidate_tuple = (float(candidate["timestep"]), float(candidate["tdamp"]), float(candidate["pdamp"]))
        require(all(math.isclose(got, expected, rel_tol=0.0, abs_tol=1.0e-12)
                    for got, expected in zip(candidate_tuple, expected_tuple)),
                f"{case}: hidden preflight fallback/settings drift was executed: {candidate_tuple!r}")

    core = as_map(preflight.get("core_repulsion"), f"{case}.preflight.core_repulsion")
    if spec["core"]:
        require(core.get("enabled") is True and core.get("applied") is True and core.get("success") is True,
                f"{case}: the requested C2 core was not successfully applied")
        joins = as_list(core.get("resolved_pair_joins"), f"{case}.resolved_pair_joins")
        check_count(len(joins), 3, f"{case} pair-resolved core join count")
        require(core.get("r_inner_r_outer_role") == "global_calibration_request",
                f"{case}: legacy global radii are not labelled as calibration-only")
        expected_pairs = {("Si", "Si"), ("Si", "O"), ("O", "O")}
        observed_pairs: set[tuple[str, str]] = set()
        for join in joins:
            row = as_map(join, f"{case}.resolved core join")
            species = tuple(str(x) for x in as_list(row.get("species"), f"{case}.core join species"))
            require(len(species) == 2, f"{case}: malformed pair-resolved species label")
            observed_pairs.add(species)
            for key in ("requested_r_in", "requested_r_out", "resolved_r_in", "resolved_r_out"):
                require(finite(row.get(key)) and float(row[key]) > 0.0, f"{case}: non-finite/invalid {key}")
            require(float(row["requested_r_in"]) < float(row["requested_r_out"]), f"{case}: invalid requested core interval")
            require(float(row["resolved_r_in"]) < float(row["resolved_r_out"]), f"{case}: invalid resolved core interval")
        require(observed_pairs == expected_pairs, f"{case}: core join report does not cover all Si/O pairs")
        _check_tabulated_core_evidence(root, case, joins=joins)
    else:
        require(core.get("enabled") is False, f"{case}: unexpected core-repulsion activation")

    tm = as_map(data.get("tm_scan"), f"{case}.tm_scan")
    temps = as_list(tm.get("temps"), f"{case}.tm_scan.temps")
    t_min, t_max, d_t = (float(value) for value in spec["temperature_grid"])
    n_points = int(round((t_max - t_min) / d_t)) + 1
    expected_temps = [t_min + index * d_t for index in range(n_points)]
    temps = _require_float_sequence(temps, expected_temps, f"{case}.tm_scan.temps")
    require(all(temps[index] < temps[index + 1] for index in range(n_points - 1)),
            f"{case}: temperature grid is not strictly ascending")
    check_count(tm.get("replicates_per_temp"), 3, f"{case}.tm_scan.replicates_per_temp")
    outcomes = as_list(tm.get("outcomes"), f"{case}.tm_scan.outcomes")
    check_count(len(outcomes), 3 * n_points, f"{case} Tm outcome count")
    tm_identities: set[tuple[float, int]] = set()
    tm_seeds: set[int] = set()
    for temperature in temps:
        rows = [as_map(o, "Tm outcome") for o in outcomes
                if finite(as_map(o, "Tm outcome").get("temperature_start")) and
                math.isclose(float(as_map(o, "Tm outcome")["temperature_start"]), temperature,
                             rel_tol=0.0, abs_tol=1.0e-12)]
        check_count(len(rows), 3, f"{case} replicas at {temperature:g} K")
        replica_ids = {_positive_int(row.get("rep_id"), f"{case} Tm rep_id at {temperature:g} K") for row in rows}
        require(replica_ids == {1, 2, 3},
                f"{case}: Tm replica IDs at {temperature:g} K are incomplete/duplicated")
        for row in rows:
            rep_id = int(row["rep_id"])
            identity = (float(temperature), rep_id)
            require(identity not in tm_identities, f"{case}: duplicate Tm temperature/replica identity {identity!r}")
            tm_identities.add(identity)
            seed = _positive_int(row.get("seed"), f"{case} Tm seed at {temperature:g} K rep {rep_id}")
            require(seed not in tm_seeds, f"{case}: duplicate Tm random seed {seed}")
            tm_seeds.add(seed)
            check_count(row.get("n_atoms"), int(spec["atoms"]),
                        f"{case} Tm n_atoms at {temperature:g} K rep {rep_id}")
        require(all(finite(row.get("D")) and finite(row.get("density_mean")) for row in rows),
                f"{case}: non-finite Tm diffusion/density at {temperature:g} K")
        require(all(finite(row.get(key)) for row in rows for key in ("gr_peak_r", "gr_peak_height", "gr_peak_fwhm")),
                f"{case}: Tm g(r) indicator plumbing is absent/non-finite at {temperature:g} K")

    high = as_map(data.get("highT"), f"{case}.highT")
    check_count(high.get("replicates"), 3, f"{case}.highT.replicates")
    high_outcomes = as_list(high.get("outcomes"), f"{case}.highT.outcomes")
    check_count(len(high_outcomes), 3, f"{case} high-T outcome count")
    high_rows = [as_map(row, f"{case} high-T outcome") for row in high_outcomes]
    require({_positive_int(row.get("rep_id"), f"{case} high-T rep_id") for row in high_rows} == {1, 2, 3},
            f"{case}: high-T replica IDs are incomplete/duplicated")
    high_seeds: set[int] = set()
    for row in high_rows:
        seed = _positive_int(row.get("seed"), f"{case} high-T seed")
        require(seed not in high_seeds and seed not in tm_seeds, f"{case}: reused Tm/high-T random seed {seed}")
        high_seeds.add(seed)
        check_count(row.get("n_atoms"), int(spec["atoms"]), f"{case} high-T n_atoms")
        require(finite(row.get("density_mean")), f"{case}: a high-T outcome has no finite density")

    rate_scan = as_map(data.get("rate_scan"), f"{case}.rate_scan")
    rates = as_list(rate_scan.get("rates"), f"{case}.rate_scan.rates")
    check_count(len(rates), 3, f"{case} cooling-rate count")
    rate_values = _require_float_sequence(
        [as_map(row, f"{case}.rate").get("rate_K_per_ps") for row in rates],
        [float(value) for value in spec["rates"]],
        f"{case}.rate_scan.rate_K_per_ps",
    )
    require(all(rate_values[index] > rate_values[index + 1] > 0.0 for index in range(2)),
            f"{case}: cooling rates are not unique, positive, and strictly descending")
    rate_artifact_paths: set[Path] = set()
    for index, raw in enumerate(rates, start=1):
        row = as_map(raw, f"{case}.rate[{index}]")
        require(finite(row.get("rate_K_per_ps")), f"{case}: rate entry {index} has no finite K/ps rate")
        check_count(row.get("nrep"), 3, f"{case}.rate[{index}].nrep")
        reps = as_list(row.get("replicates"), f"{case}.rate[{index}].replicates")
        check_count(len(reps), 3, f"{case}.rate[{index}] replica rows")
        require(finite(row.get("density_mean")), f"{case}: rate entry {index} has no finite density")
        _assert_metric_families(row.get("metrics_mean"), f"{case}.rate[{index}].metrics_mean")
        mean_metrics = as_map(row.get("metrics_mean"), "rate metrics mean")
        stderr_metrics = as_map(row.get("metrics_stderr"), f"{case}.rate[{index}].metrics_stderr")
        require(set(stderr_metrics) == set(mean_metrics),
                f"{case}: rate entry {index} metric uncertainty coverage differs from its means")
        # A structural-metric uncertainty may be non-finite ONLY when the metric
        # itself is unavailable (non-finite mean) AND the descriptor is one that
        # can be legitimately undefined (see _metric_may_be_undefined). An
        # available metric (finite mean) must always carry a finite uncertainty,
        # and any other metric dropping out is still flagged -- so this tolerates
        # the physically-undefined S(q) peak width without masking real
        # regressions or a finite metric with a corrupt uncertainty.
        for _name, _stderr in stderr_metrics.items():
            if finite(mean_metrics.get(_name)):
                require(finite(_stderr),
                        f"{case}: rate entry {index} metric {_name!r} is available but its "
                        f"structural-metric uncertainty is non-finite")
            else:
                require(_metric_may_be_undefined(_name),
                        f"{case}: rate entry {index} metric {_name!r} is unexpectedly non-finite "
                        f"(not a known conditionally-defined descriptor)")
        amorphous_summary = as_map(row.get("amorphous_summary"), f"{case}.rate[{index}].amorphous_summary")
        require("accepted" in amorphous_summary and isinstance(amorphous_summary.get("criteria_summary"), Mapping),
                f"{case}: rate entry {index} has no amorphous acceptance evidence")
        for rep_index, rep_raw in enumerate(reps, start=1):
            rep = as_map(rep_raw, f"{case}.rate[{index}].replicate[{rep_index}]")
            require(finite(rep.get("density")), f"{case}: rate {index} replicate {rep_index} density is non-finite")
            require(math.isclose(float(rep.get("cooling_rate_K_per_ps", float("nan"))), rate_values[index - 1],
                                 rel_tol=0.0, abs_tol=1.0e-12),
                    f"{case}: rate {index} replicate {rep_index} carries a mismatched cooling rate")
            final_data = _result_artifact(root, rep.get("final_data"),
                                          f"{case} rate {index} replicate {rep_index} final data")
            check_count(_lammps_data_atom_count(final_data, f"{case} rate replicate final data"),
                        int(spec["atoms"]), f"{case} rate {index} replicate {rep_index} atom count")
            dump = _result_artifact(root, rep.get("dump"), f"{case} rate {index} replicate {rep_index} dump")
            for artifact in (final_data, dump):
                require(artifact not in rate_artifact_paths,
                        f"{case}: rate replicas reuse an output artifact: {artifact}")
                rate_artifact_paths.add(artifact)
            _assert_metric_families(rep.get("metrics"),
                                    f"{case}.rate[{index}].replicate[{rep_index}].metrics")
            require(bool(as_map(rep.get("amorphous"), f"{case}.rate[{index}].replicate[{rep_index}].amorphous")),
                    f"{case}: rate replicate amorphous diagnostics are empty")

    sizes = as_map(data.get("size_scan"), f"{case}.size_scan")
    require(sizes.get("skipped") is True, f"{case}: finite-size stage must remain disabled")
    require(as_list(sizes.get("sizes"), f"{case}.size_scan.sizes") == [], f"{case}: disabled finite-size stage emitted scan results")
    if spec["engine"] == "cp2k":
        require(sizes.get("skip_reason") == "cp2k engine: size scan disabled", f"{case}: unexpected CP2K size skip reason")
    else:
        require(sizes.get("skip_reason") == "autotune.size.enabled=false", f"{case}: size stage was not skipped by explicit configuration")

    fingerprint = as_map(data.get("resume_fingerprint"), f"{case}.resume_fingerprint")
    payload = as_map(fingerprint.get("payload"), f"{case}.resume_fingerprint.payload")
    effective = as_map(payload.get("effective_config"), f"{case}.effective_config")
    effective_autotune = as_map(effective.get("autotune"), f"{case}.effective_config.autotune")
    effective_size = as_map(effective_autotune.get("size"), f"{case}.effective_config.autotune.size")
    require(effective_size.get("enabled") is False, f"{case}: effective configuration re-enabled finite-size scanning")

    prod = as_map(data.get("production"), f"{case}.production")
    require(prod.get("enabled") is True, f"{case}: production is not enabled")
    require(prod.get("execution_status") == "completed", f"{case}: production execution did not complete")
    require(prod.get("check_convergence") is True, f"{case}: runtime adaptive convergence was disabled")
    check_count(prod.get("min_boxes"), 6, f"{case}.production runtime min_boxes")
    check_count(prod.get("batch_boxes"), 4, f"{case}.production runtime batch_boxes")
    check_count(prod.get("max_boxes"), 10, f"{case}.production runtime max_boxes")
    check_count(prod.get("n_boxes"), 10, f"{case}.production.n_boxes")
    check_count(prod.get("n_boxes_accepted"), 10, f"{case}.production.n_boxes_accepted")
    check_count(prod.get("n_boxes_total"), 10, f"{case}.production.n_boxes_total")
    check_count(prod.get("n_boxes_rejected"), 0, f"{case}.production.n_boxes_rejected")
    require(prod.get("converged") is True, f"{case}: adaptive production did not converge")
    require(prod.get("convergence_status") == "converged", f"{case}: convergence_status is not 'converged'")
    require(prod.get("convergence_streak") == 2, f"{case}: expected two consecutive convergence checks")
    require(prod.get("required_convergence_streak") == 2, f"{case}: wrong required convergence streak")
    check_count(prod.get("last_convergence_evaluated_n_boxes_total"), 10,
                f"{case}.production.last_convergence_evaluated_n_boxes_total")
    check_count(prod.get("last_convergence_evaluated_n_boxes_accepted"), 10,
                f"{case}.production.last_convergence_evaluated_n_boxes_accepted")
    inference = str(prod.get("convergence_inference_status", ""))
    require("criterion_met" in inference, f"{case}: convergence inference is not criterion-met: {inference!r}")
    convergence = as_map(prod.get("convergence"), f"{case}.production.convergence")
    _assert_convergence_degree(convergence, f"{case}.production.convergence")
    _assert_effective_convergence_families(convergence, spec, f"{case}.production.convergence")
    _assert_convergence_evidence_coverage(
        convergence,
        n_boxes=10,
        label=f"{case}.production.convergence",
    )
    _assert_criterion_coverage(prod.get("convergence_criterion_coverage"),
                               f"{case}.production.convergence_criterion_coverage")
    require(isinstance(prod.get("achieved_convergence_degree"), Mapping),
            f"{case}: production achieved-convergence diagnostic is absent")
    boxes = as_list(prod.get("boxes"), f"{case}.production.boxes")
    check_count(len(boxes), 10, f"{case} production box rows")
    box_ids: set[int] = set()
    production_seeds: set[int] = set()
    for index, raw in enumerate(boxes, start=1):
        box = as_map(raw, f"{case}.box[{index}]")
        box_id = _positive_int(box.get("box"), f"{case}.box[{index}].box")
        require(box_id not in box_ids, f"{case}: duplicate production box ID {box_id}")
        box_ids.add(box_id)
        for role in ("warmup", "melt", "quench", "relax"):
            seed = _positive_int(box.get(f"seed_{role}"), f"{case}.box[{index}].seed_{role}")
            require(seed not in production_seeds and seed not in tm_seeds and seed not in high_seeds,
                    f"{case}: reused random seed {seed} in production box {box_id}")
            production_seeds.add(seed)
        require(finite(box.get("density")), f"{case}: box {index} density is non-finite")
        _assert_metric_families(box.get("metrics"), f"{case}.box[{index}].metrics")
        _assert_distribution_families(box.get("distributions"), spec,
                                      f"{case}.box[{index}].distributions")
        require(isinstance(box.get("structure"), Mapping), f"{case}: box {index} embedded structure is absent")
        structure = as_map(box.get("structure"), f"{case}.box[{index}].structure")
        check_count(structure.get("n_atoms"), int(spec["atoms"]), f"{case}.box[{index}] atom count")
        paths = as_map(box.get("paths"), f"{case}.box[{index}].paths")
        for key in ("melt_dir", "quench_dir", "relax_dir"):
            require(isinstance(paths.get(key), str) and paths[key], f"{case}: box {index} missing paths.{key}")
            resolved = root / paths[key] if not Path(paths[key]).is_absolute() else Path(paths[key])
            require(_strictly_under(resolved, root) and resolved.is_dir(), f"{case}: invalid/missing {key}: {resolved}")
        _check_coordination_and_amorphous_artifacts(
            root,
            box,
            case=case,
            box_index=index,
        )
        _check_stage_metrics_artifacts(root, box.get("stage_metrics"), case=case, box_index=index, engine=spec["engine"])
        if spec["engine"] == "lammps":
            _check_elastic_artifacts(root, box, case=case, box_index=index)
    require(box_ids == set(range(1, 11)), f"{case}: production box IDs are not exactly 1..10")
    producer_metric_coverage = _assert_metric_plumbing_coverage(
        convergence,
        prod.get("metrics_checked"),
        label=f"{case}.production.convergence",
    )
    metric_coverage = _assert_majority_of_emitted_metrics_enter_convergence(
        {**dict(convergence), "metrics_checked": prod.get("metrics_checked")},
        [as_map(box, f"{case}.production.box") for box in boxes],
        label=f"{case}.production",
    )

    if expect_cell_refinement:
        require(case == "si_cp2k", "CELL_OPT result validation is defined only for si_cp2k")
        dft_summary = as_map(prod.get("dft_opt"), f"{case}.production.dft_opt")
        require(
            dft_summary.get("enabled") is True
            and dft_summary.get("optimizer") == "LBFGS"
            and dft_summary.get("keep_angles") is True
            and dft_summary.get("print_level") == "LOW",
            f"{case}: production CELL_OPT execution contract drifted",
        )
        check_count(dft_summary.get("max_iter"), 200, f"{case}.production.dft_opt.max_iter")
        check_count(dft_summary.get("traj_every"), 1, f"{case}.production.dft_opt.traj_every")
        require(
            finite(dft_summary.get("external_pressure_bar"))
            and math.isclose(float(dft_summary["external_pressure_bar"]), 1.0),
            f"{case}: CELL_OPT external pressure is not 1 bar",
        )
        for field, expected in (
            ("n_boxes_ok", 10),
            ("n_boxes_failed", 0),
            ("n_boxes_rejected_coordination_defects", 0),
            ("n_boxes_not_run", 0),
            ("n_boxes_accepted", 10),
        ):
            check_count(dft_summary.get(field), expected, f"{case}.production.dft_opt.{field}")
        require(prod.get("converged_md") is True and prod.get("converged_dft") is True,
                f"{case}: MD and refined DFT ensembles did not both converge")
        md_convergence = as_map(prod.get("convergence_md"), f"{case}.production.convergence_md")
        _assert_convergence_degree(md_convergence, f"{case}.production.convergence_md")
        _assert_effective_convergence_families(
            md_convergence,
            spec,
            f"{case}.production.convergence_md",
        )
        _assert_convergence_evidence_coverage(
            md_convergence,
            n_boxes=10,
            label=f"{case}.production.convergence_md",
        )
        _assert_metric_plumbing_coverage(
            md_convergence,
            prod.get("metrics_checked"),
            label=f"{case}.production.convergence_md",
        )
        dft_convergence = as_map(prod.get("convergence_dft"), f"{case}.production.convergence_dft")
        _assert_convergence_degree(dft_convergence, f"{case}.production.convergence_dft")
        _assert_effective_convergence_families(
            dft_convergence,
            spec,
            f"{case}.production.convergence_dft",
        )
        _assert_convergence_evidence_coverage(
            dft_convergence,
            n_boxes=10,
            label=f"{case}.production.convergence_dft",
        )
        _assert_metric_plumbing_coverage(
            dft_convergence,
            prod.get("metrics_checked"),
            label=f"{case}.production.convergence_dft",
        )
        refined_ids = as_list(prod.get("boxes_dft_final"), f"{case}.production.boxes_dft_final")
        require(refined_ids == list(range(1, 11)), f"{case}: refined accepted box IDs are not exactly 1..10")
        check_count(prod.get("n_boxes_dft_accepted"), 10, f"{case}.production.n_boxes_dft_accepted")
        require(as_list(prod.get("rejected_boxes_dft"), f"{case}.production.rejected_boxes_dft") == [],
                f"{case}: at least one CELL_OPT box was rejected")
        for index, raw in enumerate(boxes, start=1):
            dft = as_map(as_map(raw, f"{case}.box[{index}]").get("dft_opt"), f"{case}.box[{index}].dft_opt")
            require(dft.get("status") == "ok", f"{case}: box {index} CELL_OPT status is not ok")
            require(finite(dft.get("density")) and float(dft["density"]) > 0.0,
                    f"{case}: box {index} refined density is invalid")
            require(finite(dft.get("density_stderr")) and float(dft["density_stderr"]) == 0.0,
                    f"{case}: box {index} refined single-structure density uncertainty is invalid")
            _assert_metric_families(dft.get("metrics"), f"{case}.box[{index}].dft_opt.metrics")
            _assert_distribution_families(
                dft.get("distributions"),
                spec,
                f"{case}.box[{index}].dft_opt.distributions",
            )
            dft_paths = as_map(dft.get("paths"), f"{case}.box[{index}].dft_opt.paths")
            for key in ("dft_input", "dft_output", "dft_scf_diagnostics", "dft_traj", "dft_data"):
                artifact = _result_artifact(root, dft_paths.get(key), f"{case} box {index} {key}")
                if key == "dft_data":
                    check_count(
                        _lammps_data_atom_count(artifact, f"{case} box {index} refined data"),
                        int(spec["atoms"]),
                        f"{case}.box[{index}] refined atom count",
                    )
            output_path = _result_artifact(root, dft_paths.get("dft_output"), f"{case} box {index} dft_output")
            output_text = output_path.read_text(encoding="utf-8", errors="replace")
            require(
                "CELL OPTIMIZATION COMPLETED" in output_text.upper()
                or "GEOMETRY OPTIMIZATION COMPLETED" in output_text.upper(),
                f"{case}: box {index} CELL_OPT output lacks a positive completion marker",
            )
            scf = as_map(
                load_json(_result_artifact(root, dft_paths.get("dft_scf_diagnostics"), f"{case} box {index} SCF diagnostics")),
                f"{case}.box[{index}].dft_scf_diagnostics",
            )
            require(
                isinstance(scf.get("unconverged_scf_cycles"), int)
                and int(scf["unconverged_scf_cycles"]) >= 0
                and bool(str(scf.get("policy", "")).strip()),
                f"{case}: box {index} CELL_OPT SCF continuation provenance is incomplete",
            )
        _assert_majority_of_emitted_metrics_enter_convergence(
            {**dict(dft_convergence), "metrics_checked": prod.get("metrics_checked")},
            [
                as_map(as_map(raw, f"{case}.box").get("dft_opt"), f"{case}.box.dft_opt")
                for raw in boxes
            ],
            label=f"{case}.production.dft_opt",
        )
    else:
        require(prod.get("dft_opt") is None, f"{case}: unexpected DFT refinement summary is present")
        require(prod.get("converged_dft") is None and prod.get("convergence_dft") is None,
                f"{case}: unexpected DFT convergence evidence is present")

    recommendation = as_map(data.get("recommendation"), f"{case}.recommendation")
    ensemble = as_map(recommendation.get("final_ensemble_convergence"), f"{case}.final_ensemble_convergence")
    require(ensemble.get("converged") is True, f"{case}: recommendation does not carry converged ensemble status")
    check_count(ensemble.get("n_boxes"), 10, f"{case}.recommendation final n_boxes")
    require(ensemble.get("sequentially_valid") is False and "criterion_met" in str(ensemble.get("status", "")) and
            "criterion_met" in str(ensemble.get("convergence_inference_status", "")),
            f"{case}: recommendation convergence is not correctly inference-qualified")
    _assert_fixed_look_contract(ensemble.get("inference_contract"),
                                f"{case}.recommendation.final_ensemble_convergence.inference_contract")
    _assert_criterion_coverage(ensemble.get("convergence_criterion_coverage"),
                               f"{case}.recommendation.final_ensemble_convergence.convergence_criterion_coverage")
    achieved = as_map(ensemble.get("achieved_convergence_degree"),
                      f"{case}.recommendation.final_ensemble_convergence.achieved_convergence_degree")
    check_count(achieved.get("n_boxes"), 10, f"{case}.recommendation achieved n_boxes")
    check_count(achieved.get("convergence_streak"), 2, f"{case}.recommendation convergence streak")
    check_count(achieved.get("required_convergence_streak"), 2,
                f"{case}.recommendation required convergence streak")

    summary = {
        "case": case,
        "engine": spec["engine"],
        "atoms": spec["atoms"],
        "temperatures": temps,
        "tm_replicas": len(outcomes),
        "highT_replicas": len(high_outcomes),
        "cooling_rates": len(rates),
        "rate_replicas": sum(len(row.get("replicates", [])) for row in rates),
        "production_boxes": len(boxes),
        "convergence_inference_status": inference,
        "producer_metric_plumbing_coverage": producer_metric_coverage,
        "metric_convergence_coverage": metric_coverage,
        "cell_refinement_enabled": bool(expect_cell_refinement),
    }
    return summary


def check_analysis(path: Path, case: str, expected_boxes: int = 10) -> dict[str, Any]:
    spec = CASE_SPEC.get(case)
    require(spec is not None, f"unknown case {case!r}")
    data = as_map(load_json(path), str(path))
    root = path.parent
    require(data.get("schema") == "vitriflow.analysis_results.v2", f"{path}: wrong analysis schema")
    require(data.get("status") == "ok", f"{path}: analysis status is {data.get('status')!r}")
    require(data.get("errors") == [], f"{path}: analysis errors are present")
    require(data.get("converged") is True, f"{path}: top-level descriptor convergence did not pass")
    require(data.get("check_convergence") is True and data.get("convergence_advisory") is True,
            f"{path}: analysis convergence was not run as an advisory descriptor diagnostic")
    check_count(data.get("n_boxes"), expected_boxes, f"{path}.n_boxes")
    check_count(data.get("n_boxes_accepted"), expected_boxes, f"{path}.n_boxes_accepted")
    check_count(data.get("n_boxes_rejected"), 0, f"{path}.n_boxes_rejected")
    check_count(data.get("n_boxes_total"), expected_boxes, f"{path}.n_boxes_total")
    boxes = as_list(data.get("boxes"), f"{path}.boxes")
    check_count(len(boxes), expected_boxes, f"{path} accepted box rows")
    box_ids: set[int] = set()
    for index, raw in enumerate(boxes, start=1):
        box = as_map(raw, f"{path}.boxes[{index}]")
        box_id = _positive_int(box.get("box"), f"{path}.boxes[{index}].box")
        require(box_id not in box_ids, f"{path}: duplicate analysed box ID {box_id}")
        box_ids.add(box_id)
        require(finite(box.get("density")), f"{path}: analysed box {box_id} density is non-finite")
        _assert_metric_families(box.get("metrics"), f"{path}.boxes[{index}].metrics")
        _assert_distribution_families(box.get("distributions"), spec,
                                      f"{path}.boxes[{index}].distributions")
        structure = as_map(box.get("structure"), f"{path}.boxes[{index}].structure")
        require(structure.get("schema") == "vitriflow.structure_snapshot.v1"
                and isinstance(structure.get("n_atoms"), int) and int(structure["n_atoms"]) == int(spec["atoms"]),
                f"{path}: analysed box {index} does not contain an embedded structure")
        _check_coordination_and_amorphous_artifacts(
            path.parent,
            box,
            case=case,
            box_index=index,
        )
    require(box_ids == set(range(1, expected_boxes + 1)),
            f"{path}: analysed box IDs are not exactly 1..{expected_boxes}")
    require(not any(str(as_map(row, "analysis rejection").get("reason", "")) == "analysis_failed"
                    for row in as_list(data.get("rejected_boxes"), f"{path}.rejected_boxes")),
            f"{path}: at least one box failed analysis")
    convergence = as_map(data.get("convergence"), f"{path}.convergence")
    require(convergence.get("status") == "ok", f"{path}: convergence status is not ok/evaluated")
    require(convergence.get("advisory") is True, f"{path}: analysis convergence is not labelled advisory")
    degree = _assert_convergence_degree(convergence, f"{path}.convergence")
    _assert_effective_convergence_families(convergence, spec, f"{path}.convergence")
    _assert_convergence_evidence_coverage(
        convergence,
        n_boxes=expected_boxes,
        label=f"{path}.convergence",
    )
    producer_metric_coverage = _assert_metric_plumbing_coverage(
        convergence,
        data.get("metrics_checked"),
        label=f"{path}.convergence",
    )
    metric_coverage = _assert_majority_of_emitted_metrics_enter_convergence(
        {**dict(convergence), "metrics_checked": data.get("metrics_checked")},
        [as_map(box, f"{path}.box") for box in boxes],
        label=str(path),
    )
    parity = as_map(data.get("convergence_parity"), f"{path}.convergence_parity")
    require(
        parity.get("schema") == "vitriflow.convergence_parity.v1"
        and parity.get("comparable") is False
        and parity.get("equivalent") is None
        and "graph-rule overrides" in str(parity.get("reason", "")),
        f"{path}: graph-override analysis is not explicitly distinguished from exact convergence replay",
    )

    integrity_path = path.parent / "sidecar_integrity.json"
    integrity = as_map(load_json(integrity_path), str(integrity_path))
    require(integrity.get("schema") == "vitriflow.sidecar_integrity.v2", f"{integrity_path}: wrong schema")
    for key in ("all_present", "all_content_hashed", "all_valid"):
        require(integrity.get(key) is True, f"{integrity_path}: {key} is not true")
    require(integrity.get("missing") == [], f"{integrity_path}: missing sidecars are reported")
    sidecars = as_map(integrity.get("sidecars"), f"{integrity_path}.sidecars")
    require(sidecars, f"{integrity_path}: no sidecars were audited")
    audited_paths: set[Path] = set()
    for name, raw_record in sorted(sidecars.items()):
        record = as_map(raw_record, f"{integrity_path}.sidecars.{name}")
        require(record.get("exists") is True and record.get("valid") is True and
                record.get("status") == "verified",
                f"{integrity_path}: sidecar {name!r} was not verified")
        artifact = _result_artifact(root, record.get("path"), f"{integrity_path} sidecar {name!r}")
        require(artifact not in audited_paths, f"{integrity_path}: duplicate sidecar path {artifact}")
        audited_paths.add(artifact)
        check_count(record.get("size_bytes"), artifact.stat().st_size,
                    f"{integrity_path}.sidecars.{name}.size_bytes")
        digest = record.get("sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"{integrity_path}: sidecar {name!r} has no valid SHA-256")
        require(hashlib.sha256(artifact.read_bytes()).hexdigest() == digest,
                f"{integrity_path}: sidecar {name!r} bytes do not match its SHA-256")
        if artifact.suffix.lower() == ".json":
            payload = load_json(artifact)
            if record.get("schema") is not None:
                require(isinstance(payload, Mapping) and payload.get("schema") == record.get("schema"),
                        f"{integrity_path}: sidecar {name!r} schema record disagrees with its content")

    graph_outputs = as_map(data.get("graph_outputs"), f"{path}.graph_outputs")
    required_graph = {
        "graph_rules", "representation_rules", "graph_metric_by_rule",
        "coordination_stability", "shell_separability", "graph_uncertainty_summary",
    }
    require(required_graph.issubset(graph_outputs), f"{path}: graph-analysis sidecar set is incomplete")
    graph_rules_path = _result_artifact(root, graph_outputs["graph_rules"], f"{path} graph rules")
    graph_rules = as_map(load_json(graph_rules_path), str(graph_rules_path))
    require(graph_rules.get("schema") == "vitriflow.graph_rules.v1", f"{graph_rules_path}: wrong schema")
    rules = as_list(graph_rules.get("graph_rules"), f"{graph_rules_path}.graph_rules")
    require(len(rules) >= 5, f"{graph_rules_path}: hard/interval/soft CLI graph rules were not all exercised")
    rule_rows = [as_map(rule, f"{graph_rules_path}.graph_rules") for rule in rules]
    rule_names = [str(rule.get("name", "")) for rule in rule_rows]
    require(len(set(rule_names)) == len(rule_names) and all(rule_names),
            f"{graph_rules_path}: graph rule names are empty or duplicated")
    primary = [rule for rule in rule_rows if rule.get("name") == "cli_hard_cutoff_1"]
    check_count(len(primary), 1, f"{graph_rules_path} primary CLI hard-cutoff rules")
    primary_params = as_map(primary[0].get("parameters"), f"{graph_rules_path}.cli_hard_cutoff_1.parameters")
    require(primary[0].get("kind") == "hard_cutoff" and
            primary[0].get("provenance") == "cli:--graph-cutoff" and
            finite(primary_params.get("cutoff")) and float(primary_params["cutoff"]) > 0.0,
            f"{graph_rules_path}: primary CLI hard-cutoff rule is malformed")
    interval_rules = [
        rule for rule in rule_rows
        if rule.get("kind") == "hard_cutoff" and
        as_map(rule.get("parameters"), f"{graph_rules_path}.interval.parameters").get("parent_rule_kind")
        == "hard_cutoff_interval"
    ]
    check_count(len(interval_rules), 3, f"{graph_rules_path} interval-expanded hard-cutoff rules")
    interval_cutoffs: list[float] = []
    for rule in interval_rules:
        params = as_map(rule.get("parameters"), f"{graph_rules_path}.interval.parameters")
        require(rule.get("provenance") == "cli:--graph-cutoff-interval" and
                params.get("interval_points") == 3,
                f"{graph_rules_path}: interval graph rule lacks CLI/three-point provenance")
        interval = as_list(params.get("interval"), f"{graph_rules_path}.interval")
        require(len(interval) == 2 and all(finite(value) for value in interval) and
                0.0 < float(interval[0]) < float(interval[1]),
                f"{graph_rules_path}: interval graph rule has invalid bounds")
        require(finite(params.get("cutoff")), f"{graph_rules_path}: interval graph cutoff is non-finite")
        interval_cutoffs.append(float(params["cutoff"]))
    require(len(set(interval_cutoffs)) == 3 and interval_cutoffs == sorted(interval_cutoffs),
            f"{graph_rules_path}: interval graph cutoffs are not three unique ordered samples")
    soft = [rule for rule in rule_rows if rule.get("name") == "cli_soft_logistic"]
    check_count(len(soft), 1, f"{graph_rules_path} soft-logistic CLI rules")
    soft_params = as_map(soft[0].get("parameters"), f"{graph_rules_path}.cli_soft_logistic.parameters")
    require(soft[0].get("kind") == "soft_logistic" and
            soft[0].get("provenance") == "cli:--soft-logistic" and
            finite(soft_params.get("r0")) and float(soft_params["r0"]) > 0.0 and
            finite(soft_params.get("sigma")) and float(soft_params["sigma"]) > 0.0,
            f"{graph_rules_path}: soft-logistic CLI rule is malformed")
    representation_path = _result_artifact(root, graph_outputs["representation_rules"], f"{path} representation rules")
    representation = as_map(load_json(representation_path), str(representation_path))
    require(representation.get("schema") == "vitriflow.representation_rules.v1", f"{representation_path}: wrong schema")
    require(bool(as_list(representation.get("representation_rules"), f"{representation_path}.representation_rules")),
            f"{representation_path}: representation provenance is empty")
    graph_metric_path = _result_artifact(root, graph_outputs["graph_metric_by_rule"], f"{path} graph metrics")
    metric_fields, metric_rows = _csv_has_data(graph_metric_path, f"{path} graph metrics")
    require({"box_id", "graph_rule_name", "graph_rule_kind"}.issubset(metric_fields),
            f"{graph_metric_path}: rule/box identity columns are missing")
    metric_box_ids = {_positive_int(int(row["box_id"]), f"{graph_metric_path}.box_id") for row in metric_rows}
    require(metric_box_ids == set(range(1, expected_boxes + 1)),
            f"{graph_metric_path}: graph metrics do not cover all analysed boxes")
    require(set(rule_names).issubset({row["graph_rule_name"] for row in metric_rows}),
            f"{graph_metric_path}: at least one declared graph rule has no metric row")
    stability_path = _result_artifact(root, graph_outputs["coordination_stability"], f"{path} coordination stability")
    stability_fields, stability_rows = _csv_has_data(stability_path, f"{path} coordination stability")
    require({"box_id", "graph_rule_kind"}.issubset(stability_fields),
            f"{stability_path}: interval-stability identity columns are missing")
    require({_positive_int(int(row["box_id"]), f"{stability_path}.box_id") for row in stability_rows}
            == set(range(1, expected_boxes + 1)),
            f"{stability_path}: interval coordination stability does not cover all boxes")
    uncertainty_path = _result_artifact(root, graph_outputs["graph_uncertainty_summary"], f"{path} graph uncertainty")
    _csv_has_data(uncertainty_path, f"{path} graph uncertainty")
    shell_path = _result_artifact(root, graph_outputs["shell_separability"], f"{path} shell separability")
    shell_fields, shell_rows = _csv_has_data(shell_path, f"{path} shell separability")
    require({"box_id", "graph_rule_kind"}.issubset(shell_fields),
            f"{shell_path}: interval-shell identity columns are missing")
    require({_positive_int(int(row["box_id"]), f"{shell_path}.box_id") for row in shell_rows}
            == set(range(1, expected_boxes + 1)),
            f"{shell_path}: shell-separability diagnostics do not cover all boxes")
    return {
        "analysis_schema": data.get("schema"),
        "n_boxes": data.get("n_boxes"),
        "converged": data.get("converged"),
        "sidecars": len(sidecars),
        "graph_rules": len(rules),
        "producer_metric_plumbing_coverage": producer_metric_coverage,
        "metric_convergence_coverage": metric_coverage,
    }


def check_replay_parity(
    source_path: Path,
    analysis_path: Path,
    case: str,
    *,
    expect_cell_refinement: bool = False,
) -> dict[str, Any]:
    """Prove analyze-output reproduces production's canonical numerical result."""

    spec = CASE_SPEC.get(case)
    require(spec is not None, f"unknown case {case!r}")
    source = as_map(load_json(source_path), str(source_path))
    production = as_map(source.get("production"), f"{source_path}.production")
    analysis = as_map(load_json(analysis_path), str(analysis_path))
    require(
        analysis.get("schema") == "vitriflow.analysis_results.v2"
        and analysis.get("status") == "ok"
        and analysis.get("errors") == [],
        f"{analysis_path}: replay analysis did not complete without errors",
    )
    check_count(analysis.get("n_boxes"), 10, f"{analysis_path}.n_boxes")
    check_count(analysis.get("n_boxes_accepted"), 10, f"{analysis_path}.n_boxes_accepted")
    check_count(analysis.get("n_boxes_rejected"), 0, f"{analysis_path}.n_boxes_rejected")
    check_count(analysis.get("n_boxes_total"), 10, f"{analysis_path}.n_boxes_total")

    parity = as_map(analysis.get("convergence_parity"), f"{analysis_path}.convergence_parity")
    require(
        parity.get("schema") == "vitriflow.convergence_parity.v1"
        and parity.get("comparable") is True
        and parity.get("equivalent") is True
        and parity.get("comparison") == "exact_canonical_numerical_contract",
        f"{analysis_path}: canonical production/analyze-output convergence parity failed",
    )
    check_count(parity.get("n_differences"), 0, f"{analysis_path}.convergence_parity.n_differences")
    require(as_list(parity.get("differences"), f"{analysis_path}.convergence_parity.differences") == [],
            f"{analysis_path}: canonical convergence replay has differences")
    expected_source_role = "dft_opt_final" if expect_cell_refinement else "production_relax_ensemble"
    require(parity.get("source_role") == expected_source_role,
            f"{analysis_path}: convergence parity used the wrong source role")

    source_spec = as_map(production.get("convergence_spec"), f"{source_path}.production.convergence_spec")
    require(analysis.get("convergence_spec") == source_spec,
            f"{analysis_path}: analyze-output convergence specification differs from production")
    require(analysis.get("metrics_checked") == production.get("metrics_checked"),
            f"{analysis_path}: analyze-output convergence metric inventory differs from production")

    source_boxes_raw = as_list(production.get("boxes"), f"{source_path}.production.boxes")
    analysis_boxes_raw = as_list(analysis.get("boxes"), f"{analysis_path}.boxes")
    check_count(len(source_boxes_raw), 10, f"{source_path} production boxes")
    check_count(len(analysis_boxes_raw), 10, f"{analysis_path} replay boxes")
    source_by_id: dict[int, Mapping[str, Any]] = {}
    for raw in source_boxes_raw:
        row = as_map(raw, f"{source_path}.production.box")
        box_id = _positive_int(row.get("box"), f"{source_path}.production.box.box")
        if expect_cell_refinement:
            row = as_map(row.get("dft_opt"), f"{source_path}.production.box[{box_id}].dft_opt")
            require(row.get("status") == "ok", f"{source_path}: refined box {box_id} is not ok")
        source_by_id[box_id] = row
    analysis_by_id = {
        _positive_int(as_map(raw, f"{analysis_path}.box").get("box"), f"{analysis_path}.box.box"):
        as_map(raw, f"{analysis_path}.box")
        for raw in analysis_boxes_raw
    }
    require(set(source_by_id) == set(analysis_by_id) == set(range(1, 11)),
            f"{analysis_path}: replay/source box identities are not exactly 1..10")
    compared_fields = ("density", "density_stderr", "metrics", "distributions")
    if not expect_cell_refinement:
        compared_fields = (*compared_fields, "structure")
    for box_id in range(1, 11):
        expected = source_by_id[box_id]
        actual = analysis_by_id[box_id]
        for field in compared_fields:
            if expected.get(field) != actual.get(field):
                differences = _first_differences(
                    expected.get(field),
                    actual.get(field),
                    path=f"box[{box_id}].{field}",
                )
                fail(
                    f"{analysis_path}: exact production/analyze-output {field} parity failed for box {box_id}:\n  "
                    + "\n  ".join(differences)
                )
        _assert_metric_families(actual.get("metrics"), f"{analysis_path}.box[{box_id}].metrics")
        _assert_distribution_families(
            actual.get("distributions"),
            spec,
            f"{analysis_path}.box[{box_id}].distributions",
        )
        structure = as_map(actual.get("structure"), f"{analysis_path}.box[{box_id}].structure")
        check_count(structure.get("n_atoms"), int(spec["atoms"]), f"{analysis_path}.box[{box_id}].n_atoms")
        _check_coordination_and_amorphous_artifacts(
            analysis_path.parent,
            actual,
            case=case,
            box_index=box_id,
        )

    roles = as_map(analysis.get("analysis_source_roles"), f"{analysis_path}.analysis_source_roles")
    if expect_cell_refinement:
        require(roles == {"dft_opt_final": 10},
                f"{analysis_path}: refined replay did not exclusively use ten dft_opt_final sources")
    else:
        require(sum(int(value) for value in roles.values()) == 10 and set(roles).issubset({"relax_trajectory"}),
                f"{analysis_path}: production replay did not exclusively use relaxation trajectories")

    convergence = as_map(analysis.get("convergence"), f"{analysis_path}.convergence")
    _assert_convergence_degree(convergence, f"{analysis_path}.convergence")
    _assert_effective_convergence_families(convergence, spec, f"{analysis_path}.convergence")
    evidence = _assert_convergence_evidence_coverage(
        convergence,
        n_boxes=10,
        label=f"{analysis_path}.convergence",
    )
    producer_metric_coverage = _assert_metric_plumbing_coverage(
        convergence,
        analysis.get("metrics_checked"),
        label=f"{analysis_path}.convergence",
    )
    metric_coverage = _assert_majority_of_emitted_metrics_enter_convergence(
        {**dict(convergence), "metrics_checked": analysis.get("metrics_checked")},
        [analysis_by_id[index] for index in range(1, 11)],
        label=str(analysis_path),
    )

    integrity_path = analysis_path.parent / "sidecar_integrity.json"
    integrity = as_map(load_json(integrity_path), str(integrity_path))
    require(
        integrity.get("schema") == "vitriflow.sidecar_integrity.v2"
        and integrity.get("all_present") is True
        and integrity.get("all_content_hashed") is True
        and integrity.get("all_valid") is True
        and integrity.get("missing") == [],
        f"{integrity_path}: replay sidecar integrity did not pass",
    )
    diagnostics = as_map(analysis.get("diagnostics"), f"{analysis_path}.diagnostics")
    source_integrity = as_map(diagnostics.get("source_integrity"), f"{analysis_path}.diagnostics.source_integrity")
    require(
        source_integrity.get("manifest_locked") is True
        and source_integrity.get("all_sources_verified") is True
        and source_integrity.get("all_source_artifacts_verified") is True
        and source_integrity.get("all_pbc_source_verified") is True,
        f"{analysis_path}: replay source provenance is not fully locked and verified",
    )
    return {
        "case": case,
        "status": "identical",
        "boxes_compared": 10,
        "cell_refinement": bool(expect_cell_refinement),
        "canonical_convergence_parity": True,
        "evidence_coverage": evidence,
        "producer_metric_plumbing_coverage": producer_metric_coverage,
        "metric_convergence_coverage": metric_coverage,
    }


def check_hpc_run(path: Path, source_path: Path, case: str) -> dict[str, Any]:
    """Validate Slurm task execution, collection, diagnostics, and local parity."""

    spec = CASE_SPEC.get(case)
    require(spec is not None, f"unknown case {case!r}")
    engine = str(spec["engine"])
    root = path.parent
    data = as_map(load_json(path), str(path))
    require(data.get("status") == "ok" and data.get("execution_status") == "completed",
            f"{path}: external full-run did not complete successfully")
    production = as_map(data.get("production"), f"{path}.production")
    execution = as_map(production.get("execution"), f"{path}.production.execution")
    require(execution.get("mode") == "full-run", f"{path}: production was not collected in external full-run mode")
    check_count(execution.get("planned_boxes"), 10, f"{path}.production.execution.planned_boxes")
    require(production.get("converged") is True and production.get("convergence_streak") == 2,
            f"{path}: external adaptive convergence did not complete both looks")
    for field, expected in (
        ("n_boxes", 10), ("n_boxes_accepted", 10), ("n_boxes_rejected", 0), ("n_boxes_total", 10),
        ("min_boxes", 6), ("batch_boxes", 4), ("max_boxes", 10),
        ("last_convergence_evaluated_n_boxes_total", 10),
        ("last_convergence_evaluated_n_boxes_accepted", 10),
    ):
        check_count(production.get(field), expected, f"{path}.production.{field}")

    tasks_index_path = root / "production" / "tasks.json"
    tasks_index = as_map(load_json(tasks_index_path), str(tasks_index_path))
    require(tasks_index.get("schema") == "vitriflow.task_index.v1", f"{tasks_index_path}: wrong schema")
    task_records = as_list(tasks_index.get("tasks"), f"{tasks_index_path}.tasks")
    check_count(len(task_records), 10, f"{tasks_index_path} task count")
    task_ids: set[int] = set()
    for raw in task_records:
        record = as_map(raw, f"{tasks_index_path}.task")
        box_id = _positive_int(record.get("box"), f"{tasks_index_path}.task.box")
        require(box_id not in task_ids, f"{tasks_index_path}: duplicate task box {box_id}")
        task_ids.add(box_id)
        task_json = _result_artifact(root / "production", record.get("task_json"), f"{case} task {box_id} manifest")
        task_result_path = _result_artifact(root / "production", record.get("task_result"), f"{case} task {box_id} result")
        task = as_map(load_json(task_json), str(task_json))
        result = as_map(load_json(task_result_path), str(task_result_path))
        require(task.get("schema") == "vitriflow.box_task.v1", f"{task_json}: wrong task schema")
        require(result.get("schema") == "vitriflow.box_task_result.v1" and result.get("status") == "ok",
                f"{task_result_path}: external task did not complete successfully")
        check_count(result.get("box"), box_id, f"{task_result_path}.box")
        declared_plan = as_map(task.get("diagnostic_plan"), f"{task_json}.diagnostic_plan")
        require(declared_plan.get("schema") == "vitriflow.production_task_diagnostic_plan.v1",
                f"{task_json}: task diagnostic plan is absent")
        diagnostics = as_map(result.get("diagnostics"), f"{task_result_path}.diagnostics")
        require(
            diagnostics.get("schema") == "vitriflow.production_task_diagnostics.v1"
            and diagnostics.get("status") == "ok"
            and diagnostics.get("path_base") == "task_box"
            and diagnostics.get("plan") == declared_plan,
            f"{task_result_path}: task diagnostic execution differs from its declared plan",
        )
        _check_stage_metrics_artifacts(
            task_result_path.parent,
            diagnostics.get("stage_metrics"),
            case=case,
            box_index=box_id,
            engine=engine,
        )
        screens = as_map(diagnostics.get("elastic_screens"), f"{task_result_path}.elastic_screens")
        require(set(screens) == {"melt", "relax"}, f"{task_result_path}: elastic screen role coverage is incomplete")
        series = as_map(diagnostics.get("elastic_timeseries"), f"{task_result_path}.elastic_timeseries")
        require(set(series) == {"melt", "quench", "relax"},
                f"{task_result_path}: elastic timeseries role coverage is incomplete")
        if engine == "lammps":
            for role in ("melt", "relax"):
                screen = as_map(screens.get(role), f"{task_result_path}.elastic_screens.{role}")
                require(screen.get("status") == "ok", f"{task_result_path}: {role} elastic screen is not ok")
                _result_artifact(task_result_path.parent, screen.get("summary"), f"{case} task {box_id} {role} elastic summary")
                _result_artifact(task_result_path.parent, screen.get("plot"), f"{case} task {box_id} {role} elastic plot")
            for role in ("melt", "quench", "relax"):
                item = as_map(series.get(role), f"{task_result_path}.elastic_timeseries.{role}")
                require(item.get("status") == "ok", f"{task_result_path}: {role} elastic timeseries is not ok")
        else:
            require(all(screens.get(role) is None for role in ("melt", "relax")),
                    f"{task_result_path}: unsupported CP2K elastic screens were unexpectedly executed")
            require(all(series.get(role) is None for role in ("melt", "quench", "relax")),
                    f"{task_result_path}: unsupported CP2K elastic timeseries were unexpectedly executed")

        artifact_manifest = as_map(result.get("artifact_manifest"), f"{task_result_path}.artifact_manifest")
        require(artifact_manifest.get("schema") == "vitriflow.task_artifacts.v1",
                f"{task_result_path}: task artifact manifest schema is wrong")
        artifact_rows = as_list(artifact_manifest.get("files"), f"{task_result_path}.artifact_manifest.files")
        require(artifact_rows, f"{task_result_path}: task artifact manifest is empty")
        observed_paths: set[str] = set()
        for artifact_raw in artifact_rows:
            artifact = as_map(artifact_raw, f"{task_result_path}.artifact")
            rel = str(artifact.get("path", ""))
            require(rel and rel not in observed_paths, f"{task_result_path}: duplicate/blank artifact path {rel!r}")
            observed_paths.add(rel)
            resolved = (task_result_path.parent / rel).resolve(strict=False)
            require(_strictly_under(resolved, task_result_path.parent) and resolved.is_file(),
                    f"{task_result_path}: manifested artifact is missing or escapes the task root: {rel}")
            check_count(artifact.get("size_bytes"), resolved.stat().st_size,
                        f"{task_result_path}.artifact[{rel}].size_bytes")
            digest = str(artifact.get("sha256", ""))
            require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                    and hashlib.sha256(resolved.read_bytes()).hexdigest() == digest,
                    f"{task_result_path}: artifact hash mismatch for {rel}")
    require(task_ids == set(range(1, 11)), f"{tasks_index_path}: task IDs are not exactly 1..10")

    boxes = [as_map(row, f"{path}.production.box") for row in as_list(production.get("boxes"), f"{path}.production.boxes")]
    check_count(len(boxes), 10, f"{path} production box count")
    source = as_map(load_json(source_path), str(source_path))
    source_prod = as_map(source.get("production"), f"{source_path}.production")
    source_boxes = {
        _positive_int(as_map(row, f"{source_path}.box").get("box"), f"{source_path}.box.box"):
        as_map(row, f"{source_path}.box")
        for row in as_list(source_prod.get("boxes"), f"{source_path}.production.boxes")
    }
    for index, box in enumerate(boxes, start=1):
        box_id = _positive_int(box.get("box"), f"{path}.production.box[{index}].box")
        expected = as_map(source_boxes.get(box_id), f"{source_path}.production.box[{box_id}]")
        for field in ("density", "density_stderr", "metrics", "distributions", "structure"):
            if expected.get(field) != box.get(field):
                fail(
                    f"{path}: Slurm/local exact {field} parity failed for box {box_id}:\n  "
                    + "\n  ".join(_first_differences(expected.get(field), box.get(field), path=f"box[{box_id}].{field}"))
                )
        _assert_metric_families(box.get("metrics"), f"{path}.box[{box_id}].metrics")
        _assert_distribution_families(box.get("distributions"), spec, f"{path}.box[{box_id}].distributions")
        _check_coordination_and_amorphous_artifacts(root, box, case=case, box_index=box_id)
        _check_stage_metrics_artifacts(root, box.get("stage_metrics"), case=case, box_index=box_id, engine=engine)
        if engine == "lammps":
            _check_elastic_artifacts(root, box, case=case, box_index=box_id)
        else:
            require(box.get("elastic_relax") is None and box.get("elastic_timeseries") is None,
                    f"{path}: unsupported CP2K elastic diagnostics were unexpectedly surfaced")

    try:
        from vitriflow.workflows.production_common import compare_convergence_assessments
    except Exception as exc:
        fail(f"cannot import canonical convergence comparator: {exc}")
    convergence = as_map(production.get("convergence"), f"{path}.production.convergence")
    source_convergence = as_map(source_prod.get("convergence"), f"{source_path}.production.convergence")
    convergence_parity = as_map(
        compare_convergence_assessments(source_convergence, convergence),
        f"{path}.canonical_convergence_parity",
    )
    require(convergence_parity.get("equivalent") is True,
            f"{path}: Slurm/local canonical convergence differs: {convergence_parity.get('differences')!r}")
    _assert_convergence_degree(convergence, f"{path}.production.convergence")
    _assert_effective_convergence_families(convergence, spec, f"{path}.production.convergence")
    _assert_convergence_evidence_coverage(convergence, n_boxes=10, label=f"{path}.production.convergence")
    producer_metric_coverage = _assert_metric_plumbing_coverage(
        convergence,
        production.get("metrics_checked"),
        label=f"{path}.production.convergence",
    )
    metric_coverage = _assert_majority_of_emitted_metrics_enter_convergence(
        {**dict(convergence), "metrics_checked": production.get("metrics_checked")},
        boxes,
        label=f"{path}.production",
    )
    return {
        "case": case,
        "mode": "slurm_full_run",
        "tasks": 10,
        "boxes": 10,
        "local_box_parity": True,
        "canonical_convergence_parity": True,
        "producer_metric_plumbing_coverage": producer_metric_coverage,
        "metric_convergence_coverage": metric_coverage,
    }


def check_external_task_results_ready(root: Path, case: str) -> dict[str, Any]:
    """Fail before full-run collection if Slurm omitted or failed any task."""

    spec = CASE_SPEC.get(case)
    require(spec is not None, f"unknown case {case!r}")
    production_dir = root / "production"
    index_path = production_dir / "tasks.json"
    index = as_map(load_json(index_path), str(index_path))
    require(index.get("schema") == "vitriflow.task_index.v1", f"{index_path}: wrong schema")
    records = as_list(index.get("tasks"), f"{index_path}.tasks")
    check_count(len(records), 10, f"{index_path} task count")
    ready: list[int] = []
    for raw in records:
        record = as_map(raw, f"{index_path}.task")
        box_id = _positive_int(record.get("box"), f"{index_path}.task.box")
        task_path = _result_artifact(
            production_dir,
            record.get("task_json"),
            f"{case} task {box_id} manifest",
        )
        result_path = _result_artifact(production_dir, record.get("task_result"), f"{case} task {box_id} result")
        task = as_map(load_json(task_path), str(task_path))
        result = as_map(load_json(result_path), str(result_path))
        require(task.get("schema") == "vitriflow.box_task.v1", f"{task_path}: wrong task schema")
        require(
            result.get("schema") == "vitriflow.box_task_result.v1"
            and result.get("status") == "ok"
            and int(result.get("box", -1)) == box_id,
            f"{result_path}: Slurm task failed or produced the wrong result",
        )
        diagnostics = as_map(result.get("diagnostics"), f"{result_path}.diagnostics")
        declared_plan = as_map(task.get("diagnostic_plan"), f"{task_path}.diagnostic_plan")
        require(
            diagnostics.get("schema") == "vitriflow.production_task_diagnostics.v1"
            and diagnostics.get("status") == "ok"
            and diagnostics.get("path_base") == "task_box"
            and declared_plan.get("schema") == "vitriflow.production_task_diagnostic_plan.v1"
            and diagnostics.get("plan") == declared_plan,
            f"{result_path}: Slurm task diagnostics are incomplete/degraded",
        )
        _check_stage_metrics_artifacts(
            result_path.parent,
            diagnostics.get("stage_metrics"),
            case=case,
            box_index=box_id,
            engine=str(spec["engine"]),
        )
        screens = as_map(diagnostics.get("elastic_screens"), f"{result_path}.elastic_screens")
        series = as_map(diagnostics.get("elastic_timeseries"), f"{result_path}.elastic_timeseries")
        require(set(screens) == {"melt", "relax"},
                f"{result_path}: elastic screen role coverage is incomplete")
        require(set(series) == {"melt", "quench", "relax"},
                f"{result_path}: elastic timeseries role coverage is incomplete")
        if str(spec["engine"]) == "lammps":
            for role in ("melt", "relax"):
                screen = as_map(screens.get(role), f"{result_path}.elastic_screens.{role}")
                require(screen.get("status") == "ok", f"{result_path}: {role} elastic screen is not ok")
                _result_artifact(result_path.parent, screen.get("summary"), f"{case} task {box_id} {role} elastic summary")
                _result_artifact(result_path.parent, screen.get("plot"), f"{case} task {box_id} {role} elastic plot")
            for role in ("melt", "quench", "relax"):
                require(
                    as_map(series.get(role), f"{result_path}.elastic_timeseries.{role}").get("status") == "ok",
                    f"{result_path}: {role} elastic timeseries is not ok",
                )
        else:
            require(all(screens.get(role) is None for role in ("melt", "relax")),
                    f"{result_path}: unsupported CP2K elastic screens were unexpectedly executed")
            require(all(series.get(role) is None for role in ("melt", "quench", "relax")),
                    f"{result_path}: unsupported CP2K elastic timeseries were unexpectedly executed")

        artifact_manifest = as_map(result.get("artifact_manifest"), f"{result_path}.artifact_manifest")
        require(artifact_manifest.get("schema") == "vitriflow.task_artifacts.v1",
                f"{result_path}: task artifact manifest schema is wrong")
        artifacts = as_list(artifact_manifest.get("files"), f"{result_path}.artifact_manifest.files")
        require(artifacts, f"{result_path}: task artifact manifest is empty")
        observed: set[str] = set()
        for raw_artifact in artifacts:
            artifact = as_map(raw_artifact, f"{result_path}.artifact")
            rel = str(artifact.get("path", ""))
            require(rel and rel not in observed, f"{result_path}: duplicate/blank artifact path {rel!r}")
            observed.add(rel)
            resolved = (result_path.parent / rel).resolve(strict=False)
            require(_strictly_under(resolved, result_path.parent) and resolved.is_file(),
                    f"{result_path}: manifested artifact is missing or escapes the task root: {rel}")
            check_count(artifact.get("size_bytes"), resolved.stat().st_size,
                        f"{result_path}.artifact[{rel}].size_bytes")
            digest = str(artifact.get("sha256", ""))
            require(
                re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and hashlib.sha256(resolved.read_bytes()).hexdigest() == digest,
                f"{result_path}: artifact hash mismatch for {rel}",
            )
        ready.append(box_id)
    require(sorted(ready) == list(range(1, 11)), f"{index_path}: completed Slurm task IDs are not exactly 1..10")
    return {"case": case, "status": "ready_for_collection", "tasks": ready}


def check_metrics_csv(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty metrics CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    require(rows, f"metrics CSV has no data rows: {path}")
    require("Step" in fields and "time" in fields, f"metrics CSV lacks Step/time columns: {path}")
    metric_fields = [name for name in fields if name not in {"Step", "time"}]
    require(metric_fields, f"metrics CSV has no metric columns: {path}")
    for prefix in METRIC_PREFIXES:
        require(any(str(name).startswith(prefix) for name in metric_fields),
                f"metrics CSV lacks analytical family {prefix!r}: {path}")
    previous_step: float | None = None
    previous_time: float | None = None
    finite_by_family = {prefix: False for prefix in METRIC_PREFIXES}
    for row_index, row in enumerate(rows, start=2):
        try:
            step = float(row["Step"])
            time = float(row["time"])
        except Exception as exc:
            fail(f"{path}:{row_index}: invalid Step/time: {exc}")
        require(math.isfinite(step) and math.isfinite(time), f"{path}:{row_index}: non-finite Step/time")
        if previous_step is not None:
            require(step > previous_step and time > previous_time, f"{path}:{row_index}: Step/time is not strictly increasing")
        previous_step, previous_time = step, time
        for name in metric_fields:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                for prefix in METRIC_PREFIXES:
                    if str(name).startswith(prefix):
                        finite_by_family[prefix] = True
    missing_finite = [prefix for prefix, present in finite_by_family.items() if not present]
    require(not missing_finite,
            f"metrics CSV has no finite value for analytical families {missing_finite!r}: {path}")
    return {"rows": len(rows), "metric_columns": len(metric_fields)}


def _png_ok(path: Path) -> bool:
    if not (path.is_file() and path.stat().st_size > 100 and
            path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"):
        return False
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as image:
            image.load()
            if image.width < 64 or image.height < 64:
                return False
            rgb = image.convert("RGB")
            extrema = rgb.getextrema()
            variance = ImageStat.Stat(rgb).var
            return any(high > low for low, high in extrema) and max(variance) > 0.25
    except Exception:
        return False


def _png_has_chromatic_data(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as image:
            chromatic = 0
            for red, green, blue in image.convert("RGB").getdata():
                if max(red, green, blue) - min(red, green, blue) >= 18 and min(red, green, blue) < 245:
                    chromatic += 1
                    if chromatic >= 20:
                        return True
    except Exception:
        return False
    return False


def check_page_directory(root: Path, *, kind: str, min_pages: int) -> dict[str, Any]:
    require(root.is_dir(), f"page output directory is missing: {root}")
    suffix = ".png" if kind == "png" else ".pdf"
    pages = sorted(root.glob(f"*{suffix}"))
    require(len(pages) >= int(min_pages), f"expected at least {min_pages} {kind.upper()} pages under {root}, found {len(pages)}")
    validator = _png_ok if kind == "png" else _pdf_ok
    require(all(validator(page) for page in pages), f"invalid/empty {kind.upper()} page under {root}")
    return {"directory": str(root), "kind": kind, "pages": len(pages)}


def _production_payload_for_page_contract(path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    data = as_map(load_json(path), str(path))
    raw_production = data.get("production")
    if isinstance(raw_production, Mapping) and raw_production.get("enabled") is True:
        production = raw_production
    else:
        production = data
    boxes = [as_map(row, f"{path}.box") for row in as_list(production.get("boxes"), f"{path}.boxes")]
    require(boxes, f"{path}: no boxes for production page contract")
    return production, boxes


def _expected_production_pages(path: Path) -> int:
    production, boxes = _production_payload_for_page_contract(path)
    convergence = as_map(production.get("convergence"), f"{path}.convergence")
    spec = as_map(
        convergence.get("convergence_spec_effective", production.get("convergence_spec")),
        f"{path}.convergence_spec_effective",
    )
    ring_keys = [str(value) for value in list(spec.get("ring_keys", []) or [])]
    scalar_keys: set[str] = set()
    sweep_names: set[str] = set()
    for box in boxes:
        scalar_keys.update(str(name) for name in as_map(box.get("metrics"), f"{path}.metrics"))
        details = as_map(box.get("coordination_defect_details"), f"{path}.coordination_defect_details")
        sweep_names.update(
            str(name)
            for name, detail in details.items()
            if isinstance(detail, Mapping) and isinstance(detail.get("coordination_sweep"), Mapping)
        )
    scalar_keys = {
        name
        for name in scalar_keys
        if not name.startswith("ring_frac_") and name != "ring_mean_size"
    }
    curves = sum(
        len(list(spec.get(field, []) or []))
        for field in (
            "bondlen_names", "angle_names", "coord_names",
            "void_names", "gr_labels", "sq_labels",
        )
    )
    pages = (
        2
        + int(bool(ring_keys))
        + int(bool(spec.get("ring_has_mean_size", False)))
        + len(scalar_keys)
        + len(sweep_names)
        + curves
    )

    # An autotune production payload with nested accepted dft_opt results also
    # emits the MD-vs-refined comparison pages in the same public command.
    dft_boxes = [
        as_map(box.get("dft_opt"), f"{path}.dft_opt")
        for box in boxes
        if isinstance(box.get("dft_opt"), Mapping)
        and str(as_map(box.get("dft_opt"), f"{path}.dft_opt").get("status", "")) == "ok"
    ]
    if dft_boxes:
        pages += 1 + int(bool(ring_keys))
        first_distributions = as_map(dft_boxes[0].get("distributions"), f"{path}.dft_opt.distributions")
        for family, field in (
            ("bondlen", "bondlen_names"),
            ("angle", "angle_names"),
            ("void", "void_names"),
            ("coord", "coord_names"),
            ("gr", "gr_labels"),
            ("sq", "sq_labels"),
        ):
            available = as_map(first_distributions.get(family), f"{path}.dft_opt.distributions.{family}")
            pages += sum(str(name) in available for name in list(spec.get(field, []) or []))
    return int(pages)


def _expected_production_comparison_pages(paths: Sequence[Path]) -> int:
    """Mirror the public comparison plot's complete common-page contract."""

    require(len(paths) >= 2, "production comparison page audit needs at least two inputs")
    datasets: list[tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]] = []
    for path in paths:
        production, boxes = _production_payload_for_page_contract(path)
        # plot-production-compare intentionally consumes the persisted
        # production convergence specification rather than a first-box guess.
        spec = as_map(production.get("convergence_spec"), f"{path}.convergence_spec")
        datasets.append((production, boxes, spec))

    metric_sets = [
        {
            str(name)
            for box in boxes
            for name in as_map(box.get("metrics"), f"{path}.metrics")
        }
        | {
            str(name)
            for name in as_map(
                as_map(production.get("convergence"), f"{path}.convergence").get("scalars"),
                f"{path}.convergence.scalars",
            )
        }
        for path, (production, boxes, _spec) in zip(paths, datasets)
    ]
    comparison_metrics = set.union(*metric_sets)
    comparison_metrics = {
        name
        for name in comparison_metrics
        if name not in {"density", "ring_mean_size"}
        and not str(name).startswith("ring_frac_")
    }

    ring_sets = [
        {
            *[str(value) for value in list(spec.get("ring_keys", []) or [])],
            *[
                str(name)
                for name in metric_set
                if str(name).startswith("ring_frac_")
            ],
        }
        for (_production, _boxes, spec), metric_set in zip(datasets, metric_sets)
    ]
    comparison_ring_keys = set.union(*ring_sets)
    has_comparison_ring_mean = any(
        bool(spec.get("ring_has_mean_size", False))
        or "ring_mean_size" in metric_set
        for (_production, _boxes, spec), metric_set in zip(datasets, metric_sets)
    )

    sweep_sets = [
        {
            str(name)
            for box in boxes
            for name, raw in as_map(
                box.get("coordination_defect_details"),
                "comparison coordination_defect_details",
            ).items()
            if isinstance(raw, Mapping)
            and isinstance(raw.get("coordination_sweep"), Mapping)
        }
        for _production, boxes, _spec in datasets
    ]
    comparison_sweeps = set.union(*sweep_sets)

    distribution_fields = (
        ("bondlen", "bondlen_names"),
        ("angle", "angle_names"),
        ("coord", "coord_names"),
        ("void", "void_names"),
        ("gr", "gr_labels"),
        ("sq", "sq_labels"),
    )
    common_curve_count = 0
    for family, field in distribution_fields:
        name_sets: list[set[str]] = []
        for _production, boxes, spec in datasets:
            declared = {str(value) for value in list(spec.get(field, []) or [])}
            emitted = {
                str(name)
                for box in boxes
                for name in as_map(
                    as_map(
                        box.get("distributions"),
                        "comparison box distributions",
                    ).get(family),
                    f"comparison box distributions.{family}",
                )
            }
            name_sets.append(declared | emitted)
        common_curve_count += len(set.union(*name_sets))

    # Convergence and density are unconditional.  A scalar that is emitted but
    # unavailable in every dataset still owns an explicit unavailable page;
    # it must not disappear from this count.
    return int(
        2
        + int(bool(comparison_ring_keys))
        + int(bool(has_comparison_ring_mean))
        + len(comparison_metrics)
        + len(comparison_sweeps)
        + common_curve_count
    )


def check_comparison_plots(root: Path, inputs: Sequence[Path]) -> dict[str, Any]:
    expected = _expected_production_comparison_pages(inputs)
    report = check_page_directory(root, kind="png", min_pages=expected)
    check_count(report.get("pages"), expected, f"{root} exact comparison page count")
    return {"pages": expected, "inputs": [str(path) for path in inputs]}


def check_plots(
    root: Path,
    *,
    engine: str,
    result: Path,
    parity_analysis: Path,
    graph_analysis: Path,
    metrics_csv: Path,
) -> dict[str, Any]:
    require(root.is_dir(), f"plot output directory is missing: {root}")
    required_files = [
        "autotune.png",
        "metric_tm_density.png",
        "metric_rate_density.png",
        "metric_production_density.png",
        "stage.png",
        "voids.png",
    ]
    if engine == "lammps":
        required_files.append("elastic.png")
    for name in required_files:
        path = root / name
        require(_png_ok(path), f"missing, empty, or invalid PNG: {path}")
    for name in ("metric_tm_density.png", "metric_rate_density.png", "metric_production_density.png"):
        require(_png_has_chromatic_data(root / name),
                f"required metric plot contains no chromatic data trace: {root / name}")
    _fields, metric_rows = _csv_has_data(metrics_csv, "public metrics-timeseries")
    metric_fields = [name for name in _fields if name not in {"Step", "time"}]
    require(metric_fields, "public metrics-timeseries CSV has no metric columns")
    required_dirs = {
        "production_pages": _expected_production_pages(result),
        "analysis_parity_production_pages": _expected_production_pages(parity_analysis),
        "analysis_graph_production_pages": _expected_production_pages(graph_analysis),
        "metrics_pages": len(metric_fields),
    }
    page_counts: dict[str, int] = {}
    for name, expected in required_dirs.items():
        report = check_page_directory(root / name, kind="png", min_pages=expected)
        check_count(report.get("pages"), int(expected), f"{root / name} exact public page count")
        page_counts[name] = int(report["pages"])
    return {
        "single_plots": len(required_files),
        **page_counts,
    }


def audit_installed_plotting_contract() -> dict[str, Any]:
    """Fail fast on required public-interface contracts before expensive MD."""
    try:
        import vitriflow.cli as cli
        import vitriflow.plotting as plotting
    except Exception as exc:
        fail(f"cannot import installed Vitriflow validation surface: {exc}")

    package_root = Path(str(cli.__file__)).resolve(strict=True).parent
    try:
        production_common_source = (
            package_root / "workflows" / "production_common.py"
        ).read_text(encoding="utf-8")
        analysis_source = (
            package_root / "workflows" / "output_analysis.py"
        ).read_text(encoding="utf-8")
        hpc_source = (
            package_root / "workflows" / "hpc.py"
        ).read_text(encoding="utf-8")
    except Exception as exc:
        fail(f"cannot inspect installed Vitriflow workflow sources: {exc}")

    issues: list[str] = []
    cli_source = inspect.getsource(cli.main)
    void_start = cli_source.find('if args.cmd == "plot-voids":')
    elastic_start = cli_source.find('if args.cmd == "plot-elastic":', void_start + 1)
    require(void_start >= 0 and elastic_start > void_start, "cannot locate plot-voids CLI branch")
    void_branch = cli_source[void_start:elastic_start]
    if "type_to_species=t2s" in void_branch and "t2s =" not in void_branch:
        issues.append("plot-voids CLI uses uninitialised t2s")

    scan_source = inspect.getsource(plotting.plot_scan_metric)
    rate_marker = 'r.get("rate_K_per_time", float("nan"))'
    if rate_marker in scan_source and not any(
        marker in scan_source for marker in ('r.get("rate",', 'r.get("rate_K_per_ps",')
    ):
        issues.append("plot-metric rate_scan reader accepts neither emitted 'rate' nor 'rate_K_per_ps'")

    production_source = inspect.getsource(plotting.plot_production_results)
    convergence_display_source = inspect.getsource(plotting._production_convergence_display)
    familywise_annotation_source = inspect.getsource(plotting._familywise_error_annotation)
    if "FWER={1.0-alpha_family" in production_source or "FWER={1.0 - alpha_family" in production_source:
        issues.append("production plot reverses alpha_family when labelling FWER")
    if 'conv_flag = bool(prod.get("converged", False))' in production_source:
        issues.append("production plot coerces an unassessed convergence value to false")
    if (
        "_production_convergence_display" not in production_source
        or "convergence_inference_status" not in convergence_display_source
        or "inference_contract" not in convergence_display_source
    ):
        issues.append("production plot omits inference-qualified convergence metadata")
    if (
        "_familywise_error_annotation" not in production_source
        or not re.search(r"FWER(?: alpha)?=\{\s*alpha_family", familywise_annotation_source)
    ):
        issues.append("production plot does not label FWER with alpha_family directly")

    comparison_source = inspect.getsource(plotting.plot_production_comparison_results)
    comparison_impl = getattr(plotting, "_plot_production_comparison_results_impl", None)
    if callable(comparison_impl):
        comparison_source += "\n" + inspect.getsource(comparison_impl)
    if 'else f"{lab} (not converged)"' in comparison_source:
        issues.append("production comparison labels unassessed/prefix-only evidence as not converged")
    if (
        "_production_convergence_display" not in comparison_source
        or "convergence_inference_status" not in convergence_display_source
        or "inference_contract" not in convergence_display_source
    ):
        issues.append("production comparison omits inference-qualified convergence metadata")

    if not callable(getattr(cli, "_result_exit_code", None)):
        issues.append("CLI has no result-aware exit-status policy")

    for token, description in (
        ("vitriflow.convergence_evidence_coverage.v1", "strict-majority convergence evidence"),
        ("vitriflow.metric_plumbing_coverage.v1", "emitted-metric plumbing coverage"),
        ("canonical_convergence_assessment", "canonical convergence assessment"),
        ("compare_convergence_assessments", "canonical convergence comparator"),
    ):
        if token not in production_common_source:
            issues.append(f"production convergence surface omits {description}")

    if "vitriflow.convergence_parity.v1" not in analysis_source:
        issues.append("analyze-output omits canonical production convergence parity")
    if "dft_opt_final" not in analysis_source:
        issues.append("analyze-output cannot identify a positively completed CELL_OPT source")

    for token, description in (
        ("vitriflow.production_task_diagnostic_plan.v1", "external-task diagnostic planning"),
        ("vitriflow.production_task_diagnostics.v1", "external-task diagnostic results"),
        ("vitriflow.task_artifacts.v1", "external-task artifact integrity"),
    ):
        if token not in hpc_source:
            issues.append(f"HPC surface omits {description}")
    if issues:
        fail("installed application interface has release blockers:\n  - " + "\n  - ".join(issues))
    return {
        "application_interface_contract": "present",
        "plotting_contract": "known-defect patterns absent",
        "convergence_parity_contract": "present",
        "metric_plumbing_coverage_contract": "present",
        "hpc_diagnostic_contract": "present",
        "cell_opt_analysis_source_contract": "present",
    }


def _replace_hash_tokens(value: str, digest_tokens: Mapping[str, str]) -> str:
    return re.sub(
        r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])",
        lambda match: digest_tokens.get(match.group(0).lower(), match.group(0)),
        value,
    )


def _normalise(value: Any, root: Path, digest_tokens: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "$NONDETERMINISTIC"
                if str(key) in NONDETERMINISTIC_JSON_KEYS
                else _normalise(item, root, digest_tokens)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise(item, root, digest_tokens) for item in value]
    if isinstance(value, str):
        raw = str(root.resolve(strict=False))
        return _replace_hash_tokens(value.replace(raw, "$RUN_ROOT"), digest_tokens)
    return value


def _first_differences(a: Any, b: Any, path: str = "$", limit: int = 20) -> list[str]:
    out: list[str] = []

    def walk(x: Any, y: Any, where: str) -> None:
        if len(out) >= limit:
            return
        if type(x) is not type(y):
            out.append(f"{where}: type {type(x).__name__} != {type(y).__name__}")
            return
        if isinstance(x, Mapping):
            xk, yk = set(x), set(y)
            for key in sorted(xk - yk):
                out.append(f"{where}.{key}: only in reference")
            for key in sorted(yk - xk):
                out.append(f"{where}.{key}: only in comparison")
            for key in sorted(xk & yk):
                walk(x[key], y[key], f"{where}.{key}")
            return
        if isinstance(x, list):
            if len(x) != len(y):
                out.append(f"{where}: length {len(x)} != {len(y)}")
            for index, (left, right) in enumerate(zip(x, y)):
                walk(left, right, f"{where}[{index}]")
            return
        if x != y:
            out.append(f"{where}: {x!r} != {y!r}")

    walk(a, b, path)
    return out


# NB: `.ener` (CP2K native energy log) is intentionally NOT compared — it carries a
# wall-clock `UsedTime` column that is non-deterministic run-to-run. The reproducible
# physics it records is already covered byte-for-byte by the engine-neutral
# `thermo.csv` / `msd.csv` / `final.extxyz`, which ARE compared.
DETERMINISTIC_SUFFIXES = {
    ".json", ".csv", ".data", ".extxyz", ".lammpstrj", ".table",
    ".npy", ".npz", ".png", ".pdf", ".xyz", ".in", ".inp",
    ".txt", ".yaml", ".yml", ".cif", ".pdb", ".lammps", ".sh", ".slurm",
}

TEXT_SUFFIXES = {
    ".csv", ".data", ".extxyz", ".lammpstrj", ".table", ".xyz",
    ".in", ".inp", ".txt", ".yaml", ".yml", ".cif", ".pdb",
    ".lammps", ".sh", ".slurm",
}

EXCLUDED_RUNTIME_NAMES = {"stdout.txt", "stderr.txt", "screen.out"}

# JSON keys whose values are inherently non-reproducible and are normalised out of
# the byte-compare. `restart_sha256` is the sha256 of CP2K's binary `.wfn` wavefunction
# restart file: the physics it encodes is reproduced bit-for-bit (thermo/positions/MSD
# all compare identical), but the binary `.wfn` container itself is not byte-stable, and
# the `.wfn` is already excluded from the artifact inventory, so its recorded hash carries
# no independent determinism signal.
NONDETERMINISTIC_JSON_KEYS = {"restart_sha256"}


def _artifact_inventory(root: Path) -> dict[str, Path]:
    inventory: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        lammps_input = path.name == "in.lammps" or path.name.endswith(".in.lammps")
        if (
            not path.is_file()
            or path.name in EXCLUDED_RUNTIME_NAMES
            or path.suffix.lower() not in DETERMINISTIC_SUFFIXES
            or (path.suffix.lower() == ".lammps" and not lammps_input)
        ):
            continue
        rel = path.relative_to(root).as_posix()
        inventory[rel] = path
    return inventory


def _digest_token_map(inventory: Mapping[str, Path]) -> dict[str, str]:
    by_digest: dict[str, list[str]] = {}
    for rel, path in inventory.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_digest.setdefault(digest, []).append(rel)
    return {
        digest: "$ARTIFACT_SHA256:" + "|".join(sorted(paths))
        for digest, paths in by_digest.items()
    }


def _normalised_text(path: Path, root: Path, digest_tokens: Mapping[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace(str(root.resolve(strict=False)), "$RUN_ROOT")
    return _replace_hash_tokens(text, digest_tokens)


def _normalised_pdf(path: Path, root: Path, digest_tokens: Mapping[str, str]) -> bytes:
    payload = path.read_bytes().replace(str(root.resolve(strict=False)).encode("utf-8"), b"$RUN_ROOT")
    payload = re.sub(rb"/(?:CreationDate|ModDate)\s*\(D:[^)]*\)", b"/CreationDate (D:$NORMALISED)", payload)
    payload = re.sub(
        rb"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])",
        lambda match: digest_tokens.get(match.group(0).decode("ascii").lower(), match.group(0).decode("ascii")).encode("utf-8"),
        payload,
    )
    return payload


def compare_roots(reference: Path, comparison: Path, report_path: Path) -> dict[str, Any]:
    require(reference.is_dir(), f"reference root missing: {reference}")
    require(comparison.is_dir(), f"comparison root missing: {comparison}")
    left = _artifact_inventory(reference)
    right = _artifact_inventory(comparison)
    require(set(left) == set(right),
            "artifact inventories differ:\n  reference-only=" + repr(sorted(set(left) - set(right))[:20]) +
            "\n  comparison-only=" + repr(sorted(set(right) - set(left))[:20]))
    left_tokens = _digest_token_map(left)
    right_tokens = _digest_token_map(right)

    compared = 0
    total_bytes = 0
    digest = hashlib.sha256()
    for rel in sorted(left):
        a, b = left[rel], right[rel]
        suffix = a.suffix.lower()
        if suffix == ".json":
            av = _normalise(load_json(a), reference, left_tokens)
            bv = _normalise(load_json(b), comparison, right_tokens)
            if av != bv:
                fail(f"JSON mismatch in {rel}:\n  " + "\n  ".join(_first_differences(av, bv)))
            payload = json.dumps(av, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        elif suffix in TEXT_SUFFIXES:
            av = _normalised_text(a, reference, left_tokens)
            bv = _normalised_text(b, comparison, right_tokens)
            require(av == bv, f"exact text/numeric artifact mismatch: {rel}")
            payload = av.encode("utf-8")
        elif suffix == ".pdf":
            avb = _normalised_pdf(a, reference, left_tokens)
            bvb = _normalised_pdf(b, comparison, right_tokens)
            require(avb == bvb, f"exact PDF content mismatch after metadata normalisation: {rel}")
            payload = avb
        else:
            avb, bvb = a.read_bytes(), b.read_bytes()
            require(avb == bvb, f"exact binary/image artifact mismatch: {rel}")
            payload = avb
        compared += 1
        total_bytes += len(payload)
        digest.update(rel.encode("utf-8") + b"\0" + payload + b"\0")

    report = {
        "schema": "vitriflow.application_validation.comparison.v1",
        "status": "identical",
        "comparison_mode": "exact_after_run_root_path_derived_hash_and_pdf_date_normalisation",
        "artifact_suffix_policy": sorted(DETERMINISTIC_SUFFIXES),
        "excluded_runtime_artifacts": ["*.log", "*.out", "*.ener (CP2K wall-clock column)", "non-input *.lammps", "stdout.txt", "stderr.txt", "screen.out"],
        "normalised_json_keys": sorted(NONDETERMINISTIC_JSON_KEYS),
        "artifact_count": compared,
        "normalised_bytes": total_bytes,
        "canonical_sha256": digest.hexdigest(),
        "reference": str(reference),
        "comparison": str(comparison),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def emit(value: Mapping[str, Any] | Sequence[Any] | str) -> None:
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit-code")
    a.set_defaults(run=lambda _args: audit_installed_plotting_contract())

    r = sub.add_parser("check-run")
    r.add_argument("--result", type=Path, required=True)
    r.add_argument("--case", choices=sorted(CASE_SPEC), required=True)
    r.add_argument("--expect-cell-refinement", action="store_true")
    r.set_defaults(
        run=lambda args: check_run(
            args.result,
            args.case,
            expect_cell_refinement=bool(args.expect_cell_refinement),
        )
    )

    cfg = sub.add_parser("check-config")
    cfg.add_argument("--config", type=Path, required=True)
    cfg.add_argument("--case", choices=sorted(CASE_SPEC), required=True)
    cfg.add_argument("--expect-cell-refinement", action="store_true")
    cfg.set_defaults(
        run=lambda args: check_config(
            args.config,
            args.case,
            expect_cell_refinement=bool(args.expect_cell_refinement),
        )
    )

    s = sub.add_parser("stage-dir")
    s.add_argument("--result", type=Path, required=True)
    s.set_defaults(run=lambda args: str(stage_dir_from_result(args.result)))

    an = sub.add_parser("check-analysis")
    an.add_argument("--result", type=Path, required=True)
    an.add_argument("--case", choices=sorted(CASE_SPEC), required=True)
    an.add_argument("--boxes", type=int, default=10)
    an.set_defaults(run=lambda args: check_analysis(args.result, args.case, args.boxes))

    parity = sub.add_parser("check-parity")
    parity.add_argument("--source", type=Path, required=True)
    parity.add_argument("--analysis", type=Path, required=True)
    parity.add_argument("--case", choices=sorted(CASE_SPEC), required=True)
    parity.add_argument("--expect-cell-refinement", action="store_true")
    parity.set_defaults(
        run=lambda args: check_replay_parity(
            args.source,
            args.analysis,
            args.case,
            expect_cell_refinement=bool(args.expect_cell_refinement),
        )
    )

    hpc = sub.add_parser("check-hpc")
    hpc.add_argument("--result", type=Path, required=True)
    hpc.add_argument("--source", type=Path, required=True)
    hpc.add_argument("--case", choices=sorted(CASE_SPEC), required=True)
    hpc.set_defaults(run=lambda args: check_hpc_run(args.result, args.source, args.case))

    ready = sub.add_parser("check-task-results-ready")
    ready.add_argument("--root", type=Path, required=True)
    ready.add_argument("--case", choices=sorted(CASE_SPEC), required=True)
    ready.set_defaults(run=lambda args: check_external_task_results_ready(args.root, args.case))

    m = sub.add_parser("check-metrics-csv")
    m.add_argument("--input", type=Path, required=True)
    m.set_defaults(run=lambda args: check_metrics_csv(args.input))

    pl = sub.add_parser("check-plots")
    pl.add_argument("--dir", type=Path, required=True)
    pl.add_argument("--engine", choices=["lammps", "cp2k"], required=True)
    pl.add_argument("--result", type=Path, required=True)
    pl.add_argument("--parity-analysis", type=Path, required=True)
    pl.add_argument("--graph-analysis", type=Path, required=True)
    pl.add_argument("--metrics-csv", type=Path, required=True)
    pl.set_defaults(
        run=lambda args: check_plots(
            args.dir,
            engine=args.engine,
            result=args.result,
            parity_analysis=args.parity_analysis,
            graph_analysis=args.graph_analysis,
            metrics_csv=args.metrics_csv,
        )
    )

    pages = sub.add_parser("check-pages")
    pages.add_argument("--dir", type=Path, required=True)
    pages.add_argument("--kind", choices=["png", "pdf"], default="png")
    pages.add_argument("--min-pages", type=int, default=1)
    pages.set_defaults(run=lambda args: check_page_directory(args.dir, kind=args.kind, min_pages=args.min_pages))

    comparison_pages = sub.add_parser("check-comparison-plots")
    comparison_pages.add_argument("--dir", type=Path, required=True)
    comparison_pages.add_argument("--input", dest="inputs", type=Path, nargs="+", required=True)
    comparison_pages.set_defaults(
        run=lambda args: check_comparison_plots(args.dir, args.inputs)
    )

    c = sub.add_parser("compare")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--comparison", type=Path, required=True)
    c.add_argument("--report", type=Path, required=True)
    c.set_defaults(run=lambda args: compare_roots(args.reference, args.comparison, args.report))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        expected_package = os.environ.get("VITRIFLOW_VALIDATION_EXPECTED_PACKAGE", "").strip()
        if expected_package:
            try:
                import vitriflow
                actual = Path(str(vitriflow.__file__)).resolve(strict=True)
                expected = Path(expected_package).resolve(strict=True)
            except Exception as exc:
                fail(f"cannot resolve the expected Vitriflow package: {exc}")
            require(actual == expected,
                    f"imported Vitriflow package is shadowed: expected {expected}, imported {actual}")
        args = parser().parse_args(argv)
        result = args.run(args)
        emit(result)
        return 0
    except ValidationError as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
