from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest


def _commands(*, hybrid: bool) -> list[str]:
    if hybrid:
        return [
            "pair_style hybrid/overlay coul/long 15.0 buck 15.0 morse 15.0",
            "pair_coeff 1 1 coul/long",
            "pair_coeff 1 1 buck 139349.01 0.21 171.08",
            "pair_coeff 1 2 coul/long",
            "pair_coeff 1 2 buck 412.55 0.30 0.0",
            "pair_coeff 1 2 morse 0.44 2.57 1.91",
            "pair_coeff 2 2 coul/long",
            "pair_coeff 2 2 buck 1388.77 0.36 175.00",
            "kspace_style pppm 1.0e-6",
        ]
    return [
        "pair_style buck/coul/long 16.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 907.89 0.345 10.0",
        "pair_coeff 2 2 22764.0 0.149 0.0",
        "kspace_style pppm 1.0e-6",
    ]


def _case(*, hybrid: bool):
    from vitriflow.config import RunConfig
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )

    commands = _commands(hybrid=hybrid)
    charges = {"Ga": 1.8, "O": -1.2} if hybrid else {"Ga": 3.0, "O": -2.0}
    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Ga", "O"],
                "commands": commands,
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "table_points": 4096,
                    "table_points_max": 4096,
                    "table_verify_points": 2001,
                    "table_filename": "core.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                },
            },
            "structure": {
                "charges": charges,
                "generate": {
                    "method": "random",
                    "formula": "Ga2O3",
                    "n_formula_units": 1,
                },
            },
            "md": {"atom_style": "charge"},
        }
    )
    lines = build_tabulated_buckingham_core_lines(
        commands,
        species=["Ga", "O"],
        units_style="metal",
        r_in=1.0,
        r_out=1.4,
        table_points=4096,
        table_filename="core.table",
        table_r_min=0.1,
        charges=charges,
        gewald=0.224358,
        has_bonded_topology=False,
        table_style="spline",
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    return cfg, commands, spec


def _exact_fake_pairwrite(*, perturb_zero: float = 0.0, perturb_charged: float = 0.0):
    from vitriflow.workflows.preflight import _source_equivalence_component_references

    calls: list[dict] = []

    def fake(*args, **kwargs):
        spec = kwargs["spec"]
        full, _noncoul, _coul, _scale = _source_equivalence_component_references(
            spec,
            npoints=kwargs["npoints"],
        )
        sections = copy.deepcopy(full)
        is_zero = all(
            float(pair.get("q_i", 0.0) or 0.0) == 0.0
            and float(pair.get("q_j", 0.0) or 0.0) == 0.0
            for pair in spec["pairs"]
        )
        perturbation = float(perturb_zero if is_zero else perturb_charged)
        if perturbation:
            # A smooth, finite error over the interior cannot be dismissed as
            # a cutoff-endpoint or serialization artifact.
            for section in sections.values():
                n = len(section["r"])
                window = np.sin(np.linspace(0.0, np.pi, n, dtype=float))
                section["energy"] = np.asarray(section["energy"]) + perturbation * window
                section["force"] = np.asarray(section["force"]) + perturbation * window
        rendered = list(kwargs["potential_lines"])
        match = next(
            (
                re.search(r"pair_modify\s+table\s+(\d+)", str(line))
                for line in rendered
                if "pair_modify" in str(line)
            ),
            None,
        )
        calls.append(
            {
                "zero_charge": is_zero,
                "table_bits": None if match is None else int(match.group(1)),
                "potential_lines": rendered,
            }
        )
        return {
            "path": Path(kwargs["stage_dir"]) / kwargs["output_name"],
            "sections": sections,
            "warnings": [],
        }

    return fake, calls


def test_component_reference_evaluates_runtime_coulomb_without_cancellation() -> None:
    from vitriflow.potential import _pair_coulomb_energy_derivatives
    from vitriflow.workflows.preflight import _source_equivalence_component_references

    pair = {
        "section": "P2_2",
        "source_audit_r_min": 1.7,
        "r_out": 1.7,
        "pair_cutoff": 10.0,
        "buck_cutoff": 10.0,
        "A": 1388.773,
        "rho": 0.362319,
        "C": 175.0,
        "shift_buck": True,
        "morse_terms": [],
        "coul_mode": "long",
        "coul_cutoff": 10.0,
        "q_i": -1.2,
        "q_j": -1.2,
        "gewald": 0.8,
    }
    _full, _noncoul, coul, _scale = _source_equivalence_component_references(
        {"units": "metal", "pairs": [pair]},
        npoints=50_001,
    )
    radii = np.asarray(coul["P2_2"]["r"], dtype=float)
    evaluation_r = radii.copy()
    evaluation_r[-1] = np.nextafter(10.0, 1.7)
    expected_energy, expected_derivative, _ = _pair_coulomb_energy_derivatives(
        evaluation_r,
        pair=pair,
        units_style="metal",
        representation="runtime",
    )
    expected_force = -expected_derivative
    expected_energy[-1] = 0.0
    expected_force[-1] = 0.0

    # This screened tail is non-zero but much smaller than the Buckingham
    # operand.  An implementation that forms (Buck+Coul)-Buck loses bits here.
    tail = int(np.argmin(np.abs(radii - 8.15)))
    assert 0.0 < abs(float(expected_energy[tail])) < 1.0e-17
    np.testing.assert_array_equal(coul["P2_2"]["energy"], expected_energy)
    np.testing.assert_array_equal(coul["P2_2"]["force"], expected_force)


@pytest.mark.parametrize(
    ("noise_factor", "expected_pass"),
    [(0.5, True), (2.0, False)],
)
def test_subtracted_pairwrite_roundoff_bound_is_narrow_and_auditable(
    noise_factor: float,
    expected_pass: bool,
) -> None:
    from vitriflow.workflows.preflight import _compare_pair_table_sections

    radii = np.asarray([7.9, 8.1, 8.3], dtype=float)
    reference_energy = np.asarray([2.0e-20, 1.0e-20, 0.5e-20], dtype=float)
    reference_force = np.asarray([2.0e-19, 1.0e-19, 0.5e-19], dtype=float)
    energy_operand_scale = np.full(3, 8.0e-4, dtype=float)
    force_operand_scale = np.full(3, 9.0e-4, dtype=float)
    energy_bound = 64.0 * np.finfo(float).eps * energy_operand_scale
    force_bound = 64.0 * np.finfo(float).eps * force_operand_scale
    reference = {
        "P2_2": {
            "r": radii,
            "energy": reference_energy,
            "force": reference_force,
        }
    }
    realized = {
        "P2_2": {
            "r": radii.copy(),
            "energy": reference_energy + float(noise_factor) * energy_bound,
            "force": reference_force + float(noise_factor) * force_bound,
        }
    }
    comparison = _compare_pair_table_sections(
        reference,
        realized,
        rel_tol=0.0,
        abs_tol_frac=0.0,
        subtraction_roundoff_scale_sections={
            "P2_2": {
                "energy": energy_operand_scale,
                "force": force_operand_scale,
            }
        },
    )

    assert comparison["passed"] is expected_pass
    pair_report = comparison["pairs"]["P2_2"]
    assert pair_report["energy_tolerance_scale"][
        "subtraction_roundoff_factor_eps"
    ] == 64.0
    assert pair_report["energy_tolerance_scale"][
        "subtraction_roundoff_limited_points"
    ] == 3
    assert pair_report["energy_tolerance_scale"]["auxiliary_max_scale"] == 0.0
    if expected_pass:
        assert pair_report["max_energy_ratio"] == pytest.approx(noise_factor)
        assert pair_report["max_force_ratio"] == pytest.approx(noise_factor)
    else:
        assert pair_report["max_energy_ratio"] > 1.0
        assert pair_report["max_force_ratio"] > 1.0


def test_subtraction_roundoff_bound_does_not_mask_resolved_coulomb_error() -> None:
    from vitriflow.workflows.preflight import _compare_pair_table_sections

    radii = np.asarray([1.5, 2.0, 2.5], dtype=float)
    reference_energy = np.asarray([2.0, 1.0, 0.5], dtype=float)
    reference_force = np.asarray([3.0, 1.5, 0.75], dtype=float)
    operand_scale = np.ones(3, dtype=float)
    comparison = _compare_pair_table_sections(
        {"P1_2": {"r": radii, "energy": reference_energy, "force": reference_force}},
        {
            "P1_2": {
                "r": radii.copy(),
                "energy": 1.01 * reference_energy,
                "force": 1.01 * reference_force,
            }
        },
        rel_tol=5.0e-5,
        abs_tol_frac=1.0e-7,
        subtraction_roundoff_scale_sections={
            "P1_2": {"energy": operand_scale, "force": operand_scale}
        },
    )

    assert comparison["passed"] is False
    pair_report = comparison["pairs"]["P1_2"]
    assert pair_report["max_energy_ratio"] > 100.0
    assert pair_report["max_force_ratio"] > 100.0
    assert pair_report["energy_tolerance_scale"][
        "subtraction_roundoff_limited_points"
    ] == 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_component_audit_accepts_direct_and_hybrid_without_runtime_override_leak(
    monkeypatch,
    tmp_path: Path,
    hybrid: bool,
) -> None:
    from vitriflow.workflows.preflight import _audit_original_potential_above_core_joins

    cfg, commands, spec = _case(hybrid=hybrid)
    original_commands = list(commands)
    fake, calls = _exact_fake_pairwrite()
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves", fake
    )

    report = _audit_original_potential_above_core_joins(
        object(),
        cfg,
        outdir=tmp_path,
        source_potential_lines=commands,
        spec=spec,
    )

    assert report["passed"] is True
    component = report["component_audit"]
    assert component["accepted_table_bits"] == 18
    assert component["audit_only_pair_modify"] is True
    assert component["runtime_potential_commands_modified"] is False
    assert commands == original_commands
    for candidate in component["candidates"]:
        for pair_report in candidate["coulomb"]["pairs"].values():
            assert (
                pair_report["energy_tolerance_scale"]["auxiliary_max_scale"]
                == 0.0
            )
            assert (
                pair_report["force_tolerance_scale"]["auxiliary_max_scale"]
                == 0.0
            )
            assert (
                pair_report["energy_tolerance_scale"][
                    "subtraction_operand_max_scale"
                ]
                > 0.0
            )
            assert pair_report["energy_tolerance_scale"][
                "subtraction_roundoff_factor_eps"
            ] == 64.0
    assert [(call["table_bits"], call["zero_charge"]) for call in calls] == [
        (16, False),
        (16, True),
        (18, False),
        (18, True),
    ]
    assert all(
        any("pair_modify table" in line for line in call["potential_lines"])
        for call in calls
    )


def test_component_audit_rejects_buckingham_or_morse_source_disagreement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import (
        PreflightError,
        _audit_original_potential_above_core_joins,
    )

    cfg, commands, spec = _case(hybrid=True)
    fake, _calls = _exact_fake_pairwrite(perturb_zero=0.05)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves", fake
    )

    with pytest.raises(PreflightError, match="component-wise agreement"):
        _audit_original_potential_above_core_joins(
            object(),
            cfg,
            outdir=tmp_path,
            source_potential_lines=commands,
            spec=spec,
        )
    report = (tmp_path / "preflight" / "source_equivalence" / "source_equivalence_report.json")
    assert '"passed": false' in report.read_text()


def test_component_audit_rejects_charge_or_coulomb_source_disagreement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import (
        PreflightError,
        _audit_original_potential_above_core_joins,
    )

    cfg, commands, spec = _case(hybrid=False)
    fake, _calls = _exact_fake_pairwrite(perturb_charged=0.05)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves", fake
    )

    with pytest.raises(PreflightError, match="Coulomb-resolution convergence"):
        _audit_original_potential_above_core_joins(
            object(),
            cfg,
            outdir=tmp_path,
            source_potential_lines=commands,
            spec=spec,
        )


def test_component_audit_preserves_audited_kim_fixed_charge_semantics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from vitriflow.config import RunConfig
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
        update_tabulated_core_metadata_lines,
    )
    from vitriflow.workflows.preflight import _audit_original_potential_above_core_joins

    set_lines = ["set type 1 charge 2.4", "set type 2 charge -1.2"]
    commands = [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381",
        "pair_coeff 2 2 1388.7730 0.362319 175.0000",
        "kspace_style pppm 1.0e-6",
        *set_lines,
    ]
    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": commands,
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "table_points": 4096,
                    "table_points_max": 4096,
                    "table_verify_points": 2001,
                    "table_filename": "core.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                },
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "SiO2",
                    "n_formula_units": 1,
                }
            },
            "md": {"atom_style": "charge"},
        }
    )
    lines = build_tabulated_buckingham_core_lines(
        commands,
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=4096,
        table_filename="core.table",
        table_r_min=0.1,
        charges={"Si": 2.4, "O": -1.2},
        gewald=0.224358,
        has_bonded_topology=False,
        table_style="spline",
    )
    audit = {
        "passed": True,
        "effective_charges_e": {"Si": 2.4, "O": -1.2},
        "fixed_set_commands": [
            {"command": set_lines[0], "atom_type": 1, "charge_native": 2.4},
            {"command": set_lines[1], "atom_type": 2, "charge_native": -1.2},
        ],
    }
    lines = update_tabulated_core_metadata_lines(lines, runtime_charge_audit=audit)
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    fake, calls = _exact_fake_pairwrite()
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves", fake
    )

    report = _audit_original_potential_above_core_joins(
        object(),
        cfg,
        outdir=tmp_path,
        source_potential_lines=commands,
        spec=spec,
    )

    assert report["passed"] is True
    assert report["component_audit"]["accepted_table_bits"] == 18
    assert all(
        set_lines[0] not in call["potential_lines"]
        and set_lines[1] not in call["potential_lines"]
        for call in calls
    )
