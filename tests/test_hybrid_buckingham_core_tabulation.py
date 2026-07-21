from __future__ import annotations

import math

import numpy as np
import pytest


def _ga2o3_hybrid_commands() -> list[str]:
    return [
        "pair_style hybrid/overlay coul/long 15.0 buck 15.0 morse 15.0",
        "kspace_style pppm 1e-6",
        "pair_coeff 1 1 coul/long",
        "pair_coeff 1 1 buck 139349.01 0.21 171.08",
        "pair_coeff 1 2 coul/long",
        "pair_coeff 1 2 buck 412.55 0.3 0",
        "pair_coeff 1 2 morse 0.44 2.57 1.91",
        "pair_coeff 2 2 coul/long",
        "pair_coeff 2 2 buck 1388.77 0.36 175.00",
    ]


def _ga2o3_spec():
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )

    lines = build_tabulated_buckingham_core_lines(
        _ga2o3_hybrid_commands(),
        species=["Ga", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.2,
        table_points=4000,
        table_filename="ga2o3_core.table",
        table_r_min=0.1,
        charges={"Ga": 1.8, "O": -1.2},
        gewald=0.2,
        has_bonded_topology=False,
        table_style="spline",
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    return lines, spec


def test_ga2o3_additive_hybrid_is_replaced_by_one_kspace_table() -> None:
    lines, spec = _ga2o3_spec()

    runtime = "\n".join(line for line in lines if not line.startswith("#"))
    assert "pair_style table spline 4000 pppm" in runtime
    assert "pair_style hybrid/overlay" not in runtime
    assert "pair_coeff 1 2 ga2o3_core.table P1_2 15" in runtime
    assert "kspace_style pppm 1e-6" in runtime
    assert "kspace_modify gewald 0.2" in runtime

    assert spec["version"] == 10
    assert spec["kind"] == "additive_hybrid_buckingham_zbl_table"
    assert spec["source_pair_style"] == (
        "pair_style hybrid/overlay coul/long 15.0 buck 15.0 morse 15.0"
    )
    assert spec["source_hybrid_components"] == ["buck", "coul/long", "morse"]
    assert spec["regularized_components"] == ["buckingham"]
    assert spec["morse_policy"] == "preserved_unchanged_at_all_r"
    assert spec["ewald_split_invariant_regularization"] is True
    assert spec["common_kspace_cutoff"] == pytest.approx(15.0)
    assert [pair["pair"] for pair in spec["pairs"]] == [[1, 1], [1, 2], [2, 2]]
    cross = next(pair for pair in spec["pairs"] if pair["pair"] == [1, 2])
    assert cross["morse_terms"] == [
        {"D0": 0.44, "alpha": 2.57, "r0": 1.91, "cutoff": 15.0}
    ]
    for pair in spec["pairs"]:
        assert pair["pair_cutoff"] == pytest.approx(15.0)
        assert pair["validation"]["c2_join_validated"] is True
        assert pair["validation"]["repulsive_through_r_out"] is True
        assert pair["validation"]["minimum_core_force"] >= 0.0
        assert pair["validation"]["morse_policy"] == (
            "preserved_unchanged_at_all_r"
            if pair["pair"] == [1, 2]
            else "not_present"
        )


def test_hybrid_morse_overlay_is_preserved_unchanged_below_the_core_join() -> None:
    from vitriflow.potential import (
        _pair_coulomb_energy_derivatives,
        _pair_morse_energy_derivatives,
        _regularized_buckingham_component_energy_derivatives,
        _repulsive_regularized_pair_energy_derivatives,
    )

    _lines, spec = _ga2o3_spec()
    pair = next(pair for pair in spec["pairs"] if pair["pair"] == [1, 2])
    radius = np.asarray(
        [0.15, 0.5, float(pair["r_in"]), 0.5 * (pair["r_in"] + pair["r_out"])],
        dtype=float,
    )
    runtime = _repulsive_regularized_pair_energy_derivatives(
        radius, pair=pair, units_style="metal"
    )
    buck_core = _regularized_buckingham_component_energy_derivatives(
        radius, pair=pair, units_style="metal"
    )
    real_coulomb = _pair_coulomb_energy_derivatives(
        radius,
        pair=pair,
        units_style="metal",
        representation="runtime",
    )
    original_morse = _pair_morse_energy_derivatives(
        radius, pair=pair, units_style="metal"
    )
    for realized, buck, coulomb, morse in zip(
        runtime, buck_core, real_coulomb, original_morse
    ):
        assert np.allclose(
            realized - buck - coulomb,
            morse,
            rtol=2.0e-14,
            atol=2.0e-12,
        )


def test_ga2o3_hybrid_full_base_is_exactly_retained_above_each_join() -> None:
    from vitriflow.potential import (
        _pair_base_energy_derivatives,
        _repulsive_regularized_pair_energy_derivatives,
        tabulated_buckingham_unregularized_reference_sections,
    )

    _lines, spec = _ga2o3_spec()
    for pair in spec["pairs"]:
        radii = np.asarray(
            [
                float(pair["r_out"]),
                float(pair["r_out"]) + 0.1,
                5.0,
                14.0,
            ]
        )
        base = _pair_base_energy_derivatives(radii, pair=pair, units_style="metal")
        regularized = _repulsive_regularized_pair_energy_derivatives(
            radii, pair=pair, units_style="metal"
        )
        for actual, expected in zip(regularized, base):
            assert actual == pytest.approx(expected, rel=2.0e-13, abs=2.0e-13)

    reference = tabulated_buckingham_unregularized_reference_sections(spec, npoints=257)
    assert set(reference) == {"P1_1", "P1_2", "P2_2"}
    for pair in spec["pairs"]:
        section = reference[pair["section"]]
        assert section["r"][0] == pytest.approx(pair["r_out"])
        assert section["r"][-1] == pytest.approx(15.0)
        assert section["energy"][-1] == 0.0
        assert section["force"][-1] == 0.0


def test_ga_o_base_contains_buckingham_morse_and_screened_coulomb() -> None:
    from vitriflow.potential import (
        _coulomb_prefactor_energy_distance,
        _pair_base_energy_derivatives,
    )

    _lines, spec = _ga2o3_spec()
    pair = next(pair for pair in spec["pairs"] if pair["pair"] == [1, 2])
    radius = 2.3
    energy, derivative, second = _pair_base_energy_derivatives(
        np.asarray([radius]), pair=pair, units_style="metal"
    )

    A, rho, C = 412.55, 0.3, 0.0
    D0, alpha, r0 = 0.44, 2.57, 1.91
    x = math.exp(-alpha * (radius - r0))
    pref = _coulomb_prefactor_energy_distance("metal") * 1.8 * -1.2
    erfc = math.erfc(0.2 * radius)
    expg = math.exp(-((0.2 * radius) ** 2))
    a = 2.0 * 0.2 / math.sqrt(math.pi)
    expected_energy = (
        A * math.exp(-radius / rho)
        - C / radius**6
        + D0 * (x * x - 2.0 * x)
        + pref * erfc / radius
    )
    expected_derivative = (
        -A * math.exp(-radius / rho) / rho
        + 6.0 * C / radius**7
        + 2.0 * D0 * alpha * (x - x * x)
        + pref * (-a * expg / radius - erfc / radius**2)
    )
    expected_second = (
        A * math.exp(-radius / rho) / rho**2
        - 42.0 * C / radius**8
        + 2.0 * D0 * alpha**2 * (2.0 * x * x - x)
        + pref
        * (
            2.0 * a * 0.2**2 * expg
            + 2.0 * a * expg / radius**2
            + 2.0 * erfc / radius**3
        )
    )
    assert energy[0] == pytest.approx(expected_energy, rel=3.0e-15)
    assert derivative[0] == pytest.approx(expected_derivative, rel=3.0e-15)
    assert second[0] == pytest.approx(expected_second, rel=3.0e-15)


def test_hybrid_coul_cut_is_representable_when_all_component_cutoffs_match() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    commands = [
        line.replace("coul/long", "coul/cut")
        for line in _ga2o3_hybrid_commands()
        if not line.startswith("kspace_style")
    ]
    lines = build_tabulated_buckingham_core_lines(
        commands,
        species=["Ga", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.2,
        table_points=2000,
        table_filename="ga2o3_cut_core.table",
        table_r_min=0.1,
        charges={"Ga": 1.8, "O": -1.2},
        has_bonded_topology=False,
    )
    runtime = "\n".join(line for line in lines if not line.startswith("#"))
    assert "pair_style table linear 2000\n" in runtime
    assert "pppm" not in runtime
    assert "kspace_style" not in runtime


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda lines: [
                line.replace("pair_coeff 1 1 buck", "pair_coeff * * buck")
                if line.startswith("pair_coeff 1 1 buck")
                else line
                for line in lines
            ],
            "wildcard/range selectors",
        ),
        (
            lambda lines: [
                line for line in lines if line != "pair_coeff 2 2 coul/long"
            ],
            "explicit coul/long pair_coeff coverage",
        ),
        (
            lambda lines: [
                line.replace("morse 15.0", "morse 12.0")
                if line.startswith("pair_style")
                else line
                for line in lines
            ],
            "must exactly match the common Coulomb/KSpace cutoff",
        ),
        (
            lambda lines: (
                [lines[0].replace("morse 15.0", "buck 15.0 morse 15.0")] + lines[1:]
            ),
            "multiple instances",
        ),
        (
            lambda lines: [lines[0] + " lj/cut 15.0"] + lines[1:],
            "supports only one Buckingham substyle",
        ),
    ],
)
def test_hybrid_parser_rejects_unprovable_or_lossy_models(mutator, message) -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    with pytest.raises(ValueError, match=message):
        build_tabulated_buckingham_core_lines(
            mutator(_ga2o3_hybrid_commands()),
            species=["Ga", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.2,
            table_points=2000,
            table_filename="ga2o3_core.table",
            table_r_min=0.1,
            charges={"Ga": 1.8, "O": -1.2},
            gewald=0.2,
            has_bonded_topology=False,
        )


def test_hybrid_combined_table_rejects_incompatible_special_weights_and_topology() -> (
    None
):
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    kwargs = dict(
        species=["Ga", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.2,
        table_points=2000,
        table_filename="ga2o3_core.table",
        table_r_min=0.1,
        charges={"Ga": 1.8, "O": -1.2},
        gewald=0.2,
    )
    with pytest.raises(ValueError, match="identical special_lj and special_coul"):
        build_tabulated_buckingham_core_lines(
            _ga2o3_hybrid_commands() + ["special_bonds amber"],
            has_bonded_topology=False,
            **kwargs,
        )
    with pytest.raises(ValueError, match="bonded topology cannot be represented"):
        build_tabulated_buckingham_core_lines(
            _ga2o3_hybrid_commands(),
            has_bonded_topology=True,
            **kwargs,
        )


def test_autocore_style_inspection_distinguishes_nonbuck_and_fails_closed() -> None:
    from vitriflow.potential import inspect_buckingham_core_compatibility

    supported = inspect_buckingham_core_compatibility(_ga2o3_hybrid_commands())
    assert supported["is_buckingham"] is True
    assert supported["supported"] is True
    assert supported["parsed"]["hybrid"] is True

    unrelated = inspect_buckingham_core_compatibility(
        ["pair_style tersoff", "pair_coeff * * SiC.tersoff Si C"]
    )
    assert unrelated["is_buckingham"] is False
    assert unrelated["supported"] is False

    malformed = list(_ga2o3_hybrid_commands())
    malformed[0] += " lj/cut 8.0"
    with pytest.raises(ValueError, match="supports only one Buckingham substyle"):
        inspect_buckingham_core_compatibility(malformed)


def test_direct_buckingham_rejects_ambiguous_or_inapplicable_kspace_commands() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    common = dict(
        species=["Si"],
        units_style="metal",
        r_in=0.8,
        r_out=1.2,
        table_points=2000,
        table_filename="direct_core.table",
        table_r_min=0.1,
        charges={"Si": 2.4},
        gewald=0.2,
        has_bonded_topology=False,
    )
    long_commands = [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 1000.0 0.3 10.0",
        "kspace_style pppm 1.0e-6",
        "kspace_style ewald 1.0e-6",
    ]
    with pytest.raises(ValueError, match="exactly one effective kspace_style"):
        build_tabulated_buckingham_core_lines(long_commands, **common)

    nonlong_commands = [
        "pair_style buck 10.0",
        "pair_coeff 1 1 1000.0 0.3 10.0",
        "kspace_style pppm 1.0e-6",
    ]
    with pytest.raises(ValueError, match="without coul/long cannot retain"):
        build_tabulated_buckingham_core_lines(nonlong_commands, **common)


def test_table_pair_coeff_preserves_noninteger_cutoff_precision() -> None:
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )

    cutoff = 10.123456789012345
    lines = build_tabulated_buckingham_core_lines(
        [
            f"pair_style buck {cutoff:.17g}",
            "pair_coeff 1 1 1000.0 0.3 10.0",
        ],
        species=["Si"],
        units_style="metal",
        r_in=0.8,
        r_out=1.2,
        table_points=2000,
        table_filename="precise_core.table",
        table_r_min=0.1,
        has_bonded_topology=False,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    assert spec["pairs"][0]["pair_cutoff"] == cutoff
    assert (
        f"pair_coeff 1 1 precise_core.table P1_1 {cutoff:.17g}" in lines
    )
