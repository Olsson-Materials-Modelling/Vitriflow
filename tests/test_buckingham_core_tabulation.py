from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _base_buck_commands() -> list[str]:
    return [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381",
        "pair_coeff 2 2 1388.7730 0.362319 175.0000",
        "pair_modify shift yes",
        "kspace_style pppm 1.0e-5",
    ]


def _charges() -> dict[str, float]:
    return {"Si": 2.4, "O": -1.2}


def _legacy_kim_buck_commands() -> list[str]:
    """Commands emitted by the legacy Carre-Horbach-Ispas KIM model."""

    return [
        "pair_style buck/coul/long 11.0",
        "pair_coeff 1 1 3150.462646 0.3506986443 626.751953",
        "pair_coeff 1 2 27029.419922 0.19385081938 148.099091",
        "pair_coeff 2 2 659.595398 0.3860905475 26.836679",
        "kspace_style pppm 1.0e-5",
        "set type 1 charge 1.910",
        "set type 2 charge -0.955",
    ]


def _tabulated_bks_spec(*, table_points: int = 2000, force_mode: str = "analytic"):
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )

    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=table_points,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    spec["force_mode"] = force_mode
    return spec


def test_validated_table_search_accepts_outdir_alias_but_rejects_internal_symlink(
    tmp_path: Path,
) -> None:
    from vitriflow.potential import _find_validated_tabulated_core_source

    payload = b"authenticated autocore table\n"
    digest = hashlib.sha256(payload).hexdigest()
    spec = {"filename": "core.table", "sha256": digest}

    real_root = tmp_path / "real_result"
    stage = real_root / "production" / "box_001" / "melt"
    table = real_root / "preflight" / "potential_override" / "core.table"
    stage.mkdir(parents=True)
    table.parent.mkdir(parents=True)
    table.write_bytes(payload)
    alias = tmp_path / "scratch_alias"
    try:
        alias.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform policy, not code logic
        pytest.skip(f"symlink creation is unavailable: {exc}")

    # Entering the same canonical result tree through a user-facing alias is
    # valid and common on HPC scratch filesystems.
    found = _find_validated_tabulated_core_source(
        alias / "production" / "box_001" / "melt", spec
    )
    assert found == table.resolve(strict=True)

    escaped_root = tmp_path / "escaped_result"
    escaped_stage = escaped_root / "production" / "box_001" / "melt"
    escaped_stage.mkdir(parents=True)
    outside = tmp_path / "outside" / "potential_override"
    outside.mkdir(parents=True)
    (outside / "core.table").write_bytes(payload)
    (escaped_root / "preflight").symlink_to(
        outside.parent, target_is_directory=True
    )

    # A symlink introduced *inside* the canonical result tree is an escape
    # and remains fail-closed even though the external bytes have the expected
    # digest.
    with pytest.raises(ValueError, match="must not contain symbolic links"):
        _find_validated_tabulated_core_source(escaped_stage, spec)


def test_verified_potential_copy_rejects_bytes_changed_after_source_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vitriflow.potential as module

    source = tmp_path / "source.table"
    destination = tmp_path / "stage" / "core.table"
    source.write_bytes(b"authenticated table bytes\n")
    real_copy = module.shutil.copyfileobj

    def corrupt_copy(src, dst, length=0):
        real_copy(src, dst, length=length)
        dst.write(b"concurrent replacement")

    monkeypatch.setattr(module.shutil, "copyfileobj", corrupt_copy)
    with pytest.raises(ValueError, match="do not match"):
        module._atomic_copy_verified_regular_file(source, destination)
    assert not destination.exists()


def test_source_table_stores_left_limit_at_exact_cutoff_coordinate(tmp_path: Path) -> None:
    import numpy as np

    from vitriflow.potential import (
        _repulsive_regularized_pair_energy_derivatives,
        write_tabulated_buckingham_core_table,
    )
    from vitriflow.workflows.preflight import _parse_pair_table_file

    spec = _tabulated_bks_spec()
    pair = next(pair for pair in spec["pairs"] if pair["pair"] == [1, 1])
    cutoff = float(pair["pair_cutoff"])
    left_radius = np.nextafter(cutoff, float(spec["r_min"]))
    left_u, left_du, _left_d2u = _repulsive_regularized_pair_energy_derivatives(
        np.asarray([left_radius]), pair=pair, units_style="metal"
    )
    exact_u, exact_du, _exact_d2u = _repulsive_regularized_pair_energy_derivatives(
        np.asarray([cutoff]), pair=pair, units_style="metal"
    )
    assert exact_u[0] == 0.0
    assert exact_du[0] == 0.0
    assert left_u[0] != 0.0
    assert left_du[0] != 0.0

    table_path = tmp_path / "buck_core.table"
    write_tabulated_buckingham_core_table(table_path, spec)
    section = _parse_pair_table_file(table_path)[str(pair["section"])]
    assert section["r"][-1] == cutoff
    assert section["energy"][-1] == pytest.approx(left_u[0], rel=5.0e-15)
    assert section["force"][-1] == pytest.approx(-left_du[0], rel=5.0e-15)


def test_lammps_rsq_grid_rejects_collapsed_binary64_coordinates() -> None:
    import numpy as np

    from vitriflow.potential import _lammps_rsq_grid

    lower = 1.0
    upper = float(np.nextafter(lower, np.inf))
    with pytest.raises(ValueError, match="strictly increasing binary64 radii"):
        _lammps_rsq_grid(lower, upper, 3)


def test_table_generation_rejects_unrepresentable_cutoff_left_limit() -> None:
    import numpy as np

    from vitriflow.potential import _tabulated_buckingham_section_arrays

    r_min = 1.0
    pair_cutoff = float(np.nextafter(r_min, np.inf))
    pair = {
        "section": "P1_1",
        "pair_cutoff": pair_cutoff,
        "join_energy_component": "buckingham",
        "join_energy": 0.0,
        "r_out": r_min,
        "A": 0.0,
        "rho": 1.0,
        "C": 0.0,
        "buck_cutoff": pair_cutoff,
        "shift_buck": False,
    }
    with pytest.raises(ValueError, match="no representable interior left-limit"):
        _tabulated_buckingham_section_arrays(
            pair,
            spec={
                "version": 10,
                "kind": "buckingham_zbl_table",
                "pairs": [pair],
                "points": 2,
                "r_min": r_min,
                "units": "metal",
            },
        )


def test_repulsive_join_rejects_unresolved_subgrid_transition_cleanly() -> None:
    from vitriflow.potential import _resolve_repulsive_core_join

    pair = {
        "section": "P1_1",
        "A": 1000.0,
        "rho": 0.2,
        "C": 0.0,
        "buck_cutoff": 10.0,
        "pair_cutoff": 10.0,
        "requested_r_in": 1.0,
        "requested_r_out": 1.0 + 1.0e-10,
        "shift_buck": False,
        "morse_terms": [],
        "coul_mode": None,
    }
    with pytest.raises(ValueError, match="no numerically resolved repulsive transition"):
        _resolve_repulsive_core_join(pair, units_style="metal", r_min=0.1)


def test_quintic_partition_remains_nonnegative_and_monotone_at_outer_join() -> None:
    import numpy as np

    from vitriflow.potential import _quintic_partition_derivatives

    x = np.asarray(
        [
            0.0,
            0.5,
            float(np.nextafter(1.0, 0.0)),
            1.0,
        ],
        dtype=float,
    )
    weight, first, second = _quintic_partition_derivatives(x)

    assert np.all(np.isfinite(weight))
    assert np.all(weight >= 0.0)
    assert np.all(weight <= 1.0)
    assert np.all(first <= 0.0)
    assert weight[0] == 1.0
    assert weight[1] == 0.5
    assert weight[-2] > 0.0
    assert weight[-1] == 0.0
    assert first[0] == first[-1] == 0.0
    assert second[0] == second[-1] == 0.0


def test_analytic_pair_write_reference_is_zero_at_exact_cutoff() -> None:
    from vitriflow.potential import tabulated_buckingham_reference_sections

    spec = _tabulated_bks_spec()
    reference = tabulated_buckingham_reference_sections(spec, npoints=257)
    for pair in spec["pairs"]:
        section = reference[str(pair["section"])]
        assert section["r"][-1] == float(pair["pair_cutoff"])
        assert section["energy"][-1] == 0.0
        assert section["force"][-1] == 0.0


@pytest.mark.parametrize("version", [None, 8, 9, 10.0, 999, True])
def test_analytic_autocore_generation_rejects_non_v10_metadata(version) -> None:
    from vitriflow.potential import tabulated_buckingham_reference_sections

    spec = _tabulated_bks_spec()
    if version is None:
        spec.pop("version")
    else:
        spec["version"] = version

    with pytest.raises(ValueError, match="metadata version 10"):
        tabulated_buckingham_reference_sections(spec, npoints=257)


def test_analytic_autocore_rejects_reinterpreted_join_energy_metadata() -> None:
    from vitriflow.potential import tabulated_buckingham_reference_sections

    spec = _tabulated_bks_spec()
    spec["pairs"][0]["join_energy_component"] = "combined_hamiltonian"
    with pytest.raises(ValueError, match="join_energy_component='buckingham'"):
        tabulated_buckingham_reference_sections(spec, npoints=257)

    spec = _tabulated_bks_spec()
    spec["pairs"][0]["join_energy"] += 1.0e-4
    with pytest.raises(ValueError, match="join_energy is inconsistent"):
        tabulated_buckingham_reference_sections(spec, npoints=257)


def test_pair_write_warning_audit_rejects_stale_diagnostics_with_screen_none(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from vitriflow.runner import RunResult
    from vitriflow.workflows.preflight import _pair_write_potential_curves

    stale_warning = (
        "WARNING: 99 of 100 force values in table STALE "
        "are inconsistent with -dE/dr."
    )
    current_warning = (
        "WARNING: 1 of 2 force values in table CURRENT "
        "are inconsistent with -dE/dr."
    )
    output_name = "realized.table"
    log_name = "log_pairwrite_table.lammps"
    for name in (
        output_name,
        log_name,
        "screen.out",
        "stdout.txt",
        "stderr.txt",
        "log.lammps",
    ):
        (tmp_path / name).write_text(stale_warning + "\n")

    monkeypatch.setattr(
        "vitriflow.workflows.preflight.prepare_potential_files",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._render_pair_write_script",
        lambda *args, **kwargs: "# pair_write test\n",
    )

    class FakeRunner:
        cfg = SimpleNamespace(mpi_cmd=None)

        def run(self, script, workdir, requested_log_name, timeout_sec=None):
            workdir = Path(workdir)
            assert requested_log_name == log_name
            # Every conventional artifact from the prior refinement candidate
            # must be gone before this invocation starts.  An unrelated legacy
            # log is deliberately retained, but warning collection must ignore it.
            for name in (
                output_name,
                log_name,
                "screen.out",
                "stdout.txt",
                "stderr.txt",
            ):
                assert not (workdir / name).exists()
            (workdir / output_name).write_text(
                "CURRENT\nN 2\n\n"
                "1 0.1 1.0 2.0\n"
                "2 1.0 0.0 0.0\n"
            )
            current_log = workdir / requested_log_name
            current_log.write_text(current_warning + "\n")
            return RunResult(
                cmd=["lmp", "-screen", "none"],
                returncode=0,
                stdout="",
                stderr="",
                log_file=current_log,
            )

    config = SimpleNamespace(
        kim=SimpleNamespace(),
        autotune=SimpleNamespace(
            preflight=SimpleNamespace(timeout_sec=1.0),
        ),
    )
    result = _pair_write_potential_curves(
        FakeRunner(),
        config,
        stage_dir=tmp_path,
        potential_lines=[],
        spec={},
        npoints=2,
        output_name=output_name,
        log_name=log_name,
    )

    assert result["warnings"] == [current_warning]
    assert stale_warning not in result["warnings"]


def test_penultimate_inflection_probe_uses_included_cutoff_side(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import numpy as np

    from vitriflow.potential import _lammps_rsq_grid
    from vitriflow.workflows.preflight import _audit_lammps_inflection_warnings

    cutoff = 1.0
    table_path = tmp_path / "source.table"
    # Only the penultimate force is outside both adjacent energy secants, so
    # LAMMPS's predicate flags index N-2.
    knot_radius = _lammps_rsq_grid(0.1, cutoff, 4)
    table_path.write_text(
        "P1_1\nN 4\n\n"
        + "\n".join(
            f"{index} {radius:.17g} 0.0 {force:.1f}"
            for index, (radius, force) in enumerate(
                zip(knot_radius, (0.0, 0.0, 1.0, 0.0)), start=1
            )
        )
        + "\n"
    )
    spec = {
        "points": 4,
        "r_min": 0.1,
        "units": "metal",
        "pairs": [{"section": "P1_1", "pair_cutoff": cutoff}],
    }

    def fake_derivatives(radius, *, pair, units_style):
        radius = np.asarray(radius, dtype=float)
        zeros = np.zeros_like(radius)
        # Reproduce the dangerous topology: the included (left) curvature is
        # strictly positive, while the piecewise function reports zero only at
        # the exact, excluded cutoff.  Sampling that exact endpoint would
        # falsely classify the triplet as bracketing an inflection.
        curvature = np.where(radius == cutoff, 0.0, 1.0)
        return zeros, zeros, curvature

    monkeypatch.setattr(
        "vitriflow.workflows.preflight._repulsive_regularized_pair_energy_derivatives",
        fake_derivatives,
    )
    warning = (
        "WARNING: 1 of 4 force values in table P1_1 "
        "are inconsistent with -dE/dr."
    )
    report = _audit_lammps_inflection_warnings(
        spec=spec,
        table_path=table_path,
        observed_warnings=[warning],
    )

    flagged = report["pairs"]["P1_1"]["flagged_knots"]
    assert [row["index"] for row in flagged] == [2]
    assert flagged[0]["curvature"] == [1.0, 1.0, 1.0]
    assert flagged[0]["brackets_analytic_inflection"] is False
    assert report["passed"] is False
    assert any("non-inflection knot" in item for item in report["blocking_warnings"])


def test_fd_consistent_endpoint_force_and_fprime_use_analytic_one_sided_values() -> None:
    import numpy as np

    from vitriflow.potential import (
        _repulsive_regularized_pair_energy_derivatives,
        _tabulated_buckingham_section_arrays,
    )

    spec = _tabulated_bks_spec(force_mode="fd_consistent")
    pair = next(pair for pair in spec["pairs"] if pair["pair"] == [1, 2])
    cutoff = float(pair["pair_cutoff"])
    endpoints = np.asarray(
        [float(spec["r_min"]), np.nextafter(cutoff, float(spec["r_min"]))]
    )
    _u, du, d2u = _repulsive_regularized_pair_energy_derivatives(
        endpoints, pair=pair, units_style="metal"
    )
    section = _tabulated_buckingham_section_arrays(pair, spec=spec)

    assert section["r"][0] == float(spec["r_min"])
    assert section["r"][-1] == cutoff
    assert section["force"][0] == pytest.approx(-du[0], rel=2.0e-14)
    assert section["force"][-1] == pytest.approx(-du[-1], rel=2.0e-14)
    assert section["fprime_lo"] == pytest.approx(-d2u[0], rel=2.0e-14)
    assert section["fprime_hi"] == pytest.approx(-d2u[-1], rel=2.0e-14)


def test_lammps_force_warnings_are_advisory_only_at_proven_inflections(
    tmp_path: Path,
) -> None:
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
        write_tabulated_buckingham_core_table,
    )
    from vitriflow.workflows.preflight import _audit_lammps_inflection_warnings

    lines = build_tabulated_buckingham_core_lines(
        _legacy_kim_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.6794319849528359,
        r_out=0.8492899811910447,
        table_points=4000,
        table_filename="sio2_bks_core.table",
        table_r_min=0.1,
        charges={"Si": 1.91, "O": -0.955},
        gewald=0.2951741,
        has_bonded_topology=False,
        table_style="spline",
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    table_path = tmp_path / "sio2_bks_core.table"
    write_tabulated_buckingham_core_table(table_path, spec)

    predicted = _audit_lammps_inflection_warnings(
        spec=spec, table_path=table_path, observed_warnings=[]
    )
    warnings = []
    for section, pair_report in predicted["pairs"].items():
        count = int(pair_report["predicted_warning_count"])
        assert pair_report["all_flagged_knots_bracket_analytic_inflections"]
        if count:
            warnings.append(
                f"WARNING: {count} of 4000 force values in table {section} "
                "are inconsistent with -dE/dr."
            )

    audited = _audit_lammps_inflection_warnings(
        spec=spec, table_path=table_path, observed_warnings=warnings
    )
    assert audited["passed"] is True
    assert audited["blocking_warnings"] == []
    assert audited["advisory_warnings"] == warnings

    rejected = _audit_lammps_inflection_warnings(
        spec=spec,
        table_path=table_path,
        observed_warnings=warnings + ["WARNING: unexpected tabulated defect"],
    )
    assert rejected["passed"] is False
    assert rejected["blocking_warnings"] == [
        "WARNING: unexpected tabulated defect"
    ]


def test_spline_table_representation_is_explicit_in_commands_and_metadata() -> None:
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )

    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
        table_style="spline",
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    assert spec["table_style"] == "spline"
    assert "pair_style table spline 3000 pppm" in lines

    with pytest.raises(ValueError, match="interpolation style"):
        build_tabulated_buckingham_core_lines(
            _base_buck_commands(),
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="buck_core.table",
            table_r_min=0.1,
            charges=_charges(),
            gewald=0.224358,
            has_bonded_topology=False,
            table_style="quadratic",
        )


def test_legacy_kim_buckingham_spec_resolves_expected_charges_and_c2_joins() -> None:
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
        update_tabulated_core_metadata_lines,
    )

    lines = build_tabulated_buckingham_core_lines(
        _legacy_kim_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.6794319849528359,
        r_out=0.8492899811910447,
        table_points=4000,
        table_filename="sio2_bks_core.table",
        table_r_min=0.1,
        charges={"Si": 1.91, "O": -0.955},
        gewald=0.2951741,
        has_bonded_topology=False,
        table_style="spline",
    )
    lines = update_tabulated_core_metadata_lines(lines, force_mode="analytic")
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    assert spec["version"] == 10
    assert spec["table_style"] == "spline"
    assert spec["force_mode"] == "analytic"
    assert spec["gewald"] == pytest.approx(0.2951741)

    expected = {
        "P1_1": (1.7207962034497148, 2.0323150632815583, 1.91, 1.91),
        "P1_2": (1.098257446188321, 1.201542663463054, 1.91, -0.955),
        "P2_2": (1.03006591787489, 1.2527694700985093, -0.955, -0.955),
    }
    for pair in spec["pairs"]:
        r_in, r_out, q_i, q_j = expected[str(pair["section"])]
        assert pair["pair_cutoff"] == pytest.approx(11.0)
        assert pair["r_in"] == pytest.approx(r_in, rel=2.0e-14)
        assert pair["r_out"] == pytest.approx(r_out, rel=2.0e-14)
        assert pair["q_i"] == pytest.approx(q_i)
        assert pair["q_j"] == pytest.approx(q_j)
        assert pair["validation"]["c2_join_validated"] is True
        assert pair["validation"]["repulsive_through_r_out"] is True


def test_table_verifier_covers_all_rsq_knots_and_midpoints() -> None:
    from vitriflow.workflows.preflight import (
        _tabulated_realizations,
        _tabulated_verification_points,
    )

    assert _tabulated_verification_points(
        configured_points=50001, table_points=128000
    ) == 255999
    assert _tabulated_verification_points(
        configured_points=600001, table_points=128000
    ) == 767995
    for configured in (2, 50001, 600001, 1_000_000):
        realized = _tabulated_verification_points(
            configured_points=configured, table_points=128000
        )
        assert realized >= configured
        assert (realized - 1) % (2 * (128000 - 1)) == 0
    assert _tabulated_realizations()[0] == ("spline", "analytic")


def test_table_work_check_excludes_only_the_final_hard_cutoff_interval() -> None:
    import numpy as np

    from vitriflow.workflows.preflight import _table_force_energy_consistency_report

    r = np.asarray([1.0, 2.0, 3.0, 4.0])
    force = np.ones_like(r)
    # dU/dr = -F through the interior.  The last value represents the exact
    # hard-cutoff side and may jump because LAMMPS applies the cutoff before
    # evaluating the table at r_c.
    cutoff_jump = {"P": {"r": r, "energy": np.asarray([3.0, 2.0, 1.0, 50.0]), "force": force}}
    report = _table_force_energy_consistency_report(
        cutoff_jump, rel_tol=5.0e-5, abs_tol_frac=1.0e-7
    )
    assert report["passed"] is True
    assert report["pairs"]["P"]["n_intervals_checked"] == 2

    # Moving the same discontinuity one knot into the physical range must not
    # receive the hard-cutoff exemption.
    interior_jump = {"P": {"r": r, "energy": np.asarray([3.0, 2.0, 50.0, 0.0]), "force": force}}
    report = _table_force_energy_consistency_report(
        interior_jump, rel_tol=5.0e-5, abs_tol_frac=1.0e-7
    )
    assert report["passed"] is False
    assert report["pairs"]["P"]["n_fail"] == 1


def test_table_work_check_cannot_report_overall_pass_for_invalid_short_section() -> None:
    import numpy as np

    from vitriflow.workflows.preflight import _table_force_energy_consistency_report

    one_point = {
        "P": {
            "r": np.asarray([1.0]),
            "energy": np.asarray([2.0]),
            "force": np.asarray([3.0]),
        }
    }
    report = _table_force_energy_consistency_report(
        one_point, rel_tol=5.0e-5, abs_tol_frac=1.0e-7
    )
    assert report["passed"] is False
    assert report["pairs"]["P"]["passed"] is False


def test_pair_table_comparison_rejects_corrupted_penultimate_cutoff_value() -> None:
    import numpy as np

    from vitriflow.workflows.preflight import _compare_pair_table_sections

    r = np.linspace(0.1, 3.0, 101)
    energy = 1.0 / r
    force = 1.0 / (r * r)
    # Exact-cutoff semantics are zero; the penultimate point remains on the
    # interacting left branch and must still be verified.
    energy[-1] = 0.0
    force[-1] = 0.0
    reference = {"P": {"r": r, "energy": energy, "force": force}}
    corrupted_energy = energy.copy()
    corrupted_force = force.copy()
    corrupted_energy[-2] += 1.0
    corrupted_force[-2] += 1.0
    realized = {"P": {"r": r, "energy": corrupted_energy, "force": corrupted_force}}
    report = _compare_pair_table_sections(
        reference,
        realized,
        rel_tol=5.0e-5,
        abs_tol_frac=1.0e-7,
    )
    assert report["passed"] is False
    assert report["pairs"]["P"]["n_energy_fail"] >= 1
    assert report["pairs"]["P"]["n_force_fail"] >= 1


def test_build_tabulated_buckingham_core_lines_and_materialize_table(tmp_path: Path) -> None:
    from vitriflow.config import LammpsPotentialConfig
    from vitriflow.potential import build_tabulated_buckingham_core_lines, prepare_potential_files

    pot_cfg = LammpsPotentialConfig(
        interactions=["Si", "O"],
        commands=_base_buck_commands(),
    )
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=4000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )

    joined = "\n".join(lines)
    assert lines[0].startswith("# vitriflow_core_table_begin ")
    assert "pair_style table linear 4000 pppm" in joined
    assert "pair_coeff 1 2 buck_core.table P1_2 10" in joined
    assert "pair_style hybrid/overlay" not in joined
    assert "pair_modify pair coul/long compute no" not in joined
    assert "kspace_modify gewald 0.224358" in joined
    assert "kspace_style pppm 1.0e-5" in joined
    assert "pair_modify shift yes" not in joined

    prepare_potential_files(pot_cfg, tmp_path, lines)
    table_path = tmp_path / "buck_core.table"
    assert table_path.exists()
    text = table_path.read_text()
    # LAMMPS compares the value token after UNITS: literally.  Punctuation on
    # that token (for example ``metal;``) makes an otherwise valid table fail
    # at runtime as a unit mismatch.
    assert text.splitlines()[0] == "# UNITS: metal"
    assert "P1_1" in text
    assert "P1_2" in text
    assert "P2_2" in text
    assert "N 4000 RSQ 0.1 10 FPRIME " in text
    sec = text.split("P1_1", 1)[1].split("P1_2", 1)[0]
    rows = []
    for ln in sec.splitlines():
        toks = ln.split()
        if len(toks) == 4 and toks[0].isdigit():
            rows.append((float(toks[1]), float(toks[2]), float(toks[3])))
    assert rows
    assert any((r > 1.5 and abs(u) > 1.0e-6) for r, u, _f in rows)


def test_kspace_table_rejects_shorter_pair_specific_buckingham_cutoffs() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    commands = [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0 8.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381 9.0",
        "pair_coeff 2 2 1388.7730 0.362319 175.0 10.0",
        "kspace_style pppm 1.0e-5",
    ]
    with pytest.raises(ValueError, match="unrepresentable internal hard cutoff"):
        build_tabulated_buckingham_core_lines(
            commands,
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="core.table",
            table_r_min=0.1,
            charges=_charges(),
            gewald=0.224358,
            has_bonded_topology=False,
        )


def test_kspace_table_rejects_different_global_buckingham_and_coulomb_cutoffs() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    commands = [
        "pair_style buck/coul/long 8.0 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381",
        "pair_coeff 2 2 1388.7730 0.362319 175.0",
        "kspace_style pppm 1.0e-5",
    ]
    with pytest.raises(ValueError, match="must exactly match the common Coulomb/KSpace cutoff"):
        build_tabulated_buckingham_core_lines(
            commands,
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="core.table",
            table_r_min=0.1,
            charges=_charges(),
            gewald=0.224358,
            has_bonded_topology=False,
        )


def test_kspace_table_rejects_buckingham_cutoff_beyond_common_split() -> None:
    import pytest

    from vitriflow.potential import build_tabulated_buckingham_core_lines

    commands = [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381 11.0",
        "pair_coeff 2 2 1388.7730 0.362319 175.0",
        "kspace_style pppm 1.0e-5",
    ]
    with pytest.raises(ValueError, match="cannot preserve the KSpace split"):
        build_tabulated_buckingham_core_lines(
            commands,
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="core.table",
            table_r_min=0.1,
            charges=_charges(),
            gewald=0.224358,
        has_bonded_topology=False,
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("pair_modify tail yes", "cannot preserve pair_modify tail yes"),
        ("dielectric 2.0", "requires dielectric 1.0"),
    ],
)
def test_table_conversion_rejects_unrepresented_global_corrections(
    extra: str, message: str
) -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    with pytest.raises(ValueError, match=message):
        build_tabulated_buckingham_core_lines(
            _base_buck_commands() + [extra],
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="core.table",
            table_r_min=0.1,
            charges=_charges(),
            gewald=0.224358,
        has_bonded_topology=False,
        )


def test_table_conversion_accepts_and_omits_default_tail_and_dielectric() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands() + ["pair_modify tail no", "dielectric 1.0"],
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )
    assert not any("tail" in line for line in lines)
    assert not any(line.startswith("dielectric") for line in lines)


@pytest.mark.parametrize(
    "commands",
    [
        ["special_bonds amber"],
        ["special_bonds lj 0 0 0.5 coul 0 0 0.833333333333"],
        # Each command resets both arrays; the final lj-only command therefore
        # leaves Coulomb at zero rather than carrying over the prior value.
        ["special_bonds coul 0 0 0.5", "special_bonds lj 0 0 0.5"],
    ],
)
def test_combined_table_rejects_unequal_special_lj_and_coul_weights(commands) -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    with pytest.raises(ValueError, match="requires identical special_lj and special_coul"):
        build_tabulated_buckingham_core_lines(
            _base_buck_commands() + list(commands),
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="core.table",
            table_r_min=0.1,
            charges=_charges(),
            gewald=0.224358,
        has_bonded_topology=False,
        )


@pytest.mark.parametrize(
    ("command", "weights"),
    [
        ("special_bonds default", [0.0, 0.0, 0.0]),
        ("special_bonds charmm", [0.0, 0.0, 0.0]),
        ("special_bonds dreiding", [0.0, 0.0, 1.0]),
        ("special_bonds fene", [0.0, 1.0, 1.0]),
        ("special_bonds lj/coul 0.1 0.2 0.3 angle yes", [0.1, 0.2, 0.3]),
    ],
)
def test_combined_table_accepts_equal_special_bonds_weights(command, weights) -> None:
    from vitriflow.potential import _parse_tabulated_core_spec, build_tabulated_buckingham_core_lines

    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands() + [command],
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    assert spec["special_lj"] == pytest.approx(weights)
    assert spec["special_coul"] == pytest.approx(weights)


def test_coul_long_requires_known_topology_and_full_weights_when_bonded() -> None:
    from vitriflow.potential import _parse_tabulated_core_spec, build_tabulated_buckingham_core_lines

    kwargs = dict(
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
    )
    with pytest.raises(ValueError, match="explicit bonded-topology determination"):
        build_tabulated_buckingham_core_lines(_base_buck_commands(), **kwargs)
    with pytest.raises(ValueError, match="KSpace .* bonded correction is missing"):
        build_tabulated_buckingham_core_lines(
            _base_buck_commands(), has_bonded_topology=True, **kwargs
        )
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands() + ["special_bonds lj/coul 1 1 1"],
        has_bonded_topology=True,
        **kwargs,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    assert spec["has_bonded_topology"] is True
    assert spec["special_lj"] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(("count_line", "expected"), [("0 bonds", False), ("2 bonds", True)])
def test_preflight_reads_bonded_topology_from_data_header(
    tmp_path: Path, count_line: str, expected: bool
) -> None:
    from vitriflow.workflows.preflight import _datafile_has_bonded_topology

    data = tmp_path / "input.data"
    data.write_text(
        "LAMMPS data\n\n2 atoms\n"
        f"{count_line}\n"
        "1 atom types\n\n0 2 xlo xhi\n0 2 ylo yhi\n0 2 zlo zhi\n\nAtoms\n"
    )
    assert _datafile_has_bonded_topology(data) is expected


def test_buckingham_zbl_replacement_is_repulsive_and_c2_at_both_joins() -> None:
    import numpy as np
    import pytest

    from vitriflow.potential import (
        _buckingham_energy_derivatives,
        _parse_tabulated_core_spec,
        _regularized_buckingham_component_energy_derivatives,
        _repulsive_regularized_total_energy_derivatives,
        _zbl_base_energy_derivatives,
        build_tabulated_buckingham_core_lines,
    )

    spec = _parse_tabulated_core_spec(build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=4000,
        table_filename="core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    ))
    assert spec is not None
    assert spec["preserved_components"] == [
        "full_coulomb_via_real_plus_kspace"
    ]
    assert spec["morse_policy"] == "not_present"
    assert all(
        pair["validation"]["morse_policy"] == "not_present"
        for pair in spec["pairs"]
    )
    pair = next(p for p in spec["pairs"] if p["pair"] == [1, 2])
    assert pair["r_out"] > pair["requested_r_out"]
    assert pair["validation"]["c2_join_validated"] is True

    # The actual emitted path approaches shifted ZBL (+infinity) at short
    # range and is repulsive throughout the resolved core interval.
    radii = np.geomspace(1.0e-5, float(pair["r_out"]), 20001)
    u, du, d2u = _repulsive_regularized_total_energy_derivatives(
        radii, pair=pair, units_style="metal"
    )
    assert np.all(np.isfinite(u))
    assert u[0] > 1.0e6 and u[0] > u[-1]
    assert u[0] > u[1] > u[2]
    assert np.all(du < 0.0)
    assert np.all(np.isfinite(d2u))

    joins = np.asarray([pair["r_in"], pair["r_out"]], dtype=float)
    uj, duj, d2uj = _regularized_buckingham_component_energy_derivatives(
        joins, pair=pair, units_style="metal"
    )
    uz, duz, d2uz = _zbl_base_energy_derivatives(
        joins[:1], z_i=14, z_j=8, units_style="metal"
    )
    ub, dub, d2ub = _buckingham_energy_derivatives(
        joins[1:],
        A=pair["A"],
        rho=pair["rho"],
        C=pair["C"],
        cutoff=pair["buck_cutoff"],
        shift=pair["shift_buck"],
    )
    # The inner energy differs from bare ZBL by the constant needed to match
    # the outer base energy; force and curvature are identical.
    assert duj[0] == pytest.approx(duz[0], rel=2.0e-14)
    assert d2uj[0] == pytest.approx(d2uz[0], rel=2.0e-14)
    assert uj[1] == pytest.approx(ub[0], rel=2.0e-14)
    assert duj[1] == pytest.approx(dub[0], rel=2.0e-14)
    assert d2uj[1] == pytest.approx(d2ub[0], rel=2.0e-14)


def test_nacl_long_range_autocore_is_ewald_split_invariant_and_total_repulsive() -> None:
    import numpy as np

    from vitriflow.potential import (
        _pair_coulomb_energy_derivatives,
        _parse_tabulated_core_spec,
        _repulsive_regularized_pair_energy_derivatives,
        _repulsive_regularized_total_energy_derivatives,
        build_tabulated_buckingham_core_lines,
    )

    commands = [
        "pair_style buck/coul/long 10.0",
        "pair_coeff * * 10000.0 0.2 0.0",
        "kspace_style pppm 1.0e-6",
    ]

    def build(gewald: float) -> dict:
        lines = build_tabulated_buckingham_core_lines(
            commands,
            species=["Na", "Cl"],
            units_style="metal",
            r_in=0.8,
            r_out=1.2,
            table_points=2000,
            table_filename="nacl_core.table",
            table_r_min=0.1,
            charges={"Na": 1.0, "Cl": -1.0},
            gewald=gewald,
            has_bonded_topology=False,
        )
        spec = _parse_tabulated_core_spec(lines)
        assert spec is not None
        return spec

    low_g = build(0.12)
    high_g = build(0.45)
    assert low_g["regularized_components"] == ["buckingham"]
    assert low_g["ewald_split_invariant_regularization"] is True

    for pair_low, pair_high in zip(low_g["pairs"], high_g["pairs"]):
        # Join selection uses full Coulomb, so changing only the numerical
        # Ewald partition cannot move either endpoint.
        assert pair_low["r_in"] == pair_high["r_in"]
        assert pair_low["r_out"] == pair_high["r_out"]
        assert pair_low["join_energy"] == pair_high["join_energy"]
        assert pair_low["join_resolution"]["join_selection_ewald_split_invariant"] is True

        radius = np.unique(
            np.concatenate(
                (
                    np.geomspace(0.1, float(pair_low["r_out"]), 1001),
                    np.linspace(
                        float(pair_low["r_in"]),
                        float(pair_low["pair_cutoff"]) * 0.999,
                        1001,
                    ),
                )
            )
        )
        total_low = _repulsive_regularized_total_energy_derivatives(
            radius, pair=pair_low, units_style="metal"
        )
        total_high = _repulsive_regularized_total_energy_derivatives(
            radius, pair=pair_high, units_style="metal"
        )
        for left, right in zip(total_low, total_high):
            assert np.array_equal(left, right)

        # The emitted table differs with G only by the real-space partition.
        # Adding the exact complementary erf contribution reconstructs the
        # same total energy, force, and curvature over the whole domain.
        for pair, total in ((pair_low, total_low), (pair_high, total_high)):
            runtime = _repulsive_regularized_pair_energy_derivatives(
                radius, pair=pair, units_style="metal"
            )
            full_coul = _pair_coulomb_energy_derivatives(
                radius,
                pair=pair,
                units_style="metal",
                representation="full",
            )
            real_coul = _pair_coulomb_energy_derivatives(
                radius,
                pair=pair,
                units_style="metal",
                representation="runtime",
            )
            for realized, full, real, target in zip(
                runtime, full_coul, real_coul, total
            ):
                assert np.allclose(
                    realized + full - real,
                    target,
                    rtol=2.0e-14,
                    atol=2.0e-12,
                )

            core = radius <= float(pair["r_out"])
            assert float(np.min(-total[1][core])) > 0.0
            assert pair["validation"]["repulsive_through_r_out"] is True


def test_unphysical_coulomb_without_repulsive_base_branch_fails_closed() -> None:
    import pytest

    from vitriflow.potential import (
        build_tabulated_buckingham_core_lines,
    )

    cmds = [
        "pair_style buck/coul/cut 10.0",
        "pair_coeff 1 1 1000.0 0.2 100.0",
        "pair_coeff 1 2 1000.0 0.2 100.0",
        "pair_coeff 2 2 1000.0 0.2 100.0",
    ]
    with pytest.raises(ValueError, match="no resolved repulsive base branch"):
        build_tabulated_buckingham_core_lines(
            cmds,
            species=["Na", "Cl"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=2000,
            table_filename="core.table",
            table_r_min=1.0e-4,
            # Deliberately unphysical charges erase the base potential's
            # repulsive branch.  Such a model cannot be joined smoothly while
            # preserving its long-range physics, so table generation rejects it.
            charges={"Na": 1.0e6, "Cl": -1.0e6},
        )


def test_core_repulsion_tabulate_requires_zbl_style() -> None:
    from pydantic import ValidationError

    from vitriflow.config import RunConfig

    try:
        RunConfig.model_validate(
            {
                "potential": {
                    "kind": "lammps",
                    "user_units": "metal",
                    "interactions": ["Si", "O"],
                    "commands": _base_buck_commands(),
                    "core_repulsion": {
                        "enabled": True,
                        "style": "lj_repulsive",
                        "tabulate": True,
                    },
                },
                "structure": {"generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1}},
            }
        )
    except ValidationError as exc:
        assert "tabulate supports only style='zbl'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected validation failure for lj_repulsive tabulation")


def test_enabled_zbl_validates_consumed_table_controls_when_legacy_flag_is_false() -> None:
    from pydantic import ValidationError

    from vitriflow.config import CoreRepulsionConfig

    with pytest.raises(ValidationError, match="table_points must be >= 2000"):
        CoreRepulsionConfig.model_validate(
            {
                "enabled": True,
                "style": "zbl",
                "tabulate": False,
                "table_points": 1999,
            }
        )

    # The same dormant controls are irrelevant while autocore is disabled.
    disabled = CoreRepulsionConfig.model_validate(
        {
            "enabled": False,
            "style": "zbl",
            "tabulate": False,
            "table_points": 1999,
        }
    )
    assert disabled.table_points == 1999


def test_preflight_core_repulsion_tabulate_returns_table_override(monkeypatch, tmp_path: Path) -> None:
    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.potential import _parse_tabulated_core_spec
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.preflight import _maybe_apply_core_repulsion

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_points": 3000,
                    "table_points_max": 3000,
                    "table_filename": "buck_zbl.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                    "dt_candidates": [0.0005],
                    "r_out_factor": 0.5,
                    "r_out_min": 0.6,
                    "r_out_max": 1.6,
                    "r_in_factor": 0.8,
                    "test_run_steps": 10,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge", "timestep": 0.0005},
            "autotune": {
                "preflight": {"enabled": True},
                "tm_scan": {"equil_steps": 10, "sample_steps": 10},
            },
        }
    )

    input_data = tmp_path / "input.data"
    _write_two_atom_charge_data(input_data, units="metal")

    calls: list[str] = []

    monkeypatch.setattr("vitriflow.workflows.preflight._read_nn_median_from_datafile", lambda *a, **k: 2.0)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._probe_original_potential_gewald",
        lambda *a, **k: 0.224358,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._audit_original_potential_above_core_joins",
        lambda *a, **k: {"passed": True, "comparison": {"overall": {}}},
    )

    def _fake_stability_test(runner, config, input_data, *, outdir, potential_lines, temperature, timestep, label=""):
        text = "\n".join(str(x) for x in potential_lines)
        calls.append(text)
        return True

    def _fake_pair_write(*args, **kwargs):
        import numpy as np

        spec = kwargs["spec"]
        sections = {}
        for pair in spec["pairs"]:
            sections[pair["section"]] = {
                "r": np.asarray([0.1, float(pair["pair_cutoff"])], dtype=float),
                "energy": np.asarray([1.0, 0.0], dtype=float),
                "force": np.asarray([1.0, 0.0], dtype=float),
            }
        return {"path": Path(kwargs["stage_dir"]) / kwargs["output_name"], "sections": sections, "warnings": []}

    monkeypatch.setattr("vitriflow.workflows.preflight._run_stability_test", _fake_stability_test)
    monkeypatch.setattr("vitriflow.workflows.preflight._pair_write_potential_curves", _fake_pair_write)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._verify_tabulated_core_against_source",
        lambda *a, **k: {"passed": True, "warnings": [], "comparison": {"overall": {"max_energy_ratio": 0.0, "max_force_ratio": 0.0}}},
    )

    potential_lines, core_res, dt_sel = _maybe_apply_core_repulsion(
        LammpsRunner(LammpsConfig()),
        cfg,
        input_data,
        tmp_path,
        T_test=4000.0,
    )

    assert dt_sel == 0.0005
    assert potential_lines is not None
    joined = "\n".join(potential_lines)
    spec = _parse_tabulated_core_spec(potential_lines)
    assert spec is not None
    assert len(calls) == 1
    assert sum(
        line.strip().startswith("pair_style ") for line in calls[0].splitlines()
    ) == 1
    assert "pair_style table spline 3000 pppm" in calls[0]
    assert "hybrid/overlay" not in calls[0]
    assert joined.startswith("# vitriflow_core_table_begin ")
    assert "pair_coeff 1 2 buck_zbl.table P1_2 10" in joined
    assert "pair_style hybrid/overlay" not in joined
    assert "pair_modify pair coul/long compute no" not in joined
    assert "kspace_modify gewald 0.224358" in joined
    assert spec.get("generated_by") == "vitriflow_generated"
    assert spec.get("force_mode") == "analytic"
    assert spec.get("table_style") == "spline"
    assert bool(spec.get("include_fprime", False)) is True
    assert isinstance(spec.get("sha256"), str) and len(spec["sha256"]) == 64
    assert core_res.success is True
    assert "accepted after refinement" in core_res.note
    assert core_res.r_inner == pytest.approx(0.8)
    assert core_res.r_outer == pytest.approx(1.0)
    assert core_res.r_inner_r_outer_role == "global_calibration_request"
    assert core_res.requested_r_inner == pytest.approx(core_res.r_inner)
    assert core_res.requested_r_outer == pytest.approx(core_res.r_outer)
    assert core_res.join_radii_lammps_units_style == "metal"
    assert len(core_res.resolved_pair_joins) == 3
    reported_by_pair = {
        tuple(row["pair"]): row for row in core_res.resolved_pair_joins
    }
    metadata_by_pair = {tuple(pair["pair"]): pair for pair in spec["pairs"]}
    assert set(reported_by_pair) == set(metadata_by_pair)
    for pair, row in reported_by_pair.items():
        metadata = metadata_by_pair[pair]
        assert row["species"] == metadata["species"]
        assert row["requested_r_in"] == pytest.approx(metadata["requested_r_in"])
        assert row["requested_r_out"] == pytest.approx(metadata["requested_r_out"])
        assert row["resolved_r_in"] == pytest.approx(metadata["r_in"])
        assert row["resolved_r_out"] == pytest.approx(metadata["r_out"])
        assert row["resolver_adjusted"] is (
            not (
                metadata["r_in"] == pytest.approx(metadata["requested_r_in"])
                and metadata["r_out"] == pytest.approx(metadata["requested_r_out"])
            )
        )
    assert any(row["resolver_adjusted"] for row in core_res.resolved_pair_joins)


def test_preflight_kim_fixed_type_charges_drive_table_without_structure_charges(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.preflight import _maybe_apply_core_repulsion

    cfg = RunConfig.model_validate(
        {
            "kim": {
                "model": "Sim_LAMMPS_Buckingham_legacy_fixture__SM_000000000000_000",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "table_points": 3000,
                    "table_points_max": 3000,
                    "table_filename": "buck_zbl.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                    "dt_candidates": [0.0005],
                    "r_out_factor": 0.5,
                    "r_out_min": 0.6,
                    "r_out_max": 1.6,
                    "r_in_factor": 0.8,
                    "test_run_steps": 10,
                },
            },
            "structure": {
                "generate": {
                    "method": "random",
                    "formula": "SiO2",
                    "n_formula_units": 1,
                },
            },
            "md": {"atom_style": "charge", "timestep": 0.0005},
            "autotune": {
                "preflight": {"enabled": True},
                "tm_scan": {"equil_steps": 10, "sample_steps": 10},
            },
        }
    )
    assert cfg.structure.charges is None

    input_data = tmp_path / "input.data"
    _write_two_atom_charge_data(
        input_data,
        units="metal",
        q_si_e=0.0,
        q_o_e=0.0,
    )
    kim_commands = _base_buck_commands() + [
        "set type 1 charge 2.4",
        "set type 2 charge -1.2",
    ]
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._kim_extract_commands",
        lambda *args, **kwargs: list(kim_commands),
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._read_nn_median_from_datafile",
        lambda *args, **kwargs: 2.0,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._probe_original_potential_gewald",
        lambda *args, **kwargs: 0.224358,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._audit_original_potential_above_core_joins",
        lambda *args, **kwargs: {"passed": True, "comparison": {"overall": {}}},
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._run_stability_test",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._verify_tabulated_core_against_source",
        lambda *args, **kwargs: {
            "passed": True,
            "warnings": [],
            "comparison": {
                "overall": {"max_energy_ratio": 0.0, "max_force_ratio": 0.0}
            },
            "self_consistency": {"overall": {"max_force_ratio": 0.0}},
        },
    )

    builder_charge_calls: list[dict[str, float]] = []

    def _recording_builder(*args, **kwargs):
        builder_charge_calls.append(dict(kwargs["charges"]))
        return build_tabulated_buckingham_core_lines(*args, **kwargs)

    monkeypatch.setattr(
        "vitriflow.workflows.preflight.build_tabulated_buckingham_core_lines",
        _recording_builder,
    )

    potential_lines, core_result, timestep = _maybe_apply_core_repulsion(
        LammpsRunner(LammpsConfig()),
        cfg,
        input_data,
        tmp_path,
        T_test=4000.0,
    )

    assert core_result.success is True
    assert timestep == 0.0005
    assert potential_lines is not None
    assert len(builder_charge_calls) >= 2
    assert all(charges == pytest.approx(_charges()) for charges in builder_charge_calls)
    assert "set type 1 charge 2.4" in potential_lines
    assert "set type 2 charge -1.2" in potential_lines

    spec = _parse_tabulated_core_spec(potential_lines)
    assert spec is not None
    audit = spec["runtime_charge_audit"]
    assert audit["configured_charges_e"] is None
    assert audit["effective_charges_e"] == pytest.approx(_charges())
    assert audit["input_charges_overwritten_for_species"] == ["Si", "O"]
    assert all(pair["q_i"] is not None and pair["q_j"] is not None for pair in spec["pairs"])


def test_fixed_kim_charge_commands_render_after_data_and_replication_before_run() -> None:
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import StageSpec, render_stage

    stage = StageSpec(
        name="kim_charge_order",
        input_data=Path("input.data"),
        output_data=Path("output.data"),
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=1,
        run_steps=1,
        seed=12345,
        replicate=(2, 2, 2),
        write_dump=False,
        potential_lines=_base_buck_commands()
        + ["set type 1 charge 2.4", "set type 2 charge -1.2"],
    )
    script = render_stage(
        KimConfig(model="legacy_fixture", interactions=["Si", "O"]),
        MDConfig(atom_style="charge", ensemble="nvt", timestep=0.0005),
        stage,
    )

    assert script.index("read_data input.data") < script.index("replicate 2 2 2")
    assert script.index("replicate 2 2 2") < script.index("set type 1 charge 2.4")
    assert script.index("set type 1 charge 2.4") < script.index("set type 2 charge -1.2")
    assert script.index("set type 2 charge -1.2") < script.index("run 1")


def test_core_repulsion_result_additive_join_fields_have_compatible_defaults() -> None:
    from vitriflow.workflows.preflight import CoreRepulsionResult

    result = CoreRepulsionResult(
        enabled=False,
        applied=False,
        style="zbl",
        base_pair_style=None,
        r_inner=None,
        r_outer=None,
        attempts=0,
        success=True,
        note="disabled",
    )
    assert result.r_inner_r_outer_role == "not_applicable"
    assert result.requested_r_inner is None
    assert result.requested_r_outer is None
    assert result.join_radii_lammps_units_style is None
    assert result.resolved_pair_joins == ()


def test_analytic_core_execution_lines_label_legacy_radii_as_applied_join() -> None:
    from vitriflow.workflows.preflight import (
        _core_join_report_from_execution_lines,
    )

    role, pair_joins = _core_join_report_from_execution_lines(
        [
            "pair_style hybrid/overlay buck/coul/long 10.0 zbl 0.8 1.0",
            "pair_coeff * * buck/coul/long 0.0 1.0 0.0",
            "pair_coeff * * zbl 14 8",
        ]
    )
    assert role == "applied_global_join"
    assert pair_joins == ()




def test_preflight_core_repulsion_rejects_unbounded_analytic_fallback(monkeypatch, tmp_path: Path) -> None:
    import pytest

    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.preflight import PreflightError, _maybe_apply_core_repulsion

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_points": 3000,
                    "table_points_max": 6000,
                    "table_filename": "buck_zbl.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                    "dt_candidates": [0.0005],
                    "r_out_factor": 0.5,
                    "r_out_min": 0.6,
                    "r_out_max": 1.6,
                    "r_in_factor": 0.8,
                    "test_run_steps": 10,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge", "timestep": 0.0005},
            "autotune": {
                "preflight": {"enabled": True},
                "tm_scan": {"equil_steps": 10, "sample_steps": 10},
            },
        }
    )

    input_data = tmp_path / "input.data"
    _write_two_atom_charge_data(input_data, units="metal")

    monkeypatch.setattr("vitriflow.workflows.preflight._read_nn_median_from_datafile", lambda *a, **k: 2.0)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._probe_original_potential_gewald",
        lambda *a, **k: 0.224358,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._audit_original_potential_above_core_joins",
        lambda *a, **k: {"passed": True, "comparison": {"overall": {}}},
    )

    def _fake_stability_test(runner, config, input_data, *, outdir, potential_lines, temperature, timestep, label=""):
        return "pair_style hybrid/overlay" in "\n".join(str(x) for x in potential_lines)

    def _fake_pair_write(*args, **kwargs):
        import numpy as np

        spec = kwargs["spec"]
        sections = {}
        for pair in spec["pairs"]:
            sections[pair["section"]] = {
                "r": np.asarray([0.1, float(pair["pair_cutoff"])], dtype=float),
                "energy": np.asarray([1.0, 0.0], dtype=float),
                "force": np.asarray([1.0, 0.0], dtype=float),
            }
        return {"path": Path(kwargs["stage_dir"]) / kwargs["output_name"], "sections": sections, "warnings": []}

    monkeypatch.setattr("vitriflow.workflows.preflight._run_stability_test", _fake_stability_test)
    monkeypatch.setattr("vitriflow.workflows.preflight._pair_write_potential_curves", _fake_pair_write)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._verify_tabulated_core_against_source",
        lambda *a, **k: {
            "passed": False,
            "warnings": ["WARNING: force values in table are inconsistent with -dE/dr."],
            "comparison": {"overall": {"max_energy_ratio": 9.0, "max_force_ratio": 11.0}},
            "self_consistency": {"overall": {"max_force_ratio": 25.0}},
        },
    )

    with pytest.raises(PreflightError, match="refusing the unbounded analytic"):
        _maybe_apply_core_repulsion(
            LammpsRunner(LammpsConfig()),
            cfg,
            input_data,
            tmp_path,
            T_test=4000.0,
        )

    summary_txt = tmp_path / "preflight" / "table_verify" / "refinement_summary.txt"
    summary_json = tmp_path / "preflight" / "table_verify" / "refinement_report.json"
    assert summary_txt.exists()
    assert summary_json.exists()
    summary_text = summary_txt.read_text()
    assert "status: rejected_fail_closed" in summary_text
    assert "fallback_to_analytic: False" in summary_text
    assert "verify=fail, stability=not_run" in summary_text

    import json

    report = json.loads(summary_json.read_text())
    assert report["candidates"]
    assert all(candidate["verify_passed"] is False for candidate in report["candidates"])
    assert all(candidate["stability_ok"] is None for candidate in report["candidates"])
    assert all(
        candidate["stability_status"] == "not_run"
        for candidate in report["candidates"]
    )


def test_preflight_reports_attempted_table_stability_failure_as_fail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import json

    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.preflight import PreflightError, _maybe_apply_core_repulsion

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_points": 3000,
                    "table_points_max": 3000,
                    "table_filename": "buck_zbl.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                    "dt_candidates": [0.0005],
                    "r_out_factor": 0.5,
                    "r_out_min": 0.6,
                    "r_out_max": 1.6,
                    "r_in_factor": 0.8,
                    "test_run_steps": 10,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {
                    "method": "random",
                    "formula": "SiO2",
                    "n_formula_units": 1,
                },
            },
            "md": {"atom_style": "charge", "timestep": 0.0005},
            "autotune": {
                "preflight": {"enabled": True},
                "tm_scan": {"equil_steps": 10, "sample_steps": 10},
            },
        }
    )

    input_data = tmp_path / "input.data"
    _write_two_atom_charge_data(input_data, units="metal")
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._read_nn_median_from_datafile",
        lambda *a, **k: 2.0,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._probe_original_potential_gewald",
        lambda *a, **k: 0.224358,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._audit_original_potential_above_core_joins",
        lambda *a, **k: {"passed": True, "comparison": {"overall": {}}},
    )

    stability_calls: list[bool] = []

    def _fake_stability_test(
        runner,
        config,
        input_data,
        *,
        outdir,
        potential_lines,
        temperature,
        timestep,
        label="",
    ):
        is_table = "pair_style table" in "\n".join(str(x) for x in potential_lines)
        stability_calls.append(is_table)
        return not is_table

    monkeypatch.setattr(
        "vitriflow.workflows.preflight._run_stability_test",
        _fake_stability_test,
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._verify_tabulated_core_against_source",
        lambda *a, **k: {
            "passed": True,
            "warnings": [],
            "comparison": {
                "overall": {"max_energy_ratio": 0.0, "max_force_ratio": 0.0}
            },
            "self_consistency": {"overall": {"max_force_ratio": 0.0}},
        },
    )

    with pytest.raises(PreflightError, match="passed strict energy/force verification and stability"):
        _maybe_apply_core_repulsion(
            LammpsRunner(LammpsConfig()),
            cfg,
            input_data,
            tmp_path,
            T_test=4000.0,
        )

    # Every stability call uses a pair-write-verified bounded table.  The
    # unbounded analytic overlay is never integrated.
    from vitriflow.workflows.preflight import _tabulated_realizations

    assert stability_calls == [True] * len(_tabulated_realizations())
    summary_path = tmp_path / "preflight" / "table_verify" / "refinement_summary.txt"
    report_path = tmp_path / "preflight" / "table_verify" / "refinement_report.json"
    summary_text = summary_path.read_text()
    assert "verify=pass, stability=fail" in summary_text
    assert "stability=not_run" not in summary_text

    report = json.loads(report_path.read_text())
    assert report["candidates"]
    assert all(candidate["verify_passed"] is True for candidate in report["candidates"])
    assert all(candidate["stability_ok"] is False for candidate in report["candidates"])
    assert all(
        candidate["stability_status"] == "fail"
        for candidate in report["candidates"]
    )

def test_build_tabulated_buckingham_core_lines_assigns_coul_cut_substyle() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    cmds = [
        "pair_style buck/coul/cut 8.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381",
        "pair_coeff 2 2 1388.7730 0.362319 175.0000",
    ]

    lines = build_tabulated_buckingham_core_lines(
        cmds,
        species=["Na", "Cl"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges={"Na": 2.4, "Cl": -1.2},
    )

    joined = "\n".join(lines)
    assert "pair_style table linear 3000" in joined
    assert "pair_style hybrid/overlay" not in joined
    assert "pair_coeff * * coul/cut" not in joined


@pytest.mark.parametrize(
    "commands",
    [
        [
            "pair_style buck/coul/cut 8.0 10.0",
            "pair_coeff 1 1 0.0 1.0 0.0",
            "pair_coeff 1 2 18003.7572 0.205205 133.5381",
            "pair_coeff 2 2 1388.7730 0.362319 175.0",
        ],
        [
            "pair_style buck/coul/cut 10.0",
            "pair_coeff 1 1 0.0 1.0 0.0 8.0 10.0",
            "pair_coeff 1 2 18003.7572 0.205205 133.5381",
            "pair_coeff 2 2 1388.7730 0.362319 175.0",
        ],
    ],
)
def test_coul_cut_table_rejects_internal_component_cutoff(
    commands: list[str],
) -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    with pytest.raises(ValueError, match="unrepresentable internal hard cutoff"):
        build_tabulated_buckingham_core_lines(
            commands,
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="buck_core.table",
            table_r_min=0.1,
            charges=_charges(),
        )


def test_pure_buck_table_rejects_core_join_at_or_beyond_cutoff() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    with pytest.raises(ValueError, match="strictly below every component cutoff"):
        build_tabulated_buckingham_core_lines(
            [
                "pair_style buck 1.0",
                "pair_coeff 1 1 1000.0 0.3 10.0",
            ],
            species=["Si"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="buck_core.table",
            table_r_min=0.1,
        )


def test_build_tabulated_buckingham_core_lines_requires_fixed_charges_for_coulombic_styles() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    try:
        build_tabulated_buckingham_core_lines(
            _base_buck_commands(),
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="buck_core.table",
            table_r_min=0.1,
            gewald=0.224358,
        has_bonded_topology=False,
        )
    except ValueError as exc:
        assert "requires fixed species charges" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing-charge failure for tabulated buck/coul/long")


@pytest.mark.parametrize(
    ("commands", "message"),
    [
        (
            _base_buck_commands() + ["pair_style buck/coul/long 10.0"],
            "exactly one effective pair_style",
        ),
        (
            ["pair_style buck/coul/long 10.0 10.0 surplus"]
            + _base_buck_commands()[1:],
            "exactly one or two cutoff arguments",
        ),
        (
            [
                _base_buck_commands()[0],
                "pair_coeff 1 1 0.0 1.0 0.0 10.0 surplus",
                *_base_buck_commands()[2:],
            ],
            "surplus Buckingham pair_coeff arguments",
        ),
    ],
)
def test_buckingham_command_parser_rejects_ambiguous_or_surplus_arguments(
    commands: list[str], message: str
) -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    with pytest.raises(ValueError, match=message):
        build_tabulated_buckingham_core_lines(
            commands,
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="core.table",
            table_r_min=0.1,
            charges=_charges(),
            gewald=0.224358,
            has_bonded_topology=False,
        )


def _write_two_atom_charge_data(
    path: Path,
    *,
    units: str,
    q_si_e: float = 2.4,
    q_o_e: float = -1.2,
    atoms_style: str = "charge",
    omit_second_charge: bool = False,
) -> None:
    from vitriflow.lammps_units import (
        charge_from_elementary_factor,
        length_from_angstrom_factor,
        mass_from_amu_factor,
    )

    lf = length_from_angstrom_factor(units)
    mf = mass_from_amu_factor(units)
    qf = charge_from_elementary_factor(units)
    if atoms_style == "charge":
        first = f"1 1 {q_si_e * qf:.16g} {1.0 * lf:.16g} {1.0 * lf:.16g} {1.0 * lf:.16g}"
        second = (
            f"2 2 {q_o_e * qf:.16g} {2.0 * lf:.16g} {2.0 * lf:.16g} {2.0 * lf:.16g}"
            if not omit_second_charge
            else f"2 2 {2.0 * lf:.16g} {2.0 * lf:.16g} {2.0 * lf:.16g}"
        )
    else:
        first = f"1 1 {1.0 * lf:.16g} {1.0 * lf:.16g} {1.0 * lf:.16g}"
        second = f"2 2 {2.0 * lf:.16g} {2.0 * lf:.16g} {2.0 * lf:.16g}"
    path.write_text(
        "charge audit fixture\n\n"
        "2 atoms\n2 atom types\n\n"
        f"0 {10.0 * lf:.16g} xlo xhi\n"
        f"0 {10.0 * lf:.16g} ylo yhi\n"
        f"0 {10.0 * lf:.16g} zlo zhi\n\n"
        "Masses\n\n"
        f"1 {28.085 * mf:.16g}\n2 {15.999 * mf:.16g}\n\n"
        f"Atoms # {atoms_style}\n\n{first}\n{second}\n"
    )


def test_tabulated_coulomb_runtime_charge_audit_accepts_valid_si_units(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _validate_tabulated_coulomb_runtime_charges

    data = tmp_path / "si.data"
    _write_two_atom_charge_data(data, units="si")
    report = _validate_tabulated_coulomb_runtime_charges(
        data,
        atom_style="charge",
        species=["Si", "O"],
        configured_charges=_charges(),
        units_style="si",
        potential_commands=_base_buck_commands(),
    )
    assert report["passed"] is True
    assert report["n_atoms"] == 2
    assert report["configured_charges_e"] == _charges()


@pytest.mark.parametrize("units", ["metal", "si", "cgs", "micro"])
def test_tabulated_coulomb_runtime_charge_audit_accepts_fixed_kim_type_charges(
    tmp_path: Path,
    units: str,
) -> None:
    from vitriflow.lammps_units import charge_from_elementary_factor
    from vitriflow.workflows.preflight import (
        _validate_tabulated_coulomb_runtime_charges,
    )

    data = tmp_path / f"kim-{units}.data"
    _write_two_atom_charge_data(data, units=units, q_si_e=0.0, q_o_e=0.0)
    native_per_e = charge_from_elementary_factor(units)
    commands = _base_buck_commands() + [
        f"set type 1 charge {2.4 * native_per_e:.16g}",
        f"set type 2 charge {-1.2 * native_per_e:.16g}",
    ]

    report = _validate_tabulated_coulomb_runtime_charges(
        data,
        atom_style="charge",
        species=["Si", "O"],
        configured_charges=None,
        units_style=units,
        potential_commands=commands,
    )

    assert report["passed"] is True
    assert report["configured_charges_e"] is None
    assert report["effective_charges_e"] == pytest.approx(_charges())
    assert report["fixed_set_type_charges_e"] == pytest.approx(_charges())
    assert report["source_by_species"] == {
        "Si": "kim_fixed_set_type_charge",
        "O": "kim_fixed_set_type_charge",
    }
    assert report["input_charges_overwritten_for_species"] == ["Si", "O"]
    assert [row["atom_type"] for row in report["fixed_set_commands"]] == [1, 2]


def test_tabulated_coulomb_runtime_charge_audit_accepts_uniform_input_without_yaml_charges(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import (
        _validate_tabulated_coulomb_runtime_charges,
    )

    data = tmp_path / "data-authority.data"
    _write_two_atom_charge_data(data, units="metal")
    report = _validate_tabulated_coulomb_runtime_charges(
        data,
        atom_style="charge",
        species=["Si", "O"],
        configured_charges=None,
        units_style="metal",
        potential_commands=_base_buck_commands(),
    )

    assert report["effective_charges_e"] == pytest.approx(_charges())
    assert report["source_by_species"] == {
        "Si": "input.data Atoms # charge",
        "O": "input.data Atoms # charge",
    }


def test_tabulated_coulomb_runtime_charge_audit_uses_last_fixed_type_assignment(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import (
        _validate_tabulated_coulomb_runtime_charges,
    )

    data = tmp_path / "ordered-set.data"
    _write_two_atom_charge_data(data, units="metal", q_si_e=0.0, q_o_e=0.0)
    report = _validate_tabulated_coulomb_runtime_charges(
        data,
        atom_style="charge",
        species=["Si", "O"],
        configured_charges=None,
        units_style="metal",
        potential_commands=_base_buck_commands()
        + [
            "set type 1 charge 99.0",
            "set type 1 charge 2.4",
            "set type 2 charge -1.2",
        ],
    )

    assert report["effective_charges_e"] == pytest.approx(_charges())
    assert len(report["fixed_set_commands"]) == 3


@pytest.mark.parametrize(
    ("fixture_kwargs", "atom_style", "commands", "message"),
    [
        ({"q_o_e": -1.1}, "charge", _base_buck_commands(), "conflicts"),
        ({"omit_second_charge": True}, "charge", _base_buck_commands(), "Malformed charge atom row"),
        ({"atoms_style": "atomic"}, "atomic", _base_buck_commands(), "atom_style='charge'"),
        (
            {},
            "charge",
            _base_buck_commands() + ["fix fq all qeq/point 1 10 1e-6 100 params"],
            "KIM-generated fix command",
        ),
        (
            {"q_si_e": 0.0, "q_o_e": 0.0},
            "charge",
            _base_buck_commands() + ["set group all charge 0.0"],
            "only fixed literal",
        ),
        (
            {"q_si_e": 0.0, "q_o_e": 0.0},
            "charge",
            _base_buck_commands() + ["set type 1 charge v_q"],
            "finite numeric literal",
        ),
        (
            {"q_si_e": 0.0, "q_o_e": 0.0},
            "charge",
            _base_buck_commands() + ["set type 1 type 2"],
            "only fixed literal",
        ),
        (
            {},
            "charge",
            _base_buck_commands() + ["fix fswap all atom/swap 10 1 2 12345"],
            "KIM-generated fix command",
        ),
    ],
)
def test_tabulated_coulomb_runtime_charge_audit_fails_closed(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    atom_style: str,
    commands: list[str],
    message: str,
) -> None:
    from vitriflow.workflows.preflight import _validate_tabulated_coulomb_runtime_charges

    data = tmp_path / "bad.data"
    _write_two_atom_charge_data(data, units="nano", **fixture_kwargs)
    with pytest.raises(ValueError, match=message):
        _validate_tabulated_coulomb_runtime_charges(
            data,
            atom_style=atom_style,
            species=["Si", "O"],
            configured_charges=_charges(),
            units_style="nano",
            potential_commands=commands,
        )


def test_tabulated_coulomb_runtime_charge_audit_rejects_yaml_assertion_conflict(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import (
        _validate_tabulated_coulomb_runtime_charges,
    )

    data = tmp_path / "conflicting-assertion.data"
    _write_two_atom_charge_data(data, units="metal", q_si_e=0.0, q_o_e=0.0)
    commands = _base_buck_commands() + [
        "set type 1 charge 2.4",
        "set type 2 charge -1.2",
    ]
    with pytest.raises(ValueError, match="conflicts with structure.charges"):
        _validate_tabulated_coulomb_runtime_charges(
            data,
            atom_style="charge",
            species=["Si", "O"],
            configured_charges={"Si": 2.3, "O": -1.2},
            units_style="metal",
            potential_commands=commands,
        )


def test_generated_kim_charge_audit_requires_fixed_set_coverage_for_every_present_type(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import (
        _validate_tabulated_coulomb_runtime_charges,
    )

    data = tmp_path / "incomplete-kim-set.data"
    _write_two_atom_charge_data(data, units="metal", q_si_e=0.0, q_o_e=0.0)
    with pytest.raises(ValueError, match="explicit fixed KIM.*set type 2"):
        _validate_tabulated_coulomb_runtime_charges(
            data,
            atom_style="charge",
            species=["Si", "O"],
            configured_charges=None,
            units_style="metal",
            potential_commands=_base_buck_commands() + ["set type 1 charge 2.4"],
            require_explicit_set_for_present_types=True,
        )


def test_kim_interaction_extractor_preserves_charge_mutating_fix_for_rejection(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "# BEGIN KIM INTERACTIONS\n"
        "pair_style buck/coul/long 10.0\n"
        "set type 1 charge 2.4\n"
        "fix fq all qeq/point 1 10 1e-6 100 params\n"
        "# END KIM INTERACTIONS\n"
    )

    assert _extract_kim_interactions_block(log) == [
        "pair_style buck/coul/long 10.0",
        "set type 1 charge 2.4",
        "fix fq all qeq/point 1 10 1e-6 100 params",
    ]


def test_kim_interaction_extractor_resolves_real_simulator_model_control_block(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "#=== BEGIN KIM INTERACTIONS ===\n"
        "variable kim_update equal 0\n"
        "variable kim_periodic equal 1\n"
        'if "${kim_periodic} && ${kim_update}" then '
        '"pair_style buck 10.0"\n'
        'if "${kim_periodic} && !${kim_update}" then '
        '"pair_style buck/coul/long 10.0"\n'
        # LAMMPS echoes the command selected by the second conditional.  It
        # must not appear twice in the extracted replay block.
        "pair_style buck/coul/long 10.0\n"
        "pair_coeff 1 1 1000.0 0.3 10.0\n"
        "set type 1 charge 2.4\n"
        "Setting atom values ...\n"
        "  64 settings made for charge\n"
        "#=== END KIM INTERACTIONS ===\n"
    )

    assert _extract_kim_interactions_block(log) == [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 1000.0 0.3 10.0",
        "set type 1 charge 2.4",
    ]


def test_kim_interaction_extractor_retains_fired_command_without_echo(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "# BEGIN KIM INTERACTIONS\n"
        "variable use_long internal 1\n"
        'if "${use_long} == 1" then "pair_style buck/coul/long 10.0" '
        'else "pair_style buck 10.0"\n'
        "pair_coeff 1 1 1000.0 0.3 10.0\n"
        "# END KIM INTERACTIONS\n"
    )

    assert _extract_kim_interactions_block(log) == [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 1000.0 0.3 10.0",
    ]


def test_kim_interaction_extractor_preserves_repeated_direct_commands_for_rejection(
    tmp_path: Path,
) -> None:
    from vitriflow.potential import inspect_buckingham_core_compatibility
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "# BEGIN KIM INTERACTIONS\n"
        "pair_style buck 10.0\n"
        "pair_coeff 1 1 1000.0 0.3 10.0\n"
        # A second pair_style is not redundant in LAMMPS: it resets all pair
        # coefficients.  Extraction must preserve it so the exact parser can
        # reject this ambiguous replay block.
        "pair_style buck 10.0\n"
        "pair_coeff 1 1 1000.0 0.3 10.0\n"
        "# END KIM INTERACTIONS\n"
    )

    commands = _extract_kim_interactions_block(log)
    assert commands == [
        "pair_style buck 10.0",
        "pair_coeff 1 1 1000.0 0.3 10.0",
        "pair_style buck 10.0",
        "pair_coeff 1 1 1000.0 0.3 10.0",
    ]
    with pytest.raises(ValueError, match="exactly one effective pair_style"):
        inspect_buckingham_core_compatibility(commands)


def test_kim_interaction_extractor_rejects_chained_comparison_semantics(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "# BEGIN KIM INTERACTIONS\n"
        'if "2 < 1 < 2" then "pair_style buck 8.0" '
        'else "pair_style buck/coul/long 10.0"\n'
        "# END KIM INTERACTIONS\n"
    )

    with pytest.raises(ValueError, match="Unsupported KIM-generated if condition"):
        _extract_kim_interactions_block(log)


def test_kim_interaction_extractor_rejects_unresolved_condition_even_with_echo(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "# BEGIN KIM INTERACTIONS\n"
        'if "${expired}" then "pair_style buck 8.0" '
        'else "pair_style buck/coul/long 10.0"\n'
        "pair_style buck/coul/long 10.0\n"
        "pair_coeff 1 1 1000.0 0.3 10.0\n"
        "# END KIM INTERACTIONS\n"
    )

    with pytest.raises(ValueError, match="Cannot resolve KIM-generated variable"):
        _extract_kim_interactions_block(log)


def test_kim_interaction_extractor_rejects_selected_unknown_conditional_command(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "# BEGIN KIM INTERACTIONS\n"
        "variable select_hidden equal 1\n"
        'if "${select_hidden}" then "bond_style harmonic"\n'
        "pair_style buck/coul/long 10.0\n"
        "# END KIM INTERACTIONS\n"
    )

    with pytest.raises(ValueError, match="Unsupported KIM-generated material command"):
        _extract_kim_interactions_block(log)


@pytest.mark.parametrize(
    "unknown_command",
    ["bond_style harmonic", "include hidden-material-commands.in"],
)
def test_kim_interaction_extractor_rejects_unknown_material_commands(
    tmp_path: Path,
    unknown_command: str,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "# BEGIN KIM INTERACTIONS\n"
        "pair_style buck/coul/long 10.0\n"
        f"{unknown_command}\n"
        "# END KIM INTERACTIONS\n"
    )
    with pytest.raises(ValueError, match="Unsupported KIM-generated material command"):
        _extract_kim_interactions_block(log)


def test_kim_interaction_extractor_requires_a_complete_unique_marker_pair(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.preflight import _extract_kim_interactions_block

    log = tmp_path / "log.lammps"
    log.write_text(
        "kim interactions\n"
        "pair_style buck/coul/long 10.0\n"
        "Step Temp\n"
    )
    with pytest.raises(ValueError, match="exactly one ordered BEGIN/END"):
        _extract_kim_interactions_block(log)


def test_table_comparison_tail_guard_is_not_masked_by_huge_repulsive_core() -> None:
    import numpy as np

    from vitriflow.workflows.preflight import _compare_pair_table_sections

    r = np.linspace(0.1, 10.0, 2001)
    # The first point is ~1e12 while the physical tail is near zero.  The old
    # global-max absolute allowance was ~1e5 and accepted an O(1) tail defect.
    energy = 1.0e12 * np.exp(-25.0 * (r - r[0]))
    force = 25.0 * energy
    reference = {"P1_1": {"r": r, "energy": energy, "force": force}}
    corrupted_energy = energy.copy()
    corrupted_force = force.copy()
    corrupted_energy[-20] += 0.5
    corrupted_force[-20] -= 0.25
    realized = {
        "P1_1": {"r": r, "energy": corrupted_energy, "force": corrupted_force}
    }
    report = _compare_pair_table_sections(
        reference,
        realized,
        rel_tol=5.0e-5,
        abs_tol_frac=1.0e-7,
        critical_radii_by_section={
            "P1_1": [{"name": "pair_cutoff", "r": float(r[-20])}]
        },
    )
    assert report["passed"] is False
    pair = report["pairs"]["P1_1"]
    assert pair["max_energy_ratio"] > 1.0
    assert pair["critical_radius_neighborhoods"][0]["passed"] is False


def test_parse_gewald_from_modern_lammps_log_format(tmp_path: Path) -> None:
    from vitriflow.workflows.preflight import _parse_gewald_from_log

    log = tmp_path / "log.lammps"
    log.write_text(
        """
PPPM initialization ...
  using 12-bit tables for long-range coulomb
  G vector (1/distance) = 0.278015
  grid = 96 90 128
"""
    )

    val = _parse_gewald_from_log(log)
    assert val is not None
    assert abs(val - 0.278015) < 1.0e-12


def test_prepare_potential_files_prefers_validated_pairwrite_source(monkeypatch, tmp_path: Path) -> None:
    import hashlib

    from vitriflow.config import LammpsPotentialConfig
    from vitriflow.potential import build_tabulated_buckingham_core_lines, prepare_potential_files, update_tabulated_core_metadata_lines

    pot_cfg = LammpsPotentialConfig(interactions=["Si", "O"], commands=_base_buck_commands())
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )
    preflight_dir = tmp_path / "preflight" / "potential_override"
    preflight_dir.mkdir(parents=True)
    src = preflight_dir / "buck_core.table"
    src.write_text("validated table content\n")
    lines = update_tabulated_core_metadata_lines(
        lines,
        generated_by="lammps_pair_write",
        sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("fallback writer should not be used when validated source exists")

    monkeypatch.setattr("vitriflow.potential.write_tabulated_buckingham_core_table", _boom)
    stage_dir = tmp_path / "stages" / "prod_box_001" / "stage1"
    prepare_potential_files(pot_cfg, stage_dir, lines)
    copied = stage_dir / "buck_core.table"
    assert copied.exists()
    assert copied.read_text() == "validated table content\n"


def test_prepare_potential_files_requires_validated_pairwrite_source_when_declared(tmp_path: Path) -> None:
    from vitriflow.config import LammpsPotentialConfig
    from vitriflow.potential import build_tabulated_buckingham_core_lines, prepare_potential_files, update_tabulated_core_metadata_lines

    pot_cfg = LammpsPotentialConfig(interactions=["Si", "O"], commands=_base_buck_commands())
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )
    lines = update_tabulated_core_metadata_lines(lines, generated_by="lammps_pair_write", sha256="0" * 64)

    try:
        prepare_potential_files(pot_cfg, tmp_path / "stage", lines)
    except FileNotFoundError as exc:
        assert "validated tabulated-core source file not found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing validated-source failure")


def test_render_pair_write_script_injects_explicit_gewald_for_long_range_tabulation() -> None:
    from vitriflow.config import RunConfig
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )
    from vitriflow.workflows.preflight import _render_pair_write_script

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_gewald": 0.224358,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge"},
        }
    )

    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None

    script = _render_pair_write_script(
        cfg,
        potential_lines=[
            "pair_style hybrid/overlay buck/coul/long 10.0 zbl 0.8 1.0",
            "pair_coeff 1 1 buck/coul/long 0.0 1.0 0.0",
            "pair_coeff 1 2 buck/coul/long 18003.7572 0.205205 133.5381",
            "pair_coeff 2 2 buck/coul/long 1388.7730 0.362319 175.0000",
            "pair_coeff 1 1 zbl 14 14",
            "pair_coeff 1 2 zbl 14 8",
            "pair_coeff 2 2 zbl 8 8",
            "kspace_style pppm 1.0e-5",
        ],
        spec=spec,
        npoints=1001,
        output_name="reference.table",
    )

    assert "kspace_modify gewald 0.224358" in script
    assert "pair_write 1 1 1001 rsq 0.1 10 reference.table P1_1 2.4 2.4" in script


def test_atomless_pair_write_omits_only_audited_kim_fixed_charges() -> None:
    from vitriflow.config import RunConfig
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
        update_tabulated_core_metadata_lines,
    )
    from vitriflow.workflows.preflight import _render_pair_write_script

    charges = {"Si": 1.91, "O": -0.955}
    set_commands = [
        "set type 1 charge 1.910",
        "set type 2 charge -0.955",
    ]
    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands() + set_commands,
                "core_repulsion": {"enabled": True, "style": "zbl"},
            },
            "structure": {
                "charges": charges,
                "generate": {
                    "method": "random",
                    "formula": "SiO2",
                    "n_formula_units": 1,
                },
            },
            "md": {"atom_style": "charge"},
        }
    )
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands() + set_commands,
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=charges,
        gewald=0.224358,
        has_bonded_topology=False,
    )
    audit = {
        "passed": True,
        "effective_charges_e": charges,
        "fixed_set_commands": [
            {
                "command": set_commands[0],
                "command_index": 6,
                "atom_type": 1,
                "species": "Si",
                "charge_native": 1.91,
                "charge_e": 1.91,
            },
            {
                "command": set_commands[1],
                "command_index": 7,
                "atom_type": 2,
                "species": "O",
                "charge_native": -0.955,
                "charge_e": -0.955,
            },
        ],
    }
    lines = update_tabulated_core_metadata_lines(
        lines, runtime_charge_audit=audit
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None

    # These lines remain part of the accepted runtime potential and are only
    # omitted while rendering the synthetic zero-atom pair_write script.
    assert set_commands[0] in lines
    assert set_commands[1] in lines
    script = _render_pair_write_script(
        cfg,
        potential_lines=lines,
        spec=spec,
        npoints=1001,
        output_name="realized.table",
    )
    script_lines = script.splitlines()
    assert set_commands[0] not in script_lines
    assert set_commands[1] not in script_lines
    assert (
        "pair_write 1 1 1001 rsq 0.1 10 realized.table P1_1 1.91 1.91"
        in script_lines
    )
    assert (
        "pair_write 1 2 1001 rsq 0.1 10 realized.table P1_2 1.91 -0.955"
        in script_lines
    )
    assert (
        "pair_write 2 2 1001 rsq 0.1 10 realized.table P2_2 -0.955 -0.955"
        in script_lines
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_source_coulomb_lookup_controls_do_not_leak_into_replacement_runtime(
    hybrid: bool,
) -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    if hybrid:
        commands = [
            "pair_style hybrid/overlay coul/long 15.0 buck 15.0 morse 15.0",
            "pair_coeff 1 1 coul/long",
            "pair_coeff 1 1 buck 139349.01 0.21 171.08",
            "pair_coeff 1 2 coul/long",
            "pair_coeff 1 2 buck 412.55 0.3 0.0",
            "pair_coeff 1 2 morse 0.44 2.57 1.91",
            "pair_coeff 2 2 coul/long",
            "pair_coeff 2 2 buck 1388.77 0.36 175.0",
            "kspace_style pppm 1.0e-6",
            "pair_modify table 12 tabinner 1.41421356237 shift no",
        ]
        charges = {"Si": 1.8, "O": -1.2}
        cutoff = 15.0
    else:
        commands = _base_buck_commands() + [
            "pair_modify table 12 tabinner 1.41421356237"
        ]
        charges = _charges()
        cutoff = 10.0

    lines = build_tabulated_buckingham_core_lines(
        commands,
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="core.table",
        table_r_min=0.1,
        charges=charges,
        gewald=0.224358,
        has_bonded_topology=False,
        table_style="spline",
    )

    joined = "\n".join(lines)
    assert "pair_modify table" not in joined
    assert "tabinner" not in joined
    assert f" {cutoff:g}" in joined


def test_atomless_pair_write_rejects_unaudited_or_mismatched_charge_state() -> None:
    from vitriflow.config import RunConfig
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
        update_tabulated_core_metadata_lines,
    )
    from vitriflow.workflows.preflight import _render_pair_write_script

    charges = {"Si": 1.91, "O": -0.955}
    set_commands = [
        "set type 1 charge 1.910",
        "set type 2 charge -0.955",
    ]
    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands() + set_commands,
            },
            "structure": {
                "charges": charges,
                "generate": {
                    "method": "random",
                    "formula": "SiO2",
                    "n_formula_units": 1,
                },
            },
            "md": {"atom_style": "charge"},
        }
    )
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands() + set_commands,
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=charges,
        gewald=0.224358,
        has_bonded_topology=False,
    )
    unaudited_spec = _parse_tabulated_core_spec(lines)
    assert unaudited_spec is not None
    with pytest.raises(ValueError, match="without a passed runtime-charge audit"):
        _render_pair_write_script(
            cfg,
            potential_lines=lines,
            spec=unaudited_spec,
            npoints=101,
            output_name="bad.table",
        )

    audit = {
        "passed": True,
        "effective_charges_e": charges,
        "fixed_set_commands": [
            {"command": set_commands[0], "atom_type": 1, "charge_native": 1.91},
            {"command": set_commands[1], "atom_type": 2, "charge_native": -0.955},
        ],
    }
    audited_lines = update_tabulated_core_metadata_lines(
        lines, runtime_charge_audit=audit
    )
    audited_spec = _parse_tabulated_core_spec(audited_lines)
    assert audited_spec is not None
    mismatched_spec = dict(audited_spec)
    mismatched_spec["pairs"] = [dict(pair) for pair in audited_spec["pairs"]]
    mismatched_spec["pairs"][0]["q_i"] = 99.0
    with pytest.raises(ValueError, match="disagrees with the audited runtime charge"):
        _render_pair_write_script(
            cfg,
            potential_lines=audited_lines,
            spec=mismatched_spec,
            npoints=101,
            output_name="bad.table",
        )

    with pytest.raises(ValueError, match="ordered fixed charge commands differ"):
        _render_pair_write_script(
            cfg,
            potential_lines=audited_lines + ["set type 1 charge 9.9"],
            spec=audited_spec,
            npoints=101,
            output_name="bad.table",
        )

    reordered_lines = list(audited_lines)
    set_indices = [
        index
        for index, line in enumerate(reordered_lines)
        if str(line).strip().lower().startswith("set type ")
    ]
    assert len(set_indices) == 2
    first, second = set_indices
    reordered_lines[first], reordered_lines[second] = (
        reordered_lines[second],
        reordered_lines[first],
    )
    with pytest.raises(ValueError, match="ordered fixed charge commands differ"):
        _render_pair_write_script(
            cfg,
            potential_lines=reordered_lines,
            spec=audited_spec,
            npoints=101,
            output_name="bad.table",
        )


def test_verify_tabulated_core_report_includes_force_energy_self_consistency(monkeypatch, tmp_path: Path) -> None:
    import json
    import numpy as np

    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
        tabulated_buckingham_reference_sections,
    )
    from vitriflow.workflows.preflight import _verify_tabulated_core_against_source

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_gewald": 0.224358,
                    "table_verify_points": 2001,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge"},
            "autotune": {"preflight": {"enabled": True}, "tm_scan": {"equil_steps": 10, "sample_steps": 10}},
        }
    )
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
        has_bonded_topology=False,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None

    # N=3000 requires all knots and midpoints: 2*N-1 = 5999 points.
    sections = tabulated_buckingham_reference_sections(spec, npoints=5999)

    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves",
        lambda *a, **k: {"path": Path(k["stage_dir"]) / k["output_name"], "sections": sections, "warnings": []},
    )
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._audit_lammps_inflection_warnings",
        lambda **k: {
            "passed": True,
            "advisory_warnings": [],
            "blocking_warnings": [],
            "pairs": {},
        },
    )

    report = _verify_tabulated_core_against_source(
        LammpsRunner(LammpsConfig()),
        cfg,
        outdir=tmp_path,
        table_potential_lines=lines,
        reference_sections=sections,
        spec=spec,
    )

    assert report["passed"] is True
    assert report["configured_verify_points"] == 2001
    assert report["verify_points"] == 5999
    assert report["coverage"] == "all_table_knots_and_interval_midpoints"
    assert report["self_consistency"]["overall"]["max_force_ratio"] <= 1.0
    saved = json.loads((tmp_path / "preflight" / "table_verify" / "verification_report.json").read_text())
    assert "self_consistency" in saved

    inconsistent = {
        key: {**value, "force": np.zeros_like(value["force"])}
        for key, value in sections.items()
    }
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves",
        lambda *a, **k: {
            "path": Path(k["stage_dir"]) / k["output_name"],
            "sections": inconsistent,
            "warnings": [],
        },
    )
    rejected = _verify_tabulated_core_against_source(
        LammpsRunner(LammpsConfig()),
        cfg,
        outdir=tmp_path / "inconsistent",
        table_potential_lines=lines,
        reference_sections=inconsistent,
        spec=spec,
    )
    assert rejected["comparison"]["passed"] is True
    assert rejected["self_consistency"]["passed"] is False
    assert rejected["passed"] is False
