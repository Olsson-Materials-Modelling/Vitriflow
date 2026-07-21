from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_born21_roundtrip_and_isotropic_projection():
    from vitriflow.analysis.elastic import (
        born21_to_matrix,
        isotropic_projection_voigt,
        matrix_to_born21,
    )

    K = 50.0
    G = 20.0
    C = np.array(
        [
            [K + 4.0 * G / 3.0, K - 2.0 * G / 3.0, K - 2.0 * G / 3.0, 0.0, 0.0, 0.0],
            [K - 2.0 * G / 3.0, K + 4.0 * G / 3.0, K - 2.0 * G / 3.0, 0.0, 0.0, 0.0],
            [K - 2.0 * G / 3.0, K - 2.0 * G / 3.0, K + 4.0 * G / 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, G, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, G, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, G],
        ],
        dtype=float,
    )
    b21 = matrix_to_born21(C)
    C2 = born21_to_matrix(b21)
    assert np.allclose(C, C2)

    iso, K2, G2 = isotropic_projection_voigt(C)
    assert np.allclose(C, iso)
    assert K2 == K
    assert G2 == G

    # The uniform-strain Voigt shear average includes deviatoric normal
    # stiffness.  Averaging only C44/C55/C66 would incorrectly return 20.
    anisotropic = np.diag([100.0, 100.0, 100.0, 20.0, 20.0, 20.0])
    _iso_a, K_a, G_a = isotropic_projection_voigt(anisotropic)
    assert K_a == pytest.approx(100.0 / 3.0)
    assert G_a == pytest.approx(32.0)


def test_affine_isotropization_strain_orthorhombic_box():
    from vitriflow.analysis.elastic import affine_isotropization_strain

    cell = np.diag([2.0, 4.0, 8.0])
    out = affine_isotropization_strain(cell)
    assert out["volume"] == pytest.approx(64.0)
    assert out["target_cubic_length"] == pytest.approx(4.0)
    eps = np.asarray(out["small_strain"], dtype=float)
    assert np.allclose(eps, np.diag([1.0, 0.0, -0.5]))


def test_affine_isotropization_reports_column_vector_F_for_triclinic_cell():
    from vitriflow.analysis.elastic import affine_isotropization_strain

    # Row-vector cell convention, deliberately non-symmetric/triclinic so a
    # transposed deformation gradient cannot pass accidentally.
    h0 = np.asarray(
        [[3.0, 0.0, 0.0], [0.7, 4.0, 0.0], [-0.2, 0.9, 5.0]], dtype=float
    )
    out = affine_isotropization_strain(h0)
    length = float(out["target_cubic_length"])
    h1 = np.eye(3) * length
    F = np.asarray(out["F"], dtype=float)
    assert F @ h0.T == pytest.approx(h1.T, rel=2.0e-15, abs=2.0e-15)
    assert not np.allclose(F.T @ h0.T, h1.T)


def test_read_single_custom_dump_and_build_summary(tmp_path: Path):
    from vitriflow.analysis.elastic import (
        build_elastic_screen_summary,
        parse_born_stress_raw,
        read_single_custom_dump,
    )

    raw = tmp_path / "born_raw.txt"
    raw.write_text(
        " ".join(["vol=1000", "S1=100", "S2=100", "S3=100", "S4=0", "S5=0", "S6=0"] + [f"B{i}={2000 if i <= 3 else (500 if i <= 6 else 1000 if i in (7,8,12) else 0)}" for i in range(1, 22)])
    )
    dump = tmp_path / "local_stress.dump"
    dump.write_text(
        """
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_pst[1] c_pst[2] c_pst[3] c_pst[4] c_pst[5] c_pst[6]
1 1 1 2 3 -10 -10 -10 0 0 0
2 1 4 5 6 -20 -20 -20 0 0 0
""".lstrip()
    )

    r = parse_born_stress_raw(raw)
    d = read_single_custom_dump(dump)
    cols = d["data"]
    pos = np.column_stack([cols["x"], cols["y"], cols["z"]])
    typ = np.asarray(cols["type"], dtype=int)
    pst = np.column_stack([cols[f"c_pst[{i}]" ] for i in range(1, 7)])
    summary = build_elastic_screen_summary(
        born21=np.asarray(r["born21"], dtype=float),
        global_stress_voigt=np.asarray(r["global_stress_voigt"], dtype=float),
        volume=float(r["volume"]),
        n_atoms=2,
        local_positions=pos,
        local_types=typ,
        local_stress_volume=pst,
        units_style="metal",
        force_isotropic=True,
        input_cell=np.diag([8.0, 10.0, 12.5]),
    )

    assert summary["status"] == "ok"
    assert summary["units"]["pressure_native"] == "bar"
    assert summary["born_matrix_GPa"] is not None
    # B1=2000 eV and V=1000 A^3 -> C11=2 eV/A^3.
    assert summary["born_matrix_GPa"][0][0] == pytest.approx(2.0 * 160.2176634, rel=2.0e-10)
    # compute pressure already reports bar; it is not divided by V.
    assert summary["global_stress_voigt_native"][0] == pytest.approx(100.0)
    assert summary["global_stress_voigt_GPa"][0] == pytest.approx(0.01)
    assert summary["units"]["born_energy_density_to_pressure_factor"] == pytest.approx(1.602176634e6)
    assert summary["volume_native"] == pytest.approx(1000.0)
    assert summary["volume_A3"] == pytest.approx(1000.0)
    assert summary["kind"] == "static_affine_born_snapshot_diagnostic"
    assert summary["method"]["thermodynamic_elastic_tensor"] is False
    assert summary["method"]["relaxed_elastic_modulus"] is False
    assert "stress_fluctuation_covariance" in summary["method"]["omitted_terms"]
    assert "non_affine_internal_relaxation" in summary["method"]["omitted_terms"]
    assert "not a finite-temperature thermodynamic elastic tensor" in summary["note"]
    assert "voigt_born_bulk_response_native" in summary
    assert "voigt_born_shear_response_native" in summary
    assert "voigt_bulk_modulus_native" not in summary
    assert "affine_isotropization" in summary
    proxy = summary["average_volume_normalized_virial_proxy_summary"]
    assert proxy["von_mises_proxy_native"]["max"] >= 0.0
    assert proxy["atomic_volume_partition"] is False
    assert proxy["unique_local_cauchy_stress"] is False
    assert "not uniquely defined local Cauchy stresses" in summary["note"]


@pytest.mark.parametrize("units", ["metal", "real", "nano"])
def test_local_stress_volume_is_already_native_pressure_times_volume(units: str):
    from vitriflow.analysis.elastic import build_elastic_screen_summary

    summary = build_elastic_screen_summary(
        born21=np.ones(21),
        global_stress_voigt=np.zeros(6),
        volume=1.0,
        n_atoms=1,
        local_positions=np.zeros((1, 3)),
        local_types=np.ones(1, dtype=int),
        # LAMMPS stress/atom is Cauchy stress*volume in native pressure-volume
        # units; it is not compute born/matrix energy.
        local_stress_volume=np.asarray([[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]]),
        units_style=units,
    )
    assert summary["average_volume_normalized_virial_proxy_summary"][
        "hydrostatic_proxy_native"
    ]["p50"] == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    ("natoms", "bounds", "atom_rows", "message"),
    [
        (1, "0 10\n0 10\n0 10", "1.5 1 1 2 3", "positive integers"),
        (2, "0 10\n0 10\n0 10", "1 1 1 2 3\n1 1 4 5 6", "must be unique"),
        (1, "0 10\n0 10\n0 10", "1 0 1 2 3", "types.*positive integers"),
        (1, "0 inf\n0 10\n0 10", "1 1 1 2 3", "must be finite"),
        (1, "0 10\n0 10\n0 10", "1 1 1 2 3\n2 1 4 5 6", "extra data"),
        (2, "0 10\n0 10\n0 10", "1 1 1 2 3", "ends after 1 rows"),
    ],
)
def test_single_custom_dump_parser_rejects_structural_corruption(
    tmp_path: Path,
    natoms: int,
    bounds: str,
    atom_rows: str,
    message: str,
) -> None:
    from vitriflow.analysis.elastic import read_single_custom_dump

    path = tmp_path / "bad.dump"
    path.write_text(
        "ITEM: TIMESTEP\n0\n"
        f"ITEM: NUMBER OF ATOMS\n{natoms}\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        f"{bounds}\n"
        "ITEM: ATOMS id type x y z\n"
        f"{atom_rows}\n"
    )
    with pytest.raises(ValueError, match=message):
        read_single_custom_dump(path)


def test_lammps_stress_atom_sign_is_preserved_before_hydrostatic_conversion():
    from vitriflow.analysis.elastic import hydrostatic_from_stress, stress_volume_to_pressure_like

    stress = stress_volume_to_pressure_like(
        np.asarray([[-2.0, -2.0, -2.0, 0.0, 0.0, 0.0]]),
        volume=1.0,
        n_atoms=1,
    )
    assert stress[0, 0] == pytest.approx(-2.0)
    assert hydrostatic_from_stress(stress)[0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("born21", np.asarray([np.nan] + [1.0] * 20)),
        ("global_stress_voigt", np.asarray([np.inf] + [0.0] * 5)),
        ("local_positions", np.asarray([[np.nan, 0.0, 0.0]])),
        ("local_stress_volume", np.asarray([[np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]])),
    ],
)
def test_elastic_summary_rejects_nonfinite_physical_evidence(field: str, value: np.ndarray):
    from vitriflow.analysis.elastic import build_elastic_screen_summary

    kwargs = {
        "born21": np.ones(21),
        "global_stress_voigt": np.zeros(6),
        "volume": 1.0,
        "n_atoms": 1,
        "local_positions": np.zeros((1, 3)),
        "local_types": np.ones(1, dtype=int),
        "local_stress_volume": np.zeros((1, 6)),
        "units_style": "metal",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="finite"):
        build_elastic_screen_summary(**kwargs)


def test_elastic_summary_rejects_atom_count_normalization_mismatch():
    from vitriflow.analysis.elastic import build_elastic_screen_summary

    with pytest.raises(ValueError, match="n_atoms must match"):
        build_elastic_screen_summary(
            born21=np.ones(21),
            global_stress_voigt=np.zeros(6),
            volume=2.0,
            n_atoms=2,
            local_positions=np.zeros((1, 3)),
            local_types=np.ones(1, dtype=int),
            local_stress_volume=np.zeros((1, 6)),
            units_style="metal",
        )


@pytest.mark.parametrize("units", ["metal", "real", "nano"])
def test_local_stress_csv_is_canonical_and_unit_explicit(tmp_path: Path, units: str):
    from vitriflow.analysis.elastic import write_local_stress_csv
    from vitriflow.lammps_units import (
        length_from_angstrom_factor,
        pressure_to_gpa_factor,
    )

    out = tmp_path / f"{units}.csv"
    native_position = np.asarray([[1.0, 2.0, 3.0]]) * length_from_angstrom_factor(units)
    stress = np.asarray([[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]])
    write_local_stress_csv(
        out,
        ids=np.asarray([1]),
        types=np.asarray([1]),
        positions=native_position,
        stress_volume=np.zeros((1, 6)),
        stress_native=stress,
        hydrostatic_native=np.asarray([1.0]),
        von_mises_native=np.asarray([2.0]),
        units_style=units,
        normalization_volume_native_per_atom=8.0,
    )
    data = np.genfromtxt(out, delimiter=",", names=True)
    assert set(data.dtype.names or ()) == {
        "id", "type", "x_A", "y_A", "z_A", "normalization_volume_A3_per_atom",
        "virial_proxy_xx_GPa", "virial_proxy_yy_GPa", "virial_proxy_zz_GPa",
        "virial_proxy_xy_GPa", "virial_proxy_xz_GPa", "virial_proxy_yz_GPa",
        "hydrostatic_virial_proxy_GPa", "von_mises_virial_proxy_GPa",
    }
    assert float(data["x_A"]) == pytest.approx(1.0)
    assert float(data["virial_proxy_xx_GPa"]) == pytest.approx(-pressure_to_gpa_factor(units))
    assert float(data["hydrostatic_virial_proxy_GPa"]) == pytest.approx(pressure_to_gpa_factor(units))
    assert float(data["von_mises_virial_proxy_GPa"]) == pytest.approx(2.0 * pressure_to_gpa_factor(units))


def test_local_stress_csv_legacy_call_retains_native_schema(tmp_path: Path):
    from vitriflow.analysis.elastic import write_local_stress_csv

    out = tmp_path / "legacy.csv"
    with pytest.warns(DeprecationWarning, match="legacy-only schema"):
        write_local_stress_csv(
            out,
            ids=np.asarray([7]),
            types=np.asarray([2]),
            positions=np.asarray([[1.0, 2.0, 3.0]]),
            stress_volume=np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            stress_native=np.asarray([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
            hydrostatic_native=np.asarray([-5.0]),
            von_mises_native=np.asarray([4.0]),
        )

    header = out.read_text().splitlines()[0]
    assert header == (
        "id,type,x,y,z,sv_xx,sv_yy,sv_zz,sv_xy,sv_xz,sv_yz,"
        "s_xx,s_yy,s_zz,s_xy,s_xz,s_yz,hydrostatic_native,von_mises_native"
    )
    data = np.genfromtxt(out, delimiter=",", names=True)
    assert float(data["x"]) == pytest.approx(1.0)
    assert float(data["sv_yz"]) == pytest.approx(6.0)
    assert float(data["s_xx"]) == pytest.approx(6.0)


def test_local_stress_csv_rejects_partial_canonical_provenance(tmp_path: Path):
    from vitriflow.analysis.elastic import write_local_stress_csv

    arrays = {
        "ids": np.asarray([1]),
        "types": np.asarray([1]),
        "positions": np.zeros((1, 3)),
        "stress_volume": np.zeros((1, 6)),
        "stress_native": np.zeros((1, 6)),
        "hydrostatic_native": np.zeros(1),
        "von_mises_native": np.zeros(1),
    }
    with pytest.raises(ValueError, match="must be provided together"):
        write_local_stress_csv(tmp_path / "partial.csv", **arrays, units_style="metal")


def test_plot_elastic_screen_generates_file(tmp_path: Path):
    from vitriflow.plotting import plot_elastic_screen

    elastic_dir = tmp_path / "elastic"
    elastic_dir.mkdir()
    (elastic_dir / "elastic_screen.json").write_text(
        '{\n'
        '  "status": "ok",\n'
        '  "units": {"pressure_native": "bar", "pressure_to_GPa_factor": 0.0001},\n'
        '  "born_matrix_native": [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],\n'
        '  "born_matrix_GPa": [[0.0001,0,0,0,0,0],[0,0.0001,0,0,0,0],[0,0,0.0001,0,0,0],[0,0,0,0.0001,0,0],[0,0,0,0,0.0001,0],[0,0,0,0,0,0.0001]],\n'
        '  "isotropy_residual": 0.0,\n'
        '  "normal_shear_coupling_norm": 0.0,\n'
        '  "voigt_bulk_modulus_native": 1.0,\n'
        '  "voigt_bulk_modulus_GPa": 0.0001,\n'
        '  "voigt_shear_modulus_native": 1.0,\n'
        '  "voigt_shear_modulus_GPa": 0.0001,\n'
        '  "local_stress_summary": {"von_mises_native": {"max_over_median": 2.0}},\n'
        '  "flags": []\n'
        '}'
    )
    (elastic_dir / "local_stress.csv").write_text(
        "id,type,x,y,z,sv_xx,sv_yy,sv_zz,sv_xy,sv_xz,sv_yz,s_xx,s_yy,s_zz,s_xy,s_xz,s_yz,hydrostatic_native,von_mises_native\n"
        "1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,-1,0.5\n"
        "2,1,1,1,1,0,0,0,0,0,0,2,2,2,0,0,0,-2,1.5\n"
    )
    out = tmp_path / "elastic.png"
    plot_elastic_screen(elastic_dir, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_elastic_screen_default_is_output_root_independent(tmp_path: Path):
    """Reference and reproduction roots must not change plot pixels."""

    import shutil

    from vitriflow.plotting import plot_elastic_screen

    source = tmp_path / "reference" / "production" / "box_001" / "relax" / "elastic"
    source.mkdir(parents=True)
    (source / "elastic_screen.json").write_text(
        '{\n'
        '  "status": "ok",\n'
        '  "units": {"pressure_native": "bar", "pressure_to_GPa_factor": 0.0001},\n'
        '  "born_matrix_native": [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],\n'
        '  "born_matrix_GPa": [[0.0001,0,0,0,0,0],[0,0.0001,0,0,0,0],[0,0,0.0001,0,0,0],[0,0,0,0.0001,0,0],[0,0,0,0,0.0001,0],[0,0,0,0,0,0.0001]],\n'
        '  "isotropy_residual": 0.0,\n'
        '  "normal_shear_coupling_norm": 0.0,\n'
        '  "voigt_bulk_modulus_native": 1.0,\n'
        '  "voigt_bulk_modulus_GPa": 0.0001,\n'
        '  "voigt_shear_modulus_native": 1.0,\n'
        '  "voigt_shear_modulus_GPa": 0.0001,\n'
        '  "local_stress_summary": {"von_mises_native": {"max_over_median": 2.0}},\n'
        '  "flags": []\n'
        '}'
    )
    (source / "local_stress.csv").write_text(
        "id,type,x,y,z,sv_xx,sv_yy,sv_zz,sv_xy,sv_xz,sv_yz,s_xx,s_yy,s_zz,s_xy,s_xz,s_yz,hydrostatic_native,von_mises_native\n"
        "1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,-1,0.5\n"
        "2,1,1,1,1,0,0,0,0,0,0,2,2,2,0,0,0,-2,1.5\n"
    )
    reproduced = (
        tmp_path / "comparison" / "production" / "box_001" / "relax" / "elastic"
    )
    reproduced.parent.mkdir(parents=True)
    shutil.copytree(source, reproduced)

    ref_png = source / "elastic_screen.png"
    cmp_png = reproduced / "elastic_screen.png"
    plot_elastic_screen(source, ref_png, dpi=100)
    plot_elastic_screen(reproduced, cmp_png, dpi=100)

    assert ref_png.read_bytes() == cmp_png.read_bytes()
