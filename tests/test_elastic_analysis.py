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


def test_affine_isotropization_strain_orthorhombic_box():
    from vitriflow.analysis.elastic import affine_isotropization_strain

    cell = np.diag([2.0, 4.0, 8.0])
    out = affine_isotropization_strain(cell)
    assert out["volume"] == pytest.approx(64.0)
    assert out["target_cubic_length"] == pytest.approx(4.0)
    eps = np.asarray(out["small_strain"], dtype=float)
    assert np.allclose(eps, np.diag([1.0, 0.0, -0.5]))


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
    assert "affine_isotropization" in summary
    assert summary["local_stress_summary"]["von_mises_native"]["max"] >= 0.0


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
