from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("units", "expected_gpa"),
    [
        ("metal", 160.2176634),              # eV / A^3
        ("real", 6.947695457055374),         # kcal mol^-1 / A^3
        ("electron", 29421.0156965221),      # Ha / bohr^3
        ("nano", 1.0e-3),                    # 1e-21 J / nm^3
        ("si", 1.0e-9),                      # J / m^3
        ("cgs", 1.0e-10),                    # erg / cm^3
        ("micro", 1.0e-6),                   # 1e-15 J / um^3
    ],
)
def test_native_energy_density_has_absolute_gpa_reference(units: str, expected_gpa: float) -> None:
    from vitriflow.lammps_units import energy_density_to_pressure_factor, pressure_to_gpa_factor

    got = energy_density_to_pressure_factor(units) * pressure_to_gpa_factor(units)
    assert got == pytest.approx(expected_gpa, rel=3.0e-10)


def test_electron_nano_and_native_charge_constants_have_absolute_references() -> None:
    from vitriflow.lammps_units import (
        boltzmann_constant_native,
        native_charge_coulomb_prefactor,
        zbl_coulomb_prefactor,
    )

    assert boltzmann_constant_native("electron") == pytest.approx(3.166811563455557e-6, rel=3.0e-12)
    assert boltzmann_constant_native("nano") == pytest.approx(1.380649e-2, rel=1.0e-15)
    assert native_charge_coulomb_prefactor("nano") == pytest.approx(230.70775523517024, rel=2.0e-12)
    assert native_charge_coulomb_prefactor("si") == pytest.approx(8.987551792261171e9, rel=2.0e-12)
    assert zbl_coulomb_prefactor("si") == pytest.approx(2.3070775523417355e-28, rel=3.0e-10)


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ("metal", 14.399645),
        ("real", 332.06371),
        ("electron", 1.0),
        ("nano", 230.7078669),
        ("si", 8.9876e9),
        ("cgs", 1.0),
        ("micro", 8.987556e6),
    ],
)
def test_engine_facing_coulomb_prefactor_matches_lammps_serialized_constants(
    units: str,
    expected: float,
) -> None:
    from vitriflow.lammps_units import lammps_charge_coulomb_prefactor

    assert lammps_charge_coulomb_prefactor(units) == expected


@pytest.mark.parametrize(
    "units", ["metal", "real", "electron", "nano", "si", "cgs", "micro"]
)
def test_autocore_zbl_and_ewald_real_derivatives_in_every_unit_style(
    units: str,
) -> None:
    from vitriflow.lammps_units import (
        charge_from_elementary_factor,
        length_from_angstrom_factor,
    )
    from vitriflow.potential import (
        _pair_coulomb_energy_derivatives,
        _zbl_base_energy_derivatives,
    )

    length_scale = float(length_from_angstrom_factor(units))
    charge_scale = float(charge_from_elementary_factor(units))
    radius = np.asarray([0.3, 0.8, 2.0], dtype=float) * length_scale

    pair = {
        "coul_mode": "long",
        "coul_cutoff": 15.0 * length_scale,
        "pair_cutoff": 15.0 * length_scale,
        "q_i": 1.8 * charge_scale,
        "q_j": -1.2 * charge_scale,
        # G*r is dimensionless, so G transforms as inverse length.
        "gewald": 0.25 / length_scale,
    }
    components = (
        lambda value: _zbl_base_energy_derivatives(
            value, z_i=31, z_j=8, units_style=units
        ),
        lambda value: _pair_coulomb_energy_derivatives(
            value, pair=pair, units_style=units, representation="runtime"
        ),
    )
    for evaluate in components:
        energy, derivative, second = evaluate(radius)
        assert np.all(np.isfinite(energy))
        assert np.all(np.isfinite(derivative))
        assert np.all(np.isfinite(second))

        h_first = radius * 1.0e-6
        numeric_first = (
            evaluate(radius + h_first)[0] - evaluate(radius - h_first)[0]
        ) / (2.0 * h_first)
        assert np.allclose(numeric_first, derivative, rtol=2.0e-8, atol=0.0)

        h_second = radius * 2.0e-5
        numeric_second = (
            evaluate(radius + h_second)[1] - evaluate(radius - h_second)[1]
        ) / (2.0 * h_second)
        assert np.allclose(numeric_second, second, rtol=2.0e-6, atol=0.0)


def test_nano_and_micro_pressure_are_coherent_derived_units_not_pascals() -> None:
    from vitriflow.lammps_units import energy_density_to_pressure_factor, pressure_to_gpa_factor

    # E/V is already the coherent native pressure in both styles.
    assert energy_density_to_pressure_factor("nano") == pytest.approx(1.0)
    assert energy_density_to_pressure_factor("micro") == pytest.approx(1.0)
    # 1 ag/(nm ns^2) = 1e6 Pa; 1 pg/(um us^2) = 1e3 Pa.
    assert pressure_to_gpa_factor("nano") == pytest.approx(1.0e-3)
    assert pressure_to_gpa_factor("micro") == pytest.approx(1.0e-6)


@pytest.mark.parametrize(
    ("units", "length_factor", "mass_factor", "charge_factor"),
    [
        ("metal", 1.0, 1.0, 1.0),
        ("real", 1.0, 1.0, 1.0),
        ("electron", 1.8897261246257702, 1.0, 1.0),
        ("nano", 0.1, 1.66053906660e-6, 1.0),
        ("si", 1.0e-10, 1.66053906660e-27, 1.602176634e-19),
        ("cgs", 1.0e-8, 1.66053906660e-24, 4.803204712570263e-10),
        ("micro", 1.0e-4, 1.66053906660e-12, 1.602176634e-7),
    ],
)
def test_lammps_data_writer_converts_every_dimensional_field(
    tmp_path: Path,
    units: str,
    length_factor: float,
    mass_factor: float,
    charge_factor: float,
) -> None:
    ase = pytest.importorskip("ase")
    from ase.data import atomic_masses, atomic_numbers

    # Some legacy source tests temporarily install an ASE stub and reload this
    # module; reload against the restored real ASE module so this contract test
    # is order independent.
    import importlib
    import vitriflow.structuregen as structuregen

    structuregen = importlib.reload(structuregen)

    atoms = ase.Atoms(
        "H",
        positions=[[1.0, 0.5, 0.25]],
        cell=np.diag([2.0, 3.0, 4.0]),
        pbc=True,
    )
    atoms.set_initial_charges([1.0])
    out = tmp_path / f"{units}.data"
    structuregen.write_lammps_data(out, atoms, atom_style="charge", units_style=units)

    lines = out.read_text().splitlines()
    assert f"LAMMPS units {units}" in lines[0]
    xline = next(line for line in lines if line.endswith("xlo xhi"))
    assert float(xline.split()[1]) == pytest.approx(2.0 * length_factor, rel=3.0e-12)

    mass_start = lines.index("Masses")
    mass_line = next(line for line in lines[mass_start + 1 :] if line.strip())
    written_mass = float(mass_line.split()[1])
    expected_mass = float(atomic_masses[atomic_numbers["H"]]) * mass_factor
    assert written_mass != 0.0
    assert written_mass == pytest.approx(expected_mass, rel=3.0e-15)

    atoms_start = lines.index("Atoms # charge")
    atom_line = next(line for line in lines[atoms_start + 1 :] if line.strip())
    tokens = atom_line.split()
    assert float(tokens[2]) != 0.0
    assert float(tokens[2]) == pytest.approx(charge_factor, rel=3.0e-15)
    assert float(tokens[3]) == pytest.approx(1.0 * length_factor, rel=3.0e-12)

    from vitriflow.io.lammps_data_minimal import read_lammps_data_minimal

    roundtrip = read_lammps_data_minimal(
        out,
        atom_style="charge",
        specorder=["H"],
        units_style=units,
    )
    assert roundtrip.get_positions()[0] == pytest.approx([1.0, 0.5, 0.25], rel=3.0e-12)
    assert np.asarray(roundtrip.get_cell()) == pytest.approx(np.diag([2.0, 3.0, 4.0]), rel=3.0e-12)
    assert roundtrip.get_masses()[0] == pytest.approx(float(atomic_masses[atomic_numbers["H"]]), rel=3.0e-15)
    assert roundtrip.get_initial_charges()[0] == pytest.approx(1.0, rel=3.0e-15)


def _minimal_charge_data(*, mass: str = "1.008", charge: str = "1.0") -> str:
    return (
        "minimal charged structure\n\n"
        "1 atoms\n"
        "1 atom types\n\n"
        "0 2 xlo xhi\n"
        "0 2 ylo yhi\n"
        "0 2 zlo zhi\n\n"
        "Masses\n\n"
        f"1 {mass}\n\n"
        "Atoms # charge\n\n"
        f"1 1 {charge} 0.5 0.5 0.5\n"
    )


@pytest.mark.parametrize("bad_mass", ["0", "-1", "nan", "inf", "not-a-mass"])
def test_minimal_data_reader_rejects_invalid_native_masses(
    tmp_path: Path, bad_mass: str
) -> None:
    from vitriflow.io.lammps_data_minimal import read_lammps_data_minimal

    path = tmp_path / "bad-mass.data"
    path.write_text(_minimal_charge_data(mass=bad_mass))
    with pytest.raises(ValueError, match="[Mm]ass"):
        read_lammps_data_minimal(
            path, atom_style="charge", specorder=["H"], units_style="metal"
        )


@pytest.mark.parametrize("bad_charge", ["nan", "inf", "-inf", "not-a-charge"])
def test_minimal_data_reader_rejects_invalid_native_charges(
    tmp_path: Path, bad_charge: str
) -> None:
    from vitriflow.io.lammps_data_minimal import read_lammps_data_minimal

    path = tmp_path / "bad-charge.data"
    path.write_text(_minimal_charge_data(charge=bad_charge))
    with pytest.raises(ValueError, match="[Cc]harge"):
        read_lammps_data_minimal(
            path, atom_style="charge", specorder=["H"], units_style="metal"
        )


def test_minimal_data_reader_does_not_substitute_missing_explicit_mass(
    tmp_path: Path,
) -> None:
    from vitriflow.io.lammps_data_minimal import read_lammps_data_minimal

    path = tmp_path / "missing-mass.data"
    path.write_text(_minimal_charge_data().replace("1 1.008\n", ""))
    with pytest.raises(ValueError, match="does not cover used atom types"):
        read_lammps_data_minimal(
            path, atom_style="charge", specorder=["H"], units_style="metal"
        )


@pytest.mark.parametrize(
    "replacement",
    ["0 inf xlo xhi", "0 nan ylo yhi", "nan 0 0 xy xz yz"],
)
def test_minimal_data_reader_rejects_nonfinite_box_geometry(
    tmp_path: Path, replacement: str
) -> None:
    from vitriflow.io.lammps_data_minimal import read_lammps_data_minimal

    text = _minimal_charge_data()
    if replacement.endswith("xlo xhi"):
        text = text.replace("0 2 xlo xhi", replacement)
    elif replacement.endswith("ylo yhi"):
        text = text.replace("0 2 ylo yhi", replacement)
    else:
        text = text.replace("0 2 zlo zhi\n", "0 2 zlo zhi\n" + replacement + "\n")
    path = tmp_path / "bad-cell.data"
    path.write_text(text)
    with pytest.raises(ValueError, match="bounds and tilt factors.*finite"):
        read_lammps_data_minimal(
            path, atom_style="charge", specorder=["H"], units_style="metal"
        )


@pytest.mark.parametrize("bad_field", ["position", "origin", "charge", "mass", "id", "type"])
def test_dumpframe_data_writer_rejects_nonfinite_or_invalid_fields(
    tmp_path: Path, bad_field: str
) -> None:
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.io.lammps_data_minimal import write_dumpframe_lammps_data

    ids: object = np.asarray([1])
    types: object = np.asarray([1])
    positions = np.asarray([[0.5, 0.5, 0.5]])
    origin = np.zeros(3)
    charges = {1: 0.0}
    masses = {1: 1.0}
    if bad_field == "position":
        positions[0, 0] = np.nan
    elif bad_field == "origin":
        origin[0] = np.inf
    elif bad_field == "charge":
        charges[1] = np.nan
    elif bad_field == "mass":
        masses[1] = -1.0
    elif bad_field == "id":
        ids = np.asarray([1.5])
    elif bad_field == "type":
        types = np.asarray([0])
    frame = DumpFrame(
        timestep=0,
        ids=ids,  # type: ignore[arg-type]
        types=types,  # type: ignore[arg-type]
        positions=positions,
        cell=np.eye(3),
        origin=origin,
    )
    with pytest.raises(ValueError):
        write_dumpframe_lammps_data(
            tmp_path / "bad.data",
            frame,
            atom_style="charge",
            masses_by_type=masses,
            charges_by_id=charges,
        )


def _first_table_row(path: Path, section: str) -> tuple[float, float, float]:
    lines = path.read_text().splitlines()
    start = lines.index(section)
    for line in lines[start + 1 :]:
        toks = line.split()
        if len(toks) == 4 and toks[0].isdigit():
            return float(toks[1]), float(toks[2]), float(toks[3])
    raise AssertionError(f"No row found for {section}")


def test_mg2_table_is_the_same_physical_potential_in_metal_and_nano(tmp_path: Path) -> None:
    from vitriflow.config import MG2SiNPotentialConfig
    from vitriflow.potential import mg2_sin_commands, write_mg2_sin_table

    metal = MG2SiNPotentialConfig(kind="mg2_sin", user_units="metal", table_points=1000)
    nano = MG2SiNPotentialConfig(kind="mg2_sin", user_units="nano", table_points=1000)
    p_metal = tmp_path / "metal.table"
    p_nano = tmp_path / "nano.table"
    write_mg2_sin_table(p_metal, metal)
    write_mg2_sin_table(p_nano, nano)
    assert p_metal.read_text().splitlines()[0] == "# UNITS: metal"
    assert p_nano.read_text().splitlines()[0] == "# UNITS: nano"

    r_m, u_m, f_m = _first_table_row(p_metal, "SiN")
    r_n, u_n, f_n = _first_table_row(p_nano, "SiN")
    assert r_n == pytest.approx(0.1 * r_m, rel=2.0e-12)
    assert u_n == pytest.approx(160.2176634 * u_m, rel=2.0e-10)
    assert f_n == pytest.approx(1602.176634 * f_m, rel=2.0e-10)
    assert " 0.58" in mg2_sin_commands(nano)[1]


def test_mg2_rejects_reduced_lj_units_without_reference_scales() -> None:
    from pydantic import ValidationError

    from vitriflow.config import MG2SiNPotentialConfig

    with pytest.raises(ValidationError, match="Reduced 'lj' units require explicit user reference scales"):
        MG2SiNPotentialConfig(kind="mg2_sin", user_units="lj")


@pytest.mark.parametrize("kind", ["kim", "lammps"])
def test_all_lammps_potential_configs_reject_unscaled_lj_units(kind: str) -> None:
    from pydantic import ValidationError

    from vitriflow.config import KimConfig, LammpsPotentialConfig

    with pytest.raises(ValidationError, match="Reduced 'lj' units require explicit user reference scales"):
        if kind == "kim":
            KimConfig(model="dummy", interactions=["Si"], user_units="lj")
        else:
            LammpsPotentialConfig(
                interactions=["Si"],
                commands=["pair_style zero 5.0", "pair_coeff * *"],
                user_units="lj",
            )


@pytest.mark.parametrize("units", ["metal", "nano", "si"])
def test_generated_lammps_mass_commands_use_native_mass_units(units: str) -> None:
    from types import SimpleNamespace

    from ase.data import atomic_masses, atomic_numbers

    from vitriflow.lammps_input import _mass_lines_from_interactions
    from vitriflow.lammps_units import mass_from_amu_factor

    line = _mass_lines_from_interactions(
        ["Si"],
        md=SimpleNamespace(mass_mode="kim"),
        units_style=units,
    )
    value = float(line.split()[2])
    expected = float(atomic_masses[atomic_numbers["Si"]]) * mass_from_amu_factor(units)
    assert value != 0.0
    assert value == pytest.approx(expected, rel=3.0e-15)


def test_cp2k_stage_outcome_converts_fs_diffusivity_and_bar_pressure(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from vitriflow.lammps_input import StageSpec
    from vitriflow.workflows.stage_runner import StageArtifacts, stage_outcome_from_artifacts

    thermo = tmp_path / "thermo.csv"
    thermo.write_text(
        "Step,Temp,Press,PotEng,Volume,Density\n"
        "0,300,0.1,-1,100,2.5\n"
        "1,300,0.1,-1,100,2.5\n"
    )
    msd = tmp_path / "cp2k.msd.dat"
    msd.write_text("".join(f"{i} {0.006 * i:.16g}\n" for i in range(20)))
    output = tmp_path / "output.data"
    output.write_text("vitriflow\n\n1 atoms\n")
    art = StageArtifacts(
        stage_dir=tmp_path,
        input_local=tmp_path / "input.data",
        output_local=output,
        log_path=tmp_path / "log",
        msd_path=msd,
        dump_path=None,
        neighbor_skin=float("nan"),
        neighbor_skin_retries=0,
        thermo_csv=thermo,
        msd_csv=tmp_path / "msd.csv",
        traj_extxyz=None,
        final_extxyz=tmp_path / "final.extxyz",
        engine="cp2k",
    )
    stage = StageSpec(
        name="cp2k",
        input_data=tmp_path / "input.data",
        output_data=Path("output.data"),
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=1000.0,
        equil_steps=0,
        run_steps=19,
        seed=1,
    )
    outcome = stage_outcome_from_artifacts(
        art,
        md_cfg=SimpleNamespace(timestep=1.0),
        stage=stage,
    )
    assert outcome.D == pytest.approx(1.0, rel=2.0e-14)  # A^2/ps
    assert outcome.pressure == pytest.approx(0.1)  # GPa


def test_cp2k_raw_thermo_pressure_conversion_is_bar_to_gpa() -> None:
    from vitriflow.parse import ThermoTable
    from vitriflow.workflows.stage_runner import _canonical_cp2k_thermo_table

    table = ThermoTable(columns=["Step", "Press"], data=np.asarray([[0.0, 12345.0]]))
    converted = _canonical_cp2k_thermo_table(table)
    assert converted.data[0, 1] == pytest.approx(1.2345)


def test_canonical_elastic_frame_is_converted_back_to_native_continuation_geometry(
    tmp_path: Path,
) -> None:
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.io.lammps_data_minimal import write_dumpframe_lammps_data

    frame = DumpFrame(
        timestep=0,
        ids=np.asarray([1]),
        types=np.asarray([1]),
        positions=np.asarray([[1.0, 0.5, 0.25]]),
        cell=np.diag([2.0, 3.0, 4.0]),
        origin=np.zeros(3),
    )
    out = tmp_path / "nano.data"
    write_dumpframe_lammps_data(
        out,
        frame,
        masses_by_type={1: 1.0},
        canonical_to_lammps_units_style="nano",
    )
    lines = out.read_text().splitlines()
    assert float(next(x for x in lines if x.endswith("xlo xhi")).split()[1]) == pytest.approx(0.2)
    atom_start = lines.index("Atoms # atomic")
    atom = next(x for x in lines[atom_start + 1 :] if x.strip()).split()
    assert float(atom[2]) == pytest.approx(0.1)


def test_cp2k_stage_metrics_use_fs_to_ps_and_propagate_dump_units(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    captured = {}

    class FakeSeries:
        columns = ["Step", "time"]
        data = np.asarray([[0.0, 0.0], [10.0, 0.02]])
        metadata = {}

        def to_csv(self, path):
            Path(path).write_text("Step,time\n0,0\n10,0.02\n")

    def fake_compute(**kwargs):
        captured.update(kwargs)
        return FakeSeries()

    monkeypatch.setattr(
        "vitriflow.analysis.timeseries.compute_metrics_timeseries", fake_compute
    )
    from vitriflow.workflows.stage_metrics import collect_stage_metrics_timeseries

    cfg = SimpleNamespace(
        stage_timeseries_frame_stride=1,
        stage_timeseries_max_frames=8,
        stage_timeseries_make_plot=False,
        quench_tail_focus_fraction=0.67,
        quench_tail_min_frames=4,
        quench_tail_fallback_fraction=0.4,
    )
    report = collect_stage_metrics_timeseries(
        stage_dir=tmp_path,
        metrics_cfg=cfg,
        cutoffs={},
        md_timestep=2.0,
        lammps_units_style="nano",
        engine="cp2k",
    )
    assert captured["md_timestep"] == pytest.approx(0.002)
    assert captured["trajectory_lammps_units_style"] == "nano"
    assert report["engine"] == "cp2k"
    assert report["time_unit"] == "ps"


@pytest.mark.parametrize(
    "filename",
    ["autotune_results.json", "run_results.json", "analysis_results.json", "output_dataset.json"],
)
def test_public_result_json_embeds_canonical_reporting_contract(
    tmp_path: Path, filename: str
) -> None:
    import json

    from vitriflow.workflows.progress import atomic_write_json

    path = tmp_path / filename
    atomic_write_json(path, {"status": "failed", "units": {"lammps_units": "nano"}})
    units = json.loads(path.read_text())["units"]
    assert units["lammps_units"] == "nano"
    assert units["reporting_contract"] == "vitriflow.canonical_physical_units.v1"
    assert units == {
        **units,
        "length": "Å",
        "volume": "Å^3",
        "density": "g/cm^3",
        "energy": "eV",
        "pressure": "GPa",
        "time": "ps",
        "msd": "Å^2",
        "diffusion": "Å^2/ps",
    }
