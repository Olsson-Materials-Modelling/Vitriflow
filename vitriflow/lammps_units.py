"""Exact dimensional conversions between ASE and LAMMPS unit styles.

ASE stores positions in Angstrom, masses in atomic-mass units, and charges as
multiples of the elementary charge.  A LAMMPS data file, in contrast, is
unitless: every number is interpreted according to the ``units`` command that
precedes ``read_data``.  Keeping these conversions in one module prevents a
data file and the potential that consumes it from silently using different
dimensions.

Constants below use the exact SI definitions of the elementary charge and
Avogadro constant and the CODATA 2018 atomic-mass constant used by ASE/LAMMPS.
"""

from __future__ import annotations

import math


ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906660e-27
ANGSTROM_M = 1.0e-10
BOHR_ANGSTROM = 0.529177210903
EV_J = ELEMENTARY_CHARGE_C
HARTREE_EV = 27.211386245988
AVOGADRO = 6.02214076e23
STATCOULOMB_C = 3.3356409519815204e-10

PHYSICAL_LAMMPS_UNIT_STYLES = frozenset(
    {"metal", "real", "electron", "nano", "si", "cgs", "micro"}
)

CANONICAL_REPORTING_CONTRACT = "vitriflow.canonical_physical_units.v1"


def canonical_reporting_units() -> dict[str, str]:
    """Public engine-neutral numeric contract embedded in result JSON."""

    return {
        "reporting_contract": CANONICAL_REPORTING_CONTRACT,
        "length": "Å",
        "volume": "Å^3",
        "density": "g/cm^3",
        "temperature": "K",
        "energy": "eV",
        "pressure": "GPa",
        "time": "ps",
        "msd": "Å^2",
        "diffusion": "Å^2/ps",
    }


def normalize_lammps_units_style(units_style: str) -> str:
    """Return a supported dimensional LAMMPS unit style.

    ``lj`` is deliberately excluded: an ASE structure has physical dimensions,
    so there is no unique conversion to reduced LJ units without user-supplied
    reference scales for length, mass, energy, and charge.
    """

    units = str(units_style or "").strip().lower()
    if units not in PHYSICAL_LAMMPS_UNIT_STYLES:
        supported = ", ".join(sorted(PHYSICAL_LAMMPS_UNIT_STYLES))
        raise ValueError(
            f"Unsupported dimensional LAMMPS units style {units_style!r}; "
            f"supported styles are {supported}. Reduced 'lj' units require "
            "explicit user reference scales and cannot be inferred from ASE data."
        )
    return units


def length_from_angstrom_factor(units_style: str) -> float:
    """Multiply an Angstrom value by this factor for a LAMMPS data file."""

    units = normalize_lammps_units_style(units_style)
    return {
        "metal": 1.0,
        "real": 1.0,
        "electron": 1.0 / BOHR_ANGSTROM,
        "nano": 0.1,
        "si": ANGSTROM_M,
        "cgs": 1.0e-8,
        "micro": 1.0e-4,
    }[units]


def length_to_angstrom_factor(units_style: str) -> float:
    """Multiply a native LAMMPS length by this factor to obtain Angstrom."""

    return 1.0 / length_from_angstrom_factor(units_style)


def mass_from_amu_factor(units_style: str) -> float:
    """Multiply an atomic mass in u by this factor for LAMMPS ``Masses``."""

    units = normalize_lammps_units_style(units_style)
    return {
        # g/mol is numerically identical to u per particle.
        "metal": 1.0,
        "real": 1.0,
        "electron": 1.0,
        # LAMMPS nano mass is attograms (1 ag = 1e-21 kg).
        "nano": ATOMIC_MASS_KG / 1.0e-21,
        "si": ATOMIC_MASS_KG,
        "cgs": ATOMIC_MASS_KG * 1.0e3,
        # LAMMPS micro mass is picograms (1 pg = 1e-15 kg).
        "micro": ATOMIC_MASS_KG / 1.0e-15,
    }[units]


def mass_to_amu_factor(units_style: str) -> float:
    """Multiply a native LAMMPS mass by this factor to obtain atomic-mass units."""

    return 1.0 / mass_from_amu_factor(units_style)


def charge_from_elementary_factor(units_style: str) -> float:
    """Multiply a charge in units of ``e`` by this factor for LAMMPS."""

    units = normalize_lammps_units_style(units_style)
    return {
        "metal": 1.0,
        "real": 1.0,
        "electron": 1.0,
        "nano": 1.0,
        "si": ELEMENTARY_CHARGE_C,
        "cgs": ELEMENTARY_CHARGE_C / STATCOULOMB_C,
        # LAMMPS micro charge is picocoulombs.
        "micro": ELEMENTARY_CHARGE_C / 1.0e-12,
    }[units]


def charge_to_elementary_factor(units_style: str) -> float:
    """Multiply a native LAMMPS charge by this factor to obtain units of e."""

    return 1.0 / charge_from_elementary_factor(units_style)


def energy_from_ev_factor(units_style: str) -> float:
    """Multiply an energy in eV by this factor for LAMMPS."""

    units = normalize_lammps_units_style(units_style)
    return {
        "metal": 1.0,
        "real": EV_J * AVOGADRO / 4184.0,
        "electron": 1.0 / HARTREE_EV,
        # nano energy: ag nm^2/ns^2 = 1e-21 J
        "nano": EV_J / 1.0e-21,
        "si": EV_J,
        "cgs": EV_J * 1.0e7,
        # micro energy: pg um^2/us^2 = 1e-15 J
        "micro": EV_J / 1.0e-15,
    }[units]


def energy_to_ev_factor(units_style: str) -> float:
    """Multiply a native LAMMPS energy by this factor to obtain eV."""

    return 1.0 / energy_from_ev_factor(units_style)


def time_to_ps_factor(units_style: str) -> float:
    """Multiply native LAMMPS time by this factor to obtain picoseconds."""

    units = normalize_lammps_units_style(units_style)
    return {
        "metal": 1.0,
        "real": 1.0e-3,
        "electron": 1.0e-3,
        "nano": 1.0e3,
        "si": 1.0e12,
        "cgs": 1.0e12,
        "micro": 1.0e6,
    }[units]


def density_to_g_cm3_factor(units_style: str) -> float:
    """Multiply native LAMMPS mass density by this factor for g/cm^3.

    LAMMPS explicitly reports ``metal`` and ``real`` thermo density in g/cm^3.
    Other dimensional styles use their coherent mass/volume definitions.
    """

    units = normalize_lammps_units_style(units_style)
    if units in {"metal", "real"}:
        return 1.0
    native_mass_kg = {
        "electron": ATOMIC_MASS_KG,
        "nano": 1.0e-21,
        "si": 1.0,
        "cgs": 1.0e-3,
        "micro": 1.0e-15,
    }[units]
    native_length_m = {
        "electron": BOHR_ANGSTROM * 1.0e-10,
        "nano": 1.0e-9,
        "si": 1.0,
        "cgs": 1.0e-2,
        "micro": 1.0e-6,
    }[units]
    # kg/m^3 -> g/cm^3 contributes 1e-3.
    return (native_mass_kg / native_length_m**3) * 1.0e-3


def volume_to_angstrom3_factor(units_style: str) -> float:
    """Multiply native LAMMPS volume by this factor to obtain Angstrom^3."""

    return length_to_angstrom_factor(units_style) ** 3


def msd_to_angstrom2_factor(units_style: str) -> float:
    """Multiply native squared displacement by this factor to obtain A^2."""

    return length_to_angstrom_factor(units_style) ** 2


def diffusivity_to_angstrom2_per_ps_factor(units_style: str) -> float:
    """Convert native length^2/native-time diffusivity to A^2/ps."""

    return msd_to_angstrom2_factor(units_style) / time_to_ps_factor(units_style)


def energy_density_to_pressure_factor(units_style: str) -> float:
    """Convert native energy/native-volume to LAMMPS native pressure.

    ``compute born/matrix`` reports energy, not pressure.  After division by
    volume this factor is therefore required before comparing the result with
    ``compute pressure`` or labelling it in the unit style's pressure units.
    """

    units = normalize_lammps_units_style(units_style)
    energy_si = {
        "metal": EV_J,
        "real": 4184.0 / AVOGADRO,
        "electron": HARTREE_EV * EV_J,
        "nano": 1.0e-21,
        "si": 1.0,
        "cgs": 1.0e-7,
        "micro": 1.0e-15,
    }[units]
    length_si = {
        "metal": 1.0e-10,
        "real": 1.0e-10,
        "electron": BOHR_ANGSTROM * 1.0e-10,
        "nano": 1.0e-9,
        "si": 1.0,
        "cgs": 1.0e-2,
        "micro": 1.0e-6,
    }[units]
    pressure_pa = energy_si / (length_si**3)
    pa_per_native_pressure = {
        "metal": 1.0e5,  # bar
        "real": 101325.0,  # atm
        "electron": 1.0,
        "nano": 1.0e6,  # ag/(nm ns^2)
        "si": 1.0,
        "cgs": 0.1,  # dyne/cm^2 (barye)
        "micro": 1.0e3,  # pg/(um us^2)
    }[units]
    factor = pressure_pa / pa_per_native_pressure
    if not (math.isfinite(factor) and factor > 0.0):  # pragma: no cover
        raise RuntimeError(f"Invalid energy-density conversion for {units!r}")
    return factor


def pressure_to_gpa_factor(units_style: str) -> float:
    """Multiply native LAMMPS pressure by this factor to obtain GPa."""

    units = normalize_lammps_units_style(units_style)
    return {
        "metal": 1.0e-4,
        "real": 1.01325e-4,
        "electron": 1.0e-9,
        "nano": 1.0e-3,
        "si": 1.0e-9,
        "cgs": 1.0e-10,
        "micro": 1.0e-6,
    }[units]


def native_charge_coulomb_prefactor(units_style: str) -> float:
    """Return physical ``k_e`` in native energy*length/native-charge^2 units."""

    units = normalize_lammps_units_style(units_style)
    # Styles whose charge unit is the elementary charge include e^2 in qqrd2e.
    if units == "metal":
        return 14.3996454784255
    if units == "real":
        return 332.06371329919216
    if units == "electron":
        return 1.0
    if units == "nano":
        return 230.70775523517024
    if units == "si":
        return 8.987551792261171e9
    if units == "cgs":
        return 1.0
    if units == "micro":
        return 8.987551792261171e6
    raise AssertionError(units)  # pragma: no cover


def lammps_charge_coulomb_prefactor(units_style: str) -> float:
    """Return the exact ``force->qqr2e`` constant used by LAMMPS.

    LAMMPS intentionally serializes a finite set of unit constants rather than
    recomputing them from CODATA at runtime.  A KSpace-compatible real-space
    table must use those same constants; otherwise its real/reciprocal split
    differs slightly from the engine even when both formulas are correct.

    Values match LAMMPS stable_22Jul2025_update4 ``Update::set_units``.
    """

    units = normalize_lammps_units_style(units_style)
    return {
        "metal": 14.399645,
        "real": 332.06371,
        "electron": 1.0,
        "nano": 230.7078669,
        "si": 8.9876e9,
        "cgs": 1.0,
        "micro": 8.987556e6,
    }[units]


def zbl_coulomb_prefactor(units_style: str) -> float:
    """Return ``k_e e^2`` in native energy*length units for nuclear charges."""

    q_native = charge_from_elementary_factor(units_style)
    return native_charge_coulomb_prefactor(units_style) * q_native * q_native


def boltzmann_constant_native(units_style: str) -> float:
    """Boltzmann constant in the LAMMPS unit style's energy/K."""

    units = normalize_lammps_units_style(units_style)
    # Exact k_B in J/K converted through the native energy unit.
    k_b_j = 1.380649e-23
    joule_per_native_energy = EV_J / energy_from_ev_factor(units)
    return k_b_j / joule_per_native_energy
