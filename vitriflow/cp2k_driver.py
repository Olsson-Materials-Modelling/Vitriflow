from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .config import Cp2kConfig, MDConfig, ThermostatConfig


HARTREE_TO_EV = 27.211386245988  # codata


@dataclass(frozen=True)
class Cp2kEnerTable:
    """Cp2k ener table."""

    step: np.ndarray  # n
    time_fs: np.ndarray  # n
    temperature_K: np.ndarray  # n
    potential_Ha: np.ndarray  # n

    @property
    def potential_eV(self) -> np.ndarray:
        return self.potential_Ha * HARTREE_TO_EV


def _fmt_vec(v: Sequence[float]) -> str:
    return f"{float(v[0]): .12f} {float(v[1]): .12f} {float(v[2]): .12f}"


def _render_thermostat(th_cfg: ThermostatConfig, tdamp_fs: float) -> str:
    """Thermostat."""

    tdamp_fs = float(tdamp_fs)
    if tdamp_fs <= 0 or (not math.isfinite(tdamp_fs)):
        raise ValueError("tdamp_fs must be finite and > 0")

    style = str(th_cfg.style).strip().lower()
    if style == "csvr":
        return (
            "    &THERMOSTAT\n"
            "      TYPE CSVR\n"
            "      &CSVR\n"
            f"        TIMECON [fs] {tdamp_fs:.6f}\n"
            "      &END CSVR\n"
            "    &END THERMOSTAT\n"
        )

    if style in ("nose-hoover", "nose"):
        # length chain cp2
        return (
            "    &THERMOSTAT\n"
            "      TYPE NOSE\n"
            "      &NOSE\n"
            f"        TIMECON [fs] {tdamp_fs:.6f}\n"
            "        LENGTH 3\n"
            "      &END NOSE\n"
            "    &END THERMOSTAT\n"
        )

    raise ValueError(f"Unsupported thermostat style for CP2K: {th_cfg.style}")


def render_cp2k_md_input(
    *,
    atoms,  # ase atoms
    cfg: Cp2kConfig,
    md_cfg: MDConfig,
    # basis pseudopotential
    # override basis potential
    # robust conda deployments
    # search differ environments
    basis_set_file_name: str | None = None,
    potential_file_name: str | None = None,
    ensemble: str,
    temperature_K: float,
    steps: int,
    timestep_fs: float,
    tdamp_fs: float,
    project: str,
    energy_every: int,
    traj_every: int,
    traj_file: str,
    ener_file: str,
    restart_file: str | None = None,
    pressure_bar: float | None = None,
    pdamp_fs: float | None = None,
    print_level: str = "LOW",
) -> str:
    """Cp2k md input."""

    # import importing ase
    from ase.data import chemical_symbols

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if energy_every < 1 or traj_every < 1:
        raise ValueError("print frequencies must be >= 1")

    T = float(temperature_K)
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError("temperature_K must be finite and > 0")

    dt = float(timestep_fs)
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("timestep_fs must be finite and > 0")

    ens = str(ensemble).strip().lower()
    if ens not in ("nvt", "npt"):
        raise ValueError("ensemble must be 'nvt' or 'npt'")

    basis_fn = str(cfg.basis_set_file_name) if basis_set_file_name is None else str(basis_set_file_name)
    pot_fn = str(cfg.potential_file_name) if potential_file_name is None else str(potential_file_name)

    barostat_block = ""
    stress_tensor_line = ""
    cp2k_ensemble = "NVT"

    if ens == "npt":
        cp2k_ensemble = "NPT_I"
        P = float(md_cfg.pressure if pressure_bar is None else pressure_bar)
        pd = float(md_cfg.barostat.pdamp if pdamp_fs is None else pdamp_fs)
        if not math.isfinite(P):
            raise ValueError("pressure must be finite for NPT")
        if not (math.isfinite(pd) and pd > 0.0):
            raise ValueError("pdamp_fs must be finite and > 0 for NPT")
        barostat_block = (
            "    &BAROSTAT\n"
            f"      PRESSURE [bar] {P:.6f}\n"
            f"      TIMECON [fs] {pd:.6f}\n"
            "    &END BAROSTAT\n"
        )
        # npt stress tensor
        stress_tensor_line = "  STRESS_TENSOR ANALYTICAL\n"

    ext_restart_block = ""
    if restart_file is not None:
        rf = str(restart_file)
        # velocities cell thermostat
        baro_restart = ""
        if ens == "npt":
            baro_restart = (
                "  RESTART_BAROSTAT T\n"
                "  RESTART_BAROSTAT_THERMOSTAT T\n"
            )
        ext_restart_block = (
            "&EXT_RESTART\n"
            f"  RESTART_FILE_NAME {rf}\n"
            "  RESTART_DEFAULT F\n"
            "  RESTART_POS T\n"
            "  RESTART_VEL T\n"
            "  RESTART_CELL T\n"
            "  RESTART_COUNTERS F\n"
            "  RESTART_THERMOSTAT T\n"
            + baro_restart
            + "&END EXT_RESTART\n"
            "\n"
        )

    # guard
    symbols = list(atoms.get_chemical_symbols())
    uniq = sorted(set(symbols))
    missing = [s for s in uniq if s not in cfg.kind_settings]
    if missing:
        raise ValueError("CP2K kind_settings is missing entries for: " + ", ".join(missing))

    # coord
    coord_lines: list[str] = []
    pos = atoms.get_positions()  # angstrom
    for sym, r in zip(symbols, pos):
        if sym not in chemical_symbols:
            raise ValueError(f"Invalid chemical symbol in atoms: {sym!r}")
        coord_lines.append(
            f"      {sym} {float(r[0]): .12f} {float(r[1]): .12f} {float(r[2]): .12f}"
        )

    # cell vectors explicitly
    cell = np.array(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("atoms.get_cell() must be 3x3")
    a = cell[0]
    b = cell[1]
    c = cell[2]

    # dft core
    ot_block = ""
    if cfg.use_ot:
        ot_block = (
            "      &OT ON\n"
            f"        MINIMIZER {cfg.ot_minimizer}\n"
            f"        PRECONDITIONER {cfg.ot_preconditioner}\n"
            "        ENERGY_GAP 0.001\n"
            "      &END OT\n"
        )

    kind_blocks: list[str] = []
    for sym in uniq:
        kc = cfg.kind_settings[sym]
        kind_blocks.append(
            "    &KIND "
            + sym
            + "\n"
            + f"      BASIS_SET {kc.basis_set}\n"
            + f"      POTENTIAL {kc.potential}\n"
            + "    &END KIND\n"
        )

    thermostat_block = _render_thermostat(md_cfg.thermostat, float(tdamp_fs))

    # trajectory extxyz standard
    # trajectory dcd variant
    # cell information extxyz
    # analysis
    tf = str(traj_file).strip().lower()
    if tf.endswith(".dcd"):
        # dcd cell ase
        # orthorhombic cubic aligned
        # equivalent aligned generally
        traj_format = "DCD_ALIGNED_CELL"
    elif tf.endswith(".xyz"):
        traj_format = "XYZ"
    elif tf.endswith(".pdb"):
        traj_format = "PDB"
    else:
        # fallback
        traj_format = "XYZ"


    inp = "".join(
        [
            "&GLOBAL\n",
            f"  PROJECT {project}\n",
            "  RUN_TYPE MD\n",
            f"  PRINT_LEVEL {print_level}\n",
            "&END GLOBAL\n",
            "\n",
            ext_restart_block,
            "&FORCE_EVAL\n",
            "  METHOD QS\n",
            stress_tensor_line,
            "  &DFT\n",
            f"    BASIS_SET_FILE_NAME {basis_fn}\n",
            f"    POTENTIAL_FILE_NAME {pot_fn}\n",
            "    CHARGE 0\n",
            "    MULTIPLICITY 1\n",
            "    &MGRID\n",
            f"      CUTOFF [Ry] {cfg.cutoff_Ry:.6f}\n",
            f"      REL_CUTOFF [Ry] {cfg.rel_cutoff_Ry:.6f}\n",
            f"      NGRIDS {int(cfg.ngrids)}\n",
            "    &END MGRID\n",
            "    &QS\n",
            "      EPS_DEFAULT 1.0E-12\n",
            "      EXTRAPOLATION PS\n",
            "      EXTRAPOLATION_ORDER 3\n",
            "    &END QS\n",
            "    &SCF\n",
            # robust occasional convergence
            # aggressive preflight aborting
            # warn
            # ignore convergence failure
            "      IGNORE_CONVERGENCE_FAILURE T\n",
            f"      EPS_SCF {cfg.eps_scf:.6e}\n",
            f"      MAX_SCF {int(cfg.max_scf)}\n",
            f"      SCF_GUESS {cfg.scf_guess}\n",
            ot_block,
            "    &END SCF\n",
            "    &XC\n",
            f"      &XC_FUNCTIONAL {cfg.xc_functional}\n",
            f"      &END XC_FUNCTIONAL\n",
            "    &END XC\n",
            "  &END DFT\n",
            "  &SUBSYS\n",
            "    &CELL\n",
            f"      A [angstrom] {_fmt_vec(a)}\n",
            f"      B [angstrom] {_fmt_vec(b)}\n",
            f"      C [angstrom] {_fmt_vec(c)}\n",
            "      PERIODIC XYZ\n",
            "    &END CELL\n",
            "    &COORD\n",
            "\n".join(coord_lines),
            "\n",
            "    &END COORD\n",
            "".join(kind_blocks),
            "  &END SUBSYS\n",
            "&END FORCE_EVAL\n",
            "\n",
            "&MOTION\n",
            "  &MD\n",
            f"    ENSEMBLE {cp2k_ensemble}\n",
            f"    TIMESTEP [fs] {dt:.6f}\n",
            f"    STEPS {int(steps)}\n",
            f"    TEMPERATURE [K] {T:.6f}\n",
            barostat_block,
            thermostat_block,
            # printing controls energy
            "    &PRINT\n",
            "      &ENERGY\n",
            "        &EACH\n",
            f"          MD {int(energy_every)}\n",
            "        &END EACH\n",
            f"        FILENAME ={ener_file}\n",
            "      &END ENERGY\n",
            "    &END PRINT\n",
            "  &END MD\n",
            # printing controls trajectory
            "  &PRINT\n",
            "    &TRAJECTORY\n",
            f"      FORMAT {traj_format}\n",
            "      &EACH\n",
            f"        MD {int(traj_every)}\n",
            "      &END EACH\n",
            f"      FILENAME ={traj_file}\n",
            "      ADD_LAST NUMERIC\n",
            "    &END TRAJECTORY\n",
            "    &RESTART\n",
            "      &EACH\n",
            "        MD 0\n",
            "      &END EACH\n",
            "      ADD_LAST NUMERIC\n",
            "    &END RESTART\n",
            "  &END PRINT\n",
            "&END MOTION\n",
        ]
    )

    return inp


def render_cp2k_cell_opt_input(
    *,
    atoms,  # ase atoms
    cfg: Cp2kConfig,
    # basis pseudopotential
    basis_set_file_name: str | None = None,
    potential_file_name: str | None = None,
    project: str,
    optimizer: str = "LBFGS",
    max_iter: int = 200,
    keep_angles: bool = True,
    external_pressure_bar: float = 0.0,
    traj_every: int = 1,
    traj_file: str = "cell_opt.dcd",
    restart_file: str | None = None,
    print_level: str = "LOW",
) -> str:
    """Cp2k cell opt."""

    # import importing ase
    from ase.data import chemical_symbols

    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1")
    if int(traj_every) < 1:
        raise ValueError("traj_every must be >= 1")
    if not bool(keep_angles):
        raise ValueError("keep_angles must be True (KEEP_ANGLES is enforced)")
    P = float(external_pressure_bar)
    if not math.isfinite(P):
        raise ValueError("external_pressure_bar must be finite")

    opt = str(optimizer).strip().upper()
    if opt not in ("LBFGS", "BFGS", "CG"):
        raise ValueError("optimizer must be one of {'LBFGS','BFGS','CG'}")

    basis_fn = str(cfg.basis_set_file_name) if basis_set_file_name is None else str(basis_set_file_name)
    pot_fn = str(cfg.potential_file_name) if potential_file_name is None else str(potential_file_name)

    ext_restart_block = ""
    if restart_file is not None:
        rf = str(restart_file)
        ext_restart_block = (
            "&EXT_RESTART\n"
            f"  RESTART_FILE_NAME {rf}\n"
            "  RESTART_DEFAULT F\n"
            "  RESTART_POS T\n"
            "  RESTART_CELL T\n"
            "  RESTART_COUNTERS F\n"
            "&END EXT_RESTART\n"
            "\n"
        )

    # guard
    symbols = list(atoms.get_chemical_symbols())
    uniq = sorted(set(symbols))
    missing = [s for s in uniq if s not in cfg.kind_settings]
    if missing:
        raise ValueError("CP2K kind_settings is missing entries for: " + ", ".join(missing))

    # coord
    coord_lines: list[str] = []
    pos = atoms.get_positions()  # angstrom
    for sym, r in zip(symbols, pos):
        if sym not in chemical_symbols:
            raise ValueError(f"Invalid chemical symbol in atoms: {sym!r}")
        coord_lines.append(
            f"      {sym} {float(r[0]): .12f} {float(r[1]): .12f} {float(r[2]): .12f}"
        )

    # cell vectors explicitly
    cell = np.array(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("atoms.get_cell() must be 3x3")
    a = cell[0]
    b = cell[1]
    c = cell[2]

    # dft core
    ot_block = ""
    if cfg.use_ot:
        ot_block = (
            "      &OT ON\n"
            f"        MINIMIZER {cfg.ot_minimizer}\n"
            f"        PRECONDITIONER {cfg.ot_preconditioner}\n"
            "        ENERGY_GAP 0.001\n"
            "      &END OT\n"
        )

    kind_blocks: list[str] = []
    for sym in uniq:
        kc = cfg.kind_settings[sym]
        kind_blocks.append(
            "    &KIND "
            + sym
            + "\n"
            + f"      BASIS_SET {kc.basis_set}\n"
            + f"      POTENTIAL {kc.potential}\n"
            + "    &END KIND\n"
        )

    # trajectory dcd cell
    tf = str(traj_file).strip().lower()
    if tf.endswith(".dcd"):
        traj_format = "DCD_ALIGNED_CELL"
    elif tf.endswith(".xyz"):
        traj_format = "XYZ"
    elif tf.endswith(".pdb"):
        traj_format = "PDB"
    else:
        traj_format = "XYZ"

    inp = "".join(
        [
            "&GLOBAL\n",
            f"  PROJECT {project}\n",
            "  RUN_TYPE CELL_OPT\n",
            f"  PRINT_LEVEL {print_level}\n",
            "&END GLOBAL\n",
            "\n",
            ext_restart_block,
            "&FORCE_EVAL\n",
            "  METHOD QS\n",
            "  STRESS_TENSOR ANALYTICAL\n",
            "  &DFT\n",
            f"    BASIS_SET_FILE_NAME {basis_fn}\n",
            f"    POTENTIAL_FILE_NAME {pot_fn}\n",
            "    CHARGE 0\n",
            "    MULTIPLICITY 1\n",
            "    &MGRID\n",
            f"      CUTOFF [Ry] {cfg.cutoff_Ry:.6f}\n",
            f"      REL_CUTOFF [Ry] {cfg.rel_cutoff_Ry:.6f}\n",
            f"      NGRIDS {int(cfg.ngrids)}\n",
            "    &END MGRID\n",
            "    &QS\n",
            "      EPS_DEFAULT 1.0E-12\n",
            "      EXTRAPOLATION PS\n",
            "      EXTRAPOLATION_ORDER 3\n",
            "    &END QS\n",
            "    &SCF\n",
            f"      EPS_SCF {cfg.eps_scf:.6e}\n",
            f"      MAX_SCF {int(cfg.max_scf)}\n",
            f"      SCF_GUESS {cfg.scf_guess}\n",
            ot_block,
            "    &END SCF\n",
            "    &XC\n",
            f"      &XC_FUNCTIONAL {cfg.xc_functional}\n",
            f"      &END XC_FUNCTIONAL\n",
            "    &END XC\n",
            "  &END DFT\n",
            "  &SUBSYS\n",
            "    &CELL\n",
            f"      A [angstrom] {_fmt_vec(a)}\n",
            f"      B [angstrom] {_fmt_vec(b)}\n",
            f"      C [angstrom] {_fmt_vec(c)}\n",
            "      PERIODIC XYZ\n",
            "    &END CELL\n",
            "    &COORD\n",
            "\n".join(coord_lines),
            "\n",
            "    &END COORD\n",
            "".join(kind_blocks),
            "  &END SUBSYS\n",
            "&END FORCE_EVAL\n",
            "\n",
            "&MOTION\n",
            "  &CELL_OPT\n",
            "    TYPE DIRECT_CELL_OPT\n",
            f"    OPTIMIZER {opt}\n",
            f"    MAX_ITER {int(max_iter)}\n",
            "    KEEP_ANGLES TRUE\n",
            f"    EXTERNAL_PRESSURE [bar] {P:.6f}\n",
            "  &END CELL_OPT\n",
            "  &PRINT\n",
            "    &TRAJECTORY\n",
            f"      FORMAT {traj_format}\n",
            "      &EACH\n",
            f"        CELL_OPT {int(traj_every)}\n",
            "      &END EACH\n",
            f"      FILENAME ={traj_file}\n",
            "      ADD_LAST NUMERIC\n",
            "    &END TRAJECTORY\n",
            "    &RESTART\n",
            "      &EACH\n",
            "        CELL_OPT 0\n",
            "      &END EACH\n",
            "      ADD_LAST NUMERIC\n",
            "    &END RESTART\n",
            "  &END PRINT\n",
            "&END MOTION\n",
        ]
    )

    return inp

def parse_cp2k_ener(path: Path) -> Cp2kEnerTable:
    """Cp2k ener."""

    if not path.exists():
        raise FileNotFoundError(str(path))

    step: list[int] = []
    time_fs: list[float] = []
    temp: list[float] = []
    pot: list[float] = []

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            # unexpected skip defensively
            continue
        try:
            step.append(int(float(parts[0])))
            time_fs.append(float(parts[1]))
            # parts temperature docs
            temp.append(float(parts[3]))
            # potential energy hartree
            pot.append(float(parts[4]))
        except Exception:
            continue

    if len(step) == 0:
        raise ValueError(f"No data rows parsed from CP2K energy file: {path}")

    return Cp2kEnerTable(
        step=np.asarray(step, dtype=int),
        time_fs=np.asarray(time_fs, dtype=float),
        temperature_K=np.asarray(temp, dtype=float),
        potential_Ha=np.asarray(pot, dtype=float),
    )


def density_g_cm3_from_atoms(atoms) -> float:
    """Density g cm3."""

    # masses amu volume
    m_amu = float(np.sum(atoms.get_masses()))
    vol_a3 = float(atoms.get_volume())
    if vol_a3 <= 0.0:
        raise ValueError("atoms volume must be > 0")
    # amu 66053906660e g
    # 1e cm
    return 1.66053906660 * m_amu / vol_a3


def unwrap_positions_fractional(frac: np.ndarray) -> np.ndarray:
    """Unwrap positions fractional."""

    if frac.ndim != 3 or frac.shape[-1] != 3:
        raise ValueError("frac must have shape (n_frames, n_atoms, 3)")

    out = np.empty_like(frac, dtype=float)
    out[0] = frac[0]
    for t in range(1, frac.shape[0]):
        df = frac[t] - frac[t - 1]
        # component nearest integer
        df = df - np.round(df)
        out[t] = out[t - 1] + df
    return out


def compute_msd(
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    unwrap: bool = True,
) -> np.ndarray:
    """Msd."""

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")

    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("cell must be shape (3,3)")

    if unwrap:
        # ase cells cell
        inv = np.linalg.inv(cell)
        frac = pos @ inv  # t i
        frac = frac - np.floor(frac)
        ufrac = unwrap_positions_fractional(frac)
        pos_u = ufrac @ cell
    else:
        pos_u = pos

    dr = pos_u - pos_u[0:1]
    msd = np.mean(np.sum(dr * dr, axis=-1), axis=1)
    return msd
