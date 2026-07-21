from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .config import Cp2kConfig, MDConfig, ThermostatConfig


HARTREE_TO_EV = 27.211386245988  # codata


def cp2k_scf_continuation_policy(cp2k_version: Sequence[int] | None) -> tuple[str, str]:
    """Return the version-correct SCF continuation input and audit label.

    CP2K 2024.1 introduced ``IGNORE_CONVERGENCE_FAILURE`` with a default of
    false.  Older releases do not know that keyword and retain their legacy
    continue-after-warning behaviour.  A live VitriFlow workflow therefore
    has to query the executable before rendering its input.  ``None`` is kept
    as an offline-rendering compatibility mode and deliberately emits no
    version-specific keyword; live workflow call sites must pass a detected
    version.
    """

    if cp2k_version is None:
        return "", "version_unspecified_keyword_omitted"
    if isinstance(cp2k_version, (str, bytes)):
        raise TypeError("cp2k_version must be a sequence of integer components")
    try:
        components = tuple(int(value) for value in cp2k_version)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cp2k_version must contain integer components") from exc
    if not components or any(value < 0 for value in components):
        raise ValueError("cp2k_version must contain at least one nonnegative component")
    if any(float(raw) != float(value) for raw, value in zip(cp2k_version, components)):
        raise ValueError("cp2k_version components must be exact integers")
    if components[0] >= 2024:
        return "      IGNORE_CONVERGENCE_FAILURE T\n", "explicit_ignore_convergence_failure"
    return "", "legacy_keyword_omitted"


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


def _validated_cp2k_atoms_geometry(atoms, *, n_symbols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return finite positions/cell suitable for a periodic CP2K calculation."""

    try:
        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = np.asarray(atoms.get_cell(), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("CP2K atom positions and cell must be numeric") from exc
    if positions.shape != (int(n_symbols), 3):
        raise ValueError(
            f"atoms.get_positions() must have shape ({int(n_symbols)}, 3), "
            f"got {positions.shape}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("CP2K atom positions must all be finite")
    if cell.shape != (3, 3):
        raise ValueError("atoms.get_cell() must be 3x3")
    if not np.all(np.isfinite(cell)):
        raise ValueError("CP2K cell vectors must all be finite")
    scale = float(np.max(np.abs(cell)))
    determinant = float(np.linalg.det(cell))
    tolerance = 128.0 * np.finfo(float).eps * max(scale**3, np.finfo(float).tiny)
    if not math.isfinite(determinant) or abs(determinant) <= tolerance:
        raise ValueError("CP2K periodic cell must be nonsingular")
    return positions, cell


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
    restart_wfn_file: str | None = None,
    restart_barostat: bool | None = None,
    seed: int = 2000,
    pressure_bar: float | None = None,
    pdamp_fs: float | None = None,
    print_level: str = "LOW",
    cp2k_version: Sequence[int] | None = None,
) -> str:
    """Cp2k md input."""

    # import importing ase
    from ase.data import chemical_symbols

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if energy_every < 1 or traj_every < 1:
        raise ValueError("print frequencies must be >= 1")

    # A ramp segment can legitimately be shorter than the workflow-wide
    # thermo stride.  CP2K then writes only the step-zero energy row, while
    # instantaneous NPT pressures begin at propagated step one.  Keep the
    # downstream exact-step pressure/energy contract and make the segment
    # producer observable instead of inventing a nearest-step association.
    effective_energy_every = min(int(energy_every), int(steps))

    try:
        seed_value = float(seed)
        rng_seed = int(seed_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("seed must be a positive 32-bit integer") from exc
    if (
        isinstance(seed, bool)
        or not math.isfinite(seed_value)
        or seed_value != float(rng_seed)
        or not (1 <= rng_seed <= 2_147_483_647)
    ):
        raise ValueError("seed must be a positive 32-bit integer")

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
        if ens == "npt" and restart_barostat is not False:
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
            "  RESTART_RANDOMG T\n"
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
    pos, cell = _validated_cp2k_atoms_geometry(atoms, n_symbols=len(symbols))
    for sym, r in zip(symbols, pos):
        if sym not in chemical_symbols:
            raise ValueError(f"Invalid chemical symbol in atoms: {sym!r}")
        coord_lines.append(
            f"      {sym} {float(r[0]): .12f} {float(r[1]): .12f} {float(r[2]): .12f}"
        )

    # cell vectors explicitly
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
    wfn_restart_line = (
        ""
        if restart_wfn_file is None
        else f"    WFN_RESTART_FILE_NAME {str(restart_wfn_file)}\n"
    )
    # Cross-invocation WFN continuity is permitted only through the explicit,
    # checksum-validated restart bundle supplied by the stage runner.  Falling
    # back to cfg.scf_guess=RESTART without WFN_RESTART_FILE_NAME would let
    # CP2K discover same-project residue from an interrupted invocation.
    scf_guess = "RESTART" if restart_wfn_file is not None else "ATOMIC"

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


    scf_continuation_line, _ = cp2k_scf_continuation_policy(cp2k_version)

    inp = "".join(
        [
            "&GLOBAL\n",
            f"  PROJECT {project}\n",
            "  RUN_TYPE MD\n",
            f"  SEED {rng_seed}\n",
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
            wfn_restart_line,
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
            scf_continuation_line,
            f"      EPS_SCF {cfg.eps_scf:.6e}\n",
            f"      MAX_SCF {int(cfg.max_scf)}\n",
            f"      SCF_GUESS {scf_guess}\n",
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
            f"          MD {effective_energy_every}\n",
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
    cp2k_version: Sequence[int] | None = None,
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
    pos, cell = _validated_cp2k_atoms_geometry(atoms, n_symbols=len(symbols))
    for sym, r in zip(symbols, pos):
        if sym not in chemical_symbols:
            raise ValueError(f"Invalid chemical symbol in atoms: {sym!r}")
        coord_lines.append(
            f"      {sym} {float(r[0]): .12f} {float(r[1]): .12f} {float(r[2]): .12f}"
        )

    # cell vectors explicitly
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

    scf_continuation_line, _ = cp2k_scf_continuation_policy(cp2k_version)

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
            # CELL_OPT restart authentication currently binds the CP2K
            # geometry/cell restart, not a wavefunction file.  An implicit
            # project-RESTART.wfn lookup would therefore consume state outside
            # the manifest.  Start the electronic problem from atoms for both
            # fresh and authenticated geometry-restart CELL_OPT launches.
            # Segmented CP2K MD retains its explicit, authenticated WFN
            # continuity path in render_cp2k_md_input.
            "      SCF_GUESS ATOMIC\n",
            scf_continuation_line,
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
    """Parse CP2K ``.ener`` rows atomically and fail closed on corruption."""

    if not path.exists():
        raise FileNotFoundError(str(path))

    step: list[int] = []
    time_fs: list[float] = []
    temp: list[float] = []
    pot: list[float] = []

    previous_step: int | None = None
    previous_time: float | None = None
    for line_number, raw in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(
                f"Malformed CP2K energy row at {path}:{line_number}: expected at least 6 columns"
            )
        try:
            step_numeric = float(parts[0].replace("D", "E").replace("d", "e"))
            row_step = int(step_numeric)
            row_time = float(parts[1].replace("D", "E").replace("d", "e"))
            row_temp = float(parts[3].replace("D", "E").replace("d", "e"))
            row_pot = float(parts[4].replace("D", "E").replace("d", "e"))
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"Malformed numeric CP2K energy row at {path}:{line_number}: {line}"
            ) from exc
        if not math.isfinite(step_numeric) or step_numeric != float(row_step) or row_step < 0:
            raise ValueError(
                f"CP2K energy step must be a nonnegative integer at {path}:{line_number}"
            )
        if not all(math.isfinite(x) for x in (row_time, row_temp, row_pot)):
            raise ValueError(
                f"CP2K energy time, temperature, and potential must be finite at "
                f"{path}:{line_number}"
            )
        if row_time < 0.0 or row_temp < 0.0:
            raise ValueError(
                f"CP2K energy time and temperature must be nonnegative at {path}:{line_number}"
            )
        if previous_step is not None and row_step <= previous_step:
            raise ValueError(
                f"CP2K energy steps must be strictly increasing at {path}:{line_number}"
            )
        if previous_time is not None and row_time <= previous_time:
            raise ValueError(
                f"CP2K energy times must be strictly increasing at {path}:{line_number}"
            )
        step.append(row_step)
        time_fs.append(row_time)
        temp.append(row_temp)
        pot.append(row_pot)
        previous_step = row_step
        previous_time = row_time

    if len(step) == 0:
        raise ValueError(f"No data rows parsed from CP2K energy file: {path}")

    return Cp2kEnerTable(
        step=np.asarray(step, dtype=int),
        time_fs=np.asarray(time_fs, dtype=float),
        temperature_K=np.asarray(temp, dtype=float),
        potential_Ha=np.asarray(pot, dtype=float),
    )


_CP2K_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_CP2K_SCF_FAILURE = re.compile(r"\bSCF(?:\s+run)?\s+NOT\s+converged\b", re.IGNORECASE)
_CP2K_CELL_OPT_SUCCESS = re.compile(
    r"\b(?:GEOMETRY|CELL)\s+OPTIMIZATION\s+COMPLETED\b", re.IGNORECASE
)


def _cp2k_output_contains(path: Path, pattern: re.Pattern[str]) -> bool:
    with Path(path).open("r", errors="replace") as handle:
        return any(pattern.search(line) is not None for line in handle)


def assert_cp2k_scf_converged(path: Path) -> None:
    """Reject a CP2K output containing an explicitly unconverged SCF cycle."""

    output = Path(path)
    if not output.is_file():
        raise FileNotFoundError(str(output))
    if _cp2k_output_contains(output, _CP2K_SCF_FAILURE):
        raise RuntimeError(
            f"CP2K reported an unconverged SCF cycle in {output}; "
            "the caller's opt-in strict SCF-convergence contract was not met"
        )


def count_cp2k_scf_failures(path: Path) -> int:
    """Count explicitly reported unconverged SCF cycles without rejecting a run.

    This is the normal workflow policy: CP2K is instructed (where supported)
    to continue, while VitriFlow preserves a machine-readable count so the
    event is visible rather than silently discarded.  The strict assertion
    above remains available to callers that intentionally require it.
    """

    output = Path(path)
    if not output.is_file():
        raise FileNotFoundError(str(output))
    count = 0
    with output.open("r", errors="replace") as handle:
        for line in handle:
            if _CP2K_SCF_FAILURE.search(line) is not None:
                count += 1
    return count


def assert_cp2k_cell_opt_converged(path: Path) -> None:
    """Require CP2K's positive optimisation-completion marker.

    A zero process return code only says CP2K terminated normally.  In
    particular, reaching ``MAX_ITER`` is a normal termination and must not be
    confused with a converged CELL_OPT result.
    """

    output = Path(path)
    if not output.is_file():
        raise FileNotFoundError(str(output))
    if not _cp2k_output_contains(output, _CP2K_CELL_OPT_SUCCESS):
        raise RuntimeError(
            f"CP2K CELL_OPT did not report optimisation convergence in {output}"
        )


def parse_cp2k_md_step_pressures(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse MD step numbers and instantaneous NPT pressures in bar.

    CP2K's run-information line contains both an instantaneous and an average
    pressure.  Input-summary lines contain only the target pressure, so those
    are deliberately excluded.
    """

    output = Path(path)
    if not output.is_file():
        raise FileNotFoundError(str(output))
    steps: list[int] = []
    values: list[float] = []
    step_re = re.compile(r"(?:MD\|\s*)?STEP\s+NUMBER\s*(?:=\s*)?(\d+)\b", re.IGNORECASE)
    runtime_re = re.compile(
        rf"PRESSURE\s*\[bar\]\s*(?:=\s*)?({_CP2K_FLOAT})(?:\s+({_CP2K_FLOAT}))?",
        re.IGNORECASE,
    )
    current_step: int | None = None
    with output.open("r", errors="replace") as handle:
        lines = handle
        for line in lines:
            step_match = step_re.search(line)
            if step_match is not None:
                current_step = int(step_match.group(1))
                continue
            match = runtime_re.search(line)
            if match is None:
                continue
            # Runtime output is either ``PRESSURE [bar] = inst [avg]`` or
            # ``MD| Pressure [bar] inst [avg]``.  CP2K versions/builds differ
            # in whether the running average is present.  Requiring a step
            # marker is what distinguishes either runtime form from the
            # one-number target-pressure line in the input summary.
            is_runtime = current_step is not None and (
                "=" in line[match.start() : match.end()] or "MD|" in line.upper()
            )
            if not is_runtime:
                continue
            try:
                value = float(match.group(1).replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            if math.isfinite(value) and current_step is not None:
                steps.append(current_step)
                values.append(value)
                current_step = None
    return np.asarray(steps, dtype=int), np.asarray(values, dtype=float)


def map_cp2k_pressures_to_energy_steps(
    energy_steps: Sequence[int],
    pressure_steps: Sequence[int],
    pressures_bar: Sequence[float],
    *,
    source: Path | None = None,
) -> dict[int, float]:
    """Build a finite pressure map aligned with every propagated energy row.

    CP2K pressure samples come from the text output, while temperatures and
    energies come from the ``.ener`` file.  Local step zero is the initial
    state before an NPT propagation step and may legitimately have no reported
    instantaneous pressure.  Every positive energy step must have a finite
    aligned pressure; otherwise the resulting thermo series would silently
    mix measured pressures with missing values.
    """

    label = f" in {Path(source)}" if source is not None else ""

    def _exact_nonnegative_steps(values: Sequence[int], *, field: str) -> np.ndarray:
        raw_values = np.asarray(values, dtype=object).reshape(-1)
        parsed: list[int] = []
        maximum = int(np.iinfo(np.intp).max)
        for index, raw in enumerate(raw_values.tolist()):
            if isinstance(raw, (bool, np.bool_)):
                raise RuntimeError(
                    f"CP2K {field}[{index}] must be a nonnegative integer{label}"
                )
            try:
                numeric = Decimal(str(raw).strip())
            except (InvalidOperation, ValueError, AttributeError) as exc:
                raise RuntimeError(
                    f"CP2K {field}[{index}] must be a nonnegative integer{label}"
                ) from exc
            if (
                not numeric.is_finite()
                or numeric != numeric.to_integral_value()
                or numeric < 0
                or numeric > maximum
            ):
                raise RuntimeError(
                    f"CP2K {field}[{index}] must be a nonnegative integer{label}"
                )
            parsed.append(int(numeric))
        if len(set(parsed)) != len(parsed):
            raise RuntimeError(f"CP2K {field} contains duplicate step identifiers{label}")
        return np.asarray(parsed, dtype=np.intp)

    e_steps = _exact_nonnegative_steps(energy_steps, field="energy_steps")
    p_steps = _exact_nonnegative_steps(pressure_steps, field="pressure_steps")
    try:
        p_values = np.asarray(pressures_bar, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"CP2K pressure values must be numeric{label}") from exc

    if p_steps.size != p_values.size:
        raise RuntimeError(
            "CP2K pressure step/value arrays have different lengths"
            f"{label}: {p_steps.size} != {p_values.size}"
        )

    pressure_by_step = {
        int(step): float(value)
        for step, value in zip(p_steps, p_values)
        if math.isfinite(float(value))
    }
    aligned_steps = set(int(step) for step in e_steps).intersection(pressure_by_step)
    if not aligned_steps:
        raise RuntimeError(
            "CP2K NPT output contains no finite pressure sample aligned with "
            f"an energy step{label}"
        )
    missing_positive_steps = sorted(
        int(step)
        for step in set(int(step) for step in e_steps if int(step) > 0)
        if int(step) not in pressure_by_step
    )
    if missing_positive_steps:
        raise RuntimeError(
            "CP2K NPT output is missing finite pressure samples for positive "
            f"energy steps {missing_positive_steps}{label}"
        )
    return pressure_by_step


def parse_cp2k_md_pressures(path: Path) -> np.ndarray:
    """Parse instantaneous NPT pressures, in bar, from CP2K MD output."""

    _steps, pressures = parse_cp2k_md_step_pressures(path)
    return pressures


def _read_cp2k_dcd_last_validated(
    path: Path,
    *,
    ref_atoms,
    aligned: bool,
):
    """Read and validate the final frame of a CP2K DCD trajectory.

    ASE 3.29's direct ``read_cp2k_dcd`` function does not normalize a negative
    integer index before calculating its byte offset.  Its public iterator,
    ``iread_cp2k_dcd``, does normalize negative indices and yields only the
    requested frame, so use that constant-memory path and require exactly one
    result.
    """

    from ase.io.cp2k import iread_cp2k_dcd

    trajectory = Path(path)
    if not trajectory.is_file():
        raise FileNotFoundError(str(trajectory))
    with trajectory.open("rb") as handle:
        frames = iter(
            iread_cp2k_dcd(
                handle,
                indices=-1,
                ref_atoms=ref_atoms,
                aligned=bool(aligned),
            )
        )
        try:
            last = next(frames)
        except StopIteration as exc:
            raise RuntimeError(
                f"No frames read from CP2K DCD file: {trajectory}"
            ) from exc
        try:
            next(frames)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                f"Final-frame CP2K DCD read returned more than one frame: {trajectory}"
            )
    try:
        n_atoms = len(last)
    except TypeError as exc:
        raise RuntimeError(
            f"Final frame in CP2K DCD file is malformed: {trajectory}"
        ) from exc
    if n_atoms <= 0:
        raise RuntimeError(f"Final frame in CP2K DCD file is empty: {trajectory}")
    try:
        _validated_cp2k_atoms_geometry(last, n_symbols=n_atoms)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Final frame in CP2K DCD file has invalid coordinates or cell: {trajectory}"
        ) from exc
    return last


def read_cp2k_dcd_last_aligned(path: Path, *, ref_atoms):
    """Read the final frame of a CP2K ``DCD_ALIGNED_CELL`` trajectory."""

    return _read_cp2k_dcd_last_validated(
        path,
        ref_atoms=ref_atoms,
        aligned=True,
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
    masses: np.ndarray | None = None,
    remove_com: bool = False,
) -> np.ndarray:
    """Mean-squared displacement with optional COM-drift removal.

    The COM displacement is mass-weighted, matching LAMMPS ``compute msd
    ... com yes``.  The per-atom squared displacements themselves retain the
    usual unweighted average.
    """

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
    if bool(remove_com):
        if masses is None:
            weights = np.ones((pos.shape[1],), dtype=float)
        else:
            weights = np.asarray(masses, dtype=float)
            if weights.shape != (pos.shape[1],):
                raise ValueError("masses must have shape (n_atoms,)")
            if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("masses must contain only finite values > 0")
        com_dr = np.sum(dr * weights[None, :, None], axis=1) / float(np.sum(weights))
        dr = dr - com_dr[:, None, :]
    msd = np.mean(np.sum(dr * dr, axis=-1), axis=1)
    return msd
