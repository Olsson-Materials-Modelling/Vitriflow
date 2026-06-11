from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ..analysis.datafile import strip_lammps_data_pair_coeff_sections
from ..analysis.stats import window_mean_stderr
from ..config import BarostatConfig, MDConfig, RunConfig, ThermostatConfig, KimConfig
from ..lammps_input import StageSpec, render_stage
from ..potential import (
    _parse_tabulated_core_spec,
    _sha256_path,
    build_tabulated_buckingham_core_lines,
    kim_init_line,
    kim_interactions_line,
    potential_default_lines,
    potential_init_lines,
    prepare_potential_files,
    update_tabulated_core_metadata_lines,
    write_tabulated_buckingham_core_table,
)
from ..parse import parse_last_thermo_table
from ..runner import Cp2kRunner, LammpsRunner
from .stage_runner import run_stage_local
from ..utils import ExternalCommandError, ensure_dir, scale_steps_for_timestep
from .autoskin import run_with_neighbor_skin_autotune
from .progress import CondensedProgressLog


class PreflightError(RuntimeError):
    """Preflight error."""

    def __init__(self, message: str, candidates: Optional[list[Any]] = None):
        super().__init__(message)
        self.candidates = candidates or []

@dataclass(frozen=True)
class CoreRepulsionResult:
    enabled: bool
    applied: bool
    style: str
    base_pair_style: Optional[str]
    r_inner: Optional[float]
    r_outer: Optional[float]
    attempts: int
    success: bool
    note: str


@dataclass(frozen=True)
class ThermoCandidateResult:
    timestep: float
    ensemble: str
    tdamp: float
    pdamp: Optional[float]
    ok: bool
    score: float
    details: dict[str, Any]


@dataclass(frozen=True)
class PreflightResult:
    selected_ensemble: str
    selected_timestep: float
    selected_tdamp: float
    selected_pdamp: Optional[float]
    # override scripts
    potential_lines: Optional[list[str]]
    core_repulsion: CoreRepulsionResult
    candidates: list[ThermoCandidateResult]


def _clean_dt_candidates(values: Optional[Sequence[Any]]) -> list[float]:
    out: list[float] = []
    if values is None:
        return out
    seen: set[float] = set()
    for x in values:
        try:
            v = float(x)
        except Exception:
            continue
        if v > 0.0 and math.isfinite(v) and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _implicit_dt_fallbacks(dt0: float, *, cp2k: bool = False) -> list[float]:
    vals = [dt0, dt0 / 2.0, dt0 / 4.0] if cp2k else [dt0, dt0 / 2.0, dt0 / 4.0, dt0 / 8.0]
    return _clean_dt_candidates(vals)


def _core_dt_candidates(config: RunConfig) -> tuple[list[float], str]:
    pf = config.autotune.preflight
    core = getattr(config.kim, "core_repulsion", None)
    md0 = float(config.md.timestep)
    pf_explicit = _clean_dt_candidates(getattr(pf, "dt_candidates", None))
    core_explicit = _clean_dt_candidates(getattr(core, "dt_candidates", None) if core is not None else None)

    if pf_explicit and core_explicit:
        inter = [x for x in core_explicit if x in set(pf_explicit)]
        if not inter:
            raise ValueError(
                "Explicit autotune.preflight.dt_candidates and potential.core_repulsion.dt_candidates are disjoint; "
                "core preflight cannot select an undeclared timestep."
            )
        return sorted(inter, reverse=True), "intersection(autotune.preflight.dt_candidates, potential.core_repulsion.dt_candidates)"
    if core_explicit:
        return sorted(core_explicit, reverse=True), "potential.core_repulsion.dt_candidates"
    if pf_explicit:
        return sorted(pf_explicit, reverse=True), "autotune.preflight.dt_candidates"
    if bool(getattr(pf, "allow_implicit_dt_fallback", False)):
        return sorted(_implicit_dt_fallbacks(md0), reverse=True), "implicit_dt_fallback"
    return [md0], "md.timestep"


def _preflight_dt_candidates(config: RunConfig, *, dt_cap: Optional[float] = None, cp2k: bool = False) -> tuple[list[float], str]:
    pf = config.autotune.preflight
    core = getattr(config.kim, "core_repulsion", None)
    md0 = float(config.md.timestep)
    pf_explicit = _clean_dt_candidates(getattr(pf, "dt_candidates", None))
    core_explicit = _clean_dt_candidates(getattr(core, "dt_candidates", None) if (core is not None and not cp2k) else None)

    if pf_explicit:
        vals = pf_explicit
        source = "autotune.preflight.dt_candidates"
    elif core_explicit:
        vals = core_explicit
        source = "potential.core_repulsion.dt_candidates"
    elif bool(getattr(pf, "allow_implicit_dt_fallback", False)):
        vals = _implicit_dt_fallbacks(md0, cp2k=cp2k)
        source = "implicit_dt_fallback"
    else:
        vals = [md0]
        source = "md.timestep"

    vals = _clean_dt_candidates(vals)
    if dt_cap is not None:
        cap = float(dt_cap)
        vals = [x for x in vals if x <= cap + 1.0e-15]
    vals = sorted(vals, reverse=True)
    return vals, source


def _atomic_number(symbol: str) -> int:
    sym = str(symbol).strip()
    try:
        from ase.data import atomic_numbers

        return int(atomic_numbers[sym])
    except Exception:  # pragma: no cover
        _fallback = {"H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Si": 14, "Fe": 26}
        if sym not in _fallback:
            raise KeyError(f"Unknown atomic number for symbol '{sym}'")
        return _fallback[sym]


def _read_nn_median_from_datafile(data_path: Path, *, atom_style: str) -> float:
    """Nn median from."""
    try:
        from ase.io import read as ase_read
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ASE is required for preflight distance estimation") from e

    # style ase version
    # fall back
    try:
        atoms = ase_read(str(data_path), format="lammps-data", style=str(atom_style))
    except Exception:
        atoms = ase_read(str(data_path), format="lammps-data")

    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    nn = np.min(d, axis=1)
    nn = nn[np.isfinite(nn) & (nn > 1.0e-8)]
    if len(nn) == 0:
        raise ValueError(f"Failed to compute nearest-neighbour distances from {data_path}")
    return float(np.median(nn))


def _extract_kim_interactions_block(log_path: Path) -> list[str]:
    """Extract kim interactions."""
    lines = log_path.read_text(errors="replace").splitlines()
    low = [ln.lower() for ln in lines]

    def clean(ln: str) -> str:
        if "#" in ln:
            ln = ln.split("#", 1)[0]
        return ln.strip()

    b = None
    e = None
    for i, ln in enumerate(low):
        if "begin kim interactions" in ln:
            b = i
        if b is not None and "end kim interactions" in ln:
            e = i
            break
    allowed_heads = {
        "pair_style",
        "pair_coeff",
        "kspace_style",
        "kspace_modify",
        "pair_modify",
        "special_bonds",
        "neighbor",
        "neigh_modify",
        "dielectric",
        "set",
    }

    if b is not None and e is not None and e > b:
        # kim interaction lammps
        # commands output produced
        # atom look valid
        # lammps commands
        out: list[str] = []
        for ln in lines[b + 1 : e]:
            c = clean(ln)
            if not c:
                continue
            head = c.split()[0].lower()
            if head in allowed_heads:
                out.append(c)
        if out:
            return out

    # fallback
    ki = None
    for i, ln in enumerate(low):
        if "kim interactions" in ln:
            ki = i
    if ki is None:
        raise ValueError(f"No 'kim interactions' marker found in log: {log_path}")

    out: list[str] = []
    for ln in lines[ki + 1 :]:
        if ln.strip().startswith("Step "):
            break
        c = clean(ln)
        if not c:
            continue
        head = c.split()[0].lower()
        if head in allowed_heads:
            out.append(c)

    if not out:
        raise ValueError(f"Could not extract KIM-generated commands from log: {log_path}")
    return out


def _localize_input_data_for_preflight(*, source: Path, destination: Path) -> Path:
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(source).read_bytes())
    strip_lammps_data_pair_coeff_sections(dst)
    return dst


def _kim_extract_commands(runner: LammpsRunner, config: RunConfig, input_data: Path, outdir: Path) -> list[str]:
    """Kim extract commands."""
    stage_dir = outdir / "preflight" / "kim_extract"
    ensure_dir(stage_dir)

    _localize_input_data_for_preflight(source=input_data, destination=stage_dir / "input.data")

    script = f"""# vitriflow preflight: extract KIM-generated interaction commands
{kim_init_line(config.kim)}
atom_style {config.md.atom_style}
boundary p p p
atom_modify map array

read_data input.data

{kim_interactions_line(config.kim)}

run 0
"""
    runner.run(script, stage_dir, "log.lammps", timeout_sec=float(config.autotune.preflight.timeout_sec))
    cmds = _extract_kim_interactions_block(stage_dir / "log.lammps")
    (stage_dir / "kim_generated_commands.txt").write_text("\n".join(cmds) + "\n")
    return cmds


def _rewrite_for_hybrid_overlay_zbl(
    base_cmds: list[str],
    *,
    species: list[str],
    r_in: float,
    r_out: float,
) -> tuple[list[str], str]:
    """Rewrite for hybrid."""
    pair_style_line = None
    for ln in base_cmds:
        if ln.strip().startswith("pair_style"):
            pair_style_line = ln.strip()
            break
    if pair_style_line is None:
        raise ValueError("No pair_style line found in KIM-generated command block")

    toks = pair_style_line.split()
    if len(toks) < 2:
        raise ValueError(f"Malformed pair_style line: {pair_style_line}")
    base_style = toks[1]
    base_args = toks[2:]

    if "buck" not in base_style.lower():
        raise ValueError(f"Base pair_style is not Buckingham-like: {base_style}")

    base_coeff: list[str] = []
    other: list[str] = []
    for ln in base_cmds:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("pair_style"):
            continue
        if s.startswith("pair_coeff"):
            tt = s.split()
            # hybrid syntax inserted
            if len(tt) >= 4 and tt[3] == base_style:
                base_coeff.append(s)
            else:
                base_coeff.append(" ".join(tt[:3] + [base_style] + tt[3:]))
            continue
        other.append(s)

    hybrid_pair_style = "pair_style hybrid/overlay " + " ".join(
        [base_style] + base_args + ["zbl", f"{r_in:.6g}", f"{r_out:.6g}"]
    )

    zbl_coeff: list[str] = []
    Z = [_atomic_number(s) for s in species]
    ntypes = len(Z)
    for i in range(1, ntypes + 1):
        for j in range(i, ntypes + 1):
            zbl_coeff.append(f"pair_coeff {i} {j} zbl {Z[i-1]} {Z[j-1]}")

    # order pair remaining
    return [hybrid_pair_style] + base_coeff + zbl_coeff + other, base_style


def _rewrite_for_hybrid_overlay_lj_repulsive(
    base_cmds: list[str],
    *,
    species: list[str],
    r_in: float,
    r_out: float,
    u_target_eV: float,
) -> tuple[list[str], str]:
    """Rewrite for hybrid."""
    pair_style_line = None
    for ln in base_cmds:
        if ln.strip().startswith("pair_style"):
            pair_style_line = ln.strip()
            break
    if pair_style_line is None:
        raise ValueError("No pair_style line found in KIM-generated command block")

    toks = pair_style_line.split()
    if len(toks) < 2:
        raise ValueError(f"Malformed pair_style line: {pair_style_line}")
    base_style = toks[1]
    base_args = toks[2:]

    if "buck" not in base_style.lower():
        raise ValueError(f"Base pair_style is not Buckingham-like: {base_style}")

    # prepare coefficients hybrid
    base_coeff: list[str] = []
    other: list[str] = []
    for ln in base_cmds:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("pair_style"):
            continue
        if s.startswith("pair_coeff"):
            tt = s.split()
            if len(tt) >= 4 and tt[3] == base_style:
                base_coeff.append(s)
            else:
                base_coeff.append(" ".join(tt[:3] + [base_style] + tt[3:]))
            continue
        other.append(s)

    # repulsive minimum force
    if r_out <= 0 or r_in <= 0 or r_in >= r_out:
        raise ValueError(f"Invalid LJ core radii: r_in={r_in}, r_out={r_out}")
    sigma = float(r_out) / (2.0 ** (1.0 / 6.0))
    x = float(sigma) / float(r_in)
    denom = 4.0 * (x ** 12 - x ** 6) + 1.0
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError("Failed to compute LJ calibration denominator")
    eps = float(u_target_eV) / float(denom)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid LJ epsilon computed")

    hybrid_pair_style = "pair_style hybrid/overlay " + " ".join(
        [base_style] + base_args + ["lj/cut", f"{r_out:.6g}"]
    )

    lj_coeff: list[str] = []
    ntypes = len(species)
    for i in range(1, ntypes + 1):
        for j in range(i, ntypes + 1):
            lj_coeff.append(f"pair_coeff {i} {j} lj/cut {eps:.6g} {sigma:.6g}")

    # order pair remaining
    return [hybrid_pair_style, "pair_modify shift yes"] + base_coeff + lj_coeff + other, base_style


def _run_stability_test(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    *,
    outdir: Path,
    potential_lines: list[str],
    temperature: float,
    timestep: float,
    label: str = "",
) -> bool:
    """Stability test."""

    # unique directory preserve
    stage_dir = _core_stability_stage_dir(outdir, temperature=temperature, label=label)
    ensure_dir(stage_dir)
    _localize_input_data_for_preflight(source=input_data, destination=stage_dir / "input.data")

    dt = float(timestep)
    Te = float(temperature)

    core = config.kim.core_repulsion
    pf = config.autotune.preflight

    # thermostat time constants
    tdamp = max(float(config.md.thermostat.tdamp), 100.0 * dt)
    langevin_damp = float(core.langevin_damp) if core.langevin_damp is not None else max(100.0 * dt, 0.05)

    # conservative metal distance
    limit_max_disp = float(getattr(core, "limit_max_disp", 0.02))
    ramp_steps = int(getattr(core, "ramp_steps", 5000))
    limit_hold_steps = int(getattr(core, "limit_hold_steps", 2000))

    # important preserve physical
    # otherwise smaller candidates
    # appear spuriously stable
    dt_ref = float(config.md.timestep)
    ramp_steps = scale_steps_for_timestep(ramp_steps, dt_ref, dt, min_steps=1)
    limit_hold_steps = scale_steps_for_timestep(limit_hold_steps, dt_ref, dt, min_steps=1)

    # stability highest scan
    nvt_steps = int(core.test_equil_steps) + int(core.test_run_steps)
    try:
        nvt_steps = max(nvt_steps, int(config.autotune.tm_scan.equil_steps) + int(config.autotune.tm_scan.sample_steps))
    except Exception:
        pass
    nvt_steps = max(2000, int(nvt_steps))
    nvt_steps = scale_steps_for_timestep(int(nvt_steps), dt_ref, dt, min_steps=2000)

    pot_block = "\n".join([str(x).strip() for x in potential_lines if str(x).strip()])
    T0 = float(getattr(pf, "T_low", None) or float(config.md.temperature) or 300.0)
    if T0 <= 0.0 or not math.isfinite(T0):
        T0 = 300.0
    T0 = min(T0, Te)

    # deterministic decouples timestep
    # pathologies volume neighbor
    # thermo scan confirmation
    final_hold = f"""# Nose-Hoover NVT hold
fix int all nvt temp {Te} {Te} {tdamp}
run {nvt_steps}
unfix int"""

    # auxiliary potential present
    prepare_potential_files(config.kim, stage_dir, potential_lines)

    init_block = "\n".join(potential_init_lines(config.kim))

    def _build_script(md_use: MDConfig) -> str:
        return f"""# vitriflow preflight: core stability (high T)

    {init_block}
    atom_style {config.md.atom_style}
    boundary p p p
    atom_modify map array

    read_data input.data

    # potential force field
    {pot_block}

    # conservative neighbour stability
    neighbor {md_use.neighbor_skin} bin
    neigh_modify every 1 delay 0 check yes

    timestep {dt}

    thermo_style custom step time temp press pe ke etotal vol density lx ly lz
    thermo {max(50, int(config.md.thermo_every))}
    thermo_modify flush yes

    # minimisation relieve contacts
    min_style cg
    minimize 1.0e-6 1.0e-8 500 5000

    # velocities low ramp
    velocity all create {T0} 12345 mom yes rot yes dist gaussian

    # langevin displacement limiting
    fix lang all langevin {T0} {Te} {langevin_damp} 12345
    fix lim all nve/limit {limit_max_disp}
    run {ramp_steps}
    unfix lim
    unfix lang

    # short limited langevin
    fix lang2 all langevin {Te} {Te} {langevin_damp} 54321
    fix lim2 all nve/limit {limit_max_disp}
    run {limit_hold_steps}
    unfix lim2
    unfix lang2

    {final_hold}

    write_data output.data nocoeff
    """

    try:
        run_with_neighbor_skin_autotune(
            runner,
            _build_script,
            stage_dir,
            config.md,
            log_name="log.lammps",
            timeout_sec=float(config.autotune.preflight.timeout_sec),
            cleanup_paths=[stage_dir / "output.data"],
        )
    except Exception:
        return False

    # sanity thermo table
    try:
        tbl = parse_last_thermo_table(stage_dir / "log.lammps").as_dict()
        T = np.asarray(tbl.get("Temp", []), dtype=float)
        if len(T) == 0 or not np.all(np.isfinite(T)):
            return False
        if float(np.nanmax(T)) > float(pf.max_temp_factor) * float(temperature):
            return False
        P = np.asarray(tbl.get("Press", []), dtype=float)
        if len(P) > 0 and np.any(np.isfinite(P)):
            if float(np.nanmax(np.abs(P))) > float(pf.max_press_abs):
                return False
    except Exception:
        return False
    return True


def _core_stability_stage_dir(outdir: Path, *, temperature: float, label: str = "") -> Path:
    safe = str(label).strip()
    if safe:
        safe = safe.replace(".", "p").replace("-", "m").replace("+", "")
        safe = "_" + safe
    return Path(outdir) / "preflight" / f"core_stability_T{int(round(float(temperature)))}{safe}"


def _parse_gewald_from_log(log_path: Path) -> Optional[float]:
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    patterns = [
        r"G\s+vector(?:\s*\([^)]*\))?\s*=\s*([0-9Ee+\-.]+)",
        r"g[_-]?ewald\s*=\s*([0-9Ee+\-.]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _resolve_tabulated_gewald(
    config: RunConfig,
    base_cmds: Sequence[str],
    *,
    stage_log: Optional[Path] = None,
) -> Optional[float]:
    core = config.kim.core_repulsion
    explicit = getattr(core, "table_gewald", None)
    if explicit is not None:
        return float(explicit)
    for ln in base_cmds:
        toks = str(ln).split()
        if len(toks) < 3 or toks[0] != "kspace_modify":
            continue
        i = 1
        while i < len(toks) - 1:
            if toks[i].lower() == "gewald":
                return float(toks[i + 1])
            i += 1
    if stage_log is not None:
        val = _parse_gewald_from_log(stage_log)
        if val is not None:
            return val
        try:
            screen = Path(stage_log).with_name("screen.out")
        except Exception:
            screen = None
        if screen is not None:
            return _parse_gewald_from_log(screen)
    return None


def _parse_pair_table_file(path: Path) -> dict[str, dict[str, np.ndarray]]:
    lines = Path(path).read_text(errors="ignore").splitlines()
    out: dict[str, dict[str, np.ndarray]] = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        section = s.split()[0]
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        hdr = lines[i].split()
        if len(hdr) < 2 or hdr[0] != "N":
            raise ValueError(f"Malformed pair_write/table section header for {section!r} in {path}")
        n_expected = int(hdr[1])
        i += 1
        rows_r: list[float] = []
        rows_u: list[float] = []
        rows_f: list[float] = []
        while i < len(lines):
            row = lines[i].strip()
            if not row:
                i += 1
                if len(rows_r) >= n_expected:
                    break
                continue
            if row.startswith("#"):
                i += 1
                continue
            toks = row.split()
            if len(toks) >= 4 and toks[0].isdigit():
                rows_r.append(float(toks[1]))
                rows_u.append(float(toks[2]))
                rows_f.append(float(toks[3]))
                i += 1
                if len(rows_r) >= n_expected:
                    break
                continue
            break
        if len(rows_r) != n_expected:
            raise ValueError(
                f"Expected {n_expected} rows for pair-table section {section!r} in {path}, found {len(rows_r)}"
            )
        out[section] = {
            "r": np.asarray(rows_r, dtype=float),
            "energy": np.asarray(rows_u, dtype=float),
            "force": np.asarray(rows_f, dtype=float),
        }
    if not out:
        raise ValueError(f"No pair-table sections found in {path}")
    return out


_TABLE_WARNING_PATTERNS = (
    "force values in table",
    "distance values in table",
    "dE/dr",
    "tabulated",
)


def _table_warning_lines(stage_dir: Path) -> list[str]:
    msgs: list[str] = []
    seen: set[str] = set()
    for name in ("screen.out", "stdout.txt", "stderr.txt", "log.lammps"):
        path = Path(stage_dir) / name
        if not path.exists():
            continue
        for raw in path.read_text(errors="ignore").splitlines():
            s = raw.strip()
            low = s.lower()
            if "warning" not in low:
                continue
            if not any(pat.lower() in low for pat in _TABLE_WARNING_PATTERNS):
                continue
            if s not in seen:
                seen.add(s)
                msgs.append(s)
    return msgs


def _render_pair_write_script(
    config: RunConfig,
    *,
    potential_lines: Sequence[str],
    spec: dict[str, Any],
    npoints: int,
    output_name: str,
) -> str:
    ntypes = len(list(config.kim.interactions))
    units_style = str(config.kim.user_units).strip() or "metal"
    atom_style = str(config.md.atom_style).strip() or "atomic"
    lines: list[str] = [
        "# vitriflow preflight: pair_write potential verification",
        f"units {units_style}",
        "dimension 3",
        f"atom_style {atom_style}",
        "boundary p p p",
        "region box block 0 20 0 20 0 20",
        f"create_box {ntypes} box",
    ]
    for i in range(1, ntypes + 1):
        lines.append(f"mass {i} 1.0")
    lines.append("")
    lines.extend(str(x) for x in potential_lines)
    gewald = spec.get("gewald", None)
    if gewald is not None:
        # pair charge box
        # fixed defines tabulated
        # reference generated identical
        lines.append(f"kspace_modify gewald {float(gewald):.16g}")
    lines.append("")
    rmin = float(spec["r_min"])
    for pair in list(spec.get("pairs", [])):
        i, j = int(pair["pair"][0]), int(pair["pair"][1])
        pair_cut = float(pair["pair_cutoff"])
        cmd = f"pair_write {i} {j} {int(npoints)} rsq {rmin:.16g} {pair_cut:.16g} {output_name} {pair['section']}"
        if pair.get("q_i", None) is not None and pair.get("q_j", None) is not None:
            cmd += f" {float(pair['q_i']):.16g} {float(pair['q_j']):.16g}"
        lines.append(cmd)
    lines.append("")
    return "\n".join(lines) + "\n"



def _pair_write_potential_curves(
    runner: LammpsRunner,
    config: RunConfig,
    *,
    stage_dir: Path,
    potential_lines: Sequence[str],
    spec: dict[str, Any],
    npoints: int,
    output_name: str,
    log_name: str,
) -> dict[str, Any]:
    stage_dir = Path(stage_dir)
    ensure_dir(stage_dir)
    out_path = stage_dir / output_name
    try:
        out_path.unlink()
    except Exception:
        pass
    prepare_potential_files(config.kim, stage_dir, potential_lines)
    script = _render_pair_write_script(
        config,
        potential_lines=potential_lines,
        spec=spec,
        npoints=int(npoints),
        output_name=str(Path(output_name).name),
    )
    pairwrite_runner = runner
    try:
        if getattr(runner.cfg, "mpi_cmd", None):
            pairwrite_runner = LammpsRunner(runner.cfg.model_copy(update={"nprocs": 1}))
    except Exception:
        pairwrite_runner = runner
    pairwrite_runner.run(script, stage_dir, log_name, timeout_sec=float(config.autotune.preflight.timeout_sec))
    if not out_path.exists():
        raise PreflightError(f"pair_write did not create expected output file: {out_path}")
    return {
        "path": out_path,
        "sections": _parse_pair_table_file(out_path),
        "warnings": _table_warning_lines(stage_dir),
    }




def _compare_pair_table_sections(
    reference: dict[str, dict[str, np.ndarray]],
    realized: dict[str, dict[str, np.ndarray]],
    *,
    rel_tol: float,
    abs_tol_frac: float,
) -> dict[str, Any]:
    pair_reports: dict[str, Any] = {}
    overall_energy_ratio = 0.0
    overall_force_ratio = 0.0
    overall_r_error = 0.0
    worst_energy_section: Optional[str] = None
    worst_force_section: Optional[str] = None
    passed = True
    missing = sorted(set(reference) ^ set(realized))
    if missing:
        return {"passed": False, "missing_sections": missing, "pairs": {}, "overall": {}}
    for section in sorted(reference):
        ref = reference[section]
        got = realized[section]
        r_ref = np.asarray(ref["r"], dtype=float)
        r_got = np.asarray(got["r"], dtype=float)
        if r_ref.shape != r_got.shape:
            return {
                "passed": False,
                "missing_sections": [],
                "pairs": pair_reports,
                "overall": {"shape_mismatch": {section: [int(r_ref.size), int(r_got.size)]}},
            }
        r_err = float(np.max(np.abs(r_got - r_ref))) if r_ref.size else 0.0
        overall_r_error = max(overall_r_error, r_err)
        u_ref = np.asarray(ref["energy"], dtype=float)
        u_got = np.asarray(got["energy"], dtype=float)
        f_ref = np.asarray(ref["force"], dtype=float)
        f_got = np.asarray(got["force"], dtype=float)
        du = u_got - u_ref
        df = f_got - f_ref
        u_scale = max(1.0, float(np.max(np.abs(u_ref))) if u_ref.size else 0.0)
        f_scale = max(1.0, float(np.max(np.abs(f_ref))) if f_ref.size else 0.0)
        u_allow = float(abs_tol_frac) * u_scale + float(rel_tol) * np.abs(u_ref)
        f_allow = float(abs_tol_frac) * f_scale + float(rel_tol) * np.abs(f_ref)
        u_ratio_arr = np.abs(du) / u_allow if u_ref.size else np.zeros(0, dtype=float)
        f_ratio_arr = np.abs(df) / f_allow if f_ref.size else np.zeros(0, dtype=float)
        u_ratio = float(np.max(u_ratio_arr)) if u_ratio_arr.size else 0.0
        f_ratio = float(np.max(f_ratio_arr)) if f_ratio_arr.size else 0.0
        overall_energy_ratio = max(overall_energy_ratio, u_ratio)
        overall_force_ratio = max(overall_force_ratio, f_ratio)
        if u_ratio >= overall_energy_ratio:
            worst_energy_section = section
        if f_ratio >= overall_force_ratio:
            worst_force_section = section
        u_mask = np.abs(u_ref) > float(abs_tol_frac) * u_scale
        f_mask = np.abs(f_ref) > float(abs_tol_frac) * f_scale
        max_rel_u = float(np.max(np.abs(du[u_mask]) / np.abs(u_ref[u_mask]))) if np.any(u_mask) else 0.0
        max_rel_f = float(np.max(np.abs(df[f_mask]) / np.abs(f_ref[f_mask]))) if np.any(f_mask) else 0.0
        idx_u = int(np.argmax(u_ratio_arr)) if u_ratio_arr.size else 0
        idx_f = int(np.argmax(f_ratio_arr)) if f_ratio_arr.size else 0
        n_energy_fail = int(np.count_nonzero(u_ratio_arr > 1.0))
        n_force_fail = int(np.count_nonzero(f_ratio_arr > 1.0))
        pair_ok = bool((u_ratio <= 1.0) and (f_ratio <= 1.0) and (r_err <= 1.0e-12))
        passed = passed and pair_ok
        pair_reports[section] = {
            "passed": pair_ok,
            "max_abs_energy_error": float(np.max(np.abs(du))) if du.size else 0.0,
            "max_abs_force_error": float(np.max(np.abs(df))) if df.size else 0.0,
            "max_rel_energy_error": max_rel_u,
            "max_rel_force_error": max_rel_f,
            "max_energy_ratio": u_ratio,
            "max_force_ratio": f_ratio,
            "max_r_error": r_err,
            "n_points": int(r_ref.size),
            "n_energy_fail": n_energy_fail,
            "n_force_fail": n_force_fail,
            "energy_fail_fraction": (float(n_energy_fail) / float(r_ref.size)) if r_ref.size else 0.0,
            "force_fail_fraction": (float(n_force_fail) / float(r_ref.size)) if r_ref.size else 0.0,
            "worst_energy_point": {
                "index": idx_u,
                "r": float(r_ref[idx_u]) if r_ref.size else 0.0,
                "ref_energy": float(u_ref[idx_u]) if u_ref.size else 0.0,
                "got_energy": float(u_got[idx_u]) if u_got.size else 0.0,
                "abs_error": float(abs(du[idx_u])) if du.size else 0.0,
                "allow": float(u_allow[idx_u]) if u_allow.size else 0.0,
            },
            "worst_force_point": {
                "index": idx_f,
                "r": float(r_ref[idx_f]) if r_ref.size else 0.0,
                "ref_force": float(f_ref[idx_f]) if f_ref.size else 0.0,
                "got_force": float(f_got[idx_f]) if f_got.size else 0.0,
                "abs_error": float(abs(df[idx_f])) if df.size else 0.0,
                "allow": float(f_allow[idx_f]) if f_allow.size else 0.0,
            },
        }
    return {
        "passed": bool(passed),
        "missing_sections": [],
        "pairs": pair_reports,
        "overall": {
            "max_energy_ratio": overall_energy_ratio,
            "max_force_ratio": overall_force_ratio,
            "max_r_error": overall_r_error,
            "worst_energy_section": worst_energy_section,
            "worst_force_section": worst_force_section,
        },
    }



def _table_force_energy_consistency_report(
    sections: dict[str, dict[str, np.ndarray]],
    *,
    rel_tol: float,
    abs_tol_frac: float,
) -> dict[str, Any]:
    """Table force energy."""
    pair_reports: dict[str, Any] = {}
    overall_ratio = 0.0
    worst_section: Optional[str] = None
    passed = True
    for section in sorted(sections):
        data = sections[section]
        r = np.asarray(data.get("r", []), dtype=float)
        u = np.asarray(data.get("energy", []), dtype=float)
        f = np.asarray(data.get("force", []), dtype=float)
        if r.size < 3 or u.size != r.size or f.size != r.size:
            pair_reports[section] = {
                "passed": bool(r.size == u.size == f.size and r.size >= 2),
                "n_points": int(r.size),
                "max_force_ratio": 0.0,
                "max_abs_force_error": 0.0,
                "n_fail": 0,
                "fail_fraction": 0.0,
                "note": "insufficient points for interior finite-difference check",
            }
            continue
        dU = np.gradient(u, r, edge_order=2)
        f_from_u = -np.asarray(dU, dtype=float)
        df = f - f_from_u
        scale = max(1.0, float(np.max(np.abs(f))) if f.size else 0.0, float(np.max(np.abs(f_from_u))) if f_from_u.size else 0.0)
        allow = float(abs_tol_frac) * scale + float(rel_tol) * np.abs(f_from_u)
        ratio_arr = np.abs(df) / allow if df.size else np.zeros(0, dtype=float)
        ratio = float(np.max(ratio_arr)) if ratio_arr.size else 0.0
        overall_ratio = max(overall_ratio, ratio)
        if ratio >= overall_ratio:
            worst_section = section
        mask = np.abs(f_from_u) > float(abs_tol_frac) * scale
        idx = int(np.argmax(ratio_arr)) if ratio_arr.size else 0
        n_fail = int(np.count_nonzero(ratio_arr > 1.0))
        pair_ok = bool(ratio <= 1.0)
        passed = passed and pair_ok
        pair_reports[section] = {
            "passed": pair_ok,
            "n_points": int(r.size),
            "max_abs_force_error": float(np.max(np.abs(df))) if df.size else 0.0,
            "max_rel_force_error": float(np.max(np.abs(df[mask]) / np.abs(f_from_u[mask]))) if np.any(mask) else 0.0,
            "max_force_ratio": ratio,
            "n_fail": n_fail,
            "fail_fraction": (float(n_fail) / float(r.size)) if r.size else 0.0,
            "worst_point": {
                "index": idx,
                "r": float(r[idx]) if r.size else 0.0,
                "table_force": float(f[idx]) if f.size else 0.0,
                "energy_derived_force": float(f_from_u[idx]) if f_from_u.size else 0.0,
                "abs_error": float(abs(df[idx])) if df.size else 0.0,
                "allow": float(allow[idx]) if allow.size else 0.0,
            },
        }
    return {
        "passed": bool(passed),
        "pairs": pair_reports,
        "overall": {"max_force_ratio": overall_ratio, "worst_section": worst_section},
    }


def _materialize_pairwrite_tabulated_core_source(
    runner: LammpsRunner,
    config: RunConfig,
    *,
    outdir: Path,
    source_potential_lines: Sequence[str],
    spec: dict[str, Any],
) -> Path:
    potdir = Path(outdir) / "preflight" / "potential_override"
    ensure_dir(potdir)
    fname = Path(str(spec.get("filename", "")).strip() or "buckingham_core.table").name
    data = _pair_write_potential_curves(
        runner,
        config,
        stage_dir=potdir,
        potential_lines=source_potential_lines,
        spec=spec,
        npoints=int(spec["points"]),
        output_name=fname,
        log_name="log_pairwrite_source.lammps",
    )
    if data["warnings"]:
        raise PreflightError(
            "analytic source pair_write emitted table-related warnings: " + "; ".join(data["warnings"])
        )
    return Path(data["path"])



def _verify_tabulated_core_against_source(
    runner: LammpsRunner,
    config: RunConfig,
    *,
    outdir: Path,
    table_potential_lines: Sequence[str],
    reference_sections: dict[str, dict[str, np.ndarray]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    verify_dir = Path(outdir) / "preflight" / "table_verify"
    ensure_dir(verify_dir)
    core = config.kim.core_repulsion
    realized = _pair_write_potential_curves(
        runner,
        config,
        stage_dir=verify_dir / "table",
        potential_lines=table_potential_lines,
        spec=spec,
        npoints=int(core.table_verify_points),
        output_name="realized.table",
        log_name="log_pairwrite_table.lammps",
    )
    cmp = _compare_pair_table_sections(
        reference_sections,
        realized["sections"],
        rel_tol=float(core.table_verify_rel_tol),
        abs_tol_frac=float(core.table_verify_abs_tol_frac),
    )
    warnings = list(realized["warnings"])
    self_consistency = _table_force_energy_consistency_report(
        realized["sections"],
        rel_tol=float(core.table_verify_rel_tol),
        abs_tol_frac=float(core.table_verify_abs_tol_frac),
    )
    ok = bool(cmp.get("passed", False))
    if bool(getattr(core, "table_require_warning_free", True)) and warnings:
        ok = False
    report = {
        "passed": ok,
        "warnings": warnings,
        "comparison": cmp,
        "self_consistency": self_consistency,
        "verify_points": int(core.table_verify_points),
        "rel_tol": float(core.table_verify_rel_tol),
        "abs_tol_frac": float(core.table_verify_abs_tol_frac),
    }
    (verify_dir / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report



def _materialize_generated_tabulated_core_source(
    *,
    outdir: Path,
    spec: dict[str, Any],
) -> Path:
    potdir = Path(outdir) / "preflight" / "potential_override"
    ensure_dir(potdir)
    fname = Path(str(spec.get("filename", "")).strip() or "buckingham_core.table").name
    out = potdir / fname
    write_tabulated_buckingham_core_table(out, spec)
    return out


def _tabulation_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    comparison = dict((candidate.get("comparison") or {}).get("overall", {}) or {})
    warnings = list(candidate.get("warnings", []) or [])
    verify_ok = bool(candidate.get("verify_passed", False))
    stability_ok = bool(candidate.get("stability_ok", False))
    max_energy = float(comparison.get("max_energy_ratio", float("inf")))
    max_force = float(comparison.get("max_force_ratio", float("inf")))
    max_self = float(((candidate.get("self_consistency") or {}).get("overall", {}) or {}).get("max_force_ratio", float("inf")))
    penalty = 0.0
    if not verify_ok:
        penalty += 1.0e6
    if not stability_ok:
        penalty += 1.0e5
    penalty += 1.0e3 * float(len(warnings))
    return (
        penalty + max(max_energy, max_force, max_self),
        max(max_energy, max_force, max_self),
        max_energy + max_force + max_self,
        float(candidate.get("table_points", float("inf"))),
        0.0 if str(candidate.get("force_mode", "")) == "fd_consistent" else 1.0,
    )


def _format_tabulation_candidate_summary(candidate: Mapping[str, Any]) -> str:
    comparison = dict((candidate.get("comparison") or {}).get("overall", {}) or {})
    warnings = list(candidate.get("warnings", []) or [])
    return (
        f"mode={candidate.get('force_mode', '?')}, "
        f"table_points={candidate.get('table_points', '?')}, "
        f"verify={'pass' if bool(candidate.get('verify_passed', False)) else 'fail'}, "
        f"stability={'pass' if bool(candidate.get('stability_ok', False)) else 'fail'}, "
        f"max_energy_ratio={float(comparison.get('max_energy_ratio', float('nan'))):.3g}, "
        f"max_force_ratio={float(comparison.get('max_force_ratio', float('nan'))):.3g}, "
        f"warnings={len(warnings)}"
    )


def _write_tabulation_refinement_report(
    *,
    outdir: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    verify_dir = Path(outdir) / "preflight" / "table_verify"
    ensure_dir(verify_dir)
    json_path = verify_dir / "refinement_report.json"
    txt_path = verify_dir / "refinement_summary.txt"
    json_path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
    summary_lines: list[str] = []
    summary_lines.append(f"status: {report.get('status', 'unknown')}")
    summary_lines.append(f"fallback_to_analytic: {bool(report.get('fallback_to_analytic', False))}")
    best = report.get("best_candidate", None)
    if isinstance(best, Mapping):
        summary_lines.append("best_candidate: " + _format_tabulation_candidate_summary(best))
    accepted = report.get("accepted_candidate", None)
    if isinstance(accepted, Mapping):
        summary_lines.append("accepted_candidate: " + _format_tabulation_candidate_summary(accepted))
    reason = str(report.get("reason", "") or "").strip()
    if reason:
        summary_lines.append("reason: " + reason)
    cand_list = list(report.get("candidates", []) or [])
    if cand_list:
        summary_lines.append("candidates:")
        for cand in cand_list:
            if isinstance(cand, Mapping):
                summary_lines.append("  - " + _format_tabulation_candidate_summary(cand))
                err = str(cand.get("error", "") or "").strip()
                if err:
                    summary_lines.append("    error: " + err)
    txt_path.write_text("\n".join(summary_lines) + "\n")
    return json_path, txt_path


def _append_condensed_preflight_warning(outdir: Path, message: str) -> None:
    try:
        CondensedProgressLog(Path(outdir) / "condensed.log").warn("preflight", str(message))
    except Exception:
        pass


def _append_condensed_preflight_info(outdir: Path, message: str) -> None:
    try:
        CondensedProgressLog(Path(outdir) / "condensed.log").info("preflight", str(message))
    except Exception:
        pass


def _append_preflight_warning(outdir: Path, warnings_list: list[str], message: str) -> None:
    msg = str(message).strip()
    if not msg:
        return
    if msg not in warnings_list:
        warnings_list.append(msg)
    _append_condensed_preflight_warning(outdir, msg)


def _fmt_dt_values(values: Sequence[float]) -> str:
    vals = [float(x) for x in values]
    return "[" + ", ".join(f"{v:g}" for v in vals) + "]"


def _make_timestep_fallback_warning(
    *,
    context: str,
    source: str,
    tried: Sequence[float],
    selected: Optional[float],
) -> str:
    tried_vals = [float(x) for x in tried]
    tried_txt = _fmt_dt_values(tried_vals)
    if selected is None:
        return (
            f"{context}: timestep fallback activated; source={source}; tried={tried_txt}; "
            "no stable candidate passed"
        )
    return (
        f"{context}: timestep fallback activated; source={source}; tried={tried_txt}; "
        f"selected dt={float(selected):g} after larger candidate(s) failed"
    )


def _tabulated_force_modes() -> tuple[str, ...]:
    return ("analytic", "fd_consistent")

def _maybe_apply_core_repulsion(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    outdir: Path,
    *,
    T_test: float,
) -> tuple[Optional[list[str]], CoreRepulsionResult, float]:
    """Maybe apply core."""
    core = config.kim.core_repulsion
    dt0 = float(config.md.timestep)

    # disabled return immediately
    if not core.enabled:
        return (
            None,
            CoreRepulsionResult(
                enabled=False,
                applied=False,
                style=str(core.style),
                base_pair_style=None,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=True,
                note="disabled",
            ),
            dt0,
        )

    try:
        if isinstance(config.kim, KimConfig):
            base_cmds = _kim_extract_commands(runner, config, input_data, outdir)
        else:
            # kim potentials directly
            base_cmds = potential_default_lines(config.kim)
    except Exception as e:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=None,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note=f"failed to obtain base potential commands: {e}",
            ),
            dt0,
        )

    base_pair_style = None
    for ln in base_cmds:
        if ln.strip().startswith("pair_style"):
            tt = ln.split()
            if len(tt) >= 2:
                base_pair_style = tt[1]
            break

    # buckingham modify potential
    # timestep scan here
    if base_pair_style is None or "buck" not in str(base_pair_style).lower():
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=True,
                note="base potential is not Buckingham-like; core overlay not applied",
            ),
            dt0,
        )

    species = (
        list(config.kim.interactions)
        if isinstance(config.kim.interactions, list) and len(config.kim.interactions) > 0
        else []
    )
    if len(species) == 0:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note="kim.interactions must be a non-empty list for core overlay parameterisation",
            ),
            dt0,
        )

    # candidate timesteps declared
    # fallback
    # explicitly enabled
    try:
        dt_list, dt_source = _core_dt_candidates(config)
    except ValueError as e:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note=str(e),
            ),
            dt0,
        )

    try:
        r_nn = _read_nn_median_from_datafile(input_data, atom_style=str(config.md.atom_style))
    except Exception as e:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note=f"failed to estimate nearest-neighbour distance: {e}",
            ),
            dt0,
        )

    # cutoff guess structure
    r_out = float(core.r_out_factor) * float(r_nn)
    r_out = max(float(core.r_out_min), min(float(core.r_out_max), r_out))

    style = str(core.style).strip().lower()
    # dimensionless target configured
    # lammps style repulsive
    units_style = str(config.kim.user_units).strip().lower()

    def _kB_energy_per_K(units_style: str) -> float:
        # boltzmann constant lammps
        # lammps documentation base
        if units_style in ("metal", "electron", "nano"):
            return 8.617333262145e-5  # e v k
        if units_style == "real":
            return 0.00198720425864083  # kcal mol k
        if units_style == "si":
            return 1.380649e-23  # j k
        if units_style == "cgs":
            return 1.380649e-16  # erg k
        raise ValueError(
            f"core_repulsion.lj_repulsive does not currently support LAMMPS units '{units_style}'. "
            "Use kim.user_units='metal' (recommended) or disable core_repulsion."
        )

    kB = _kB_energy_per_K(units_style)
    u_target_eV = float(core.u_target_kT) * float(kB) * float(T_test)

    # loop
    for k in range(1, int(core.max_attempts) + 1):
        r_in = float(core.r_in_factor) * float(r_out)

        try:
            if style == "zbl":
                pot_lines, base_style = _rewrite_for_hybrid_overlay_zbl(
                    base_cmds, species=species, r_in=r_in, r_out=r_out
                )
            elif style == "lj_repulsive":
                pot_lines, base_style = _rewrite_for_hybrid_overlay_lj_repulsive(
                    base_cmds, species=species, r_in=r_in, r_out=r_out, u_target_eV=u_target_eV
                )
            else:
                raise ValueError(f"Unsupported core_repulsion.style: {style}")
        except Exception as e:
            return (
                None,
                CoreRepulsionResult(
                    enabled=True,
                    applied=False,
                    style=str(core.style),
                    base_pair_style=base_pair_style,
                    r_inner=None,
                    r_outer=None,
                    attempts=k,
                    success=False,
                    note=f"failed to rewrite KIM commands: {e}",
                ),
                dt0,
            )

        # timestep scan stability
        # purely width limited
        dt_selected: Optional[float] = None
        dt_tried: list[float] = []
        for dt in dt_list:
            dt_tried.append(float(dt))
            ok = _run_stability_test(
                runner,
                config,
                input_data,
                outdir=outdir,
                potential_lines=pot_lines,
                temperature=float(T_test),
                timestep=float(dt),
                label=f"try{k}_dt{dt:.6g}",
            )
            if ok:
                dt_selected = float(dt)
                break

        if dt_selected is not None:
            final_lines = pot_lines
            note = f"calibrated at T={T_test:g} K with dt={dt_selected:g}"
            if len(dt_tried) > 1:
                fb_msg = _make_timestep_fallback_warning(
                    context="core-repulsion calibration",
                    source=dt_source,
                    tried=dt_tried,
                    selected=float(dt_selected),
                )
                _append_condensed_preflight_warning(outdir, fb_msg)
                note += f" ({fb_msg})"
            if bool(getattr(core, "tabulate", False)):
                gewald = _resolve_tabulated_gewald(
                    config,
                    base_cmds,
                    stage_log=_core_stability_stage_dir(
                        outdir,
                        temperature=float(T_test),
                        label=f"try{k}_dt{dt_selected:.6g}",
                    )
                    / "log.lammps",
                )
                charges = None if getattr(config.structure, "charges", None) is None else dict(config.structure.charges)
                refinement_candidates: list[dict[str, Any]] = []
                best_candidate: Optional[dict[str, Any]] = None
                accepted_candidate: Optional[dict[str, Any]] = None
                accepted_lines: Optional[list[str]] = None
                summary_reason = ""
                point_schedule: list[int] = []
                tp = int(core.table_points)
                tp_max = int(core.table_points_max)
                while True:
                    point_schedule.append(int(tp))
                    if tp >= tp_max:
                        break
                    tp = min(tp_max, max(int(tp) + 1, int(math.ceil(float(tp) * 2.0))))
                try:
                    seed_lines = build_tabulated_buckingham_core_lines(
                        base_cmds,
                        species=species,
                        units_style=units_style,
                        r_in=float(r_in),
                        r_out=float(r_out),
                        table_points=int(core.table_points),
                        table_filename=str(core.table_filename),
                        table_r_min=float(core.table_r_min),
                        charges=charges,
                        gewald=gewald,
                    )
                    seed_spec = _parse_tabulated_core_spec(seed_lines)
                    if seed_spec is None:
                        raise ValueError("failed to parse tabulated-core metadata after construction")
                    reference_data = _pair_write_potential_curves(
                        runner,
                        config,
                        stage_dir=Path(outdir) / "preflight" / "table_verify" / "reference",
                        potential_lines=pot_lines,
                        spec=seed_spec,
                        npoints=int(core.table_verify_points),
                        output_name="reference.table",
                        log_name="log_pairwrite_reference.lammps",
                    )
                    if reference_data["warnings"]:
                        raise PreflightError(
                            "analytic source pair_write emitted table-related warnings: "
                            + "; ".join(reference_data["warnings"])
                        )
                except Exception as e:
                    summary_reason = f"failed to initialise tabulated Buckingham verification: {e}"
                    report = {
                        "status": "fallback_analytic",
                        "fallback_to_analytic": True,
                        "reason": summary_reason,
                        "accepted_candidate": None,
                        "best_candidate": None,
                        "candidates": [],
                    }
                    _json_path, summary_path = _write_tabulation_refinement_report(outdir=outdir, report=report)
                    try:
                        summary_rel = str(summary_path.relative_to(outdir))
                    except Exception:
                        summary_rel = str(summary_path)
                    _append_condensed_preflight_warning(
                        outdir,
                        "tabulated Buckingham real-space pair potential unavailable during preflight; "
                        f"falling back to analytic hybrid/overlay core ({summary_reason}); see {summary_rel}",
                    )
                    note += (
                        " (tabulated Buckingham real-space pair-potential refinement unavailable; "
                        f"fell back to analytic hybrid/overlay core, see {summary_rel})"
                    )
                    final_lines = pot_lines
                else:
                    for force_mode in _tabulated_force_modes():
                        for table_points in point_schedule:
                            candidate: dict[str, Any] = {
                                "force_mode": str(force_mode),
                                "table_points": int(table_points),
                                "verify_passed": False,
                                "stability_ok": False,
                                "passed": False,
                                "warnings": [],
                            }
                            candidate_lines: Optional[list[str]] = None
                            last_report: Optional[dict[str, Any]] = None
                            try:
                                candidate_lines = build_tabulated_buckingham_core_lines(
                                    base_cmds,
                                    species=species,
                                    units_style=units_style,
                                    r_in=float(r_in),
                                    r_out=float(r_out),
                                    table_points=int(table_points),
                                    table_filename=str(core.table_filename),
                                    table_r_min=float(core.table_r_min),
                                    charges=charges,
                                    gewald=gewald,
                                )
                                candidate_lines = update_tabulated_core_metadata_lines(
                                    candidate_lines,
                                    force_mode=str(force_mode),
                                    include_fprime=True,
                                )
                                spec = _parse_tabulated_core_spec(candidate_lines)
                                if spec is None:
                                    raise ValueError("failed to parse tabulated-core metadata after construction")
                                table_path = _materialize_generated_tabulated_core_source(outdir=outdir, spec=spec)
                                candidate_lines = update_tabulated_core_metadata_lines(
                                    candidate_lines,
                                    generated_by="vitriflow_generated",
                                    sha256=_sha256_path(table_path),
                                    force_mode=str(force_mode),
                                    include_fprime=True,
                                )
                                spec = _parse_tabulated_core_spec(candidate_lines)
                                if spec is None:
                                    raise ValueError("failed to parse tabulated-core metadata after source materialisation")
                                last_report = _verify_tabulated_core_against_source(
                                    runner,
                                    config,
                                    outdir=outdir,
                                    table_potential_lines=candidate_lines,
                                    reference_sections=reference_data["sections"],
                                    spec=spec,
                                )
                                candidate["warnings"] = list(last_report.get("warnings", []))
                                candidate["comparison"] = dict(last_report.get("comparison", {}) or {})
                                candidate["self_consistency"] = dict(last_report.get("self_consistency", {}) or {})
                                candidate["verify_passed"] = bool(last_report.get("passed", False))
                                if bool(candidate["verify_passed"]):
                                    candidate["stability_ok"] = bool(
                                        _run_stability_test(
                                            runner,
                                            config,
                                            input_data,
                                            outdir=outdir,
                                            potential_lines=candidate_lines,
                                            temperature=float(T_test),
                                            timestep=float(dt_selected),
                                            label=(
                                                f"try{k}_dt{dt_selected:.6g}_table_"
                                                f"{str(force_mode).replace('-', '_')}_{int(table_points)}"
                                            ),
                                        )
                                    )
                                candidate["passed"] = bool(candidate["verify_passed"] and candidate["stability_ok"])
                            except Exception as e:
                                candidate["error"] = str(e)
                                candidate["verify_passed"] = False
                                candidate["stability_ok"] = False
                                candidate["passed"] = False
                            refinement_candidates.append(candidate)
                            if best_candidate is None or _tabulation_candidate_sort_key(candidate) < _tabulation_candidate_sort_key(best_candidate):
                                best_candidate = dict(candidate)
                            if bool(candidate.get("passed", False)) and candidate_lines is not None:
                                accepted_candidate = dict(candidate)
                                accepted_lines = list(candidate_lines)
                                break
                        if accepted_candidate is not None:
                            break

                    if accepted_candidate is not None and accepted_lines is not None:
                        final_lines = accepted_lines
                        overall = dict((accepted_candidate.get("comparison") or {}).get("overall", {}) or {})
                        summary_reason = "accepted verified tabulated real-space table"
                        report = {
                            "status": "tabulated_verified",
                            "fallback_to_analytic": False,
                            "reason": summary_reason,
                            "accepted_candidate": accepted_candidate,
                            "best_candidate": best_candidate,
                            "candidates": refinement_candidates,
                        }
                        _json_path, summary_path = _write_tabulation_refinement_report(outdir=outdir, report=report)
                        try:
                            summary_rel = str(summary_path.relative_to(outdir))
                        except Exception:
                            summary_rel = str(summary_path)
                        _append_condensed_preflight_info(
                            outdir,
                            "tabulated Buckingham real-space pair potential accepted after refinement; "
                            + _format_tabulation_candidate_summary(accepted_candidate)
                            + f"; see {summary_rel}",
                        )
                        note += (
                            " (tabulated Buckingham real-space pair potential accepted after refinement; "
                            f"pair_write-verified on {int(core.table_verify_points)} RSQ points, "
                            f"force_mode={accepted_candidate.get('force_mode', '?')}, "
                            f"table_points={int(accepted_candidate.get('table_points', table_points))}, "
                            f"max_energy_ratio={float(overall.get('max_energy_ratio', 0.0)):.3g}, "
                            f"max_force_ratio={float(overall.get('max_force_ratio', 0.0)):.3g}; "
                            f"see {summary_rel})"
                        )
                    else:
                        summary_reason = (
                            "tabulated Buckingham real-space pair-potential refinement did not find a "
                            "candidate that passed strict energy/force verification and calibrated stability"
                        )
                        report = {
                            "status": "fallback_analytic",
                            "fallback_to_analytic": True,
                            "reason": summary_reason,
                            "accepted_candidate": None,
                            "best_candidate": best_candidate,
                            "candidates": refinement_candidates,
                        }
                        _json_path, summary_path = _write_tabulation_refinement_report(outdir=outdir, report=report)
                        try:
                            summary_rel = str(summary_path.relative_to(outdir))
                        except Exception:
                            summary_rel = str(summary_path)
                        best_txt = _format_tabulation_candidate_summary(best_candidate) if isinstance(best_candidate, dict) else "no viable candidate"
                        _append_condensed_preflight_warning(
                            outdir,
                            "tabulated Buckingham real-space pair potential rejected after refinement; "
                            f"falling back to analytic hybrid/overlay core. best candidate: {best_txt}; "
                            f"see {summary_rel}",
                        )
                        note += (
                            " (tabulated Buckingham real-space pair-potential refinement failed; "
                            f"fell back to analytic hybrid/overlay core. best candidate: {best_txt}; "
                            f"see {summary_rel})"
                        )
                        final_lines = pot_lines

            pdir = outdir / "preflight" / "potential_override"
            ensure_dir(pdir)
            (pdir / "potential_lines.lmp").write_text("\n".join(final_lines) + "\n")
            return (
                final_lines,
                CoreRepulsionResult(
                    enabled=True,
                    applied=True,
                    style=str(core.style),
                    base_pair_style=str(base_style),
                    r_inner=float(r_in),
                    r_outer=float(r_out),
                    attempts=k,
                    success=True,
                    note=note,
                ),
                float(dt_selected),
            )

        # calibration update
        # strengthen repulsion widen
        if style == "lj_repulsive":
            u_target_eV = float(u_target_eV) * float(getattr(core, "strength_grow_factor", 2.0))
        r_out = max(float(core.r_out_min), min(float(core.r_out_max), float(r_out) * float(core.grow_factor)))

    return (
        None,
        CoreRepulsionResult(
            enabled=True,
            applied=True,
            style=str(core.style),
            base_pair_style=str(base_pair_style),
            r_inner=None,
            r_outer=float(r_out),
            attempts=int(core.max_attempts),
            success=False,
            note=f"failed to find stable {style} core settings after {int(core.max_attempts)} attempts",
        ),
        float(dt_list[-1]) if dt_list else dt0,
    )


def _evaluate_thermo_table(
    tbl: dict[str, np.ndarray],
    *,
    T_target: float,
    P_target: float,
    ensemble: str,
    config: RunConfig,
) -> tuple[bool, dict[str, Any], float]:
    pf = config.autotune.preflight

    if "Temp" not in tbl:
        return False, {"reason": "Temp column missing"}, float("inf")
    T = np.asarray(tbl["Temp"], dtype=float)
    if len(T) < 3 or not np.all(np.isfinite(T)):
        return False, {"reason": "Temp invalid"}, float("inf")

    Tw = window_mean_stderr(T, start_fraction=0.5)
    eT = abs(Tw.mean - float(T_target)) / max(1.0, float(T_target))

    if float(np.nanmax(T)) > float(pf.max_temp_factor) * float(T_target):
        return False, {"reason": "Temp exploded", "T_max": float(np.nanmax(T))}, float("inf")

    ok = eT <= float(pf.temp_rel_tol)
    score = float(eT)
    details: dict[str, Any] = {
        "T_target": float(T_target),
        "T_mean": float(Tw.mean),
        "T_stderr": float(Tw.stderr),
        "T_rel_err": float(eT),
    }

    if ensemble == "npt":
        # pressure
        if "Press" in tbl:
            P = np.asarray(tbl["Press"], dtype=float)
            Pw = window_mean_stderr(P, start_fraction=0.5)
            eP = abs(Pw.mean - float(P_target))
            details.update(
                {
                    "P_target": float(P_target),
                    "P_mean": float(Pw.mean),
                    "P_stderr": float(Pw.stderr),
                    "P_abs_err": float(eP),
                }
            )
            if not np.isfinite(eP) or abs(float(Pw.mean)) > float(pf.max_press_abs):
                return False, {**details, "reason": "Press invalid/exploded"}, float("inf")
            # pressure convergence objective
            # score hard pass
            # rejecting otherwise stable
            # pressure relaxes slowly
            # cells enable hard
            # preflight pressure tolerance
            if bool(getattr(pf, "require_pressure_tolerance", False)) and eP > float(pf.press_abs_tol):
                ok = False
            score += float(eP) / max(1.0, float(pf.press_abs_tol))
        else:
            ok = False
            details["reason"] = "Press column missing"

        # volume runaway barostat
        V = None
        if "Vol" in tbl:
            V = np.asarray(tbl["Vol"], dtype=float)
        elif "Density" in tbl:
            den = np.asarray(tbl["Density"], dtype=float)
            if np.all(np.isfinite(den)) and float(np.nanmin(den)) > 0:
                V = 1.0 / den

        if V is not None and len(V) >= 3 and np.all(np.isfinite(V)):
            V0 = float(V[0])
            Vw = window_mean_stderr(V, start_fraction=0.5)
            Vmean = float(Vw.mean)
            if V0 > 0 and Vmean > 0:
                ratio = max(Vmean / V0, V0 / Vmean)
                details.update(
                    {
                        "V0": float(V0),
                        "V_mean": float(Vmean),
                        "V_ratio_tail_vs_start": float(ratio),
                    }
                )
                if ratio > float(pf.max_vol_ratio):
                    return False, {**details, "reason": "Volume drift too large"}, float("inf")
                score += float(max(0.0, ratio - 1.0)) / max(1.0, float(pf.max_vol_ratio) - 1.0)

    return ok, details, float(score)


def _run_candidate(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    *,
    outdir: Path,
    potential_lines: Optional[list[str]],
    md: MDConfig,
    temperature: float,
    name: str,
    equil_steps: int,
    run_steps: int,
) -> tuple[bool, dict[str, Any], float]:
    stage_dir = outdir / "preflight" / "thermo_scan" / name / f"T{int(round(temperature))}"
    ensure_dir(stage_dir)
    _localize_input_data_for_preflight(source=input_data, destination=stage_dir / "input.data")

    # preserve candidate timestep
    # preflight selection smaller
    dt_ref = float(config.md.timestep)
    equil_steps_s = scale_steps_for_timestep(int(equil_steps), dt_ref, float(md.timestep), min_steps=0)
    run_steps_s = scale_steps_for_timestep(int(run_steps), dt_ref, float(md.timestep), min_steps=1)

    stage = StageSpec(
        name=f"preflight_{name}",
        input_data=stage_dir / "input.data",
        output_data=stage_dir / "output.data",
        temperature_start=float(temperature),
        temperature_stop=float(temperature),
        pressure=float(md.pressure),
        equil_steps=int(equil_steps_s),
        run_steps=int(run_steps_s),
        seed=12345,
        replicate=(1, 1, 1),
        write_dump=False,
        dump_every=1000,
        msd_every=200,
        potential_lines=potential_lines,
    )
    script = render_stage(config.kim, md, stage)
    try:
        prepare_potential_files(config.kim, stage_dir, potential_lines)
        cleanup = [stage_dir / "output.data", stage_dir / f"{stage.name}.msd.dat"]
        run_with_neighbor_skin_autotune(
            runner,
            lambda md_use: render_stage(config.kim, md_use, stage),
            stage_dir,
            md,
            log_name="log.lammps",
            timeout_sec=float(config.autotune.preflight.timeout_sec),
            cleanup_paths=cleanup,
        )
    except (ExternalCommandError, Exception) as e:
        return False, {"reason": f"lammps failed: {e}"}, float("inf")

    try:
        tbl = parse_last_thermo_table(stage_dir / "log.lammps").as_dict()
    except Exception as e:
        return False, {"reason": f"failed to parse thermo: {e}"}, float("inf")

    ok, details, score = _evaluate_thermo_table(
        tbl,
        T_target=float(temperature),
        P_target=float(md.pressure),
        ensemble=str(md.ensemble),
        config=config,
    )
    return ok, details, score


def _select_md_settings(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    outdir: Path,
    *,
    potential_lines: Optional[list[str]],
    timestep: Optional[float] = None,
) -> tuple[MDConfig, list[ThermoCandidateResult]]:
    pf = config.autotune.preflight
    T_low = float(pf.T_low) if pf.T_low is not None else float(config.autotune.tm_scan.t_min)
    T_high = float(pf.T_high) if pf.T_high is not None else float(config.autotune.tm_scan.t_max)

    ensembles = pf.ensembles if pf.ensembles is not None else [config.md.ensemble]
    ensembles = [str(e) for e in ensembles]

    dt = float(timestep) if timestep is not None else float(config.md.timestep)
    dt_tag = f"dt{dt:.6g}".replace(".", "p")
    tdamps = [float(f) * dt for f in pf.tdamp_factors]
    pdamps = [float(f) * dt for f in pf.pdamp_factors]

    candidates: list[ThermoCandidateResult] = []

    for ens in ensembles:
        for tdamp in tdamps:
            if ens == "npt":
                for pdamp in pdamps:
                    md = config.md.model_copy(
                        deep=True,
                        update={
                            "timestep": float(dt),
                            "ensemble": "npt",
                            "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(tdamp)),
                            "barostat": BarostatConfig(style=config.md.barostat.style, pdamp=float(pdamp)),
                        },
                    )
                    name = f"{dt_tag}_{ens}_td{tdamp:.3g}_pd{pdamp:.3g}".replace(".", "p")
                    okL, detL, scL = _run_candidate(
                        runner,
                        config,
                        input_data,
                        outdir=outdir,
                        potential_lines=potential_lines,
                        md=md,
                        temperature=T_low,
                        name=name,
                        equil_steps=pf.equil_steps,
                        run_steps=pf.run_steps,
                    )
                    okH, detH, scH = _run_candidate(
                        runner,
                        config,
                        input_data,
                        outdir=outdir,
                        potential_lines=potential_lines,
                        md=md,
                        temperature=T_high,
                        name=name,
                        equil_steps=pf.equil_steps,
                        run_steps=pf.run_steps,
                    )
                    ok = bool(okL and okH)
                    score = float(scL + scH)
                    candidates.append(
                        ThermoCandidateResult(
                            timestep=float(dt),
                            ensemble=ens,
                            tdamp=float(tdamp),
                            pdamp=float(pdamp),
                            ok=ok,
                            score=score,
                            details={"low": detL, "high": detH},
                        )
                    )
            else:
                md = config.md.model_copy(
                    deep=True,
                    update={
                        "timestep": float(dt),
                        "ensemble": "nvt",
                        "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(tdamp)),
                    },
                )
                name = f"{dt_tag}_{ens}_td{tdamp:.3g}".replace(".", "p")
                okL, detL, scL = _run_candidate(
                    runner,
                    config,
                    input_data,
                    outdir=outdir,
                    potential_lines=potential_lines,
                    md=md,
                    temperature=T_low,
                    name=name,
                    equil_steps=pf.equil_steps,
                    run_steps=pf.run_steps,
                )
                okH, detH, scH = _run_candidate(
                    runner,
                    config,
                    input_data,
                    outdir=outdir,
                    potential_lines=potential_lines,
                    md=md,
                    temperature=T_high,
                    name=name,
                    equil_steps=pf.equil_steps,
                    run_steps=pf.run_steps,
                )
                ok = bool(okL and okH)
                score = float(scL + scH)
                candidates.append(
                    ThermoCandidateResult(
                        timestep=float(dt),
                        ensemble=ens,
                        tdamp=float(tdamp),
                        pdamp=None,
                        ok=ok,
                        score=score,
                        details={"low": detL, "high": detH},
                    )
                )

    ok_cands = [c for c in candidates if c.ok]
    if not ok_cands:
        raise PreflightError(
            "Preflight failed: no thermostat/barostat candidates passed. "
            "All probe runs either crashed or violated stability/accuracy tolerances.",
            candidates=candidates,
        )

    # ensemble consistency production
    # candidate quench volume
    # transient introduce rate
    prod_ens = str(config.md.ensemble).strip().lower()
    ok_use = ok_cands
    if prod_ens == "npt":
        ok_npt = [c for c in ok_cands if str(c.ensemble) == "npt"]
        if ok_npt:
            ok_use = ok_npt
        else:
            if bool(getattr(pf, "allow_nvt_fallback", False)):
                ok_use = ok_cands
            else:
                raise PreflightError(
                    "Preflight failed: production ensemble is NPT but no NPT thermostat/barostat candidates passed.",
                    candidates=candidates,
                )

    # screening instabilities barostat
    # conservative candidates selecting
    # selection melt quench
    # selection melt quench
    # production ensemble conservative
    # aggressive volume neighbor
    # candidates comparable stability

    prod_ens = str(config.md.ensemble).strip().lower()
    chosen = None

    if prod_ens == "npt":
        # prefer conservative pdamp
        min_pdamp = float(getattr(pf, "min_pdamp_ps_highT", 0.0) or 0.0)
        ok_use_npt = [c for c in ok_use if str(c.ensemble)=="npt" and c.pdamp is not None]
        if min_pdamp > 0.0:
            ok_ge = [c for c in ok_use_npt if float(c.pdamp) >= min_pdamp]
            if ok_ge:
                ok_use_npt = ok_ge

        # order candidates stability
        ok_use_npt = sorted(ok_use_npt, key=lambda c: (-float(c.pdamp or 0.0), -float(c.tdamp), float(c.score)))

        topk = int(getattr(pf, "confirm_topk", 5) or 5)
        topk = max(1, topk)

        # confirm temperatures include
        confirm_temps = list(getattr(pf, "confirm_temps", None) or [])
        if not confirm_temps:
            confirm_temps = [T_high]
        else:
            if all(abs(float(x) - float(T_high)) > 1.0e-6 for x in confirm_temps):
                confirm_temps.append(T_high)
        confirm_temps = [float(x) for x in confirm_temps]

        # confirmation length scan
        tm_total = int(getattr(config.autotune.tm_scan, "equil_steps", 0)) + int(getattr(config.autotune.tm_scan, "sample_steps", 0))
        confirm_run = int(getattr(pf, "confirm_run_steps", max(pf.run_steps, 10000)))
        confirm_run = max(confirm_run, tm_total)
        confirm_equil = int(getattr(pf, "confirm_equil_steps", pf.equil_steps))

        passed = []
        for c in ok_use_npt[:topk]:
            all_ok = True
            for Tconf in confirm_temps:
                md_c = config.md.model_copy(
                    deep=True,
                    update={
                        "timestep": float(dt),
                        "ensemble": "npt",
                        "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(c.tdamp)),
                        "barostat": BarostatConfig(style=config.md.barostat.style, pdamp=float(c.pdamp)),
                    },
                )
                okC, detC, scC = _run_candidate(
                    runner,
                    config,
                    input_data,
                    outdir=outdir,
                    potential_lines=potential_lines,
                    md=md_c,
                    temperature=float(Tconf),
                    name=f"{dt_tag}_confirm_{int(round(Tconf))}K_td{float(c.tdamp):.3g}_pd{float(c.pdamp):.3g}".replace(".", "p"),
                    equil_steps=confirm_equil,
                    run_steps=confirm_run,
                )
                try:
                    c.details[f"confirm_{int(round(Tconf))}K"] = detC
                    c.details[f"confirm_{int(round(Tconf))}K_score"] = float(scC)
                    c.details[f"confirm_{int(round(Tconf))}K_ok"] = bool(okC)
                except Exception:
                    pass
                if not okC:
                    all_ok = False
                    break
            if all_ok:
                passed.append(c)

        if not passed:
            raise PreflightError(
                "Preflight failed: screening found stable NPT candidates, but none passed the long confirmation run(s). "
                "This indicates timestep and/or barostat damping is too aggressive for high-T NPT (or the system is too small).",
                candidates=candidates,
            )

        # choose conservative confirm
        passed = sorted(passed, key=lambda c: (-float(c.pdamp or 0.0), -float(c.tdamp), float(c.score)))
        chosen = passed[0]

    else:
        ok_use = sorted(ok_use, key=lambda c: float(c.score))
        chosen = ok_use[0]

    md_sel = config.md.model_copy(
        deep=True,
        update={
            "timestep": float(dt),
            "ensemble": str(chosen.ensemble),
            "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(chosen.tdamp)),
            "barostat": (
                BarostatConfig(style=config.md.barostat.style, pdamp=float(chosen.pdamp))
                if str(chosen.ensemble) == "npt" and chosen.pdamp is not None
                else config.md.barostat
            ),
        },
    )
    return md_sel, candidates


def run_preflight(runner: Union[LammpsRunner, Cp2kRunner], config: RunConfig, input_data: Path, outdir: Path) -> PreflightResult:
    """Preflight."""
    ensure_dir(outdir / "preflight")

    pf = config.autotune.preflight

    # preflight timestep thermostat
    if getattr(config, "engine", "lammps") == "cp2k" or isinstance(runner, Cp2kRunner):
        return _run_preflight_cp2k(runner, config, input_data, outdir)
    if not pf.enabled:
        core_res = CoreRepulsionResult(
            enabled=bool(config.kim.core_repulsion.enabled),
            applied=False,
            style=str(config.kim.core_repulsion.style),
            base_pair_style=None,
            r_inner=None,
            r_outer=None,
            attempts=0,
            success=True,
            note="preflight disabled",
        )
        return PreflightResult(
            selected_ensemble=str(config.md.ensemble),
            selected_timestep=float(config.md.timestep),
            selected_tdamp=float(config.md.thermostat.tdamp),
            selected_pdamp=float(config.md.barostat.pdamp) if str(config.md.ensemble) == "npt" else None,
            potential_lines=None,
            core_repulsion=core_res,
            candidates=[],
        )

    T_test = float(pf.T_high) if pf.T_high is not None else float(config.autotune.tm_scan.t_max)
    pot_lines, core_res, dt_sel = _maybe_apply_core_repulsion(runner, config, input_data, outdir, T_test=T_test)

    # requested buckingham overlay
    # parameterisation continuing misleading
    if bool(config.kim.core_repulsion.enabled) and (not bool(core_res.success)):
        fail = {
            "success": False,
            "reason": f"Core repulsion preflight failed: {core_res.note}",
            "potential_lines": pot_lines,
            "core_repulsion": asdict(core_res),
            "candidates": [],
        }
        (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(fail, indent=2))
        raise PreflightError(f"Core repulsion preflight failed: {core_res.note}")

    # timestep thermo preflight
    # fallback
    # explicitly enabled
    dt_base = float(dt_sel) if dt_sel is not None else float(config.md.timestep)
    dt_list, dt_source = _preflight_dt_candidates(config, dt_cap=(dt_base if (bool(core_res.applied) and bool(core_res.success)) else None))
    preflight_warnings: list[str] = []
    if len(dt_list) == 0:
        msg = (
            f"Preflight cannot continue: the calibrated core-repulsion timestep ({dt_base:g}) is smaller than every allowed "
            f"preflight timestep from {dt_source}. Declare a compatible autotune.preflight.dt_candidates list."
        )
        fail = {
            "success": False,
            "reason": msg,
            "timestep_candidate_source": dt_source,
            "timestep_candidates_tried": [],
            "potential_lines": pot_lines,
            "core_repulsion": asdict(core_res),
            "candidates": [],
        }
        (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(fail, indent=2))
        raise PreflightError(msg)

    md_sel = None
    candidates_all: list[ThermoCandidateResult] = []
    last_err: Optional[PreflightError] = None
    dt_tried: list[float] = []
    for dt in dt_list:
        dt_tried.append(float(dt))
        try:
            md_sel_i, cands_i = _select_md_settings(
                runner,
                config,
                input_data,
                outdir,
                potential_lines=pot_lines,
                timestep=float(dt),
            )
            md_sel = md_sel_i
            candidates_all.extend(cands_i)
            break
        except PreflightError as e:
            last_err = e
            try:
                candidates_all.extend(list(getattr(e, "candidates", []) or []))
            except Exception:
                pass
            continue

    if len(dt_tried) > 1:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            _make_timestep_fallback_warning(
                context="thermo preflight",
                source=dt_source,
                tried=dt_tried,
                selected=(None if md_sel is None else float(getattr(md_sel, "timestep", dt_tried[-1]))),
            ),
        )

    if md_sel is None:
        prod_ens = str(config.md.ensemble).strip().lower()

        # summarise misleading diagnoses
        ok_total = 0
        ok_by_ens: dict[str, int] = {}
        scanned_ens: set[str] = set()
        for c in candidates_all:
            try:
                ens = str(getattr(c, "ensemble", "")).strip().lower() or "unknown"
            except Exception:
                ens = "unknown"
            scanned_ens.add(ens)
            if bool(getattr(c, "ok", False)):
                ok_total += 1
                ok_by_ens[ens] = int(ok_by_ens.get(ens, 0)) + 1

        ok_npt = int(ok_by_ens.get("npt", 0))
        ok_nvt = int(ok_by_ens.get("nvt", 0))
        scanned_sorted = sorted(scanned_ens)

        # actionable production scan
        # unstable preflight scan
        # configured test nvt
        if prod_ens == "npt" and ok_total > 0 and ok_npt == 0 and ok_nvt > 0:
            msg = (
                "Preflight failed: production ensemble is NPT but no stable NPT thermostat/barostat candidates were found. "
                f"Found {ok_nvt} stable NVT candidates. "
                "Ensure autotune.preflight.ensembles includes 'npt' (or leave it unset to follow md.ensemble), "
                "or set autotune.preflight.allow_nvt_fallback=true to permit NVT selection."
            )
        else:
            # otherwise informative underlying
            if last_err is not None:
                msg = f"Preflight failed after timestep fallback: {last_err}"
            else:
                msg = (
                    "Preflight failed: no stable thermostat/barostat candidates passed, even after timestep fallback. "
                    f"Requested production ensemble: {prod_ens}."
                )

        fail = {
            "success": False,
            "reason": msg,
            "requested_production_ensemble": prod_ens,
            "ensembles_scanned": scanned_sorted,
            "ok_counts_by_ensemble": ok_by_ens,
            "timestep_candidate_source": dt_source,
            "timestep_candidates_tried": [float(x) for x in dt_list],
            "warnings": list(preflight_warnings),
            "potential_lines": pot_lines,
            "core_repulsion": asdict(core_res),
            "candidates": [asdict(c) for c in candidates_all],
        }
        (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(fail, indent=2))
        raise PreflightError(msg, candidates=candidates_all) from last_err

    candidates = candidates_all

    requested_ensemble = str(config.md.ensemble).strip().lower()
    if str(md_sel.ensemble).strip().lower() != requested_ensemble:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            f"thermo preflight: ensemble fallback activated; requested ensemble={requested_ensemble}; "
            f"selected ensemble={str(md_sel.ensemble).strip().lower()}",
        )

    res = PreflightResult(
        selected_ensemble=str(md_sel.ensemble),
        selected_timestep=float(md_sel.timestep),
        selected_tdamp=float(md_sel.thermostat.tdamp),
        selected_pdamp=float(md_sel.barostat.pdamp) if str(md_sel.ensemble) == "npt" else None,
        potential_lines=pot_lines,
        core_repulsion=core_res,
        candidates=candidates,
    )
    report = asdict(res)
    report["timestep_candidate_source"] = dt_source
    report["timestep_candidates_tried"] = [float(x) for x in dt_list]
    report["selected_timestep_reason"] = "largest stable user-declared candidate"
    report["warnings"] = list(preflight_warnings)
    (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(report, indent=2))
    return res



def _run_preflight_cp2k(
    runner: Union[LammpsRunner, Cp2kRunner],
    config: RunConfig,
    input_data: Path,
    outdir: Path,
) -> PreflightResult:
    """Preflight cp2k."""

    if not isinstance(runner, Cp2kRunner):
        raise ValueError("engine='cp2k' requires a Cp2kRunner")

    pf = config.autotune.preflight
    md_ref = config.md

    import numpy as np

    # mapping reading lammps
    type_to_species = config.autotune.metrics.type_to_species
    if type_to_species is None:
        raise ValueError("CP2K preflight requires autotune.metrics.type_to_species")

    # preflight disabled stability
    if not pf.enabled:
        core_res = CoreRepulsionResult(
            enabled=False,
            applied=False,
            style="none",
            base_pair_style=None,
            r_inner=None,
            r_outer=None,
            attempts=0,
            success=True,
            note="preflight disabled (cp2k backend)",
        )
        return PreflightResult(
            selected_ensemble=str(md_ref.ensemble),
            selected_timestep=float(md_ref.timestep),
            selected_tdamp=float(md_ref.thermostat.tdamp),
            selected_pdamp=float(md_ref.barostat.pdamp) if str(md_ref.ensemble).lower() == "npt" else None,
            potential_lines=None,
            core_repulsion=core_res,
            candidates=[],
        )

    # temperatures scan conservative
    T_low = float(config.autotune.tm_scan.t_min)
    T_high = float(config.autotune.tm_scan.t_max)
    if not (math.isfinite(T_low) and math.isfinite(T_high) and T_low > 0 and T_high > T_low):
        raise ValueError("Invalid tm_scan temperature range for preflight")

    # candidate timesteps permitted
    # fallback
    dt_list, dt_source = _preflight_dt_candidates(config, cp2k=True)
    preflight_warnings: list[str] = []
    dt_attempted: list[float] = []

    td_factors = [float(x) for x in pf.tdamp_factors]
    td_factors = [x for x in td_factors if math.isfinite(x) and x > 0]
    if len(td_factors) == 0:
        td_factors = [100.0]

    pd_factors = [float(x) for x in pf.pdamp_factors]
    pd_factors = [x for x in pd_factors if math.isfinite(x) and x > 0]
    if len(pd_factors) == 0:
        pd_factors = [1000.0]

    min_pdamp_fs_highT = float(pf.min_pdamp_ps_highT) * 1000.0

    # evaluation helpers
    def _tail_stats(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        n = int(min(int(pf.tail_window), int(x.size)))
        tail = x[-n:]
        return float(np.mean(tail)), float(np.std(tail))

    def _ok_temp(temp: np.ndarray, target: float) -> bool:
        temp = np.asarray(temp, dtype=float)
        temp = temp[np.isfinite(temp)]
        if temp.size == 0:
            return False
        meanT, _ = _tail_stats(temp)
        if not math.isfinite(meanT):
            return False
        # consistent lammps preflight
        if abs(meanT - target) / target > float(pf.temp_rel_tol):
            return False
        if float(np.max(temp)) > float(pf.max_temp_factor) * target:
            return False
        return True

    def _ok_volume(vol: np.ndarray) -> bool:
        vol = np.asarray(vol, dtype=float)
        vol = vol[np.isfinite(vol)]
        if vol.size == 0:
            return False
        n = int(min(int(pf.tail_window), int(vol.size)))
        tail = vol[-n:]
        vmin = float(np.min(tail))
        vmax = float(np.max(tail))
        if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin > 0.0):
            return False
        if vmax / vmin > float(pf.max_vol_ratio):
            return False
        return True

    candidates: list[ThermoCandidateResult] = []

    # ensembles configured order
    ens_order = [str(e).lower() for e in (pf.ensembles or [md_ref.ensemble])]
    ens_order = [e for e in ens_order if e in ("nvt", "npt")]
    if len(ens_order) == 0:
        ens_order = [str(md_ref.ensemble).lower()]

    best_overall: Optional[ThermoCandidateResult] = None
    best_overall_hard: Optional[ThermoCandidateResult] = None

    for ens in ens_order:
        best: Optional[ThermoCandidateResult] = None
        best_hard: Optional[ThermoCandidateResult] = None

        for dt in dt_list:
            if not dt_attempted or float(dt_attempted[-1]) != float(dt):
                dt_attempted.append(float(dt))
            for fac in td_factors:
                tdamp = fac * dt

                if ens == "npt":
                    for p_fac in pd_factors:
                        pdamp = max(p_fac * dt, min_pdamp_fs_highT)

                        md_use = md_ref.model_copy(deep=True)
                        md_use.timestep = float(dt)
                        md_use.ensemble = "npt"
                        md_use.thermostat.tdamp = float(tdamp)
                        md_use.barostat.pdamp = float(pdamp)

                        stage = StageSpec(
                            name="preflight_highT",
                            input_data=input_data,
                            output_data=outdir / "preflight" / "preflight_highT_out.data",
                            sample_ensemble=None,
                            temperature_start=T_high,
                            temperature_stop=T_high,
                            pressure=float(md_ref.pressure),
                            equil_steps=int(pf.equil_steps),
                            run_steps=int(pf.run_steps),
                            seed=int(config.random_seed),
                            replicate=None,
                            write_dump=False,
                            dump_every=md_use.dump_every,
                            tail_dump_frames=0,
                            tail_dump_stride=1,
                            # preflight diffusion estimates
                            # msd sampling thermo
                            msd_every=int(md_use.thermo_every),
                            potential_lines=None,
                        )

                        cand_dir = outdir / "preflight" / f"npt_dt_{dt:g}_td_{tdamp:g}_pd_{pdamp:g}"
                        ok = False
                        last_err: Optional[Exception] = None
                        meanT = float("nan")
                        maxT = float("nan")
                        tstd = float("nan")
                        vmean = float("nan")
                        vstd = float("nan")
                        vratio = float("nan")
                        score = float("inf")
                        vol_ok = False
                        hard_temp_ok = False
                        strict_temp_ok = False
                        hard_ok = False
                        try:
                            art = run_stage_local(
                                runner,
                                None,
                                md_use,
                                stage,
                                cand_dir,
                                log_name="log.lammps",
                                type_to_species=list(type_to_species),
                            )
                            # prefer neutral thermo
                            try:
                                from ..io.thermo import parse_thermo_csv

                                table = parse_thermo_csv(art.thermo_csv)
                            except Exception:
                                table = parse_last_thermo_table(art.log_path)
                            # thermo table dict
                            # mapping column access
                            tdict = table.as_dict()
                            temp = np.asarray(tdict.get("Temp", []), dtype=float)
                            vol = np.asarray(tdict.get("Volume", []), dtype=float)

                            if temp.size > 0:
                                meanT, tstd = _tail_stats(temp)
                                maxT = float(np.max(temp[np.isfinite(temp)])) if np.any(np.isfinite(temp)) else float("nan")

                            if vol.size > 0:
                                vmean, vstd = _tail_stats(vol)
                                n = int(min(int(pf.tail_window), int(np.isfinite(vol).sum())))
                                if n > 0:
                                    tail = vol[np.isfinite(vol)][-n:]
                                    vmin = float(np.min(tail))
                                    vmax = float(np.max(tail))
                                    vratio = vmax / vmin if vmin > 0 else float("nan")

                            vol_ok = _ok_volume(vol)
                            # strict stability tolerance
                            # enforces bounded statistics
                            hard_temp_ok = bool(temp.size > 0 and math.isfinite(meanT) and math.isfinite(maxT) and maxT <= float(pf.max_temp_factor) * T_high)
                            strict_temp_ok = bool(hard_temp_ok and abs(meanT - T_high) / T_high <= float(pf.temp_rel_tol))
                            hard_ok = bool(hard_temp_ok and vol_ok)
                            ok = bool(strict_temp_ok and vol_ok)

                            # temperature volume fluctuation
                            vol_pen = 0.0
                            if math.isfinite(vratio):
                                vol_pen = max(0.0, vratio - 1.0)
                            score = -float(dt) + abs(meanT - T_high) / T_high + 0.1 * vol_pen

                        except Exception as e:
                            last_err = e
                            ok = False
                            # debug
                            try:
                                import traceback

                                ensure_dir(cand_dir)
                                (cand_dir / "vitriflow_error.txt").write_text(traceback.format_exc())
                            except Exception:
                                pass

                        details: dict[str, Any] = {
                            "T_target": float(T_high),
                            "mean_temp": float(meanT),
                            "temp_tail_std": float(tstd),
                            "max_temp": float(maxT),
                            "mean_volume": float(vmean),
                            "volume_tail_std": float(vstd),
                            "volume_ratio": float(vratio),
                        }
                        details.update(
                            {
                                "temp_rel_err": float(abs(meanT - T_high) / T_high) if math.isfinite(meanT) else float("nan"),
                                "strict_temp_ok": bool(strict_temp_ok),
                                "hard_temp_ok": bool(hard_temp_ok),
                                "vol_ok": bool(vol_ok),
                                "hard_ok": bool(hard_ok),
                            }
                        )
                        if last_err is not None:
                            details["last_error"] = str(last_err)

                        cand = ThermoCandidateResult(
                            timestep=float(dt),
                            ensemble="npt",
                            tdamp=float(tdamp),
                            pdamp=float(pdamp),
                            ok=bool(ok),
                            score=float(score),
                            details=details,
                        )
                        candidates.append(cand)

                        if ok and (best is None or cand.score < best.score):
                            best = cand
                        if hard_ok and (best_hard is None or cand.score < best_hard.score):
                            best_hard = cand

                else:  # nvt
                    md_use = md_ref.model_copy(deep=True)
                    md_use.timestep = float(dt)
                    md_use.ensemble = "nvt"
                    md_use.thermostat.tdamp = float(tdamp)

                    stage = StageSpec(
                        name="preflight_highT",
                        input_data=input_data,
                        output_data=outdir / "preflight" / "preflight_highT_out.data",
                        sample_ensemble=None,
                        temperature_start=T_high,
                        temperature_stop=T_high,
                        pressure=float(md_ref.pressure),
                        equil_steps=int(pf.equil_steps),
                        run_steps=int(pf.run_steps),
                        seed=int(config.random_seed),
                        replicate=None,
                        write_dump=False,
                        dump_every=md_use.dump_every,
                        tail_dump_frames=0,
                        tail_dump_stride=1,
                        msd_every=int(md_use.thermo_every),
                        potential_lines=None,
                    )

                    cand_dir = outdir / "preflight" / f"nvt_dt_{dt:g}_td_{tdamp:g}"
                    ok = False
                    last_err: Optional[Exception] = None
                    meanT = float("nan")
                    maxT = float("nan")
                    tstd = float("nan")
                    score = float("inf")
                    hard_temp_ok = False
                    strict_temp_ok = False
                    hard_ok = False
                    try:
                        art = run_stage_local(
                            runner,
                            None,
                            md_use,
                            stage,
                            cand_dir,
                            log_name="log.lammps",
                            type_to_species=list(type_to_species),
                        )
                        try:
                            from ..io.thermo import parse_thermo_csv

                            table = parse_thermo_csv(art.thermo_csv)
                        except Exception:
                            table = parse_last_thermo_table(art.log_path)
                        tdict = table.as_dict()
                        temp = np.asarray(tdict.get("Temp", []), dtype=float)
                        if temp.size > 0:
                            meanT, tstd = _tail_stats(temp)
                            maxT = float(np.max(temp[np.isfinite(temp)])) if np.any(np.isfinite(temp)) else float("nan")
                        # strict stability volume
                        hard_temp_ok = bool(temp.size > 0 and math.isfinite(meanT) and math.isfinite(maxT) and maxT <= float(pf.max_temp_factor) * T_high)
                        strict_temp_ok = bool(hard_temp_ok and abs(meanT - T_high) / T_high <= float(pf.temp_rel_tol))
                        hard_ok = bool(hard_temp_ok)
                        ok = bool(strict_temp_ok)
                        score = -float(dt) + abs(meanT - T_high) / T_high
                    except Exception as e:
                        last_err = e
                        ok = False
                        try:
                            import traceback

                            ensure_dir(cand_dir)
                            (cand_dir / "vitriflow_error.txt").write_text(traceback.format_exc())
                        except Exception:
                            pass

                    details: dict[str, Any] = {
                        "T_target": float(T_high),
                        "mean_temp": float(meanT),
                        "temp_tail_std": float(tstd),
                        "max_temp": float(maxT),
                    }
                    details.update(
                        {
                            "temp_rel_err": float(abs(meanT - T_high) / T_high) if math.isfinite(meanT) else float("nan"),
                            "strict_temp_ok": bool(strict_temp_ok),
                            "hard_temp_ok": bool(hard_temp_ok),
                            "hard_ok": bool(hard_ok),
                        }
                    )
                    if last_err is not None:
                        details["last_error"] = str(last_err)

                    cand = ThermoCandidateResult(
                        timestep=float(dt),
                        ensemble="nvt",
                        tdamp=float(tdamp),
                        pdamp=None,
                        ok=bool(ok),
                        score=float(score),
                        details=details,
                    )
                    candidates.append(cand)

                    if ok and (best is None or cand.score < best.score):
                        best = cand
                    if hard_ok and (best_hard is None or cand.score < best_hard.score):
                        best_hard = cand

            # candidate largest searching
            if best is not None and best.timestep == float(dt):
                break

        if best is not None:
            best_overall = best
            break
        if best_overall_hard is None and best_hard is not None:
            best_overall_hard = best_hard

    core_res = CoreRepulsionResult(
        enabled=False,
        applied=False,
        style="none",
        base_pair_style=None,
        r_inner=None,
        r_outer=None,
        attempts=0,
        success=True,
        note="core repulsion not applicable for cp2k backend",
    )
    if best_overall is None and best_overall_hard is not None:
        # fallback
        # candidate bounded volume
        d = dict(best_overall_hard.details or {})
        d["selected_by"] = "hard_fallback"
        d["note"] = "No candidate met strict mean-T tolerance; selected best hard-stable candidate."
        best_overall = ThermoCandidateResult(
            timestep=float(best_overall_hard.timestep),
            ensemble=str(best_overall_hard.ensemble),
            tdamp=float(best_overall_hard.tdamp),
            pdamp=float(best_overall_hard.pdamp) if best_overall_hard.pdamp is not None else None,
            ok=True,
            score=float(best_overall_hard.score),
            details=d,
        )

    if len(dt_attempted) > 1:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            _make_timestep_fallback_warning(
                context="cp2k preflight",
                source=dt_source,
                tried=dt_attempted,
                selected=(None if best_overall is None else float(best_overall.timestep)),
            ),
        )

    if best_overall is None:
        # candidate exception structure
        # representative misleading diagnosis
        errs = []
        for c in candidates:
            try:
                if isinstance(c, ThermoCandidateResult) and isinstance(c.details, dict) and "last_error" in c.details:
                    errs.append(str(c.details.get("last_error")))
            except Exception:
                pass
        if errs and len(errs) == len(candidates):
            msg = (
                "CP2K preflight failed: all candidates errored before producing usable output; "
                f"example error: {errs[0]}"
            )
        else:
            msg = "CP2K preflight failed: no stable thermostat/barostat candidates passed at high temperature"
        raise PreflightError(msg, candidates=candidates)

    requested_ensemble = str(md_ref.ensemble).lower()
    if str(best_overall.ensemble).lower() != requested_ensemble:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            f"cp2k preflight: ensemble fallback activated; requested ensemble={requested_ensemble}; "
            f"selected ensemble={str(best_overall.ensemble).lower()}",
        )
    if str((best_overall.details or {}).get("selected_by", "")).strip().lower() == "hard_fallback":
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            "cp2k preflight: hard stability fallback activated; no candidate met strict mean-temperature tolerance; "
            "selected best hard-stable candidate",
        )

    res = PreflightResult(
        selected_ensemble=str(best_overall.ensemble),
        selected_timestep=float(best_overall.timestep),
        selected_tdamp=float(best_overall.tdamp),
        selected_pdamp=float(best_overall.pdamp) if str(best_overall.ensemble).lower() == "npt" else None,
        potential_lines=None,
        core_repulsion=core_res,
        candidates=candidates,
    )
    report = asdict(res)
    report["timestep_candidate_source"] = dt_source
    report["timestep_candidates_tried"] = [float(x) for x in dt_list]
    report["selected_timestep_reason"] = "largest stable user-declared candidate"
    report["warnings"] = list(preflight_warnings)
    (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(report, indent=2))
    return res
