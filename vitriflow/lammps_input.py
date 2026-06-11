from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Optional, Sequence

from .config import MDConfig, PotentialConfig
from .potential import potential_default_lines, potential_init_lines, potential_interactions_list


@dataclass(frozen=True)
class StageSpec:
    name: str
    input_data: Path
    output_data: Path
    temperature_start: float
    temperature_stop: float
    pressure: float
    equil_steps: int
    run_steps: int
    seed: int

    # velocities start
    # initialize boltzmann velocities
    # initialize boltzmann velocities
    # seed
    # preserve velocities present
    # chained preserve trajectory
    velocity_mode: Literal["create", "preserve"] = "create"

    # equal volume intended
    # melt disordering remove
    # anisotropic triclinic box
    force_isotropic: bool = False

    replicate: Optional[tuple[int, int, int]] = None
    write_dump: bool = True
    dump_every: Optional[int] = None
    # dump frames spaced
    # dump structural metrics
    tail_dump_frames: Optional[int] = None
    tail_dump_stride: Optional[int] = None
    msd_every: int = 100


    # override msd volume
    # equilibration ensemble sampling
    sample_ensemble: Optional[str] = None

    # override potential force
    # potential lines
    potential_lines: Optional[Sequence[str]] = None



def _force_isotropic_block(stage: StageSpec) -> str:
    if not bool(getattr(stage, "force_isotropic", False)):
        return ""
    return (
        "# Optional: remap current periodic box to an equal-volume cubic box before melt\n"
        "change_box all triclinic "
        "xy final 0.0 xz final 0.0 yz final 0.0 "
        "x final 0.0 $(vol^(1.0/3.0)) "
        "y final 0.0 $(vol^(1.0/3.0)) "
        "z final 0.0 $(vol^(1.0/3.0)) remap units box"
    )


def _mass_lines_from_interactions(
    interactions: Sequence[str], *, md: MDConfig, datafile: Optional[Path] = None
) -> str:
    """Mass lines from."""
    if not interactions:
        return ""

    mode = str(getattr(md, 'mass_mode', 'auto')).strip().lower()
    if mode == 'data':
        return ""

    if mode == 'auto' and datafile is not None:
        try:
            from .analysis.datafile import datafile_has_masses

            if datafile_has_masses(Path(datafile)):
                return ""
        except Exception:
            # determine presence emitting
            pass

    try:
        from ase.data import atomic_numbers, atomic_masses
    except Exception:  # pragma: no cover
        return ""

    lines: list[str] = []
    for t, sp in enumerate(list(interactions), start=1):
        sym = str(sp)
        if sym not in atomic_numbers:
            continue
        Z = int(atomic_numbers[sym])
        m = float(atomic_masses[Z])
        if m > 0:
            lines.append(f"mass {t} {m:.6f}")
    return "\n".join(lines)

def _style_name(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


def _thermostat_fix_lines(
    style: str,
    *,
    Tstart: float,
    Tstop: float,
    tdamp: float,
    seed: int,
) -> list[str]:
    """LAMMPS thermostat fixes that do not perform time integration."""

    if style == "csvr":
        return [f"fix th all temp/csvr {Tstart} {Tstop} {tdamp} {int(seed)}"]
    if style == "langevin":
        return [f"fix th all langevin {Tstart} {Tstop} {tdamp} {int(seed)}"]
    if style == "berendsen":
        return [f"fix th all temp/berendsen {Tstart} {Tstop} {tdamp}"]
    raise ValueError(f"Unsupported LAMMPS thermostat style: {style}")


def _lammps_integration_block(
    md: MDConfig,
    ensemble: str,
    *,
    Tstart: float,
    Tstop: float,
    pressure: float,
    seed: int,
) -> tuple[str, str]:
    """Return (fix block, cleanup block) for a LAMMPS sampling segment.

    Nose-Hoover remains the compact/default path via fix nvt/npt. Alternative
    thermostat/barostat choices are represented explicitly with separate
    time-integration, thermostat, and/or barostat fixes.
    """

    ens = _style_name(ensemble)
    if ens not in {"nvt", "npt"}:
        raise ValueError(f"Unsupported LAMMPS ensemble: {ensemble}")

    th_style = _style_name(getattr(md.thermostat, "style", "nose-hoover"))
    bar_style = _style_name(getattr(md.barostat, "style", "nose-hoover"))
    tdamp = float(getattr(md.thermostat, "tdamp", 0.0))
    pdamp = float(getattr(md.barostat, "pdamp", 0.0))
    if not math.isfinite(tdamp) or tdamp <= 0.0:
        raise ValueError("md.thermostat.tdamp must be finite and > 0")
    if ens == "npt" and (not math.isfinite(pdamp) or pdamp <= 0.0):
        raise ValueError("md.barostat.pdamp must be finite and > 0 for NPT")

    lines: list[str] = []
    cleanup: list[str] = []

    if ens == "nvt":
        if th_style == "nose-hoover":
            lines.append(f"fix int all nvt temp {Tstart} {Tstop} {tdamp}")
            cleanup.append("unfix int")
        else:
            lines.append("fix int all nve")
            lines.extend(_thermostat_fix_lines(th_style, Tstart=Tstart, Tstop=Tstop, tdamp=tdamp, seed=seed))
            cleanup.extend(["unfix th", "unfix int"])
        return "\n".join(lines), "\n".join(cleanup)

    mode = str(getattr(md.barostat, "mode", "iso")).strip().lower()
    if th_style == "nose-hoover" and bar_style == "nose-hoover":
        lines.append(f"fix int all npt temp {Tstart} {Tstop} {tdamp} {mode} {pressure} {pressure} {pdamp}")
        cleanup.append("unfix int")
    elif th_style == "nose-hoover" and bar_style == "berendsen":
        lines.append(f"fix int all nvt temp {Tstart} {Tstop} {tdamp}")
        lines.append(f"fix bar all press/berendsen {mode} {pressure} {pressure} {pdamp}")
        cleanup.extend(["unfix bar", "unfix int"])
    elif th_style != "nose-hoover" and bar_style == "nose-hoover":
        lines.append(f"fix int all nph {mode} {pressure} {pressure} {pdamp}")
        lines.extend(_thermostat_fix_lines(th_style, Tstart=Tstart, Tstop=Tstop, tdamp=tdamp, seed=seed))
        cleanup.extend(["unfix th", "unfix int"])
    elif th_style != "nose-hoover" and bar_style == "berendsen":
        lines.append("fix int all nve")
        lines.extend(_thermostat_fix_lines(th_style, Tstart=Tstart, Tstop=Tstop, tdamp=tdamp, seed=seed))
        lines.append(f"fix bar all press/berendsen {mode} {pressure} {pressure} {pdamp}")
        cleanup.extend(["unfix bar", "unfix th", "unfix int"])
    else:
        raise ValueError(f"Unsupported LAMMPS barostat style: {bar_style}")

    return "\n".join(lines), "\n".join(cleanup)


def render_stage(pot: PotentialConfig, md: MDConfig, stage: StageSpec) -> str:
    """Stage."""
    dump_every = stage.dump_every if stage.dump_every is not None else md.dump_every

    fix_line_equil, unfix_line_equil = _lammps_integration_block(
        md,
        md.ensemble,
        Tstart=float(stage.temperature_start),
        Tstop=float(stage.temperature_start),
        pressure=float(stage.pressure),
        seed=int(stage.seed),
    )
    sample_ens = stage.sample_ensemble if stage.sample_ensemble is not None else md.ensemble
    fix_line_sample, unfix_line_sample = _lammps_integration_block(
        md,
        sample_ens,
        Tstart=float(stage.temperature_start),
        Tstop=float(stage.temperature_stop),
        pressure=float(stage.pressure),
        seed=int(stage.seed),
    )

    rep_line = ""
    if stage.replicate is not None:
        nx, ny, nz = stage.replicate
        rep_line = f"replicate {nx} {ny} {nz}"

    force_iso_block = _force_isotropic_block(stage)

    # velocity policy start
    vel_mode = str(getattr(stage, "velocity_mode", "create")).strip().lower()
    if vel_mode == "preserve":
        velocity_line = "# initial velocities: preserved from input.data"
    else:
        velocity_line = (
            f"velocity all create {stage.temperature_start} {stage.seed} mom yes rot yes dist gaussian"
        )

    default_lines = potential_default_lines(pot)
    interactions = potential_interactions_list(pot)
    mass_block = _mass_lines_from_interactions(interactions, md=md, datafile=stage.input_data)
    mass_section = ""
    if mass_block:
        mode = str(getattr(md, 'mass_mode', 'auto')).strip().lower()
        mass_section = f"# Masses ({mode})\n" + str(mass_block)

    potential_block = "\n".join(default_lines)
    if stage.potential_lines is not None:
        lines = [str(x).strip() for x in stage.potential_lines if str(x).strip() != ""]
        if len(lines) > 0:
            potential_block = "\n".join(lines)

    # dump strategy
    tail_dump = bool(stage.write_dump and stage.tail_dump_frames and stage.tail_dump_stride)
    dump_lines = ""
    run_lines = ""

    if not stage.write_dump:
        run_lines = f"run {stage.run_steps}"
    else:
        if tail_dump:
            frames = int(stage.tail_dump_frames or 0)
            stride = int(stage.tail_dump_stride or 0)
            if frames < 1:
                frames = 1
            if stride < 1:
                stride = max(1, int(dump_every))
            tail_steps = int(frames * stride)
            if stage.run_steps <= tail_steps:
                # Run is shorter than the requested tail window; dump throughout.
                # `first yes` is added as a robustness measure so step 0 is
                # always emitted as a frame -- without it, very short runs
                # whose stride exceeds run_steps could yield zero frames at
                # all. It is NOT a frame-count guarantee: the actual number
                # of frames depends on stride_eff and run_steps and may
                # overshoot the requested {frames} by one (e.g. run_steps=10,
                # frames=5 -> stride_eff=2 -> frames at 0,2,4,6,8,10 = 6).
                stride_eff = max(1, int(math.ceil(stage.run_steps / max(1, frames))))
                dump_lines = f"""
# trajectory dump disabled
dump d1 all custom {stride_eff} {stage.name}.lammpstrj id type xu yu zu
 dump_modify d1 sort id first yes
""".rstrip()
                run_lines = f"run {stage.run_steps}\nundump d1"
            else:
                pre_steps = int(stage.run_steps - tail_steps)
                dump_lines = f"""
# trajectory tail
# pre dump
run {pre_steps}
dump d1 all custom {stride} {stage.name}.lammpstrj id type xu yu zu
 dump_modify d1 sort id
run {tail_steps}
undump d1
""".rstrip()
                run_lines = ""  # included inside dump
        else:
            dump_lines = f"""
# trajectory unwrapped coords
dump d1 all custom {dump_every} {stage.name}.lammpstrj id type xu yu zu
 dump_modify d1 sort id
""".rstrip()
            run_lines = f"run {stage.run_steps}\nundump d1"
    init_block = "\n".join(potential_init_lines(pot))

    script = f"""# vitriflow generated stage: {stage.name}

{init_block}
atom_style {md.atom_style}
boundary p p p
atom_modify map array

read_data {stage.input_data.name}
{rep_line}
{force_iso_block}

{mass_section}

# potential force field
{potential_block}

# neighbour conservative stability
neighbor {md.neighbor_skin} bin
neigh_modify every 1 delay 0 check yes

timestep {md.timestep}

thermo_style custom step time temp press pe ke etotal vol density lx ly lz
thermo {md.thermo_every}
thermo_modify flush yes

# velocities
{velocity_line}

{fix_line_equil}
run {stage.equil_steps}
{unfix_line_equil}

# sampling msd
reset_timestep 0
{fix_line_sample}

compute msd_all all msd com yes
fix msd_out all ave/time {stage.msd_every} 1 {stage.msd_every} c_msd_all[4] file {stage.name}.msd.dat

{dump_lines}

{run_lines}

# cleanup
unfix msd_out
uncompute msd_all
{unfix_line_sample}

write_data {stage.output_data.name} nocoeff
"""
    # remove blank caused
    lines = [ln.rstrip() for ln in script.splitlines()]
    # exactly consecutive blank
    cleaned = []
    for ln in lines:
        if ln.strip() == "" and (len(cleaned) == 0 or cleaned[-1].strip() == ""):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip() + "\n"


def _render_dump_and_run(
    stage: StageSpec,
    *,
    dump_filename: str,
) -> tuple[str, str]:
    """Dump and run."""

    # dump strategy
    tail_dump = bool(stage.write_dump and stage.tail_dump_frames and stage.tail_dump_stride)
    dump_lines = ""
    run_lines = ""

    dump_every = stage.dump_every
    if dump_every is None:
        # dumping enabled frequency
        dump_every = 100

    if not stage.write_dump:
        run_lines = f"run {stage.run_steps}"
        return dump_lines, run_lines

    if tail_dump:
        frames = int(stage.tail_dump_frames or 0)
        stride = int(stage.tail_dump_stride or 0)
        if frames < 1:
            frames = 1
        if stride < 1:
            stride = max(1, int(dump_every))
        tail_steps = int(frames * stride)
        if stage.run_steps <= tail_steps:
            # See render_stage(): `first yes` is robustness-only -- it
            # guarantees step 0 is a frame so very short runs are not silently
            # left with zero dumps. The frame count is NOT pinned to the
            # requested {frames}; it can overshoot by one when run_steps does
            # not divide cleanly by stride_eff.
            stride_eff = max(1, int(math.ceil(stage.run_steps / max(1, frames))))
            dump_lines = f"""
# trajectory dump disabled
dump d1 all custom {stride_eff} {dump_filename} id type xu yu zu
 dump_modify d1 sort id first yes
""".rstrip()
            run_lines = f"run {stage.run_steps}\nundump d1"
        else:
            pre_steps = int(stage.run_steps - tail_steps)
            dump_lines = f"""
# trajectory tail
# pre dump
run {pre_steps}
dump d1 all custom {stride} {dump_filename} id type xu yu zu
 dump_modify d1 sort id
run {tail_steps}
undump d1
""".rstrip()
            run_lines = ""  # included inside dump
        return dump_lines, run_lines

    # dump throughout
    dump_lines = f"""
# trajectory unwrapped coords
dump d1 all custom {int(dump_every)} {dump_filename} id type xu yu zu
 dump_modify d1 sort id
""".rstrip()
    run_lines = f"run {stage.run_steps}\nundump d1"
    return dump_lines, run_lines


def render_continuous_stages(
    pot: PotentialConfig,
    md: MDConfig,
    stages: Sequence[StageSpec],
    *,
    stage_dir_prefixes: dict[str, str],
    log_name: str = "log.lammps",
) -> str:
    """Continuous stages."""

    if not stages:
        raise ValueError("render_continuous_stages: no stages provided")

    # guard
    # directories exist
    for i, st in enumerate(stages):
        if str(st.name) not in stage_dir_prefixes:
            raise ValueError(f"Missing stage_dir_prefix for stage '{st.name}'")
        vmode = str(getattr(st, "velocity_mode", "create")).strip().lower()
        if vmode not in {"create", "preserve"}:
            raise ValueError(f"Unknown velocity_mode for stage '{st.name}': {getattr(st, 'velocity_mode', None)!r}")
        if i > 0 and vmode == "create":
            raise ValueError(
                "Continuous LAMMPS pipelines only support velocity creation on the first stage; "
                f"stage '{st.name}' requested velocity_mode='create'"
            )

    # potential consistent pipeline
    def _pot_lines_for(s: StageSpec) -> list[str]:
        if s.potential_lines is not None:
            return [str(x).strip() for x in s.potential_lines if str(x).strip()]
        return list(potential_default_lines(pot))

    base_pot_lines = _pot_lines_for(stages[0])
    for st in stages[1:]:
        if _pot_lines_for(st) != base_pot_lines:
            raise ValueError(
                "Continuous LAMMPS pipeline requires identical potential lines across stages. "
                f"Stage '{stages[0].name}' and '{st.name}' differ."
            )

    interactions = potential_interactions_list(pot)
    mass_block = _mass_lines_from_interactions(interactions, md=md, datafile=stages[0].input_data)
    mass_section = ""
    if mass_block:
        mode = str(getattr(md, "mass_mode", "auto")).strip().lower()
        mass_section = f"# Masses ({mode})\n" + str(mass_block)

    init_block = "\n".join(potential_init_lines(pot))
    potential_block = "\n".join(base_pot_lines)

    # replicate immediately structure
    rep_line = ""
    rep = stages[0].replicate
    if rep is not None:
        rx, ry, rz = (int(rep[0]), int(rep[1]), int(rep[2]))
        rep_line = f"replicate {rx} {ry} {rz}"

    # velocities beginning pipeline
    vmode0 = str(getattr(stages[0], "velocity_mode", "create")).strip().lower()
    if vmode0 == "create":
        velocity_line = f"velocity all create {stages[0].temperature_start} {stages[0].seed} mom yes rot yes dist gaussian"
    elif vmode0 == "preserve":
        velocity_line = "# initial velocities: preserved (read from data file)"
    else:
        raise ValueError(f"Unknown velocity_mode: {stages[0].velocity_mode}")

    # blocks
    stage_blocks: list[str] = []
    for st in stages:
        sdir = str(stage_dir_prefixes[str(st.name)])
        dump_file = f"{sdir}/{st.name}.lammpstrj"
        dump_lines, run_lines = _render_dump_and_run(st, dump_filename=dump_file)

        # integration sample mirrors the configured thermostat/barostat styles.
        fix_line_equil, unfix_line_equil = _lammps_integration_block(
            md,
            md.ensemble,
            Tstart=float(st.temperature_start),
            Tstop=float(st.temperature_start),
            pressure=float(st.pressure),
            seed=int(st.seed),
        )
        sample_ens = st.sample_ensemble or md.ensemble
        fix_line_sample, unfix_line_sample = _lammps_integration_block(
            md,
            sample_ens,
            Tstart=float(st.temperature_start),
            Tstop=float(st.temperature_stop),
            pressure=float(st.pressure),
            seed=int(st.seed),
        )

        force_iso_block = _force_isotropic_block(st)

        block = f"""# VITRIFLOW_STAGE: {st.name}

# redirect log output
log {sdir}/{log_name}

# origin compatibility plotting
reset_timestep 0
{force_iso_block}

{fix_line_equil}
run {st.equil_steps}
{unfix_line_equil}

# sampling msd
reset_timestep 0
{fix_line_sample}

compute msd_all all msd com yes
fix msd_out all ave/time {st.msd_every} 1 {st.msd_every} c_msd_all[4] file {sdir}/{st.name}.msd.dat

{dump_lines}

{run_lines}

# snapshot output materialization
write_dump all custom {sdir}/{st.name}.final.lammpstrj id type xu yu zu modify sort id

# cleanup
unfix msd_out
uncompute msd_all
{unfix_line_sample}
"""
        stage_blocks.append(block)

    script = f"""# vitriflow generated continuous pipeline

{init_block}
atom_style {md.atom_style}
boundary p p p
atom_modify map array

read_data {stages[0].input_data.name}
{rep_line}

{mass_section}

# potential
{potential_block}

neighbor {md.neighbor_skin} bin
neigh_modify every 1 delay 0 check yes

timestep {md.timestep}

thermo_style custom step time temp press pe ke etotal vol density lx ly lz
thermo {md.thermo_every}
thermo_modify flush yes

# velocities applied once
{velocity_line}

""" + "\n\n".join(stage_blocks)

    # clean blank return
    lines = [ln.rstrip() for ln in script.splitlines()]
    cleaned: list[str] = []
    for ln in lines:
        if ln.strip() == "" and (len(cleaned) == 0 or cleaned[-1].strip() == ""):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip() + "\n"


def render_elastic_screen(
    pot: PotentialConfig,
    md: MDConfig,
    *,
    input_data: Path,
    born_delta: float = 1.0e-5,
    raw_output_name: str = "born_raw.txt",
    stress_dump_name: str = "local_stress.dump",
    potential_lines: Optional[Sequence[str]] = None,
) -> str:
    """Elastic screen."""

    if not (math.isfinite(float(born_delta)) and float(born_delta) > 0.0):
        raise ValueError("born_delta must be finite and > 0")

    default_lines = potential_default_lines(pot)
    interactions = potential_interactions_list(pot)
    mass_block = _mass_lines_from_interactions(interactions, md=md, datafile=input_data)
    mass_section = ""
    if mass_block:
        mode = str(getattr(md, 'mass_mode', 'auto')).strip().lower()
        mass_section = f"# Masses ({mode})\n" + str(mass_block)

    potential_block = "\n".join(default_lines)
    if potential_lines is not None:
        lines = [str(x).strip() for x in potential_lines if str(x).strip() != ""]
        if len(lines) > 0:
            potential_block = "\n".join(lines)

    init_block = "\n".join(potential_init_lines(pot))
    born_vars = "\n".join([f"variable B{i} equal c_born[{i}]" for i in range(1, 22)])
    print_tokens = " ".join(["vol=${VOL}"] + [f"S{i}=${{S{i}}}" for i in range(1, 7)] + [f"B{i}=${{B{i}}}" for i in range(1, 22)])

    script = f"""# vitriflow elastic screen

{init_block}
atom_style {md.atom_style}
boundary p p p
atom_modify map array

read_data {input_data.name}

{mass_section}

# potential force field
{potential_block}

neighbor {md.neighbor_skin} bin
neigh_modify every 1 delay 0 check yes

timestep {md.timestep}

thermo_style custom step temp press pe vol lx ly lz
thermo 1
thermo_modify flush yes

compute vir all pressure NULL virial
compute born all born/matrix numdiff {float(born_delta):.8g} vir
compute pst all stress/atom NULL virial

run 0

# stress standard order
# xx yy zz
variable S1 equal -c_vir[1]
variable S2 equal -c_vir[2]
variable S3 equal -c_vir[3]
variable S4 equal -c_vir[6]
variable S5 equal -c_vir[5]
variable S6 equal -c_vir[4]
variable VOL equal vol
{born_vars}

print "{print_tokens}" file {raw_output_name} screen no
write_dump all custom {stress_dump_name} id type x y z c_pst[1] c_pst[2] c_pst[3] c_pst[4] c_pst[5] c_pst[6] modify sort id
"""

    lines = [ln.rstrip() for ln in script.splitlines()]
    cleaned: list[str] = []
    for ln in lines:
        if ln.strip() == "" and (len(cleaned) == 0 or cleaned[-1].strip() == ""):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip() + "\n"
