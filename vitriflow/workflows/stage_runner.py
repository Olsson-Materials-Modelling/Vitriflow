from __future__ import annotations

"""Stage execution utilities shared by multiple workflows."""

import csv
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from ..lammps_input import StageSpec, render_continuous_stages, render_stage
from ..parse import ThermoTable, parse_last_thermo_table, parse_msd_file
from ..io.thermo import write_thermo_csv, write_msd_csv
from ..io.extxyz import write_extxyz_frames, write_extxyz_iter, write_extxyz_single
from ..io.lammps_data_minimal import write_dumpframe_lammps_data
from ..analysis.dump import read_last_dump_frame
from ..analysis.msd import estimate_diffusion_from_msd
from ..analysis.stats import window_mean_stderr
from ..analysis.datafile import (
    count_atoms_in_datafile,
    datafile_has_velocities,
    read_datafile_charges,
    read_datafile_frame,
    read_datafile_masses,
    strip_lammps_data_pair_coeff_sections,
)
from ..potential import prepare_potential_files
from ..runner import Cp2kRunner, LammpsRunner
from ..utils import ensure_dir
from ..cp2k_driver import (
    compute_msd,
    density_g_cm3_from_atoms,
    parse_cp2k_ener,
    render_cp2k_md_input,
    unwrap_positions_fractional,
)
from .autoskin import run_with_neighbor_skin_autotune


LOG = logging.getLogger(__name__)

_ARTIFACT_IO_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    ValueError,
    TypeError,
    IndexError,
    KeyError,
    RuntimeError,
    csv.Error,
)
_ARTIFACT_IMPORT_EXCEPTIONS = (ImportError, ModuleNotFoundError)
_ARTIFACT_EXPORT_EXCEPTIONS = _ARTIFACT_IO_EXCEPTIONS + _ARTIFACT_IMPORT_EXCEPTIONS
_ARTIFACT_COPY_EXCEPTIONS = (OSError, RuntimeError, shutil.Error)


@dataclass(frozen=True)
class StageArtifacts:
    """Stage artifacts."""

    stage_dir: Path
    input_local: Path
    output_local: Path
    log_path: Path
    msd_path: Path
    dump_path: Optional[Path]
    neighbor_skin: float
    neighbor_skin_retries: int

    # neutral outputs analysis
    thermo_csv: Path
    msd_csv: Path
    traj_extxyz: Optional[Path]
    final_extxyz: Path


@dataclass(frozen=True)
class StageOutcome:
    """Stage outcome."""

    name: str
    temperature_start: float
    temperature_stop: float
    pressure: float
    equil_steps: int
    run_steps: int
    seed: int
    n_atoms: int
    vol_last: float
    density_mean: float
    density_stderr: float
    pe_mean: float
    pe_stderr: float
    D: float
    D_stderr: float
    msd_rms_last: float
    output_data: str  # relative path
    dump: Optional[str] = None  # relative path

    neighbor_skin: float = float("nan")
    neighbor_skin_retries: int = 0

    gr_peak_r: float = float("nan")
    gr_peak_height: float = float("nan")
    gr_peak_fwhm: float = float("nan")

    rep_id: Optional[int] = None


@dataclass(frozen=True)
class _LocalizedLammpsStage:
    """Localized lammps stage."""

    requested_stage: StageSpec
    local_stage: StageSpec
    stage_dir: Path
    runner_input: Path
    artifact_input: Path
    output_local: Path
    dump_path: Optional[Path]
    final_dump_path: Path

    def log_path(self, log_name: str) -> Path:
        return self.stage_dir / str(log_name)

    @property
    def msd_path(self) -> Path:
        return self.stage_dir / f"{self.local_stage.name}.msd.dat"

    @property
    def thermo_csv_path(self) -> Path:
        return self.stage_dir / "thermo.csv"

    @property
    def msd_csv_path(self) -> Path:
        return self.stage_dir / "msd.csv"

    @property
    def traj_extxyz_path(self) -> Path:
        return self.stage_dir / "traj.extxyz"

    @property
    def final_extxyz_path(self) -> Path:
        return self.stage_dir / "final.extxyz"

    def cleanup_paths(self, *, log_name: str, include_postprocessed: bool) -> list[Path]:
        out = [self.output_local, self.msd_path, self.final_dump_path]
        if include_postprocessed:
            out.extend(
                [
                    self.log_path(log_name),
                    self.thermo_csv_path,
                    self.msd_csv_path,
                    self.traj_extxyz_path,
                    self.final_extxyz_path,
                ]
            )
        if self.dump_path is not None:
            out.append(self.dump_path)
        return out


class ArtifactMaterializationError(RuntimeError):
    """Artifact materialization error."""


class ThermoArtifactError(ArtifactMaterializationError):
    """Thermo artifact error."""


class MSDArtifactError(ArtifactMaterializationError):
    """Msdartifact error."""


class OutputLocalizationError(ArtifactMaterializationError):
    """Output localization error."""


class InputSnapshotError(ArtifactMaterializationError):
    """Input snapshot error."""


def _log_best_effort_failure(exc: BaseException) -> None:
    """Log best effort."""

    cause = exc.__cause__
    if cause is None:
        LOG.warning("%s", exc)
        return
    LOG.warning("%s (caused by %s: %s)", exc, type(cause).__name__, cause)
    LOG.debug(
        "Best-effort artifact failure traceback",
        exc_info=(type(cause), cause, cause.__traceback__),
    )


def _write_placeholder_artifact(path: Path, *, label: str) -> Path:
    """Placeholder artifact."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    except OSError as exc:
        LOG.warning(
            "Failed to create empty %s placeholder at %s (%s: %s)",
            label,
            path,
            type(exc).__name__,
            exc,
        )
    return path


def _remove_partial_artifact(path: Path) -> None:
    """Remove partial artifact."""

    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        LOG.debug(
            "Failed to remove partial artifact %s (%s: %s)",
            path,
            type(exc).__name__,
            exc,
        )


def _localized_output_name(output_data: Optional[Union[str, Path]]) -> str:
    if output_data is None:
        return "output.data"
    return Path(output_data).name


def _resolved_velocity_mode(stage: StageSpec, *, input_data: Optional[Path]) -> str:
    vel_mode = str(getattr(stage, "velocity_mode", "create")).strip().lower()
    if input_data is not None and vel_mode == "preserve" and not datafile_has_velocities(input_data):
        return "create"
    return vel_mode


def _localize_lammps_stage(
    stage: StageSpec,
    *,
    runner_input: Path,
    artifact_input: Path,
    stage_dir: Path,
    potential_lines: Optional[list[str]],
    check_input_velocities: bool,
) -> _LocalizedLammpsStage:
    """Localize lammps stage."""

    vel_mode = _resolved_velocity_mode(
        stage,
        input_data=runner_input if check_input_velocities else None,
    )
    output_local = Path(stage_dir) / _localized_output_name(stage.output_data)
    local_stage = StageSpec(
        name=stage.name,
        input_data=Path(runner_input),
        output_data=Path(output_local),
        sample_ensemble=stage.sample_ensemble,
        temperature_start=stage.temperature_start,
        temperature_stop=stage.temperature_stop,
        pressure=stage.pressure,
        equil_steps=stage.equil_steps,
        run_steps=stage.run_steps,
        seed=stage.seed,
        velocity_mode=vel_mode,
        force_isotropic=bool(getattr(stage, "force_isotropic", False)),
        replicate=stage.replicate,
        write_dump=stage.write_dump,
        dump_every=stage.dump_every,
        tail_dump_frames=stage.tail_dump_frames,
        tail_dump_stride=stage.tail_dump_stride,
        msd_every=stage.msd_every,
        potential_lines=(stage.potential_lines if stage.potential_lines is not None else potential_lines),
    )
    dump_path = Path(stage_dir) / f"{stage.name}.lammpstrj" if stage.write_dump else None
    return _LocalizedLammpsStage(
        requested_stage=stage,
        local_stage=local_stage,
        stage_dir=Path(stage_dir),
        runner_input=Path(runner_input),
        artifact_input=Path(artifact_input),
        output_local=Path(output_local),
        dump_path=dump_path,
        final_dump_path=Path(stage_dir) / f"{stage.name}.final.lammpstrj",
    )


def _materialize_output_from_final_dump(
    *,
    output_local: Path,
    final_dump_path: Path,
    template_input: Path,
    atom_style: str,
) -> Path:
    """Materialize output from."""

    frame = read_last_dump_frame(Path(final_dump_path))
    masses_by_type = read_datafile_masses(Path(template_input))
    charges_by_id = read_datafile_charges(Path(template_input), atom_style=str(atom_style))

    write_dumpframe_lammps_data(
        Path(output_local),
        frame,
        atom_style=str(atom_style),
        masses_by_type=(masses_by_type if masses_by_type else None),
        charges_by_id=(charges_by_id if str(atom_style).strip().lower() == "charge" else None),
    )
    return Path(output_local)


def _materialize_thermo_csv_from_log(*, log_path: Path, thermo_csv: Path) -> Path:
    """Materialize thermo csv."""

    try:
        table = parse_last_thermo_table(log_path)
    except _ARTIFACT_IO_EXCEPTIONS as exc:
        raise ThermoArtifactError(
            f"Failed to parse thermo table from {log_path} for {thermo_csv}"
        ) from exc
    try:
        write_thermo_csv(thermo_csv, table)
    except _ARTIFACT_IO_EXCEPTIONS as exc:
        raise ThermoArtifactError(
            f"Failed to write engine-neutral thermo CSV {thermo_csv} from {log_path}"
        ) from exc
    return thermo_csv


def _materialize_thermo_csv_from_table(*, table: ThermoTable, thermo_csv: Path) -> Path:
    """Materialize thermo csv."""

    try:
        write_thermo_csv(thermo_csv, table)
    except _ARTIFACT_IO_EXCEPTIONS as exc:
        raise ThermoArtifactError(f"Failed to write engine-neutral thermo CSV {thermo_csv}") from exc
    return thermo_csv


def _materialize_msd_csv_from_file(*, msd_path: Path, msd_csv: Path) -> Path:
    """Materialize msd csv."""

    try:
        msd = parse_msd_file(msd_path)
    except _ARTIFACT_IO_EXCEPTIONS as exc:
        raise MSDArtifactError(f"Failed to parse MSD series from {msd_path} for {msd_csv}") from exc
    try:
        write_msd_csv(msd_csv, step=msd.step, msd=msd.msd)
    except _ARTIFACT_IO_EXCEPTIONS as exc:
        raise MSDArtifactError(
            f"Failed to write engine-neutral MSD CSV {msd_csv} from {msd_path}"
        ) from exc
    return msd_csv


def _write_input_snapshot(*, source: Path, destination: Path) -> Path:
    """Input snapshot."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
    except (FileNotFoundError, OSError) as exc:
        raise InputSnapshotError(f"Failed to write stage input snapshot {destination} from {source}") from exc
    return destination


def _copy_requested_output_if_needed(
    *,
    requested_output: Optional[Union[str, Path]],
    output_local: Path,
    stage_dir: Path,
) -> None:
    """Copy requested output."""

    if requested_output is None:
        return

    requested = Path(requested_output).expanduser()
    if not (requested.is_absolute() or requested.parent != stage_dir):
        return

    try:
        requested.parent.mkdir(parents=True, exist_ok=True)
        if requested.resolve() == output_local.resolve():
            return
        shutil.copy2(output_local, requested)
    except _ARTIFACT_COPY_EXCEPTIONS as exc:
        raise OutputLocalizationError(
            f"Failed to copy localized output {output_local} to requested path {requested}"
        ) from exc


def _postprocess_lammps_stage(
    plan: _LocalizedLammpsStage,
    *,
    md_cfg,
    log_name: str,
    type_to_species: Optional[Sequence[str]],
    neighbor_skin: float,
    neighbor_skin_retries: int,
    input_snapshot_source: Optional[Path] = None,
) -> StageArtifacts:
    """Postprocess lammps stage."""

    log_path = plan.log_path(log_name)
    msd_path = plan.msd_path
    thermo_csv = plan.thermo_csv_path
    msd_csv = plan.msd_csv_path

    if input_snapshot_source is not None:
        try:
            _write_input_snapshot(source=input_snapshot_source, destination=plan.artifact_input)
        except InputSnapshotError as exc:
            _log_best_effort_failure(exc)

    if not Path(plan.output_local).exists() and Path(plan.final_dump_path).exists():
        _materialize_output_from_final_dump(
            output_local=plan.output_local,
            final_dump_path=plan.final_dump_path,
            template_input=plan.artifact_input,
            atom_style=str(md_cfg.atom_style),
        )

    try:
        _materialize_thermo_csv_from_log(log_path=log_path, thermo_csv=thermo_csv)
    except ThermoArtifactError as exc:
        _log_best_effort_failure(exc)
        thermo_csv = _write_placeholder_artifact(thermo_csv, label="thermo.csv")

    try:
        _materialize_msd_csv_from_file(msd_path=msd_path, msd_csv=msd_csv)
    except MSDArtifactError as exc:
        _log_best_effort_failure(exc)
        msd_csv = _write_placeholder_artifact(msd_csv, label="msd.csv")

    traj_extxyz, final_extxyz = _materialize_lammps_engine_neutral_outputs(
        stage_dir=plan.stage_dir,
        output_data=plan.output_local,
        dump_path=plan.dump_path,
        md_cfg=md_cfg,
        type_to_species=type_to_species,
    )

    try:
        _copy_requested_output_if_needed(
            requested_output=plan.requested_stage.output_data,
            output_local=plan.output_local,
            stage_dir=plan.stage_dir,
        )
    except OutputLocalizationError as exc:
        _log_best_effort_failure(exc)

    return StageArtifacts(
        stage_dir=plan.stage_dir,
        input_local=plan.artifact_input,
        output_local=plan.output_local,
        log_path=log_path,
        msd_path=msd_path,
        dump_path=plan.dump_path,
        neighbor_skin=float(neighbor_skin),
        neighbor_skin_retries=int(neighbor_skin_retries),
        thermo_csv=thermo_csv,
        msd_csv=msd_csv,
        traj_extxyz=traj_extxyz,
        final_extxyz=final_extxyz,
    )


def _materialize_lammps_engine_neutral_outputs(
    *,
    stage_dir: Path,
    output_data: Path,
    dump_path: Optional[Path],
    md_cfg,
    type_to_species: Optional[Sequence[str]] = None,
) -> tuple[Optional[Path], Path]:
    """Materialize lammps engine."""

    traj_extxyz: Optional[Path] = None
    final_extxyz = Path(stage_dir) / "final.extxyz"

    last_dump_frame = None
    if dump_path is not None and Path(dump_path).exists():
        _traj = Path(stage_dir) / "traj.extxyz"
        try:
            from ..analysis.dump import iter_dump_frames

            last_dump_frame = write_extxyz_iter(
                _traj,
                iter_dump_frames(Path(dump_path)),
                type_to_species=type_to_species,
            )
            traj_extxyz = _traj
        except _ARTIFACT_EXPORT_EXCEPTIONS as exc:
            _remove_partial_artifact(_traj)
            LOG.warning(
                "Failed to convert LAMMPS dump %s to traj.extxyz in %s; falling back to last-frame extraction (%s: %s)",
                dump_path,
                stage_dir,
                type(exc).__name__,
                exc,
            )
            try:
                last_dump_frame = read_last_dump_frame(Path(dump_path))
            except _ARTIFACT_EXPORT_EXCEPTIONS as frame_exc:
                LOG.warning(
                    "Failed to recover the last frame from LAMMPS dump %s for final.extxyz fallback (%s: %s)",
                    dump_path,
                    type(frame_exc).__name__,
                    frame_exc,
                )
                last_dump_frame = None

    # prefer snapshot extxyz
    try:
        fr = read_datafile_frame(Path(output_data), atom_style=str(md_cfg.atom_style))
        write_extxyz_single(final_extxyz, fr, type_to_species=type_to_species)
        return traj_extxyz, final_extxyz
    except _ARTIFACT_EXPORT_EXCEPTIONS as exc:
        _remove_partial_artifact(final_extxyz)
        LOG.warning(
            "Failed to export final.extxyz from %s in %s; attempting fallbacks (%s: %s)",
            output_data,
            stage_dir,
            type(exc).__name__,
            exc,
        )

    # fallback
    if last_dump_frame is not None:
        try:
            write_extxyz_single(final_extxyz, last_dump_frame, type_to_species=type_to_species)
            return traj_extxyz, final_extxyz
        except _ARTIFACT_EXPORT_EXCEPTIONS as exc:
            _remove_partial_artifact(final_extxyz)
            LOG.warning(
                "Failed dump-frame fallback while exporting final.extxyz from %s (%s: %s)",
                output_data,
                type(exc).__name__,
                exc,
            )

    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write

        at = ase_read(
            str(Path(output_data)),
            format="lammps-data",
            style=str(md_cfg.atom_style),
            specorder=list(type_to_species) if type_to_species is not None else None,
        )
        ase_write(str(final_extxyz), at, format="extxyz")
    except _ARTIFACT_EXPORT_EXCEPTIONS as exc:
        _remove_partial_artifact(final_extxyz)
        LOG.warning(
            "Failed ASE fallback while exporting final.extxyz from %s in %s (%s: %s)",
            output_data,
            stage_dir,
            type(exc).__name__,
            exc,
        )

    return traj_extxyz, final_extxyz


def run_stage_local(
    runner: Union[LammpsRunner, Cp2kRunner],
    pot_cfg,
    md_cfg,
    stage: StageSpec,
    stage_dir: Path,
    *,
    potential_lines: Optional[list[str]] = None,
    log_name: str = "log.lammps",
    type_to_species: Optional[Sequence[str]] = None,
) -> StageArtifacts:
    """Stage local."""

    if isinstance(runner, Cp2kRunner):
        if type_to_species is None:
            raise ValueError("CP2K stages require type_to_species (LAMMPS type -> element symbols)")
        return _run_stage_local_cp2k(
            runner,
            md_cfg,
            stage,
            stage_dir,
            type_to_species=type_to_species,
            log_name=log_name,
        )

    if not isinstance(runner, LammpsRunner):
        raise TypeError(f"Unsupported runner type: {type(runner)}")

    if pot_cfg is None:
        raise ValueError("LAMMPS stages require a potential configuration (kim/potential)")

    ensure_dir(stage_dir)

    input_local = stage_dir / "input.data"
    input_local.write_bytes(Path(stage.input_data).read_bytes())
    strip_lammps_data_pair_coeff_sections(input_local)

    plan = _localize_lammps_stage(
        stage,
        runner_input=input_local,
        artifact_input=input_local,
        stage_dir=stage_dir,
        potential_lines=potential_lines,
        check_input_velocities=True,
    )

    prepare_potential_files(pot_cfg, stage_dir, potential_lines)

    skin_used, skin_retries = run_with_neighbor_skin_autotune(
        runner,
        lambda md_use: render_stage(pot_cfg, md_use, plan.local_stage),
        stage_dir,
        md_cfg,
        log_name=log_name,
        cleanup_paths=plan.cleanup_paths(log_name=log_name, include_postprocessed=False),
    )

    return _postprocess_lammps_stage(
        plan,
        md_cfg=md_cfg,
        log_name=log_name,
        type_to_species=type_to_species,
        neighbor_skin=float(skin_used),
        neighbor_skin_retries=int(skin_retries),
    )


def run_stages_continuous_lammps(
    runner: LammpsRunner,
    pot_cfg,
    md_cfg,
    stages: Sequence[StageSpec],
    stage_dirs: Sequence[Path],
    workdir: Path,
    *,
    potential_lines: Optional[list[str]] = None,
    log_name: str = "log.lammps",
    type_to_species: Optional[Sequence[str]] = None,
) -> list[StageArtifacts]:
    """Stages continuous lammps."""

    if not isinstance(runner, LammpsRunner):
        raise TypeError("run_stages_continuous_lammps requires a LammpsRunner")
    if pot_cfg is None:
        raise ValueError("LAMMPS stages require a potential configuration (kim/potential)")
    if len(stages) < 1:
        raise ValueError("No stages provided")
    if len(stage_dirs) != len(stages):
        raise ValueError("stage_dirs length must match stages length")

    workdir = Path(workdir)
    ensure_dir(workdir)

    # localize structure workdir
    input0 = Path(stages[0].input_data)
    input_local0 = workdir / "input.data"
    input_local0.write_bytes(input0.read_bytes())
    strip_lammps_data_pair_coeff_sections(input_local0)

    # continuous pipelines replication
    for st in stages[1:]:
        if getattr(st, "replicate", None) is not None:
            raise ValueError("Continuous LAMMPS pipeline supports replicate only on the first stage")

    # directories exist
    for sd in stage_dirs:
        ensure_dir(sd)

    # potential working directory
    prepare_potential_files(pot_cfg, workdir, potential_lines)

    # local specs rendering
    stage_plans: list[_LocalizedLammpsStage] = []
    stage_dir_prefixes: dict[str, str] = {}

    for idx, (st, sd) in enumerate(zip(stages, stage_dirs)):
        # relative workdir directory
        # lammps script
        rel = os.path.relpath(str(sd), start=str(workdir))
        stage_dir_prefixes[str(st.name)] = rel.replace(os.sep, "/")

        # continuous pipeline input
        # input consumed renderer
        stage_plans.append(
            _localize_lammps_stage(
                st,
                runner_input=input_local0,
                artifact_input=Path(sd) / "input.data",
                stage_dir=Path(sd),
                potential_lines=potential_lines,
                check_input_velocities=(idx == 0),
            )
        )

    # snapshot compatibility provenance
    try:
        _write_input_snapshot(source=input_local0, destination=stage_plans[0].artifact_input)
    except InputSnapshotError as exc:
        _log_best_effort_failure(exc)

    # neighbor skin partially
    cleanup: list[Path] = []
    for plan in stage_plans:
        cleanup.extend(plan.cleanup_paths(log_name=log_name, include_postprocessed=True))

    # workdir screen outputs
    cleanup.append(workdir / log_name)
    cleanup.append(workdir / "screen.out")

    skin_used, skin_retries = run_with_neighbor_skin_autotune(
        runner,
        lambda md_use: render_continuous_stages(
            pot_cfg,
            md_use,
            [plan.local_stage for plan in stage_plans],
            stage_dir_prefixes=stage_dir_prefixes,
            log_name=log_name,
        ),
        workdir,
        md_cfg,
        log_name=log_name,
        cleanup_paths=cleanup,
    )

    # thermo msd extxyz
    arts: list[StageArtifacts] = []
    for i, plan in enumerate(stage_plans):
        prev_output = stage_plans[i - 1].output_local if i > 0 else None
        arts.append(
            _postprocess_lammps_stage(
                plan,
                md_cfg=md_cfg,
                log_name=log_name,
                type_to_species=type_to_species,
                neighbor_skin=float(skin_used),
                neighbor_skin_retries=int(skin_retries),
                input_snapshot_source=prev_output,
            )
        )

    return arts


def _gcd_many(vals: Sequence[int]) -> int:
    import math

    g = 0
    for v in vals:
        g = math.gcd(g, int(v))
    return int(g)


def _distribute_steps(total: int, nseg: int) -> list[int]:
    if nseg < 1:
        raise ValueError("nseg must be >= 1")
    if total < 0:
        raise ValueError("total must be >= 0")
    base = total // nseg
    rem = total % nseg
    out = [base + (1 if i < rem else 0) for i in range(nseg)]
    assert sum(out) == total
    return out


def _orthorhombic_box_from_cell(cell) -> tuple[float, float, float]:
    import numpy as np

    c = np.asarray(cell, dtype=float)
    if c.shape != (3, 3):
        raise ValueError("cell must be 3x3")
    # accept orthorhombic cells
    off = c.copy()
    off[0, 0] = 0.0
    off[1, 1] = 0.0
    off[2, 2] = 0.0
    if np.max(np.abs(off)) > 1.0e-6:
        raise ValueError("CP2K driver currently supports orthorhombic cells only")
    lx = float(c[0, 0])
    ly = float(c[1, 1])
    lz = float(c[2, 2])
    if not (lx > 0 and ly > 0 and lz > 0):
        raise ValueError("Cell lengths must be > 0")
    return lx, ly, lz



def _write_lammpstrj(
    path: Path,
    *,
    steps: Sequence[int],
    frac_positions: "np.ndarray",  # n frames atoms
    symbols: Sequence[str],
    type_to_species: Sequence[str],
    cells: "np.ndarray",  # n frames
) -> None:
    """Lammpstrj."""

    import numpy as np

    sym_to_type = {sym: i + 1 for i, sym in enumerate(type_to_species)}
    n_atoms = int(len(symbols))

    frac_positions = np.asarray(frac_positions, dtype=float)
    cells = np.asarray(cells, dtype=float)

    if frac_positions.ndim != 3 or frac_positions.shape[-1] != 3:
        raise ValueError("frac_positions must have shape (n_frames, n_atoms, 3)")
    if cells.ndim != 3 or cells.shape[-2:] != (3, 3):
        raise ValueError("cells must have shape (n_frames, 3, 3)")
    if frac_positions.shape[1] != n_atoms:
        raise ValueError("frac_positions atom dimension mismatch")
    if frac_positions.shape[0] != len(steps):
        raise ValueError("frac_positions frame dimension mismatch")
    if cells.shape[0] != len(steps):
        raise ValueError("cells frame dimension mismatch")

    with path.open("w") as f:
        for istep, sxyz, cell in zip(steps, frac_positions, cells):
            lx, ly, lz = _orthorhombic_box_from_cell(cell)
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{int(istep)}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n_atoms}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"0.0 {lx:.12f}\n")
            f.write(f"0.0 {ly:.12f}\n")
            f.write(f"0.0 {lz:.12f}\n")
            f.write("ITEM: ATOMS id type xs ys zs\n")
            for i, (sym, s) in enumerate(zip(symbols, sxyz), start=1):
                itype = sym_to_type.get(sym)
                if itype is None:
                    raise ValueError(f"Unknown symbol {sym!r} (missing from type_to_species)")
                sw = s - np.floor(s)
                f.write(
                    f"{i} {itype} {float(sw[0]): .12f} {float(sw[1]): .12f} {float(sw[2]): .12f}\n"
                )


def _build_cp2k_ramp_schedule(
    *,
    T_start: float,
    T_stop: float,
    total_steps: int,
    max_deltaT_K: float,
    max_segments: int,
) -> list[tuple[float, int]]:
    """Cp2k ramp schedule."""

    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")

    T0 = float(T_start)
    T1 = float(T_stop)
    dT = T1 - T0

    if abs(dT) == 0.0:
        return [(T1, int(total_steps))]

    max_deltaT = float(max_deltaT_K)
    if not (math.isfinite(max_deltaT) and max_deltaT > 0.0):
        raise ValueError("max_deltaT_K must be finite and > 0")

    mmax = int(max_segments)
    if mmax < 1:
        raise ValueError("max_segments must be >= 1")

    # lower bound step
    m0 = int(math.ceil(abs(dT) / max_deltaT))
    m0 = max(1, m0)

    if m0 > mmax:
        raise ValueError(
            "CP2K ramp discretization infeasible: "
            f"|ΔT|={abs(dT):g} K requires at least {m0} segments for "
            f"max_deltaT_K={max_deltaT:g}, but ramp_max_segments={mmax}."
        )

    if m0 > total_steps:
        # segment satisfy constraint
        raise ValueError(
            "CP2K ramp discretization infeasible: "
            f"total_steps={total_steps} is too small to realize |ΔT|={abs(dT):g} K "
            f"with max_deltaT_K={max_deltaT:g}. Increase quench duration or relax max_deltaT_K."
        )

    # smallest satisfies distribution
    for m in range(m0, mmax + 1):
        if m > total_steps:
            break
        steps_list = _distribute_steps(int(total_steps), int(m))
        max_jump = abs(dT) * (max(steps_list) / float(total_steps))
        if max_jump <= max_deltaT * (1.0 + 1.0e-12) + 1.0e-12:
            schedule: list[tuple[float, int]] = []
            cum = 0
            for n in steps_list:
                Ti = T0 + dT * (cum / float(total_steps))
                schedule.append((float(Ti), int(n)))
                cum += int(n)
            return schedule

    raise ValueError(
        "CP2K ramp discretization infeasible with current constraints. "
        "Increase cp2k.ramp_max_segments or relax cp2k.ramp_max_deltaT_K."
    )


def _select_cp2k_export_indices(
    *,
    steps_all: "np.ndarray",
    run_steps: int,
    tail_dump_frames: Optional[int],
    tail_dump_stride: Optional[int],
    dump_every: int,
    traj_every: int,
    write_dump: bool,
) -> list[int]:
    """Cp2k export indices."""

    import numpy as np

    if not bool(write_dump):
        return []

    run_steps = int(run_steps)
    frames_req = int(tail_dump_frames or 0)
    stride_req = int(tail_dump_stride or 0)
    dump_stride = int(dump_every) if int(dump_every) > 0 else int(traj_every)

    desired_steps: list[int]
    if frames_req > 0 and stride_req > 0:
        # exactly targets spaced
        start = max(0, int(run_steps - (frames_req - 1) * stride_req))
        desired_steps = [int(start + i * stride_req) for i in range(int(frames_req))]
        if not desired_steps or int(desired_steps[-1]) != int(run_steps):
            desired_steps.append(int(run_steps))
    else:
        # cadence
        if dump_stride < 1:
            dump_stride = 1
        desired_steps = list(range(0, int(run_steps) + 1, int(dump_stride)))
        if not desired_steps:
            desired_steps = [int(run_steps)]
        if int(desired_steps[-1]) != int(run_steps):
            desired_steps.append(int(run_steps))

    selected_out: list[int] = []
    for ds in desired_steps:
        j = int(np.searchsorted(steps_all, int(ds), side="right") - 1)
        j = max(0, min(j, len(steps_all) - 1))
        if not selected_out or j != selected_out[-1]:
            selected_out.append(j)

    # always include frame
    if selected_out and selected_out[-1] != len(steps_all) - 1:
        selected_out.append(len(steps_all) - 1)
    if not selected_out:
        selected_out = [len(steps_all) - 1]
    return selected_out


def _materialize_cp2k_engine_neutral_outputs(
    *,
    stage_dir: Path,
    steps_all: "np.ndarray",
    pos_all: "np.ndarray",
    cells_all: "np.ndarray",
    symbols: Sequence[str],
    type_to_species: Sequence[str],
    selected_out: Sequence[int],
    write_dump: bool,
) -> tuple[Optional[Path], Path]:
    """Materialize cp2k engine."""

    import numpy as np

    traj_path = Path(stage_dir) / "traj.extxyz"
    final_extxyz = Path(stage_dir) / "final.extxyz"
    traj_extxyz: Optional[Path] = None
    traj_written = False

    try:
        from ..analysis.dump import DumpFrame

        # symbols species ordering
        sym_to_type = {str(sym): i + 1 for i, sym in enumerate(list(type_to_species))}
        ids = np.arange(1, int(len(symbols)) + 1, dtype=int)
        types = np.asarray([sym_to_type.get(str(sym), 0) for sym in symbols], dtype=int)

        def _mk_frame(idx: int) -> DumpFrame:
            return DumpFrame(
                timestep=int(steps_all[idx]),
                ids=np.asarray(ids, dtype=int),
                types=np.asarray(types, dtype=int),
                positions=np.asarray(pos_all[idx], dtype=float),
                cell=np.asarray(cells_all[idx], dtype=float),
                origin=np.zeros((3,), dtype=float),
            )

        last_frame = _mk_frame(len(steps_all) - 1)

        if bool(write_dump) and selected_out:
            write_extxyz_frames(traj_path, [_mk_frame(i) for i in selected_out], type_to_species=type_to_species)
            traj_extxyz = traj_path
            traj_written = True

        write_extxyz_single(final_extxyz, last_frame, type_to_species=type_to_species)
    except _ARTIFACT_EXPORT_EXCEPTIONS as exc:
        if not traj_written:
            _remove_partial_artifact(traj_path)
            traj_extxyz = None
        _remove_partial_artifact(final_extxyz)
        LOG.warning(
            "Failed to materialize CP2K engine-neutral EXTXYZ outputs in %s (%s: %s)",
            stage_dir,
            type(exc).__name__,
            exc,
        )

    return traj_extxyz, final_extxyz


def _run_stage_local_cp2k(
    runner: Cp2kRunner,
    md_cfg,
    stage: StageSpec,
    stage_dir: Path,
    *,
    type_to_species: Sequence[str],
    log_name: str,
) -> StageArtifacts:
    """Stage local cp2k."""

    # ase import lightweight
    from ase.io import read as ase_read
    from ase.io.cp2k import read_cp2k_dcd
    import numpy as np

    ensure_dir(stage_dir)

    # locally basis pseudopotentials
    # absolute input subtle
    # cp2k env
    runner._ensure_data_files_present(stage_dir)
    import os
    basis_cfg = str(runner.cfg.basis_set_file_name)
    pot_cfg = str(runner.cfg.potential_file_name)

    # working directory filenames
    # supplied contained reliance
    # cp2 internal search
    if os.path.isabs(basis_cfg) or ("/" in basis_cfg) or ("\\" in basis_cfg):
        basis_file = basis_cfg
        basis_path = Path(basis_cfg)
    else:
        basis_path = stage_dir / Path(basis_cfg).name
        basis_file = basis_path.name

    if os.path.isabs(pot_cfg) or ("/" in pot_cfg) or ("\\" in pot_cfg):
        pot_file = pot_cfg
        pot_path = Path(pot_cfg)
    else:
        pot_path = stage_dir / Path(pot_cfg).name
        pot_file = pot_path.name

    # message basis potential
    # otherwise parsing actionable
    if not basis_path.exists():
        raise FileNotFoundError(
            "CP2K basis set file was not staged into the working directory. "
            f"Expected: {basis_path}. "
            "Either set cp2k.basis_set_file_name to an absolute path, or ensure the file is available via vitriflow/data/cp2k or CP2K_DATA_DIR."
        )
    if not pot_path.exists():
        raise FileNotFoundError(
            "CP2K potential file was not staged into the working directory. "
            f"Expected: {pot_path}. "
            "Either set cp2k.potential_file_name to an absolute path, or ensure the file is available via vitriflow/data/cp2k or CP2K_DATA_DIR."
        )

    input_local = stage_dir / "input.data"
    input_local.write_bytes(Path(stage.input_data).read_bytes())
    strip_lammps_data_pair_coeff_sections(input_local)

    out_name = Path(stage.output_data).name if stage.output_data is not None else "output.data"
    output_local = stage_dir / out_name

    # backward compatibility outputs
    # derived thermo table
    log_path = stage_dir / str(log_name)
    msd_path = stage_dir / f"{stage.name}.msd.dat"
    dump_path = stage_dir / f"{stage.name}.lammpstrj" if stage.write_dump else None

    thermo_csv = stage_dir / "thermo.csv"
    msd_csv = stage_dir / "msd.csv"
    traj_extxyz: Optional[Path] = None
    final_extxyz = stage_dir / "final.extxyz"

    # structure element species
    # ase lammps parser
    # ase parsing sensitive
    try:
        atoms = ase_read(
            str(input_local),
            format="lammps-data",
            style=str(md_cfg.atom_style),
            specorder=list(type_to_species),
        )
    except _ARTIFACT_EXPORT_EXCEPTIONS:
        from ..io.lammps_data_minimal import read_lammps_data_minimal

        atoms = read_lammps_data_minimal(
            input_local,
            atom_style=str(md_cfg.atom_style),
            specorder=list(type_to_species),
        )
    atoms.pbc = True

    # replicate requested
    if stage.replicate is not None:
        rx, ry, rz = stage.replicate
        atoms = atoms.repeat((int(rx), int(ry), int(rz)))
        atoms.pbc = True

    dt_fs = float(md_cfg.timestep)
    tdamp_fs = float(md_cfg.thermostat.tdamp)
    pdamp_fs = float(md_cfg.barostat.pdamp)

    if not (math.isfinite(dt_fs) and dt_fs > 0.0):
        raise ValueError("md.timestep must be finite and > 0 (interpreted as fs for CP2K)")
    if not (math.isfinite(tdamp_fs) and tdamp_fs > 0.0):
        raise ValueError("thermostat.tdamp must be finite and > 0 (interpreted as fs for CP2K)")
    if not (math.isfinite(pdamp_fs) and pdamp_fs > 0.0):
        raise ValueError("barostat.pdamp must be finite and > 0 (interpreted as fs for CP2K)")

    # ensembles ensemble sampling
    eq_ens = str(md_cfg.ensemble).strip().lower()
    sample_override = stage.sample_ensemble
    if isinstance(sample_override, bool):
        # defensive boolean certainly
        sample_override = None
    samp_ens = str(sample_override).strip().lower() if sample_override is not None else eq_ens

    if eq_ens not in ("nvt", "npt"):
        raise ValueError("CP2K backend supports md.ensemble in {'nvt','npt'} only")
    if samp_ens not in ("nvt", "npt"):
        raise ValueError("CP2K backend supports stage.sample_ensemble in {'nvt','npt'} only")

    pressure_bar = float(stage.pressure)
    if not math.isfinite(pressure_bar):
        raise ValueError("stage.pressure must be finite")

    energy_every = int(md_cfg.thermo_every)
    msd_every = int(stage.msd_every)
    # trajectory volume density
    # msd export trajectory
    # dump export lammps
    # dump export lammps
    # printing dump disabled
    dump_every = int(stage.dump_every) if stage.dump_every is not None else int(getattr(md_cfg, "dump_every", md_cfg.thermo_every))
    stride_reqs = [energy_every, msd_every]
    if bool(stage.write_dump):
        if stage.tail_dump_frames is not None and stage.tail_dump_stride is not None:
            # dump selection trajectory
            stride_reqs.append(int(stage.tail_dump_stride))
        else:
            stride_reqs.append(int(dump_every))
    traj_every = _gcd_many(stride_reqs)
    if traj_every < 1:
        traj_every = 1

    restart_prev: str | None = None

    def _read_cp2k_dcd_all(path: Path, *, ref_atoms, aligned: bool = True):
        """Cp2k dcd all."""

        with path.open("rb") as fd:
            frames = list(read_cp2k_dcd(fd, index=slice(None), ref_atoms=ref_atoms, aligned=bool(aligned)))
        return frames

    def _read_cp2k_dcd_last(path: Path, *, ref_atoms, aligned: bool = True):
        """Cp2k dcd last."""

        with path.open("rb") as fd:
            # cp2k dcd generator
            last = None
            for at in read_cp2k_dcd(fd, index=-1, ref_atoms=ref_atoms, aligned=bool(aligned)):
                last = at
            if last is None:
                raise RuntimeError(f"No frames read from CP2K DCD file: {path}")
            return last

    def _infer_cp2k_traj_steps(nsteps: int, stride: int, nframes: int) -> list[int]:
        """Cp2k traj steps."""

        nsteps = int(nsteps)
        stride = int(stride)
        nframes = int(nframes)
        if nsteps < 0:
            raise ValueError("nsteps must be >= 0")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if nframes < 1:
            raise ValueError("nframes must be >= 1")

        k, r = divmod(nsteps, stride)

        # r step stride
        if r == 0:
            if nframes == k + 1:
                return [i * stride for i in range(k + 1)]
            if nframes == k and k > 0:
                return [(i + 1) * stride for i in range(k)]

        # extra frame nsteps
        if r != 0:
            if nframes == k + 2:
                return [i * stride for i in range(k + 1)] + [nsteps]
            if nframes == k + 1:
                return [(i + 1) * stride for i in range(k)] + [nsteps]

        # fallback
        if nframes == 1:
            return [nsteps]
        grid = np.linspace(0.0, float(nsteps), num=nframes)
        out = [int(round(x)) for x in grid]
        # enforce monotonicity endpoints
        out[0] = min(out[0], 0)
        out[-1] = nsteps
        for i in range(1, len(out)):
            out[i] = max(out[i], out[i - 1])
        return out

    # equilibration start ens
    if stage.equil_steps > 0:
        equil_inp = stage_dir / f"{stage.name}_equil.inp"
        equil_traj = f"{stage.name}_equil.dcd"
        equil_ener = f"{stage.name}_equil.ener"
        equil_inp.write_text(
            render_cp2k_md_input(
                atoms=atoms,
                cfg=runner.cfg,
                md_cfg=md_cfg,
                basis_set_file_name=basis_file,
                potential_file_name=pot_file,
                ensemble=eq_ens,
                temperature_K=float(stage.temperature_start),
                steps=int(stage.equil_steps),
                timestep_fs=dt_fs,
                tdamp_fs=tdamp_fs,
                pdamp_fs=pdamp_fs,
                pressure_bar=pressure_bar,
                project=f"{stage.name}_equil",
                energy_every=energy_every,
                traj_every=traj_every,
                traj_file=equil_traj,
                ener_file=equil_ener,
            )
        )
        runner.run(equil_inp, stage_dir, output_name=f"{stage.name}_equil.out")
        atoms = _read_cp2k_dcd_last(stage_dir / equil_traj, ref_atoms=atoms, aligned=True)
        atoms.pbc = True

        equil_restart = stage_dir / f"{stage.name}_equil-1.restart"
        if equil_restart.exists():
            restart_prev = equil_restart.name

    # sampling discretize ramp
    T0 = float(stage.temperature_start)
    T1 = float(stage.temperature_stop)
    if abs(T1 - T0) > 0.0:
        schedule = _build_cp2k_ramp_schedule(
            T_start=T0,
            T_stop=T1,
            total_steps=int(stage.run_steps),
            max_deltaT_K=float(runner.cfg.ramp_max_deltaT_K),
            max_segments=int(runner.cfg.ramp_max_segments),
        )
    else:
        schedule = [(T1, int(stage.run_steps))]

    ener_tables: list[tuple[int, object]] = []
    traj_frames: list[tuple[int, np.ndarray, np.ndarray]] = []
    step_offset = 0
    symbols = list(atoms.get_chemical_symbols())

    for iseg, (Tseg, nsteps) in enumerate(schedule):
        if int(nsteps) <= 0:
            continue
        seg_tag = f"seg{iseg:03d}"
        inp_path = stage_dir / f"{stage.name}_{seg_tag}.inp"
        traj_file = f"{stage.name}_{seg_tag}.dcd"
        ener_file = f"{stage.name}_{seg_tag}.ener"

        inp_path.write_text(
            render_cp2k_md_input(
                atoms=atoms,
                cfg=runner.cfg,
                md_cfg=md_cfg,
                basis_set_file_name=basis_file,
                potential_file_name=pot_file,
                ensemble=samp_ens,
                temperature_K=float(Tseg),
                steps=int(nsteps),
                timestep_fs=dt_fs,
                tdamp_fs=tdamp_fs,
                pdamp_fs=pdamp_fs,
                pressure_bar=pressure_bar,
                project=f"{stage.name}_{seg_tag}",
                energy_every=energy_every,
                traj_every=traj_every,
                traj_file=traj_file,
                ener_file=ener_file,
                restart_file=restart_prev,
            )
        )
        runner.run(inp_path, stage_dir, output_name=f"{stage.name}_{seg_tag}.out")

        seg_restart = stage_dir / f"{stage.name}_{seg_tag}-1.restart"
        if seg_restart.exists():
            restart_prev = seg_restart.name
        else:
            if iseg < len(schedule) - 1:
                raise RuntimeError(
                    f"CP2K did not produce restart file {seg_restart.name!r}; required to continue a segmented temperature ramp"
                )
            restart_prev = None

        etab = parse_cp2k_ener(stage_dir / ener_file)
        ener_tables.append((step_offset, etab))

        dcd_path = stage_dir / traj_file
        frames = _read_cp2k_dcd_all(dcd_path, ref_atoms=atoms, aligned=True)
        if len(frames) == 0:
            raise RuntimeError(f"No frames read from CP2K trajectory: {dcd_path.name}")

        local_steps = _infer_cp2k_traj_steps(int(nsteps), int(traj_every), int(len(frames)))
        if len(local_steps) != len(frames):
            raise RuntimeError(
                "Internal error: inferred CP2K trajectory steps do not match frame count "
                f"(nsteps={nsteps}, stride={traj_every}, nframes={len(frames)}, inferred={len(local_steps)})"
            )

        # duplicate boundary segmented
        start_i = 0
        if traj_frames and local_steps and int(local_steps[0]) == 0:
            if int(traj_frames[-1][0]) == int(step_offset):
                start_i = 1

        for iframe in range(start_i, len(frames)):
            at = frames[iframe]
            gstep = int(step_offset + int(local_steps[iframe]))
            traj_frames.append(
                (gstep, np.asarray(at.get_positions(), dtype=float), np.asarray(at.get_cell(), dtype=float))
            )

        atoms = frames[-1]
        atoms.pbc = True
        step_offset += int(nsteps)

    if len(traj_frames) == 0:
        raise RuntimeError("No CP2K trajectory frames were produced")

    # cell lammps writer
    _orthorhombic_box_from_cell(np.asarray(atoms.get_cell(), dtype=float))

    # output structure cell
    try:
        frac_last = atoms.get_scaled_positions(wrap=True)
        atoms.set_scaled_positions(frac_last)
    except _ARTIFACT_IO_EXCEPTIONS:
        pass

    # import ase lammps
    from ..structuregen import write_lammps_data

    write_lammps_data(
        output_local,
        atoms,
        specorder=list(type_to_species),
        atom_style=str(md_cfg.atom_style),
    )

    # requested outside materialize
    try:
        _copy_requested_output_if_needed(
            requested_output=stage.output_data,
            output_local=output_local,
            stage_dir=stage_dir,
        )
    except OutputLocalizationError as exc:
        _log_best_effort_failure(exc)

    # lammps thermo table
    # volume density trajectory
    traj_frames.sort(key=lambda x: x[0])
    steps_all = np.asarray([t[0] for t in traj_frames], dtype=int)
    pos_all = np.asarray([t[1] for t in traj_frames], dtype=float)
    cells_all = np.asarray([t[2] for t in traj_frames], dtype=float)

    # guard
    for c in cells_all:
        _orthorhombic_box_from_cell(c)

    mass_amu = float(np.sum(atoms.get_masses()))
    if not (math.isfinite(mass_amu) and mass_amu > 0.0):
        raise ValueError("Invalid total mass from atoms")

    # frame volumes densities
    vols = np.abs(np.linalg.det(cells_all)).astype(float)
    if np.any(~np.isfinite(vols)) or np.any(vols <= 0.0):
        raise ValueError("Non-finite or non-positive cell volume encountered in CP2K trajectory")
    rhos = (1.66053906660 * mass_amu / vols).astype(float)

    # interpolate volume density
    step_to_idx = {int(s): i for i, s in enumerate(steps_all)}

    thermo_rows: list[tuple[int, float, float, float, float, float]] = []
    press = float("nan")

    for offset, etab in ener_tables:
        s = etab.step.astype(int) + int(offset)
        for si, Ti, pe in zip(s, etab.temperature_K, etab.potential_eV):
            s_int = int(si)
            if s_int in step_to_idx:
                j = step_to_idx[s_int]
            else:
                # fallback
                j = int(np.searchsorted(steps_all, s_int, side="right") - 1)
                j = max(0, min(j, len(steps_all) - 1))
            thermo_rows.append((
                s_int,
                float(Ti),
                press,
                float(pe),
                float(vols[j]),
                float(rhos[j]),
            ))

    thermo_rows.sort(key=lambda x: x[0])

    # lammps thermo table
    with log_path.open("w") as f:
        f.write("Step Temp Press PotEng Volume Density\n")
        for row in thermo_rows:
            f.write(
                f"{row[0]} {row[1]:.8f} {row[2]:.8f} {row[3]:.10f} {row[4]:.10f} {row[5]:.10f}\n"
            )

    # engine neutral thermo
    try:
        cols = ["Step", "Temp", "Press", "PotEng", "Volume", "Density"]
        data = np.asarray([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in thermo_rows], dtype=float)
        _materialize_thermo_csv_from_table(
            table=ThermoTable(columns=cols, data=data),
            thermo_csv=thermo_csv,
        )
    except ThermoArtifactError as exc:
        _log_best_effort_failure(exc)
        thermo_csv = _write_placeholder_artifact(thermo_csv, label="thermo.csv")

    # trajectory derived msd
    cell_ref = np.asarray(cells_all[0], dtype=float)
    _orthorhombic_box_from_cell(cell_ref)

    if samp_ens == "nvt":
        msd_all = compute_msd(pos_all, cell_ref, unwrap=True)
    else:
        # cell fractional instantaneous
        # reference cell box
        inv_cells = np.linalg.inv(cells_all)
        frac = np.einsum("tnj,tjk->tnk", pos_all, inv_cells)
        frac = frac - np.floor(frac)
        ufrac = unwrap_positions_fractional(frac)
        pos_u = np.einsum("tnj,jk->tnk", ufrac, cell_ref)
        dr = pos_u - pos_u[0:1]
        msd_all = np.mean(np.sum(dr * dr, axis=-1), axis=1)

    idx = [i for i, s in enumerate(steps_all) if int(s) % int(msd_every) == 0]
    if len(idx) < 3:
        idx = list(range(len(steps_all)))
    with msd_path.open("w") as f:
        for i in idx:
            f.write(f"{int(steps_all[i])} {float(msd_all[i]):.10f}\n")

    # engine neutral msd
    try:
        _materialize_msd_csv_from_file(msd_path=msd_path, msd_csv=msd_csv)
    except MSDArtifactError as exc:
        _log_best_effort_failure(exc)
        msd_csv = _write_placeholder_artifact(msd_csv, label="msd.csv")

    # lammps dump extxyz
    # trajectory thermo msd
    # analysis exported dump
    selected_out = _select_cp2k_export_indices(
        steps_all=steps_all,
        run_steps=int(stage.run_steps),
        tail_dump_frames=stage.tail_dump_frames,
        tail_dump_stride=stage.tail_dump_stride,
        dump_every=int(dump_every),
        traj_every=int(traj_every),
        write_dump=bool(stage.write_dump),
    )

    # lammps dump trajectory
    if dump_path is not None and selected_out:
        sel_steps = [int(steps_all[i]) for i in selected_out]
        sel_cells = cells_all[selected_out]
        inv_sel = np.linalg.inv(sel_cells)
        sel_frac = np.einsum("tnj,tjk->tnk", pos_all[selected_out], inv_sel)
        sel_frac = sel_frac - np.floor(sel_frac)
        _write_lammpstrj(
            dump_path,
            steps=sel_steps,
            frac_positions=sel_frac,
            symbols=symbols,
            type_to_species=type_to_species,
            cells=sel_cells,
        )

    # trajectory structure extxyz
    traj_extxyz, final_extxyz = _materialize_cp2k_engine_neutral_outputs(
        stage_dir=stage_dir,
        steps_all=steps_all,
        pos_all=pos_all,
        cells_all=cells_all,
        symbols=symbols,
        type_to_species=type_to_species,
        selected_out=selected_out,
        write_dump=bool(stage.write_dump),
    )

    return StageArtifacts(
        stage_dir=stage_dir,
        input_local=input_local,
        output_local=output_local,
        log_path=log_path,
        msd_path=msd_path,
        dump_path=dump_path,
        neighbor_skin=float("nan"),
        neighbor_skin_retries=0,
        thermo_csv=thermo_csv,
        msd_csv=msd_csv,
        traj_extxyz=traj_extxyz,
        final_extxyz=final_extxyz,
    )


def stage_outcome_from_artifacts(art: StageArtifacts, *, md_cfg, stage: StageSpec) -> StageOutcome:
    """Stage outcome from."""

    # prefer neutral thermo
    try:
        from ..io.thermo import parse_thermo_csv

        thermo = parse_thermo_csv(art.thermo_csv).as_dict()
    except _ARTIFACT_IO_EXCEPTIONS:
        thermo = parse_last_thermo_table(art.log_path).as_dict()

    # msd always return
    D = float("nan")
    D_stderr = float("nan")
    msd_rms_last = float("nan")
    try:
        msd = parse_msd_file(art.msd_path)
        diff = estimate_diffusion_from_msd(msd.step, msd.msd, timestep=md_cfg.timestep, fit_start_fraction=0.5)
        D = float(diff.D)
        D_stderr = float(diff.D_stderr)
        msd_rms_last = float(math.sqrt(max(0.0, float(msd.msd[-1]))))
    except _ARTIFACT_IO_EXCEPTIONS:
        pass

    dens = window_mean_stderr(thermo.get("Density"), start_fraction=0.5) if "Density" in thermo else window_mean_stderr([], start_fraction=0.5)
    pe = window_mean_stderr(thermo.get("PotEng"), start_fraction=0.5) if "PotEng" in thermo else window_mean_stderr([], start_fraction=0.5)

    vol_last = float(thermo["Volume"][-1]) if "Volume" in thermo and len(thermo["Volume"]) > 0 else float("nan")
    n_atoms = count_atoms_in_datafile(art.output_local)

    dump_rel = None
    if art.dump_path is not None:
        try:
            dump_rel = str(art.dump_path.relative_to(art.stage_dir))
        except (OSError, RuntimeError, ValueError):
            dump_rel = str(art.dump_path)

    return StageOutcome(
        name=str(stage.name),
        temperature_start=float(stage.temperature_start),
        temperature_stop=float(stage.temperature_stop),
        pressure=float(stage.pressure),
        equil_steps=int(stage.equil_steps),
        run_steps=int(stage.run_steps),
        seed=int(stage.seed),
        n_atoms=int(n_atoms),
        vol_last=vol_last,
        density_mean=float(dens.mean),
        density_stderr=float(dens.stderr),
        pe_mean=float(pe.mean),
        pe_stderr=float(pe.stderr),
        D=float(D),
        D_stderr=float(D_stderr),
        msd_rms_last=float(msd_rms_last),
        output_data=str(art.output_local.relative_to(art.stage_dir)),
        dump=dump_rel,
        neighbor_skin=float(art.neighbor_skin),
        neighbor_skin_retries=int(art.neighbor_skin_retries),
    )
