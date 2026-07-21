from __future__ import annotations

"""Stage execution utilities shared by multiple workflows."""

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import stat as stat_module
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from ..lammps_units import (
    density_to_g_cm3_factor,
    diffusivity_to_angstrom2_per_ps_factor,
    energy_to_ev_factor,
    length_to_angstrom_factor,
    msd_to_angstrom2_factor,
    pressure_to_gpa_factor,
    time_to_ps_factor,
    volume_to_angstrom3_factor,
)

from ..lammps_input import (
    StageSpec,
    render_continuous_stages,
    render_stage,
    validate_stage_name,
)
from ..parse import ThermoTable, parse_last_thermo_table, parse_msd_file
from ..io.thermo import parse_msd_csv, parse_thermo_csv, write_thermo_csv, write_msd_csv
from ..io.stage_manifest import (
    STAGE_ARTIFACT_MANIFEST_NAME,
    load_stage_artifact_manifest,
    verify_manifest_artifact,
    write_stage_artifact_manifest,
)
from ..io.extxyz import write_extxyz_frames, write_extxyz_iter, write_extxyz_single
from ..io.lammps_data_minimal import write_dumpframe_lammps_data
from ..io.ase_compat import ase_read_lammps_data
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
from ..potential import prepare_potential_files, _parse_tabulated_core_spec
from ..config import validated_lammps_localized_filename
from ..runner import Cp2kRunner, LammpsRunner
from ..utils import ensure_dir, stable_file_identity
from ..analysis.provenance import write_json_strict
from ..cp2k_driver import (
    count_cp2k_scf_failures,
    cp2k_scf_continuation_policy,
    compute_msd,
    density_g_cm3_from_atoms,
    map_cp2k_pressures_to_energy_steps,
    parse_cp2k_ener,
    parse_cp2k_md_step_pressures,
    _read_cp2k_dcd_last_validated,
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


def _infer_cp2k_traj_steps(nsteps: int, stride: int, nframes: int) -> list[int]:
    """Map CP2K trajectory frames only for explicit stride/ADD_LAST patterns.

    CP2K may include or omit the initial frame; ``ADD_LAST NUMERIC`` adds the
    final step when it is off-stride.  Any other count is ambiguous and must
    not be assigned an invented time grid.
    """

    values = {"nsteps": nsteps, "stride": stride, "nframes": nframes}
    parsed: dict[str, int] = {}
    for name, raw in values.items():
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")
        try:
            numeric = float(raw)
            integer = int(numeric)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not math.isfinite(numeric) or numeric != float(integer):
            raise ValueError(f"{name} must be an integer")
        parsed[name] = integer
    nsteps_i = parsed["nsteps"]
    stride_i = parsed["stride"]
    nframes_i = parsed["nframes"]
    if nsteps_i < 0:
        raise ValueError("nsteps must be >= 0")
    if stride_i < 1:
        raise ValueError("stride must be >= 1")
    if nframes_i < 1:
        raise ValueError("nframes must be >= 1")

    k, remainder = divmod(nsteps_i, stride_i)
    valid: dict[int, list[list[int]]] = {}

    def _add(candidate: list[int]) -> None:
        valid.setdefault(len(candidate), [])
        if candidate not in valid[len(candidate)]:
            valid[len(candidate)].append(candidate)

    if remainder == 0:
        _add([i * stride_i for i in range(k + 1)])  # initial frame included
        if k > 0:
            _add([(i + 1) * stride_i for i in range(k)])  # initial omitted
    else:
        _add([i * stride_i for i in range(k + 1)] + [nsteps_i])
        _add([(i + 1) * stride_i for i in range(k)] + [nsteps_i])

    candidates = valid.get(nframes_i, [])
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Unexpected CP2K trajectory frame count; cannot infer steps without "
        "inventing timestamps "
        f"(nsteps={nsteps_i}, stride={stride_i}, nframes={nframes_i}, "
        f"valid_counts={sorted(valid)})"
    )


def _merge_cp2k_segment_thermo_rows(
    rows: Sequence[tuple[int, float, float, float, float, float]],
    *,
    segment_ids: Optional[Sequence[int]] = None,
    segment_scf_failures: Optional[Mapping[int, int]] = None,
    segment_labels: Optional[Mapping[int, str]] = None,
    boundary_diagnostics: Optional[list[dict[str, object]]] = None,
) -> list[tuple[int, float, float, float, float, float]]:
    """Merge repeated CP2K segment-boundary observations without losing evidence.

    A segment's local step zero and its predecessor's final step describe the
    same *nuclear* state but are distinct electronic observations: CP2K runs a
    fresh SCF calculation after loading the restart.  In particular, an
    explicitly continued unconverged SCF calculation is not required to
    reproduce the preceding potential energy or electronic pressure.  Treating
    those values as exact duplicates caused otherwise completed segmented MD
    runs to be discarded.

    The canonical series retains the preceding segment's propagated boundary
    row, matching the retained trajectory frame.  Temperature, volume, and
    density are continuity invariants and must agree within output roundoff.
    Potential energy and finite pressure are independent SCF observations and
    are not equality invariants, even when both calculations converged.  Both
    observations, their deltas, and adjacent-segment SCF diagnostics are
    preserved in ``boundary_diagnostics``.  A missing restart-side pressure is
    expected at local step zero; the finite side is retained.

    ``segment_ids`` is required by production callers so duplicate steps can
    only cross segment boundaries.  Omitting it retains the strict standalone
    helper behaviour used by legacy callers and tests.
    """

    if segment_ids is not None and len(segment_ids) != len(rows):
        raise ValueError("segment_ids must have exactly one entry per CP2K thermo row")
    failures: dict[int, int] = {}
    if segment_scf_failures is not None:
        for raw_segment, raw_count in segment_scf_failures.items():
            if isinstance(raw_segment, bool) or isinstance(raw_count, bool):
                raise ValueError("CP2K segment identifiers and SCF failure counts must be integers")
            segment = int(raw_segment)
            count = int(raw_count)
            if float(raw_segment) != float(segment) or segment < 0:
                raise ValueError("CP2K segment identifiers must be nonnegative integers")
            if float(raw_count) != float(count) or count < 0:
                raise ValueError("CP2K SCF failure counts must be nonnegative integers")
            failures[segment] = count

    labels = {int(key): str(value) for key, value in (segment_labels or {}).items()}
    merged: list[tuple[int, float, float, float, float, float]] = []
    merged_segments: list[Optional[int]] = []
    merged_boundary_steps: set[int] = set()
    previous_input_segment: Optional[int] = None
    field_names = (
        "temperature_K",
        "pressure_bar",
        "potential_eV",
        "volume_A3",
        "density_g_cm3",
    )

    def _json_number(value: float) -> Optional[float]:
        return float(value) if math.isfinite(value) else None

    for row_index, raw in enumerate(rows):
        row = tuple(raw)
        if len(row) != 6:
            raise ValueError("CP2K thermo rows must contain six fields")
        step = int(row[0])
        if float(row[0]) != float(step) or step < 0:
            raise ValueError("CP2K global thermo steps must be nonnegative integers")
        normalized = (
            step,
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        )
        if not all(math.isfinite(normalized[index]) for index in (1, 3, 4, 5)):
            raise ValueError(
                f"CP2K thermo row at global step {step} has non-finite required observables"
            )
        if normalized[1] < 0.0 or normalized[4] <= 0.0 or normalized[5] <= 0.0:
            raise ValueError(
                f"CP2K thermo row at global step {step} has invalid temperature/volume/density"
            )
        if math.isinf(normalized[2]):
            raise ValueError(f"CP2K thermo pressure at global step {step} cannot be infinite")

        current_segment: Optional[int] = None
        if segment_ids is not None:
            raw_segment = segment_ids[row_index]
            if isinstance(raw_segment, bool):
                raise ValueError("CP2K segment identifiers must be nonnegative integers")
            current_segment = int(raw_segment)
            if float(raw_segment) != float(current_segment) or current_segment < 0:
                raise ValueError("CP2K segment identifiers must be nonnegative integers")
            if previous_input_segment is not None and current_segment < previous_input_segment:
                raise ValueError("CP2K thermo segment identifiers must be nondecreasing")
            previous_input_segment = current_segment

        if merged and step < merged[-1][0]:
            raise ValueError(
                "CP2K segmented thermo steps are nonmonotone: "
                f"{step} follows {merged[-1][0]}"
            )
        if merged and step == merged[-1][0]:
            if step in merged_boundary_steps:
                raise ValueError(
                    "CP2K segmented thermo contains more than one restart-side "
                    f"observation at global step {step}"
                )
            previous = merged[-1]
            previous_segment = merged_segments[-1]
            if segment_ids is not None and (
                previous_segment is None
                or current_segment is None
                or current_segment <= previous_segment
            ):
                raise ValueError(
                    "Duplicate CP2K thermo step occurred within one segment rather than "
                    f"at a restart boundary (global step {step})"
                )
            if (
                segment_ids is not None
                and previous_segment is not None
                and current_segment is not None
                and current_segment != previous_segment + 1
            ):
                raise ValueError(
                    "Duplicate CP2K thermo step crossed non-adjacent segments at "
                    f"global step {step}: {previous_segment} -> {current_segment}"
                )

            electronic_reobservation = bool(
                segment_ids is not None
                and previous_segment is not None
                and current_segment is not None
                and current_segment > previous_segment
            )
            inconsistent: list[int] = []
            # The previous row is the propagated observation and corresponds
            # to the retained trajectory frame.  Only fill an optional missing
            # pressure from the restart-side observation.
            resolved = list(previous)
            field_audit: dict[str, object] = {}
            for index, (old, new) in enumerate(zip(previous[1:], normalized[1:]), start=1):
                old_finite = math.isfinite(old)
                new_finite = math.isfinite(new)
                consistent = False
                if not old_finite or not new_finite:
                    if math.isnan(old) and math.isnan(new):
                        consistent = True
                    elif math.isnan(new) and old_finite:
                        consistent = index == 2
                    elif math.isnan(old) and new_finite:
                        if index == 2:
                            resolved[index] = new
                            consistent = True
                    else:
                        inconsistent.append(index)
                else:
                    consistent = math.isclose(old, new, rel_tol=1.0e-10, abs_tol=1.0e-8)
                    if not consistent:
                        # A new CP2K segment performs a new SCF calculation at
                        # the same restarted nuclear state.  Pressure and
                        # potential energy are therefore distinct electronic
                        # observations even when both SCFs converged; EPS_SCF,
                        # extrapolation and diagonalisation history make a
                        # fixed output-roundoff equality test unphysical.
                        # Retain the preceding propagated observation (which
                        # corresponds to the retained DCD frame), audit the
                        # delta, and keep nuclear T/V/density strict.
                        if index not in (2, 3) or not electronic_reobservation:
                            inconsistent.append(index)

                field_audit[field_names[index - 1]] = {
                    "preceding": _json_number(old),
                    "restart": _json_number(new),
                    "absolute_delta": (
                        abs(float(new) - float(old)) if old_finite and new_finite else None
                    ),
                    "within_output_roundoff": bool(consistent),
                    "selected": _json_number(float(resolved[index])),
                }
            if inconsistent:
                raise ValueError(
                    "Inconsistent duplicate CP2K segment-boundary thermo row at "
                    f"global step {step}; differing field indices {inconsistent}; "
                    "electronic fields may differ only across an explicitly identified "
                    "restart boundary, while nuclear fields must remain continuous"
                )
            merged[-1] = tuple(resolved)  # type: ignore[assignment]
            merged_boundary_steps.add(step)
            if boundary_diagnostics is not None:
                boundary_diagnostics.append(
                    {
                        "global_step": step,
                        "preceding_segment": previous_segment,
                        "preceding_label": labels.get(previous_segment, str(previous_segment)),
                        "restart_segment": current_segment,
                        "restart_label": labels.get(current_segment, str(current_segment)),
                        "preceding_unconverged_scf_cycles": failures.get(previous_segment, 0),
                        "restart_unconverged_scf_cycles": failures.get(current_segment, 0),
                        "electronic_mismatch_allowed": electronic_reobservation,
                        "electronic_mismatch_basis": (
                            "independent_scf_reobservation_at_restart_boundary"
                        ),
                        "canonical_observation": "preceding_propagated_row",
                        "fields": field_audit,
                    }
                )
            continue
        merged.append(normalized)
        merged_segments.append(current_segment)
    if not merged:
        raise ValueError("No CP2K thermo rows were produced")
    return merged


def _audit_cp2k_segment_trajectory_boundary(
    *,
    global_step: int,
    preceding_positions: np.ndarray,
    preceding_cell: np.ndarray,
    restart_positions: np.ndarray,
    restart_cell: np.ndarray,
) -> dict[str, object]:
    """Verify nuclear continuity across a CP2K DCD restart boundary.

    CP2K's DCD trajectory is commonly single precision.  The comparison is
    therefore exact up to a scale-aware bound of 64 binary32 ulps, not an
    arbitrary MD displacement tolerance.  Positions are compared modulo the
    periodic lattice so harmless wrapping at a segment boundary is accepted;
    the cell itself must remain continuous independently.
    """

    step = int(global_step)
    if isinstance(global_step, bool) or float(global_step) != float(step) or step < 0:
        raise ValueError("CP2K trajectory boundary step must be a nonnegative integer")

    p0 = np.asarray(preceding_positions, dtype=float)
    p1 = np.asarray(restart_positions, dtype=float)
    c0 = np.asarray(preceding_cell, dtype=float)
    c1 = np.asarray(restart_cell, dtype=float)
    if p0.ndim != 2 or p0.shape[1:] != (3,) or p1.shape != p0.shape:
        raise RuntimeError(
            f"CP2K trajectory atom layout changed at global step {step}: "
            f"{p0.shape} -> {p1.shape}"
        )
    if c0.shape != (3, 3) or c1.shape != (3, 3):
        raise RuntimeError(f"CP2K trajectory cell has invalid shape at global step {step}")
    if not all(np.all(np.isfinite(value)) for value in (p0, p1, c0, c1)):
        raise RuntimeError(f"CP2K trajectory boundary contains non-finite geometry at global step {step}")
    det0 = float(np.linalg.det(c0))
    det1 = float(np.linalg.det(c1))
    if not (math.isfinite(det0) and math.isfinite(det1) and abs(det0) > 0.0 and abs(det1) > 0.0):
        raise RuntimeError(f"CP2K trajectory boundary has a singular cell at global step {step}")

    scale = max(
        1.0,
        float(np.max(np.abs(c0))),
        float(np.max(np.abs(c1))),
        float(np.max(np.abs(p0))) if p0.size else 0.0,
        float(np.max(np.abs(p1))) if p1.size else 0.0,
    )
    tolerance_A = max(1.0e-6, 64.0 * float(np.finfo(np.float32).eps) * scale)
    max_cell_delta_A = float(np.max(np.abs(c1 - c0)))

    # ASE uses lattice vectors as rows, hence Cartesian = fractional @ cell.
    frac0 = p0 @ np.linalg.inv(c0)
    frac1 = p1 @ np.linalg.inv(c1)
    delta_fractional = frac1 - frac0
    delta_fractional -= np.rint(delta_fractional)
    comparison_cell = 0.5 * (c0 + c1)
    delta_cartesian = delta_fractional @ comparison_cell
    max_position_delta_A = (
        float(np.max(np.linalg.norm(delta_cartesian, axis=1))) if p0.shape[0] else 0.0
    )

    if max_cell_delta_A > tolerance_A or max_position_delta_A > tolerance_A:
        raise RuntimeError(
            "CP2K restart trajectory is discontinuous at global step "
            f"{step}: max_position_delta_A={max_position_delta_A:.9g}, "
            f"max_cell_delta_A={max_cell_delta_A:.9g}, "
            f"binary32_roundoff_bound_A={tolerance_A:.9g}"
        )
    return {
        "global_step": step,
        "atom_count": int(p0.shape[0]),
        "max_position_delta_A": max_position_delta_A,
        "max_cell_delta_A": max_cell_delta_A,
        "binary32_roundoff_bound_A": tolerance_A,
        "periodic_wrapping_normalized": True,
        "status": "continuous",
    }


def _validate_cp2k_npt_pressure_rows(
    rows: Sequence[tuple[int, float, float, float, float, float]],
) -> None:
    """Require pressure evidence for every propagated NPT thermo state.

    CP2K may omit instantaneous pressure for the initial global step zero,
    which precedes propagation.  A local step-zero row from a later ramp
    segment is not a new initial state and therefore receives no exemption.
    """

    for index, row in enumerate(rows):
        step = int(row[0])
        pressure = float(row[2])
        if math.isfinite(pressure):
            continue
        if index == 0 and step == 0 and math.isnan(pressure):
            continue
        raise RuntimeError(
            "CP2K NPT thermo output lacks a finite instantaneous pressure at "
            f"global step {step}; only the initial global step zero may omit pressure"
        )


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
    restart_path: Optional[Path] = None
    wfn_restart_path: Optional[Path] = None
    # None means the engine-neutral files were already canonical on creation
    # (CP2K).  A LAMMPS style records how raw log/MSD fallbacks must convert.
    lammps_units_style: Optional[str] = None
    engine: str = "lammps"
    manifest_path: Optional[Path] = None
    cp2k_version: Optional[str] = None
    cp2k_scf_policy: Optional[str] = None
    cp2k_unconverged_scf_cycles: int = 0
    cp2k_scf_diagnostics_path: Optional[Path] = None


@dataclass(frozen=True)
class _Cp2kRestartState:
    restart_file: Path
    ensemble: str
    wfn_file: Optional[Path] = None


def _strict_cp2k_restart_bundle_identity(
    path: Path,
    *,
    bundle_dir: Path,
    expected_name: str,
) -> dict[str, object]:
    """Hash one exact, unaliased restart-bundle direct child."""

    directory = Path(bundle_dir)
    candidate = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(
            f"CP2K restart bundle directory must be a real directory: {directory}"
        )
    canonical_dir = directory.resolve(strict=True)
    if (
        candidate.name != str(expected_name)
        or candidate.parent.resolve(strict=False) != canonical_dir
    ):
        raise RuntimeError(
            "CP2K restart bundle artifact must use the exact direct-child name "
            f"{expected_name!r}: {candidate}"
        )
    if candidate.is_symlink():
        raise RuntimeError(
            f"CP2K restart bundle artifact must not be a symbolic link: {candidate}"
        )
    try:
        before = candidate.stat()
    except OSError as exc:
        raise RuntimeError(
            f"CP2K restart bundle artifact is missing: {candidate}"
        ) from exc
    if not stat_module.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise RuntimeError(
            "CP2K restart bundle artifact must be a unique regular file: "
            f"{candidate}"
        )
    identity = stable_file_identity(candidate, reject_final_symlink=True)
    after = candidate.stat()
    if (
        int(after.st_nlink) != 1
        or int(after.st_dev) != int(identity["device"])
        or int(after.st_ino) != int(identity["inode"])
    ):
        raise RuntimeError(
            f"CP2K restart bundle artifact changed link identity: {candidate}"
        )
    return identity


def _atomic_publish_cp2k_restart_artifact(
    source: Path,
    destination: Path,
    *,
    bundle_dir: Path,
) -> dict[str, object]:
    """Copy verified bytes to a reserved inode and atomically publish them."""

    source_path = Path(source)
    directory = Path(bundle_dir)
    source_identity = _strict_cp2k_restart_bundle_identity(
        source_path,
        bundle_dir=directory,
        expected_name=source_path.name,
    )
    fd, raw_tmp = tempfile.mkstemp(
        dir=str(directory),
        prefix=f".{Path(destination).name}.",
        suffix=".tmp",
    )
    tmp_path: Optional[Path] = Path(raw_tmp)
    try:
        with source_path.open("rb") as source_handle, os.fdopen(
            fd, "wb", closefd=True
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        copied_identity = stable_file_identity(
            tmp_path,
            reject_final_symlink=True,
        )
        source_after = _strict_cp2k_restart_bundle_identity(
            source_path,
            bundle_dir=directory,
            expected_name=source_path.name,
        )
        stable_fields = ("device", "inode", "size_bytes", "sha256")
        if any(source_after[field] != source_identity[field] for field in stable_fields):
            raise RuntimeError(
                f"CP2K restart source changed while being published: {source_path}"
            )
        if any(
            copied_identity[field] != source_identity[field]
            for field in ("size_bytes", "sha256")
        ):
            raise RuntimeError(
                f"CP2K restart copy does not match its source: {source_path}"
            )
        os.replace(tmp_path, destination)
        tmp_path = None
        published = _strict_cp2k_restart_bundle_identity(
            Path(destination),
            bundle_dir=directory,
            expected_name=Path(destination).name,
        )
        if any(
            published[field] != source_identity[field]
            for field in ("size_bytes", "sha256")
        ):
            raise RuntimeError(
                f"Published CP2K restart differs from its source: {destination}"
            )
        return published
    finally:
        if tmp_path is not None and (tmp_path.exists() or tmp_path.is_symlink()):
            tmp_path.unlink()


def _load_cp2k_restart_state_for_stage(stage: StageSpec) -> Optional[_Cp2kRestartState]:
    """Resolve the prior CP2K state required by velocity_mode='preserve'.

    Directory invariant (must hold if CP2K custom schedules are ever enabled):
    ``_publish_cp2k_restart_state`` writes ``cp2k.restart`` + ``cp2k.restart.json``
    into ``stage_dir`` next to the stage's produced coordinates file
    (``output_local = stage_dir / output_data.name``). This loader reads the
    bundle from the directory holding *this* stage's ``input_data`` — which the
    orchestrator must set to the prior stage's produced coordinates file, so the
    bundle travels with the coordinates. A mismatch fails loud below (missing
    metadata -> RuntimeError), never a silent coordinate-only continuation.
    """

    if str(getattr(stage, "velocity_mode", "create")).strip().lower() != "preserve":
        return None
    # The coordinate file and restart bundle form one state.  Resolve the
    # parent, never the final input component: resolving a final symlink first
    # would pivot bundle discovery into the symlink target's directory.
    input_path = Path(stage.input_data).expanduser()
    bundle_dir = input_path.parent.resolve(strict=True)
    coordinate_identity = _strict_cp2k_restart_bundle_identity(
        input_path,
        bundle_dir=bundle_dir,
        expected_name=input_path.name,
    )
    metadata_path = bundle_dir / "cp2k.restart.json"
    if not metadata_path.exists() and not metadata_path.is_symlink():
        raise RuntimeError(
            "CP2K stage requested velocity_mode='preserve', but the previous "
            f"stage has no restart metadata: {metadata_path}. Coordinate-only "
            "continuation is not a valid CP2K MD restart."
        )
    try:
        metadata_identity = _strict_cp2k_restart_bundle_identity(
            metadata_path,
            bundle_dir=bundle_dir,
            expected_name="cp2k.restart.json",
        )
        metadata_bytes = metadata_path.read_bytes()
        metadata_identity_after = _strict_cp2k_restart_bundle_identity(
            metadata_path,
            bundle_dir=bundle_dir,
            expected_name="cp2k.restart.json",
        )
        if any(
            metadata_identity[field] != metadata_identity_after[field]
            for field in ("device", "inode", "size_bytes", "sha256")
        ) or hashlib.sha256(metadata_bytes).hexdigest() != str(
            metadata_identity["sha256"]
        ):
            raise RuntimeError(
                f"CP2K restart metadata changed while being read: {metadata_path}"
            )
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Malformed CP2K restart metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != "vitriflow.cp2k_restart.v2":
        raise RuntimeError(f"Unsupported CP2K restart metadata: {metadata_path}")
    coordinate_name = metadata.get("coordinates_file")
    if (
        not isinstance(coordinate_name, str)
        or coordinate_name in {".", ".."}
        or Path(coordinate_name).name != coordinate_name
        or coordinate_name != input_path.name
    ):
        raise RuntimeError(
            "CP2K restart metadata does not name the exact input coordinate "
            f"file {input_path.name!r}: {metadata_path}"
        )
    raw_coordinate_size = metadata.get("coordinates_size_bytes")
    if type(raw_coordinate_size) is not int:
        raise RuntimeError(
            f"CP2K restart metadata has invalid coordinate size: {metadata_path}"
        )
    expected_coordinate_size = raw_coordinate_size
    raw_coordinate_sha = metadata.get("coordinates_sha256")
    expected_coordinate_sha = (
        raw_coordinate_sha.strip().lower()
        if isinstance(raw_coordinate_sha, str)
        else ""
    )
    if (
        expected_coordinate_size <= 0
        or expected_coordinate_size != int(coordinate_identity["size_bytes"])
        or len(expected_coordinate_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_coordinate_sha)
        or expected_coordinate_sha != str(coordinate_identity["sha256"])
    ):
        raise RuntimeError(
            f"CP2K restart bundle does not match its input coordinates: {input_path}"
        )
    if metadata.get("restart_file") != "cp2k.restart":
        raise RuntimeError(
            f"CP2K restart metadata must name exact local file 'cp2k.restart': {metadata_path}"
        )
    restart_file = bundle_dir / "cp2k.restart"
    restart_identity = _strict_cp2k_restart_bundle_identity(
        restart_file,
        bundle_dir=bundle_dir,
        expected_name="cp2k.restart",
    )
    if int(restart_identity["size_bytes"]) <= 0:
        raise RuntimeError(f"CP2K restart artifact is missing or empty: {restart_file}")
    raw_restart_sha = metadata.get("restart_sha256")
    expected_sha = (
        raw_restart_sha.strip().lower()
        if isinstance(raw_restart_sha, str)
        else ""
    )
    if (
        len(expected_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha)
        or str(restart_identity["sha256"]) != expected_sha
    ):
        raise RuntimeError(f"CP2K restart artifact checksum mismatch: {restart_file}")
    raw_ensemble = metadata.get("ensemble")
    ensemble = raw_ensemble.strip().lower() if isinstance(raw_ensemble, str) else ""
    if ensemble not in {"nvt", "npt"}:
        raise RuntimeError(f"CP2K restart metadata has invalid ensemble: {metadata_path}")
    wfn_file: Optional[Path] = None
    wfn_name = metadata.get("wfn_restart_file")
    if wfn_name is not None:
        if wfn_name != "cp2k-RESTART.wfn":
            raise RuntimeError(
                "CP2K restart metadata must name exact local WFN file "
                f"'cp2k-RESTART.wfn': {metadata_path}"
            )
        candidate = bundle_dir / "cp2k-RESTART.wfn"
        raw_wfn_sha = metadata.get("wfn_restart_sha256")
        expected_wfn_sha = (
            raw_wfn_sha.strip().lower()
            if isinstance(raw_wfn_sha, str)
            else ""
        )
        wfn_identity = _strict_cp2k_restart_bundle_identity(
            candidate,
            bundle_dir=bundle_dir,
            expected_name="cp2k-RESTART.wfn",
        )
        if (
            int(wfn_identity["size_bytes"]) <= 0
            or len(expected_wfn_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_wfn_sha)
            or str(wfn_identity["sha256"]) != expected_wfn_sha
        ):
            raise RuntimeError(f"CP2K wavefunction restart artifact is inconsistent: {candidate}")
        wfn_file = candidate
    elif metadata.get("wfn_restart_sha256") is not None:
        raise RuntimeError(
            f"CP2K restart metadata has a WFN hash without a WFN file: {metadata_path}"
        )
    elif (bundle_dir / "cp2k-RESTART.wfn").exists() or (
        bundle_dir / "cp2k-RESTART.wfn"
    ).is_symlink():
        raise RuntimeError(
            "CP2K restart bundle contains an unbound wavefunction artifact: "
            f"{bundle_dir / 'cp2k-RESTART.wfn'}"
        )
    return _Cp2kRestartState(
        restart_file=restart_file.resolve(strict=True),
        ensemble=ensemble,
        wfn_file=(None if wfn_file is None else wfn_file.resolve(strict=True)),
    )


def _publish_cp2k_restart_state(
    stage_dir: Path,
    *,
    coordinate_source: Path,
    restart_source: Path,
    ensemble: str,
    wfn_source: Optional[Path] = None,
) -> _Cp2kRestartState:
    """Publish a checksum-locked restart bundle for the next CP2K stage."""

    directory = Path(stage_dir).resolve(strict=True)
    if Path(stage_dir).is_symlink() or not directory.is_dir():
        raise RuntimeError(
            f"CP2K restart bundle directory must be a real directory: {stage_dir}"
        )
    source = Path(restart_source)
    source_identity = _strict_cp2k_restart_bundle_identity(
        source,
        bundle_dir=directory,
        expected_name=source.name,
    )
    if int(source_identity["size_bytes"]) <= 0:
        raise RuntimeError(f"CP2K did not produce a usable restart file: {source}")
    coordinate_path = Path(coordinate_source)
    reserved_bundle_names = {
        "cp2k.restart",
        "cp2k-RESTART.wfn",
        "cp2k.restart.json",
    }
    if coordinate_path.name in reserved_bundle_names:
        raise RuntimeError(
            "CP2K restart coordinates collide with a reserved restart-bundle "
            f"name: {coordinate_path.name}"
        )
    coordinate_identity = _strict_cp2k_restart_bundle_identity(
        coordinate_path,
        bundle_dir=directory,
        expected_name=coordinate_path.name,
    )
    if int(coordinate_identity["size_bytes"]) <= 0:
        raise RuntimeError(
            f"CP2K did not produce usable restart coordinates: {coordinate_path}"
        )
    ensemble_norm = str(ensemble).strip().lower()
    if ensemble_norm not in {"nvt", "npt"}:
        raise ValueError("CP2K restart ensemble must be 'nvt' or 'npt'")
    wfn_source_path: Optional[Path] = None
    if wfn_source is not None:
        wfn_source_path = Path(wfn_source)
        wfn_source_identity = _strict_cp2k_restart_bundle_identity(
            wfn_source_path,
            bundle_dir=directory,
            expected_name=wfn_source_path.name,
        )
        if int(wfn_source_identity["size_bytes"]) <= 0:
            raise RuntimeError(
                f"CP2K produced an empty wavefunction restart: {wfn_source_path}"
            )

    # Validate every source member before replacing any published bundle
    # member.  The metadata remains the commit record, but this ordering also
    # prevents a known-invalid WFN from partially replacing an older bundle.
    metadata_path = directory / "cp2k.restart.json"
    if metadata_path.exists() or metadata_path.is_symlink():
        try:
            metadata_path.unlink()
        except OSError as exc:
            raise RuntimeError(
                "Cannot invalidate the previous CP2K restart-bundle commit "
                f"record: {metadata_path}"
            ) from exc
    restart_out = directory / "cp2k.restart"
    restart_identity = _atomic_publish_cp2k_restart_artifact(
        source,
        restart_out,
        bundle_dir=directory,
    )
    wfn_out: Optional[Path] = None
    wfn_identity: Optional[dict[str, object]] = None
    published_wfn = directory / "cp2k-RESTART.wfn"
    if wfn_source_path is not None:
        wfn_out = published_wfn
        wfn_identity = _atomic_publish_cp2k_restart_artifact(
            wfn_source_path,
            wfn_out,
            bundle_dir=directory,
        )
    else:
        if published_wfn.exists() or published_wfn.is_symlink():
            try:
                published_wfn.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot remove stale CP2K restart WFN artifact: {published_wfn}"
                ) from exc

    coordinate_after = _strict_cp2k_restart_bundle_identity(
        coordinate_path,
        bundle_dir=directory,
        expected_name=coordinate_path.name,
    )
    if any(
        coordinate_after[field] != coordinate_identity[field]
        for field in ("device", "inode", "size_bytes", "sha256")
    ):
        raise RuntimeError(
            "CP2K restart coordinates changed while the restart bundle was "
            f"being published: {coordinate_path}"
        )
    metadata = {
        "schema": "vitriflow.cp2k_restart.v2",
        "coordinates_file": coordinate_path.name,
        "coordinates_size_bytes": int(coordinate_identity["size_bytes"]),
        "coordinates_sha256": str(coordinate_identity["sha256"]),
        "restart_file": restart_out.name,
        "restart_sha256": str(restart_identity["sha256"]),
        "ensemble": ensemble_norm,
        "wfn_restart_file": (None if wfn_out is None else wfn_out.name),
        "wfn_restart_sha256": (
            None if wfn_identity is None else str(wfn_identity["sha256"])
        ),
    }
    write_json_strict(
        metadata_path,
        metadata,
        indent=2,
        sort_keys=True,
    )
    _strict_cp2k_restart_bundle_identity(
        metadata_path,
        bundle_dir=directory,
        expected_name="cp2k.restart.json",
    )
    return _Cp2kRestartState(
        restart_file=restart_out,
        ensemble=ensemble_norm,
        wfn_file=wfn_out,
    )


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

    # Additive diagnostics for the physical D >= 0 boundary constraint.  The
    # unconstrained value remains in canonical diffusion units for auditability.
    D_unconstrained: float = float("nan")
    D_boundary_constrained: bool = False

    neighbor_skin: float = float("nan")
    neighbor_skin_retries: int = 0

    gr_peak_r: float = float("nan")
    gr_peak_height: float = float("nan")
    gr_peak_fwhm: float = float("nan")

    rep_id: Optional[int] = None

    # CP2K continuation is intentional, but never silent.  These additive
    # fields label the executable/policy and count every explicit SCF warning.
    cp2k_version: Optional[str] = None
    cp2k_scf_policy: Optional[str] = None
    cp2k_unconverged_scf_cycles: int = 0


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

    @property
    def manifest_path(self) -> Path:
        return self.stage_dir / STAGE_ARTIFACT_MANIFEST_NAME

    def cleanup_paths(self, *, log_name: str, include_postprocessed: bool) -> list[Path]:
        # The manifest is the authority that marks neutral CSV values as
        # canonical.  Remove it before every attempt so a failed rerun cannot
        # lend stale provenance to artifacts from a prior invocation.
        out = [self.output_local, self.msd_path, self.final_dump_path, self.manifest_path]
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
        if path.exists() or path.is_symlink():
            path.unlink()
    except OSError as exc:
        LOG.debug(
            "Failed to remove partial artifact %s (%s: %s)",
            path,
            type(exc).__name__,
            exc,
        )


def _atomic_copy_lammps_stage_input(source: Path, destination: Path) -> Path:
    """Stage one input snapshot without following an existing output alias."""

    src = Path(source)
    dst = Path(destination)
    expected = stable_file_identity(src)
    try:
        payload = src.read_bytes()
    except OSError as exc:
        raise InputSnapshotError(f"Failed to read LAMMPS stage input {src}") from exc
    if (
        len(payload) != int(expected["size_bytes"])
        or hashlib.sha256(payload).hexdigest() != str(expected["sha256"])
    ):
        raise InputSnapshotError(
            f"LAMMPS stage input changed while it was being localized: {src}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{dst.name}.",
            suffix=".input.tmp",
            dir=dst.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, dst)
        temporary = None
    except OSError as exc:
        raise InputSnapshotError(
            f"Failed to publish localized LAMMPS stage input {dst}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass

    actual = stable_file_identity(dst, reject_final_symlink=True)
    if (
        int(actual["size_bytes"]) != int(expected["size_bytes"])
        or str(actual["sha256"]) != str(expected["sha256"])
    ):
        raise InputSnapshotError(
            f"Localized LAMMPS stage input failed content verification: {dst}"
        )
    return dst


def _clear_cp2k_project_outputs(
    stage_dir: Path,
    *,
    project: str,
    trajectory_file: str,
    energy_file: str,
    output_file: str,
) -> None:
    """Remove deterministic outputs before a CP2K project invocation.

    CP2K may terminate normally without producing every expected artifact.
    Clearing them first prevents a prior invocation from satisfying the
    current run's restart/trajectory/energy checks.
    """

    def _unlink_required(path: Path) -> None:
        if not (path.exists() or path.is_symlink()):
            return
        try:
            path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot remove stale CP2K project artifact before execution: {path}"
            ) from exc

    for name in (
        str(trajectory_file),
        str(energy_file),
        str(output_file),
        f"{project}-1.restart",
    ):
        _unlink_required(Path(stage_dir) / name)

    # CP2K can retain rolling WFN backups.  No implicit same-project WFN is
    # admissible: render_cp2k_md_input uses ATOMIC unless an independently
    # authenticated WFN_RESTART_FILE_NAME is supplied.  Clear the fixed
    # direct-child project namespace, including broken symlinks, and fail
    # before execution if any candidate cannot be removed.
    root = Path(stage_dir)
    for candidate in sorted(root.glob(f"{project}-RESTART.wfn*")):
        if candidate.parent != root:
            raise RuntimeError(
                f"CP2K WFN cleanup escaped its stage directory: {candidate}"
            )
        _unlink_required(candidate)


def _atomic_write_cp2k_stage_input(path: Path, text: str) -> Path:
    """Publish a generated CP2K input without following a stale file alias."""

    destination = Path(path)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".inp.tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise RuntimeError(
            f"Failed to publish generated CP2K input safely: {destination}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return destination


def _ensure_real_cp2k_stage_dir(stage_dir: Path) -> Path:
    """Create or validate a CP2K stage directory without accepting a symlink."""

    directory = Path(stage_dir)
    try:
        info = directory.lstat()
    except FileNotFoundError:
        ensure_dir(directory)
        try:
            info = directory.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"CP2K stage directory was not created safely: {directory}"
            ) from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot inspect CP2K stage directory: {directory}") from exc
    if stat_module.S_ISLNK(info.st_mode) or not stat_module.S_ISDIR(info.st_mode):
        raise RuntimeError(
            f"CP2K stage_dir must be a real non-symlink directory: {directory}"
        )
    return directory


def _unlink_cp2k_stage_artifact(path: Path) -> None:
    """Strictly clear one deterministic CP2K stage artifact entry."""

    candidate = Path(path)
    try:
        candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(
            f"Cannot inspect stale CP2K stage artifact before execution: {candidate}"
        ) from exc
    try:
        candidate.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot remove stale CP2K stage artifact before execution: {candidate}"
        ) from exc


def _validate_cp2k_stage_artifact_namespace(
    stage: StageSpec,
    *,
    cfg: object,
    log_name: str,
    stage_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Validate the complete direct-child CP2K stage namespace up front.

    CP2K creates several files from each project name in addition to the
    filenames explicitly passed in the input.  User-controlled output, log,
    and localized data names must therefore be disjoint from both fixed
    VitriFlow artifacts and the reserved ``<stage>_...`` project namespace.
    The check is pure and is called before the stage directory is created or
    any BASIS/POTENTIAL/input file is staged.
    """

    stage_name = validate_stage_name(stage.name, context="CP2K stage name")
    output_name = _localized_output_name(stage.output_data)
    validate_stage_name(output_name, context="CP2K output-data basename")
    if Path(output_name).suffix.lower() != ".data":
        raise ValueError(
            "CP2K stage output_data must localize to a path-safe '.data' basename"
        )
    log_basename = validate_stage_name(log_name, context="CP2K log_name")

    roles: dict[str, str] = {
        "localized structure input": "input.data",
        "localized structure output": output_name,
        "synthetic thermo log": log_basename,
        "native MSD": f"{stage_name}.msd.dat",
        "canonical thermo": "thermo.csv",
        "canonical MSD": "msd.csv",
        "neutral trajectory": "traj.extxyz",
        "neutral final structure": "final.extxyz",
        "artifact manifest": STAGE_ARTIFACT_MANIFEST_NAME,
        "SCF diagnostics": "cp2k_scf_diagnostics.json",
        "restart bundle coordinates": "cp2k.restart",
        "restart bundle metadata": "cp2k.restart.json",
        "restart bundle wavefunction": "cp2k-RESTART.wfn",
        "runner screen": "screen.out",
        "runner stdout": "stdout.txt",
        "runner stderr": "stderr.txt",
    }
    if bool(stage.write_dump):
        roles["LAMMPS-compatible trajectory"] = f"{stage_name}.lammpstrj"

    user_owned_roles = {
        "localized structure output": output_name,
        "synthetic thermo log": log_basename,
    }
    for role, attribute in (
        ("localized BASIS file", "basis_set_file_name"),
        ("localized potential file", "potential_file_name"),
    ):
        raw = str(getattr(cfg, attribute))
        configured_path = Path(raw)
        qualified = configured_path.is_absolute() or "/" in raw or "\\" in raw
        if qualified:
            if stage_dir is None:
                continue
            try:
                same_namespace = (
                    configured_path.parent.resolve(strict=False)
                    == Path(stage_dir).resolve(strict=False)
                )
            except OSError as exc:
                raise ValueError(
                    f"Cannot resolve CP2K {role} parent for namespace validation: {raw}"
                ) from exc
            if not same_namespace:
                continue
            basename = validate_stage_name(
                configured_path.name, context=f"CP2K in-stage absolute {role}"
            )
        else:
            basename = validate_stage_name(raw, context=f"CP2K {role}")
        roles[role] = basename
        user_owned_roles[role] = basename

    internal_prefix = f"{stage_name}_"
    prefix_collisions = {
        role: basename
        for role, basename in user_owned_roles.items()
        if basename.startswith(internal_prefix)
    }
    if prefix_collisions:
        details = "; ".join(
            f"{role}: {name!r}" for role, name in sorted(prefix_collisions.items())
        )
        raise ValueError(
            "CP2K user-controlled stage filenames must not occupy the reserved "
            f"project prefix {internal_prefix!r}; {details}"
        )

    by_name: dict[str, list[str]] = {}
    for role, basename in roles.items():
        validate_stage_name(basename, context=f"CP2K {role} basename")
        by_name.setdefault(basename, []).append(role)
    collisions = {
        basename: owners for basename, owners in by_name.items() if len(owners) > 1
    }
    if collisions:
        details = "; ".join(
            f"{basename!r}: {', '.join(owners)}"
            for basename, owners in sorted(collisions.items())
        )
        raise ValueError(
            "CP2K stage artifact basenames must be disjoint; collision(s): " + details
        )

    return roles


def _localized_output_name(output_data: Optional[Union[str, Path]]) -> str:
    if output_data is None:
        return "output.data"
    return Path(output_data).name


def _validate_cp2k_species_coverage(
    symbols: Sequence[str],
    type_to_species: Sequence[str],
) -> None:
    """Require every CP2K-stage species to have a LAMMPS type mapping."""

    known = {str(x) for x in type_to_species}
    unknown = sorted({str(symbol) for symbol in symbols if str(symbol) not in known})
    if unknown:
        raise ValueError(
            f"CP2K stage species {unknown!r} are not covered by "
            f"type_to_species={list(type_to_species)!r}. Update the YAML so the "
            "LAMMPS type ordering covers every species in the input structure."
        )


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
    log_name: str = "log.lammps",
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
    plan = _LocalizedLammpsStage(
        requested_stage=stage,
        local_stage=local_stage,
        stage_dir=Path(stage_dir),
        runner_input=Path(runner_input),
        artifact_input=Path(artifact_input),
        output_local=Path(output_local),
        dump_path=dump_path,
        final_dump_path=Path(stage_dir) / f"{stage.name}.final.lammpstrj",
    )
    _validate_lammps_stage_artifact_namespace(plan, log_name=log_name)
    return plan


def _validate_lammps_stage_artifact_namespace(
    plan: _LocalizedLammpsStage,
    *,
    log_name: str,
    potential_asset_names: Sequence[str] = (),
) -> None:
    """Require every deterministic stage artifact to own a distinct basename."""

    log_basename = validate_stage_name(log_name, context="LAMMPS log_name")
    roles: dict[str, str] = {
        "input snapshot": Path(plan.artifact_input).name,
        "output data": Path(plan.output_local).name,
        "MSD native output": Path(plan.msd_path).name,
        "final dump": Path(plan.final_dump_path).name,
        "LAMMPS input script": "in.lammps",
        "LAMMPS log": log_basename,
        "LAMMPS screen": "screen.out",
        "captured stdout": "stdout.txt",
        "captured stderr": "stderr.txt",
        "canonical thermo": "thermo.csv",
        "canonical MSD": "msd.csv",
        "neutral trajectory": "traj.extxyz",
        "neutral final structure": "final.extxyz",
        "artifact manifest": STAGE_ARTIFACT_MANIFEST_NAME,
    }
    if plan.dump_path is not None:
        roles["trajectory dump"] = Path(plan.dump_path).name
    for index, raw_name in enumerate(potential_asset_names):
        name = validated_lammps_localized_filename(
            raw_name,
            field_name=f"potential asset {index} basename",
        )
        if any(
            name.startswith(f"{runner_name}.failed_skin")
            for runner_name in (log_basename, "screen.out", "stdout.txt", "stderr.txt")
        ):
            raise ValueError(
                f"potential asset {name!r} collides with the neighbor-skin "
                "failure-diagnostic namespace"
            )
        roles[f"potential asset {index}"] = name

    for role, basename in roles.items():
        validate_stage_name(basename, context=f"{role} basename")

    by_name: dict[str, list[str]] = {}
    for role, basename in roles.items():
        by_name.setdefault(basename, []).append(role)
    collisions = {
        basename: owners
        for basename, owners in by_name.items()
        if len(owners) > 1
    }
    if collisions:
        details = "; ".join(
            f"{basename!r}: {', '.join(owners)}"
            for basename, owners in sorted(collisions.items())
        )
        raise ValueError(
            "LAMMPS stage artifact basenames must be disjoint; collision(s): "
            + details
        )


def _lammps_potential_asset_basenames(
    pot_cfg,
    potential_lines: Optional[Sequence[str]],
) -> list[str]:
    """Resolve every file that potential staging will create in the workdir."""

    names: list[str] = []
    for index, source in enumerate(list(getattr(pot_cfg, "files", []) or [])):
        names.append(
            validated_lammps_localized_filename(
                Path(source).name,
                field_name=f"potential.files[{index}] basename",
            )
        )
    if str(getattr(pot_cfg, "kind", "")).strip().lower() == "mg2_sin":
        names.append(
            validated_lammps_localized_filename(
                getattr(pot_cfg, "table_filename", "mg2_sin.table"),
                field_name="MG2 table_filename",
            )
        )
    if potential_lines is not None:
        spec = _parse_tabulated_core_spec(potential_lines)
        if spec is not None:
            raw_name = spec.get("filename", "")
            names.append(
                validated_lammps_localized_filename(
                    "buckingham_core.table" if raw_name == "" else raw_name,
                    field_name="tabulated core filename",
                )
            )
    return names


def _validate_planned_lammps_directory(path: Path, *, context: str) -> Path:
    """Validate one runner-owned directory without creating or following it.

    Stage/work directory names are later embedded in generated LAMMPS paths.
    Keep each one to a single portable component and reject an existing final
    symlink before any input or potential artifact is materialized.
    """

    candidate = Path(path)
    validated_lammps_localized_filename(candidate.name, field_name=context)
    if ".." in candidate.parts:
        raise ValueError(f"{context} must not contain a '..' path component: {candidate}")
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        return candidate
    except OSError as exc:
        raise ValueError(f"Cannot inspect {context}: {candidate}") from exc
    if stat_module.S_ISLNK(info.st_mode):
        raise ValueError(f"{context} must not be a symbolic link: {candidate}")
    if not stat_module.S_ISDIR(info.st_mode):
        raise ValueError(f"{context} must be a directory: {candidate}")
    return candidate


def _validate_continuous_lammps_directories(
    *,
    workdir: Path,
    stage_dirs: Sequence[Path],
) -> tuple[Path, list[Path]]:
    """Bind a continuous run to distinct sibling directories under one root."""

    work = _validate_planned_lammps_directory(
        Path(workdir), context="continuous LAMMPS workdir basename"
    )
    stages = [
        _validate_planned_lammps_directory(
            Path(path), context=f"continuous stage_dirs[{index}] basename"
        )
        for index, path in enumerate(stage_dirs)
    ]
    all_paths = [work, *stages]
    canonical_parents = [path.parent.resolve(strict=False) for path in all_paths]
    if len(set(canonical_parents)) != 1:
        raise ValueError(
            "Continuous LAMMPS workdir and stage_dirs must be direct children "
            "of the same canonical parent directory"
        )
    canonical_paths = [path.resolve(strict=False) for path in all_paths]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError(
            "Continuous LAMMPS workdir and stage_dirs must be mutually distinct"
        )
    return work, stages


def _validate_lammps_workdir_asset_namespace(
    *,
    log_name: str,
    potential_asset_names: Sequence[str],
) -> None:
    """Keep potential assets disjoint from continuous-run root artifacts."""

    log_basename = validate_stage_name(log_name, context="LAMMPS log_name")
    roles: dict[str, str] = {
        "localized input": "input.data",
        "LAMMPS input script": "in.lammps",
        "LAMMPS log": log_basename,
        "LAMMPS screen": "screen.out",
        "captured stdout": "stdout.txt",
        "captured stderr": "stderr.txt",
    }
    for index, raw_name in enumerate(potential_asset_names):
        name = validated_lammps_localized_filename(
            raw_name,
            field_name=f"potential asset {index} basename",
        )
        if any(
            name.startswith(f"{runner_name}.failed_skin")
            for runner_name in (log_basename, "screen.out", "stdout.txt", "stderr.txt")
        ):
            raise ValueError(
                f"potential asset {name!r} collides with the neighbor-skin "
                "failure-diagnostic namespace"
            )
        roles[f"potential asset {index}"] = name

    by_name: dict[str, list[str]] = {}
    for role, name in roles.items():
        by_name.setdefault(name, []).append(role)
    collisions = {
        name: owners for name, owners in by_name.items() if len(owners) > 1
    }
    if collisions:
        details = "; ".join(
            f"{name!r}: {', '.join(owners)}"
            for name, owners in sorted(collisions.items())
        )
        raise ValueError(
            "LAMMPS workdir artifact basenames must be disjoint; collision(s): "
            + details
        )


def _materialize_output_from_final_dump(
    *,
    output_local: Path,
    final_dump_path: Path,
    template_input: Path,
    atom_style: str,
) -> Path:
    """Materialize output from."""

    # This path reconstructs a continuation data file, not an analysis frame:
    # preserve every native number exactly so the next LAMMPS read_data sees
    # the same unit style.  Engine-neutral exports canonicalize separately.
    frame = read_last_dump_frame(Path(final_dump_path), units_style=None)
    masses_by_type = read_datafile_masses(Path(template_input))
    charges_by_id = read_datafile_charges(Path(template_input), atom_style=str(atom_style))
    if str(atom_style).strip().lower() == "charge" and frame.charges is not None:
        frame_charges = np.asarray(frame.charges, dtype=float).reshape(-1)
        frame_ids = np.asarray(frame.ids, dtype=int).reshape(-1)
        if frame_charges.size != frame_ids.size or not np.all(np.isfinite(frame_charges)):
            raise ValueError(
                f"Invalid per-atom charge column in final dump {final_dump_path}; "
                "charged output cannot be reconstructed safely"
            )
        # The final dump is authoritative after `replicate`: replicated atom
        # ids do not exist in the original data-file charge map, while LAMMPS
        # has already copied the correct charge onto every image atom.
        charges_by_id = {
            int(atom_id): float(charge)
            for atom_id, charge in zip(frame_ids.tolist(), frame_charges.tolist())
        }
    elif str(atom_style).strip().lower() == "charge":
        missing_ids = [
            int(atom_id)
            for atom_id in np.asarray(frame.ids, dtype=int).reshape(-1).tolist()
            if int(atom_id) not in charges_by_id
        ]
        if missing_ids:
            raise ValueError(
                "Charged final dump has no q column and its atom ids are not "
                "fully covered by the input data-file charge map; this commonly "
                "indicates a replicated continuous run. Regenerate the dump with "
                f"q included (uncovered ids include {missing_ids[:5]})."
            )

    write_dumpframe_lammps_data(
        Path(output_local),
        frame,
        atom_style=str(atom_style),
        masses_by_type=(masses_by_type if masses_by_type else None),
        charges_by_id=(charges_by_id if str(atom_style).strip().lower() == "charge" else None),
    )
    return Path(output_local)


def _canonical_lammps_thermo_table(table: ThermoTable, *, units_style: str) -> ThermoTable:
    """Convert LAMMPS thermo columns to the engine-neutral reporting contract."""

    columns = list(table.columns)
    data = np.asarray(table.data, dtype=float).copy()
    energy_factor = energy_to_ev_factor(units_style)
    length_factor = length_to_angstrom_factor(units_style)
    pressure_factor = pressure_to_gpa_factor(units_style)
    factors: dict[str, float] = {
        "Time": time_to_ps_factor(units_style),
        "Volume": volume_to_angstrom3_factor(units_style),
        "Vol": volume_to_angstrom3_factor(units_style),
        "Density": density_to_g_cm3_factor(units_style),
        "Press": pressure_factor,
        "Pxx": pressure_factor,
        "Pyy": pressure_factor,
        "Pzz": pressure_factor,
        "Pxy": pressure_factor,
        "Pxz": pressure_factor,
        "Pyz": pressure_factor,
        "Lx": length_factor,
        "Ly": length_factor,
        "Lz": length_factor,
        "Xlo": length_factor,
        "Xhi": length_factor,
        "Ylo": length_factor,
        "Yhi": length_factor,
        "Zlo": length_factor,
        "Zhi": length_factor,
    }
    energy_columns = {
        "PotEng", "KinEng", "TotEng", "Enthalpy", "E_pair", "E_mol",
        "E_vdwl", "E_coul", "E_long", "E_bond", "E_angle", "E_dihed",
        "E_impro", "E_tail", "Pe", "Ke", "Etot",
    }
    for index, name in enumerate(columns):
        factor = energy_factor if name in energy_columns else factors.get(name)
        if factor is not None:
            data[:, index] *= float(factor)
    return ThermoTable(columns=columns, data=data)


def _canonical_cp2k_thermo_table(table: ThermoTable) -> ThermoTable:
    """Convert CP2K's bar pressure to the neutral GPa thermo contract."""

    columns = list(table.columns)
    data = np.asarray(table.data, dtype=float).copy()
    pressure_factor = pressure_to_gpa_factor("metal")  # bar -> GPa
    pressure_columns = {"Press", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"}
    for index, name in enumerate(columns):
        if name in pressure_columns:
            data[:, index] *= float(pressure_factor)
    return ThermoTable(columns=columns, data=data)


def _materialize_thermo_csv_from_log(
    *,
    log_path: Path,
    thermo_csv: Path,
    units_style: str = "metal",
) -> Path:
    """Materialize thermo csv."""

    try:
        table = _canonical_lammps_thermo_table(
            parse_last_thermo_table(log_path),
            units_style=units_style,
        )
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


def _materialize_msd_csv_from_file(
    *,
    msd_path: Path,
    msd_csv: Path,
    units_style: Optional[str] = None,
) -> Path:
    """Materialize msd csv."""

    try:
        msd = parse_msd_file(msd_path)
    except _ARTIFACT_IO_EXCEPTIONS as exc:
        raise MSDArtifactError(f"Failed to parse MSD series from {msd_path} for {msd_csv}") from exc
    try:
        values = np.asarray(msd.msd, dtype=float)
        if units_style is not None:
            values = values * float(msd_to_angstrom2_factor(units_style))
        write_msd_csv(msd_csv, step=msd.step, msd=values)
    except _ARTIFACT_IO_EXCEPTIONS as exc:
        raise MSDArtifactError(
            f"Failed to write engine-neutral MSD CSV {msd_csv} from {msd_path}"
        ) from exc
    return msd_csv


def _write_input_snapshot(*, source: Path, destination: Path) -> Path:
    """Input snapshot."""

    try:
        _atomic_copy_lammps_stage_input(Path(source), Path(destination))
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
    units_style: str = "metal",
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
        _materialize_thermo_csv_from_log(
            log_path=log_path,
            thermo_csv=thermo_csv,
            units_style=units_style,
        )
    except ThermoArtifactError as exc:
        _log_best_effort_failure(exc)
        thermo_csv = _write_placeholder_artifact(thermo_csv, label="thermo.csv")

    try:
        _materialize_msd_csv_from_file(
            msd_path=msd_path,
            msd_csv=msd_csv,
            units_style=units_style,
        )
    except MSDArtifactError as exc:
        _log_best_effort_failure(exc)
        msd_csv = _write_placeholder_artifact(msd_csv, label="msd.csv")

    traj_extxyz, final_extxyz = _materialize_lammps_engine_neutral_outputs(
        stage_dir=plan.stage_dir,
        output_data=plan.output_local,
        dump_path=plan.dump_path,
        md_cfg=md_cfg,
        type_to_species=type_to_species,
        units_style=units_style,
    )

    manifest_path = write_stage_artifact_manifest(
        plan.stage_dir,
        engine="lammps",
        timestep_ps=float(md_cfg.timestep) * float(time_to_ps_factor(units_style)),
        thermo_csv=thermo_csv,
        msd_csv=msd_csv,
        lammps_units_style=units_style,
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
        lammps_units_style=str(units_style),
        manifest_path=manifest_path,
    )


def _materialize_lammps_engine_neutral_outputs(
    *,
    stage_dir: Path,
    output_data: Path,
    dump_path: Optional[Path],
    md_cfg,
    type_to_species: Optional[Sequence[str]] = None,
    units_style: str = "metal",
) -> tuple[Optional[Path], Path]:
    """Materialize lammps engine."""

    traj_extxyz: Optional[Path] = None
    final_extxyz = Path(stage_dir) / "final.extxyz"

    # Never let artifacts from a reused output directory masquerade as
    # products of the current invocation.  A dump-less run must not retain an
    # old traj.extxyz.
    _remove_partial_artifact(Path(stage_dir) / "traj.extxyz")
    _remove_partial_artifact(final_extxyz)

    last_dump_frame = None
    if dump_path is not None and Path(dump_path).exists():
        _traj = Path(stage_dir) / "traj.extxyz"
        try:
            from ..analysis.dump import iter_dump_frames

            last_dump_frame = write_extxyz_iter(
                _traj,
                iter_dump_frames(Path(dump_path), units_style=units_style),
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
                last_dump_frame = read_last_dump_frame(Path(dump_path), units_style=units_style)
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
        fr = read_datafile_frame(
            Path(output_data),
            atom_style=str(md_cfg.atom_style),
            units_style=units_style,
        )
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
        from ase.io import write as ase_write

        at = ase_read_lammps_data(
            Path(output_data),
            atom_style=str(md_cfg.atom_style),
            specorder=list(type_to_species) if type_to_species is not None else None,
            units=str(units_style),
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
            lammps_units_style=str(getattr(pot_cfg, "user_units", "metal") or "metal"),
        )

    if not isinstance(runner, LammpsRunner):
        raise TypeError(f"Unsupported runner type: {type(runner)}")

    if pot_cfg is None:
        raise ValueError("LAMMPS stages require a potential configuration (kim/potential)")

    # Resolve every deterministic direct-child name before creating the stage
    # directory or copying input/potential bytes.  This makes namespace errors
    # fail without partially mutating a requested result directory.
    stage_dir = _validate_planned_lammps_directory(
        Path(stage_dir), context="LAMMPS stage_dir basename"
    )
    potential_assets = _lammps_potential_asset_basenames(
        pot_cfg, potential_lines
    )
    input_local = stage_dir / "input.data"
    preflight_plan = _localize_lammps_stage(
        stage,
        runner_input=input_local,
        artifact_input=input_local,
        stage_dir=stage_dir,
        potential_lines=potential_lines,
        check_input_velocities=False,
        log_name=log_name,
    )
    _validate_lammps_stage_artifact_namespace(
        preflight_plan,
        log_name=log_name,
        potential_asset_names=potential_assets,
    )

    ensure_dir(stage_dir)
    _validate_planned_lammps_directory(
        stage_dir, context="LAMMPS stage_dir basename"
    )
    _atomic_copy_lammps_stage_input(Path(stage.input_data), input_local)
    strip_lammps_data_pair_coeff_sections(input_local)

    plan = _localize_lammps_stage(
        stage,
        runner_input=input_local,
        artifact_input=input_local,
        stage_dir=stage_dir,
        potential_lines=potential_lines,
        check_input_velocities=True,
        log_name=log_name,
    )
    _validate_lammps_stage_artifact_namespace(
        plan,
        log_name=log_name,
        potential_asset_names=potential_assets,
    )
    _remove_partial_artifact(plan.manifest_path)

    prepare_potential_files(pot_cfg, stage_dir, potential_lines)

    skin_used, skin_retries = run_with_neighbor_skin_autotune(
        runner,
        lambda md_use: render_stage(pot_cfg, md_use, plan.local_stage),
        stage_dir,
        md_cfg,
        log_name=log_name,
        cleanup_paths=plan.cleanup_paths(log_name=log_name, include_postprocessed=True),
    )

    return _postprocess_lammps_stage(
        plan,
        md_cfg=md_cfg,
        log_name=log_name,
        type_to_species=type_to_species,
        neighbor_skin=float(skin_used),
        neighbor_skin_retries=int(skin_retries),
        units_style=str(getattr(pot_cfg, "user_units", "metal") or "metal"),
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

    stage_names = [
        validate_stage_name(stage.name, context=f"stages[{index}].name")
        for index, stage in enumerate(stages)
    ]
    if len(set(stage_names)) != len(stage_names):
        duplicates = sorted(
            {name for name in stage_names if stage_names.count(name) > 1}
        )
        raise ValueError(
            "Continuous LAMMPS pipeline requires unique stage names; duplicate(s): "
            + ", ".join(repr(name) for name in duplicates)
        )

    # continuous pipelines replication
    for st in stages[1:]:
        if getattr(st, "replicate", None) is not None:
            raise ValueError("Continuous LAMMPS pipeline supports replicate only on the first stage")

    # Validate the complete workdir/stage namespace before the first mkdir or
    # file copy.  The generated script may then use only bounded ``../stage``
    # paths within one authenticated box root.
    workdir, stage_dirs_list = _validate_continuous_lammps_directories(
        workdir=Path(workdir),
        stage_dirs=[Path(path) for path in stage_dirs],
    )
    potential_assets = _lammps_potential_asset_basenames(
        pot_cfg, potential_lines
    )
    _validate_lammps_workdir_asset_namespace(
        log_name=log_name,
        potential_asset_names=potential_assets,
    )
    input_local0 = workdir / "input.data"
    preflight_plans: list[_LocalizedLammpsStage] = []
    for st, sd in zip(stages, stage_dirs_list):
        preflight_plans.append(
            _localize_lammps_stage(
                st,
                runner_input=input_local0,
                artifact_input=sd / "input.data",
                stage_dir=sd,
                potential_lines=potential_lines,
                check_input_velocities=False,
                log_name=log_name,
            )
        )

    ensure_dir(workdir)
    _validate_planned_lammps_directory(
        workdir, context="continuous LAMMPS workdir basename"
    )

    # localize structure workdir
    input0 = Path(stages[0].input_data)
    _atomic_copy_lammps_stage_input(input0, input_local0)
    strip_lammps_data_pair_coeff_sections(input_local0)

    # directories exist
    for sd in stage_dirs_list:
        ensure_dir(sd)
        _validate_planned_lammps_directory(
            sd, context="continuous LAMMPS stage_dir basename"
        )

    # potential working directory
    prepare_potential_files(pot_cfg, workdir, potential_lines)

    # local specs rendering
    stage_plans: list[_LocalizedLammpsStage] = []
    stage_dir_prefixes: dict[str, str] = {}

    for idx, (st, sd) in enumerate(zip(stages, stage_dirs_list)):
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
                log_name=log_name,
            )
        )

    for plan in stage_plans:
        _remove_partial_artifact(plan.manifest_path)

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
                units_style=str(getattr(pot_cfg, "user_units", "metal") or "metal"),
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
    units_style: str = "metal",
) -> None:
    """Lammpstrj."""

    import numpy as np

    sym_to_type = {sym: i + 1 for i, sym in enumerate(type_to_species)}
    n_atoms = int(len(symbols))

    frac_positions = np.asarray(frac_positions, dtype=float)
    # CP2K/ASE coordinates are Angstrom; a LAMMPS dump has no embedded unit
    # declaration and downstream readers interpret it using the configured
    # LAMMPS style, so serialize native lengths here.
    cells = np.asarray(cells, dtype=float) / float(length_to_angstrom_factor(units_style))

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
                # Right-edge targets ensure the final segment applies T_stop.
                # Left-edge targets omit the requested endpoint entirely.
                Ti = T0 + dT * ((cum + int(n)) / float(total_steps))
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

    _remove_partial_artifact(traj_path)
    _remove_partial_artifact(final_extxyz)

    from ..analysis.dump import DumpFrame

    # Species coverage is a hard contract, not a best-effort artifact concern.
    # Direct indexing ensures an unexpected contract violation raises instead
    # of emitting invalid LAMMPS type zero.
    sym_to_type = {str(sym): i + 1 for i, sym in enumerate(list(type_to_species))}
    ids = np.arange(1, int(len(symbols)) + 1, dtype=int)
    types = np.asarray([sym_to_type[str(sym)] for sym in symbols], dtype=int)

    def _mk_frame(idx: int) -> DumpFrame:
        return DumpFrame(
            timestep=int(steps_all[idx]),
            ids=np.asarray(ids, dtype=int),
            types=np.asarray(types, dtype=int),
            positions=np.asarray(pos_all[idx], dtype=float),
            cell=np.asarray(cells_all[idx], dtype=float),
            origin=np.zeros((3,), dtype=float),
        )

    try:
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


def _read_cp2k_dcd_last(path: Path, *, ref_atoms, aligned: bool = True):
    """Read the final CP2K DCD frame through the validated shared reader."""

    return _read_cp2k_dcd_last_validated(
        path,
        ref_atoms=ref_atoms,
        aligned=bool(aligned),
    )


def _run_stage_local_cp2k(
    runner: Cp2kRunner,
    md_cfg,
    stage: StageSpec,
    stage_dir: Path,
    *,
    type_to_species: Sequence[str],
    log_name: str,
    lammps_units_style: str = "metal",
) -> StageArtifacts:
    """Stage local cp2k."""

    # Validate every deterministic direct-child name before creating the
    # directory, staging CP2K data, clearing prior artifacts, or writing an
    # input.  This keeps configuration errors side-effect free and prevents a
    # BASIS/POTENTIAL/output name from clobbering scientific state.
    _validate_cp2k_stage_artifact_namespace(
        stage,
        cfg=runner.cfg,
        log_name=log_name,
        stage_dir=stage_dir,
    )

    # ase import lightweight
    from ase.io.cp2k import read_cp2k_dcd
    import numpy as np

    stage_dir = _ensure_real_cp2k_stage_dir(stage_dir)
    # Do not let a failed rerun retain the canonical-unit authority from an
    # earlier successful invocation in the same stage directory.
    _unlink_cp2k_stage_artifact(Path(stage_dir) / STAGE_ARTIFACT_MANIFEST_NAME)
    _unlink_cp2k_stage_artifact(Path(stage_dir) / "cp2k_scf_diagnostics.json")

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
    _atomic_copy_lammps_stage_input(Path(stage.input_data), input_local)
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

    # The input snapshot is now independent of any requested in-place output.
    # Clear every derived/public stage artifact before the first engine call so
    # an early failure cannot leave stale science in an apparently current
    # directory, and ordinary writers below cannot follow a prior symlink or
    # hardlink entry.
    stale_stage_artifacts = [
        output_local,
        log_path,
        msd_path,
        thermo_csv,
        msd_csv,
        stage_dir / "traj.extxyz",
        final_extxyz,
        stage_dir / STAGE_ARTIFACT_MANIFEST_NAME,
        stage_dir / "cp2k_scf_diagnostics.json",
        stage_dir / "cp2k.restart",
        stage_dir / "cp2k.restart.json",
        stage_dir / "cp2k-RESTART.wfn",
    ]
    if dump_path is not None:
        stale_stage_artifacts.append(dump_path)
    for stale in stale_stage_artifacts:
        _unlink_cp2k_stage_artifact(stale)

    # structure element species
    # ase lammps parser
    # ase parsing sensitive
    try:
        atoms = ase_read_lammps_data(
            input_local,
            atom_style=str(md_cfg.atom_style),
            specorder=list(type_to_species),
            units=str(lammps_units_style),
        )
    except _ARTIFACT_EXPORT_EXCEPTIONS:
        from ..io.lammps_data_minimal import read_lammps_data_minimal

        atoms = read_lammps_data_minimal(
            input_local,
            atom_style=str(md_cfg.atom_style),
            specorder=list(type_to_species),
            units_style=str(lammps_units_style),
        )
    atoms.pbc = True

    # replicate requested
    if stage.replicate is not None:
        rx, ry, rz = stage.replicate
        atoms = atoms.repeat((int(rx), int(ry), int(rz)))
        atoms.pbc = True

    # Reject incomplete mappings before any expensive CP2K invocation.
    _validate_cp2k_species_coverage(atoms.get_chemical_symbols(), type_to_species)

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

    # CP2K 2024 changed the availability/default semantics of the SCF
    # continuation keyword.  Run all engine-independent validation first, then
    # query the exact configured executable before any input file is rendered
    # or committed to disk; an unknown version is never guessed.
    cp2k_version = runner.query_version(stage_dir)
    cp2k_version_text = ".".join(str(value) for value in cp2k_version)
    _, cp2k_scf_policy = cp2k_scf_continuation_policy(cp2k_version)
    scf_output_diagnostics: list[dict[str, object]] = []

    def _record_scf_diagnostics(output: Path, *, phase: str) -> int:
        failures = count_cp2k_scf_failures(output)
        scf_output_diagnostics.append(
            {
                "phase": str(phase),
                "output": str(Path(output).name),
                "unconverged_scf_cycles": int(failures),
            }
        )
        if failures:
            LOG.warning(
                "CP2K %s continued after %d unconverged SCF cycle(s) in %s (%s)",
                cp2k_version_text,
                failures,
                output,
                cp2k_scf_policy,
            )
        return int(failures)

    initial_restart = _load_cp2k_restart_state_for_stage(stage)
    restart_prev: str | None = (
        None if initial_restart is None else str(initial_restart.restart_file)
    )
    wfn_prev: str | None = (
        None
        if initial_restart is None or initial_restart.wfn_file is None
        else str(initial_restart.wfn_file)
    )
    restart_ensemble: str | None = (
        None if initial_restart is None else str(initial_restart.ensemble)
    )

    def _read_cp2k_dcd_all(path: Path, *, ref_atoms, aligned: bool = True):
        """Cp2k dcd all."""

        with path.open("rb") as fd:
            frames = list(read_cp2k_dcd(fd, index=slice(None), ref_atoms=ref_atoms, aligned=bool(aligned)))
        return frames

    # equilibration start ens
    if stage.equil_steps > 0:
        equil_inp = stage_dir / f"{stage.name}_equil.inp"
        equil_traj = f"{stage.name}_equil.dcd"
        equil_ener = f"{stage.name}_equil.ener"
        equil_project = f"{stage.name}_equil"
        equil_output = stage_dir / f"{stage.name}_equil.out"
        _clear_cp2k_project_outputs(
            stage_dir,
            project=equil_project,
            trajectory_file=equil_traj,
            energy_file=equil_ener,
            output_file=equil_output.name,
        )
        _atomic_write_cp2k_stage_input(
            equil_inp,
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
                project=equil_project,
                energy_every=energy_every,
                traj_every=traj_every,
                traj_file=equil_traj,
                ener_file=equil_ener,
                restart_file=restart_prev,
                restart_wfn_file=wfn_prev,
                restart_barostat=(restart_ensemble == "npt"),
                seed=int(stage.seed),
                cp2k_version=cp2k_version,
            ),
        )
        runner.run(equil_inp, stage_dir, output_name=equil_output.name)
        _record_scf_diagnostics(equil_output, phase="equilibration")
        atoms = _read_cp2k_dcd_last(stage_dir / equil_traj, ref_atoms=atoms, aligned=True)
        atoms.pbc = True

        equil_restart = stage_dir / f"{stage.name}_equil-1.restart"
        if not equil_restart.is_file() or equil_restart.stat().st_size <= 0:
            raise RuntimeError(
                f"CP2K did not produce restart file {equil_restart.name!r}; "
                "sampling cannot continue with velocities/thermostat state intact"
            )
        restart_prev = str(equil_restart.resolve(strict=True))
        equil_wfn = stage_dir / f"{stage.name}_equil-RESTART.wfn"
        wfn_prev = str(equil_wfn.resolve(strict=True)) if equil_wfn.is_file() else None
        restart_ensemble = eq_ens

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

    ener_tables: list[tuple[int, object, dict[int, float], int, int, str]] = []
    # The sampling input is authoritative for global step zero.  Including it
    # prevents a CP2K build that omits the initial DCD frame from assigning the
    # first propagated frame's cell/volume to the initial thermo row.
    traj_frames: list[tuple[int, np.ndarray, np.ndarray]] = [
        (
            0,
            np.asarray(atoms.get_positions(), dtype=float),
            np.asarray(atoms.get_cell(), dtype=float),
        )
    ]
    trajectory_boundary_diagnostics: list[dict[str, object]] = []
    thermo_boundary_diagnostics: list[dict[str, object]] = []
    segment_failure_counts: dict[int, int] = {}
    segment_labels: dict[int, str] = {}
    step_offset = 0
    symbols = list(atoms.get_chemical_symbols())

    for iseg, (Tseg, nsteps) in enumerate(schedule):
        if int(nsteps) <= 0:
            continue
        seg_tag = f"seg{iseg:03d}"
        inp_path = stage_dir / f"{stage.name}_{seg_tag}.inp"
        traj_file = f"{stage.name}_{seg_tag}.dcd"
        ener_file = f"{stage.name}_{seg_tag}.ener"
        segment_project = f"{stage.name}_{seg_tag}"
        segment_output = stage_dir / f"{stage.name}_{seg_tag}.out"
        _clear_cp2k_project_outputs(
            stage_dir,
            project=segment_project,
            trajectory_file=traj_file,
            energy_file=ener_file,
            output_file=segment_output.name,
        )

        _atomic_write_cp2k_stage_input(
            inp_path,
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
                project=segment_project,
                energy_every=energy_every,
                traj_every=traj_every,
                traj_file=traj_file,
                ener_file=ener_file,
                restart_file=restart_prev,
                restart_wfn_file=wfn_prev,
                restart_barostat=(restart_ensemble == "npt"),
                seed=int(stage.seed),
                cp2k_version=cp2k_version,
            ),
        )
        runner.run(inp_path, stage_dir, output_name=segment_output.name)
        segment_failures = _record_scf_diagnostics(
            segment_output, phase=f"sampling_segment_{iseg:03d}"
        )
        segment_failure_counts[int(iseg)] = int(segment_failures)
        segment_labels[int(iseg)] = str(segment_output.name)

        seg_restart = stage_dir / f"{stage.name}_{seg_tag}-1.restart"
        if not seg_restart.is_file() or seg_restart.stat().st_size <= 0:
            raise RuntimeError(
                f"CP2K did not produce restart file {seg_restart.name!r}; "
                "restart continuity and the next stage cannot be guaranteed"
            )
        restart_prev = str(seg_restart.resolve(strict=True))
        seg_wfn = stage_dir / f"{stage.name}_{seg_tag}-RESTART.wfn"
        wfn_prev = str(seg_wfn.resolve(strict=True)) if seg_wfn.is_file() else None
        restart_ensemble = samp_ens

        etab = parse_cp2k_ener(stage_dir / ener_file)
        pressure_by_step: dict[int, float] = {}
        if samp_ens == "npt":
            p_steps, p_values = parse_cp2k_md_step_pressures(segment_output)
            pressure_by_step = map_cp2k_pressures_to_energy_steps(
                etab.step,
                p_steps,
                p_values,
                source=segment_output,
            )
        ener_tables.append(
            (
                step_offset,
                etab,
                pressure_by_step,
                int(iseg),
                int(segment_failures),
                str(segment_output.name),
            )
        )

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

        # A local step-zero DCD frame is a second serialization of the restart
        # boundary.  Verify it modulo periodic wrapping before discarding it.
        # If CP2K omits that optional frame, the already-retained propagated
        # predecessor (or the sampling input at global zero) remains canonical.
        start_i = 0
        if not traj_frames or int(traj_frames[-1][0]) != int(step_offset):
            raise RuntimeError(
                "CP2K trajectory lacks the propagated frame required at segment "
                f"boundary global step {step_offset}"
            )
        if local_steps and int(local_steps[0]) == 0:
            first = frames[0]
            audit = _audit_cp2k_segment_trajectory_boundary(
                global_step=int(step_offset),
                preceding_positions=np.asarray(traj_frames[-1][1], dtype=float),
                preceding_cell=np.asarray(traj_frames[-1][2], dtype=float),
                restart_positions=np.asarray(first.get_positions(), dtype=float),
                restart_cell=np.asarray(first.get_cell(), dtype=float),
            )
            audit.update(
                {
                    "segment": int(iseg),
                    "output": str(segment_output.name),
                    "restart_frame_emitted": True,
                }
            )
            trajectory_boundary_diagnostics.append(audit)
            start_i = 1
        else:
            trajectory_boundary_diagnostics.append(
                {
                    "global_step": int(step_offset),
                    "segment": int(iseg),
                    "output": str(segment_output.name),
                    "restart_frame_emitted": False,
                    "status": "canonical_preceding_frame_retained",
                }
            )

        for iframe in range(start_i, len(frames)):
            at = frames[iframe]
            gstep = int(step_offset + int(local_steps[iframe]))
            if gstep <= int(traj_frames[-1][0]):
                raise RuntimeError(
                    "CP2K segmented trajectory steps are not strictly increasing after "
                    f"boundary reconciliation: {gstep} follows {traj_frames[-1][0]}"
                )
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
        units_style=str(lammps_units_style),
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
    steps_all = np.asarray([t[0] for t in traj_frames], dtype=int)
    if steps_all.size < 1 or np.any(np.diff(steps_all) <= 0):
        raise RuntimeError(
            "CP2K segmented trajectory did not produce a strictly increasing global step axis"
        )
    pos_all = np.asarray([t[1] for t in traj_frames], dtype=float)
    cells_all = np.asarray([t[2] for t in traj_frames], dtype=float)

    # guard
    for c in cells_all:
        _orthorhombic_box_from_cell(c)

    atom_masses = np.asarray(atoms.get_masses(), dtype=float)
    mass_amu = float(np.sum(atom_masses))
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
    thermo_row_segment_ids: list[int] = []
    for offset, etab, pressure_by_step, segment_id, _failures, _output_name in ener_tables:
        s = etab.step.astype(int) + int(offset)
        for local_si, si, Ti, pe in zip(etab.step.astype(int), s, etab.temperature_K, etab.potential_eV):
            s_int = int(si)
            if s_int in step_to_idx:
                j = step_to_idx[s_int]
            else:
                # fallback
                j = int(np.searchsorted(steps_all, s_int, side="right") - 1)
                j = max(0, min(j, len(steps_all) - 1))
            local_step = int(local_si)
            if samp_ens == "npt":
                # Alignment was validated when the segment was parsed.  Only
                # the pre-propagation step zero may lack instantaneous pressure.
                pressure_value = (
                    float(pressure_by_step.get(0, float("nan")))
                    if local_step == 0
                    else float(pressure_by_step[local_step])
                )
            else:
                pressure_value = float("nan")
            thermo_rows.append((
                s_int,
                float(Ti),
                pressure_value,
                float(pe),
                float(vols[j]),
                float(rhos[j]),
            ))
            thermo_row_segment_ids.append(int(segment_id))

    thermo_rows = _merge_cp2k_segment_thermo_rows(
        thermo_rows,
        segment_ids=thermo_row_segment_ids,
        segment_scf_failures=segment_failure_counts,
        segment_labels=segment_labels,
        boundary_diagnostics=thermo_boundary_diagnostics,
    )
    if samp_ens == "npt":
        _validate_cp2k_npt_pressure_rows(thermo_rows)

    # lammps thermo table
    with log_path.open("w") as f:
        f.write("Step Temp Press PotEng Volume Density\n")
        for row in thermo_rows:
            f.write(
                f"{row[0]} {row[1]:.8f} {row[2]:.8f} {row[3]:.10f} {row[4]:.10f} {row[5]:.10f}\n"
            )

    # engine neutral thermo
    cols = ["Step", "Temp", "Press", "PotEng", "Volume", "Density"]
    data = np.asarray(
        [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in thermo_rows], dtype=float
    )
    # Parsed CP2K thermodynamics are an authoritative stage result, not a
    # best-effort visualization.  Never replace valid native data with an empty
    # placeholder if canonical serialization fails.
    _materialize_thermo_csv_from_table(
        table=_canonical_cp2k_thermo_table(ThermoTable(columns=cols, data=data)),
        thermo_csv=thermo_csv,
    )

    # trajectory derived msd
    cell_ref = np.asarray(cells_all[0], dtype=float)
    _orthorhombic_box_from_cell(cell_ref)

    if samp_ens == "nvt":
        msd_all = compute_msd(
            pos_all,
            cell_ref,
            unwrap=True,
            masses=atom_masses,
            remove_com=True,
        )
    else:
        # cell fractional instantaneous
        # reference cell box
        inv_cells = np.linalg.inv(cells_all)
        frac = np.einsum("tnj,tjk->tnk", pos_all, inv_cells)
        frac = frac - np.floor(frac)
        ufrac = unwrap_positions_fractional(frac)
        pos_u = np.einsum("tnj,jk->tnk", ufrac, cell_ref)
        dr = pos_u - pos_u[0:1]
        com_dr = np.sum(dr * atom_masses[None, :, None], axis=1) / mass_amu
        dr = dr - com_dr[:, None, :]
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
            units_style=str(lammps_units_style),
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

    if restart_prev is None or restart_ensemble is None:
        raise RuntimeError(
            "CP2K stage completed without a final restart state; subsequent "
            "velocity-preserving continuation is unsafe"
        )
    published_restart = _publish_cp2k_restart_state(
        stage_dir,
        coordinate_source=output_local,
        restart_source=Path(restart_prev),
        ensemble=restart_ensemble,
        wfn_source=(None if wfn_prev is None else Path(wfn_prev)),
    )

    manifest_path = write_stage_artifact_manifest(
        stage_dir,
        engine="cp2k",
        timestep_ps=float(md_cfg.timestep) * 1.0e-3,
        thermo_csv=thermo_csv,
        msd_csv=msd_csv,
    )

    scf_diagnostics_path = stage_dir / "cp2k_scf_diagnostics.json"
    total_scf_failures = int(
        sum(int(item["unconverged_scf_cycles"]) for item in scf_output_diagnostics)
    )
    write_json_strict(
        scf_diagnostics_path,
        {
            "schema": "vitriflow.cp2k_scf_diagnostics.v1",
            "cp2k_version": cp2k_version_text,
            "policy": cp2k_scf_policy,
            "unconverged_scf_cycles": total_scf_failures,
            "outputs": scf_output_diagnostics,
            "segment_thermo_boundaries": thermo_boundary_diagnostics,
            "segment_trajectory_boundaries": trajectory_boundary_diagnostics,
        },
        indent=2,
        sort_keys=True,
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
        restart_path=published_restart.restart_file,
        wfn_restart_path=published_restart.wfn_file,
        engine="cp2k",
        manifest_path=manifest_path,
        cp2k_version=cp2k_version_text,
        cp2k_scf_policy=cp2k_scf_policy,
        cp2k_unconverged_scf_cycles=total_scf_failures,
        cp2k_scf_diagnostics_path=scf_diagnostics_path,
    )


def stage_outcome_from_artifacts(
    art: StageArtifacts,
    *,
    md_cfg,
    stage: StageSpec,
    lammps_units_style: Optional[str] = None,
) -> StageOutcome:
    """Stage outcome from."""

    if lammps_units_style is None:
        lammps_units_style = getattr(art, "lammps_units_style", None)
    engine = str(getattr(art, "engine", "lammps") or "lammps").strip().lower()

    manifest: Optional[dict[str, object]] = None
    if art.manifest_path is not None:
        manifest_path = Path(art.manifest_path)
        if manifest_path.parent.resolve() != Path(art.stage_dir).resolve():
            raise ValueError(
                f"Stage artifact manifest must be located directly under {art.stage_dir}: "
                f"{manifest_path}"
            )
        manifest = load_stage_artifact_manifest(manifest_path)
        if str(manifest["engine"]) != engine:
            raise ValueError(
                "Stage artifact engine disagrees with its manifest: "
                f"{engine!r} != {manifest['engine']!r}"
            )
        native_units = manifest.get("native_source_units", {})
        manifest_lammps_units = (
            str(native_units.get("lammps_units_style", ""))
            if engine == "lammps" and isinstance(native_units, dict)
            else None
        )
        if manifest_lammps_units:
            if lammps_units_style is not None and str(lammps_units_style).strip().lower() != manifest_lammps_units:
                raise ValueError(
                    "LAMMPS units style disagrees with the stage manifest: "
                    f"{lammps_units_style!r} != {manifest_lammps_units!r}"
                )
            lammps_units_style = manifest_lammps_units
        expected_timestep_ps = float(md_cfg.timestep) * (
            float(time_to_ps_factor(lammps_units_style))
            if engine == "lammps" and lammps_units_style is not None
            else 1.0e-3
        )
        if not math.isclose(
            float(manifest["timestep_ps"]),
            expected_timestep_ps,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Stage timestep disagrees with the canonical manifest: "
                f"{expected_timestep_ps!r} != {manifest['timestep_ps']!r} ps"
            )

    # A current manifest makes the canonical CSV authoritative.  Raw-log
    # fallback is retained only for explicitly pre-manifest StageArtifacts.
    if manifest is not None:
        expected_thermo = Path(art.stage_dir) / "thermo.csv"
        if Path(art.thermo_csv).resolve() != expected_thermo.resolve():
            raise ValueError(
                f"Stage thermo path disagrees with its manifest: {art.thermo_csv}"
            )
        if not verify_manifest_artifact(
            stage_dir=Path(art.stage_dir),
            manifest=manifest,
            artifact_key="thermo_csv",
        ):
            raise ThermoArtifactError(
                f"Current stage has no valid canonical thermo artifact: {expected_thermo}"
            )
        try:
            thermo = parse_thermo_csv(expected_thermo).as_dict()
        except _ARTIFACT_IO_EXCEPTIONS as exc:
            raise ThermoArtifactError(
                f"Manifest-bound thermo artifact failed strict parsing: {expected_thermo}"
            ) from exc
    else:
        try:
            thermo = parse_thermo_csv(art.thermo_csv).as_dict()
        except _ARTIFACT_IO_EXCEPTIONS:
            raw_table = parse_last_thermo_table(art.log_path)
            if lammps_units_style is not None:
                raw_table = _canonical_lammps_thermo_table(
                    raw_table,
                    units_style=lammps_units_style,
                )
            elif engine == "cp2k":
                raw_table = _canonical_cp2k_thermo_table(raw_table)
            thermo = raw_table.as_dict()

    # msd always return
    D = float("nan")
    D_stderr = float("nan")
    D_unconstrained = float("nan")
    D_boundary_constrained = False
    msd_rms_last = float("nan")
    if manifest is not None:
        expected_msd = Path(art.stage_dir) / "msd.csv"
        if Path(art.msd_csv).resolve() != expected_msd.resolve():
            raise ValueError(
                f"Stage MSD path disagrees with its manifest: {art.msd_csv}"
            )
        msd_available = verify_manifest_artifact(
            stage_dir=Path(art.stage_dir),
            manifest=manifest,
            artifact_key="msd_csv",
        )
        if msd_available:
            # Identity verification above and strict parsing below are both
            # authoritative.  Neither may fall back to the native raw file.
            msd_step, msd_values = parse_msd_csv(expected_msd)
            msd_rms_last = float(math.sqrt(float(msd_values[-1])))
            if msd_step.size >= 5:
                diff = estimate_diffusion_from_msd(
                    msd_step,
                    msd_values,
                    timestep=float(manifest["timestep_ps"]),
                    fit_start_fraction=0.5,
                )
                D = float(diff.D)
                D_stderr = float(diff.D_stderr)
                D_unconstrained = float(diff.D_unconstrained)
                D_boundary_constrained = bool(diff.boundary_constrained)
    else:
        try:
            msd = parse_msd_file(art.msd_path)
            msd_step, msd_values = msd.step, msd.msd
            diff = estimate_diffusion_from_msd(
                msd_step,
                msd_values,
                timestep=float(md_cfg.timestep),
                fit_start_fraction=0.5,
            )
            D = float(diff.D)
            D_stderr = float(diff.D_stderr)
            D_unconstrained = float(diff.D_unconstrained)
            D_boundary_constrained = bool(diff.boundary_constrained)
            msd_rms_last = float(math.sqrt(max(0.0, float(msd_values[-1]))))
            if lammps_units_style is not None:
                diff_factor = diffusivity_to_angstrom2_per_ps_factor(lammps_units_style)
                length_factor = length_to_angstrom_factor(lammps_units_style)
                D *= float(diff_factor)
                D_stderr *= float(diff_factor)
                D_unconstrained *= float(diff_factor)
                msd_rms_last *= float(length_factor)
            elif engine == "cp2k":
                # CP2K trajectory displacement is already A^2, while
                # md.timestep is fs; the estimator returns A^2/fs.
                D *= 1000.0
                D_stderr *= 1000.0
                D_unconstrained *= 1000.0
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
        pressure=(
            float(stage.pressure) * 1.0e-4
            if engine == "cp2k"
            else float(stage.pressure)
            * float(pressure_to_gpa_factor(lammps_units_style or "metal"))
        ),
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
        D_unconstrained=float(D_unconstrained),
        D_boundary_constrained=bool(D_boundary_constrained),
        msd_rms_last=float(msd_rms_last),
        output_data=str(art.output_local.relative_to(art.stage_dir)),
        dump=dump_rel,
        neighbor_skin=float(art.neighbor_skin),
        neighbor_skin_retries=int(art.neighbor_skin_retries),
        cp2k_version=getattr(art, "cp2k_version", None),
        cp2k_scf_policy=getattr(art, "cp2k_scf_policy", None),
        cp2k_unconverged_scf_cycles=int(
            getattr(art, "cp2k_unconverged_scf_cycles", 0)
        ),
    )


def recover_completed_stage_outcome(
    stage_dir: Path,
    *,
    md_cfg,
    stage: StageSpec,
    expected_engine: str,
    lammps_units_style: Optional[str] = None,
) -> StageOutcome:
    """Recover a completed stage for post-processing-only resume.

    This path is used when engine execution completed but a later box-analysis
    or plotting step raised before the box entered the production checkpoint.
    Canonical thermo/MSD bytes remain authenticated by the stage manifest;
    required structure/trajectory files must be direct, non-symlink,
    single-link regular files and are parsed again before reuse.
    """

    directory = Path(stage_dir).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(
            f"Completed-stage recovery requires a real stage directory: {directory}"
        )
    directory = directory.resolve(strict=True)

    def _direct_regular(name: str, *, required: bool) -> Optional[Path]:
        candidate = directory / validate_stage_name(
            name,
            context="completed-stage recovery artifact basename",
        )
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            if required:
                raise RuntimeError(
                    f"Completed-stage recovery artifact is missing: {candidate}"
                )
            return None
        if (
            candidate.is_symlink()
            or not stat_module.S_ISREG(info.st_mode)
            or int(info.st_nlink) != 1
            or int(info.st_size) < 1
        ):
            raise RuntimeError(
                "Completed-stage recovery artifact must be a non-empty direct, "
                f"single-link regular file: {candidate}"
            )
        return candidate

    manifest_path = _direct_regular(STAGE_ARTIFACT_MANIFEST_NAME, required=True)
    assert manifest_path is not None
    manifest = load_stage_artifact_manifest(manifest_path)
    engine = str(expected_engine or "").strip().lower()
    if engine not in {"lammps", "cp2k"} or str(manifest.get("engine")) != engine:
        raise RuntimeError(
            "Completed-stage recovery engine disagrees with the stage manifest"
        )

    output_name = _localized_output_name(stage.output_data)
    output_local = _direct_regular(output_name, required=True)
    assert output_local is not None
    input_local = _direct_regular("input.data", required=True)
    assert input_local is not None
    dump_path = _direct_regular(
        f"{stage.name}.lammpstrj",
        required=bool(stage.write_dump),
    )
    traj_extxyz = _direct_regular("traj.extxyz", required=False)
    final_extxyz = _direct_regular("final.extxyz", required=False)

    cp2k_version: Optional[str] = None
    cp2k_scf_policy: Optional[str] = None
    cp2k_unconverged = 0
    cp2k_diag = _direct_regular("cp2k_scf_diagnostics.json", required=False)
    if engine == "cp2k" and cp2k_diag is not None:
        try:
            diagnostic = json.loads(cp2k_diag.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError(
                f"Completed CP2K stage has invalid SCF diagnostics: {cp2k_diag}"
            ) from exc
        if not isinstance(diagnostic, Mapping):
            raise RuntimeError(
                f"Completed CP2K stage has malformed SCF diagnostics: {cp2k_diag}"
            )
        cp2k_version = str(diagnostic.get("cp2k_version") or "") or None
        cp2k_scf_policy = str(diagnostic.get("policy") or "") or None
        cp2k_unconverged = int(diagnostic.get("unconverged_scf_cycles", 0) or 0)

    artifacts = StageArtifacts(
        stage_dir=directory,
        input_local=input_local,
        output_local=output_local,
        log_path=directory / "log.lammps",
        msd_path=directory / f"{stage.name}.msd.dat",
        dump_path=dump_path,
        neighbor_skin=float(getattr(md_cfg, "neighbor_skin", float("nan"))),
        neighbor_skin_retries=0,
        thermo_csv=directory / "thermo.csv",
        msd_csv=directory / "msd.csv",
        traj_extxyz=traj_extxyz,
        final_extxyz=(final_extxyz or directory / "final.extxyz"),
        lammps_units_style=(lammps_units_style if engine == "lammps" else None),
        engine=engine,
        manifest_path=manifest_path,
        cp2k_version=cp2k_version,
        cp2k_scf_policy=cp2k_scf_policy,
        cp2k_unconverged_scf_cycles=cp2k_unconverged,
        cp2k_scf_diagnostics_path=cp2k_diag,
    )
    return stage_outcome_from_artifacts(
        artifacts,
        md_cfg=md_cfg,
        stage=stage,
        lammps_units_style=(lammps_units_style if engine == "lammps" else None),
    )
