from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import warnings
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import TypeAdapter

from ..config import (
    RunConfig,
    MDConfig,
    ThermostatConfig,
    BarostatConfig,
    KimConfig,
    StructureMetricsConfig,
    ProductionEnsembleConfig,
    ConvergenceConfig,
    PotentialConfig,
)
from ..kim import (
    ensure_model_installed,
    ensure_potential_model_installed,
)
from ..engine_identity import (
    assert_engine_build_identity_bundle_unchanged,
    query_engine_build_identities,
    validate_engine_build_identity_bundle,
)
from ..lammps_input import StageSpec
from ..lammps_units import (
    pressure_to_gpa_factor,
)
from ..runner import Cp2kRunner, LammpsRunner
from ..utils import (
    ensure_dir,
    quarantine_uncommitted_box_directories,
    scale_steps_for_timestep,
    stable_file_identity,
)
from ..parse import parse_all_thermo_tables, parse_last_thermo_table, parse_msd_file
from ..io.thermo import parse_thermo_csv
from ..io.extxyz import write_extxyz_single_with_species
from ..io.ase_compat import ase_read_lammps_data
from ..analysis.tm import estimate_tm, estimate_tm_from_diffusion
from ..analysis.stats import early_late_change

from ..analysis.datafile import count_atoms_in_datafile
from ..analysis.dump import DumpFrame
from ..analysis.provenance import write_json_strict
from ..analysis.amorphous import analyse_amorphous_state, summarize_rate_amorphous_acceptance
from ..analysis.motif_summary import summarize_production_crystal_motifs
from ..analysis.convergence import allowed_delta, choose_fastest_converged
from ..analysis.trajectory import read_last_frames_auto, quench_window_steps
from ..analysis.gr import compute_first_peak_gr, compute_gr
from ..analysis.structure import (
    compute_structure_metrics,
    compute_structure_metrics_timeavg,
    compute_structure_distributions_timeavg,
    compute_coordination_defects,
    compute_coordination_defect_details,
    estimate_pair_cutoffs,
    fixed_cutoffs_from_metrics,
    required_pairs_from_metrics,
)

from ..cp2k_driver import (
    assert_cp2k_cell_opt_converged,
    count_cp2k_scf_failures,
    cp2k_scf_continuation_policy,
    density_g_cm3_from_atoms,
    read_cp2k_dcd_last_aligned,
    render_cp2k_cell_opt_input,
)

from ..structuregen import prepare_initial_structure, prepare_size_scan_base_structure
from .elastic_screen import (
    build_elastic_sampling_hint,
    estimate_diffusion_freeze_temperature,
    run_elastic_screen_lammps,
    run_elastic_screen_timeseries_lammps,
    should_collect_elastic_stage_timeseries,
    should_run_elastic_screen,
)
from .stage_metrics import collect_stage_metrics_timeseries, should_collect_stage_metrics_timeseries
from .metrics_policy import resolve_effective_metrics_config
from .production_common import (
    analyse_production_box,
    assess_fixed_count_convergence_posthoc,
    build_production_convergence_spec,
    check_production_convergence,
    graph_analysis_requested,
    make_production_plan,
    metrics_checked_from_conv_spec,
    plan_production_stage_diagnostics,
    production_plan_from_dict,
    production_plan_to_dict,
    resolve_production_relax_dump_settings,
    resolve_production_time_unit_ps,
    resolve_production_warmup_duration_ps,
    resolve_production_warmup_start_temperature,
    resolve_production_warmup_steps,
    validate_production_entry_against_spec,
    write_graph_analysis_outputs,
)
from .progress import CondensedProgressLog, write_autotune_outputs
from .workflow_lock import locked_output_workflow, workflow_payload_entries
from .preflight import run_preflight
from .quench_rates import quench_steps_for_rate, resolve_quench_rates_K_per_time
from .stage_runner import (
    StageOutcome,
    _atomic_write_cp2k_stage_input,
    recover_completed_stage_outcome,
    run_stage_local,
    run_stages_continuous_lammps,
    stage_outcome_from_artifacts,
)
from .step_counts import extend_highT_steps_for_force_isotropic, resolve_lammps_units_style, resolve_md_pressure
from .resume_integrity import (
    attach_production_state_integrity as _attach_production_state_integrity,
    canonical_json_sha256 as _canonical_json_sha256,
    is_zero_committed_active_resume_state,
    potential_command_file_paths,
    prepare_release_resume_migration,
    production_final_status as _production_final_status,
    resolve_result_path as _resolve_result_path,
    seal_release_resume_migration,
    strict_file_identity as _strict_file_identity,
    validate_release_resume_migration,
    validate_production_resume_state as _validate_production_resume_state,
)
from ..runtime_identity import runtime_identity


_AUTOTUNE_RESUME_FINGERPRINT_SCHEMA = "vitriflow.autotune.resume_fingerprint.v3"
_POTENTIAL_ADAPTER = TypeAdapter(PotentialConfig)
_AUTOTUNE_SEED_SCHEME = (
    "stateful_scan_streams_v1+unique_rate_replica_stage_seeds_v1+"
    "production_stage_stream_v1"
)
_AUTOTUNE_RESUME_POLICY = "explicit_state_required_clean_restart_v1"
_CP2K_SCF_DIAGNOSTICS_SCHEMA = "vitriflow.cp2k_scf_diagnostics.v1"
_CP2K_CELL_OPT_IDENTITY_SCHEMA = "vitriflow.cp2k_cell_opt.identity.v1"
_CP2K_CELL_OPT_CALCULATION_SCHEMA = "vitriflow.cp2k_cell_opt.calculation.v1"
_CP2K_CELL_OPT_RESTART_RE = re.compile(r"^dft_opt-([1-9][0-9]*)\.restart$")


def _cp2k_cell_opt_restart_index(path: Path) -> int:
    """Return the canonical CP2K restart index or reject an ambiguous name."""

    name = Path(path).name
    match = _CP2K_CELL_OPT_RESTART_RE.fullmatch(name)
    if match is None:
        raise RuntimeError(
            "Cannot safely resume CP2K CELL_OPT: restart filename must match "
            "'dft_opt-<positive integer>.restart'"
        )
    return int(match.group(1))


def _resolve_autotune_resume_mode(
    *,
    outdir: Path,
    results_path: Path,
    resume: bool | None,
) -> bool:
    """Resolve autotune restart semantics without mixing stale stage state.

    Auto mode resumes only from the protected top-level result bundle.  An
    explicit ``--resume`` requires that bundle, while ``--no-resume`` requires
    a clean directory.  Orphan stage files are never treated as a trustworthy
    checkpoint because they do not bind the complete configuration, selected
    structure, RNG stream, and production convergence state.
    """

    result_path = Path(results_path)
    if result_path.is_symlink():
        raise RuntimeError(
            "Cannot trust autotune_results.json for resume because it is a symbolic link"
        )
    exists = result_path.is_file()
    if resume is True:
        if not exists:
            raise RuntimeError(
                "Cannot resume: --resume was requested but autotune_results.json is "
                "missing; use a fresh empty output directory with --no-resume to "
                "start a new autotune run"
            )
        return True
    if resume is False and exists:
        raise RuntimeError(
            "Cannot start with --no-resume: autotune_results.json already exists in "
            "the output directory; choose a fresh empty output directory"
        )
    if exists:
        return True
    leftovers = sorted(path.name for path in workflow_payload_entries(outdir))
    if leftovers:
        preview = ", ".join(leftovers[:8])
        suffix = " ..." if len(leftovers) > 8 else ""
        raise RuntimeError(
            "Cannot safely start autotune without a protected "
            "autotune_results.json in a non-empty output directory "
            f"({preview}{suffix}); choose a fresh empty output directory"
        )
    return False


def _json_model_payload(value: Any) -> Any:
    """Return a deterministic JSON-compatible payload for validated config."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Cannot build a CP2K CELL_OPT identity from {type(value).__name__}")


def _build_cp2k_cell_opt_calculation_identity(
    *,
    parent_relax_data: Path,
    dft_config: Any,
    cp2k_config: Any,
    cp2k_version: Sequence[int],
    basis_path: Path,
    potential_path: Path,
    base_input_text: str,
    external_pressure_bar: float,
    atom_style: str,
    type_to_species: Optional[Sequence[str]],
    lammps_units_style: str,
) -> dict[str, Any]:
    """Identify the exact physical calculation to which CELL_OPT artifacts belong.

    ``base_input_text`` is rendered without ``EXT_RESTART``.  Restart selection is
    an execution-state detail and is separately content-hashed in the artifact
    manifest; excluding it from the calculation identity lets a verified restart
    continue the same Hamiltonian without pretending to be a different problem.
    """

    version = tuple(int(component) for component in cp2k_version)
    if not version or any(component < 0 for component in version):
        raise ValueError("CP2K CELL_OPT identity requires a parsed non-negative version")
    pressure = float(external_pressure_bar)
    if not math.isfinite(pressure):
        raise ValueError("CP2K CELL_OPT identity requires finite external pressure")
    payload = {
        "schema": _CP2K_CELL_OPT_CALCULATION_SCHEMA,
        "runtime": runtime_identity(),
        "parent_relax_data": _strict_file_identity(
            Path(parent_relax_data), configured_path=Path(parent_relax_data).name
        ),
        "dft_opt_config": _json_model_payload(dft_config),
        "cp2k_config": _json_model_payload(cp2k_config),
        "cp2k_version": [int(component) for component in version],
        "staged_cp2k_data": {
            "basis_set": _strict_file_identity(
                Path(basis_path), configured_path=Path(basis_path).name
            ),
            "potential": _strict_file_identity(
                Path(potential_path), configured_path=Path(potential_path).name
            ),
        },
        "base_rendered_input_sha256": _canonical_json_sha256(
            {"cell_opt_input": str(base_input_text)}
        ),
        "external_pressure_bar": pressure,
        "analysis_bridge": {
            "atom_style": str(atom_style),
            "type_to_species": (
                None
                if type_to_species is None
                else [str(species) for species in type_to_species]
            ),
            "lammps_units_style": str(lammps_units_style),
        },
    }
    return {
        "schema": _CP2K_CELL_OPT_CALCULATION_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "sha256": _canonical_json_sha256(payload),
        "payload": payload,
    }


def _write_cp2k_cell_opt_identity_manifest(
    path: Path,
    *,
    calculation: Mapping[str, Any],
    status: str,
    artifacts: Mapping[str, Path],
    restart_paths: Sequence[Path] = (),
) -> None:
    """Persist an authenticated CELL_OPT calculation/artifact relationship."""

    state = str(status).strip().lower()
    if state not in {"running", "failed", "completed"}:
        raise ValueError(f"Unsupported CP2K CELL_OPT manifest status: {status!r}")
    directory = Path(path).parent
    artifact_rows = {
        str(role): _cp2k_cell_opt_direct_file_identity(
            Path(artifact),
            directory=directory,
        )
        for role, artifact in sorted(artifacts.items())
    }
    restart_candidates = [Path(value) for value in restart_paths]
    restart_candidates.sort(
        key=lambda value: (_cp2k_cell_opt_restart_index(value), value.name)
    )
    restart_rows = [
        _cp2k_cell_opt_direct_file_identity(restart, directory=directory)
        for restart in restart_candidates
    ]
    payload = {
        "schema": _CP2K_CELL_OPT_IDENTITY_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "status": state,
        "calculation": dict(calculation),
        "artifacts": artifact_rows,
        "restart_artifacts": restart_rows,
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    write_json_strict(Path(path), payload, indent=2, sort_keys=True)


def _cp2k_cell_opt_direct_file_identity(
    path: Path,
    *,
    directory: Path,
) -> dict[str, Any]:
    """Hash a unique regular CELL_OPT artifact that is a direct child.

    CELL_OPT outputs and restart inputs are calculation-local mutable state,
    not user-supplied potential/data aliases.  Accepting a final symlink or a
    hard-linked inode would let bytes outside the protected result tree be
    authenticated and later consumed by ``EXT_RESTART``.  Require an
    unaliased direct child and recheck that property around the stable read.
    """

    candidate = Path(path).expanduser()
    parent = Path(directory).expanduser()
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(
            f"CP2K CELL_OPT artifact directory must be a real directory: {parent}"
        )
    try:
        canonical_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"CP2K CELL_OPT artifact directory cannot be resolved: {parent}"
        ) from exc
    if candidate.parent.resolve(strict=False) != canonical_parent:
        raise RuntimeError(
            f"CP2K CELL_OPT artifact must be a direct child of {canonical_parent}: {candidate}"
        )
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(
            f"CP2K CELL_OPT artifact must be a regular non-symlink file: {candidate}"
        )
    before = candidate.stat()
    if int(before.st_nlink) != 1:
        raise RuntimeError(
            f"CP2K CELL_OPT artifact must not be hard-linked: {candidate}"
        )
    identity = stable_file_identity(candidate, reject_final_symlink=True)
    after = candidate.stat()
    if (
        int(after.st_nlink) != 1
        or int(after.st_dev) != int(identity["device"])
        or int(after.st_ino) != int(identity["inode"])
    ):
        raise RuntimeError(
            f"CP2K CELL_OPT artifact link identity changed while hashing: {candidate}"
        )
    return {
        "path": candidate.name,
        "filename": candidate.name,
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def _cp2k_cell_opt_file_identity_matches(
    stored: Mapping[str, Any],
    path: Path,
) -> bool:
    try:
        current = _cp2k_cell_opt_direct_file_identity(
            Path(path),
            directory=Path(path).parent,
        )
        return bool(
            str(stored.get("filename", "")) == str(current["filename"])
            and int(stored.get("size_bytes", -1)) == int(current["size_bytes"])
            and str(stored.get("sha256", "")).lower()
            == str(current["sha256"]).lower()
        )
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return False


def _resolve_cp2k_cell_opt_resume(
    dft_dir: Path,
    *,
    calculation: Mapping[str, Any],
    allow_resume: bool,
) -> dict[str, Any]:
    """Resolve only cryptographically verified CELL_OPT reuse/restart state.

    The result is one of ``fresh``, ``recover_fresh``, ``completed`` or
    ``restart``.  ``recover_fresh`` means an interrupted, uncommitted attempt
    must be quarantined and recomputed from the authenticated classical parent;
    none of its scientific artifacts may be consumed.  Modified completed or
    authenticated-restart evidence remains a hard error.  When resume is not
    requested, no existing artifact is inspected or consumed; the caller starts
    from a clean CELL_OPT directory.
    """

    directory = Path(dft_dir)
    if not allow_resume:
        return {"mode": "fresh", "reason": "resume_not_requested"}

    manifest_path = directory / "cell_opt_identity.json"
    if manifest_path.is_symlink():
        raise RuntimeError(
            "Cannot safely resume CP2K CELL_OPT: identity manifest must not be a symbolic link"
        )
    if manifest_path.exists() and not manifest_path.is_file():
        raise RuntimeError(
            "Cannot safely resume CP2K CELL_OPT: identity manifest is not a regular file"
        )
    scientific_paths = [
        directory / "dft_opt.data",
        directory / "cp2k.out",
        directory / "cell_opt.inp",
        directory / "traj.dcd",
        *sorted(directory.glob("dft_opt-*.restart")),
    ]
    any_scientific_artifact = any(path.exists() or path.is_symlink() for path in scientific_paths)
    if not manifest_path.is_file():
        if any_scientific_artifact:
            return {
                "mode": "recover_fresh",
                "reason": "uncommitted_artifacts_without_identity_manifest",
            }
        return {"mode": "fresh", "reason": "no_prior_artifacts"}

    try:
        manifest_identity = _cp2k_cell_opt_direct_file_identity(
            manifest_path,
            directory=directory,
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest_identity_after = _cp2k_cell_opt_direct_file_identity(
            manifest_path,
            directory=directory,
        )
        if (
            manifest_identity != manifest_identity_after
            or len(manifest_bytes) != int(manifest_identity["size_bytes"])
            or hashlib.sha256(manifest_bytes).hexdigest()
            != str(manifest_identity["sha256"])
        ):
            raise RuntimeError(
                "Cannot safely resume CP2K CELL_OPT: identity manifest changed "
                "while it was read"
            )
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: identity manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: identity manifest is not an object")
    if manifest.get("schema") != _CP2K_CELL_OPT_IDENTITY_SCHEMA:
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: unsupported identity manifest schema")
    recorded_manifest_sha = str(manifest.get("manifest_sha256", "")).lower()
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_sha256", None)
    if recorded_manifest_sha != _canonical_json_sha256(unhashed_manifest):
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: identity manifest was modified")

    stored_calculation = manifest.get("calculation")
    if not isinstance(stored_calculation, Mapping):
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: calculation identity is missing")
    stored_payload = stored_calculation.get("payload")
    if not isinstance(stored_payload, Mapping):
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: calculation payload is malformed")
    if str(stored_calculation.get("sha256", "")).lower() != _canonical_json_sha256(
        dict(stored_payload)
    ):
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: calculation payload was modified")
    expected_payload = calculation.get("payload")
    if not isinstance(expected_payload, Mapping) or str(calculation.get("sha256", "")).lower() != _canonical_json_sha256(
        dict(expected_payload)
    ):
        raise RuntimeError("Internal CP2K CELL_OPT calculation identity is malformed")
    if str(stored_calculation.get("sha256", "")).lower() != str(
        calculation.get("sha256", "")
    ).lower():
        raise RuntimeError(
            "Cannot safely resume CP2K CELL_OPT: parent structure, configuration, "
            "runtime, CP2K version, data files, or rendered input changed"
        )

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: artifact identities are malformed")

    def _require_artifact(role: str, filename: str) -> Path:
        stored = artifacts.get(role)
        path = directory / filename
        if not isinstance(stored, Mapping) or not _cp2k_cell_opt_file_identity_matches(stored, path):
            raise RuntimeError(
                f"Cannot safely resume CP2K CELL_OPT: {role} artifact is missing or changed"
            )
        return path

    state = str(manifest.get("status", "")).strip().lower()
    if state == "completed":
        return {
            "mode": "completed",
            "input": _require_artifact("input", "cell_opt.inp"),
            "output": _require_artifact("output", "cp2k.out"),
            "scf_diagnostics": _require_artifact(
                "scf_diagnostics", "cp2k_scf_diagnostics.json"
            ),
            "trajectory": _require_artifact("trajectory", "traj.dcd"),
            "data": _require_artifact("data", "dft_opt.data"),
            "manifest": manifest_path,
        }
    if state == "running":
        # The input identity proves which calculation was launched, but output
        # and restart files can change after this running manifest is sealed.
        # They are therefore uncommitted and must be quarantined, never reused.
        _require_artifact("input", "cell_opt.inp")
        return {
            "mode": "recover_fresh",
            "reason": "interrupted_running_attempt_without_committed_artifacts",
            "manifest": manifest_path,
        }
    if state != "failed":
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: invalid manifest status")

    restart_rows = manifest.get("restart_artifacts", [])
    if not isinstance(restart_rows, list):
        raise RuntimeError("Cannot safely resume CP2K CELL_OPT: restart identities are malformed")
    current_restart_names = {path.name for path in directory.glob("dft_opt-*.restart")}
    recorded_restart_names: set[str] = set()
    verified_restarts: list[tuple[int, Path]] = []
    for row in restart_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("Cannot safely resume CP2K CELL_OPT: restart identity is malformed")
        name = str(row.get("filename", ""))
        if not name or Path(name).name != name:
            raise RuntimeError("Cannot safely resume CP2K CELL_OPT: restart filename is invalid")
        restart_index = _cp2k_cell_opt_restart_index(Path(name))
        if name in recorded_restart_names:
            raise RuntimeError("Cannot safely resume CP2K CELL_OPT: duplicate restart identity")
        recorded_restart_names.add(name)
        restart_path = directory / name
        if not _cp2k_cell_opt_file_identity_matches(row, restart_path):
            raise RuntimeError(
                f"Cannot safely resume CP2K CELL_OPT: restart artifact {name!r} is missing or changed"
            )
        verified_restarts.append((restart_index, restart_path))
    if current_restart_names != recorded_restart_names:
        raise RuntimeError(
            "Cannot safely resume CP2K CELL_OPT: unbound or missing restart artifacts are present"
        )
    if not verified_restarts:
        return {"mode": "fresh", "reason": "verified_failure_without_restart"}
    _require_artifact("input", "cell_opt.inp")
    selected_index, selected_restart = max(
        verified_restarts, key=lambda value: (value[0], value[1].name)
    )
    return {
        "mode": "restart",
        "restart": selected_restart,
        "restart_index": int(selected_index),
        "manifest": manifest_path,
    }


def _contained_cp2k_cell_opt_path(
    dft_dir: Path,
    *,
    result_root: Path,
) -> tuple[Path, Path, Path]:
    """Return canonical root/path and relative path after link-safe containment."""

    root_alias = Path(result_root).expanduser().absolute()
    directory_alias = Path(dft_dir).expanduser().absolute()
    try:
        relative = directory_alias.relative_to(root_alias)
    except ValueError as exc:
        raise RuntimeError(
            "Interrupted CP2K CELL_OPT directory must remain inside the result tree"
        ) from exc
    if not relative.parts or relative.name != "dft_opt" or ".." in relative.parts:
        raise RuntimeError(
            "Interrupted CP2K CELL_OPT quarantine received an invalid directory path"
        )

    try:
        canonical_root = root_alias.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot resolve CP2K CELL_OPT result root for quarantine: {root_alias}"
        ) from exc
    if not canonical_root.is_dir():
        raise RuntimeError(
            f"CP2K CELL_OPT result root is not a directory: {canonical_root}"
        )

    cursor = canonical_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(
                "CP2K CELL_OPT path must not contain symbolic links: "
                f"{cursor}"
            )
    try:
        cursor.resolve(strict=False).relative_to(canonical_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError("CP2K CELL_OPT path escaped the result tree") from exc
    return canonical_root, cursor, relative


def _quarantine_interrupted_cp2k_cell_opt(
    dft_dir: Path,
    *,
    result_root: Path,
) -> Path:
    """Atomically quarantine an uncommitted CELL_OPT tree without following links.

    The user-facing result root may itself be a symlink to scratch storage, but
    every component below that canonical root must be a real directory.  The
    interrupted tree is moved to an atomically reserved destination on the same
    filesystem, then an empty replacement directory is created for a fresh run.
    """

    canonical_root, interrupted, relative = _contained_cp2k_cell_opt_path(
        dft_dir,
        result_root=result_root,
    )
    if not interrupted.exists() or not interrupted.is_dir():
        raise RuntimeError(
            f"Interrupted CP2K CELL_OPT path is not a real directory: {interrupted}"
        )

    quarantine_relative = (
        Path("interrupted_attempts")
        / "cp2k_cell_opt"
        / relative.parent.name
    )
    quarantine_parent = canonical_root
    for part in quarantine_relative.parts:
        quarantine_parent = quarantine_parent / part
        if quarantine_parent.is_symlink():
            raise RuntimeError(
                "CP2K CELL_OPT quarantine must not contain symbolic links: "
                f"{quarantine_parent}"
            )
    quarantine_parent.mkdir(parents=True, exist_ok=True)
    if quarantine_parent.is_symlink() or not quarantine_parent.is_dir():
        raise RuntimeError(
            f"CP2K CELL_OPT quarantine is not a real directory: {quarantine_parent}"
        )
    try:
        quarantine_parent.resolve(strict=True).relative_to(canonical_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "CP2K CELL_OPT quarantine escaped the result tree"
        ) from exc

    reservation = Path(
        tempfile.mkdtemp(prefix="attempt-", dir=str(quarantine_parent))
    )
    destination = reservation / "dft_opt"
    try:
        os.replace(interrupted, destination)
    except Exception:
        reservation.rmdir()
        raise
    interrupted.mkdir()
    return destination


def _clear_cp2k_cell_opt_artifacts(dft_dir: Path) -> None:
    """Remove every potentially consumable artifact before a fresh CELL_OPT."""

    directory = Path(dft_dir)
    candidates = [
        directory / "dft_opt.data",
        directory / "cp2k.out",
        directory / "cell_opt.inp",
        directory / "traj.dcd",
        directory / "cp2k_scf_diagnostics.json",
        directory / "cell_opt_identity.json",
        *sorted(directory.glob("dft_opt-*.restart")),
    ]
    for path in candidates:
        if path.exists() or path.is_symlink():
            path.unlink()
    _clear_cp2k_cell_opt_wavefunction_restarts(directory)


def _clear_cp2k_cell_opt_wavefunction_restarts(dft_dir: Path) -> None:
    """Remove unbound project-local CELL_OPT WFN state without following links.

    CP2K can discover ``<PROJECT>-RESTART.wfn`` and its backup variants
    implicitly when ``SCF_GUESS RESTART`` is present.  CELL_OPT deliberately
    authenticates only geometry/cell restarts in this release, so no WFN from
    a previous attempt may survive into either a fresh or geometry-restart
    launch.  The fixed direct-child glob cannot escape ``dft_dir``; unlinking a
    symlink removes the link itself and never touches its target.
    """

    directory = Path(dft_dir)
    for path in sorted(directory.glob("dft_opt-RESTART.wfn*")):
        if path.parent != directory:
            raise RuntimeError(
                f"CP2K CELL_OPT WFN cleanup escaped its working directory: {path}"
            )
        if path.exists() or path.is_symlink():
            path.unlink()


def _reconcile_dft_coordination_rejections(
    rejected_rows: Sequence[Mapping[str, Any]],
    boxes: Sequence[Mapping[str, Any]],
    *,
    exclude_defects: bool,
) -> list[dict[str, Any]]:
    """Rebuild derived DFT coordination rejections idempotently from box state."""

    reason = "coordination_defects_dft"
    reconciled: list[dict[str, Any]] = []
    for row in rejected_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("Cannot reconcile a malformed DFT rejection row")
        if str(row.get("reason", "")) != reason:
            reconciled.append(dict(row))
    if not exclude_defects:
        return reconciled

    defective_ids: set[int] = set()
    for entry in boxes:
        if not isinstance(entry, Mapping):
            raise RuntimeError("Cannot reconcile a malformed DFT production box")
        dft = entry.get("dft_opt", {})
        if not isinstance(dft, Mapping) or str(dft.get("status", "")) != "ok":
            continue
        if not bool(dft.get("has_coordination_defects", False)):
            continue
        box_id = int(entry.get("box", 0) or 0)
        if box_id < 1:
            raise RuntimeError(
                "Cannot reconcile DFT coordination rejections for a non-positive box id"
            )
        if box_id in defective_ids:
            raise RuntimeError(
                f"Cannot reconcile duplicate DFT production box id {box_id}"
            )
        defective_ids.add(box_id)
    reconciled.extend(
        {"box": box_id, "reason": reason} for box_id in sorted(defective_ids)
    )
    return reconciled


def _write_cp2k_cell_opt_scf_diagnostics(
    diagnostics_path: Path,
    *,
    output_path: Path,
    failures: int,
    cp2k_version: Optional[str],
    policy: str,
    recovered_from_existing_output: bool,
) -> None:
    """Write the auditable SCF status for one CELL_OPT output."""

    write_json_strict(
        Path(diagnostics_path),
        {
            "schema": _CP2K_SCF_DIAGNOSTICS_SCHEMA,
            "cp2k_version": cp2k_version,
            "policy": str(policy),
            "recovered_from_existing_output": bool(recovered_from_existing_output),
            "unconverged_scf_cycles": int(failures),
            "outputs": [
                {
                    "phase": "cell_optimization",
                    "output": Path(output_path).name,
                    "unconverged_scf_cycles": int(failures),
                }
            ],
        },
        indent=2,
        sort_keys=True,
    )


def _ensure_recovered_cp2k_cell_opt_scf_diagnostics(
    output_path: Path,
    diagnostics_path: Path,
) -> int:
    """Preserve or reconstruct SCF diagnostics when reusing CELL_OPT output.

    A newly queried executable version cannot safely be attributed to an older
    output. If its original sidecar is missing or inconsistent, record the
    counted failure status with an explicitly unknown version/policy instead.
    """

    output = Path(output_path)
    diagnostics = Path(diagnostics_path)
    failures = int(count_cp2k_scf_failures(output))
    valid_existing = False
    if diagnostics.is_file():
        try:
            payload = json.loads(diagnostics.read_text())
            outputs = payload.get("outputs", []) if isinstance(payload, Mapping) else []
            matching_output = any(
                isinstance(item, Mapping)
                and str(item.get("output", "")) == output.name
                and int(item.get("unconverged_scf_cycles", -1)) == failures
                for item in outputs
            )
            valid_existing = bool(
                isinstance(payload, Mapping)
                and payload.get("schema") == _CP2K_SCF_DIAGNOSTICS_SCHEMA
                and int(payload.get("unconverged_scf_cycles", -1)) == failures
                and matching_output
            )
        except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
            valid_existing = False
    if not valid_existing:
        _write_cp2k_cell_opt_scf_diagnostics(
            diagnostics,
            output_path=output,
            failures=failures,
            cp2k_version=None,
            policy="recovered_output_version_unknown",
            recovered_from_existing_output=True,
        )
    return failures


def _compute_dft_distribution_curves(
    frames: Sequence[DumpFrame],
    *,
    metrics_cfg: Any,
    type_to_species: Optional[Sequence[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build DFT g(r)/S(q) payloads without permitting key overwrites."""

    def _slug(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()

    gr_curves: dict[str, Any] = {}
    for gm in list(getattr(metrics_cfg, "gr", [])):
        label = "all" if gm.pair is None else f"{gm.pair[0]}-{gm.pair[1]}"
        key = f"gr_{_slug(label)}"
        if key in gr_curves:
            previous_label = str(gr_curves[key].get("label", ""))
            raise ValueError(
                "Duplicate generated DFT g(r) curve key "
                f"'{key}' for labels '{previous_label}' and '{label}'; "
                "metric labels must be unique after slug normalization"
            )
        r, g, _l = compute_gr(
            frames,
            r_max=float(gm.r_max),
            nbins=int(gm.nbins),
            pair=gm.pair,
            type_to_species=type_to_species,
        )
        gr_curves[key] = {
            "label": str(label),
            "r": [float(v) for v in r.tolist()],
            "g": [float(v) for v in g.tolist()],
        }

    sq_curves: dict[str, Any] = {}
    if hasattr(metrics_cfg, "sq"):
        from ..analysis.sq import compute_sq

        for sm in list(getattr(metrics_cfg, "sq", [])):
            label = "all" if sm.pair is None else f"{sm.pair[0]}-{sm.pair[1]}"
            key = f"sq_{_slug(label)}"
            if key in sq_curves:
                previous_label = str(sq_curves[key].get("label", ""))
                raise ValueError(
                    "Duplicate generated DFT S(q) curve key "
                    f"'{key}' for labels '{previous_label}' and '{label}'; "
                    "metric labels must be unique after slug normalization"
                )
            q, s, representation = compute_sq(
                frames,
                q_max=float(sm.q_max),
                nq=int(sm.nq),
                r_max=float(sm.r_max),
                nbins=int(sm.nbins),
                pair=sm.pair,
                type_to_species=type_to_species,
                window=str(getattr(sm, "window", "lorch")),
                return_metadata=True,
            )
            sq_curves[key] = {
                "label": str(label),
                "q": [float(v) for v in q.tolist()],
                "s": [float(v) for v in s.tolist()],
                "representation": dict(representation),
            }

    return gr_curves, sq_curves


def _config_input_identities(config: RunConfig, *, workdir: Path) -> dict[str, Any]:
    """Hash every user-supplied structure/potential file used by autotune."""

    structure: dict[str, Any] = {}
    lammps_data = getattr(config.structure, "lammps_data", None)
    if lammps_data is not None:
        structure["lammps_data"] = _strict_file_identity(
            Path(lammps_data), configured_path=str(lammps_data)
        )
    generated = getattr(config.structure, "generate", None)
    poscar = getattr(generated, "poscar_path", None) if generated is not None else None
    if poscar is not None:
        structure["poscar_path"] = _strict_file_identity(
            Path(poscar), configured_path=str(poscar)
        )

    potential_cfg = getattr(config, "kim", None)
    potential_paths = [Path(raw) for raw in list(getattr(potential_cfg, "files", None) or [])]
    potential: list[dict[str, Any]] = [
        _strict_file_identity(path, configured_path=str(path)) for path in potential_paths
    ]
    command_files: list[dict[str, Any]] = []
    seen_command_paths = {str(path.resolve(strict=False)) for path in potential_paths}
    potential_payload = (
        potential_cfg.model_dump(mode="json")
        if hasattr(potential_cfg, "model_dump")
        else {}
    )
    for candidate in potential_command_file_paths(
        potential=potential_payload,
        plan={},
        declared_values=potential_paths,
        base_dir=Path.cwd(),
    ):
        key = str(candidate)
        if key in seen_command_paths:
            continue
        seen_command_paths.add(key)
        command_files.append(
            _strict_file_identity(candidate, configured_path=str(candidate))
        )

    cp2k_data: list[dict[str, Any]] = []
    cp2k = getattr(config, "cp2k", None)
    dft_enabled = bool(
        getattr(
            getattr(getattr(config.autotune, "production", None), "dft_opt", None),
            "enabled",
            False,
        )
    )
    if cp2k is not None and (
        str(getattr(config, "engine", "lammps")).strip().lower() == "cp2k" or dft_enabled
    ):
        resolved = Cp2kRunner(cp2k).resolved_data_files(Path(workdir), require=True)
        for role in sorted(resolved):
            item = resolved[role]
            path = Path(item["resolved_path"])
            cp2k_data.append(
                {
                    "role": str(role),
                    "configured_name": str(item["configured_name"]),
                    **_strict_file_identity(path, configured_path=str(path)),
                }
            )
    return {
        "structure_files": structure,
        "potential_files": potential,
        "potential_command_files": command_files,
        "cp2k_data_files": cp2k_data,
    }


def _build_autotune_resume_fingerprint(
    *,
    config: RunConfig,
    outdir: Path,
    selected_structure: Path,
    production_plan: Mapping[str, Any],
    engine_build_identities: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Bind resumable autotune state to the complete effective calculation."""

    runtime = runtime_identity()
    if hasattr(config, "model_dump"):
        effective_config = config.model_dump(mode="json")
    else:
        raise TypeError("Autotune resume fingerprinting requires a validated RunConfig")
    selected = Path(selected_structure)
    # The CLI accepts a relative output directory.  Fresh execution commonly
    # reaches this builder with ``selected_structure`` expressed relative to
    # the process CWD (for example ``out/structure/size_base.data``), whereas
    # resume resolves the path to an absolute filename before rebuilding the
    # fingerprint.  Canonicalise both operands before deriving the protected
    # spelling so those two representations identify the same file.  The
    # content identity below remains authoritative and still rejects any byte
    # change.
    selected_resolved = selected.expanduser().resolve(strict=False)
    outdir_resolved = Path(outdir).expanduser().resolve(strict=False)
    try:
        selected_configured_path = str(selected_resolved.relative_to(outdir_resolved))
    except ValueError:
        selected_configured_path = str(selected_resolved)
    payload = {
        "schema": _AUTOTUNE_RESUME_FINGERPRINT_SCHEMA,
        "workflow": "autotune",
        "vitriflow_version": str(runtime["vitriflow_version"]),
        "runtime": runtime,
        "engine_build_identities": (
            dict(engine_build_identities)
            if engine_build_identities is not None
            else {
                "status": "not_supplied_to_internal_builder",
                "primary_engine": str(config.engine),
            }
        ),
        "seed_scheme": _AUTOTUNE_SEED_SCHEME,
        "resume_policy": _AUTOTUNE_RESUME_POLICY,
        "effective_config": effective_config,
        "input_identities": _config_input_identities(config, workdir=outdir),
        "production_plan": json.loads(
            json.dumps(dict(production_plan), sort_keys=True, allow_nan=False)
        ),
        "selected_structure": _strict_file_identity(
            selected_resolved,
            configured_path=selected_configured_path,
        ),
    }
    return {
        "schema": _AUTOTUNE_RESUME_FINGERPRINT_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "sha256": _canonical_json_sha256(payload),
        "payload": payload,
    }


def _migrate_0_4_35_1_autotune_resume_fingerprint(
    prev: Mapping[str, Any],
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    outdir: Path,
) -> dict[str, Any] | None:
    """Authenticate the exact first-box 0.4.35.1 plotting-hotfix resume."""

    prepared = prepare_release_resume_migration(
        stored,
        current,
        workflow="autotune",
    )
    if prepared is None or not is_zero_committed_active_resume_state(prev):
        return None
    normalized_stored, normalized_current, record = prepared
    old_identity = normalized_stored.get("selected_structure")
    new_identity = normalized_current.get("selected_structure")
    if not isinstance(old_identity, Mapping) or not isinstance(new_identity, Mapping):
        return None
    old_without_path = dict(old_identity)
    new_without_path = dict(new_identity)
    old_path = old_without_path.pop("path", None)
    new_path = new_without_path.pop("path", None)
    if old_path is None or new_path is None or old_without_path != new_without_path:
        return None
    if _resolve_result_path(old_path, outdir=outdir) != _resolve_result_path(
        new_path,
        outdir=outdir,
    ):
        return None
    normalized_stored["selected_structure"] = {
        **dict(old_identity),
        "path": str(new_path),
    }
    if _canonical_json_sha256(normalized_stored) != _canonical_json_sha256(
        normalized_current
    ):
        return None

    record["canonicalized_path_fields"] = (
        ["selected_structure.path"] if str(old_path) != str(new_path) else []
    )
    migrated = dict(current)
    migrated["release_resume_migration"] = seal_release_resume_migration(record)
    return migrated


def _assert_autotune_terminal_fingerprint_unchanged(
    initial: Mapping[str, Any],
    terminal: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject a terminal bundle assembled across changing scientific inputs.

    Fresh and resumed autotune deliberately share this exact gate so neither
    branch can drift to weaker end-of-execution authentication.
    """

    initial_sha = str(initial.get("sha256", "")).strip().lower()
    terminal_sha = str(terminal.get("sha256", "")).strip().lower()
    if not initial_sha or not terminal_sha or terminal_sha != initial_sha:
        raise RuntimeError(
            "Autotune configuration or scientific input bytes changed during "
            f"{context}; refusing to write a terminal result that could combine "
            "different structures, potentials, command includes, or CP2K data"
        )


def _selected_structure_from_autotune_results(prev: Mapping[str, Any], *, outdir: Path) -> Path:
    size = prev.get("size_scan", {})
    rec = prev.get("recommendation", {})
    value = size.get("base_data") if isinstance(size, Mapping) else None
    if value is None and isinstance(rec, Mapping):
        value = rec.get("structure_data")
    if value is None:
        raise RuntimeError("Cannot safely resume autotune: no selected structure path is stored")
    return _resolve_result_path(value, outdir=outdir)


def _validate_autotune_resume_fingerprint(
    prev: Mapping[str, Any],
    *,
    config: RunConfig,
    outdir: Path,
    engine_build_identities: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    stored = prev.get("resume_fingerprint")
    if not isinstance(stored, Mapping):
        raise RuntimeError(
            "Cannot safely resume autotune: autotune_results.json has no provenance fingerprint; "
            "use a fresh output directory for pre-fingerprint results"
        )
    if stored.get("schema") != _AUTOTUNE_RESUME_FINGERPRINT_SCHEMA:
        raise RuntimeError("Cannot safely resume autotune: unsupported provenance fingerprint schema")
    payload = stored.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Cannot safely resume autotune: malformed provenance fingerprint payload")
    stored_sha = str(stored.get("sha256", "")).strip().lower()
    if stored_sha != _canonical_json_sha256(dict(payload)):
        raise RuntimeError("Cannot safely resume autotune: stored fingerprint payload was modified")
    previous_plan = prev.get("production_plan")
    protected_plan = payload.get("production_plan")
    if not isinstance(previous_plan, Mapping) or not isinstance(protected_plan, Mapping):
        raise RuntimeError("Cannot safely resume autotune: no protected production plan is stored")
    if _canonical_json_sha256(dict(previous_plan)) != _canonical_json_sha256(dict(protected_plan)):
        raise RuntimeError("Cannot safely resume autotune: stored production plan was modified")
    if engine_build_identities is None:
        dft_enabled = bool(
            getattr(
                getattr(getattr(config.autotune, "production", None), "dft_opt", None),
                "enabled",
                False,
            )
        )
        engine_build_identities = query_engine_build_identities(
            config,
            workdir=outdir,
            primary_engine=str(config.engine),
            include_cp2k_refinement=dft_enabled,
        )
    validate_engine_build_identity_bundle(engine_build_identities)
    current = _build_autotune_resume_fingerprint(
        config=config,
        outdir=outdir,
        selected_structure=_selected_structure_from_autotune_results(prev, outdir=outdir),
        production_plan=previous_plan,
        engine_build_identities=engine_build_identities,
    )
    if stored_sha != str(current.get("sha256", "")).strip().lower():
        migrated = _migrate_0_4_35_1_autotune_resume_fingerprint(
            prev,
            stored,
            current,
            outdir=outdir,
        )
        if migrated is not None:
            return migrated
        raise RuntimeError(
            "Cannot safely resume autotune: effective configuration, package/seed scheme, selected "
            "structure, or potential contents differ from the existing calculation"
        )
    return current


def _autotune_workflow_status(production: Mapping[str, Any]) -> str:
    if not bool(production.get("enabled", False)):
        return "ok"
    return str(production.get("status", "incomplete"))


def _initial_production_checkpoint_status(*, enabled: bool) -> str:
    """Return the only valid initial status for the requested production mode."""

    return "starting" if bool(enabled) else "not_requested"


def _protected_potential_from_plan(
    plan: Mapping[str, Any],
    *,
    fallback: Any,
) -> Any:
    """Resolve the immutable production-plan potential for resume execution.

    A current YAML/configuration is required to authenticate the resume
    fingerprint, but it is not the execution source once a production plan
    has been protected.  Validate the serialized discriminated union again so
    KIM installation and stage rendering dispatch on the recorded potential
    kind.  Falling back to the current validated configuration is permitted
    only for an older plan that genuinely omitted ``potential_config``.
    """

    protected = plan.get("potential_config")
    if protected is None:
        return fallback
    if not isinstance(protected, Mapping) or not protected:
        raise RuntimeError(
            "Cannot safely resume autotune: protected potential configuration is malformed"
        )
    try:
        return _POTENTIAL_ADAPTER.validate_python(dict(protected))
    except Exception as exc:
        raise RuntimeError(
            "Cannot safely resume autotune: protected potential configuration is invalid"
        ) from exc


def _protected_type_to_species_from_plan(
    plan: Mapping[str, Any],
    *,
    fallback: Optional[Sequence[str]],
) -> Optional[list[str]]:
    """Resolve the immutable type ordering used by production diagnostics."""

    if "type_to_species" not in plan:
        return None if fallback is None else [str(value) for value in fallback]
    protected = plan.get("type_to_species")
    if protected is None:
        return None
    if not isinstance(protected, Sequence) or isinstance(
        protected, (str, bytes, bytearray)
    ):
        raise RuntimeError(
            "Cannot safely resume autotune: protected type_to_species must be a sequence"
        )
    values = list(protected)
    if not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise RuntimeError(
            "Cannot safely resume autotune: protected type_to_species must contain non-empty strings"
        )
    return [value.strip() for value in values]


def _kim_install_jsonable(value: Any) -> Optional[dict[str, Any]]:
    """Return KIM installation state in a strict-JSON-compatible form."""

    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "KIM installation state must be a dataclass, mapping, or None; "
        f"got {type(value).__name__}"
    )


def _production_resume_seed_draw_count(entry: Any) -> int:
    """Validate one persisted production RNG record and return its draw count.

    Production always draws one seed for each of warmup, melt, quench, and
    relax.  The warmup draw is consumed even when its step count is zero, so a
    three-draw fallback would shift every subsequent resumed box.
    """

    if not isinstance(entry, Mapping):
        raise RuntimeError("Cannot safely resume: malformed production box entry")
    expected = {"warmup", "melt", "quench", "relax"}
    nested = entry.get("seeds")
    legacy = {
        str(key)[5:]: value
        for key, value in entry.items()
        if str(key).startswith("seed_")
    }
    if nested is not None and not isinstance(nested, Mapping):
        raise RuntimeError("Cannot safely resume: production seeds record is malformed")
    if isinstance(nested, Mapping):
        seeds = {str(key): value for key, value in nested.items()}
        if legacy and (set(legacy) != expected or legacy != seeds):
            raise RuntimeError(
                "Cannot safely resume: nested and legacy production RNG records disagree"
            )
    else:
        seeds = legacy
    if set(seeds) != expected:
        raise RuntimeError(
            "Cannot safely resume: every production box must record exactly the "
            "warmup, melt, quench, and relax RNG seeds"
        )
    for role in sorted(expected):
        value = seeds.get(role)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"Cannot safely resume: production {role} seed is not an integer"
            )
        seed = int(value)
        if seed < 1 or seed >= 2**31 - 1:
            raise RuntimeError(
                f"Cannot safely resume: production {role} seed is outside the "
                "recorded RNG domain"
            )
    return 4


def _count_production_resume_seed_draws(entries: Sequence[Any]) -> int:
    return sum(_production_resume_seed_draw_count(entry) for entry in entries)


def _format_rate_amorphous_criteria_summary(summary: Mapping[str, Any]) -> str:
    crit = dict(summary.get("criteria_summary", {}) or {})
    if not crit:
        return ""
    parts: list[str] = []
    for name in sorted(crit.keys()):
        payload = dict(crit.get(name, {}) or {})
        n_failed = int(payload.get("n_failed", 0) or 0)
        n_eval = int(payload.get("n_evaluated", 0) or 0)
        if n_eval <= 0 or n_failed <= 0:
            continue
        thr = payload.get("threshold", None)
        mean = payload.get("mean", None)
        vmax = payload.get("max", None)
        seg = f"{name}: failed={n_failed}/{n_eval}"
        if mean is not None:
            try:
                seg += f", mean={float(mean):.4g}"
            except Exception:
                pass
        if vmax is not None:
            try:
                seg += f", max={float(vmax):.4g}"
            except Exception:
                pass
        if thr is not None:
            try:
                seg += f", thr={float(thr):.4g}"
            except Exception:
                pass
        parts.append(seg)
    return "; ".join(parts)


def _write_rate_scan_failure_snapshot(
    *,
    outdir: Path,
    config: RunConfig,
    pot_cfg: Any,
    kim_install: Any,
    preflight: Any,
    T: Sequence[float],
    D: Sequence[float],
    D_mu: Sequence[float],
    D_se: Sequence[float],
    D_med: Sequence[float],
    tm_cfg: Any,
    tm_summary: Any,
    tm_outcomes_all: Sequence[Any],
    tm_est: Any,
    time_unit_ps: Optional[float],
    T_high: float,
    high_total_steps: int,
    force_iso_active: bool,
    high_cfg: Any,
    high_stationarity_summary: Any,
    high_rep_summaries: Sequence[Any],
    high_outcomes: Sequence[Any],
    melt_pool: Sequence[Any],
    melt_data: Any,
    rate_results: Sequence[Mapping[str, Any]],
    cutoffs_rate: Mapping[tuple[int, int], float],
    metric_warnings: Sequence[str],
    metrics_summary: Mapping[str, Any],
    failure_message: str,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    results = {
        "status": "failed",
        "failure": {
            "stage": "rate_scan",
            "reason": str(failure_message),
            "kind": "amorphous_gate",
        },
        "units": {
            "engine": str(getattr(config, "engine", "lammps")),
            "lammps_units": resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal"),
            "time_unit_ps": float(time_unit_ps) if time_unit_ps is not None else None,
        },
        "kim_install": _kim_install_jsonable(kim_install),
        "preflight": asdict(preflight),
        "tm_scan": {
            "temps": [float(t) for t in T],
            "replicates_per_temp": int(getattr(tm_cfg, 'replicates_per_temp', 1)),
            "D": [float(x) for x in D],
            "D_mean": [float(x) for x in D_mu],
            "D_stderr": [float(x) for x in D_se],
            "D_median": [float(x) for x in D_med],
            "summary": tm_summary,
            "outcomes": [asdict(o) for o in tm_outcomes_all],
            "Tm_estimate": {
                "Tm": float(tm_est.Tm),
                "T_liquid": float(getattr(tm_est, "T_liquid", float("nan"))),
                "D_liquid_target": float(getattr(tm_est, "D_liquid_target", float("nan"))),
                "method": str(tm_est.method),
                "score": float(tm_est.score),
                "idx": int(tm_est.idx),
            },
        },
        "highT": {
            "T_high": float(T_high),
            "total_steps": int(high_total_steps),
            "force_isotropic_extension_factor": 1.5 if bool(force_iso_active) else 1.0,
            "replicates": int(getattr(high_cfg, 'replicates', 1)),
            "stationarity": high_stationarity_summary,
            "rep_summaries": list(high_rep_summaries),
            "outcomes": [asdict(o) for o in high_outcomes],
            "melt_pool": [
                str(Path(p).relative_to(outdir)) if Path(p).is_relative_to(outdir) else str(p)
                for p in melt_pool
            ],
            "melt_data": str(Path(melt_data).relative_to(outdir)) if Path(melt_data).is_relative_to(outdir) else str(melt_data),
        },
        "rate_scan": {
            "rates": [dict(rr) for rr in rate_results],
            "decision_density": None,
            "decision_multi": None,
            "cutoffs": [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(cutoffs_rate.items())],
            "accepted_rates": [float(rr.get("rate", float("nan"))) for rr in rate_results if bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))],
            "rejected_rates": [float(rr.get("rate", float("nan"))) for rr in rate_results if not bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))],
        },
        "size_scan": {
            "skipped": True,
            "skip_reason": "rate_scan_failed",
            "base_data": None,
            "base_natoms": None,
            "initial_repeat": None,
            "sizes": [],
            "decision_density": None,
            "decision_multi": None,
            "cutoffs": [],
        },
        "production": {
            "enabled": bool(getattr(getattr(config.autotune, "production", None), "enabled", False)),
            "status": "not_started",
            "converged": False,
            "n_boxes": 0,
            "n_boxes_total": 0,
            "convergence": None,
            "rejected_boxes": [],
        },
        "production_plan": None,
        "recommendation": None,
        "metric_warnings": list(metric_warnings),
        "effective_metrics": dict(metrics_summary),
        "paths": {
            "autotune_results": "autotune_results.json",
            "autotune": "autotune.json",
            "condensed_log": "condensed.log",
        },
    }
    write_autotune_outputs(outdir, results)
    if progress is not None:
        progress.error("rate_scan", f"{failure_message} Diagnostics written to autotune_results.json")
    return results


def _get_type_to_species(config: RunConfig) -> Optional[list[str]]:
    m = config.autotune.metrics
    if m.type_to_species is not None:
        return list(m.type_to_species)
    if config.kim is not None and config.kim.interactions != "fixed_types":
        return list(config.kim.interactions)
    if getattr(config, "engine", "lammps") == "cp2k":
        raise ValueError("engine='cp2k' requires autotune.metrics.type_to_species")
    return None


def _complete_mean_stderr(values: Sequence[float]) -> tuple[float, float]:
    """Mean/SE for a complete replicate set, failing closed when unassessed.

    A missing/non-finite replicate is not exchangeable with a measured value
    and must not be discarded after observing the data.  Likewise, one
    realization has no estimable between-replicate standard error; representing
    that uncertainty as zero would allow a scan point to pass spuriously.
    """
    try:
        arr = np.asarray([float(v) for v in values], dtype=float)
    except Exception:
        return float("nan"), float("nan")
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, float("nan")
    stderr = float(np.std(arr, ddof=1) / math.sqrt(float(arr.size)))
    return mean, stderr


def _complete_tm_replicate_summary(
    values: Sequence[float],
    *,
    require_nonnegative: bool = False,
) -> tuple[float, float, float]:
    """Complete-evidence mean, SE, and median for a TM scan point.

    No replicate may disappear after its value is observed.  A single finite
    realization has an estimable centre but no between-replicate SE, and
    physically constrained quantities must already have been constrained and
    labelled at their estimator boundary rather than clipped here.
    """

    try:
        arr = np.asarray([float(v) for v in values], dtype=float)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return float("nan"), float("nan"), float("nan")
    if require_nonnegative and np.any(arr < 0.0):
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    stderr = (
        float(np.std(arr, ddof=1) / math.sqrt(float(arr.size)))
        if arr.size >= 2
        else float("nan")
    )
    return mean, stderr, median


def _aggregate_scalar_metrics(reps: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate scalar metrics without deleting incomplete replicates."""

    keys: set[str] = set()
    for r in reps:
        keys.update(r.keys())

    mu: dict[str, float] = {}
    se: dict[str, float] = {}

    for k in sorted(keys):
        values: list[float] = []
        for r in reps:
            try:
                values.append(float(r[k]) if k in r else float("nan"))
            except Exception:
                values.append(float("nan"))
        mu[k], se[k] = _complete_mean_stderr(values)
    return mu, se


def _resolve_replicate_traj_path(*, outdir: Path, rep_entry: Mapping[str, Any]) -> Path:
    p = Path(outdir) / Path(rep_entry["dump"])
    cand = Path(p).parent / "traj.extxyz"
    return cand if cand.exists() else p


def _collect_scan_tail_frames(
    scan_results: Sequence[Mapping[str, Any]],
    *,
    outdir: Path,
    metrics_cfg,
    type_to_species: Optional[Sequence[str]],
    lammps_units_style: Optional[str] = "metal",
) -> dict[str, list[Any]]:
    """Scan tail frames."""

    n_frames = max(1, int(getattr(metrics_cfg, "time_average_frames", 1) or 1))
    frames_by_path: dict[str, list[Any]] = {}
    for row in list(scan_results or []):
        if not isinstance(row, Mapping):
            continue
        for rep_entry in list(row.get("replicates", []) or []):
            if not isinstance(rep_entry, Mapping) or "dump" not in rep_entry:
                continue
            traj_path = _resolve_replicate_traj_path(outdir=outdir, rep_entry=rep_entry)
            key = str(traj_path)
            if key in frames_by_path:
                continue
            frames_by_path[key] = read_last_frames_auto(
                traj_path,
                n_frames,
                type_to_species=type_to_species,
                units_style=lammps_units_style,
            )
    return frames_by_path


def _collect_rate_scan_cutoff_reference_frames(
    *,
    rate_results: Sequence[Mapping[str, Any]],
    outdir: Path,
    metrics_cfg,
    type_to_species: Optional[Sequence[str]],
    lammps_units_style: Optional[str] = "metal",
) -> list[Any]:
    """Rate scan cutoff."""

    frames_by_path = _collect_scan_tail_frames(
        rate_results,
        outdir=outdir,
        metrics_cfg=metrics_cfg,
        type_to_species=type_to_species,
        lammps_units_style=lammps_units_style,
    )
    out: list[Any] = []
    for frames in frames_by_path.values():
        out.extend(list(frames or []))
    return out


def _estimate_pooled_scan_cutoffs(
    scan_results: Sequence[Mapping[str, Any]],
    *,
    outdir: Path,
    metrics_cfg,
    required_pairs: Sequence[Tuple[int, int]],
    fixed_cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]],
    lammps_units_style: Optional[str] = "metal",
) -> tuple[dict[tuple[int, int], float], dict[str, list[Any]]]:
    """Pooled scan cutoffs."""

    frames_by_path = _collect_scan_tail_frames(
        scan_results,
        outdir=outdir,
        metrics_cfg=metrics_cfg,
        type_to_species=type_to_species,
        lammps_units_style=lammps_units_style,
    )
    pooled_frames: list[Any] = []
    for frames in frames_by_path.values():
        pooled_frames.extend(list(frames or []))

    missing_pairs = []
    for pair in list(required_pairs or []):
        try:
            a = int(pair[0])
            b = int(pair[1])
        except Exception:
            continue
        key = (a, b) if a <= b else (b, a)
        if key not in fixed_cutoffs:
            missing_pairs.append(key)
    if len(missing_pairs) > 0 and len(pooled_frames) == 0:
        raise ValueError("No trajectory frames available to estimate pooled scan cutoffs.")

    cutoffs = estimate_pair_cutoffs(
        pooled_frames,
        required_pairs,
        auto=metrics_cfg.auto_cutoff,
        fixed_cutoffs=fixed_cutoffs,
    )
    return cutoffs, frames_by_path


def _cutoffs_any_to_dict(obj: Any) -> dict[tuple[int, int], float]:
    """Cutoffs any to."""

    if obj is None:
        return {}

    if isinstance(obj, dict):
        out: dict[tuple[int, int], float] = {}
        for k, v in obj.items():
            if isinstance(k, (list, tuple)) and len(k) == 2:
                a, b = int(k[0]), int(k[1])
            else:
                s = str(k).strip().lstrip("(").rstrip(")")
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) != 2:
                    continue
                a, b = int(parts[0]), int(parts[1])
            out[(min(a, b), max(a, b))] = float(v)
        return out

    if isinstance(obj, list):
        out: dict[tuple[int, int], float] = {}
        for ent in obj:
            if not isinstance(ent, dict):
                continue
            pair = ent.get("pair", None)
            cutoff = ent.get("cutoff", None)
            if pair is None or cutoff is None:
                continue
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = int(pair[0]), int(pair[1])
                out[(min(a, b), max(a, b))] = float(cutoff)
        return out

    return {}


class _UnsupportedScalarConvergenceMetric(ValueError):
    """A calculated scalar with no rate/size-scan tolerance family."""


def _tol_for_metric(name: str, conv) -> tuple[float, float]:
    """Tol for metric."""
    if name == "density":
        return float(conv.density_rel_tol), float(conv.density_abs_tol)
    if name.startswith("coord_"):
        return float(conv.coord_rel_tol), float(conv.coord_abs_tol)
    if name.startswith("bondlen_"):
        return float(conv.bondlen_rel_tol), float(conv.bondlen_abs_tol)
    if name.startswith("angle_"):
        return float(conv.angle_rel_tol), float(conv.angle_abs_tol)
    if name.startswith("ring_frac_"):
        return float(conv.ring_rel_tol), float(conv.ring_abs_tol)
    if name == "ring_mean_size":
        return float(conv.ring_size_rel_tol), float(conv.ring_size_abs_tol)
    # absolute tolerance uncertainty
    if name.startswith("gr_") and name.endswith("_peak_r"):
        return conv.gr_peak_r_rel_tol, conv.gr_peak_r_abs_tol
    if name.startswith("gr_") and name.endswith("_peak_height"):
        return conv.gr_peak_height_rel_tol, conv.gr_peak_height_abs_tol
    if name.startswith("gr_") and name.endswith("_peak_fwhm"):
        return conv.gr_peak_fwhm_rel_tol, conv.gr_peak_fwhm_abs_tol
    raise _UnsupportedScalarConvergenceMetric(
        f"No scalar-scan convergence tolerance defined for metric {name!r}"
    )

def _multimetric_decision(
    x: list[float],
    mu_maps: list[dict[str, float]],
    se_maps: list[dict[str, float]],
    *,
    conv,
    kind: str,
) -> dict[str, Any]:
    """Select the earliest scan point with a complete converged tail.

    ``combined_passed`` remains the historical pointwise candidate/reference
    comparison.  Selection uses ``combined_tail_passed``: the candidate and
    every successively higher-fidelity point through the reference must pass.
    """

    if len(x) < 2:
        raise ValueError("Need >= 2 points for convergence decision")
    if not (len(x) == len(mu_maps) == len(se_maps)):
        raise ValueError("x, mu_maps, se_maps must have same length")

    ref = len(x) - 1
    mu_norm = [{str(k): v for k, v in row.items()} for row in mu_maps]
    se_norm = [{str(k): v for k, v in row.items()} for row in se_maps]
    # Only observables with a defined scalar scan tolerance participate in the
    # rate/size decision.  The metric payload also contains useful calculated
    # diagnostics (for example bond incidence, S(q) peaks, and void summaries)
    # which must remain in results without silently becoming convergence
    # criteria.  Keep the strict resolver for explicit callers and catch only
    # its dedicated unsupported-metric exception here.
    # Derive eligibility from the union of all observed mean/stderr keys.  A
    # missing or non-finite reference value is evidence that the configured
    # criterion could not be assessed, not permission to delete that metric.
    candidate_metrics = sorted(
        {
            str(k)
            for row in [*mu_norm, *se_norm]
            for k in row.keys()
        }
    )
    metric_tolerances: dict[str, tuple[float, float]] = {}
    skipped_metrics: list[dict[str, str]] = []
    for metric_name in candidate_metrics:
        try:
            metric_tolerances[metric_name] = _tol_for_metric(metric_name, conv)
        except _UnsupportedScalarConvergenceMetric:
            skipped_metrics.append(
                {
                    "kind": "scalar_scan",
                    "name": str(metric_name),
                    "reason": (
                        "auxiliary metric has no defined scalar-scan tolerance "
                        "and was excluded from selection"
                    ),
                }
            )
    metrics = sorted(metric_tolerances)
    if not metrics:
        return {
            "kind": kind,
            "chosen_index": int(ref),
            "chosen_value": float(x[ref]),
            "reference_index": int(ref),
            "metrics": {},
            "skipped_metrics": skipped_metrics,
            "combined_passed": [False for _ in x],
            "combined_tail_passed": [False for _ in x],
            "point_criteria_complete": [False for _ in x],
            "tail_criteria_complete": [False for _ in x],
            "criteria_complete": False,
            "selection_criteria_complete": False,
            "blocking_metrics": [],
            "selection_converged": False,
            "fallback_used": True,
            "selection_status": "no_eligible_metrics_unassessed",
            "selection_reason": (
                "no scalar observables with configured convergence tolerances "
                "were present in the scan payload"
            ),
        }

    per_metric: dict[str, dict[str, Any]] = {}
    for m in metrics:
        per_metric[m] = {"deltas": [], "allowed": [], "passed": []}

    metric_values: dict[str, tuple[list[float], list[float]]] = {}
    blocking_metrics: list[dict[str, Any]] = []
    point_criteria_complete = [True for _ in x]
    for m in metrics:
        means: list[float] = []
        stderrs: list[float] = []
        for i in range(len(x)):
            invalid_fields: list[str] = []
            for field_name, row, target in (
                ("mean", mu_norm[i], means),
                ("stderr", se_norm[i], stderrs),
            ):
                try:
                    value = float(row[m]) if m in row else float("nan")
                except Exception:
                    value = float("nan")
                target.append(value)
                if not np.isfinite(value) or (
                    field_name == "stderr" and value < 0.0
                ):
                    invalid_fields.append(field_name)
            if invalid_fields:
                point_criteria_complete[i] = False
                blocking_metrics.append(
                    {
                        "kind": "scalar_scan",
                        "name": str(m),
                        "index": int(i),
                        "x": float(x[i]),
                        "fields": invalid_fields,
                        "reason": (
                            "eligible metric payload is missing, non-numeric, non-finite, "
                            "or has negative standard error"
                        ),
                    }
                )
        metric_values[m] = (means, stderrs)

    combined_pass: list[bool] = []

    for i in range(len(x)):
        ok_all = True
        for m in metrics:
            means, stderrs = metric_values[m]
            mu_i = means[i]
            se_i = stderrs[i]
            mu_r = means[ref]
            se_r = stderrs[ref]

            if not (
                np.isfinite(mu_i)
                and np.isfinite(se_i)
                and se_i >= 0.0
                and np.isfinite(mu_r)
                and np.isfinite(se_r)
                and se_r >= 0.0
            ):
                d = float("nan")
                a = float("nan")
                ok = False
            else:
                rel_tol, abs_tol = metric_tolerances[m]
                d = abs(mu_i - mu_r)
                a = allowed_delta(mu_r, se_i, se_r, rel_tol, abs_tol, float(conv.zscore))
                ok = bool(d <= a)

            # Keep the public decision strict-JSON compatible while the
            # blocking diagnostics preserve why this point was unassessed.
            per_metric[m]["deltas"].append(float(d) if np.isfinite(d) else None)
            per_metric[m]["allowed"].append(float(a) if np.isfinite(a) else None)
            per_metric[m]["passed"].append(bool(ok))
            if not ok:
                ok_all = False

        combined_pass.append(bool(ok_all))

    for m in metrics:
        point_pass = list(per_metric[m]["passed"])
        per_metric[m]["tail_passed"] = [
            bool(all(point_pass[i:])) for i in range(len(point_pass))
        ]
    combined_tail_pass = [
        bool(all(combined_pass[i:])) for i in range(len(combined_pass))
    ]
    tail_criteria_complete = [
        bool(all(point_criteria_complete[i:]))
        for i in range(len(point_criteria_complete))
    ]
    criteria_complete = bool(all(point_criteria_complete))
    chosen: Optional[int] = None
    chosen = next(
        (
            i
            for i, ok in enumerate(combined_tail_pass)
            if ok and tail_criteria_complete[i]
        ),
        None,
    )
    selection_converged = chosen is not None
    if chosen is None:
        # Keep the reference as the safest available fallback, but do not
        # relabel it as converged merely because the scan is exhausted.
        chosen = ref

    return {
        "kind": kind,
        "chosen_index": int(chosen),
        "chosen_value": float(x[chosen]),
        "reference_index": int(ref),
        "metrics": per_metric,
        "skipped_metrics": skipped_metrics,
        "combined_passed": combined_pass,
        "combined_tail_passed": combined_tail_pass,
        "point_criteria_complete": point_criteria_complete,
        "tail_criteria_complete": tail_criteria_complete,
        "criteria_complete": bool(criteria_complete),
        "selection_criteria_complete": bool(tail_criteria_complete[chosen]),
        "blocking_metrics": blocking_metrics,
        "selection_converged": bool(selection_converged),
        "fallback_used": bool(not selection_converged),
        "selection_status": (
            "incomplete_eligible_metrics_unassessed"
            if not tail_criteria_complete[chosen]
            else ("converged" if selection_converged else "fallback_unconverged")
        ),
        "selection_reason": (
            "the selected fallback tail contains missing or non-finite tolerance-supported metric payloads"
            if not tail_criteria_complete[chosen]
            else None
        ),
    }


def _stage_run(
    runner: Union[LammpsRunner, Cp2kRunner],
    pot_cfg,
    md_cfg,
    stage: StageSpec,
    stage_dir: Path,
    *,
    potential_lines: Optional[list[str]] = None,
    type_to_species: Optional[list[str]] = None,
) -> StageOutcome:
    """Stage run."""

    art = run_stage_local(
        runner,
        pot_cfg,
        md_cfg,
        stage,
        stage_dir,
        potential_lines=potential_lines,
        log_name="log.lammps",
        type_to_species=type_to_species,
    )
    return stage_outcome_from_artifacts(
        art,
        md_cfg=md_cfg,
        stage=stage,
        lammps_units_style=(
            str(getattr(pot_cfg, "user_units", "metal") or "metal")
            if isinstance(runner, LammpsRunner)
            else None
        ),
    )






class _ProductionEnsembleRunner:
    """Production ensemble runner."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self) -> dict[str, Any]:
        config = self.config
        outdir = self.outdir
        runner = self.runner
        pot_cfg = self.pot_cfg
        md_use = self.md_use
        potential_lines = self.potential_lines
        type_to_species = self.type_to_species
        metrics_cfg = self.metrics_cfg
        tm_cfg = self.tm_cfg
        q_cfg = self.q_cfg
        size_base_data = self.size_base_data
        chosen_replicate = self.chosen_replicate
        chosen_rate = self.chosen_rate
        dt_ref = self.dt_ref
        dt_mq = self.dt_mq
        cooling_rate_ps = self.cooling_rate_ps
        cutoffs_rate = self.cutoffs_rate
        cutoffs_size = self.cutoffs_size
        T_high = self.T_high
        high_total_steps = self.high_total_steps
        resume_state = self.resume_state
        sampling_hint = self.sampling_hint
        progress = self.progress
        checkpoint_cb = self.checkpoint_cb
        pressure_override = self.pressure_override
        seed_base = self.seed_base
        prod_cfg_override = self.prod_cfg_override
        conv_cfg_override = self.conv_cfg_override
        quench_steps_override = self.quench_steps_override
        relax_steps_override = self.relax_steps_override
        time_unit_ps_override = getattr(self, "time_unit_ps_override", None)
        if progress is None:
            progress = CondensedProgressLog(outdir / "condensed.log")
        progress.info("production", "initialising production ensemble")
        # production ensemble
        production: dict[str, Any] = {"enabled": False}
        prod_cfg = (prod_cfg_override if prod_cfg_override is not None else getattr(config.autotune, "production", None))
        if prod_cfg is not None and bool(getattr(prod_cfg, "enabled", False)):
            if resume_state is not None:
                _validate_production_resume_state(resume_state, outdir=outdir)
            warmup_start_temperature = resolve_production_warmup_start_temperature(
                prod_cfg=prod_cfg,
                T_high=float(T_high),
            )
            warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
            warmup_time_unit_ps = resolve_production_time_unit_ps(
                config=config,
                engine=str(getattr(config, "engine", "lammps") or "lammps"),
                pot_cfg=pot_cfg,
                time_unit_ps=time_unit_ps_override,
            )
            warmup_steps = resolve_production_warmup_steps(
                prod_cfg=prod_cfg,
                md_timestep=float(dt_mq),
                time_unit_ps=warmup_time_unit_ps,
            )
            prod_dir = outdir / "production"
            ensure_dir(prod_dir)

            # selected rate size
            prod_rate = float(chosen_rate)  # k lammps time
            nx, ny, nz = (int(chosen_replicate[0]), int(chosen_replicate[1]), int(chosen_replicate[2]))
            if nx < 1 or ny < 1 or nz < 1:
                raise ValueError(f"Invalid production replicate factors: {chosen_replicate}")

            # quench selected rate
            # ceil high rate
            dT = float(T_high - q_cfg.t_final)
            n_quench_prod = (
                int(quench_steps_override) if quench_steps_override is not None else quench_steps_for_rate(float(dT), float(prod_rate), float(dt_mq), min_steps=1)
            )

            # relax temperature timestep
            relax_steps_prod = (
                int(relax_steps_override) if relax_steps_override is not None else scale_steps_for_timestep(int(q_cfg.relax_steps), dt_ref, dt_mq, min_steps=1)
            )

            # resuming existing production
            # guarantee consistency yaml
            if resume_state is not None:
                rs_q = resume_state.get("quench_steps", None)
                rs_r = resume_state.get("relax_steps", None)
                if isinstance(rs_q, int) and rs_q > 0:
                    n_quench_prod = int(rs_q)
                if isinstance(rs_r, int) and rs_r > 0:
                    relax_steps_prod = int(rs_r)

            # seed
            rng_prod = random.Random(int(seed_base) if seed_base is not None else (int(config.random_seed) + 13579))

            # production warmup initializes
            # downstream preserve recreate
            cont = str(getattr(md_use, "stage_continuity", "discontinuous")).strip().lower()
            vel_next = "preserve" if cont == "continuous" else "create"
            melt_force_iso = bool(getattr(md_use, "force_isotropic", False))

            md_pressure = float(resolve_md_pressure(config, md_use=md_use, override=pressure_override, default=0.0))

            def _maybe_elastic(
                stage_role: str,
                *,
                stage_dir: Path,
                structure_data: Path,
                input_data: Path,
                force_iso_context: bool,
            ) -> Optional[dict[str, Any]]:
                run_screen, strict, _cfg = should_run_elastic_screen(
                    metrics_cfg,
                    runner=runner,
                    stage_role=stage_role,
                    force_isotropic=bool(force_iso_context),
                )
                if not run_screen:
                    return None
                try:
                    return run_elastic_screen_lammps(
                        runner,
                        pot_cfg,
                        md_use,
                        structure_data=structure_data,
                        stage_dir=stage_dir,
                        potential_lines=potential_lines,
                        metrics_cfg=metrics_cfg,
                        force_isotropic=bool(force_iso_context),
                        input_data_for_affine_strain=input_data if bool(force_iso_context) else None,
                        outdir=outdir,
                    )
                except Exception:
                    if strict:
                        raise
                    return None

            def _maybe_elastic_series(
                stage_role: str,
                *,
                stage_dir: Path,
                stage_output_data: Path,
                force_iso_context: bool,
                sampling_hint: Optional[dict[str, float]] = None,
            ) -> Optional[dict[str, Any]]:
                run_series, strict, _cfg = should_collect_elastic_stage_timeseries(
                    metrics_cfg,
                    runner=runner,
                    stage_role=stage_role,
                    force_isotropic=bool(force_iso_context),
                )
                if not run_series:
                    return None
                try:
                    return run_elastic_screen_timeseries_lammps(
                        runner,
                        pot_cfg,
                        md_use,
                        stage_dir=stage_dir,
                        stage_output_data=stage_output_data,
                        stage_role=stage_role,
                        potential_lines=potential_lines,
                        metrics_cfg=metrics_cfg,
                        force_isotropic=bool(force_iso_context),
                        outdir=outdir,
                        sampling_hint=sampling_hint,
                    )
                except Exception:
                    if strict:
                        raise
                    return None

            # structural metrics convergence
            if not bool(metrics_cfg.enabled):
                raise RuntimeError(
                    "Production ensemble generation requires autotune.metrics.enabled=true "
                    "(needed for bond/angle/coordination/ring/g(r) convergence)."
                )

            # scan cutoffs rate
            # possibly downstream analysis
            prod_cutoffs: dict[tuple[int, int], float] = {}
            if isinstance(cutoffs_size, dict) and len(cutoffs_size) > 0:
                prod_cutoffs = dict(cutoffs_size)
            elif isinstance(cutoffs_rate, dict) and len(cutoffs_rate) > 0:
                prod_cutoffs = dict(cutoffs_rate)

            # cutoffs production analysis
            if resume_state is not None:
                rs_cut = _cutoffs_any_to_dict(resume_state.get("cutoffs", None))
                if len(rs_cut) > 0:
                    prod_cutoffs = dict(rs_cut)

            required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
            fixed_cut = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)

            # convergence helpers distribution
            conv_cfg = (conv_cfg_override if conv_cfg_override is not None else config.autotune.convergence)

            def _alpha_from_z(z: float) -> float:
                # sided alpha normal
                p = float(NormalDist().cdf(abs(float(z))))
                a = 2.0 * max(0.0, 1.0 - float(p))
                return float(min(1.0, max(0.0, a)))

            def _critical_value(n: int, alpha: float) -> tuple[float, str]:
                """Critical value."""
                a = float(min(1.0, max(0.0, alpha)))
                if int(n) < 2:
                    return float("inf"), "n<2"
                try:
                    from scipy.stats import t as _t  # type: ignore

                    crit = float(_t.ppf(1.0 - a / 2.0, df=int(n) - 1))
                    if math.isfinite(crit):
                        return crit, "t"
                except Exception:
                    pass
                # fallback
                crit = float(NormalDist().inv_cdf(1.0 - a / 2.0))
                return crit, "z"

            def _tol_scalar(name: str) -> tuple[float, float]:
                if name == "density":
                    return float(conv_cfg.density_rel_tol), float(conv_cfg.density_abs_tol)
                return _tol_for_metric(name, conv_cfg)

            def _tol_curve(kind: str) -> tuple[float, float]:
                if kind == "bondlen_cdf":
                    return float(conv_cfg.bondlen_cdf_rel_tol), float(conv_cfg.bondlen_cdf_abs_tol)
                if kind == "angle_cdf":
                    return float(conv_cfg.angle_cdf_rel_tol), float(conv_cfg.angle_cdf_abs_tol)
                if kind == "coord_cdf":
                    return float(conv_cfg.coord_cdf_rel_tol), float(conv_cfg.coord_cdf_abs_tol)
                if kind == "gr_curve":
                    return float(conv_cfg.gr_curve_rel_tol), float(conv_cfg.gr_curve_abs_tol)
                if kind == "sq_curve":
                    return float(conv_cfg.sq_curve_rel_tol), float(conv_cfg.sq_curve_abs_tol)
                if kind == "void_cdf":
                    return float(conv_cfg.void_cdf_rel_tol), float(conv_cfg.void_cdf_abs_tol)
                return 0.0, 0.0

            def _metric_group(kind: str) -> str:
                if kind in ("density", "gr_curve", "sq_curve", "void_cdf"):
                    return "long"
                if kind == "ring":
                    return "medium"
                if kind in ("bondlen_cdf", "angle_cdf", "coord_cdf"):
                    return "short"
                if kind == "ring_mean_size":
                    return "medium"
                return "other"

            def _vector_stats(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
                """Vector stats."""
                arr = np.asarray(mat, dtype=float)
                if arr.ndim != 2:
                    raise ValueError("mat must be 2D")
                if not np.all(np.isfinite(arr)):
                    raise ValueError("non-finite values in convergence matrix")
                n = int(arr.shape[0])
                mu = np.mean(arr, axis=0)
                if n < 2:
                    sd = np.full_like(mu, np.nan, dtype=float)
                    se = np.full_like(mu, np.nan, dtype=float)
                    return mu, sd, se, n
                sd = np.std(arr, axis=0, ddof=1)
                se = sd / math.sqrt(float(n))
                return mu, sd, se, n

            def _check_convergence(boxes: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
                return check_production_convergence(boxes, spec, conv_cfg)

            # adaptive box converged
            target = int(getattr(prod_cfg, "min_boxes", 10))
            if target < 1:
                target = 1
            minimum_accepted = int(target)

            # boxes hard unset
            # generating convergence achieved
            max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
            max_boxes: int | None
            if max_boxes_raw is None:
                max_boxes = None
            else:
                try:
                    max_boxes = int(max_boxes_raw)
                except Exception:
                    max_boxes = None
            if max_boxes is not None and int(max_boxes) <= 0:
                max_boxes = None
            if max_boxes is not None and int(max_boxes) < target:
                max_boxes = target

            # loop
            # active boxes unset
            HARD_MAX_BOXES = 10000
            batch = int(getattr(prod_cfg, "batch_boxes", 5))
            if batch < 1:
                batch = 1
            do_converge = bool(getattr(prod_cfg, "check_convergence", True))
            stage_diag = plan_production_stage_diagnostics(
                prod_cfg=prod_cfg,
                metrics_cfg=metrics_cfg,
                runner=runner,
                force_isotropic=bool(melt_force_iso),
                total_quench_steps=int(n_quench_prod),
                temperature_start=float(T_high),
                temperature_stop=float(q_cfg.t_final),
                sampling_hint=sampling_hint,
            )
            dump_traj = bool(stage_diag["dump_traj"])
            dump_every = int(stage_diag["dump_every"])
            collect_stage_metric_series = bool(stage_diag["collect_stage_metric_series"])
            collect_elastic_series = dict(stage_diag["collect_elastic_series"])
            need_stage_dump = dict(stage_diag["need_stage_dump"])
            quench_dump_every = int(stage_diag["quench_dump_every"])
            quench_window_steps_range = stage_diag["quench_window_steps_range"]
            relax_dump_settings = resolve_production_relax_dump_settings(stage_diag=stage_diag, metrics_cfg=metrics_cfg)
            if sampling_hint is not None:
                progress.info(
                    "production",
                    f"quench analysis focus: Tm={(sampling_hint or {}).get('Tm')} -> freeze={(sampling_hint or {}).get('freeze_temperature')} ; dump_every={quench_dump_every}",
                )

            bondlen_cdf_points = int(getattr(prod_cfg, "bondlen_cdf_points", 200))
            angle_cdf_points = int(getattr(prod_cfg, "angle_cdf_points", 180))
            store_distributions = bool(getattr(prod_cfg, "store_distributions", True))

            # refinement cell optimisation
            dft_cfg = getattr(prod_cfg, "dft_opt", None)
            dft_enabled = bool(getattr(dft_cfg, "enabled", False)) if dft_cfg is not None else False
            cp2k_opt_runner: Cp2kRunner | None = None
            if dft_enabled:
                # runner backend otherwise
                # dedicated runner refinement
                if isinstance(runner, Cp2kRunner):
                    cp2k_opt_runner = runner
                else:
                    if getattr(config, "cp2k", None) is None:
                        raise RuntimeError(
                            "autotune.production.dft_opt.enabled=true but no cp2k configuration was provided"
                        )
                    cp2k_opt_runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]

            exclude_defects = bool(getattr(prod_cfg, "exclude_coordination_defects", False))
            rejects_subdir = str(getattr(prod_cfg, "rejects_subdir", "rejects") or "rejects")
            rejects_dir = prod_dir / rejects_subdir
            if exclude_defects:
                ensure_dir(rejects_dir)

            # rejected failure coordination
            rejected_boxes_dft: list[dict[str, Any]] = []
            if resume_state is not None:
                prev_rej_dft = resume_state.get("rejected_boxes_dft", None)
                if isinstance(prev_rej_dft, list):
                    rejected_boxes_dft = deepcopy(prev_rej_dft)

            # density conversion lammps
            lammps_units_style = resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal")
            analysis_lammps_units_style: Optional[str] = (
                str(lammps_units_style)
                if isinstance(runner, LammpsRunner)
                else None
            )

            def _density_from_cp2k_atoms(atoms) -> float:
                # CP2K/ASE structures are Angstrom/u and the public analysis
                # contract is g/cm^3, independent of the LAMMPS input style.
                return float(density_g_cm3_from_atoms(atoms))

            def _pressure_to_bar(P: float) -> float:
                """Pressure to bar."""
                p = float(P)
                # native -> GPa -> bar
                return p * float(pressure_to_gpa_factor(lammps_units_style)) * 1.0e4

            def _atoms_to_dumpframe(atoms, *, type_to_species: Optional[list[str]] = None) -> "DumpFrame":
                """Atoms to dumpframe."""
                syms = list(atoms.get_chemical_symbols())
                n = len(syms)
                if int(n) < 1:
                    raise ValueError("DFT optimisation produced an empty structure")

                # symbols lammps integer
                if type_to_species is None:
                    # unique symbols order
                    uniq = sorted(set(syms))
                    sym_to_type = {s: i + 1 for i, s in enumerate(uniq)}
                else:
                    sym_to_type = {str(s): i + 1 for i, s in enumerate(list(type_to_species))}
                try:
                    types = np.asarray([int(sym_to_type[str(s)]) for s in syms], dtype=int)
                except KeyError as e:
                    raise ValueError(f"DFT structure contains symbol not present in type_to_species: {e}") from e

                pos = np.asarray(atoms.get_positions(), dtype=float)
                cell = np.asarray(atoms.get_cell(), dtype=float)
                if cell.shape != (3, 3):
                    raise ValueError("DFT structure has invalid cell shape")
                origin = np.zeros(3, dtype=float)
                ids = np.arange(1, int(n) + 1, dtype=int)
                return DumpFrame(timestep=0, cell=cell, origin=origin, types=types, positions=pos, ids=ids)

            def _run_dft_cell_opt_for_box(entry: dict[str, Any]) -> dict[str, Any]:
                """Dft cell opt."""
                assert cp2k_opt_runner is not None
                assert conv_spec is not None
                assert prod_cutoffs is not None

                box_id = int(entry.get("box", 0) or 0)
                box_dir = prod_dir / f"box_{box_id:03d}"
                dft_dir = box_dir / "dft_opt"
                _contained_cp2k_cell_opt_path(dft_dir, result_root=outdir)
                ensure_dir(dft_dir)

                def _read_lammps_data_to_atoms(p: Path):
                    atoms = None
                    try:
                        # Preserve the data file's exact Masses section.  ASE's
                        # LAMMPS reader maps types back to elemental reference
                        # masses, whose last digits can differ and would make
                        # live and replay density/convergence inputs diverge.
                        from ..io.lammps_data_minimal import read_lammps_data_minimal

                        atoms = read_lammps_data_minimal(
                            p,
                            atom_style=str(md_use.atom_style),
                            specorder=type_to_species,
                            units_style=str(lammps_units_style),
                        )
                    except Exception:
                        try:
                            atoms = ase_read_lammps_data(
                                p,
                                atom_style=str(md_use.atom_style),
                                specorder=type_to_species,
                                units=str(lammps_units_style),
                            )
                        except Exception as e:
                            raise RuntimeError(f"Failed to read LAMMPS data: {p}") from e
                    assert atoms is not None
                    return atoms

                def _analyse_atoms_final(paths: dict[str, str]) -> dict[str, Any]:
                    """Replay the shared production analysis on the refined cell.

                    DFT refinement changes both coordinates and the cell, so
                    every structure-derived descriptor, filter, graph rule,
                    coordination sweep, and amorphous classifier must be
                    recomputed.  Calling the same public box-analysis path used
                    for the MD ensemble prevents a second, slowly diverging
                    implementation of those semantics.
                    """
                    source_path_dft = Path(str(paths.get("dft_data", "")))
                    if not source_path_dft.is_absolute():
                        source_path_dft = outdir / source_path_dft
                    if not source_path_dft.is_file():
                        raise FileNotFoundError(
                            f"DFT final analysis source is missing: {source_path_dft}"
                        )
                    # The serialized LAMMPS-data bridge is the canonical DFT
                    # analysis artifact.  Reload it before calculating density
                    # so live autotune and analyze-output consume identical
                    # cell/mass precision and therefore produce bit-identical
                    # convergence inputs.
                    canonical_atoms = _read_lammps_data_to_atoms(source_path_dft)
                    density_dft = float(_density_from_cp2k_atoms(canonical_atoms))

                    elastic_cfg = getattr(metrics_cfg, "elastic", None)
                    if elastic_cfg is not None and hasattr(elastic_cfg, "model_copy"):
                        elastic_cfg = elastic_cfg.model_copy(update={"enabled": False})
                    metrics_cfg_dft = StructureMetricsConfig.model_validate(
                        metrics_cfg.model_copy(
                            deep=True,
                            update={
                                "collect_during_production_stages": False,
                                "stage_timeseries_make_plot": False,
                                "elastic": elastic_cfg,
                            },
                        )
                    )
                    result_dft, _ = analyse_production_box(
                        box_id=int(box_id),
                        outdir=outdir,
                        melt_stage_dir=dft_dir,
                        quench_stage_dir=dft_dir,
                        relax_stage_dir=dft_dir,
                        relax_data_path=source_path_dft,
                        density_mean=float(density_dft),
                        density_stderr=0.0,
                        metrics_cfg=metrics_cfg_dft,
                        cutoffs=prod_cutoffs,
                        required_pairs=required_pairs,
                        fixed_cutoffs=fixed_cut,
                        type_to_species=type_to_species,
                        md_timestep=float(md_use.timestep),
                        quench_window_steps_range=None,
                        sampling_hint=None,
                        bondlen_cdf_points=bondlen_cdf_points,
                        angle_cdf_points=angle_cdf_points,
                        seeds=None,
                        melt_elastic=None,
                        relax_elastic=None,
                        elastic_timeseries=None,
                        exclude_coordination_defects=exclude_defects,
                        rejects_dir=rejects_dir,
                        relax_dump_path=dft_dir / "no_relax_dump",
                        relax_traj_path=source_path_dft,
                        analysis_source_path=source_path_dft,
                        analysis_source_role="dft_opt_final",
                        atom_style=str(md_use.atom_style),
                        embed_structures=bool(getattr(prod_cfg, "embed_structures", True)),
                        lammps_units_style=str(lammps_units_style),
                        engine="lammps",
                    )
                    validate_production_entry_against_spec(
                        result_dft,
                        conv_spec,
                        box_label=f"DFT {box_id}",
                    )
                    result_dft["status"] = "ok"
                    result_dft["analysis_source_role"] = "dft_opt_final"
                    result_dft["paths"] = {
                        **dict(result_dft.get("paths", {}) or {}),
                        **dict(paths),
                    }
                    return result_dft

                # structure classical relax
                relax_data_rel = Path(entry["paths"]["relax_data"])
                relax_data = (outdir / relax_data_rel) if not relax_data_rel.is_absolute() else relax_data_rel
                if not relax_data.exists():
                    raise FileNotFoundError(str(relax_data))

                atoms0 = _read_lammps_data_to_atoms(relax_data)

                # locally basis pseudopotentials
                # reference explicitly input
                cp2k_opt_runner._ensure_data_files_present(dft_dir)
                import os
                basis_cfg = str(cp2k_opt_runner.cfg.basis_set_file_name)
                pot_cfg = str(cp2k_opt_runner.cfg.potential_file_name)

                if os.path.isabs(basis_cfg) or ("/" in basis_cfg) or ("\\" in basis_cfg):
                    basis_file = basis_cfg
                    basis_path = Path(basis_cfg)
                else:
                    basis_path = dft_dir / Path(basis_cfg).name
                    basis_file = basis_path.name

                if os.path.isabs(pot_cfg) or ("/" in pot_cfg) or ("\\" in pot_cfg):
                    pot_file = pot_cfg
                    pot_path = Path(pot_cfg)
                else:
                    pot_path = dft_dir / Path(pot_cfg).name
                    pot_file = pot_path.name

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

                # determine external pressure
                if getattr(dft_cfg, "external_pressure_bar", None) is not None:
                    extP = float(getattr(dft_cfg, "external_pressure_bar"))
                else:
                    extP = float(_pressure_to_bar(float(md_pressure)))

                cp2k_version = cp2k_opt_runner.query_version(dft_dir)
                cp2k_version_text = ".".join(str(value) for value in cp2k_version)
                _, cp2k_scf_policy = cp2k_scf_continuation_policy(cp2k_version)
                base_inp_txt = render_cp2k_cell_opt_input(
                    atoms=atoms0,
                    cfg=cp2k_opt_runner.cfg,
                    basis_set_file_name=basis_file,
                    potential_file_name=pot_file,
                    project="dft_opt",
                    optimizer=str(getattr(dft_cfg, "optimizer", "LBFGS")),
                    max_iter=int(getattr(dft_cfg, "max_iter", 200)),
                    keep_angles=bool(getattr(dft_cfg, "keep_angles", True)),
                    external_pressure_bar=float(extP),
                    traj_every=int(getattr(dft_cfg, "traj_every", 1)),
                    traj_file="traj.dcd",
                    print_level=str(getattr(dft_cfg, "print_level", "LOW")),
                    restart_file=None,
                    cp2k_version=cp2k_version,
                )
                calculation_identity = _build_cp2k_cell_opt_calculation_identity(
                    parent_relax_data=relax_data,
                    dft_config=dft_cfg,
                    cp2k_config=cp2k_opt_runner.cfg,
                    cp2k_version=cp2k_version,
                    basis_path=basis_path,
                    potential_path=pot_path,
                    base_input_text=base_inp_txt,
                    external_pressure_bar=float(extP),
                    atom_style=str(md_use.atom_style),
                    type_to_species=type_to_species,
                    lammps_units_style=str(lammps_units_style),
                )
                resume_decision = _resolve_cp2k_cell_opt_resume(
                    dft_dir,
                    calculation=calculation_identity,
                    allow_resume=bool(resume_state is not None),
                )
                if str(resume_decision.get("mode")) == "recover_fresh":
                    recovery_reason = str(resume_decision.get("reason", "interrupted_attempt"))
                    quarantined = _quarantine_interrupted_cp2k_cell_opt(
                        dft_dir,
                        result_root=outdir,
                    )
                    # The complete directory was moved atomically, including
                    # locally staged basis/potential files.  Restage them and
                    # prove that the calculation identity is unchanged before
                    # launching a fresh CELL_OPT from the protected MD parent.
                    cp2k_opt_runner._ensure_data_files_present(dft_dir)
                    if not basis_path.is_file() or not pot_path.is_file():
                        raise RuntimeError(
                            "CP2K data files were not restored after interrupted "
                            "CELL_OPT quarantine"
                        )
                    recovered_calculation_identity = (
                        _build_cp2k_cell_opt_calculation_identity(
                            parent_relax_data=relax_data,
                            dft_config=dft_cfg,
                            cp2k_config=cp2k_opt_runner.cfg,
                            cp2k_version=cp2k_version,
                            basis_path=basis_path,
                            potential_path=pot_path,
                            base_input_text=base_inp_txt,
                            external_pressure_bar=float(extP),
                            atom_style=str(md_use.atom_style),
                            type_to_species=type_to_species,
                            lammps_units_style=str(lammps_units_style),
                        )
                    )
                    if str(recovered_calculation_identity.get("sha256", "")).lower() != str(
                        calculation_identity.get("sha256", "")
                    ).lower():
                        raise RuntimeError(
                            "CP2K CELL_OPT scientific inputs changed while recovering "
                            "an interrupted attempt"
                        )
                    warnings.warn(
                        "Quarantined an uncommitted CP2K CELL_OPT attempt and will "
                        f"rerun it from the authenticated parent ({recovery_reason}; "
                        f"preserved at {quarantined})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    resume_decision = {
                        "mode": "fresh",
                        "reason": f"recovered_{recovery_reason}",
                        "quarantined": quarantined,
                    }
                identity_path = dft_dir / "cell_opt_identity.json"
                if str(resume_decision.get("mode")) == "completed":
                    out_path0 = Path(resume_decision["output"])
                    existing_data = Path(resume_decision["data"])
                    assert_cp2k_cell_opt_converged(out_path0)
                    scf_diagnostics_path0 = dft_dir / "cp2k_scf_diagnostics.json"
                    recovered_scf_failures = _ensure_recovered_cp2k_cell_opt_scf_diagnostics(
                        out_path0,
                        scf_diagnostics_path0,
                    )
                    if recovered_scf_failures:
                        warnings.warn(
                            "Reused CP2K CELL_OPT output contains "
                            f"{recovered_scf_failures} unconverged SCF cycle(s); "
                            "the positive optimization-completion marker was still required",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    paths: dict[str, str] = {
                        "dft_dir": str(dft_dir.relative_to(outdir)),
                        "dft_data": str(existing_data.relative_to(outdir)),
                        "dft_input": str(Path(resume_decision["input"]).relative_to(outdir)),
                        "dft_output": str(out_path0.relative_to(outdir)),
                        "dft_scf_diagnostics": str(
                            scf_diagnostics_path0.relative_to(outdir)
                        ),
                        "dft_identity": str(identity_path.relative_to(outdir)),
                    }
                    traj_path0 = dft_dir / "traj.dcd"
                    if traj_path0.exists():
                        paths["dft_traj"] = str(traj_path0.relative_to(outdir))
                    return _analyse_atoms_final(paths)

                restart_path = resume_decision.get("restart")
                # Geometry/cell restarts are manifest-bound above.  WFN state
                # is not, and CP2K otherwise discovers it implicitly by
                # project name, so clear regular files, backups and symlinks
                # before every launch regardless of fresh/restart mode.
                _clear_cp2k_cell_opt_wavefunction_restarts(dft_dir)
                if str(resume_decision.get("mode")) == "restart":
                    # Retain only the authenticated restart selected above.
                    for stale_path in (
                        dft_dir / "dft_opt.data",
                        dft_dir / "cp2k.out",
                        dft_dir / "traj.dcd",
                        dft_dir / "cp2k_scf_diagnostics.json",
                    ):
                        if stale_path.exists() or stale_path.is_symlink():
                            stale_path.unlink()
                else:
                    _clear_cp2k_cell_opt_artifacts(dft_dir)

                restart_file = (
                    None if restart_path is None else Path(restart_path).name
                )
                inp_txt = (
                    base_inp_txt
                    if restart_file is None
                    else render_cp2k_cell_opt_input(
                        atoms=atoms0,
                        cfg=cp2k_opt_runner.cfg,
                        basis_set_file_name=basis_file,
                        potential_file_name=pot_file,
                        project="dft_opt",
                        optimizer=str(getattr(dft_cfg, "optimizer", "LBFGS")),
                        max_iter=int(getattr(dft_cfg, "max_iter", 200)),
                        keep_angles=bool(getattr(dft_cfg, "keep_angles", True)),
                        external_pressure_bar=float(extP),
                        traj_every=int(getattr(dft_cfg, "traj_every", 1)),
                        traj_file="traj.dcd",
                        print_level=str(getattr(dft_cfg, "print_level", "LOW")),
                        restart_file=restart_file,
                        cp2k_version=cp2k_version,
                    )
                )
                inp_path = dft_dir / "cell_opt.inp"
                _atomic_write_cp2k_stage_input(inp_path, inp_txt)
                _write_cp2k_cell_opt_identity_manifest(
                    identity_path,
                    calculation=calculation_identity,
                    status="running",
                    artifacts={"input": inp_path},
                    restart_paths=([] if restart_path is None else [Path(restart_path)]),
                )

                # execute cp2 k
                try:
                    rr = cp2k_opt_runner.run(inp_path, dft_dir, output_name="cp2k.out")
                    if int(rr.returncode) != 0:
                        raise RuntimeError(
                            f"CP2K CELL_OPT failed (returncode={rr.returncode})"
                        )
                    assert_cp2k_cell_opt_converged(dft_dir / "cp2k.out")
                    scf_failures = count_cp2k_scf_failures(dft_dir / "cp2k.out")
                    scf_diagnostics_path = dft_dir / "cp2k_scf_diagnostics.json"
                    _write_cp2k_cell_opt_scf_diagnostics(
                        scf_diagnostics_path,
                        output_path=dft_dir / "cp2k.out",
                        failures=int(scf_failures),
                        cp2k_version=cp2k_version_text,
                        policy=cp2k_scf_policy,
                        recovered_from_existing_output=False,
                    )
                    if scf_failures:
                        warnings.warn(
                            f"CP2K {cp2k_version_text} CELL_OPT continued after "
                            f"{scf_failures} unconverged SCF cycle(s); the positive "
                            "optimization-completion marker was still required",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                    traj_path = dft_dir / "traj.dcd"
                    if not traj_path.exists():
                        raise RuntimeError(
                            "CP2K CELL_OPT did not produce trajectory file 'traj.dcd'"
                        )
                    try:
                        atoms_final = read_cp2k_dcd_last_aligned(
                            traj_path,
                            ref_atoms=atoms0,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to read CP2K trajectory: {traj_path}"
                        ) from exc
                    try:
                        sp = atoms_final.get_scaled_positions(wrap=True)
                        atoms_final.set_scaled_positions(sp)
                    except Exception:
                        pass

                    from ..structuregen import write_lammps_data

                    out_data = dft_dir / "dft_opt.data"
                    write_lammps_data(
                        out_data,
                        atoms_final,
                        atom_style=str(md_use.atom_style),
                        specorder=type_to_species,
                        units_style=str(lammps_units_style),
                    )
                    _write_cp2k_cell_opt_identity_manifest(
                        identity_path,
                        calculation=calculation_identity,
                        status="completed",
                        artifacts={
                            "input": inp_path,
                            "output": dft_dir / "cp2k.out",
                            "scf_diagnostics": scf_diagnostics_path,
                            "trajectory": traj_path,
                            "data": out_data,
                        },
                    )
                except Exception:
                    failed_artifacts = {
                        role: path
                        for role, path in {
                            "input": inp_path,
                            "output": dft_dir / "cp2k.out",
                            "trajectory": dft_dir / "traj.dcd",
                            "scf_diagnostics": dft_dir / "cp2k_scf_diagnostics.json",
                            "data": dft_dir / "dft_opt.data",
                        }.items()
                        if path.is_file()
                    }
                    _write_cp2k_cell_opt_identity_manifest(
                        identity_path,
                        calculation=calculation_identity,
                        status="failed",
                        artifacts=failed_artifacts,
                        restart_paths=sorted(dft_dir.glob("dft_opt-*.restart")),
                    )
                    raise

                return _analyse_atoms_final(
                    {
                        "dft_dir": str(dft_dir.relative_to(outdir)),
                        "dft_input": str(inp_path.relative_to(outdir)),
                        "dft_output": str((dft_dir / "cp2k.out").relative_to(outdir)),
                        "dft_scf_diagnostics": str(scf_diagnostics_path.relative_to(outdir)),
                        "dft_traj": str(traj_path.relative_to(outdir)),
                        "dft_data": str((dft_dir / "dft_opt.data").relative_to(outdir)),
                        "dft_identity": str(identity_path.relative_to(outdir)),
                    }
                )

            def _ensure_dft_results() -> None:
                """Dft results."""
                if not dft_enabled:
                    return
                assert cp2k_opt_runner is not None
                for ent in boxes:
                    if ent.get("dft_opt", {}).get("status") == "ok":
                        continue
                    current_box = int(ent.get("box", 0) or 0)
                    rejected_boxes_dft[:] = [
                        row
                        for row in rejected_boxes_dft
                        if int(row.get("box", 0) or 0) != current_box
                    ]
                    try:
                        ent["dft_opt"] = _run_dft_cell_opt_for_box(ent)
                    except Exception as e:
                        ent["dft_opt"] = {
                            "status": "failed",
                            "error": str(e),
                            "paths": {"dft_dir": str((prod_dir / f"box_{int(ent.get('box',0)):03d}" / 'dft_opt').relative_to(outdir))},
                        }
                        rejected_boxes_dft.append({"box": int(ent.get("box", 0) or 0), "reason": "cp2k_failed", "error": str(e)})

            def _dft_accepted_boxes_view() -> list[dict[str, Any]]:
                """Dft accepted boxes."""
                out: list[dict[str, Any]] = []
                for ent in boxes:
                    d = ent.get("dft_opt", {})
                    if str(d.get("status", "")) != "ok":
                        continue
                    if bool(d.get("reject", False)):
                        continue
                    if exclude_defects and bool(d.get("has_coordination_defects", False)):
                        continue
                    out.append({"density": float(d["density"]), "metrics": d["metrics"], "distributions": d["distributions"]})
                return out

            boxes: list[dict[str, Any]] = []
            rejected_boxes: list[dict[str, Any]] = []
            conv_spec: dict[str, Any] | None = None

            # resume production regenerating
            next_box_id = 1
            recoverable_uncommitted_box_id: Optional[int] = None
            if resume_state is not None:
                prev_boxes = resume_state.get("boxes", [])
                prev_rejected = resume_state.get("rejected_boxes", [])
                if isinstance(prev_boxes, list):
                    boxes = deepcopy(prev_boxes)
                if isinstance(prev_rejected, list):
                    rejected_boxes = deepcopy(prev_rejected)

                def _box_id(ent: dict[str, Any]) -> int:
                    try:
                        return int(ent.get("box", 0) or 0)
                    except Exception:
                        return 0

                boxes.sort(key=_box_id)
                rejected_boxes.sort(key=_box_id)

                rs_spec = resume_state.get("convergence_spec", None)
                if isinstance(rs_spec, dict) and len(rs_spec) > 0:
                    conv_spec = deepcopy(rs_spec)

                # coordination exclusion retroactively
                if exclude_defects and len(boxes) > 0:
                    kept: list[dict[str, Any]] = []
                    for ent in boxes:
                        if bool(ent.get("has_coordination_defects", False)):
                            ent_rej = deepcopy(ent)
                            ent_rej["reason"] = "coordination_defects"
                            rejected_boxes.append(ent_rej)
                        else:
                            kept.append(ent)
                    boxes = kept

                ids: list[int] = []
                for ent in list(boxes) + list(rejected_boxes):
                    try:
                        ids.append(int(ent.get("box", 0) or 0))
                    except Exception:
                        pass
                if len(ids) > 0:
                    next_box_id = max(ids) + 1

                # The next box can contain a fully completed engine pipeline
                # when only post-processing/plotting failed before the box was
                # checkpointed.  Preserve exactly that candidate for strict
                # stage-artifact recovery; quarantine any other uncommitted
                # box tree because it cannot belong to the deterministic next
                # seed position.
                candidate_dir = prod_dir / f"box_{int(next_box_id):03d}"
                if candidate_dir.is_dir() and not candidate_dir.is_symlink():
                    recoverable_uncommitted_box_id = int(next_box_id)

                quarantine_uncommitted_box_directories(
                    prod_dir,
                    committed_box_ids=(
                        ids
                        + (
                            [int(recoverable_uncommitted_box_id)]
                            if recoverable_uncommitted_box_id is not None
                            else []
                        )
                    ),
                    quarantine_root=(
                        Path(outdir) / "interrupted_attempts" / "production"
                    ).resolve(strict=False),
                )

                # seed
                prev_entries = []
                if isinstance(prev_boxes, list):
                    prev_entries.extend(prev_boxes)
                if isinstance(prev_rejected, list):
                    prev_entries.extend(prev_rejected)
                consumed_draws = _count_production_resume_seed_draws(prev_entries)
                for _ in range(int(consumed_draws)):
                    rng_prod.randrange(1, 2**31 - 1)

            if dft_enabled:
                # Normalize restored derived audit rows before any resumed
                # checkpoint can persist them again.  The terminal pass below
                # repeats the same idempotent operation after newly refined
                # boxes have been added.
                rejected_boxes_dft[:] = _reconcile_dft_coordination_rejections(
                    rejected_boxes_dft,
                    boxes,
                    exclude_defects=exclude_defects,
                )

            # resuming reconstruct convergence
            # accepted box
            if conv_spec is None and len(boxes) > 0:
                b0 = boxes[0]
                if not isinstance(b0.get("distributions", None), dict) or not isinstance(b0.get("metrics", None), dict):
                    raise RuntimeError("Cannot resume: production boxes lack stored metrics/distributions")
                conv_spec = build_production_convergence_spec(b0, metrics_cfg)

            conv_report_md: dict[str, Any] = {}
            conv_report_dft: dict[str, Any] = {}
            conv_spec_dft: dict[str, Any] | None = None
            converged_md = False
            converged_dft: bool | None = None
            converged = False

            required_streak = max(1, int(getattr(prod_cfg, "consecutive_converged_checks", 1)))
            converged_streak = 0
            last_convergence_evaluated_n_boxes_total: int | None = None
            last_convergence_evaluated_n_boxes_accepted: int | None = None
            if resume_state is not None:
                stored_required = int(
                    resume_state.get("required_convergence_streak", required_streak)
                )
                if stored_required != required_streak:
                    raise RuntimeError(
                        "Cannot safely resume: required convergence streak changed"
                    )
                converged_streak = int(resume_state.get("convergence_streak", 0))
                last_total = resume_state.get(
                    "last_convergence_evaluated_n_boxes_total"
                )
                last_accepted = resume_state.get(
                    "last_convergence_evaluated_n_boxes_accepted"
                )
                last_convergence_evaluated_n_boxes_total = (
                    None if last_total is None else int(last_total)
                )
                last_convergence_evaluated_n_boxes_accepted = (
                    None if last_accepted is None else int(last_accepted)
                )
                if isinstance(resume_state.get("convergence_md"), Mapping):
                    conv_report_md = deepcopy(resume_state.get("convergence_md", {}))
                converged_md = bool(resume_state.get("converged_md", False))
                if isinstance(resume_state.get("convergence_dft"), Mapping):
                    conv_report_dft = deepcopy(resume_state.get("convergence_dft", {}))
                if isinstance(resume_state.get("convergence_spec_dft"), Mapping):
                    conv_spec_dft = deepcopy(resume_state.get("convergence_spec_dft", {}))
                if resume_state.get("converged_dft") is not None:
                    converged_dft = bool(resume_state.get("converged_dft"))

            dft_summary: dict[str, Any] | None = None
            boxes_dft_final: list[int] | None = None
            integrity_file_cache: dict[str, tuple[int, int, int, int, int, str]] = {}

            def _production_state(
                *,
                status: str,
                error: Optional[str] = None,
                graph_output_paths: Optional[Mapping[str, Any]] = None,
            ) -> dict[str, Any]:
                final_conv_spec = (
                    conv_spec_dft
                    if dft_enabled and converged_dft is not None and conv_spec_dft is not None
                    else conv_spec
                )
                metrics_checked = metrics_checked_from_conv_spec(final_conv_spec)
                final_conv_report: dict[str, Any] = conv_report_md
                if dft_enabled and converged_dft is not None:
                    final_conv_report = conv_report_dft
                motif_summary = summarize_production_crystal_motifs(boxes, rejected_boxes=rejected_boxes)
                graph_paths = dict(graph_output_paths or {})
                terminal = str(status) in {"ok", "incomplete", "not_converged"}
                resumable = not (terminal and do_converge and not store_distributions)
                report_boxes = boxes
                report_rejected_boxes = rejected_boxes
                if terminal and not store_distributions:
                    report_boxes = deepcopy(boxes)
                    report_rejected_boxes = deepcopy(rejected_boxes)
                    for entry in report_boxes:
                        entry.pop("distributions", None)
                        dft_entry = entry.get("dft_opt")
                        if isinstance(dft_entry, dict):
                            dft_entry.pop("distributions", None)
                    for entry in report_rejected_boxes:
                        entry.pop("distributions", None)
                if not do_converge:
                    convergence_inference_status = (
                        "fixed_n_terminal_posthoc_not_sequentially_valid"
                        if bool(final_conv_report.get("assessment_performed", False))
                        else "fixed_count_unassessed"
                    )
                elif bool(converged):
                    convergence_inference_status = (
                        "criterion_met_repeated_looks_not_sequentially_valid"
                    )
                elif final_conv_report:
                    convergence_inference_status = (
                        "criterion_not_met_or_unassessed_repeated_looks_not_sequentially_valid"
                    )
                else:
                    convergence_inference_status = "not_yet_assessed"
                convergence_degree = (
                    dict(final_conv_report.get("convergence_degree", {}) or {})
                    if isinstance(final_conv_report, Mapping)
                    else {}
                )
                criterion_coverage = {
                    key: dict(convergence_degree.get(key, {}) or {})
                    for key in ("ci", "stability", "criteria_integrity", "overall")
                    if isinstance(convergence_degree.get(key), Mapping)
                }
                state = {
                    "enabled": True,
                    "status": str(status),
                    "execution_status": (
                        "completed" if str(status) in {"ok", "incomplete", "not_converged"}
                        else str(status)
                    ),
                    "error": (str(error) if error is not None else None),
                    "converged": (bool(converged) if do_converge else None),
                    "convergence_status": (
                        "converged"
                        if do_converge and bool(converged)
                        else ("not_converged" if do_converge else "fixed_count_unassessed")
                    ),
                    # Additive inference-qualified status; retain the legacy
                    # convergence_status token for existing API consumers.
                    "convergence_inference_status": str(
                        convergence_inference_status
                    ),
                    "achieved_convergence_degree": (
                        dict(final_conv_report.get("achieved_convergence_degree", {}) or {})
                        if isinstance(final_conv_report, Mapping)
                        and isinstance(
                            final_conv_report.get("achieved_convergence_degree"),
                            Mapping,
                        )
                        else None
                    ),
                    "posthoc_convergence_criterion_met": (
                        final_conv_report.get("posthoc_criterion_met")
                        if not do_converge and isinstance(final_conv_report, Mapping)
                        else None
                    ),
                    "posthoc_convergence_failed_items": (
                        list(final_conv_report.get("posthoc_failed_items", []) or [])
                        if not do_converge and isinstance(final_conv_report, Mapping)
                        else None
                    ),
                    "convergence_criterion_coverage": (
                        criterion_coverage if criterion_coverage else None
                    ),
                    "n_boxes": int(len(boxes)),
                    "n_boxes_accepted": int(len(boxes)),
                    "n_boxes_rejected": int(len(rejected_boxes)),
                    "n_boxes_total": int(len(boxes) + len(rejected_boxes)),
                    "min_boxes": int(minimum_accepted),
                    "max_boxes": int(max_boxes) if max_boxes is not None else None,
                    "batch_boxes": int(batch),
                    "check_convergence": bool(do_converge),
                    "resumable": bool(resumable),
                    "non_resumable_reason": (
                        "adaptive convergence distributions were omitted from the terminal result"
                        if not resumable
                        else None
                    ),
                    "convergence_streak": int(converged_streak),
                    "required_convergence_streak": int(required_streak),
                    "last_convergence_evaluated_n_boxes_total": (
                        None
                        if last_convergence_evaluated_n_boxes_total is None
                        else int(last_convergence_evaluated_n_boxes_total)
                    ),
                    "last_convergence_evaluated_n_boxes_accepted": (
                        None
                        if last_convergence_evaluated_n_boxes_accepted is None
                        else int(last_convergence_evaluated_n_boxes_accepted)
                    ),
                    "dump_trajectory": bool(dump_traj),
                    "dump_every_steps": int(dump_every),
                    "rate_K_per_time": float(prod_rate),
                    "rate_K_per_ps": float(cooling_rate_ps) if cooling_rate_ps is not None else None,
                    "replicate": [int(nx), int(ny), int(nz)],
                    "structure_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
                    "exclude_coordination_defects": bool(exclude_defects),
                    "rejects_subdir": str(rejects_subdir) if bool(exclude_defects) else None,
                    "rejects_dir": str(rejects_dir.relative_to(outdir)) if bool(exclude_defects) and rejects_dir.exists() else None,
                    "warmup_start_temperature": float(warmup_start_temperature),
                    "warmup_duration_ps": float(warmup_duration_ps),
                    "warmup_steps": int(warmup_steps),
                    "T_high": float(T_high),
                    "t_final": float(q_cfg.t_final),
                    "quench_steps": int(n_quench_prod),
                    "highT_steps": int(high_total_steps),
                    "relax_steps": int(relax_steps_prod),
                    "cutoffs": ([{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted((prod_cutoffs or {}).items())] if isinstance(prod_cutoffs, dict) else None),
                    "metrics_checked": metrics_checked,
                    "convergence_spec": final_conv_spec,
                    "convergence_spec_md": conv_spec,
                    "convergence_spec_dft": conv_spec_dft,
                    "converged_md": (bool(converged_md) if do_converge else None),
                    "convergence_md": conv_report_md,
                    "converged_dft": (bool(converged_dft) if dft_enabled and converged_dft is not None else None),
                    "convergence_dft": (conv_report_dft if dft_enabled and converged_dft is not None else None),
                    "convergence": final_conv_report,
                    "crystal_motifs": motif_summary,
                    "dft_opt": dft_summary,
                    "boxes_dft_final": boxes_dft_final,
                    "n_boxes_dft_accepted": (int(len(boxes_dft_final)) if isinstance(boxes_dft_final, list) else None),
                    "rejected_boxes_dft": rejected_boxes_dft if dft_enabled else None,
                    "boxes": report_boxes,
                    "rejected_boxes": report_rejected_boxes,
                    "graph_outputs": graph_paths,
                    "paths": dict(graph_paths),
                    "ensemble_dir": str(prod_dir.relative_to(outdir)),
                }
                return _attach_production_state_integrity(
                    state,
                    outdir=outdir,
                    identity_cache=integrity_file_cache,
                    force_rehash=terminal,
                )

            def _checkpoint(*, status: str, error: Optional[str] = None) -> None:
                if checkpoint_cb is None:
                    return
                checkpoint_cb(_production_state(status=status, error=error))

            _checkpoint(status="starting")
            while True:
                while len(boxes) < target:
                    total_boxes = len(boxes) + len(rejected_boxes)
                    if max_boxes is not None and total_boxes >= int(max_boxes):
                        break
                    if max_boxes is None and total_boxes >= int(HARD_MAX_BOXES):
                        raise RuntimeError(
                            f"Production ensemble failed to converge after {HARD_MAX_BOXES} boxes. "
                            "This indicates tolerances are likely too strict or metrics are ill-posed."
                        )
                    b = int(next_box_id)
                    next_box_id += 1
                    bdir = prod_dir / f"box_{b:03d}"
                    ensure_dir(bdir)
                    progress.info("production", f"box {b}: starting ({len(boxes)+len(rejected_boxes)+1} total attempted)")

                    # distinct warm followed
                    # independent melt preserves
                    # high duration physics
                    seed_warmup = int(rng_prod.randrange(1, 2**31 - 1))
                    warmup_stage = StageSpec(
                        name="warmup",
                        input_data=size_base_data,
                        output_data=bdir / "warmup.data",
                        temperature_start=float(warmup_start_temperature),
                        temperature_stop=float(T_high),
                        pressure=float(md_pressure),
                        equil_steps=0,
                        run_steps=int(warmup_steps),
                        seed=seed_warmup,
                        velocity_mode="create",
                        force_isotropic=melt_force_iso,
                        replicate=(nx, ny, nz),
                        write_dump=bool(need_stage_dump["melt"]),
                        dump_every=int(dump_every) if need_stage_dump["melt"] else None,
                        msd_every=int(tm_cfg.msd_every),
                    )

                    seed_melt = int(rng_prod.randrange(1, 2**31 - 1))
                    melt_stage = StageSpec(
                        name="melt",
                        input_data=bdir / "warmup.data",
                        output_data=bdir / "melt.data",
                        temperature_start=float(T_high),
                        temperature_stop=float(T_high),
                        pressure=float(md_pressure),
                        equil_steps=0,
                        run_steps=int(high_total_steps),
                        seed=seed_melt,
                        velocity_mode="preserve",
                        force_isotropic=False,
                        replicate=None,
                        write_dump=bool(need_stage_dump["melt"]),
                        dump_every=int(dump_every) if need_stage_dump["melt"] else None,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    recovered_stage_execution = False
                    prepared_seed_quench: Optional[int] = None
                    prepared_seed_relax: Optional[int] = None
                    if int(b) == int(recoverable_uncommitted_box_id or -1):
                        prepared_seed_quench = int(rng_prod.randrange(1, 2**31 - 1))
                        prepared_seed_relax = int(rng_prod.randrange(1, 2**31 - 1))
                        recovery_quench_stage = StageSpec(
                            name="quench",
                            input_data=bdir / "melt.data",
                            output_data=bdir / "quench.data",
                            temperature_start=T_high,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(n_quench_prod),
                            seed=int(prepared_seed_quench),
                            velocity_mode=vel_next,
                            replicate=None,
                            write_dump=bool(need_stage_dump["quench"]),
                            dump_every=(
                                int(quench_dump_every)
                                if need_stage_dump["quench"]
                                else None
                            ),
                            msd_every=int(tm_cfg.msd_every),
                        )
                        recovery_relax_stage = StageSpec(
                            name="relax",
                            input_data=bdir / "quench.data",
                            output_data=bdir / "relax.data",
                            temperature_start=q_cfg.t_final,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(relax_steps_prod),
                            seed=int(prepared_seed_relax),
                            velocity_mode=vel_next,
                            replicate=None,
                            write_dump=bool(relax_dump_settings["write_dump"]),
                            dump_every=relax_dump_settings["dump_every"],
                            tail_dump_frames=relax_dump_settings["tail_dump_frames"],
                            tail_dump_stride=relax_dump_settings["tail_dump_stride"],
                            msd_every=int(tm_cfg.msd_every),
                        )
                        recovery_stages = (
                            warmup_stage,
                            melt_stage,
                            recovery_quench_stage,
                            recovery_relax_stage,
                        )
                        recovery_dirs = (
                            bdir / "warmup",
                            bdir / "melt",
                            bdir / "quench",
                            bdir / "relax",
                        )
                        recovery_engine = (
                            "cp2k" if isinstance(runner, Cp2kRunner) else "lammps"
                        )
                        recovery_units = str(
                            getattr(pot_cfg, "user_units", "metal") or "metal"
                        )
                        try:
                            recovered_outcomes = [
                                recover_completed_stage_outcome(
                                    stage_directory,
                                    md_cfg=md_use,
                                    stage=stage_spec,
                                    expected_engine=recovery_engine,
                                    lammps_units_style=recovery_units,
                                )
                                for stage_directory, stage_spec in zip(
                                    recovery_dirs,
                                    recovery_stages,
                                )
                            ]
                        except Exception as exc:
                            progress.warn(
                                "production",
                                "box "
                                f"{b}: completed-stage recovery was not valid; "
                                f"quarantining it before deterministic rerun ({exc})",
                            )
                            committed_ids = [
                                int(entry.get("box", 0) or 0)
                                for entry in list(boxes) + list(rejected_boxes)
                            ]
                            quarantine_uncommitted_box_directories(
                                prod_dir,
                                committed_box_ids=committed_ids,
                                quarantine_root=(
                                    Path(outdir)
                                    / "interrupted_attempts"
                                    / "production"
                                ).resolve(strict=False),
                            )
                            ensure_dir(bdir)
                        else:
                            (
                                warmup_out,
                                melt_out,
                                quench_out,
                                relax_out,
                            ) = recovered_outcomes
                            # These are the exact third and fourth draws from
                            # the deterministic production RNG stream.  Keep
                            # the public box provenance identical to an
                            # uninterrupted execution; the engine is not
                            # relaunched merely to reconstruct seed metadata.
                            seed_quench = int(prepared_seed_quench)
                            seed_relax = int(prepared_seed_relax)
                            quench_stage = recovery_quench_stage
                            relax_stage = recovery_relax_stage
                            recovered_stage_execution = True
                            progress.info(
                                "production",
                                f"box {b}: validated completed stage artifacts; retrying analysis only",
                            )
                        recoverable_uncommitted_box_id = None

                    if recovered_stage_execution:
                        relax_dir = bdir / "relax"
                    elif cont == "continuous" and isinstance(runner, LammpsRunner):
                        progress.info("production", f"box {b}: continuous warmup→melt→quench→relax")
                        # lammps warmup quench
                        # directories populated analysis
                        seed_quench = (
                            int(prepared_seed_quench)
                            if prepared_seed_quench is not None
                            else int(rng_prod.randrange(1, 2**31 - 1))
                        )
                        quench_stage = StageSpec(
                            name="quench",
                            input_data=bdir / "melt.data",  # ignored continuous rendering
                            output_data=bdir / "quench.data",
                            temperature_start=T_high,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(n_quench_prod),
                            seed=seed_quench,
                            velocity_mode="preserve",
                            replicate=None,
                            write_dump=bool(need_stage_dump["quench"]),
                            dump_every=int(quench_dump_every) if need_stage_dump["quench"] else None,
                            msd_every=int(tm_cfg.msd_every),
                        )

                        seed_relax = (
                            int(prepared_seed_relax)
                            if prepared_seed_relax is not None
                            else int(rng_prod.randrange(1, 2**31 - 1))
                        )
                        relax_stage = StageSpec(
                            name="relax",
                            input_data=bdir / "quench.data",  # ignored continuous rendering
                            output_data=bdir / "relax.data",
                            temperature_start=q_cfg.t_final,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(relax_steps_prod),
                            seed=seed_relax,
                            velocity_mode="preserve",
                            replicate=None,
                            write_dump=bool(relax_dump_settings["write_dump"]),
                            dump_every=relax_dump_settings["dump_every"],
                            tail_dump_frames=relax_dump_settings["tail_dump_frames"],
                            tail_dump_stride=relax_dump_settings["tail_dump_stride"],
                            msd_every=int(tm_cfg.msd_every),
                        )
                        relax_dir = bdir / "relax"

                        arts = run_stages_continuous_lammps(
                            runner,
                            pot_cfg,
                            md_use,
                            [warmup_stage, melt_stage, quench_stage, relax_stage],
                            [bdir / "warmup", bdir / "melt", bdir / "quench", relax_dir],
                            bdir / "continuous",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )
                        outcome_kwargs = {
                            "lammps_units_style": str(getattr(pot_cfg, "user_units", "metal") or "metal")
                        }
                        warmup_out = stage_outcome_from_artifacts(arts[0], md_cfg=md_use, stage=warmup_stage, **outcome_kwargs)
                        melt_out = stage_outcome_from_artifacts(arts[1], md_cfg=md_use, stage=melt_stage, **outcome_kwargs)
                        quench_out = stage_outcome_from_artifacts(arts[2], md_cfg=md_use, stage=quench_stage, **outcome_kwargs)
                        relax_out = stage_outcome_from_artifacts(arts[3], md_cfg=md_use, stage=relax_stage, **outcome_kwargs)
                    else:
                        progress.info("production", f"box {b}: warmup")
                        warmup_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            warmup_stage,
                            bdir / "warmup",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                        melt_stage = StageSpec(
                            name="melt",
                            input_data=(bdir / "warmup" / warmup_out.output_data),
                            output_data=bdir / "melt.data",
                            temperature_start=float(T_high),
                            temperature_stop=float(T_high),
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(high_total_steps),
                            seed=seed_melt,
                            velocity_mode="preserve",
                            force_isotropic=False,
                            replicate=None,
                            write_dump=bool(need_stage_dump["melt"]),
                            dump_every=int(dump_every) if need_stage_dump["melt"] else None,
                            msd_every=int(tm_cfg.msd_every),
                        )

                        progress.info("production", f"box {b}: melt")
                        melt_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            melt_stage,
                            bdir / "melt",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                        # quench selected rate
                        progress.info("production", f"box {b}: quench")
                        seed_quench = (
                            int(prepared_seed_quench)
                            if prepared_seed_quench is not None
                            else int(rng_prod.randrange(1, 2**31 - 1))
                        )
                        quench_stage = StageSpec(
                            name="quench",
                            input_data=(bdir / "melt" / melt_out.output_data),
                            output_data=bdir / "quench.data",
                            temperature_start=T_high,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(n_quench_prod),
                            seed=seed_quench,
                            velocity_mode=vel_next,
                            replicate=None,
                            write_dump=bool(need_stage_dump["quench"]),
                            dump_every=int(quench_dump_every) if need_stage_dump["quench"] else None,
                            msd_every=int(tm_cfg.msd_every),
                        )
                        quench_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            quench_stage,
                            bdir / "quench",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                        # relax dump metrics
                        progress.info("production", f"box {b}: relax")
                        seed_relax = (
                            int(prepared_seed_relax)
                            if prepared_seed_relax is not None
                            else int(rng_prod.randrange(1, 2**31 - 1))
                        )
                        relax_stage = StageSpec(
                            name="relax",
                            input_data=(bdir / "quench" / quench_out.output_data),
                            output_data=bdir / "relax.data",
                            temperature_start=q_cfg.t_final,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(relax_steps_prod),
                            seed=seed_relax,
                            velocity_mode=vel_next,
                            replicate=None,
                            write_dump=bool(relax_dump_settings["write_dump"]),
                            dump_every=relax_dump_settings["dump_every"],
                            tail_dump_frames=relax_dump_settings["tail_dump_frames"],
                            tail_dump_stride=relax_dump_settings["tail_dump_stride"],
                            msd_every=int(tm_cfg.msd_every),
                        )
                        relax_dir = bdir / "relax"
                        relax_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            relax_stage,
                            relax_dir,
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                    progress.info("production", f"box {b}: stage execution complete")
                    melt_stage_dir = bdir / "melt"
                    melt_elastic = _maybe_elastic(
                        "melt",
                        stage_dir=melt_stage_dir,
                        structure_data=Path(melt_out.output_data) if isinstance(melt_out.output_data, Path) and Path(melt_out.output_data).is_absolute() else (melt_stage_dir / melt_out.output_data),
                        input_data=(bdir / "warmup" / warmup_out.output_data),
                        force_iso_context=bool(melt_force_iso),
                    )
                    quench_stage_dir = bdir / "quench"
                    relax_elastic = _maybe_elastic(
                        "relax",
                        stage_dir=relax_dir,
                        structure_data=(relax_dir / relax_out.output_data),
                        input_data=(quench_stage_dir / quench_out.output_data),
                        force_iso_context=bool(melt_force_iso),
                    )
                    elastic_timeseries = {
                        "melt": _maybe_elastic_series(
                            "melt",
                            stage_dir=melt_stage_dir,
                            stage_output_data=(
                                Path(melt_out.output_data)
                                if isinstance(melt_out.output_data, Path) and Path(melt_out.output_data).is_absolute()
                                else (melt_stage_dir / melt_out.output_data)
                            ),
                            force_iso_context=bool(melt_force_iso),
                        ),
                        "quench": _maybe_elastic_series(
                            "quench",
                            stage_dir=quench_stage_dir,
                            stage_output_data=(quench_stage_dir / quench_out.output_data),
                            force_iso_context=bool(melt_force_iso),
                            sampling_hint=sampling_hint,
                        ),
                        "relax": _maybe_elastic_series(
                            "relax",
                            stage_dir=relax_dir,
                            stage_output_data=(relax_dir / relax_out.output_data),
                            force_iso_context=bool(melt_force_iso),
                        ),
                    }

                    dump_path = relax_dir / f"{relax_stage.name}.lammpstrj"
                    cand = relax_dir / "traj.extxyz"
                    traj_path = cand if cand.exists() else dump_path
                    entry, prod_cutoffs = analyse_production_box(
                        box_id=int(b),
                        outdir=outdir,
                        melt_stage_dir=melt_stage_dir,
                        quench_stage_dir=quench_stage_dir,
                        relax_stage_dir=relax_dir,
                        relax_data_path=(relax_dir / relax_out.output_data),
                        density_mean=float(relax_out.density_mean),
                        density_stderr=float(relax_out.density_stderr),
                        metrics_cfg=metrics_cfg,
                        cutoffs=prod_cutoffs,
                        required_pairs=required_pairs,
                        fixed_cutoffs=fixed_cut,
                        type_to_species=type_to_species,
                        md_timestep=float(md_use.timestep),
                        quench_window_steps_range=quench_window_steps_range,
                        sampling_hint=sampling_hint,
                        bondlen_cdf_points=int(bondlen_cdf_points),
                        angle_cdf_points=int(angle_cdf_points),
                        seeds={"warmup": int(seed_warmup), "melt": int(seed_melt), "quench": int(seed_quench), "relax": int(seed_relax)},
                        melt_elastic=melt_elastic,
                        relax_elastic=relax_elastic,
                        elastic_timeseries=elastic_timeseries,
                        exclude_coordination_defects=bool(exclude_defects),
                        rejects_dir=(rejects_dir if bool(exclude_defects) else None),
                        relax_dump_path=dump_path,
                        relax_traj_path=traj_path,
                        embed_structures=bool(getattr(prod_cfg, "embed_structures", True)),
                        lammps_units_style=analysis_lammps_units_style,
                        engine=("cp2k" if isinstance(runner, Cp2kRunner) else "lammps"),
                    )
                    if recovered_stage_execution:
                        entry["resume_recovery"] = {
                            "mode": "postprocessing_only",
                            "engine_stages_reused": [
                                "warmup",
                                "melt",
                                "quench",
                                "relax",
                            ],
                            "reason": (
                                "engine execution completed before the prior "
                                "analysis/plotting failure"
                            ),
                        }

                    if conv_spec is None:
                        conv_spec = build_production_convergence_spec(entry, metrics_cfg)
                    else:
                        validate_production_entry_against_spec(entry, conv_spec, box_label=b)

                    if bool(entry.get("reject")):
                        rejected_boxes.append(entry)
                        rej = dict(entry.get("reject", {}) or {})
                        reason = str(rej.get("reason", "rejected"))
                        progress.warn("production", f"box {b}: rejected ({reason})")
                    else:
                        boxes.append(entry)
                        progress.info("production", f"box {b}: accepted ({len(boxes)} accepted total)")
                    _checkpoint(status="running")

                if not do_converge:
                    # Fixed-count execution is complete only when the requested
                    # number of accepted boxes exists. Reaching a cap with only
                    # rejected boxes is execution-complete but scientifically
                    # incomplete, not "converged".
                    fixed_count_complete = len(boxes) >= target
                    # Meeting a requested count completes execution; it is not
                    # a statistical convergence result.  Keep the adaptive
                    # convergence flag false internally for final-status logic
                    # and expose the unassessed state explicitly below.
                    converged = False
                    converged_md = False
                    conv_report_md = assess_fixed_count_convergence_posthoc(
                        boxes,
                        conv_spec,
                        conv_cfg,
                        execution_target_met=bool(fixed_count_complete),
                        min_boxes=int(minimum_accepted),
                    )
                    if not fixed_count_complete:
                        conv_report_md["error"] = (
                            f"accepted {len(boxes)} boxes, below min_boxes={minimum_accepted}"
                        )
                    progress.convergence("production-posthoc", conv_report_md)
                    break

                current_total = len(boxes) + len(rejected_boxes)
                current_accepted = len(boxes)
                already_evaluated = (
                    last_convergence_evaluated_n_boxes_total == current_total
                    and last_convergence_evaluated_n_boxes_accepted == current_accepted
                )
                if not already_evaluated:
                    previous_accepted = last_convergence_evaluated_n_boxes_accepted
                    accepted_ensemble_changed = (
                        previous_accepted is None
                        or current_accepted != int(previous_accepted)
                    )
                    new_accepted_evidence = (
                        previous_accepted is None
                        or current_accepted > int(previous_accepted)
                    )
                    # Rejected-only attempts change bookkeeping, not the
                    # accepted ensemble.  Re-evaluating that identical sample
                    # and advancing a "consecutive" streak would manufacture a
                    # second passing look without any new statistical evidence.
                    if accepted_ensemble_changed:
                        if conv_spec is None:
                            converged_md = False
                            conv_report_md = {
                                "error": "no distributions available for convergence"
                            }
                        elif len(boxes) < 1:
                            converged_md = False
                            conv_report_md = {
                                "error": "no accepted boxes (all rejected)"
                            }
                        else:
                            converged_md, conv_report_md = _check_convergence(boxes, conv_spec)

                        if new_accepted_evidence and converged_md:
                            converged_streak += 1
                        else:
                            # A reduced/otherwise changed accepted ensemble is
                            # a new criterion history, so an old streak cannot
                            # be carried into it.
                            converged_streak = 0
                    last_convergence_evaluated_n_boxes_total = current_total
                    last_convergence_evaluated_n_boxes_accepted = current_accepted
                    progress.convergence("production", conv_report_md)
                    progress.info(
                        "production",
                        f"MD convergence streak: {converged_streak}/{required_streak}",
                    )
                    _checkpoint(status="running")


                # refinement disabled ensemble
                # remained converged requested
                if (not dft_enabled) and converged_md and converged_streak >= required_streak:
                    converged = True
                    break

                # refinement enabled refined
                # ensemble converged expensive
                # ensemble requested convergence
                if dft_enabled and converged_md and converged_streak >= required_streak:
                    # box dft optimisation
                    _ensure_dft_results()
                    dft_view = _dft_accepted_boxes_view()
                    if len(dft_view) < 1:
                        converged_dft = False
                        conv_report_dft = {"error": "no DFT-accepted boxes (all failed or rejected)"}
                        converged = False
                    else:
                        conv_spec_dft = build_production_convergence_spec(
                            dft_view[0], metrics_cfg
                        )
                        for dft_box in dft_view[1:]:
                            validate_production_entry_against_spec(
                                dft_box,
                                conv_spec_dft,
                                box_label=f"DFT {dft_box.get('box', '?')}",
                            )
                        converged_dft, conv_report_dft = _check_convergence(
                            dft_view, conv_spec_dft
                        )
                        converged = bool(converged_dft)

                    if converged:
                        break

                    # loop
                    # add structures
                    converged_streak = 0
                    _checkpoint(status="running")
                if max_boxes is not None and (len(boxes) + len(rejected_boxes)) >= int(max_boxes):
                    break
                if max_boxes is None and (len(boxes) + len(rejected_boxes)) >= int(HARD_MAX_BOXES):
                    raise RuntimeError(
                        f"Production ensemble failed to converge after {HARD_MAX_BOXES} boxes. "
                        "Relax convergence tolerances or set production.max_boxes to impose a cap."
                    )

                if max_boxes is None:
                    target = len(boxes) + batch
                else:
                    target = min(int(max_boxes), len(boxes) + batch)

            metrics_checked = metrics_checked_from_conv_spec(conv_spec)

            # summarise refinement outcomes
            dft_summary: dict[str, Any] | None = None
            boxes_dft_final: list[int] | None = None
            if dft_enabled:
                rejected_boxes_dft[:] = _reconcile_dft_coordination_rejections(
                    rejected_boxes_dft,
                    boxes,
                    exclude_defects=exclude_defects,
                )
                n_ok = 0
                n_failed = 0
                n_defects = 0
                n_not_run = 0
                final_ids: list[int] = []
                for ent in boxes:
                    d = ent.get("dft_opt", {}) if isinstance(ent.get("dft_opt"), dict) else {}
                    st = str(d.get("status", ""))
                    if st == "ok":
                        n_ok += 1
                        if exclude_defects and bool(d.get("has_coordination_defects", False)):
                            n_defects += 1
                        else:
                            final_ids.append(int(ent.get("box", 0) or 0))
                    elif st == "failed":
                        n_failed += 1
                    else:
                        n_not_run += 1

                boxes_dft_final = sorted(final_ids)
                extP_used = (
                    float(getattr(dft_cfg, "external_pressure_bar", 0.0))
                    if getattr(dft_cfg, "external_pressure_bar", None) is not None
                    else float(_pressure_to_bar(float(md_pressure)))
                )
                dft_summary = {
                    "enabled": True,
                    "optimizer": str(getattr(dft_cfg, "optimizer", "LBFGS")),
                    "max_iter": int(getattr(dft_cfg, "max_iter", 200)),
                    "keep_angles": True,
                    "external_pressure_bar": float(extP_used),
                    "traj_every": int(getattr(dft_cfg, "traj_every", 1)),
                    "print_level": str(getattr(dft_cfg, "print_level", "LOW")),
                    "n_boxes_ok": int(n_ok),
                    "n_boxes_failed": int(n_failed),
                    "n_boxes_rejected_coordination_defects": int(n_defects),
                    "n_boxes_not_run": int(n_not_run),
                    "n_boxes_accepted": int(len(final_ids)),
                }

            terminal_graph_outputs: dict[str, Any] = {}
            if graph_analysis_requested(metrics_cfg):
                terminal_graph_outputs = dict(
                    write_graph_analysis_outputs(
                        outdir,
                        boxes=boxes,
                        rejected_boxes=rejected_boxes,
                        metrics=metrics_cfg,
                        type_to_species=type_to_species,
                        legacy_cutoffs=prod_cutoffs,
                    )
                )
            accepted_for_status = (
                len(boxes_dft_final)
                if dft_enabled and converged_dft is not None and isinstance(boxes_dft_final, list)
                else len(boxes)
            )
            final_status, final_error = _production_final_status(
                n_accepted=accepted_for_status,
                min_boxes=minimum_accepted,
                check_convergence=do_converge,
                converged=converged,
                max_boxes=max_boxes,
                n_total=len(boxes) + len(rejected_boxes),
            )
            production = _production_state(
                status=final_status,
                error=final_error,
                graph_output_paths=terminal_graph_outputs,
            )

        if not bool(production.get("enabled", False)):
            production = _attach_production_state_integrity(
                {
                    "enabled": False,
                    "status": "not_requested",
                    "execution_status": "not_requested",
                    "error": None,
                    "converged": False,
                    "n_boxes": 0,
                    "n_boxes_accepted": 0,
                    "n_boxes_rejected": 0,
                    "n_boxes_total": 0,
                    "boxes": [],
                    "rejected_boxes": [],
                },
                outdir=outdir,
            )
        return production

def _run_production_ensemble(
    *,
    config: RunConfig,
    outdir: Path,
    runner: Union[LammpsRunner, Cp2kRunner],
    pot_cfg: KimConfig,
    md_use: MDConfig,
    potential_lines: Optional[list[str]],
    type_to_species: Optional[list[str]],
    metrics_cfg,
    tm_cfg,
    q_cfg,
    size_base_data: Path,
    chosen_replicate: list[int],
    chosen_rate: float,
    dt_ref: float,
    dt_mq: float,
    cooling_rate_ps: Optional[float],
    cutoffs_rate: dict[tuple[int, int], float],
    cutoffs_size: dict[tuple[int, int], float],
    T_high: float,
    high_total_steps: int,
    resume_state: Optional[dict[str, Any]] = None,
    sampling_hint: Optional[dict[str, float]] = None,
    progress: Optional[CondensedProgressLog] = None,
    checkpoint_cb=None,
    pressure_override: Optional[float] = None,
    seed_base: Optional[int] = None,
    time_unit_ps_override: Optional[float] = None,
    prod_cfg_override=None,
    conv_cfg_override=None,
    quench_steps_override: Optional[int] = None,
    relax_steps_override: Optional[int] = None,
) -> dict[str, Any]:
    """Production ensemble."""
    return _ProductionEnsembleRunner(
        config=config,
        outdir=outdir,
        runner=runner,
        pot_cfg=pot_cfg,
        md_use=md_use,
        potential_lines=potential_lines,
        type_to_species=type_to_species,
        metrics_cfg=metrics_cfg,
        tm_cfg=tm_cfg,
        q_cfg=q_cfg,
        size_base_data=size_base_data,
        chosen_replicate=chosen_replicate,
        chosen_rate=chosen_rate,
        dt_ref=dt_ref,
        dt_mq=dt_mq,
        cooling_rate_ps=cooling_rate_ps,
        cutoffs_rate=cutoffs_rate,
        cutoffs_size=cutoffs_size,
        T_high=T_high,
        high_total_steps=high_total_steps,
        resume_state=resume_state,
        sampling_hint=sampling_hint,
        progress=progress,
        checkpoint_cb=checkpoint_cb,
        pressure_override=pressure_override,
        seed_base=seed_base,
        time_unit_ps_override=time_unit_ps_override,
        prod_cfg_override=prod_cfg_override,
        conv_cfg_override=conv_cfg_override,
        quench_steps_override=quench_steps_override,
        relax_steps_override=relax_steps_override,
    ).run()


def _autotune_resume_from_results(
    *,
    config: RunConfig,
    outdir: Path,
    prev: dict[str, Any],
    engine_build_identities: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Autotune resume from."""

    dft_identity_enabled = bool(
        getattr(
            getattr(getattr(config.autotune, "production", None), "dft_opt", None),
            "enabled",
            False,
        )
    )
    if engine_build_identities is None:
        engine_build_identities = query_engine_build_identities(
            config,
            workdir=outdir,
            primary_engine=str(config.engine),
            include_cp2k_refinement=dft_identity_enabled,
        )
    validate_engine_build_identity_bundle(engine_build_identities)

    # sanity resuming production
    prev_prod = prev.get("production", None)
    if not isinstance(prev_prod, dict):
        raise RuntimeError("Cannot resume: existing autotune_results.json has no 'production' state")
    existing_migrations = prev.get("release_resume_migrations", [])
    if not isinstance(existing_migrations, list) or not all(
        isinstance(item, Mapping) for item in existing_migrations
    ):
        raise RuntimeError(
            "Cannot safely resume autotune: release-resume migration history is malformed"
        )
    for migration_record in existing_migrations:
        validate_release_resume_migration(migration_record)
    validated_resume_fingerprint = _validate_autotune_resume_fingerprint(
        prev,
        config=config,
        outdir=outdir,
        engine_build_identities=engine_build_identities,
    )
    release_resume_migration = validated_resume_fingerprint.pop(
        "release_resume_migration",
        None,
    )
    _validate_production_resume_state(prev_prod, outdir=outdir)
    expected_top_status = _autotune_workflow_status(prev_prod)
    actual_top_status = str(prev.get("status", "")).strip().lower()
    production_status = str(prev_prod.get("status", "")).strip().lower()
    active_statuses = {"starting", "running"}
    disabled_preproduction_checkpoint = bool(
        not bool(prev_prod.get("enabled", True))
        and production_status == "not_requested"
        and actual_top_status in active_statuses
    )
    statuses_consistent = actual_top_status == expected_top_status.lower()
    if actual_top_status in active_statuses and production_status in active_statuses:
        statuses_consistent = True
    if disabled_preproduction_checkpoint:
        # This is the exact checkpoint written after every expensive autotune
        # stage has completed but before the final engine-identity check.  No
        # production work is requested.  Once the fingerprint, engine identity,
        # and sealed disabled state above have all been revalidated, completing
        # the top-level status is the only remaining operation.
        statuses_consistent = True
    if not statuses_consistent:
        raise RuntimeError("Cannot safely resume autotune: top-level and production statuses disagree")
    expected_top_execution = (
        "completed"
        if actual_top_status in {"ok", "incomplete", "not_converged"}
        else actual_top_status
    )
    if str(prev.get("execution_status", "")).strip().lower() != expected_top_execution:
        raise RuntimeError("Cannot safely resume autotune: top-level execution_status is inconsistent")
    if disabled_preproduction_checkpoint:
        completed = dict(prev)
        completed["status"] = "ok"
        completed["execution_status"] = "completed"
        write_autotune_outputs(outdir, completed)
        return completed
    if production_status in {"ok", "not_requested"}:
        return dict(prev)
    if (
        production_status in {"incomplete", "not_converged"}
        and not bool(prev_prod.get("resumable", True))
    ):
        # The terminal bundle is valid and scientifically complete as a
        # report, but its distributions were intentionally omitted.  Never
        # pretend it can be extended from less information than convergence
        # originally consumed.
        return dict(prev)

    # Validate the engine early, but defer potential dispatch until the
    # protected plan has been parsed.  In particular, never install the KIM
    # model named by the current config when the immutable plan records a
    # direct/hybrid potential (or vice versa).
    engine = str(getattr(config, "engine", "lammps") or "lammps").strip().lower()
    if engine not in {"lammps", "cp2k"}:
        raise ValueError(f"Unsupported engine '{engine}'")

    progress = CondensedProgressLog(outdir / "condensed.log")
    progress.info("autotune", "resuming from existing autotune_results.json")

    metric_warnings: list[str] = list(prev.get("metric_warnings", []) or [])

    def _warn_metric(msg: str) -> None:
        if str(msg) not in metric_warnings:
            metric_warnings.append(str(msg))
        warnings.warn(str(msg), stacklevel=2)
        progress.warn("metrics", str(msg))

    protected_plan = prev.get("production_plan")
    if not isinstance(protected_plan, Mapping):
        raise RuntimeError("Cannot safely resume autotune: protected production plan is missing")
    plan = dict(protected_plan)
    try:
        production_plan_from_dict(plan, base_dir=outdir)
    except (TypeError, ValueError, OSError) as exc:
        raise RuntimeError(
            "Cannot safely resume autotune: protected production plan is invalid"
        ) from exc
    if str(plan.get("engine", "")).strip().lower() != engine:
        raise RuntimeError("Cannot safely resume autotune: production-plan engine is inconsistent")

    pot_cfg = _protected_potential_from_plan(plan, fallback=config.kim)
    type_to_species = _protected_type_to_species_from_plan(
        plan,
        fallback=_get_type_to_species(config),
    )
    if engine == "lammps":
        if pot_cfg is None:
            raise RuntimeError(
                "Cannot safely resume autotune: LAMMPS production plan has no potential configuration"
            )
        runner: Union[LammpsRunner, Cp2kRunner] = LammpsRunner(config.lammps)
        kim_install = ensure_potential_model_installed(
            pot_cfg,
            installer=ensure_model_installed,
        )
    else:
        runner = Cp2kRunner(config.cp2k)
        kim_install = None

    md_use = MDConfig.model_validate(plan.get("md_use", {}))
    potential_lines_raw = plan.get("potential_lines")
    potential_lines = (
        None if potential_lines_raw is None else [str(x) for x in list(potential_lines_raw)]
    )
    chosen_rate = float(plan["chosen_rate"])
    chosen_replicate = [int(x) for x in list(plan["replicate"])]
    if len(chosen_replicate) != 3:
        raise RuntimeError("Cannot safely resume autotune: production-plan replicate is invalid")
    T_high = float(plan["T_high"])
    high_total_steps = int(plan["high_total_steps"])
    time_unit_ps = plan.get("time_unit_ps")
    time_unit_ps = None if time_unit_ps is None else float(time_unit_ps)
    cooling_rate_ps = plan.get("cooling_rate_ps")
    cooling_rate_ps = None if cooling_rate_ps is None else float(cooling_rate_ps)
    size_base_data = _resolve_result_path(plan["structure_data"], outdir=outdir)

    # The mutable status/checkpoint block and immutable plan must describe the
    # same calculation. Both are independently digested; these cross-checks
    # prevent a valid but internally contradictory result bundle.
    if float(prev_prod.get("rate_K_per_time")) != chosen_rate:
        raise RuntimeError("Cannot safely resume autotune: production rate disagrees with protected plan")
    if [int(x) for x in list(prev_prod.get("replicate", []))] != chosen_replicate:
        raise RuntimeError("Cannot safely resume autotune: replicate disagrees with protected plan")
    if float(prev_prod.get("T_high")) != T_high or int(prev_prod.get("highT_steps")) != high_total_steps:
        raise RuntimeError("Cannot safely resume autotune: high-temperature state disagrees with protected plan")
    state_structure = _resolve_result_path(prev_prod.get("structure_data"), outdir=outdir)
    if state_structure != size_base_data:
        raise RuntimeError("Cannot safely resume autotune: structure path disagrees with protected plan")

    metrics_cfg = StructureMetricsConfig.model_validate(plan.get("metrics_cfg", {}))
    metrics_summary = dict(plan.get("effective_metrics", {}) or {})
    progress.info("metrics", f"effective metrics summary: {metrics_summary}")

    cutoffs_rate = _cutoffs_any_to_dict(plan.get("cutoffs_rate"))
    cutoffs_size = _cutoffs_any_to_dict(
        plan.get("preferred_cutoffs") or plan.get("cutoffs_size") or plan.get("cutoffs_rate")
    )

    q_cfg = config.autotune.quench.model_copy(
        update={"t_final": float(plan["t_final"]), "relax_steps": int(plan["relax_steps"])}
    )
    tm_cfg = config.autotune.tm_scan.model_copy(update={"msd_every": int(plan["msd_every"])})
    prod_cfg_override = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    conv_cfg_override = ConvergenceConfig.model_validate(plan.get("convergence_cfg", {}))
    expected_max = getattr(prod_cfg_override, "max_boxes", None)
    state_max = prev_prod.get("max_boxes")
    if (
        bool(prev_prod.get("check_convergence"))
        != bool(prod_cfg_override.check_convergence)
        or int(prev_prod.get("min_boxes")) != int(prod_cfg_override.min_boxes)
        or int(prev_prod.get("batch_boxes")) != int(prod_cfg_override.batch_boxes)
        or (None if state_max is None else int(state_max))
        != (None if expected_max is None else int(expected_max))
        or int(prev_prod.get("required_convergence_streak"))
        != int(prod_cfg_override.consecutive_converged_checks)
    ):
        raise RuntimeError(
            "Cannot safely resume autotune: production checkpoint settings disagree with protected plan"
        )

    if release_resume_migration is not None:
        prev["resume_fingerprint"] = dict(validated_resume_fingerprint)
        prev["release_resume_migrations"] = [
            *list(existing_migrations),
            dict(release_resume_migration),
        ]
        progress.info(
            "autotune",
            "authenticated exact 0.4.35.1→0.4.36.0 zero-box checkpoint migration",
        )

    dt_ref = float(getattr(config.md, "timestep", md_use.timestep))
    dt_mq = float(md_use.timestep)

    prev["kim_install"] = _kim_install_jsonable(kim_install)
    prev["metric_warnings"] = list(metric_warnings)
    prev["effective_metrics"] = dict(metrics_summary)
    prev["paths"] = {
        "autotune_results": "autotune_results.json",
        "autotune": "autotune.json",
        "condensed_log": "condensed.log",
    }

    def _checkpoint_production(prod_state: dict[str, Any]) -> None:
        prev["status"] = "running"
        prev["execution_status"] = "running"
        prev["production"] = dict(prod_state)
        prev["metric_warnings"] = list(metric_warnings)
        prev["effective_metrics"] = dict(metrics_summary)
        write_autotune_outputs(outdir, prev)

    write_autotune_outputs(outdir, prev)
    progress.info("production", "resuming production ensemble")
    production = _run_production_ensemble(
        config=config,
        outdir=outdir,
        runner=runner,
        pot_cfg=pot_cfg,
        md_use=md_use,
        potential_lines=potential_lines,
        type_to_species=type_to_species,
        metrics_cfg=metrics_cfg,
        tm_cfg=tm_cfg,
        q_cfg=q_cfg,
        size_base_data=size_base_data,
        chosen_replicate=chosen_replicate,
        chosen_rate=chosen_rate,
        dt_ref=dt_ref,
        dt_mq=dt_mq,
        cooling_rate_ps=cooling_rate_ps,
        cutoffs_rate=cutoffs_rate,
        cutoffs_size=cutoffs_size,
        T_high=T_high,
        high_total_steps=high_total_steps,
        resume_state=prev_prod,
        progress=progress,
        checkpoint_cb=_checkpoint_production,
        pressure_override=float(plan["pressure"]),
        seed_base=int(plan["seed_base"]),
        time_unit_ps_override=(None if time_unit_ps is None else float(time_unit_ps)),
        prod_cfg_override=prod_cfg_override,
        conv_cfg_override=conv_cfg_override,
        quench_steps_override=int(plan["quench_steps"]),
        relax_steps_override=int(plan["relax_steps"]),
        sampling_hint=(
            None if plan.get("sampling_hint") is None else dict(plan.get("sampling_hint"))
        ),
    )

    final_engine_build_identities = query_engine_build_identities(
        config,
        workdir=outdir,
        primary_engine=str(config.engine),
        include_cp2k_refinement=dft_identity_enabled,
    )
    assert_engine_build_identity_bundle_unchanged(
        engine_build_identities,
        final_engine_build_identities,
        context="during resumed autotune production execution",
    )
    terminal_fingerprint = _build_autotune_resume_fingerprint(
        config=config,
        outdir=outdir,
        selected_structure=size_base_data,
        production_plan=plan,
        engine_build_identities=final_engine_build_identities,
    )
    stored_fingerprint = prev.get("resume_fingerprint")
    if not isinstance(stored_fingerprint, Mapping):
        raise RuntimeError(
            "Autotune configuration or scientific input bytes changed during "
            "resumed production; refusing to seal a mixed-input result"
        )
    _assert_autotune_terminal_fingerprint_unchanged(
        stored_fingerprint,
        terminal_fingerprint,
        context="resumed production",
    )

    prev["status"] = _autotune_workflow_status(production)
    prev["execution_status"] = (
        "completed"
        if prev["status"] in {"ok", "incomplete", "not_converged"}
        else str(prev["status"])
    )
    prev["production"] = production
    prev["metric_warnings"] = list(metric_warnings)
    prev["effective_metrics"] = dict(metrics_summary)
    prev["paths"] = {
        "autotune_results": "autotune_results.json",
        "autotune": "autotune.json",
        "condensed_log": "condensed.log",
        **dict(production.get("graph_outputs", {}) or {}),
    }
    write_autotune_outputs(outdir, prev)
    return prev




class _AutotuneWorkflow:
    """Autotune workflow."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self) -> dict[str, Any]:
        config = self.config
        outdir = self.outdir
        resume = self.resume
        ensure_dir(outdir)
        results_path = outdir / "autotune_results.json"
        do_resume = _resolve_autotune_resume_mode(
            outdir=outdir,
            results_path=results_path,
            resume=resume,
        )
        dft_identity_enabled = bool(
            getattr(
                getattr(getattr(config.autotune, "production", None), "dft_opt", None),
                "enabled",
                False,
            )
        )
        # Probe before any simulation or checkpoint is created.  The first
        # resumable result therefore binds the engine build that produced all
        # preceding scientific decisions, and a switched build fails before
        # a resumed stage can run.
        engine_build_identities = query_engine_build_identities(
            config,
            workdir=outdir,
            primary_engine=str(config.engine),
            include_cp2k_refinement=dft_identity_enabled,
        )
        validate_engine_build_identity_bundle(engine_build_identities)
        initial_config_input_identities = _config_input_identities(
            config,
            workdir=outdir,
        )
        if do_resume:
            prev = json.loads(results_path.read_text())
            return _autotune_resume_from_results(
                config=config,
                outdir=outdir,
                prev=prev,
                engine_build_identities=engine_build_identities,
            )

        # starting structure benchmarks
        initial_data = prepare_initial_structure(config, outdir)

        type_to_species = _get_type_to_species(config)

        if getattr(config, "engine", "lammps") == "cp2k":
            kim_install = None
            runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]
        else:
            # KIM installation is meaningful only for an explicitly tagged
            # KIM potential.  Analytic/hybrid LAMMPS configurations must not
            # acquire a hidden KIM dependency on either a fresh run or resume.
            kim_install = ensure_potential_model_installed(
                config.kim,
                installer=ensure_model_installed,
            )
            runner = LammpsRunner(config.lammps)

        if bool(getattr(config.md, "force_isotropic", False)) and (not isinstance(runner, LammpsRunner)):
            raise ValueError(
                "md.force_isotropic is currently supported only for engine='lammps'"
            )

        progress = CondensedProgressLog(outdir / "condensed.log")
        progress.info("autotune", "initialising workflow")

        metric_warnings: list[str] = []

        def _warn_metric(msg: str) -> None:
            metric_warnings.append(str(msg))
            warnings.warn(str(msg), stacklevel=2)
            progress.warn("metrics", str(msg))

        pot_cfg = config.kim
        analysis_lammps_units_style: Optional[str] = (
            resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal")
            if isinstance(runner, LammpsRunner)
            else None
        )

        metrics_cfg, metric_auto_defaults, metrics_summary = resolve_effective_metrics_config(
            config.autotune.metrics,
            structure_data=Path(initial_data),
            type_to_species=type_to_species,
            lammps_units_style=analysis_lammps_units_style,
            warn_fn=_warn_metric,
            context="autotune production",
        )
        progress.info("metrics", f"effective metrics summary: {metrics_summary}")

        rng = random.Random(config.random_seed)

        # preflight
        # preflight
        # buckingham stabilization overlay
        # thermostat parameter scan
        # progress info preflight
        progress.info("preflight", "starting preflight checks")
        preflight = run_preflight(runner, config, initial_data, outdir)

        # thermo dump frequencies
        # applying preflight selected
        md_use = config.md.model_copy(
            deep=True,
            update={
                "timestep": float(preflight.selected_timestep),
                "ensemble": str(preflight.selected_ensemble),
                "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(preflight.selected_tdamp)),
                "barostat": (
                    BarostatConfig(style=config.md.barostat.style, pdamp=float(preflight.selected_pdamp))
                    if str(preflight.selected_ensemble) == "npt" and preflight.selected_pdamp is not None
                    else config.md.barostat
                ),
            },
        )

        potential_lines = preflight.potential_lines

        def _maybe_elastic(
            stage_role: str,
            *,
            stage_dir: Path,
            structure_data: Path,
            input_data: Path,
            force_iso_context: bool,
        ) -> Optional[dict[str, Any]]:
            run_screen, strict, _cfg = should_run_elastic_screen(
                metrics_cfg,
                runner=runner,
                stage_role=stage_role,
                force_isotropic=bool(force_iso_context),
            )
            if not run_screen:
                return None
            try:
                return run_elastic_screen_lammps(
                    runner,
                    pot_cfg,
                    md_use,
                    structure_data=structure_data,
                    stage_dir=stage_dir,
                    potential_lines=potential_lines,
                    metrics_cfg=metrics_cfg,
                    force_isotropic=bool(force_iso_context),
                    input_data_for_affine_strain=input_data if bool(force_iso_context) else None,
                    outdir=outdir,
                )
            except Exception:
                if strict:
                    raise
                return None

        # progress info tm
        progress.info("tm_scan", "starting operational melting scan")
        # melting scan
        # tm cfg
        tm_cfg = config.autotune.tm_scan
        scan_dir = outdir / "tm_scan"
        ensure_dir(scan_dir)

        T_values = np.arange(tm_cfg.t_min, tm_cfg.t_max + 0.5 * tm_cfg.dT, tm_cfg.dT, dtype=float)

        # melt scan statistically
        # reliable isolated outliers
        tm_nrep = int(getattr(tm_cfg, 'replicates_per_temp', 1))
        if tm_nrep < 1:
            tm_nrep = 1

        tm_outcomes_all: list[StageOutcome] = []
        tm_by_T: dict[float, list[StageOutcome]] = {float(t): [] for t in T_values.tolist()}
        tm_end_data: list[Path] = []

        # scan ensemble overridden
        md_scan = md_use
        ens_override = getattr(tm_cfg, 'ensemble', None)
        if ens_override is not None:
            ens_override = str(ens_override).strip().lower()
            if ens_override in ('nvt','npt') and ens_override != str(md_use.ensemble):
                md_scan = md_use.model_copy(deep=True, update={"ensemble": ens_override})


        gr_enabled = bool(getattr(tm_cfg, "gr", None) and tm_cfg.gr.enabled)

        # physical equilibration sampling
        # timestep adjusted preflight
        dt_ref = float(config.md.timestep)
        dt_scan = float(md_scan.timestep)
        equil_steps = scale_steps_for_timestep(int(tm_cfg.equil_steps), dt_ref, dt_scan, min_steps=0)
        sample_steps = scale_steps_for_timestep(int(tm_cfg.sample_steps), dt_ref, dt_scan, min_steps=1)
        stride_steps = None
        if gr_enabled:
            stride_steps = scale_steps_for_timestep(int(tm_cfg.gr.stride), dt_ref, dt_scan, min_steps=1)

        # independent heating replicas
        for rep_id in range(1, tm_nrep + 1):
            prev_data = initial_data
            for k, T in enumerate(T_values):
                if tm_nrep == 1:
                    stage_name = f"tm_{int(round(T))}K"
                else:
                    stage_name = f"tm_{int(round(T))}K_rep{rep_id}"
                stage_dir = scan_dir / stage_name

                seed = rng.randrange(1, 2**31 - 1)

                stage = StageSpec(
                    name=stage_name,
                    input_data=prev_data,
                    output_data=stage_dir / "output.data",
                    temperature_start=float(T),
                    temperature_stop=float(T),
                    pressure=float(md_use.pressure),
                    equil_steps=int(equil_steps),
                    run_steps=int(sample_steps),
                    seed=int(seed),
                    replicate=None,
                    write_dump=bool(gr_enabled),
                    tail_dump_frames=int(tm_cfg.gr.frames) if gr_enabled else None,
                    tail_dump_stride=int(stride_steps) if gr_enabled else None,
                    msd_every=int(tm_cfg.msd_every),
                    sample_ensemble=(
                        "nvt"
                        if (getattr(tm_cfg, "sample_in_nvt", False) and md_scan.ensemble == "npt")
                        else None
                    ),
                )
                out = _stage_run(
                    runner,
                    pot_cfg,
                    md_scan,
                    stage,
                    stage_dir,
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )

                # g melt indicator
                if gr_enabled:
                    dump_path = stage_dir / f"{stage_name}.lammpstrj"
                    try:
                        cand = stage_dir / "traj.extxyz"
                        traj_path = cand if cand.exists() else dump_path
                        frames = read_last_frames_auto(
                            traj_path,
                            int(tm_cfg.gr.frames),
                            type_to_species=type_to_species,
                            units_style=analysis_lammps_units_style,
                        )
                        gr = compute_first_peak_gr(
                            frames,
                            r_max=float(tm_cfg.gr.r_max),
                            nbins=int(tm_cfg.gr.nbins),
                            smooth=int(tm_cfg.gr.smooth),
                            pair=tm_cfg.gr.pair,
                            type_to_species=type_to_species,
                            r_ignore_factor=float(tm_cfg.gr.r_ignore_factor),
                            r_search_factor=float(tm_cfg.gr.r_search_factor),
                        )
                        out = StageOutcome(
                            **{
                                **asdict(out),
                                "gr_peak_r": float(gr.peak_r),
                                "gr_peak_height": float(gr.peak_height),
                                "gr_peak_fwhm": float(gr.peak_fwhm),
                            }
                        )
                    except Exception:
                        # scan analysis fails
                        pass

                # downstream aggregation reporting
                out = StageOutcome(**{**asdict(out), "rep_id": int(rep_id)})

                tm_outcomes_all.append(out)
                tm_by_T[float(T)].append(out)
                prev_data = stage_dir / out.output_data  # output inside dir

            tm_end_data.append(Path(prev_data))

        # aggregate temperature statistics
        def _agg(x: list[float], *, clamp_nonneg: bool = False) -> tuple[float, float, float]:
            return _complete_tm_replicate_summary(
                x,
                require_nonnegative=bool(clamp_nonneg),
            )

        T = np.asarray([float(t) for t in T_values.tolist()], dtype=float)
        tm_summary: list[dict[str, Any]] = []
        D_mu: list[float] = []
        D_se: list[float] = []
        D_med: list[float] = []
        H_mu: list[float] = []
        H_se: list[float] = []
        H_med: list[float] = []
        W_mu: list[float] = []
        W_se: list[float] = []
        W_med: list[float] = []
        msd_mu: list[float] = []
        msd_se: list[float] = []
        msd_med: list[float] = []
        vol_mu: list[float] = []
        vol_se: list[float] = []
        vol_med: list[float] = []

        for t in T.tolist():
            reps = tm_by_T.get(float(t), [])
            Dm, Ds, Dmd = _agg([o.D for o in reps], clamp_nonneg=True)
            D_mu.append(Dm)
            D_se.append(Ds)
            D_med.append(Dmd)
            Hm, Hs, Hmd = _agg([o.gr_peak_height for o in reps])
            Wm, Ws, Wmd = _agg([o.gr_peak_fwhm for o in reps])
            H_mu.append(Hm)
            H_se.append(Hs)
            H_med.append(Hmd)
            W_mu.append(Wm)
            W_se.append(Ws)
            W_med.append(Wmd)
            mm, ms, mmd = _agg([o.msd_rms_last for o in reps], clamp_nonneg=True)
            vm, vs, vmd = _agg([o.vol_last for o in reps], clamp_nonneg=True)
            msd_mu.append(mm)
            msd_se.append(ms)
            msd_med.append(mmd)
            vol_mu.append(vm)
            vol_se.append(vs)
            vol_med.append(vmd)
            tm_summary.append(
                {
                    "T": float(t),
                    "nrep": int(len(reps)),
                    "D_mean": float(Dm),
                    "D_stderr": float(Ds),
                    "D_median": float(Dmd),
                    "D_boundary_constrained_replicates": int(
                        sum(bool(o.D_boundary_constrained) for o in reps)
                    ),
                    "gr_peak_height_mean": float(Hm),
                    "gr_peak_height_stderr": float(Hs),
                    "gr_peak_height_median": float(Hmd),
                    "gr_peak_fwhm_mean": float(Wm),
                    "gr_peak_fwhm_stderr": float(Ws),
                    "gr_peak_fwhm_median": float(Wmd),
                    "msd_rms_last_mean": float(mm),
                    "msd_rms_last_stderr": float(ms),
                    "msd_rms_last_median": float(mmd),
                    "vol_last_mean": float(vm),
                    "vol_last_stderr": float(vs),
                    "vol_last_median": float(vmd),
                }
            )

        # replicas representative selection
        D = np.asarray(D_med, dtype=float)

        # estimate diffusion combined
        if bool(getattr(tm_cfg, "gr", None) and tm_cfg.gr.enabled):
            H = np.asarray(H_med, dtype=float)
            W = np.asarray(W_med, dtype=float)
            msd_rms = np.asarray(msd_med, dtype=float)
            vol_last = np.asarray(vol_med, dtype=float)
            natoms = int(count_atoms_in_datafile(initial_data))
            tm_est = estimate_tm(
                T,
                D,
                gr_peak_height=H,
                gr_peak_fwhm=W,
                msd_rms_last=msd_rms,
                vol_last=vol_last,
                natoms=natoms,
                w_diffusion=float(tm_cfg.gr.w_diffusion),
                w_peak_height=float(tm_cfg.gr.w_peak_height),
                w_peak_fwhm=float(tm_cfg.gr.w_peak_fwhm),
                liquid_D_frac=float(getattr(tm_cfg, 'liquid_D_frac', 0.2)),
                liquid_top_k=int(getattr(tm_cfg, 'liquid_top_k', 3)),
                liquid_min_consecutive=int(getattr(tm_cfg, 'liquid_min_consecutive', 2)),
            )
        else:
            tm_est = estimate_tm_from_diffusion(T, D)

        # temperature melt operational
        T_high_base = float(tm_est.Tm)
        if getattr(tm_cfg, 'highT_mode', 'onset') == 'liquid' and hasattr(tm_est, 'T_liquid') and np.isfinite(tm_est.T_liquid):
            T_high_base = float(tm_est.T_liquid)
        T_high = float(T_high_base + config.autotune.highT.margin)

        # warn
        # inherently stability diagnostics
        # evaluated higher temperature
        try:
            t_scan_max = float(getattr(tm_cfg, 't_max', float('nan')))
            if math.isfinite(t_scan_max) and T_high > t_scan_max:
                warnings.warn(
                    f"Selected T_high={T_high:g} K exceeds tm_scan.t_max={t_scan_max:g} K. "
                    "Consider extending the scan range if unexpected instability occurs.",
                    stacklevel=2,
                )
        except Exception:
            pass

        # progress info high
        progress.info("highT", "starting high-temperature disordering")
        # high equilibration time
        # high cfg
        high_cfg = config.autotune.highT
        high_dir = outdir / "highT"
        ensure_dir(high_dir)

        # equilibration production ensemble
        # volume transient quench
        dt_mq = float(md_use.timestep)
        force_iso_active = bool(getattr(md_use, "force_isotropic", False))
        chunk_steps = scale_steps_for_timestep(int(high_cfg.chunk_steps), dt_ref, dt_mq, min_steps=1)
        min_total_steps = scale_steps_for_timestep(int(high_cfg.min_total_steps), dt_ref, dt_mq, min_steps=1)
        min_total_steps = extend_highT_steps_for_force_isotropic(int(min_total_steps), force_isotropic=force_iso_active)

        # continuous segment replica
        # chunk sizing equivalent
        # loop
        max_chunks = int(getattr(high_cfg, "max_chunks", 20))
        if max_chunks < 1:
            max_chunks = 1
        max_total_steps = int(max(int(min_total_steps), int(chunk_steps) * int(max_chunks)))
        max_total_steps = extend_highT_steps_for_force_isotropic(int(max_total_steps), force_isotropic=force_iso_active)

        # starting disordering scan
        # scan temperature structure
        start_pool: list[Path] = [Path(p) for p in tm_end_data if Path(p).exists()]
        if len(start_pool) == 0:
            start_pool = [Path(initial_data)]

        high_outcomes: list[StageOutcome] = []
        high_rep_summaries: list[dict[str, Any]] = []
        melt_pool: list[Path] = []
        high_steps: list[int] = []

        n_high_rep = int(getattr(high_cfg, 'replicates', 1))
        if n_high_rep < 1:
            n_high_rep = 1

        for rep_id in range(1, n_high_rep + 1):
            start_data = start_pool[(rep_id - 1) % len(start_pool)]
            if n_high_rep == 1:
                stage_name = "highT"
            else:
                stage_name = f"highT_rep{rep_id}"
            stage_dir = high_dir / stage_name
            seed = rng.randrange(1, 2**31 - 1)

            # coordinate msd equilibration
            # followed equilibrated volume
            msd_every_high = max(1, int(tm_cfg.msd_every))

            equil_steps_high = 0
            run_steps_high = int(max_total_steps)
            sample_ensemble_high = None
            if str(md_use.ensemble).lower() == "npt":
                sample_ensemble_high = "nvt"
                equil_steps_high = int(min_total_steps)
                run_steps_high = int(max_total_steps) - int(equil_steps_high)

            # sampling segment msd
            if int(run_steps_high) < 3:
                shift = 3 - int(run_steps_high)
                run_steps_high = 3
                equil_steps_high = max(0, int(equil_steps_high) - int(shift))

            if int(run_steps_high) < 3 * int(msd_every_high):
                msd_every_high = max(1, int(run_steps_high) // 3)

            total_steps_high = int(equil_steps_high) + int(run_steps_high)

            stage = StageSpec(
                name=stage_name,
                input_data=start_data,
                output_data=stage_dir / "output.data",
                temperature_start=T_high,
                temperature_stop=T_high,
                pressure=float(md_use.pressure),
                equil_steps=int(equil_steps_high),
                run_steps=int(run_steps_high),
                seed=int(seed),
                force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
                replicate=None,
                write_dump=False,
                msd_every=int(msd_every_high),
                sample_ensemble=sample_ensemble_high,
            )
            out = _stage_run(
                runner,
                pot_cfg,
                md_use,
                stage,
                stage_dir,
                potential_lines=potential_lines,
                type_to_species=type_to_species,
            )
            out = StageOutcome(**{**asdict(out), "rep_id": int(rep_id)})
            high_outcomes.append(out)

            elastic_high = _maybe_elastic(
                "highT",
                stage_dir=stage_dir,
                structure_data=(stage_dir / out.output_data),
                input_data=Path(start_data),
                force_iso_context=bool(force_iso_active),
            )

            melt_data_rep = stage_dir / out.output_data
            melt_pool.append(Path(melt_data_rep))

            # spacing estimate volume
            if out.n_atoms > 0 and out.vol_last > 0:
                l = float((out.vol_last / out.n_atoms) ** (1.0 / 3.0))
            else:
                l = float("nan")
            thresh = float(high_cfg.rms_multiple) * float(l) if math.isfinite(l) else float("nan")

            # determine continuous msd
            disorder_step = int(total_steps_high)
            disorder_rms = float("nan")
            reached_rms = False
            try:
                msd = parse_msd_file(stage_dir / f"{stage_name}.msd.dat")
                steps = np.asarray(msd.step, dtype=int)
                total_steps = steps + int(equil_steps_high)
                rms = np.sqrt(np.maximum(0.0, np.asarray(msd.msd, dtype=float)))
                if math.isfinite(thresh) and thresh > 0:
                    m = (total_steps >= int(min_total_steps)) & (rms >= float(thresh))
                    if np.any(m):
                        j = int(np.where(m)[0][0])
                        disorder_step = int(total_steps[j])
                        disorder_rms = float(rms[j])
                        reached_rms = True
            except Exception:
                pass
            # stationarity diagnostic density
            dens_early = float('nan')
            dens_late = float('nan')
            dens_rel = float('nan')
            pe_early = float('nan')
            pe_late = float('nan')
            pe_rel = float('nan')
            stationarity_ok = False
            stationarity_density_segment = "sample"
            stationarity_pe_segment = "sample"
            try:
                # prefer parsing handle
                # particular equilibration followed
                # thermo table density
                # stationarity diagnostics density
                # segment pot sampling

                n_tbl = 1
                try:
                    tbls = parse_all_thermo_tables(stage_dir / 'log.lammps')
                    n_tbl = int(len(tbls))
                    thermo_equil = tbls[0].as_dict()
                    thermo_sample = tbls[-1].as_dict()
                except Exception:
                    thermo_sample = parse_thermo_csv(stage_dir / 'thermo.csv').as_dict()
                    thermo_equil = thermo_sample

                use_split = (
                    int(n_tbl) >= 2
                    and sample_ensemble_high is not None
                    and str(md_use.ensemble).strip().lower() == 'npt'
                    and str(sample_ensemble_high).strip().lower() == 'nvt'
                    and int(equil_steps_high) > 0
                )

                dens_series = (thermo_equil.get('Density', []) if use_split else thermo_sample.get('Density', []))
                pe_series = thermo_sample.get('PotEng', [])

                stationarity_density_segment = 'equil' if use_split else 'sample'
                stationarity_pe_segment = 'sample'

                dens_chg = early_late_change(dens_series, split_fraction=0.5, denom='late')
                pe_chg = early_late_change(pe_series, split_fraction=0.5, denom='late')
                dens_early = float(dens_chg.early_mean)
                dens_late = float(dens_chg.late_mean)
                dens_rel = float(dens_chg.rel_change)
                pe_early = float(pe_chg.early_mean)
                pe_late = float(pe_chg.late_mean)
                pe_rel = float(pe_chg.rel_change)
                tol = float(high_cfg.stationarity_tol)
                stationarity_ok = (
                    math.isfinite(dens_rel)
                    and math.isfinite(pe_rel)
                    and (dens_rel <= tol)
                    and (pe_rel <= tol)
                )
            except Exception:
                stationarity_ok = False

            # criterion triggers immediately
            disorder_step = int(max(int(min_total_steps), int(disorder_step)))
            high_steps.append(int(disorder_step))

            high_rep_summaries.append(
                {
                    "rep_id": int(rep_id),
                    "seed": int(seed),
                    "start_data": str(Path(start_data).relative_to(outdir)) if Path(start_data).is_relative_to(outdir) else str(start_data),
                    "equil_steps": int(equil_steps_high),
                    "sample_steps": int(run_steps_high),
                    "run_steps": int(total_steps_high),
                    "max_total_steps": int(max_total_steps),
                    "sample_ensemble": str(sample_ensemble_high) if sample_ensemble_high is not None else str(md_use.ensemble),
                    "min_total_steps": int(min_total_steps),
                    "disorder_step": int(disorder_step),
                    "reached_rms": bool(reached_rms),
                    "rms_threshold": float(thresh),
                    "rms_at_disorder": float(disorder_rms),
                    "spacing_l": float(l),
                    "density_mean": float(out.density_mean),
                    "density_stderr": float(out.density_stderr),
                    "density_early_mean": float(dens_early),
                    "density_late_mean": float(dens_late),
                    "density_rel_change": float(dens_rel),
                    "pe_mean": float(out.pe_mean),
                    "pe_stderr": float(out.pe_stderr),
                    "pe_early_mean": float(pe_early),
                    "pe_late_mean": float(pe_late),
                    "pe_rel_change": float(pe_rel),
                    "stationarity_ok": bool(stationarity_ok),
                    "stationarity_density_segment": str(stationarity_density_segment),
                    "stationarity_pe_segment": str(stationarity_pe_segment),
                    "melt_data": str(Path(melt_data_rep).relative_to(outdir)) if Path(melt_data_rep).is_relative_to(outdir) else str(melt_data_rep),
                    "elastic": elastic_high,
                }
            )

        # recommended disordering observed
        high_total_steps = int(max(high_steps) if len(high_steps) > 0 else int(min_total_steps))

        # conservative melt subsequent
        # longest time disorder
        idx_max: Optional[int] = None
        if len(melt_pool) > 0 and len(high_steps) > 0:
            idx_max = int(np.argmax(np.asarray(high_steps, dtype=int)))
            melt_data = melt_pool[idx_max]
        else:
            melt_data = Path(start_pool[-1])

        # stationarity summary enforcement
        stationarity_flags = [bool(r.get("stationarity_ok", False)) for r in high_rep_summaries]
        n_stat = int(len(stationarity_flags))
        n_ok = int(sum(1 for x in stationarity_flags if x))
        ok_fraction = float(n_ok) / float(n_stat) if n_stat > 0 else float("nan")

        dens_rel_arr = np.asarray([float(r.get("density_rel_change", float("nan"))) for r in high_rep_summaries], dtype=float)
        pe_rel_arr = np.asarray([float(r.get("pe_rel_change", float("nan"))) for r in high_rep_summaries], dtype=float)
        dens_rel_max = float(np.nanmax(dens_rel_arr)) if np.any(np.isfinite(dens_rel_arr)) else float("nan")
        pe_rel_max = float(np.nanmax(pe_rel_arr)) if np.any(np.isfinite(pe_rel_arr)) else float("nan")

        high_stationarity_summary = {
            "tol": float(high_cfg.stationarity_tol),
            "enforced": bool(getattr(high_cfg, "enforce_stationarity", False)),
            "ok_fraction": float(ok_fraction),
            "ok_count": int(n_ok),
            "n": int(n_stat),
            "density_rel_change_max": float(dens_rel_max),
            "pe_rel_change_max": float(pe_rel_max),
            "rep_id_max_disorder": int(high_rep_summaries[idx_max]["rep_id"]) if idx_max is not None and idx_max < len(high_rep_summaries) else None,
            "stationarity_ok_max_disorder": bool(high_rep_summaries[idx_max]["stationarity_ok"]) if idx_max is not None and idx_max < len(high_rep_summaries) else None,
        }

        if bool(getattr(high_cfg, "enforce_stationarity", False)):
            if idx_max is None or idx_max >= len(high_rep_summaries):
                raise ValueError("High-T stationarity enforcement requested, but no replicate summary is available")
            if not bool(high_rep_summaries[idx_max].get("stationarity_ok", False)):
                tol = float(high_cfg.stationarity_tol)
                dens_rel = float(high_rep_summaries[idx_max].get("density_rel_change", float("nan")))
                pe_rel = float(high_rep_summaries[idx_max].get("pe_rel_change", float("nan")))
                raise ValueError(
                    f"High-T stationarity check failed for the max-disorder replica (rep_id={high_rep_summaries[idx_max].get('rep_id')}). "
                    f"density_rel_change={dens_rel:g}, pe_rel_change={pe_rel:g}, tol={tol:g}. "
                    "Increase highT.min_total_steps/max_chunks or relax highT.stationarity_tol."
                )
        else:
            # warn
            if idx_max is not None and idx_max < len(high_rep_summaries):
                if not bool(high_rep_summaries[idx_max].get("stationarity_ok", False)):
                    tol = float(high_cfg.stationarity_tol)
                    dens_rel = float(high_rep_summaries[idx_max].get("density_rel_change", float("nan")))
                    pe_rel = float(high_rep_summaries[idx_max].get("pe_rel_change", float("nan")))
                    warnings.warn(
                        f"High-T stationarity check failed for the max-disorder replica (rep_id={high_rep_summaries[idx_max].get('rep_id')}). "
                        f"density_rel_change={dens_rel:g}, pe_rel_change={pe_rel:g}, tol={tol:g}. "
                        "Proceeding because highT.enforce_stationarity=false.",
                        stacklevel=2,
                    )

        # progress info rate
        progress.info("rate_scan", "starting quench-rate scan")
        # rate scan density
        # q cfg
        q_cfg = config.autotune.quench
        rates, time_unit_ps, rates_ps_sorted = resolve_quench_rates_K_per_time(config)
        rate_dir = outdir / "rates"
        ensure_dir(rate_dir)

        n_rep_rate = int(config.autotune.quench.replicates_per_rate)
        rate_results: list[dict[str, Any]] = []
        used_rate_stage_seeds: set[int] = set()

        def _draw_rate_stage_seed() -> int:
            while True:
                value = int(rng.randrange(1, 2**31 - 1))
                if value not in used_rate_stage_seeds:
                    used_rate_stage_seeds.add(value)
                    return value

        for idx_r, r in enumerate(rates):
            dens_reps: list[float] = []
            rep_entries: list[dict[str, Any]] = []
            r_ps = float("nan")
            if time_unit_ps is not None:
                r_ps = float(r) / float(time_unit_ps)
            elif rates_ps_sorted is not None and idx_r < len(rates_ps_sorted):
                r_ps = float(rates_ps_sorted[idx_r])
            # quench steps rate
            # important timestep yaml
            # timestep requested rate
            dT = T_high - q_cfg.t_final
            n_quench = quench_steps_for_rate(float(dT), float(r), float(dt_mq), min_steps=1)

            for rep in range(n_rep_rate):
                rtag = f"r{r:g}_rep{rep+1}"
                rdir = rate_dir / rtag
                ensure_dir(rdir)

                # quench once rate

                seed_melt = _draw_rate_stage_seed()
                # diverse starting snapshots
                # correlation between replicates
                melt_seed_data = melt_data
                melt_pool_index: Optional[int] = None
                if len(melt_pool) > 0:
                    melt_pool_index = int(rng.randrange(0, len(melt_pool)))
                    melt_seed_data = melt_pool[melt_pool_index]
                melt_stage = StageSpec(
                    name=f"melt_r{r:g}",
                    input_data=Path(melt_seed_data),
                    output_data=rdir / "melt.data",
                    temperature_start=T_high,
                    temperature_stop=T_high,
                    pressure=float(md_use.pressure),
                    equil_steps=0,
                    run_steps=int(high_total_steps),
                    seed=int(seed_melt),
                    force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
                    replicate=None,
                    write_dump=False,
                    msd_every=int(tm_cfg.msd_every),
                )
                melt_out = _stage_run(
                    runner,
                    pot_cfg,
                    md_use,
                    melt_stage,
                    rdir / "melt",
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )

                seed = _draw_rate_stage_seed()
                quench_stage = StageSpec(
                    name=f"quench_r{r:g}",
                    input_data=(rdir / "melt" / melt_out.output_data),
                    output_data=rdir / "quench.data",
                    temperature_start=T_high,
                    temperature_stop=q_cfg.t_final,
                    pressure=float(md_use.pressure),
                    equil_steps=0,
                    run_steps=n_quench,
                    seed=int(seed),
                    replicate=None,
                    write_dump=False,
                    msd_every=int(tm_cfg.msd_every),
                )
                quench_out = _stage_run(
                    runner,
                    pot_cfg,
                    md_use,
                    quench_stage,
                    rdir / "quench",
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )

                seed2 = _draw_rate_stage_seed()
                mcfg = config.autotune.metrics
                relax_steps = scale_steps_for_timestep(int(q_cfg.relax_steps), dt_ref, dt_mq, min_steps=1)
                relax_stage = StageSpec(
                    name=f"relax_r{r:g}",
                    input_data=(rdir / "quench" / quench_out.output_data),
                    output_data=rdir / "relax.data",
                    temperature_start=q_cfg.t_final,
                    temperature_stop=q_cfg.t_final,
                    pressure=float(md_use.pressure),
                    equil_steps=0,
                    run_steps=int(relax_steps),
                    seed=int(seed2),
                    replicate=None,
                    write_dump=True,
                    dump_every=int(relax_steps) if not mcfg.enabled else None,
                    tail_dump_frames=int(mcfg.time_average_frames) if mcfg.enabled else None,
                    tail_dump_stride=int(mcfg.time_average_stride) if mcfg.enabled else None,
                    msd_every=int(tm_cfg.msd_every),
                )
                relax_dir = rdir / "relax"
                relax_out = _stage_run(
                    runner,
                    pot_cfg,
                    md_use,
                    relax_stage,
                    relax_dir,
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )
                melt_elastic = _maybe_elastic(
                    "melt",
                    stage_dir=rdir / "melt",
                    structure_data=(rdir / "melt" / melt_out.output_data),
                    input_data=Path(melt_seed_data),
                    force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                )
                relax_elastic = _maybe_elastic(
                    "relax",
                    stage_dir=relax_dir,
                    structure_data=(relax_dir / relax_out.output_data),
                    input_data=(rdir / "quench" / quench_out.output_data),
                    force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                )
                dens_reps.append(relax_out.density_mean)
                rep_entries.append(
                    {
                        "replicate_id": int(rep + 1),
                        "seeds": {
                            "melt": int(seed_melt),
                            "quench": int(seed),
                            "relax": int(seed2),
                        },
                        "melt_source": str(melt_seed_data),
                        "melt_pool_index": melt_pool_index,
                        "density": float(relax_out.density_mean),
                        "n_quench_steps": int(n_quench),
                        "cooling_rate_K_per_time": float(r),
                        "cooling_rate_K_per_ps": float(r_ps) if math.isfinite(r_ps) else None,
                        "final_data": str((relax_dir / relax_out.output_data).relative_to(outdir))
                        if (relax_dir / relax_out.output_data).is_relative_to(outdir)
                        else str(relax_dir / relax_out.output_data),
                        "dump": str((relax_dir / f"{relax_stage.name}.lammpstrj").relative_to(outdir))
                        if (relax_dir / f"{relax_stage.name}.lammpstrj").is_relative_to(outdir)
                        else str(relax_dir / f"{relax_stage.name}.lammpstrj"),
                        "elastic_melt": melt_elastic,
                        "elastic_relax": relax_elastic,
                    }
                )

            mu, se = _complete_mean_stderr(dens_reps)
            rate_results.append(
                {
                    "rate": float(r),
                    "rate_K_per_ps": float(r_ps) if math.isfinite(r_ps) else None,
                    "n_quench_steps": int(n_quench),
                    "density_mean": mu,
                    "density_stderr": se,
                    "nrep": len(dens_reps),
                    "replicates": rep_entries,
                }
            )

        # density structure metrics
        # effective metrics start
        type_to_species = _get_type_to_species(config)
        cutoffs_rate: dict[tuple[int, int], float] = {}
        rate_results_for_selection = list(rate_results)

        if metrics_cfg.enabled:
            required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
            fixed_cut = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)

            cutoffs_rate, frames_by_path = _estimate_pooled_scan_cutoffs(
                rate_results,
                outdir=outdir,
                metrics_cfg=metrics_cfg,
                required_pairs=required_pairs,
                fixed_cutoffs=fixed_cut,
                type_to_species=type_to_species,
                lammps_units_style=analysis_lammps_units_style,
            )

            # replicate metrics rate
            for rr in rate_results:
                for rep_entry in rr["replicates"]:
                    traj_path = _resolve_replicate_traj_path(outdir=outdir, rep_entry=rep_entry)
                    frames = frames_by_path.get(str(traj_path))
                    if frames is None:
                        frames = read_last_frames_auto(
                            traj_path,
                            int(metrics_cfg.time_average_frames),
                            type_to_species=type_to_species,
                            units_style=analysis_lammps_units_style,
                        )
                        frames_by_path[str(traj_path)] = frames
                    sm = compute_structure_metrics_timeavg(frames, metrics_cfg, cutoffs=cutoffs_rate, type_to_species=type_to_species)
                    rep_entry["metrics"] = dict(sm.values)
                    if bool(getattr(getattr(metrics_cfg, "amorphous", None), "enabled", False)):
                        rep_entry["amorphous"] = analyse_amorphous_state(
                            frames,
                            metrics_cfg=metrics_cfg,
                            cutoffs=cutoffs_rate,
                            type_to_species=type_to_species,
                            cache_dir=(outdir / "amorphous_references"),
                            progress=progress,
                        )

            # aggregate rate
            for rr in rate_results:
                rep_metrics = [re.get("metrics", {}) for re in rr["replicates"]]
                mu_m, se_m = _aggregate_scalar_metrics(rep_metrics)
                rr["metrics_mean"] = mu_m
                rr["metrics_stderr"] = se_m

            amorph_cfg = getattr(metrics_cfg, "amorphous", None)
            if bool(getattr(amorph_cfg, "enabled", False)):
                rate_amorphous = summarize_rate_amorphous_acceptance(rate_results, amorph_cfg=amorph_cfg)
                for rr, summary in zip(rate_results, rate_amorphous):
                    rr["amorphous_summary"] = dict(summary)
                if bool(getattr(amorph_cfg, "enforce_during_rate_scan", False)):
                    rate_results_for_selection = [
                        rr for rr in rate_results if bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))
                    ]
                    rejected_rates = [
                        rr for rr in rate_results if not bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))
                    ]
                    for rr in rejected_rates:
                        summ = dict(rr.get("amorphous_summary", {}) or {})
                        crit_txt = _format_rate_amorphous_criteria_summary(summ)
                        msg = (
                            f"rate {float(rr['rate']):g}: rejected by amorphous gate "
                            f"(pass_fraction={float(summ.get('pass_fraction', float('nan'))):.3g}, required={float(summ.get('required_pass_fraction', float('nan'))):.3g})"
                        )
                        if crit_txt:
                            msg += f"; {crit_txt}"
                        progress.warn("rate_scan", msg)
                    if len(rate_results_for_selection) == 0:
                        failure_message = (
                            "No cooling rates satisfied the amorphous acceptance gate. "
                            "Relax amorphous thresholds or increase quench rates."
                        )
                        _write_rate_scan_failure_snapshot(
                            outdir=outdir,
                            config=config,
                            pot_cfg=pot_cfg,
                            kim_install=kim_install,
                            preflight=preflight,
                            T=T,
                            D=D,
                            D_mu=D_mu,
                            D_se=D_se,
                            D_med=D_med,
                            tm_cfg=tm_cfg,
                            tm_summary=tm_summary,
                            tm_outcomes_all=tm_outcomes_all,
                            tm_est=tm_est,
                            time_unit_ps=time_unit_ps,
                            T_high=T_high,
                            high_total_steps=high_total_steps,
                            force_iso_active=force_iso_active,
                            high_cfg=high_cfg,
                            high_stationarity_summary=high_stationarity_summary,
                            high_rep_summaries=high_rep_summaries,
                            high_outcomes=high_outcomes,
                            melt_pool=melt_pool,
                            melt_data=melt_data,
                            rate_results=rate_results,
                            cutoffs_rate=cutoffs_rate,
                            metric_warnings=metric_warnings,
                            metrics_summary=metrics_summary,
                            failure_message=failure_message,
                            progress=progress,
                        )
                        raise ValueError(failure_message)

        # density convergence rate
        if len(rate_results_for_selection) >= 2:
            decision_rate_density = choose_fastest_converged(
                [rr["rate"] for rr in rate_results_for_selection],
                [rr["density_mean"] for rr in rate_results_for_selection],
                [rr["density_stderr"] for rr in rate_results_for_selection],
                rel_tol=config.autotune.convergence.density_rel_tol,
                abs_tol=config.autotune.convergence.density_abs_tol,
                z=config.autotune.convergence.zscore,
                kind="rate",
            )
        else:
            rr0 = rate_results_for_selection[0]
            decision_rate_density = {
                "kind": "rate",
                "chosen_index": 0,
                "chosen_value": float(rr0["rate"]),
                "reference_value": float(rr0["density_mean"]),
                "deltas": [0.0],
                "allowed": [float("nan")],
                "passed": [False],
                "accepted_subset": True,
                "selection_converged": False,
                "fallback_used": True,
                "selection_status": "single_candidate_unassessed",
                "selection_reason": "at least two scan points are required to assess convergence",
            }

        if metrics_cfg.enabled:
            # metric decision accepted
            if len(rate_results_for_selection) >= 2:
                x = [float(rr["rate"]) for rr in rate_results_for_selection]
                mu_maps: list[dict[str, float]] = []
                se_maps: list[dict[str, float]] = []
                for rr in rate_results_for_selection:
                    mu_map = {"density": float(rr["density_mean"])}
                    se_map = {"density": float(rr["density_stderr"])}
                    for k, v in rr.get("metrics_mean", {}).items():
                        mu_map[str(k)] = float(v)
                    for k, v in rr.get("metrics_stderr", {}).items():
                        se_map[str(k)] = float(v)
                    mu_maps.append(mu_map)
                    se_maps.append(se_map)

                decision_rate_multi = _multimetric_decision(x, mu_maps, se_maps, conv=config.autotune.convergence, kind="rate")
                chosen_rate = float(decision_rate_multi["chosen_value"])
            else:
                rr0 = rate_results_for_selection[0]
                decision_rate_multi = {
                    "kind": "rate",
                    "chosen_index": 0,
                    "chosen_value": float(rr0["rate"]),
                    "reference_index": 0,
                    "metrics": {},
                    "skipped_metrics": [],
                    "combined_passed": [False],
                    "accepted_subset": True,
                    "selection_converged": False,
                    "fallback_used": True,
                    "selection_status": "single_candidate_unassessed",
                    "selection_reason": "at least two scan points are required to assess convergence",
                }
                chosen_rate = float(rr0["rate"])
        else:
            decision_rate_multi = None
            chosen_rate = float(decision_rate_density["chosen_value"] if isinstance(decision_rate_density, dict) else decision_rate_density.chosen_value)

        # progress info size
        progress.info("size_scan", "starting size scan")
        # box scan density
        # size cfg
        size_cfg = config.autotune.size
        size_dir = outdir / "sizes"
        ensure_dir(size_dir)

        # construct cell production
        # anisotropic expansions misleading
        size_base_data, initial_repeat = prepare_size_scan_base_structure(config, outdir, initial_data)
        base_natoms = count_atoms_in_datafile(size_base_data)

        engine_name = str(getattr(config, "engine", "lammps")).strip().lower()
        size_enabled = bool(getattr(size_cfg, "enabled", False))
        if engine_name == "cp2k":
            size_scan_skipped = True
            size_scan_reason = "cp2k engine: size scan disabled"
        elif not size_enabled:
            size_scan_skipped = True
            size_scan_reason = "autotune.size.enabled=false"
        else:
            size_scan_skipped = False
            size_scan_reason = None

        # containers skipping
        size_results: list[dict[str, Any]] = []
        decision_size_density = None
        decision_size_multi = None
        cutoffs_size: dict[tuple[int, int], float] = {}

        if size_scan_skipped:
            # scans disabled expensive
            # structure size downstream
            chosen_replicate = [int(initial_repeat[0]), int(initial_repeat[1]), int(initial_repeat[2])]
        else:
            rx0, ry0, rz0 = (int(initial_repeat[0]), int(initial_repeat[1]), int(initial_repeat[2]))
            n0 = 1
            if rx0 == ry0 == rz0 and rx0 >= 1:
                n0 = int(rx0)
            else:
                # approximation scan isotropic
                prod_rep = max(1, int(rx0 * ry0 * rz0))
                n0 = int(max(1, round(float(prod_rep) ** (1.0 / 3.0))))

            # autotune configured isotropic
            # additional isotropic increasing
            max_atoms = int(getattr(size_cfg, "max_atoms", 0) or 0)

            def _natoms_for_n(n: int) -> int:
                return int(base_natoms) * int(n) * int(n) * int(n)

            # interpret yaml multipliers
            # enforce isotropic replication
            n_candidates: list[int] = []
            for r in list(size_cfg.replicas):
                try:
                    mx, my, mz = (int(r[0]), int(r[1]), int(r[2]))
                except Exception:
                    continue
                if mx < 1 or my < 1 or mz < 1:
                    continue
                if mx != my or mx != mz:
                    # isotropic replicas alternatives
                    continue
                n_candidates.append(int(n0 * mx))

            if int(n0) not in n_candidates:
                n_candidates.insert(0, int(n0))

            # filter count always
            if max_atoms > 0 and base_natoms > 0:
                max_n = int(math.floor((float(max_atoms) / float(base_natoms)) ** (1.0 / 3.0)))
                max_n = max(1, int(max_n))
                n_filtered: list[int] = []
                for n in n_candidates:
                    n = int(n)
                    if n == int(n0):
                        n_filtered.append(n)
                        continue
                    if n <= max_n and _natoms_for_n(n) <= int(max_atoms):
                        n_filtered.append(n)
                n_candidates = n_filtered

                # isotropic sizes total
                if len(set(n_candidates)) < 2 and max_n > int(n0):
                    for n in range(int(n0) + 1, int(max_n) + 1):
                        n_candidates.append(int(n))
                        if len(set(n_candidates)) >= 3:
                            break

            # isotropic scan sorted
            n_list = sorted(set(int(n) for n in n_candidates if int(n) >= 1))
            replicas = [(int(n), int(n), int(n)) for n in n_list]

            size_results = []
            for repfac in replicas:
                nx, ny, nz = repfac
                dens_reps: list[float] = []
                rep_entries: list[dict[str, Any]] = []
                n_rep_size = int(getattr(size_cfg, "replicates_per_size", 1))
                if n_rep_size < 1:
                    n_rep_size = 1
                for rep in range(n_rep_size):
                    stag = f"size_{nx}x{ny}x{nz}_rep{rep+1}"
                    sdir = size_dir / stag
                    ensure_dir(sdir)

                    # melt total replicate
                    seed = rng.randrange(1, 2**31 - 1)
                    melt_stage = StageSpec(
                        name="melt",
                        input_data=size_base_data,
                        output_data=sdir / "melt.data",
                        temperature_start=T_high,
                        temperature_stop=T_high,
                        pressure=float(md_use.pressure),
                        equil_steps=0,
                        run_steps=int(high_total_steps),
                        seed=int(seed),
                        force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
                        replicate=(nx, ny, nz),
                        write_dump=False,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    melt_out = _stage_run(
                        runner,
                        pot_cfg,
                        md_use,
                        melt_stage,
                        sdir / "melt",
                        potential_lines=potential_lines,
                        type_to_species=type_to_species,
                    )

                    # quench rate selected
                    dT = T_high - q_cfg.t_final
                    n_quench = quench_steps_for_rate(float(dT), float(chosen_rate), float(dt_mq), min_steps=1)

                    seed2 = rng.randrange(1, 2**31 - 1)
                    quench_stage = StageSpec(
                        name="quench",
                        input_data=(sdir / "melt" / melt_out.output_data),
                        output_data=sdir / "quench.data",
                        temperature_start=T_high,
                        temperature_stop=q_cfg.t_final,
                        pressure=float(md_use.pressure),
                        equil_steps=0,
                        run_steps=n_quench,
                        seed=int(seed2),
                        replicate=None,
                        write_dump=False,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    quench_out = _stage_run(
                        runner,
                        pot_cfg,
                        md_use,
                        quench_stage,
                        sdir / "quench",
                        potential_lines=potential_lines,
                        type_to_species=type_to_species,
                    )

                    seed3 = rng.randrange(1, 2**31 - 1)
                    mcfg = config.autotune.metrics
                    relax_steps = scale_steps_for_timestep(int(q_cfg.relax_steps), dt_ref, dt_mq, min_steps=1)
                    relax_stage = StageSpec(
                        name="relax",
                        input_data=(sdir / "quench" / quench_out.output_data),
                        output_data=sdir / "final.data",
                        temperature_start=q_cfg.t_final,
                        temperature_stop=q_cfg.t_final,
                        pressure=float(md_use.pressure),
                        equil_steps=0,
                        run_steps=int(relax_steps),
                        seed=int(seed3),
                        replicate=None,
                        write_dump=True,
                        dump_every=int(relax_steps) if not mcfg.enabled else None,
                        tail_dump_frames=int(mcfg.time_average_frames) if mcfg.enabled else None,
                        tail_dump_stride=int(mcfg.time_average_stride) if mcfg.enabled else None,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    relax_dir = sdir / "relax"
                    relax_out = _stage_run(
                        runner,
                        pot_cfg,
                        md_use,
                        relax_stage,
                        relax_dir,
                        potential_lines=potential_lines,
                        type_to_species=type_to_species,
                    )
                    melt_elastic = _maybe_elastic(
                        "melt",
                        stage_dir=sdir / "melt",
                        structure_data=(sdir / "melt" / melt_out.output_data),
                        input_data=Path(size_base_data),
                        force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                    )
                    relax_elastic = _maybe_elastic(
                        "relax",
                        stage_dir=relax_dir,
                        structure_data=(relax_dir / relax_out.output_data),
                        input_data=(sdir / "quench" / quench_out.output_data),
                        force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                    )
                    dens_reps.append(relax_out.density_mean)
                    rep_entries.append(
                        {
                            "density": float(relax_out.density_mean),
                            "n_atoms": int(_natoms_for_n(int(nx))) if base_natoms > 0 else None,
                            "final_data": str((relax_dir / relax_out.output_data).relative_to(outdir))
                            if (relax_dir / relax_out.output_data).is_relative_to(outdir)
                            else str(relax_dir / relax_out.output_data),
                            "dump": str((relax_dir / f"{relax_stage.name}.lammpstrj").relative_to(outdir))
                            if (relax_dir / f"{relax_stage.name}.lammpstrj").is_relative_to(outdir)
                            else str(relax_dir / f"{relax_stage.name}.lammpstrj"),
                            "elastic_melt": melt_elastic,
                            "elastic_relax": relax_elastic,
                        }
                    )

                mu, se = _complete_mean_stderr(dens_reps)
                size_results.append(
                    {
                        "replicate": [nx, ny, nz],
                        "multiplier": int(nx * ny * nz),
                        "n_atoms": int(_natoms_for_n(int(nx))) if base_natoms > 0 else None,
                        "density_mean": mu,
                        "density_stderr": se,
                        "nrep": len(dens_reps),
                        "replicates": rep_entries,
                    }
                )

            # decide converged point
            # cell already atoms
            decision_size_density = None
            decision_size_multi = None
            cutoffs_size = {}

            if not size_results:
                chosen_multiplier = 1.0
                chosen_replicate = [int(n0), int(n0), int(n0)]
            elif len(size_results) < 2:
                chosen_multiplier = float(size_results[0].get("n_atoms") or size_results[0]["multiplier"])
                chosen_replicate = size_results[0]["replicate"]
            else:
                x_size = [float(sr.get("n_atoms") or sr["multiplier"]) for sr in size_results]
                decision_size_density = choose_fastest_converged(
                    x_size,
                    [sr["density_mean"] for sr in size_results],
                    [sr["density_stderr"] for sr in size_results],
                    rel_tol=config.autotune.convergence.density_rel_tol,
                    abs_tol=config.autotune.convergence.density_abs_tol,
                    z=config.autotune.convergence.zscore,
                    kind="size",
                )

                if metrics_cfg.enabled:
                    required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
                    fixed_cut = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)

                    cutoffs_size, frames_by_path = _estimate_pooled_scan_cutoffs(
                        size_results,
                        outdir=outdir,
                        metrics_cfg=metrics_cfg,
                        required_pairs=required_pairs,
                        fixed_cutoffs=fixed_cut,
                        type_to_species=type_to_species,
                        lammps_units_style=analysis_lammps_units_style,
                    )

                    for sr in size_results:
                        for rep_entry in sr["replicates"]:
                            traj_path = _resolve_replicate_traj_path(outdir=outdir, rep_entry=rep_entry)
                            frames = frames_by_path.get(str(traj_path))
                            if frames is None:
                                frames = read_last_frames_auto(
                                    traj_path,
                                    int(metrics_cfg.time_average_frames),
                                    type_to_species=type_to_species,
                                    units_style=analysis_lammps_units_style,
                                )
                                frames_by_path[str(traj_path)] = frames
                            sm = compute_structure_metrics_timeavg(
                                frames,
                                metrics_cfg,
                                cutoffs=cutoffs_size,
                                type_to_species=type_to_species,
                            )
                            rep_entry["metrics"] = dict(sm.values)

                    for sr in size_results:
                        rep_metrics = [re.get("metrics", {}) for re in sr["replicates"]]
                        mu_m, se_m = _aggregate_scalar_metrics(rep_metrics)
                        sr["metrics_mean"] = mu_m
                        sr["metrics_stderr"] = se_m

                    x = [float(sr.get("n_atoms") or sr["multiplier"]) for sr in size_results]
                    mu_maps = []
                    se_maps = []
                    for sr in size_results:
                        mu_map = {"density": float(sr["density_mean"])}
                        se_map = {"density": float(sr["density_stderr"])}
                        for k, v in sr.get("metrics_mean", {}).items():
                            mu_map[str(k)] = float(v)
                        for k, v in sr.get("metrics_stderr", {}).items():
                            se_map[str(k)] = float(v)
                        mu_maps.append(mu_map)
                        se_maps.append(se_map)

                    decision_size_multi = _multimetric_decision(x, mu_maps, se_maps, conv=config.autotune.convergence, kind="size")
                    chosen_multiplier = float(decision_size_multi["chosen_value"])
                else:
                    chosen_multiplier = float(decision_size_density.chosen_value)

                chosen_replicate = None
                for sr in size_results:
                    xval = float(sr.get("n_atoms") or sr["multiplier"])
                    if float(xval) == float(chosen_multiplier):
                        chosen_replicate = sr["replicate"]
                        break
                if chosen_replicate is None:
                    chosen_replicate = size_results[-1]["replicate"]
        # rate conversion lammps
        cooling_rate_ps = None
        if time_unit_ps is not None:
            cooling_rate_ps = float(chosen_rate) / float(time_unit_ps)

        def _build_results(*, production_state: dict[str, Any], status: str) -> dict[str, Any]:
            def _selection_summary(decision: Any, *, skipped: bool = False) -> dict[str, Any]:
                if skipped:
                    return {
                        "status": "not_applicable",
                        "converged": None,
                        "fallback_used": None,
                    }
                if decision is None:
                    return {
                        "status": "single_candidate_unassessed",
                        "converged": False,
                        "fallback_used": True,
                    }
                if isinstance(decision, Mapping):
                    converged_selection = bool(decision.get("selection_converged", False))
                    fallback = bool(decision.get("fallback_used", not converged_selection))
                    decision_status = str(
                        decision.get(
                            "selection_status",
                            "converged" if converged_selection else "fallback_unconverged",
                        )
                    )
                else:
                    converged_selection = bool(getattr(decision, "selection_converged", False))
                    fallback = bool(getattr(decision, "fallback_used", not converged_selection))
                    decision_status = str(
                        getattr(
                            decision,
                            "selection_status",
                            "converged" if converged_selection else "fallback_unconverged",
                        )
                    )
                return {
                    "status": decision_status,
                    "converged": bool(converged_selection),
                    "fallback_used": bool(fallback),
                }

            rate_selection_source = (
                decision_rate_multi if decision_rate_multi is not None else decision_rate_density
            )
            size_selection_source = (
                decision_size_multi if decision_size_multi is not None else decision_size_density
            )
            rate_selection_summary = _selection_summary(rate_selection_source)
            size_selection_summary = _selection_summary(
                size_selection_source,
                skipped=bool(size_scan_skipped),
            )
            production_enabled = bool(production_state.get("enabled", False))
            ensemble_stopping_assessed = bool(
                production_enabled
                and production_state.get("check_convergence", False)
            )
            ensemble_diagnostics = (
                production_state.get("convergence", {}) if production_enabled else None
            )
            ensemble_posthoc_assessed = bool(
                production_enabled
                and not ensemble_stopping_assessed
                and isinstance(ensemble_diagnostics, Mapping)
                and ensemble_diagnostics.get("assessment_performed", False)
                and ensemble_diagnostics.get("assessment_role")
                == "terminal_posthoc_diagnostic"
            )
            ensemble_diagnostic_assessed = bool(
                ensemble_stopping_assessed or ensemble_posthoc_assessed
            )
            inference_contract = (
                dict((ensemble_diagnostics or {}).get("inference_contract", {}) or {})
                if isinstance(ensemble_diagnostics, Mapping)
                else {}
            )
            sequentially_valid = (
                inference_contract.get("sequentially_valid")
                if ensemble_diagnostic_assessed
                else None
            )
            ensemble_converged = (
                bool(production_state.get("converged", False))
                if ensemble_stopping_assessed
                else None
            )
            achieved_degree_raw = (
                (ensemble_diagnostics or {}).get("achieved_convergence_degree")
                if isinstance(ensemble_diagnostics, Mapping)
                else None
            )
            achieved_degree = (
                dict(achieved_degree_raw)
                if isinstance(achieved_degree_raw, Mapping)
                else None
            )
            if achieved_degree is not None:
                achieved_degree.update(
                    {
                        "n_boxes": int(production_state.get("n_boxes", 0) or 0),
                        "convergence_streak": int(
                            production_state.get("convergence_streak", 0) or 0
                        ),
                        "required_convergence_streak": int(
                            production_state.get("required_convergence_streak", 0) or 0
                        ),
                    }
                )
            ensemble_inference_status = production_state.get(
                "convergence_inference_status"
            )
            if not isinstance(ensemble_inference_status, str) or not str(
                ensemble_inference_status
            ).strip():
                ensemble_inference_status = (
                    "not_applicable"
                    if not production_enabled
                    else (
                        "fixed_count_unassessed"
                        if not ensemble_stopping_assessed
                        else (
                            "criterion_met_repeated_looks_not_sequentially_valid"
                            if ensemble_converged
                            else "criterion_not_met_or_unassessed_repeated_looks_not_sequentially_valid"
                        )
                    )
                )
            criterion_coverage_raw = production_state.get(
                "convergence_criterion_coverage"
            )
            criterion_coverage = (
                {
                    str(key): dict(value)
                    for key, value in criterion_coverage_raw.items()
                    if isinstance(value, Mapping)
                }
                if isinstance(criterion_coverage_raw, Mapping)
                else None
            )
            ensemble_convergence = {
                "status": (
                    "not_applicable"
                    if not production_enabled
                    else (
                        "fixed_count_unassessed"
                        if not ensemble_stopping_assessed
                        else (
                            "criterion_met_repeated_looks_not_sequentially_valid"
                            if ensemble_converged and sequentially_valid is False
                            else ("converged" if ensemble_converged else "not_converged")
                        )
                    )
                ),
                # Additive inference-qualified label.  ``status`` above is
                # retained for compatibility with existing recommendation
                # consumers, including its older converged/not_converged
                # tokens.
                "convergence_inference_status": str(
                    ensemble_inference_status
                ),
                "converged": ensemble_converged,
                "sequentially_valid": sequentially_valid,
                "assessment_role": (
                    ensemble_diagnostics.get("assessment_role")
                    if isinstance(ensemble_diagnostics, Mapping)
                    else None
                ),
                "interval_method": (
                    inference_contract.get("interval_method")
                    if ensemble_diagnostic_assessed
                    else None
                ),
                "inference_contract": inference_contract if production_enabled else None,
                "achieved_convergence_degree": achieved_degree,
                "convergence_criterion_coverage": criterion_coverage,
                "posthoc_criterion_met": (
                    ensemble_diagnostics.get("posthoc_criterion_met")
                    if ensemble_posthoc_assessed
                    and isinstance(ensemble_diagnostics, Mapping)
                    else None
                ),
                "posthoc_failed_items": (
                    list(ensemble_diagnostics.get("posthoc_failed_items", []) or [])
                    if ensemble_posthoc_assessed
                    and isinstance(ensemble_diagnostics, Mapping)
                    else None
                ),
                "md_converged": (
                    production_state.get("converged_md")
                    if ensemble_stopping_assessed
                    else None
                ),
                "dft_converged": (
                    production_state.get("converged_dft") if production_enabled else None
                ),
                "n_boxes": (
                    int(production_state.get("n_boxes", 0) or 0) if production_enabled else None
                ),
                "convergence_streak": (
                    int(production_state.get("convergence_streak", 0) or 0)
                    if production_enabled
                    else None
                ),
                "required_convergence_streak": (
                    int(production_state.get("required_convergence_streak", 0) or 0)
                    if production_enabled
                    else None
                ),
                "diagnostics": ensemble_diagnostics,
            }
            return {
                "status": str(status),
                "execution_status": (
                    "completed"
                    if str(status) in {"ok", "incomplete", "not_converged"}
                    else str(status)
                ),
                "units": {
                    "engine": str(getattr(config, "engine", "lammps")),
                    "lammps_units": resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal"),
                    "time_unit_ps": float(time_unit_ps) if time_unit_ps is not None else None,
                },
                "kim_install": _kim_install_jsonable(kim_install),
                "preflight": asdict(preflight),
                "tm_scan": {
                    "temps": [float(t) for t in T],
                    "replicates_per_temp": int(getattr(tm_cfg, 'replicates_per_temp', 1)),
                    "D": [float(x) for x in D],
                    "D_mean": [float(x) for x in D_mu],
                    "D_stderr": [float(x) for x in D_se],
                    "D_median": [float(x) for x in D_med],
                    "summary": tm_summary,
                    "outcomes": [asdict(o) for o in tm_outcomes_all],
                    "Tm_estimate": {
                        "Tm": float(tm_est.Tm),
                        "T_liquid": float(getattr(tm_est, "T_liquid", float("nan"))),
                        "D_liquid_target": float(getattr(tm_est, "D_liquid_target", float("nan"))),
                        "method": str(tm_est.method),
                        "score": float(tm_est.score),
                        "idx": int(tm_est.idx),
                    },
                },
                "highT": {
                    "T_high": float(T_high),
                    "total_steps": int(high_total_steps),
                    "force_isotropic_extension_factor": 1.5 if bool(force_iso_active) else 1.0,
                    "replicates": int(getattr(high_cfg, 'replicates', 1)),
                    "stationarity": high_stationarity_summary,
                    "rep_summaries": high_rep_summaries,
                    "outcomes": [asdict(o) for o in high_outcomes],
                    "melt_pool": [
                        str(Path(p).relative_to(outdir)) if Path(p).is_relative_to(outdir) else str(p)
                        for p in melt_pool
                    ],
                    "melt_data": str(Path(melt_data).relative_to(outdir)) if Path(melt_data).is_relative_to(outdir) else str(melt_data),
                },
                "rate_scan": {
                    "rates": rate_results,
                    "decision_density": (dict(decision_rate_density) if isinstance(decision_rate_density, dict) else asdict(decision_rate_density)),
                    "decision_multi": decision_rate_multi,
                    "cutoffs": [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(cutoffs_rate.items())],
                },
                "size_scan": {
                    "skipped": bool(size_scan_skipped),
                    "skip_reason": str(size_scan_reason) if size_scan_reason is not None else None,
                    "base_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
                    "base_natoms": int(base_natoms),
                    "initial_repeat": [int(initial_repeat[0]), int(initial_repeat[1]), int(initial_repeat[2])],
                    "sizes": size_results,
                    "decision_density": asdict(decision_size_density) if decision_size_density is not None else None,
                    "decision_multi": decision_size_multi,
                    "cutoffs": [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(cutoffs_size.items())],
                },
                "production": production_state,
                "production_plan": production_plan,
                "recommendation": {
                    "T_high": T_high,
                    "Tm_operational": tm_est.Tm,
                    "T_liquid": float(getattr(tm_est, "T_liquid", float("nan"))),
                    "highT_steps": high_total_steps,
                    "force_isotropic_extension_factor": 1.5 if bool(force_iso_active) else 1.0,
                    "cooling_rate_K_per_time": chosen_rate,
                    "cooling_rate_K_per_ps": cooling_rate_ps,
                    "replicate": chosen_replicate,
                    "structure_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
                    "t_final": q_cfg.t_final,
                    "pressure": config.md.pressure,
                    "md": {
                        "ensemble": str(md_use.ensemble),
                        "timestep": float(md_use.timestep),
                        "atom_style": str(md_use.atom_style),
                        "force_isotropic": bool(getattr(md_use, "force_isotropic", False)),
                        "thermostat": {"style": str(md_use.thermostat.style), "tdamp": float(md_use.thermostat.tdamp)},
                        "barostat": {"style": str(md_use.barostat.style), "pdamp": float(md_use.barostat.pdamp)},
                    },
                    "core_repulsion": asdict(preflight.core_repulsion),
                    # The recommendation remains usable when a finite scan is
                    # exhausted, but its epistemic status travels with it.
                    "selection_convergence": {
                        "rate": rate_selection_summary,
                        "size": size_selection_summary,
                        "all_applicable_converged": bool(
                            bool(rate_selection_summary.get("converged", False))
                            and (
                                size_selection_summary.get("converged") is None
                                or bool(size_selection_summary.get("converged", False))
                            )
                        ),
                    },
                    "final_ensemble_convergence": ensemble_convergence,
                },
                "metric_warnings": list(metric_warnings),
                "effective_metrics": dict(metrics_summary),
                "resume_fingerprint": dict(resume_fingerprint),
                "paths": {
                    "autotune_results": "autotune_results.json",
                    "autotune": "autotune.json",
                    "condensed_log": "condensed.log",
                    **dict((production_state.get("graph_outputs", {}) if isinstance(production_state, dict) else {}) or {}),
                },
            }

        prod_cfg = getattr(config.autotune, "production", None)
        pre_n_quench_prod = quench_steps_for_rate(float(T_high - q_cfg.t_final), float(chosen_rate), float(dt_mq), min_steps=1)
        pre_relax_steps_prod = scale_steps_for_timestep(int(q_cfg.relax_steps), float(dt_ref), float(dt_mq), min_steps=1)
        pre_prod_cutoffs = dict(cutoffs_size) if len(cutoffs_size) > 0 else dict(cutoffs_rate)
        production_plan = production_plan_to_dict(
            make_production_plan(
                engine=str(getattr(config, "engine", "lammps")),
                structure_data=Path(size_base_data),
                T_high=float(T_high),
                high_total_steps=int(high_total_steps),
                t_final=float(q_cfg.t_final),
                chosen_rate=float(chosen_rate),
                cooling_rate_ps=(None if cooling_rate_ps is None else float(cooling_rate_ps)),
                replicate=chosen_replicate,
                pressure=float(md_use.pressure),
                md_use=md_use.model_dump(mode="json"),
                potential_config=(pot_cfg.model_dump(mode="json") if hasattr(pot_cfg, "model_dump") else None),
                potential_lines=potential_lines,
                core_repulsion=asdict(preflight.core_repulsion),
                type_to_species=type_to_species,
                metrics_cfg=metrics_cfg.model_dump(mode="json"),
                effective_metrics=dict(metrics_summary),
                production_cfg=(prod_cfg.model_dump(mode="json") if prod_cfg is not None else {}),
                convergence_cfg=config.autotune.convergence.model_dump(mode="json"),
                cutoffs_rate=cutoffs_rate,
                cutoffs_size=cutoffs_size,
                preferred_cutoffs=pre_prod_cutoffs,
                quench_steps=int(pre_n_quench_prod),
                relax_steps=int(pre_relax_steps_prod),
                msd_every=int(tm_cfg.msd_every),
                seed_base=int(config.random_seed) + 13579,
                time_unit_ps=(None if time_unit_ps is None else float(time_unit_ps)),
                sampling_hint=None,
                execution_mode=("adaptive" if bool(prod_cfg is not None and getattr(prod_cfg, "check_convergence", True)) else "fixed"),
                source_kind="autotune",
            ),
            relative_to=outdir,
        )
        current_engine_build_identities = query_engine_build_identities(
            config,
            workdir=outdir,
            primary_engine=str(config.engine),
            include_cp2k_refinement=dft_identity_enabled,
        )
        validate_engine_build_identity_bundle(current_engine_build_identities)
        if (
            current_engine_build_identities.get("identity_sha256")
            != engine_build_identities.get("identity_sha256")
        ):
            raise RuntimeError(
                "The configured engine build changed during autotune before the "
                "production plan was protected; refusing to combine recommendations "
                "from heterogeneous engine builds"
            )
        if _config_input_identities(config, workdir=outdir) != initial_config_input_identities:
            raise RuntimeError(
                "Scientific input bytes changed during autotune before the "
                "production plan was protected; refusing recommendations that "
                "could combine different potentials, command includes, structures, "
                "or CP2K data"
            )
        resume_fingerprint = _build_autotune_resume_fingerprint(
            config=config,
            outdir=outdir,
            selected_structure=Path(size_base_data),
            production_plan=production_plan,
            engine_build_identities=engine_build_identities,
        )
        preprod_warmup_start_temperature = 300.0
        preprod_warmup_duration_ps = 5.0
        preprod_warmup_steps = None
        if prod_cfg is not None and bool(getattr(prod_cfg, "enabled", False)):
            preprod_warmup_start_temperature = resolve_production_warmup_start_temperature(
                prod_cfg=prod_cfg,
                T_high=float(T_high),
            )
            preprod_warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
            preprod_warmup_steps = resolve_production_warmup_steps(
                prod_cfg=prod_cfg,
                md_timestep=float(dt_mq),
                time_unit_ps=resolve_production_time_unit_ps(
                    config=config,
                    engine=str(getattr(config, "engine", "lammps") or "lammps"),
                    pot_cfg=pot_cfg,
                    time_unit_ps=time_unit_ps,
                ),
            )

        preprod_enabled = bool(
            prod_cfg is not None and bool(getattr(prod_cfg, "enabled", False))
        )
        preprod_initial_status = _initial_production_checkpoint_status(
            enabled=preprod_enabled
        )
        preprod_state = {
            "enabled": preprod_enabled,
            "status": preprod_initial_status,
            "execution_status": preprod_initial_status,
            "error": None,
            "converged": False,
            "n_boxes": 0,
            "n_boxes_accepted": 0,
            "n_boxes_rejected": 0,
            "n_boxes_total": 0,
            "min_boxes": int(getattr(prod_cfg, "min_boxes", 0)) if prod_cfg is not None else 0,
            "max_boxes": (int(getattr(prod_cfg, "max_boxes", 0)) if (prod_cfg is not None and getattr(prod_cfg, "max_boxes", None) not in (None, 0)) else None),
            "batch_boxes": int(getattr(prod_cfg, "batch_boxes", 1)) if prod_cfg is not None else 1,
            "check_convergence": bool(getattr(prod_cfg, "check_convergence", True)) if prod_cfg is not None else False,
            "resumable": True,
            "non_resumable_reason": None,
            "convergence_streak": 0,
            "required_convergence_streak": max(
                1,
                int(getattr(prod_cfg, "consecutive_converged_checks", 1)),
            ) if prod_cfg is not None else 1,
            "last_convergence_evaluated_n_boxes_total": None,
            "last_convergence_evaluated_n_boxes_accepted": None,
            "dump_trajectory": bool(getattr(prod_cfg, "dump_trajectory", True)) if prod_cfg is not None else False,
            "dump_every_steps": int(getattr(prod_cfg, "dump_every_steps", 5000) or 5000) if prod_cfg is not None else 5000,
            "rate_K_per_time": float(chosen_rate),
            "rate_K_per_ps": float(cooling_rate_ps) if cooling_rate_ps is not None else None,
            "replicate": [int(x) for x in chosen_replicate],
            "structure_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
            "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)) if prod_cfg is not None else False,
            "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")) if prod_cfg is not None else None,
            "rejects_dir": None,
            "warmup_start_temperature": float(preprod_warmup_start_temperature),
            "warmup_duration_ps": float(preprod_warmup_duration_ps),
            "warmup_steps": (int(preprod_warmup_steps) if preprod_warmup_steps is not None else None),
            "T_high": float(T_high),
            "t_final": float(q_cfg.t_final),
            "quench_steps": int(pre_n_quench_prod),
            "highT_steps": int(high_total_steps),
            "relax_steps": int(pre_relax_steps_prod),
            "cutoffs": ([{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(pre_prod_cutoffs.items())] if len(pre_prod_cutoffs) > 0 else None),
            "metrics_checked": None,
            "convergence_spec": None,
            "converged_md": False,
            "convergence_md": {},
            "converged_dft": None,
            "convergence_dft": None,
            "convergence": {},
            "dft_opt": None,
            "boxes_dft_final": None,
            "n_boxes_dft_accepted": None,
            "rejected_boxes_dft": None,
            "boxes": [],
            "rejected_boxes": [],
            "graph_outputs": {},
            "paths": {},
            "ensemble_dir": "production",
        }
        preprod_state = _attach_production_state_integrity(preprod_state, outdir=outdir)

        write_autotune_outputs(outdir, _build_results(production_state=preprod_state, status="running"))

        def _checkpoint_production(prod_state: dict[str, Any]) -> None:
            write_autotune_outputs(outdir, _build_results(production_state=dict(prod_state), status="running"))

        progress.info("production", "starting production ensemble")
        production = _run_production_ensemble(
            config=config,
            outdir=outdir,
            runner=runner,
            pot_cfg=pot_cfg,
            md_use=md_use,
            potential_lines=potential_lines,
            type_to_species=type_to_species,
            metrics_cfg=metrics_cfg,
            tm_cfg=tm_cfg,
            q_cfg=q_cfg,
            size_base_data=Path(size_base_data),
            chosen_replicate=[int(x) for x in chosen_replicate],
            chosen_rate=float(chosen_rate),
            dt_ref=float(dt_ref),
            dt_mq=float(dt_mq),
            cooling_rate_ps=(float(cooling_rate_ps) if cooling_rate_ps is not None else None),
            cutoffs_rate=dict(cutoffs_rate),
            cutoffs_size=dict(cutoffs_size),
            T_high=float(T_high),
            high_total_steps=int(high_total_steps),
            resume_state=None,
            progress=progress,
            checkpoint_cb=_checkpoint_production,
            time_unit_ps_override=(None if time_unit_ps is None else float(time_unit_ps)),
        )

        final_engine_build_identities = query_engine_build_identities(
            config,
            workdir=outdir,
            primary_engine=str(config.engine),
            include_cp2k_refinement=dft_identity_enabled,
        )
        assert_engine_build_identity_bundle_unchanged(
            engine_build_identities,
            final_engine_build_identities,
            context="during autotune execution",
        )
        terminal_fingerprint = _build_autotune_resume_fingerprint(
            config=config,
            outdir=outdir,
            selected_structure=Path(size_base_data),
            production_plan=production_plan,
            engine_build_identities=final_engine_build_identities,
        )
        _assert_autotune_terminal_fingerprint_unchanged(
            resume_fingerprint,
            terminal_fingerprint,
            context="execution",
        )

        results = _build_results(
            production_state=production,
            status=_autotune_workflow_status(production),
        )
        write_autotune_outputs(outdir, results)
        return results

@locked_output_workflow("autotune workflow")
def autotune(config: RunConfig, outdir: Path, *, resume: bool | None = None) -> dict[str, Any]:
    """Autotune."""
    return _AutotuneWorkflow(
        config=config,
        outdir=outdir,
        resume=resume,
    ).run()
