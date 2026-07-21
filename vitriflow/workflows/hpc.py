"""External/HPC-compatible production task orchestration.

The external layer keeps the *planning* and *analysis* steps in vitriflow while
allowing production boxes to be executed independently from task manifests.

Modes
-----
- dry-run: create box directories, input snapshots, task manifests, preview
  inputs, and optional Slurm submission scripts without executing MD.
- full-run: create task manifests and execute them in batched waves, then run
  the generic output-analysis pipeline to compute metrics and convergence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from numbers import Integral
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import TypeAdapter

from ..config import MDConfig, PotentialConfig, ProductionEnsembleConfig, RunConfig, StructureMetricsConfig
from ..engine_identity import (
    assert_homogeneous_engine_build_identities,
    homogeneous_successful_task_engine_identity,
    query_engine_build_identities,
    validate_engine_build_identity,
)
from ..kim import ensure_model_installed, ensure_potential_model_installed
from ..lammps_input import StageSpec, render_continuous_stages, render_stage
from ..potential import (
    _atomic_copy_verified_regular_file,
    prepare_potential_files,
    validated_tabulated_core_path,
)
from ..runner import Cp2kRunner, LammpsRunner
from ..runtime_identity import runtime_identity
from ..utils import ensure_dir, stable_file_identity
from ..analysis.provenance import json_sanitize
from .metric_requirements import fixed_cutoffs_from_metrics, required_pairs_from_metrics
from .output_analysis import analyze_output_data
from .elastic_screen import (
    run_elastic_screen_lammps,
    run_elastic_screen_timeseries_lammps,
    should_collect_elastic_stage_timeseries,
    should_run_elastic_screen,
)
from .production_common import (
    plan_production_stage_diagnostics,
    production_plan_to_dict,
    resolve_production_relax_dump_settings,
    resolve_production_time_unit_ps,
    resolve_production_warmup_duration_ps,
    resolve_production_warmup_start_temperature,
    resolve_production_warmup_steps,
)
from .stage_metrics import (
    collect_stage_metrics_timeseries,
    should_collect_stage_metrics_timeseries,
)
from .progress import CondensedProgressLog, atomic_write_json
from .resume_integrity import (
    TASK_RESULT_SCHEMA,
    potential_command_file_paths,
    production_final_status,
    seal_task_result,
    task_manifest_sha256,
    validate_task_result_integrity,
)
from .stage_runner import run_stage_local, run_stages_continuous_lammps, stage_outcome_from_artifacts
from .workflow_lock import exclusive_workflow_lock


_POTENTIAL_ADAPTER = TypeAdapter(PotentialConfig)
_HARD_MAX_EXTERNAL_BOXES = 10_000


def _prefix_analysis_output_path(value: str) -> str:
    path = Path(str(value))
    if ".." in path.parts:
        raise ValueError(f"External analysis path escapes its production root: {value!r}")
    if path.is_absolute() or path.parts[:1] == ("production",):
        return str(path)
    return str(Path("production") / path)


def _is_declared_path_key(key: str) -> bool:
    name = str(key).strip().lower()
    return name in {
        "analysis_source",
        "box_dir",
        "database",
        "input_data",
        "relax_data",
        "relax_dump",
        "relax_traj",
        "structure_manifest",
        "structure_snapshot",
    } or name.endswith(
        (
            "_data",
            "_dir",
            "_dump",
            "_extxyz",
            "_json",
            "_manifest",
            "_snapshot",
            "_traj",
        )
    )


def _normalise_analysis_entry_paths(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix only fields whose path-map key declares path semantics."""

    entry = deepcopy(dict(raw))
    paths = entry.get("paths")

    def visit(value: Any, *, key: str) -> Any:
        if isinstance(value, Mapping):
            return {
                str(child_key): visit(child, key=str(child_key))
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [visit(child, key=key) for child in value]
        if isinstance(value, str) and value.strip() and _is_declared_path_key(key):
            return _prefix_analysis_output_path(value)
        return value

    if isinstance(paths, Mapping):
        entry["paths"] = {
            str(key): visit(value, key=str(key)) for key, value in paths.items()
        }

    def normalise_provenance_mapping(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        out = deepcopy(dict(value))
        for path_key in ("source_path", "manifest_sidecar", "structure_reference"):
            path_value = out.get(path_key)
            if isinstance(path_value, str) and path_value.strip():
                out[path_key] = _prefix_analysis_output_path(path_value)
        for identity_key in ("source_file_identity", "source_file_verification"):
            identity = out.get(identity_key)
            if isinstance(identity, Mapping):
                identity_out = dict(identity)
                identity_path = identity_out.get("path")
                if isinstance(identity_path, str) and identity_path.strip():
                    identity_out["path"] = _prefix_analysis_output_path(identity_path)
                out[identity_key] = identity_out
        return out

    for provenance_key in (
        "structure_manifest",
        "structure",
        "structure_embedding",
    ):
        if isinstance(entry.get(provenance_key), Mapping):
            entry[provenance_key] = normalise_provenance_mapping(
                entry[provenance_key]
            )
    return entry


def _normalise_analysis_graph_outputs(value: Any) -> Any:
    """Graph-output mappings are schema-declared path trees."""

    if isinstance(value, Mapping):
        return {
            str(key): _normalise_analysis_graph_outputs(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_normalise_analysis_graph_outputs(child) for child in value]
    if isinstance(value, str) and value.strip():
        return _prefix_analysis_output_path(value)
    return value


def _runtime_identity() -> dict[str, Any]:
    return runtime_identity()


def _task_runtime_is_current(task_data: Mapping[str, Any]) -> bool:
    runtime = task_data.get("runtime", None)
    return isinstance(runtime, Mapping) and dict(runtime) == _runtime_identity()


def _potential_from_dict(data: Optional[Mapping[str, Any]], fallback: Any) -> Any:
    if not isinstance(data, Mapping) or len(data) == 0:
        return fallback
    return _POTENTIAL_ADAPTER.validate_python(dict(data))


def _task_engine_build_identity(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
    workdir: Path,
) -> dict[str, Any]:
    """Probe the exact primary engine used by one external worker task."""

    engine = str(plan.get("engine", config.engine)).strip().lower()
    bundle = query_engine_build_identities(
        config,
        workdir=workdir,
        primary_engine=engine,
        include_cp2k_refinement=False,
    )
    identities = bundle.get("engines", {})
    if not isinstance(identities, Mapping) or not isinstance(
        identities.get(engine), Mapping
    ):
        raise RuntimeError(
            f"No verified worker build identity was produced for engine {engine!r}"
        )
    return validate_engine_build_identity(
        identities[engine], expected_engine=engine
    )


def _assert_task_engine_build_identity_unchanged(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    initial_checked = validate_engine_build_identity(initial)
    final_checked = validate_engine_build_identity(
        final,
        expected_engine=str(initial_checked.get("engine", "")),
    )
    initial_digest = str(initial_checked.get("identity_sha256", ""))
    final_digest = str(final_checked.get("identity_sha256", ""))
    if final_digest != initial_digest:
        raise RuntimeError(
            "Worker engine build changed during production-box execution; "
            "refusing to seal a mixed-build success "
            f"(initial={initial_digest}, final={final_digest})"
        )
    return final_checked


def _relpath_or_str(path: Path | None, base: Path) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    b = Path(base)
    try:
        return str(p.relative_to(b))
    except Exception:
        return str(p)


def _task_manifest_digest(task_data: Mapping[str, Any]) -> str:
    return task_manifest_sha256(task_data)


def _strict_integral_box_id(value: Any, *, context: str) -> int:
    """Parse an exact, non-boolean, positive integer box id."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(f"{context} has a non-integral box id {value!r}")
    box_id = int(value)
    if box_id < 1:
        raise RuntimeError(f"{context} has a non-positive box id {box_id}")
    return box_id


def _strict_nonnegative_integer(value: Any, *, context: str) -> int:
    """Parse an exact JSON-style integer used by protected resume state."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(f"{context} must be a non-boolean integer")
    parsed = int(value)
    if parsed < 0:
        raise RuntimeError(f"{context} must be non-negative")
    return parsed


def _accepted_evidence_identity(
    analysis: Mapping[str, Any],
) -> tuple[list[int], str]:
    """Identify the exact accepted ensemble used for one convergence look.

    Counts alone are insufficient resume evidence: a rejected box can be
    replaced by another accepted box without changing the count.  Bind every
    look to the canonical accepted entries, including their structure and
    metric evidence, so an interrupted external run cannot count a changed or
    repeated ensemble as a new consecutive look.
    """

    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in list(analysis.get("boxes", []) or []):
        if not isinstance(raw, Mapping):
            raise RuntimeError("External analysis returned a non-mapping accepted box")
        box_id = _strict_integral_box_id(
            raw.get("box", raw.get("box_id", 0)),
            context="External analysis",
        )
        if box_id in seen:
            raise RuntimeError(
                f"External analysis returned a duplicate accepted box id {box_id}"
            )
        seen.add(box_id)
        entries.append({"box": box_id, "entry": json_sanitize(dict(raw))})
    entries.sort(key=lambda row: int(row["box"]))
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return [int(row["box"]) for row in entries], hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _validated_convergence_look_history(
    value: Any,
) -> list[dict[str, Any]]:
    """Validate persisted external convergence looks before resume."""

    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise RuntimeError(
            "Cannot safely resume external production: convergence look history is malformed"
        )
    out: list[dict[str, Any]] = []
    prior_total = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "Cannot safely resume external production: convergence look history is malformed"
            )
        try:
            total = _strict_nonnegative_integer(
                raw.get("n_boxes_total"),
                context=f"External convergence look history record {index} n_boxes_total",
            )
            accepted = _strict_nonnegative_integer(
                raw.get("n_boxes_accepted"),
                context=f"External convergence look history record {index} n_boxes_accepted",
            )
            streak_after = _strict_nonnegative_integer(
                raw.get("convergence_streak_after"),
                context=(
                    "External convergence look history record "
                    f"{index} convergence_streak_after"
                ),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Cannot safely resume external production: invalid convergence "
                f"look history record {index}: {exc}"
            ) from exc
        ids_raw = raw.get("accepted_box_ids", [])
        if not isinstance(ids_raw, list):
            raise RuntimeError(
                "Cannot safely resume external production: invalid convergence look "
                f"history record {index}: accepted_box_ids must be a list"
            )
        ids = [
            _strict_integral_box_id(
                item,
                context=f"External convergence look history record {index}",
            )
            for item in list(ids_raw or [])
        ]
        digest = str(raw.get("accepted_evidence_sha256", "")).strip().lower()
        criterion_met = raw.get("criterion_met")
        advanced_streak = raw.get("advanced_streak")
        if (
            total <= prior_total
            or accepted < 0
            or accepted > total
            or ids != sorted(set(ids))
            or len(ids) != accepted
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or streak_after < 0
            or not isinstance(criterion_met, bool)
            or not isinstance(advanced_streak, bool)
        ):
            raise RuntimeError(
                "Cannot safely resume external production: invalid convergence look "
                f"history record {index}"
            )
        out.append(
            {
                "n_boxes_total": total,
                "n_boxes_accepted": accepted,
                "accepted_box_ids": ids,
                "accepted_evidence_sha256": digest,
                "criterion_met": criterion_met,
                "advanced_streak": advanced_streak,
                "convergence_streak_after": streak_after,
            }
        )
        prior_total = total
    return out


def _sha256_file(path: Path) -> str:
    return str(stable_file_identity(Path(path))["sha256"])


def _file_identity(path: Path, *, recorded_path: Optional[str] = None) -> dict[str, Any]:
    p = Path(path)
    current = stable_file_identity(p)
    return {
        "path": str(p if recorded_path is None else recorded_path),
        "size_bytes": int(current["size_bytes"]),
        "sha256": str(current["sha256"]),
    }


def _identity_matches(path: Path, identity: Mapping[str, Any]) -> bool:
    try:
        p = Path(path)
        expected_size = int(identity.get("size_bytes"))
        expected_sha = str(identity.get("sha256", "")).strip().lower()
        current = stable_file_identity(p)
        return (
            int(current["size_bytes"]) == expected_size
            and len(expected_sha) == 64
            and str(current["sha256"]).lower() == expected_sha
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _resolve_configured_file(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve(strict=False)


def _task_input_manifest(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
    input_snapshot: Path,
    base_dir: Path,
) -> dict[str, Any]:
    """Bind a task manifest to every content-bearing scientific input."""

    plan_potential = plan.get("potential_config", None)
    if isinstance(plan_potential, Mapping) and len(plan_potential) > 0:
        potential: Mapping[str, Any] = plan_potential
    else:
        config_potential = getattr(config, "kim", None)
        potential = (
            config_potential.model_dump(mode="json")
            if hasattr(config_potential, "model_dump")
            else {}
        )
    dependency_values: list[Any] = list(potential.get("files", []) or [])
    if len(potential) > 0:
        for path in potential_command_file_paths(
            potential=potential,
            plan=plan,
            declared_values=dependency_values,
            base_dir=base_dir,
        ):
            if str(path) not in {str(x) for x in dependency_values}:
                dependency_values.append(path)
    generated_table = validated_tabulated_core_path(
        plan.get("potential_lines"), root=base_dir
    )
    if generated_table is not None and str(generated_table) not in {
        str(x) for x in dependency_values
    }:
        dependency_values.append(generated_table)

    dependencies: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in dependency_values:
        if value is None:
            raise ValueError(
                "potential.files contains a null entry; provide a path or remove the entry"
            )
        if not str(value).strip():
            raise ValueError(
                "potential.files contains a blank entry; provide a path or remove the entry"
            )
        path = _resolve_configured_file(value, base_dir=base_dir)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dependencies.append(_file_identity(path, recorded_path=key))

    cp2k_data_files: list[dict[str, Any]] = []
    engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()
    if engine == "cp2k":
        cp2k_cfg = getattr(config, "cp2k", None)
        if cp2k_cfg is None:
            raise ValueError("CP2K task materialisation requires a cp2k configuration")
        resolved = Cp2kRunner(cp2k_cfg).resolved_data_files(base_dir, require=True)
        for role in ("basis_set", "potential"):
            record = resolved.get(role, None)
            if not isinstance(record, Mapping):
                raise FileNotFoundError(f"Could not resolve required CP2K {role} input")
            path = Path(str(record.get("resolved_path", "")))
            identity = _file_identity(path, recorded_path=str(path.resolve(strict=False)))
            identity.update(
                {
                    "role": role,
                    "configured_name": str(record.get("configured_name", "")),
                }
            )
            cp2k_data_files.append(identity)

    return {
        "schema": "vitriflow.task_inputs.v2",
        "structure_snapshot": _file_identity(
            input_snapshot,
            recorded_path=str(Path(input_snapshot).resolve(strict=False)),
        ),
        "dependencies": dependencies,
        "cp2k_data_files": cp2k_data_files,
    }


def _task_inputs_are_current(task_data: Mapping[str, Any]) -> bool:
    manifest = task_data.get("input_manifest", None)
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "vitriflow.task_inputs.v2":
        return False
    structure = manifest.get("structure_snapshot", None)
    if not isinstance(structure, Mapping):
        return False
    if not _identity_matches(Path(str(structure.get("path", ""))), structure):
        return False
    dependencies = manifest.get("dependencies", None)
    if not isinstance(dependencies, list):
        return False
    if not all(
        isinstance(identity, Mapping)
        and _identity_matches(Path(str(identity.get("path", ""))), identity)
        for identity in dependencies
    ):
        return False

    plan = task_data.get("production_plan", {})
    config_data = task_data.get("config", {})
    if not isinstance(plan, Mapping) or not isinstance(config_data, Mapping):
        engine = "lammps"
    else:
        engine = str(plan.get("engine", config_data.get("engine", "lammps"))).strip().lower()
    cp2k_records = manifest.get("cp2k_data_files", None)
    if not isinstance(cp2k_records, list):
        return False
    if engine != "cp2k":
        return len(cp2k_records) == 0

    if len(cp2k_records) != 2:
        return False
    by_role: dict[str, Mapping[str, Any]] = {}
    for identity in cp2k_records:
        if not isinstance(identity, Mapping):
            return False
        role = str(identity.get("role", "")).strip()
        path = Path(str(identity.get("path", "")))
        if role not in {"basis_set", "potential"} or role in by_role:
            return False
        if not _identity_matches(path, identity):
            return False
        by_role[role] = identity
    if set(by_role) != {"basis_set", "potential"}:
        return False

    try:
        config = RunConfig.model_validate(dict(config_data))
        cp2k_cfg = getattr(config, "cp2k", None)
        if cp2k_cfg is None:
            return False
        task_meta = task_data.get("task", {})
        workdir = Path(str(task_meta.get("box_dir", "."))) if isinstance(task_meta, Mapping) else Path(".")
        resolved = Cp2kRunner(cp2k_cfg).resolved_data_files(workdir, require=True)
    except Exception:
        return False
    for role, expected in by_role.items():
        current = resolved.get(role, None)
        if not isinstance(current, Mapping):
            return False
        if str(current.get("configured_name", "")) != str(expected.get("configured_name", "")):
            return False
        try:
            current_path = Path(str(current.get("resolved_path", ""))).resolve(strict=True)
            expected_path = Path(str(expected.get("path", ""))).resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        if current_path != expected_path or not _identity_matches(current_path, expected):
            return False
    return True


def _safe_box_artifact_path(box_dir: Path, relative_path: str) -> Optional[Path]:
    rel = Path(str(relative_path))
    if rel.is_absolute():
        return None
    try:
        base = Path(box_dir).resolve(strict=False)
        path = (base / rel).resolve(strict=False)
        path.relative_to(base)
        return path
    except (OSError, RuntimeError, ValueError):
        return None


def _build_task_artifact_manifest(
    *,
    box_dir: Path,
    outcomes: Mapping[str, Mapping[str, Any]],
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(relative_path: str, *, required: bool) -> None:
        rel = str(Path(relative_path))
        if rel in seen:
            return
        path = _safe_box_artifact_path(box_dir, rel)
        if path is None or not path.is_file():
            if required:
                raise FileNotFoundError(f"Required production artifact is missing: {relative_path}")
            return
        seen.add(rel)
        identity = _file_identity(path, recorded_path=rel)
        identity["required"] = bool(required)
        files.append(identity)

    for stage_name, payload in outcomes.items():
        if not isinstance(payload, Mapping):
            raise TypeError(f"Stage outcome {stage_name!r} is not a mapping")
        stage_dir = Path(str(stage_name))
        output_data = payload.get("output_data", None)
        if not isinstance(output_data, str) or not output_data.strip():
            raise ValueError(f"Stage outcome {stage_name!r} has no output_data")
        add(str(stage_dir / output_data), required=True)
        dump = payload.get("dump", None)
        if isinstance(dump, str) and dump.strip():
            add(str(stage_dir / dump), required=True)
        for optional_name in (
            "thermo.csv",
            "msd.csv",
            "stage_artifacts.json",
            "final.extxyz",
        ):
            add(str(stage_dir / optional_name), required=False)

    def add_diagnostic_payload(value: Any, *, key: Optional[str] = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                add_diagnostic_payload(child_value, key=str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                add_diagnostic_payload(child, key=key)
            return
        if not isinstance(value, str) or not value.strip():
            return
        if key in {"csv", "summary", "plot"}:
            add(value, required=True)
        elif key == "dir":
            directory = _safe_box_artifact_path(box_dir, value)
            if directory is None or not directory.is_dir():
                raise FileNotFoundError(
                    f"Required diagnostic artifact directory is missing: {value}"
                )
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    add(
                        str(path.relative_to(Path(box_dir).resolve(strict=False))),
                        required=True,
                    )

    if diagnostics is not None:
        add_diagnostic_payload(diagnostics)

    return {
        "schema": "vitriflow.task_artifacts.v1",
        "files": files,
    }


def _task_artifacts_are_current(
    *,
    box_dir: Path,
    cached: Mapping[str, Any],
) -> bool:
    manifest = cached.get("artifact_manifest", None)
    outcomes = cached.get("outcomes", None)
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "vitriflow.task_artifacts.v1":
        return False
    if not isinstance(outcomes, Mapping) or not outcomes:
        return False
    files = manifest.get("files", None)
    if not isinstance(files, list) or not files:
        return False
    identities: dict[str, Mapping[str, Any]] = {}
    for identity in files:
        if not isinstance(identity, Mapping):
            return False
        rel = str(identity.get("path", ""))
        path = _safe_box_artifact_path(box_dir, rel)
        if not rel or path is None or rel in identities or not _identity_matches(path, identity):
            return False
        identities[rel] = identity

    # Every stage output recorded in the result is required for cache reuse;
    # a truncated manifest cannot make an incomplete task appear valid.
    for stage_name, payload in outcomes.items():
        if not isinstance(payload, Mapping):
            return False
        output_data = payload.get("output_data", None)
        if not isinstance(output_data, str) or not output_data.strip():
            return False
        required = str(Path(str(stage_name)) / output_data)
        if required not in identities or not bool(identities[required].get("required", False)):
            return False
        dump = payload.get("dump", None)
        if isinstance(dump, str) and dump.strip():
            required_dump = str(Path(str(stage_name)) / dump)
            if required_dump not in identities or not bool(identities[required_dump].get("required", False)):
                return False

    def declared_diagnostic_paths(value: Any, *, key: Optional[str] = None) -> list[str]:
        out: list[str] = []
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                out.extend(
                    declared_diagnostic_paths(child_value, key=str(child_key))
                )
        elif isinstance(value, list):
            for child in value:
                out.extend(declared_diagnostic_paths(child, key=key))
        elif isinstance(value, str) and value.strip() and key in {
            "csv",
            "summary",
            "plot",
        }:
            out.append(str(Path(value)))
        return out

    diagnostics = cached.get("diagnostics", None)
    if diagnostics is not None:
        if not isinstance(diagnostics, Mapping):
            return False
        try:
            _validate_completed_task_diagnostics(diagnostics)
        except (TypeError, ValueError, RuntimeError):
            return False
        for required_diag in declared_diagnostic_paths(diagnostics):
            if required_diag not in identities or not bool(
                identities[required_diag].get("required", False)
            ):
                return False
    return True


def _cached_task_result_is_reusable(
    *,
    task_data: Mapping[str, Any],
    cached: Mapping[str, Any],
    current_engine_identity: Optional[Mapping[str, Any]] = None,
) -> bool:
    try:
        validate_task_result_integrity(cached, require_current=True)
    except (RuntimeError, TypeError, ValueError):
        return False
    task_meta = task_data.get("task", {})
    if not isinstance(task_meta, Mapping):
        return False
    status = str(cached.get("status", "")).strip().lower()
    if status not in {"ok", "success"}:
        return False
    cached_engine_identity = cached.get("engine_build_identity")
    if not isinstance(cached_engine_identity, Mapping):
        return False
    try:
        checked_cached_identity = validate_engine_build_identity(
            cached_engine_identity
        )
        if current_engine_identity is not None:
            checked_current_identity = validate_engine_build_identity(
                current_engine_identity
            )
            if (
                checked_cached_identity.get("identity_sha256")
                != checked_current_identity.get("identity_sha256")
            ):
                return False
    except (RuntimeError, TypeError, ValueError):
        return False
    if not _task_runtime_is_current(task_data):
        return False
    if str(cached.get("task_manifest_sha256", "") or "").strip().lower() != _task_manifest_digest(task_data):
        return False
    if not _task_inputs_are_current(task_data):
        return False
    declared_diagnostic_plan = task_data.get("diagnostic_plan", None)
    if declared_diagnostic_plan is not None:
        diagnostics = cached.get("diagnostics", None)
        if not isinstance(declared_diagnostic_plan, Mapping) or not isinstance(
            diagnostics, Mapping
        ):
            return False
        if dict(diagnostics.get("plan", {}) or {}) != dict(
            declared_diagnostic_plan
        ):
            return False
    box_dir = Path(str(task_meta.get("box_dir", "")))
    return _task_artifacts_are_current(box_dir=box_dir, cached=cached)


def validate_external_task_engine_identities(
    *,
    production_dir: Path,
    box_ids: Sequence[int],
    expected_current: Optional[Mapping[str, Any]] = None,
    resume_state: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Authenticate current task results and enforce one worker engine build."""

    prod_dir = Path(production_dir)
    results: list[Mapping[str, Any]] = []
    for box_id in box_ids:
        box_dir = prod_dir / f"box_{int(box_id):03d}"
        task_path = box_dir / "task.json"
        result_path = box_dir / "task_result.json"
        if (
            task_path.is_symlink()
            or result_path.is_symlink()
            or not task_path.is_file()
            or not result_path.is_file()
        ):
            raise RuntimeError(
                f"Missing or non-regular external task/result for box {int(box_id)}"
            )
        try:
            task_data = json.loads(task_path.read_text())
            result = json.loads(result_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Malformed external task/result for box {int(box_id)}"
            ) from exc
        if not isinstance(task_data, Mapping) or not isinstance(result, Mapping):
            raise RuntimeError(
                f"Malformed external task/result mapping for box {int(box_id)}"
            )
        validate_task_result_integrity(result, require_current=True)
        if str(result.get("task_manifest_sha256", "")).strip().lower() != _task_manifest_digest(task_data):
            raise RuntimeError(
                f"External task result for box {int(box_id)} belongs to a different task manifest"
            )
        status = str(result.get("status", "")).strip().lower()
        if status in {"ok", "success"} and not _cached_task_result_is_reusable(
            task_data=task_data,
            cached=result,
            current_engine_identity=expected_current,
        ):
            raise RuntimeError(
                f"Successful external task result for box {int(box_id)} is not current or reusable"
            )
        results.append(result)

    expected_resume: Optional[Mapping[str, Any]] = None
    if resume_state is not None:
        raw_expected = resume_state.get("engine_build_identity")
        accepted = int(
            resume_state.get(
                "n_boxes_accepted", resume_state.get("n_boxes", 0)
            )
            or 0
        )
        if raw_expected is not None:
            if not isinstance(raw_expected, Mapping):
                raise RuntimeError(
                    "Cannot safely resume external production: malformed stored engine identity"
                )
            expected_resume = validate_engine_build_identity(raw_expected)
        elif accepted > 0:
            raise RuntimeError(
                "Cannot safely resume external production: accepted checkpoint boxes "
                "have no protected worker engine identity"
            )

    identity = homogeneous_successful_task_engine_identity(
        results,
        expected=expected_resume,
    )
    if identity is not None and expected_current is not None:
        # The cache checks above already compare each success.  Repeat through
        # the public identity validator so this invariant remains explicit if
        # cache policy is refactored later.
        assert_homogeneous_engine_build_identities(
            [identity], expected=expected_current
        )
    return identity


def _ensure_real_directory_within(
    path: Path,
    *,
    root: Path,
) -> Path:
    """Create/check a writable directory without following child symlinks.

    External task trees are later populated with engine inputs and shell
    scripts.  ``Path.mkdir(..., exist_ok=True)`` accepts a symlink to a
    directory, which would allow a stale or hostile task tree to redirect
    those writes outside the calculation root.  Walk every component below a
    trusted, canonical root, reject symlinks and non-directories, and prove
    the final directory remains contained after creation.

    The caller must pass a canonical calculation root.  Existing symlink
    aliases above that root are deliberately outside this helper's boundary;
    no symlink is permitted at or below ``root``.
    """

    root = Path(root).expanduser()
    path = Path(path).expanduser()
    if not root.is_absolute():
        root = Path(os.path.abspath(root))
    if not path.is_absolute():
        path = root / path

    if root.is_symlink():
        raise RuntimeError(
            f"External production root must be a real directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RuntimeError(
            f"External production root must be a real directory: {root}"
        )
    root_resolved = root.resolve(strict=True)

    # Use the lexical path for component checks so resolving the candidate
    # cannot erase evidence of an intermediate symlink.
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise RuntimeError(
            f"External task directory escapes the production root: {path}"
        ) from exc

    cursor = lexical_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(
                "External task directory must not contain a symbolic link: "
                f"{cursor}"
            )
        if cursor.exists():
            if not cursor.is_dir():
                raise RuntimeError(
                    f"External task directory component is not a directory: {cursor}"
                )
        else:
            cursor.mkdir()
        try:
            cursor.resolve(strict=True).relative_to(root_resolved)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"External task directory escapes the production root: {cursor}"
            ) from exc
    return lexical_path


def _reject_symlinks_in_task_tree(path: Path) -> None:
    """Fail closed if an existing task tree contains redirected entries."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"External task box must be a real directory: {root}")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            candidate = base / name
            if candidate.is_symlink():
                raise RuntimeError(
                    "External task tree must not contain a symbolic link: "
                    f"{candidate}"
                )


def _copy_input_snapshot(
    src: Path,
    dst: Path,
    *,
    containment_root: Optional[Path] = None,
) -> Path:
    src = Path(src)
    dst = Path(dst)
    before = stable_file_identity(src, reject_final_symlink=True)
    if containment_root is not None:
        _ensure_real_directory_within(dst.parent, root=Path(containment_root))
    if dst.is_symlink() or dst.parent.is_symlink():
        raise RuntimeError(
            f"Input snapshot destination must not contain a symbolic link: {dst}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        existing = stable_file_identity(dst, reject_final_symlink=True)
        if (
            str(existing["sha256"]) == str(before["sha256"])
            and int(existing["size_bytes"]) == int(before["size_bytes"])
        ):
            after = stable_file_identity(src, reject_final_symlink=True)
            if after != before:
                raise RuntimeError(f"Input structure changed while snapshotting: {src}")
            return dst

    try:
        _atomic_copy_verified_regular_file(
            src,
            dst,
            expected_sha256=str(before["sha256"]),
            expected_size_bytes=int(before["size_bytes"]),
        )
        after = stable_file_identity(src, reject_final_symlink=True)
        if after != before:
            raise RuntimeError(f"Input structure changed while snapshotting: {src}")
    except Exception:
        # Never leave a snapshot which the caller could mistake for a
        # successfully authenticated task input.
        if dst.exists() and not dst.is_symlink():
            try:
                copied = stable_file_identity(dst, reject_final_symlink=True)
                if (
                    str(copied["sha256"]) == str(before["sha256"])
                    and int(copied["size_bytes"]) == int(before["size_bytes"])
                ):
                    dst.unlink()
            except (OSError, RuntimeError):
                pass
        raise
    return dst


def _box_seed_map(seed_base: int, box_id: int) -> dict[str, int]:
    if int(box_id) < 1:
        raise ValueError("box_id must be >= 1")
    rng = random.Random(int(seed_base))
    for _ in range((int(box_id) - 1) * 4):
        rng.randrange(1, 2**31 - 1)
    return {
        "warmup": int(rng.randrange(1, 2**31 - 1)),
        "melt": int(rng.randrange(1, 2**31 - 1)),
        "quench": int(rng.randrange(1, 2**31 - 1)),
        "relax": int(rng.randrange(1, 2**31 - 1)),
    }


def _planned_box_count(prod_cfg: ProductionEnsembleConfig) -> int:
    target = max(1, int(getattr(prod_cfg, "min_boxes", 1) or 1))
    max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
    if max_boxes_raw is None:
        return int(target)
    try:
        max_boxes = int(max_boxes_raw)
    except Exception:
        return int(target)
    if max_boxes <= 0:
        return int(target)
    return int(max(target, max_boxes))


def _guard_external_production_feature_parity(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
) -> None:
    """Reject only production features the external task schema cannot run."""

    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    if bool(getattr(getattr(prod_cfg, "dft_opt", None), "enabled", False)):
        raise ValueError(
            "external dry/full-run does not support production.dft_opt cell "
            "refinement; run the production plan locally or disable "
            "autotune.production.dft_opt.enabled before materialising Slurm tasks"
        )

    metrics_cfg = _task_metrics_cfg(plan)
    if should_collect_stage_metrics_timeseries(metrics_cfg):
        _task_metric_cutoffs(
            plan=plan,
            metrics_cfg=metrics_cfg,
            type_to_species=_type_to_species_from_plan(config, plan),
        )



def _task_metrics_cfg(plan: Mapping[str, Any]) -> StructureMetricsConfig:
    """Return the exact protected metrics configuration for task execution."""

    return StructureMetricsConfig.model_validate(plan.get("metrics_cfg", {}))


def _type_to_species_from_plan(config: RunConfig, plan: Mapping[str, Any]) -> Optional[list[str]]:
    val = plan.get("type_to_species", None)
    if val is not None:
        return [str(x) for x in val]
    metrics = config.autotune.metrics
    if metrics.type_to_species is not None:
        return [str(x) for x in metrics.type_to_species]
    pot = getattr(config, "kim", None)
    interactions = getattr(pot, "interactions", None)
    if interactions is not None and interactions != "fixed_types":
        return [str(x) for x in interactions]
    return None


def _task_metric_cutoffs(
    *,
    plan: Mapping[str, Any],
    metrics_cfg: StructureMetricsConfig,
    type_to_species: Optional[Sequence[str]],
) -> dict[tuple[int, int], float]:
    """Resolve and validate the protected cutoff map used by task diagnostics."""

    raw = (
        plan.get("preferred_cutoffs")
        or plan.get("cutoffs_size")
        or plan.get("cutoffs_rate")
        or []
    )
    out: dict[tuple[int, int], float] = {}

    def _insert(a_raw: Any, b_raw: Any, value_raw: Any, *, label: str) -> None:
        try:
            a, b = int(a_raw), int(b_raw)
            value = float(value_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid production-plan cutoff {label}") from exc
        if a <= 0 or b <= 0 or not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                f"Production-plan cutoff {label} must use positive atom types "
                "and a finite value > 0"
            )
        key = (min(a, b), max(a, b))
        if key in out:
            if not math.isclose(
                float(out[key]), value, rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                raise ValueError(f"Conflicting production-plan cutoff for pair {key}")
            raise ValueError(f"Duplicate production-plan cutoff for pair {key}")
        out[key] = value

    if isinstance(raw, Mapping):
        for key_raw, value_raw in raw.items():
            if isinstance(key_raw, (list, tuple)) and len(key_raw) == 2:
                a_raw, b_raw = key_raw
            else:
                text = str(key_raw).strip().lstrip("(").rstrip(")")
                parts = [part.strip() for part in text.replace("_", ",").split(",")]
                if len(parts) != 2 or not all(parts):
                    raise ValueError(
                        f"Invalid production-plan cutoff pair key: {key_raw!r}"
                    )
                a_raw, b_raw = parts
            _insert(a_raw, b_raw, value_raw, label=repr(key_raw))
    elif isinstance(raw, list):
        for idx, entry in enumerate(raw):
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Production-plan cutoff entry {idx} must be an object"
                )
            pair = entry.get("pair")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"Production-plan cutoff entry {idx} has an invalid pair"
                )
            if "cutoff" not in entry:
                raise ValueError(
                    f"Production-plan cutoff entry {idx} is missing cutoff"
                )
            _insert(pair[0], pair[1], entry.get("cutoff"), label=str(idx))
    else:
        raise ValueError("Production-plan cutoffs must be a list or mapping")

    fixed = fixed_cutoffs_from_metrics(
        metrics_cfg,
        type_to_species=(
            None if type_to_species is None else [str(x) for x in type_to_species]
        ),
    )
    for key, value in fixed.items():
        norm = (min(int(key[0]), int(key[1])), max(int(key[0]), int(key[1])))
        if norm in out and not math.isclose(
            float(out[norm]), float(value), rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Production-plan cutoff {norm} disagrees with the fixed metrics cutoff"
            )
        out[norm] = float(value)

    required = required_pairs_from_metrics(
        metrics_cfg,
        type_to_species=(
            None if type_to_species is None else [str(x) for x in type_to_species]
        ),
    )
    missing = sorted(set(required) - set(out))
    if missing:
        raise ValueError(
            "External task stage diagnostics require protected cutoffs for every "
            f"configured neighbour metric; missing pairs: {missing}"
        )
    return out


def _stage_specs_for_box(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
    box_dir: Path,
    input_snapshot: Path,
) -> tuple[list[StageSpec], dict[str, Any], Any, Any, Optional[list[str]], Optional[list[str]], bool, str]:
    engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()
    md_use = MDConfig.model_validate(plan.get("md_use", {}))
    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    warmup_start_temperature = resolve_production_warmup_start_temperature(
        prod_cfg=prod_cfg,
        T_high=float(plan.get("T_high")),
    )
    warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    metrics_cfg = _task_metrics_cfg(plan)
    potential_lines = plan.get("potential_lines", None)
    if potential_lines is not None:
        potential_lines = [str(x) for x in potential_lines]
    pot_cfg = _potential_from_dict(plan.get("potential_config", None), config.kim)
    type_to_species = _type_to_species_from_plan(config, plan)

    if engine == "cp2k":
        runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]
    else:
        runner = LammpsRunner(config.lammps)

    T_high = float(plan.get("T_high"))
    warmup_time_unit_ps = resolve_production_time_unit_ps(
        config=config,
        engine=engine,
        pot_cfg=pot_cfg,
        time_unit_ps=plan.get("time_unit_ps", None),
    )
    t_final = float(plan.get("t_final"))
    quench_steps = int(plan.get("quench_steps"))
    relax_steps = int(plan.get("relax_steps"))
    high_total_steps = int(plan.get("high_total_steps"))
    warmup_steps = resolve_production_warmup_steps(
        prod_cfg=prod_cfg,
        md_timestep=float(md_use.timestep),
        time_unit_ps=warmup_time_unit_ps,
    )
    replicate = [int(x) for x in plan.get("replicate", [1, 1, 1])]
    if len(replicate) != 3:
        raise ValueError("production plan replicate must have length 3")
    md_pressure = float(plan.get("pressure"))
    melt_force_iso = bool(getattr(md_use, "force_isotropic", False))
    cont = str(getattr(md_use, "stage_continuity", "discontinuous")).strip().lower()
    vel_next = "preserve" if cont == "continuous" else "create"
    stage_diag = plan_production_stage_diagnostics(
        prod_cfg=prod_cfg,
        metrics_cfg=metrics_cfg,
        runner=runner,
        force_isotropic=bool(melt_force_iso),
        total_quench_steps=int(quench_steps),
        temperature_start=float(T_high),
        temperature_stop=float(t_final),
        sampling_hint=(plan.get("sampling_hint", None) if isinstance(plan.get("sampling_hint", None), Mapping) else None),
    )
    need_stage_dump = dict(stage_diag.get("need_stage_dump", {}))
    dump_every = int(stage_diag.get("dump_every", getattr(prod_cfg, "dump_every_steps", 5000) or 5000))
    quench_dump_every = int(stage_diag.get("quench_dump_every", dump_every))
    msd_every = int(plan.get("msd_every", config.autotune.tm_scan.msd_every))
    relax_dump_settings = resolve_production_relax_dump_settings(stage_diag=stage_diag, metrics_cfg=metrics_cfg)

    seeds = _box_seed_map(int(plan.get("seed_base", config.random_seed + 13579)), int(box_dir.name.split("_")[-1]))

    warmup_stage = StageSpec(
        name="warmup",
        input_data=Path(input_snapshot),
        output_data=box_dir / "warmup.data",
        temperature_start=float(warmup_start_temperature),
        temperature_stop=float(T_high),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(warmup_steps),
        seed=int(seeds["warmup"]),
        velocity_mode="create",
        force_isotropic=bool(melt_force_iso),
        replicate=(int(replicate[0]), int(replicate[1]), int(replicate[2])),
        write_dump=bool(need_stage_dump.get("melt", True)),
        dump_every=int(dump_every) if bool(need_stage_dump.get("melt", True)) else None,
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    melt_stage = StageSpec(
        name="melt",
        input_data=box_dir / "warmup" / "warmup.data",
        output_data=box_dir / "melt.data",
        temperature_start=float(T_high),
        temperature_stop=float(T_high),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(high_total_steps),
        seed=int(seeds["melt"]),
        velocity_mode="preserve",
        force_isotropic=False,
        replicate=None,
        write_dump=bool(need_stage_dump.get("melt", True)),
        dump_every=int(dump_every) if bool(need_stage_dump.get("melt", True)) else None,
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    quench_stage = StageSpec(
        name="quench",
        input_data=box_dir / "melt" / "melt.data",
        output_data=box_dir / "quench.data",
        temperature_start=float(T_high),
        temperature_stop=float(t_final),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(quench_steps),
        seed=int(seeds["quench"]),
        velocity_mode=("preserve" if cont == "continuous" else vel_next),
        replicate=None,
        write_dump=bool(need_stage_dump.get("quench", True)),
        dump_every=int(quench_dump_every) if bool(need_stage_dump.get("quench", True)) else None,
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    relax_stage = StageSpec(
        name="relax",
        input_data=box_dir / "quench" / "quench.data",
        output_data=box_dir / "relax.data",
        temperature_start=float(t_final),
        temperature_stop=float(t_final),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(relax_steps),
        seed=int(seeds["relax"]),
        velocity_mode=("preserve" if cont == "continuous" else vel_next),
        replicate=None,
        write_dump=bool(relax_dump_settings["write_dump"]),
        dump_every=relax_dump_settings["dump_every"],
        tail_dump_frames=relax_dump_settings["tail_dump_frames"],
        tail_dump_stride=relax_dump_settings["tail_dump_stride"],
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    return [warmup_stage, melt_stage, quench_stage, relax_stage], stage_diag, runner, pot_cfg, type_to_species, potential_lines, bool(melt_force_iso), cont


def _task_diagnostic_plan(
    *,
    metrics_cfg: StructureMetricsConfig,
    runner: Any,
    stage_diag: Mapping[str, Any],
    force_isotropic: bool,
    engine: str,
) -> dict[str, Any]:
    """Describe every diagnostic callback a materialised task must execute."""

    collect_stage = should_collect_stage_metrics_timeseries(metrics_cfg)
    elastic_cfg = getattr(metrics_cfg, "elastic", None)
    screen_roles: dict[str, dict[str, bool]] = {}
    series_roles: dict[str, dict[str, bool]] = {}
    if str(engine).strip().lower() == "lammps":
        for role in ("melt", "relax"):
            enabled, strict, _ = should_run_elastic_screen(
                metrics_cfg,
                runner=runner,
                stage_role=role,
                force_isotropic=bool(force_isotropic),
            )
            screen_roles[role] = {
                "enabled": bool(enabled),
                "strict": bool(strict),
                "plot_required": bool(
                    enabled
                    and elastic_cfg is not None
                    and bool(getattr(elastic_cfg, "make_plot", True))
                ),
            }
        for role in ("melt", "quench", "relax"):
            enabled, strict, _ = should_collect_elastic_stage_timeseries(
                metrics_cfg,
                runner=runner,
                stage_role=role,
                force_isotropic=bool(force_isotropic),
            )
            planned_enabled = bool(
                (stage_diag.get("collect_elastic_series", {}) or {}).get(
                    role, enabled
                )
            )
            if planned_enabled != bool(enabled):
                raise RuntimeError(
                    f"External diagnostic plan disagrees for elastic timeseries role={role}"
                )
            series_roles[role] = {
                "enabled": bool(enabled),
                "strict": bool(strict),
                "plot_required": bool(
                    enabled
                    and elastic_cfg is not None
                    and bool(
                        getattr(elastic_cfg, "stage_timeseries_make_plot", True)
                    )
                ),
            }
    else:
        screen_roles = {
            role: {"enabled": False, "strict": False, "plot_required": False}
            for role in ("melt", "relax")
        }
        series_roles = {
            role: {"enabled": False, "strict": False, "plot_required": False}
            for role in ("melt", "quench", "relax")
        }

    return {
        "schema": "vitriflow.production_task_diagnostic_plan.v1",
        "engine": str(engine).strip().lower(),
        "stage_metrics": {
            "enabled": bool(collect_stage),
            "roles": ["melt", "quench", "relax"] if collect_stage else [],
            "plot_required": bool(
                collect_stage
                and bool(getattr(metrics_cfg, "stage_timeseries_make_plot", False))
            ),
        },
        "elastic_screens": {
            "supported": str(engine).strip().lower() == "lammps",
            "roles": screen_roles,
        },
        "elastic_timeseries": {
            "supported": str(engine).strip().lower() == "lammps",
            "roles": series_roles,
        },
    }


def _stage_output_path(
    *, box_dir: Path, stage_name: str, outcome: Mapping[str, Any]
) -> Path:
    raw = outcome.get("output_data")
    if raw in (None, ""):
        raise ValueError(f"Stage {stage_name!r} outcome has no output_data")
    path = Path(str(raw))
    if path.is_absolute():
        resolved = path
    else:
        in_stage = Path(box_dir) / str(stage_name) / path
        in_box = Path(box_dir) / path
        resolved = in_stage if in_stage.is_file() or not in_box.is_file() else in_box
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Stage {stage_name!r} output data is missing: {resolved}"
        )
    return resolved


def _qualified_optional_diagnostic_failure(
    *, family: str, stage_role: str, exc: Exception
) -> dict[str, Any]:
    return {
        "status": "failed",
        "family": str(family),
        "stage_role": str(stage_role),
        "error": str(exc),
    }


def _validate_completed_task_diagnostics(diagnostics: Mapping[str, Any]) -> None:
    """Validate task callbacks against the protected diagnostic plan."""

    if diagnostics.get("schema") != "vitriflow.production_task_diagnostics.v1":
        raise ValueError("Unsupported completed task diagnostics schema")
    if diagnostics.get("path_base") != "task_box":
        raise ValueError(
            "Completed task diagnostic paths must be relative to path_base='task_box'"
        )
    plan = diagnostics.get("plan")
    if not isinstance(plan, Mapping) or plan.get("schema") != (
        "vitriflow.production_task_diagnostic_plan.v1"
    ):
        raise ValueError("Completed task diagnostics are missing their protected plan")

    observed_statuses: list[str] = []
    stage_plan = plan.get("stage_metrics", {})
    stage_results = diagnostics.get("stage_metrics")
    if bool(stage_plan.get("enabled", False)):
        if not isinstance(stage_results, Mapping):
            raise RuntimeError("Requested stage metrics are absent from task diagnostics")
        stage_roles = [str(role) for role in list(stage_plan.get("roles", []) or [])]
        if set(str(key) for key in stage_results) != set(stage_roles):
            raise RuntimeError(
                "Task stage-metrics roles disagree with the protected diagnostic plan"
            )
        for role in stage_roles:
            result = stage_results.get(str(role))
            if not isinstance(result, Mapping) or str(
                result.get("status", "")
            ).strip().lower() != "ok":
                raise RuntimeError(
                    f"Requested stage metrics did not complete successfully for role={role}"
                )
            observed_statuses.append("ok")
            for key in ("csv", "summary"):
                if not isinstance(result.get(key), str) or not str(
                    result.get(key)
                ).strip():
                    raise RuntimeError(
                        f"Requested stage metrics role={role} is missing {key}"
                    )
            if bool(stage_plan.get("plot_required", False)) and (
                not isinstance(result.get("plot"), str)
                or not str(result.get("plot")).strip()
            ):
                raise RuntimeError(
                    f"Requested stage metrics role={role} is missing its plot"
                )
    elif stage_results is not None:
        raise RuntimeError("Task produced stage metrics that were disabled in its plan")

    for family_key, required_paths in (
        ("elastic_screens", ("dir", "summary")),
        ("elastic_timeseries", ("dir", "csv", "summary")),
    ):
        family_plan = plan.get(family_key, {})
        family_results = diagnostics.get(family_key)
        if not isinstance(family_results, Mapping):
            raise RuntimeError(f"Task diagnostics are missing {family_key}")
        role_plan = family_plan.get("roles", {})
        if not isinstance(role_plan, Mapping):
            raise ValueError(f"Diagnostic plan has invalid {family_key} roles")
        if set(str(key) for key in family_results) != set(
            str(key) for key in role_plan
        ):
            raise RuntimeError(
                f"Task {family_key} roles disagree with the protected diagnostic plan"
            )
        for role, settings in role_plan.items():
            if not isinstance(settings, Mapping):
                raise ValueError(
                    f"Diagnostic plan has invalid {family_key} role={role}"
                )
            enabled = bool(settings.get("enabled", False))
            result = family_results.get(str(role))
            if not enabled:
                if result is not None:
                    raise RuntimeError(
                        f"Task produced disabled {family_key} role={role}"
                    )
                continue
            if not isinstance(result, Mapping):
                raise RuntimeError(
                    f"Requested {family_key} role={role} is absent"
                )
            status = str(result.get("status", "")).strip().lower()
            if status not in {"ok", "degraded", "failed"}:
                raise RuntimeError(
                    f"Requested {family_key} role={role} has invalid status={status!r}"
                )
            if bool(settings.get("strict", False)) and status != "ok":
                raise RuntimeError(
                    f"Strict {family_key} role={role} did not complete successfully"
                )
            observed_statuses.append(status)
            if status == "ok":
                for key in required_paths:
                    if not isinstance(result.get(key), str) or not str(
                        result.get(key)
                    ).strip():
                        raise RuntimeError(
                            f"Successful {family_key} role={role} is missing {key}"
                        )
                if bool(settings.get("plot_required", False)) and (
                    not isinstance(result.get("plot"), str)
                    or not str(result.get("plot")).strip()
                ):
                    raise RuntimeError(
                        f"Successful {family_key} role={role} is missing its plot"
                    )

    expected_status = (
        "ok" if all(status == "ok" for status in observed_statuses) else "degraded"
    )
    if str(diagnostics.get("status", "")).strip().lower() != expected_status:
        raise RuntimeError(
            "Completed task diagnostic status disagrees with its role results: "
            f"reported={diagnostics.get('status')!r} expected={expected_status!r}"
        )


def _run_task_production_diagnostics(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
    box_dir: Path,
    stage_diag: Mapping[str, Any],
    runner: Any,
    pot_cfg: Any,
    md_use: MDConfig,
    type_to_species: Optional[Sequence[str]],
    potential_lines: Optional[list[str]],
    force_isotropic: bool,
    outcomes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute local-equivalent stage, plot, and elastic callbacks in a task."""

    metrics_cfg = _task_metrics_cfg(plan)
    engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()
    diagnostic_plan = _task_diagnostic_plan(
        metrics_cfg=metrics_cfg,
        runner=runner,
        stage_diag=stage_diag,
        force_isotropic=bool(force_isotropic),
        engine=engine,
    )
    stage_dirs = {
        role: Path(box_dir) / role for role in ("warmup", "melt", "quench", "relax")
    }
    output_paths = {
        role: _stage_output_path(
            box_dir=box_dir,
            stage_name=role,
            outcome=outcomes[role],
        )
        for role in stage_dirs
    }

    elastic_screens: dict[str, Any] = {"melt": None, "relax": None}
    elastic_timeseries: dict[str, Any] = {
        "melt": None,
        "quench": None,
        "relax": None,
    }
    if engine == "lammps":
        for role, input_role in (("melt", "warmup"), ("relax", "quench")):
            role_plan = diagnostic_plan["elastic_screens"]["roles"][role]
            if not bool(role_plan["enabled"]):
                continue
            try:
                elastic_screens[role] = run_elastic_screen_lammps(
                    runner,
                    pot_cfg,
                    md_use,
                    structure_data=output_paths[role],
                    stage_dir=stage_dirs[role],
                    potential_lines=potential_lines,
                    metrics_cfg=metrics_cfg,
                    force_isotropic=bool(force_isotropic),
                    input_data_for_affine_strain=output_paths[input_role],
                    outdir=box_dir,
                )
            except Exception as exc:
                if bool(role_plan["strict"]):
                    raise
                elastic_screens[role] = _qualified_optional_diagnostic_failure(
                    family="elastic_screen", stage_role=role, exc=exc
                )

        for role in ("melt", "quench", "relax"):
            role_plan = diagnostic_plan["elastic_timeseries"]["roles"][role]
            if not bool(role_plan["enabled"]):
                continue
            try:
                elastic_timeseries[role] = run_elastic_screen_timeseries_lammps(
                    runner,
                    pot_cfg,
                    md_use,
                    stage_dir=stage_dirs[role],
                    stage_output_data=output_paths[role],
                    stage_role=role,
                    potential_lines=potential_lines,
                    metrics_cfg=metrics_cfg,
                    force_isotropic=bool(force_isotropic),
                    outdir=box_dir,
                    sampling_hint=(
                        dict(plan.get("sampling_hint", {}) or {})
                        if role == "quench"
                        else None
                    ),
                )
            except Exception as exc:
                if bool(role_plan["strict"]):
                    raise
                elastic_timeseries[role] = _qualified_optional_diagnostic_failure(
                    family="elastic_timeseries", stage_role=role, exc=exc
                )

    stage_metrics = None
    if bool(diagnostic_plan["stage_metrics"]["enabled"]):
        cutoffs = _task_metric_cutoffs(
            plan=plan,
            metrics_cfg=metrics_cfg,
            type_to_species=type_to_species,
        )
        lammps_units_style = (
            str(getattr(pot_cfg, "user_units", "metal") or "metal")
            if engine == "lammps"
            else None
        )
        stage_metrics = {}
        for role in ("melt", "quench", "relax"):
            stage_metrics[role] = collect_stage_metrics_timeseries(
                stage_dir=stage_dirs[role],
                metrics_cfg=metrics_cfg,
                cutoffs=cutoffs,
                md_timestep=float(md_use.timestep),
                type_to_species=type_to_species,
                outdir=box_dir,
                stage_role=role,
                quench_window_steps_range=(
                    stage_diag.get("quench_window_steps_range")
                    if role == "quench"
                    else None
                ),
                sampling_hint=(
                    dict(plan.get("sampling_hint", {}) or {})
                    if role == "quench"
                    else None
                ),
                lammps_units_style=lammps_units_style,
                engine=engine,
            )

    statuses: list[str] = []
    for family in (elastic_screens, elastic_timeseries, stage_metrics or {}):
        for value in family.values():
            if isinstance(value, Mapping):
                statuses.append(str(value.get("status", "unknown")).strip().lower())
    status = "ok" if all(value == "ok" for value in statuses) else "degraded"
    completed = {
        "schema": "vitriflow.production_task_diagnostics.v1",
        "status": status,
        "path_base": "task_box",
        "plan": diagnostic_plan,
        "stage_metrics": stage_metrics,
        "elastic_screens": elastic_screens,
        "elastic_timeseries": elastic_timeseries,
    }
    _validate_completed_task_diagnostics(completed)
    return completed


def _render_preview_inputs(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
    box_dir: Path,
    stages: Sequence[StageSpec],
    potential_cfg: Any,
    stage_continuity: str,
) -> list[str]:
    preview_dir = Path(box_dir) / "preview"
    ensure_dir(preview_dir)
    written: list[str] = []
    engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()

    if engine == "lammps":
        if potential_cfg is not None:
            prepare_potential_files(potential_cfg, preview_dir, plan.get("potential_lines", None))
        md_use = MDConfig.model_validate(plan.get("md_use", {}))
        if stage_continuity == "continuous":
            script = render_continuous_stages(
                potential_cfg,
                md_use,
                stages,
                stage_dir_prefixes={s.name: s.name for s in stages},
                log_name="log.lammps",
            )
            path = preview_dir / "continuous.in.lammps"
            path.write_text(script)
            written.append(str(path))
        else:
            for stage in stages:
                path = preview_dir / f"{stage.name}.in.lammps"
                path.write_text(render_stage(potential_cfg, md_use, stage))
                written.append(str(path))
    else:
        # preview limited execution
        # segmentation depends backend

        def _jsonable(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            return value

        preview = {
            "schema": "vitriflow.preview_stage_specs.v1",
            "engine": engine,
            "stages": [_jsonable(asdict(stage)) for stage in stages],
            "note": "Exact CP2K native input files are materialised during task execution.",
        }
        path = preview_dir / "stage_specs.json"
        path.write_text(json.dumps(preview, indent=2))
        written.append(str(path))
    return written


def _render_template(template_text: str, mapping: Mapping[str, Any]) -> str:
    out = str(template_text)
    for key, value in mapping.items():
        sval = str(value)
        out = out.replace(f"{{{{{key}}}}}", sval)
        out = out.replace(f"${{{key}}}", sval)
    return out


_SLURM_TEMPLATE_KEYS = (
    "TASK_JSON",
    "TASK_DIR",
    "TASK_JSON_RAW",
    "TASK_DIR_RAW",
    "BOX_ID",
    "JOB_NAME",
    "EXECUTE_CMD",
)


def _validate_job_template_text(template_text: str, *, rendered: bool = False) -> None:
    """Validate the small, public Slurm-template contract.

    A template without ``EXECUTE_CMD`` can be submitted successfully while
    never running ``vitriflow execute-task``.  That is an application-facing
    false success, so require the command placeholder before materialisation.
    After rendering, reject unresolved ``{{NAME}}`` tokens and unresolved
    VitriFlow mapping tokens.  Ordinary shell variables remain valid.
    """

    text = str(template_text)
    known = set(_SLURM_TEMPLATE_KEYS)
    if not rendered and not any(
        token in text for token in ("{{EXECUTE_CMD}}", "${EXECUTE_CMD}")
    ):
        raise ValueError(
            "Slurm job template must contain {{EXECUTE_CMD}} or ${EXECUTE_CMD}; "
            "otherwise submitted jobs do not execute their VitriFlow task"
        )
    if rendered:
        unresolved_braces = sorted(
            set(re.findall(r"\{\{\s*([A-Z_][A-Z0-9_]*)\s*\}\}", text))
        )
        unresolved_known_shell = sorted(
            key for key in known if f"${{{key}}}" in text
        )
        unresolved = sorted(set(unresolved_braces + unresolved_known_shell))
        if unresolved:
            raise ValueError(
                "Slurm job template contains unresolved placeholder(s): "
                + ", ".join(unresolved)
            )


def _write_submission_script(task_json: Path, template_path: Optional[Path]) -> Optional[Path]:
    if template_path is None:
        return None
    template = Path(template_path)
    text = template.read_text()
    _validate_job_template_text(text, rendered=False)
    box_dir = Path(task_json).parent
    box_id = int(box_dir.name.split("_")[-1])
    mapping = {
        # The canonical placeholders are shell-quoted because submission
        # templates are shell scripts.  Explicit *_RAW aliases preserve access
        # to the literal path for non-shell metadata fields.
        "TASK_JSON": shlex.quote(str(task_json)),
        "TASK_DIR": shlex.quote(str(box_dir)),
        "TASK_JSON_RAW": str(task_json),
        "TASK_DIR_RAW": str(box_dir),
        "BOX_ID": str(box_id),
        "JOB_NAME": f"vitriflow_box_{box_id:03d}",
        "EXECUTE_CMD": f"vitriflow execute-task --task {shlex.quote(str(task_json))}",
    }
    rendered = _render_template(text, mapping)
    _validate_job_template_text(rendered, rendered=True)
    submit_path = box_dir / "submit.slurm"
    submit_path.write_text(rendered)
    return submit_path


def _task_record(base_dir: Path, task_json: Path, task_result: Path, submit_script: Optional[Path]) -> dict[str, Any]:
    box_dir = task_json.parent
    return {
        "box": int(box_dir.name.split("_")[-1]),
        "box_dir": _relpath_or_str(box_dir, base_dir),
        "task_json": _relpath_or_str(task_json, base_dir),
        "task_result": _relpath_or_str(task_result, base_dir),
        "submit_script": _relpath_or_str(submit_script, base_dir),
    }


def materialize_external_production(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    job_template: Optional[Path] = None,
    n_boxes: Optional[int] = None,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    """Materialize external production."""

    outdir = Path(outdir).expanduser().resolve(strict=False)
    outdir.mkdir(parents=True, exist_ok=True)
    if outdir.is_symlink() or not outdir.is_dir():
        raise RuntimeError(
            f"External calculation root must be a real directory: {outdir}"
        )
    _guard_external_production_feature_parity(config=config, plan=plan)
    if job_template is not None:
        template_path = Path(job_template)
        template_text = template_path.read_text()
        _validate_job_template_text(template_text, rendered=False)
        rendered_probe = _render_template(
            template_text,
            {key: f"__VITRIFLOW_{key}__" for key in _SLURM_TEMPLATE_KEYS},
        )
        _validate_job_template_text(rendered_probe, rendered=True)
    prod_dir = _ensure_real_directory_within(outdir / "production", root=outdir)
    if progress is None:
        progress = CondensedProgressLog(outdir / "condensed.log")

    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))

    planned = int(n_boxes if n_boxes is not None else _planned_box_count(prod_cfg))
    if planned < 1:
        planned = 1

    structure_src = Path(str(plan.get("structure_data", ""))).expanduser()
    if not structure_src.is_absolute():
        structure_src = (outdir / structure_src).resolve(strict=False)
    if not structure_src.exists():
        raise FileNotFoundError(f"Production-plan structure_data not found: {structure_src}")

    task_records: list[dict[str, Any]] = []
    materialized_diagnostic_plan: Optional[dict[str, Any]] = None
    for box_id in range(1, int(planned) + 1):
        box_dir = _ensure_real_directory_within(
            prod_dir / f"box_{box_id:03d}", root=prod_dir
        )
        # A repeated materialisation may encounter a partial task tree.  It is
        # safe to reuse ordinary files/directories after content checks, but
        # never follow a redirected entry for previews, inputs, or scripts.
        _reject_symlinks_in_task_tree(box_dir)
        input_snapshot = _copy_input_snapshot(
            structure_src,
            box_dir / "input" / structure_src.name,
            containment_root=box_dir,
        )
        _ensure_real_directory_within(box_dir / "preview", root=box_dir)
        stages, stage_diag, task_runner, pot_cfg, _type_to_species, _potential_lines, force_iso, stage_cont = _stage_specs_for_box(
            config=config,
            plan=plan,
            box_dir=box_dir,
            input_snapshot=input_snapshot,
        )
        diagnostic_plan = _task_diagnostic_plan(
            metrics_cfg=_task_metrics_cfg(plan),
            runner=task_runner,
            stage_diag=stage_diag,
            force_isotropic=bool(force_iso),
            engine=str(plan.get("engine", getattr(config, "engine", "lammps"))),
        )
        if materialized_diagnostic_plan is None:
            materialized_diagnostic_plan = dict(diagnostic_plan)
        elif diagnostic_plan != materialized_diagnostic_plan:
            raise RuntimeError(
                "External per-box diagnostic plans are inconsistent"
            )
        preview_paths = _render_preview_inputs(
            config=config,
            plan=plan,
            box_dir=box_dir,
            stages=stages,
            potential_cfg=pot_cfg,
            stage_continuity=stage_cont,
        )
        task_json = box_dir / "task.json"
        task_result = box_dir / "task_result.json"
        task = {
            "schema": "vitriflow.box_task.v1",
            "runtime": _runtime_identity(),
            "config": config.model_dump(mode="json"),
            "production_plan": dict(plan),
            "input_manifest": _task_input_manifest(
                config=config,
                plan=plan,
                input_snapshot=input_snapshot,
                base_dir=outdir,
            ),
            "diagnostic_plan": diagnostic_plan,
            "task": {
                "box": int(box_id),
                "box_dir": str(box_dir),
                "input_snapshot": str(input_snapshot),
                "task_json": str(task_json),
                "task_result": str(task_result),
                "preview_inputs": list(preview_paths),
            },
        }
        atomic_write_json(task_json, task)
        submit_script = _write_submission_script(task_json, job_template)
        # output_dataset.json and tasks.json both live in ``prod_dir``.  Keep
        # every relative task path anchored to that directory so consumers do
        # not accidentally resolve ``production/box_...`` as
        # ``production/production/box_...``.
        task_records.append(_task_record(prod_dir, task_json, task_result, submit_script))
        progress.info("external", f"materialised task for box {box_id}")

    submit_all = prod_dir / "submit_all.sh"
    if submit_all.is_symlink():
        raise RuntimeError(
            f"External submission script must not be a symbolic link: {submit_all}"
        )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd -- "$(dirname -- "$0")"',
    ]
    for rec in task_records:
        if rec.get("submit_script"):
            lines.append(f"sbatch {shlex.quote(str(rec['submit_script']))}")
    submit_all.write_text("\n".join(lines) + "\n")
    submit_all.chmod(0o755)

    dataset = {
        "schema": "vitriflow.output_dataset.v1",
        "source_root": str(prod_dir),
        "boxes": [
            {
                "box": int(rec["box"]),
                "box_dir": rec["box_dir"],
                "task_result": rec["task_result"],
            }
            for rec in task_records
        ],
        "planned_boxes": int(planned),
        "job_template": (None if job_template is None else str(job_template)),
        "diagnostic_plan": materialized_diagnostic_plan,
    }
    atomic_write_json(prod_dir / "output_dataset.json", dataset)
    atomic_write_json(prod_dir / "tasks.json", {"schema": "vitriflow.task_index.v1", "tasks": task_records})
    progress.info("external", f"wrote {len(task_records)} task manifests")

    return {
        "planned_boxes": int(planned),
        "tasks": task_records,
        "output_dataset": _relpath_or_str(prod_dir / "output_dataset.json", outdir),
        "task_index": _relpath_or_str(prod_dir / "tasks.json", outdir),
        "submit_all": _relpath_or_str(submit_all, outdir),
        "diagnostic_plan": materialized_diagnostic_plan,
    }


def _read_stable_external_task_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    """Read one regular task manifest without accepting a replaced path."""

    configured = Path(path).expanduser()
    if configured.is_symlink() or not configured.is_file():
        raise RuntimeError(
            f"External task manifest must be a regular non-symlink file: {configured}"
        )
    before = stable_file_identity(configured, reject_final_symlink=True)
    try:
        payload = configured.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Cannot read external task manifest: {configured}") from exc
    if (
        len(payload) != int(before["size_bytes"])
        or hashlib.sha256(payload).hexdigest() != str(before["sha256"])
    ):
        raise RuntimeError(
            f"External task manifest changed while it was being read: {configured}"
        )
    after = stable_file_identity(configured, reject_final_symlink=True)
    if before != after:
        raise RuntimeError(
            f"External task manifest changed while it was being read: {configured}"
        )
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"External task manifest is not valid JSON: {configured}") from exc
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"External task manifest root is not a mapping: {configured}")
    return Path(str(before["resolved_path"])), dict(raw)


def _validated_external_task_context(
    task: Path | Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Authenticate manifest path adjacency before choosing the task lock.

    Scientific input hashes are validated by the worker after it owns the
    lock.  Here we establish the filesystem authority boundary itself: the
    manifest must be the regular file adjacent to its real ``box_NNN``
    directory, and every worker-writable/read input path must be the exact
    generated location within that directory.
    """

    if isinstance(task, Mapping):
        supplied = dict(task)
        supplied_meta = supplied.get("task", {})
        if not isinstance(supplied_meta, Mapping):
            raise RuntimeError("External task manifest has no task metadata mapping")
        task_json_raw = supplied_meta.get("task_json")
        if task_json_raw is None or not str(task_json_raw).strip():
            raise RuntimeError(
                "In-memory external task manifests require their canonical task_json path"
            )
        task_path, task_data = _read_stable_external_task_manifest(
            Path(str(task_json_raw))
        )
        if task_data != supplied:
            raise RuntimeError(
                "In-memory external task manifest disagrees with its canonical task.json"
            )
    else:
        task_path, task_data = _read_stable_external_task_manifest(Path(task))

    if str(task_data.get("schema", "")).strip().lower() != "vitriflow.box_task.v1":
        raise RuntimeError("Unsupported external task manifest schema")
    task_meta = task_data.get("task", {})
    if not isinstance(task_meta, Mapping):
        raise RuntimeError("External task manifest has no task metadata mapping")

    box_id = _strict_integral_box_id(
        task_meta.get("box"), context="External task manifest"
    )
    box_dir_raw = task_meta.get("box_dir")
    if box_dir_raw is None or not str(box_dir_raw).strip():
        raise RuntimeError("External task manifest has no box_dir")
    box_dir_configured = Path(str(box_dir_raw)).expanduser()
    if not box_dir_configured.is_absolute():
        raise RuntimeError("External task manifest box_dir must be an absolute path")
    if box_dir_configured.is_symlink() or not box_dir_configured.is_dir():
        raise RuntimeError(
            f"External task box_dir must be a real directory: {box_dir_configured}"
        )
    box_dir = box_dir_configured.resolve(strict=True)
    if box_dir != task_path.parent:
        raise RuntimeError(
            "External task box_dir must be the directory adjacent to task.json"
        )
    match = re.fullmatch(r"box_([0-9]+)", box_dir.name)
    if match is None or int(match.group(1)) < 1 or int(match.group(1)) != box_id:
        raise RuntimeError(
            "External task box id disagrees with its canonical box_NNN directory"
        )

    recorded_task_path = Path(str(task_meta.get("task_json", ""))).expanduser()
    if (
        not recorded_task_path.is_absolute()
        or recorded_task_path.is_symlink()
        or not recorded_task_path.is_file()
        or recorded_task_path.resolve(strict=True) != task_path
    ):
        raise RuntimeError(
            "External task task_json path must identify the supplied adjacent regular manifest"
        )

    expected_result = box_dir / "task_result.json"
    recorded_result = Path(str(task_meta.get("task_result", ""))).expanduser()
    if not recorded_result.is_absolute() or recorded_result.resolve(
        strict=False
    ) != expected_result:
        raise RuntimeError(
            "External task task_result path must be the adjacent task_result.json"
        )
    if recorded_result.is_symlink():
        raise RuntimeError("External task task_result path must not be a symbolic link")

    input_root = box_dir / "input"
    recorded_input = Path(str(task_meta.get("input_snapshot", ""))).expanduser()
    if (
        not recorded_input.is_absolute()
        or input_root.is_symlink()
        or not input_root.is_dir()
        or recorded_input.is_symlink()
        or not recorded_input.is_file()
    ):
        raise RuntimeError(
            "External task input_snapshot must be a regular file inside its input directory"
        )
    input_root_resolved = input_root.resolve(strict=True)
    input_resolved = recorded_input.resolve(strict=True)
    try:
        input_resolved.relative_to(input_root_resolved)
    except ValueError as exc:
        raise RuntimeError(
            "External task input_snapshot escapes its adjacent input directory"
        ) from exc

    input_manifest = task_data.get("input_manifest", {})
    structure_identity = (
        input_manifest.get("structure_snapshot", {})
        if isinstance(input_manifest, Mapping)
        else {}
    )
    manifest_input = (
        Path(str(structure_identity.get("path", ""))).expanduser()
        if isinstance(structure_identity, Mapping)
        else Path("")
    )
    if (
        not manifest_input.is_absolute()
        or manifest_input.is_symlink()
        or not manifest_input.is_file()
        or manifest_input.resolve(strict=True) != input_resolved
    ):
        raise RuntimeError(
            "External task input_snapshot disagrees with its protected input manifest"
        )
    return task_path, box_dir, task_data


def _assert_external_task_manifest_unchanged(task_data: Mapping[str, Any]) -> None:
    """Prove the adjacent task bytes still describe the executing task."""

    task_meta = task_data.get("task", {})
    if not isinstance(task_meta, Mapping):
        raise RuntimeError("External task manifest has no task metadata mapping")
    _path, current = _read_stable_external_task_manifest(
        Path(str(task_meta.get("task_json", "")))
    )
    if current != dict(task_data):
        raise RuntimeError(
            "External task manifest changed during production-box execution"
        )


def _locked_production_box_task(function):
    """Serialize duplicate local/Slurm invocations of one box task."""

    @wraps(function)
    def wrapped(
        task: Path | Mapping[str, Any],
        *,
        retry_existing_result: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        task_path, box_dir, task_data = _validated_external_task_context(task)
        initial_manifest_sha256 = _task_manifest_digest(task_data)
        task_meta = task_data["task"]
        box_label = task_meta.get("box", "?")
        with exclusive_workflow_lock(
            box_dir,
            purpose=f"production box task {box_label}",
        ):
            _locked_task_path, locked_box_dir, locked_task_data = (
                _validated_external_task_context(task_path)
            )
            if (
                locked_box_dir != box_dir
                or _task_manifest_digest(locked_task_data)
                != initial_manifest_sha256
            ):
                raise RuntimeError(
                    "External task manifest changed before its task lock was acquired"
                )
            return function(
                locked_task_data,
                retry_existing_result=retry_existing_result,
            )

    return wrapped


def _task_result_retry_authorization(
    *,
    task_data: Mapping[str, Any],
    task_result: Path,
) -> dict[str, str]:
    """Bind an orchestrator retry to the exact result it inspected."""

    result_path = Path(task_result)
    if result_path.is_symlink() or not result_path.is_file():
        raise RuntimeError(
            f"Cannot authorize retry from a non-regular task result: {result_path}"
        )
    return {
        "schema": "vitriflow.task_result_retry.v1",
        "task_manifest_sha256": _task_manifest_digest(task_data),
        "existing_result_sha256": _sha256_file(result_path),
    }


def _consume_task_result_retry_authorization(
    *,
    task_data: Mapping[str, Any],
    task_result: Path,
    authorization: Optional[Mapping[str, Any]],
) -> Path:
    """Validate and archive one exact non-reusable result before retrying."""

    result_path = Path(task_result)
    if result_path.is_symlink() or not result_path.is_file():
        raise RuntimeError(
            f"Refusing to replace non-regular task result: {result_path}"
        )
    if not isinstance(authorization, Mapping):
        raise RuntimeError(
            "Refusing to overwrite an existing non-reusable task result. "
            "Retry through `vitriflow run --external-mode full-run --resume`, "
            "which authenticates and archives the exact prior result."
        )
    if str(authorization.get("schema", "")) != "vitriflow.task_result_retry.v1":
        raise RuntimeError("Invalid external task-result retry authorization schema")
    task_sha = _task_manifest_digest(task_data)
    if str(authorization.get("task_manifest_sha256", "")).lower() != task_sha:
        raise RuntimeError(
            "External task-result retry authorization belongs to a different task manifest"
        )
    existing_sha = _sha256_file(result_path)
    if str(authorization.get("existing_result_sha256", "")).lower() != existing_sha:
        raise RuntimeError(
            "External task result changed after retry authorization; refusing overwrite"
        )

    archive = result_path.with_name(
        f"{result_path.name}.superseded-{existing_sha[:16]}"
    )
    if archive.exists() or archive.is_symlink():
        if archive.is_symlink() or not archive.is_file() or _sha256_file(archive) != existing_sha:
            raise RuntimeError(
                f"Cannot preserve superseded task result without clobbering {archive}"
            )
        result_path.unlink()
    else:
        os.replace(result_path, archive)
    return archive


def _quarantine_incomplete_task_execution(box_dir: Path) -> Optional[Path]:
    """Preserve partial stage trees before an uncommitted task is rerun."""

    box = Path(box_dir)
    if box.is_symlink() or not box.is_dir():
        raise RuntimeError(f"External task box must be a real directory: {box}")
    box = box.resolve(strict=True)
    children = [
        box / name
        for name in ("warmup", "melt", "quench", "relax", "continuous")
        if (box / name).exists() or (box / name).is_symlink()
    ]
    if not children:
        return None
    production_root = box.parent.resolve(strict=True)
    quarantine_parent = production_root / "interrupted_task_attempts"
    quarantine_root = quarantine_parent / box.name
    for candidate in (quarantine_parent, quarantine_root):
        if candidate.is_symlink():
            raise RuntimeError(
                "External task quarantine must not contain a symbolic link: "
                f"{candidate}"
            )
        try:
            candidate.resolve(strict=False).relative_to(production_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "External task quarantine must remain inside the production root"
            ) from exc
    quarantine_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        destination = quarantine_root / f"attempt_{attempt:03d}"
        if not destination.exists() and not destination.is_symlink():
            break
        attempt += 1
    destination.mkdir()
    for child in children:
        os.replace(child, destination / child.name)
    return destination


@_locked_production_box_task
def execute_production_box_task(
    task: Path | Mapping[str, Any],
    *,
    retry_existing_result: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Production box task."""

    task_data = dict(task) if isinstance(task, Mapping) else json.loads(Path(task).read_text())
    if str(task_data.get("schema", "")).strip().lower() != "vitriflow.box_task.v1":
        raise ValueError("Unsupported task manifest schema")

    task_manifest_sha256 = _task_manifest_digest(task_data)

    task_meta = dict(task_data.get("task", {}) or {})
    box_id = int(task_meta.get("box"))
    box_dir = Path(str(task_meta.get("box_dir")))
    input_snapshot = Path(str(task_meta.get("input_snapshot")))
    task_result = Path(str(task_meta.get("task_result", box_dir / "task_result.json")))

    if task_result.is_symlink():
        raise RuntimeError(
            f"Refusing symbolic-link task result: {task_result}"
        )

    cached: Optional[Mapping[str, Any]] = None
    if task_result.exists():
        try:
            candidate = json.loads(task_result.read_text())
            if isinstance(candidate, Mapping) and _cached_task_result_is_reusable(
                task_data=task_data,
                cached=candidate,
            ):
                cached = candidate
        except Exception:
            cached = None
        # A manifest/result mismatch must be authorized and archived before
        # validating (and potentially executing) any command from the new task.
        if cached is None:
            _consume_task_result_retry_authorization(
                task_data=task_data,
                task_result=task_result,
                authorization=retry_existing_result,
            )
    elif retry_existing_result is not None:
        raise RuntimeError(
            "External task result disappeared after retry authorization; refusing "
            "an unaudited retry"
        )

    # Authenticate the task/runtime and query the concrete worker build before
    # returning a cached success.  Otherwise the same task_result.json could be
    # silently reused after a module, container, PATH, or executable switch.
    if not _task_runtime_is_current(task_data):
        task_runtime = task_data.get("runtime", {})
        recorded = task_runtime.get("vitriflow_version", "missing") if isinstance(task_runtime, Mapping) else "missing"
        recorded_content = (
            task_runtime.get("package_content", {}).get("sha256", "missing")
            if isinstance(task_runtime, Mapping)
            and isinstance(task_runtime.get("package_content"), Mapping)
            else "missing"
        )
        executing_runtime = _runtime_identity()
        raise RuntimeError(
            "task runtime identity mismatch: "
            f"manifest_version={recorded!r}, "
            f"executing_version={executing_runtime['vitriflow_version']!r}, "
            f"manifest_content={recorded_content!r}, "
            f"executing_content={executing_runtime['package_content']['sha256']!r}; "
            "rematerialise the task with the installed VitriFlow package"
        )
    if not _task_inputs_are_current(task_data):
        raise RuntimeError(
            "task input fingerprint mismatch: structure or potential files changed after task materialisation"
        )
    config = RunConfig.model_validate(task_data.get("config", {}))
    plan = dict(task_data.get("production_plan", {}) or {})
    engine_build_identity = _task_engine_build_identity(
        config=config,
        plan=plan,
        workdir=box_dir,
    )
    if cached is not None:
        if _cached_task_result_is_reusable(
            task_data=task_data,
            cached=cached,
            current_engine_identity=engine_build_identity,
        ):
            return dict(cached)
        _consume_task_result_retry_authorization(
            task_data=task_data,
            task_result=task_result,
            authorization=retry_existing_result,
        )

    _quarantine_incomplete_task_execution(box_dir)

    try:
        stages, stage_diag, runner, pot_cfg, type_to_species, potential_lines, force_iso, stage_cont = _stage_specs_for_box(
            config=config,
            plan=plan,
            box_dir=box_dir,
            input_snapshot=input_snapshot,
        )
        engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()
        md_use = MDConfig.model_validate(plan.get("md_use", {}))

        if engine == "lammps":
            ensure_potential_model_installed(
                pot_cfg,
                installer=ensure_model_installed,
            )
            if stage_cont == "continuous" and isinstance(runner, LammpsRunner):
                arts = run_stages_continuous_lammps(
                    runner,
                    pot_cfg,
                    md_use,
                    list(stages),
                    [box_dir / "warmup", box_dir / "melt", box_dir / "quench", box_dir / "relax"],
                    box_dir / "continuous",
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )
                outcomes = [stage_outcome_from_artifacts(art, md_cfg=md_use, stage=stage) for art, stage in zip(arts, stages)]
            else:
                outcomes = []
                for stage in stages:
                    stage_dir = box_dir / str(stage.name)
                    art = run_stage_local(
                        runner,
                        pot_cfg,
                        md_use,
                        stage,
                        stage_dir,
                        potential_lines=potential_lines,
                        log_name="log.lammps",
                        type_to_species=type_to_species,
                    )
                    outcomes.append(stage_outcome_from_artifacts(art, md_cfg=md_use, stage=stage))
        else:
            outcomes = []
            for stage in stages:
                stage_dir = box_dir / str(stage.name)
                art = run_stage_local(
                    runner,
                    pot_cfg,
                    md_use,
                    stage,
                    stage_dir,
                    potential_lines=None,
                    log_name="cp2k.out",
                    type_to_species=type_to_species,
                )
                outcomes.append(stage_outcome_from_artifacts(art, md_cfg=md_use, stage=stage))

        outcome_map = {str(stage.name): asdict(out) for stage, out in zip(stages, outcomes)}
        diagnostics = _run_task_production_diagnostics(
            config=config,
            plan=plan,
            box_dir=box_dir,
            stage_diag=stage_diag,
            runner=runner,
            pot_cfg=pot_cfg,
            md_use=md_use,
            type_to_species=type_to_species,
            potential_lines=potential_lines,
            force_isotropic=bool(force_iso),
            outcomes=outcome_map,
        )
        declared_diagnostic_plan = task_data.get("diagnostic_plan", None)
        if not isinstance(declared_diagnostic_plan, Mapping):
            raise RuntimeError(
                "task manifest is missing the protected diagnostic plan"
            )
        if dict(declared_diagnostic_plan) != dict(diagnostics.get("plan", {})):
            raise RuntimeError(
                "task diagnostic execution plan disagrees with its manifest"
            )
        if not _task_inputs_are_current(task_data):
            raise RuntimeError(
                "task input fingerprint changed during execution: structure, "
                "potential, command include, or CP2K data bytes no longer match "
                "the protected task manifest"
            )
        _assert_external_task_manifest_unchanged(task_data)
        final_engine_build_identity = _task_engine_build_identity(
            config=config,
            plan=plan,
            workdir=box_dir,
        )
        _assert_task_engine_build_identity_unchanged(
            engine_build_identity,
            final_engine_build_identity,
        )
        relax_out = outcomes[-1]
        result = {
            "schema": TASK_RESULT_SCHEMA,
            "runtime": _runtime_identity(),
            "status": "ok",
            "box": int(box_id),
            "engine": engine,
            "engine_build_identity": engine_build_identity,
            "engine_build_identity_end_verified": True,
            "task": task_meta,
            "task_manifest_sha256": task_manifest_sha256,
            "seeds": _box_seed_map(int(plan.get("seed_base", config.random_seed + 13579)), int(box_id)),
            "outcomes": outcome_map,
            "diagnostics": diagnostics,
            "artifact_manifest": _build_task_artifact_manifest(
                box_dir=box_dir,
                outcomes=outcome_map,
                diagnostics=diagnostics,
            ),
            "density": float(relax_out.density_mean),
            "density_stderr": float(relax_out.density_stderr),
            "paths": {
                "box_dir": str(box_dir),
                "warmup_dir": str(box_dir / "warmup"),
                "melt_dir": str(box_dir / "melt"),
                "quench_dir": str(box_dir / "quench"),
                "relax_dir": str(box_dir / "relax"),
                "task_json": str(task_meta.get("task_json", "")),
                "input_snapshot": str(input_snapshot),
            },
        }
        result = seal_task_result(result)
        atomic_write_json(task_result, result)
        return result
    except Exception as exc:
        final_engine_build_identity: Optional[dict[str, Any]] = None
        engine_end_error: Optional[str] = None
        try:
            final_engine_build_identity = _task_engine_build_identity(
                config=config,
                plan=plan,
                workdir=box_dir,
            )
            _assert_task_engine_build_identity_unchanged(
                engine_build_identity,
                final_engine_build_identity,
            )
        except Exception as identity_exc:
            engine_end_error = (
                "worker end-of-task engine verification failed: "
                f"{type(identity_exc).__name__}: {identity_exc}"
            )
        error_text = str(exc)
        if engine_end_error is not None:
            error_text = f"{error_text}; {engine_end_error}"
        failed = {
            "schema": TASK_RESULT_SCHEMA,
            "runtime": _runtime_identity(),
            "status": "failed",
            "box": int(box_id),
            "engine_build_identity": engine_build_identity,
            "engine_build_identity_end_verified": engine_end_error is None,
            "engine_build_identity_final": final_engine_build_identity,
            "task": task_meta,
            "task_manifest_sha256": task_manifest_sha256,
            "error": error_text,
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(task_result, seal_task_result(failed))
        raise RuntimeError(
            f"production box task {box_id} failed: {error_text}"
        ) from exc


def _execute_task_path(
    task_json: Path,
    retry_existing_result: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if retry_existing_result is None:
        return execute_production_box_task(task_json)
    return execute_production_box_task(
        task_json,
        retry_existing_result=retry_existing_result,
    )


def _summarise_analysis_as_production_state(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    prod_cfg: ProductionEnsembleConfig,
    analysis: Mapping[str, Any],
    mode: str,
    max_parallel_boxes: int,
    job_template: Optional[Path],
    planned_boxes: int,
    convergence_streak: int,
    required_convergence_streak: int,
    last_convergence_evaluated_n_boxes_total: int | None,
    last_convergence_evaluated_n_boxes_accepted: int | None,
    convergence_look_history: Sequence[Mapping[str, Any]],
    converged_md: bool,
    converged: bool,
    engine_build_identity: Optional[Mapping[str, Any]] = None,
    status_override: str | None = None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    prod_dir = outdir / "production"
    md_use = MDConfig.model_validate(plan.get("md_use", {}))
    pot_cfg = _potential_from_dict(plan.get("potential_config", None), config.kim)
    warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    warmup_steps = resolve_production_warmup_steps(
        prod_cfg=prod_cfg,
        md_timestep=float(md_use.timestep),
        time_unit_ps=resolve_production_time_unit_ps(
            config=config,
            engine=str(plan.get("engine", getattr(config, "engine", "lammps"))),
            pot_cfg=pot_cfg,
            time_unit_ps=plan.get("time_unit_ps", None),
        ),
    )
    max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
    max_boxes: int | None
    if max_boxes_raw is None:
        max_boxes = None
    else:
        try:
            max_boxes = int(max_boxes_raw)
        except Exception:
            max_boxes = None
        if max_boxes is not None and max_boxes <= 0:
            max_boxes = None
    check_convergence = bool(getattr(prod_cfg, "check_convergence", True))
    n_accepted = int(analysis.get("n_boxes_accepted", analysis.get("n_boxes", 0)))
    n_rejected = int(analysis.get("n_boxes_rejected", 0))
    n_total = int(analysis.get("n_boxes_total", n_accepted + n_rejected))
    final_status, final_error = production_final_status(
        n_accepted=n_accepted,
        min_boxes=int(getattr(prod_cfg, "min_boxes", 0)),
        check_convergence=check_convergence,
        converged=bool(converged),
        max_boxes=max_boxes,
        n_total=n_total,
    )
    status = str(status_override) if status_override is not None else final_status
    terminal = status in {"ok", "incomplete", "not_converged"}
    store_distributions = bool(getattr(prod_cfg, "store_distributions", True))
    resumable = not (terminal and check_convergence and not store_distributions)
    report_boxes = [
        _normalise_analysis_entry_paths(entry)
        for entry in list(analysis.get("boxes", []))
    ]
    report_rejected = [
        _normalise_analysis_entry_paths(entry)
        for entry in list(analysis.get("rejected_boxes", []))
    ]
    if terminal and not store_distributions:
        for entry in report_boxes + report_rejected:
            entry.pop("distributions", None)
            dft_entry = entry.get("dft_opt")
            if isinstance(dft_entry, dict):
                dft_entry.pop("distributions", None)
    graph_outputs = _normalise_analysis_graph_outputs(
        dict(analysis.get("graph_outputs", {}) or {})
    )
    convergence_report = dict(analysis.get("convergence", {}) or {})
    if not check_convergence:
        # analyze_output_data performs the one terminal fixed-n diagnostic.
        # Preserve that evidence here while keeping the stopping decision
        # explicitly unassessed for backward-compatible production semantics.
        convergence_report.setdefault(
            "status", "fixed_n_terminal_posthoc_unassessed"
        )
        convergence_report.setdefault("sampling_design", "fixed_n")
        convergence_report.setdefault(
            "assessment_role", "terminal_posthoc_diagnostic"
        )
        convergence_report.setdefault("assessment_performed", False)
        convergence_report["check_convergence"] = False
        convergence_report["used_for_stopping"] = False
        convergence_report["stopping_assessment_performed"] = False
        convergence_report["stopping_status"] = "fixed_count_unassessed"
        convergence_report.setdefault(
            "sequential_inference_status", "not_sequentially_valid"
        )
        convergence_report["execution_target_met"] = bool(
            n_accepted >= int(getattr(prod_cfg, "min_boxes", 0))
        )
    if not check_convergence:
        convergence_inference_status = (
            "fixed_n_terminal_posthoc_not_sequentially_valid"
            if bool(convergence_report.get("assessment_performed", False))
            else "fixed_count_unassessed"
        )
    elif bool(converged):
        convergence_inference_status = (
            "criterion_met_repeated_looks_not_sequentially_valid"
        )
    elif convergence_report:
        convergence_inference_status = (
            "criterion_not_met_or_unassessed_repeated_looks_not_sequentially_valid"
        )
    else:
        convergence_inference_status = "not_yet_assessed"
    convergence_degree = dict(convergence_report.get("convergence_degree", {}) or {})
    criterion_coverage = {
        key: dict(convergence_degree.get(key, {}) or {})
        for key in ("ci", "stability", "criteria_integrity", "overall")
        if isinstance(convergence_degree.get(key), Mapping)
    }
    return {
        "enabled": True,
        "status": status,
        "execution_status": "completed" if terminal else status,
        "error": final_error if terminal else analysis.get("error", None),
        "converged": (bool(converged) if check_convergence else None),
        "convergence_status": (
            "converged"
            if check_convergence and bool(converged)
            else (
                "not_converged"
                if check_convergence
                else "fixed_count_unassessed"
            )
        ),
        "convergence_inference_status": str(convergence_inference_status),
        "achieved_convergence_degree": (
            dict(convergence_report.get("achieved_convergence_degree", {}) or {})
            if isinstance(convergence_report.get("achieved_convergence_degree"), Mapping)
            else None
        ),
        "posthoc_convergence_criterion_met": (
            convergence_report.get("posthoc_criterion_met")
            if not check_convergence
            else None
        ),
        "posthoc_convergence_failed_items": (
            list(convergence_report.get("posthoc_failed_items", []) or [])
            if not check_convergence
            else None
        ),
        "convergence_criterion_coverage": (
            criterion_coverage if criterion_coverage else None
        ),
        "n_boxes": int(analysis.get("n_boxes", 0)),
        "n_boxes_accepted": int(analysis.get("n_boxes_accepted", analysis.get("n_boxes", 0))),
        "n_boxes_rejected": int(analysis.get("n_boxes_rejected", 0)),
        "n_boxes_total": int(analysis.get("n_boxes_total", 0)),
        "min_boxes": int(getattr(prod_cfg, "min_boxes", 0)),
        "max_boxes": max_boxes,
        "batch_boxes": int(getattr(prod_cfg, "batch_boxes", 1)),
        "check_convergence": check_convergence,
        "resumable": bool(resumable),
        "non_resumable_reason": (
            "adaptive convergence distributions were omitted from the terminal result"
            if not resumable
            else None
        ),
        "convergence_streak": int(convergence_streak),
        "required_convergence_streak": int(required_convergence_streak),
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
        "convergence_look_history": [
            dict(record) for record in convergence_look_history
        ],
        "engine_build_identity": (
            None
            if engine_build_identity is None
            else validate_engine_build_identity(engine_build_identity)
        ),
        "engine_build_identity_status": (
            "no_successful_task_results"
            if engine_build_identity is None
            else "verified_homogeneous_workers"
        ),
        "dump_trajectory": bool(getattr(prod_cfg, "dump_trajectory", True)),
        "dump_every_steps": int(getattr(prod_cfg, "dump_every_steps", 5000)),
        "rate_K_per_time": float(plan.get("chosen_rate")),
        "rate_K_per_ps": (None if plan.get("cooling_rate_ps", None) is None else float(plan.get("cooling_rate_ps"))),
        "replicate": [int(x) for x in plan.get("replicate", [1, 1, 1])],
        "structure_data": _relpath_or_str(Path(str(plan.get("structure_data"))), outdir),
        "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
        "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")),
        "rejects_dir": (_relpath_or_str(prod_dir / str(getattr(prod_cfg, "rejects_subdir", "rejects")), outdir) if (prod_dir / str(getattr(prod_cfg, "rejects_subdir", "rejects"))).exists() else None),
        "warmup_start_temperature": float(getattr(prod_cfg, "warmup_start_temperature", 300.0)),
        "warmup_duration_ps": float(warmup_duration_ps),
        "warmup_steps": int(warmup_steps),
        "T_high": float(plan.get("T_high")),
        "t_final": float(plan.get("t_final")),
        "quench_steps": int(plan.get("quench_steps")),
        "highT_steps": int(plan.get("high_total_steps")),
        "relax_steps": int(plan.get("relax_steps")),
        "cutoffs": list(analysis.get("cutoffs", [])),
        "metrics_checked": analysis.get("metrics_checked", None),
        "convergence_spec": analysis.get("convergence_spec", None),
        "converged_md": (bool(converged_md) if check_convergence else None),
        "convergence_md": convergence_report,
        "converged_dft": None,
        "convergence_dft": None,
        "convergence": convergence_report,
        "dft_opt": None,
        "boxes_dft_final": None,
        "n_boxes_dft_accepted": None,
        "rejected_boxes_dft": None,
        "boxes": report_boxes,
        "rejected_boxes": report_rejected,
        "graph_outputs": graph_outputs,
        "paths": graph_outputs,
        "ensemble_dir": "production",
        "execution": {
            "mode": str(mode),
            "planned_boxes": int(planned_boxes),
            "max_parallel_boxes": int(max_parallel_boxes),
            "job_template": (None if job_template is None else str(job_template)),
            "output_dataset": "production/output_dataset.json",
            "analysis_results": "production/analysis_results.json",
        },
    }


def dry_run_external_production(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    job_template: Optional[Path] = None,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    _guard_external_production_feature_parity(config=config, plan=plan)
    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    md_use = MDConfig.model_validate(plan.get("md_use", {}))
    pot_cfg = _potential_from_dict(plan.get("potential_config", None), config.kim)
    warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    warmup_steps = resolve_production_warmup_steps(
        prod_cfg=prod_cfg,
        md_timestep=float(md_use.timestep),
        time_unit_ps=resolve_production_time_unit_ps(
            config=config,
            engine=str(plan.get("engine", getattr(config, "engine", "lammps"))),
            pot_cfg=pot_cfg,
            time_unit_ps=plan.get("time_unit_ps", None),
        ),
    )
    materialized = materialize_external_production(
        config=config,
        outdir=outdir,
        plan=plan,
        job_template=job_template,
        progress=progress,
    )
    return {
        "enabled": True,
        "status": "planned",
        "execution_status": "planned",
        "error": None,
        "converged": False,
        "n_boxes": 0,
        "n_boxes_accepted": 0,
        "n_boxes_rejected": 0,
        # Planned tasks are not attempted production boxes.  Reporting them as
        # n_boxes_total made dry-run summaries scientifically misleading.
        "n_boxes_total": 0,
        "min_boxes": int(getattr(prod_cfg, "min_boxes", 0)),
        "max_boxes": getattr(prod_cfg, "max_boxes", None),
        "batch_boxes": int(getattr(prod_cfg, "batch_boxes", 1)),
        "check_convergence": bool(getattr(prod_cfg, "check_convergence", True)),
        "resumable": True,
        "non_resumable_reason": None,
        "convergence_streak": 0,
        "required_convergence_streak": max(
            1, int(getattr(prod_cfg, "consecutive_converged_checks", 1))
        ),
        "last_convergence_evaluated_n_boxes_total": None,
        "last_convergence_evaluated_n_boxes_accepted": None,
        "engine_build_identity": None,
        "engine_build_identity_status": "deferred_to_external_worker",
        "dump_trajectory": bool(getattr(prod_cfg, "dump_trajectory", True)),
        "dump_every_steps": int(getattr(prod_cfg, "dump_every_steps", 5000)),
        "rate_K_per_time": float(plan.get("chosen_rate")),
        "rate_K_per_ps": (None if plan.get("cooling_rate_ps", None) is None else float(plan.get("cooling_rate_ps"))),
        "replicate": [int(x) for x in plan.get("replicate", [1, 1, 1])],
        "structure_data": _relpath_or_str(Path(str(plan.get("structure_data"))), Path(outdir)),
        "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
        "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")),
        "rejects_dir": None,
        "warmup_start_temperature": float(getattr(prod_cfg, "warmup_start_temperature", 300.0)),
        "warmup_duration_ps": float(warmup_duration_ps),
        "warmup_steps": int(warmup_steps),
        "T_high": float(plan.get("T_high")),
        "t_final": float(plan.get("t_final")),
        "quench_steps": int(plan.get("quench_steps")),
        "highT_steps": int(plan.get("high_total_steps")),
        "relax_steps": int(plan.get("relax_steps")),
        "cutoffs": list(plan.get("preferred_cutoffs", []) or plan.get("cutoffs_size", []) or plan.get("cutoffs_rate", [])),
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
        "execution": {
            "mode": "dry-run",
            **materialized,
        },
    }


def full_run_external_production(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    job_template: Optional[Path] = None,
    max_parallel_boxes: int = 1,
    progress: Optional[CondensedProgressLog] = None,
    resume_state: Optional[Mapping[str, Any]] = None,
    checkpoint_cb=None,
) -> dict[str, Any]:
    _guard_external_production_feature_parity(config=config, plan=plan)
    outdir = Path(outdir)
    prod_dir = outdir / "production"
    ensure_dir(prod_dir)
    if progress is None:
        progress = CondensedProgressLog(outdir / "condensed.log")

    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    dft_enabled = bool(getattr(getattr(prod_cfg, "dft_opt", None), "enabled", False))
    if dft_enabled:
        raise ValueError("external dry/full-run does not support production.dft_opt refinement")
    # Do not probe the engine on the orchestration/login node.  A Slurm job
    # template may load the configured LAMMPS/CP2K module only inside the
    # worker allocation, so the login node is deliberately outside the
    # scientific execution-identity boundary.  Each successful v3 task proves
    # its own start/end worker identity below; the collector then requires one
    # homogeneous identity and, on resume, equality with the identity already
    # protected by the checkpoint.

    target = max(1, int(getattr(prod_cfg, "min_boxes", 1) or 1))
    batch = max(1, int(getattr(prod_cfg, "batch_boxes", 1) or 1))
    do_converge = bool(getattr(prod_cfg, "check_convergence", True))
    required_streak = max(
        1, int(getattr(prod_cfg, "consecutive_converged_checks", 1))
    )
    converged_streak = 0
    converged_now = False
    converged = False
    last_convergence_evaluated_n_boxes_total: int | None = None
    last_convergence_evaluated_n_boxes_accepted: int | None = None
    convergence_look_history: list[dict[str, Any]] = []
    if resume_state is not None:
        stored_required = int(
            resume_state.get("required_convergence_streak", required_streak)
        )
        if stored_required != required_streak:
            raise RuntimeError(
                "Cannot safely resume external production: required convergence streak changed"
            )
        converged_streak = int(resume_state.get("convergence_streak", 0))
        converged_now = bool(resume_state.get("converged_md", False))
        last_total = resume_state.get("last_convergence_evaluated_n_boxes_total")
        last_accepted = resume_state.get(
            "last_convergence_evaluated_n_boxes_accepted"
        )
        last_convergence_evaluated_n_boxes_total = (
            None if last_total is None else int(last_total)
        )
        last_convergence_evaluated_n_boxes_accepted = (
            None if last_accepted is None else int(last_accepted)
        )
        convergence_look_history = _validated_convergence_look_history(
            resume_state.get("convergence_look_history", [])
        )
        if convergence_look_history:
            last_look = convergence_look_history[-1]
            if (
                last_convergence_evaluated_n_boxes_total
                != int(last_look["n_boxes_total"])
                or last_convergence_evaluated_n_boxes_accepted
                != int(last_look["n_boxes_accepted"])
                or converged_streak
                != int(last_look["convergence_streak_after"])
            ):
                raise RuntimeError(
                    "Cannot safely resume external production: convergence look "
                    "history disagrees with the checkpoint counters"
                )
            if bool(converged_now) != bool(last_look["criterion_met"]):
                raise RuntimeError(
                    "Cannot safely resume external production: the persisted "
                    "convergence decision disagrees with the final look-history record"
                )

    max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
    max_boxes: Optional[int]
    if max_boxes_raw is None:
        max_boxes = None
    else:
        try:
            max_boxes = int(max_boxes_raw)
        except Exception:
            max_boxes = None
        if max_boxes is not None and max_boxes <= 0:
            max_boxes = None
        if max_boxes is not None and max_boxes < target:
            max_boxes = target

    if last_convergence_evaluated_n_boxes_total is not None:
        target = max(target, int(last_convergence_evaluated_n_boxes_total))
    if max_boxes is None and target > _HARD_MAX_EXTERNAL_BOXES:
        raise RuntimeError(
            "Cannot safely resume external production: checkpoint exceeds the hard "
            f"limit of {_HARD_MAX_EXTERNAL_BOXES} attempted boxes"
        )
    planned_initial = int(target)
    materialize_external_production(
        config=config,
        outdir=outdir,
        plan=plan,
        job_template=job_template,
        n_boxes=planned_initial,
        progress=progress,
    )

    analysis: dict[str, Any] = {
        "status": "ok",
        "converged": False,
        "n_boxes": 0,
        "n_boxes_total": 0,
        "boxes": [],
        "rejected_boxes": [],
        "convergence": {},
        "cutoffs": list(plan.get("preferred_cutoffs", []) or plan.get("cutoffs_size", []) or plan.get("cutoffs_rate", [])),
    }

    while True:
        task_attempts: list[tuple[Path, Optional[dict[str, str]]]] = []
        for box_id in range(1, int(target) + 1):
            task_json = prod_dir / f"box_{box_id:03d}" / "task.json"
            if not task_json.exists():
                raise FileNotFoundError(f"Missing task manifest: {task_json}")
            task_result = prod_dir / f"box_{box_id:03d}" / "task_result.json"
            if task_result.is_symlink():
                raise RuntimeError(
                    f"Refusing symbolic-link task result during external resume: {task_result}"
                )
            if not task_result.exists():
                task_attempts.append((task_json, None))
                continue
            try:
                task_data = json.loads(task_json.read_text())
                cached = json.loads(task_result.read_text())
                if not (
                    isinstance(task_data, Mapping)
                    and isinstance(cached, Mapping)
                    and _cached_task_result_is_reusable(
                        task_data=task_data,
                        cached=cached,
                    )
                ):
                    if not isinstance(task_data, Mapping):
                        raise RuntimeError(f"Malformed task manifest: {task_json}")
                    task_attempts.append(
                        (
                            task_json,
                            _task_result_retry_authorization(
                                task_data=task_data,
                                task_result=task_result,
                            ),
                        )
                    )
            except (OSError, UnicodeError, json.JSONDecodeError):
                try:
                    task_data = json.loads(task_json.read_text())
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Malformed task manifest: {task_json}") from exc
                if not isinstance(task_data, Mapping):
                    raise RuntimeError(f"Malformed task manifest: {task_json}")
                task_attempts.append(
                    (
                        task_json,
                        _task_result_retry_authorization(
                            task_data=task_data,
                            task_result=task_result,
                        ),
                    )
                )

        if task_attempts:
            workers = max(1, int(max_parallel_boxes))
            progress.info("external", f"executing {len(task_attempts)} task(s) with max_parallel_boxes={workers}")
            if workers == 1:
                for task_json, retry_authorization in task_attempts:
                    _execute_task_path(task_json, retry_authorization)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = {
                        pool.submit(
                            _execute_task_path,
                            task_json,
                            retry_authorization,
                        ): task_json
                        for task_json, retry_authorization in task_attempts
                    }
                    for fut in as_completed(futs):
                        fut.result()

        worker_engine_identity = validate_external_task_engine_identities(
            production_dir=prod_dir,
            box_ids=list(range(1, int(target) + 1)),
            resume_state=resume_state,
        )
        analysis = analyze_output_data(
            config=config,
            input_path=prod_dir,
            outdir=prod_dir,
            plan=plan,
            progress=progress,
        )

        total_boxes = int(analysis.get("n_boxes_total", 0))
        accepted_boxes = int(analysis.get("n_boxes_accepted", analysis.get("n_boxes", 0)))
        if total_boxes != int(target):
            raise RuntimeError(
                "External production convergence must be evaluated on the exact "
                f"configured prefix of {int(target)} attempted boxes; analysis "
                f"discovered {total_boxes}. Inspect output_dataset.json, task results, "
                "and analysis source selection."
            )
        accepted_box_ids, accepted_evidence_sha256 = _accepted_evidence_identity(
            analysis
        )
        if len(accepted_box_ids) != accepted_boxes:
            raise RuntimeError(
                "External production analysis accepted-box count disagrees with "
                "its accepted box entries"
            )
        already_evaluated = (
            last_convergence_evaluated_n_boxes_total == total_boxes
            and last_convergence_evaluated_n_boxes_accepted == accepted_boxes
        )
        if already_evaluated and convergence_look_history:
            prior = convergence_look_history[-1]
            if (
                list(prior["accepted_box_ids"]) != accepted_box_ids
                or str(prior["accepted_evidence_sha256"])
                != accepted_evidence_sha256
            ):
                raise RuntimeError(
                    "Cannot safely resume external production: the already-evaluated "
                    "prefix now contains different accepted evidence"
                )
        if not already_evaluated:
            previous_accepted = last_convergence_evaluated_n_boxes_accepted
            previous_ids = (
                set(convergence_look_history[-1]["accepted_box_ids"])
                if convergence_look_history
                else set()
            )
            current_ids = set(accepted_box_ids)
            if convergence_look_history:
                new_accepted_evidence = bool(
                    current_ids > previous_ids
                    and all(
                        int(box_id) in current_ids for box_id in previous_ids
                    )
                )
                accepted_ensemble_changed = (
                    current_ids != previous_ids
                    or str(convergence_look_history[-1]["accepted_evidence_sha256"])
                    != accepted_evidence_sha256
                )
            else:
                # Legacy checkpoints did not persist accepted ids.  Count
                # growth is the only safely provable new evidence; an equal
                # count can never advance the streak.
                accepted_ensemble_changed = (
                    previous_accepted is None
                    or accepted_boxes != int(previous_accepted)
                )
                new_accepted_evidence = (
                    previous_accepted is None
                    or accepted_boxes > int(previous_accepted)
                )
            # External analysis is rerun when attempted tasks change, but a
            # rejected-only task leaves the accepted ensemble identical.  It
            # must not advance a consecutive-passing-look streak.
            if accepted_ensemble_changed:
                converged_now = bool(analysis.get("converged", False))
                if (
                    new_accepted_evidence
                    and converged_now
                    and accepted_boxes >= int(getattr(prod_cfg, "min_boxes", 0))
                ):
                    converged_streak += 1
                else:
                    converged_streak = 0
            last_convergence_evaluated_n_boxes_total = total_boxes
            last_convergence_evaluated_n_boxes_accepted = accepted_boxes
            convergence_look_history.append(
                {
                    "n_boxes_total": int(total_boxes),
                    "n_boxes_accepted": int(accepted_boxes),
                    "accepted_box_ids": list(accepted_box_ids),
                    "accepted_evidence_sha256": str(
                        accepted_evidence_sha256
                    ),
                    "criterion_met": bool(converged_now),
                    "advanced_streak": bool(
                        new_accepted_evidence
                        and converged_now
                        and accepted_boxes
                        >= int(getattr(prod_cfg, "min_boxes", 0))
                    ),
                    "convergence_streak_after": int(converged_streak),
                }
            )

        converged = (
            accepted_boxes >= int(getattr(prod_cfg, "min_boxes", 0))
            if not do_converge
            else bool(converged_now and converged_streak >= required_streak)
        )
        if checkpoint_cb is not None:
            checkpoint_cb(
                _summarise_analysis_as_production_state(
                    config=config,
                    outdir=outdir,
                    plan=plan,
                    prod_cfg=prod_cfg,
                    analysis=analysis,
                    mode="full-run",
                    max_parallel_boxes=max(1, int(max_parallel_boxes)),
                    job_template=job_template,
                    planned_boxes=planned_initial,
                    convergence_streak=converged_streak,
                    required_convergence_streak=required_streak,
                    last_convergence_evaluated_n_boxes_total=last_convergence_evaluated_n_boxes_total,
                    last_convergence_evaluated_n_boxes_accepted=last_convergence_evaluated_n_boxes_accepted,
                    convergence_look_history=convergence_look_history,
                    converged_md=converged_now,
                    converged=converged,
                    engine_build_identity=worker_engine_identity,
                    status_override="running",
                )
            )
        if not do_converge:
            break
        if converged:
            break
        if max_boxes is not None and total_boxes >= int(max_boxes):
            break
        if max_boxes is None and total_boxes >= _HARD_MAX_EXTERNAL_BOXES:
            raise RuntimeError(
                "External production failed to converge after "
                f"{_HARD_MAX_EXTERNAL_BOXES} attempted boxes. Relax convergence "
                "tolerances or set production.max_boxes to impose a lower cap."
            )
        if max_boxes is None and total_boxes >= target:
            target = min(_HARD_MAX_EXTERNAL_BOXES, total_boxes + batch)
        elif max_boxes is not None and total_boxes >= target:
            target = min(int(max_boxes), total_boxes + batch)
        if max_boxes is not None and target > int(max_boxes):
            target = int(max_boxes)
        if total_boxes >= target and task_attempts == []:
            break
        if target > planned_initial:
            materialize_external_production(
                config=config,
                outdir=outdir,
                plan=plan,
                job_template=job_template,
                n_boxes=target,
                progress=progress,
            )
            planned_initial = max(planned_initial, target)

    return _summarise_analysis_as_production_state(
        config=config,
        outdir=outdir,
        plan=plan,
        prod_cfg=prod_cfg,
        analysis=analysis,
        mode="full-run",
        max_parallel_boxes=max(1, int(max_parallel_boxes)),
        job_template=job_template,
        planned_boxes=planned_initial,
        convergence_streak=converged_streak,
        required_convergence_streak=required_streak,
        last_convergence_evaluated_n_boxes_total=last_convergence_evaluated_n_boxes_total,
        last_convergence_evaluated_n_boxes_accepted=last_convergence_evaluated_n_boxes_accepted,
        convergence_look_history=convergence_look_history,
        converged_md=converged_now,
        converged=converged,
        engine_build_identity=worker_engine_identity,
    )
