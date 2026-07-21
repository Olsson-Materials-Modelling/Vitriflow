from __future__ import annotations

import warnings
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import TypeAdapter

from ..config import (
    RunConfig,
    MDConfig,
    ThermostatConfig,
    BarostatConfig,
    PotentialConfig,
    StructureMetricsConfig,
    ProductionEnsembleConfig,
    ConvergenceConfig,
)
from ..kim import ensure_model_installed, ensure_potential_model_installed
from ..engine_identity import (
    assert_engine_build_identity_bundle_unchanged,
    deferred_engine_build_identities,
    query_engine_build_identities,
    validate_engine_build_identity_bundle,
)
from ..potential import (
    stage_validated_tabulated_core_for_replay,
    validated_tabulated_core_path,
)
from ..runner import Cp2kRunner, LammpsRunner
from ..structuregen import prepare_initial_structure
from ..utils import ensure_dir, stable_file_identity
from .metrics_policy import resolve_effective_metrics_config
from .preflight import run_preflight
from .progress import CondensedProgressLog, atomic_write_json
from .workflow_lock import locked_output_workflow, workflow_payload_entries
import importlib
from .quench_rates import quench_steps_for_rate, resolve_quench_rates_K_per_time
from .step_counts import extend_highT_steps_for_force_isotropic, resolve_lammps_units_style, resolve_md_pressure
from .elastic_screen import build_elastic_sampling_hint
from .resume_integrity import (
    attach_production_state_integrity as _attach_production_state_integrity,
    is_zero_committed_active_resume_state,
    potential_command_file_paths,
    prepare_release_resume_migration,
    resolve_result_path as _resolve_result_path,
    seal_release_resume_migration,
    validate_release_resume_migration,
    validate_production_resume_state as _validate_production_resume_state,
)
from ..runtime_identity import runtime_identity


_RUN_RESUME_FINGERPRINT_SCHEMA = "vitriflow.run.resume_fingerprint.v5"
_RUN_SEED_SCHEME = "stateful_stage_seed_stream_v1"
_RUN_RESUME_POLICY = "explicit_state_required_clean_restart_v1"


def _resolve_run_resume_mode(
    *,
    outdir: Path,
    results_path: Path,
    resume: bool | None,
) -> bool:
    """Resolve run restart semantics without ever mixing trusted and stale state.

    ``None`` retains the documented auto-resume behavior.  An explicit
    ``--resume`` requires the protected result bundle to exist; an explicit
    ``--no-resume`` requires a clean output directory.  Starting without a
    result bundle in a non-empty directory is rejected because stage files
    alone do not carry enough provenance to decide whether they belong to the
    requested calculation.
    """

    result_path = Path(results_path)
    if result_path.is_symlink():
        raise RuntimeError(
            "Cannot trust run_results.json for resume because it is a symbolic link"
        )
    exists = result_path.is_file()
    if resume is True:
        if not exists:
            raise RuntimeError(
                "Cannot resume: --resume was requested but run_results.json is missing; "
                "use a fresh empty output directory with --no-resume to start a new run"
            )
        return True
    if resume is False and exists:
        raise RuntimeError(
            "Cannot start with --no-resume: run_results.json already exists in the output "
            "directory; choose a fresh empty output directory"
        )
    if exists:
        return True
    leftovers = sorted(path.name for path in workflow_payload_entries(outdir))
    if leftovers:
        preview = ", ".join(leftovers[:8])
        suffix = " ..." if len(leftovers) > 8 else ""
        raise RuntimeError(
            "Cannot safely start run without a protected run_results.json in a non-empty "
            f"output directory ({preview}{suffix}); choose a fresh empty output directory"
        )
    return False


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return str(stable_file_identity(Path(path))["sha256"])


def _resume_file_identity(path: Path, *, configured_path: str) -> dict[str, Any]:
    p = Path(path)
    identity = stable_file_identity(p)
    return {
        "configured_path": str(configured_path),
        "filename": p.name,
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def _resolve_resume_dependency(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve(strict=False)


def _build_run_resume_fingerprint(
    *,
    config: RunConfig,
    production_plan: Mapping[str, Any],
    outdir: Path,
    external_mode: str,
    job_template: Optional[Path] = None,
    engine_build_identities: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Bind resumable run state to its scientific and execution inputs."""

    outdir = Path(outdir).expanduser().resolve(strict=False)

    plan = json.loads(json.dumps(dict(production_plan), allow_nan=False))
    structure_value = plan.get("structure_data", None)
    if structure_value is None or not str(structure_value).strip():
        raise ValueError("Production plan has no structure_data for resume fingerprinting")
    structure_path = _resolve_resume_dependency(structure_value, base_dir=outdir)

    potential = plan.get("potential_config", None)
    potential_source: Mapping[str, Any]
    if isinstance(potential, Mapping) and len(potential) > 0:
        potential_source = potential
    else:
        potential_source = _model_dump_jsonlike(getattr(config, "kim", None))
    potential_files: list[Any] = []
    if isinstance(potential_source, Mapping):
        potential_files.extend(list(potential_source.get("files", []) or []))

    dependency_identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in potential_files:
        path = _resolve_resume_dependency(value, base_dir=outdir)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dependency_identities.append(
            _resume_file_identity(path, configured_path=str(value))
        )

    command_file_identities: list[dict[str, Any]] = []
    for candidate in potential_command_file_paths(
        potential=potential_source,
        plan=plan,
        declared_values=potential_files,
        base_dir=outdir,
    ):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        command_file_identities.append(
            _resume_file_identity(candidate, configured_path=str(candidate))
        )

    generated_potential_identities: list[dict[str, Any]] = []
    generated_table = validated_tabulated_core_path(
        plan.get("potential_lines"), root=outdir
    )
    if generated_table is not None:
        generated_potential_identities.append(
            _resume_file_identity(
                generated_table,
                configured_path=str(generated_table.relative_to(outdir)),
            )
        )

    engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()
    if engine == "cp2k":
        execution_config = _model_dump_jsonlike(getattr(config, "cp2k", None))
    else:
        execution_config = _model_dump_jsonlike(getattr(config, "lammps", None))

    cp2k_data_identities: list[dict[str, Any]] = []
    plan_prod_cfg = plan.get("production_cfg", {})
    plan_dft_cfg = plan_prod_cfg.get("dft_opt", {}) if isinstance(plan_prod_cfg, Mapping) else {}
    uses_cp2k = engine == "cp2k" or (
        isinstance(plan_dft_cfg, Mapping) and bool(plan_dft_cfg.get("enabled", False))
    )
    cp2k_execution_config: Optional[dict[str, Any]] = None
    if uses_cp2k:
        cp2k_cfg = getattr(config, "cp2k", None)
        if cp2k_cfg is None:
            raise RuntimeError("CP2K execution/refinement requires a cp2k configuration")
        cp2k_execution_config = _model_dump_jsonlike(cp2k_cfg)
        resolved = Cp2kRunner(cp2k_cfg).resolved_data_files(outdir, require=True)
        for role in sorted(resolved):
            item = resolved[role]
            path = Path(item["resolved_path"])
            cp2k_data_identities.append(
                {
                    "role": str(role),
                    "configured_name": str(item["configured_name"]),
                    **_resume_file_identity(path, configured_path=str(path)),
                }
            )

    runtime = runtime_identity()

    job_template_identity: Optional[dict[str, Any]] = None
    if job_template is not None:
        if str(external_mode).strip().lower() == "local":
            raise ValueError("job_template is meaningful only for external dry-run/full-run execution")
        template_path = Path(job_template).expanduser().resolve(strict=False)
        job_template_identity = _resume_file_identity(
            template_path,
            configured_path=str(job_template),
        )

    payload = {
        "schema": _RUN_RESUME_FINGERPRINT_SCHEMA,
        "workflow": "run_meltquench",
        "vitriflow_version": str(runtime["vitriflow_version"]),
        "runtime": runtime,
        "external_mode": str(external_mode),
        "seed_scheme": _RUN_SEED_SCHEME,
        "resume_policy": _RUN_RESUME_POLICY,
        "production_plan": plan,
        "execution_config": execution_config,
        "cp2k_execution_config": cp2k_execution_config,
        # Public workflow entry points always provide a verified local bundle
        # or an explicit external-worker deferral marker.  The fallback keeps
        # this internal builder backwards-compatible for callers that only
        # compare non-execution fingerprint fields.
        "engine_build_identities": (
            dict(engine_build_identities)
            if engine_build_identities is not None
            else {
                "status": "not_supplied_to_internal_builder",
                "primary_engine": engine,
            }
        ),
        "job_template": job_template_identity,
        "fallback_potential_config": (
            _model_dump_jsonlike(getattr(config, "kim", None))
            if not isinstance(potential, Mapping) or len(potential) == 0
            else None
        ),
        "input_identities": {
            "structure_data": _resume_file_identity(
                structure_path,
                configured_path=str(structure_value),
            ),
            "potential_files": dependency_identities,
            "potential_command_files": command_file_identities,
            "generated_potential_files": generated_potential_identities,
            "cp2k_data_files": cp2k_data_identities,
        },
    }
    return {
        "schema": _RUN_RESUME_FINGERPRINT_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "sha256": _canonical_json_sha256(payload),
        "payload": payload,
    }


def _migrate_0_4_35_1_run_resume_fingerprint(
    previous: Mapping[str, Any],
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    outdir: Path,
) -> dict[str, Any] | None:
    """Authenticate the exact first-box 0.4.35.1 plotting-hotfix resume."""

    prepared = prepare_release_resume_migration(
        stored,
        current,
        workflow="run_meltquench",
    )
    if prepared is None or not is_zero_committed_active_resume_state(previous):
        return None
    normalized_stored, normalized_current, record = prepared
    canonicalized: list[str] = []

    old_plan = normalized_stored.get("production_plan")
    new_plan = normalized_current.get("production_plan")
    if not isinstance(old_plan, Mapping) or not isinstance(new_plan, Mapping):
        return None
    old_structure = old_plan.get("structure_data")
    new_structure = new_plan.get("structure_data")
    if old_structure is None or new_structure is None:
        return None
    if _resolve_result_path(old_structure, outdir=outdir) != _resolve_result_path(
        new_structure,
        outdir=outdir,
    ):
        return None
    normalized_old_plan = dict(old_plan)
    normalized_old_plan["structure_data"] = str(new_structure)
    normalized_stored["production_plan"] = normalized_old_plan
    if str(old_structure) != str(new_structure):
        canonicalized.append("production_plan.structure_data")

    old_inputs = normalized_stored.get("input_identities")
    new_inputs = normalized_current.get("input_identities")
    if not isinstance(old_inputs, Mapping) or not isinstance(new_inputs, Mapping):
        return None
    old_identity = old_inputs.get("structure_data")
    new_identity = new_inputs.get("structure_data")
    if not isinstance(old_identity, Mapping) or not isinstance(new_identity, Mapping):
        return None
    old_without_path = dict(old_identity)
    new_without_path = dict(new_identity)
    old_identity_path = old_without_path.pop("configured_path", None)
    new_identity_path = new_without_path.pop("configured_path", None)
    if (
        old_identity_path is None
        or new_identity_path is None
        or old_without_path != new_without_path
        or _resolve_result_path(old_identity_path, outdir=outdir)
        != _resolve_result_path(new_identity_path, outdir=outdir)
    ):
        return None
    normalized_old_inputs = dict(old_inputs)
    normalized_old_inputs["structure_data"] = {
        **dict(old_identity),
        "configured_path": str(new_identity_path),
    }
    normalized_stored["input_identities"] = normalized_old_inputs
    if str(old_identity_path) != str(new_identity_path):
        canonicalized.append("input_identities.structure_data.configured_path")

    if _canonical_json_sha256(normalized_stored) != _canonical_json_sha256(
        normalized_current
    ):
        return None
    record["canonicalized_path_fields"] = canonicalized
    return seal_release_resume_migration(record)


def _validate_run_resume_fingerprint(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    allow_dry_run_to_full_run: bool = False,
    outdir: Path | None = None,
) -> dict[str, Any] | None:
    stored = previous.get("resume_fingerprint", None)
    if not isinstance(stored, Mapping):
        raise RuntimeError(
            "Cannot safely resume: run_results.json has no run resume fingerprint. "
            "Use a fresh output directory for pre-fingerprint results."
        )
    if stored.get("schema") != _RUN_RESUME_FINGERPRINT_SCHEMA:
        raise RuntimeError("Cannot safely resume: unsupported run resume fingerprint schema")
    stored_payload = stored.get("payload", None)
    if not isinstance(stored_payload, Mapping):
        raise RuntimeError("Cannot safely resume: malformed run resume fingerprint payload")
    stored_sha = str(stored.get("sha256", "")).strip().lower()
    if stored_sha != _canonical_json_sha256(dict(stored_payload)):
        raise RuntimeError("Cannot safely resume: stored run resume fingerprint is internally inconsistent")
    current_payload = current.get("payload", None)
    if not isinstance(current_payload, Mapping):
        raise RuntimeError("Cannot safely resume: current run resume fingerprint is malformed")
    current_sha = str(current.get("sha256", "")).strip().lower()
    if current_sha != _canonical_json_sha256(dict(current_payload)):
        raise RuntimeError("Cannot safely resume: current run resume fingerprint is internally inconsistent")
    if stored_sha != current_sha:
        if outdir is not None:
            migrated = _migrate_0_4_35_1_run_resume_fingerprint(
                previous,
                stored,
                current,
                outdir=Path(outdir),
            )
            if migrated is not None:
                return migrated
        previous_prod = previous.get("production", {})
        stored_without_mode = dict(stored_payload)
        current_without_mode = dict(current_payload)
        stored_mode = str(stored_without_mode.pop("external_mode", "")).strip().lower()
        current_mode = str(current_without_mode.pop("external_mode", "")).strip().lower()
        valid_materialisation_transition = (
            allow_dry_run_to_full_run
            and stored_mode == "dry-run"
            and current_mode == "full-run"
            and _canonical_json_sha256(stored_without_mode)
            == _canonical_json_sha256(current_without_mode)
            and isinstance(previous_prod, Mapping)
            and str(previous.get("status", "")).strip().lower() == "planned"
            and str(previous.get("execution_status", "")).strip().lower() == "planned"
            and str(previous_prod.get("status", "")).strip().lower() == "planned"
            and str(previous_prod.get("execution_status", "")).strip().lower() == "planned"
            and int(previous_prod.get("n_boxes_total", -1)) == 0
            and int(previous_prod.get("n_boxes_accepted", -1)) == 0
            and int(previous_prod.get("n_boxes_rejected", -1)) == 0
            and bool(previous_prod.get("resumable", False))
            and isinstance(previous_prod.get("execution"), Mapping)
            and str(previous_prod.get("execution", {}).get("mode", "")).strip().lower()
            == "dry-run"
        )
        if valid_materialisation_transition:
            return None
        raise RuntimeError(
            "Cannot safely resume: production plan, execution configuration, package version, "
            "or input file contents differ from the existing run"
        )
    return None


def _get_type_to_species(config: RunConfig) -> Optional[list[str]]:
    m = config.autotune.metrics
    if m.type_to_species is not None:
        return list(m.type_to_species)
    if config.kim is not None and config.kim.interactions != "fixed_types":
        return list(config.kim.interactions)
    if getattr(config, "engine", "lammps") == "cp2k":
        raise ValueError("engine='cp2k' requires autotune.metrics.type_to_species")
    return None


def _model_dump_jsonlike(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump(mode="json"))
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _production_common_module():
    return importlib.import_module("vitriflow.workflows.production_common")


def _run_production_executor(**kwargs):
    # Routes through `production_common.run_production_ensemble`, which is a
    # TRANSITIONAL SHIM that lazy-imports `autotune._run_production_ensemble`.
    # This is NOT an architecture fix for the project design guide's run/autotune/run-schedule
    # separation rule -- the cross-runner dependency still exists, just
    # renamed. The finding is intentionally left open; see the docstring on
    # `production_common.run_production_ensemble` for why we kept the shim
    # rather than physically migrating the runner this release.
    return _production_common_module().run_production_ensemble(**kwargs)


def _strip_distributions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ent in entries:
        d = dict(ent)
        d.pop("distributions", None)
        out.append(d)
    return out


def _resolve_path_from_source(value: Any, *, base_dir: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = (Path(base_dir) / p).expanduser()
    return p.resolve(strict=False)


_POTENTIAL_ADAPTER = TypeAdapter(PotentialConfig)


def _potential_from_dict(data: Optional[Mapping[str, Any]], fallback: Any) -> Any:
    if not isinstance(data, Mapping) or len(data) == 0:
        return fallback
    return _POTENTIAL_ADAPTER.validate_python(dict(data))


def _fixed_production_cfg(base_cfg: ProductionEnsembleConfig, n_replicates: int) -> ProductionEnsembleConfig:
    n = max(1, int(n_replicates))
    data = _model_dump_jsonlike(base_cfg)
    data.update(
        {
            "enabled": True,
            "min_boxes": n,
            "max_boxes": n,
            "batch_boxes": n,
            "consecutive_converged_checks": min(max(1, int(data.get("consecutive_converged_checks", 1) or 1)), n),
        }
    )
    return ProductionEnsembleConfig.model_validate(data)


def _production_cfg_from_summary(
    base_cfg: ProductionEnsembleConfig,
    summary: Mapping[str, Any],
    *,
    fallback_n_replicates: Optional[int] = None,
) -> ProductionEnsembleConfig:
    if fallback_n_replicates is not None:
        return _fixed_production_cfg(base_cfg, fallback_n_replicates)

    data = _model_dump_jsonlike(base_cfg)
    data["enabled"] = bool(summary.get("enabled", True))
    for key in (
        "min_boxes",
        "max_boxes",
        "batch_boxes",
        "dump_trajectory",
        "dump_every_steps",
        "check_convergence",
        "consecutive_converged_checks",
        "bondlen_cdf_points",
        "angle_cdf_points",
        "store_distributions",
        "warmup_start_temperature",
        "warmup_duration_ps",
        "exclude_coordination_defects",
        "rejects_subdir",
    ):
        if key in summary and summary.get(key) is not None:
            data[key] = summary.get(key)
    dft_summary = summary.get("dft_opt", None)
    if isinstance(dft_summary, Mapping):
        dft_cfg = dict(_model_dump_jsonlike(base_cfg.dft_opt))
        dft_cfg["enabled"] = bool(dft_summary.get("enabled", False))
        for key in ("optimizer", "max_iter", "keep_angles", "external_pressure_bar", "traj_every", "print_level"):
            if key in dft_summary and dft_summary.get(key) is not None:
                dft_cfg[key] = dft_summary.get(key)
        data["dft_opt"] = dft_cfg
    return ProductionEnsembleConfig.model_validate(data)


def _legacy_md_use_and_potential_lines(config: RunConfig, source: Mapping[str, Any]) -> tuple[Optional[MDConfig], Optional[list[str]]]:
    rec = source.get("recommendation", source) if isinstance(source, Mapping) else {}
    if not isinstance(rec, Mapping):
        rec = {}
    pf = source.get("preflight", {}) if isinstance(source.get("preflight", {}), Mapping) else {}
    md_rec = rec.get("md", {}) if isinstance(rec.get("md", {}), Mapping) else {}

    timestep = md_rec.get("timestep", pf.get("selected_timestep", None))
    ensemble = md_rec.get("ensemble", pf.get("selected_ensemble", None))
    tdamp = (md_rec.get("thermostat", {}) or {}).get("tdamp", pf.get("selected_tdamp", None))
    pdamp = (md_rec.get("barostat", {}) or {}).get("pdamp", pf.get("selected_pdamp", None))
    force_isotropic = md_rec.get("force_isotropic", None)

    if timestep is None and ensemble is None and tdamp is None and pdamp is None and force_isotropic is None:
        return None, None

    updates: dict[str, Any] = {}
    if timestep is not None:
        updates["timestep"] = float(timestep)
    if ensemble is not None:
        updates["ensemble"] = str(ensemble)
    if tdamp is not None:
        updates["thermostat"] = config.md.thermostat.model_copy(update={"tdamp": float(tdamp)})
    if pdamp is not None and config.md.barostat is not None:
        updates["barostat"] = config.md.barostat.model_copy(update={"pdamp": float(pdamp)})
    if force_isotropic is not None:
        updates["force_isotropic"] = bool(force_isotropic)
    md_use = config.md.model_copy(update=updates) if updates else config.md

    potential_lines = pf.get("potential_lines", None)
    if potential_lines is not None:
        potential_lines = [str(x) for x in potential_lines]
    return md_use, potential_lines


def _legacy_rate_and_structure(config: RunConfig, source: Mapping[str, Any], base_dir: Optional[Path]) -> tuple[Optional[Path], dict[str, Any], dict[str, Any], Optional[Path]]:
    rec = source.get("recommendation", source) if isinstance(source, Mapping) else {}
    if not isinstance(rec, Mapping):
        rec = {}
    size_scan = source.get("size_scan", {}) if isinstance(source.get("size_scan", {}), Mapping) else {}
    production = source.get("production", {}) if isinstance(source.get("production", {}), Mapping) else {}

    structure_override = rec.get("structure_data", None)
    if structure_override is None and isinstance(size_scan.get("base_data", None), str):
        structure_override = size_scan.get("base_data")
    structure_path = _resolve_path_from_source(structure_override, base_dir=base_dir)
    return structure_path, dict(rec), dict(production), _resolve_path_from_source(size_scan.get("base_data", None), base_dir=base_dir)


@locked_output_workflow("run melt-quench workflow")
def run_meltquench(
    config: RunConfig,
    outdir: Path,
    *,
    production_source: Optional[dict[str, Any]] = None,
    recommendation: Optional[dict[str, Any]] = None,
    recommendation_base_dir: Optional[Path] = None,
    n_replicates: int = 1,
    external_mode: str = "local",
    job_template: Optional[Path] = None,
    max_parallel_boxes: int = 1,
    resume: bool | None = None,
) -> dict[str, Any]:
    ensure_dir(outdir)
    results_path = outdir / "run_results.json"
    do_resume = _resolve_run_resume_mode(
        outdir=outdir,
        results_path=results_path,
        resume=resume,
    )
    progress = CondensedProgressLog(outdir / "condensed.log")
    progress.info("run", "initialising melt-quench workflow")

    metric_warnings: list[str] = []
    run_warnings: list[str] = []

    def _warn_metric(msg: str) -> None:
        metric_warnings.append(str(msg))
        warnings.warn(str(msg), stacklevel=2)
        progress.warn("metrics", str(msg))

    def _warn_run(msg: str) -> None:
        run_warnings.append(str(msg))
        warnings.warn(str(msg), stacklevel=2)
        progress.warn("run", str(msg))

    external_mode_norm = str(external_mode).strip().lower()
    if external_mode_norm not in {"local", "dry-run", "full-run"}:
        raise ValueError(f"Unsupported external_mode={external_mode!r}; expected one of local, dry-run, full-run")

    source = production_source if production_source is not None else recommendation
    requested_source = source
    requested_source_base_dir = recommendation_base_dir
    production_common = _production_common_module()
    resume_state: Optional[dict[str, Any]] = None
    previous_results: Optional[dict[str, Any]] = None
    previous_complete = False
    release_resume_migrations: list[dict[str, Any]] = []
    if do_resume:
        prev = json.loads(results_path.read_text())
        if not isinstance(prev, dict):
            raise RuntimeError("Cannot resume: run_results.json does not contain a JSON object")
        previous_results = prev
        previous_migrations = prev.get("release_resume_migrations", [])
        if not isinstance(previous_migrations, list) or not all(
            isinstance(item, Mapping) for item in previous_migrations
        ):
            raise RuntimeError(
                "Cannot safely resume: release-resume migration history is malformed"
            )
        release_resume_migrations = [dict(item) for item in previous_migrations]
        for migration_record in release_resume_migrations:
            validate_release_resume_migration(migration_record)
        prev_prod = prev.get("production", None)
        if not isinstance(prev_prod, Mapping):
            raise RuntimeError("Cannot resume: existing run_results.json has no 'production' state")
        prev_status = str(prev_prod.get("status", prev.get("status", "")) or "").strip().lower()
        previous_complete = prev_status == "ok"
        resume_state = dict(prev_prod)
        if isinstance(prev.get("metric_warnings", None), list):
            metric_warnings.extend(str(x) for x in prev.get("metric_warnings", []) or [])
        if isinstance(prev.get("run_warnings", None), list):
            run_warnings.extend(str(x) for x in prev.get("run_warnings", []) or [])
        prev_plan = production_common.production_plan_from_source(prev, base_dir=outdir)
        if prev_plan is not None:
            if requested_source is not None:
                requested_plan = production_common.production_plan_from_source(
                    requested_source,
                    base_dir=requested_source_base_dir,
                )
                if requested_plan is None:
                    raise RuntimeError(
                        "Cannot safely resume: the supplied --use-autotune source does not "
                        "contain a complete production plan"
                    )
                protected_plan_payload = production_common.production_plan_to_dict(
                    prev_plan,
                    relative_to=None,
                )
                requested_plan_payload = production_common.production_plan_to_dict(
                    requested_plan,
                    relative_to=None,
                )
                if _canonical_json_sha256(protected_plan_payload) != _canonical_json_sha256(
                    requested_plan_payload
                ):
                    raise RuntimeError(
                        "Cannot safely resume: the supplied --use-autotune production plan "
                        "differs from the plan protected by run_results.json"
                    )
            source = prev
            recommendation_base_dir = outdir
            progress.info(
                "run",
                "resuming production from existing run_results.json "
                f"({int(prev_prod.get('n_boxes_total', 0) or 0)} boxes already attempted)",
            )
        else:
            raise RuntimeError(
                "Cannot safely resume: existing run_results.json has no stored production_plan. "
                "Use a fresh output directory rather than mixing an unverified plan with prior boxes."
            )

    plan = (
        None
        if source is None
        else production_common.production_plan_from_source(source, base_dir=recommendation_base_dir)
    )
    source_kind = "none"

    if plan is not None:
        source_kind = plan.source_kind
        type_to_species = list(plan.type_to_species) if plan.type_to_species is not None else _get_type_to_species(config)
        metrics_cfg = StructureMetricsConfig.model_validate(plan.metrics_cfg or _model_dump_jsonlike(config.autotune.metrics))
        metrics_summary = dict(plan.effective_metrics or {})
        progress.info("metrics", f"effective metrics summary: {metrics_summary}")
        prod_cfg_override = ProductionEnsembleConfig.model_validate(plan.production_cfg)
        conv_cfg_override = ConvergenceConfig.model_validate(plan.convergence_cfg)
        md_use = MDConfig.model_validate(plan.md_use)
        q_cfg = config.autotune.quench.model_copy(update={"t_final": float(plan.t_final), "relax_steps": int(plan.relax_steps)})
        tm_cfg = config.autotune.tm_scan.model_copy(update={"msd_every": int(plan.msd_every)})
        pot_cfg = _potential_from_dict(plan.potential_config, config.kim)
        potential_lines = (None if plan.potential_lines is None else [str(x) for x in plan.potential_lines])
        cutoffs_rate = production_common.cutoffs_dict_from_any(plan.cutoffs_rate)
        cutoffs_size = production_common.cutoffs_dict_from_any(plan.preferred_cutoffs or plan.cutoffs_size or plan.cutoffs_rate)
        if int(n_replicates) != 1:
            _warn_run("run --n-replicates is ignored when an explicit production plan is supplied")
        engine = str(plan.engine).strip().lower() or str(getattr(config, "engine", "lammps"))
        if engine != str(getattr(config, "engine", "lammps")).strip().lower():
            _warn_run(f"production plan engine='{engine}' overrides config.engine='{config.engine}'")
    else:
        engine = str(getattr(config, "engine", "lammps")).strip().lower()
        type_to_species = _get_type_to_species(config)

        structure_data: Optional[Path] = None
        legacy_rec: dict[str, Any] = {}
        legacy_prod: dict[str, Any] = {}
        if isinstance(source, Mapping):
            structure_data, legacy_rec, legacy_prod, _legacy_size_base = _legacy_rate_and_structure(config, source, recommendation_base_dir)
        if structure_data is None:
            structure_data = Path(prepare_initial_structure(config, outdir))

        metrics_cfg, _auto_defaults, metrics_summary = resolve_effective_metrics_config(
            config.autotune.metrics,
            structure_data=Path(structure_data),
            type_to_species=type_to_species,
            lammps_units_style=(
                str(getattr(config.kim, "user_units", "") or "")
                if engine == "lammps"
                else None
            ),
            warn_fn=_warn_metric,
            context="run production",
        )
        progress.info("metrics", f"effective metrics summary: {metrics_summary}")

        legacy_md_use, legacy_potential_lines = (
            _legacy_md_use_and_potential_lines(config, source) if isinstance(source, Mapping) else (None, None)
        )
        if legacy_md_use is not None:
            md_use = legacy_md_use
            potential_lines = legacy_potential_lines
            _warn_run("legacy autotune results do not carry a full production plan; replay uses stored recommendation/preflight state plus current metrics/convergence settings")
        else:
            runner_for_preflight = Cp2kRunner(config.cp2k) if engine == "cp2k" else LammpsRunner(config.lammps)
            preflight = run_preflight(runner_for_preflight, config, str(structure_data), outdir)
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

        rates_time, time_unit_ps, _rates_ps = resolve_quench_rates_K_per_time(config)
        default_rate_time = float(rates_time[-1])
        T_high = float(legacy_rec.get("T_high", config.autotune.tm_scan.t_max))
        high_steps = int(legacy_rec.get("highT_steps", config.autotune.highT.min_total_steps))
        t_final = float(legacy_prod.get("t_final", legacy_rec.get("t_final", config.autotune.quench.t_final)))
        if "cooling_rate_K_per_time" in legacy_rec:
            chosen_rate = float(legacy_rec.get("cooling_rate_K_per_time"))
        elif "rate_K_per_time" in legacy_prod:
            chosen_rate = float(legacy_prod.get("rate_K_per_time"))
        elif "cooling_rate_K_per_ps" in legacy_rec and time_unit_ps is not None:
            chosen_rate = float(legacy_rec.get("cooling_rate_K_per_ps")) * float(time_unit_ps)
        else:
            chosen_rate = float(default_rate_time)
        cooling_rate_ps = None if time_unit_ps is None else float(chosen_rate) / float(time_unit_ps)
        replicate = legacy_prod.get("replicate", legacy_rec.get("replicate", [1, 1, 1])) or [1, 1, 1]
        high_steps = extend_highT_steps_for_force_isotropic(
            int(high_steps),
            force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
        )
        quench_sampling_hint = build_elastic_sampling_hint(
            Tm=legacy_rec.get("Tm_operational", legacy_rec.get("Tm", None)),
            freeze_temperature=legacy_rec.get("T_diffusion_freeze", legacy_rec.get("diffusion_freeze_temperature", None)),
            threshold_A2_per_ps=float(getattr(metrics_cfg.elastic, "diffusion_freeze_threshold_A2_per_ps", 0.1)),
        )
        quench_steps = int(legacy_prod.get("quench_steps", quench_steps_for_rate(float(T_high - t_final), float(chosen_rate), float(md_use.timestep), min_steps=1)))
        relax_steps = int(legacy_prod.get("relax_steps", config.autotune.quench.relax_steps))
        pot_cfg = config.kim
        cutoffs_rate = production_common.cutoffs_dict_from_any((source or {}).get("rate_scan", {}).get("cutoffs", None) if isinstance(source, Mapping) else None)
        cutoffs_size = production_common.cutoffs_dict_from_any((source or {}).get("size_scan", {}).get("cutoffs", None) if isinstance(source, Mapping) else None)
        preferred_cutoffs = production_common.cutoffs_dict_from_any(legacy_prod.get("cutoffs", None))
        if len(preferred_cutoffs) > 0:
            cutoffs_size = dict(preferred_cutoffs)
        if isinstance(source, Mapping) and ("production" in source):
            prod_cfg_override = _production_cfg_from_summary(config.autotune.production, legacy_prod)
            source_kind = "autotune_legacy"
        else:
            prod_cfg_override = _fixed_production_cfg(config.autotune.production, n_replicates)
            source_kind = "live_config"
        conv_cfg_override = ConvergenceConfig.model_validate(_model_dump_jsonlike(config.autotune.convergence))
        plan = production_common.make_production_plan(
            engine=engine,
            structure_data=Path(structure_data),
            T_high=float(T_high),
            high_total_steps=int(high_steps),
            t_final=float(t_final),
            chosen_rate=float(chosen_rate),
            cooling_rate_ps=cooling_rate_ps,
            replicate=replicate,
            pressure=float(md_use.pressure),
            md_use=md_use.model_dump(mode="json"),
            potential_config=(_model_dump_jsonlike(pot_cfg) if pot_cfg is not None else None),
            potential_lines=potential_lines,
            core_repulsion=_model_dump_jsonlike(getattr(pot_cfg, "core_repulsion", None)),
            type_to_species=type_to_species,
            metrics_cfg=metrics_cfg.model_dump(mode="json"),
            effective_metrics=dict(metrics_summary),
            production_cfg=prod_cfg_override.model_dump(mode="json"),
            convergence_cfg=conv_cfg_override.model_dump(mode="json"),
            cutoffs_rate=cutoffs_rate,
            cutoffs_size=cutoffs_size,
            preferred_cutoffs=(cutoffs_size if len(cutoffs_size) > 0 else cutoffs_rate),
            quench_steps=int(quench_steps),
            relax_steps=int(relax_steps),
            msd_every=int(config.autotune.tm_scan.msd_every),
            seed_base=int(config.random_seed) + 13579,
            time_unit_ps=time_unit_ps,
            sampling_hint=quench_sampling_hint,
            execution_mode=("adaptive" if isinstance(source, Mapping) and ("production" in source) else "fixed"),
            source_kind=source_kind,
        )

    if plan is None:
        raise RuntimeError("failed to construct a production plan")

    # A protected autocore table is a realized numerical artifact, not merely
    # a set of analytic parameters.  Portable --use-autotune replay copies and
    # authenticates that exact table before fingerprinting or task creation.
    if plan.potential_lines is not None:
        table_source_root = (
            Path(recommendation_base_dir)
            if recommendation_base_dir is not None
            else Path(outdir)
        )
        stage_validated_tabulated_core_for_replay(
            plan.potential_lines,
            source_root=table_source_root,
            target_root=outdir,
        )

    plan_for_fingerprint = production_common.production_plan_to_dict(plan, relative_to=outdir)
    plan_prod_for_identity = plan_for_fingerprint.get("production_cfg", {})
    plan_dft_for_identity = (
        plan_prod_for_identity.get("dft_opt", {})
        if isinstance(plan_prod_for_identity, Mapping)
        else {}
    )
    if external_mode_norm == "local":
        engine_build_identities = query_engine_build_identities(
            config,
            workdir=outdir,
            primary_engine=str(plan.engine),
            include_cp2k_refinement=bool(
                isinstance(plan_dft_for_identity, Mapping)
                and plan_dft_for_identity.get("enabled", False)
            ),
        )
        validate_engine_build_identity_bundle(engine_build_identities)
    else:
        # A planning/login node is not required to expose the compute-node
        # engine.  Every executed task records and authenticates the concrete
        # worker identity, which the external collector checks for homogeneity.
        engine_build_identities = deferred_engine_build_identities(str(plan.engine))
    resume_fingerprint = _build_run_resume_fingerprint(
        config=config,
        production_plan=plan_for_fingerprint,
        outdir=outdir,
        external_mode=external_mode_norm,
        job_template=job_template,
        engine_build_identities=engine_build_identities,
    )
    if previous_results is not None:
        release_resume_migration = _validate_run_resume_fingerprint(
            previous_results,
            resume_fingerprint,
            allow_dry_run_to_full_run=(external_mode_norm == "full-run"),
            outdir=outdir,
        )
        assert resume_state is not None
        _validate_production_resume_state(resume_state, outdir=outdir)
        if str(previous_results.get("status", "")).strip().lower() != str(
            resume_state.get("status", "")
        ).strip().lower():
            raise RuntimeError("Cannot safely resume: top-level and production statuses disagree")
        if str(previous_results.get("execution_status", "")).strip().lower() != str(
            resume_state.get("execution_status", "")
        ).strip().lower():
            raise RuntimeError(
                "Cannot safely resume: top-level and production execution statuses disagree"
            )
        if release_resume_migration is not None:
            release_resume_migrations.append(dict(release_resume_migration))
            progress.info(
                "run",
                "authenticated exact 0.4.35.1→0.4.36.0 zero-box checkpoint migration",
            )
        terminal_non_resumable = (
            str(resume_state.get("status", "")).strip().lower()
            in {"incomplete", "not_converged"}
            and not bool(resume_state.get("resumable", True))
        )
        if previous_complete or terminal_non_resumable:
            progress.info(
                "run",
                "existing terminal run result and its resume fingerprint are valid; returning cached summary",
            )
            return dict(previous_results)

    engine = str(plan.engine).strip().lower()
    # Resolve pot_cfg from the plan first so a replayed production matches the
    # original potential regardless of engine. The previous "pot_cfg = config.kim"
    # reset discarded plan.potential_config for the CP2K path; harmless today
    # but a silent desync for any future code that reads pot_cfg under CP2K.
    pot_cfg = (
        _potential_from_dict(plan.potential_config, config.kim)
        if plan.potential_config is not None
        else config.kim
    )
    runner = None
    if engine == "cp2k":
        if external_mode_norm == "local":
            runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]
    else:
        if external_mode_norm == "local":
            ensure_potential_model_installed(
                pot_cfg,
                installer=ensure_model_installed,
            )
            runner = LammpsRunner(config.lammps)

    if bool(getattr(md_use, "force_isotropic", False)) and engine != "lammps":
        raise ValueError("md.force_isotropic is currently supported only for engine='lammps'")

    progress.info("run", f"executing shared production plan ({plan.execution_mode}; source={source_kind})")
    if isinstance(source, Mapping) and isinstance(source.get("production", None), Mapping):
        rec_conv = source.get("production", {}).get("convergence", None)
        if isinstance(rec_conv, Mapping):
            progress.convergence("recommendation", dict(rec_conv))

    q_cfg_exec = config.autotune.quench.model_copy(update={"t_final": float(plan.t_final), "relax_steps": int(plan.relax_steps)})
    tm_cfg_exec = config.autotune.tm_scan.model_copy(update={"msd_every": int(plan.msd_every)})
    store_distributions = bool(getattr(prod_cfg_override, "store_distributions", True))
    integrity_file_cache: dict[str, tuple[int, int, int, int, int, str]] = {}

    def _build_summary(prod_state: dict[str, Any]) -> dict[str, Any]:
        prod_status = str(prod_state.get("status", "incomplete"))
        terminal = prod_status in {"ok", "incomplete", "not_converged"}
        prod_state.setdefault("execution_status", "completed" if terminal else prod_status)
        prod_state = _attach_production_state_integrity(
            prod_state,
            outdir=outdir,
            identity_cache=integrity_file_cache,
            force_rehash=terminal,
        )
        all_entries = list(prod_state.get("boxes", [])) + list(prod_state.get("rejected_boxes", []))
        rep_entries = all_entries if store_distributions else _strip_distributions(all_entries)
        return {
            "status": prod_status,
            "execution_status": "completed" if terminal else prod_status,
            "parameters": {
                "engine": str(engine),
                "lammps_units": resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal"),
                "time_unit_ps": (None if plan.time_unit_ps is None else float(plan.time_unit_ps)),
                "warmup_start_temperature": float(getattr(prod_cfg_override, "warmup_start_temperature", 300.0)),
                "warmup_duration_ps": float(getattr(prod_cfg_override, "warmup_duration_ps", 5.0)),
                "warmup_steps": (int(prod_state.get("warmup_steps")) if prod_state.get("warmup_steps", None) is not None else None),
                "T_high": float(plan.T_high),
                "highT_steps": int(plan.high_total_steps),
                "t_final": float(plan.t_final),
                "cooling_rate_K_per_time": float(plan.chosen_rate),
                "cooling_rate_K_per_ps": (None if plan.cooling_rate_ps is None else float(plan.cooling_rate_ps)),
                "n_quench_steps": int(plan.quench_steps),
                "replicate": [int(plan.replicate[0]), int(plan.replicate[1]), int(plan.replicate[2])],
                "pressure": float(plan.pressure),
                "md": dict(plan.md_use),
                "seed_base": int(plan.seed_base),
                "execution_mode": str(plan.execution_mode),
            },
            "replicates": rep_entries,
            "production": prod_state,
            "graph_outputs": dict((prod_state.get("graph_outputs", {}) if isinstance(prod_state, Mapping) else {}) or {}),
            "cutoffs": list(plan.preferred_cutoffs),
            "metric_warnings": list(metric_warnings),
            "run_warnings": list(run_warnings),
            "effective_metrics": dict(plan.effective_metrics or {}),
            "production_plan": production_common.production_plan_to_dict(plan, relative_to=outdir),
            "resume_fingerprint": dict(resume_fingerprint),
            **(
                {"release_resume_migrations": list(release_resume_migrations)}
                if release_resume_migrations
                else {}
            ),
            "paths": {
                "condensed_log": "condensed.log",
                "run_results": "run_results.json",
                **dict((prod_state.get("graph_outputs", {}) if isinstance(prod_state, Mapping) else {}) or {}),
            },
        }

    def _checkpoint(prod_state: dict[str, Any]) -> None:
        atomic_write_json(outdir / "run_results.json", _build_summary(dict(prod_state)))

    if external_mode_norm == "local":
        production = _run_production_executor(
            config=config,
            outdir=outdir,
            runner=runner,
            pot_cfg=pot_cfg,
            md_use=md_use,
            potential_lines=(None if plan.potential_lines is None else [str(x) for x in plan.potential_lines]),
            type_to_species=type_to_species,
            metrics_cfg=metrics_cfg,
            tm_cfg=tm_cfg_exec,
            q_cfg=q_cfg_exec,
            size_base_data=Path(plan.structure_data),
            chosen_replicate=[int(plan.replicate[0]), int(plan.replicate[1]), int(plan.replicate[2])],
            chosen_rate=float(plan.chosen_rate),
            dt_ref=float(md_use.timestep),
            dt_mq=float(md_use.timestep),
            cooling_rate_ps=plan.cooling_rate_ps,
            cutoffs_rate=production_common.cutoffs_dict_from_any(plan.cutoffs_rate),
            cutoffs_size=production_common.cutoffs_dict_from_any(plan.preferred_cutoffs or plan.cutoffs_size or plan.cutoffs_rate),
            T_high=float(plan.T_high),
            high_total_steps=int(plan.high_total_steps),
            resume_state=resume_state,
            sampling_hint=(None if plan.sampling_hint is None else dict(plan.sampling_hint)),
            progress=progress,
            checkpoint_cb=_checkpoint,
            pressure_override=float(plan.pressure),
            seed_base=int(plan.seed_base),
            time_unit_ps_override=(None if plan.time_unit_ps is None else float(plan.time_unit_ps)),
            prod_cfg_override=prod_cfg_override,
            conv_cfg_override=conv_cfg_override,
            quench_steps_override=int(plan.quench_steps),
            relax_steps_override=int(plan.relax_steps),
        )
    else:
        from .hpc import dry_run_external_production, full_run_external_production

        plan_dict = production_common.production_plan_to_dict(plan, relative_to=outdir)
        progress.info("external", f"dispatching production via {external_mode_norm}")
        if external_mode_norm == "dry-run":
            production = dry_run_external_production(
                config=config,
                outdir=outdir,
                plan=plan_dict,
                job_template=job_template,
                progress=progress,
            )
        else:
            production = full_run_external_production(
                config=config,
                outdir=outdir,
                plan=plan_dict,
                job_template=job_template,
                max_parallel_boxes=max(1, int(max_parallel_boxes)),
                progress=progress,
                resume_state=resume_state,
                checkpoint_cb=_checkpoint,
            )

    terminal_engine_build_identities = engine_build_identities
    if external_mode_norm == "local":
        final_engine_build_identities = query_engine_build_identities(
            config,
            workdir=outdir,
            primary_engine=str(plan.engine),
            include_cp2k_refinement=bool(
                isinstance(plan_dft_for_identity, Mapping)
                and plan_dft_for_identity.get("enabled", False)
            ),
        )
        assert_engine_build_identity_bundle_unchanged(
            engine_build_identities,
            final_engine_build_identities,
            context="during local run execution",
        )
        terminal_engine_build_identities = final_engine_build_identities
    terminal_fingerprint = _build_run_resume_fingerprint(
        config=config,
        production_plan=production_common.production_plan_to_dict(
            plan, relative_to=outdir
        ),
        outdir=outdir,
        external_mode=external_mode_norm,
        job_template=job_template,
        engine_build_identities=terminal_engine_build_identities,
    )
    if str(terminal_fingerprint.get("sha256", "")) != str(
        resume_fingerprint.get("sha256", "")
    ):
        raise RuntimeError(
            "Run configuration or scientific input bytes changed during execution; "
            "refusing to write a terminal result that could combine different "
            "structures, potentials, command includes, or CP2K data"
        )
    summary = _build_summary(dict(production))
    atomic_write_json(outdir / "run_results.json", summary)
    progress.info("run", "workflow complete; wrote run_results.json")
    return summary
