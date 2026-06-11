from __future__ import annotations

"""Generic fixed custom-stage production workflow.

This module is deliberately separate from the standard ``run`` and
``autotune`` workflows.  It allows a YAML file to define an arbitrary
continuous LAMMPS NVT/NPT stage schedule while reusing VitriFlow's structure
preparation, metrics, production-convergence and output-analysis machinery.
"""

import json
import math
import re
import shlex
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from ..config import MDConfig, RunConfig

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class CustomStageConfig:
    """One user-defined MD stage in a custom schedule."""

    name: str
    temperature_start_K: float
    temperature_stop_K: float
    time_ps: Optional[float] = None
    steps: Optional[int] = None
    role: Optional[str] = None
    velocity_mode: Optional[str] = None
    equil_steps: int = 0
    write_dump: Optional[bool] = None
    dump_every_steps: Optional[int] = None
    tail_dump_frames: Optional[int] = None
    tail_dump_stride: Optional[int] = None
    force_isotropic: Optional[bool] = None
    sample_ensemble: Optional[str] = None
    msd_every: Optional[int] = None


@dataclass(frozen=True)
class CustomSchedule:
    """A continuous custom-stage schedule."""

    stages: tuple[CustomStageConfig, ...]
    enforce_temperature_continuity: bool = True
    temperature_tolerance_K: float = 1.0e-8
    analysis_roles: dict[str, str] = field(default_factory=dict)
    sampling_hint: dict[str, float] = field(default_factory=dict)
    workflow_label: str = "custom_stage_schedule"
    description: Optional[str] = None


def _load_raw_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(data)


def _first(src: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in src:
            return src[name]
    return default


def _finite_float(value: Any, *, field_name: str, positive: bool = False) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"custom_schedule.{field_name} must be finite")
    if positive and x <= 0.0:
        raise ValueError(f"custom_schedule.{field_name} must be > 0")
    return x


def _bool_from_any(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        x = float(value)
        if x == 1.0:
            return True
        if x == 0.0:
            return False
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "y", "on", "1"}:
        return True
    if s in {"false", "f", "no", "n", "off", "0"}:
        return False
    raise ValueError(f"custom_schedule.{field_name} must be boolean-like")


def _optional_positive_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"custom_schedule.{field_name} must be finite when set")
    n = int(round(x))
    if abs(x - float(n)) > 1.0e-9:
        raise ValueError(f"custom_schedule.{field_name} must be an integer when set")
    if n < 1:
        raise ValueError(f"custom_schedule.{field_name} must be >= 1 when set")
    return n


def _optional_nonnegative_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"custom_schedule.{field_name} must be finite when set")
    n = int(round(x))
    if abs(x - float(n)) > 1.0e-9:
        raise ValueError(f"custom_schedule.{field_name} must be an integer when set")
    if n < 0:
        raise ValueError(f"custom_schedule.{field_name} must be >= 0 when set")
    return n


def _stage_from_mapping(item: Mapping[str, Any], idx: int) -> CustomStageConfig:
    raw_name = _first(item, ["name", "stage", "label"], None)
    if raw_name is None:
        raise ValueError(f"custom_schedule.stages[{idx}] requires a name")
    name = str(raw_name).strip()
    if not name:
        raise ValueError(f"custom_schedule.stages[{idx}].name must be non-empty")
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"custom_schedule.stages[{idx}].name={name!r} is not path-safe; use letters, digits, '_', '-' or '.'"
        )

    hold_T = _first(item, ["temperature_K", "T_K", "temperature", "T"], None)
    t0 = _first(item, ["temperature_start_K", "start_temperature_K", "T_start_K", "T_start", "temperature_start"], hold_T)
    t1 = _first(item, ["temperature_stop_K", "stop_temperature_K", "T_stop_K", "T_stop", "temperature_stop", "temperature_final_K"], hold_T)
    if t0 is None or t1 is None:
        raise ValueError(f"custom_schedule.stages[{idx}] requires either temperature_K or both temperature_start_K and temperature_stop_K")

    time_raw = _first(item, ["time_ps", "duration_ps", "run_time_ps"], None)
    steps_raw = _first(item, ["steps", "run_steps"], None)
    steps = _optional_positive_int(steps_raw, field_name=f"stages[{idx}].steps")
    time_ps = None if time_raw is None else _finite_float(time_raw, field_name=f"stages[{idx}].time_ps", positive=True)
    if steps is None and time_ps is None:
        raise ValueError(f"custom_schedule.stages[{idx}] requires either time_ps or steps")

    role_raw = _first(item, ["role", "analysis_role"], None)
    role = None if role_raw is None else str(role_raw).strip().lower()
    if role == "":
        role = None

    vel_raw = _first(item, ["velocity_mode", "velocities"], None)
    velocity_mode = None if vel_raw is None else str(vel_raw).strip().lower()
    if velocity_mode is not None and velocity_mode not in {"create", "preserve"}:
        raise ValueError(f"custom_schedule.stages[{idx}].velocity_mode must be 'create' or 'preserve'")

    sample_raw = _first(item, ["sample_ensemble", "sampling_ensemble"], None)
    sample_ensemble = None if sample_raw is None else str(sample_raw).strip().lower()
    if sample_ensemble == "":
        sample_ensemble = None
    if sample_ensemble is not None and sample_ensemble not in {"nvt", "npt"}:
        raise ValueError(f"custom_schedule.stages[{idx}].sample_ensemble must be 'nvt' or 'npt'")

    return CustomStageConfig(
        name=name,
        temperature_start_K=_finite_float(t0, field_name=f"stages[{idx}].temperature_start_K", positive=True),
        temperature_stop_K=_finite_float(t1, field_name=f"stages[{idx}].temperature_stop_K", positive=True),
        time_ps=time_ps,
        steps=steps,
        role=role,
        velocity_mode=velocity_mode,
        equil_steps=int(_optional_nonnegative_int(_first(item, ["equil_steps", "equilibration_steps"], 0), field_name=f"stages[{idx}].equil_steps") or 0),
        write_dump=(None if "write_dump" not in item else _bool_from_any(item.get("write_dump"), field_name=f"stages[{idx}].write_dump")),
        dump_every_steps=_optional_positive_int(_first(item, ["dump_every_steps", "dump_every"], None), field_name=f"stages[{idx}].dump_every_steps"),
        tail_dump_frames=_optional_positive_int(_first(item, ["tail_dump_frames"], None), field_name=f"stages[{idx}].tail_dump_frames"),
        tail_dump_stride=_optional_positive_int(_first(item, ["tail_dump_stride"], None), field_name=f"stages[{idx}].tail_dump_stride"),
        force_isotropic=(None if "force_isotropic" not in item else _bool_from_any(item.get("force_isotropic"), field_name=f"stages[{idx}].force_isotropic")),
        sample_ensemble=sample_ensemble,
        msd_every=_optional_positive_int(_first(item, ["msd_every"], None), field_name=f"stages[{idx}].msd_every"),
    )


def _hardcarbon_schedule_to_custom(raw_hc: Mapping[str, Any]) -> CustomSchedule:
    """Backward-compatible conversion of the old HC-only schema."""

    def f(name: str, default: float) -> float:
        return _finite_float(raw_hc.get(name, default), field_name=f"hardcarbon_schedule.{name}", positive=True)

    def steps(name: str) -> Optional[int]:
        return _optional_positive_int(raw_hc.get(name, None), field_name=f"hardcarbon_schedule.{name}")

    random_T = f("random_temperature_K", 9000.0)
    graph_T = f("graph_temperature_K", 3500.0)
    final_T = f("final_temperature_K", 300.0)
    pre_start = f("prequench_start_K", random_T)
    pre_stop = f("prequench_stop_K", graph_T)
    stages = (
        CustomStageConfig("randomisation", random_T, random_T, f("random_time_ps", 10.0), steps("random_steps"), role="warmup", velocity_mode="create"),
        CustomStageConfig("prequench", pre_start, pre_stop, f("prequench_time_ps", 6.0), steps("prequench_steps"), role="prequench", velocity_mode="preserve"),
        CustomStageConfig("graphitisation", graph_T, graph_T, f("graph_time_ps", 400.0), steps("graph_steps"), role="melt", velocity_mode="preserve"),
        CustomStageConfig("quench", graph_T, final_T, f("final_quench_time_ps", 20.0), steps("final_quench_steps"), role="quench", velocity_mode="preserve"),
        CustomStageConfig("relax", final_T, final_T, f("relax_time_ps", 20.0), steps("relax_steps"), role="relax", velocity_mode="preserve"),
    )
    return CustomSchedule(
        stages=stages,
        enforce_temperature_continuity=True,
        analysis_roles={"melt": "graphitisation", "quench": "quench", "relax": "relax"},
        sampling_hint={"Tm": float(graph_T), "freeze_temperature": float(final_T)},
        workflow_label="hardcarbon_gap20ugr_legacy_schedule",
        description="Converted from legacy hardcarbon_schedule",
    )


def _schedule_from_raw(raw: Mapping[str, Any]) -> CustomSchedule:
    src = raw.get("custom_schedule", None)
    if src is None:
        hc = raw.get("hardcarbon_schedule", None)
        if isinstance(hc, Mapping):
            return _hardcarbon_schedule_to_custom(hc)
        raise ValueError("run-custom requires a custom_schedule.stages list in the YAML")
    if not isinstance(src, Mapping):
        raise ValueError("custom_schedule must be a mapping")
    stages_raw = src.get("stages", None)
    if not isinstance(stages_raw, Sequence) or isinstance(stages_raw, (str, bytes)) or len(stages_raw) < 1:
        raise ValueError("custom_schedule.stages must be a non-empty list")
    stages = []
    for i, item in enumerate(stages_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"custom_schedule.stages[{i}] must be a mapping")
        stages.append(_stage_from_mapping(item, i))

    roles_raw = src.get("analysis_roles", src.get("analysis_stages", {}))
    if roles_raw is None:
        roles_raw = {}
    if not isinstance(roles_raw, Mapping):
        raise ValueError("custom_schedule.analysis_roles/analysis_stages must be a mapping when present")
    roles = {str(k).strip().lower(): str(v).strip() for k, v in roles_raw.items() if str(k).strip() and str(v).strip()}

    hint_raw = src.get("sampling_hint", {}) or {}
    if not isinstance(hint_raw, Mapping):
        raise ValueError("custom_schedule.sampling_hint must be a mapping when present")
    hint = {str(k): _finite_float(v, field_name=f"sampling_hint.{k}", positive=False) for k, v in hint_raw.items()}
    label = str(src.get("workflow_label", "custom_stage_schedule") or "custom_stage_schedule").strip()
    if not label:
        label = "custom_stage_schedule"

    return CustomSchedule(
        stages=tuple(stages),
        enforce_temperature_continuity=_bool_from_any(src.get("enforce_temperature_continuity", src.get("require_continuity", True)), field_name="enforce_temperature_continuity"),
        temperature_tolerance_K=_finite_float(src.get("temperature_tolerance_K", 1.0e-8), field_name="temperature_tolerance_K", positive=False),
        analysis_roles=roles,
        sampling_hint=hint,
        workflow_label=label,
        description=(None if src.get("description", None) is None else str(src.get("description"))),
    )


def _resolve_analysis_roles(schedule: CustomSchedule) -> dict[str, str]:
    """Resolve melt/quench/relax roles to concrete stage names."""

    names = [s.name for s in schedule.stages]
    name_set = set(names)
    out: dict[str, str] = {}
    for role in ("melt", "quench", "relax"):
        explicit = schedule.analysis_roles.get(role)
        if explicit:
            if explicit not in name_set:
                raise ValueError(f"custom_schedule.analysis_roles.{role}={explicit!r} does not match a stage name")
            out[role] = explicit
            continue
        matches = [s.name for s in schedule.stages if (s.role or "").lower() == role]
        if len(matches) > 1:
            raise ValueError(f"Multiple custom_schedule stages have role={role!r}; set analysis_roles.{role}")
        if len(matches) == 1:
            out[role] = matches[0]
            continue
        if role in name_set:
            out[role] = role

    missing = [r for r in ("melt", "quench", "relax") if r not in out]
    if missing:
        if len(schedule.stages) >= 3 and not schedule.analysis_roles:
            # Generic fallback: use the last three stages as the analysis stages.
            last3 = schedule.stages[-3:]
            fallback = {"melt": last3[0].name, "quench": last3[1].name, "relax": last3[2].name}
            for r in missing:
                out[r] = fallback[r]
        else:
            raise ValueError(
                "custom_schedule must identify analysis roles melt/quench/relax via stage.role, stage names, "
                "or custom_schedule.analysis_roles"
            )

    idx = {s.name: i for i, s in enumerate(schedule.stages)}
    if not (idx[out["melt"]] < idx[out["quench"]] < idx[out["relax"]]):
        raise ValueError("custom_schedule analysis stages must be ordered melt before quench before relax")
    return out


def _validate_schedule(schedule: CustomSchedule) -> dict[str, str]:
    if len(schedule.stages) < 1:
        raise ValueError("custom_schedule.stages must be non-empty")
    seen: set[str] = set()
    for i, st in enumerate(schedule.stages):
        if st.name in seen:
            raise ValueError(f"Duplicate custom_schedule stage name: {st.name!r}")
        seen.add(st.name)
        if st.steps is None and st.time_ps is None:
            raise ValueError(f"custom_schedule.stages[{i}] requires time_ps or steps")
        if st.steps is not None and int(st.steps) < 1:
            raise ValueError(f"custom_schedule.stages[{i}].steps must be >= 1")
        if st.time_ps is not None and not (math.isfinite(float(st.time_ps)) and float(st.time_ps) > 0.0):
            raise ValueError(f"custom_schedule.stages[{i}].time_ps must be finite and > 0")
        if int(st.equil_steps) < 0:
            raise ValueError(f"custom_schedule.stages[{i}].equil_steps must be >= 0")
        if st.velocity_mode is not None and st.velocity_mode not in {"create", "preserve"}:
            raise ValueError(f"custom_schedule.stages[{i}].velocity_mode must be 'create' or 'preserve'")
        if i > 0 and st.velocity_mode == "create":
            raise ValueError(
                "continuous custom schedules only support velocity creation on the first stage; "
                f"stage {st.name!r} requested velocity_mode='create'"
            )
    if schedule.enforce_temperature_continuity:
        tol = float(schedule.temperature_tolerance_K)
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError("custom_schedule.temperature_tolerance_K must be finite and >= 0")
        for a, b in zip(schedule.stages[:-1], schedule.stages[1:]):
            if abs(float(a.temperature_stop_K) - float(b.temperature_start_K)) > tol:
                raise ValueError(
                    "custom_schedule is discontinuous: "
                    f"{a.name}.temperature_stop_K={a.temperature_stop_K} differs from "
                    f"{b.name}.temperature_start_K={b.temperature_start_K}"
                )
    return _resolve_analysis_roles(schedule)


def _positive_steps_from_ps(time_ps: float, *, md_timestep: float, time_unit_ps: float, field_name: str) -> int:
    t_ps = float(time_ps)
    if not (math.isfinite(t_ps) and t_ps > 0.0):
        raise ValueError(f"custom_schedule.{field_name} must be finite and > 0 when step override is absent")
    dt_ps = float(md_timestep) * float(time_unit_ps)
    if not (math.isfinite(dt_ps) and dt_ps > 0.0):
        raise ValueError(f"invalid MD timestep/time unit while computing {field_name}: dt_ps={dt_ps}")
    n = int(round(t_ps / dt_ps))
    if n < 1:
        raise ValueError(f"custom_schedule.{field_name} is shorter than one MD step")
    return int(n)


def _type_to_species(config: RunConfig) -> Optional[list[str]]:
    m = config.autotune.metrics
    if m.type_to_species is not None:
        return [str(x) for x in m.type_to_species]
    if config.kim is not None and getattr(config.kim, "interactions", None) != "fixed_types":
        return [str(x) for x in config.kim.interactions]
    return None


def _strip_distributions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ent in entries:
        d = dict(ent)
        d.pop("distributions", None)
        out.append(d)
    return out


def _clone_config_with_structure_seed(config: RunConfig, seed: int) -> RunConfig:
    """Return a config copy with structure.generate.seed changed when present."""
    if config.structure.generate is None:
        return config
    gen = config.structure.generate.model_copy(update={"seed": int(seed)})
    struct = config.structure.model_copy(update={"generate": gen})
    return config.model_copy(deep=True, update={"structure": struct})


def _relpath(path: Path, base: Path) -> str:
    try:
        return str(Path(path).relative_to(Path(base)))
    except Exception:
        return str(path)


def _schedule_steps(schedule: CustomSchedule, *, md_use: MDConfig, time_unit_ps: float) -> dict[str, int]:
    dt = float(md_use.timestep)
    out: dict[str, int] = {}
    for st in schedule.stages:
        out[st.name] = int(st.steps) if st.steps is not None else _positive_steps_from_ps(
            float(st.time_ps), md_timestep=dt, time_unit_ps=time_unit_ps, field_name=f"stages.{st.name}.time_ps"
        )
    return out


def _schedule_report(
    schedule: CustomSchedule,
    steps: Mapping[str, int],
    *,
    md_use: MDConfig,
    time_unit_ps: float,
    analysis_roles: Mapping[str, str],
) -> dict[str, Any]:
    dt_ps = float(md_use.timestep) * float(time_unit_ps)
    stage_reports: list[dict[str, Any]] = []
    role_by_name: dict[str, str] = {v: k for k, v in analysis_roles.items()}
    for idx, st in enumerate(schedule.stages):
        n = int(steps[st.name])
        t_ps = float(n * dt_ps)
        dT = float(st.temperature_stop_K) - float(st.temperature_start_K)
        rate = None if t_ps <= 0.0 else float(abs(dT) / t_ps)
        stage_reports.append(
            {
                "name": st.name,
                "role": role_by_name.get(st.name, st.role),
                "temperature_start_K": float(st.temperature_start_K),
                "temperature_stop_K": float(st.temperature_stop_K),
                "steps": n,
                "time_ps": t_ps,
                "rate_K_per_ps": rate,
                "velocity_mode": st.velocity_mode or ("create" if idx == 0 else "preserve"),
                "equil_steps": int(st.equil_steps),
            }
        )
    return {
        "kind": "custom_stage_schedule_run",
        "workflow_label": str(schedule.workflow_label),
        "description": schedule.description,
        "sampling_hint": {str(k): float(v) for k, v in dict(schedule.sampling_hint).items()},
        "time_unit_ps": float(time_unit_ps),
        "timestep": float(md_use.timestep),
        "enforce_temperature_continuity": bool(schedule.enforce_temperature_continuity),
        "analysis_roles": dict(analysis_roles),
        "stages": stage_reports,
    }




_RESUME_FINGERPRINT_SCHEMA = "vitriflow.custom_schedule.resume_fingerprint.v2"
_RESUME_FINGERPRINT_SIDECAR = "custom_schedule_resume_fingerprint.json"

# Tag identifying the algorithm used to derive per-box seeds. v2 fingerprints
# carry this so a run started under one scheme cannot be resumed by a runner
# that draws seeds differently. See _derive_box_seeds in run_custom_schedule.
_SEED_SCHEME = "sha256_box_slot_v1"


def _normalise_for_fingerprint(value: Any) -> Any:
    """Convert pydantic/dataclass/path containers into canonical JSON values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("resume fingerprint cannot encode non-finite float values")
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalise_for_fingerprint(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _normalise_for_fingerprint(model_dump(mode="json"))
        except TypeError:
            return _normalise_for_fingerprint(model_dump())
    if isinstance(value, Mapping):
        return {
            str(k): _normalise_for_fingerprint(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, set):
        vals = [_normalise_for_fingerprint(v) for v in value]
        return sorted(vals, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if isinstance(value, (list, tuple)):
        return [_normalise_for_fingerprint(v) for v in value]
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalise_for_fingerprint(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_canonical_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_identity_for_fingerprint(path: Path) -> dict[str, Any]:
    """Return content identity without binding the digest to an absolute path."""

    path = Path(path)
    ident: dict[str, Any] = {
        "filename": str(path.name),
        "exists": bool(path.exists()),
    }
    if path.exists() and path.is_file():
        try:
            ident["size_bytes"] = int(path.stat().st_size)
            ident["sha256"] = _sha256_file(path)
        except Exception as exc:
            ident["size_bytes"] = None
            ident["sha256"] = None
            ident["hash_error"] = f"{type(exc).__name__}: {exc}"
    else:
        ident["size_bytes"] = None
        ident["sha256"] = None
    return ident


def _split_lammps_command_tokens(line: str) -> list[str]:
    try:
        return shlex.split(str(line), posix=True)
    except Exception:
        return str(line).split()


def _resolve_command_file_token(token: str, *, files_by_name: Mapping[str, Path], config_path: Optional[Path]) -> Path:
    raw = str(token).strip().strip("'\"")
    name = Path(raw).name
    if name in files_by_name:
        return Path(files_by_name[name])
    p = Path(raw)
    if p.is_absolute():
        return p
    if config_path is not None:
        return Path(config_path).parent / p
    return p


def _potential_fingerprint_payload(pot_cfg: Any, *, config_path: Optional[Path]) -> dict[str, Any]:
    """Capture potential commands, configured filenames, and GAP XML content identity."""

    cfg_payload = _normalise_for_fingerprint(pot_cfg)
    file_paths = [Path(x) for x in (getattr(pot_cfg, "files", None) or [])]
    if isinstance(cfg_payload, dict) and "files" in cfg_payload:
        cfg_payload = dict(cfg_payload)
        cfg_payload["files"] = [str(Path(x).name) for x in file_paths]

    file_identities = [_file_identity_for_fingerprint(p) for p in file_paths]
    files_by_name = {Path(p).name: Path(p) for p in file_paths}
    commands = [str(x).strip() for x in (getattr(pot_cfg, "commands", None) or []) if str(x).strip()]
    xml_refs: list[dict[str, Any]] = []
    seen_xml_files: set[str] = set()
    for idx, line in enumerate(commands):
        labels = re.findall(r"\bxml_label=([^\s\"']+)", str(line))
        for tok in _split_lammps_command_tokens(line):
            clean = str(tok).strip().strip("'\"")
            if ".xml" not in clean.lower():
                continue
            p = _resolve_command_file_token(clean, files_by_name=files_by_name, config_path=config_path)
            seen_key = f"cmd:{idx}:{Path(clean).name}:{str(p)}"
            if seen_key in seen_xml_files:
                continue
            seen_xml_files.add(seen_key)
            xml_refs.append(
                {
                    "command_index": int(idx),
                    "token": clean,
                    "filename": str(Path(clean).name),
                    "xml_labels": list(labels),
                    "file": _file_identity_for_fingerprint(p),
                }
            )

    xml_files = []
    for p in file_paths:
        if Path(p).suffix.lower() == ".xml":
            xml_files.append(_file_identity_for_fingerprint(p))

    return {
        "kind": str(getattr(pot_cfg, "kind", type(pot_cfg).__name__)),
        "config": cfg_payload,
        "commands": commands,
        "files": file_identities,
        "gap_xml_identity": {
            "configured_xml_files": xml_files,
            "command_xml_references": xml_refs,
        },
    }


def _structure_fingerprint_payload(structure_cfg: Any) -> dict[str, Any]:
    payload = _normalise_for_fingerprint(structure_cfg)
    identities: dict[str, Any] = {}
    lammps_data = getattr(structure_cfg, "lammps_data", None)
    if lammps_data is not None:
        identities["lammps_data"] = _file_identity_for_fingerprint(Path(lammps_data))
    gen = getattr(structure_cfg, "generate", None)
    poscar_path = getattr(gen, "poscar_path", None) if gen is not None else None
    if poscar_path is not None:
        identities["poscar_path"] = _file_identity_for_fingerprint(Path(poscar_path))
    return {"config": payload, "file_identities": identities}


def _build_resume_fingerprint(
    *,
    config: RunConfig,
    schedule: CustomSchedule,
    analysis_roles: Mapping[str, str],
    steps: Mapping[str, int],
    sched_report: Mapping[str, Any],
    time_unit_ps: float,
    md_pressure: float,
    lammps_units: str,
    config_path: Optional[Path],
) -> dict[str, Any]:
    """Build the deterministic custom-schedule resume/provenance fingerprint."""

    try:
        from .. import __version__ as vitriflow_version
    except Exception:
        vitriflow_version = "unknown"

    payload = {
        "schema": _RESUME_FINGERPRINT_SCHEMA,
        "workflow": "custom_stage_schedule",
        "runner": {
            "name": "run_custom_schedule",
            "execution": "continuous_lammps",
            "vitriflow_version": str(vitriflow_version),
            "engine": str(config.engine),
            "lammps": _normalise_for_fingerprint(config.lammps),
        },
        "custom_schedule": _normalise_for_fingerprint(schedule),
        "custom_schedule_derived": {
            "analysis_roles": {str(k): str(v) for k, v in sorted(dict(analysis_roles).items())},
            "steps": {str(k): int(v) for k, v in sorted(dict(steps).items())},
            "schedule_report": _normalise_for_fingerprint(sched_report),
            "time_unit_ps": float(time_unit_ps),
        },
        "md": _normalise_for_fingerprint(config.md),
        "potential": _potential_fingerprint_payload(config.kim, config_path=config_path),
        "metrics": _normalise_for_fingerprint(config.autotune.metrics),
        "convergence": _normalise_for_fingerprint(config.autotune.convergence),
        "production_acceptance": _normalise_for_fingerprint(config.autotune.production),
        "structure": _structure_fingerprint_payload(config.structure),
        "random_seed": int(config.random_seed),
        # Identifies which seed-derivation algorithm produced the per-box
        # seeds in this run. A change here invalidates resume against any
        # output that was generated under a different scheme; see
        # _RESUME_FINGERPRINT_SCHEMA bump (v1 -> v2) and _derive_box_seeds.
        "seed_scheme": _SEED_SCHEME,
        "resolved_context": {
            "pressure": float(md_pressure),
            "lammps_units": str(lammps_units),
        },
    }
    return {
        "schema": _RESUME_FINGERPRINT_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "sha256": _sha256_canonical_json(payload),
        "payload": payload,
    }


def _extract_resume_fingerprint(previous: Mapping[str, Any], *, outdir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    candidates = [
        previous.get("resume_fingerprint"),
        previous.get("fingerprint"),
    ]
    provenance = previous.get("provenance", {})
    if isinstance(provenance, Mapping):
        candidates.append(provenance.get("resume_fingerprint"))
    production = previous.get("production", {})
    if isinstance(production, Mapping):
        candidates.append(production.get("resume_fingerprint"))
    for cand in candidates:
        if isinstance(cand, Mapping):
            return dict(cand)
    if outdir is not None:
        sidecar = Path(outdir) / _RESUME_FINGERPRINT_SIDECAR
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text())
                if isinstance(data, Mapping):
                    return dict(data)
            except Exception:
                return None
    return None


def _fingerprint_sha256(fingerprint: Mapping[str, Any]) -> Optional[str]:
    sha = fingerprint.get("sha256", None)
    if isinstance(sha, str) and sha.strip():
        return sha.strip().lower()
    payload = fingerprint.get("payload", None)
    if isinstance(payload, Mapping):
        return _sha256_canonical_json(payload)
    return None


def _short_fingerprint_value(value: Any, max_len: int = 120) -> str:
    try:
        txt = json.dumps(_normalise_for_fingerprint(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except Exception:
        txt = repr(value)
    if len(txt) > max_len:
        return txt[: max_len - 3] + "..."
    return txt


def _diff_fingerprint_payloads(a: Any, b: Any, *, max_diffs: int = 12) -> list[str]:
    diffs: list[str] = []

    def rec(x: Any, y: Any, path: str) -> None:
        if len(diffs) >= max_diffs:
            return
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            keys = sorted(set(str(k) for k in x.keys()) | set(str(k) for k in y.keys()))
            sx = {str(k): v for k, v in x.items()}
            sy = {str(k): v for k, v in y.items()}
            for key in keys:
                if len(diffs) >= max_diffs:
                    return
                child = f"{path}.{key}" if path else key
                if key not in sx:
                    diffs.append(f"{child}: stored=<missing> current={_short_fingerprint_value(sy[key])}")
                elif key not in sy:
                    diffs.append(f"{child}: stored={_short_fingerprint_value(sx[key])} current=<missing>")
                else:
                    rec(sx[key], sy[key], child)
            return
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"{path}: stored length={len(x)} current length={len(y)}")
                return
            for i, (xi, yi) in enumerate(zip(x, y)):
                if len(diffs) >= max_diffs:
                    return
                rec(xi, yi, f"{path}[{i}]")
            return
        if x != y:
            diffs.append(f"{path}: stored={_short_fingerprint_value(x)} current={_short_fingerprint_value(y)}")

    rec(_normalise_for_fingerprint(a), _normalise_for_fingerprint(b), "")
    return diffs


def _validate_resume_fingerprint_or_raise(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    outdir: Path,
) -> None:
    stored = _extract_resume_fingerprint(previous, outdir=outdir)
    if stored is None:
        raise RuntimeError(
            "run-schedule cannot resume this output directory because no custom-schedule provenance "
            "fingerprint was found in run_results.json or custom_schedule_resume_fingerprint.json. "
            "Use a fresh output directory or remove the stale run_results.json."
        )
    stored_sha = _fingerprint_sha256(stored)
    current_sha = _fingerprint_sha256(current)
    if stored_sha is None or current_sha is None:
        raise RuntimeError("run-schedule resume fingerprint is malformed; use a fresh output directory")
    if stored_sha == current_sha:
        return
    stored_payload = stored.get("payload", {}) if isinstance(stored, Mapping) else {}
    current_payload = current.get("payload", {}) if isinstance(current, Mapping) else {}
    diffs = _diff_fingerprint_payloads(stored_payload, current_payload, max_diffs=12)
    diff_text = ""
    if diffs:
        diff_text = " First differences: " + "; ".join(diffs)
    raise RuntimeError(
        "run-schedule resume fingerprint mismatch: the current YAML/configuration does not match the "
        "existing run_results.json. Resuming would mix boxes from non-equivalent schedules or convergence "
        f"settings. stored={stored_sha} current={current_sha}.{diff_text} "
        "Use a fresh output directory or remove run_results.json to start a new run."
    )


def _final_status(
    *,
    n_accepted: int,
    min_boxes: int,
    check_convergence: bool,
    converged: bool,
    max_boxes: Optional[int],
    n_total: int,
) -> tuple[str, Optional[str]]:
    if int(n_accepted) < int(min_boxes):
        return "incomplete", f"accepted {int(n_accepted)} boxes, below min_boxes={int(min_boxes)}"
    if bool(check_convergence) and not bool(converged):
        cap = ""
        if max_boxes is not None and int(n_total) >= int(max_boxes):
            cap = f" after reaching max_boxes={int(max_boxes)}"
        return "not_converged", f"production convergence criteria were not satisfied{cap}"
    return "ok", None

def _stage_by_name(schedule: CustomSchedule) -> dict[str, CustomStageConfig]:
    return {s.name: s for s in schedule.stages}


def _role_by_stage_name(analysis_roles: Mapping[str, str]) -> dict[str, str]:
    return {str(v): str(k) for k, v in analysis_roles.items()}


def _guard_custom_schedule_runner_scope(config: RunConfig) -> None:
    """Fail early for runner modes the custom schedule cannot execute equivalently."""

    if str(getattr(config, "engine", "")).strip().lower() != "lammps":
        raise ValueError(
            "run-schedule currently supports engine='lammps' only; "
            "custom schedule execution is a continuous LAMMPS pipeline"
        )
    if getattr(config, "kim", None) is None:
        raise ValueError("run-schedule requires a LAMMPS potential block")
    md = getattr(config, "md", None)
    if str(getattr(md, "stage_continuity", "")).strip().lower() != "continuous":
        raise ValueError("run-schedule requires md.stage_continuity: continuous")


def _guard_custom_schedule_supported_equivalence_paths(
    *,
    config: RunConfig,
    metrics_cfg: Any,
    runner: Any,
    force_isotropic: bool,
) -> None:
    """Fail loudly for standard-runner features custom schedules do not yet reproduce.

    The custom runner currently shares the production structural-metric and MD
    convergence machinery, but it does not run the standard production DFT
    refinement or elastic-screen/timeseries paths. Guard those paths before any
    production boxes are generated so a custom schedule cannot silently claim a
    non-equivalent result.
    """

    prod_cfg = getattr(getattr(config, "autotune", None), "production", None)
    dft_opt = getattr(prod_cfg, "dft_opt", None)
    if bool(getattr(dft_opt, "enabled", False)):
        raise ValueError(
            "run-schedule does not yet support autotune.production.dft_opt.enabled=true; "
            "disable DFT refinement for custom schedules or use the standard production workflow"
        )

    elastic_cfg = getattr(metrics_cfg, "elastic", None)
    if elastic_cfg is None:
        return

    enabled = getattr(elastic_cfg, "enabled", "auto")
    if enabled is False:
        return

    from .elastic_screen import should_collect_elastic_stage_timeseries, should_run_elastic_screen

    requested: list[str] = []
    for role in ("melt", "quench", "relax"):
        run_screen, _strict_screen, _cfg_screen = should_run_elastic_screen(
            metrics_cfg,
            runner=runner,
            stage_role=role,
            force_isotropic=bool(force_isotropic),
        )
        run_series, _strict_series, _cfg_series = should_collect_elastic_stage_timeseries(
            metrics_cfg,
            runner=runner,
            stage_role=role,
            force_isotropic=bool(force_isotropic),
        )
        if bool(run_screen):
            requested.append(f"{role} elastic screen")
        if bool(run_series):
            requested.append(f"{role} elastic timeseries")

    if enabled is True or requested:
        detail = "explicitly enabled"
        if requested:
            detail = "would request " + ", ".join(requested)
        raise ValueError(
            "run-schedule does not yet support elastic production screens/timeseries; "
            f"the current elastic configuration {detail}. "
            "Set autotune.metrics.elastic.enabled=false for custom schedules, or use the standard production workflow."
        )


def _make_production_stage_specs(
    *,
    schedule: CustomSchedule,
    steps: Mapping[str, int],
    bdir: Path,
    input_data: Path,
    md_use: MDConfig,
    md_pressure: float,
    seed_stage: Mapping[str, int],
    role_by_name: Mapping[str, str],
    dump_every: int,
    need_stage_dump: Mapping[str, bool],
    quench_dump_every: int,
    relax_dump_settings: Mapping[str, Any],
    default_msd_every: int,
) -> list[Any]:
    from ..lammps_input import StageSpec

    specs: list[Any] = []
    prev_output: Path = input_data
    for i, st in enumerate(schedule.stages):
        role = role_by_name.get(st.name)
        output = bdir / f"{st.name}.data"
        velocity_mode = st.velocity_mode or ("create" if i == 0 else "preserve")
        write_dump: bool
        dump_every_stage: Optional[int]
        tail_frames: Optional[int] = None
        tail_stride: Optional[int] = None
        if role == "relax" and st.write_dump is None:
            write_dump = bool(relax_dump_settings.get("write_dump", False))
            dump_every_stage = relax_dump_settings.get("dump_every", None)
            tail_frames = relax_dump_settings.get("tail_dump_frames", None)
            tail_stride = relax_dump_settings.get("tail_dump_stride", None)
        elif st.write_dump is not None:
            write_dump = bool(st.write_dump)
            dump_every_stage = int(st.dump_every_steps) if st.dump_every_steps is not None else int(dump_every)
        elif role == "melt":
            write_dump = bool(need_stage_dump.get("melt", False))
            dump_every_stage = int(dump_every) if write_dump else None
        elif role == "quench":
            write_dump = bool(need_stage_dump.get("quench", False))
            dump_every_stage = int(quench_dump_every) if write_dump else None
        else:
            write_dump = False
            dump_every_stage = None
        if st.tail_dump_frames is not None:
            tail_frames = int(st.tail_dump_frames)
        if st.tail_dump_stride is not None:
            tail_stride = int(st.tail_dump_stride)
        if st.dump_every_steps is not None:
            dump_every_stage = int(st.dump_every_steps)
        specs.append(
            StageSpec(
                name=st.name,
                input_data=prev_output,
                output_data=output,
                temperature_start=float(st.temperature_start_K),
                temperature_stop=float(st.temperature_stop_K),
                pressure=float(md_pressure),
                equil_steps=int(st.equil_steps),
                run_steps=int(steps[st.name]),
                seed=int(seed_stage[st.name]),
                velocity_mode=velocity_mode,  # type: ignore[arg-type]
                force_isotropic=bool(st.force_isotropic) if st.force_isotropic is not None else (bool(getattr(md_use, "force_isotropic", False)) and i == 0),
                replicate=None,
                write_dump=bool(write_dump),
                dump_every=dump_every_stage,
                tail_dump_frames=tail_frames,
                tail_dump_stride=tail_stride,
                msd_every=int(st.msd_every) if st.msd_every is not None else int(default_msd_every),
                sample_ensemble=st.sample_ensemble,
            )
        )
        prev_output = output
    return specs


def run_custom_schedule(
    config: RunConfig,
    outdir: Path,
    *,
    config_path: Optional[Path] = None,
    resume: bool | None = None,
) -> dict[str, Any]:
    """Run a YAML-defined custom stage schedule.

    The standard ``run`` and ``autotune`` workflows are not called or modified.
    This function reuses production metrics/convergence but the temperature
    programme is taken literally from ``custom_schedule.stages``.
    """

    from ..analysis.motif_summary import summarize_production_crystal_motifs
    from ..runner import LammpsRunner
    from ..utils import ensure_dir
    from .metric_requirements import fixed_cutoffs_from_metrics, required_pairs_from_metrics
    from .metrics_policy import resolve_effective_metrics_config
    from .production_common import (
        analyse_production_box,
        build_production_convergence_spec,
        check_production_convergence,
        metrics_checked_from_conv_spec,
        plan_production_stage_diagnostics,
        resolve_production_relax_dump_settings,
        resolve_production_time_unit_ps,
        validate_production_entry_against_spec,
    )
    from .progress import CondensedProgressLog, atomic_write_json
    from .stage_runner import run_stages_continuous_lammps, stage_outcome_from_artifacts
    from .step_counts import resolve_lammps_units_style, resolve_md_pressure

    _guard_custom_schedule_runner_scope(config)
    runner = LammpsRunner(config.lammps)
    pot_cfg = config.kim
    md_use = config.md
    prod_cfg = config.autotune.production
    if not bool(getattr(prod_cfg, "enabled", False)):
        raise RuntimeError("run-schedule requires autotune.production.enabled=true")
    _guard_custom_schedule_supported_equivalence_paths(
        config=config,
        metrics_cfg=config.autotune.metrics,
        runner=runner,
        force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
    )

    raw = _load_raw_yaml(config_path) if config_path is not None else {}
    schedule = _schedule_from_raw(raw)
    analysis_roles = _validate_schedule(schedule)
    role_by_name = _role_by_stage_name(analysis_roles)
    stages_by_name = _stage_by_name(schedule)

    type_to_species = _type_to_species(config)
    time_unit_ps = resolve_production_time_unit_ps(config=config, engine="lammps", pot_cfg=pot_cfg)
    if time_unit_ps is None:
        raise RuntimeError("Unable to determine LAMMPS time unit in ps for custom schedule")
    time_unit_ps = float(time_unit_ps)
    steps = _schedule_steps(schedule, md_use=md_use, time_unit_ps=time_unit_ps)
    sched_report = _schedule_report(schedule, steps, md_use=md_use, time_unit_ps=time_unit_ps, analysis_roles=analysis_roles)
    md_pressure = float(resolve_md_pressure(config, md_use=md_use, override=None, default=0.0))
    lammps_units = resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal")
    resume_fingerprint = _build_resume_fingerprint(
        config=config,
        schedule=schedule,
        analysis_roles=analysis_roles,
        steps=steps,
        sched_report=sched_report,
        time_unit_ps=float(time_unit_ps),
        md_pressure=float(md_pressure),
        lammps_units=str(lammps_units),
        config_path=config_path,
    )

    outdir = Path(outdir)
    ensure_dir(outdir)
    progress = CondensedProgressLog(outdir / "condensed.log")
    progress.info("custom", "initialising custom fixed-schedule run")

    results_path = outdir / "run_results.json"
    previous_results: Optional[dict[str, Any]] = None
    if results_path.exists():
        prev = json.loads(results_path.read_text())
        if not isinstance(prev, Mapping):
            raise RuntimeError(f"run_results.json in {outdir} is not a JSON object")
        prod_status_src = prev.get("production", {})
        prod_status = prod_status_src.get("status", "") if isinstance(prod_status_src, Mapping) else ""
        prev_status = str(prev.get("status", prod_status) or "").lower()
        if resume is None or bool(resume):
            _validate_resume_fingerprint_or_raise(prev, resume_fingerprint, outdir=outdir)
            previous_results = dict(prev)
            if prev_status == "ok":
                progress.info("custom", "existing run_results.json fingerprint matches and is complete; returning cached result")
                return dict(prev)
        elif not bool(resume):
            raise RuntimeError(f"run_results.json already exists in {outdir}; use --resume or remove the directory")

    from ..structuregen import prepare_initial_structure

    metric_warnings: list[str] = []

    def _warn_metric(msg: str) -> None:
        metric_warnings.append(str(msg))
        progress.warn("metrics", str(msg))

    # Generate one representative initial structure for metrics defaults only.
    initial_ref = Path(prepare_initial_structure(config, outdir / "initial_reference"))
    metrics_cfg, _auto_defaults, metrics_summary = resolve_effective_metrics_config(
        config.autotune.metrics,
        structure_data=initial_ref,
        type_to_species=type_to_species,
        warn_fn=_warn_metric,
        context="run-schedule production",
    )
    if not bool(metrics_cfg.enabled):
        raise RuntimeError("run-schedule requires autotune.metrics.enabled=true for convergence-aware production")

    conv_cfg = config.autotune.convergence

    prod_dir = outdir / "production"
    ensure_dir(prod_dir)

    min_boxes = max(1, int(getattr(prod_cfg, "min_boxes", 10)))
    max_raw = getattr(prod_cfg, "max_boxes", None)
    max_boxes = None if max_raw is None else max(min_boxes, int(max_raw))
    batch = max(1, int(getattr(prod_cfg, "batch_boxes", 5)))
    check_convergence = bool(getattr(prod_cfg, "check_convergence", True))
    required_streak = max(1, int(getattr(prod_cfg, "consecutive_converged_checks", 1)))
    store_distributions = bool(getattr(prod_cfg, "store_distributions", True))
    exclude_defects = bool(getattr(prod_cfg, "exclude_coordination_defects", False))
    rejects_subdir = str(getattr(prod_cfg, "rejects_subdir", "rejects") or "rejects")
    rejects_dir = prod_dir / rejects_subdir
    if exclude_defects:
        ensure_dir(rejects_dir)

    melt_stage_cfg = stages_by_name[analysis_roles["melt"]]
    quench_stage_cfg = stages_by_name[analysis_roles["quench"]]
    relax_stage_cfg = stages_by_name[analysis_roles["relax"]]

    sampling_hint = {
        "Tm": float(melt_stage_cfg.temperature_stop_K),
        "freeze_temperature": float(relax_stage_cfg.temperature_stop_K),
        "custom_stage_schedule": 1.0,
    }
    sampling_hint.update({str(k): float(v) for k, v in dict(schedule.sampling_hint).items()})
    sched_report["sampling_hint"] = dict(sampling_hint)
    stage_diag = plan_production_stage_diagnostics(
        prod_cfg=prod_cfg,
        metrics_cfg=metrics_cfg,
        runner=runner,
        force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
        total_quench_steps=int(steps[quench_stage_cfg.name]),
        temperature_start=float(quench_stage_cfg.temperature_start_K),
        temperature_stop=float(quench_stage_cfg.temperature_stop_K),
        sampling_hint=sampling_hint,
    )
    dump_traj = bool(stage_diag["dump_traj"])
    dump_every = int(stage_diag["dump_every"])
    need_stage_dump = dict(stage_diag["need_stage_dump"])
    quench_dump_every = int(stage_diag["quench_dump_every"])
    quench_window_steps_range = stage_diag["quench_window_steps_range"]
    relax_dump_settings = resolve_production_relax_dump_settings(stage_diag=stage_diag, metrics_cfg=metrics_cfg)

    required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
    fixed_cut = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)
    prod_cutoffs: dict[tuple[int, int], float] = dict(fixed_cut or {})

    boxes: list[dict[str, Any]] = []
    rejected_boxes: list[dict[str, Any]] = []
    conv_spec: Optional[dict[str, Any]] = None
    conv_report: dict[str, Any] = {}
    converged = False
    converged_streak = 0
    next_box_id = 0

    if results_path.exists() and (resume is None or bool(resume)):
        try:
            prev = previous_results if previous_results is not None else json.loads(results_path.read_text())
            prod_prev = prev.get("production", {}) if isinstance(prev.get("production", {}), Mapping) else {}
            boxes = [dict(x) for x in prod_prev.get("boxes", []) if isinstance(x, Mapping)]
            rejected_boxes = [dict(x) for x in prod_prev.get("rejected_boxes", []) if isinstance(x, Mapping)]
            if isinstance(prod_prev.get("convergence_spec", None), Mapping):
                conv_spec = dict(prod_prev.get("convergence_spec", {}))
            if isinstance(prod_prev.get("convergence", None), Mapping):
                conv_report = dict(prod_prev.get("convergence", {}))
            existing_ids = [int(x.get("box", -1)) for x in boxes + rejected_boxes if isinstance(x.get("box", None), int)]
            next_box_id = (max(existing_ids) + 1) if existing_ids else 0
            progress.info("custom", f"resuming from {len(boxes)} accepted and {len(rejected_boxes)} rejected boxes")
        except Exception as exc:
            progress.warn("custom", f"failed to read previous run_results.json for resume: {type(exc).__name__}: {exc}")

    # Per-box seeds are derived deterministically from (seed_base, box_id, slot)
    # via a SHA-based stream rather than advancing a single rng cursor by box
    # count. The cursor scheme silently diverges from a fresh run when a prior
    # run was killed *between* `next_box_id += 1` and seed consumption, because
    # the resume code restores cursor position from the count of recorded
    # boxes only. With deterministic derivation, box k's seeds depend on k
    # alone, so resume reproducibility no longer depends on whether the
    # previous run was interrupted mid-box.
    #
    # The algorithm tag below is mirrored into the resume fingerprint as
    # `seed_scheme = _SEED_SCHEME`; changing the algorithm requires bumping
    # _SEED_SCHEME so any pre-existing run_results.json fingerprint mismatches
    # against the new runner and refuses to resume.
    assert _SEED_SCHEME == "sha256_box_slot_v1"
    seed_base = int(config.random_seed) + 97531
    stage_names = [str(st.name) for st in schedule.stages]

    def _derive_box_seeds(box_id: int) -> tuple[int, dict[str, int]]:
        import hashlib

        def _seed_for(slot: str) -> int:
            # Format is part of the sha256_box_slot_v1 contract: any change
            # here MUST bump _SEED_SCHEME (and the fingerprint schema bump
            # follows automatically because seed_scheme is in the payload).
            payload = f"{seed_base}:{int(box_id)}:{slot}".encode("utf-8")
            digest = hashlib.sha256(payload).digest()
            # Mask to 31 bits so the value is always representable as a signed
            # int LAMMPS seed and never zero (LAMMPS rejects seed=0 for some fixes).
            v = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            if v < 1:
                v = 1
            return v

        pack_seed = _seed_for("packmol")
        stage_seeds = {name: _seed_for(f"stage:{name}") for name in stage_names}
        return pack_seed, stage_seeds

    target = max(min_boxes, len(boxes))
    hard_max_boxes = 10000

    def _production_state(status: str, error: Optional[str] = None) -> dict[str, Any]:
        report_boxes = boxes if store_distributions else _strip_distributions(boxes)
        report_rej = rejected_boxes if store_distributions else _strip_distributions(rejected_boxes)
        motif_summary = summarize_production_crystal_motifs(boxes, rejected_boxes=rejected_boxes)
        return {
            "enabled": True,
            "workflow": "custom_stage_schedule",
            "workflow_label": str(schedule.workflow_label),
            "resume_fingerprint_sha256": str(resume_fingerprint.get("sha256", "")),
            "status": str(status),
            "error": None if error is None else str(error),
            "converged": bool(converged),
            "n_boxes": int(len(boxes)),
            "n_boxes_accepted": int(len(boxes)),
            "n_boxes_rejected": int(len(rejected_boxes)),
            "n_boxes_total": int(len(boxes) + len(rejected_boxes)),
            "min_boxes": int(min_boxes),
            "max_boxes": (None if max_boxes is None else int(max_boxes)),
            "batch_boxes": int(batch),
            "check_convergence": bool(check_convergence),
            "convergence_streak": int(converged_streak),
            "required_convergence_streak": int(required_streak),
            "dump_trajectory": bool(dump_traj),
            "dump_every_steps": int(dump_every),
            "schedule": dict(sched_report),
            "pressure": float(md_pressure),
            "lammps_units": str(lammps_units),
            "md": md_use.model_dump(mode="json"),
            "structure_generation": "per_box_independent_seed" if config.structure.generate is not None else "fixed_lammps_data",
            "exclude_coordination_defects": bool(exclude_defects),
            "rejects_subdir": str(rejects_subdir) if exclude_defects else None,
            "rejects_dir": (_relpath(rejects_dir, outdir) if exclude_defects and rejects_dir.exists() else None),
            "cutoffs": [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted((prod_cutoffs or {}).items())],
            "metrics_checked": metrics_checked_from_conv_spec(conv_spec),
            "convergence_spec": conv_spec,
            "converged_md": bool(converged),
            "convergence_md": conv_report,
            "convergence": conv_report,
            "crystal_motifs": motif_summary,
            "boxes": report_boxes,
            "rejected_boxes": report_rej,
            "ensemble_dir": _relpath(prod_dir, outdir),
        }

    def _summary(status: str, error: Optional[str] = None) -> dict[str, Any]:
        prod_state = _production_state(status, error=error)
        all_entries = list(prod_state.get("boxes", [])) + list(prod_state.get("rejected_boxes", []))
        return {
            "status": str(status),
            "error": None if error is None else str(error),
            "workflow": "custom_stage_schedule",
            "workflow_label": str(schedule.workflow_label),
            "parameters": {
                "engine": "lammps",
                "lammps_units": str(lammps_units),
                "time_unit_ps": float(time_unit_ps),
                "schedule": dict(sched_report),
                "pressure": float(md_pressure),
                "md": md_use.model_dump(mode="json"),
                "seed_base": int(config.random_seed) + 97531,
            },
            "replicates": all_entries,
            "production": prod_state,
            "resume_fingerprint": dict(resume_fingerprint),
            "provenance": {
                "resume_fingerprint_schema": str(resume_fingerprint.get("schema", _RESUME_FINGERPRINT_SCHEMA)),
                "resume_fingerprint_sha256": str(resume_fingerprint.get("sha256", "")),
                "resume_fingerprint_sidecar": _RESUME_FINGERPRINT_SIDECAR,
            },
            "metric_warnings": list(metric_warnings),
            "effective_metrics": dict(metrics_summary),
            "paths": {
                "condensed_log": "condensed.log",
                "run_results": "run_results.json",
                "resume_fingerprint": _RESUME_FINGERPRINT_SIDECAR,
            },
        }

    def _checkpoint(status: str, error: Optional[str] = None) -> None:
        atomic_write_json(outdir / _RESUME_FINGERPRINT_SIDECAR, resume_fingerprint)
        atomic_write_json(outdir / "run_results.json", _summary(status, error=error))

    _checkpoint("starting")

    while True:
        while len(boxes) < target:
            total_attempted = len(boxes) + len(rejected_boxes)
            if max_boxes is not None and total_attempted >= max_boxes:
                break
            if max_boxes is None and total_attempted >= hard_max_boxes:
                raise RuntimeError(
                    f"Custom production failed to converge after {hard_max_boxes} attempted boxes. "
                    "Set autotune.production.max_boxes or relax convergence tolerances."
                )
            b = int(next_box_id)
            next_box_id += 1
            bdir = prod_dir / f"box_{b:03d}"
            ensure_dir(bdir)
            progress.info("custom", f"box {b}: starting ({total_attempted + 1} total attempted)")

            seed_pack, seed_stage = _derive_box_seeds(b)

            cfg_box = _clone_config_with_structure_seed(config, seed_pack)
            input_data = Path(prepare_initial_structure(cfg_box, bdir / "initial"))

            stage_specs = _make_production_stage_specs(
                schedule=schedule,
                steps=steps,
                bdir=bdir,
                input_data=input_data,
                md_use=md_use,
                md_pressure=md_pressure,
                seed_stage=seed_stage,
                role_by_name=role_by_name,
                dump_every=dump_every,
                need_stage_dump=need_stage_dump,
                quench_dump_every=quench_dump_every,
                relax_dump_settings=relax_dump_settings,
                default_msd_every=int(config.autotune.tm_scan.msd_every),
            )
            stage_dirs = [bdir / st.name for st in stage_specs]
            stage_index = {st.name: i for i, st in enumerate(stage_specs)}

            pretty = " -> ".join(
                f"{s.temperature_start_K:g}->{s.temperature_stop_K:g}K:{s.name}" for s in schedule.stages
            )
            progress.info("custom", f"box {b}: continuous custom schedule {pretty}")
            arts = run_stages_continuous_lammps(
                runner,
                pot_cfg,
                md_use,
                stage_specs,
                stage_dirs,
                bdir / "continuous",
                potential_lines=None,
                type_to_species=type_to_species,
            )
            outs = [stage_outcome_from_artifacts(a, md_cfg=md_use, stage=s) for a, s in zip(arts, stage_specs)]

            melt_name = analysis_roles["melt"]
            quench_name = analysis_roles["quench"]
            relax_name = analysis_roles["relax"]
            relax_dir = bdir / relax_name
            relax_out = outs[stage_index[relax_name]]
            dump_path = relax_dir / f"{relax_name}.lammpstrj"
            traj_path = (relax_dir / "traj.extxyz") if (relax_dir / "traj.extxyz").exists() else dump_path
            entry, prod_cutoffs = analyse_production_box(
                box_id=int(b),
                outdir=outdir,
                melt_stage_dir=bdir / melt_name,
                quench_stage_dir=bdir / quench_name,
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
                bondlen_cdf_points=int(getattr(prod_cfg, "bondlen_cdf_points", 200)),
                angle_cdf_points=int(getattr(prod_cfg, "angle_cdf_points", 180)),
                seeds={"packmol": int(seed_pack), **{name: int(seed_stage[name]) for name in sorted(seed_stage)}},
                melt_elastic=None,
                relax_elastic=None,
                elastic_timeseries=None,
                exclude_coordination_defects=bool(exclude_defects),
                rejects_dir=(rejects_dir if exclude_defects else None),
                relax_dump_path=dump_path,
                relax_traj_path=traj_path,
            )
            entry["schedule"] = dict(sched_report)
            entry.setdefault("paths", {})["initial_data"] = _relpath(input_data, outdir)
            entry.setdefault("paths", {})["analysis_melt_dir"] = _relpath(bdir / melt_name, outdir)
            entry.setdefault("paths", {})["analysis_quench_dir"] = _relpath(bdir / quench_name, outdir)
            for st in schedule.stages:
                entry.setdefault("paths", {})[f"stage_{st.name}_dir"] = _relpath(bdir / st.name, outdir)

            if conv_spec is None:
                conv_spec = build_production_convergence_spec(entry)
            else:
                validate_production_entry_against_spec(entry, conv_spec, box_label=b)

            if bool(entry.get("reject")):
                rejected_boxes.append(entry)
                reason = str((entry.get("reject", {}) or {}).get("reason", "rejected"))
                progress.warn("custom", f"box {b}: rejected ({reason})")
            else:
                boxes.append(entry)
                progress.info("custom", f"box {b}: accepted ({len(boxes)} accepted total)")
            _checkpoint("running")

        if not check_convergence:
            converged = True
            conv_report = {}
            break
        if conv_spec is None:
            converged = False
            conv_report = {"error": "no convergence spec available"}
            break
        if len(boxes) < 1:
            converged = False
            conv_report = {"error": "no accepted boxes"}
            break

        converged_now, conv_report = check_production_convergence(boxes, conv_spec, conv_cfg)
        if converged_now:
            converged_streak += 1
        else:
            converged_streak = 0
        progress.convergence("custom", conv_report)
        progress.info("custom", f"convergence streak: {converged_streak}/{required_streak}")
        _checkpoint("running")

        if converged_now and converged_streak >= required_streak and len(boxes) >= min_boxes:
            converged = True
            break
        if max_boxes is not None and (len(boxes) + len(rejected_boxes)) >= max_boxes:
            break
        if max_boxes is None and (len(boxes) + len(rejected_boxes)) >= hard_max_boxes:
            raise RuntimeError(
                f"Custom production failed to converge after {hard_max_boxes} attempted boxes. "
                "Set autotune.production.max_boxes or relax convergence tolerances."
            )
        target = (len(boxes) + batch) if max_boxes is None else min(max_boxes, len(boxes) + batch)

    status, error = _final_status(
        n_accepted=len(boxes),
        min_boxes=min_boxes,
        check_convergence=check_convergence,
        converged=converged,
        max_boxes=max_boxes,
        n_total=(len(boxes) + len(rejected_boxes)),
    )
    summary = _summary(status, error=error)
    atomic_write_json(outdir / _RESUME_FINGERPRINT_SIDECAR, resume_fingerprint)
    atomic_write_json(outdir / "run_results.json", summary)
    if status == "ok":
        progress.info("custom", "custom schedule run complete; wrote run_results.json")
    else:
        progress.warn("custom", f"custom schedule run ended with status={status}: {error}")
    return summary


# Backward-compatible alias used by earlier hard-carbon demonstrator releases.
def run_hardcarbon(
    config: RunConfig,
    outdir: Path,
    *,
    config_path: Optional[Path] = None,
    resume: bool | None = None,
) -> dict[str, Any]:
    return run_custom_schedule(config, outdir, config_path=config_path, resume=resume)
