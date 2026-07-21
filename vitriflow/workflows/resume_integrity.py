"""Neutral, fail-closed integrity primitives for resumable workflows."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..analysis.provenance import json_sanitize
from ..utils import stable_file_identity


PRODUCTION_STATE_INTEGRITY_SCHEMA = "vitriflow.production_state_integrity.v1"
TASK_RESULT_SCHEMA = "vitriflow.box_task_result.v3"
TASK_RESULT_INTEGRITY_SCHEMA = "vitriflow.box_task_result_integrity.v1"

# Release-to-release execution-state compatibility is intentionally an
# allowlist, not a semantic-version range.  0.4.36.0 changes only diagnostic
# plotting/audit handling and canonical path spelling relative to the exact
# released 0.4.35.1 execution package.  The old package identity is embedded
# here so a locally modified 0.4.35.1 tree cannot enter this migration path.
RELEASE_RESUME_MIGRATION_SCHEMA = "vitriflow.release_resume_migration.v1"
_RELEASE_RESUME_MIGRATION_POLICY = (
    "exact_0.4.35.1_to_0.4.36.0_zero_committed_box_hotfix_v1"
)
_RELEASED_0_4_35_1_RUNTIME = {
    "schema": "vitriflow.runtime.v2",
    "vitriflow_version": "0.4.35.1",
    "package_content": {
        "schema": "vitriflow.package_content.v1",
        "algorithm": "sha256:length-prefixed-relative-path-and-content:v1",
        "sha256": "48f804d12295796a928a1257e56b6d91a7c250291cc4a67b092fc1346ca5f445",
        "file_count": 72,
    },
}

POTENTIAL_FILE_SUFFIXES = (
    ".ace", ".airebo", ".alloy", ".bop", ".eam", ".edip", ".json",
    ".meam", ".model", ".mtp", ".nn", ".pace", ".pb", ".pot",
    ".reax", ".snap", ".sw", ".table", ".tersoff", ".xml",
)


def canonical_json_sha256(value: Any) -> str:
    """Hash the exact strict-JSON representation persisted by VitriFlow."""

    return hashlib.sha256(
        json.dumps(
            json_sanitize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def prepare_release_resume_migration(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    workflow: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Prepare the sole authenticated cross-release resume transition.

    The caller must still compare the complete normalized payloads and may
    canonicalize only a workflow-specific spelling of an already content-
    authenticated structure path.  Returning ``None`` means that the ordinary
    exact-fingerprint rule remains mandatory.
    """

    stored_payload = stored.get("payload")
    current_payload = current.get("payload")
    if not isinstance(stored_payload, Mapping) or not isinstance(current_payload, Mapping):
        return None
    stored_sha = str(stored.get("sha256", "")).strip().lower()
    current_sha = str(current.get("sha256", "")).strip().lower()
    if (
        stored_sha != canonical_json_sha256(dict(stored_payload))
        or current_sha != canonical_json_sha256(dict(current_payload))
    ):
        return None

    source_version = str(stored_payload.get("vitriflow_version", ""))
    target_version = str(current_payload.get("vitriflow_version", ""))
    if source_version != "0.4.35.1" or target_version != "0.4.36.0":
        return None
    if str(stored_payload.get("workflow", "")) != str(workflow):
        return None
    if str(current_payload.get("workflow", "")) != str(workflow):
        return None

    stored_runtime = stored_payload.get("runtime")
    current_runtime = current_payload.get("runtime")
    if not isinstance(stored_runtime, Mapping) or not isinstance(current_runtime, Mapping):
        return None
    if dict(stored_runtime) != _RELEASED_0_4_35_1_RUNTIME:
        return None
    if str(current_runtime.get("schema", "")) != "vitriflow.runtime.v2":
        return None
    if str(current_runtime.get("vitriflow_version", "")) != "0.4.36.0":
        return None
    current_package = current_runtime.get("package_content")
    if not isinstance(current_package, Mapping):
        return None
    current_package_sha = str(current_package.get("sha256", "")).strip().lower()
    try:
        current_file_count = int(current_package.get("file_count", 0) or 0)
    except (TypeError, ValueError):
        return None
    if (
        str(current_package.get("schema", "")) != "vitriflow.package_content.v1"
        or str(current_package.get("algorithm", ""))
        != "sha256:length-prefixed-relative-path-and-content:v1"
        or re.fullmatch(r"[0-9a-f]{64}", current_package_sha) is None
        or current_file_count <= 0
    ):
        return None

    normalized_stored = deepcopy(dict(stored_payload))
    normalized_current = deepcopy(dict(current_payload))
    # These are the only release fields the common policy permits to differ.
    # Workflow-specific code must prove that every remaining field is equal.
    normalized_stored["vitriflow_version"] = target_version
    normalized_stored["runtime"] = deepcopy(dict(current_runtime))
    record = {
        "schema": RELEASE_RESUME_MIGRATION_SCHEMA,
        "policy": _RELEASE_RESUME_MIGRATION_POLICY,
        "workflow": str(workflow),
        "from_version": source_version,
        "to_version": target_version,
        "from_fingerprint_sha256": stored_sha,
        "to_fingerprint_sha256": current_sha,
        "from_package_content_sha256": str(
            _RELEASED_0_4_35_1_RUNTIME["package_content"]["sha256"]
        ),
        "to_package_content_sha256": current_package_sha,
        "allowed_release_field_changes": ["vitriflow_version", "runtime"],
    }
    return normalized_stored, normalized_current, record


def seal_release_resume_migration(record: Mapping[str, Any]) -> dict[str, Any]:
    """Self-digest a persisted release-resume migration audit record."""

    payload = dict(record)
    payload.pop("integrity", None)
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256:c14n-json:v1",
            "payload_sha256": canonical_json_sha256(payload),
        },
    }


def validate_release_resume_migration(record: Mapping[str, Any]) -> None:
    """Reject a malformed or modified persisted migration audit record."""

    if str(record.get("schema", "")) != RELEASE_RESUME_MIGRATION_SCHEMA:
        raise RuntimeError("Release-resume migration record has an unsupported schema")
    if str(record.get("policy", "")) != _RELEASE_RESUME_MIGRATION_POLICY:
        raise RuntimeError("Release-resume migration record has an unsupported policy")
    integrity = record.get("integrity")
    if not isinstance(integrity, Mapping):
        raise RuntimeError("Release-resume migration record has no integrity digest")
    if str(integrity.get("algorithm", "")) != "sha256:c14n-json:v1":
        raise RuntimeError("Release-resume migration record has an unsupported digest")
    payload = dict(record)
    payload.pop("integrity", None)
    if str(integrity.get("payload_sha256", "")).strip().lower() != canonical_json_sha256(
        payload
    ):
        raise RuntimeError("Release-resume migration record was modified")


def is_zero_committed_active_resume_state(previous: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint is the narrow first-box hotfix state.

    A stage directory may already exist, but no box may have entered the
    protected ensemble.  The resume executor may reuse a complete next-box
    stage pipeline after validating its canonical artifacts, then retry the
    failed post-processing with the unchanged deterministic seed stream.
    """

    production = previous.get("production")
    if not isinstance(production, Mapping):
        return False
    active = {"starting", "running"}
    if str(previous.get("status", "")).strip().lower() not in active:
        return False
    if str(previous.get("execution_status", "")).strip().lower() not in active:
        return False
    if str(production.get("status", "")).strip().lower() not in active:
        return False
    if str(production.get("execution_status", "")).strip().lower() not in active:
        return False
    if production.get("enabled") is not True or production.get("resumable") is not True:
        return False
    for key in ("n_boxes", "n_boxes_total", "n_boxes_accepted", "n_boxes_rejected"):
        try:
            if int(production.get(key, 0) or 0) != 0:
                return False
        except (TypeError, ValueError):
            return False
    for key in ("boxes", "rejected_boxes", "boxes_dft_final", "rejected_boxes_dft"):
        value = production.get(key)
        if value not in (None, []):
            return False
    return True


def seal_task_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach an internal content digest to a current external task result."""

    clean = dict(json_sanitize(dict(result)))
    clean.pop("result_integrity", None)
    if str(clean.get("schema", "")) != TASK_RESULT_SCHEMA:
        raise ValueError(
            f"Current task results must use schema {TASK_RESULT_SCHEMA!r}"
        )
    clean["result_integrity"] = {
        "schema": TASK_RESULT_INTEGRITY_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "payload_sha256": canonical_json_sha256(clean),
    }
    return clean


def validate_task_result_integrity(
    result: Mapping[str, Any],
    *,
    require_current: bool,
) -> bool:
    """Validate a task result digest, optionally rejecting legacy results.

    Version-1/version-2 task results remain readable by the generic analysis
    command, but they are never reusable execution caches.  Every successful result
    written by the current executor carries an exact digest and proof that the
    worker re-queried the engine build after execution.
    """

    schema = str(result.get("schema", ""))
    legacy_integrity_schema = "vitriflow.box_task_result.v2"
    if schema not in {TASK_RESULT_SCHEMA, legacy_integrity_schema}:
        if require_current:
            raise RuntimeError(
                "External task result is legacy or has an unsupported schema"
            )
        return False
    if schema == legacy_integrity_schema and require_current:
        raise RuntimeError(
            "External task result is legacy or has an unsupported schema"
        )
    integrity = result.get("result_integrity")
    if not isinstance(integrity, Mapping):
        raise RuntimeError("External task result has no content-integrity record")
    if str(integrity.get("schema", "")) != TASK_RESULT_INTEGRITY_SCHEMA:
        raise RuntimeError("External task result has an unsupported integrity schema")
    if str(integrity.get("algorithm", "")) != "sha256:c14n-json:v1":
        raise RuntimeError("External task result has an unsupported integrity algorithm")
    clean = dict(result)
    clean.pop("result_integrity", None)
    expected = canonical_json_sha256(clean)
    if str(integrity.get("payload_sha256", "")).strip().lower() != expected:
        raise RuntimeError("External task result content was modified or corrupted")
    if (
        schema == TASK_RESULT_SCHEMA
        and
        str(result.get("status", "")).strip().lower() in {"ok", "success"}
        and result.get("engine_build_identity_end_verified") is not True
    ):
        raise RuntimeError(
            "Successful external task result has no end-of-task engine verification"
        )
    return schema == TASK_RESULT_SCHEMA


def task_manifest_sha256(task_data: Mapping[str, Any]) -> str:
    """Return the digest persisted by the external task executor.

    This intentionally matches the task-manifest serialization contract used
    by :mod:`vitriflow.workflows.hpc`.  It is kept in this neutral module so a
    read-only ``analyze-output`` replay can authenticate ``task.json`` without
    introducing an ``hpc``/``output_analysis`` import cycle.
    """

    payload = json.dumps(
        dict(task_data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contained_task_artifact_path(box_dir: Path, relative_path: Any) -> tuple[str, Path]:
    """Resolve one manifest path without permitting absolute/path escapes."""

    raw = str(relative_path)
    rel = Path(raw)
    if not raw.strip() or rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(
            f"External task artifact path is not task-box-relative: {raw!r}"
        )
    normalized = str(rel)
    try:
        base = Path(box_dir).resolve(strict=False)
        resolved = (base / rel).resolve(strict=False)
        resolved.relative_to(base)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"External task artifact path escapes its task box: {raw!r}"
        ) from exc
    return normalized, resolved


def _task_artifact_identity_matches(path: Path, identity: Mapping[str, Any]) -> bool:
    try:
        artifact = Path(path)
        expected_size = int(identity.get("size_bytes"))
        expected_sha = str(identity.get("sha256", "")).strip().lower()
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        ):
            return False
        before = artifact.stat()
        if int(before.st_size) != expected_size:
            return False
        observed_sha = sha256_file(artifact).lower()
        after = artifact.stat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            return False
        return observed_sha == expected_sha
    except (OSError, TypeError, ValueError):
        return False


def validate_task_result_replay_artifacts(
    *,
    result: Mapping[str, Any],
    task_data: Mapping[str, Any],
    box_dir: Path,
) -> None:
    """Authenticate a current successful task bundle for read-only replay.

    The task result's self-digest protects only the JSON payload.  Replaying
    its diagnostics also depends on the exact ``task.json`` and every stage or
    diagnostic file recorded when the worker completed.  This validator binds
    all three layers and rejects truncated artifact manifests, path escapes,
    missing files, and content changes.
    """

    validate_task_result_integrity(result, require_current=True)
    if str(result.get("status", "")).strip().lower() not in {"ok", "success"}:
        raise RuntimeError("Only successful external task results can be replayed")
    if str(task_data.get("schema", "")) != "vitriflow.box_task.v1":
        raise RuntimeError("External task manifest has an unsupported schema")
    expected_task_sha = task_manifest_sha256(task_data)
    recorded_task_sha = str(result.get("task_manifest_sha256", "")).strip().lower()
    if recorded_task_sha != expected_task_sha:
        raise RuntimeError(
            "External task result does not authenticate the adjacent task manifest"
        )

    manifest = result.get("artifact_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != (
        "vitriflow.task_artifacts.v1"
    ):
        raise RuntimeError("External task result has no supported artifact manifest")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("External task artifact manifest is empty or malformed")

    identities: dict[str, Mapping[str, Any]] = {}
    for raw_identity in records:
        if not isinstance(raw_identity, Mapping):
            raise RuntimeError("External task artifact identity is malformed")
        rel, path = _contained_task_artifact_path(
            box_dir, raw_identity.get("path", "")
        )
        if rel in identities:
            raise RuntimeError(
                f"External task artifact manifest contains duplicate path {rel!r}"
            )
        if not _task_artifact_identity_matches(path, raw_identity):
            raise RuntimeError(
                f"External task artifact is missing or changed: {rel}"
            )
        identities[rel] = raw_identity

    # Reconstruct the exact set the worker-side manifest builder is required
    # to emit.  This prevents a shortened but otherwise internally consistent
    # manifest from silently dropping a stage source or diagnostic artifact.
    expected: dict[str, bool] = {}

    def add_expected(value: Any, *, required: bool) -> None:
        rel, path = _contained_task_artifact_path(box_dir, value)
        if rel in expected:
            expected[rel] = bool(expected[rel] or required)
            return
        if required and not path.is_file():
            raise RuntimeError(f"Required external task artifact is missing: {rel}")
        if path.is_file():
            expected[rel] = bool(required)

    outcomes = result.get("outcomes")
    if not isinstance(outcomes, Mapping) or not outcomes:
        raise RuntimeError("External task result has no completed stage outcomes")
    for stage_name, raw_outcome in outcomes.items():
        if not isinstance(raw_outcome, Mapping):
            raise RuntimeError(
                f"External task stage outcome {stage_name!r} is malformed"
            )
        output_data = raw_outcome.get("output_data")
        if not isinstance(output_data, str) or not output_data.strip():
            raise RuntimeError(
                f"External task stage outcome {stage_name!r} has no output_data"
            )
        stage_dir = Path(str(stage_name))
        add_expected(stage_dir / output_data, required=True)
        dump = raw_outcome.get("dump")
        if isinstance(dump, str) and dump.strip():
            add_expected(stage_dir / dump, required=True)
        for optional_name in (
            "thermo.csv",
            "msd.csv",
            "stage_artifacts.json",
            "final.extxyz",
        ):
            add_expected(stage_dir / optional_name, required=False)

    diagnostics = result.get("diagnostics")

    def add_diagnostic_artifacts(value: Any, *, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                add_diagnostic_artifacts(child_value, key=str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                add_diagnostic_artifacts(child, key=key)
            return
        if not isinstance(value, str) or not value.strip():
            return
        if key in {"csv", "summary", "plot"}:
            add_expected(value, required=True)
        elif key == "dir":
            _rel, directory = _contained_task_artifact_path(box_dir, value)
            if not directory.is_dir():
                raise RuntimeError(
                    f"Required external task diagnostic directory is missing: {value}"
                )
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    try:
                        relative = path.relative_to(
                            Path(box_dir).resolve(strict=False)
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise RuntimeError(
                            f"External task diagnostic directory escapes its task box: {value}"
                        ) from exc
                    add_expected(relative, required=True)

    if diagnostics is not None:
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError("External task diagnostics are malformed")
        add_diagnostic_artifacts(diagnostics)

    if set(identities) != set(expected):
        missing = sorted(set(expected) - set(identities))
        unexpected = sorted(set(identities) - set(expected))
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise RuntimeError(
            "External task artifact manifest is incomplete or non-canonical"
            + (": " + "; ".join(details) if details else "")
        )
    for rel, required in expected.items():
        if bool(identities[rel].get("required", False)) != bool(required):
            raise RuntimeError(
                f"External task artifact requiredness disagrees for {rel}"
            )


def sha256_file(path: Path) -> str:
    return str(stable_file_identity(Path(path))["sha256"])


def strict_file_identity(path: Path, *, configured_path: str | None = None) -> dict[str, Any]:
    p = Path(path)
    identity = stable_file_identity(p)
    return {
        "path": str(configured_path if configured_path is not None else p),
        "filename": p.name,
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def resolve_result_path(value: Any, *, outdir: Path) -> Path:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = Path(outdir) / p
    return p.resolve(strict=False)


def potential_command_file_paths(
    *,
    potential: Mapping[str, Any],
    plan: Mapping[str, Any],
    declared_values: Sequence[Any],
    base_dir: Path,
) -> list[Path]:
    """Resolve unambiguous command-file tokens or fail closed.

    Declared basenames are already covered by ``declared_values``. Absolute
    command paths are returned for independent hashing. Identifiable relative
    file references that were not declared are rejected because they are not
    guaranteed to be staged consistently across local/HPC execution.
    """

    declared_by_name: dict[str, list[Path]] = {}
    for value in declared_values:
        if value is None or not str(value).strip():
            continue
        path = resolve_result_path(value, outdir=base_dir)
        declared_by_name.setdefault(path.name, []).append(path)

    generated_names: set[str] = set()
    if str(potential.get("kind", "")).strip().lower() == "mg2_sin":
        generated_names.add(Path(str(potential.get("table_filename", "mg2_sin.table"))).name)
    for core in (potential.get("core_repulsion"), plan.get("core_repulsion")):
        realizes_zbl_table = (
            isinstance(core, Mapping)
            and bool(core.get("enabled", False))
            and str(core.get("style", "")).strip().lower() == "zbl"
        )
        if isinstance(core, Mapping) and (
            bool(core.get("tabulate", False)) or realizes_zbl_table
        ):
            generated_names.add(Path(str(core.get("table_filename", "buckingham_core.table"))).name)

    lines = [
        str(raw).strip()
        for raw in list(potential.get("commands", []) or [])
        + list(plan.get("potential_lines", []) or [])
        if str(raw).strip()
    ]
    variables: dict[str, str] = {}
    parsed_lines: list[tuple[str, list[str]]] = []
    for line in lines:
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"Could not parse potential command for provenance: {line!r}") from exc
        parsed_lines.append((line, tokens))
        if len(tokens) >= 4 and tokens[0].lower() == "variable" and tokens[2].lower() == "string":
            variables[str(tokens[1])] = " ".join(str(x) for x in tokens[3:])

    variable_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z])")

    def _expand_variables(value: str, *, line: str) -> str:
        expanded = str(value)
        for _ in range(16):
            matches = list(variable_pattern.finditer(expanded))
            if not matches:
                return expanded
            missing = [
                (match.group(1) or match.group(2) or "")
                for match in matches
                if (match.group(1) or match.group(2) or "") not in variables
            ]
            if missing:
                raise ValueError(
                    "Potential command contains an unresolved LAMMPS variable in a "
                    f"file-bearing position: {sorted(set(missing))} in {line!r}"
                )
            expanded = variable_pattern.sub(
                lambda match: variables[match.group(1) or match.group(2) or ""],
                expanded,
            )
        raise ValueError(f"Potential command variable expansion is cyclic or too deep: {line!r}")

    discovered: list[Path] = []
    seen: set[str] = set()
    hybrid_pair_style = False
    for line, tokens in parsed_lines:
        if tokens and tokens[0].lower() == "pair_style":
            if len(tokens) < 2:
                raise ValueError(f"Malformed pair_style command: {line!r}")
            active_style = tokens[1].lower()
            hybrid_pair_style = active_style == "hybrid" or active_style.startswith(
                "hybrid/"
            )
            continue
        if not tokens or tokens[0].lower() not in {"include", "pair_coeff"}:
            continue
        command = tokens[0].lower()
        # Hybrid pair_coeff syntax inserts a sub-style after the two type
        # selectors.  Tokens such as ``coul/long`` are style identifiers, not
        # filesystem paths; file-bearing arguments start after that token.
        candidates = (
            tokens[1:]
            if command == "include"
            else tokens[4:] if hybrid_pair_style else tokens[3:]
        )
        for token in candidates:
            value = _expand_variables(str(token).strip(), line=line)
            if not value or value in {"*", "NULL"}:
                continue
            token_path = Path(value).expanduser()
            name = token_path.name
            declared = declared_by_name.get(name, [])
            is_absolute = token_path.is_absolute()
            if declared and not is_absolute and "/" not in value and "\\" not in value:
                if len({str(path) for path in declared}) > 1:
                    raise ValueError(
                        f"Potential command token {value!r} matches multiple declared files; "
                        "use unique basenames"
                    )
                continue
            path_qualified = is_absolute or "/" in value or "\\" in value
            if name in generated_names and not path_qualified:
                continue
            looks_like_file = (
                command == "include"
                or path_qualified
                or value.lower().startswith("ffield")
                or value.lower().endswith(POTENTIAL_FILE_SUFFIXES)
            )
            path = (
                token_path.resolve(strict=False)
                if is_absolute
                else resolve_result_path(value, outdir=base_dir)
            )
            if is_absolute and path.is_file():
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    discovered.append(path)
                continue
            if looks_like_file:
                raise FileNotFoundError(
                    "Potential execution command references a file that was not materialised: "
                    f"{value!r} in {line!r}. Add it to potential.files or provide a valid path."
                )
    return discovered


def production_artifact_identities(
    state: Mapping[str, Any],
    *,
    outdir: Path,
    identity_cache: dict[str, tuple[int, int, int, int, int, str]] | None = None,
    force_rehash: bool = False,
) -> list[dict[str, Any]]:
    """Hash every source artifact required to trust persisted box results."""

    records: list[dict[str, Any]] = []
    seen_boxes: set[int] = set()

    outdir_resolved = Path(outdir).resolve(strict=False)

    def _contained_output_path(value: Any, *, label: str) -> Path:
        path = resolve_result_path(value, outdir=outdir)
        try:
            path.relative_to(outdir_resolved)
        except ValueError as exc:
            raise RuntimeError(f"Cannot trust {label}: path escapes the output directory: {value}") from exc
        return path

    def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot trust {label}: sidecar is not valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Cannot trust {label}: sidecar root is not a JSON object: {path}")
        return payload

    def _artifact_identity(path: Path, *, configured_path: str) -> dict[str, Any]:
        p = Path(path)
        if p.is_symlink():
            raise RuntimeError(
                f"Required resume/provenance artifact must not be a symbolic link: {p}"
            )
        try:
            stat = p.stat()
        except OSError as exc:
            raise RuntimeError(
                f"Required resume/provenance input is missing or not a file: {p}"
            ) from exc
        key = str(p.resolve(strict=False))
        cached = identity_cache.get(key) if identity_cache is not None else None
        if (
            not force_rehash
            and cached is not None
            and int(cached[0]) == int(stat.st_dev)
            and int(cached[1]) == int(stat.st_ino)
            and int(cached[2]) == int(stat.st_size)
            and int(cached[3]) == int(stat.st_mtime_ns)
            and int(cached[4]) == int(stat.st_ctime_ns)
        ):
            digest = str(cached[5])
        else:
            # ``sha256_file`` performs an fd-based stable-inode read.  Keep
            # this public callsite so checkpoint-cache tests and downstream
            # instrumentation can observe actual rehashes.
            digest = sha256_file(p)
            after = p.stat()
            if any(
                getattr(stat, name) != getattr(after, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            ):
                raise RuntimeError(
                    f"Required resume/provenance input changed while hashing: {p}"
                )
            stat = after
            if identity_cache is not None:
                identity_cache[key] = (
                    int(stat.st_dev),
                    int(stat.st_ino),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    int(stat.st_ctime_ns),
                    digest,
                )
        return {
            "path": str(configured_path),
            "filename": p.name,
            "size_bytes": int(stat.st_size),
            "sha256": digest,
        }

    def _is_sha256(value: Any) -> bool:
        text = str(value or "")
        return len(text) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in text)

    def _strict_manifest_row(row: Mapping[str, Any]) -> bool:
        # Pre-0.4.31 synthetic/legacy checkpoints did not identify the row
        # schema.  They remain readable under the existing hash envelope, while
        # every row written by current VitriFlow receives the stronger semantic
        # validation below.
        return str(row.get("schema", "")) == "vitriflow.structure_manifest.row.v2"

    def _snapshot_hashes(snapshot: Mapping[str, Any], *, box: int) -> dict[str, str] | None:
        schema = str(snapshot.get("schema", ""))
        keys = ("structure_hash", "cell_hash", "positions_hash", "symbols_hash")
        if schema == "vitriflow.structure_snapshot_ref.v1":
            hashes = {key: str(snapshot.get(key, "")) for key in keys}
            if not all(_is_sha256(value) for value in hashes.values()):
                raise RuntimeError(
                    f"Cannot trust production box {box}: referenced structure snapshot has malformed hashes"
                )
            return hashes
        if schema != "vitriflow.structure_snapshot.v1":
            return None

        # Old test fixtures and legacy snapshots may contain only n_atoms.  A
        # complete embedded v1 snapshot is independently re-hashed so the
        # snapshot and manifest cannot describe different structures.
        lattice = snapshot.get("lattice")
        if not isinstance(lattice, Mapping) or not all(
            key in snapshot for key in ("positions", "species")
        ):
            return None
        cell = lattice.get("cell")
        pbc = lattice.get("pbc")
        positions = snapshot.get("positions")
        species = snapshot.get("species")
        if not isinstance(cell, list) or not isinstance(pbc, list):
            raise RuntimeError(f"Cannot trust production box {box}: embedded structure snapshot is malformed")
        structure = {
            "cell": cell,
            "species": species,
            "positions": positions,
            "pbc": pbc,
        }
        return {
            "structure_hash": canonical_json_sha256(structure),
            "cell_hash": canonical_json_sha256({"cell": cell}),
            "positions_hash": canonical_json_sha256({"positions": positions}),
            "symbols_hash": canonical_json_sha256({"species": species}),
        }

    def _verify_structure_semantics(
        *,
        box: int,
        raw: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        manifest_row: Mapping[str, Any],
    ) -> None:
        if not _strict_manifest_row(manifest_row):
            return
        try:
            manifest_box = int(manifest_row.get("box_id"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot trust production box {box}: manifest box_id is invalid") from exc
        if manifest_box != int(box):
            raise RuntimeError(
                f"Cannot trust production box {box}: manifest box_id={manifest_box} disagrees with state"
            )

        hash_keys = ("structure_hash", "cell_hash", "positions_hash", "symbols_hash")
        for key in hash_keys:
            if not _is_sha256(manifest_row.get(key)):
                raise RuntimeError(f"Cannot trust production box {box}: manifest {key} is malformed")

        entry_manifest = raw.get("structure_manifest")
        if not isinstance(entry_manifest, Mapping):
            raise RuntimeError(f"Cannot trust production box {box}: state has no structure_manifest row")
        for key in (
            "schema",
            "box_id",
            *hash_keys,
            "n_atoms",
            "source_path",
            "source_role",
            "source_file_identity",
        ):
            if entry_manifest.get(key) != manifest_row.get(key):
                raise RuntimeError(
                    f"Cannot trust production box {box}: manifest {key} disagrees with state"
                )

        snap_hashes = _snapshot_hashes(snapshot, box=box)
        if snap_hashes is None:
            raise RuntimeError(
                f"Cannot trust production box {box}: current manifest requires a complete structure snapshot"
            )
        for key in hash_keys:
            if str(snap_hashes[key]) != str(manifest_row.get(key)):
                raise RuntimeError(
                    f"Cannot trust production box {box}: snapshot {key} disagrees with manifest"
                )

        try:
            manifest_n = int(manifest_row.get("n_atoms"))
            snapshot_n = int(snapshot.get("n_atoms"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot trust production box {box}: invalid n_atoms provenance") from exc
        if manifest_n < 1 or snapshot_n != manifest_n:
            raise RuntimeError(
                f"Cannot trust production box {box}: snapshot n_atoms disagrees with manifest"
            )

        source_value = manifest_row.get("source_path")
        recorded_source = manifest_row.get("source_file_identity")
        if source_value in (None, "") or not isinstance(recorded_source, Mapping):
            raise RuntimeError(f"Cannot trust production box {box}: manifest source identity is missing")
        manifest_source = resolve_result_path(source_value, outdir=outdir)
        actual_source = _artifact_identity(manifest_source, configured_path=str(source_value))
        try:
            recorded_size = int(recorded_source.get("size_bytes", -1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Cannot trust production box {box}: manifest source identity has an invalid size"
            ) from exc
        if (
            not _is_sha256(recorded_source.get("sha256"))
            or str(recorded_source.get("sha256")) != str(actual_source["sha256"])
            or recorded_size != int(actual_source["size_bytes"])
        ):
            raise RuntimeError(
                f"Cannot trust production box {box}: manifest source identity disagrees with source artifact"
            )
        snapshot_source = snapshot.get("source_file_identity")
        if snapshot_source is not None and snapshot_source != recorded_source:
            raise RuntimeError(
                f"Cannot trust production box {box}: snapshot source identity disagrees with manifest"
            )
        records.append(
            {
                "collection": collection,
                "box": box,
                "role": "manifest_source",
                **actual_source,
            }
        )

    def _declared_diagnostic_artifacts(
        value: Any,
        *,
        prefix: str,
        key: str | None = None,
    ):
        """Yield only schema-defined diagnostic artifact path fields.

        Diagnostic payloads also contain arbitrary labels, status strings and
        physical metadata.  Treating every string as a path would be both
        incorrect and brittle, so resume integrity follows the same explicit
        ``csv``/``summary``/``plot`` contract used by the task artifact
        manifest.
        """

        if isinstance(value, Mapping):
            for child_key in sorted(value, key=str):
                child_prefix = f"{prefix}.{child_key}" if prefix else str(child_key)
                yield from _declared_diagnostic_artifacts(
                    value[child_key],
                    prefix=child_prefix,
                    key=str(child_key),
                )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from _declared_diagnostic_artifacts(
                    child,
                    prefix=f"{prefix}[{index}]",
                    key=key,
                )
        elif (
            key in {"csv", "summary", "plot"}
            and isinstance(value, (str, Path))
            and str(value).strip()
        ):
            yield prefix, str(value)

    def _declared_named_artifacts(
        value: Any,
        *,
        prefix: str,
        path_keys: set[str],
        key: str | None = None,
    ):
        """Yield paths only from an explicit artifact-field allow-list."""

        if isinstance(value, Mapping):
            for child_key in sorted(value, key=str):
                child_prefix = f"{prefix}.{child_key}" if prefix else str(child_key)
                yield from _declared_named_artifacts(
                    value[child_key],
                    prefix=child_prefix,
                    path_keys=path_keys,
                    key=str(child_key),
                )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from _declared_named_artifacts(
                    child,
                    prefix=f"{prefix}[{index}]",
                    path_keys=path_keys,
                    key=key,
                )
        elif (
            key in path_keys
            and isinstance(value, (str, Path))
            and str(value).strip()
        ):
            yield prefix, str(value)

    for collection in ("boxes", "rejected_boxes"):
        entries = state.get(collection, [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Cannot trust production state: {collection} is not a list")
        for raw in entries:
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"Cannot trust production state: malformed entry in {collection}")
            try:
                box = int(raw.get("box"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Cannot trust production state: invalid box id in {collection}") from exc
            # Standard production boxes are one-based, while the public
            # custom-schedule workflow deliberately represents its sole
            # continuous trajectory as box_000.  Both are valid identities;
            # negative or duplicate identifiers are not.
            if box < 0:
                raise RuntimeError(
                    f"Cannot trust production state: box id must be non-negative in {collection}"
                )
            if box in seen_boxes:
                raise RuntimeError(f"Cannot trust production state: duplicate box id {box}")
            seen_boxes.add(box)
            paths = raw.get("paths")
            if not isinstance(paths, Mapping):
                raise RuntimeError(f"Cannot trust production box {box}: missing paths mapping")
            source_value = paths.get("relax_data") or paths.get("analysis_source")
            if source_value is None:
                raise RuntimeError(
                    f"Cannot trust production box {box}: no relax_data or analysis_source artifact is recorded"
                )
            source_path = resolve_result_path(source_value, outdir=outdir)
            identity = _artifact_identity(source_path, configured_path=str(source_value))
            parent_source_identity = dict(identity)
            records.append({"collection": collection, "box": box, "role": "source", **identity})

            analysis_value = paths.get("analysis_source")
            if analysis_value is not None:
                analysis_path = resolve_result_path(analysis_value, outdir=outdir)
                if analysis_path != source_path:
                    identity = _artifact_identity(analysis_path, configured_path=str(analysis_value))
                    records.append(
                        {"collection": collection, "box": box, "role": "analysis_source", **identity}
                    )

            # Per-box diagnostics are scientific evidence consumed by plots,
            # public analysis and convergence reporting.  Bind every declared
            # artifact, not only the final structure, so a terminal cached run
            # cannot remain apparently valid after a CSV/summary/plot is
            # deleted or replaced.
            seen_diagnostic_paths: set[str] = set()
            for family in (
                "stage_metrics",
                "elastic_melt",
                "elastic_relax",
                "elastic_timeseries",
            ):
                payload = raw.get(family)
                if payload is None:
                    continue
                if not isinstance(payload, (Mapping, list, tuple)):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: {family} diagnostics are malformed"
                    )
                for role, value in _declared_diagnostic_artifacts(
                    payload,
                    prefix=family,
                ):
                    path = _contained_output_path(
                        value,
                        label=f"production box {box} diagnostic {role}",
                    )
                    path_key = str(path)
                    if path_key in seen_diagnostic_paths:
                        continue
                    seen_diagnostic_paths.add(path_key)
                    identity = _artifact_identity(path, configured_path=str(value))
                    records.append(
                        {
                            "collection": collection,
                            "box": box,
                            "role": f"diagnostic.{role}",
                            **identity,
                        }
                    )

            provenance = raw.get("task_diagnostics_provenance")
            if provenance is not None:
                if not isinstance(provenance, Mapping):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: task diagnostic provenance is malformed"
                    )
                if (
                    str(provenance.get("schema", ""))
                    != "vitriflow.reused_task_diagnostics.v1"
                    or str(provenance.get("mode", ""))
                    != "validated_read_only_reuse"
                ):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: unsupported task diagnostic provenance"
                    )
                for provenance_role in ("task_result", "task_manifest"):
                    value = provenance.get(provenance_role)
                    if not isinstance(value, (str, Path)) or not str(value).strip():
                        raise RuntimeError(
                            f"Cannot trust production box {box}: task diagnostic provenance lacks {provenance_role}"
                        )
                    path = _contained_output_path(
                        value,
                        label=(
                            f"production box {box} task diagnostic {provenance_role}"
                        ),
                    )
                    path_key = str(path)
                    if path_key in seen_diagnostic_paths:
                        continue
                    seen_diagnostic_paths.add(path_key)
                    identity = _artifact_identity(path, configured_path=str(value))
                    records.append(
                        {
                            "collection": collection,
                            "box": box,
                            "role": f"diagnostic_provenance.{provenance_role}",
                            **identity,
                        }
                    )

            # Bind the remaining per-box analysis sidecars.  These files are
            # public metric/graph evidence and, for streamed graph analysis,
            # are also required to reconstruct aggregate outputs after a
            # resume.  The path allow-lists deliberately exclude status,
            # reason and graph-rule strings.
            sidecar_sources = (
                (
                    "paths.coord_defects",
                    paths.get("coord_defects"),
                    {"detail_json", "marked_extxyz", "shell_extxyz"},
                ),
                (
                    "paths.amorphous",
                    paths.get("amorphous"),
                    {"state_json"},
                ),
                (
                    "graph_analysis.chunk_paths",
                    (
                        raw.get("graph_analysis", {}).get("chunk_paths")
                        if isinstance(raw.get("graph_analysis"), Mapping)
                        else None
                    ),
                    {
                        "manifest",
                        "graph_rules",
                        "adaptive_graph_rules",
                        "representation_rules",
                        "adaptive_graph_rule_derivations",
                        "graph_metric_by_rule",
                        "metric_results",
                        "coordination_stability",
                        "shell_separability",
                    },
                ),
                (
                    "reject",
                    raw.get("reject"),
                    {"relax_data", "relax_dump"},
                ),
            )
            for prefix, payload, path_keys in sidecar_sources:
                if payload is None:
                    continue
                if not isinstance(payload, (Mapping, list, tuple)):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: {prefix} artifacts are malformed"
                    )
                for role, value in _declared_named_artifacts(
                    payload,
                    prefix=prefix,
                    path_keys=set(path_keys),
                ):
                    path = _contained_output_path(
                        value,
                        label=f"production box {box} sidecar {role}",
                    )
                    path_key = str(path)
                    if path_key in seen_diagnostic_paths:
                        continue
                    seen_diagnostic_paths.add(path_key)
                    identity = _artifact_identity(path, configured_path=str(value))
                    records.append(
                        {
                            "collection": collection,
                            "box": box,
                            "role": f"sidecar.{role}",
                            **identity,
                        }
                    )

            reject = raw.get("reject")
            reject_dir_value = (
                reject.get("reject_dir") if isinstance(reject, Mapping) else None
            )
            if reject_dir_value not in (None, ""):
                reject_dir = _contained_output_path(
                    reject_dir_value,
                    label=f"production box {box} reject_dir",
                )
                if reject_dir.is_symlink() or not reject_dir.is_dir():
                    raise RuntimeError(
                        f"Cannot trust production box {box}: reject_dir is not a regular directory"
                    )
                for artifact in sorted(reject_dir.rglob("*")):
                    if artifact.is_symlink():
                        raise RuntimeError(
                            f"Cannot trust production box {box}: reject_dir contains a symbolic link"
                        )
                    if not artifact.is_file():
                        continue
                    artifact_key = str(artifact.resolve(strict=False))
                    if artifact_key in seen_diagnostic_paths:
                        continue
                    seen_diagnostic_paths.add(artifact_key)
                    configured = str(artifact.relative_to(outdir_resolved))
                    identity = _artifact_identity(
                        artifact,
                        configured_path=configured,
                    )
                    records.append(
                        {
                            "collection": collection,
                            "box": box,
                            "role": "sidecar.reject_dir",
                            **identity,
                        }
                    )

            dft = raw.get("dft_opt")
            if isinstance(dft, Mapping) and str(dft.get("status", "")).strip().lower() == "ok":
                dft_paths = dft.get("paths")
                required_dft_paths = {
                    "dft_data": "data",
                    "dft_input": "input",
                    "dft_output": "output",
                    "dft_scf_diagnostics": "scf_diagnostics",
                    "dft_traj": "trajectory",
                    "dft_identity": "identity_manifest",
                }
                if not isinstance(dft_paths, Mapping):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: accepted DFT result has no paths mapping"
                    )
                missing_dft_paths = sorted(
                    role for role in required_dft_paths if dft_paths.get(role) is None
                )
                if missing_dft_paths:
                    raise RuntimeError(
                        f"Cannot trust production box {box}: accepted DFT result is missing "
                        + ", ".join(missing_dft_paths)
                    )

                dft_identities: dict[str, dict[str, Any]] = {}
                for role in required_dft_paths:
                    value = dft_paths.get(role)
                    path = resolve_result_path(value, outdir=outdir)
                    identity = _artifact_identity(path, configured_path=str(value))
                    dft_identities[role] = dict(identity)
                    records.append({"collection": collection, "box": box, "role": role, **identity})

                identity_value = dft_paths["dft_identity"]
                identity_path = _contained_output_path(
                    identity_value,
                    label=f"production box {box} DFT identity manifest",
                )
                identity_payload = _read_json_object(
                    identity_path,
                    label=f"production box {box} DFT identity manifest",
                )
                if str(identity_payload.get("schema", "")) != "vitriflow.cp2k_cell_opt.identity.v1":
                    raise RuntimeError(
                        f"Cannot trust production box {box}: unsupported DFT identity schema"
                    )
                if str(identity_payload.get("status", "")).strip().lower() != "completed":
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT identity is not completed"
                    )
                manifest_sha = str(identity_payload.get("manifest_sha256", "")).lower()
                manifest_unhashed = dict(identity_payload)
                manifest_unhashed.pop("manifest_sha256", None)
                if not _is_sha256(manifest_sha) or manifest_sha != canonical_json_sha256(
                    manifest_unhashed
                ):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT identity manifest was modified"
                    )
                calculation = identity_payload.get("calculation")
                calculation_payload = (
                    calculation.get("payload") if isinstance(calculation, Mapping) else None
                )
                if not isinstance(calculation, Mapping) or not isinstance(
                    calculation_payload, Mapping
                ):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT calculation identity is malformed"
                    )
                if (
                    str(calculation.get("schema", ""))
                    != "vitriflow.cp2k_cell_opt.calculation.v1"
                    or str(calculation_payload.get("schema", ""))
                    != "vitriflow.cp2k_cell_opt.calculation.v1"
                ):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: unsupported DFT calculation identity schema"
                    )
                calculation_sha = str(calculation.get("sha256", "")).lower()
                if not _is_sha256(calculation_sha) or calculation_sha != canonical_json_sha256(
                    dict(calculation_payload)
                ):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT calculation identity was modified"
                    )
                recorded_parent = calculation_payload.get("parent_relax_data")
                if not isinstance(recorded_parent, Mapping):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT parent identity is missing"
                    )
                if (
                    str(recorded_parent.get("sha256", "")).lower()
                    != str(parent_source_identity["sha256"]).lower()
                    or int(recorded_parent.get("size_bytes", -1))
                    != int(parent_source_identity["size_bytes"])
                ):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT calculation is bound to a different parent structure"
                    )
                manifest_artifacts = identity_payload.get("artifacts")
                if not isinstance(manifest_artifacts, Mapping):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT artifact identities are malformed"
                    )
                for path_role, manifest_role in required_dft_paths.items():
                    if path_role == "dft_identity":
                        continue
                    recorded = manifest_artifacts.get(manifest_role)
                    actual = dft_identities[path_role]
                    if not isinstance(recorded, Mapping) or (
                        str(recorded.get("sha256", "")).lower()
                        != str(actual["sha256"]).lower()
                        or int(recorded.get("size_bytes", -1))
                        != int(actual["size_bytes"])
                        or str(recorded.get("filename", "")) != str(actual["filename"])
                    ):
                        raise RuntimeError(
                            f"Cannot trust production box {box}: DFT {path_role} disagrees with its identity manifest"
                        )

                from ..cp2k_driver import (
                    assert_cp2k_cell_opt_converged,
                    count_cp2k_scf_failures,
                )

                dft_output_path = resolve_result_path(
                    dft_paths["dft_output"], outdir=outdir
                )
                assert_cp2k_cell_opt_converged(dft_output_path)
                scf_payload = _read_json_object(
                    resolve_result_path(
                        dft_paths["dft_scf_diagnostics"], outdir=outdir
                    ),
                    label=f"production box {box} DFT SCF diagnostics",
                )
                if str(scf_payload.get("schema", "")) != "vitriflow.cp2k_scf_diagnostics.v1":
                    raise RuntimeError(
                        f"Cannot trust production box {box}: unsupported DFT SCF diagnostics schema"
                    )
                diagnostic_outputs = scf_payload.get("outputs")
                if not isinstance(diagnostic_outputs, list) or not any(
                    isinstance(row, Mapping)
                    and str(row.get("phase", "")) == "cell_optimization"
                    and str(row.get("output", ""))
                    == Path(str(dft_paths["dft_output"])).name
                    for row in diagnostic_outputs
                ):
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT SCF diagnostics do not identify CELL_OPT output"
                    )
                counted_scf_failures = int(count_cp2k_scf_failures(dft_output_path))
                try:
                    recorded_scf_failures = int(
                        scf_payload.get("unconverged_scf_cycles", -1)
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT SCF failure count is invalid"
                    ) from exc
                if recorded_scf_failures != counted_scf_failures:
                    raise RuntimeError(
                        f"Cannot trust production box {box}: DFT SCF diagnostics disagree with output"
                    )

                dft_snapshot_payload: dict[str, Any] | None = None
                dft_manifest_row: Mapping[str, Any] | None = None
                for role, expected_schemas in (
                    (
                        "structure_snapshot",
                        {"vitriflow.structure_snapshot.v1", "vitriflow.structure_snapshot_ref.v1"},
                    ),
                    ("structure_manifest", {"vitriflow.structure_manifest.v2"}),
                ):
                    value = dft_paths.get(role)
                    if value is None:
                        raise RuntimeError(
                            f"Cannot trust production box {box}: accepted DFT result lacks {role}"
                        )
                    path = _contained_output_path(
                        value, label=f"production box {box} DFT {role}"
                    )
                    payload = _read_json_object(
                        path, label=f"production box {box} DFT {role}"
                    )
                    if str(payload.get("schema", "")) not in expected_schemas:
                        raise RuntimeError(
                            f"Cannot trust production box {box}: DFT {role} has unsupported schema"
                        )
                    if role == "structure_manifest":
                        structures = payload.get("structures")
                        if (
                            not isinstance(structures, list)
                            or len(structures) != 1
                            or not isinstance(structures[0], Mapping)
                        ):
                            raise RuntimeError(
                                f"Cannot trust production box {box}: DFT structure_manifest must contain one row"
                            )
                        dft_manifest_row = structures[0]
                    else:
                        dft_snapshot_payload = payload
                    identity = _artifact_identity(path, configured_path=str(value))
                    records.append(
                        {
                            "collection": collection,
                            "box": box,
                            "role": f"dft_{role}",
                            **identity,
                        }
                    )
                if dft_snapshot_payload is None or dft_manifest_row is None:
                    raise RuntimeError(
                        f"Cannot trust production box {box}: incomplete DFT structure provenance"
                    )
                _verify_structure_semantics(
                    box=box,
                    raw=dft,
                    snapshot=dft_snapshot_payload,
                    manifest_row=dft_manifest_row,
                )

            snapshot_payload: dict[str, Any] | None = None
            manifest_row: Mapping[str, Any] | None = None
            for role, expected_schemas in (
                (
                    "structure_snapshot",
                    {"vitriflow.structure_snapshot.v1", "vitriflow.structure_snapshot_ref.v1"},
                ),
                ("structure_manifest", {"vitriflow.structure_manifest.v2"}),
            ):
                value = paths.get(role)
                if value is None:
                    raise RuntimeError(f"Cannot trust production box {box}: mandatory {role} is not recorded")
                path = _contained_output_path(value, label=f"production box {box} {role}")
                payload = _read_json_object(path, label=f"production box {box} {role}")
                if str(payload.get("schema", "")) not in expected_schemas:
                    raise RuntimeError(
                        f"Cannot trust production box {box}: {role} has unsupported schema "
                        f"{payload.get('schema')!r}"
                    )
                if role == "structure_manifest":
                    structures = payload.get("structures")
                    if not isinstance(structures, list) or len(structures) != 1 or not isinstance(structures[0], Mapping):
                        raise RuntimeError(
                            f"Cannot trust production box {box}: structure_manifest must contain exactly one row"
                        )
                    manifest_row = structures[0]
                    entry_manifest = raw.get("structure_manifest")
                    if isinstance(entry_manifest, Mapping):
                        for hash_key in ("structure_hash", "cell_hash", "positions_hash", "symbols_hash"):
                            expected = entry_manifest.get(hash_key)
                            actual = structures[0].get(hash_key)
                            if expected is not None and str(actual) != str(expected):
                                raise RuntimeError(
                                    f"Cannot trust production box {box}: manifest {hash_key} disagrees with state"
                                )
                else:
                    snapshot_payload = payload
                identity = _artifact_identity(path, configured_path=str(value))
                records.append({"collection": collection, "box": box, "role": role, **identity})

            if snapshot_payload is None or manifest_row is None:
                raise RuntimeError(f"Cannot trust production box {box}: incomplete structure provenance")
            _verify_structure_semantics(
                box=box,
                raw=raw,
                snapshot=snapshot_payload,
                manifest_row=manifest_row,
            )

    graph_outputs = state.get("graph_outputs", {})
    if graph_outputs is None:
        graph_outputs = {}
    if not isinstance(graph_outputs, Mapping):
        raise RuntimeError("Cannot trust production state: graph_outputs is not a mapping")

    def _graph_paths(value: Any, prefix: str = ""):
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                child = f"{prefix}.{key}" if prefix else str(key)
                yield from _graph_paths(value[key], child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from _graph_paths(item, f"{prefix}[{index}]")
        elif isinstance(value, (str, Path)) and str(value).strip():
            yield prefix, str(value)

    seen_graph_paths: set[str] = set()
    for key, value in _graph_paths(graph_outputs):
        path = _contained_output_path(value, label=f"graph output {key}")
        resolved_key = str(path)
        if resolved_key in seen_graph_paths:
            continue
        seen_graph_paths.add(resolved_key)
        identity = _artifact_identity(path, configured_path=str(value))
        records.append({"collection": "graph_outputs", "box": -1, "role": str(key), **identity})
    records.sort(
        key=lambda item: (
            str(item["collection"]),
            int(item["box"]),
            str(item["role"]),
            str(item["path"]),
        )
    )
    return records


def validate_production_state_semantics(state: Mapping[str, Any]) -> None:
    boxes = state.get("boxes", [])
    rejected = state.get("rejected_boxes", [])
    if not isinstance(boxes, list) or not isinstance(rejected, list):
        raise RuntimeError("Cannot trust production state: box collections are malformed")
    attempt_ids: list[int] = []
    for collection_name, entries in (("boxes", boxes), ("rejected_boxes", rejected)):
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError(
                    f"Cannot trust production state: malformed entry in {collection_name}"
                )
            raw_box = entry.get("box")
            if isinstance(raw_box, bool):
                raise RuntimeError(
                    f"Cannot trust production state: invalid box id in {collection_name}"
                )
            try:
                box_id = int(raw_box)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"Cannot trust production state: invalid box id in {collection_name}"
                ) from exc
            if box_id < 0 or raw_box != box_id:
                raise RuntimeError(
                    f"Cannot trust production state: invalid box id in {collection_name}"
                )
            attempt_ids.append(box_id)
    ordered_ids = sorted(attempt_ids)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise RuntimeError("Cannot trust production state: duplicate box ids")
    if ordered_ids:
        # The public run/autotune workflows use one-based box ids, while the
        # legacy-compatible custom-schedule workflow deliberately starts at
        # zero.  Both are valid, but in either convention the committed
        # attempts must form the complete prefix.  The old ``[0]`` special
        # case accidentally rejected every valid custom checkpoint after its
        # first box.
        expected_ids = list(
            range(0, len(ordered_ids))
            if ordered_ids[0] == 0
            else range(1, len(ordered_ids) + 1)
        )
        if ordered_ids != expected_ids:
            raise RuntimeError(
                "Cannot safely resume production: attempted box ids are not a "
                "complete contiguous prefix, so exact RNG replay is impossible"
            )
    n_accepted = len(boxes)
    n_rejected = len(rejected)
    for key, expected in (
        ("n_boxes", n_accepted),
        ("n_boxes_accepted", n_accepted),
        ("n_boxes_rejected", n_rejected),
        ("n_boxes_total", n_accepted + n_rejected),
    ):
        if key in state and int(state.get(key, -1)) != int(expected):
            raise RuntimeError(f"Cannot trust production state: production count {key} is inconsistent")

    status = str(state.get("status", "")).strip().lower()
    enabled = bool(state.get("enabled", True))
    terminal_statuses = {"ok", "incomplete", "not_converged"}
    if status in terminal_statuses:
        required = {
            "execution_status",
            "converged",
            "check_convergence",
            "min_boxes",
            "n_boxes",
            "n_boxes_accepted",
            "n_boxes_rejected",
            "n_boxes_total",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise RuntimeError(
                "Cannot trust terminal production state: missing required fields " + ", ".join(missing)
            )
    if status:
        expected_execution = "completed" if status in terminal_statuses else status
        if str(state.get("execution_status", "")).strip().lower() != expected_execution:
            raise RuntimeError(
                "Cannot trust production state: production execution_status is missing or inconsistent"
            )
    if not enabled:
        if n_accepted or n_rejected or status not in {"not_requested", "disabled"}:
            raise RuntimeError("Cannot trust production state: disabled production state is inconsistent")
        return

    min_boxes = max(0, int(state.get("min_boxes", 0) or 0))
    check_convergence = bool(state.get("check_convergence", False))
    converged = bool(state.get("converged", False))
    resumable = bool(state.get("resumable", True))
    if not resumable and status not in terminal_statuses:
        raise RuntimeError(
            "Cannot trust production state: a non-terminal checkpoint cannot be marked non-resumable"
        )
    if check_convergence:
        for key in (
            "convergence_streak",
            "required_convergence_streak",
            "last_convergence_evaluated_n_boxes_total",
            "last_convergence_evaluated_n_boxes_accepted",
        ):
            if key not in state:
                raise RuntimeError(
                    "Cannot trust adaptive production state: missing exact convergence progress field "
                    f"{key}"
                )
        streak = int(state.get("convergence_streak", -1))
        required_streak = int(state.get("required_convergence_streak", 0))
        if streak < 0 or required_streak < 1 or streak > required_streak:
            raise RuntimeError(
                "Cannot trust adaptive production state: convergence streak is invalid"
            )
        last_total_raw = state.get("last_convergence_evaluated_n_boxes_total")
        last_accepted_raw = state.get("last_convergence_evaluated_n_boxes_accepted")
        if (last_total_raw is None) != (last_accepted_raw is None):
            raise RuntimeError(
                "Cannot trust adaptive production state: convergence evaluation counts are incomplete"
            )
        if last_total_raw is not None:
            last_total = int(last_total_raw)
            last_accepted = int(last_accepted_raw)
            if (
                last_total < 0
                or last_accepted < 0
                or last_accepted > last_total
                or last_total > n_accepted + n_rejected
                or last_accepted > n_accepted
            ):
                raise RuntimeError(
                    "Cannot trust adaptive production state: convergence evaluation counts are invalid"
                )
    if resumable and check_convergence and n_accepted:
        if any("distributions" not in entry for entry in boxes if isinstance(entry, Mapping)):
            raise RuntimeError(
                "Cannot safely resume adaptive production: stored distributions were omitted; "
                "use a fresh output directory or retain distributions for resumable runs"
            )
    if status == "ok":
        if n_accepted < min_boxes or (check_convergence and not converged):
            raise RuntimeError("Cannot trust production state: status='ok' is scientifically inconsistent")
    elif status == "incomplete" and n_accepted >= min_boxes:
        raise RuntimeError("Cannot trust production state: status='incomplete' is inconsistent")
    elif status == "not_converged":
        if n_accepted < min_boxes or not check_convergence or converged:
            raise RuntimeError("Cannot trust production state: status='not_converged' is inconsistent")


def attach_production_state_integrity(
    state: Mapping[str, Any],
    *,
    outdir: Path,
    identity_cache: dict[str, tuple[int, int, int, int, int, str]] | None = None,
    force_rehash: bool = False,
) -> dict[str, Any]:
    clean = dict(state)
    clean.pop("state_integrity", None)
    validate_production_state_semantics(clean)
    integrity = {
        "schema": PRODUCTION_STATE_INTEGRITY_SCHEMA,
        "algorithm": "sha256:c14n-json:v1",
        "state_sha256": canonical_json_sha256(clean),
        "artifacts": production_artifact_identities(
            clean,
            outdir=outdir,
            identity_cache=identity_cache,
            force_rehash=force_rehash,
        ),
    }
    integrity["integrity_sha256"] = canonical_json_sha256(integrity)
    clean["state_integrity"] = integrity
    return clean


def validate_production_resume_state(state: Mapping[str, Any], *, outdir: Path) -> None:
    stored = state.get("state_integrity")
    if not isinstance(stored, Mapping):
        raise RuntimeError(
            "Cannot safely resume: production checkpoint has no state/artifact integrity record; "
            "use a fresh output directory"
        )
    if stored.get("schema") != PRODUCTION_STATE_INTEGRITY_SCHEMA:
        raise RuntimeError("Cannot safely resume: unsupported production state-integrity schema")
    stored_envelope = dict(stored)
    stored_envelope_sha = str(stored_envelope.pop("integrity_sha256", "")).strip().lower()
    if stored_envelope_sha != canonical_json_sha256(stored_envelope):
        raise RuntimeError("Cannot safely resume: production state-integrity record was modified")
    clean = dict(state)
    clean.pop("state_integrity", None)
    if str(stored.get("state_sha256", "")).strip().lower() != canonical_json_sha256(clean):
        raise RuntimeError("Cannot safely resume: stored production checkpoint state was modified")
    if stored.get("artifacts") != production_artifact_identities(clean, outdir=outdir):
        raise RuntimeError(
            "Cannot safely resume: a production source artifact is missing or its contents changed"
        )

    validate_production_state_semantics(clean)


def production_final_status(
    *,
    n_accepted: int,
    min_boxes: int,
    check_convergence: bool,
    converged: bool,
    max_boxes: int | None,
    n_total: int,
) -> tuple[str, str | None]:
    if int(n_accepted) < int(min_boxes):
        return "incomplete", f"accepted {int(n_accepted)} boxes, below min_boxes={int(min_boxes)}"
    if bool(check_convergence) and not bool(converged):
        suffix = (
            f" after reaching max_boxes={int(max_boxes)}"
            if max_boxes is not None and int(n_total) >= int(max_boxes)
            else ""
        )
        return "not_converged", f"production convergence criteria were not satisfied{suffix}"
    return "ok", None


__all__ = [
    "PRODUCTION_STATE_INTEGRITY_SCHEMA",
    "attach_production_state_integrity",
    "canonical_json_sha256",
    "production_artifact_identities",
    "potential_command_file_paths",
    "production_final_status",
    "resolve_result_path",
    "sha256_file",
    "strict_file_identity",
    "task_manifest_sha256",
    "TASK_RESULT_SCHEMA",
    "TASK_RESULT_INTEGRITY_SCHEMA",
    "seal_task_result",
    "validate_task_result_replay_artifacts",
    "validate_task_result_integrity",
    "validate_production_resume_state",
    "validate_production_state_semantics",
]
