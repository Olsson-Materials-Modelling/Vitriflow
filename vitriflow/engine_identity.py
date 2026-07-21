"""Fail-closed identities for the external molecular-dynamics engines.

Configuration strings are not build identities.  Two executables with the
same name may implement different releases, packages, compiler options, or
patch levels.  This module therefore binds a calculation to both its exact
configured invocation and an authoritative, successful engine banner query.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Optional, Sequence

from .config import Cp2kConfig, LammpsConfig, RunConfig
from .utils import ensure_dir, run_cmd, stable_file_identity


ENGINE_BUILD_IDENTITY_SCHEMA = "vitriflow.engine_build_identity.v1"
ENGINE_BUILD_IDENTITIES_SCHEMA = "vitriflow.engine_build_identities.v1"
_IDENTITY_ALGORITHM = "sha256:c14n-json:v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _command_tokens(value: str | Sequence[str], *, name: str) -> list[str]:
    if isinstance(value, str):
        tokens = [value.strip()]
    else:
        tokens = [str(token).strip() for token in value]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must contain only non-empty command tokens")
    return tokens


def _normalise_banner(text: str) -> str:
    """Canonicalise platform newlines without erasing build information."""

    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return ("\n".join(lines) + "\n") if lines else ""


def _resolve_executable(token: str) -> Optional[Path]:
    raw = str(token)
    candidate: Optional[str]
    if "/" in raw or "\\" in raw:
        path = Path(raw).expanduser()
        candidate = str(path) if path.exists() else None
    else:
        candidate = shutil.which(raw)
    if candidate is None:
        return None
    try:
        resolved = Path(candidate).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    return resolved


_SCRIPT_SUFFIXES = frozenset({".py", ".pyw", ".sh", ".bash", ".pl", ".rb"})


def _looks_like_file_operand(token: str) -> bool:
    raw = str(token).strip()
    return bool(
        raw
        and not raw.startswith("-")
        and Path(raw).suffix.lower() in _SCRIPT_SUFFIXES
    )


def _regular_command_token(
    token: str,
    *,
    workdir: Path,
    executable: bool,
) -> Optional[Path]:
    raw = str(token)
    if executable:
        resolved = _resolve_executable(raw)
        if resolved is None:
            raise RuntimeError(f"Configured command executable cannot be resolved: {raw!r}")
        return resolved

    configured = Path(raw).expanduser()
    candidates = (
        [configured]
        if configured.is_absolute()
        else [Path(workdir) / configured, Path.cwd() / configured]
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise RuntimeError(
                f"Configured command file operand must not be a symbolic link: {raw!r}"
            )
        if candidate.exists():
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Configured command file operand cannot be resolved: {raw!r}"
                ) from exc
            if not resolved.is_file():
                raise RuntimeError(
                    f"Configured command file operand is not a regular file: {raw!r}"
                )
            return resolved
    if _looks_like_file_operand(raw):
        raise RuntimeError(
            f"Configured command contains an unbound file/script operand: {raw!r}"
        )
    return None


def _command_token_identities(
    tokens: Sequence[str],
    *,
    role: str,
    workdir: Path,
    executable_indices: Sequence[int],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve/hash every executable and every existing file command token."""

    executable_set = {int(index) for index in executable_indices}
    # ``python -m package`` executes mutable code which cannot be bound to a
    # single regular-file operand.  A concrete script path is safe because it
    # is discovered and hashed below.  Fail closed rather than pretending the
    # interpreter binary alone identifies the launched program.
    for executable_index in executable_set:
        if not (0 <= executable_index < len(tokens)):
            raise RuntimeError(
                f"Configured {role} has an invalid executable token index "
                f"{executable_index}"
            )
        executable_path = _resolve_executable(str(tokens[executable_index]))
        executable_name = (
            executable_path.name.lower()
            if executable_path is not None
            else Path(str(tokens[executable_index])).name.lower()
        )
        if executable_name.startswith(("python", "pypy")):
            suffix = [str(token) for token in tokens[executable_index + 1 :]]
            if "-m" in suffix:
                raise RuntimeError(
                    f"Configured {role} uses an unbound Python module invocation; "
                    "configure a concrete script/executable path instead"
                )
    resolved_tokens = [str(token) for token in tokens]
    identities: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        path = _regular_command_token(
            str(token),
            workdir=workdir,
            executable=index in executable_set,
        )
        if path is None:
            continue
        stable = stable_file_identity(path, reject_final_symlink=True)
        resolved_tokens[index] = str(stable["resolved_path"])
        identities.append(
            {
                "role": str(role),
                "token_index": int(index),
                "configured_token": str(token),
                "resolved_path": str(stable["resolved_path"]),
                "size_bytes": int(stable["size_bytes"]),
                "sha256": str(stable["sha256"]),
            }
        )
    return resolved_tokens, identities


def _assert_command_tokens_unchanged(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    engine: str,
) -> None:
    if list(before) != list(after):
        raise RuntimeError(
            f"Configured {engine} command files changed during build-identity query"
        )


def _probe(
    command: list[str],
    *,
    workdir: Path,
    engine: str,
) -> tuple[str, str]:
    try:
        rc, stdout, stderr = run_cmd(
            command,
            cwd=workdir,
            check=False,
            capture=True,
            timeout=30.0,
            kill_grace_sec=5.0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not query the configured {engine} build identity using "
            f"{command!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if int(rc) != 0:
        raise RuntimeError(
            f"Could not query the configured {engine} build identity using "
            f"{command!r}: return code {int(rc)}"
        )
    stdout_norm = _normalise_banner(stdout)
    stderr_norm = _normalise_banner(stderr)
    if not stdout_norm and not stderr_norm:
        raise RuntimeError(
            f"Configured {engine} build-identity query returned no banner output"
        )
    return stdout_norm, stderr_norm


def _banner_digest(stdout: str, stderr: str) -> dict[str, Any]:
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")
    combined = (
        len(stdout_bytes).to_bytes(8, "big")
        + stdout_bytes
        + len(stderr_bytes).to_bytes(8, "big")
        + stderr_bytes
    )
    return {
        "normalisation": "utf8:lf:strip-trailing-space-and-outer-blank-lines:v1",
        "stdout_size_bytes": len(stdout_bytes),
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_size_bytes": len(stderr_bytes),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "combined_sha256": hashlib.sha256(combined).hexdigest(),
    }


def _seal_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["algorithm"] = _IDENTITY_ALGORITHM
    result["identity_sha256"] = _canonical_sha256(payload)
    return result


def query_lammps_build_identity(
    config: LammpsConfig,
    *,
    workdir: Path,
) -> dict[str, Any]:
    """Return an authenticated identity for the configured LAMMPS build."""

    workdir = Path(workdir).expanduser().resolve(strict=False)
    ensure_dir(workdir)
    engine_command = _command_tokens(config.lammps_cmd, name="lammps.lammps_cmd")
    execution_command: list[str] = []
    if config.mpi_cmd:
        execution_command += [str(config.mpi_cmd), "-np", str(int(config.nprocs))]
    execution_command += engine_command
    execution_command += [str(arg) for arg in config.extra_args]

    # Query the engine directly.  MPI launchers commonly add node-, fabric-,
    # or scheduler-specific diagnostics which are not properties of the
    # LAMMPS build and would create false heterogeneous-worker failures.
    probe_command = engine_command + ["-h"]
    resolved_engine, engine_files = _command_token_identities(
        engine_command,
        role="engine_command",
        workdir=workdir,
        executable_indices=[0],
    )
    execution_engine_index = 3 if config.mpi_cmd else 0
    resolved_execution, execution_files = _command_token_identities(
        execution_command,
        role="execution_command",
        workdir=workdir,
        executable_indices=(
            [0, execution_engine_index]
            if config.mpi_cmd
            else [execution_engine_index]
        ),
    )
    resolved_probe, probe_files = _command_token_identities(
        probe_command,
        role="probe_command",
        workdir=workdir,
        executable_indices=[0],
    )
    stdout, stderr = _probe(resolved_probe, workdir=workdir, engine="LAMMPS")
    # Re-identify after the process exits.  ``stable_file_identity`` also
    # checks each individual read; this second pass closes replacement races
    # spanning process creation and banner parsing.
    _, engine_files_after = _command_token_identities(
        engine_command,
        role="engine_command",
        workdir=workdir,
        executable_indices=[0],
    )
    _, execution_files_after = _command_token_identities(
        execution_command,
        role="execution_command",
        workdir=workdir,
        executable_indices=(
            [0, execution_engine_index]
            if config.mpi_cmd
            else [execution_engine_index]
        ),
    )
    _, probe_files_after = _command_token_identities(
        probe_command,
        role="probe_command",
        workdir=workdir,
        executable_indices=[0],
    )
    _assert_command_tokens_unchanged(engine_files, engine_files_after, engine="LAMMPS")
    _assert_command_tokens_unchanged(
        execution_files, execution_files_after, engine="LAMMPS"
    )
    _assert_command_tokens_unchanged(probe_files, probe_files_after, engine="LAMMPS")

    combined_text = stdout + "\n" + stderr
    banner_pattern = re.compile(
        r"^\s*Large-scale Atomic/Molecular Massively Parallel Simulator\s*-\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    releases = {
        match.strip()
        for match in banner_pattern.findall(combined_text)
        if match.strip()
    }
    if len(releases) != 1:
        detail = "none" if not releases else repr(sorted(releases))
        raise RuntimeError(
            "Configured LAMMPS -h output did not contain exactly one unambiguous "
            f"LAMMPS release banner (found {detail})"
        )

    return _seal_identity(
        {
            "schema": ENGINE_BUILD_IDENTITY_SCHEMA,
            "status": "verified",
            "engine": "lammps",
            "configured_engine_command": engine_command,
            "resolved_engine_command": resolved_engine,
            "configured_execution_command": execution_command,
            "resolved_execution_command": resolved_execution,
            "probe": {
                "flag": "-h",
                "configured_command": probe_command,
                "resolved_command": resolved_probe,
                "release_banner": next(iter(releases)),
                "output": _banner_digest(stdout, stderr),
            },
            "command_file_identities": (
                engine_files + execution_files + probe_files
            ),
        }
    )


_CP2K_VERSION_PATTERN = re.compile(
    r"\bCP2K\s+version\s+(\d+(?:\.\d+)+)(?![.\d])",
    re.IGNORECASE,
)


def query_cp2k_build_identity(
    config: Cp2kConfig,
    *,
    workdir: Path,
) -> dict[str, Any]:
    """Return an authenticated identity for the configured CP2K build."""

    workdir = Path(workdir).expanduser().resolve(strict=False)
    ensure_dir(workdir)
    prefix = [str(token) for token in config.exec_prefix]
    engine_command = _command_tokens(config.cp2k_cmd, name="cp2k.cp2k_cmd")
    execution_command = list(prefix)
    if config.mpi_cmd:
        execution_command += [str(config.mpi_cmd), "-np", str(int(config.nprocs))]
    execution_command += engine_command
    execution_command += [str(arg) for arg in config.extra_args]

    # An execution prefix may select an environment whose CP2K (and MPI)
    # executables are intentionally absent from this process' PATH.  Resolving
    # those downstream executable names locally would either reject a valid
    # configuration or, worse, bind an unrelated executable from the caller's
    # environment.  In the prefixed case the prefix executable is the local
    # trust anchor: hash it and every concrete file operand, retain the exact
    # downstream tokens, and bind their selected build through the successful
    # prefixed version/banner probe below.
    delegated_by_prefix = bool(prefix)

    def _path_qualified(token: str) -> bool:
        raw = str(token)
        return Path(raw).expanduser().is_absolute() or "/" in raw or "\\" in raw

    if delegated_by_prefix:
        delegated_name = Path(engine_command[0]).name.lower()
        if delegated_name.startswith(("python", "pypy")) and "-m" in engine_command[1:]:
            raise RuntimeError(
                "Configured engine_command uses an unbound Python module invocation; "
                "configure a concrete script/executable path instead"
            )

    # Explicit paths remain locally resolvable and hashable even with a
    # prefix.  Only bare executable names are delegated to the selected PATH.
    engine_executable_indices = (
        [0] if not delegated_by_prefix or _path_qualified(engine_command[0]) else []
    )
    resolved_engine, engine_files = _command_token_identities(
        engine_command,
        role="engine_command",
        workdir=workdir,
        executable_indices=engine_executable_indices,
    )
    execution_engine_index = len(prefix) + (3 if config.mpi_cmd else 0)
    if delegated_by_prefix:
        # Only the first prefix token is launched by the current environment.
        # MPI and CP2K are launched after the prefix has selected its runtime
        # environment, so bare names must remain unresolved here.  Absolute or
        # otherwise concrete file tokens are still discovered and hashed as
        # ordinary command operands by ``_command_token_identities``.
        execution_executable_indices = [0]
        if config.mpi_cmd and _path_qualified(str(config.mpi_cmd)):
            execution_executable_indices.append(len(prefix))
        if _path_qualified(engine_command[0]):
            execution_executable_indices.append(execution_engine_index)
    else:
        execution_executable_indices = [execution_engine_index]
        if config.mpi_cmd:
            execution_executable_indices.append(0)
    resolved_execution, execution_files = _command_token_identities(
        execution_command,
        role="execution_command",
        workdir=workdir,
        executable_indices=execution_executable_indices,
    )

    # Query through the same selected environment as production.  For an MPI
    # build use one rank, matching ``Cp2kRunner.query_version``: this exercises
    # the configured MPI/CP2K launch chain without launching production's full
    # rank count.  Without an execution prefix, retain the established direct
    # CP2K query so scheduler/network diagnostics cannot contaminate identity.
    base_probe = list(prefix)
    if delegated_by_prefix and config.mpi_cmd:
        base_probe += [str(config.mpi_cmd), "-np", "1"]
    base_probe += engine_command
    probe_engine_index = len(prefix) + (
        3 if delegated_by_prefix and config.mpi_cmd else 0
    )
    probe_executable_indices = [0]
    if delegated_by_prefix and config.mpi_cmd and _path_qualified(str(config.mpi_cmd)):
        probe_executable_indices.append(len(prefix))
    if delegated_by_prefix and _path_qualified(engine_command[0]):
        probe_executable_indices.append(probe_engine_index)
    attempts: list[str] = []
    selected: Optional[
        tuple[str, list[str], list[str], list[dict[str, Any]], str, str, str]
    ] = None
    for flag in ("--version", "-v"):
        probe_command = base_probe + [flag]
        resolved_probe, probe_files = _command_token_identities(
            probe_command,
            role="probe_command",
            workdir=workdir,
            executable_indices=probe_executable_indices,
        )
        try:
            stdout, stderr = _probe(resolved_probe, workdir=workdir, engine="CP2K")
        except RuntimeError as exc:
            _, probe_files_after = _command_token_identities(
                probe_command,
                role="probe_command",
                workdir=workdir,
                executable_indices=probe_executable_indices,
            )
            _assert_command_tokens_unchanged(
                probe_files, probe_files_after, engine="CP2K"
            )
            attempts.append(str(exc))
            continue
        _, probe_files_after = _command_token_identities(
            probe_command,
            role="probe_command",
            workdir=workdir,
            executable_indices=probe_executable_indices,
        )
        _assert_command_tokens_unchanged(probe_files, probe_files_after, engine="CP2K")
        versions = set(_CP2K_VERSION_PATTERN.findall(stdout + "\n" + stderr))
        if len(versions) == 1:
            selected = (
                flag,
                probe_command,
                resolved_probe,
                probe_files,
                stdout,
                stderr,
                next(iter(versions)),
            )
            break
        detail = "none" if not versions else repr(sorted(versions))
        attempts.append(f"{probe_command!r}: expected one CP2K version, found {detail}")
    if selected is None:
        raise RuntimeError(
            "Could not query an unambiguous CP2K version/build banner from the "
            "configured executable: " + "; ".join(attempts)
        )
    flag, probe_command, resolved_probe, probe_files, stdout, stderr, version = selected
    _, engine_files_after = _command_token_identities(
        engine_command,
        role="engine_command",
        workdir=workdir,
        executable_indices=engine_executable_indices,
    )
    _, execution_files_after = _command_token_identities(
        execution_command,
        role="execution_command",
        workdir=workdir,
        executable_indices=execution_executable_indices,
    )
    _, selected_probe_files_after = _command_token_identities(
        probe_command,
        role="probe_command",
        workdir=workdir,
        executable_indices=probe_executable_indices,
    )
    _assert_command_tokens_unchanged(engine_files, engine_files_after, engine="CP2K")
    _assert_command_tokens_unchanged(
        execution_files, execution_files_after, engine="CP2K"
    )
    _assert_command_tokens_unchanged(
        probe_files, selected_probe_files_after, engine="CP2K"
    )

    payload: dict[str, Any] = {
        "schema": ENGINE_BUILD_IDENTITY_SCHEMA,
        "status": "verified",
        "engine": "cp2k",
        "configured_engine_command": engine_command,
        "resolved_engine_command": resolved_engine,
        "configured_execution_command": execution_command,
        "resolved_execution_command": resolved_execution,
        "probe": {
            "flag": flag,
            "configured_command": probe_command,
            "resolved_command": resolved_probe,
            "version": version,
            "output": _banner_digest(stdout, stderr),
        },
        "command_file_identities": engine_files + execution_files + probe_files,
    }
    if delegated_by_prefix:
        payload["exec_prefix_binding"] = {
            "mode": "runtime_environment_delegation_v1",
            "configured_prefix": prefix,
            "prefix_executable_index": 0,
            "execution_cp2k_index": execution_engine_index,
            "probe_cp2k_index": probe_engine_index,
            "execution_mpi_index": (len(prefix) if config.mpi_cmd else None),
            "probe_mpi_index": (len(prefix) if config.mpi_cmd else None),
            "probe_mpi_nprocs": (1 if config.mpi_cmd else None),
        }
    return _seal_identity(payload)


def validate_engine_build_identity(
    value: Mapping[str, Any],
    *,
    expected_engine: Optional[str] = None,
) -> dict[str, Any]:
    """Validate a self-hashed verified engine identity and return a copy."""

    if not isinstance(value, Mapping):
        raise RuntimeError("Engine build identity is not a mapping")
    data = dict(value)
    digest = str(data.pop("identity_sha256", "")).strip().lower()
    algorithm = str(data.pop("algorithm", ""))
    if data.get("schema") != ENGINE_BUILD_IDENTITY_SCHEMA:
        raise RuntimeError("Unsupported engine build identity schema")
    if data.get("status") != "verified":
        raise RuntimeError("Engine build identity is not verified")
    engine = str(data.get("engine", "")).strip().lower()
    if engine not in {"lammps", "cp2k"}:
        raise RuntimeError("Engine build identity has an unsupported engine")
    if expected_engine is not None and engine != str(expected_engine).strip().lower():
        raise RuntimeError(
            f"Engine build identity is for {engine}, expected {expected_engine}"
        )
    if algorithm != _IDENTITY_ALGORITHM or digest != _canonical_sha256(data):
        raise RuntimeError("Engine build identity digest is missing or inconsistent")
    command_fields = (
        ("engine_command", "configured_engine_command", "resolved_engine_command"),
        (
            "execution_command",
            "configured_execution_command",
            "resolved_execution_command",
        ),
    )
    command_values: dict[str, tuple[list[Any], list[Any]]] = {}
    for role, configured_key, resolved_key in command_fields:
        configured = data.get(configured_key)
        resolved = data.get(resolved_key)
        if (
            not isinstance(configured, list)
            or not configured
            or not isinstance(resolved, list)
            or len(configured) != len(resolved)
        ):
            raise RuntimeError(
                f"Engine build identity has malformed {role} command evidence"
            )
        command_values[role] = (configured, resolved)
    probe = data.get("probe")
    if not isinstance(probe, Mapping):
        raise RuntimeError("Engine build identity has no probe evidence")
    configured_probe = probe.get("configured_command")
    resolved_probe = probe.get("resolved_command")
    if (
        not isinstance(configured_probe, list)
        or not configured_probe
        or not isinstance(resolved_probe, list)
        or len(configured_probe) != len(resolved_probe)
    ):
        raise RuntimeError("Engine build identity has malformed probe command evidence")
    command_values["probe_command"] = (configured_probe, resolved_probe)
    output = probe.get("output")
    if not isinstance(output, Mapping) or not re.fullmatch(
        r"[0-9a-f]{64}", str(output.get("combined_sha256", "")).lower()
    ):
        raise RuntimeError("Engine build identity has malformed banner evidence")
    command_files = data.get("command_file_identities")
    if not isinstance(command_files, list) or not command_files:
        raise RuntimeError("Engine build identity has no command-file evidence")
    seen: set[tuple[str, int]] = set()
    for row in command_files:
        if not isinstance(row, Mapping):
            raise RuntimeError(
                "Engine build identity has malformed command-file evidence"
            )
        role = str(row.get("role", ""))
        if role not in command_values:
            raise RuntimeError(
                f"Engine build identity has an unsupported command-file role {role!r}"
            )
        try:
            token_index = int(row.get("token_index"))
            size_bytes = int(row.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Engine build identity has malformed command-file indices"
            ) from exc
        configured, resolved = command_values[role]
        key = (role, token_index)
        if key in seen or not (0 <= token_index < len(configured)):
            raise RuntimeError(
                "Engine build identity has duplicate or out-of-range command-file evidence"
            )
        seen.add(key)
        if (
            str(row.get("configured_token", "")) != str(configured[token_index])
            or str(row.get("resolved_path", "")) != str(resolved[token_index])
            or not Path(str(row.get("resolved_path", ""))).is_absolute()
            or size_bytes < 0
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("sha256", "")).lower()
            )
        ):
            raise RuntimeError(
                "Engine build identity has inconsistent command-file evidence"
            )
    engine_file_bound = ("engine_command", 0) in seen
    prefix_binding = data.get("exec_prefix_binding")
    if not engine_file_bound or prefix_binding is not None:
        # A prefixed CP2K command can only be resolved after the prefix has
        # selected its environment.  Accept that deliberately delegated
        # binding only when the signed command layout is internally exact,
        # both actual launch commands use the same non-empty prefix, and the
        # locally launched prefix executable is itself file-bound in both
        # command roles.  The selected CP2K build remains bound by the exact
        # downstream token plus the successful version/banner evidence.
        if engine != "cp2k" or not isinstance(prefix_binding, Mapping):
            raise RuntimeError(
                "Engine build identity does not bind its configured engine executable"
            )
        if str(prefix_binding.get("mode", "")) != ("runtime_environment_delegation_v1"):
            raise RuntimeError("CP2K engine identity has invalid exec-prefix binding")
        configured_prefix = prefix_binding.get("configured_prefix")
        if (
            not isinstance(configured_prefix, list)
            or not configured_prefix
            or any(
                not isinstance(token, str) or not token or token != token.strip()
                for token in configured_prefix
            )
        ):
            raise RuntimeError("CP2K engine identity has malformed exec-prefix binding")
        execution_configured, _ = command_values["execution_command"]
        probe_configured, _ = command_values["probe_command"]
        if (
            execution_configured[: len(configured_prefix)] != configured_prefix
            or probe_configured[: len(configured_prefix)] != configured_prefix
        ):
            raise RuntimeError("CP2K engine identity exec-prefix commands disagree")
        try:
            prefix_executable_index = int(prefix_binding.get("prefix_executable_index"))
            execution_cp2k_index = int(prefix_binding.get("execution_cp2k_index"))
            probe_cp2k_index = int(prefix_binding.get("probe_cp2k_index"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "CP2K engine identity has malformed exec-prefix indices"
            ) from exc
        if prefix_executable_index != 0 or not {
            ("execution_command", 0),
            ("probe_command", 0),
        }.issubset(seen):
            raise RuntimeError(
                "CP2K engine identity does not bind its exec-prefix executable"
            )
        engine_configured, _ = command_values["engine_command"]
        probe_flag = str(probe.get("flag", ""))
        if (
            execution_cp2k_index < len(configured_prefix)
            or probe_cp2k_index < len(configured_prefix)
            or execution_configured[
                execution_cp2k_index : execution_cp2k_index + len(engine_configured)
            ]
            != engine_configured
            or probe_configured[
                probe_cp2k_index : probe_cp2k_index + len(engine_configured)
            ]
            != engine_configured
            or not probe_flag
            or probe_configured[probe_cp2k_index + len(engine_configured) :]
            != [probe_flag]
        ):
            raise RuntimeError(
                "CP2K engine identity does not bind the delegated engine command"
            )
        cp2k_token = str(engine_configured[0])
        cp2k_path_qualified = (
            Path(cp2k_token).expanduser().is_absolute()
            or "/" in cp2k_token
            or "\\" in cp2k_token
        )
        if cp2k_path_qualified and not {
            ("engine_command", 0),
            ("execution_command", execution_cp2k_index),
            ("probe_command", probe_cp2k_index),
        }.issubset(seen):
            raise RuntimeError(
                "CP2K engine identity does not retain file evidence for its "
                "path-qualified delegated engine executable"
            )
        execution_mpi_index = prefix_binding.get("execution_mpi_index")
        probe_mpi_index = prefix_binding.get("probe_mpi_index")
        probe_mpi_nprocs = prefix_binding.get("probe_mpi_nprocs")
        mpi_values = (execution_mpi_index, probe_mpi_index, probe_mpi_nprocs)
        if any(value is None for value in mpi_values):
            if not all(value is None for value in mpi_values):
                raise RuntimeError(
                    "CP2K engine identity has inconsistent delegated MPI binding"
                )
            if execution_cp2k_index != len(
                configured_prefix
            ) or probe_cp2k_index != len(configured_prefix):
                raise RuntimeError(
                    "CP2K engine identity has unexpected tokens before delegated CP2K"
                )
        else:
            try:
                execution_mpi_index_int = int(execution_mpi_index)
                probe_mpi_index_int = int(probe_mpi_index)
                probe_mpi_nprocs_int = int(probe_mpi_nprocs)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "CP2K engine identity has malformed delegated MPI binding"
                ) from exc
            expected_mpi_index = len(configured_prefix)
            execution_mpi_nprocs: Optional[int]
            try:
                execution_mpi_nprocs = int(
                    execution_configured[execution_mpi_index_int + 2]
                )
            except (IndexError, TypeError, ValueError):
                execution_mpi_nprocs = None
            if (
                execution_mpi_index_int != expected_mpi_index
                or probe_mpi_index_int != expected_mpi_index
                or probe_mpi_nprocs_int != 1
                or execution_mpi_nprocs is None
                or execution_mpi_nprocs < 1
                or execution_cp2k_index != expected_mpi_index + 3
                or probe_cp2k_index != expected_mpi_index + 3
                or execution_mpi_index_int + 1 >= len(execution_configured)
                or probe_mpi_index_int + 1 >= len(probe_configured)
                or execution_configured[execution_mpi_index_int + 1] != "-np"
                or probe_configured[probe_mpi_index_int + 1 : probe_cp2k_index]
                != ["-np", "1"]
                or execution_configured[execution_mpi_index_int]
                != probe_configured[probe_mpi_index_int]
            ):
                raise RuntimeError(
                    "CP2K engine identity has inconsistent delegated MPI command"
                )
            mpi_token = str(execution_configured[execution_mpi_index_int])
            mpi_path_qualified = (
                Path(mpi_token).expanduser().is_absolute()
                or "/" in mpi_token
                or "\\" in mpi_token
            )
            if mpi_path_qualified and not {
                ("execution_command", execution_mpi_index_int),
                ("probe_command", probe_mpi_index_int),
            }.issubset(seen):
                raise RuntimeError(
                    "CP2K engine identity does not retain file evidence for its "
                    "path-qualified delegated MPI executable"
                )
    return dict(value)


def engine_build_identity_bundle(
    *,
    primary_engine: str,
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    primary = str(primary_engine).strip().lower()
    if primary not in {"lammps", "cp2k"}:
        raise ValueError(f"Unsupported primary engine {primary_engine!r}")
    checked: dict[str, Any] = {}
    for engine in sorted(identities):
        key = str(engine).strip().lower()
        checked[key] = validate_engine_build_identity(
            identities[engine], expected_engine=key
        )
    if primary not in checked:
        raise RuntimeError(f"No verified identity was supplied for primary engine {primary}")
    payload = {
        "schema": ENGINE_BUILD_IDENTITIES_SCHEMA,
        "status": "verified",
        "primary_engine": primary,
        "engines": checked,
    }
    return {
        **payload,
        "algorithm": _IDENTITY_ALGORITHM,
        "identity_sha256": _canonical_sha256(payload),
    }


def validate_engine_build_identity_bundle(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Engine identity bundle is not a mapping")
    data = dict(value)
    digest = str(data.pop("identity_sha256", "")).strip().lower()
    algorithm = str(data.pop("algorithm", ""))
    if data.get("schema") != ENGINE_BUILD_IDENTITIES_SCHEMA or data.get("status") != "verified":
        raise RuntimeError("Unsupported or unverified engine identity bundle")
    engines = data.get("engines")
    if not isinstance(engines, Mapping) or not engines:
        raise RuntimeError("Engine identity bundle has no engine identities")
    for name, identity in engines.items():
        validate_engine_build_identity(identity, expected_engine=str(name))
    primary = str(data.get("primary_engine", "")).strip().lower()
    if primary not in engines:
        raise RuntimeError("Engine identity bundle does not contain its primary engine")
    if algorithm != _IDENTITY_ALGORITHM or digest != _canonical_sha256(data):
        raise RuntimeError("Engine identity bundle digest is missing or inconsistent")
    return dict(value)


def assert_engine_build_identity_bundle_unchanged(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate two bundles and fail closed unless they are byte-identical identities."""

    expected_checked = validate_engine_build_identity_bundle(expected)
    observed_checked = validate_engine_build_identity_bundle(observed)
    expected_digest = str(expected_checked.get("identity_sha256", ""))
    observed_digest = str(observed_checked.get("identity_sha256", ""))
    if observed_digest != expected_digest:
        raise RuntimeError(
            f"Configured engine build changed {context}; refusing to seal a result "
            f"from mixed engine builds (initial={expected_digest}, "
            f"final={observed_digest})"
        )
    return observed_checked


def query_engine_build_identities(
    config: RunConfig,
    *,
    workdir: Path,
    primary_engine: Optional[str] = None,
    include_cp2k_refinement: bool = False,
) -> dict[str, Any]:
    primary = str(primary_engine or config.engine).strip().lower()
    identities: dict[str, Mapping[str, Any]] = {}
    if primary == "lammps":
        identities["lammps"] = query_lammps_build_identity(
            config.lammps, workdir=workdir
        )
    elif primary == "cp2k":
        if config.cp2k is None:
            raise RuntimeError("engine='cp2k' requires CP2K configuration")
        identities["cp2k"] = query_cp2k_build_identity(
            config.cp2k, workdir=workdir
        )
    else:
        raise ValueError(f"Unsupported engine {primary!r}")
    if include_cp2k_refinement and "cp2k" not in identities:
        if config.cp2k is None:
            raise RuntimeError("CP2K refinement requires CP2K configuration")
        identities["cp2k"] = query_cp2k_build_identity(
            config.cp2k, workdir=workdir
        )
    return engine_build_identity_bundle(
        primary_engine=primary,
        identities=identities,
    )


def deferred_engine_build_identities(primary_engine: str) -> dict[str, Any]:
    """Explicit marker used only while external tasks have not run yet."""

    primary = str(primary_engine).strip().lower()
    if primary not in {"lammps", "cp2k"}:
        raise ValueError(f"Unsupported engine {primary_engine!r}")
    payload = {
        "schema": ENGINE_BUILD_IDENTITIES_SCHEMA,
        "status": "deferred_to_external_worker",
        "primary_engine": primary,
        "engines": {},
    }
    return {
        **payload,
        "algorithm": _IDENTITY_ALGORITHM,
        "identity_sha256": _canonical_sha256(payload),
    }


def assert_homogeneous_engine_build_identities(
    values: Sequence[Mapping[str, Any]],
    *,
    expected: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Fail unless all verified task identities describe the exact same build."""

    if not values:
        raise RuntimeError("No successful task engine identities were available")
    checked = [validate_engine_build_identity(value) for value in values]
    first = checked[0]
    first_digest = str(first["identity_sha256"])
    for index, identity in enumerate(checked[1:], start=2):
        if str(identity["identity_sha256"]) != first_digest:
            raise RuntimeError(
                "External production task results were produced by heterogeneous "
                f"engine builds (first={first_digest}, task#{index}="
                f"{identity['identity_sha256']})"
            )
    if expected is not None:
        expected_checked = validate_engine_build_identity(expected)
        if str(expected_checked["identity_sha256"]) != first_digest:
            raise RuntimeError(
                "External task engine identity does not match the resumed "
                "production-state engine identity"
            )
    return first


def homogeneous_successful_task_engine_identity(
    task_results: Sequence[Mapping[str, Any]],
    *,
    expected: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Validate engine evidence on every successful external task result.

    Result sealing/runtime/artifact validation remains the caller's job.  This
    deliberately engine-neutral helper is suitable for both the HPC collector
    and the standalone ``analyze-output`` ingestion path.
    """

    identities: list[Mapping[str, Any]] = []
    for index, result in enumerate(task_results, start=1):
        if not isinstance(result, Mapping):
            raise RuntimeError(f"External task result #{index} is not a mapping")
        status = str(result.get("status", "")).strip().lower()
        if status not in {"ok", "success"}:
            continue
        identity = result.get("engine_build_identity")
        if not isinstance(identity, Mapping):
            raise RuntimeError(
                f"Successful external task result #{index} has no verified engine identity"
            )
        identities.append(identity)
    if not identities:
        if expected is not None:
            validate_engine_build_identity(expected)
        return None
    return assert_homogeneous_engine_build_identities(
        identities,
        expected=expected,
    )


__all__ = [
    "ENGINE_BUILD_IDENTITIES_SCHEMA",
    "ENGINE_BUILD_IDENTITY_SCHEMA",
    "assert_homogeneous_engine_build_identities",
    "assert_engine_build_identity_bundle_unchanged",
    "deferred_engine_build_identities",
    "engine_build_identity_bundle",
    "homogeneous_successful_task_engine_identity",
    "query_cp2k_build_identity",
    "query_engine_build_identities",
    "query_lammps_build_identity",
    "validate_engine_build_identity",
    "validate_engine_build_identity_bundle",
]
