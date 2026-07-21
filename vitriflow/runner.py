from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Optional

from .config import (
    LammpsConfig,
    Cp2kConfig,
    validate_cp2k_command,
    validate_cp2k_extra_args,
    validated_lammps_command,
    validated_lammps_extra_args,
)
from .utils import (
    ensure_dir,
    run_cmd,
    stable_file_identity,
    ExternalCommandError,
    CommandFailureContext,
    _tail_lines,
)


def _tail_file(path: Path, *, max_bytes: int = 200_000, n_lines: int = 80) -> str:
    """Tail file."""
    try:
        if not path.exists():
            return ""
        # read chunk bytes
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(-max_bytes, os.SEEK_END)
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
        return _tail_lines(text, n=int(n_lines))
    except Exception:
        return ""


@dataclass(frozen=True)
class RunResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    log_file: Path


_SAFE_LAMMPS_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_LAMMPS_RUNNER_FIXED_NAMES = frozenset(
    {"in.lammps", "screen.out", "stdout.txt", "stderr.txt"}
)

_SAFE_CP2K_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_CP2K_RUNNER_FIXED_NAMES = frozenset({"screen.out", "stdout.txt", "stderr.txt"})


def _validated_cp2k_artifact_name(value: object, *, role: str) -> str:
    """Return one unambiguous CP2K direct-child artifact basename."""

    raw = str(value)
    name = raw
    if (
        not name
        or name != name.strip()
        or name in {".", ".."}
        or Path(name).name != name
        or _SAFE_CP2K_ARTIFACT_NAME.fullmatch(name) is None
    ):
        raise ValueError(
            f"CP2K {role} must be one path-safe basename made from letters, "
            "digits, '_', '-' or '.'"
        )
    return name


def _validated_cp2k_input_file(input_file: Path, *, workdir: Path) -> Path:
    """Bind the invoked CP2K input to one direct, unaliased workdir file."""

    raw = Path(input_file).expanduser()
    candidate = Path(workdir) / raw if raw.parent == Path(".") else raw
    name = _validated_cp2k_artifact_name(candidate.name, role="input filename")
    root = Path(workdir).resolve(strict=True)
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"CP2K input parent cannot be resolved: {candidate.parent}") from exc
    if parent != root or candidate.name != name:
        raise ValueError(
            "CP2K input_file must be a direct child of the execution workdir; "
            f"got {candidate} for {root}"
        )
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise FileNotFoundError(str(candidate)) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(
            f"CP2K input_file must be a direct regular non-symlink file: {candidate}"
        )
    if int(info.st_nlink) != 1:
        raise ValueError(f"CP2K input_file must not be hard-linked: {candidate}")
    return candidate


def _unlink_cp2k_attempt_artifact(path: Path) -> None:
    """Remove one prior CP2K runner artifact without following aliases."""

    candidate = Path(path)
    try:
        candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(
            f"Cannot inspect stale CP2K attempt artifact before execution: {candidate}"
        ) from exc
    try:
        candidate.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot remove stale CP2K attempt artifact before execution: {candidate}"
        ) from exc


def _atomic_write_cp2k_runner_text(path: Path, text: str) -> None:
    """Publish runner diagnostics without truncating an existing alias target."""

    destination = Path(path)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
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
            f"Cannot publish CP2K runner artifact safely: {destination}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _validated_lammps_log_name(value: object) -> str:
    """Return one unambiguous direct-child log basename."""

    raw_name = str(value)
    name = raw_name.strip()
    if (
        not name
        or raw_name != name
        or name in {".", ".."}
        or Path(name).name != name
        or _SAFE_LAMMPS_ARTIFACT_NAME.fullmatch(name) is None
    ):
        raise ValueError(
            "LAMMPS log_name must be one path-safe basename made from letters, "
            "digits, '_', '-' or '.'"
        )
    if name in _LAMMPS_RUNNER_FIXED_NAMES:
        raise ValueError(
            f"LAMMPS log_name {name!r} collides with a runner-owned artifact"
        )
    return name


def _unlink_lammps_attempt_artifact(path: Path) -> None:
    """Remove one prior-attempt artifact without following aliases."""

    candidate = Path(path)
    try:
        candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(
            f"Cannot inspect stale LAMMPS attempt artifact before execution: {candidate}"
        ) from exc
    try:
        candidate.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot remove stale LAMMPS attempt artifact before execution: {candidate}"
        ) from exc


def _atomic_write_lammps_runner_text(path: Path, text: str) -> None:
    """Publish runner-owned text without truncating a link target."""

    destination = Path(path)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
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
            f"Cannot publish LAMMPS runner artifact safely: {destination}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _validated_lammps_mpi_command(value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "LAMMPS mpi_cmd must be null or one exact non-empty executable token"
        )
    return value


def _validated_lammps_nprocs(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("LAMMPS nprocs must be an integer >= 1")
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LAMMPS nprocs must be an integer >= 1") from exc
    if not math.isfinite(numeric) or numeric != float(integer) or integer < 1:
        raise ValueError("LAMMPS nprocs must be an integer >= 1")
    return integer


def _validated_lammps_timeout(value: object, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"LAMMPS {field_name} must be finite and > 0")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"LAMMPS {field_name} must be finite and > 0") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"LAMMPS {field_name} must be finite and > 0")
    return numeric


def _validated_lammps_kill_grace(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("LAMMPS kill_grace_sec must be finite and >= 0")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LAMMPS kill_grace_sec must be finite and >= 0") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError("LAMMPS kill_grace_sec must be finite and >= 0")
    return numeric


def _validated_current_lammps_log(path: Path) -> None:
    """Prove that the successful attempt created one usable direct log file."""

    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise RuntimeError(
            "LAMMPS returned success without creating its current log file: "
            f"{candidate}"
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or int(before.st_size) <= 0
    ):
        raise RuntimeError(
            "LAMMPS returned success but its current log is not a nonempty, "
            f"direct, single-link regular file: {candidate}"
        )
    identity = stable_file_identity(candidate, reject_final_symlink=True)
    try:
        after = candidate.lstat()
    except OSError as exc:
        raise RuntimeError(f"LAMMPS log changed during validation: {candidate}") from exc
    if (
        int(after.st_dev) != int(identity["device"])
        or int(after.st_ino) != int(identity["inode"])
        or int(after.st_size) != int(identity["size_bytes"])
        or int(after.st_nlink) != 1
        or int(after.st_size) <= 0
    ):
        raise RuntimeError(f"LAMMPS log changed during validation: {candidate}")


class LammpsRunner:
    def __init__(self, cfg: LammpsConfig):
        self.cfg = cfg

    def run(self, input_script: str, workdir: Path, log_name: str, *, timeout_sec: Optional[float] = None) -> RunResult:
        engine_command = validated_lammps_command(
            getattr(self.cfg, "lammps_cmd", "lmp")
        )
        extra_args = validated_lammps_extra_args(
            getattr(self.cfg, "extra_args", [])
        )
        log_basename = _validated_lammps_log_name(log_name)
        mpi_command = _validated_lammps_mpi_command(
            getattr(self.cfg, "mpi_cmd", None)
        )
        nprocs = _validated_lammps_nprocs(getattr(self.cfg, "nprocs", 1))
        configured_timeout = _validated_lammps_timeout(
            getattr(self.cfg, "timeout_sec", None), field_name="timeout_sec"
        )
        explicit_timeout = _validated_lammps_timeout(
            timeout_sec, field_name="timeout_sec override"
        )
        kill_grace = _validated_lammps_kill_grace(
            getattr(self.cfg, "kill_grace_sec", 5.0)
        )
        workdir = Path(workdir)
        if workdir.is_symlink():
            raise ValueError(f"LAMMPS workdir must not be a symbolic link: {workdir}")
        ensure_dir(workdir)
        if workdir.is_symlink() or not workdir.is_dir():
            raise ValueError(f"LAMMPS workdir must be a real directory: {workdir}")
        in_path = workdir / "in.lammps"
        log_file = workdir / log_basename
        screen_file = workdir / "screen.out"
        stdout_file = workdir / "stdout.txt"
        stderr_file = workdir / "stderr.txt"

        # Every invocation is a new provenance attempt.  Remove the complete
        # deterministic runner namespace first, including broken symlinks and
        # hardlink directory entries.  Failure is fatal: retaining any prior
        # diagnostic could make a current failure look like a successful run.
        for artifact in (in_path, log_file, screen_file, stdout_file, stderr_file):
            _unlink_lammps_attempt_artifact(artifact)
        _atomic_write_lammps_runner_text(in_path, input_script)

        cmd = []
        if mpi_command is not None:
            cmd += [mpi_command, "-np", str(nprocs)]
        # lammps string tokenized
        if isinstance(engine_command, list):
            cmd += engine_command
        else:
            cmd += [engine_command]
        cmd += ["-in", str(in_path.name), "-log", str(log_file.name)]
        cmd += ["-screen", str(screen_file.name)]
        cmd += extra_args

        timeout_use = explicit_timeout if explicit_timeout is not None else configured_timeout
        rc, out, err = run_cmd(
            cmd,
            cwd=workdir,
            check=False,
            capture=True,
            timeout=timeout_use,
            kill_grace_sec=kill_grace,
        )
        # captured regardless mpi
        # suppress screen canonical
        _atomic_write_lammps_runner_text(stdout_file, out)
        _atomic_write_lammps_runner_text(stderr_file, err)
        if rc != 0:
            # include screen exception
            ctx = CommandFailureContext(
                screen_tail=_tail_file(screen_file),
                log_tail=_tail_file(log_file),
                stdout_tail=_tail_lines(out, n=80),
                stderr_tail=_tail_lines(err, n=80),
            )
            raise ExternalCommandError(cmd, rc, out, err, context=ctx)
        _validated_current_lammps_log(log_file)
        return RunResult(cmd=cmd, returncode=rc, stdout=out, stderr=err, log_file=log_file)


class Cp2kRunner:
    """Cp2k runner."""

    def __init__(self, cfg: Cp2kConfig):
        self.cfg = cfg
        # Detection depends on mutable configuration and environment state.
        # Cache both successful and unsuccessful lookups against those inputs
        # so changing CP2K_DATA_DIR or the executable cannot reuse stale data.
        self._cached_data_dir: Optional[tuple[tuple[Any, ...], Optional[Path]]] = None

    def _validated_runtime_config(self) -> Cp2kConfig:
        """Rebuild a validated execution snapshot from a possibly mutated model."""

        if not isinstance(self.cfg, Cp2kConfig):
            raise TypeError("Cp2kRunner requires a Cp2kConfig instance")
        snapshot = self.cfg.model_copy(deep=True)
        snapshot.cp2k_cmd = validate_cp2k_command(snapshot.cp2k_cmd)
        snapshot.extra_args = validate_cp2k_extra_args(snapshot.extra_args)
        snapshot.exec_prefix = Cp2kConfig._exec_prefix_valid(snapshot.exec_prefix)
        snapshot.mpi_cmd = Cp2kConfig._mpi_cmd_strip_cp2k(snapshot.mpi_cmd)
        snapshot.nprocs = Cp2kConfig._nprocs_positive_cp2k(snapshot.nprocs)
        snapshot.omp_num_threads = Cp2kConfig._omp_threads_valid(
            snapshot.omp_num_threads
        )
        snapshot.timeout_sec = Cp2kConfig._cp2k_timeout_valid(snapshot.timeout_sec)
        snapshot.kill_grace_sec = Cp2kConfig._cp2k_kill_grace_valid(
            snapshot.kill_grace_sec
        )
        return snapshot

    def _cp2k_executable_tokens(self, cfg: Optional[Cp2kConfig] = None) -> list[str]:
        """Resolve the same CP2K command payload used for production runs."""

        import shutil

        effective = self.cfg if cfg is None else cfg
        command = validate_cp2k_command(effective.cp2k_cmd)
        if isinstance(command, list):
            return list(command)

        executable = str(command)
        if getattr(effective, "exec_prefix", None):
            # The prefix may select an environment in which the executable is
            # not visible to the current process.  Resolution must happen in
            # that selected environment.
            return [executable]

        resolved = shutil.which(executable)
        if resolved is None and executable == "cp2k":
            for candidate in ("cp2k.psmp", "cp2k.popt", "cp2k.sopt", "cp2k.ssmp"):
                resolved = shutil.which(candidate)
                if resolved is not None:
                    break
        if resolved is None:
            raise FileNotFoundError(
                f"CP2K executable not found: {executable!r}. Set cp2k.cp2k_cmd "
                "to the executable used for the run."
            )
        return [resolved]

    def query_version(self, workdir: Path) -> tuple[int, ...]:
        """Query and parse the configured CP2K executable before input rendering.

        The 2024 release changed SCF non-convergence control, so guessing a
        version is unsafe: an unparseable or unreachable executable is a hard
        pre-run error.  Probe on every stage entry rather than caching across
        stages: PATH, an environment selected by ``exec_prefix``, or the binary
        itself can change during a long workflow, and stale version policy is
        not an acceptable optimization.
        """

        import re

        runtime_cfg = self._validated_runtime_config()
        ensure_dir(workdir)
        prefix = [str(token) for token in (getattr(runtime_cfg, "exec_prefix", None) or [])]
        executable = self._cp2k_executable_tokens(runtime_cfg)
        payload = list(prefix)
        if runtime_cfg.mpi_cmd:
            payload += [str(runtime_cfg.mpi_cmd), "-np", "1"]
        payload += executable

        # CP2K releases in scope print strings such as ``CP2K version 2024.1``.
        # Require a dotted release token and reject conflicting banners rather
        # than silently taking the first match from a wrapper's combined
        # stdout/stderr.
        version_pattern = re.compile(
            r"\bCP2K\s+version\s+(\d+(?:\.\d+)+)(?![.\d])",
            re.IGNORECASE,
        )
        attempts: list[str] = []
        for flag in ("--version", "-v"):
            command = payload + [flag]
            try:
                rc, out, err = run_cmd(
                    command,
                    cwd=workdir,
                    check=False,
                    capture=True,
                    timeout=30.0,
                )
            except Exception as exc:
                attempts.append(f"{' '.join(command)}: {type(exc).__name__}: {exc}")
                continue
            text = (out or "") + "\n" + (err or "")
            matches = {
                tuple(int(component) for component in token.split("."))
                for token in version_pattern.findall(text)
            }
            if rc == 0 and len(matches) == 1:
                return next(iter(matches))
            reason = (
                "conflicting CP2K version strings"
                if len(matches) > 1
                else "no parseable CP2K version string"
            )
            attempts.append(
                f"{' '.join(command)}: returncode={rc}, {reason}, "
                f"output={_tail_lines(text, n=8)!r}"
            )

        detail = "; ".join(attempts)
        raise RuntimeError(
            "Could not query an unambiguous CP2K version from the configured "
            "executable before rendering input"
            + (f": {detail}" if detail else "")
        )

    def _detection_cache_key(self, cfg: Optional[Cp2kConfig] = None) -> tuple[Any, ...]:
        effective = self.cfg if cfg is None else cfg
        cfg_data_dir = getattr(effective, "data_dir", None)
        env_data_dir = os.environ.get("CP2K_DATA_DIR", "").strip()
        if isinstance(effective.cp2k_cmd, list):
            command = tuple(str(x) for x in effective.cp2k_cmd)
        else:
            command = (str(effective.cp2k_cmd),)
        prefix = tuple(str(x) for x in (getattr(effective, "exec_prefix", None) or []))
        mpi_command = str(getattr(effective, "mpi_cmd", "") or "")
        return (
            str(cfg_data_dir) if cfg_data_dir else "",
            env_data_dir,
            command,
            prefix,
            mpi_command,
        )

    def _detect_data_dir(
        self,
        workdir: Path,
        cfg: Optional[Cp2kConfig] = None,
    ) -> Optional[Path]:
        """Detect data dir."""

        effective = self.cfg if cfg is None else cfg
        key = self._detection_cache_key(effective)
        if self._cached_data_dir is not None and self._cached_data_dir[0] == key:
            cached = self._cached_data_dir[1]
            if cached is None or cached.is_dir():
                return cached
            self._cached_data_dir = None

        # override
        if getattr(effective, "data_dir", None):
            p = Path(str(effective.data_dir)).expanduser()
            if p.is_dir():
                self._cached_data_dir = (key, p)
                return p
            # An explicit data directory is authoritative.  If it disappears,
            # do not silently switch to CP2K_DATA_DIR or executable/package
            # discovery and consume different scientific inputs.
            self._cached_data_dir = (key, None)
            return None

        import os
        env_dd = os.environ.get("CP2K_DATA_DIR", "").strip()
        if env_dd:
            p = Path(env_dd).expanduser()
            if p.is_dir():
                self._cached_data_dir = (key, p)
                return p

        # lightweight command version
        import re

        def _cp2k_exe_token() -> str:
            if isinstance(effective.cp2k_cmd, list):
                if len(effective.cp2k_cmd) == 0:
                    return "cp2k.psmp"
                return str(effective.cp2k_cmd[0])
            return str(effective.cp2k_cmd)

        prefix: list[str] = []
        if getattr(effective, "exec_prefix", None):
            prefix += list(effective.exec_prefix)

        # execution mpi binaries
        cand_cmds: list[list[str]] = [prefix + [_cp2k_exe_token(), "-v"]]
        if effective.mpi_cmd:
            cand_cmds.append(prefix + [effective.mpi_cmd, "-np", "1", _cp2k_exe_token(), "-v"])

        for cmd in cand_cmds:
            try:
                rc, out, err = run_cmd(cmd, cwd=workdir, check=False, capture=True, timeout=30.0)
            except Exception:
                continue
            if rc != 0:
                continue
            txt = (out or "") + "\n" + (err or "")
            m = re.search(r"__DATA_DIR=\"([^\"]+)\"", txt)
            if m:
                p = Path(m.group(1)).expanduser()
                if p.is_dir():
                    self._cached_data_dir = (key, p)
                    return p
            # fallback
            m2 = re.search(r"Data directory path\s+([^\s]+)", txt)
            if m2:
                p = Path(m2.group(1)).expanduser()
                if p.is_dir():
                    self._cached_data_dir = (key, p)
                    return p

        # reach here existed
        if env_dd:
            p = Path(env_dd).expanduser()
            if p.is_dir():
                self._cached_data_dir = (key, p)
                return p
        self._cached_data_dir = (key, None)
        return None

    @staticmethod
    def _packaged_data_dir() -> Optional[Path]:
        """Return the installed VitriFlow CP2K data directory, if present."""

        try:
            path = Path(__file__).resolve().parent / "data" / "cp2k"
            return path if path.is_dir() else None
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _is_path_qualified(filename: str) -> bool:
        return os.path.isabs(filename) or ("/" in filename) or ("\\" in filename)

    def _resolve_data_file_source(
        self,
        filename: str,
        workdir: Path,
        *,
        cfg: Optional[Cp2kConfig] = None,
    ) -> Optional[Path]:
        """Resolve the exact CP2K data file that a run will consume.

        Explicit file paths are used verbatim.  For ordinary CP2K filenames an
        explicitly configured ``cp2k.data_dir`` is authoritative, followed by
        the environment/executable data directory and finally VitriFlow's
        packaged fallback.  In particular, packaged data must never mask a
        user-configured file with the same basename.
        """

        name = str(filename).strip()
        if not name:
            return None
        if self._is_path_qualified(name):
            path = Path(name).expanduser()
            if not path.is_absolute():
                raise ValueError(
                    "Path-qualified CP2K data filenames must be absolute; "
                    f"got {name!r}. Load relative paths through RunConfig.from_yaml "
                    "or supply an absolute path."
                )
            return path.resolve(strict=False) if path.is_file() else None

        effective = self.cfg if cfg is None else cfg
        configured = getattr(effective, "data_dir", None)
        if configured:
            configured_dir = Path(str(configured)).expanduser()
            candidate = configured_dir / name
            # An explicit directory is a scientific input, not a hint.  Do not
            # silently replace a missing configured file with bundled data.
            return candidate.resolve(strict=False) if candidate.is_file() else None

        detected = self._detect_data_dir(workdir, effective)
        if detected is not None:
            candidate = detected / name
            if candidate.is_file():
                return candidate.resolve(strict=False)

        packaged = self._packaged_data_dir()
        if packaged is not None:
            candidate = packaged / name
            if candidate.is_file():
                return candidate.resolve(strict=False)
        return None

    def resolved_data_files(
        self,
        workdir: Path,
        *,
        require: bool = True,
        _cfg: Optional[Cp2kConfig] = None,
    ) -> dict[str, dict[str, str]]:
        """Return exact BASIS/POTENTIAL sources used by this runner.

        The result is intentionally JSON-friendly so workflow manifests can
        bind execution and cache reuse to the resolved file contents.
        """

        effective = self._validated_runtime_config() if _cfg is None else _cfg
        specs = {
            "basis_set": str(effective.basis_set_file_name),
            "potential": str(effective.potential_file_name),
        }
        resolved: dict[str, dict[str, str]] = {}
        for role, configured_name in specs.items():
            source = self._resolve_data_file_source(
                configured_name,
                workdir,
                cfg=effective,
            )
            if source is None:
                if require:
                    data_dir = getattr(effective, "data_dir", None)
                    location = f" configured cp2k.data_dir={data_dir!r}" if data_dir else " available CP2K/package data"
                    raise FileNotFoundError(
                        f"Could not resolve CP2K {role} file {configured_name!r} from{location}"
                    )
                continue
            resolved[role] = {
                "configured_name": configured_name,
                "resolved_path": str(source),
            }
        return resolved

    def _ensure_data_files_present(
        self,
        workdir: Path,
        *,
        _cfg: Optional[Cp2kConfig] = None,
    ) -> Optional[Path]:
        """Stage and verify the exact configured CP2K data files.

        Existing files in a reused work directory are never accepted merely by
        basename.  The authoritative sources are resolved first, each bare-name
        destination is refreshed, and its content is verified before CP2K can
        start.
        """

        import shutil

        effective = self._validated_runtime_config() if _cfg is None else _cfg

        ensure_dir(workdir)

        def _digest(path: Path) -> str:
            value = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(chunk)
            return value.hexdigest()

        def _clear_stale_destinations() -> None:
            for configured_name in (
                str(effective.basis_set_file_name),
                str(effective.potential_file_name),
            ):
                if self._is_path_qualified(configured_name):
                    continue
                destination = workdir / configured_name
                if not (destination.exists() or destination.is_symlink()):
                    continue
                configured_dir = getattr(effective, "data_dir", None)
                if configured_dir is not None:
                    candidate = Path(str(configured_dir)).expanduser() / configured_name
                    try:
                        if candidate.resolve(strict=False) == destination.resolve(strict=False):
                            continue
                    except OSError:
                        pass
                destination.unlink()

        try:
            resolved = self.resolved_data_files(workdir, require=True, _cfg=effective)
        except (FileNotFoundError, OSError):
            _clear_stale_destinations()
            raise

        def _stage_file(src: Path, dst: Path) -> None:
            """Stage file."""
            if not src.is_file():
                raise FileNotFoundError(f"Resolved CP2K data source is not a file: {src}")

            # data_dir may intentionally be the stage directory. In that
            # case source and destination are the same regular file; never
            # unlink the scientific input in an attempt to stage it onto
            # itself.
            try:
                if src.resolve(strict=False) == dst.resolve(strict=False):
                    return
            except OSError:
                pass

            if dst.exists() or dst.is_symlink():
                dst.unlink()

            try:
                dst.symlink_to(src)
            except OSError:
                try:
                    shutil.copy2(str(src), str(dst))
                except OSError as exc:
                    raise OSError(f"Could not stage CP2K data file {src} at {dst}") from exc

            if not dst.is_file() or dst.stat().st_size != src.stat().st_size or _digest(dst) != _digest(src):
                try:
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                finally:
                    raise OSError(f"Staged CP2K data file does not match authoritative source: {src} -> {dst}")

        dd = self._detect_data_dir(workdir, effective)
        for role, details in resolved.items():
            fname = str(details["configured_name"])
            # Explicit paths are passed directly to CP2K and are not staged.
            if self._is_path_qualified(fname):
                continue
            dst = workdir / fname
            source = Path(details["resolved_path"])
            try:
                _stage_file(source, dst)
            except (FileNotFoundError, OSError) as exc:
                raise type(exc)(f"Failed to stage CP2K {role} file {fname!r}: {exc}") from exc

        return dd

    def run(self, input_file: Path, workdir: Path, output_name: str = "cp2k.out", *, timeout_sec: Optional[float] = None) -> RunResult:
        runtime_cfg = self._validated_runtime_config()
        if timeout_sec is not None:
            if isinstance(timeout_sec, bool):
                raise ValueError("CP2K timeout_sec override must be numeric")
            timeout_value = float(timeout_sec)
            if not math.isfinite(timeout_value) or timeout_value <= 0.0:
                raise ValueError("CP2K timeout_sec override must be finite and > 0")
        else:
            timeout_value = None
        workdir = Path(workdir)
        ensure_dir(workdir)

        # Revalidate at the execution boundary because the configuration model
        # is mutable and may also have been constructed without validation.
        validate_cp2k_command(runtime_cfg.cp2k_cmd)
        extra_args = validate_cp2k_extra_args(runtime_cfg.extra_args)
        input_path = _validated_cp2k_input_file(Path(input_file), workdir=workdir)
        output_basename = _validated_cp2k_artifact_name(
            output_name, role="output filename"
        )
        if output_basename in _CP2K_RUNNER_FIXED_NAMES:
            raise ValueError(
                f"CP2K output filename {output_basename!r} collides with a "
                "runner-owned diagnostic"
            )
        if output_basename == input_path.name:
            raise ValueError("CP2K input and output filenames must be distinct")

        # Bare BASIS/POTENTIAL names are materialized into this same directory.
        # Refuse a public runner invocation that would overwrite or reinterpret
        # one of them as the calculation input/output or a diagnostic.
        local_data_names: list[str] = []
        for role, configured in (
            ("basis-set", runtime_cfg.basis_set_file_name),
            ("potential", runtime_cfg.potential_file_name),
        ):
            raw = str(configured)
            if Path(raw).is_absolute() or "/" in raw or "\\" in raw:
                continue
            name = _validated_cp2k_artifact_name(raw, role=f"{role} data filename")
            local_data_names.append(name)
        owned = [input_path.name, output_basename, *_CP2K_RUNNER_FIXED_NAMES]
        collisions = sorted(set(local_data_names).intersection(owned))
        if collisions or len(set(local_data_names)) != len(local_data_names):
            raise ValueError(
                "CP2K input/output, diagnostics, and localized BASIS/POTENTIAL "
                f"filenames must be disjoint; collision(s): {collisions or local_data_names}"
            )

        out_file = workdir / output_basename
        screen_file = workdir / "screen.out"
        stdout_file = workdir / "stdout.txt"
        stderr_file = workdir / "stderr.txt"

        # A zero-return utility invocation or interrupted prior attempt must
        # never leave an old CP2K output looking current.  Unlink the complete
        # runner-owned output namespace, including link directory entries,
        # before staging data or starting the executable.
        for artifact in (out_file, screen_file, stdout_file, stderr_file):
            _unlink_cp2k_attempt_artifact(artifact)

        cmd: list[str] = []

        # basis potential present
        # detected relying shell
        import os
        dd = self._ensure_data_files_present(workdir, _cfg=runtime_cfg)
        env = os.environ.copy()
        if dd is not None:
            env["CP2K_DATA_DIR"] = str(dd)
        # mp oversubscription mpi
        if getattr(runtime_cfg, "omp_num_threads", None) is not None:
            env["OMP_NUM_THREADS"] = str(int(runtime_cfg.omp_num_threads))
        else:
            if runtime_cfg.mpi_cmd and int(runtime_cfg.nprocs) > 1:
                env.setdefault("OMP_NUM_THREADS", "1")

        # execution prefix env
        # intentionally resolving executables
        # environment resolution prefixed
        # execution context
        if getattr(runtime_cfg, "exec_prefix", None):
            cmd += list(runtime_cfg.exec_prefix)
        if runtime_cfg.mpi_cmd:
            cmd += [runtime_cfg.mpi_cmd, "-np", str(runtime_cfg.nprocs)]

        cmd += self._cp2k_executable_tokens(runtime_cfg)

        # cp2 input output
        cmd += ["-i", str(input_path.name), "-o", str(out_file.name)]
        cmd += extra_args

        timeout_use = timeout_value if timeout_value is not None else runtime_cfg.timeout_sec
        rc, out, err = run_cmd(
            cmd,
            cwd=workdir,
            env=env,
            check=False,
            capture=True,
            timeout=timeout_use,
            kill_grace_sec=float(runtime_cfg.kill_grace_sec),
        )
        # debug
        _atomic_write_cp2k_runner_text(stdout_file, out)
        _atomic_write_cp2k_runner_text(stderr_file, err)
        _atomic_write_cp2k_runner_text(screen_file, out)

        if rc != 0:
            ctx = CommandFailureContext(
                screen_tail=_tail_file(screen_file),
                log_tail=_tail_file(out_file),
                stdout_tail=_tail_lines(out, n=80),
                stderr_tail=_tail_lines(err, n=80),
            )
            raise ExternalCommandError(cmd, rc, out, err, context=ctx)

        try:
            output_info = out_file.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"CP2K returned success without creating current output {out_file}"
            ) from exc
        if (
            stat.S_ISLNK(output_info.st_mode)
            or not stat.S_ISREG(output_info.st_mode)
            or int(output_info.st_nlink) != 1
            or int(output_info.st_size) < 1
        ):
            raise RuntimeError(
                "CP2K output must be a non-empty, direct, single-link regular "
                f"file from the current invocation: {out_file}"
            )
        return RunResult(cmd=cmd, returncode=rc, stdout=out, stderr=err, log_file=out_file)
