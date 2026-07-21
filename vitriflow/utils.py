from __future__ import annotations

import hashlib
import os
import shlex
import stat as stat_module
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


def _tail_lines(text: str, n: int = 50, *, max_chars: int = 8000) -> str:
    """Tail lines."""
    try:
        lines = str(text).splitlines()
    except Exception:
        return ""
    tail = "\n".join(lines[-int(n) :])
    if len(tail) > int(max_chars):
        tail = tail[-int(max_chars) :]
    return tail


@dataclass(frozen=True)
class CommandFailureContext:
    """Command failure context."""

    screen_tail: str = ""
    log_tail: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


class ExternalCommandError(RuntimeError):
    def __init__(
        self,
        cmd: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        *,
        context: Optional[CommandFailureContext] = None,
    ):
        self.cmd = list(cmd)
        self.returncode = int(returncode)
        self.stdout = str(stdout)
        self.stderr = str(stderr)
        self.context = context

        msg = f"Command failed (code {returncode}): {' '.join(shlex.quote(c) for c in cmd)}"
        if context is not None:
            blocks: list[str] = []
            if context.screen_tail.strip():
                blocks.append("--- screen.out (tail) ---\n" + context.screen_tail.rstrip())
            if context.log_tail.strip():
                blocks.append("--- log.lammps (tail) ---\n" + context.log_tail.rstrip())
            if context.stderr_tail.strip():
                blocks.append("--- stderr (tail) ---\n" + context.stderr_tail.rstrip())
            if context.stdout_tail.strip():
                blocks.append("--- stdout (tail) ---\n" + context.stdout_tail.rstrip())
            if blocks:
                msg += "\n" + "\n".join(blocks)
        super().__init__(msg)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def scale_steps_for_timestep(steps_at_ref_dt: int, dt_ref: float, dt_new: float, *, min_steps: int = 1) -> int:
    """Scale steps for."""
    import math

    s0 = int(steps_at_ref_dt)
    if s0 <= 0:
        return int(min_steps)
    dt0 = float(dt_ref)
    dt1 = float(dt_new)
    if not (math.isfinite(dt0) and dt0 > 0.0 and math.isfinite(dt1) and dt1 > 0.0):
        # inputs invalid callers
        # guard
        return max(int(min_steps), s0)

    t = s0 * dt0
    s1 = int(math.ceil(t / dt1))
    return max(int(min_steps), s1)


def stable_file_identity(
    path: Path,
    *,
    reject_final_symlink: bool = False,
) -> dict[str, object]:
    """Hash one stable regular-file inode and verify its path still names it.

    A plain ``stat(); open(); hash()`` sequence can combine metadata from one
    file with bytes from another if a producer replaces the path concurrently.
    This helper resolves the configured path once, opens the canonical final
    component with ``O_NOFOLLOW`` where available, hashes that open inode, and
    compares device, inode, type, size, mtime and ctime before and after the
    read.  It then proves that both the canonical name and the original path
    still resolve to the same inode.  Directory symlink aliases remain usable;
    callers protecting result artifacts may additionally reject a symlink in
    the final configured component.
    """

    configured = Path(path).expanduser()
    try:
        if reject_final_symlink and configured.is_symlink():
            raise RuntimeError(f"Protected file must not be a symbolic link: {configured}")
        resolved = configured.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Required file is missing or cannot be resolved: {configured}") from exc

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= int(getattr(os, "O_BINARY"))
    if hasattr(os, "O_CLOEXEC"):
        flags |= int(getattr(os, "O_CLOEXEC"))
    if hasattr(os, "O_NOFOLLOW"):
        flags |= int(getattr(os, "O_NOFOLLOW"))

    fd = os.open(str(resolved), flags)
    try:
        before = os.fstat(fd)
        if not stat_module.S_ISREG(before.st_mode):
            raise RuntimeError(f"Required path is not a regular file: {configured}")
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)

    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise RuntimeError(f"Required file changed while hashing: {configured}")

    try:
        named = resolved.stat()
        resolved_again = configured.resolve(strict=True)
        original_named = resolved_again.stat()
    except OSError as exc:
        raise RuntimeError(f"Required file path changed while hashing: {configured}") from exc
    if resolved_again != resolved or any(
        getattr(after, name) != getattr(named, name)
        or getattr(after, name) != getattr(original_named, name)
        for name in fields
    ):
        raise RuntimeError(f"Required file path was replaced while hashing: {configured}")

    return {
        "resolved_path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
        # Process-local stability evidence.  Callers deliberately do not need
        # to persist these filesystem-specific values in portable manifests.
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
    }


def sha256sum(path: Path) -> str:
    return str(stable_file_identity(path)["sha256"])


def quarantine_uncommitted_box_directories(
    production_dir: Path,
    *,
    committed_box_ids: Sequence[int],
    quarantine_root: Path,
) -> list[Path]:
    """Atomically move orphan ``box_NNN`` trees out of a resumed ensemble.

    A checkpoint commits only accepted/rejected box identifiers.  A process
    killed during the next box can leave engine outputs that must not be
    interpreted as part of that checkpoint or reused by the rerun.  This
    helper preserves such trees for diagnosis under a non-discoverable
    quarantine root and returns their new paths.
    """

    production = Path(production_dir).expanduser()
    if not production.exists():
        return []
    if production.is_symlink() or not production.is_dir():
        raise RuntimeError(
            f"Production directory must be a real directory before resume: {production}"
        )
    committed = {int(value) for value in committed_box_ids}
    if any(value < 0 for value in committed):
        raise ValueError("Committed production box ids must be non-negative")

    production = production.resolve(strict=True)
    calculation_root = production.parent
    quarantine = Path(quarantine_root).expanduser()
    if not quarantine.is_absolute():
        quarantine = calculation_root / quarantine
    raw_cursor = quarantine
    while True:
        try:
            if raw_cursor.resolve(strict=False) == calculation_root:
                break
        except (OSError, RuntimeError):
            pass
        if raw_cursor.is_symlink():
            raise RuntimeError(
                "Interrupted-attempt quarantine must not contain symbolic links: "
                f"{raw_cursor}"
            )
        if raw_cursor.parent == raw_cursor:
            break
        raw_cursor = raw_cursor.parent
    quarantine_resolved = quarantine.resolve(strict=False)
    try:
        quarantine_resolved.relative_to(calculation_root)
    except ValueError as exc:
        raise RuntimeError(
            "Interrupted-attempt quarantine must remain inside the calculation root: "
            f"{quarantine}"
        ) from exc
    cursor = calculation_root
    for part in quarantine_resolved.relative_to(calculation_root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(
                f"Interrupted-attempt quarantine must not contain symbolic links: {cursor}"
            )
    quarantine = quarantine_resolved
    quarantine.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    import re

    for candidate in sorted(production.iterdir(), key=lambda item: item.name):
        match = re.fullmatch(r"box_(\d+)", candidate.name)
        if match is None or int(match.group(1)) in committed:
            continue
        index = 1
        while True:
            destination = quarantine / f"{candidate.name}.interrupted-{index:03d}"
            if not destination.exists() and not destination.is_symlink():
                break
            index += 1
        os.replace(candidate, destination)
        moved.append(destination)
    return moved


def run_cmd(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture: bool = True,
    timeout: Optional[float] = None,
    kill_grace_sec: float = 5.0,
) -> tuple[int, str, str]:
    """Cmd."""
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.PIPE if capture else None

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,  # os killpg posix
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timeout_msg = f"TIMEOUT after {timeout} s"
        # session mpi hangs
        try:
            if os.name != "nt":
                import signal
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass
        try:
            out, err = proc.communicate(timeout=kill_grace_sec)
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt":
                    import signal
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
            out, err = proc.communicate()

        out = out or ""
        err = (err or "")
        if err:
            err += "\n"
        err += timeout_msg

        rc = proc.returncode if proc.returncode is not None else -9
        if check:
            raise ExternalCommandError(cmd, rc, out, err)
        return rc, out, err

    rc = proc.returncode if proc.returncode is not None else 0
    out = out or ""
    err = err or ""
    if check and rc != 0:
        raise ExternalCommandError(cmd, rc, out, err)
    return rc, out, err


def which(cmd: str) -> Optional[str]:
    import shutil
    return shutil.which(cmd)  # type: ignore[name-defined]
