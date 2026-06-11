from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
