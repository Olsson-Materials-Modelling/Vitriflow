from __future__ import annotations

"""Crash-safe advisory locks for mutable workflow output trees.

The checkpoint files are atomically replaced, but atomic replacement alone
does not serialize two orchestrators that both read the same old checkpoint.
These locks prevent concurrent ``run``/``autotune`` controllers and duplicate
external box tasks from writing the same stage tree at the same time.

Lock files live inside the protected directory.  This is important on HPC
filesystems where a job can own its calculation directory while the shared
parent is deliberately read-only.  Public clean-start admission ignores only
this exact internal control file.  The inode is retained after use: unlinking
a locked file permits a third process to lock a new inode while the first
still owns the old one.  ``flock`` ownership itself is released by the kernel
on normal exit, exception, signal, or process death.
"""

import json
import os
import stat
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar, cast


_F = TypeVar("_F", bound=Callable[..., Any])
WORKFLOW_LOCK_FILENAME = ".vitriflow.lock"


def _lock_path_for(target: Path) -> Path:
    resolved = Path(target).expanduser().resolve(strict=False)
    return resolved / WORKFLOW_LOCK_FILENAME


def workflow_payload_entries(target: Path) -> tuple[Path, ...]:
    """Return user/scientific entries, excluding only our internal lock.

    The lock is a control inode rather than calculation state.  Centralising
    this exact-name exclusion prevents the clean-start resolvers from growing
    broad hidden-file exceptions which could conceal stale scientific output.
    """

    return tuple(
        path
        for path in Path(target).iterdir()
        if path.name != WORKFLOW_LOCK_FILENAME
    )


@contextmanager
def exclusive_workflow_lock(target: Path, *, purpose: str) -> Iterator[Path]:
    """Hold a non-blocking exclusive lock associated with ``target``.

    A failure to provide a real advisory lock is fail-closed.  Waiting is not
    used because a duplicate Slurm task or controller can legitimately run for
    days; reporting the collision immediately is safer than silently queuing a
    second scientific calculation.
    """

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - supported HPC targets are POSIX
        raise RuntimeError(
            "Safe workflow locking is unavailable on this platform; refusing "
            f"to start {purpose!r}"
        ) from exc

    target_path = Path(target).expanduser().resolve(strict=False)
    target_path.mkdir(parents=True, exist_ok=True)
    if not target_path.is_dir():
        raise RuntimeError(
            f"Workflow output target is not a directory for {purpose!r}: {target_path}"
        )
    lock_path = _lock_path_for(target_path)
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot open trusted workflow lock for {purpose!r}: {lock_path}: {exc}"
        ) from exc

    acquired = False
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise RuntimeError(
                f"Workflow lock path is not a regular file: {lock_path}"
            )
        # O_NOFOLLOW closes the symbolic-link path, but a pre-existing hard
        # link would still make the metadata truncate/write below modify an
        # unrelated inode.  A workflow lock has no legitimate reason to have
        # more than one directory entry, so reject it before acquiring or
        # modifying the file.  Recheck after flock to close a concurrent-link
        # race before the first destructive operation.
        if int(lock_stat.st_nlink) != 1:
            raise RuntimeError(
                "Workflow lock path must have exactly one hard link; refusing "
                f"to modify an aliased inode: {lock_path}"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Cannot start {purpose!r}: another VitriFlow process holds "
                f"the output lock {lock_path}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Cannot acquire a reliable workflow lock for {purpose!r}: "
                f"{lock_path}: {exc}"
            ) from exc

        locked_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(locked_stat.st_mode)
            or int(locked_stat.st_nlink) != 1
            or int(locked_stat.st_dev) != int(lock_stat.st_dev)
            or int(locked_stat.st_ino) != int(lock_stat.st_ino)
        ):
            raise RuntimeError(
                "Workflow lock inode changed or acquired a hard-link alias "
                f"while being secured: {lock_path}"
            )

        metadata = json.dumps(
            {
                "pid": int(os.getpid()),
                "purpose": str(purpose),
                "target": str(target_path),
                "acquired_unix_time": float(time.time()),
            },
            sort_keys=True,
        ) + "\n"
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, metadata.encode("utf-8"))
        os.fsync(fd)
        yield lock_path
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def locked_output_workflow(purpose: str) -> Callable[[_F], _F]:
    """Decorate a public ``(config, outdir, ...)`` workflow entry point."""

    def decorate(function: _F) -> _F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if "outdir" in kwargs:
                output = kwargs["outdir"]
            elif len(args) >= 2:
                output = args[1]
            else:  # pragma: no cover - Python's signature error normally wins
                raise TypeError(f"{function.__name__} requires an outdir argument")
            with exclusive_workflow_lock(Path(output), purpose=str(purpose)):
                return function(*args, **kwargs)

        return cast(_F, wrapped)

    return decorate


__all__ = [
    "WORKFLOW_LOCK_FILENAME",
    "exclusive_workflow_lock",
    "locked_output_workflow",
    "workflow_payload_entries",
]
