from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from ..config import MDConfig
from ..runner import LammpsRunner
from ..utils import ExternalCommandError


_PPPM_OOR_RE = re.compile(r"out\s+of\s+range\s+atoms.*pppm", re.IGNORECASE)
_PPPM_GENERIC_RE = re.compile(r"out\s+of\s+range\s+atoms", re.IGNORECASE)


def _tail_file(path: Path, *, max_bytes: int = 200_000) -> str:
    """Tail file."""
    try:
        if not path.exists():
            return ""
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(-max_bytes, os.SEEK_END)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def script_uses_pppm(script: str) -> bool:
    s = str(script).lower()
    # conservative trigger kspace
    return ("kspace_style" in s) and ("pppm" in s)


def is_pppm_out_of_range(err: ExternalCommandError, workdir: Path) -> bool:
    """Is pppm out."""
    haystacks: list[str] = []
    ctx = getattr(err, "context", None)
    if ctx is not None:
        for k in ("screen_tail", "log_tail", "stdout_tail", "stderr_tail"):
            try:
                v = str(getattr(ctx, k, "") or "")
            except Exception:
                v = ""
            if v:
                haystacks.append(v)

    # inspect persisted runner
    for fn in ("screen.out", "log.lammps", "stdout.txt", "stderr.txt"):
        haystacks.append(_tail_file(workdir / fn))

    for h in haystacks:
        if not h:
            continue
        if _PPPM_OOR_RE.search(h):
            return True

    # fallback
    # signature mentioned somewhere
    merged = "\n".join(haystacks)
    if _PPPM_GENERIC_RE.search(merged) and ("pppm" in merged.lower()):
        return True

    return False


def _archive_file(path: Path, suffix: str) -> None:
    try:
        if not path.exists():
            return
        dst = path.with_name(path.name + suffix)
        # clobbering repeated
        if dst.exists():
            dst.unlink()
        path.rename(dst)
    except Exception:
        return


def _cleanup(paths: Sequence[Path]) -> None:
    for p in paths:
        path = Path(p)
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(
                f"Cannot inspect stale LAMMPS stage artifact before execution: {path}"
            ) from exc
        try:
            # unlink() removes symbolic and hardlink directory entries without
            # following them.  Directories and permission failures are hard
            # errors because a stale artifact must never survive into a new
            # provenance attempt.
            path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot remove stale LAMMPS stage artifact before execution: {path}"
            ) from exc


def run_with_neighbor_skin_autotune(
    runner: LammpsRunner,
    script_builder: Callable[[MDConfig], str],
    workdir: Path,
    md_cfg: MDConfig,
    *,
    log_name: str = "log.lammps",
    timeout_sec: Optional[float] = None,
    cleanup_paths: Optional[Sequence[Path]] = None,
) -> Tuple[float, int]:
    """With neighbor skin."""

    cleanup = list(cleanup_paths or [])

    # once decide pppm
    script0 = script_builder(md_cfg)
    if not md_cfg.neighbor_skin_autotune or not script_uses_pppm(script0):
        _cleanup(cleanup)
        runner.run(script0, workdir=workdir, log_name=log_name, timeout_sec=timeout_sec)
        return float(getattr(md_cfg, "neighbor_skin", 0.0)), 0

    skin = float(md_cfg.neighbor_skin)
    step = float(md_cfg.neighbor_skin_step)
    skin_max = float(md_cfg.neighbor_skin_max)

    n_retry = 0
    while True:
        md_try = md_cfg.model_copy(update={"neighbor_skin": float(skin)})
        script = script_builder(md_try)
        try:
            _cleanup(cleanup)
            runner.run(script, workdir=workdir, log_name=log_name, timeout_sec=timeout_sec)
            # propagate skin subsequent
            # reuse stable
            try:
                md_cfg.neighbor_skin = float(skin)
            except Exception:
                pass
            return float(skin), int(n_retry)
        except ExternalCommandError as e:
            if not is_pppm_out_of_range(e, workdir):
                raise
            nxt = float(skin) + float(step)
            if nxt > skin_max + 1.0e-12:
                raise

            # debug
            tag = f".failed_skin{skin:.3f}".rstrip("0").rstrip(".")
            _archive_file(workdir / log_name, tag)
            _archive_file(workdir / "screen.out", tag)
            _archive_file(workdir / "stdout.txt", tag)
            _archive_file(workdir / "stderr.txt", tag)

            skin = nxt
            n_retry += 1
