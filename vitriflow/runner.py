from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from .config import LammpsConfig, Cp2kConfig
from .utils import ensure_dir, run_cmd, ExternalCommandError, CommandFailureContext, _tail_lines


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


class LammpsRunner:
    def __init__(self, cfg: LammpsConfig):
        self.cfg = cfg

    def run(self, input_script: str, workdir: Path, log_name: str, *, timeout_sec: Optional[float] = None) -> RunResult:
        ensure_dir(workdir)
        in_path = workdir / "in.lammps"
        in_path.write_text(input_script)

        log_file = workdir / log_name
        screen_file = workdir / "screen.out"

        cmd = []
        if self.cfg.mpi_cmd:
            cmd += [self.cfg.mpi_cmd, "-np", str(self.cfg.nprocs)]
        # lammps string tokenized
        if isinstance(self.cfg.lammps_cmd, list):
            cmd += list(self.cfg.lammps_cmd)
        else:
            cmd += [str(self.cfg.lammps_cmd)]
        cmd += ["-in", str(in_path.name), "-log", str(log_file.name)]
        # debug
        if "-screen" not in self.cfg.extra_args:
            cmd += ["-screen", str(screen_file.name)]
        cmd += self.cfg.extra_args

        timeout_use = timeout_sec if timeout_sec is not None else self.cfg.timeout_sec
        rc, out, err = run_cmd(
            cmd,
            cwd=workdir,
            check=False,
            capture=True,
            timeout=timeout_use,
            kill_grace_sec=float(self.cfg.kill_grace_sec),
        )
        # captured regardless mpi
        # suppress screen canonical
        (workdir / "stdout.txt").write_text(out)
        (workdir / "stderr.txt").write_text(err)
        if rc != 0:
            # include screen exception
            ctx = CommandFailureContext(
                screen_tail=_tail_file(screen_file),
                log_tail=_tail_file(log_file),
                stdout_tail=_tail_lines(out, n=80),
                stderr_tail=_tail_lines(err, n=80),
            )
            raise ExternalCommandError(cmd, rc, out, err, context=ctx)
        return RunResult(cmd=cmd, returncode=rc, stdout=out, stderr=err, log_file=log_file)


class Cp2kRunner:
    """Cp2k runner."""

    def __init__(self, cfg: Cp2kConfig):
        self.cfg = cfg
        self._cached_data_dir: Optional[Path] = None

    def _detect_data_dir(self, workdir: Path) -> Optional[Path]:
        """Detect data dir."""

        if self._cached_data_dir is not None:
            return self._cached_data_dir

        # override
        if getattr(self.cfg, "data_dir", None):
            p = Path(str(self.cfg.data_dir)).expanduser()
            if p.is_dir():
                self._cached_data_dir = p
                return p

        import os
        env_dd = os.environ.get("CP2K_DATA_DIR", "").strip()
        if env_dd:
            p = Path(env_dd).expanduser()
            if p.is_dir():
                # cache infer executable
                # override stale env
                pass

        # lightweight command version
        import re

        def _cp2k_exe_token() -> str:
            if isinstance(self.cfg.cp2k_cmd, list):
                if len(self.cfg.cp2k_cmd) == 0:
                    return "cp2k.psmp"
                return str(self.cfg.cp2k_cmd[0])
            return str(self.cfg.cp2k_cmd)

        prefix: list[str] = []
        if getattr(self.cfg, "exec_prefix", None):
            prefix += list(self.cfg.exec_prefix)

        # execution mpi binaries
        cand_cmds: list[list[str]] = [prefix + [_cp2k_exe_token(), "-v"]]
        if self.cfg.mpi_cmd:
            cand_cmds.append(prefix + [self.cfg.mpi_cmd, "-np", "1", _cp2k_exe_token(), "-v"])

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
                    self._cached_data_dir = p
                    return p
            # fallback
            m2 = re.search(r"Data directory path\s+([^\s]+)", txt)
            if m2:
                p = Path(m2.group(1)).expanduser()
                if p.is_dir():
                    self._cached_data_dir = p
                    return p

        # reach here existed
        if env_dd:
            p = Path(env_dd).expanduser()
            if p.is_dir():
                self._cached_data_dir = p
                return p
        return None

    def _ensure_data_files_present(self, workdir: Path) -> Optional[Path]:
        """Data files present."""

        import os
        import shutil

        def _packaged_cp2k_dir() -> Optional[Path]:
            try:
                p = (Path(__file__).resolve().parent / "data" / "cp2k")
                if p.is_dir():
                    return p
            except Exception:
                return None
            return None

        def _stage_file(src: Path, dst: Path) -> bool:
            """Stage file."""
            if not src.exists():
                return False

            try:
                if dst.exists() or dst.is_symlink():
                    try:
                        if dst.is_symlink() and dst.resolve() == src.resolve():
                            return True
                    except Exception:
                        pass
                    try:
                        dst.unlink()
                    except Exception:
                        return False
            except Exception:
                pass

            try:
                dst.symlink_to(src)
                return dst.exists() or dst.is_symlink()
            except Exception:
                try:
                    shutil.copy2(str(src), str(dst))
                    return dst.exists()
                except Exception:
                    return False

        pkg_dd = _packaged_cp2k_dir()
        dd = self._detect_data_dir(workdir)

        for fname in (str(self.cfg.basis_set_file_name), str(self.cfg.potential_file_name)):
            # already alter
            if os.path.isabs(fname) or ("/" in fname) or ("\\" in fname):
                continue
            dst = workdir / fname

            # prefer packaged reference
            if pkg_dd is not None:
                if _stage_file(pkg_dd / fname, dst):
                    continue

            # fall installation directory
            if dd is not None:
                _stage_file(dd / fname, dst)

        return dd

    def run(self, input_file: Path, workdir: Path, output_name: str = "cp2k.out", *, timeout_sec: Optional[float] = None) -> RunResult:
        ensure_dir(workdir)
        if not input_file.exists():
            raise FileNotFoundError(str(input_file))

        out_file = workdir / output_name
        screen_file = workdir / "screen.out"

        import shutil

        cmd: list[str] = []

        # basis potential present
        # detected relying shell
        import os
        dd = self._ensure_data_files_present(workdir)
        env = os.environ.copy()
        if dd is not None:
            env["CP2K_DATA_DIR"] = str(dd)
        # mp oversubscription mpi
        if getattr(self.cfg, "omp_num_threads", None) is not None:
            env["OMP_NUM_THREADS"] = str(int(self.cfg.omp_num_threads))
        else:
            if self.cfg.mpi_cmd and int(self.cfg.nprocs) > 1:
                env.setdefault("OMP_NUM_THREADS", "1")

        # execution prefix env
        # intentionally resolving executables
        # environment resolution prefixed
        # execution context
        if getattr(self.cfg, "exec_prefix", None):
            cmd += list(self.cfg.exec_prefix)
        if self.cfg.mpi_cmd:
            cmd += [self.cfg.mpi_cmd, "-np", str(self.cfg.nprocs)]

        if isinstance(self.cfg.cp2k_cmd, list):
            # command list cp2k
            cmd += list(self.cfg.cp2k_cmd)
        else:
            exe = str(self.cfg.cp2k_cmd)

            if getattr(self.cfg, "exec_prefix", None):
                # resolution prefixed environment
                cmd += [exe]
            else:
                exe_path = shutil.which(exe)
                # conda variants cp2k
                if exe_path is None and exe == "cp2k":
                    for cand in ("cp2k.psmp", "cp2k.popt", "cp2k.sopt", "cp2k.ssmp"):
                        exe_path = shutil.which(cand)
                        if exe_path is not None:
                            break
                if exe_path is None:
                    raise FileNotFoundError(
                        f"CP2K executable not found: {exe!r}. Set cp2k.cp2k_cmd to your CP2K binary (or set cp2k.exec_prefix, e.g. ['conda','run','-n','<env>'], to execute in a separate environment)."
                    )
                cmd += [exe_path]

        # cp2 input output
        cmd += ["-i", str(input_file.name), "-o", str(out_file.name)]
        cmd += list(self.cfg.extra_args)

        timeout_use = timeout_sec if timeout_sec is not None else self.cfg.timeout_sec
        rc, out, err = run_cmd(
            cmd,
            cwd=workdir,
            env=env,
            check=False,
            capture=True,
            timeout=timeout_use,
            kill_grace_sec=float(self.cfg.kill_grace_sec),
        )
        # debug
        (workdir / "stdout.txt").write_text(out)
        (workdir / "stderr.txt").write_text(err)
        # dump captured screen
        try:
            screen_file.write_text(out)
        except Exception:
            pass

        if rc != 0:
            ctx = CommandFailureContext(
                screen_tail=_tail_file(screen_file),
                log_tail=_tail_file(out_file),
                stdout_tail=_tail_lines(out, n=80),
                stderr_tail=_tail_lines(err, n=80),
            )
            raise ExternalCommandError(cmd, rc, out, err, context=ctx)
        return RunResult(cmd=cmd, returncode=rc, stdout=out, stderr=err, log_file=out_file)
