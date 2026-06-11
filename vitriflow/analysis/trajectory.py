from __future__ import annotations

"""Trajectory loading and stage-frame selection helpers."""

import math
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .dump import DumpFrame, read_dump_frames, read_last_dump_frames
from ..io.extxyz import read_extxyz_frames


def _read_text_head(path: Path, *, max_lines: int = 80) -> list[str]:
    try:
        return Path(path).read_text(errors="replace").splitlines()[: int(max_lines)]
    except Exception:
        return []


def _looks_like_lammps_dump(path: Path) -> bool:
    p = Path(path)
    if p.suffix.lower() in {".lammpstrj", ".dump", ".trj"}:
        return True
    head = _read_text_head(p, max_lines=80)
    if not head:
        return False
    up = [str(ln).strip().upper() for ln in head]
    return bool(any(ln.startswith("ITEM: TIMESTEP") for ln in up) and any(ln.startswith("ITEM: ATOMS") for ln in up))


def _looks_like_lammps_data(path: Path) -> bool:
    p = Path(path)
    name = p.name.lower()
    if name in {"relax.data", "output.data", "input.data", "structure.data"}:
        return True
    if p.suffix.lower() in {".data", ".lmp", ".dat"}:
        return True
    head = _read_text_head(p, max_lines=80)
    if not head:
        return False
    low = [str(ln).lower() for ln in head]
    atoms_hdr = any(" atoms" in ln for ln in low)
    types_hdr = any(" atom types" in ln for ln in low)
    bounds_hdr = any("xlo xhi" in ln for ln in low)
    atoms_section = any(str(ln).strip().lower().startswith("atoms") for ln in low)
    return bool(atoms_hdr and bounds_hdr and (types_hdr or atoms_section))


def _atoms_to_dumpframe(atoms, *, type_to_species: Optional[Sequence[str]], timestep: int) -> DumpFrame:
    syms = [str(s) for s in atoms.get_chemical_symbols()]
    if len(syms) < 1:
        raise ValueError("Structure source produced zero atoms")

    if type_to_species is not None:
        mapping = {str(sym): i + 1 for i, sym in enumerate(list(type_to_species))}
    else:
        mapping = {str(sym): i + 1 for i, sym in enumerate(sorted(set(syms)))}
    try:
        types = np.asarray([int(mapping[str(sym)]) for sym in syms], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Structure contains symbol not present in type_to_species: {exc}") from exc

    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("Structure source has invalid cell shape")
    if abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError("Structure source is missing a valid periodic cell")

    ids = np.arange(1, int(len(syms)) + 1, dtype=int)
    return DumpFrame(
        timestep=int(timestep),
        ids=np.asarray(ids, dtype=int),
        types=np.asarray(types, dtype=int),
        positions=np.asarray(pos, dtype=float),
        cell=np.asarray(cell, dtype=float),
        origin=np.zeros((3,), dtype=float),
    )


def _frames_from_ase_source(
    path: Path,
    *,
    last_n: Optional[int],
    type_to_species: Optional[Sequence[str]],
    atom_style: str,
) -> list[DumpFrame]:
    p = Path(path)

    if _looks_like_lammps_data(p):
        atoms = None
        try:
            from ase.io import read as ase_read

            atoms = ase_read(
                str(p),
                format="lammps-data",
                style=str(atom_style),
                specorder=(None if type_to_species is None else list(type_to_species)),
            )
        except Exception:
            try:
                from ..io.lammps_data_minimal import read_lammps_data_minimal

                atoms = read_lammps_data_minimal(
                    p,
                    atom_style=str(atom_style),
                    specorder=(None if type_to_species is None else list(type_to_species)),
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to read LAMMPS data structure: {p}") from exc
        assert atoms is not None
        return [_atoms_to_dumpframe(atoms, type_to_species=type_to_species, timestep=0)]

    from ase.io import read as ase_read

    images = None
    if last_n is not None:
        try:
            images = ase_read(str(p), index=slice(-int(last_n), None))
        except Exception:
            images = None
    if images is None:
        try:
            images = ase_read(str(p), index=":")
        except Exception:
            images = ase_read(str(p))

    if isinstance(images, (list, tuple)):
        seq = list(images)
    else:
        seq = [images]
    if last_n is not None and len(seq) > int(last_n):
        seq = seq[-int(last_n) :]

    frames: list[DumpFrame] = []
    for i, atoms in enumerate(seq):
        info = getattr(atoms, "info", None)
        step = int(i)
        if isinstance(info, dict):
            for key in ("Step", "step", "timestep", "Timestep"):
                val = info.get(key, None)
                if val is None:
                    continue
                try:
                    step = int(float(val))
                    break
                except Exception:
                    continue
        frames.append(_atoms_to_dumpframe(atoms, type_to_species=type_to_species, timestep=step))
    return frames


def read_frames_auto(
    path: Path,
    *,
    last_n: Optional[int] = None,
    type_to_species: Optional[Sequence[str]] = None,
    atom_style: str = "atomic",
) -> list[DumpFrame]:
    """Frames auto."""

    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".extxyz":
        return read_extxyz_frames(p, last_n=last_n, type_to_species=type_to_species)
    if suf == ".xyz":
        try:
            return read_extxyz_frames(p, last_n=last_n, type_to_species=type_to_species)
        except Exception:
            return _frames_from_ase_source(
                p,
                last_n=last_n,
                type_to_species=type_to_species,
                atom_style=atom_style,
            )
    if _looks_like_lammps_dump(p):
        return read_dump_frames(p, last_n=last_n)
    return _frames_from_ase_source(
        p,
        last_n=last_n,
        type_to_species=type_to_species,
        atom_style=atom_style,
    )


def read_last_frames_auto(
    path: Path,
    n: int,
    *,
    type_to_species: Optional[Sequence[str]] = None,
    atom_style: str = "atomic",
) -> list[DumpFrame]:
    """Last frames auto."""

    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".extxyz":
        return read_extxyz_frames(p, last_n=int(n), type_to_species=type_to_species)
    if suf == ".xyz":
        try:
            return read_extxyz_frames(p, last_n=int(n), type_to_species=type_to_species)
        except Exception:
            return _frames_from_ase_source(
                p,
                last_n=int(n),
                type_to_species=type_to_species,
                atom_style=atom_style,
            )
    if _looks_like_lammps_dump(p):
        return read_last_dump_frames(p, int(n))
    return _frames_from_ase_source(
        p,
        last_n=int(n),
        type_to_species=type_to_species,
        atom_style=atom_style,
    )


def stage_trajectory_path(stage_dir: Path) -> Optional[Path]:
    """Stage trajectory path."""
    d = Path(stage_dir)
    cand = d / "traj.extxyz"
    if cand.exists():
        return cand
    for nm in d.glob("*.lammpstrj"):
        return nm
    return None


def evenly_sample_indices(n: int, k: Optional[int]) -> list[int]:
    """Evenly sample indices."""

    n = int(n)
    if n <= 0:
        return []
    if k is None:
        return list(range(n))
    k = int(k)
    if k <= 0 or k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    idx = np.linspace(0, n - 1, k, dtype=int)
    out: list[int] = []
    seen: set[int] = set()
    for i in idx.tolist():
        ii = int(i)
        if ii not in seen:
            out.append(ii)
            seen.add(ii)
    if out[0] != 0:
        out.insert(0, 0)
    if out[-1] != n - 1:
        out.append(n - 1)
    return sorted(set(out), key=out.index)


def quench_window_steps(
    *,
    T_start: float,
    T_stop: float,
    total_steps: int,
    T_upper: Optional[float],
    T_lower: Optional[float],
) -> Optional[Tuple[float, float]]:
    """Quench window steps."""

    if int(total_steps) <= 0:
        return None
    if T_upper is None or T_lower is None:
        return None
    Tu = float(T_upper)
    Tl = float(T_lower)
    if not (np.isfinite(Tu) and np.isfinite(Tl)):
        return None
    if float(T_start) == float(T_stop):
        return None
    loT = min(float(T_start), float(T_stop))
    hiT = max(float(T_start), float(T_stop))
    Tu = min(max(Tu, loT), hiT)
    Tl = min(max(Tl, loT), hiT)
    if Tu < Tl:
        Tu, Tl = Tl, Tu

    def _step_for_T(T: float) -> float:
        frac = (float(T) - float(T_start)) / (float(T_stop) - float(T_start))
        return float(frac * float(total_steps))

    s1 = _step_for_T(Tu)
    s2 = _step_for_T(Tl)
    a = max(0.0, min(float(total_steps), min(s1, s2)))
    b = max(0.0, min(float(total_steps), max(s1, s2)))
    if b <= a:
        return None
    return (float(a), float(b))


def _select_dense_window_indices(
    steps: np.ndarray,
    *,
    window: Tuple[float, float],
    quench_tail_min_frames: int,
    max_frames: Optional[int],
) -> list[int]:
    dense_idx = [int(i) for i, s in enumerate(steps.tolist()) if float(window[0]) <= float(s) <= float(window[1])]
    if len(dense_idx) == 0:
        return []

    n_dense_target = max(int(quench_tail_min_frames), 2)
    if max_frames is not None and int(max_frames) > 0:
        n_dense_target = min(max(int(quench_tail_min_frames), 2), int(max_frames))
    dense_rel = evenly_sample_indices(len(dense_idx), n_dense_target)
    chosen_dense = [dense_idx[i] for i in dense_rel]

    chosen: list[int] = []
    for idx in [0] + chosen_dense + [len(steps) - 1]:
        if idx not in chosen:
            chosen.append(idx)

    if max_frames is not None and int(max_frames) > 0:
        target = max(int(max_frames), len(chosen), n_dense_target)
        if len(chosen) < target:
            spare = [i for i in evenly_sample_indices(len(steps), target) if i not in chosen]
            for i in spare:
                chosen.append(i)
                if len(chosen) >= target:
                    break
        if len(chosen) > target:
            dense_set = set(chosen_dense)
            keep = [i for i in chosen if i in dense_set or i in {0, len(steps) - 1}]
            for i in chosen:
                if i not in keep:
                    keep.append(i)
                if len(keep) >= target:
                    break
            chosen = keep[:target]

    return sorted(set(chosen))


def select_stage_frames(
    frames_all: Sequence[DumpFrame],
    *,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    stage_role: Optional[str] = None,
    quench_window_steps_range: Optional[Tuple[float, float]] = None,
    temperatures: Optional[Sequence[float]] = None,
    tm_temperature: Optional[float] = None,
    diffusion_freeze_temperature: Optional[float] = None,
    quench_tail_fraction: float = 0.67,
    quench_tail_min_frames: int = 8,
    quench_tail_fallback_fraction: float = 0.40,
) -> tuple[list[DumpFrame], dict[str, Any]]:
    """Stage frames."""

    if int(frame_stride) < 1:
        raise ValueError("frame_stride must be >= 1")

    base = list(frames_all)[:: int(frame_stride)]
    meta: dict[str, Any] = {
        "selection": "uniform",
        "frame_stride": int(frame_stride),
        "max_frames": int(max_frames) if max_frames is not None else None,
    }
    if not base:
        return [], meta

    steps = np.asarray([float(fr.timestep) for fr in base], dtype=float)
    role = str(stage_role or "").strip().lower()

    dense_window: Optional[Tuple[float, float]] = None
    selection = "uniform"

    if role == "quench":
        if temperatures is not None and tm_temperature is not None and diffusion_freeze_temperature is not None:
            try:
                temps = np.asarray([float(x) for x in temperatures], dtype=float)
            except Exception:
                temps = np.full((len(base),), np.nan, dtype=float)
            if temps.shape[0] == len(base) and np.isfinite(temps).any():
                t_hi = max(float(tm_temperature), float(diffusion_freeze_temperature))
                t_lo = min(float(tm_temperature), float(diffusion_freeze_temperature))
                idx = [int(i) for i, T in enumerate(temps.tolist()) if t_lo <= float(T) <= t_hi]
                if idx:
                    dense_window = (float(steps[min(idx)]), float(steps[max(idx)]))
                    selection = "quench_tm_freeze_dense"
                    meta["tm_temperature"] = float(tm_temperature)
                    meta["diffusion_freeze_temperature"] = float(diffusion_freeze_temperature)
                    meta["dense_window_temperature"] = [float(t_lo), float(t_hi)]
        if dense_window is None and quench_window_steps_range is not None:
            dense_window = (float(quench_window_steps_range[0]), float(quench_window_steps_range[1]))
            selection = "quench_step_window_dense"
        if dense_window is None:
            frac = min(max(float(quench_tail_fallback_fraction), 0.05), 0.95)
            start_idx = int(max(0, math.floor((1.0 - frac) * len(base))))
            dense_window = (float(steps[start_idx]), float(steps[-1]))
            selection = "quench_tail_dense"
            meta["dense_window_fraction"] = float(frac)

        chosen_idx = _select_dense_window_indices(
            steps,
            window=dense_window,
            quench_tail_min_frames=max(int(quench_tail_min_frames), 1),
            max_frames=max_frames,
        )
        meta["selection"] = selection
        meta["dense_window_steps"] = [float(dense_window[0]), float(dense_window[1])]
        meta["quench_tail_fraction"] = float(quench_tail_fraction)
        meta["quench_tail_min_frames"] = int(quench_tail_min_frames)
        meta["quench_tail_fallback_fraction"] = float(quench_tail_fallback_fraction)
    else:
        chosen_idx = evenly_sample_indices(len(base), max_frames)

    selected = [base[int(i)] for i in chosen_idx]
    meta["n_selected"] = int(len(selected))
    meta["selected_steps"] = [int(base[int(i)].timestep) for i in chosen_idx]
    return selected, meta
