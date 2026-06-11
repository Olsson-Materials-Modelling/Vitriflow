from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class DumpFrame:
    """Dump frame."""

    timestep: int
    ids: np.ndarray  # n int
    types: np.ndarray  # n int
    positions: np.ndarray  # n float cartesian
    cell: np.ndarray  # float row vectors
    origin: np.ndarray  # float

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    def cell_inv(self) -> np.ndarray:
        return np.linalg.inv(self.cell)


def _find_col(cols: list[str], names: Iterable[str]) -> Optional[int]:
    for nm in names:
        if nm in cols:
            return cols.index(nm)
    return None


def read_last_dump_frame(path: Path) -> DumpFrame:
    """Last dump frame."""

    frames = read_dump_frames(path, last_n=1)
    if not frames:
        raise ValueError(f"Failed to parse last frame from dump: {path}")
    return frames[-1]


def read_last_dump_frames(path: Path, n: int) -> list[DumpFrame]:
    """Last dump frames."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return read_dump_frames(path, last_n=int(n))


def read_dump_frames(path: Path, *, last_n: Optional[int] = None) -> list[DumpFrame]:
    """Dump frames."""

    frames: "deque[DumpFrame]" = deque(maxlen=int(last_n) if last_n is not None else None)
    for fr in iter_dump_frames(path):
        frames.append(fr)
    return list(frames)


def iter_dump_frames(path: Path) -> Iterable[DumpFrame]:
    """Iter dump frames."""

    with Path(path).open("r", errors="replace") as f:
        it = iter(f)
        while True:
            try:
                ln = next(it)
            except StopIteration:
                break
            ln = ln.strip()
            if not ln or not ln.startswith("ITEM: TIMESTEP"):
                continue

            # timestep
            try:
                timestep = int(next(it).strip())
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e

            # number atoms
            try:
                ln2 = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            if not ln2.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Unexpected dump format in {path}: expected NUMBER OF ATOMS, got: {ln2}")
            try:
                natoms = int(next(it).strip())
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e

            # box bounds
            try:
                ln3 = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            if not ln3.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected dump format in {path}: expected BOX BOUNDS, got: {ln3}")

            bounds = []
            try:
                for _ in range(3):
                    toks = next(it).split()
                    bounds.append([float(x) for x in toks])
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e

            if len(bounds[0]) == 2:
                xlo, xhi = bounds[0]
                ylo, yhi = bounds[1]
                zlo, zhi = bounds[2]
                lx, ly, lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)
                cell = np.array([[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]], dtype=float)
                origin = np.array([xlo, ylo, zlo], dtype=float)
            elif len(bounds[0]) >= 3:
                # triclinic bounding limits
                xlo_b, xhi_b, xy = float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2])
                ylo_b, yhi_b, xz = float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2])
                zlo_b, zhi_b, yz = float(bounds[2][0]), float(bounds[2][1]), float(bounds[2][2])

                x_corr_min = min(0.0, xy, xz, xy + xz)
                x_corr_max = max(0.0, xy, xz, xy + xz)
                y_corr_min = min(0.0, yz)
                y_corr_max = max(0.0, yz)

                xlo, xhi = (xlo_b - x_corr_min), (xhi_b - x_corr_max)
                ylo, yhi = (ylo_b - y_corr_min), (yhi_b - y_corr_max)
                zlo, zhi = zlo_b, zhi_b

                lx, ly, lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)
                cell = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)
                origin = np.array([xlo, ylo, zlo], dtype=float)
            else:
                raise ValueError(f"Unrecognized BOX BOUNDS line format in {path}")

            # atoms
            try:
                ln4 = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            if not ln4.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected dump format in {path}: expected ATOMS, got: {ln4}")
            cols = ln4.split()[2:]

            atom_lines: list[list[str]] = []
            try:
                for _ in range(natoms):
                    atom_lines.append(next(it).split())
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e

            yield _parse_dump_frame_from_table(timestep, natoms, cell, origin, cols, atom_lines, Path(path))


def _parse_dump_frame_from_table(
    timestep: int,
    natoms: int,
    cell: np.ndarray,
    origin: np.ndarray,
    cols: list[str],
    atom_lines: list[list[str]],
    path: Path,
) -> DumpFrame:
    """Dump frame from."""

    id_col = _find_col(cols, ["id"])
    type_col = _find_col(cols, ["type"])
    x_col = _find_col(cols, ["xu", "x"])
    y_col = _find_col(cols, ["yu", "y"])
    z_col = _find_col(cols, ["zu", "z"])
    xs_col = _find_col(cols, ["xs"])
    ys_col = _find_col(cols, ["ys"])
    zs_col = _find_col(cols, ["zs"])

    if id_col is None or type_col is None:
        raise ValueError(f"Dump {path} missing required columns 'id' and/or 'type' (got: {cols})")

    use_scaled = (x_col is None or y_col is None or z_col is None) and (xs_col is not None and ys_col is not None and zs_col is not None)
    if not use_scaled and (x_col is None or y_col is None or z_col is None):
        raise ValueError(
            f"Dump {path} must contain cartesian columns x/y/z or xu/yu/zu, or scaled xs/ys/zs. Got: {cols}"
        )

    ids = np.empty((natoms,), dtype=int)
    types = np.empty((natoms,), dtype=int)
    pos = np.empty((natoms, 3), dtype=float)

    if use_scaled:
        H = cell
        for i, toks in enumerate(atom_lines):
            ids[i] = int(float(toks[id_col]))
            types[i] = int(float(toks[type_col]))
            s = np.array([float(toks[xs_col]), float(toks[ys_col]), float(toks[zs_col])], dtype=float)
            pos[i, :] = origin + s @ H
    else:
        for i, toks in enumerate(atom_lines):
            ids[i] = int(float(toks[id_col]))
            types[i] = int(float(toks[type_col]))
            pos[i, :] = [float(toks[x_col]), float(toks[y_col]), float(toks[z_col])]

    order = np.argsort(ids)
    ids = ids[order]
    types = types[order]
    pos = pos[order]

    return DumpFrame(
        timestep=int(timestep),
        ids=ids,
        types=types,
        positions=pos,
        cell=np.asarray(cell, dtype=float),
        origin=np.asarray(origin, dtype=float),
    )
