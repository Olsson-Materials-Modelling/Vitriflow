from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..lammps_units import (
    energy_density_to_pressure_factor,
    length_to_angstrom_factor,
    pressure_to_gpa_factor as _pressure_to_gpa_factor,
    volume_to_angstrom3_factor,
)


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
LOCAL_STRESS_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")


def born21_to_matrix(born21: np.ndarray) -> np.ndarray:
    """Born21 to matrix."""

    arr = np.asarray(born21, dtype=float).reshape(-1)
    if arr.size != 21:
        raise ValueError(f"born21 must have length 21, got {arr.size}")
    mat = np.zeros((6, 6), dtype=float)
    idx_pairs = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5),
    ]
    for v, (i, j) in zip(arr.tolist(), idx_pairs):
        mat[i, j] = float(v)
        mat[j, i] = float(v)
    return mat


def matrix_to_born21(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    if arr.shape != (6, 6):
        raise ValueError(f"matrix_to_born21 expects shape (6,6), got {arr.shape}")
    idx_pairs = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5),
    ]
    return np.asarray([float(arr[i, j]) for (i, j) in idx_pairs], dtype=float)


def pressure_unit_label(units_style: str) -> str:
    u = str(units_style or "").strip().lower()
    if u == "metal":
        return "bar"
    if u == "real":
        return "atm"
    if u in {"si", "electron"}:
        return "Pa"
    if u == "nano":
        return "ag/(nm ns^2)"
    if u == "micro":
        return "pg/(um us^2)"
    if u == "cgs":
        return "dyne/cm^2"
    return "native"


def length_unit_label(units_style: str) -> str:
    return {
        "metal": "Å",
        "real": "Å",
        "electron": "bohr",
        "nano": "nm",
        "si": "m",
        "cgs": "cm",
        "micro": "µm",
    }.get(str(units_style or "").strip().lower(), "native_length")


def pressure_to_gpa_factor(units_style: str) -> Optional[float]:
    try:
        return float(_pressure_to_gpa_factor(units_style))
    except ValueError:
        return None


def isotropic_projection_voigt(C: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Isotropic projection voigt."""

    M = np.asarray(C, dtype=float)
    if M.shape != (6, 6):
        raise ValueError(f"Expected (6,6) matrix, got {M.shape}")
    M = 0.5 * (M + M.T)
    K = float((M[0, 0] + M[1, 1] + M[2, 2] + 2.0 * (M[0, 1] + M[0, 2] + M[1, 2])) / 9.0)
    # Uniform-strain (Voigt) orientational average in engineering-shear Voigt
    # notation.  The diagonal-shear average alone is only valid after the
    # tensor is already isotropic.
    G = float(
        (
            M[0, 0]
            + M[1, 1]
            + M[2, 2]
            - M[0, 1]
            - M[0, 2]
            - M[1, 2]
            + 3.0 * (M[3, 3] + M[4, 4] + M[5, 5])
        )
        / 15.0
    )
    iso = np.array(
        [
            [K + 4.0 * G / 3.0, K - 2.0 * G / 3.0, K - 2.0 * G / 3.0, 0.0, 0.0, 0.0],
            [K - 2.0 * G / 3.0, K + 4.0 * G / 3.0, K - 2.0 * G / 3.0, 0.0, 0.0, 0.0],
            [K - 2.0 * G / 3.0, K - 2.0 * G / 3.0, K + 4.0 * G / 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, G, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, G, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, G],
        ],
        dtype=float,
    )
    return iso, K, G


def hydrostatic_from_stress(stress_voigt: np.ndarray) -> np.ndarray:
    s = np.asarray(stress_voigt, dtype=float)
    return -np.mean(s[..., 0:3], axis=-1)


def von_mises_from_stress(stress_voigt: np.ndarray) -> np.ndarray:
    s = np.asarray(stress_voigt, dtype=float)
    sxx, syy, szz, sxy, sxz, syz = [s[..., i] for i in range(6)]
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + sxz**2 + syz**2)
    return np.sqrt(np.maximum(vm2, 0.0))


def stress_volume_to_pressure_like(stress_volume: np.ndarray, *, volume: float, n_atoms: int) -> np.ndarray:
    """Normalize per-atom virials by the global mean volume ``V/N``.

    The result has pressure dimensions and is useful as a spatial virial
    hotspot *proxy*.  It is not a uniquely defined local Cauchy stress: LAMMPS
    does not supply an atomic-volume partition, and using the same ``V/N`` for
    every atom is only a diagnostic normalization.
    """

    arr = np.asarray(stress_volume, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"stress_volume must have shape (N,6), got {arr.shape}")
    vol = float(volume)
    nat = int(n_atoms)
    if not (math.isfinite(vol) and vol > 0.0 and nat > 0):
        raise ValueError("volume must be >0 and n_atoms must be >0")
    vbar = float(vol) / float(nat)
    # LAMMPS compute stress/atom uses the stress sign convention: its diagonal
    # is negative under positive compression and has units pressure*volume.
    # Divide by the *mean* volume per atom; do not negate it or describe the
    # result as a uniquely partitioned local Cauchy stress.
    return arr / float(vbar)


def _safe_rel_spread(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mu = float(np.mean(np.abs(arr)))
    if not (math.isfinite(mu) and mu > 0.0):
        return 0.0
    return float(np.max(arr) - np.min(arr)) / mu


def _nanpercentile(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanpercentile(arr, float(q)))


def parse_born_stress_raw(path: Path) -> dict[str, Any]:
    txt = Path(path).read_text(errors="replace").strip().split()
    if len(txt) == 0:
        raise ValueError(f"Empty born/stress raw file: {path}")
    kv: dict[str, float] = {}
    for tok in txt:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        kv[str(k)] = float(v)
    born = np.asarray([kv[f"B{i}"] for i in range(1, 22)], dtype=float)
    stress = np.asarray([kv[f"S{i}"] for i in range(1, 7)], dtype=float)
    vol = float(kv.get("vol", float("nan")))
    if not np.all(np.isfinite(born)) or not np.all(np.isfinite(stress)):
        raise ValueError(f"Born/stress raw values in {path} must be finite")
    if not math.isfinite(vol) or vol <= 0.0:
        raise ValueError(f"Born/stress volume in {path} must be finite and > 0")
    return {
        "born21": born,
        "global_stress_voigt": stress,
        "volume": vol,
    }


def read_single_custom_dump(path: Path) -> dict[str, Any]:
    """Read exactly one finite, structurally valid LAMMPS custom-dump frame."""

    def _strict_integer(raw: str, *, field: str, line_number: int, minimum: int) -> int:
        try:
            numeric = float(raw)
            value = int(numeric)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {field} at {path}:{line_number}: {raw!r}") from exc
        if not math.isfinite(numeric) or numeric != float(value) or value < minimum:
            raise ValueError(
                f"{field} at {path}:{line_number} must be an integer >= {minimum}; "
                f"got {raw!r}"
            )
        return value

    lines = Path(path).read_text(errors="replace").splitlines()
    if len(lines) < 9:
        raise ValueError(f"Custom dump too short: {path}")
    i = 0
    if not lines[i].strip().startswith("ITEM: TIMESTEP"):
        raise ValueError(f"Expected ITEM: TIMESTEP at line 1 in {path}")
    i += 1
    timestep = _strict_integer(
        lines[i].strip(), field="timestep", line_number=i + 1, minimum=0
    )
    i += 1
    if not lines[i].strip().startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError(f"Expected ITEM: NUMBER OF ATOMS in {path}")
    i += 1
    natoms = _strict_integer(
        lines[i].strip(), field="number of atoms", line_number=i + 1, minimum=1
    )
    i += 1
    hdr = lines[i].strip()
    if not hdr.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"Expected ITEM: BOX BOUNDS in {path}")
    i += 1
    bounds: list[list[float]] = []
    for _ in range(3):
        if i >= len(lines):
            raise ValueError(f"Truncated BOX BOUNDS block in {path}")
        toks = lines[i].split()
        if len(toks) not in {2, 3}:
            raise ValueError(
                f"BOX BOUNDS row at {path}:{i + 1} must have 2 or 3 numeric values"
            )
        try:
            row = [float(x) for x in toks]
        except ValueError as exc:
            raise ValueError(f"Invalid BOX BOUNDS row at {path}:{i + 1}") from exc
        if not np.all(np.isfinite(np.asarray(row, dtype=float))):
            raise ValueError(f"BOX BOUNDS values at {path}:{i + 1} must be finite")
        bounds.append(row)
        i += 1
    row_lengths = {len(row) for row in bounds}
    if len(row_lengths) != 1:
        raise ValueError(f"Inconsistent BOX BOUNDS row widths in {path}")
    if len(bounds[0]) == 2:
        xlo, xhi = bounds[0]
        ylo, yhi = bounds[1]
        zlo, zhi = bounds[2]
        lx, ly, lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)
        cell = np.array([[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]], dtype=float)
        origin = np.array([xlo, ylo, zlo], dtype=float)
    elif len(bounds[0]) == 3:
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
        raise ValueError(f"Unrecognized BOX BOUNDS format in {path}")
    if not (lx > 0.0 and ly > 0.0 and lz > 0.0):
        raise ValueError(f"Custom-dump box lengths in {path} must be positive")
    vol = float(abs(np.linalg.det(cell)))
    scale = float(np.max(np.abs(cell)))
    det_tol = 128.0 * np.finfo(float).eps * max(scale**3, np.finfo(float).tiny)
    if (
        not np.all(np.isfinite(cell))
        or not np.all(np.isfinite(origin))
        or not math.isfinite(vol)
        or vol <= det_tol
    ):
        raise ValueError(f"Custom-dump cell in {path} must be finite and nonsingular")

    if i >= len(lines):
        raise ValueError(f"Missing ITEM: ATOMS block in {path}")
    hdr = lines[i].strip()
    if not hdr.startswith("ITEM: ATOMS"):
        raise ValueError(f"Expected ITEM: ATOMS in {path}")
    cols = hdr.split()[2:]
    if not cols or len(set(cols)) != len(cols):
        raise ValueError(f"ITEM: ATOMS columns in {path} must be non-empty and unique")
    if "id" not in cols or "type" not in cols:
        raise ValueError(f"ITEM: ATOMS in {path} must contain id and type columns")
    i += 1
    data_map: dict[str, np.ndarray] = {}
    table = [[] for _ in cols]
    for atom_row in range(int(natoms)):
        if i >= len(lines):
            raise ValueError(
                f"Custom dump {path} declares {natoms} atoms but ends after {atom_row} rows"
            )
        toks = lines[i].split()
        if len(toks) != len(cols):
            raise ValueError(
                f"Atom row at {path}:{i + 1} has {len(toks)} values; expected {len(cols)}"
            )
        try:
            values = [float(tok) for tok in toks]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric atom row at {path}:{i + 1}") from exc
        if not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError(f"Atom values at {path}:{i + 1} must be finite")
        for j, value in enumerate(values):
            table[j].append(value)
        i += 1
    trailing = [line for line in lines[i:] if line.strip()]
    if trailing:
        raise ValueError(
            f"Custom dump {path} contains extra data after the declared {natoms} atom rows"
        )
    for c, vals in zip(cols, table):
        data_map[str(c)] = np.asarray(vals, dtype=float)
    ids_raw = data_map["id"]
    types_raw = data_map["type"]
    if np.any(ids_raw != np.floor(ids_raw)) or np.any(ids_raw <= 0.0):
        raise ValueError(f"Atom ids in {path} must be positive integers")
    if np.unique(ids_raw).size != ids_raw.size:
        raise ValueError(f"Atom ids in {path} must be unique")
    if np.any(types_raw != np.floor(types_raw)) or np.any(types_raw <= 0.0):
        raise ValueError(f"Atom types in {path} must be positive integers")
    data_map["id"] = ids_raw.astype(int)
    data_map["type"] = types_raw.astype(int)
    return {
        "timestep": int(timestep),
        "natoms": int(natoms),
        "cell": cell,
        "origin": origin,
        "volume": vol,
        "columns": list(cols),
        "data": data_map,
    }


def affine_isotropization_strain(cell: np.ndarray) -> dict[str, Any]:
    H0 = np.asarray(cell, dtype=float)
    if H0.shape != (3, 3):
        raise ValueError(f"Expected input cell shape (3,3), got {H0.shape}")
    vol = float(abs(np.linalg.det(H0)))
    if not (math.isfinite(vol) and vol > 0.0):
        raise ValueError("Input cell volume must be finite and > 0")
    L = float(vol ** (1.0 / 3.0))
    H1 = np.diag([L, L, L]).astype(float)
    # ASE/LAMMPS cell vectors are rows.  In the conventional column-vector
    # deformation convention B1 = F B0, with B0=H0.T and B1=H1.T, hence
    # F = H1.T H0^-T.  The old solve(H0.T,H1.T).T reported its transpose for a
    # triclinic cell (the symmetric small strain happened to hide that error
    # for a cubic target).
    F = np.linalg.solve(H0, H1).T
    eps = 0.5 * (F + F.T) - np.eye(3)
    return {
        "volume": vol,
        "target_cubic_length": L,
        "F": F.tolist(),
        "small_strain": eps.tolist(),
        "frobenius_norm": float(np.linalg.norm(eps)),
        "principal_small_strains": [float(x) for x in np.linalg.eigvalsh(eps).tolist()],
    }


def build_elastic_screen_summary(
    *,
    born21: np.ndarray,
    global_stress_voigt: np.ndarray,
    volume: float,
    n_atoms: int,
    local_positions: np.ndarray,
    local_types: np.ndarray,
    local_stress_volume: np.ndarray,
    units_style: str,
    isotropy_warn_threshold: float = 0.15,
    coupling_warn_threshold: float = 0.10,
    hotspot_warn_multiple: float = 5.0,
    force_isotropic: bool = False,
    input_cell: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    born21 = np.asarray(born21, dtype=float).reshape(-1)
    global_stress_voigt = np.asarray(global_stress_voigt, dtype=float).reshape(-1)
    pos = np.asarray(local_positions, dtype=float)
    typ_numeric = np.asarray(local_types, dtype=float).reshape(-1)
    stress_vol = np.asarray(local_stress_volume, dtype=float)
    if born21.size != 21:
        raise ValueError(f"born21 must have length 21, got {born21.size}")
    if global_stress_voigt.size != 6:
        raise ValueError(f"global_stress_voigt must have length 6, got {global_stress_voigt.size}")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"local_positions must have shape (N,3), got {pos.shape}")
    if stress_vol.shape != (pos.shape[0], 6):
        raise ValueError(f"local_stress_volume must have shape (N,6), got {stress_vol.shape}")
    if typ_numeric.size != pos.shape[0]:
        raise ValueError("local_types length mismatch")
    if not np.all(np.isfinite(born21)):
        raise ValueError("born21 must contain only finite values")
    if not np.all(np.isfinite(global_stress_voigt)):
        raise ValueError("global_stress_voigt must contain only finite values")
    if not np.all(np.isfinite(pos)):
        raise ValueError("local_positions must contain only finite values")
    if not np.all(np.isfinite(stress_vol)):
        raise ValueError("local_stress_volume must contain only finite values")
    if not np.all(np.isfinite(typ_numeric)) or np.any(typ_numeric != np.floor(typ_numeric)):
        raise ValueError("local_types must contain finite integer LAMMPS type indices")
    if np.any(typ_numeric < 1):
        raise ValueError("local_types must contain positive LAMMPS type indices")
    typ = typ_numeric.astype(int)
    vol = float(volume)
    try:
        nat_numeric = float(n_atoms)
        nat = int(nat_numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_atoms must be a positive integer") from exc
    if not math.isfinite(nat_numeric) or nat_numeric != float(nat) or nat < 1:
        raise ValueError("n_atoms must be a positive integer")
    if not (math.isfinite(vol) and vol > 0.0 and nat > 0):
        raise ValueError("volume must be finite and > 0; n_atoms must be > 0")
    if nat != int(pos.shape[0]):
        raise ValueError(
            "n_atoms must match the per-atom virial rows; global-mean-volume "
            f"normalization would otherwise be invalid ({nat} != {pos.shape[0]})"
        )
    if input_cell is not None:
        input_cell_arr = np.asarray(input_cell, dtype=float)
        if input_cell_arr.shape != (3, 3) or not np.all(np.isfinite(input_cell_arr)):
            raise ValueError("input_cell must be a finite 3x3 matrix")

    # compute born/matrix is extensive and reports energy.  E/V must be
    # converted to the unit style's pressure unit (e.g. eV/A^3 -> bar in
    # ``metal``).  Conversely, compute pressure is already intensive and must
    # not be divided by volume a second time.
    C_energy = born21_to_matrix(born21)
    born_density_to_pressure = energy_density_to_pressure_factor(units_style)
    C_native = (C_energy / vol) * float(born_density_to_pressure)
    stress_native = np.asarray(global_stress_voigt, dtype=float)
    # compute stress/atom is already native pressure*volume, unlike
    # compute born/matrix (energy).  Dividing by V/N gives a pressure-like
    # virial hotspot proxy, not a uniquely defined local Cauchy stress.
    local_native = stress_volume_to_pressure_like(stress_vol, volume=vol, n_atoms=nat)

    iso_proj, K, G = isotropic_projection_voigt(C_native)
    C_sym = 0.5 * (C_native + C_native.T)
    denom = float(np.linalg.norm(C_sym))
    isotropy_residual = float(np.linalg.norm(C_sym - iso_proj) / denom) if denom > 0.0 else 0.0
    coupling_norm = float(np.linalg.norm(C_sym[0:3, 3:6]))
    normal_norm = float(np.linalg.norm(C_sym[0:3, 0:3]))
    normal_shear_coupling_norm = coupling_norm / normal_norm if normal_norm > 0.0 else 0.0

    diag_spread_rel = _safe_rel_spread(np.asarray([C_sym[0, 0], C_sym[1, 1], C_sym[2, 2]], dtype=float))
    offdiag_spread_rel = _safe_rel_spread(np.asarray([C_sym[0, 1], C_sym[0, 2], C_sym[1, 2]], dtype=float))
    shear_spread_rel = _safe_rel_spread(np.asarray([C_sym[3, 3], C_sym[4, 4], C_sym[5, 5]], dtype=float))
    eig = np.linalg.eigvalsh(C_sym)

    hydro = hydrostatic_from_stress(local_native)
    vm = von_mises_from_stress(local_native)
    vm_med = float(np.nanmedian(vm)) if vm.size > 0 else float("nan")
    vm_max = float(np.nanmax(vm)) if np.any(np.isfinite(vm)) else float("nan")
    hotspot_ratio = float(vm_max / vm_med) if math.isfinite(vm_max) and math.isfinite(vm_med) and vm_med > 0.0 else float("inf")

    gpa_fac = pressure_to_gpa_factor(units_style)
    units = {
        "lammps_units": str(units_style or ""),
        "pressure_native": pressure_unit_label(units_style),
        "pressure_to_GPa_factor": float(gpa_fac) if gpa_fac is not None else None,
        "born_energy_density_to_pressure_factor": float(born_density_to_pressure),
        "length_native": length_unit_label(units_style),
        "length_native_to_A_factor": float(length_to_angstrom_factor(units_style)),
        "volume_native": f"{length_unit_label(units_style)}^3",
        "volume_native_to_A3_factor": float(volume_to_angstrom3_factor(units_style)),
        "per_atom_virial_native": f"{pressure_unit_label(units_style)}*{length_unit_label(units_style)}^3",
        "average_volume_normalized_virial_proxy_native": pressure_unit_label(units_style),
        "reporting_length": "Å",
        "reporting_volume": "Å^3",
        "reporting_pressure": "GPa",
    }

    C_gpa = (C_native * float(gpa_fac)).tolist() if gpa_fac is not None else None
    stress_gpa = (stress_native * float(gpa_fac)).tolist() if gpa_fac is not None else None

    flags: list[str] = []
    if np.any(eig <= 0.0):
        flags.append("non_positive_born_eigenvalue")
    if math.isfinite(isotropy_residual) and isotropy_residual > float(isotropy_warn_threshold):
        flags.append("born_anisotropy")
    if math.isfinite(normal_shear_coupling_norm) and normal_shear_coupling_norm > float(coupling_warn_threshold):
        flags.append("normal_shear_coupling")
    if math.isfinite(hotspot_ratio) and hotspot_ratio > float(hotspot_warn_multiple):
        flags.append("virial_proxy_hotspots")

    mean_volume_native = float(vol) / float(nat)
    proxy_summary = {
        "quantity": "per_atom_virial_divided_by_global_mean_volume",
        "normalization": "V_global/N_atoms",
        "normalization_volume_native_per_atom": mean_volume_native,
        "normalization_volume_A3_per_atom": float(
            mean_volume_native * volume_to_angstrom3_factor(units_style)
        ),
        "atomic_volume_partition": False,
        "unique_local_cauchy_stress": False,
        "valid_interpretation": "spatial_virial_hotspot_proxy",
        "hydrostatic_proxy_native": {
            "p05": _nanpercentile(hydro, 5.0),
            "p50": _nanpercentile(hydro, 50.0),
            "p95": _nanpercentile(hydro, 95.0),
        },
        "hydrostatic_proxy_GPa": {
            "p05": _nanpercentile(hydro * float(gpa_fac), 5.0),
            "p50": _nanpercentile(hydro * float(gpa_fac), 50.0),
            "p95": _nanpercentile(hydro * float(gpa_fac), 95.0),
        },
        "von_mises_proxy_GPa": {
            "p05": _nanpercentile(vm * float(gpa_fac), 5.0),
            "p50": _nanpercentile(vm * float(gpa_fac), 50.0),
            "p95": _nanpercentile(vm * float(gpa_fac), 95.0),
            "max": float(vm_max * float(gpa_fac)),
            "max_over_median": float(hotspot_ratio),
        },
        "von_mises_proxy_native": {
            "p05": _nanpercentile(vm, 5.0),
            "p50": _nanpercentile(vm, 50.0),
            "p95": _nanpercentile(vm, 95.0),
            "max": float(vm_max),
            "max_over_median": float(hotspot_ratio),
        },
    }

    summary: dict[str, Any] = {
        "status": "ok",
        "kind": "static_affine_born_snapshot_diagnostic",
        "note": (
            "This is a single-configuration affine potential-energy Born curvature and a local "
            "virial hotspot diagnostic, not a finite-temperature thermodynamic elastic tensor "
            "or relaxed elastic modulus. Per-atom virials are divided by the global mean volume "
            "V/N only; without an atomic-volume partition they are not uniquely defined local "
            "Cauchy stresses. Kinetic, stress-covariance, and non-affine relaxation terms are "
            "not included."
        ),
        "method": {
            "estimator": "single_snapshot_affine_potential_energy_born_curvature",
            "thermodynamic_elastic_tensor": False,
            "relaxed_elastic_modulus": False,
            "included_terms": [
                "affine_potential_energy_born_curvature",
                "instantaneous_global_virial_stress",
                "per_atom_virial_divided_by_global_mean_volume_proxy",
            ],
            "omitted_terms": [
                "kinetic_elastic_contribution",
                "stress_fluctuation_covariance",
                "non_affine_internal_relaxation",
                "finite_strain_relaxed_response",
            ],
            "valid_interpretation": "static_affine_stiffness_fingerprint_and_stability_screen",
            "per_atom_virial_atomic_volume_partition": False,
            "per_atom_virial_is_unique_local_cauchy_stress": False,
        },
        "units": units,
        "volume": float(vol),  # legacy alias, explicitly native via units.volume_native
        "volume_native": float(vol),
        "volume_A3": float(vol * volume_to_angstrom3_factor(units_style)),
        "n_atoms": int(nat),
        "born_matrix_native": C_native.tolist(),
        "born_matrix_GPa": C_gpa,
        "global_stress_voigt_native": stress_native.tolist(),
        "global_stress_voigt_GPa": stress_gpa,
        "voigt_born_bulk_response_native": float(K),
        "voigt_born_bulk_response_GPa": float(K * gpa_fac) if gpa_fac is not None else None,
        "voigt_born_shear_response_native": float(G),
        "voigt_born_shear_response_GPa": float(G * gpa_fac) if gpa_fac is not None else None,
        "isotropy_residual": float(isotropy_residual),
        "normal_shear_coupling_norm": float(normal_shear_coupling_norm),
        "diag_spread_rel": float(diag_spread_rel),
        "offdiag_spread_rel": float(offdiag_spread_rel),
        "shear_spread_rel": float(shear_spread_rel),
        "born_eigenvalues_native": [float(x) for x in eig.tolist()],
        "average_volume_normalized_virial_proxy_summary": proxy_summary,
        "flags": flags,
        "force_isotropic": bool(force_isotropic),
    }

    if bool(force_isotropic) and input_cell is not None:
        try:
            summary["affine_isotropization"] = affine_isotropization_strain(np.asarray(input_cell, dtype=float))
            summary["affine_isotropization"]["length_unit"] = "Å"
            summary["affine_isotropization"]["volume_unit"] = "Å^3"
        except Exception as exc:
            summary["affine_isotropization_error"] = str(exc)

    return summary


def write_born_matrix_csv(path: Path, matrix: np.ndarray, *, units_label: str) -> None:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (6, 6):
        raise ValueError(f"Expected matrix shape (6,6), got {arr.shape}")
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"units={units_label}"] + list(VOIGT_LABELS))
        for lab, row in zip(VOIGT_LABELS, arr.tolist()):
            w.writerow([lab] + [float(x) for x in row])


def write_local_stress_csv(
    path: Path,
    *,
    ids: np.ndarray,
    types: np.ndarray,
    positions: np.ndarray,
    stress_volume: np.ndarray,
    stress_native: np.ndarray,
    hydrostatic_native: np.ndarray,
    von_mises_native: np.ndarray,
    units_style: str | None = None,
    normalization_volume_native_per_atom: float | None = None,
) -> None:
    """Write an average-volume-normalized virial-proxy table.

    Supplying both unit/provenance arguments writes the canonical, physically
    explicit schema.  Omitting both retains the pre-0.4.31 native-unit CSV
    schema for callers of this public helper.  Partial specification is
    rejected so a canonical-looking artifact can never carry guessed units or
    normalization provenance.
    """
    ids_numeric = np.asarray(ids, dtype=float).reshape(-1)
    types_numeric = np.asarray(types, dtype=float).reshape(-1)
    pos = np.asarray(positions, dtype=float)
    sv = np.asarray(stress_volume, dtype=float)
    sn = np.asarray(stress_native, dtype=float)
    hydro = np.asarray(hydrostatic_native, dtype=float).reshape(-1)
    vm = np.asarray(von_mises_native, dtype=float).reshape(-1)
    n = int(ids_numeric.size)
    if pos.shape != (n, 3) or sv.shape != (n, 6) or sn.shape != (n, 6) or hydro.size != n or vm.size != n or types_numeric.size != n:
        raise ValueError("write_local_stress_csv: inconsistent array sizes")
    if n < 1:
        raise ValueError("write_local_stress_csv requires at least one atom")
    if not np.all(np.isfinite(ids_numeric)) or np.any(ids_numeric != np.floor(ids_numeric)):
        raise ValueError("write_local_stress_csv ids must be finite integers")
    if np.any(ids_numeric <= 0.0) or np.unique(ids_numeric).size != n:
        raise ValueError("write_local_stress_csv ids must be unique and positive")
    if not np.all(np.isfinite(types_numeric)) or np.any(
        types_numeric != np.floor(types_numeric)
    ):
        raise ValueError("write_local_stress_csv types must be finite integers")
    if np.any(types_numeric <= 0.0):
        raise ValueError("write_local_stress_csv types must be positive")
    if not all(np.all(np.isfinite(array)) for array in (pos, sv, sn, hydro, vm)):
        raise ValueError("write_local_stress_csv numeric arrays must be finite")
    ids = ids_numeric.astype(int)
    types = types_numeric.astype(int)

    if units_style is None and normalization_volume_native_per_atom is None:
        warnings.warn(
            "write_local_stress_csv without units_style and normalization volume "
            "writes a legacy-only schema: units, normalization, and stress semantics "
            "are caller-defined and are not validated as a local Cauchy stress. "
            "Pass both provenance arguments for release artifacts.",
            DeprecationWarning,
            stacklevel=2,
        )
        hdr = [
            "id", "type", "x", "y", "z",
            "sv_xx", "sv_yy", "sv_zz", "sv_xy", "sv_xz", "sv_yz",
            "s_xx", "s_yy", "s_zz", "s_xy", "s_xz", "s_yz",
            "hydrostatic_native", "von_mises_native",
        ]
        with Path(path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            for i in range(n):
                w.writerow(
                    [int(ids[i]), int(types[i]), float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])]
                    + [float(x) for x in sv[i].tolist()]
                    + [float(x) for x in sn[i].tolist()]
                    + [float(hydro[i]), float(vm[i])]
                )
        return
    if units_style is None or normalization_volume_native_per_atom is None:
        raise ValueError(
            "units_style and normalization_volume_native_per_atom must be "
            "provided together"
        )

    vbar_native = float(normalization_volume_native_per_atom)
    if not math.isfinite(vbar_native) or vbar_native <= 0.0:
        raise ValueError("normalization_volume_native_per_atom must be finite and > 0")
    pos_A = pos * float(length_to_angstrom_factor(units_style))
    vbar_A3 = vbar_native * float(volume_to_angstrom3_factor(units_style))
    stress_GPa = sn * float(_pressure_to_gpa_factor(units_style))
    hydro_GPa = hydro * float(_pressure_to_gpa_factor(units_style))
    vm_GPa = vm * float(_pressure_to_gpa_factor(units_style))
    hdr = [
        "id", "type", "x_A", "y_A", "z_A", "normalization_volume_A3_per_atom",
        "virial_proxy_xx_GPa", "virial_proxy_yy_GPa", "virial_proxy_zz_GPa",
        "virial_proxy_xy_GPa", "virial_proxy_xz_GPa", "virial_proxy_yz_GPa",
        "hydrostatic_virial_proxy_GPa", "von_mises_virial_proxy_GPa",
    ]
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for i in range(n):
            w.writerow(
                [
                    int(ids[i]), int(types[i]),
                    float(pos_A[i, 0]), float(pos_A[i, 1]), float(pos_A[i, 2]),
                    float(vbar_A3),
                ]
                + [float(x) for x in stress_GPa[i].tolist()]
                + [float(hydro_GPa[i]), float(vm_GPa[i])]
            )


def load_elastic_screen_summary(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
