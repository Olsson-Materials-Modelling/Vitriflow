from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


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
    if u == "si":
        return "Pa"
    if u == "cgs":
        return "dyne/cm^2"
    if u == "electron":
        return "Pa"
    return "native"


def pressure_to_gpa_factor(units_style: str) -> Optional[float]:
    u = str(units_style or "").strip().lower()
    if u == "metal":
        return 1.0e-4
    if u == "real":
        return 1.01325e-4
    if u in {"si", "electron"}:
        return 1.0e-9
    if u == "cgs":
        return 1.0e-10
    return None


def isotropic_projection_voigt(C: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Isotropic projection voigt."""

    M = np.asarray(C, dtype=float)
    if M.shape != (6, 6):
        raise ValueError(f"Expected (6,6) matrix, got {M.shape}")
    M = 0.5 * (M + M.T)
    K = float((M[0, 0] + M[1, 1] + M[2, 2] + 2.0 * (M[0, 1] + M[0, 2] + M[1, 2])) / 9.0)
    G = float((M[3, 3] + M[4, 4] + M[5, 5]) / 3.0)
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
    """Stress volume to."""

    arr = np.asarray(stress_volume, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"stress_volume must have shape (N,6), got {arr.shape}")
    vol = float(volume)
    nat = int(n_atoms)
    if not (math.isfinite(vol) and vol > 0.0 and nat > 0):
        raise ValueError("volume must be >0 and n_atoms must be >0")
    vbar = float(vol) / float(nat)
    return -arr / float(vbar)


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
    return {
        "born21": born,
        "global_stress_voigt": stress,
        "volume": vol,
    }


def read_single_custom_dump(path: Path) -> dict[str, Any]:
    """Single custom dump."""

    lines = Path(path).read_text(errors="replace").splitlines()
    if len(lines) < 9:
        raise ValueError(f"Custom dump too short: {path}")
    i = 0
    if not lines[i].strip().startswith("ITEM: TIMESTEP"):
        raise ValueError(f"Expected ITEM: TIMESTEP at line 1 in {path}")
    i += 1
    timestep = int(lines[i].strip())
    i += 1
    if not lines[i].strip().startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError(f"Expected ITEM: NUMBER OF ATOMS in {path}")
    i += 1
    natoms = int(lines[i].strip())
    i += 1
    hdr = lines[i].strip()
    if not hdr.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"Expected ITEM: BOX BOUNDS in {path}")
    i += 1
    bounds = []
    for _ in range(3):
        toks = lines[i].split()
        bounds.append([float(x) for x in toks])
        i += 1
    if len(bounds[0]) == 2:
        xlo, xhi = bounds[0]
        ylo, yhi = bounds[1]
        zlo, zhi = bounds[2]
        lx, ly, lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)
        cell = np.array([[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]], dtype=float)
        origin = np.array([xlo, ylo, zlo], dtype=float)
    elif len(bounds[0]) >= 3:
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
    vol = float(abs(np.linalg.det(cell)))

    hdr = lines[i].strip()
    if not hdr.startswith("ITEM: ATOMS"):
        raise ValueError(f"Expected ITEM: ATOMS in {path}")
    cols = hdr.split()[2:]
    i += 1
    data_map: dict[str, np.ndarray] = {}
    table = [[] for _ in cols]
    for _ in range(int(natoms)):
        toks = lines[i].split()
        if len(toks) < len(cols):
            raise ValueError(f"Short atom line in {path}: {lines[i]!r}")
        for j, tok in enumerate(toks[: len(cols)]):
            table[j].append(float(tok))
        i += 1
    for c, vals in zip(cols, table):
        data_map[str(c)] = np.asarray(vals, dtype=float)
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
    F = np.linalg.solve(H0.T, H1.T).T
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
    typ = np.asarray(local_types, dtype=int).reshape(-1)
    stress_vol = np.asarray(local_stress_volume, dtype=float)
    if born21.size != 21:
        raise ValueError(f"born21 must have length 21, got {born21.size}")
    if global_stress_voigt.size != 6:
        raise ValueError(f"global_stress_voigt must have length 6, got {global_stress_voigt.size}")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"local_positions must have shape (N,3), got {pos.shape}")
    if stress_vol.shape != (pos.shape[0], 6):
        raise ValueError(f"local_stress_volume must have shape (N,6), got {stress_vol.shape}")
    if typ.size != pos.shape[0]:
        raise ValueError("local_types length mismatch")
    vol = float(volume)
    nat = int(n_atoms)
    if nat < 1:
        nat = int(pos.shape[0])
    if not (math.isfinite(vol) and vol > 0.0 and nat > 0):
        raise ValueError("volume must be finite and > 0; n_atoms must be > 0")

    C_energy = born21_to_matrix(born21)
    C_native = C_energy / vol
    stress_native = np.asarray(global_stress_voigt, dtype=float) / vol
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
        flags.append("stress_hotspots")

    summary: dict[str, Any] = {
        "status": "ok",
        "kind": "static_born_term_plus_local_virial",
        "note": (
            "Born matrix is a global elastic-response fingerprint. The local map uses "
            "mean-volume-normalized per-atom virial stress as a hotspot proxy."
        ),
        "units": units,
        "volume": float(vol),
        "n_atoms": int(nat),
        "born_matrix_native": C_native.tolist(),
        "born_matrix_GPa": C_gpa,
        "global_stress_voigt_native": stress_native.tolist(),
        "global_stress_voigt_GPa": stress_gpa,
        "voigt_bulk_modulus_native": float(K),
        "voigt_bulk_modulus_GPa": float(K * gpa_fac) if gpa_fac is not None else None,
        "voigt_shear_modulus_native": float(G),
        "voigt_shear_modulus_GPa": float(G * gpa_fac) if gpa_fac is not None else None,
        "isotropy_residual": float(isotropy_residual),
        "normal_shear_coupling_norm": float(normal_shear_coupling_norm),
        "diag_spread_rel": float(diag_spread_rel),
        "offdiag_spread_rel": float(offdiag_spread_rel),
        "shear_spread_rel": float(shear_spread_rel),
        "born_eigenvalues_native": [float(x) for x in eig.tolist()],
        "local_stress_summary": {
            "hydrostatic_native": {
                "p05": _nanpercentile(hydro, 5.0),
                "p50": _nanpercentile(hydro, 50.0),
                "p95": _nanpercentile(hydro, 95.0),
            },
            "von_mises_native": {
                "p05": _nanpercentile(vm, 5.0),
                "p50": _nanpercentile(vm, 50.0),
                "p95": _nanpercentile(vm, 95.0),
                "max": float(vm_max),
                "max_over_median": float(hotspot_ratio),
            },
        },
        "flags": flags,
        "force_isotropic": bool(force_isotropic),
    }

    if bool(force_isotropic) and input_cell is not None:
        try:
            summary["affine_isotropization"] = affine_isotropization_strain(np.asarray(input_cell, dtype=float))
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
) -> None:
    ids = np.asarray(ids, dtype=int).reshape(-1)
    types = np.asarray(types, dtype=int).reshape(-1)
    pos = np.asarray(positions, dtype=float)
    sv = np.asarray(stress_volume, dtype=float)
    sn = np.asarray(stress_native, dtype=float)
    hydro = np.asarray(hydrostatic_native, dtype=float).reshape(-1)
    vm = np.asarray(von_mises_native, dtype=float).reshape(-1)
    n = int(ids.size)
    if pos.shape != (n, 3) or sv.shape != (n, 6) or sn.shape != (n, 6) or hydro.size != n or vm.size != n or types.size != n:
        raise ValueError("write_local_stress_csv: inconsistent array sizes")
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


def load_elastic_screen_summary(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
