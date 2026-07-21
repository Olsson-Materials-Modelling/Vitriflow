from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# avogadro constant exact
_N_A = 6.02214076e23

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.data import atomic_masses, atomic_numbers
except Exception as e:  # pragma: no cover
    raise ImportError(
        "vitriflow.structuregen requires 'ase'. Install via pip/conda: pip install ase"
    ) from e

from .config import RunConfig, StructureGenerateConfig
from .analysis.provenance import file_identity, file_identity_matches, write_json_strict
from .io.ase_compat import ase_read_lammps_data
from .lammps_units import (
    charge_from_elementary_factor,
    length_from_angstrom_factor,
    mass_from_amu_factor,
    normalize_lammps_units_style,
)
from .utils import ensure_dir


_COD_RESULT_ENDPOINT = "https://www.crystallography.net/cod/result"
_COD_CIF_BASE = "https://www.crystallography.net/cod"


@dataclass(frozen=True)
class StructureProvenance:
    method: str
    formula: str
    source: str
    cod_id: Optional[int] = None
    cif_url: Optional[str] = None
    materials_project_id: Optional[str] = None
    repeat: Optional[tuple[int, int, int]] = None
    n_atoms: Optional[int] = None
    fallback_from: Optional[str] = None
    fallback_reason: Optional[str] = None


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Dict[str, int]:
    """Formula."""
    if formula is None or str(formula).strip() == "":
        raise ValueError("Empty formula")
    tokens = _FORMULA_RE.findall(str(formula).strip())
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula}")
    counts: Dict[str, int] = {}
    for el, num in tokens:
        n = int(num) if num else 1
        counts[el] = counts.get(el, 0) + n
    return counts


def hill_formula_with_spaces(counts: Dict[str, int]) -> str:
    """Hill formula with."""
    els = list(counts.keys())
    if "C" in counts:
        order = ["C"]
        if "H" in counts:
            order.append("H")
        rest = sorted([e for e in els if e not in set(order)])
        order.extend(rest)
    else:
        order = sorted(els)

    parts = []
    for el in order:
        n = int(counts[el])
        if n == 1:
            parts.append(f"{el}")
        else:
            parts.append(f"{el}{n}")
    return " ".join(parts)


def _download_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "vitriflow/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vitriflow/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def resolve_cif_url(cfg: StructureGenerateConfig) -> str:
    """Cif url."""
    if cfg.method == "cif_url":
        assert cfg.cif_url is not None
        return str(cfg.cif_url)

    if cfg.method == "cod":
        if cfg.cod_id is not None:
            return f"{_COD_CIF_BASE}/{int(cfg.cod_id)}.cif"

        counts = parse_formula(cfg.formula)
        hill = hill_formula_with_spaces(counts)
        q = urllib.parse.urlencode({"formula": hill, "format": "urls"})
        url = f"{_COD_RESULT_ENDPOINT}?{q}"
        txt = _download_text(url)
        urls = [ln.strip() for ln in txt.splitlines() if ln.strip().startswith("http")]
        if not urls:
            raise ValueError(f"COD query returned no results for formula '{hill}'")
        return urls[0]

    raise ValueError(f"Unsupported method for CIF resolution: {cfg.method}")


def _infer_repeat_for_target_atoms(atoms: Atoms, target_atoms: int) -> tuple[int, int, int]:
    """Repeat for target."""
    n_cell = int(atoms.get_global_number_of_atoms())
    if n_cell < 1:
        return (1, 1, 1)
    if int(target_atoms) <= n_cell:
        return (1, 1, 1)

    # approximate cell lengths
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    lens = [float(np.linalg.norm(cell[i])) for i in range(3)]
    # division zero
    lens = [x if x > 1.0e-12 else 1.0 for x in lens]

    mult_target = float(target_atoms) / float(n_cell)
    m0 = int(math.ceil(mult_target ** (1.0 / 3.0)))
    nmax = max(1, m0 + 3)

    best = None
    best_score = float("inf")
    best_atoms = float("inf")
    w_anis = 0.25
    under_penalty = 10.0

    for nx in range(1, nmax + 1):
        for ny in range(1, nmax + 1):
            for nz in range(1, nmax + 1):
                prod = nx * ny * nz
                nat = n_cell * prod
                if nat < int(target_atoms):
                    # strongly discourage undershooting
                    pen = under_penalty * (float(target_atoms) - float(nat)) / float(target_atoms)
                else:
                    pen = (float(nat) - float(target_atoms)) / float(target_atoms)

                Ls = [nx * lens[0], ny * lens[1], nz * lens[2]]
                anis = max(Ls) / min(Ls)
                score = float(pen) + float(w_anis) * float(max(0.0, anis - 1.0))

                if score < best_score - 1.0e-12 or (
                    abs(score - best_score) <= 1.0e-12 and nat < best_atoms
                ):
                    best = (nx, ny, nz)
                    best_score = score
                    best_atoms = float(nat)

    if best is None:
        mm = max(1, m0)
        return (mm, mm, mm)
    return tuple(int(x) for x in best)


def _infer_repeat_for_target_atoms_cubic_first(atoms: Atoms, target_atoms: int) -> tuple[int, int, int]:
    """Repeat for target."""
    n_cell = int(atoms.get_global_number_of_atoms())
    if n_cell < 1:
        return (1, 1, 1)
    if int(target_atoms) <= n_cell:
        return (1, 1, 1)

    # approximate lattice lengths
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    lens = [float(np.linalg.norm(cell[i])) for i in range(3)]
    lens = [x if x > 1.0e-12 else 1.0 for x in lens]

    mult_target = float(target_atoms) / float(n_cell)
    m0 = int(math.ceil(mult_target ** (1.0 / 3.0)))
    # neighbourhood anisotropy improved
    # modest overshoot multipliers
    nmax = max(1, m0 + 4)

    best = None
    best_anis = float("inf")
    best_atoms = float("inf")
    eps = 1.0e-12

    for nx in range(1, nmax + 1):
        for ny in range(1, nmax + 1):
            for nz in range(1, nmax + 1):
                prod = nx * ny * nz
                nat = n_cell * prod
                if nat < int(target_atoms):
                    continue

                Ls = [nx * lens[0], ny * lens[1], nz * lens[2]]
                anis = max(Ls) / min(Ls)

                if anis < best_anis - eps or (abs(anis - best_anis) <= eps and nat < best_atoms):
                    best = (nx, ny, nz)
                    best_anis = float(anis)
                    best_atoms = float(nat)

    if best is None:
        mm = max(1, m0)
        return (mm, mm, mm)
    return tuple(int(x) for x in best)


def _try_compute_formula_units_per_cell(atoms: Atoms, formula_counts: Dict[str, int]) -> Optional[int]:
    comp: Dict[str, int] = {}
    for s in atoms.get_chemical_symbols():
        comp[s] = comp.get(s, 0) + 1
    ratios = []
    for el, n in formula_counts.items():
        if el not in comp:
            return None
        ratios.append(comp[el] / n)
    if not ratios:
        return None
    # ratios equal integer
    r0 = ratios[0]
    if any(abs(r - r0) > 1e-6 for r in ratios[1:]):
        return None
    nfu = int(round(r0))
    if abs(r0 - nfu) > 1e-6:
        return None
    return max(1, nfu)


def _scale_atoms_to_density(atoms: Atoms, rho_g_cm3: float) -> Atoms:
    """Scale atoms to."""
    rho = float(rho_g_cm3)
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("target density must be a finite, positive number")

    # volume
    V0_A3 = float(atoms.get_volume())
    if not math.isfinite(V0_A3) or V0_A3 <= 0:
        raise ValueError("invalid initial cell volume")

    # total mass atomic
    m_amu = np.asarray(atoms.get_masses(), dtype=float)
    if m_amu.size == 0 or not np.all(np.isfinite(m_amu)):
        raise ValueError("invalid atomic masses")
    m_g = float(np.sum(m_amu)) / _N_A  # g

    # target volume 1e
    Vt_A3 = (m_g / rho) / 1.0e-24
    if not math.isfinite(Vt_A3) or Vt_A3 <= 0:
        raise ValueError("invalid target volume")

    s = float((Vt_A3 / V0_A3) ** (1.0 / 3.0))
    if not math.isfinite(s) or s <= 0:
        raise ValueError("invalid density scaling factor")

    # scale cell positions
    atoms = atoms.copy()
    atoms.set_cell(atoms.get_cell() * s, scale_atoms=True)
    atoms.set_pbc(True)
    return atoms


def _packing_density_g_cm3(cfg: StructureGenerateConfig) -> float:
    if cfg.target_density_g_cm3 is not None:
        return float(cfg.target_density_g_cm3)
    if cfg.packing_density_g_cm3 is not None:
        return float(cfg.packing_density_g_cm3)
    return float(cfg.random_fallback_density_g_cm3)


def _packing_min_distance_A(cfg: StructureGenerateConfig) -> float:
    if cfg.packing_min_distance_A is not None:
        return float(cfg.packing_min_distance_A)
    return float(cfg.random_min_distance)


def _effective_formula_units(cfg: StructureGenerateConfig) -> int:
    nfu = int(cfg.n_formula_units)
    if cfg.min_atoms is not None:
        atoms_per_fu = sum(int(v) for v in parse_formula(cfg.formula).values())
        if atoms_per_fu > 0:
            nfu = max(nfu, int(math.ceil(float(cfg.min_atoms) / float(atoms_per_fu))))
    return max(1, int(nfu))


def _total_mass_g_per_mol(counts: Dict[str, int]) -> float:
    mass_g_per_mol = 0.0
    for el, n in counts.items():
        Z = atomic_numbers.get(str(el))
        if Z is None:
            raise ValueError(f"Unknown element symbol: {el}")
        mass_g_per_mol += float(atomic_masses[int(Z)]) * float(n)
    return float(mass_g_per_mol)


def _packing_box_length_A(counts: Dict[str, int], n_formula_units: int, rho_g_cm3: float) -> float:
    rho = float(rho_g_cm3)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("packing density must be finite and > 0")
    mass_g = (_total_mass_g_per_mol(counts) * float(n_formula_units)) / _N_A
    vol_cm3 = mass_g / rho
    vol_A3 = vol_cm3 * 1.0e24
    L = float(vol_A3 ** (1.0 / 3.0))
    if not math.isfinite(L) or L <= 0.0:
        raise ValueError("invalid packing box length")
    return L


def _reduced_counts(counts: Dict[str, int]) -> Dict[str, int]:
    vals = [int(v) for v in counts.values() if int(v) > 0]
    if not vals:
        return {}
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(int(g), int(v))
    if g < 1:
        g = 1
    return {str(k): int(v) // int(g) for k, v in counts.items() if int(v) > 0}


def _atoms_counts(atoms: Atoms) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in atoms.get_chemical_symbols():
        out[str(s)] = out.get(str(s), 0) + 1
    return out


def _normalize_cmd(cmd: Sequence[str] | str) -> list[str]:
    if isinstance(cmd, str):
        toks = [x for x in shlex.split(cmd) if str(x).strip() != ""]
    else:
        toks = [str(x).strip() for x in list(cmd) if str(x).strip() != ""]
    if len(toks) == 0:
        raise ValueError("Command must be non-empty")
    return toks


def _resolve_mp_api_key(cfg: StructureGenerateConfig) -> str:
    if cfg.mp_api_key is not None and str(cfg.mp_api_key).strip() != "":
        return str(cfg.mp_api_key).strip()
    env_name = str(cfg.mp_api_key_env).strip() if cfg.mp_api_key_env is not None else ""
    if env_name != "":
        val = os.environ.get(env_name)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    raise ValueError(
        "materials_project generation requires an API key via structure.generate.mp_api_key "
        "or the environment variable named by structure.generate.mp_api_key_env"
    )


def _load_materials_project_atoms(
    cfg: StructureGenerateConfig, *, outdir: Path
) -> tuple[Atoms, StructureProvenance]:
    api_key = _resolve_mp_api_key(cfg)
    mpid = str(cfg.mp_material_id).strip()
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "materials_project generation requires the optional 'mp-api' package"
        ) from e

    with MPRester(api_key) as mpr:  # type: ignore[misc]
        try:
            structure = mpr.get_structure_by_material_id(mpid, final=True, conventional_unit_cell=True)
        except TypeError:
            try:
                structure = mpr.get_structure_by_material_id(mpid, final=True)
            except TypeError:
                structure = mpr.get_structure_by_material_id(mpid)

    if structure is None:
        raise ValueError(f"Materials Project returned no structure for {mpid}")

    if hasattr(structure, "is_ordered") and not bool(getattr(structure, "is_ordered")):
        raise ValueError(f"Materials Project structure {mpid} is disordered and cannot be converted safely")

    atoms: Optional[Atoms] = None
    if hasattr(structure, "to_ase_atoms"):
        atoms = structure.to_ase_atoms()
    if atoms is None:
        try:
            from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore

            atoms = AseAtomsAdaptor.get_atoms(structure)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert Materials Project structure {mpid} to ASE atoms"
            ) from e

    if not isinstance(atoms, Atoms):
        raise TypeError(f"Materials Project structure conversion did not return ASE Atoms for {mpid}")

    atoms.set_pbc(True)

    actual_red = _reduced_counts(_atoms_counts(atoms))
    requested_red = _reduced_counts(parse_formula(cfg.formula))
    if actual_red != requested_red:
        raise ValueError(
            f"Materials Project structure {mpid} composition {actual_red} does not match requested formula {requested_red}"
        )

    try:
        (outdir / "source_mp.json").write_text(
            json.dumps({"source": "materials_project", "material_id": mpid}, indent=2)
        )
    except Exception:
        pass

    prov = StructureProvenance(
        method="materials_project",
        formula=str(cfg.formula),
        source=f"materials_project:{mpid}",
        materials_project_id=mpid,
    )
    return atoms, prov


def _load_crystal_source_atoms(
    cfg: StructureGenerateConfig, *, outdir: Path
) -> tuple[Atoms, StructureProvenance]:
    cif_url: Optional[str] = None
    cod_id: Optional[int] = None

    if cfg.method == "builtin":
        atoms = builtin_atoms(cfg)
        atoms.set_pbc(True)
        prov = StructureProvenance(
            method=str(cfg.method),
            formula=str(cfg.formula),
            source=f"builtin:{cfg.builtin_name}",
        )
        return atoms, prov

    if cfg.method == "poscar":
        poscar = Path(str(cfg.poscar_path)).expanduser().resolve()
        atoms = read(str(poscar), format="vasp")
        if not isinstance(atoms, Atoms):
            atoms = atoms[0]
        atoms.set_pbc(True)
        try:
            shutil.copy2(poscar, outdir / "source.POSCAR")
        except Exception:
            pass
        prov = StructureProvenance(
            method=str(cfg.method),
            formula=str(cfg.formula),
            source=f"poscar:{poscar}",
        )
        return atoms, prov

    if cfg.method == "materials_project":
        return _load_materials_project_atoms(cfg, outdir=outdir)

    if cfg.method not in ("cod", "cif_url"):
        raise ValueError(f"Unsupported crystal structure source: {cfg.method}")

    cif_url = resolve_cif_url(cfg)
    cif_path = outdir / "source.cif"
    cif_path.write_bytes(_download_bytes(cif_url))

    atoms = read(str(cif_path))
    if not isinstance(atoms, Atoms):
        atoms = atoms[0]
    atoms.set_pbc(True)

    source = "COD" if cfg.method == "cod" else "cif_url"
    cod_id = int(cfg.cod_id) if cfg.cod_id is not None else None
    prov = StructureProvenance(
        method=str(cfg.method),
        formula=str(cfg.formula),
        source=str(source),
        cod_id=cod_id,
        cif_url=str(cif_url) if cif_url is not None else None,
    )
    return atoms, prov


def _generate_random_atoms(
    cfg: StructureGenerateConfig,
    *,
    outdir: Path,
    fallback_from: Optional[str] = None,
    fallback_reason: Optional[str] = None,
) -> tuple[Atoms, StructureProvenance]:
    counts = parse_formula(cfg.formula)
    symbols_fu: list[str] = []
    for el, n in counts.items():
        symbols_fu.extend([el] * int(n))

    nfu = _effective_formula_units(cfg)
    symbols: list[str] = symbols_fu * int(nfu)
    rho = _packing_density_g_cm3(cfg)
    L = _packing_box_length_A(counts, nfu, rho)

    rng = np.random.default_rng(int(cfg.seed))
    pos = np.zeros((len(symbols), 3), dtype=float)
    min_d = _packing_min_distance_A(cfg)
    max_attempts = 20000

    for i in range(len(symbols)):
        ok = False
        for _ in range(max_attempts):
            p = rng.random(3) * L
            if i == 0:
                pos[i] = p
                ok = True
                break
            d = p - pos[:i]
            d -= np.round(d / L) * L
            if np.all(np.linalg.norm(d, axis=1) > min_d):
                pos[i] = p
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Failed to place atom {i+1}/{len(symbols)} with min_distance={min_d} Å. "
                "Try decreasing random_min_distance / packing_min_distance_A or increasing the box size."
            )

    atoms = Atoms(symbols=symbols, positions=pos, cell=[L, L, L], pbc=True)
    rep = None
    if cfg.repeat is not None:
        rep = tuple(int(x) for x in cfg.repeat)
        atoms = atoms.repeat(rep)
    if cfg.target_density_g_cm3 is not None:
        atoms = _scale_atoms_to_density(atoms, float(cfg.target_density_g_cm3))

    prov = StructureProvenance(
        method="random",
        formula=str(cfg.formula),
        source="random",
        repeat=rep,
        n_atoms=int(atoms.get_global_number_of_atoms()),
        fallback_from=fallback_from,
        fallback_reason=fallback_reason,
    )
    return atoms, prov


def _generate_packmol_atoms(
    cfg: StructureGenerateConfig, *, outdir: Path
) -> tuple[Atoms, StructureProvenance]:
    counts = parse_formula(cfg.formula)
    nfu = _effective_formula_units(cfg)
    rho = _packing_density_g_cm3(cfg)
    min_d = _packing_min_distance_A(cfg)
    L = _packing_box_length_A(counts, nfu, rho)

    element_counts: Dict[str, int] = {str(el): int(n) * int(nfu) for el, n in counts.items()}

    for el in sorted(element_counts):
        tpl = outdir / f"template_{el}.xyz"
        tpl.write_text(f"1\n{el} template\n{el} 0.0 0.0 0.0\n")

    lines = [
        f"tolerance {min_d:.8f}",
        "filetype xyz",
        "output packmol_output.xyz",
        f"seed {int(cfg.seed)}",
        f"pbc 0.0 0.0 0.0 {L:.8f} {L:.8f} {L:.8f}",
        "",
    ]
    for el in sorted(element_counts):
        n = int(element_counts[el])
        lines.extend(
            [
                f"structure template_{el}.xyz",
                f"  number {n}",
                f"  inside box 0.0 0.0 0.0 {L:.8f} {L:.8f} {L:.8f}",
                "end structure",
                "",
            ]
        )
    inp_text = "\n".join(lines).rstrip() + "\n"
    (outdir / "packmol.inp").write_text(inp_text)

    cmd = _normalize_cmd(cfg.packmol_cmd)
    inp_path = outdir / "packmol.inp"
    with inp_path.open("r", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            stdin=fh,
            cwd=str(outdir),
            capture_output=True,
            text=True,
            check=False,
        )
    (outdir / "packmol.stdout").write_text(proc.stdout or "")
    (outdir / "packmol.stderr").write_text(proc.stderr or "")

    out_xyz = outdir / "packmol_output.xyz"
    if proc.returncode != 0 or (not out_xyz.exists()):
        raise RuntimeError(
            "Packmol failed to generate a structure. "
            f"Return code={proc.returncode}. See {outdir / 'packmol.stdout'} and {outdir / 'packmol.stderr'}."
        )

    atoms = read(str(out_xyz))
    if not isinstance(atoms, Atoms):
        atoms = atoms[0]
    atoms.set_cell([L, L, L], scale_atoms=False)
    atoms.set_pbc(True)

    rep = None
    if cfg.repeat is not None:
        rep = tuple(int(x) for x in cfg.repeat)
        atoms = atoms.repeat(rep)

    if cfg.target_density_g_cm3 is not None:
        atoms = _scale_atoms_to_density(atoms, float(cfg.target_density_g_cm3))

    prov = StructureProvenance(
        method="packmol",
        formula=str(cfg.formula),
        source="packmol",
        repeat=rep,
        n_atoms=int(atoms.get_global_number_of_atoms()),
    )
    return atoms, prov



def builtin_atoms(cfg: StructureGenerateConfig) -> Atoms:
    """Builtin atoms."""
    name = str(cfg.builtin_name).strip().lower() if cfg.builtin_name is not None else ""
    if name in ("si_diamond", "diamond_si", "si_diamond_conventional"):
        # conventional cell diamond
        # intended lightweight smoke
        # intended lightweight smoke
        # periodic semiconductor download
        # dependency scale size
        # a
        a = 5.431  # reference k

        frac = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.25, 0.25],
                [0.00, 0.50, 0.50],
                [0.25, 0.75, 0.75],
                [0.50, 0.00, 0.50],
                [0.75, 0.25, 0.75],
                [0.50, 0.50, 0.00],
                [0.75, 0.75, 0.25],
            ],
            dtype=float,
        )
        pos = frac * float(a)
        atoms = Atoms(symbols=["Si"] * 8, positions=pos, cell=[a, a, a], pbc=True)
        atoms.set_pbc(True)
        return atoms
    if name in ("beta_cristobalite", "cristobalite_beta", "cristobalite"):
        # cristobalite conventional cell
        # rationale
        # rationale
        # several cristobalite occupancy
        # atom conventional cells
        # seed
        # external
        # construction
        # construction
        # diamond network cell
        # midpoints neighbour bonds
        # cell reference density
        # cell reference density
        # a
        a = 7.16  # reference

        # diamond fractional coordinates
        fcc = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )
        shift = np.array([0.25, 0.25, 0.25], dtype=float)
        frac_si = np.vstack([fcc, (fcc + shift) % 1.0])
        pos_si = frac_si * float(a)

        # nearest neighbour bonds
        r_nn = float(math.sqrt(3.0) * float(a) / 4.0)
        r_cut = 1.05 * r_nn

        # bond midpoints convention
        mids: list[np.ndarray] = []
        for i in range(len(pos_si)):
            for j in range(i + 1, len(pos_si)):
                d = pos_si[j] - pos_si[i]
                d -= float(a) * np.round(d / float(a))
                r = float(np.linalg.norm(d))
                if r <= r_cut:
                    m = (pos_si[i] + 0.5 * d) % float(a)
                    mids.append(m)

        # deduplicate midpoints numerical
        uniq: dict[tuple[float, float, float], np.ndarray] = {}
        for m in mids:
            key = tuple(np.round(m, 6))
            uniq[key] = m
        pos_o = np.array(list(uniq.values()), dtype=float)

        if pos_o.shape[0] != 16:
            # bond detection tolerance
            raise ValueError(
                f"builtin beta_cristobalite: expected 16 O midpoints, got {pos_o.shape[0]}"
            )

        symbols = ["Si"] * 8 + ["O"] * 16
        positions = np.vstack([pos_si, pos_o])

        atoms = Atoms(symbols=symbols, positions=positions, cell=[a, a, a], pbc=True)
        atoms.set_pbc(True)
        return atoms

    raise ValueError(f"Unknown builtin structure: {cfg.builtin_name}")


def generate_atoms(cfg: StructureGenerateConfig, *, outdir: Path) -> tuple[Atoms, StructureProvenance]:
    """Generate atoms."""
    ensure_dir(outdir)

    if cfg.method in ("cod", "cif_url", "materials_project"):
        try:
            atoms, prov0 = _load_crystal_source_atoms(cfg, outdir=outdir)
        except Exception as e:
            if bool(cfg.fallback_to_random):
                return _generate_random_atoms(
                    cfg,
                    outdir=outdir,
                    fallback_from=str(cfg.method),
                    fallback_reason=str(e),
                )
            raise
    elif cfg.method in ("builtin", "poscar"):
        atoms, prov0 = _load_crystal_source_atoms(cfg, outdir=outdir)
    elif cfg.method == "packmol":
        try:
            return _generate_packmol_atoms(cfg, outdir=outdir)
        except Exception as e:
            if bool(cfg.fallback_to_random):
                return _generate_random_atoms(
                    cfg,
                    outdir=outdir,
                    fallback_from="packmol",
                    fallback_reason=str(e),
                )
            raise
    elif cfg.method == "random":
        return _generate_random_atoms(cfg, outdir=outdir)
    else:
        raise ValueError(f"Unsupported structure generation method: {cfg.method}")

    atoms.set_pbc(True)

    formula_counts = parse_formula(cfg.formula)
    nfu_cell = _try_compute_formula_units_per_cell(atoms, formula_counts)

    # determine replication policy
    # poscars behaviour respect
    # poscars behaviour respect
    # supercell therefore expand
    # explicitly requests yaml
    # otherwise we repeat
    nfu_explicit = "n_formula_units" in getattr(cfg, "model_fields_set", set())

    if cfg.repeat is not None:
        rep = tuple(int(x) for x in cfg.repeat)
    elif cfg.method == "poscar" and (cfg.min_atoms is None) and (not nfu_explicit):
        rep = (1, 1, 1)
    else:
        atoms_per_fu = None
        if nfu_cell is not None and nfu_cell > 0:
            atoms_per_fu = atoms.get_global_number_of_atoms() / float(nfu_cell)

        if atoms_per_fu is not None:
            target_atoms = int(math.ceil(cfg.n_formula_units * atoms_per_fu))
        else:
            target_atoms = int(cfg.n_formula_units * sum(int(v) for v in formula_counts.values()))

        if cfg.min_atoms is not None:
            target_atoms = max(int(target_atoms), int(cfg.min_atoms))

        if cfg.method == "poscar":
            rep = _infer_repeat_for_target_atoms_cubic_first(atoms, int(target_atoms))
        else:
            rep = _infer_repeat_for_target_atoms(atoms, int(target_atoms))

    atoms = atoms.repeat(rep)

    if cfg.target_density_g_cm3 is not None:
        atoms = _scale_atoms_to_density(atoms, float(cfg.target_density_g_cm3))

    prov = StructureProvenance(
        method=str(prov0.method),
        formula=str(cfg.formula),
        source=str(prov0.source),
        cod_id=prov0.cod_id,
        cif_url=prov0.cif_url,
        materials_project_id=prov0.materials_project_id,
        repeat=tuple(int(x) for x in rep),
        n_atoms=int(atoms.get_global_number_of_atoms()),
        fallback_from=prov0.fallback_from,
        fallback_reason=prov0.fallback_reason,
    )
    return atoms, prov


def _infer_specorder(config: RunConfig, atoms: Atoms) -> list[str]:
    """Specorder."""
    # prefer mapping recommended
    if config.autotune.metrics.type_to_species is not None:
        return [str(x) for x in config.autotune.metrics.type_to_species]

    # lammps species order
    if config.kim is not None and config.kim.interactions != "fixed_types":
        return [str(x) for x in config.kim.interactions]

    # fallback
    seen: list[str] = []
    for s in atoms.get_chemical_symbols():
        if s not in seen:
            seen.append(s)
    return seen


def write_lammps_data(
    path: Path,
    atoms: Atoms,
    *,
    specorder: Optional[Sequence[str]] = None,
    atom_style: str = "atomic",
    units_style: str = "metal",
) -> None:
    """Write ASE atoms in the dimensions selected by ``units_style``.

    LAMMPS data files do not contain a unit declaration; values are interpreted
    by the preceding ``units`` command.  ASE coordinates are Angstrom, masses
    are atomic-mass units, and initial charges are multiples of ``e``, so each
    field must be converted before serialization for non-metal/real styles.
    """
    units = normalize_lammps_units_style(units_style)
    length_factor = length_from_angstrom_factor(units)
    mass_factor = mass_from_amu_factor(units)
    charge_factor = charge_from_elementary_factor(units)
    spec = list(specorder) if specorder is not None else []
    if len(spec) == 0:
        # fallback
        seen: list[str] = []
        for s in atoms.get_chemical_symbols():
            if s not in seen:
                seen.append(s)
        spec = seen

    # symbols lammps indexed
    sym_to_type = {str(sym): i + 1 for i, sym in enumerate(spec)}
    ntypes = len(spec)

    # lammps cell decomposition
    cell_angstrom = np.asarray(atoms.get_cell().array, dtype=float)
    if cell_angstrom.shape != (3, 3) or not np.all(np.isfinite(cell_angstrom)):
        raise ValueError("Invalid or non-finite cell for LAMMPS data output")
    singular_values = np.linalg.svd(cell_angstrom, compute_uv=False)
    if singular_values.size != 3 or float(singular_values[-1]) <= (
        np.finfo(float).eps * max(1.0, float(singular_values[0])) * 16.0
    ):
        raise ValueError("Invalid or degenerate cell for LAMMPS data output")
    cell = cell_angstrom * float(length_factor)
    B = cell.T  # columns lattice vectors

    Q, R = np.linalg.qr(B)
    # enforce positive diagonal
    for i in range(3):
        if R[i, i] < 0:
            R[i, :] *= -1.0
            Q[:, i] *= -1.0

    # rotated positions rows
    pos = np.asarray(atoms.get_positions(), dtype=float) * float(length_factor)
    pos_rot = pos @ Q

    # cell coordinates respect
    invR = np.linalg.inv(R)
    s = pos_rot @ invR.T
    s = s - np.floor(s)
    pos_wrap = s @ R.T

    # parameters lammps expects
    lx = float(R[0, 0])
    ly = float(R[1, 1])
    lz = float(R[2, 2])
    xy = float(R[0, 1])
    xz = float(R[0, 2])
    yz = float(R[1, 2])

    # masses
    masses: list[float] = []
    for sym in spec:
        Z = atomic_numbers.get(str(sym))
        if Z is None:
            masses.append(1.0 * float(mass_factor))
        else:
            m = float(atomic_masses[int(Z)])
            masses.append((m if m > 0 else 1.0) * float(mass_factor))

    # charges
    use_charge = str(atom_style).lower() == "charge"
    charges = None
    if use_charge:
        try:
            charges = np.asarray(atoms.get_initial_charges(), dtype=float)
        except Exception:
            charges = None
        if charges is None or len(charges) != len(pos_wrap):
            charges = np.zeros(len(pos_wrap), dtype=float)

    # lines
    lines: list[str] = []
    lines.append(f"(written by vitriflow; LAMMPS units {units})")
    lines.append("")
    lines.append(f"{len(pos_wrap)} atoms")
    lines.append(f"{ntypes} atom types")
    lines.append("")
    lines.append(f"0.0 {lx:.16g} xlo xhi")
    lines.append(f"0.0 {ly:.16g} ylo yhi")
    lines.append(f"0.0 {lz:.16g} zlo zhi")
    # include factors triclinic
    if abs(xy) > 1.0e-12 or abs(xz) > 1.0e-12 or abs(yz) > 1.0e-12:
        lines.append(f"{xy:.16g} {xz:.16g} {yz:.16g} xy xz yz")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    for t, m in enumerate(masses, start=1):
        lines.append(f"{t} {m:.16g}")
    lines.append("")

    style_tag = "charge" if use_charge else "atomic"
    lines.append(f"Atoms # {style_tag}")
    lines.append("")
    for i, (sym, r) in enumerate(zip(atoms.get_chemical_symbols(), pos_wrap), start=1):
        t = sym_to_type.get(str(sym))
        if t is None:
            raise ValueError(f"Symbol '{sym}' not present in specorder mapping")
        if use_charge:
            q = float(charges[i - 1]) * float(charge_factor)
            lines.append(f"{i} {t} {q:.16g} {r[0]:.16g} {r[1]:.16g} {r[2]:.16g}")
        else:
            lines.append(f"{i} {t} {r[0]:.16g} {r[1]:.16g} {r[2]:.16g}")

    path.write_text("\n".join(lines) + "\n")


def _initial_structure_cache_spec(config: RunConfig) -> dict[str, object]:
    gen = config.structure.generate
    gen_spec = None if gen is None else dict(gen.model_dump(mode="json"))

    charges = None
    if config.structure.charges is not None:
        charges = {str(k): float(v) for k, v in sorted(config.structure.charges.items())}

    try:
        metric_species = getattr(config.autotune.metrics, "type_to_species", None)
        type_to_species = None if metric_species is None else [str(x) for x in metric_species]
    except Exception:
        type_to_species = None

    try:
        kim_interactions_raw = getattr(config.kim, "interactions", None)
        if kim_interactions_raw is None or kim_interactions_raw == "fixed_types":
            kim_interactions = kim_interactions_raw
        else:
            kim_interactions = [str(x) for x in kim_interactions_raw]
    except Exception:
        kim_interactions = None

    source_files: dict[str, object] = {}
    if gen is not None and getattr(gen, "poscar_path", None) is not None:
        poscar_path = Path(gen.poscar_path)
        source_files["poscar_path"] = file_identity(
            poscar_path,
            recorded_path=str(poscar_path),
        )

    return {
        "structure_generate": gen_spec,
        "structure_charges": charges,
        "atom_style": str(config.md.atom_style),
        "type_to_species": type_to_species,
        "kim_interactions": kim_interactions,
        "lammps_units_style": str(getattr(config.kim, "user_units", "metal") or "metal").strip().lower(),
        "source_files": source_files,
    }


def prepare_initial_structure(config: RunConfig, outdir: Path) -> Path:
    """Initial structure."""
    if config.structure.lammps_data is not None:
        return Path(config.structure.lammps_data)

    gen = config.structure.generate
    if gen is None:
        raise ValueError("No structure.lammps_data and no structure.generate")

    struct_dir = outdir / "structure"
    ensure_dir(struct_dir)

    data_path = struct_dir / "initial.data"
    meta_path = struct_dir / "structure_provenance.json"
    cache_spec = _initial_structure_cache_spec(config)

    if data_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cached_identity = meta.get("initial_data_identity") if isinstance(meta, dict) else None
            if (
                isinstance(meta, dict)
                and meta.get("_cache_spec") == cache_spec
                and isinstance(cached_identity, dict)
                and file_identity_matches(data_path, cached_identity)
            ):
                return data_path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    atoms, prov = generate_atoms(gen, outdir=struct_dir)

    # species charge assignment
    if config.structure.charges is not None:
        qmap = {str(k): float(v) for k, v in config.structure.charges.items()}
        charges = []
        for sym in atoms.get_chemical_symbols():
            if sym not in qmap:
                raise ValueError(f"No charge provided for species '{sym}' in structure.charges")
            charges.append(float(qmap[sym]))
        atoms.set_initial_charges(charges)

    # charge charges initialize
    atom_style = str(config.md.atom_style)
    if atom_style == "charge" and config.structure.charges is None:
        atoms.set_initial_charges([0.0] * len(atoms))

    specorder = _infer_specorder(config, atoms)
    units_style = str(getattr(config.kim, "user_units", "metal") or "metal").strip().lower()
    write_lammps_data(
        data_path,
        atoms,
        specorder=specorder,
        atom_style=atom_style,
        units_style=units_style,
    )

    meta = dict(asdict(prov))
    meta["_cache_spec"] = cache_spec
    meta["initial_data_identity"] = file_identity(
        data_path,
        recorded_path=str(data_path.name),
    )
    write_json_strict(meta_path, meta)
    return data_path


def prepare_size_scan_base_structure(
    config: RunConfig,
    outdir: Path,
    initial_data: Path,
) -> tuple[Path, tuple[int, int, int]]:
    """Size scan base."""

    import shutil
    from collections import Counter

    struct_dir = Path(outdir) / "structure"
    ensure_dir(struct_dir)
    base_data = struct_dir / "size_base.data"
    meta_path = struct_dir / "size_base_meta.json"

    atom_style = str(config.md.atom_style)
    units_style = str(getattr(config.kim, "user_units", "metal") or "metal").strip().lower()

    # specorder ase chemical
    specorder = None
    try:
        if config.kim.interactions != "fixed_types":
            specorder = [str(x) for x in config.kim.interactions]
        elif config.autotune.metrics.type_to_species is not None:
            specorder = [str(x) for x in config.autotune.metrics.type_to_species]
    except Exception:
        specorder = None

    # ase version differences
    def _read_atoms(path: Path) -> Atoms:
        try:
            return ase_read_lammps_data(
                path,
                atom_style=str(atom_style),
                specorder=specorder,
                units=units_style,
            )
        except Exception:
            try:
                return ase_read_lammps_data(
                    path,
                    atom_style=str(atom_style),
                    units=units_style,
                )
            except Exception:
                try:
                    return read(str(path), format="lammps-data", units=units_style)
                except Exception as e:
                    # fallback parser
                    try:
                        from .io.lammps_data_minimal import read_lammps_data_minimal

                        return read_lammps_data_minimal(
                            Path(path),
                            atom_style=str(atom_style),
                            specorder=specorder,
                            units_style=units_style,
                        )
                    except Exception:
                        raise RuntimeError(f"Failed to read LAMMPS data file: {path}") from e

    atoms0 = _read_atoms(Path(initial_data))
    atoms0.set_pbc(True)

    # determine repeat structure
    initial_repeat: tuple[int, int, int] = (1, 1, 1)
    prov_path = struct_dir / "structure_provenance.json"
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
            rep = prov.get("repeat", None)
            if isinstance(rep, (list, tuple)) and len(rep) == 3:
                rr = tuple(int(x) for x in rep)
                if all(int(x) >= 1 for x in rr):
                    initial_repeat = rr  # type: ignore[assignment]
        except Exception:
            initial_repeat = (1, 1, 1)

    # repeat translational invariance
    def _infer_repeat(atoms: Atoms, *, max_k: int = 24, tol: float = 2.0e-5) -> tuple[int, int, int]:
        # work fractional coordinates
        s = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
        s = s - np.floor(s)
        Z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        if Z.size != s.shape[0]:
            Z = np.ones((s.shape[0],), dtype=int)

        def key_counter(ss: np.ndarray) -> Counter:
            rr = np.round(ss / float(tol)).astype(np.int64)
            return Counter((int(Z[i]), int(rr[i, 0]), int(rr[i, 1]), int(rr[i, 2])) for i in range(rr.shape[0]))

        base = key_counter(s)

        reps = [1, 1, 1]
        for ax in range(3):
            best = 1
            for k in range(2, int(max_k) + 1):
                shift = 1.0 / float(k)
                ss = np.array(s, copy=True)
                ss[:, ax] = (ss[:, ax] + shift) % 1.0
                if key_counter(ss) == base:
                    best = int(k)
            reps[ax] = int(best)
        return (int(reps[0]), int(reps[1]), int(reps[2]))

    if initial_repeat == (1, 1, 1):
        try:
            initial_repeat = _infer_repeat(atoms0)
        except Exception:
            initial_repeat = (1, 1, 1)

    # reduce inferred repeat
    def _reduce_by_repeat(atoms: Atoms, rep: tuple[int, int, int]) -> Optional[Atoms]:
        rx, ry, rz = (int(rep[0]), int(rep[1]), int(rep[2]))
        if rx <= 1 and ry <= 1 and rz <= 1:
            return atoms.copy()

        s = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
        s = s - np.floor(s)
        # cell supercell tiling
        eps = 1.0e-10
        ix = np.floor(s[:, 0] * float(rx) + eps).astype(int)
        iy = np.floor(s[:, 1] * float(ry) + eps).astype(int)
        iz = np.floor(s[:, 2] * float(rz) + eps).astype(int)
        m = (ix == 0) & (iy == 0) & (iz == 0)
        if not np.any(m):
            return None

        s0 = s[m, :]
        # fractional coordinates cell
        s_red = np.zeros_like(s0)
        s_red[:, 0] = (s0[:, 0] * float(rx)) % 1.0
        s_red[:, 1] = (s0[:, 1] * float(ry)) % 1.0
        s_red[:, 2] = (s0[:, 2] * float(rz)) % 1.0

        cell = np.asarray(atoms.get_cell().array, dtype=float)
        cell_red = np.array(cell, copy=True)
        cell_red[0, :] /= float(rx)
        cell_red[1, :] /= float(ry)
        cell_red[2, :] /= float(rz)

        pos_red = s_red @ cell_red

        syms = atoms.get_chemical_symbols()
        syms_red = [syms[i] for i in np.where(m)[0]]
        a2 = Atoms(symbols=syms_red, positions=pos_red, cell=cell_red, pbc=True)

        # preserve charges present
        try:
            q = np.asarray(atoms.get_initial_charges(), dtype=float)
            if q.size == len(atoms):
                a2.set_initial_charges(q[m])
        except Exception:
            pass

        return a2

    reduced = _reduce_by_repeat(atoms0, initial_repeat)

    # guard
    if reduced is None:
        reduced = atoms0.copy()
        initial_repeat = (1, 1, 1)
    else:
        try:
            n0 = int(atoms0.get_global_number_of_atoms())
            n1 = int(reduced.get_global_number_of_atoms())
            prod = int(initial_repeat[0] * initial_repeat[1] * initial_repeat[2])
            if prod <= 0 or n1 * prod != n0:
                # clean tiling reduce
                reduced = atoms0.copy()
                initial_repeat = (1, 1, 1)
        except Exception:
            reduced = atoms0.copy()
            initial_repeat = (1, 1, 1)

    # always produce reference
    try:
        spec = specorder
        if spec is None or len(spec) == 0:
            # fallback
            spec = _infer_specorder(config, reduced)
        write_lammps_data(
            base_data,
            reduced,
            specorder=spec,
            atom_style=atom_style,
            units_style=units_style,
        )
    except Exception:
        # resort copy original
        try:
            shutil.copy2(Path(initial_data), base_data)
        except Exception as e:
            raise RuntimeError(f"Failed to create size_base.data under {struct_dir}") from e
        initial_repeat = (1, 1, 1)

    meta_path.write_text(json.dumps({"initial_repeat": list(initial_repeat)}, indent=2))
    return base_data, initial_repeat
