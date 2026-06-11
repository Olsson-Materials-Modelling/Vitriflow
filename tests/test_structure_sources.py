from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _install_fake_ase(monkeypatch):
    import numpy as np

    class FakeCell:
        def __init__(self, data):
            arr = np.asarray(data, dtype=float)
            if arr.shape == (3,):
                arr = np.diag(arr)
            self.array = np.asarray(arr, dtype=float)

        def __mul__(self, other):
            return self.array * float(other)

        __rmul__ = __mul__

        def lengths(self):
            return np.asarray([float(np.linalg.norm(self.array[i])) for i in range(3)], dtype=float)

    atomic_numbers = {"Al": 13, "Si": 14, "N": 7, "Na": 11, "Cl": 17}
    atomic_masses = np.zeros(120, dtype=float)
    atomic_masses[7] = 14.0067
    atomic_masses[11] = 22.98976928
    atomic_masses[13] = 26.9815385
    atomic_masses[14] = 28.085
    atomic_masses[17] = 35.45

    class Atoms:
        def __init__(self, symbols, positions, cell=None, pbc=True):
            self._symbols = list(symbols)
            self._positions = np.asarray(positions, dtype=float)
            self._cell = FakeCell([0.0, 0.0, 0.0] if cell is None else cell)
            if isinstance(pbc, bool):
                self._pbc = np.asarray([pbc, pbc, pbc], dtype=bool)
            else:
                self._pbc = np.asarray(list(pbc), dtype=bool)
            self._charges = np.zeros(len(self._symbols), dtype=float)

        def copy(self):
            out = Atoms(self._symbols.copy(), self._positions.copy(), cell=self._cell.array.copy(), pbc=self._pbc.copy())
            out._charges = self._charges.copy()
            return out

        def set_pbc(self, pbc):
            if isinstance(pbc, bool):
                self._pbc[:] = bool(pbc)
            else:
                self._pbc = np.asarray(list(pbc), dtype=bool)

        def get_pbc(self):
            return self._pbc.copy()

        def get_chemical_symbols(self):
            return list(self._symbols)

        def __len__(self):
            return len(self._symbols)

        def get_global_number_of_atoms(self):
            return len(self._symbols)

        def get_volume(self):
            return float(abs(np.linalg.det(self._cell.array)))

        def get_masses(self):
            return np.asarray([atomic_masses[atomic_numbers[s]] for s in self._symbols], dtype=float)

        def get_cell(self):
            return self._cell

        @property
        def cell(self):
            return self._cell

        def set_cell(self, cell, scale_atoms=False):
            new_cell = FakeCell(cell)
            if scale_atoms:
                old_diag = np.diag(self._cell.array)
                new_diag = np.diag(new_cell.array)
                scale = np.divide(new_diag, old_diag, out=np.ones(3, dtype=float), where=np.abs(old_diag) > 1.0e-12)
                self._positions = self._positions * scale[None, :]
            self._cell = new_cell

        def repeat(self, rep):
            nx, ny, nz = (int(rep[0]), int(rep[1]), int(rep[2]))
            ax, by, cz = np.diag(self._cell.array)
            syms = []
            pos = []
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        shift = np.asarray([ix * ax, iy * by, iz * cz], dtype=float)
                        for s, r in zip(self._symbols, self._positions):
                            syms.append(s)
                            pos.append(np.asarray(r, dtype=float) + shift)
            return Atoms(syms, np.asarray(pos, dtype=float), cell=[ax * nx, by * ny, cz * nz], pbc=True)

        def set_initial_charges(self, charges):
            self._charges = np.asarray(charges, dtype=float)

        def get_initial_charges(self):
            return self._charges.copy()

    def read(path, format=None, **kwargs):
        txt = Path(path).read_text().splitlines()
        n = int(txt[0].strip())
        syms = []
        pos = []
        for ln in txt[2:2 + n]:
            t = ln.split()
            syms.append(t[0])
            pos.append([float(t[1]), float(t[2]), float(t[3])])
        return Atoms(syms, pos, cell=[1.0, 1.0, 1.0], pbc=True)

    def write(path, atoms, format=None):
        Path(path).write_text("")

    ase_mod = types.ModuleType("ase")
    io_mod = types.ModuleType("ase.io")
    data_mod = types.ModuleType("ase.data")
    ase_mod.Atoms = Atoms
    io_mod.read = read
    io_mod.write = write
    data_mod.atomic_numbers = atomic_numbers
    data_mod.atomic_masses = atomic_masses
    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.io", io_mod)
    monkeypatch.setitem(sys.modules, "ase.data", data_mod)
    sys.modules.pop("vitriflow.structuregen", None)
    return Atoms



def _install_fake_mp(monkeypatch, *, structure=None, error: Exception | None = None, capture: dict | None = None):
    capture = capture if capture is not None else {}

    class FakeMPRester:
        def __init__(self, api_key):
            capture["api_key"] = api_key

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_structure_by_material_id(self, material_id, **kwargs):
            capture["material_id"] = material_id
            capture["kwargs"] = dict(kwargs)
            if error is not None:
                raise error
            return structure

    mp_api_mod = types.ModuleType("mp_api")
    client_mod = types.ModuleType("mp_api.client")
    client_mod.MPRester = FakeMPRester
    mp_api_mod.client = client_mod
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_mod)
    monkeypatch.setitem(sys.modules, "mp_api.client", client_mod)
    return capture


def test_materials_project_generate_uses_env_key_and_writes_metadata(tmp_path: Path, monkeypatch):
    Atoms = _install_fake_ase(monkeypatch)
    from vitriflow.config import StructureGenerateConfig
    from vitriflow.structuregen import generate_atoms

    class FakeStructure:
        is_ordered = True

        def to_ase_atoms(self):
            return Atoms(
                symbols=["Si", "Si"],
                positions=[[0.0, 0.0, 0.0], [1.35775, 1.35775, 1.35775]],
                cell=[5.431, 5.431, 5.431],
                pbc=True,
            )

    seen: dict[str, object] = {}
    _install_fake_mp(monkeypatch, structure=FakeStructure(), capture=seen)
    monkeypatch.setenv("MP_API_KEY", "test-key")

    cfg = StructureGenerateConfig(
        method="materials_project",
        formula="Si",
        material_id="mp-149",
        mp_api_key_env="MP_API_KEY",
        n_formula_units=2,
    )

    atoms, prov = generate_atoms(cfg, outdir=tmp_path)

    assert len(atoms) == 2
    assert prov.method == "materials_project"
    assert prov.materials_project_id == "mp-149"
    assert prov.repeat == (1, 1, 1)
    assert seen["api_key"] == "test-key"
    assert seen["material_id"] == "mp-149"
    assert (tmp_path / "source_mp.json").exists()


def test_materials_project_failure_can_fallback_to_random(tmp_path: Path, monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.config import StructureGenerateConfig
    from vitriflow.structuregen import generate_atoms

    _install_fake_mp(monkeypatch, error=RuntimeError("mp unavailable"))

    cfg = StructureGenerateConfig(
        method="materials_project",
        formula="NaCl",
        mp_material_id="mp-fail",
        mp_api_key="dummy-key",
        n_formula_units=2,
        fallback_to_random=True,
        random_fallback_density_g_cm3=2.0,
        random_min_distance=0.5,
        seed=7,
    )

    atoms, prov = generate_atoms(cfg, outdir=tmp_path)

    assert len(atoms) == 4
    assert prov.method == "random"
    assert prov.fallback_from == "materials_project"
    assert prov.fallback_reason is not None
    assert "mp unavailable" in prov.fallback_reason


def test_packmol_generate_reads_output_and_repeat(tmp_path: Path, monkeypatch):
    _install_fake_ase(monkeypatch)
    import vitriflow.structuregen as sg
    from vitriflow.config import StructureGenerateConfig
    from vitriflow.structuregen import generate_atoms

    def fake_run(cmd, *, stdin, cwd, capture_output, text, check):
        assert list(cmd) == ["packmol"]
        assert capture_output is True
        assert text is True
        assert check is False
        work = Path(cwd)
        assert Path(stdin.name) == (work / "packmol.inp")
        assert stdin.read().startswith("tolerance 2.20000000")
        (work / "packmol_output.xyz").write_text(
            "2\npackmol\nAl 0.0 0.0 0.0\nAl 1.0 1.0 1.0\n"
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sg.subprocess, "run", fake_run)

    cfg = StructureGenerateConfig(
        method="packmol",
        formula="Al",
        n_formula_units=2,
        repeat=(2, 1, 1),
        packmol_cmd=["packmol"],
        packing_density_g_cm3=2.4,
        packing_min_distance_A=2.2,
        seed=11,
    )

    atoms, prov = generate_atoms(cfg, outdir=tmp_path)

    assert len(atoms) == 4
    assert prov.method == "packmol"
    assert prov.repeat == (2, 1, 1)
    assert atoms.get_pbc().all()
    assert float(atoms.cell.lengths()[0]) > 0.0
    assert (tmp_path / "packmol.inp").exists()
    assert "tolerance 2.20000000" in (tmp_path / "packmol.inp").read_text()
    assert (tmp_path / "packmol.stdout").read_text() == "ok"
    assert (tmp_path / "packmol.stderr").read_text() == ""


def test_packmol_generate_uses_seekable_input_file(tmp_path: Path, monkeypatch):
    _install_fake_ase(monkeypatch)
    import vitriflow.structuregen as sg
    from vitriflow.config import StructureGenerateConfig
    from vitriflow.structuregen import generate_atoms

    seen = {}

    def fake_run(cmd, *, stdin, cwd, capture_output, text, check):
        seen["stdin_name"] = Path(stdin.name)
        seen["stdin_head"] = stdin.read(64)
        work = Path(cwd)
        (work / "packmol_output.xyz").write_text(
            "2\npackmol\nSi 0.0 0.0 0.0\nN 1.0 1.0 1.0\n"
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sg.subprocess, "run", fake_run)

    cfg = StructureGenerateConfig(
        method="packmol",
        formula="SiN",
        n_formula_units=1,
        packmol_cmd=["packmol"],
        packing_density_g_cm3=2.5,
        packing_min_distance_A=1.6,
        seed=13,
    )

    atoms, prov = generate_atoms(cfg, outdir=tmp_path)

    assert len(atoms) == 2
    assert prov.method == "packmol"
    assert seen["stdin_name"] == (tmp_path / "packmol.inp")
    assert seen["stdin_head"].startswith("tolerance 1.60000000")


def test_prepare_initial_structure_invalidates_cache_when_structure_spec_changes(tmp_path: Path, monkeypatch):
    Atoms = _install_fake_ase(monkeypatch)
    from vitriflow.config import RunConfig
    import vitriflow.structuregen as structuregen_mod

    calls = {"count": 0}

    def _fake_generate_atoms(cfg, outdir):
        calls["count"] += 1
        atoms = Atoms(
            symbols=["Al"],
            positions=[[0.0, 0.0, 0.0]],
            cell=[4.05, 4.05, 4.05],
            pbc=True,
        )
        prov = structuregen_mod.StructureProvenance(
            method=str(cfg.method),
            formula=str(cfg.formula),
            source="fake",
            repeat=(1, 1, 1),
            n_atoms=1,
        )
        return atoms, prov

    monkeypatch.setattr(structuregen_mod, "generate_atoms", _fake_generate_atoms)

    def _fake_write_lammps_data(path, atoms, *, specorder=None, atom_style="atomic"):
        Path(path).write_text(f"Atoms # {atom_style}\n")

    monkeypatch.setattr(structuregen_mod, "write_lammps_data", _fake_write_lammps_data)

    cfg_atomic = RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "structure": {"generate": {"method": "random", "formula": "Al", "n_formula_units": 1}},
            "md": {"atom_style": "atomic"},
            "autotune": {"metrics": {"enabled": True, "type_to_species": ["Al"], "pairs": [{"pair": ["Al", "Al"]}] }},
        }
    )

    cfg_charge = RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "structure": {
                "generate": {"method": "random", "formula": "Al", "n_formula_units": 1},
                "charges": {"Al": 0.0},
            },
            "md": {"atom_style": "charge"},
            "autotune": {"metrics": {"enabled": True, "type_to_species": ["Al"], "pairs": [{"pair": ["Al", "Al"]}] }},
        }
    )

    data_atomic = structuregen_mod.prepare_initial_structure(cfg_atomic, tmp_path)
    assert "Atoms # atomic" in Path(data_atomic).read_text()

    data_charge = structuregen_mod.prepare_initial_structure(cfg_charge, tmp_path)
    txt = Path(data_charge).read_text()
    assert "Atoms # charge" in txt
    assert calls["count"] == 2

    meta = json.loads((tmp_path / "structure" / "structure_provenance.json").read_text())
    assert meta["_cache_spec"]["atom_style"] == "charge"
