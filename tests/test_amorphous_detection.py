from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _install_fake_ase(monkeypatch):
    class FakeCell:
        def __init__(self, data):
            arr = np.asarray(data, dtype=float)
            if arr.shape == (3,):
                arr = np.diag(arr)
            self.array = np.asarray(arr, dtype=float)

        def __array__(self, dtype=None):
            return np.asarray(self.array, dtype=dtype)

        def lengths(self):
            return np.asarray([float(np.linalg.norm(self.array[i])) for i in range(3)], dtype=float)

    class Atoms:
        def __init__(self, symbols=None, positions=None, *, numbers=None, cell=None, pbc=True):
            if symbols is None and numbers is not None:
                numbers = np.asarray(numbers, dtype=int).tolist()
                symbols = [f"X{int(z)}" for z in numbers]
            self._symbols = list(symbols or [])
            self._positions = np.asarray(positions if positions is not None else np.zeros((len(self._symbols), 3)), dtype=float)
            self._cell = FakeCell([1.0, 1.0, 1.0] if cell is None else cell)
            if isinstance(pbc, bool):
                self._pbc = np.asarray([pbc, pbc, pbc], dtype=bool)
            else:
                self._pbc = np.asarray(list(pbc), dtype=bool)

        def copy(self):
            return Atoms(self._symbols.copy(), self._positions.copy(), cell=self._cell.array.copy(), pbc=self._pbc.copy())

        def set_pbc(self, pbc):
            if isinstance(pbc, bool):
                self._pbc[:] = bool(pbc)
            else:
                self._pbc = np.asarray(list(pbc), dtype=bool)

        def get_chemical_symbols(self):
            return list(self._symbols)

        def get_positions(self):
            return np.asarray(self._positions, dtype=float)

        def get_cell(self):
            return self._cell

        def repeat(self, rep):
            rx, ry, rz = (int(rep[0]), int(rep[1]), int(rep[2]))
            cell = np.asarray(self._cell.array, dtype=float)
            ax, by, cz = np.diag(cell)
            syms = []
            pos = []
            for ix in range(rx):
                for iy in range(ry):
                    for iz in range(rz):
                        shift = np.asarray([ix * ax, iy * by, iz * cz], dtype=float)
                        for s, r in zip(self._symbols, self._positions):
                            syms.append(s)
                            pos.append(np.asarray(r, dtype=float) + shift)
            return Atoms(syms, np.asarray(pos, dtype=float), cell=[ax * rx, by * ry, cz * rz], pbc=True)

        def __len__(self):
            return len(self._symbols)

    def write(path, atoms, format=None):
        Path(path).write_text("fake extxyz\n")

    def read(path, format=None, **kwargs):
        return Atoms(symbols=["Sm", "O"], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)

    def neighbor_list(kind, atoms, cutoff):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    ase_mod = types.ModuleType("ase")
    io_mod = types.ModuleType("ase.io")
    nlist_mod = types.ModuleType("ase.neighborlist")
    data_mod = types.ModuleType("ase.data")
    ase_mod.Atoms = Atoms
    io_mod.write = write
    io_mod.read = read
    nlist_mod.neighbor_list = neighbor_list
    data_mod.atomic_masses = np.ones((120,), dtype=float)
    data_mod.atomic_numbers = {"Sm": 62, "O": 8, "Si": 14, "N": 7}
    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.io", io_mod)
    monkeypatch.setitem(sys.modules, "ase.neighborlist", nlist_mod)
    monkeypatch.setitem(sys.modules, "ase.data", data_mod)
    return Atoms


def _install_fake_mp(monkeypatch, *, docs, capture: dict[str, object] | None = None):
    capture = capture if capture is not None else {}

    class SummaryRester:
        def search(self, **kwargs):
            capture["search_kwargs"] = dict(kwargs)
            return list(docs)

    class MaterialsRester:
        def __init__(self):
            self.summary = SummaryRester()

    class FakeMPRester:
        def __init__(self, api_key):
            capture["api_key"] = api_key
            self.materials = MaterialsRester()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    mp_api_mod = types.ModuleType("mp_api")
    client_mod = types.ModuleType("mp_api.client")
    client_mod.MPRester = FakeMPRester
    mp_api_mod.client = client_mod
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_mod)
    monkeypatch.setitem(sys.modules, "mp_api.client", client_mod)
    return capture


def _minimal_run_config(metrics_overrides: dict):
    from vitriflow.config import RunConfig

    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Sm", "O"],
                "commands": [
                    "pair_style lj/cut 2.5",
                    "pair_coeff 1 1 1.0 1.0 2.5",
                    "pair_coeff 2 2 1.0 1.0 2.5",
                    "pair_coeff 1 2 1.0 1.0 2.5",
                ],
            },
            "structure": {"generate": {"method": "packmol", "formula": "Sm2O3", "n_formula_units": 1}},
            "autotune": {"metrics": metrics_overrides},
        }
    )


def test_metrics_policy_prepends_total_sq_for_amorphous_detection():
    from vitriflow.workflows.metrics_policy import resolve_effective_metrics_config

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "voids": {"enabled": True},
            "rings": {"enabled": False},
            "sq": [{"pair": ["Sm", "O"], "q_max": 12.0, "nq": 64, "r_max": 6.0, "nbins": 64}],
            "amorphous": {"enabled": True, "reference": {"enabled": False}},
        }
    )
    metrics_eff, warnings, summary = resolve_effective_metrics_config(
        cfg.autotune.metrics,
        structure_data=None,
        type_to_species=["Sm", "O"],
        context="autotune production",
    )

    assert metrics_eff.sq[0].pair is None
    assert metrics_eff.sq[1].pair == ("Sm", "O")
    assert summary["amorphous_enabled"] is True
    assert any("amorphous detection requires total S(q)" in w for w in warnings)


def test_summarize_rate_amorphous_acceptance_respects_fraction_threshold(monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.analysis.amorphous import summarize_rate_amorphous_acceptance

    res = summarize_rate_amorphous_acceptance(
        [
            {"rate": 50.0, "replicates": [{"amorphous": {"passed": True}}, {"amorphous": {"passed": True}}, {"amorphous": {"passed": False}}]},
            {"rate": 10.0, "replicates": [{"amorphous": {"passed": True}}, {"amorphous": {"passed": False}}]},
        ],
        amorph_cfg=SimpleNamespace(min_pass_fraction=0.75),
    )

    assert res[0]["accepted"] is False
    assert res[0]["pass_fraction"] == 2.0 / 3.0
    assert res[1]["accepted"] is False
    assert res[1]["pass_fraction"] == 0.5


def test_reference_peak_overlap_uses_one_to_one_matching(monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.analysis.amorphous import _reference_peak_overlap

    box_peaks = [
        {"q": 2.00, "prominence": 1.0, "fwhm": 0.60},
    ]
    ref_peaks = [
        {"q": 1.95, "prominence": 1.0, "fwhm": 0.05},
        {"q": 2.05, "prominence": 1.0, "fwhm": 0.05},
        {"q": 2.15, "prominence": 1.0, "fwhm": 0.05},
    ]

    ov = _reference_peak_overlap(box_peaks=box_peaks, ref_peaks=ref_peaks, q_tol=0.20)

    assert 0.0 <= ov <= 1.0
    assert ov < 0.5


def test_summarize_rate_amorphous_acceptance_includes_failed_criteria(monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.analysis.amorphous import summarize_rate_amorphous_acceptance

    res = summarize_rate_amorphous_acceptance(
        [
            {
                "rate": 50.0,
                "replicates": [
                    {
                        "amorphous": {
                            "passed": False,
                            "criteria": {
                                "reference_peak_overlap": {"value": 0.82, "threshold": 0.60, "passed": False},
                                "crystalline_fraction": {"value": 0.03, "threshold": 0.18, "passed": True},
                            },
                        }
                    },
                    {
                        "amorphous": {
                            "passed": True,
                            "criteria": {
                                "reference_peak_overlap": {"value": 0.58, "threshold": 0.60, "passed": True},
                                "crystalline_fraction": {"value": 0.02, "threshold": 0.18, "passed": True},
                            },
                        }
                    },
                ],
            }
        ],
        amorph_cfg=SimpleNamespace(min_pass_fraction=0.75),
    )

    summ = res[0]
    assert summ["accepted"] is False
    assert summ["failed_criteria"] == ["reference_peak_overlap"]
    crit = summ["criteria_summary"]["reference_peak_overlap"]
    assert crit["n_failed"] == 1
    assert crit["threshold"] == 0.60
    assert crit["max"] == 0.82


def test_write_rate_scan_failure_snapshot_persists_partial_metrics(tmp_path: Path, monkeypatch):
    from dataclasses import dataclass, field

    _install_fake_ase(monkeypatch)
    from vitriflow.workflows.autotune import _write_rate_scan_failure_snapshot

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "rings": {"enabled": False},
            "voids": {"enabled": True},
            "sq": [{"pair": None, "q_max": 12.0, "nq": 64, "r_max": 6.0, "nbins": 64}],
            "amorphous": {"enabled": True, "reference": {"enabled": False}},
        }
    )

    @dataclass
    class DummyCore:
        enabled: bool = False

    @dataclass
    class DummyPreflight:
        selected_timestep: float = 0.001
        selected_ensemble: str = "npt"
        selected_tdamp: float = 0.1
        selected_pdamp: float = 1.0
        potential_lines: list[str] | None = None
        core_repulsion: DummyCore = field(default_factory=DummyCore)

    @dataclass
    class DummyOutcome:
        label: str

    out = _write_rate_scan_failure_snapshot(
        outdir=tmp_path,
        config=cfg,
        pot_cfg=cfg.kim,
        kim_install=None,
        preflight=DummyPreflight(),
        T=[1000.0, 2000.0],
        D=[0.0, 1.0],
        D_mu=[0.0, 1.0],
        D_se=[0.0, 0.1],
        D_med=[0.0, 1.0],
        tm_cfg=cfg.autotune.tm_scan,
        tm_summary={"ok": True},
        tm_outcomes_all=[DummyOutcome("tm")],
        tm_est=SimpleNamespace(Tm=2500.0, T_liquid=3000.0, D_liquid_target=1.0, method="test", score=1.0, idx=1),
        time_unit_ps=1.0,
        T_high=3300.0,
        high_total_steps=1000,
        force_iso_active=False,
        high_cfg=cfg.autotune.highT,
        high_stationarity_summary={"ok": True},
        high_rep_summaries=[{"rep": 1}],
        high_outcomes=[DummyOutcome("high")],
        melt_pool=[tmp_path / "melt_1.data"],
        melt_data=tmp_path / "melt_1.data",
        rate_results=[
            {
                "rate": 100.0,
                "replicates": [
                    {
                        "amorphous": {
                            "passed": False,
                            "criteria": {
                                "reference_peak_overlap": {"value": 0.82, "threshold": 0.6, "passed": False}
                            },
                        }
                    }
                ],
                "amorphous_summary": {
                    "accepted": False,
                    "pass_fraction": 0.0,
                    "required_pass_fraction": 0.67,
                    "criteria_summary": {
                        "reference_peak_overlap": {"n_failed": 1, "n_evaluated": 1, "threshold": 0.6, "mean": 0.82, "max": 0.82}
                    },
                },
            }
        ],
        cutoffs_rate={(1, 2): 2.5},
        metric_warnings=["warn"],
        metrics_summary={"amorphous_enabled": True},
        failure_message="No cooling rates satisfied the amorphous acceptance gate.",
        progress=None,
    )

    assert out["status"] == "failed"
    assert out["failure"]["stage"] == "rate_scan"
    saved = json.loads((tmp_path / "autotune_results.json").read_text())
    assert saved["rate_scan"]["rates"][0]["rate"] == 100.0
    assert saved["rate_scan"]["rates"][0]["amorphous_summary"]["criteria_summary"]["reference_peak_overlap"]["max"] == 0.82


def test_build_materials_project_reference_library_filters_candidates(tmp_path: Path, monkeypatch):
    _install_fake_ase(monkeypatch)
    import vitriflow.analysis.amorphous as am
    from vitriflow.workflows.metrics_policy import resolve_effective_metrics_config

    class FakeComposition:
        def __init__(self, data):
            self._data = dict(data)

        def as_dict(self):
            return dict(self._data)

    class FakeStructure:
        is_ordered = True

        def __init__(self, tag: str):
            self.tag = tag

        def to_conventional(self):
            return self

        def to_ase_atoms(self):
            from ase import Atoms

            return Atoms(
                symbols=["Sm", "Sm", "O", "O", "O"],
                positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5], [2.0, 2.0, 0.0], [0.0, 2.0, 2.0], [2.0, 0.0, 2.0]],
                cell=[5.0, 5.0, 5.0],
                pbc=True,
            )

    docs = [
        SimpleNamespace(
            material_id="mp-good-1",
            formula_pretty="Sm2O3",
            energy_above_hull=0.0,
            is_stable=True,
            composition_reduced=FakeComposition({"Sm": 2, "O": 3}),
            structure=FakeStructure("good-1"),
        ),
        SimpleNamespace(
            material_id="mp-good-2",
            formula_pretty="Sm2O3",
            energy_above_hull=0.03,
            is_stable=True,
            composition_reduced=FakeComposition({"Sm": 2, "O": 3}),
            structure=FakeStructure("good-2"),
        ),
        SimpleNamespace(
            material_id="mp-bad-comp",
            formula_pretty="SmO",
            energy_above_hull=0.0,
            is_stable=True,
            composition_reduced=FakeComposition({"Sm": 1, "O": 1}),
            structure=FakeStructure("bad-comp"),
        ),
        SimpleNamespace(
            material_id="mp-unstable",
            formula_pretty="Sm2O3",
            energy_above_hull=0.2,
            is_stable=False,
            composition_reduced=FakeComposition({"Sm": 2, "O": 3}),
            structure=FakeStructure("unstable"),
        ),
    ]
    seen: dict[str, object] = {}
    _install_fake_mp(monkeypatch, docs=docs, capture=seen)
    monkeypatch.setenv("MP_API_KEY", "test-mp-key")

    monkeypatch.setattr(am, "compute_sq", lambda frames, **kwargs: (np.asarray([0.8, 1.6, 2.4], dtype=float), np.asarray([1.0, 1.7, 1.1], dtype=float)))
    monkeypatch.setattr(
        am,
        "_local_order_analysis",
        lambda frame, *, cutoffs, amorph_cfg: {
            "crystalline_fraction": 0.0,
            "largest_cluster_fraction": 0.0,
            "qbar": {"qbar_4_mean": 0.1},
        },
    )

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "voids": {"enabled": True},
            "rings": {"enabled": False},
            "sq": [{"pair": None, "q_max": 12.0, "nq": 64, "r_max": 6.0, "nbins": 64}],
            "amorphous": {
                "enabled": True,
                "reference": {
                    "enabled": True,
                    "mp_api_key_env": "MP_API_KEY",
                    "max_candidates": 2,
                    "min_supercell_length_A": 1.0,
                    "min_supercell_atoms": 1,
                },
            },
        }
    )
    metrics_eff, _warnings, _summary = resolve_effective_metrics_config(
        cfg.autotune.metrics,
        structure_data=None,
        type_to_species=["Sm", "O"],
        context="output analysis",
    )

    cache_dir = tmp_path / "refs"
    lib = am.build_materials_project_reference_library(
        cache_dir=cache_dir,
        formula="Sm2O3",
        type_to_species=["Sm", "O"],
        cutoffs={(1, 1): 2.5, (1, 2): 2.5, (2, 2): 2.5},
        metrics_cfg=metrics_eff,
    )

    assert seen["api_key"] == "test-mp-key"
    assert seen["search_kwargs"]["chemsys"] == "O-Sm"
    assert [ref["material_id"] for ref in lib["references"]] == ["mp-good-1", "mp-good-2"]
    assert (cache_dir / "reference_manifest.json").exists()
    assert len(list(cache_dir.glob("reference_*.extxyz"))) == 2
    manifest = json.loads((cache_dir / "reference_manifest.json").read_text())
    assert manifest["used"] is True
    assert len(manifest["references"]) == 2



def test_analyse_production_box_rejects_non_amorphous_box(tmp_path: Path, monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.workflows import production_common as pc

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "voids": {"enabled": True},
            "rings": {"enabled": False},
            "pairs": [{"pair": ["Sm", "Sm"]}],
            "coordinations": [{"central": "Sm", "neighbor": "Sm"}],
            "sq": [],
            "gr": [],
            "amorphous": {"enabled": True, "enforce_during_production": True, "reference": {"enabled": False}},
        }
    )
    metrics_cfg = cfg.autotune.metrics

    frame = DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 1], dtype=int),
        positions=np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )

    monkeypatch.setattr(pc, "read_last_frames_auto", lambda *args, **kwargs: [frame])
    monkeypatch.setattr(pc, "compute_structure_metrics_timeavg", lambda *args, **kwargs: SimpleNamespace(values={"density": 1.0}))
    monkeypatch.setattr(pc, "should_collect_stage_metrics_timeseries", lambda *args, **kwargs: False)
    monkeypatch.setattr(pc, "compute_structure_distributions_timeavg", lambda *args, **kwargs: {"bondlen": {}, "angle": {}, "coord": {}, "void": {}})
    monkeypatch.setattr(pc, "compute_coordination_defects", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pc,
        "analyse_amorphous_state",
        lambda *args, **kwargs: {
            "enabled": True,
            "passed": False,
            "criteria": {"crystalline_fraction": {"value": 0.45, "threshold": 0.15, "passed": False}},
            "scalar_metrics": {"amorphous_crystalline_fraction": 0.45},
            "reference": {"used": False},
        },
    )

    outdir = tmp_path / "out"
    melt_dir = outdir / "melt"
    quench_dir = outdir / "quench"
    relax_dir = outdir / "relax"
    rejects_dir = outdir / "rejects"
    for p in (melt_dir, quench_dir, relax_dir, rejects_dir):
        p.mkdir(parents=True, exist_ok=True)
    relax_data = relax_dir / "relax.data"
    relax_data.write_text("LAMMPS data file\n\n0 atoms\n")
    relax_dump = relax_dir / "relax.lammpstrj"
    relax_dump.write_text("")
    relax_traj = relax_dir / "traj.extxyz"
    relax_traj.write_text("")

    entry, cut_map = pc.analyse_production_box(
        box_id=1,
        outdir=outdir,
        melt_stage_dir=melt_dir,
        quench_stage_dir=quench_dir,
        relax_stage_dir=relax_dir,
        relax_data_path=relax_data,
        density_mean=2.2,
        density_stderr=0.01,
        metrics_cfg=metrics_cfg,
        cutoffs={(1, 1): 2.1},
        required_pairs=[],
        fixed_cutoffs={},
        type_to_species=["Sm"],
        md_timestep=0.001,
        rejects_dir=rejects_dir,
        relax_dump_path=relax_dump,
        relax_traj_path=relax_traj,
    )

    assert cut_map == {(1, 1): 2.1}
    assert entry["is_amorphous"] is False
    assert entry["metrics"]["amorphous_crystalline_fraction"] == 0.45
    assert entry["reject"]["reason"] == "non_amorphous"
    assert (relax_dir / "amorphous_state.json").exists()
    assert (rejects_dir / "box_001" / "amorphous_state.json").exists()


def test_analyse_amorphous_state_reports_reference_motif_matches(monkeypatch):
    import vitriflow.analysis.amorphous as am

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "rings": {"enabled": False},
            "gr": [],
            "sq": [{"pair": None, "q_max": 12.0, "nq": 64, "r_max": 6.0, "nbins": 64}],
            "amorphous": {
                "enabled": True,
                "max_reference_peak_overlap": 0.65,
                "reference": {"enabled": True, "required": False},
            },
        }
    )

    frame = am.DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 2], dtype=int),
        positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )

    monkeypatch.setattr(am, "compute_sq", lambda *args, **kwargs: (np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.8, 1.2])))
    monkeypatch.setattr(
        am,
        "_sq_peak_features",
        lambda *args, **kwargs: [
            {"q": 1.0, "prominence": 1.0, "height": 2.0, "fwhm": 0.2, "sharpness": 5.0},
            {"q": 2.0, "prominence": 0.7, "height": 1.7, "fwhm": 0.3, "sharpness": 2.3333333333},
        ],
    )
    monkeypatch.setattr(
        am,
        "_local_order_analysis",
        lambda *args, **kwargs: {
            "crystalline_fraction": 0.30,
            "largest_cluster_fraction": 0.20,
            "qbar": {"qbar_4_mean": 0.20, "qbar_6_mean": 0.50},
        },
    )
    monkeypatch.setattr(
        am,
        "build_materials_project_reference_library",
        lambda **kwargs: {
            "enabled": True,
            "used": True,
            "formula": "Sm2O3",
            "references": [
                {
                    "material_id": "mp-strong",
                    "formula_pretty": "Sm2O3",
                    "energy_above_hull": 0.0,
                    "sq": {"peaks": [{"q": 1.0, "prominence": 1.0}, {"q": 2.0, "prominence": 0.8}]},
                    "local_order": {"qbar": {"qbar_4_mean": 0.21, "qbar_6_mean": 0.48}},
                },
                {
                    "material_id": "mp-weak",
                    "formula_pretty": "Sm2O3",
                    "energy_above_hull": 0.02,
                    "sq": {"peaks": [{"q": 4.5, "prominence": 1.0}]},
                    "local_order": {"qbar": {"qbar_4_mean": 0.05, "qbar_6_mean": 0.15}},
                },
            ],
        },
    )

    report = am.analyse_amorphous_state(
        [frame],
        metrics_cfg=cfg.autotune.metrics,
        cutoffs={(1, 2): 2.1},
        type_to_species=["Sm", "O"],
        cache_dir=Path("dummy-cache"),
        formula_override="Sm2O3",
    )

    assert report["motifs"]["used"] is True
    assert report["motifs"]["thresholds"]["candidate_peak_overlap"] == 0.325
    top = report["motifs"]["top_matches"]
    assert top[0]["material_id"] == "mp-strong"
    assert top[0]["detected"] is True
    assert top[0]["motif_score"] >= top[1]["motif_score"]
    assert report["reference"]["best_material_id"] == "mp-strong"
    assert report["reference"]["best_motif_material_id"] == "mp-strong"



def test_summarize_production_crystal_motifs_aggregates_candidate_and_detected_boxes():
    from vitriflow.analysis.motif_summary import summarize_production_crystal_motifs

    accepted = [
        {
            "box": 1,
            "amorphous": {
                "motifs": {
                    "enabled": True,
                    "used": True,
                    "thresholds": {"candidate_peak_overlap": 0.325, "detected_peak_overlap": 0.65},
                    "top_matches": [
                        {
                            "material_id": "mp-a",
                            "formula_pretty": "Sm2O3",
                            "energy_above_hull": 0.0,
                            "peak_overlap": 0.72,
                            "motif_score": 0.81,
                            "candidate": True,
                            "detected": True,
                            "rank": 1,
                        }
                    ],
                }
            },
        },
        {
            "box": 2,
            "amorphous": {
                "motifs": {
                    "enabled": True,
                    "used": True,
                    "thresholds": {"candidate_peak_overlap": 0.325, "detected_peak_overlap": 0.65},
                    "top_matches": [
                        {
                            "material_id": "mp-b",
                            "formula_pretty": "Sm2O3",
                            "energy_above_hull": 0.01,
                            "peak_overlap": 0.40,
                            "motif_score": 0.46,
                            "candidate": True,
                            "detected": False,
                            "rank": 1,
                        }
                    ],
                }
            },
        },
    ]
    rejected = [
        {
            "box": 3,
            "amorphous": {
                "motifs": {
                    "enabled": True,
                    "used": True,
                    "thresholds": {"candidate_peak_overlap": 0.325, "detected_peak_overlap": 0.65},
                    "top_matches": [
                        {
                            "material_id": "mp-a",
                            "formula_pretty": "Sm2O3",
                            "energy_above_hull": 0.0,
                            "peak_overlap": 0.69,
                            "motif_score": 0.75,
                            "candidate": True,
                            "detected": True,
                            "rank": 1,
                        }
                    ],
                }
            },
        }
    ]

    summary = summarize_production_crystal_motifs(accepted, rejected_boxes=rejected)

    assert summary["used"] is True
    assert summary["n_boxes_total"] == 3
    assert summary["motifs"][0]["material_id"] == "mp-a"
    assert summary["motifs"][0]["detected_boxes"] == [1, 3]
    assert summary["motifs"][0]["accepted_detected_boxes"] == [1]
    assert summary["motifs"][0]["rejected_detected_boxes"] == [3]
    assert summary["motifs"][1]["material_id"] == "mp-b"
    assert summary["motifs"][1]["n_boxes_candidate"] == 1
    assert summary["motifs"][1]["n_boxes_detected"] == 0


def test_reference_peak_overlap_penalizes_broad_vs_sharp_match(monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.analysis.amorphous import _reference_peak_overlap

    box_peaks = [{"q": 3.50, "prominence": 1.0, "fwhm": 0.60}]
    ref_peaks = [{"q": 3.52, "prominence": 1.0, "fwhm": 0.06}]

    ov = _reference_peak_overlap(box_peaks=box_peaks, ref_peaks=ref_peaks, q_tol=0.20)

    assert 0.0 <= ov <= 1.0
    assert ov < 0.4


def test_reference_peak_overlap_details_marks_single_peak_match_non_discriminative(monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.analysis.amorphous import _reference_peak_overlap_details

    box_peaks = [{"q": 3.50, "prominence": 1.0, "fwhm": 0.55}]
    ref_peaks = [
        {"q": 3.52, "prominence": 1.0, "fwhm": 0.20},
        {"q": 5.10, "prominence": 0.6, "fwhm": 0.15},
    ]

    info = _reference_peak_overlap_details(box_peaks=box_peaks, ref_peaks=ref_peaks, q_tol=0.20)

    assert 0.0 <= info["overlap"] <= 1.0
    assert info["matched_pairs"] == 1
    assert info["matched_significant_pairs"] == 1
    assert info["discriminative"] is False
    assert info["reason"] == "insufficient_matched_peak_support"



def test_analyse_amorphous_state_skips_non_discriminative_reference_gate(monkeypatch):
    import vitriflow.analysis.amorphous as am

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "rings": {"enabled": False},
            "gr": [],
            "sq": [{"pair": None, "q_max": 12.0, "nq": 64, "r_max": 6.0, "nbins": 64}],
            "amorphous": {
                "enabled": True,
                "max_reference_peak_overlap": 0.65,
                "reference": {"enabled": True, "required": False},
            },
        }
    )

    frame = am.DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 2], dtype=int),
        positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )

    monkeypatch.setattr(am, "compute_sq", lambda *args, **kwargs: (np.asarray([1.0, 3.5, 6.0]), np.asarray([1.0, 1.8, 1.1])))
    monkeypatch.setattr(
        am,
        "_sq_peak_features",
        lambda *args, **kwargs: [
            {"q": 3.5, "prominence": 1.0, "height": 1.8, "fwhm": 0.55, "sharpness": 0.3},
        ],
    )
    monkeypatch.setattr(
        am,
        "_local_order_analysis",
        lambda *args, **kwargs: {
            "crystalline_fraction": 0.0,
            "largest_cluster_fraction": 0.0,
            "qbar": {"qbar_4_mean": 0.05, "qbar_6_mean": 0.08},
        },
    )
    monkeypatch.setattr(
        am,
        "build_materials_project_reference_library",
        lambda **kwargs: {
            "enabled": True,
            "used": True,
            "formula": "Sm2O3",
            "references": [
                {
                    "material_id": "mp-218",
                    "formula_pretty": "Sm2O3",
                    "energy_above_hull": 0.0,
                    "sq": {
                        "peaks": [
                            {"q": 3.52, "prominence": 1.0, "fwhm": 0.20},
                            {"q": 5.10, "prominence": 0.6, "fwhm": 0.15},
                        ]
                    },
                    "local_order": {"qbar": {"qbar_4_mean": 0.20, "qbar_6_mean": 0.48}},
                },
            ],
        },
    )

    report = am.analyse_amorphous_state(
        [frame],
        metrics_cfg=cfg.autotune.metrics,
        cutoffs={(1, 2): 2.1},
        type_to_species=["Sm", "O"],
        cache_dir=Path("dummy-cache"),
        formula_override="Sm2O3",
    )

    assert report["passed"] is True
    crit = report["criteria"]["reference_peak_overlap"]
    assert crit["skipped"] is True
    assert crit["passed"] is True
    assert crit["reference_discriminative"] is False
    assert "advisory_only" in crit
    top = report["motifs"]["top_matches"][0]
    assert top["candidate"] is True
    assert top["detected"] is False
    assert top["reference_discriminative"] is False


def test_local_order_analysis_requires_amplitude_support_from_reference(monkeypatch):
    _install_fake_ase(monkeypatch)
    import vitriflow.analysis.amorphous as am

    frame = am.DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2, 3, 4], dtype=int),
        types=np.asarray([1, 1, 1, 1], dtype=int),
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )

    n_atoms = 4
    nbr_ids = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    nbr_vecs = [[np.asarray([1.0, 0.0, 0.0])] * 3 for _ in range(n_atoms)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    monkeypatch.setattr(am, "_directed_neighbors", lambda *args, **kwargs: (nbr_ids, nbr_vecs, edges))

    # orientation qbar magnitude
    qlm_small = np.zeros((n_atoms, 13), dtype=complex)
    qlm_small[:, 0] = 0.02 + 0.0j
    monkeypatch.setattr(am, "_qlm_from_nbr_vecs", lambda *args, **kwargs: qlm_small.copy())

    amorph_cfg = SimpleNamespace(
        l_values=[4, 6],
        solid_like_l=6,
        solid_like_bond_threshold=0.5,
        ordered_min_neighbors=2,
        ordered_min_fraction=0.5,
    )
    refs = [
        {
            "local_order": {
                "qbar": {
                    "qbar_6_p25": 0.18,
                    "qbar_6_p50": 0.22,
                    "qbar_6_p75": 0.26,
                }
            }
        }
    ]

    out = am._local_order_analysis(frame, cutoffs={(1, 1): 2.0}, amorph_cfg=amorph_cfg, reference_refs=refs)

    assert out["solid_like_qbar_floor"] >= 0.10
    assert out["crystalline_fraction"] == 0.0
    assert out["largest_cluster_fraction"] == 0.0
    assert out["ordered_atoms"] == 0


def test_local_order_analysis_retains_high_amplitude_crystal_like_order(monkeypatch):
    _install_fake_ase(monkeypatch)
    import vitriflow.analysis.amorphous as am

    frame = am.DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2, 3, 4], dtype=int),
        types=np.asarray([1, 1, 1, 1], dtype=int),
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )

    n_atoms = 4
    nbr_ids = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    nbr_vecs = [[np.asarray([1.0, 0.0, 0.0])] * 3 for _ in range(n_atoms)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    monkeypatch.setattr(am, "_directed_neighbors", lambda *args, **kwargs: (nbr_ids, nbr_vecs, edges))

    qlm_large = np.zeros((n_atoms, 13), dtype=complex)
    qlm_large[:, 0] = 0.20 + 0.0j
    monkeypatch.setattr(am, "_qlm_from_nbr_vecs", lambda *args, **kwargs: qlm_large.copy())

    amorph_cfg = SimpleNamespace(
        l_values=[4, 6],
        solid_like_l=6,
        solid_like_bond_threshold=0.5,
        ordered_min_neighbors=2,
        ordered_min_fraction=0.5,
    )
    refs = [
        {
            "local_order": {
                "qbar": {
                    "qbar_6_p25": 0.18,
                    "qbar_6_p50": 0.22,
                    "qbar_6_p75": 0.26,
                }
            }
        }
    ]

    out = am._local_order_analysis(frame, cutoffs={(1, 1): 2.0}, amorph_cfg=amorph_cfg, reference_refs=refs)

    assert out["solid_like_qbar_floor"] >= 0.10
    assert out["crystalline_fraction"] == 1.0
    assert out["largest_cluster_fraction"] == 1.0
    assert out["ordered_atoms"] == 4



def test_reference_peak_overlap_details_accepts_many_peak_discriminative_match(monkeypatch):
    _install_fake_ase(monkeypatch)
    from vitriflow.analysis.amorphous import _reference_peak_overlap_details

    box_peaks = [
        {"q": 1.0 + 0.4 * float(i), "prominence": 1.0, "fwhm": 0.08}
        for i in range(12)
    ]
    ref_peaks = [
        {"q": 1.0 + 0.4 * float(i), "prominence": 1.0, "fwhm": 0.08}
        for i in range(12)
    ]

    info = _reference_peak_overlap_details(box_peaks=box_peaks, ref_peaks=ref_peaks, q_tol=0.10)

    assert info["discriminative"] is True
    assert info["reason"] is None
    assert info["matched_pairs"] == 12
    assert info["matched_significant_pairs"] == 12
    assert info["overlap"] == pytest.approx(1.0)
    assert info["significant_support_threshold"] < 0.10


def test_analyse_amorphous_state_uses_best_discriminative_reference_for_hard_gate(monkeypatch):
    import vitriflow.analysis.amorphous as am

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "rings": {"enabled": False},
            "gr": [],
            "sq": [{"pair": None, "q_max": 12.0, "nq": 64, "r_max": 6.0, "nbins": 64}],
            "amorphous": {
                "enabled": True,
                "max_reference_peak_overlap": 0.80,
                "reference": {"enabled": True, "required": False},
            },
        }
    )

    frame = am.DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 2], dtype=int),
        positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )

    monkeypatch.setattr(am, "compute_sq", lambda *args, **kwargs: (np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.3, 1.1])))
    monkeypatch.setattr(am, "_sq_peak_features", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        am,
        "_local_order_analysis",
        lambda *args, **kwargs: {
            "crystalline_fraction": 0.0,
            "largest_cluster_fraction": 0.0,
            "qbar": {"qbar_4_mean": 0.10, "qbar_6_mean": 0.20},
        },
    )
    monkeypatch.setattr(
        am,
        "build_materials_project_reference_library",
        lambda **kwargs: {
            "enabled": True,
            "used": True,
            "formula": "Sm2O3",
            "references": [
                {"material_id": "mp-nondisc", "formula_pretty": "Sm2O3", "sq": {"peaks": []}, "local_order": {"qbar": {}}},
                {"material_id": "mp-disc", "formula_pretty": "Sm2O3", "sq": {"peaks": []}, "local_order": {"qbar": {}}},
            ],
        },
    )
    monkeypatch.setattr(
        am,
        "_rank_reference_motifs",
        lambda **kwargs: [
            {
                "material_id": "mp-disc",
                "formula_pretty": "Sm2O3",
                "energy_above_hull": 0.0,
                "peak_overlap": 0.7852,
                "qbar_similarity": 0.90,
                "motif_score": 0.80,
                "candidate": True,
                "detected": True,
                "reference_discriminative": True,
                "matched_pairs": 2,
                "matched_significant_pairs": 2,
                "matched_significant_support_weight": 0.30,
                "box_peak_count": 3,
                "ref_peak_count": 3,
                "significant_support_threshold": 0.10,
                "significant_quality_threshold": 0.50,
                "discriminative_min_significant_support_weight": 0.20,
                "match_reason": None,
                "rank": 1,
            },
            {
                "material_id": "mp-nondisc",
                "formula_pretty": "Sm2O3",
                "energy_above_hull": 0.01,
                "peak_overlap": 0.8088,
                "qbar_similarity": 0.20,
                "motif_score": 0.55,
                "candidate": True,
                "detected": False,
                "reference_discriminative": False,
                "matched_pairs": 1,
                "matched_significant_pairs": 1,
                "matched_significant_support_weight": 0.10,
                "box_peak_count": 3,
                "ref_peak_count": 2,
                "significant_support_threshold": 0.10,
                "significant_quality_threshold": 0.50,
                "discriminative_min_significant_support_weight": 0.20,
                "match_reason": "insufficient_matched_peak_support",
                "rank": 2,
            },
        ],
    )

    report = am.analyse_amorphous_state(
        [frame],
        metrics_cfg=cfg.autotune.metrics,
        cutoffs={(1, 2): 2.1},
        type_to_species=["Sm", "O"],
        cache_dir=Path("dummy-cache"),
        formula_override="Sm2O3",
    )

    crit = report["criteria"]["reference_peak_overlap"]
    assert report["passed"] is True
    assert crit["passed"] is True
    assert crit.get("skipped", False) is False
    assert crit["best_material_id"] == "mp-disc"
    assert crit["value"] == pytest.approx(0.7852)
    assert crit["advisory_best_material_id"] == "mp-nondisc"
    assert crit["advisory_best_peak_overlap"] == pytest.approx(0.8088)
    assert report["reference"]["best_material_id"] == "mp-nondisc"
    assert report["reference"]["best_discriminative_material_id"] == "mp-disc"


def test_local_order_analysis_timeavg_averages_tail_frames(monkeypatch):
    _install_fake_ase(monkeypatch)
    import vitriflow.analysis.amorphous as am

    frame_a = am.DumpFrame(
        timestep=0,
        ids=np.asarray([1], dtype=int),
        types=np.asarray([1], dtype=int),
        positions=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )
    frame_b = am.DumpFrame(
        timestep=1,
        ids=np.asarray([1], dtype=int),
        types=np.asarray([1], dtype=int),
        positions=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
        origin=np.zeros((3,), dtype=float),
    )

    reports = {
        0: {
            "crystalline_fraction": 0.0,
            "largest_cluster_fraction": 0.0,
            "ordered_atoms": 0,
            "largest_cluster": 0,
            "solid_like_bonds": 0,
            "degrees_mean": 2.0,
            "degrees_median": 2.0,
            "solid_like_threshold": 0.5,
            "solid_like_qbar_scale": 0.2,
            "solid_like_qbar_floor": 0.1,
            "solid_like_calibration_source": "reference",
            "ordered_atom_indices": [],
            "solid_like_counts": [0],
            "qbar": {"qbar_4_mean": 0.10, "qbar_6_mean": 0.20},
        },
        1: {
            "crystalline_fraction": 0.5,
            "largest_cluster_fraction": 0.25,
            "ordered_atoms": 1,
            "largest_cluster": 1,
            "solid_like_bonds": 2,
            "degrees_mean": 4.0,
            "degrees_median": 4.0,
            "solid_like_threshold": 0.5,
            "solid_like_qbar_scale": 0.2,
            "solid_like_qbar_floor": 0.1,
            "solid_like_calibration_source": "reference",
            "ordered_atom_indices": [0],
            "solid_like_counts": [2],
            "qbar": {"qbar_4_mean": 0.30, "qbar_6_mean": 0.40},
        },
    }
    monkeypatch.setattr(
        am,
        "_local_order_analysis",
        lambda frame, **kwargs: dict(reports[int(frame.timestep)]),
    )

    out = am._local_order_analysis_timeavg(
        [frame_a, frame_b],
        cutoffs={(1, 1): 2.0},
        amorph_cfg=SimpleNamespace(),
        reference_refs=None,
    )

    assert out["crystalline_fraction"] == 0.25
    assert out["largest_cluster_fraction"] == 0.125
    assert out["ordered_atoms"] == 0
    assert out["ordered_atoms_mean"] == 0.5
    assert out["qbar"]["qbar_4_mean"] == 0.20
    assert out["qbar"]["qbar_6_mean"] == pytest.approx(0.30)
    assert out["aggregation"]["n_frames"] == 2
    assert out["aggregation"]["policy"] == "mean_over_tail_frames"


def test_analyse_amorphous_state_time_averages_local_order_over_tail_frames(monkeypatch):
    import vitriflow.analysis.amorphous as am

    cfg = _minimal_run_config(
        {
            "enabled": True,
            "rings": {"enabled": False},
            "gr": [],
            "sq": [{"pair": None, "q_max": 12.0, "nq": 64, "r_max": 6.0, "nbins": 64}],
            "amorphous": {
                "enabled": True,
                "max_bragg_sharpness": 10.0,
                "max_crystalline_fraction": 0.75,
                "max_largest_cluster_fraction": 0.75,
                "reference": {"enabled": False},
            },
        }
    )

    frames = [
        am.DumpFrame(
            timestep=0,
            ids=np.asarray([1, 2], dtype=int),
            types=np.asarray([1, 1], dtype=int),
            positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
            origin=np.zeros((3,), dtype=float),
        ),
        am.DumpFrame(
            timestep=1,
            ids=np.asarray([1, 2], dtype=int),
            types=np.asarray([1, 1], dtype=int),
            positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=float),
            origin=np.zeros((3,), dtype=float),
        ),
    ]

    monkeypatch.setattr(am, "compute_sq", lambda *args, **kwargs: (np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.1, 1.0])))
    monkeypatch.setattr(am, "_sq_peak_features", lambda *args, **kwargs: [])

    def _fake_local(frame, **kwargs):
        if int(frame.timestep) == 0:
            return {
                "crystalline_fraction": 0.0,
                "largest_cluster_fraction": 0.0,
                "ordered_atoms": 0,
                "largest_cluster": 0,
                "solid_like_bonds": 0,
                "qbar": {"qbar_6_mean": 0.0},
            }
        return {
            "crystalline_fraction": 1.0,
            "largest_cluster_fraction": 1.0,
            "ordered_atoms": 2,
            "largest_cluster": 2,
            "solid_like_bonds": 1,
            "ordered_atom_indices": [0, 1],
            "solid_like_counts": [1, 1],
            "qbar": {"qbar_6_mean": 1.0},
        }

    monkeypatch.setattr(am, "_local_order_analysis", _fake_local)

    report = am.analyse_amorphous_state(
        frames,
        metrics_cfg=cfg.autotune.metrics,
        cutoffs={(1, 1): 2.1},
        type_to_species=["Sm"],
        cache_dir=Path("dummy-cache"),
        formula_override="Sm",
    )

    assert report["passed"] is True
    assert report["criteria"]["crystalline_fraction"]["value"] == pytest.approx(0.5)
    assert report["criteria"]["largest_cluster_fraction"]["value"] == pytest.approx(0.5)
    assert report["local_order"]["qbar"]["qbar_6_mean"] == pytest.approx(0.5)
    assert report["local_order"]["aggregation"]["policy"] == "mean_over_tail_frames"
    assert report["local_order"]["aggregation"]["n_frames"] == 2
    assert report["local_order"]["aggregation"]["crystalline_fraction_per_frame"] == [0.0, 1.0]


def test_collect_rate_scan_cutoff_reference_frames_pools_all_rates(tmp_path: Path, monkeypatch):
    from vitriflow.workflows import autotune as at

    fast_dir = tmp_path / "rate_fast"
    slow_dir = tmp_path / "rate_slow"
    fast_dir.mkdir()
    slow_dir.mkdir()
    (fast_dir / "relax.lammpstrj").write_text("")
    (slow_dir / "relax.lammpstrj").write_text("")
    (slow_dir / "traj.extxyz").write_text("")

    seen: list[Path] = []

    def _fake_read(path, n_tail, *, type_to_species=None):
        seen.append(Path(path))
        return [Path(path).parent.name]

    monkeypatch.setattr(at, "read_last_frames_auto", _fake_read)

    frames = at._collect_rate_scan_cutoff_reference_frames(
        rate_results=[
            {"rate": 100.0, "replicates": [{"dump": "rate_fast/relax.lammpstrj"}]},
            {"rate": 10.0, "replicates": [{"dump": "rate_slow/relax.lammpstrj"}]},
        ],
        outdir=tmp_path,
        metrics_cfg=SimpleNamespace(time_average_frames=3),
        type_to_species=["Sm"],
    )

    assert frames == ["rate_fast", "rate_slow"]
    assert seen == [fast_dir / "relax.lammpstrj", slow_dir / "traj.extxyz"]
