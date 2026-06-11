from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def _config():
    from vitriflow.config import RunConfig

    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "structure": {"generate": {"method": "random", "formula": "Al", "n_formula_units": 1}},
            "autotune": {
                "metrics": {
                    "enabled": True,
                    "type_to_species": ["Al"],
                    "pairs": [{"pair": ["Al", "Al"]}],
                },
                "production": {"enabled": True, "min_boxes": 1, "max_boxes": 2, "batch_boxes": 1},
            },
        }
    )


def _write_box(root: Path, box_id: int) -> Path:
    box_dir = root / f"box_{box_id:03d}"
    relax_dir = box_dir / "relax"
    relax_dir.mkdir(parents=True)
    (relax_dir / "relax.data").write_text("LAMMPS data file\n\n0 atoms\n")
    (relax_dir / "traj.extxyz").write_text("")
    return box_dir


def _write_box_with_relaxation_sources(root: Path, box_id: int, *, restart_name: str = "calc-1.restart") -> Path:
    box_dir = root / f"box_{box_id:03d}"
    box_dir.mkdir(parents=True)
    (box_dir / "input.xyz").write_text("initial structure\n")
    (box_dir / "traj.xyz").write_text("relax trajectory\n")
    (box_dir / restart_name).write_text("final restart\n")
    return box_dir


def _patch_generic_analysis(monkeypatch, *, cutoffs_after: float = 3.1):
    calls: list[dict] = []

    def _fake_analyse(**kwargs):
        calls.append(dict(kwargs))
        return (
            {
                "box": int(kwargs["box_id"]),
                "metrics": {"density_mean": 2.7 + 0.01 * int(kwargs["box_id"])},
                "distributions": {},
                "paths": {"relax_dir": str(kwargs["relax_stage_dir"])},
                "reject": False,
            },
            {(1, 1): float(cutoffs_after)},
        )

    pc = types.ModuleType("vitriflow.workflows.production_common")
    pc.analyse_production_box = _fake_analyse
    pc.build_production_convergence_spec = lambda entry: {"metrics": ["density_mean"]}
    pc.validate_production_entry_against_spec = lambda *a, **k: None
    pc.check_production_convergence = lambda boxes, spec, cfg: (True, {"ok": True, "n": len(boxes)})
    pc.metrics_checked_from_conv_spec = lambda spec: list(spec.get("metrics", []))
    pc.resolve_production_time_unit_ps = lambda **kwargs: 1.0
    pc.resolve_production_warmup_duration_ps = lambda *, prod_cfg: float(getattr(prod_cfg, "warmup_duration_ps", 5.0))
    pc.resolve_production_warmup_steps = lambda *, prod_cfg, md_timestep, time_unit_ps: int(round(float(getattr(prod_cfg, "warmup_duration_ps", 5.0)) / (float(md_timestep) * float(time_unit_ps))))
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", pc)
    return calls


def _install_fake_ase_for_generic_discovery(monkeypatch):
    import numpy as np

    class _FakeAtoms:
        def __len__(self):
            return 2

        def get_global_number_of_atoms(self):
            return 2

        def get_chemical_symbols(self):
            return ["Al", "Al"]

        def get_positions(self):
            return np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)

        def get_cell(self):
            return np.diag([5.0, 5.0, 5.0])

        def get_volume(self):
            return 125.0

        def get_masses(self):
            return np.asarray([26.9815385, 26.9815385], dtype=float)

    def _fake_read(path, index=None, format=None, style=None, specorder=None):
        p = Path(path)
        if p.suffix.lower() not in {".traj", ".cfg"}:
            raise ValueError(f"unsupported fake ASE path: {p}")
        return _FakeAtoms()

    ase_mod = types.ModuleType("ase")
    io_mod = types.ModuleType("ase.io")
    io_mod.read = _fake_read
    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.io", io_mod)


def test_analyze_output_data_single_task_result_falls_back_to_raw_box(monkeypatch, tmp_path: Path):
    from vitriflow.workflows.output_analysis import analyze_output_data

    cfg = _config()
    box_dir = _write_box(tmp_path / "production", 1)
    task_result = box_dir / "task_result.json"
    task_result.write_text(
        json.dumps(
            {
                "schema": "vitriflow.box_task_result.v1",
                "status": "ok",
                "box": 1,
                "task": {"box": 1},
            },
            indent=2,
        )
    )

    calls = _patch_generic_analysis(monkeypatch)
    outdir = tmp_path / "analysis"
    res = analyze_output_data(config=cfg, input_path=task_result, outdir=outdir)

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert [int(b["box"]) for b in res["boxes"]] == [1]
    assert len(calls) == 1
    assert calls[0]["box_id"] == 1
    assert (outdir / "analysis_results.json").exists()
    assert (outdir / "output_dataset.json").exists()


def test_analyze_output_data_accepts_results_json_input(monkeypatch, tmp_path: Path):
    from vitriflow.workflows.output_analysis import analyze_output_data

    cfg = _config()
    prod_dir = tmp_path / "production"
    _write_box(prod_dir, 1)
    results_json = tmp_path / "run_results.json"
    results_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "production": {
                    "ensemble_dir": "production",
                },
            },
            indent=2,
        )
    )

    calls = _patch_generic_analysis(monkeypatch)
    res = analyze_output_data(config=cfg, input_path=results_json, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert len(calls) == 1
    assert calls[0]["box_id"] == 1


def test_analyze_output_data_reuses_shared_cutoffs_for_raw_boxes(monkeypatch, tmp_path: Path):
    from vitriflow.workflows.output_analysis import analyze_output_data

    cfg = _config()
    prod_dir = tmp_path / "production"
    _write_box(prod_dir, 1)
    _write_box(prod_dir, 2)

    seen_cutoffs: list[dict] = []

    def _fake_analyse(**kwargs):
        seen_cutoffs.append(dict(kwargs["cutoffs"]))
        box_id = int(kwargs["box_id"])
        return (
            {
                "box": box_id,
                "metrics": {"density_mean": 2.7 + 0.01 * box_id},
                "distributions": {},
                "paths": {},
                "reject": False,
            },
            {(1, 1): 3.25},
        )

    pc = types.ModuleType("vitriflow.workflows.production_common")
    pc.analyse_production_box = _fake_analyse
    pc.build_production_convergence_spec = lambda entry: {"metrics": ["density_mean"]}
    pc.validate_production_entry_against_spec = lambda *a, **k: None
    pc.check_production_convergence = lambda boxes, spec, cfg: (True, {"ok": True, "n": len(boxes)})
    pc.metrics_checked_from_conv_spec = lambda spec: list(spec.get("metrics", []))
    pc.resolve_production_time_unit_ps = lambda **kwargs: 1.0
    pc.resolve_production_warmup_duration_ps = lambda *, prod_cfg: float(getattr(prod_cfg, "warmup_duration_ps", 5.0))
    pc.resolve_production_warmup_steps = lambda *, prod_cfg, md_timestep, time_unit_ps: int(round(float(getattr(prod_cfg, "warmup_duration_ps", 5.0)) / (float(md_timestep) * float(time_unit_ps))))
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", pc)

    res = analyze_output_data(config=cfg, input_path=prod_dir, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert seen_cutoffs[0] == {}
    assert seen_cutoffs[1] == {(1, 1): 3.25}


def test_analyze_output_data_accepts_loose_box_directories_with_structure_files(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    ensemble = tmp_path / "ensemble"
    (ensemble / "sample_a").mkdir(parents=True)
    (ensemble / "sample_b").mkdir(parents=True)
    (ensemble / "sample_a" / "CONTCAR").write_text("vasp snapshot\n")
    (ensemble / "sample_b" / "POSCAR").write_text("vasp snapshot\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.55)

    res = oa.analyze_output_data(config=cfg, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert [int(c["box_id"]) for c in calls] == [1, 2]
    assert [Path(c["relax_traj_path"]).name for c in calls] == ["CONTCAR", "POSCAR"]
    assert all(float(c["density_mean"]) == pytest.approx(2.55) for c in calls)



def test_analyze_output_data_accepts_directory_of_direct_structure_files(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    (ensemble / "box_001.vasp").write_text("vasp snapshot\n")
    (ensemble / "box_002.pdb").write_text("cp2k snapshot\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.60)

    res = oa.analyze_output_data(config=cfg, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert [int(c["box_id"]) for c in calls] == [1, 2]
    assert [Path(c["relax_traj_path"]).name for c in calls] == ["box_001.vasp", "box_002.pdb"]
    assert all(float(c["density_mean"]) == pytest.approx(2.60) for c in calls)



def test_analyze_output_data_accepts_single_structure_file_input(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    source = tmp_path / "box_007.vasp"
    source.write_text("vasp snapshot\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.70)

    res = oa.analyze_output_data(config=cfg, input_path=source, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert len(calls) == 1
    assert int(calls[0]["box_id"]) == 7
    assert Path(calls[0]["relax_traj_path"]).name == "box_007.vasp"
    assert float(calls[0]["density_mean"]) == pytest.approx(2.70)


def test_analyze_output_data_accepts_directory_of_direct_ase_readable_files_with_unknown_suffixes(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    _install_fake_ase_for_generic_discovery(monkeypatch)
    cfg = _config()
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    (ensemble / "a_sample.traj").write_text("ase trajectory\n")
    (ensemble / "b_other.cfg").write_text("ase cfg\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.80)

    res = oa.analyze_output_data(config=cfg, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert [int(c["box_id"]) for c in calls] == [1, 2]
    assert [Path(c["relax_traj_path"]).name for c in calls] == ["a_sample.traj", "b_other.cfg"]
    assert all(float(c["density_mean"]) == pytest.approx(2.80) for c in calls)


def test_analyze_output_data_accepts_single_ase_readable_file_with_unknown_suffix(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    _install_fake_ase_for_generic_discovery(monkeypatch)
    cfg = _config()
    source = tmp_path / "sample.traj"
    source.write_text("ase trajectory\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.81)

    res = oa.analyze_output_data(config=cfg, input_path=source, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert len(calls) == 1
    assert int(calls[0]["box_id"]) == 1
    assert Path(calls[0]["relax_traj_path"]).name == "sample.traj"
    assert float(calls[0]["density_mean"]) == pytest.approx(2.81)


def test_analyze_output_data_single_ase_source_inside_relax_dir_uses_parent_box_id(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    _install_fake_ase_for_generic_discovery(monkeypatch)
    cfg = _config()
    source = tmp_path / "box_003" / "relax" / "sample.traj"
    source.parent.mkdir(parents=True)
    source.write_text("ase trajectory\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.82)

    res = oa.analyze_output_data(config=cfg, input_path=source, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert len(calls) == 1
    assert int(calls[0]["box_id"]) == 3
    assert Path(calls[0]["relax_stage_dir"]).name == "relax"
    assert Path(calls[0]["relax_traj_path"]).name == "sample.traj"
    assert float(calls[0]["density_mean"]) == pytest.approx(2.82)


def test_analyze_output_data_single_restart_inside_box_dir_uses_parent_box_id(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    source = tmp_path / "box_004" / "calc-1.restart"
    source.parent.mkdir(parents=True)
    source.write_text("final restart\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.83)

    res = oa.analyze_output_data(config=cfg, input_path=source, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert len(calls) == 1
    assert int(calls[0]["box_id"]) == 4
    assert Path(calls[0]["analysis_source_path"]).name == "calc-1.restart"
    assert float(calls[0]["density_mean"]) == pytest.approx(2.83)


def test_role_aware_source_resolution_prefers_restart_for_analysis(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    box_dir = _write_box_with_relaxation_sources(tmp_path, 7)

    box = oa._box_from_dirs(box_dir)

    assert box.input_structure is not None
    assert box.final_structure is not None
    assert box.relax_traj is not None
    assert box.analysis_source is not None
    assert Path(box.input_structure).name == "input.xyz"
    assert Path(box.final_structure).name == "calc-1.restart"
    assert Path(box.relax_traj).name == "traj.xyz"
    assert Path(box.analysis_source).name == "calc-1.restart"
    assert box.analysis_source_role == "final_structure"


def test_analyze_output_data_passes_restart_as_analysis_source_but_keeps_relax_traj(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    ensemble = tmp_path / "ensemble"
    _write_box_with_relaxation_sources(ensemble, 1)

    calls: list[dict] = []

    def _fake_analyse(**kwargs):
        calls.append(dict(kwargs))
        return (
            {
                "box": int(kwargs["box_id"]),
                "metrics": {"density_mean": 2.75},
                "distributions": {},
                "paths": {},
                "reject": False,
            },
            dict(kwargs["cutoffs"]),
        )

    pc = types.ModuleType("vitriflow.workflows.production_common")
    pc.analyse_production_box = _fake_analyse
    pc.build_production_convergence_spec = lambda entry: {"metrics": ["density_mean"]}
    pc.validate_production_entry_against_spec = lambda *a, **k: None
    pc.check_production_convergence = lambda boxes, spec, cfg: (True, {"ok": True, "n": len(boxes)})
    pc.metrics_checked_from_conv_spec = lambda spec: list(spec.get("metrics", []))
    pc.resolve_production_time_unit_ps = lambda **kwargs: 1.0
    pc.resolve_production_warmup_duration_ps = lambda *, prod_cfg: float(getattr(prod_cfg, "warmup_duration_ps", 5.0))
    pc.resolve_production_warmup_steps = lambda *, prod_cfg, md_timestep, time_unit_ps: int(round(float(getattr(prod_cfg, "warmup_duration_ps", 5.0)) / (float(md_timestep) * float(time_unit_ps))))
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", pc)

    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.55)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.45}, {"mode": "pooled_ensemble_auto", "plan_cutoffs_reused": False}),
    )

    res = oa.analyze_output_data(config=cfg, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["cutoff_provenance"]["mode"] == "pooled_ensemble_auto"
    assert res["analysis_source_roles"] == {"final_structure": 1}
    assert len(calls) == 1
    assert Path(calls[0]["relax_data_path"]).name == "calc-1.restart"
    assert Path(calls[0]["relax_traj_path"]).name == "traj.xyz"
    assert Path(calls[0]["analysis_source_path"]).name == "calc-1.restart"
    assert calls[0]["analysis_source_role"] == "final_structure"


def test_resolve_output_analysis_cutoffs_pools_current_ensemble_and_not_plan(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    src1 = tmp_path / "box_001.restart"
    src2 = tmp_path / "box_002.restart"
    src1.write_text("restart one\n")
    src2.write_text("restart two\n")

    raw_boxes = [oa._box_from_source_file(src1, box=1), oa._box_from_source_file(src2, box=2)]
    ctx = oa._analysis_context_from_config(cfg, cutoffs={(1, 1): 9.9})

    read_calls: list[tuple[str, int]] = []
    pooled_seen: list[list[str]] = []

    def _fake_read(path, n, *, type_to_species=None, atom_style="atomic"):
        read_calls.append((Path(path).name, int(n)))
        return [f"frame:{Path(path).name}"]

    def _fake_estimate(frames, required_pairs, *, auto, fixed_cutoffs):
        pooled_seen.append(list(frames))
        assert required_pairs == [(1, 1)]
        assert fixed_cutoffs == {}
        return {(1, 1): 3.21}

    monkeypatch.setattr(oa, "read_last_frames_auto", _fake_read)
    structure_mod = types.ModuleType("vitriflow.analysis.structure")
    structure_mod.estimate_pair_cutoffs = _fake_estimate
    monkeypatch.setitem(sys.modules, "vitriflow.analysis.structure", structure_mod)

    cutoffs, provenance = oa._resolve_output_analysis_cutoffs(
        raw_boxes=raw_boxes,
        ctx=ctx,
        required_pairs=[(1, 1)],
        fixed_cutoffs={},
    )

    assert cutoffs == {(1, 1): 3.21}
    assert provenance["mode"] == "pooled_ensemble_auto"
    assert provenance["plan_cutoffs_available"] is True
    assert provenance["plan_cutoffs_reused"] is False
    assert provenance["n_boxes_sampled"] == 2
    assert provenance["n_frames_sampled"] == 2
    assert sorted(read_calls) == [("box_001.restart", 5), ("box_002.restart", 5)]
    assert pooled_seen == [["frame:box_001.restart", "frame:box_002.restart"]]


def test_analyze_output_data_accepts_flat_lammps_final_structure_directory(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    ensemble = tmp_path / "flat_lammps"
    ensemble.mkdir()
    # Formula digits must not be concatenated into the box id.  These should
    # become boxes 1 and 2, not 34001 and 34002.
    (ensemble / "Si3N4_001.lmp").write_text("LAMMPS data placeholder\n")
    (ensemble / "Si3N4_002.lmp").write_text("LAMMPS data placeholder\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.95)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.0}, {"mode": "pooled_ensemble_auto"}),
    )

    res = oa.analyze_output_data(config=cfg, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert [int(c["box_id"]) for c in calls] == [1, 2]
    assert [Path(c["analysis_source_path"]).name for c in calls] == ["Si3N4_001.lmp", "Si3N4_002.lmp"]
    assert [c["analysis_source_role"] for c in calls] == ["final_structure", "final_structure"]

    dataset = json.loads((tmp_path / "analysis" / "output_dataset.json").read_text())
    assert dataset["metadata"]["layout"] == "flat_file_ensemble"
    assert dataset["metadata"]["n_flat_file_sources"] == 2
    assert [int(b["box"]) for b in dataset["boxes"]] == [1, 2]
    assert all(b["source_layout"] == "flat_file_ensemble" for b in dataset["boxes"])



def test_flat_extxyz_final_structure_directory_is_supported(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    ensemble = tmp_path / "flat_extxyz"
    ensemble.mkdir()
    # Minimal periodic EXTXYZ final structures. Discovery should treat direct
    # EXTXYZ files as final structures in a flat-file ensemble.
    for idx in (1, 2):
        (ensemble / f"final_{idx:03d}.extxyz").write_text(
            "2\n"
            'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
            "Al 0.0 0.0 0.0\n"
            "Al 1.0 1.0 1.0\n"
        )

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.70)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.0}, {"mode": "pooled_ensemble_auto"}),
    )

    res = oa.analyze_output_data(config=cfg, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert [int(c["box_id"]) for c in calls] == [1, 2]
    assert [Path(c["analysis_source_path"]).name for c in calls] == ["final_001.extxyz", "final_002.extxyz"]
    assert [c["analysis_source_role"] for c in calls] == ["final_structure", "final_structure"]

    dataset = json.loads((tmp_path / "analysis" / "output_dataset.json").read_text())
    assert dataset["metadata"]["layout"] == "flat_file_ensemble"
    assert all(b["source_layout"] == "flat_file_ensemble" for b in dataset["boxes"])


def test_extxyz_symbol_types_follow_configured_species_order(tmp_path: Path):
    from vitriflow.analysis.trajectory import read_last_frames_auto

    source = tmp_path / "symbols_only.extxyz"
    source.write_text(
        "2\n"
        'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        # N appears first, but configured type order below says Si=1, N=2.
        "N 0.0 0.0 0.0\n"
        "Si 1.0 1.0 1.0\n"
    )

    frames = read_last_frames_auto(source, 1, type_to_species=["Si", "N"])

    assert len(frames) == 1
    assert frames[0].types.tolist() == [2, 1]

def test_flat_final_structure_directory_uses_per_box_artifact_dirs(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    ensemble = tmp_path / "flat_lammps"
    ensemble.mkdir()
    (ensemble / "sample_1.data").write_text("LAMMPS data placeholder\n")
    (ensemble / "sample_2.data").write_text("LAMMPS data placeholder\n")

    calls: list[dict] = []

    def _fake_analyse(**kwargs):
        calls.append(dict(kwargs))
        stage = Path(kwargs["relax_stage_dir"])
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "artifact.txt").write_text(str(kwargs["box_id"]))
        return (
            {
                "box": int(kwargs["box_id"]),
                "metrics": {"density_mean": 2.7},
                "distributions": {},
                "paths": {"relax_dir": str(kwargs["relax_stage_dir"])},
                "reject": False,
            },
            dict(kwargs["cutoffs"]),
        )

    pc = types.ModuleType("vitriflow.workflows.production_common")
    pc.analyse_production_box = _fake_analyse
    pc.build_production_convergence_spec = lambda entry: {"metrics": ["density_mean"]}
    pc.validate_production_entry_against_spec = lambda *a, **k: None
    pc.check_production_convergence = lambda boxes, spec, cfg: (True, {"ok": True, "n": len(boxes)})
    pc.metrics_checked_from_conv_spec = lambda spec: list(spec.get("metrics", []))
    pc.resolve_production_time_unit_ps = lambda **kwargs: 1.0
    pc.resolve_production_warmup_duration_ps = lambda *, prod_cfg: float(getattr(prod_cfg, "warmup_duration_ps", 5.0))
    pc.resolve_production_warmup_steps = lambda *, prod_cfg, md_timestep, time_unit_ps: int(round(float(getattr(prod_cfg, "warmup_duration_ps", 5.0)) / (float(md_timestep) * float(time_unit_ps))))
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", pc)

    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.95)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.0}, {"mode": "pooled_ensemble_auto"}),
    )

    outdir = tmp_path / "analysis"
    res = oa.analyze_output_data(config=cfg, input_path=ensemble, outdir=outdir)

    assert res["status"] == "ok"
    stage_dirs = [Path(c["relax_stage_dir"]) for c in calls]
    assert stage_dirs == [outdir / "box_artifacts" / "box_001", outdir / "box_artifacts" / "box_002"]
    assert all((d / "artifact.txt").exists() for d in stage_dirs)
    assert not (ensemble / "artifact.txt").exists()


def test_unknown_suffix_lammps_data_file_is_discovered_by_header(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    source = tmp_path / "Si3N4_003.finalframe"
    source.write_text(
        "LAMMPS data file\n\n"
        "2 atoms\n"
        "2 atom types\n\n"
        "0.0 5.0 xlo xhi\n"
        "0.0 5.0 ylo yhi\n"
        "0.0 5.0 zlo zhi\n\n"
        "Atoms # atomic\n\n"
        "1 1 0.0 0.0 0.0\n"
        "2 2 1.0 1.0 1.0\n"
    )

    assert oa._is_analysis_source_candidate(source)
    raw_boxes, entries, rejected, meta = oa.discover_output_dataset(tmp_path)
    assert entries == []
    assert rejected == []
    assert meta["layout"] == "flat_file_ensemble"
    assert [b.box for b in raw_boxes] == [3]
    assert raw_boxes[0].analysis_source == source


def test_flat_discovery_skips_lammps_input_and_log_files(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    root = tmp_path / "flat"
    root.mkdir()
    (root / "final_001.lammps").write_text(
        "LAMMPS data file\n\n"
        "1 atoms\n"
        "1 atom types\n\n"
        "0.0 5.0 xlo xhi\n"
        "0.0 5.0 ylo yhi\n"
        "0.0 5.0 zlo zhi\n\n"
        "Atoms # atomic\n\n"
        "1 1 0.0 0.0 0.0\n"
    )
    (root / "melt.in.lammps").write_text("read_data input.data\nrun 100\n")
    (root / "log.lammps").write_text("LAMMPS log file\n")

    raw_boxes, entries, rejected, meta = oa.discover_output_dataset(root)

    assert entries == []
    assert rejected == []
    assert meta["layout"] == "flat_file_ensemble"
    assert [(b.box, b.analysis_source.name) for b in raw_boxes] == [(1, "final_001.lammps")]


def test_standalone_analysis_config_builds_context_for_external_structures():
    import vitriflow.workflows.output_analysis as oa

    ctx = oa.analysis_context_from_standalone_config(
        {
            "analysis": {
                "type_to_species": ["Si", "N"],
                "atom_style": "charge",
                "timestep": 0.5,
                "metrics": {
                    "enabled": True,
                    "pairs": [{"pair": ["Si", "N"]}],
                    "coordinations": [
                        {"central": "Si", "neighbor": "N", "expected": 4},
                        {"central": "N", "neighbor": "Si", "expected": 3},
                    ],
                    "rings": {
                        "enabled": True,
                        "mode": "bond_graph",
                        "nodes": ["Si", "N"],
                        "bond_pairs": [{"pair": ["Si", "N"]}],
                    },
                },
                "production": {"min_boxes": 1, "batch_boxes": 1},
            }
        }
    )

    assert ctx.type_to_species == ["Si", "N"]
    assert ctx.atom_style == "charge"
    assert ctx.md_timestep == 0.5
    assert ctx.metrics_cfg.enabled is True
    assert ctx.metrics_cfg.time_average_frames == 1
    assert ctx.prod_cfg.min_boxes == 1
    assert ctx.prod_cfg.exclude_coordination_defects is False


def test_analyze_output_data_accepts_standalone_context_for_flat_final_structures(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    ctx = oa.analysis_context_from_standalone_config(
        {
            "analysis": {
                "type_to_species": ["Si", "N"],
                "metrics": {
                    "enabled": True,
                    "pairs": [{"pair": ["Si", "N"]}],
                    "coordinations": [
                        {"central": "Si", "neighbor": "N", "expected": 4},
                        {"central": "N", "neighbor": "Si", "expected": 3},
                    ],
                    "rings": {
                        "enabled": True,
                        "mode": "bond_graph",
                        "nodes": ["Si", "N"],
                        "bond_pairs": [{"pair": ["Si", "N"]}],
                    },
                },
                "production": {"min_boxes": 1, "batch_boxes": 1},
            }
        }
    )
    ensemble = tmp_path / "external_finals"
    ensemble.mkdir()
    (ensemble / "Si3N4_001.data").write_text("LAMMPS data placeholder\n")
    (ensemble / "Si3N4_002.data").write_text("LAMMPS data placeholder\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 3.05)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 2): 2.2}, {"mode": "pooled_ensemble_auto"}),
    )

    res = oa.analyze_output_data(
        analysis_context=ctx,
        input_path=ensemble,
        outdir=tmp_path / "analysis",
    )

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert res["effective_metrics"] == {"source": "standalone_analysis_config"}
    assert [Path(c["analysis_source_path"]).name for c in calls] == ["Si3N4_001.data", "Si3N4_002.data"]
    assert all(c["type_to_species"] == ["Si", "N"] for c in calls)
    assert all(c["atom_style"] == "atomic" for c in calls)

    dataset = json.loads((tmp_path / "analysis" / "output_dataset.json").read_text())
    assert dataset["metadata"]["layout"] == "flat_file_ensemble"
    assert [int(b["box"]) for b in dataset["boxes"]] == [1, 2]


def _install_fake_ase_for_any_format_and_db(monkeypatch):
    import numpy as np

    class _FakeAtoms:
        def __len__(self):
            return 2

        def get_global_number_of_atoms(self):
            return 2

        def get_chemical_symbols(self):
            return ["Al", "Al"]

        def get_positions(self):
            return np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)

        def get_cell(self):
            return np.diag([5.0, 5.0, 5.0])

        def get_volume(self):
            return 125.0

        def get_masses(self):
            return np.asarray([26.9815385, 26.9815385], dtype=float)

    class _FakeRow:
        def __init__(self, row_id, label):
            self.id = int(row_id)
            self.name = str(label)
            self.key_value_pairs = {"label": str(label)}

        def toatoms(self):
            return _FakeAtoms()

    class _FakeDB:
        def __init__(self, path):
            self.path = str(path)
            self._rows = [_FakeRow(11, "box_011"), _FakeRow(12, "box_012")]

        def select(self, *args, **kwargs):
            return list(self._rows)

        def get(self, *args, **kwargs):
            row_id = kwargs.get("id", args[0] if args else None)
            for row in self._rows:
                if int(row.id) == int(row_id):
                    return row
            raise KeyError(row_id)

    def _fake_connect(path):
        return _FakeDB(path)

    def _fake_read(path, index=None, format=None, style=None, specorder=None):
        p = Path(path)
        if p.suffix.lower() in {".json", ".foo", ".bar"}:
            return _FakeAtoms()
        raise ValueError(f"unsupported fake ASE path: {p}")

    def _fake_write(path, atoms, format=None):
        Path(path).write_text(
            "2\n"
            'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
            "Al 0 0 0\n"
            "Al 1 1 1\n"
        )

    ase_mod = types.ModuleType("ase")
    io_mod = types.ModuleType("ase.io")
    db_mod = types.ModuleType("ase.db")
    io_mod.read = _fake_read
    io_mod.write = _fake_write
    db_mod.connect = _fake_connect
    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.io", io_mod)
    monkeypatch.setitem(sys.modules, "ase.db", db_mod)


def test_flat_directory_accepts_ase_readable_json_format(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    _install_fake_ase_for_any_format_and_db(monkeypatch)
    root = tmp_path / "flat_any_ase"
    root.mkdir()
    (root / "box_001.json").write_text('{"ase_atoms": true}')
    (root / "output_dataset.json").write_text('{"schema": "vitriflow.output_dataset.v1", "boxes": []}')

    # The reserved output_dataset.json is still not a structure source, but a
    # non-reserved ASE-readable JSON file can be used as a final structure.
    (root / "output_dataset.json").unlink()
    raw_boxes, entries, rejected, meta = oa.discover_output_dataset(root)

    assert entries == []
    assert rejected == []
    assert meta["layout"] == "flat_file_ensemble"
    assert [(b.box, b.analysis_source.name, b.analysis_source_role) for b in raw_boxes] == [
        (1, "box_001.json", "final_structure")
    ]


def test_discover_output_dataset_accepts_ase_database_file(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    _install_fake_ase_for_any_format_and_db(monkeypatch)
    db_path = tmp_path / "structures.db"
    db_path.write_text("fake sqlite db")

    raw_boxes, entries, rejected, meta = oa.discover_output_dataset(db_path)

    assert entries == []
    assert rejected == []
    assert meta["layout"] == "ase_database"
    assert meta["n_database_rows"] == 2
    assert [b.box for b in raw_boxes] == [11, 12]
    assert [b.source_layout for b in raw_boxes] == ["ase_database", "ase_database"]
    assert [b.source_record["row_id"] for b in raw_boxes] == [11, 12]
    assert all(b.analysis_source == db_path for b in raw_boxes)


def test_analyze_output_data_accepts_ase_database_file(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    _install_fake_ase_for_any_format_and_db(monkeypatch)
    cfg = _config()
    db_path = tmp_path / "structures.db"
    db_path.write_text("fake sqlite db")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.0}, {"mode": "pooled_ensemble_auto"}),
    )
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.85)

    res = oa.analyze_output_data(config=cfg, input_path=db_path, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert [int(c["box_id"]) for c in calls] == [11, 12]
    assert [Path(c["analysis_source_path"]).name for c in calls] == ["box_011.extxyz", "box_012.extxyz"]
    assert all(Path(c["analysis_source_path"]).exists() for c in calls)
    assert [c["analysis_source_role"] for c in calls] == ["final_structure", "final_structure"]

    dataset = json.loads((tmp_path / "analysis" / "output_dataset.json").read_text())
    assert dataset["metadata"]["layout"] == "ase_database"
    assert [b["source_layout"] for b in dataset["boxes"]] == ["ase_database", "ase_database"]
    assert [b["source_record"]["row_id"] for b in dataset["boxes"]] == [11, 12]


def test_original_vitriflow_directory_layout_still_uses_relax_trajectory_for_analysis(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    prod_dir = tmp_path / "production"
    _write_box(prod_dir, 1)

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.70)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.0}, {"mode": "pooled_ensemble_auto"}),
    )

    res = oa.analyze_output_data(config=cfg, input_path=prod_dir, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert len(calls) == 1
    assert Path(calls[0]["relax_data_path"]).name == "relax.data"
    assert Path(calls[0]["relax_traj_path"]).name == "traj.extxyz"
    assert Path(calls[0]["analysis_source_path"]).name == "traj.extxyz"
    assert calls[0]["analysis_source_role"] == "relax_trajectory"

    dataset = json.loads((tmp_path / "analysis" / "output_dataset.json").read_text())
    assert dataset["metadata"]["layout"] == "directory"
    assert dataset["boxes"][0]["analysis_source_role"] == "relax_trajectory"
    assert Path(dataset["boxes"][0]["analysis_source"]).name == "traj.extxyz"
