from __future__ import annotations

import json
import shutil
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
    pc.write_graph_analysis_outputs = lambda outdir, *, boxes, rejected_boxes=None: {}
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
    pc.write_graph_analysis_outputs = lambda outdir, *, boxes, rejected_boxes=None: {}
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", pc)

    res = analyze_output_data(config=cfg, input_path=prod_dir, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 2
    assert seen_cutoffs[0] == {}
    assert seen_cutoffs[1] == {(1, 1): 3.25}


def test_analyze_output_data_per_box_scope_does_not_reuse_previous_box_cutoffs(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    prod_dir = tmp_path / "production"
    _write_box(prod_dir, 1)
    _write_box(prod_dir, 2)

    calls = _patch_generic_analysis(monkeypatch, cutoffs_after=3.25)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: (
            {},
            {
                "scope": "per_box",
                "mode": "per_box_auto",
                "plan_cutoffs_reused": False,
            },
        ),
    )

    result = oa.analyze_output_data(
        config=cfg,
        input_path=prod_dir,
        outdir=tmp_path / "analysis",
    )

    assert [dict(call["cutoffs"]) for call in calls] == [{}, {}]
    assert result["cutoffs"] == []
    assert result["cutoff_provenance"]["mode"] == "per_box_auto"


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

    outdir = tmp_path / "analysis"
    res = oa.analyze_output_data(config=cfg, input_path=source, outdir=outdir)

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

    outdir = tmp_path / "analysis"
    res = oa.analyze_output_data(config=cfg, input_path=source, outdir=outdir)

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

    outdir = tmp_path / "analysis"
    res = oa.analyze_output_data(config=cfg, input_path=source, outdir=outdir)

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert len(calls) == 1
    assert int(calls[0]["box_id"]) == 3
    assert Path(calls[0]["relax_stage_dir"]) == outdir / "box_artifacts" / "box_003"
    assert Path(calls[0]["relax_traj_path"]).name == "sample.traj"
    assert float(calls[0]["density_mean"]) == pytest.approx(2.82)
    assert sorted(path.name for path in source.parent.iterdir()) == ["sample.traj"]


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


def test_strict_final_restart_names_do_not_accept_plain_one_restart(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    root = tmp_path / "restart_names"
    root.mkdir()
    bad_plain = root / "Si3N4_0101.restart"
    bad_missing_marker = root / "Si3N4_010.restart"
    good_final = root / "Si3N4_010-1.restart"
    bad_plain.write_text("plain numbered restart; not final\n")
    bad_missing_marker.write_text("intermediate restart; not final\n")
    good_final.write_text("strict final restart\n")

    assert oa._is_strict_final_restart_name(good_final.name) is True
    assert oa._is_strict_final_restart_name(bad_plain.name) is False
    assert oa._is_strict_final_restart_name(bad_missing_marker.name) is False
    assert oa._is_analysis_source_candidate(good_final) is True
    assert oa._is_analysis_source_candidate(bad_plain) is False
    assert oa._is_analysis_source_candidate(bad_missing_marker) is False
    assert oa._box_id_from_label(good_final.name) == 10

    raw_boxes, entries, rejected, meta = oa.discover_output_dataset(root)

    assert entries == []
    assert rejected == []
    assert meta["layout"] == "flat_file_ensemble"
    assert [(box.box, box.analysis_source.name) for box in raw_boxes] == [(10, "Si3N4_010-1.restart")]


def test_standalone_source_globs_restrict_flat_analysis_to_strict_restart_finals(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    ctx = oa.analysis_context_from_standalone_config(
        {
            "analysis": {
                "type_to_species": ["Si", "N"],
                "sources": {"include_globs": ["*-1.restart"]},
                "metrics": {
                    "enabled": True,
                    "pairs": [{"pair": ["Si", "N"]}],
                    "rings": {"enabled": False},
                    "gr": [],
                    "sq": [],
                },
                "production": {"min_boxes": 1, "batch_boxes": 1},
            }
        }
    )
    ensemble = tmp_path / "external_sin"
    ensemble.mkdir()
    for name in (
        "Si3N4_001-1.restart",
        "Si3N4_002-1.restart",
        "Si3N4_001.restart",
        "Si3N4_0021.restart",
        "preview.extxyz",
    ):
        (ensemble / name).write_text(f"{name}\n")

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
    assert [Path(c["analysis_source_path"]).name for c in calls] == [
        "Si3N4_001-1.restart",
        "Si3N4_002-1.restart",
    ]
    assert [int(c["box_id"]) for c in calls] == [1, 2]
    assert res["analysis_source_roles"] == {"final_structure": 2}

    dataset = json.loads((tmp_path / "analysis" / "output_dataset.json").read_text())
    assert dataset["metadata"]["source_selection"] == {"include_globs": ["*-1.restart"], "exclude_globs": []}
    assert [Path(b["analysis_source"]).name for b in dataset["boxes"]] == [
        "Si3N4_001-1.restart",
        "Si3N4_002-1.restart",
    ]

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




def test_box_from_dirs_overrides_legacy_poscar_when_strict_final_restart_exists(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    box_dir = tmp_path / "box_259"
    box_dir.mkdir()
    poscar = box_dir / "POSCAR"
    strict_restart = box_dir / "box_259_hse06-1.restart"
    poscar.write_text("legacy oxygen-containing POSCAR that should not win analysis selection\n")
    strict_restart.write_text("strict final CP2K restart placeholder\n")

    box = oa._box_from_dirs(
        box_dir,
        box=259,
        relax_dir=box_dir,
        final_structure=poscar,
        relax_data=poscar,
        analysis_source=poscar,
        analysis_source_role="final_structure",
    )

    assert Path(box.final_structure).name == "box_259_hse06-1.restart"
    assert Path(box.relax_data).name == "box_259_hse06-1.restart"
    assert Path(box.analysis_source).name == "box_259_hse06-1.restart"
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
    pc.write_graph_analysis_outputs = lambda outdir, *, boxes, rejected_boxes=None: {}
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

    def _fake_read(path, n, *, type_to_species=None, atom_style="atomic", units_style=None):
        assert units_style == "metal"
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
    pc.write_graph_analysis_outputs = lambda outdir, *, boxes, rejected_boxes=None: {}
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


def test_results_replay_uses_recorded_time_average_source_not_generic_final_frame(
    monkeypatch,
    tmp_path: Path,
):
    """A result replay must preserve production's estimator, not just its box.

    Current production records ``traj.extxyz`` as the exact time-averaging
    source while the same relax directory also contains a high-confidence
    ``final.extxyz``.  Generic source discovery prefers the latter.  Results
    replay is stricter and must bind the former or live/analyze-output
    convergence can silently describe different estimators.
    """

    import vitriflow.workflows.output_analysis as oa

    monkeypatch.setattr(
        oa, "_authenticate_current_results_replay", lambda path, payload: True
    )

    root = tmp_path / "run"
    relax = root / "production" / "box_001" / "relax"
    melt = root / "production" / "box_001" / "melt"
    quench = root / "production" / "box_001" / "quench"
    for directory in (relax, melt, quench):
        directory.mkdir(parents=True, exist_ok=True)
    trajectory = relax / "traj.extxyz"
    final = relax / "final.extxyz"
    relax_data = relax / "relax.data"
    relax_dump = relax / "relax.lammpstrj"
    for path in (trajectory, final, relax_data, relax_dump):
        path.write_text("placeholder\n")

    result_path = root / "autotune_results.json"
    result_path.write_text(
        json.dumps(
            {
                "production_plan": {"schema": "vitriflow.production_plan.v1"},
                "production": {
                    "convergence": {"passed": True},
                    "boxes": [
                        {
                            "box": 1,
                            "density": 2.2,
                            "density_stderr": 0.01,
                            "analysis_source_role": "relax_trajectory",
                            "paths": {
                                "analysis_source": "production/box_001/relax/traj.extxyz",
                                "relax_traj": "production/box_001/relax/traj.extxyz",
                                "relax_dump": "production/box_001/relax/relax.lammpstrj",
                                "relax_data": "production/box_001/relax/relax.data",
                                "relax_dir": "production/box_001/relax",
                                "melt_dir": "production/box_001/melt",
                                "quench_dir": "production/box_001/quench",
                            },
                        }
                    ],
                    "rejected_boxes": [],
                },
            }
        )
    )

    raw, entries, rejected, metadata = oa.discover_output_dataset(result_path)

    assert entries == []
    assert rejected == []
    assert len(raw) == 1
    assert raw[0].analysis_source == trajectory.resolve()
    assert raw[0].analysis_source_role == "relax_trajectory"
    assert raw[0].relax_traj == trajectory.resolve()
    assert raw[0].final_structure == final.resolve()
    assert raw[0].density == pytest.approx(2.2)
    assert raw[0].density_stderr == pytest.approx(0.01)
    assert metadata["layout"] == "production_results_recorded_sources"
    assert metadata["exact_replay_source_contract"] is True


def test_results_replay_fails_closed_when_recorded_analysis_source_is_missing(
    monkeypatch,
    tmp_path: Path,
):
    import vitriflow.workflows.output_analysis as oa

    monkeypatch.setattr(
        oa, "_authenticate_current_results_replay", lambda path, payload: True
    )

    root = tmp_path / "run"
    (root / "production" / "box_001" / "relax").mkdir(parents=True)
    result_path = root / "run_results.json"
    result_path.write_text(
        json.dumps(
            {
                "production_plan": {"schema": "vitriflow.production_plan.v1"},
                "production": {
                    "convergence": {"passed": True},
                    "boxes": [
                        {
                            "box": 1,
                            "analysis_source_role": "relax_trajectory",
                            "paths": {
                                "analysis_source": "production/box_001/relax/missing.extxyz",
                                "relax_dir": "production/box_001/relax",
                            },
                        }
                    ],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="recorded analysis source/role"):
        oa.discover_output_dataset(result_path)


def test_custom_schedule_results_replay_accepts_recorded_box_zero(
    monkeypatch,
    tmp_path: Path,
):
    """Custom schedule is intentionally zero-based; autotune/run remain positive."""

    import hashlib

    import vitriflow.workflows.output_analysis as oa

    monkeypatch.setattr(
        oa, "_authenticate_current_results_replay", lambda path, payload: True
    )

    root = tmp_path / "custom_run"
    relax = root / "production" / "box_000" / "relax"
    relax.mkdir(parents=True)
    trajectory = relax / "traj.extxyz"
    trajectory.write_text("placeholder\n")

    payload = {"workflow": "custom_stage_schedule"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    fingerprint = {
        "payload": payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }
    result_path = root / "run_results.json"
    result_path.write_text(
        json.dumps(
            {
                "workflow": "custom_stage_schedule",
                "resume_fingerprint": fingerprint,
                "production": {
                    "convergence": {"passed": True},
                    "boxes": [
                        {
                            "box": 0,
                            "density": 2.2,
                            "density_stderr": 0.01,
                            "analysis_source_role": "relax_trajectory",
                            "paths": {
                                "analysis_source": (
                                    "production/box_000/relax/traj.extxyz"
                                ),
                                "relax_traj": (
                                    "production/box_000/relax/traj.extxyz"
                                ),
                                "relax_dir": "production/box_000/relax",
                            },
                        }
                    ],
                },
            }
        )
    )

    raw, entries, rejected, metadata = oa.discover_output_dataset(result_path)

    assert entries == []
    assert rejected == []
    assert len(raw) == 1
    assert raw[0].box == 0
    assert raw[0].analysis_source == trajectory.resolve()
    assert raw[0].analysis_source_role == "relax_trajectory"
    assert metadata["exact_replay_source_contract"] is True

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.0}, {"mode": "test"}),
    )
    analysed = oa.analyze_output_data(
        config=_config(),
        input_path=result_path,
        outdir=tmp_path / "analysis",
    )
    assert analysed["status"] == "ok"
    assert [int(entry["box"]) for entry in analysed["boxes"]] == [0]
    assert [int(call["box_id"]) for call in calls] == [0]
    public_box = analysed["boxes"][0]
    assert public_box["structure_embedding"]["structure_reference"] == (
        "structure_references/box_000000.json"
    )
    structure_manifest = json.loads(
        (tmp_path / "analysis" / "structure_manifest.json").read_text()
    )
    assert [row["box_id"] for row in structure_manifest["structures"]] == [0]
    structure_references = json.loads(
        (tmp_path / "analysis" / "structure_references.json").read_text()
    )
    assert [row["box_id"] for row in structure_references["references"]] == [0]


def test_zero_box_id_survives_worker_failure_and_strict_id_validation() -> None:
    import vitriflow.workflows.output_analysis as oa

    failed = oa._analysis_worker_task({"box_id": 0})
    assert failed["ok"] is False
    assert failed["box"] == 0

    assert oa._strict_analysis_box_id(0, context="test", minimum=0) == 0
    large = 2**53 + 1
    assert oa._strict_analysis_box_id(large, context="test", minimum=0) == large
    for invalid in (True, -1, 0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            oa._strict_analysis_box_id(invalid, context="test", minimum=0)


def test_custom_schedule_convergence_parity_requires_self_hashed_matching_config():
    import hashlib

    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    payload = {
        "workflow": "custom_stage_schedule",
        "runner": {"engine": str(cfg.engine)},
        "md": cfg.md.model_dump(mode="json"),
        "potential": {"config": cfg.kim.model_dump(mode="json")},
        "metrics": cfg.autotune.metrics.model_dump(mode="json"),
        "convergence": cfg.autotune.convergence.model_dump(mode="json"),
        "production_acceptance": cfg.autotune.production.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    fingerprint = {
        "payload": payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }
    contract = {
        "workflow": "custom_stage_schedule",
        "resume_fingerprint": fingerprint,
    }

    assert oa._custom_schedule_replay_config_matches(cfg, contract) is True

    changed_cfg = cfg.model_copy(
        update={
            "autotune": cfg.autotune.model_copy(
                update={
                    "convergence": cfg.autotune.convergence.model_copy(
                        update={"density_abs_tol": cfg.autotune.convergence.density_abs_tol + 1.0}
                    )
                }
            )
        }
    )
    assert oa._custom_schedule_replay_config_matches(changed_cfg, contract) is False

    tampered = json.loads(json.dumps(contract))
    tampered["resume_fingerprint"]["payload"]["metrics"]["enabled"] = False
    assert oa._custom_schedule_replay_config_matches(cfg, tampered) is False


def _current_local_results_payload(tmp_path: Path, engine_bundle: dict) -> dict:
    from vitriflow.workflows.resume_integrity import (
        attach_production_state_integrity,
        canonical_json_sha256,
    )

    plan = {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": "base.data",
    }
    (tmp_path / "base.data").write_text("structure\n")
    production = attach_production_state_integrity(
        {
            "enabled": False,
            "status": "disabled",
            "execution_status": "disabled",
            "converged": False,
            "check_convergence": False,
            "n_boxes": 0,
            "n_boxes_accepted": 0,
            "n_boxes_rejected": 0,
            "n_boxes_total": 0,
            "boxes": [],
            "rejected_boxes": [],
        },
        outdir=tmp_path,
    )
    fingerprint_payload = {
        "schema": "vitriflow.run.resume_fingerprint.v5",
        "workflow": "run_meltquench",
        "external_mode": "local",
        "engine_build_identities": engine_bundle,
        "production_plan": plan,
    }
    return {
        "status": "disabled",
        "execution_status": "disabled",
        "production_plan": plan,
        "production": production,
        "resume_fingerprint": {
            "schema": "vitriflow.run.resume_fingerprint.v5",
            "algorithm": "sha256:c14n-json:v1",
            "sha256": canonical_json_sha256(fingerprint_payload),
            "payload": fingerprint_payload,
        },
    }


def test_current_results_exact_replay_authenticates_all_crosslinks(
    tmp_path: Path,
    mock_engine_build_identities,
):
    import vitriflow.workflows.output_analysis as oa

    payload = _current_local_results_payload(
        tmp_path,
        mock_engine_build_identities["bundle"](
            _config(), primary_engine="lammps"
        ),
    )
    result_path = tmp_path / "run_results.json"
    result_path.write_text(json.dumps(payload, sort_keys=True))
    assert oa._authenticate_current_results_replay(result_path, payload) is True

    modified_plan = json.loads(json.dumps(payload))
    modified_plan["production_plan"]["structure_data"] = "other.data"
    with pytest.raises(ValueError, match="protected production plan"):
        oa._authenticate_current_results_replay(result_path, modified_plan)

    modified_state = json.loads(json.dumps(payload))
    modified_state["production"]["converged"] = True
    with pytest.raises(ValueError, match="production authentication failed"):
        oa._authenticate_current_results_replay(result_path, modified_state)

    modified_status = json.loads(json.dumps(payload))
    modified_status["status"] = "ok"
    with pytest.raises(ValueError, match="status disagrees"):
        oa._authenticate_current_results_replay(result_path, modified_status)


def test_legacy_results_never_claim_exact_replay(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    payload = {
        "production_plan": {"schema": "vitriflow.production_plan.v1"},
        "production": {"convergence": {"passed": True}, "boxes": []},
    }
    result_path = tmp_path / "run_results.json"
    result_path.write_text(json.dumps(payload))
    assert oa._authenticate_current_results_replay(result_path, payload) is False


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


def test_output_analysis_rejects_duplicate_box_ids_before_keyed_aggregation():
    import vitriflow.workflows.output_analysis as oa

    with pytest.raises(ValueError, match="duplicate box id 7"):
        oa._require_unique_dataset_box_ids(
            [],
            [{"box": 7, "metrics": {}}, {"box": 7, "metrics": {}}],
            [],
        )


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


def test_standalone_embed_structures_false_passes_through_to_box_analysis(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    ctx = oa.analysis_context_from_standalone_config(
        {
            "analysis": {
                "embed_structures": False,
                "type_to_species": ["Al"],
                "metrics": {
                    "enabled": True,
                    "pairs": [{"pair": ["Al", "Al"]}],
                    "rings": {"enabled": False},
                    "gr": [],
                    "sq": [],
                },
                "production": {"min_boxes": 1, "batch_boxes": 1},
            }
        }
    )
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    (ensemble / "box_001.vasp").write_text("vasp snapshot\n")

    calls = _patch_generic_analysis(monkeypatch)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.70)
    monkeypatch.setattr(
        oa,
        "_resolve_output_analysis_cutoffs",
        lambda **kwargs: ({(1, 1): 3.0}, {"mode": "pooled_ensemble_auto"}),
    )

    res = oa.analyze_output_data(analysis_context=ctx, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["embed_structures"] is False
    assert calls and calls[0]["embed_structures"] is False
    written = json.loads((tmp_path / "analysis" / "analysis_results.json").read_text())
    assert written["embed_structures"] is False
    assert written["sidecar_integrity"]["exists"] is True



def test_full_config_and_plan_embed_structures_false_are_supported():
    import vitriflow.workflows.output_analysis as oa

    cfg = _config()
    prod_cfg = cfg.autotune.production.model_copy(update={"embed_structures": False})
    ctx = oa._analysis_context_from_config(cfg, prod_cfg=prod_cfg)
    assert ctx.embed_structures is False
    assert ctx.prod_cfg.embed_structures is False

    plan_ctx = oa._analysis_context_from_plan(
        cfg,
        {
            "metrics_cfg": {
                "enabled": True,
                "type_to_species": ["Al"],
                "pairs": [{"pair": ["Al", "Al"]}],
            },
            "production_cfg": {"embed_structures": False, "min_boxes": 1, "batch_boxes": 1},
            "convergence_cfg": {},
            "type_to_species": ["Al"],
            "T_high": 3000.0,
            "t_final": 300.0,
            "quench_steps": 1000,
            "md_use": {"timestep": 1.0, "atom_style": "atomic"},
        },
    )
    assert plan_ctx.embed_structures is False
    assert plan_ctx.prod_cfg.embed_structures is False

def test_analysis_only_reject_flags_are_advisory(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    ctx = oa.analysis_context_from_standalone_config(
        {
            "analysis": {
                "type_to_species": ["Al"],
                "metrics": {"enabled": True, "pairs": [{"pair": ["Al", "Al"]}], "rings": {"enabled": False}, "gr": [], "sq": []},
                "production": {"min_boxes": 1, "batch_boxes": 1, "exclude_coordination_defects": True},
            }
        }
    )
    ensemble = tmp_path / "ensemble_advisory"
    ensemble.mkdir()
    (ensemble / "box_001.vasp").write_text("vasp snapshot\n")

    def _fake_analyse(**kwargs):
        return (
            {
                "box": int(kwargs["box_id"]),
                "density": 2.7,
                "metrics": {"density_mean": 2.7},
                "distributions": {},
                "paths": {},
                "reject": {"reason": "coordination_defects", "box": int(kwargs["box_id"])},
            },
            dict(kwargs["cutoffs"]),
        )

    pc = types.ModuleType("vitriflow.workflows.production_common")
    pc.analyse_production_box = _fake_analyse
    pc.build_production_convergence_spec = lambda entry: {"metrics": ["density_mean"]}
    pc.validate_production_entry_against_spec = lambda *a, **k: None
    pc.check_production_convergence = lambda boxes, spec, cfg: (False, {"familywise": {"alpha_per_test": 0.05}, "groups": {}})
    pc.metrics_checked_from_conv_spec = lambda spec: list(spec.get("metrics", []))
    pc.resolve_production_time_unit_ps = lambda **kwargs: 1.0
    pc.resolve_production_warmup_duration_ps = lambda *, prod_cfg: 0.0
    pc.resolve_production_warmup_steps = lambda *, prod_cfg, md_timestep, time_unit_ps: 0
    pc.write_graph_analysis_outputs = lambda outdir, *, boxes, rejected_boxes=None: {}
    monkeypatch.setitem(sys.modules, "vitriflow.workflows.production_common", pc)
    monkeypatch.setattr(oa, "_estimate_density_from_source", lambda *a, **k: 2.70)
    monkeypatch.setattr(oa, "_resolve_output_analysis_cutoffs", lambda **kwargs: ({(1, 1): 3.0}, {"mode": "pooled_ensemble_auto"}))

    res = oa.analyze_output_data(analysis_context=ctx, input_path=ensemble, outdir=tmp_path / "analysis")

    assert res["status"] == "ok"
    assert res["n_boxes"] == 1
    assert res["n_boxes_rejected"] == 0
    assert res["n_boxes_would_be_rejected"] == 1
    assert res["filtering"]["mode"] == "advisory"
    assert res["boxes"][0]["reject"] is False
    assert res["boxes"][0]["reject_advisory"]["reason"] == "coordination_defects"


def _task_diagnostic_fixture(tmp_path: Path) -> tuple[Path, dict]:
    plan = {
        "schema": "vitriflow.production_task_diagnostic_plan.v1",
        "engine": "lammps",
        "stage_metrics": {
            "enabled": True,
            "roles": ["melt", "quench", "relax"],
            "plot_required": True,
        },
        "elastic_screens": {
            "supported": True,
            "roles": {
                "melt": {"enabled": True, "strict": True},
                "relax": {"enabled": False, "strict": False},
            },
        },
        "elastic_timeseries": {
            "supported": True,
            "roles": {
                "melt": {"enabled": False, "strict": False},
                "quench": {"enabled": True, "strict": True},
                "relax": {"enabled": False, "strict": False},
            },
        },
    }
    box_dir = tmp_path / "box_001"
    box_dir.mkdir()
    (box_dir / "task.json").write_text(
        json.dumps(
            {
                "schema": "vitriflow.box_task.v1",
                "diagnostic_plan": plan,
            }
        )
    )
    ok = lambda role: {"status": "ok", "stage_role": role}
    result = {
        "schema": "vitriflow.box_task_result.v1",
        "status": "ok",
        "density": 2.70,
        "density_stderr": 0.02,
        "diagnostics": {
            "schema": "vitriflow.production_task_diagnostics.v1",
            "status": "ok",
            "path_base": "task_box",
            "plan": plan,
            "stage_metrics": {
                "melt": ok("melt"),
                "quench": ok("quench"),
                "relax": ok("relax"),
            },
            "elastic_screens": {"melt": ok("melt"), "relax": None},
            "elastic_timeseries": {
                "melt": None,
                "quench": ok("quench"),
                "relax": None,
            },
        },
    }
    task_result = box_dir / "task_result.json"
    task_result.write_text(json.dumps(result))
    return task_result, result


def test_legacy_external_task_diagnostics_are_not_reused(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    task_result, _result = _task_diagnostic_fixture(tmp_path)
    assert oa._validated_task_diagnostics(task_result) is None


def test_legacy_external_task_diagnostics_remain_generic_when_mutated(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    task_result, result = _task_diagnostic_fixture(tmp_path)
    del result["diagnostics"]["stage_metrics"]["quench"]
    task_result.write_text(json.dumps(result))
    assert oa._validated_task_diagnostics(task_result) is None


def test_v2_task_diagnostics_with_mutated_csv_are_never_reused(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa
    from vitriflow.workflows.resume_integrity import canonical_json_sha256

    task_result, result = _task_diagnostic_fixture(tmp_path)
    csv_path = task_result.parent / "melt" / "metrics_timeseries.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("Step,value\n0,1.0\n")
    result["schema"] = "vitriflow.box_task_result.v2"
    result["diagnostics"]["stage_metrics"]["melt"]["csv"] = str(
        csv_path.relative_to(task_result.parent)
    )
    result["result_integrity"] = {
        "schema": "vitriflow.box_task_result_integrity.v1",
        "algorithm": "sha256:c14n-json:v1",
        "payload_sha256": canonical_json_sha256(result),
    }
    task_result.write_text(json.dumps(result, sort_keys=True))
    csv_path.write_text("Step,value\n0,999.0\n")

    assert oa._validated_task_diagnostics(task_result) is None


def _current_task_diagnostic_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """Create one worker-authentic current task/result/diagnostic bundle."""

    from vitriflow.workflows.resume_integrity import (
        TASK_RESULT_SCHEMA,
        seal_task_result,
        sha256_file,
        task_manifest_sha256,
    )

    box_dir = tmp_path / "box_001"
    box_dir.mkdir()
    plan = {
        "schema": "vitriflow.production_task_diagnostic_plan.v1",
        "engine": "lammps",
        "stage_metrics": {
            "enabled": True,
            "roles": ["melt", "quench", "relax"],
            "plot_required": False,
        },
        "elastic_screens": {"supported": True, "roles": {}},
        "elastic_timeseries": {"supported": True, "roles": {}},
    }
    task = {
        "schema": "vitriflow.box_task.v1",
        "task": {"box": 1, "box_dir": str(box_dir)},
        "diagnostic_plan": plan,
    }
    task_json = box_dir / "task.json"
    task_json.write_text(json.dumps(task, sort_keys=True))

    outcomes: dict[str, dict[str, object]] = {}
    stage_outputs: dict[str, Path] = {}
    stage_metrics: dict[str, dict[str, object]] = {}
    diagnostic_csvs: dict[str, Path] = {}
    artifact_files: list[tuple[Path, bool]] = []
    for role in ("melt", "quench", "relax"):
        stage_dir = box_dir / role
        stage_dir.mkdir()
        output = stage_dir / f"{role}.data"
        output.write_text(f"{role} structure\n")
        csv_path = stage_dir / "metrics_timeseries.csv"
        csv_path.write_text("Step,value\n0,1.0\n")
        summary = stage_dir / "metrics_timeseries.json"
        summary.write_text('{"status":"ok"}\n')
        outcomes[role] = {"output_data": output.name, "dump": None}
        stage_outputs[role] = output
        diagnostic_csvs[role] = csv_path
        stage_metrics[role] = {
            "status": "ok",
            "stage_role": role,
            "csv": str(csv_path.relative_to(box_dir)),
            "summary": str(summary.relative_to(box_dir)),
        }
        artifact_files.extend(((output, True), (csv_path, True), (summary, True)))

    identities = []
    for path, required in artifact_files:
        identities.append(
            {
                "path": str(path.relative_to(box_dir)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "required": required,
            }
        )
    diagnostics = {
        "schema": "vitriflow.production_task_diagnostics.v1",
        "status": "ok",
        "path_base": "task_box",
        "plan": plan,
        "stage_metrics": stage_metrics,
        "elastic_screens": {},
        "elastic_timeseries": {},
    }
    result = seal_task_result(
        {
            "schema": TASK_RESULT_SCHEMA,
            "status": "ok",
            "box": 1,
            "engine_build_identity_end_verified": True,
            "task_manifest_sha256": task_manifest_sha256(task),
            "outcomes": outcomes,
            "diagnostics": diagnostics,
            "artifact_manifest": {
                "schema": "vitriflow.task_artifacts.v1",
                "files": identities,
            },
        }
    )
    task_result = box_dir / "task_result.json"
    task_result.write_text(json.dumps(result, sort_keys=True))
    return task_result, task_json, stage_outputs["relax"], diagnostic_csvs["relax"]


def test_current_task_diagnostics_authenticate_task_and_all_artifacts(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    task_result, _task_json, _relax_output, _diagnostic_csv = (
        _current_task_diagnostic_fixture(tmp_path)
    )
    reused = oa._validated_task_diagnostics(task_result)

    assert reused is not None
    assert reused["provenance"]["mode"] == "validated_read_only_reuse"


def test_external_task_authentication_survives_copying_complete_tree(
    tmp_path: Path,
    mock_engine_build_identities,
):
    """Absolute worker paths must not make an authenticated result immovable."""

    import vitriflow.workflows.output_analysis as oa
    from vitriflow.workflows.resume_integrity import (
        seal_task_result,
        task_manifest_sha256,
    )

    original = tmp_path / "original"
    production_dir = original / "production"
    production_dir.mkdir(parents=True)
    task_result, task_json, _relax_output, _diagnostic_csv = (
        _current_task_diagnostic_fixture(production_dir)
    )
    box_dir = task_json.parent
    protected_plan = {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": "base.data",
    }
    protected_execution = {"lammps_cmd": "lmp", "nprocs": 1}
    task = json.loads(task_json.read_text())
    task["production_plan"] = protected_plan
    task["config"] = {"lammps": protected_execution}
    task["task"].update(
        {
            "box_dir": str(box_dir),
            "task_json": str(task_json),
            "task_result": str(task_result),
        }
    )
    task_json.write_text(json.dumps(task, sort_keys=True))

    engine_identity = mock_engine_build_identities["identities"]["lammps"]
    result = json.loads(task_result.read_text())
    result.pop("result_integrity")
    result["task_manifest_sha256"] = task_manifest_sha256(task)
    result["engine_build_identity"] = engine_identity
    task_result.write_text(json.dumps(seal_task_result(result), sort_keys=True))

    production = {
        "boxes": [{"box": 1}],
        "rejected_boxes": [],
        "n_boxes_total": 1,
        "ensemble_dir": "production",
        "engine_build_identity": engine_identity,
        "engine_build_identity_status": "verified_homogeneous_workers",
    }
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)

    checked = oa._authenticate_external_task_results(
        results_path=moved / "run_results.json",
        production=production,
        protected_plan=protected_plan,
        protected_execution_config=protected_execution,
    )
    assert checked["identity_sha256"] == engine_identity["identity_sha256"]


def test_current_task_diagnostics_reject_mutated_task_manifest(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    task_result, task_json, _relax_output, _diagnostic_csv = (
        _current_task_diagnostic_fixture(tmp_path)
    )
    task = json.loads(task_json.read_text())
    task["tampered"] = True
    task_json.write_text(json.dumps(task, sort_keys=True))

    with pytest.raises(ValueError, match="does not authenticate the adjacent task manifest"):
        oa._validated_task_diagnostics(task_result)


def test_current_task_diagnostics_reject_mutated_relax_output(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    task_result, _task_json, relax_output, _diagnostic_csv = (
        _current_task_diagnostic_fixture(tmp_path)
    )
    relax_output.write_text("modified relax structure\n")

    with pytest.raises(ValueError, match="artifact is missing or changed: relax/relax.data"):
        oa._validated_task_diagnostics(task_result)


def test_current_task_diagnostics_reject_mutated_diagnostic_csv(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    task_result, _task_json, _relax_output, diagnostic_csv = (
        _current_task_diagnostic_fixture(tmp_path)
    )
    diagnostic_csv.write_text("Step,value\n0,999.0\n")

    with pytest.raises(
        ValueError,
        match="artifact is missing or changed: relax/metrics_timeseries.csv",
    ):
        oa._validated_task_diagnostics(task_result)


def test_current_task_diagnostics_reject_resealed_truncated_manifest(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa
    from vitriflow.workflows.resume_integrity import seal_task_result

    task_result, _task_json, _relax_output, _diagnostic_csv = (
        _current_task_diagnostic_fixture(tmp_path)
    )
    result = json.loads(task_result.read_text())
    result.pop("result_integrity")
    result["artifact_manifest"]["files"] = [
        record
        for record in result["artifact_manifest"]["files"]
        if record["path"] != "quench/metrics_timeseries.json"
    ]
    task_result.write_text(json.dumps(seal_task_result(result), sort_keys=True))

    with pytest.raises(
        ValueError,
        match="artifact manifest is incomplete or non-canonical",
    ):
        oa._validated_task_diagnostics(task_result)


def test_successful_task_density_and_uncertainty_are_authoritative(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    task_result = tmp_path / "task_result.json"
    task_result.write_text(
        json.dumps(
            {
                "schema": "vitriflow.box_task_result.v1",
                "status": "ok",
                "density": 2.71,
                "density_stderr": 0.031,
            }
        )
    )

    assert oa._authoritative_task_density(task_result) == pytest.approx((2.71, 0.031))


def test_dft_replay_selects_only_positive_cp2k_cell_opt_final(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    box_dir = tmp_path / "box_001"
    relax = box_dir / "relax"
    relax.mkdir(parents=True)
    (relax / "relax.data").write_text("source")
    dft_dir = box_dir / "dft_opt"
    dft_dir.mkdir()
    (dft_dir / "dft_opt.data").write_text("refined")
    (dft_dir / "cp2k.out").write_text("CELL OPTIMIZATION COMPLETED\n")
    source_box = oa._box_from_dirs(box_dir, box=1)

    selected, rejected = oa._select_dft_refined_sources(
        [source_box], required_box_ids=[1]
    )

    assert rejected == []
    assert len(selected) == 1
    assert selected[0].analysis_source == dft_dir / "dft_opt.data"
    assert selected[0].analysis_source_role == "dft_opt_final"
    assert selected[0].density is None
    assert selected[0].density_stderr == 0.0


def test_dft_replay_rejects_normal_termination_without_cell_opt_success(tmp_path: Path):
    import vitriflow.workflows.output_analysis as oa

    box_dir = tmp_path / "box_001"
    relax = box_dir / "relax"
    relax.mkdir(parents=True)
    (relax / "relax.data").write_text("source")
    dft_dir = box_dir / "dft_opt"
    dft_dir.mkdir()
    (dft_dir / "dft_opt.data").write_text("refined")
    (dft_dir / "cp2k.out").write_text("PROGRAM ENDED AT\n")
    source_box = oa._box_from_dirs(box_dir, box=1)

    with pytest.raises(RuntimeError, match="lacks valid CP2K CELL_OPT evidence"):
        oa._select_dft_refined_sources([source_box], required_box_ids=[1])
