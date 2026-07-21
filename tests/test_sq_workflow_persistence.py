from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vitriflow.analysis.dump import DumpFrame
from vitriflow.config import StructureMetricsConfig
from vitriflow.workflows import autotune
from vitriflow.workflows import production_common as pc


def _frame() -> DumpFrame:
    return DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 1], dtype=int),
        positions=np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        cell=np.eye(3, dtype=float) * 10.0,
        origin=np.zeros(3, dtype=float),
    )


def _metrics(*, gr: list[dict] | None = None, sq: list[dict] | None = None) -> StructureMetricsConfig:
    metrics = StructureMetricsConfig.model_validate(
        {
            "enabled": True,
            "collect_during_production_stages": False,
            "graph_rules": [],
            "rings": {"enabled": False},
            "amorphous": {"enabled": False},
            "gr": list(gr or []),
            "sq": list(sq or []),
        }
    )
    # Keep these focused tests independent of the mandatory production void
    # policy; the workflow's curve-generation path is the subject under test.
    metrics.voids.enabled = False
    return metrics


def _prepare_production_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "read_last_frames_auto", lambda *args, **kwargs: [_frame()])
    monkeypatch.setattr(
        pc,
        "compute_structure_metrics_timeavg",
        lambda *args, **kwargs: SimpleNamespace(values={}),
    )
    monkeypatch.setattr(
        pc,
        "compute_structure_distributions_timeavg",
        lambda *args, **kwargs: {"bondlen": {}, "angle": {}, "coord": {}, "void": {}},
    )
    monkeypatch.setattr(pc, "should_collect_stage_metrics_timeseries", lambda *args, **kwargs: False)
    monkeypatch.setattr(pc, "compute_coordination_defects", lambda *args, **kwargs: {})


def _analyse_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metrics: StructureMetricsConfig,
    *,
    type_to_species: list[str],
) -> dict:
    _prepare_production_analysis(monkeypatch)
    source = tmp_path / "analysis.extxyz"
    source.write_text("immutable test fixture\n")
    entry, _cutoffs = pc.analyse_production_box(
        box_id=1,
        outdir=tmp_path,
        melt_stage_dir=tmp_path / "melt",
        quench_stage_dir=tmp_path / "quench",
        relax_stage_dir=tmp_path,
        relax_data_path=source,
        density_mean=2.0,
        density_stderr=0.0,
        metrics_cfg=metrics,
        cutoffs={},
        required_pairs=[],
        fixed_cutoffs={},
        type_to_species=type_to_species,
        md_timestep=1.0,
        analysis_source_path=source,
    )
    return entry


def _sq_config(pair=None) -> dict:
    return {
        "pair": pair,
        "q_max": 8.0,
        "nq": 16,
        "r_max": 4.0,
        "nbins": 64,
        "window": "hann",
        "peak_search": [0.5, 3.0],
    }


def _gr_config(pair=None) -> dict:
    return {"pair": pair, "r_max": 4.0, "nbins": 32}


def test_production_persists_sq_representation_metadata(monkeypatch, tmp_path: Path) -> None:
    from vitriflow.analysis import sq as sq_module

    representation = {
        "schema": "vitriflow.sq_representation.v1",
        "normalization": "fixture-normalization",
        "frame_aggregation": "fixture-aggregation",
    }
    calls: list[dict] = []

    def _fake_compute_sq(*args, **kwargs):
        calls.append(dict(kwargs))
        assert kwargs.get("return_metadata") is True
        return (
            np.asarray([0.0, 1.0], dtype=float),
            np.asarray([1.0, 1.5], dtype=float),
            dict(representation),
        )

    monkeypatch.setattr(sq_module, "compute_sq", _fake_compute_sq)
    entry = _analyse_production(
        monkeypatch,
        tmp_path,
        _metrics(sq=[_sq_config()]),
        type_to_species=["Si"],
    )

    assert len(calls) == 1
    payload = entry["distributions"]["sq"]["sq_all"]
    assert payload["q"] == [0.0, 1.0]
    assert payload["s"] == [1.0, 1.5]
    assert payload["representation"] == representation


@pytest.mark.parametrize("family", ["gr", "sq"])
def test_production_rejects_slug_colliding_curve_keys_before_second_compute(
    monkeypatch,
    tmp_path: Path,
    family: str,
) -> None:
    from vitriflow.analysis import sq as sq_module

    calls = 0

    def _fake_gr(*args, **kwargs):
        nonlocal calls
        calls += 1
        return np.asarray([0.5]), np.asarray([1.0]), 0.5

    def _fake_sq(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs.get("return_metadata") is True
        return np.asarray([0.0]), np.asarray([1.0]), {"schema": "vitriflow.sq_representation.v1"}

    monkeypatch.setattr(pc, "compute_gr", _fake_gr)
    monkeypatch.setattr(sq_module, "compute_sq", _fake_sq)
    collision_pairs = [["A+B", "C"], ["A B", "C"]]
    metrics = _metrics(
        gr=[_gr_config(pair) for pair in collision_pairs] if family == "gr" else [],
        sq=[_sq_config(pair) for pair in collision_pairs] if family == "sq" else [],
    )

    expected = r"Duplicate generated (?:g\(r\)|S\(q\)) curve key '" + family + r"_a_b_c'"
    with pytest.raises(ValueError, match=expected):
        _analyse_production(
            monkeypatch,
            tmp_path,
            metrics,
            type_to_species=["A+B", "A B", "C"],
        )

    assert calls == 1


def test_autotune_dft_curve_builder_persists_sq_representation_metadata(monkeypatch) -> None:
    from vitriflow.analysis import sq as sq_module

    representation = {
        "schema": "vitriflow.sq_representation.v1",
        "normalization": "fixture-dft-normalization",
    }

    def _fake_compute_sq(*args, **kwargs):
        assert kwargs.get("return_metadata") is True
        return (
            np.asarray([0.0, 2.0], dtype=float),
            np.asarray([1.0, 0.8], dtype=float),
            dict(representation),
        )

    monkeypatch.setattr(sq_module, "compute_sq", _fake_compute_sq)
    gr_curves, sq_curves = autotune._compute_dft_distribution_curves(
        [_frame()],
        metrics_cfg=_metrics(sq=[_sq_config()]),
        type_to_species=["Si"],
    )

    assert gr_curves == {}
    assert sq_curves["sq_all"]["representation"] == representation
    assert sq_curves["sq_all"]["q"] == [0.0, 2.0]
    assert sq_curves["sq_all"]["s"] == [1.0, 0.8]


@pytest.mark.parametrize("family", ["gr", "sq"])
def test_autotune_dft_curve_builder_rejects_slug_collisions_before_second_compute(
    monkeypatch,
    family: str,
) -> None:
    from vitriflow.analysis import sq as sq_module

    calls = 0

    def _fake_gr(*args, **kwargs):
        nonlocal calls
        calls += 1
        return np.asarray([0.5]), np.asarray([1.0]), 0.5

    def _fake_sq(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs.get("return_metadata") is True
        return np.asarray([0.0]), np.asarray([1.0]), {"schema": "vitriflow.sq_representation.v1"}

    monkeypatch.setattr(autotune, "compute_gr", _fake_gr)
    monkeypatch.setattr(sq_module, "compute_sq", _fake_sq)
    collision_pairs = [["A+B", "C"], ["A B", "C"]]
    metrics = _metrics(
        gr=[_gr_config(pair) for pair in collision_pairs] if family == "gr" else [],
        sq=[_sq_config(pair) for pair in collision_pairs] if family == "sq" else [],
    )

    expected = r"Duplicate generated DFT (?:g\(r\)|S\(q\)) curve key '" + family + r"_a_b_c'"
    with pytest.raises(ValueError, match=expected):
        autotune._compute_dft_distribution_curves(
            [_frame()],
            metrics_cfg=metrics,
            type_to_species=["A+B", "A B", "C"],
        )

    assert calls == 1
