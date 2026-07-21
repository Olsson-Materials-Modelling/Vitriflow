from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vitriflow.plotting import (
    plot_production_comparison_results,
    plot_production_results,
)


def _ci_halfwidth(values: list[float], factor: float = 0.75) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(factor * np.std(arr, ddof=1) / np.sqrt(float(arr.size)))


def _vector_ci_halfwidth(rows: list[list[float]], factor: float = 0.75) -> list[float]:
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return [0.0 for _ in range(int(arr.shape[-1]) if arr.ndim > 0 else 0)]
    return (factor * np.std(arr, axis=0, ddof=1) / np.sqrt(float(arr.shape[0]))).tolist()


def _make_analysis_results(path: Path, *, offset: float) -> Path:
    x_bond = [0.0, 1.0, 2.0, 3.0]
    r_gr = [0.5, 1.0, 1.5, 2.0]

    boxes = []
    densities: list[float] = []
    ring_means: list[float] = []
    ring_fracs: list[float] = []
    bond_cdfs: list[list[float]] = []
    gr_curves: list[list[float]] = []

    for box_id in range(1, 5):
        density = 2.40 + float(offset) + 0.03 * float(box_id)
        ring_frac = 0.18 + 0.01 * float(box_id) + 0.02 * float(offset)
        ring_mean = 5.40 + float(offset) + 0.08 * float(box_id)
        bond_cdf = [
            0.0,
            min(0.95, 0.18 + 0.015 * float(box_id) + 0.01 * float(offset)),
            min(0.99, 0.82 + 0.010 * float(box_id) + 0.01 * float(offset)),
            1.0,
        ]
        gr_curve = [
            0.15 + 0.02 * float(offset),
            1.10 + 0.06 * float(box_id) + 0.03 * float(offset),
            0.55 + 0.02 * float(box_id) + 0.02 * float(offset),
            0.18 + 0.01 * float(offset),
        ]

        densities.append(density)
        ring_fracs.append(ring_frac)
        ring_means.append(ring_mean)
        bond_cdfs.append(bond_cdf)
        gr_curves.append(gr_curve)

        boxes.append(
            {
                "box": box_id,
                "density": density,
                "density_stderr": 0.0,
                "metrics": {
                    "bondlen_A-B_mean": 1.52 + float(offset) + 0.01 * float(box_id),
                    "bondlen_A-B_std": 0.08 + 0.003 * float(box_id),
                    "gr_all_peak_r": 1.52 + 0.02 * float(offset) + 0.005 * float(box_id),
                    "gr_all_peak_height": 1.90 + 0.05 * float(offset) + 0.03 * float(box_id),
                    "ring_frac_3": ring_frac,
                    "ring_mean_size": ring_mean,
                },
                "distributions": {
                    "bondlen": {
                        "bondlen_A-B": {
                            "x": list(x_bond),
                            "cdf": bond_cdf,
                        }
                    },
                    "angle": {},
                    "coord": {},
                    "void": {},
                    "gr": {
                        "gr_all": {
                            "label": "all",
                            "r": list(r_gr),
                            "g": gr_curve,
                        }
                    },
                    "sq": {},
                },
                "paths": {},
                "analysis_source_role": "final_structure",
            }
        )

    bond_mean = np.mean(np.asarray(bond_cdfs, dtype=float), axis=0).tolist()
    gr_mean = np.mean(np.asarray(gr_curves, dtype=float), axis=0).tolist()

    results = {
        "schema": "vitriflow.analysis_results.v1",
        "status": "ok",
        "error": None,
        "converged": True,
        "n_boxes": len(boxes),
        "n_boxes_accepted": len(boxes),
        "n_boxes_rejected": 0,
        "n_boxes_total": len(boxes),
        "check_convergence": True,
        "exclude_coordination_defects": False,
        "rejects_subdir": "rejects",
        "warmup_start_temperature": 300.0,
        "warmup_duration_ps": 0.0,
        "warmup_steps": 0,
        "cutoffs": [],
        "cutoff_provenance": {"mode": "pooled_ensemble_auto"},
        "convergence_spec": {
            "bondlen_names": ["bondlen_A-B"],
            "angle_names": [],
            "coord_names": [],
            "ring_keys": ["ring_frac_3"],
            "ring_has_mean_size": True,
            "gr_labels": ["gr_all"],
            "sq_labels": [],
            "void_names": [],
        },
        "convergence": {
            "zscore": 1.96,
            "mode": "both",
            "n_boxes": len(boxes),
            "familywise": {
                "method": "bonferroni",
                "alpha_family": 0.05,
                "m_tests": 4,
                "alpha_per_test": 0.05,
                "crit": 1.96,
                "crit_method": "z",
                "bounded_ci_method": "t",
            },
            "scalars": {
                "density": {
                    "group": "long",
                    "mean": float(np.mean(densities)),
                    "std": float(np.std(densities, ddof=1)),
                    "stderr": float(np.std(densities, ddof=1) / np.sqrt(float(len(densities)))),
                    "ci_halfwidth": _ci_halfwidth(densities),
                    "rel_tol": 0.02,
                    "abs_tol": 0.01,
                    "tol": max(0.01, 0.02 * float(np.mean(densities))),
                    "passed": True,
                },
                "ring_mean_size": {
                    "group": "medium",
                    "mean": float(np.mean(ring_means)),
                    "std": float(np.std(ring_means, ddof=1)),
                    "stderr": float(np.std(ring_means, ddof=1) / np.sqrt(float(len(ring_means)))),
                    "ci_halfwidth": _ci_halfwidth(ring_means),
                    "rel_tol": 0.0,
                    "abs_tol": 0.50,
                    "tol": 0.50,
                    "passed": True,
                },
            },
            "distributions": {
                "ring": {
                    "group": "medium",
                    "kind": "pmf",
                    "keys": ["ring_frac_3"],
                    "mean": [float(np.mean(ring_fracs))],
                    "stderr": [float(np.std(ring_fracs, ddof=1) / np.sqrt(float(len(ring_fracs))))],
                    "ci_halfwidth": [_ci_halfwidth(ring_fracs)],
                    "rel_tol": 0.0,
                    "abs_tol": 0.05,
                    "tol": [0.05],
                    "passed": True,
                    "worst_index": 0,
                    "worst_key": "ring_frac_3",
                },
                "bondlen_A-B": {
                    "group": "short",
                    "kind": "bondlen_cdf",
                    "x": list(x_bond),
                    "mean": bond_mean,
                    "stderr": _vector_ci_halfwidth(bond_cdfs, factor=1.0 / 0.75),
                    "ci_halfwidth": _vector_ci_halfwidth(bond_cdfs),
                    "rel_tol": 0.0,
                    "abs_tol": 0.10,
                    "tol": [0.10 for _ in x_bond],
                    "passed": True,
                    "worst_index": 1,
                    "worst_x": float(x_bond[1]),
                },
                "gr_all": {
                    "group": "long",
                    "kind": "gr_curve",
                    "r": list(r_gr),
                    "mean": gr_mean,
                    "stderr": _vector_ci_halfwidth(gr_curves, factor=1.0 / 0.75),
                    "ci_halfwidth": _vector_ci_halfwidth(gr_curves),
                    "rel_tol": 0.0,
                    "abs_tol": 0.10,
                    "tol": [0.10 for _ in r_gr],
                    "passed": True,
                    "worst_index": 1,
                    "worst_r": float(r_gr[1]),
                },
            },
            "groups": {
                "short": {"passed": True, "items": ["bondlen_A-B"]},
                "medium": {"passed": True, "items": ["ring", "ring_mean_size"]},
                "long": {"passed": True, "items": ["density", "gr_all"]},
            },
            "stability": {
                "enabled": False,
            },
            "ci_converged": True,
            "stability_converged": True,
            "converged": True,
            "passed": True,
        },
        "crystal_motifs": {},
        "metrics_checked": [
            "density",
            "ring_mean_size",
            "ring_frac_3",
            "bondlen_A-B",
            "gr_all",
        ],
        "effective_metrics": {},
        "metric_warnings": [],
        "analysis_source_roles": {"final_structure": len(boxes)},
        "boxes": boxes,
        "rejected_boxes": [],
        "paths": {
            "output_dataset": "output_dataset.json",
            "analysis_results": "analysis_results.json",
            "condensed_log": "condensed.log",
        },
    }

    path.write_text(json.dumps(results))
    return path


def test_plot_production_accepts_analysis_results_json(tmp_path: Path) -> None:
    analysis_json = _make_analysis_results(tmp_path / "analysis_results.json", offset=0.0)
    out_pdf = tmp_path / "production_single.pdf"

    plot_production_results(analysis_json, out_pdf, dpi=80)

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0


def test_plot_production_accepts_native_zero_based_box_identifier(
    tmp_path: Path,
) -> None:
    """Custom/run schedules serialize their first production box as zero."""

    analysis_json = _make_analysis_results(tmp_path / "analysis_results.json", offset=0.0)
    payload = json.loads(analysis_json.read_text())
    payload["boxes"][0]["box"] = 0
    analysis_json.write_text(json.dumps(payload))
    out_pdf = tmp_path / "production_zero_based.pdf"

    plot_production_results(analysis_json, out_pdf, dpi=80)

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0


@pytest.mark.parametrize("invalid", [True, -1, 0.5, float("nan"), float("inf")])
def test_plot_production_rejects_invalid_box_identifiers(
    tmp_path: Path,
    invalid,
) -> None:
    analysis_json = _make_analysis_results(tmp_path / "analysis_results.json", offset=0.0)
    payload = json.loads(analysis_json.read_text())
    payload["boxes"][0]["box"] = invalid
    analysis_json.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="box identifier"):
        plot_production_results(
            analysis_json,
            tmp_path / "invalid.pdf",
            dpi=80,
        )


def test_plot_production_rejects_duplicate_box_identifiers(tmp_path: Path) -> None:
    analysis_json = _make_analysis_results(tmp_path / "analysis_results.json", offset=0.0)
    payload = json.loads(analysis_json.read_text())
    payload["boxes"][1]["box"] = payload["boxes"][0]["box"]
    analysis_json.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="duplicate box identifier"):
        plot_production_results(
            analysis_json,
            tmp_path / "duplicate.pdf",
            dpi=80,
        )


def test_plot_production_preserves_distinct_integer_ids_above_float_precision(
    tmp_path: Path,
) -> None:
    analysis_json = _make_analysis_results(tmp_path / "analysis_results.json", offset=0.0)
    payload = json.loads(analysis_json.read_text())
    payload["boxes"][0]["box"] = 2**53
    payload["boxes"][1]["box"] = 2**53 + 1
    analysis_json.write_text(json.dumps(payload))
    output = tmp_path / "large_ids.pdf"

    plot_production_results(analysis_json, output, dpi=80)

    assert output.is_file() and output.stat().st_size > 0


def test_plot_production_emits_dft_pages_for_native_box_identifiers(
    tmp_path: Path,
) -> None:
    """Autotune stores ``box``; plotting must not require synthetic ``box_id``."""

    source = _make_analysis_results(tmp_path / "analysis_results.json", offset=0.0)
    analysis = json.loads(source.read_text())
    boxes = analysis["boxes"]
    for entry in boxes:
        assert "box" in entry and "box_id" not in entry
        entry["dft_opt"] = {
            "status": "ok",
            "density": float(entry["density"]) + 0.05,
            "metrics": dict(entry["metrics"]),
            "distributions": dict(entry["distributions"]),
        }
    production = dict(analysis)
    production.update(
        {
            "enabled": True,
            "boxes": boxes,
            "boxes_dft_final": [1, 2, 3, 4],
            "convergence_md": dict(analysis["convergence"]),
            "convergence_dft": dict(analysis["convergence"]),
        }
    )
    result_path = tmp_path / "autotune_results.json"
    result_path.write_text(json.dumps({"production": production}))
    out_dir = tmp_path / "plots"

    plot_production_results(result_path, out_dir, dpi=80)

    generated_names = {path.name for path in out_dir.glob("*.png")}
    assert any("MD_vs_DFT:_density" in name for name in generated_names)
    assert any("MD_vs_DFT:_bondlen" in name for name in generated_names)
    assert any("MD_vs_DFT:_g(r)" in name for name in generated_names)


def test_plot_production_compare_multiple_analysis_results(tmp_path: Path) -> None:
    md = _make_analysis_results(tmp_path / "MD_analysis.json", offset=0.0)
    pbe = _make_analysis_results(tmp_path / "PBE_analysis.json", offset=0.2)
    hse = _make_analysis_results(tmp_path / "HSE_analysis.json", offset=0.4)
    out_pdf = tmp_path / "production_compare.pdf"

    plot_production_comparison_results(
        [md, pbe, hse],
        out_pdf,
        labels=["MD", "PBE", "HSE06"],
        dpi=80,
        max_pages=3,
    )

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0


def test_plot_production_analysis_results_without_convergence_familywise(tmp_path: Path) -> None:
    analysis_json = _make_analysis_results(tmp_path / "analysis_results_no_conv.json", offset=0.0)
    payload = json.loads(analysis_json.read_text())
    payload["schema"] = "vitriflow.analysis_results.v2"
    payload["converged"] = False
    payload["convergence"] = {
        "schema": "vitriflow.analysis_descriptor_convergence.v1",
        "advisory": True,
        "status": "not_evaluated",
        "reason": "legacy analysis output without production familywise convergence report",
        "groups": {
            "short": {"status": "not_evaluated", "passed": None, "items": ["bondlen_cdf:bondlen_A-B"]},
            "medium": {"status": "not_evaluated", "passed": None, "items": ["ring_mean_size", "ring_frac_3"]},
            "long": {"status": "not_evaluated", "passed": None, "items": ["density", "gr_curve:gr_all"]},
        },
    }
    analysis_json.write_text(json.dumps(payload))
    out_dir = tmp_path / "plots"

    plot_production_results(analysis_json, out_dir, dpi=80, max_pages=4)

    assert out_dir.exists()
    assert any(p.suffix == ".png" and p.stat().st_size > 0 for p in out_dir.iterdir())
