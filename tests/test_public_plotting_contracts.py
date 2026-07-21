from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import vitriflow.plotting as plotting


def test_rate_scan_current_schema_legacy_alias_and_dimensional_check() -> None:
    x, label = plotting._rate_scan_coordinates(
        [
            {"rate": 0.5, "rate_K_per_ps": 500.0},
            {"rate": 1.0, "rate_K_per_ps": 1000.0},
        ],
        time_unit_ps=0.001,
    )
    assert np.allclose(x, [500.0, 1000.0])
    assert label == "cooling rate (K/ps)"

    x_legacy, legacy_label = plotting._rate_scan_coordinates(
        [{"rate_K_per_time": 0.5}, {"rate_K_per_time": 1.0}],
        time_unit_ps=None,
    )
    assert np.allclose(x_legacy, [0.5, 1.0])
    assert legacy_label == "cooling rate (K / time unit)"

    with pytest.raises(ValueError, match="inconsistent native and K/ps rates"):
        plotting._rate_scan_coordinates(
            [{"rate": 0.5, "rate_K_per_ps": 5.0}],
            time_unit_ps=0.001,
        )


def test_convergence_display_preserves_unassessed_and_inference_roles() -> None:
    unassessed = plotting._production_convergence_display(
        {"converged": None, "check_convergence": True},
        {"status": "not_evaluated"},
    )
    assert unassessed["state"] == "unassessed"
    assert "not converged" not in unassessed["label"]

    repeated = plotting._production_convergence_display(
        {
            "converged": True,
            "check_convergence": True,
            "convergence_inference_status": (
                "criterion_met_repeated_looks_not_sequentially_valid"
            ),
        },
        {
            "status": "converged",
            "inference_contract": {"sequentially_valid": False},
            "familywise": {},
            "scalars": {"density": {}},
        },
    )
    assert repeated["state"] == "criterion_met"
    assert "not sequentially valid" in repeated["label"]

    posthoc = plotting._production_convergence_display(
        {"converged": None, "check_convergence": False},
        {
            "status": "fixed_n_terminal_posthoc_assessed",
            "assessment_role": "terminal_posthoc_diagnostic",
            "posthoc_criterion_met": True,
        },
    )
    assert posthoc["state"] == "posthoc_criterion_met"
    assert "not a stopping result" in posthoc["label"]


def test_familywise_annotation_labels_error_probability_not_confidence() -> None:
    label = plotting._familywise_error_annotation(
        {"alpha_family": 0.05, "m_tests": 12},
        alpha_per_test=0.05 / 12.0,
        bounded_ci_method="t",
    )
    assert "FWER alpha=0.05" in label
    assert "family confidence=0.950" in label
    assert "FWER=0.950" not in label


def _full_production_payload() -> dict:
    bond_x = [1.0, 1.5, 2.0, 2.5]
    angle_x = [0.0, 90.0, 180.0]
    coord_x = [0.0, 1.0, 2.0]
    void_x = [0.0, 0.5, 1.0, 1.5]
    r = [0.5, 1.0, 1.5, 2.0]
    q = [0.0, 1.0, 2.0]
    boxes = []
    for idx in range(4):
        delta = 0.01 * float(idx)
        boxes.append(
            {
                "box": idx + 1,
                "box_id": idx + 1,
                "density": 2.2 + delta,
                "metrics": {
                    "ring_frac_3": 0.2 + delta,
                    "ring_mean_size": 5.5 + delta,
                    "amorphous_order": 0.4 + delta,
                },
                "coordination_defect_details": {
                    "coord_A-B": {
                        "coordination_sweep": {
                            "delta_r": [-0.1, 0.0, 0.1],
                            "defect_fraction": [0.12 + delta, 0.08 + delta, 0.05 + delta],
                        }
                    }
                },
                "distributions": {
                    "bondlen": {
                        "bondlen_A-B": {
                            "x": bond_x,
                            "cdf": [0.0, 0.2 + delta, 0.8 + delta, 1.0],
                        }
                    },
                    "angle": {
                        "angle_A-B-A": {
                            "x": angle_x,
                            "cdf": [0.0, 0.45 + delta, 1.0],
                        }
                    },
                    "coord": {
                        "coord_A-B": {
                            "x": coord_x,
                            "cdf": [0.1 + delta, 0.8 + delta, 1.0],
                        }
                    },
                    "void": {
                        "void_all": {
                            "x": void_x,
                            "cdf": [0.0, 0.3 + delta, 0.9 + delta, 1.0],
                        }
                    },
                    "gr": {
                        "gr_all": {
                            "label": "all",
                            "r": r,
                            "g": [0.1, 1.2 + delta, 0.6 + delta, 0.2],
                        }
                    },
                    "sq": {
                        "sq_all": {
                            "q": q,
                            "s": [1.0, 1.3 + delta, 0.9 + delta],
                        }
                    },
                },
            }
        )

    def mean_rows(family: str, name: str, key: str) -> list[float]:
        return np.mean(
            np.asarray(
                [box["distributions"][family][name][key] for box in boxes],
                dtype=float,
            ),
            axis=0,
        ).tolist()

    zeros4 = [0.0] * 4
    zeros3 = [0.0] * 3
    convergence = {
        "status": "converged",
        "mode": "ci",
        "familywise": {
            "alpha_family": 0.05,
            "m_tests": 9,
            "alpha_per_test": 0.05 / 9.0,
            "bounded_ci_method": "t",
        },
        "inference_contract": {"sequentially_valid": False},
        "scalars": {
            "density": {
                "group": "long",
                "mean": float(np.mean([box["density"] for box in boxes])),
                "ci_halfwidth": 0.01,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
            "ring_mean_size": {
                "group": "medium",
                "mean": 5.515,
                "ci_halfwidth": 0.01,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
        },
        "distributions": {
            "ring": {
                "group": "medium",
                "kind": "pmf",
                "mean": [0.215],
                "ci_halfwidth": [0.01],
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
            "bondlen_A-B": {
                "group": "short",
                "kind": "bondlen_cdf",
                "x": bond_x,
                "mean": mean_rows("bondlen", "bondlen_A-B", "cdf"),
                "ci_halfwidth": zeros4,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
            "angle_A-B-A": {
                "group": "short",
                "kind": "angle_cdf",
                "x": angle_x,
                "mean": mean_rows("angle", "angle_A-B-A", "cdf"),
                "ci_halfwidth": zeros3,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
            "coord_A-B": {
                "group": "short",
                "kind": "coord_cdf",
                "x": coord_x,
                "mean": mean_rows("coord", "coord_A-B", "cdf"),
                "ci_halfwidth": zeros3,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
            "void_all": {
                "group": "medium",
                "kind": "void_cdf",
                "x": void_x,
                "mean": mean_rows("void", "void_all", "cdf"),
                "ci_halfwidth": zeros4,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
            "gr_all": {
                "group": "long",
                "kind": "gr_curve",
                "r": r,
                "mean": mean_rows("gr", "gr_all", "g"),
                "ci_halfwidth": zeros4,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
            "sq_all": {
                "group": "long",
                "kind": "sq_curve",
                "q": q,
                "mean": mean_rows("sq", "sq_all", "s"),
                "ci_halfwidth": zeros3,
                "abs_tol": 0.1,
                "rel_tol": 0.0,
            },
        },
        "groups": {
            "short": {"passed": True},
            "medium": {"passed": True},
            "long": {"passed": True},
        },
    }
    return {
        "status": "ok",
        "units": {
            "reporting_contract": "vitriflow.canonical_physical_units.v1",
            "lammps_units": "metal",
        },
        "production": {
            "enabled": True,
            "status": "ok",
            "converged": True,
            "check_convergence": True,
            "convergence_inference_status": (
                "criterion_met_repeated_looks_not_sequentially_valid"
            ),
            "n_boxes": 4,
            "n_boxes_accepted": 4,
            "n_boxes_rejected": 0,
            "n_boxes_total": 4,
            "boxes": boxes,
            "convergence_spec": {
                "bondlen_names": ["bondlen_A-B"],
                "angle_names": ["angle_A-B-A"],
                "coord_names": ["coord_A-B"],
                "ring_keys": ["ring_frac_3"],
                "ring_has_mean_size": True,
                "gr_labels": ["gr_all"],
                "sq_labels": ["sq_all"],
                "void_names": ["void_all"],
            },
            "convergence": convergence,
        },
    }


def test_production_plot_emits_every_public_metric_family_and_honest_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "autotune_results.json"
    src.write_text(json.dumps(_full_production_payload()))
    captured: dict[str, object] = {}

    def capture(pages, *_args, **_kwargs):
        captured["names"] = [name for name, _fig in pages]
        convergence_fig = pages[0][1]
        captured["suptitle"] = convergence_fig._suptitle.get_text()
        captured["axis_text"] = " ".join(
            text.get_text() for ax in convergence_fig.axes for text in ax.texts
        )
        import matplotlib.pyplot as plt

        for _name, fig in pages:
            plt.close(fig)

    monkeypatch.setattr(plotting, "_save_plot_pages", capture)
    plotting.plot_production_results(src, tmp_path / "plots", dpi=80)

    names = set(captured["names"])
    assert {
        "convergence",
        "density",
        "rings",
        "ring_mean",
        "scalar_amorphous_order",
        "coordination_sweep_coord_A-B",
        "bondlen_A-B",
        "angle_A-B-A",
        "coord_A-B",
        "void_all",
        "gr_all",
        "sq_all",
    }.issubset(names)
    assert "not sequentially valid" in str(captured["suptitle"])
    assert "FWER alpha=0.05" in str(captured["axis_text"])
    assert "family confidence=0.950" in str(captured["axis_text"])


def test_comparison_accounts_for_nonfinite_and_asymmetric_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _full_production_payload()
    comparison = _full_production_payload()
    for box in reference["production"]["boxes"]:
        box["metrics"].update(
            {
                "amorphous_reference_peak_overlap": None,
                "amorphous_reference_peak_overlap_advisory": None,
                "reference_only_metric": 1.0,
            }
        )
    for box in comparison["production"]["boxes"]:
        box["metrics"].update(
            {
                "amorphous_reference_peak_overlap": None,
                "amorphous_reference_peak_overlap_advisory": None,
            }
        )
        box["distributions"]["sq"].pop("sq_all")
    comparison["production"]["convergence_spec"]["sq_labels"] = []
    comparison["production"]["convergence"]["distributions"].pop(
        "sq_all", None
    )

    reference_path = tmp_path / "reference.json"
    comparison_path = tmp_path / "comparison.json"
    reference_path.write_text(json.dumps(reference))
    comparison_path.write_text(json.dumps(comparison))
    captures: list[dict[str, object]] = []

    def capture(pages, *_args, **_kwargs):
        names = [name for name, _fig in pages]
        legends = {
            name: fig.axes[0].get_legend_handles_labels()[1]
            for name, fig in pages
            if fig.axes
        }
        texts = {
            name: [text.get_text() for ax in fig.axes for text in ax.texts]
            for name, fig in pages
        }
        captures.append({"names": names, "legends": legends, "texts": texts})
        import matplotlib.pyplot as plt

        for _name, fig in pages:
            plt.close(fig)

    monkeypatch.setattr(plotting, "_save_plot_pages", capture)
    for paths in (
        [reference_path, comparison_path],
        [comparison_path, reference_path],
    ):
        plotting.plot_production_comparison_results(
            paths,
            tmp_path / "plots",
            labels=["reference", "comparison"],
            dpi=80,
        )

    expected_names = [
        "convergence_comparison",
        "density",
        "rings",
        "ring_mean",
        "scalar_amorphous_order",
        "scalar_amorphous_reference_peak_overlap",
        "scalar_amorphous_reference_peak_overlap_advisory",
        "scalar_reference_only_metric",
        "coordination_sweep_coord_A-B",
        "bondlen_A-B",
        "angle_A-B-A",
        "coord_A-B",
        "void_all",
        "gr_all",
        "sq_all",
    ]
    assert [capture["names"] for capture in captures] == [
        expected_names,
        expected_names,
    ]

    first = captures[0]
    unavailable_legends = first["legends"][
        "scalar_amorphous_reference_peak_overlap"
    ]
    assert unavailable_legends == [
        "reference: unavailable (no finite values)",
        "comparison: unavailable (no finite values)",
    ]
    assert first["texts"]["scalar_amorphous_reference_peak_overlap"] == [
        "No dataset contains finite per-box or summary values"
    ]
    assert "comparison: unavailable (no finite values)" in first["legends"][
        "scalar_reference_only_metric"
    ]
    assert "comparison: unavailable (not declared/emitted)" in first["legends"][
        "sq_all"
    ]


def test_comparison_rejects_declared_distribution_without_payload(
    tmp_path: Path,
) -> None:
    reference = _full_production_payload()
    malformed = _full_production_payload()
    for box in malformed["production"]["boxes"]:
        box["distributions"]["sq"].pop("sq_all")
    malformed["production"]["convergence"]["distributions"].pop("sq_all", None)

    reference_path = tmp_path / "reference.json"
    malformed_path = tmp_path / "malformed.json"
    reference_path.write_text(json.dumps(reference))
    malformed_path.write_text(json.dumps(malformed))

    with pytest.raises(ValueError, match="declared.*no usable finite payload"):
        plotting.plot_production_comparison_results(
            [reference_path, malformed_path],
            tmp_path / "plots",
            dpi=80,
        )


def test_production_payload_preserves_nullable_convergence(tmp_path: Path) -> None:
    payload = _full_production_payload()
    production = payload.pop("production")
    payload.update(
        {
            "schema": "vitriflow.analysis_results.v2",
            "boxes": production["boxes"],
            "convergence": production["convergence"],
            "convergence_spec": production["convergence_spec"],
            "converged": None,
            "check_convergence": True,
            "n_boxes": 4,
        }
    )
    src = tmp_path / "analysis_results.json"
    src.write_text(json.dumps(payload))

    _data, normalised = plotting._prepare_production_plot_payload(src)

    assert normalised["converged"] is None


def test_public_plot_max_pages_is_consistently_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_pages must be a positive integer"):
        plotting.plot_production_results(
            tmp_path / "missing.json",
            tmp_path / "plots",
            max_pages=0,
        )
    with pytest.raises(ValueError, match="max_pages must be a positive integer"):
        plotting.plot_production_comparison_results(
            [tmp_path / "a.json", tmp_path / "b.json"],
            tmp_path / "plots",
            max_pages=0,
        )


def test_stage_results_metadata_is_fail_closed_and_supports_run_schema(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "production" / "box_001" / "relax"
    stage.mkdir(parents=True)
    results = tmp_path / "run_results.json"
    results.write_text(
        json.dumps(
            {
                "parameters": {
                    "engine": "lammps",
                    "lammps_units": "metal",
                    "time_unit_ps": 1.0,
                    "md": {"timestep": 0.0005},
                }
            }
        )
    )

    metadata = plotting._stage_plot_metadata(stage, results_json=results)

    assert metadata["dt"] == pytest.approx(0.0005)
    assert metadata["units_style"] == "metal"
    assert metadata["time_unit_ps"] == pytest.approx(1.0)

    results.write_text("not-json")
    with pytest.raises(ValueError, match="Invalid results JSON"):
        plotting._stage_plot_metadata(stage, results_json=results)


def test_results_unit_metadata_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting LAMMPS unit metadata"):
        plotting._units_from_results(
            {
                "units": {"lammps_units": "metal", "time_unit_ps": 1.0},
                "parameters": {"lammps_units": "real", "time_unit_ps": 1.0},
            }
        )
    with pytest.raises(ValueError, match="conflicting time_unit_ps"):
        plotting._units_from_results(
            {
                "units": {"lammps_units": "metal", "time_unit_ps": 1.0},
                "parameters": {"lammps_units": "metal", "time_unit_ps": 0.001},
            }
        )


def test_sampled_curve_alignment_is_complete_and_representation_safe() -> None:
    x, matrix, metadata = plotting._align_sampled_plot_payloads(
        [
            {"r": [0.5, 1.0, 1.5], "g": [0.1, 1.0, 0.2]},
            {"r": [0.5, 0.75, 1.5], "g": [0.2, 0.8, 0.3]},
        ],
        family="gr",
        xkey="r",
        ykey="g",
    )
    assert x[0] == pytest.approx(0.5)
    assert x[-1] == pytest.approx(1.5)
    assert matrix.shape == (2, x.size)
    assert metadata["grid_alignment_method"] == (
        "linear_interpolation_without_extrapolation"
    )

    with pytest.raises(ValueError, match=r"mixed S\(q\) representation"):
        plotting._align_sampled_plot_payloads(
            [
                {
                    "q": [0.0, 1.0],
                    "s": [1.0, 1.1],
                    "representation": {"schema": "vitriflow.sq_representation.v1"},
                },
                {"q": [0.0, 1.0], "s": [1.0, 1.2]},
            ],
            family="sq",
            xkey="q",
            ykey="s",
        )


def test_declared_production_distribution_cannot_be_silently_omitted(
    tmp_path: Path,
) -> None:
    payload = _full_production_payload()
    del payload["production"]["boxes"][1]["distributions"]["bondlen"][
        "bondlen_A-B"
    ]
    src = tmp_path / "autotune_results.json"
    src.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="Malformed stored bondlen CDF"):
        plotting.plot_production_results(src, tmp_path / "plots", dpi=80)


def test_direct_public_plot_counts_reject_nonpositive_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        plotting.plot_voids_map(
            tmp_path,
            tmp_path / "voids.png",
            n_samples=0,
        )
    with pytest.raises(ValueError, match="dpi must be a positive integer"):
        plotting.plot_elastic_screen(
            tmp_path,
            tmp_path / "elastic.png",
            dpi=0,
        )
