from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("networkx")

from vitriflow.analysis.dump import DumpFrame
from vitriflow.analysis.graph import GraphRule, build_hard_graph
from vitriflow.analysis.structure import _compute_ring_metrics, compute_structure_metrics
from vitriflow.config import RingMetricsConfig, StructureMetricsConfig


def _frame(types, positions, *, cell=20.0):
    types = np.asarray(types, dtype=int)
    return DumpFrame(
        timestep=0,
        ids=np.arange(1, len(types) + 1, dtype=int),
        types=types,
        positions=np.asarray(positions, dtype=float),
        cell=np.eye(3, dtype=float) * float(cell),
        origin=np.zeros(3, dtype=float),
    )


def test_single_multicoordinate_bridge_does_not_create_projected_triangle():
    frame = _frame(
        [1, 1, 1, 2],
        [[2, 2, 2], [4, 2, 2], [3, 4, 2], [3, 3, 2]],
    )
    neighbors = [[3], [3], [3], [0, 1, 2]]
    ring = RingMetricsConfig(
        enabled=True, mode="projected", nodes=[1], bridge=2, max_cycle_size=6
    )
    values = _compute_ring_metrics(frame, ring, neighbors, None)
    assert values["ring_count"] == 0.0
    assert values["ring_frac_3"] == 0.0


def test_distinct_bridges_preserve_three_and_two_member_alternating_rings():
    ring = RingMetricsConfig(
        enabled=True, mode="projected", nodes=[1], bridge=2, max_cycle_size=6
    )
    frame3 = _frame([1, 1, 1, 2, 2, 2], np.zeros((6, 3)))
    # n0-b0-n1-b1-n2-b2-n0
    nbr3 = [[3, 5], [3, 4], [4, 5], [0, 1], [1, 2], [2, 0]]
    values3 = _compute_ring_metrics(frame3, ring, nbr3, None)
    assert values3["ring_count"] == 1.0
    assert values3["ring_frac_3"] == 1.0

    frame2 = _frame([1, 1, 2, 2], np.zeros((4, 3)))
    nbr2 = [[2, 3], [2, 3], [0, 1], [0, 1]]
    values2 = _compute_ring_metrics(frame2, ring, nbr2, None)
    assert values2["ring_count"] == 1.0
    assert values2["ring_frac_2"] == 1.0


def test_species_selector_pair_expansion_does_not_duplicate_canonical_edge():
    frame = _frame([1, 2], [[1, 1, 1], [2, 1, 1]])
    graph = build_hard_graph(frame, GraphRule("r", "hard_cutoff", {"cutoff": 1.1}))
    metrics = StructureMetricsConfig.model_validate(
        {"enabled": True, "pairs": [{"pair": ["Si", "Si"]}]}
    )
    metrics.voids.enabled = False
    values = compute_structure_metrics(
        frame,
        metrics,
        cutoffs={},
        type_to_species=["Si", "Si"],
        graph=graph,
    ).values
    assert values["bond_incidence_Si-Si_count"] == 1.0
    assert values["bondlen_Si-Si_mean"] == pytest.approx(1.0)


def test_cutoff_beyond_cell_length_counts_each_atom_pair_once():
    from vitriflow.analysis.graph import directed_neighbor_lists
    from vitriflow.analysis.structure import compute_structure_distributions_timeavg

    frame = _frame(
        [1, 1],
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        cell=4.0,
    )
    graph = build_hard_graph(
        frame, GraphRule("wide", "hard_cutoff", {"cutoff": 4.1})
    )

    assert graph.edges == [(0, 1)]
    nbr_ids, _vecs, _dists, _weights = directed_neighbor_lists(graph, 2)
    assert nbr_ids == [[1], [0]]

    metrics = StructureMetricsConfig.model_validate(
        {
            "enabled": True,
            "pairs": [{"pair": [1, 1], "cutoff": 4.1}],
            "coordinations": [{"central": 1, "neighbor": 1, "cutoff": 4.1}],
        }
    )
    metrics.voids.enabled = False
    distributions = compute_structure_distributions_timeavg(
        [frame], metrics, cutoffs={(1, 1): 4.1}
    )
    assert distributions["bondlen"]["bondlen_1-1"]["sample_count"] == 1
    assert distributions["coord"]["coord_1-1"]["sample_count"] == 2


def test_semantically_equal_angle_endpoint_selectors_count_once():
    frame = _frame(
        [2, 1, 1],
        [[5, 5, 5], [6, 5, 5], [5, 6, 5]],
    )
    graph = build_hard_graph(frame, GraphRule("r", "hard_cutoff", {"cutoff": 1.1}))
    metrics = StructureMetricsConfig.model_validate(
        {"enabled": True, "angles": [{"triplet": [1, 2, "Si"]}]}
    )
    metrics.voids.enabled = False
    values = compute_structure_metrics(
        frame,
        metrics,
        cutoffs={},
        type_to_species=["Si", "X"],
        graph=graph,
    ).values
    assert values["angle_1-2-Si_mean"] == pytest.approx(90.0)


def test_defect_separation_uses_periodic_minimum_image():
    from vitriflow.analysis.graph_metrics import _defect_graph_values

    frame = _frame([1, 1], [[0.1, 1.0, 1.0], [19.9, 1.0, 1.0]])
    graph = build_hard_graph(
        frame, GraphRule("none", "hard_cutoff", {"cutoff": 0.05})
    )
    metrics = StructureMetricsConfig.model_validate(
        {
            "enabled": True,
            "coordinations": [
                {"central": 1, "neighbor": 1, "expected": 1, "defect_frac_tol": 0.0}
            ],
        }
    )
    metrics.voids.enabled = False
    values = _defect_graph_values(frame, metrics, graph, type_to_species=None)
    assert values["defect_coord_1-1_defect_distance_min"] == pytest.approx(0.2)


def test_oversized_rdf_smoothing_window_is_bounded_by_grid() -> None:
    from vitriflow.analysis.gr import first_peak_features

    r = np.linspace(0.1, 4.0, 20)
    g = 1.0 + np.exp(-((r - 1.5) / 0.2) ** 2)
    features = first_peak_features(r, g, smooth=999)
    assert len(features) == 3
    assert all(np.isfinite(value) or np.isnan(value) for value in features)


def test_oversized_sq_smoothing_window_is_bounded_by_grid() -> None:
    from vitriflow.analysis.sq import first_peak_features

    q = np.linspace(0.0, 5.0, 20)
    sq = 1.0 + np.exp(-((q - 1.5) / 0.2) ** 2)
    features = first_peak_features(q, sq, smooth=999, q_min=0.1, q_max=4.0)
    assert len(features) == 3
    assert all(np.isfinite(value) or np.isnan(value) for value in features)


@pytest.mark.parametrize(
    ("field", "value"),
    [("nq", 10.5), ("nq", True), ("nbins", 50.5), ("nbins", True)],
)
def test_sq_grid_counts_require_exact_integers(field: str, value: object) -> None:
    from vitriflow.analysis.sq import compute_first_peak_sq, compute_sq

    frame = _frame([1, 1], [[1, 1, 1], [2, 1, 1]], cell=10.0)
    kwargs = {"q_max": 5.0, "nq": 10, "r_max": 4.0, "nbins": 50}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        compute_sq([frame], **kwargs)
    with pytest.raises(ValueError, match=field):
        compute_first_peak_sq([frame], **kwargs)


def test_multiframe_sq_averages_per_frame_density_transforms() -> None:
    from vitriflow.analysis.sq import compute_sq

    frac = np.asarray(
        [
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.8],
            [0.2, 0.8, 0.2],
            [0.2, 0.8, 0.8],
            [0.8, 0.2, 0.2],
            [0.8, 0.2, 0.8],
            [0.8, 0.8, 0.2],
            [0.8, 0.8, 0.8],
        ]
    )
    frames = [
        _frame(np.ones(8, dtype=int), frac * length, cell=length)
        for length in (10.0, 12.0)
    ]
    kwargs = {"q_max": 5.0, "nq": 40, "r_max": 4.0, "nbins": 60}
    q, ensemble = compute_sq(frames, **kwargs)
    q0, first = compute_sq([frames[0]], **kwargs)
    q1, second = compute_sq([frames[1]], **kwargs)

    np.testing.assert_allclose(q, q0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(q, q1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ensemble, 0.5 * (first + second), rtol=2e-14, atol=2e-14)


def test_sq_normalization_formulas_and_self_terms(monkeypatch) -> None:
    import vitriflow.analysis.sq as sq_module

    r = (np.arange(50, dtype=float) + 0.5) * 0.1
    g = np.full_like(r, 1.25)

    def _fake_gr(*args, **kwargs):
        return r.copy(), g.copy(), 1.0

    monkeypatch.setattr(sq_module, "compute_gr", _fake_gr)
    frame = _frame(
        [1, 1, 2, 2],
        [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]],
        cell=10.0,
    )
    kwargs = {
        "q_max": 2.0,
        "nq": 10,
        "r_max": 5.0,
        "nbins": 50,
        "window": "none",
    }
    integral_q0 = float(np.sum((r**2) * (g - 1.0) * 0.1))

    _q, total = sq_module.compute_sq([frame], **kwargs)
    _q, same = sq_module.compute_sq([frame], pair=(1, 1), **kwargs)
    _q, cross = sq_module.compute_sq([frame], pair=(1, 2), **kwargs)

    assert total[0] == pytest.approx(1.0 + 4.0 * np.pi * 0.004 * integral_q0)
    assert same[0] == pytest.approx(1.0 + 4.0 * np.pi * 0.002 * integral_q0)
    assert cross[0] == pytest.approx(4.0 * np.pi * 0.002 * integral_q0)


def test_sq_metadata_discloses_physical_representation_and_clipping() -> None:
    from vitriflow.analysis.sq import compute_sq

    frame = _frame(
        [1, 1, 2, 2],
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5], [3.5, 0.5, 0.5]],
        cell=6.0,
    )
    default_result = compute_sq(
        [frame], q_max=5.0, nq=20, r_max=4.0, nbins=50, window="lorch"
    )
    assert len(default_result) == 2

    _q, _s, total_meta = compute_sq(
        [frame],
        q_max=5.0,
        nq=20,
        r_max=4.0,
        nbins=50,
        window="lorch",
        return_metadata=True,
    )
    assert total_meta["schema"] == "vitriflow.sq_representation.v1"
    assert total_meta["normalization"] == "unweighted_number_number_total"
    assert total_meta["scattering_weighted"] is False
    assert total_meta["scattering_weights"] == "none"
    assert total_meta["q_unit"] == "angstrom^-1"
    assert total_meta["r_unit"] == "angstrom"
    assert total_meta["termination_window"] == "lorch"
    assert total_meta["termination_window_definition"] == "sinc(pi*r/r_support_effective)"
    assert total_meta["rdf_normalization"] == "finite_population_unordered_pair_shell_volume"
    assert total_meta["r_support_requested_A"] == 4.0
    assert total_meta["r_support_effective_A"] < 3.0
    assert total_meta["r_support_clipped_to_unique_image_radius"] is True
    assert total_meta["frame_aggregation"] == "equal_frame_mean_after_per_frame_density_transform"
    assert total_meta["n_frames_requested"] == total_meta["n_frames_used"] == 1
    assert "not_thermodynamic_compressibility" in total_meta["q_zero_semantics"]

    _q, _s, partial_meta = compute_sq(
        [frame],
        q_max=5.0,
        nq=20,
        r_max=2.0,
        nbins=50,
        pair=(1, 2),
        window="hann",
        return_metadata=True,
    )
    assert partial_meta["normalization"] == "ashcroft_langreth_partial"
    assert partial_meta["partial_kind"] == "cross"
    assert partial_meta["self_term"] == 0.0
    assert partial_meta["resolved_type_sets"] == [[1], [2]]


def test_sq_rejects_an_undefined_frame_instead_of_silently_dropping_it() -> None:
    from vitriflow.analysis.sq import compute_sq

    defined = _frame([1, 1], [[1, 1, 1], [2, 1, 1]], cell=10.0)
    undefined = _frame([1, 2], [[1, 1, 1], [2, 1, 1]], cell=10.0)
    with pytest.raises(ValueError, match=r"frame 1.*fewer than two self pairs"):
        compute_sq(
            [defined, undefined],
            q_max=5.0,
            nq=20,
            r_max=4.0,
            nbins=50,
            pair=(1, 1),
        )


def test_multiframe_sq_uses_one_common_clipped_radial_support() -> None:
    from vitriflow.analysis.sq import compute_sq

    frac = np.asarray(
        [
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.8],
            [0.2, 0.8, 0.2],
            [0.2, 0.8, 0.8],
            [0.8, 0.2, 0.2],
            [0.8, 0.2, 0.8],
            [0.8, 0.8, 0.2],
            [0.8, 0.8, 0.8],
        ]
    )
    frames = [
        _frame(np.ones(8, dtype=int), frac * length, cell=length)
        for length in (6.0, 10.0)
    ]
    q, ensemble, meta = compute_sq(
        frames,
        q_max=5.0,
        nq=40,
        r_max=4.0,
        nbins=60,
        return_metadata=True,
    )
    common_r_max = float(meta["r_support_effective_A"])
    q0, first = compute_sq(
        [frames[0]], q_max=5.0, nq=40, r_max=common_r_max, nbins=60
    )
    q1, second = compute_sq(
        [frames[1]], q_max=5.0, nq=40, r_max=common_r_max, nbins=60
    )
    np.testing.assert_allclose(q, q0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(q, q1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ensemble, 0.5 * (first + second), rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("window", [None, False, "off", "hanning"])
def test_sq_rejects_undocumented_or_non_string_windows(window: object) -> None:
    from vitriflow.analysis.sq import compute_sq

    frame = _frame([1, 1], [[1, 1, 1], [2, 1, 1]], cell=10.0)
    with pytest.raises(ValueError, match="window"):
        compute_sq(
            [frame], q_max=5.0, nq=20, r_max=4.0, nbins=50, window=window
        )


@pytest.mark.parametrize("smooth", [True, 0, 1.5, float("nan")])
def test_sq_peak_smoothing_width_is_an_exact_positive_integer(smooth: object) -> None:
    from vitriflow.analysis.sq import first_peak_features

    q = np.linspace(0.0, 5.0, 20)
    s = 1.0 + np.exp(-((q - 1.5) / 0.2) ** 2)
    with pytest.raises(ValueError, match="smooth"):
        first_peak_features(q, s, smooth=smooth)


def test_sq_and_amorphous_peak_search_must_lie_within_q_max() -> None:
    from vitriflow.config import AmorphousMetricsConfig, SqMetricConfig

    with pytest.raises(ValueError, match="peak_search"):
        SqMetricConfig(q_max=2.0, peak_search=(3.0, 4.0))
    with pytest.raises(ValueError, match="peak_search"):
        AmorphousMetricsConfig(q_max=2.0, peak_search=(3.0, 4.0))


@pytest.mark.parametrize("family", ["gr", "sq"])
def test_peak_features_do_not_fabricate_peak_from_missing_or_flat_curve(family: str) -> None:
    grid = np.linspace(0.1, 4.0, 40)
    if family == "gr":
        from vitriflow.analysis.gr import first_peak_features

        missing = first_peak_features(grid, np.full_like(grid, np.nan), smooth=7)
        flat = first_peak_features(grid, np.ones_like(grid), smooth=7)
    else:
        from vitriflow.analysis.sq import first_peak_features

        missing = first_peak_features(grid, np.full_like(grid, np.nan), smooth=7)
        flat = first_peak_features(grid, np.ones_like(grid), smooth=7)
    assert all(np.isnan(value) for value in missing)
    assert all(np.isnan(value) for value in flat)


def test_oversized_auto_cutoff_smoothing_window_is_bounded_by_grid() -> None:
    from vitriflow.analysis.structure import estimate_pair_cutoffs
    from vitriflow.config import AutoCutoffConfig

    rng = np.random.default_rng(12345)
    frame = _frame(
        np.ones(40, dtype=int),
        rng.uniform(0.0, 10.0, size=(40, 3)),
        cell=10.0,
    )
    auto = AutoCutoffConfig(
        r_max=4.0,
        nbins=20,
        smooth=999,
        peak_search=(0.1, 2.0),
        min_search=(0.2, 3.9),
    )
    result = estimate_pair_cutoffs(
        [frame], [(1, 1)], auto=auto, fixed_cutoffs={}
    )
    assert np.isfinite(result[(1, 1)])
