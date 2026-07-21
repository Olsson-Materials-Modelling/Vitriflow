from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _install_minimal_fake_ase_if_needed() -> None:
    try:
        import ase  # noqa: F401
        return
    except Exception:
        pass

    class _Atoms:
        def __init__(self, numbers, positions, cell, pbc=True):
            self.numbers = np.asarray(numbers)
            self.positions = np.asarray(positions, dtype=float)
            self.cell = np.asarray(cell, dtype=float)
            self.pbc = pbc

    def _find_mic(vectors, cell, pbc=True):
        vec = np.asarray(vectors, dtype=float)
        cell_arr = np.asarray(cell, dtype=float)
        inv = np.linalg.inv(cell_arr)
        best = []
        for v in vec.reshape((-1, 3)):
            frac = v @ inv
            frac0 = frac - np.round(frac)
            candidates = []
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for c in (-1, 0, 1):
                        candidates.append((frac0 + np.array([a, b, c], dtype=float)) @ cell_arr)
            best.append(min(candidates, key=lambda x: float(np.dot(x, x))))
        arr = np.asarray(best, dtype=float).reshape(vec.shape)
        return arr, np.linalg.norm(arr.reshape((-1, 3)), axis=1)

    def _neighbor_list(quantities, atoms, cutoff):
        if str(quantities) != "ij":
            raise NotImplementedError("minimal fake ASE only supports neighbor_list('ij', ...)")
        pos = np.asarray(atoms.positions, dtype=float)
        cell = np.asarray(atoms.cell, dtype=float)
        ii = []
        jj = []
        for i in range(pos.shape[0]):
            for j in range(pos.shape[0]):
                if i == j:
                    continue
                _dr, dd = _find_mic(pos[j] - pos[i], cell, pbc=True)
                if float(dd[0]) <= float(cutoff) + 1.0e-12:
                    ii.append(i)
                    jj.append(j)
        return np.asarray(ii, dtype=int), np.asarray(jj, dtype=int)

    ase_mod = types.ModuleType("ase")
    ase_mod.Atoms = _Atoms
    nl_mod = types.ModuleType("ase.neighborlist")
    nl_mod.neighbor_list = _neighbor_list
    geom_mod = types.ModuleType("ase.geometry")
    geom_mod.find_mic = _find_mic
    geom_geom_mod = types.ModuleType("ase.geometry.geometry")
    geom_geom_mod.find_mic = _find_mic
    sys.modules.setdefault("ase", ase_mod)
    sys.modules.setdefault("ase.neighborlist", nl_mod)
    sys.modules.setdefault("ase.geometry", geom_mod)
    sys.modules.setdefault("ase.geometry.geometry", geom_geom_mod)


_install_minimal_fake_ase_if_needed()


def _frame(types, positions, *, cell=40.0):
    from vitriflow.analysis.dump import DumpFrame

    types_arr = np.asarray(types, dtype=int)
    pos_arr = np.asarray(positions, dtype=float)
    return DumpFrame(
        timestep=0,
        ids=np.arange(1, len(types_arr) + 1, dtype=int),
        types=types_arr,
        positions=pos_arr,
        cell=np.eye(3, dtype=float) * float(cell),
        origin=np.zeros(3, dtype=float),
    )


def _metrics(**kwargs):
    from vitriflow.config import StructureMetricsConfig

    data = {"enabled": True}
    data.update(kwargs)
    cfg = StructureMetricsConfig.model_validate(data)
    # Low-level graph tests do not exercise coordinate-only void sampling.
    cfg.voids.enabled = False
    return cfg


def _row_value(rows, name):
    vals = [r["metric_value"] for r in rows if r.get("metric_name") == name]
    assert vals, f"metric row not found: {name}"
    return float(vals[0])


def test_manifest_hash_mismatch_fails_before_descriptor_analysis():
    from vitriflow.analysis.graph import manifest_row_from_frame, verify_manifest_row

    fr = _frame([1, 2], [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]])
    row = manifest_row_from_frame(
        fr,
        box_id=0,
        source_path=Path("final-1.restart"),
        source_role="final_restart",
        type_to_species=["Si", "N"],
        density=3.1,
    )

    contaminated = _frame([1, 2], [[10.0, 10.0, 10.0], [11.2, 10.0, 10.0]])
    with pytest.raises(ValueError, match="structure manifest hash mismatch"):
        verify_manifest_row(contaminated, row, type_to_species=["Si", "N"])


def test_graph_metric_outputs_carry_graph_provenance():
    from vitriflow.analysis.graph_metrics import graph_analysis_for_frame

    fr = _frame([1, 2, 2], [[10, 10, 10], [11, 10, 10], [10, 11, 10]])
    metrics = _metrics(
        coordinations=[{"central": 1, "neighbor": 2, "expected": 2}],
        angles=[{"triplet": [2, 1, 2]}],
        graph_rules=[{"name": "r12", "kind": "hard_cutoff", "parameters": {"cutoff": 1.2}, "provenance": "unit_test"}],
    )
    analysis = graph_analysis_for_frame(
        fr,
        metrics,
        box_id=3,
        type_to_species=None,
        legacy_cutoffs={},
        source_path=Path("box_003/final-1.restart"),
        source_role="final_restart",
        density=None,
    )
    rows = analysis["graph_metric_rows"]
    assert rows
    required = {
        "structure_hash",
        "graph_rule_name",
        "graph_rule_kind",
        "graph_rule_parameters",
        "graph_rule_provenance",
        "metric_family",
        "metric_name",
        "metric_value",
    }
    assert all(required.issubset(set(row)) for row in rows)
    assert {row["graph_rule_name"] for row in rows} == {"r12"}
    assert {row["graph_rule_kind"] for row in rows} == {"hard_cutoff"}


def test_changing_graph_rule_changes_coordination_in_synthetic_example():
    from vitriflow.analysis.graph import GraphRule
    from vitriflow.analysis.graph_metrics import compute_graph_metric_rows

    fr = _frame([1, 2, 2], [[10, 10, 10], [11.0, 10, 10], [11.6, 10, 10]])
    metrics = _metrics(coordinations=[{"central": 1, "neighbor": 2}])
    rows_short, _ = compute_graph_metric_rows(
        fr,
        metrics,
        box_id=0,
        graph_rules=[GraphRule("r12", "hard_cutoff", {"cutoff": 1.2}, "unit_test")],
    )
    rows_long, _ = compute_graph_metric_rows(
        fr,
        metrics,
        box_id=0,
        graph_rules=[GraphRule("r20", "hard_cutoff", {"cutoff": 2.0}, "unit_test")],
    )
    assert _row_value(rows_short, "coord_1-2_mean") == pytest.approx(1.0)
    assert _row_value(rows_long, "coord_1-2_mean") == pytest.approx(2.0)


def test_changing_graph_rule_changes_angle_counts_in_synthetic_example():
    from vitriflow.analysis.graph import GraphRule
    from vitriflow.analysis.graph_metrics import compute_graph_metric_rows

    fr = _frame([1, 2, 2], [[10, 10, 10], [11.0, 10, 10], [10, 11.6, 10]])
    metrics = _metrics(angles=[{"triplet": [2, 1, 2]}])
    rows_short, _ = compute_graph_metric_rows(
        fr,
        metrics,
        box_id=0,
        graph_rules=[GraphRule("r12", "hard_cutoff", {"cutoff": 1.2}, "unit_test")],
    )
    rows_long, _ = compute_graph_metric_rows(
        fr,
        metrics,
        box_id=0,
        graph_rules=[GraphRule("r20", "hard_cutoff", {"cutoff": 2.0}, "unit_test")],
    )
    assert _row_value(rows_short, "angle_2-1-2_count") == pytest.approx(0.0)
    assert _row_value(rows_long, "angle_2-1-2_count") == pytest.approx(1.0)


def test_changing_graph_rule_changes_ring_counts_in_synthetic_example():
    from vitriflow.analysis.graph import GraphRule
    from vitriflow.analysis.graph_metrics import compute_graph_metric_rows

    fr = _frame(
        [1, 1, 1, 1],
        [[10, 10, 10], [11, 10, 10], [11, 11, 10], [10, 11, 10]],
    )
    metrics = _metrics(
        rings={"enabled": True, "mode": "bond_graph", "algorithm": "cycle_basis", "nodes": [1], "max_cycle_size": 6}
    )
    rows_none, _ = compute_graph_metric_rows(
        fr,
        metrics,
        box_id=0,
        graph_rules=[GraphRule("r08", "hard_cutoff", {"cutoff": 0.8}, "unit_test")],
    )
    rows_ring, _ = compute_graph_metric_rows(
        fr,
        metrics,
        box_id=0,
        graph_rules=[GraphRule("r11", "hard_cutoff", {"cutoff": 1.1}, "unit_test")],
    )
    assert _row_value(rows_none, "ring_count") == pytest.approx(0.0)
    assert _row_value(rows_ring, "ring_count") == pytest.approx(1.0)


def test_robust_coordination_partition_labels_controlled_ordered_distances():
    from vitriflow.analysis.graph import GraphRule
    from vitriflow.analysis.graph_metrics import robust_coordination_partition

    positions = [
        [20, 20, 20], [21.0, 20, 20], [20, 21.0, 20],
        [60, 20, 20], [61.0, 20, 20], [61.6, 20, 20],
        [100, 20, 20], [101.0, 20, 20], [100, 21.0, 20], [100, 20, 21.0],
        [140, 20, 20], [141.0, 20, 20], [141.2, 20, 20],
    ]
    types = [1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2]
    fr = _frame(types, positions, cell=200.0)
    metrics = _metrics(coordinations=[{"central": 1, "neighbor": 2, "expected": 2}])
    rows, shell_rows = robust_coordination_partition(
        fr,
        metrics,
        box_id=0,
        interval_rule=GraphRule("interval", "hard_cutoff_interval", {"r_min": 1.1, "r_max": 1.4}, "unit_test"),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["robust_ideal_count"] == 1
    assert row["robust_undercoordinated_count"] == 1
    assert row["robust_overcoordinated_count"] == 1
    assert row["ambiguous_count"] == 1
    assert row["ambiguous_fraction"] == pytest.approx(0.25)
    assert shell_rows and shell_rows[0]["coordination_metric"] == "coord_1-2"


def test_soft_logistic_ambiguity_score_matches_finite_difference_sensitivity():
    from vitriflow.analysis.graph import GraphRule
    from vitriflow.analysis.graph_metrics import compute_graph_metric_rows

    fr = _frame([1, 2, 2], [[10, 10, 10], [10.95, 10, 10], [11.05, 10, 10]])
    metrics = _metrics(coordinations=[{"central": 1, "neighbor": 2}])
    sigma = 0.2

    def soft_coord(r0: float) -> tuple[float, float]:
        rows, _ = compute_graph_metric_rows(
            fr,
            metrics,
            box_id=0,
            graph_rules=[GraphRule("soft", "soft_logistic", {"r0": r0, "sigma": sigma}, "unit_test")],
        )
        return _row_value(rows, "soft_coord_1-2_mean"), _row_value(rows, "soft_coord_1-2_ambiguity_mean")

    eps = 1.0e-5
    c_plus, _ = soft_coord(1.0 + eps)
    c_minus, _ = soft_coord(1.0 - eps)
    c0, ambiguity = soft_coord(1.0)
    assert math.isfinite(c0)
    finite_diff = (c_plus - c_minus) / (2.0 * eps)
    assert finite_diff == pytest.approx(ambiguity / sigma, rel=1e-5, abs=1e-5)


def test_legacy_single_cutoff_mode_emits_marker():
    from vitriflow.analysis.graph_metrics import graph_analysis_for_frame

    fr = _frame([1, 2], [[10, 10, 10], [11, 10, 10]])
    metrics = _metrics(coordinations=[{"central": 1, "neighbor": 2}])
    analysis = graph_analysis_for_frame(
        fr,
        metrics,
        box_id=0,
        type_to_species=None,
        legacy_cutoffs={(1, 2): 1.2},
        source_path=None,
        source_role="unit_test",
        density=None,
    )
    assert analysis["legacy_single_cutoff"]["present"] is True
    assert analysis["legacy_single_cutoff"]["label"] == "legacy_single_cutoff"
    assert {r["graph_rule_name"] for r in analysis["graph_metric_rows"]} == {"legacy_single_cutoff"}


def test_descriptor_functions_do_not_construct_hidden_cutoffs_when_graph_supplied(monkeypatch):
    from vitriflow.analysis.graph import GraphRule, build_hard_graph
    from vitriflow.analysis import structure as structure_mod
    from vitriflow.analysis.structure import compute_structure_metrics

    fr = _frame([1, 2], [[10, 10, 10], [11, 10, 10]])
    metrics = _metrics(coordinations=[{"central": 1, "neighbor": 2}])
    graph = build_hard_graph(fr, GraphRule("explicit", "hard_cutoff", {"cutoff": 1.2}, "unit_test"))

    def _forbidden(*args, **kwargs):
        raise AssertionError("hidden cutoff graph construction was called")

    monkeypatch.setattr(structure_mod, "build_hard_graph", _forbidden)
    sm = compute_structure_metrics(fr, metrics, cutoffs={(1, 2): 99.0}, graph=graph)
    assert sm.values["coord_1-2_mean"] == pytest.approx(1.0)


def test_rdf_adaptive_graph_rule_resolves_per_structure_not_yaml_cutoff():
    from vitriflow.analysis.graph_metrics import graph_analysis_for_frame

    fr_a = _frame(
        [1, 2, 2, 2, 2],
        [[10, 10, 10], [11.0, 10, 10], [10, 11.1, 10], [12.5, 10, 10], [10, 12.8, 10]],
    )
    fr_b = _frame(
        [1, 2, 2, 2, 2],
        [[10, 10, 10], [11.0, 10, 10], [10, 11.4, 10], [12.5, 10, 10], [10, 12.8, 10]],
    )
    metrics = _metrics(
        coordinations=[{"central": 1, "neighbor": 2, "expected": 2}],
        graph_rules=[
            {
                "name": "adaptive_1_2",
                "kind": "rdf_adaptive_hard_cutoff",
                "parameters": {"pair": [1, 2], "mode": "single", "connectivity_fraction": "auto"},
                "provenance": "unit_test",
            }
        ],
    )

    ana = graph_analysis_for_frame(fr_a, metrics, box_id=1, type_to_species=None, legacy_cutoffs={}, source_path=None, source_role="unit", density=None)
    anb = graph_analysis_for_frame(fr_b, metrics, box_id=2, type_to_species=None, legacy_cutoffs={}, source_path=None, source_role="unit", density=None)
    cutoff_a = float(ana["graph_rules"][0]["parameters"]["cutoff"])
    cutoff_b = float(anb["graph_rules"][0]["parameters"]["cutoff"])

    assert cutoff_a == pytest.approx(1.1)
    assert cutoff_b == pytest.approx(1.4)
    assert cutoff_a != pytest.approx(cutoff_b)
    assert ana["adaptive_graph_rule_records"]
    assert ana["graph_rules"][0]["parameters"]["rdf_adaptive"] is True
    assert ana["graph_rules"][0]["provenance"]["source"] == "rdf_adaptive_graph_rule"

def test_rdf_adaptive_shell_objective_minimises_accidental_neighbour_inclusion():
    from vitriflow.analysis.graph_metrics import graph_analysis_for_frame

    # Two isolated 1-centred shells with expected z=1.  The distributions overlap:
    # d_z = [1.0, 1.5] and d_(z+1) = [1.2, 1.6].  A max-d_z rule would choose
    # 1.5 and include an accidental second neighbour from the first shell.  The
    # RDF-adaptive shell objective should prefer the equally good lower-loss
    # boundary with fewer accidental neighbours.
    fr = _frame(
        [1, 2, 2, 1, 2, 2],
        [
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [11.2, 10.0, 10.0],
            [30.0, 30.0, 30.0],
            [31.5, 30.0, 30.0],
            [31.6, 30.0, 30.0],
        ],
        cell=80.0,
    )
    metrics = _metrics(
        coordinations=[{"central": 1, "neighbor": 2, "expected": 1}],
        graph_rules=[
            {
                "name": "adaptive_shell_loss",
                "kind": "rdf_adaptive_hard_cutoff",
                "parameters": {"pair": [1, 2], "mode": "single", "search_radius": 3.0, "connectivity_fraction": "auto"},
            }
        ],
    )

    analysis = graph_analysis_for_frame(fr, metrics, box_id=9, type_to_species=None, legacy_cutoffs={}, source_path=None, source_role="unit", density=None)
    params = analysis["graph_rules"][0]["parameters"]
    deriv = params["derivation"][0]

    assert float(params["cutoff"]) == pytest.approx(1.0)
    assert deriv["shell_separability"]["shell_objective"] == "minimise_under_plus_accidental_neighbour_fraction"
    assert deriv["shell_separability"]["shell_objective_accidental_fraction"] == pytest.approx(0.0)
    assert deriv["shell_separability"]["d_z_max"] == pytest.approx(1.5)



def test_multi_pair_interval_rule_populates_robust_stability_rows():
    from vitriflow.analysis.graph import GraphRule
    from vitriflow.analysis.graph_metrics import robust_coordination_partition

    fr = _frame(
        [1, 2, 2, 1, 2, 2],
        [
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [30.0, 30.0, 30.0],
            [31.0, 30.0, 30.0],
            [30.0, 31.5, 30.0],
        ],
        cell=80.0,
    )
    metrics = _metrics(coordinations=[{"central": 1, "neighbor": 2, "expected": 2}])
    interval = GraphRule(
        "adaptive_interval",
        "hard_cutoff_interval",
        {
            "pair_intervals": [
                {"pair": [1, 1], "r_min": 2.0, "r_max": 2.5},
                {"pair": [1, 2], "r_min": 1.1, "r_max": 1.4},
                {"pair": [2, 2], "r_min": 1.8, "r_max": 2.3},
            ]
        },
        "unit_test",
    )
    rows, shell_rows = robust_coordination_partition(fr, metrics, box_id=4, interval_rule=interval)
    assert len(rows) == 1
    assert len(shell_rows) == 1
    assert rows[0]["coordination_metric"] == "coord_1-2"
    assert rows[0]["r_min"] == pytest.approx(1.1)
    assert rows[0]["r_max"] == pytest.approx(1.4)
    assert rows[0]["pair_intervals_used"] == [[1, 2]]


def test_adaptive_primary_graph_rule_feeds_coordination_defect_reporting():
    from vitriflow.analysis.graph import build_graph
    from vitriflow.analysis.graph_metrics import graph_analysis_for_frame
    from vitriflow.analysis.structure import compute_coordination_defects, compute_structure_distributions_for_graph
    from vitriflow.workflows.production_common import _primary_hard_graph_rule_from_analysis

    # The first Si has only one N neighbour inside the adaptive RDF/search shell;
    # the second Si supplies a well-defined two-neighbour shell used to derive the
    # per-structure cutoff.  The defect path must use that explicit graph rather
    # than returning an empty {} because no legacy cutoff map exists.
    fr = _frame(
        [1, 2, 1, 2, 2],
        [
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [30.0, 30.0, 30.0],
            [31.0, 30.0, 30.0],
            [30.0, 31.1, 30.0],
        ],
        cell=80.0,
    )
    metrics = _metrics(
        pairs=[{"pair": [1, 2]}],
        coordinations=[{"central": 1, "neighbor": 2, "expected": 2, "defect_frac_tol": 0.0}],
        graph_rules=[
            {
                "name": "adaptive_primary",
                "kind": "rdf_adaptive_hard_cutoff",
                "parameters": {"pair": [1, 2], "mode": "single", "search_radius": 3.0, "connectivity_fraction": "auto"},
                "provenance": "unit_test",
            }
        ],
    )
    analysis = graph_analysis_for_frame(fr, metrics, box_id=7, type_to_species=None, legacy_cutoffs={}, source_path=None, source_role="unit", density=None)
    rule = _primary_hard_graph_rule_from_analysis(analysis)
    assert rule is not None
    graph = build_graph(fr, rule)
    defects = compute_coordination_defects(fr, metrics, cutoffs={}, graph=graph)
    assert defects["coord_1-2"]["n_defective"] == 1
    assert defects["coord_1-2"]["has_defect"] is True
    assert defects["coord_1-2"]["graph_rule"]["parameters"]["rdf_adaptive"] is True
    dist = compute_structure_distributions_for_graph(fr, metrics, graph=graph)
    assert dist["coord"]["coord_1-2"]["available"] is True
    assert dist["coord"]["coord_1-2"]["graph_rule"]["parameters"]["rdf_adaptive"] is True


def test_rdf_adaptive_all_pairs_split_network_and_candidate_contact_families():
    from vitriflow.analysis.graph_metrics import graph_analysis_for_frame

    fr = _frame(
        [1, 1, 1, 2, 2, 2],
        [
            [10.0, 10.0, 10.0],
            [12.35, 10.0, 10.0],
            [10.0, 12.35, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [11.0, 11.0, 10.0],
        ],
        cell=50.0,
    )
    metrics = _metrics(
        pairs=[{"pair": [1, 2]}, {"pair": [1, 1]}, {"pair": [2, 2]}],
        coordinations=[{"central": 1, "neighbor": 2, "expected": 1}],
        angles=[{"triplet": [2, 1, 2]}],
        graph_rules=[
            {
                "name": "adaptive_all_pairs",
                "kind": "rdf_adaptive",
                "parameters": {"pairs": [[1, 2], [1, 1], [2, 2]], "mode": "single", "search_radius": 5.0},
            }
        ],
    )
    analysis = graph_analysis_for_frame(fr, metrics, box_id=11, type_to_species=None, legacy_cutoffs={}, source_path=None, source_role="unit", density=None)
    families = {r["parameters"].get("graph_family") for r in analysis["graph_rules"]}
    assert "network_graph" in families
    assert "candidate_contact_graph" in families

    network_rules = [r for r in analysis["graph_rules"] if r["parameters"].get("graph_family") == "network_graph"]
    defect_rules = [r for r in analysis["graph_rules"] if r["parameters"].get("graph_family") == "candidate_contact_graph"]
    assert network_rules
    assert defect_rules
    assert {tuple(row["pair"]) for row in network_rules[0]["parameters"]["pair_cutoffs"]} == {(1, 2)}
    assert {(1, 1), (1, 2), (2, 2)}.issubset({tuple(row["pair"]) for row in defect_rules[0]["parameters"]["pair_cutoffs"]})

    network_metric_names = {r["metric_name"] for r in analysis["graph_metric_rows"] if r.get("graph_family") == "network_graph"}
    defect_metric_names = {r["metric_name"] for r in analysis["graph_metric_rows"] if r.get("graph_family") == "candidate_contact_graph"}
    assert not any(name.startswith("homopolar_") for name in network_metric_names)
    assert any(name.startswith("homopolar_") for name in defect_metric_names)
    assert any(name.startswith("candidate_contact_") for name in defect_metric_names)
    assert not any(name.startswith("ring_") for name in defect_metric_names)


def test_ensemble_graph_outputs_use_ensemble_scope_and_family_columns():
    from vitriflow.analysis.graph_metrics import collect_ensemble_graph_rows_from_entries

    fr1 = _frame([1, 2, 2], [[10, 10, 10], [11.0, 10, 10], [10, 11.2, 10]], cell=40.0)
    fr2 = _frame([1, 2, 2], [[10, 10, 10], [11.1, 10, 10], [10, 11.4, 10]], cell=40.0)

    def entry(box, fr):
        return {
            "box": int(box),
            "structure": {
                "timestep": int(fr.timestep),
                "ids": [int(x) for x in fr.ids.tolist()],
                "types": [int(x) for x in fr.types.tolist()],
                "positions": [[float(x) for x in row] for row in fr.positions.tolist()],
                "lattice": {"cell": [[float(x) for x in row] for row in fr.cell.tolist()], "origin": [0.0, 0.0, 0.0]},
            },
        }

    metrics = _metrics(
        pairs=[{"pair": [1, 2]}],
        coordinations=[{"central": 1, "neighbor": 2, "expected": 2}],
        graph_rules=[
            {
                "name": "ensemble_adaptive",
                "kind": "rdf_adaptive",
                "parameters": {"pair": [1, 2], "mode": "all", "search_radius": 3.0, "points": 3},
            }
        ],
    )
    collected = collect_ensemble_graph_rows_from_entries([entry(1, fr1), entry(2, fr2)], metrics=metrics, type_to_species=None, legacy_cutoffs={})
    assert collected["graph_rules"]
    assert collected["adaptive_graph_rule_derivation_records"]
    derivation = collected["adaptive_graph_rule_derivation_records"][0]
    assert derivation["derivation_ref"].startswith("deriv:")
    assert derivation["derivation"]
    assert collected["graph_metric_by_rule"]
    assert collected["coordination_stability"]
    assert {r.get("graph_rule_scope") for r in collected["graph_metric_by_rule"]} == {"ensemble"}
    assert "network_graph" in {r.get("graph_family") for r in collected["graph_metric_by_rule"]}
    assert collected["graph_uncertainty_summary"]
    assert {r.get("graph_rule_scope") for r in collected["graph_uncertainty_summary"]} == {"ensemble"}
