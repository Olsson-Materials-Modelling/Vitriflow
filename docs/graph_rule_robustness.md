# Graph-rule descriptors

Graph-derived descriptors — coordination, rings, angles, motifs, defect
clusters, connected components, path lengths — are computed through an explicit
graph-induction rule rather than an implicit neighbour graph. For a structure
`x = (cell, species, positions)` and a rule `lambda`:

```text
graph  = G_lambda(x)
metric = M_lambda(x) = M(G_lambda(x))
```

This layer is **opt-in**. An empty or absent `metrics.graph_rules` keeps the
historical cutoff-driven path and writes no graph sidecars. Structure manifests
are always written as baseline provenance and do not, by themselves, enable
graph analysis.

## Manifest-locked structures

Each analysed structure is written to a manifest and re-verified by hash before
descriptors are evaluated, preventing first/final-frame mixups and restart
contamination from propagating into descriptor tables. The manifest hash is
`S_i = (cell_i, species_i, positions_i, pbc_i)`. Each row records box id, source
path and role, structure/cell/position/symbol hashes, PBC and its provenance,
volume and density where available, and source file size and SHA-256. A LAMMPS
data file cannot encode boundary flags, so its fully periodic PBC is recorded as
a workflow assumption rather than source-verified metadata.

## Graph rules

A rule has `name`, `kind`, `parameters`, and `provenance`. Supported kinds:

```text
hard_cutoff                          rdf_adaptive
hard_cutoff_sweep                    rdf_adaptive_hard_cutoff
hard_cutoff_interval                 rdf_adaptive_hard_cutoff_sweep
soft_logistic                        rdf_adaptive_hard_cutoff_interval
                                     rdf_adaptive_soft_logistic
```

Sweeps and intervals expand to concrete hard-cutoff rules before evaluation. A
soft logistic rule weights edges and reports soft coordination `c_soft` and a
transition-shell ambiguity score `a_i`:

```text
w_ij   = 1 / (1 + exp((d_ij - r0) / sigma))
c_soft = sum_j w_ij
a_i    = sum_j w_ij (1 - w_ij)
```

**RDF-adaptive** kinds store an induction algorithm instead of a numeric cutoff;
the concrete `lambda` is resolved per structure from that structure's partial
RDF and shell-separability diagnostics (first peak, first minimum, second-shell
onset, ordered shell bounds, optional connectivity floor). Resolved values are
written to `graph_rules.json`, `adaptive_graph_rules.json`, and the
`graph_rule_parameters` / `graph_rule_provenance` CSV columns. Use these for
amorphous systems where a fixed Angstrom cutoff would be arbitrary.

## Graph families

All-pair rules are split into named families by default
(`graph_family_strategy: network_and_defect_candidate_split`):

```text
network_graph               Backbone topology (expected-shell, ring-bond,
                            angle-edge pairs). Use for coordination, rings,
                            angles, components, path lengths.
defect_candidate_graph      Diagnostic close-contact graph; retains all
                            requested RDF-derived pairs incl. homopolar
                            (Si-Si, N-N). Use for homopolar-bond and defect
                            evidence.
soft_ambiguity_graph        Soft weighted network graph for soft coordination
                            and transition-shell ambiguity.
legacy_single_cutoff_graph  Compatibility family; only inside explicitly
                            requested graph analysis.
```

Set `graph_family_strategy: unified` or `legacy` to disable the split.

## Scopes and uncertainty

Rules are evaluated at two scopes, recorded in the `graph_rule_scope` column:
`per_structure` (rule derived independently per structure) and `ensemble` (rule
derived from the pooled ensemble and applied to all). For a rule interval,
representation uncertainty is reported separately from sampling uncertainty:

```text
width         = max_lambda mean(M_lambda) - min_lambda mean(M_lambda)
bootstrap_se  = SE over structures at a fixed rule
width_over_se = width / bootstrap_se
```

More boxes reduce `bootstrap_se` but not `width`. For expected coordination `z`
and interval `[r_min, r_max]`, each site is labelled `robust ideal`,
`robust undercoordinated`, `robust overcoordinated`, or `ambiguous` from its
ordered neighbour distances.

## Outputs

Per-structure:

```text
structure_manifest.json
graph_rules.json
adaptive_graph_rules.json
graph_metric_by_rule.csv
coordination_stability.csv
shell_separability.csv
graph_uncertainty_summary.csv
```

Ensemble:

```text
ensemble_graph_rules.json
ensemble_adaptive_graph_rules.json
ensemble_graph_metric_by_rule.csv
ensemble_coordination_stability.csv
ensemble_shell_separability.csv
ensemble_graph_uncertainty_summary.csv
```

Summaries:

```text
graph_family_summary.json
legacy_single_cutoff_summary.json
```

`graph_metric_by_rule.csv` (and its `ensemble_` form) carry one row per
structure, rule, and metric, with columns:

```text
box_id, structure_hash, graph_rule_scope, graph_family, graph_rule_name,
graph_rule_kind, graph_rule_parameters, graph_rule_provenance, metric_family,
metric_name, metric_value
```

Legacy single-cutoff fields (`boxes[].metrics`, `boxes[].distributions`, legacy
coordination-defect summaries) are retained and marked as legacy;
`graph_metric_by_rule.csv` is the provenance-carrying descriptor table.

## Configure

Via CLI:

```bash
vitriflow analyze-output -c config.yaml -i production_dir -o out --graph-cutoff 2.0
vitriflow analyze-output -c config.yaml -i production_dir -o out --graph-cutoff-sweep 1.8,1.9,2.0,2.1
vitriflow analyze-output -c config.yaml -i production_dir -o out --graph-cutoff-interval 1.8 2.1 --graph-interval-points 9
vitriflow analyze-output -c config.yaml -i production_dir -o out --soft-logistic 2.0 0.05
```

Or in YAML under `metrics.graph_rules` (run config) or top-level `graph_rules`
(standalone analysis config):

```yaml
graph_rules:
  - name: si3n4_rdf_adaptive_all_pairs
    kind: rdf_adaptive
    parameters:
      derive_from: pair_distribution_function
      pairs: [[Si, N], [Si, Si], [N, N]]
      network_pairs: [[Si, N]]
      graph_family_strategy: network_and_defect_candidate_split
      ensemble_scope: true
    provenance:
      source: analysis_yaml
```
