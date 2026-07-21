# Descriptor provenance

Amorphous-material descriptors depend on both the structure and an explicit
representation rule, and are reported with that rule attached. A descriptor value
is only meaningful together with its rule.

## Graph descriptors

Graph-derived descriptors follow `x -> G_lambda(x) -> F_lambda(x)`, where
`lambda` is the graph-induction rule; coordination, rings, angles, motifs, defect
clusters, components, and path lengths all inherit that rule's uncertainty. Over
a rule interval, representation uncertainty is reported as
`W_M = sup_lambda mean(M) - inf_lambda mean(M)`. See
[graph_rule_robustness.md](graph_rule_robustness.md) for rule kinds, families,
scopes, and the full output set.

## Void descriptors

The void representation records the probe definition, sampler/grid, clearance
definition, PBC treatment, units, and normalization. Absolute clearances carry
density/scale effects, so reduced descriptors are also reported:

```text
length_scale     = (V / N)^(1/3)
clearance_scaled = clearance / length_scale
```

Raw, density-scaled, and density-residualized void descriptors are distinct
representation rules and are stored separately.

## Learned descriptors

For model-based descriptors (`x -> model representation -> F(x)`), the
representation rule records model name, version, featurization, and checkpoint
hash.

## Structure embedding

`analyze-output` controls whether final-frame coordinates are embedded in
`analysis_results.json`:

```yaml
analysis:
  embed_structures: true    # default: full coordinates embedded
  # embed_structures: false # compact: coordinates omitted
```

With `embed_structures: false`, each box still records source path and role,
structure/cell/positions/symbols hashes, density, volume, and lattice summary;
only the coordinate arrays are dropped. Descriptors are manifest-locked to the
analysed structure hash either way.

## Convergence (analysis-only)

For analysis-only datasets, accept/reject and descriptor-set convergence are
advisory: they report how many boxes would be rejected and whether short-,
medium-, and long-range descriptor families plus ensemble size have converged,
without removing structures. Plotting shows the available descriptor data even
when a production `familywise` convergence report is absent.

## Ensemble CDFs

Per-box CDFs are not assumed to share a grid. For each family, Vitriflow builds
an explicit ensemble grid, evaluates every per-box CDF on it, and writes a
sidecar recording the per-box-mean CDF, the sample-count-weighted pooled CDF, and
whether the source grids were native-common or regridded.

## Streaming sidecars

For large ensembles, descriptor rows stay compact and carry a `derivation_ref`;
heavy RDF/shell derivations are written once to
`adaptive_graph_rule_derivations.json` or
`ensemble_adaptive_graph_rule_derivations.json`. Transient
`.analysis_stream_chunks/` files are removed after finalization.

## Scope

These outputs are descriptor-map provenance: Vitriflow declares the source space,
representation map, descriptor map, parameters, and uncertainty diagnostics.
