# Vitriflow application-facing validation contracts

The run encodes these application-facing behaviours as fail-closed runtime
assertions:

1. `plot-voids` must resolve its own species mapping.
2. rate-scan plotting must consume the emitted rate field and draw finite data.
3. production plots must report the familywise error rate correctly and
   preserve nullable, inference-qualified convergence semantics.
4. `analyze-output` must reproduce production's canonical convergence result
   exactly when no analysis overrides are supplied.
5. every emitted metric must be inventoried and plotted; strictly more than
   half must enter convergence, spanning short, medium, and long range.
6. external Slurm tasks must run the same stage, amorphous, coordination-sweep,
   void, and elastic diagnostics as local production and bind every resulting
   artifact by size and SHA-256 before collection.
7. CP2K `CELL_OPT` output is accepted only after a positive completion marker;
   the refined structure is re-analysed and converged separately from MD.
8. every public CLI result object must exit nonzero for incomplete,
   non-converged, failed, or unknown status.

`VALIDATION_PASSED.json` is emitted only after these contracts, both exact
passes, and every selected runtime path complete successfully.
