# GAP-20U+gr hard-carbon custom-schedule demonstrator

This bundle uses the generic custom-stage workflow:

```bash
vitriflow run-schedule -c CONFIG.yaml -o OUTDIR
# aliases:
vitriflow run-custom -c CONFIG.yaml -o OUTDIR
vitriflow run-custom-schedule -c CONFIG.yaml -o OUTDIR
```

`run-hardcarbon` and `run-hc` remain compatibility aliases, but the implementation is no longer hard-carbon-specific. The schedule is defined entirely by `custom_schedule.stages` in the YAML. The standard `vitriflow run` and `vitriflow autotune` workflows remain separate.

## Environment and LAMMPS/QUIP build

From the package root, use:

```bash
conda env create -f environment_quip.yml
conda activate vitriflow-quip
pip install -e .
bash scripts/build_lammps_quip.sh
```

This installs the custom `lmp_quip` executable used by the YAMLs.

## Demonstrator schedule and scope

At `md.timestep: 0.001` ps:

- 9000 K hold: 10 ps = 10000 steps
- 9000 -> 3500 K prequench: 6 ps = 6000 steps
- 3500 K graphitisation hold: 400 ps = 400000 steps
- 3500 -> 300 K quench: 20 ps = 20000 steps
- 300 K relaxation/sampling extension: 20 ps = 20000 steps

This is a literature-inspired hard-carbon schedule demonstrator: it follows the spirit of the published temperature/time scheme, but it is not a verbatim reproduction. Nose-Hoover is used intentionally as the default robust LAMMPS thermostat/barostat choice. CSVR/Bussi-style thermostatting can be selected where supported, but it is not the default target of this demonstrator.

The generated ensemble is not capped at ten boxes. Use `autotune.production.min_boxes`, `max_boxes`, and convergence settings to control how far the campaign runs.

The pilot config overrides the full step counts to test plumbing quickly.

## Thermostat/barostat selection

Default:

```yaml
md:
  ensemble: nvt
  thermostat:
    style: nose-hoover
    tdamp: 0.1
  barostat:
    style: nose-hoover
    pdamp: 1.0
    mode: iso
```

LAMMPS thermostat styles: `nose-hoover`, `csvr`, `langevin`, `berendsen`.
LAMMPS barostat styles for pressure-coupled stages: `nose-hoover`, `berendsen`.

## Use

Copy the four GAP-20U+gr files into `potentials/`:

```text
Carbon_GAP_20U+gr.xml
Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_8891
Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_8892
Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_8893
```

Verify the potential and LAMMPS/QUIP interface:

```bash
python scripts/check_gap20ugr_potential_files.py potentials/Carbon_GAP_20U+gr.xml
LMP_CMD=lmp_quip scripts/smoke_test_gap20ugr_lammps.sh potentials/Carbon_GAP_20U+gr.xml
```

Pilot:

```bash
scripts/launch_pilot_8.sh
```

Inspect schedule:

```bash
grep -n "VITRIFLOW_STAGE\|fix int all nvt temp\|fix th all\|fix bar all\|run " \
  runs/pilot_hc_custom_N128_rho130_8c/production/box_000/continuous/in.lammps
```

Production, one density:

```bash
scripts/launch_density_grid_16.sh 1.3
```

Full grid:

```bash
scripts/launch_density_grid_16.sh
```
