from __future__ import annotations

from pathlib import Path

import pytest


def _valid_plan_payload(tmp_path: Path) -> dict:
    from vitriflow.config import ConvergenceConfig, MDConfig, ProductionEnsembleConfig, StructureMetricsConfig

    structure = tmp_path / "base.data"
    structure.write_text("LAMMPS data file\n\n0 atoms\n")
    return {
        "schema": "vitriflow.production_plan.v1",
        "engine": "lammps",
        "structure_data": str(structure),
        "T_high": 1200.0,
        "high_total_steps": 5000,
        "t_final": 300.0,
        "chosen_rate": 10.0,
        "cooling_rate_ps": 10.0,
        "replicate": [2, 2, 2],
        "pressure": 0.0,
        "md_use": MDConfig().model_dump(mode="json"),
        "potential_config": {
            "kind": "kim",
            "model": "dummy",
            "user_units": "metal",
            "interactions": ["Al"],
        },
        "potential_lines": ["pair_style kim dummy"],
        "core_repulsion": {"enabled": False},
        "type_to_species": ["Al"],
        "metrics_cfg": StructureMetricsConfig().model_dump(mode="json"),
        "effective_metrics": {},
        "production_cfg": ProductionEnsembleConfig().model_dump(mode="json"),
        "convergence_cfg": ConvergenceConfig().model_dump(mode="json"),
        "cutoffs_rate": [{"pair": [1, 1], "cutoff": 3.0}],
        "cutoffs_size": [{"pair": [1, 1], "cutoff": 3.1}],
        "preferred_cutoffs": [{"pair": [1, 1], "cutoff": 3.1}],
        "quench_steps": 90,
        "relax_steps": 200,
        "msd_every": 100,
        "seed_base": 24680,
        "time_unit_ps": 1.0,
        "sampling_hint": {"Tm": 900.0},
        "execution_mode": "adaptive",
        "source_kind": "test",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema": "vitriflow.production_plan.v2"}, "Unsupported production plan schema"),
        ({"replicate": [2, 0, 2]}, "replicate"),
        ({"T_high": float("nan")}, "T_high must be finite"),
        ({"cooling_rate_ps": 9.0}, "inconsistent"),
        ({"preferred_cutoffs": [{"pair": [1, 1], "cutoff": -1.0}]}, "cutoff must be finite and > 0"),
        ({"preferred_cutoffs": [{"pair": [1.5, 1], "cutoff": 2.0}]}, "pair\\[0\\].*integer"),
        ({"quench_steps": 3.7}, "quench_steps must be an integer"),
        ({"quench_steps": 123}, "quench_steps is inconsistent"),
        ({"pressure": 1.0}, "pressure is inconsistent"),
        ({"production_cfg": {"min_boxes": 4, "max_boxes": 2, "batch_boxes": 1}}, "max_boxes must be >= production.min_boxes"),
        ({"type_to_species": None}, "type_to_species is required"),
        ({"type_to_species": ["Si"]}, "potential interaction ordering"),
        ({"type_to_species": "Si"}, "type_to_species must be a non-string sequence"),
    ],
)
def test_production_plan_rejects_unsafe_inputs(tmp_path: Path, mutation: dict, message: str):
    from vitriflow.workflows.production_common import production_plan_from_dict

    payload = _valid_plan_payload(tmp_path)
    payload.update(mutation)
    with pytest.raises(ValueError, match=message):
        production_plan_from_dict(payload)


def test_production_plan_reports_missing_required_fields(tmp_path: Path):
    from vitriflow.workflows.production_common import production_plan_from_dict

    payload = _valid_plan_payload(tmp_path)
    payload.pop("md_use")
    with pytest.raises(ValueError, match=r"missing required field.*md_use"):
        production_plan_from_dict(payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "engine", "structure_data", "T_high", "high_total_steps", "t_final",
        "chosen_rate", "cooling_rate_ps", "replicate", "pressure", "md_use",
        "potential_config", "potential_lines", "core_repulsion", "type_to_species",
        "metrics_cfg", "effective_metrics", "production_cfg", "convergence_cfg",
        "cutoffs_rate", "cutoffs_size", "preferred_cutoffs", "quench_steps",
        "relax_steps", "msd_every", "seed_base", "time_unit_ps", "sampling_hint",
        "execution_mode", "source_kind",
    ],
)
def test_versioned_production_plan_requires_every_serialized_field(tmp_path: Path, field_name: str):
    from vitriflow.workflows.production_common import production_plan_from_dict

    payload = _valid_plan_payload(tmp_path)
    payload.pop(field_name)
    with pytest.raises(ValueError, match=rf"missing required field.*{field_name}"):
        production_plan_from_dict(payload)


def test_production_plan_requires_schema_and_rejects_unknown_fields(tmp_path: Path):
    from vitriflow.workflows.production_common import production_plan_from_dict

    missing_schema = _valid_plan_payload(tmp_path)
    missing_schema.pop("schema")
    with pytest.raises(ValueError, match="Unsupported production plan schema"):
        production_plan_from_dict(missing_schema)

    unknown = _valid_plan_payload(tmp_path)
    unknown["qunch_steps"] = unknown["quench_steps"]
    with pytest.raises(ValueError, match=r"unknown field.*qunch_steps"):
        production_plan_from_dict(unknown)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"density_abs_tol": float("nan")}, "finite and >= 0"),
        ({"bondlen_rel_tol": -0.1}, "finite and >= 0"),
        ({"gr_curve_abs_tol": float("inf")}, "finite and >= 0"),
        ({"zscore": 0.0}, "finite and > 0"),
        ({"zscore": float("nan")}, "finite and > 0"),
        ({"stability_bootstrap": True}, "integer >= 0"),
        ({"stability_bootstrap": 2.5}, "integer >= 0"),
        ({"stability_quantile": float("nan")}, "finite and in"),
    ],
)
def test_convergence_config_rejects_nonsensical_numeric_controls(mutation, message):
    from pydantic import ValidationError

    from vitriflow.config import ConvergenceConfig

    with pytest.raises(ValidationError, match=message):
        ConvergenceConfig.model_validate(mutation)


@pytest.mark.parametrize(
    ("model_name", "payload"),
    [
        ("MDConfig", {"timestep": float("nan")}),
        ("MDConfig", {"temperature": -1.0}),
        ("MDConfig", {"pressure": float("inf")}),
        ("MDConfig", {"thermo_every": True}),
        ("TmScanConfig", {"t_max": float("nan")}),
        ("TmScanConfig", {"sample_steps": 0}),
        ("HighTConfig", {"chunk_steps": 0}),
        ("QuenchConfig", {"rates_K_per_time": [10.0, float("nan")]}),
        ("PairMetricConfig", {"pair": [1, 1], "cutoff": -1.0}),
        ("CoordinationMetricConfig", {"central": 1, "neighbor": 1, "cutoff": float("inf")}),
        ("GrMetricConfig", {"r_max": float("nan")}),
        ("GraphRuleConfig", {"kind": "soft_logistic", "parameters": {"r0": 2.0, "sigma": 0.0}}),
        ("GraphRuleConfig", {"kind": "hard_cutoff_sweep", "parameters": {"cutoffs": []}}),
        ("GraphRuleConfig", {"kind": "rdf_adaptive", "parameters": {"bin_width": float("nan")}}),
        ("GraphRuleConfig", {"kind": "rdf_adaptive", "parameters": {"connectivity_fraction": 1.1}}),
        ("VoidMetricsConfig", {"seed": -1}),
        ("Cp2kConfig", {"cutoff_Ry": True}),
        ("Cp2kConfig", {"ngrids": True}),
        ("Cp2kConfig", {"data_dir": ""}),
        ("ThermostatConfig", {"tdamp": True}),
        ("BarostatConfig", {"pdamp": True}),
        ("ProductionEnsembleConfig", {"warmup_duration_ps": True}),
        ("PairMetricConfig", {"pair": [True, 1]}),
        ("CoordinationMetricConfig", {"central": "", "neighbor": 1}),
        ("AngleMetricConfig", {"triplet": [1, False, 1]}),
        ("KimConfig", {"model": "dummy", "interactions": [None]}),
        ("LammpsPotentialConfig", {"interactions": ["Al"], "commands": [123]}),
        ("StructureMetricsConfig", {"type_to_species": [True]}),
        ("StructureGenerateConfig", {"method": "cod", "formula": "", "cod_id": 1}),
        ("StructureGenerateConfig", {"method": "cod", "formula": "Al", "cod_id": True}),
    ],
)
def test_execution_critical_models_reject_nonfinite_or_unsafe_values(model_name, payload):
    from pydantic import ValidationError

    import vitriflow.config as config_module

    model = getattr(config_module, model_name)
    with pytest.raises((ValidationError, ValueError)):
        model.model_validate(payload)


def test_structure_generation_seeds_counts_and_charges_are_fail_closed(tmp_path: Path):
    from pydantic import ValidationError

    from vitriflow.config import RunConfig, StructureConfig, StructureGenerateConfig

    with pytest.raises(ValidationError, match="seed"):
        StructureGenerateConfig(method="random", formula="Al", seed=True)
    with pytest.raises(ValidationError, match="min_atoms"):
        StructureGenerateConfig(method="random", formula="Al", min_atoms=0)

    structure = tmp_path / "base.data"
    structure.write_text("placeholder\n")
    with pytest.raises(ValidationError, match="must be finite"):
        StructureConfig(lammps_data=structure, charges={"Al": float("nan")})

    payload = {
        "potential": {
            "kind": "kim",
            "model": "dummy",
            "interactions": ["Al"],
        },
        "structure": {"lammps_data": str(structure)},
        "random_seed": -1,
    }
    with pytest.raises(ValidationError, match="random_seed"):
        RunConfig.model_validate(payload)


def test_yaml_relative_paths_never_search_parent_or_process_cwd(tmp_path: Path, monkeypatch):
    from pydantic import ValidationError

    from vitriflow.config import RunConfig

    # This unrelated same-name file must not rescue a missing path beside the
    # YAML. Scientific inputs resolve only against the YAML directory.
    (tmp_path / "base.data").write_text("unrelated parent artifact\n")
    config_dir = tmp_path / "nested"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "potential: {kind: kim, model: dummy, interactions: [Al]}\n"
        "structure: {lammps_data: base.data}\n"
    )
    monkeypatch.chdir(tmp_path)

    expected = str((config_dir / "base.data").resolve())
    with pytest.raises((FileNotFoundError, ValidationError), match="base.data") as excinfo:
        RunConfig.from_yaml(config_path)
    assert expected in str(excinfo.value)


def _charged_template(path: Path) -> None:
    path.write_text(
        "charged template\n\n"
        "2 atoms\n1 atom types\n\n"
        "0 4 xlo xhi\n0 4 ylo yhi\n0 4 zlo zhi\n\n"
        "Masses\n\n1 1.0\n\n"
        "Atoms # charge\n\n"
        "1 1 1.0 0 0 0\n"
        "2 1 -1.0 1 0 0\n"
    )


def _final_dump(path: Path, *, include_q: bool) -> None:
    columns = "id type q xu yu zu" if include_q else "id type xu yu zu"
    rows = (
        [
            "1 1 1.0 0 0 0",
            "2 1 -1.0 1 0 0",
            "3 1 1.0 2 0 0",
            "4 1 -1.0 3 0 0",
        ]
        if include_q
        else ["1 1 0 0 0", "2 1 1 0 0", "3 1 2 0 0", "4 1 3 0 0"]
    )
    path.write_text(
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n4\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\n"
        f"ITEM: ATOMS {columns}\n" + "\n".join(rows) + "\n"
    )


def test_charged_replicated_final_dump_preserves_authoritative_per_atom_charges(tmp_path: Path):
    from vitriflow.analysis.datafile import read_datafile_charges
    from vitriflow.workflows.stage_runner import _materialize_output_from_final_dump

    template = tmp_path / "input.data"
    dump = tmp_path / "final.lammpstrj"
    output = tmp_path / "output.data"
    _charged_template(template)
    _final_dump(dump, include_q=True)

    _materialize_output_from_final_dump(
        output_local=output,
        final_dump_path=dump,
        template_input=template,
        atom_style="charge",
    )
    assert read_datafile_charges(output, atom_style="charge") == {
        1: 1.0,
        2: -1.0,
        3: 1.0,
        4: -1.0,
    }


def test_charged_replicated_final_dump_without_q_fails_closed(tmp_path: Path):
    from vitriflow.workflows.stage_runner import _materialize_output_from_final_dump

    template = tmp_path / "input.data"
    dump = tmp_path / "final.lammpstrj"
    _charged_template(template)
    _final_dump(dump, include_q=False)
    with pytest.raises(ValueError, match="no q column.*replicated continuous run"):
        _materialize_output_from_final_dump(
            output_local=tmp_path / "output.data",
            final_dump_path=dump,
            template_input=template,
            atom_style="charge",
        )


def test_continuous_charge_renderer_includes_q_in_every_final_dump(tmp_path: Path):
    from pydantic import TypeAdapter

    from vitriflow.config import MDConfig, PotentialConfig
    from vitriflow.lammps_input import StageSpec, render_continuous_stages

    data = tmp_path / "input.data"
    _charged_template(data)
    potential = TypeAdapter(PotentialConfig).validate_python(
        {"kind": "kim", "model": "dummy", "interactions": ["Na", "Cl"]}
    )
    stage = StageSpec(
        name="melt",
        input_data=data,
        output_data=tmp_path / "melt.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
        replicate=(2, 1, 1),
    )
    script = render_continuous_stages(
        potential,
        MDConfig(atom_style="charge", stage_continuity="continuous"),
        [stage],
        stage_dir_prefixes={"melt": "melt"},
    )
    assert "write_dump all custom melt/melt.final.lammpstrj id type q xu yu zu" in script


def test_cp2k_restart_bundle_is_checksum_locked_and_required_for_preserve(tmp_path: Path):
    from vitriflow.lammps_input import StageSpec
    from vitriflow.workflows.stage_runner import (
        _load_cp2k_restart_state_for_stage,
        _publish_cp2k_restart_state,
    )

    previous = tmp_path / "warmup"
    previous.mkdir()
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("&GLOBAL\n  PROJECT warmup\n&END GLOBAL\n")
    raw_wfn = previous / "warmup-RESTART.wfn"
    raw_wfn.write_bytes(b"wfn")
    input_data = previous / "warmup.data"
    input_data.write_text("placeholder\n")
    _publish_cp2k_restart_state(
        previous,
        coordinate_source=input_data,
        restart_source=raw_restart,
        ensemble="nvt",
        wfn_source=raw_wfn,
    )
    stage = StageSpec(
        name="melt",
        input_data=input_data,
        output_data=tmp_path / "melt.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
        velocity_mode="preserve",
    )
    state = _load_cp2k_restart_state_for_stage(stage)
    assert state is not None and state.ensemble == "nvt"
    assert state.restart_file.name == "cp2k.restart"
    assert state.wfn_file is not None and state.wfn_file.name == "cp2k-RESTART.wfn"

    state.restart_file.write_text("tampered\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _load_cp2k_restart_state_for_stage(stage)


def test_cp2k_restart_bundle_is_bound_to_exact_coordinate_bytes(tmp_path: Path):
    from vitriflow.workflows.stage_runner import (
        _load_cp2k_restart_state_for_stage,
        _publish_cp2k_restart_state,
    )

    previous = tmp_path / "warmup"
    previous.mkdir()
    coordinates = previous / "warmup.data"
    coordinates.write_text("coordinates-v1\n")
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    _publish_cp2k_restart_state(
        previous,
        coordinate_source=coordinates,
        restart_source=raw_restart,
        ensemble="nvt",
    )
    stage = _cp2k_preserve_stage(previous, tmp_path / "melt.data")
    assert _load_cp2k_restart_state_for_stage(stage) is not None

    coordinates.write_text("coordinates-v2\n")
    with pytest.raises(RuntimeError, match="does not match its input coordinates"):
        _load_cp2k_restart_state_for_stage(stage)


def test_cp2k_restart_bundle_rejects_empty_wfn_before_publication(tmp_path: Path):
    from vitriflow.workflows.stage_runner import _publish_cp2k_restart_state

    previous = tmp_path / "warmup"
    previous.mkdir()
    coordinates = previous / "warmup.data"
    coordinates.write_text("coordinates\n")
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    empty_wfn = previous / "warmup-RESTART.wfn"
    empty_wfn.write_bytes(b"")

    with pytest.raises(RuntimeError, match="empty wavefunction restart"):
        _publish_cp2k_restart_state(
            previous,
            coordinate_source=coordinates,
            restart_source=raw_restart,
            ensemble="nvt",
            wfn_source=empty_wfn,
        )

    assert not (previous / "cp2k-RESTART.wfn").exists()
    assert not (previous / "cp2k.restart").exists()
    assert not (previous / "cp2k.restart.json").exists()


@pytest.mark.parametrize(
    "coordinate_name",
    ["cp2k.restart", "cp2k-RESTART.wfn", "cp2k.restart.json"],
)
def test_cp2k_restart_bundle_rejects_coordinate_namespace_collision(
    tmp_path: Path,
    coordinate_name: str,
):
    from vitriflow.workflows.stage_runner import _publish_cp2k_restart_state

    previous = tmp_path / coordinate_name.replace(".", "_")
    previous.mkdir()
    coordinates = previous / coordinate_name
    original_coordinates = b"coordinates\n"
    coordinates.write_bytes(original_coordinates)
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")

    with pytest.raises(RuntimeError, match="reserved restart-bundle name"):
        _publish_cp2k_restart_state(
            previous,
            coordinate_source=coordinates,
            restart_source=raw_restart,
            ensemble="nvt",
        )

    assert coordinates.read_bytes() == original_coordinates
    if coordinate_name != "cp2k.restart.json":
        assert not (previous / "cp2k.restart.json").exists()


def test_cp2k_restart_bundle_rejects_coordinate_change_during_publication(
    tmp_path: Path,
    monkeypatch,
):
    from vitriflow.workflows import stage_runner

    previous = tmp_path / "warmup"
    previous.mkdir()
    coordinates = previous / "warmup.data"
    coordinates.write_text("coordinates-v1\n")
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    original_publish = stage_runner._atomic_publish_cp2k_restart_artifact
    calls = 0

    def publish_then_mutate(*args, **kwargs):
        nonlocal calls
        result = original_publish(*args, **kwargs)
        calls += 1
        if calls == 1:
            coordinates.write_text("coordinates-v2\n")
        return result

    monkeypatch.setattr(
        stage_runner,
        "_atomic_publish_cp2k_restart_artifact",
        publish_then_mutate,
    )
    with pytest.raises(RuntimeError, match="coordinates changed"):
        stage_runner._publish_cp2k_restart_state(
            previous,
            coordinate_source=coordinates,
            restart_source=raw_restart,
            ensemble="nvt",
        )

    assert not (previous / "cp2k.restart.json").exists()


@pytest.mark.parametrize("unsafe_name", [" melt", "melt ", "melt\t"])
def test_stage_spec_rejects_unpreserved_whitespace_in_stage_name(
    tmp_path: Path,
    unsafe_name: str,
):
    from vitriflow.lammps_input import StageSpec

    with pytest.raises(ValueError, match="not path-safe"):
        StageSpec(
            name=unsafe_name,
            input_data=tmp_path / "input.data",
            output_data=tmp_path / "output.data",
            temperature_start=300.0,
            temperature_stop=300.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=1,
            seed=1,
        )


def _cp2k_preserve_stage(previous: Path, output: Path):
    from vitriflow.lammps_input import StageSpec

    return StageSpec(
        name="melt",
        input_data=previous / "warmup.data",
        output_data=output,
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
        velocity_mode="preserve",
    )


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_cp2k_restart_bundle_publication_replaces_aliases_without_clobbering_targets(
    tmp_path: Path,
    alias_kind: str,
):
    from vitriflow.workflows.stage_runner import (
        _load_cp2k_restart_state_for_stage,
        _publish_cp2k_restart_state,
    )

    previous = tmp_path / alias_kind / "warmup"
    previous.mkdir(parents=True)
    coordinates = previous / "warmup.data"
    coordinates.write_text("coordinates\n")
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    raw_wfn = previous / "warmup-RESTART.wfn"
    raw_wfn.write_bytes(b"wavefunction\n")

    victims: dict[str, Path] = {}
    for name in ("cp2k.restart", "cp2k-RESTART.wfn", "cp2k.restart.json"):
        victim = tmp_path / alias_kind / f"victim-{name}"
        victim.write_bytes(f"outside {name}\n".encode())
        victims[name] = victim
        destination = previous / name
        try:
            if alias_kind == "symlink":
                destination.symlink_to(victim)
            else:
                destination.hardlink_to(victim)
        except OSError as exc:  # pragma: no cover - filesystem policy
            pytest.skip(f"{alias_kind} creation unavailable: {exc}")

    before = {name: path.read_bytes() for name, path in victims.items()}
    _publish_cp2k_restart_state(
        previous,
        coordinate_source=coordinates,
        restart_source=raw_restart,
        ensemble="nvt",
        wfn_source=raw_wfn,
    )

    assert {name: path.read_bytes() for name, path in victims.items()} == before
    for name in victims:
        published = previous / name
        assert published.is_file() and not published.is_symlink()
        assert published.stat().st_nlink == 1
    state = _load_cp2k_restart_state_for_stage(
        _cp2k_preserve_stage(previous, tmp_path / alias_kind / "melt.data")
    )
    assert state is not None and state.wfn_file is not None


@pytest.mark.parametrize(
    ("artifact_name", "alias_kind"),
    [
        (name, alias)
        for name in ("cp2k.restart", "cp2k-RESTART.wfn", "cp2k.restart.json")
        for alias in ("symlink", "hardlink")
    ],
)
def test_cp2k_restart_bundle_loader_rejects_artifact_aliases(
    tmp_path: Path,
    artifact_name: str,
    alias_kind: str,
):
    from vitriflow.workflows.stage_runner import (
        _load_cp2k_restart_state_for_stage,
        _publish_cp2k_restart_state,
    )

    previous = tmp_path / f"{artifact_name}-{alias_kind}" / "warmup"
    previous.mkdir(parents=True)
    coordinates = previous / "warmup.data"
    coordinates.write_text("coordinates\n")
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    raw_wfn = previous / "warmup-RESTART.wfn"
    raw_wfn.write_bytes(b"wavefunction\n")
    _publish_cp2k_restart_state(
        previous,
        coordinate_source=coordinates,
        restart_source=raw_restart,
        ensemble="nvt",
        wfn_source=raw_wfn,
    )

    artifact = previous / artifact_name
    original = artifact.read_bytes()
    victim = tmp_path / f"victim-{artifact_name}-{alias_kind}"
    victim.write_bytes(original)
    artifact.unlink()
    try:
        if alias_kind == "symlink":
            artifact.symlink_to(victim)
        else:
            artifact.hardlink_to(victim)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"{alias_kind} creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="symbolic link|unique regular file"):
        _load_cp2k_restart_state_for_stage(
            _cp2k_preserve_stage(previous, tmp_path / "melt.data")
        )
    assert victim.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("restart_file", "../outside.restart"),
        ("restart_file", "/tmp/outside.restart"),
        ("wfn_restart_file", "../outside.wfn"),
        ("wfn_restart_file", "/tmp/outside.wfn"),
        ("coordinates_file", "../outside.data"),
        ("coordinates_file", "/tmp/outside.data"),
    ],
)
def test_cp2k_restart_bundle_metadata_rejects_path_escape(
    tmp_path: Path,
    field: str,
    unsafe_value: str,
):
    import json

    from vitriflow.workflows.stage_runner import (
        _load_cp2k_restart_state_for_stage,
        _publish_cp2k_restart_state,
    )

    previous = tmp_path / field / unsafe_value.replace("/", "_") / "warmup"
    previous.mkdir(parents=True)
    coordinates = previous / "warmup.data"
    coordinates.write_text("coordinates\n")
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    raw_wfn = previous / "warmup-RESTART.wfn"
    raw_wfn.write_bytes(b"wavefunction\n")
    _publish_cp2k_restart_state(
        previous,
        coordinate_source=coordinates,
        restart_source=raw_restart,
        ensemble="nvt",
        wfn_source=raw_wfn,
    )
    metadata_path = previous / "cp2k.restart.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = unsafe_value
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="exact|does not name"):
        _load_cp2k_restart_state_for_stage(
            _cp2k_preserve_stage(previous, tmp_path / "melt.data")
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"coordinates_size_bytes": True}, "invalid coordinate size"),
        ({"wfn_restart_file": False}, "exact local WFN file"),
        ({"wfn_restart_file": 0}, "exact local WFN file"),
        ({"wfn_restart_file": ""}, "exact local WFN file"),
        (
            {"wfn_restart_file": None, "wfn_restart_sha256": ""},
            "WFN hash without a WFN file",
        ),
    ],
)
def test_cp2k_restart_bundle_metadata_rejects_ambiguous_scalar_types(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
):
    import json

    from vitriflow.workflows.stage_runner import (
        _load_cp2k_restart_state_for_stage,
        _publish_cp2k_restart_state,
    )

    previous = tmp_path / "warmup"
    previous.mkdir()
    coordinates = previous / "warmup.data"
    coordinates.write_text("coordinates\n")
    raw_restart = previous / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    raw_wfn = previous / "warmup-RESTART.wfn"
    raw_wfn.write_bytes(b"wavefunction\n")
    _publish_cp2k_restart_state(
        previous,
        coordinate_source=coordinates,
        restart_source=raw_restart,
        ensemble="nvt",
        wfn_source=raw_wfn,
    )
    metadata_path = previous / "cp2k.restart.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(updates)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match=message):
        _load_cp2k_restart_state_for_stage(
            _cp2k_preserve_stage(previous, tmp_path / "melt.data")
        )


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_cp2k_preserve_rejects_aliased_coordinate_input(
    tmp_path: Path,
    alias_kind: str,
):
    from vitriflow.workflows.stage_runner import (
        _load_cp2k_restart_state_for_stage,
        _publish_cp2k_restart_state,
    )

    external = tmp_path / "external"
    external.mkdir()
    coordinates = external / "warmup.data"
    coordinates.write_text("coordinates\n")
    raw_restart = external / "warmup-1.restart"
    raw_restart.write_text("restart\n")
    _publish_cp2k_restart_state(
        external,
        coordinate_source=coordinates,
        restart_source=raw_restart,
        ensemble="nvt",
    )
    local = tmp_path / "local"
    local.mkdir()
    aliased_input = local / "warmup.data"
    try:
        if alias_kind == "symlink":
            aliased_input.symlink_to(coordinates)
        else:
            aliased_input.hardlink_to(coordinates)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"{alias_kind} creation unavailable: {exc}")
    stage = _cp2k_preserve_stage(local, tmp_path / "melt.data")

    with pytest.raises(RuntimeError, match="symbolic link|unique regular file"):
        _load_cp2k_restart_state_for_stage(stage)


def test_cp2k_preserve_rejects_coordinate_only_previous_stage(tmp_path: Path):
    from vitriflow.lammps_input import StageSpec
    from vitriflow.workflows.stage_runner import _load_cp2k_restart_state_for_stage

    previous = tmp_path / "warmup"
    previous.mkdir()
    input_data = previous / "warmup.data"
    input_data.write_text("coordinates only\n")
    stage = StageSpec(
        name="melt",
        input_data=input_data,
        output_data=tmp_path / "melt.data",
        temperature_start=1000.0,
        temperature_stop=1000.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=10,
        seed=123,
        velocity_mode="preserve",
    )
    with pytest.raises(RuntimeError, match="Coordinate-only continuation is not a valid CP2K MD restart"):
        _load_cp2k_restart_state_for_stage(stage)


def test_cp2k_renderer_uses_explicit_restart_state_without_invalid_barostat_restart():
    pytest.importorskip("ase")
    from ase import Atoms

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig, MDConfig
    from vitriflow.cp2k_driver import render_cp2k_md_input

    cfg = Cp2kConfig(
        kind_settings={"H": Cp2kKindConfig(basis_set="DZVP", potential="GTH-PBE")}
    )
    text = render_cp2k_md_input(
        atoms=Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        cfg=cfg,
        md_cfg=MDConfig(ensemble="npt"),
        ensemble="npt",
        temperature_K=300.0,
        steps=10,
        timestep_fs=1.0,
        tdamp_fs=100.0,
        pdamp_fs=1000.0,
        pressure_bar=0.0,
        project="next",
        energy_every=1,
        traj_every=1,
        traj_file="next.dcd",
        ener_file="next.ener",
        restart_file="/previous/cp2k.restart",
        restart_wfn_file="/previous/cp2k-RESTART.wfn",
        restart_barostat=False,
    )
    assert "RESTART_FILE_NAME /previous/cp2k.restart" in text
    assert "WFN_RESTART_FILE_NAME /previous/cp2k-RESTART.wfn" in text
    assert "SCF_GUESS RESTART" in text
    assert "RESTART_VEL T" in text
    assert "RESTART_BAROSTAT T" not in text


def test_cp2k_renderer_without_authenticated_wfn_forces_atomic_guess():
    pytest.importorskip("ase")
    from ase import Atoms

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig, MDConfig
    from vitriflow.cp2k_driver import render_cp2k_md_input

    cfg = Cp2kConfig(
        scf_guess="RESTART",
        kind_settings={"H": Cp2kKindConfig(basis_set="DZVP", potential="GTH-PBE")},
    )
    text = render_cp2k_md_input(
        atoms=Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        cfg=cfg,
        md_cfg=MDConfig(ensemble="nvt"),
        ensemble="nvt",
        temperature_K=300.0,
        steps=10,
        timestep_fs=1.0,
        tdamp_fs=100.0,
        project="fresh",
        energy_every=1,
        traj_every=1,
        traj_file="fresh.dcd",
        ener_file="fresh.ener",
        restart_file=None,
        restart_wfn_file=None,
    )
    assert "WFN_RESTART_FILE_NAME" not in text
    assert "SCF_GUESS ATOMIC" in text
    assert "SCF_GUESS RESTART" not in text
