"""potential.files null entries must raise, not silently drop.

Targeted regression for review finding #5. PyYAML parses ``null``,
``~``, and trailing ``-`` (with no value) into Python ``None``. The
earlier from_yaml fix dropped those entries silently while normalising
paths, leaving the loaded config diverging from the YAML the user wrote.
``RunConfig.from_yaml`` now raises ``ValueError`` for any null entry in
``potential.files`` (or its ``kim`` alias), naming the offending index so
the user can locate the typo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _base_config_dict(potential_files: list) -> dict:
    return {
        "engine": "lammps",
        "random_seed": 1,
        "lammps": {"lammps_cmd": "lmp", "nprocs": 1},
        "potential": {
            "kind": "lammps",
            "user_units": "metal",
            "interactions": ["C"],
            "files": potential_files,
            "commands": ["pair_style hybrid/overlay quip"],
        },
        "md": {
            "timestep": 0.001,
            "atom_style": "atomic",
            "ensemble": "nvt",
            "temperature": 300.0,
            "pressure": 0.0,
            "thermostat": {"style": "nose-hoover", "tdamp": 0.1},
            "barostat": {"style": "nose-hoover", "pdamp": 1.0},
        },
        "structure": {
            "generate": {
                "method": "random",
                "formula": "C",
                "n_formula_units": 2,
                "random_fallback_density_g_cm3": 2.0,
                "random_min_distance": 1.0,
                "seed": 1,
            }
        },
        "autotune": {
            "preflight": {"enabled": False},
            "metrics": {
                "enabled": True,
                "type_to_species": ["C"],
                "elastic": {"enabled": False},
                "pairs": [{"pair": ["C", "C"], "cutoff": 1.85}],
                "voids": {"enabled": True, "default_radius": 1.7, "radii": {"C": 1.7}},
                "amorphous": {"enabled": False},
            },
        },
    }


def _write_cfg(tmp_path: Path, files_value) -> Path:
    cfg = _base_config_dict(files_value)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def test_files_with_single_null_raises_with_index_zero(tmp_path: Path):
    from vitriflow.config import RunConfig

    cfg_path = _write_cfg(tmp_path, [None])
    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_yaml(cfg_path)
    msg = str(excinfo.value)
    assert "potential.files" in msg
    assert "index 0" in msg
    assert "null" in msg.lower()


def test_files_with_null_in_the_middle_reports_correct_index(tmp_path: Path):
    """The index should match the YAML position so the user can find the typo."""
    from vitriflow.config import RunConfig

    pot_path = tmp_path / "real.xml"
    pot_path.write_text("<GAP></GAP>")
    other = tmp_path / "other.xml"
    other.write_text("<GAP></GAP>")

    cfg_path = _write_cfg(tmp_path, [str(pot_path), None, str(other)])
    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_yaml(cfg_path)
    assert "index 1" in str(excinfo.value)


def test_yaml_null_token_triggers_the_same_error(tmp_path: Path):
    """`files: [null]`, `files: [~]`, and a trailing `-` all parse to None."""
    from vitriflow.config import RunConfig

    # `files: [~]` -- explicit YAML null shorthand
    cfg_text = (
        "engine: lammps\n"
        "random_seed: 1\n"
        "lammps: {lammps_cmd: lmp, nprocs: 1}\n"
        "potential:\n"
        "  kind: lammps\n"
        "  user_units: metal\n"
        "  interactions: [C]\n"
        "  files:\n"
        "    - ~\n"
        "  commands: [pair_style quip]\n"
        "md: {timestep: 0.001, atom_style: atomic, ensemble: nvt,\n"
        "     temperature: 300.0, pressure: 0.0,\n"
        "     thermostat: {style: nose-hoover, tdamp: 0.1},\n"
        "     barostat: {style: nose-hoover, pdamp: 1.0}}\n"
        "structure:\n"
        "  generate:\n"
        "    method: random\n"
        "    formula: C\n"
        "    n_formula_units: 2\n"
        "    random_fallback_density_g_cm3: 2.0\n"
        "    random_min_distance: 1.0\n"
        "    seed: 1\n"
        "autotune:\n"
        "  preflight: {enabled: false}\n"
        "  metrics:\n"
        "    enabled: true\n"
        "    type_to_species: [C]\n"
        "    elastic: {enabled: false}\n"
        "    pairs: [{pair: [C, C], cutoff: 1.85}]\n"
        "    voids: {enabled: true, default_radius: 1.7, radii: {C: 1.7}}\n"
        "    amorphous: {enabled: false}\n"
    )
    cfg_path = tmp_path / "tilde.yaml"
    cfg_path.write_text(cfg_text)
    with pytest.raises(ValueError, match="potential.files"):
        RunConfig.from_yaml(cfg_path)


def test_kim_alias_files_null_also_raises(tmp_path: Path):
    """The ``kim:`` block is an alias for ``potential:``; null entries must
    fail the same way under either spelling. The error message should name
    the actual key used, so the user looks at the right block."""
    from vitriflow.config import RunConfig

    cfg = _base_config_dict([None])
    # Move the potential block under the `kim:` alias key.
    cfg["kim"] = cfg.pop("potential")
    # KIM blocks default to kind="kim" + a model field; switch shape so
    # the union resolves to KimConfig.
    cfg["kim"] = {
        "kind": "kim",
        "model": "TEST_MODEL",
        "interactions": ["C"],
        "files": [None],
    }
    cfg_path = tmp_path / "kim_null.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_yaml(cfg_path)
    msg = str(excinfo.value)
    # Either spelling is acceptable, but the message must name the actual
    # YAML key so the user looks at the right block.
    assert "kim.files" in msg
    assert "index 0" in msg


def test_valid_files_list_is_unaffected(tmp_path: Path):
    """Sanity: legitimate file lists still parse fine -- we did not narrow
    behavior beyond the null-rejection contract."""
    from vitriflow.config import RunConfig

    a = tmp_path / "a.xml"
    a.write_text("<GAP></GAP>")
    b = tmp_path / "b.xml"
    b.write_text("<GAP></GAP>")
    cfg_path = _write_cfg(tmp_path, [str(a), str(b)])
    cfg = RunConfig.from_yaml(cfg_path)
    assert cfg.kim is not None
    files = list(getattr(cfg.kim, "files", []) or [])
    assert len(files) == 2
    assert {Path(p).name for p in files} == {"a.xml", "b.xml"}


def test_empty_files_list_is_unaffected(tmp_path: Path):
    """Empty lists are not the same as null entries; they must still parse."""
    from vitriflow.config import RunConfig

    cfg_path = _write_cfg(tmp_path, [])
    cfg = RunConfig.from_yaml(cfg_path)
    assert cfg.kim is not None
    files = list(getattr(cfg.kim, "files", []) or [])
    assert files == []


def test_path_rewrite_has_no_exception_swallow_and_null_still_propagates(tmp_path: Path):
    """Path ambiguity and malformed entries must never be caught and ignored."""
    import inspect

    from vitriflow.config import RunConfig

    src = inspect.getsource(RunConfig.from_yaml)
    assert "_PATH_REWRITE_EXC" not in src

    cfg_path = _write_cfg(tmp_path, [None])
    with pytest.raises(ValueError, match=r"potential\.files.*index 0.*null"):
        RunConfig.from_yaml(cfg_path)
