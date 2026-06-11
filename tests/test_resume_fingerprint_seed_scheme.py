"""Resume fingerprint must record the seed-derivation scheme.

Targeted regression for ultrareview finding #2. The custom-schedule runner
switched per-box seeds from a stateful rng cursor to a deterministic
SHA-256 derivation. The resume fingerprint now carries:

  * a bumped schema id  (`vitriflow.custom_schedule.resume_fingerprint.v2`)
  * a `seed_scheme` payload field  (`sha256_box_slot_v1`)

A fingerprint built without those tags must not be accepted as equivalent;
that contract is what protects users from silently mixing boxes generated
under different seed algorithms when they resume against an old output dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _write_fingerprint_demo_config(tmp_path: Path, *, xml_text: str = "<GAP></GAP>") -> Path:
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir()
    xml_path = pot_dir / "gap_test.xml"
    xml_path.write_text(xml_text)
    sidecar = pot_dir / "gap_test.xml.sparseX.TEST"
    sidecar.write_text("sparse data\n")
    cfg = {
        "engine": "lammps",
        "random_seed": 2468,
        "lammps": {"lammps_cmd": "lmp", "nprocs": 1},
        "potential": {
            "kind": "lammps",
            "user_units": "metal",
            "interactions": ["C"],
            "files": [str(xml_path), str(sidecar)],
            "commands": [
                "pair_style quip",
                'pair_coeff * * gap_test.xml "Potential xml_label=GAP_TEST" 6',
            ],
        },
        "md": {
            "timestep": 0.001,
            "atom_style": "atomic",
            "ensemble": "nvt",
            "temperature": 300.0,
            "pressure": 0.0,
            "stage_continuity": "continuous",
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
                "seed": 123,
            }
        },
        "custom_schedule": {
            "workflow_label": "fingerprint_test",
            "stages": [
                {"name": "melt", "temperature_K": 1000.0, "steps": 2, "role": "melt", "velocity_mode": "create"},
                {"name": "quench", "temperature_start_K": 1000.0, "temperature_stop_K": 300.0, "steps": 2, "role": "quench"},
                {"name": "relax", "temperature_K": 300.0, "steps": 2, "role": "relax"},
            ],
            "analysis_roles": {"melt": "melt", "quench": "quench", "relax": "relax"},
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
            "production": {
                "enabled": True,
                "min_boxes": 1,
                "max_boxes": 1,
                "batch_boxes": 1,
                "check_convergence": False,
                "store_distributions": True,
            },
            "convergence": {"mode": "both", "familywise": "none"},
        },
    }
    cfg_path = tmp_path / "fingerprint_demo.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _fingerprint_from_config_path(cfg_path: Path) -> dict:
    from vitriflow.config import RunConfig
    from vitriflow.workflows.custom_schedule import (
        _build_resume_fingerprint,
        _schedule_from_raw,
        _schedule_report,
        _schedule_steps,
        _validate_schedule,
    )

    cfg = RunConfig.from_yaml(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    schedule = _schedule_from_raw(raw)
    roles = _validate_schedule(schedule)
    steps = _schedule_steps(schedule, md_use=cfg.md, time_unit_ps=1.0)
    report = _schedule_report(schedule, steps, md_use=cfg.md, time_unit_ps=1.0, analysis_roles=roles)
    return _build_resume_fingerprint(
        config=cfg,
        schedule=schedule,
        analysis_roles=roles,
        steps=steps,
        sched_report=report,
        time_unit_ps=1.0,
        md_pressure=0.0,
        lammps_units="metal",
        config_path=cfg_path,
    )


def test_fingerprint_schema_is_v2():
    """The schema constant has been bumped past v1."""
    from vitriflow.workflows.custom_schedule import _RESUME_FINGERPRINT_SCHEMA

    assert _RESUME_FINGERPRINT_SCHEMA == "vitriflow.custom_schedule.resume_fingerprint.v2"


def test_seed_scheme_constant_is_versioned():
    """The seed scheme tag must be explicit and versioned."""
    from vitriflow.workflows.custom_schedule import _SEED_SCHEME

    assert _SEED_SCHEME == "sha256_box_slot_v1"


def test_fingerprint_payload_carries_seed_scheme(tmp_path: Path):
    cfg_path = _write_fingerprint_demo_config(tmp_path)
    fp = _fingerprint_from_config_path(cfg_path)

    payload = fp["payload"]
    assert payload["seed_scheme"] == "sha256_box_slot_v1"
    # The schema field on both the wrapper and the payload should advertise v2
    # so anyone inspecting the sidecar can identify the run lineage.
    assert fp["schema"] == "vitriflow.custom_schedule.resume_fingerprint.v2"
    assert payload["schema"] == "vitriflow.custom_schedule.resume_fingerprint.v2"


def test_resume_rejects_pre_v2_fingerprint_without_seed_scheme(tmp_path: Path):
    """A stored fingerprint that lacks `seed_scheme` (i.e. pre-fix output)
    must be treated as a mismatch against the current runner. This is the
    defence that prevents resuming with the new seeding algorithm against
    boxes generated under the old cursor-based scheme.
    """
    from vitriflow.workflows.custom_schedule import (
        _sha256_canonical_json,
        _validate_resume_fingerprint_or_raise,
    )

    cfg_path = _write_fingerprint_demo_config(tmp_path)
    current = _fingerprint_from_config_path(cfg_path)

    # Synthesize a "legacy" v1 fingerprint by stripping seed_scheme and
    # downgrading the schema string, then recomputing the digest the same way
    # the production code does.
    legacy_payload = dict(current["payload"])
    legacy_payload.pop("seed_scheme", None)
    legacy_payload["schema"] = "vitriflow.custom_schedule.resume_fingerprint.v1"
    legacy = {
        "schema": "vitriflow.custom_schedule.resume_fingerprint.v1",
        "algorithm": current["algorithm"],
        "sha256": _sha256_canonical_json(legacy_payload),
        "payload": legacy_payload,
    }

    # Sanity: the digests must differ; otherwise the test is vacuous.
    assert legacy["sha256"] != current["sha256"]

    with pytest.raises(RuntimeError, match="fingerprint mismatch") as excinfo:
        _validate_resume_fingerprint_or_raise(
            {"resume_fingerprint": legacy}, current, outdir=tmp_path
        )

    # The diff message should surface seed_scheme as one of the differences so
    # operators see WHY resume was refused rather than guessing.
    msg = str(excinfo.value)
    assert "seed_scheme" in msg or "schema" in msg


def test_resume_rejects_fingerprint_with_different_seed_scheme(tmp_path: Path):
    """Even if a future runner chooses a v2 fingerprint envelope, switching
    the inner `seed_scheme` tag must invalidate resume."""
    from vitriflow.workflows.custom_schedule import (
        _sha256_canonical_json,
        _validate_resume_fingerprint_or_raise,
    )

    cfg_path = _write_fingerprint_demo_config(tmp_path)
    current = _fingerprint_from_config_path(cfg_path)

    other_payload = dict(current["payload"])
    other_payload["seed_scheme"] = "sha256_box_slot_v2_hypothetical"
    other = {
        "schema": current["schema"],
        "algorithm": current["algorithm"],
        "sha256": _sha256_canonical_json(other_payload),
        "payload": other_payload,
    }

    assert other["sha256"] != current["sha256"]

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _validate_resume_fingerprint_or_raise(
            {"resume_fingerprint": other}, current, outdir=tmp_path
        )
