from __future__ import annotations

from pathlib import Path

import pytest


def _minimal_cp2k_cfg(tmp_path: Path) -> dict:
    data = tmp_path / "dummy.data"
    data.write_text("\n")
    return {
        "engine": "cp2k",
        "cp2k": {},
        "structure": {"lammps_data": str(data)},
    }


def test_force_isotropic_rejected_for_cp2k_engine(tmp_path: Path):
    from vitriflow.config import RunConfig

    cfg = _minimal_cp2k_cfg(tmp_path)
    cfg["md"] = {"force_isotropic": True}

    with pytest.raises(ValueError, match=r"force_isotropic"):
        RunConfig.model_validate(cfg)


def test_explicit_elastic_enabled_rejected_for_cp2k_engine(tmp_path: Path):
    from vitriflow.config import RunConfig

    cfg = _minimal_cp2k_cfg(tmp_path)
    cfg["autotune"] = {"metrics": {"elastic": {"enabled": True}}}

    with pytest.raises(ValueError, match=r"elastic\.enabled=true.*engine=\'lammps\'"):
        RunConfig.model_validate(cfg)


def test_stage_metric_collection_defaults_enabled_when_metrics_enabled():
    from vitriflow.config import StructureMetricsConfig
    from vitriflow.workflows.stage_metrics import should_collect_stage_metrics_timeseries

    cfg = StructureMetricsConfig(enabled=True, voids={"enabled": True})
    assert cfg.collect_during_production_stages is True
    assert should_collect_stage_metrics_timeseries(cfg) is True


def test_elastic_stage_collection_defaults_enabled_for_lammps_roles():
    from vitriflow.config import LammpsConfig, StructureMetricsConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.elastic_screen import should_collect_elastic_stage_timeseries

    cfg = StructureMetricsConfig(enabled=True, voids={"enabled": True})
    runner = LammpsRunner(LammpsConfig(lammps_cmd="lmp"))

    for role in ("melt", "quench", "relax"):
        run, strict, _ecfg = should_collect_elastic_stage_timeseries(
            cfg,
            runner=runner,
            stage_role=role,
            force_isotropic=True,
        )
        assert run is True
        assert strict is True
