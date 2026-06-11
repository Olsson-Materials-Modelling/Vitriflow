from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ase")


def _production_kwargs(tmp_path: Path) -> dict:
    base = tmp_path / "base.data"
    base.write_text("# placeholder\n")
    return {
        "config": SimpleNamespace(
            autotune=SimpleNamespace(
                production=SimpleNamespace(enabled=True),
                convergence=SimpleNamespace(),
            ),
        ),
        "outdir": tmp_path,
        "runner": object(),
        "pot_cfg": SimpleNamespace(),
        "md_use": SimpleNamespace(),
        "potential_lines": ["pair_style mock"],
        "type_to_species": ["Al"],
        "metrics_cfg": SimpleNamespace(enabled=True),
        "tm_cfg": SimpleNamespace(msd_every=100),
        "q_cfg": SimpleNamespace(t_final=300.0, relax_steps=1000),
        "size_base_data": base,
        "chosen_replicate": [1, 1, 1],
        "chosen_rate": 10.0,
        "dt_ref": 0.001,
        "dt_mq": 0.001,
        "cooling_rate_ps": 10.0,
        "cutoffs_rate": {},
        "cutoffs_size": {},
        "T_high": 1200.0,
        "high_total_steps": 5000,
        "resume_state": None,
        "sampling_hint": None,
        "progress": None,
        "checkpoint_cb": None,
        "pressure_override": None,
        "seed_base": None,
        "prod_cfg_override": None,
        "conv_cfg_override": None,
        "quench_steps_override": None,
        "relax_steps_override": None,
    }


def test_autotune_wrapper_delegates_to_stateful_workflow(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import autotune as autotune_mod

    sentinel = {"status": "ok"}

    def _fake_run(self):
        assert self.outdir == tmp_path
        assert self.resume is False
        return sentinel

    monkeypatch.setattr(autotune_mod._AutotuneWorkflow, "run", _fake_run)

    out = autotune_mod.autotune(object(), tmp_path, resume=False)
    assert out is sentinel



def test_run_and_autotune_share_production_runner(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import autotune as autotune_mod
    from vitriflow.workflows import run as run_mod

    calls: list[dict] = []

    def _fake_run(self):
        calls.append(dict(self.__dict__))
        return {"status": "ok", "replicate": list(self.chosen_replicate)}

    monkeypatch.setattr(autotune_mod._ProductionEnsembleRunner, "run", _fake_run)

    kwargs = _production_kwargs(tmp_path)
    out1 = autotune_mod._run_production_ensemble(**kwargs)
    out2 = run_mod._run_production_executor(**kwargs)

    assert out1 == {"status": "ok", "replicate": [1, 1, 1]}
    assert out2 == {"status": "ok", "replicate": [1, 1, 1]}
    assert len(calls) == 2
    assert calls[0]["chosen_rate"] == pytest.approx(calls[1]["chosen_rate"])
    assert calls[0]["chosen_replicate"] == calls[1]["chosen_replicate"] == [1, 1, 1]
