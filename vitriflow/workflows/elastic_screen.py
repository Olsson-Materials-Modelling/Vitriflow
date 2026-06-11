from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..analysis.datafile import (
    count_atoms_in_datafile,
    read_datafile_charges,
    read_datafile_frame,
    read_datafile_masses,
    strip_lammps_data_pair_coeff_sections,
)
from ..analysis.elastic import (
    build_elastic_screen_summary,
    hydrostatic_from_stress,
    load_elastic_screen_summary,
    parse_born_stress_raw,
    pressure_unit_label,
    read_single_custom_dump,
    stress_volume_to_pressure_like,
    von_mises_from_stress,
    write_born_matrix_csv,
    write_local_stress_csv,
)
from ..analysis.trajectory import read_frames_auto, stage_trajectory_path, select_stage_frames, quench_window_steps
from ..io.lammps_data_minimal import write_dumpframe_lammps_data
from ..io.thermo import parse_thermo_csv
from .step_counts import recommended_quench_dump_every
from ..lammps_input import render_elastic_screen
from ..potential import prepare_potential_files
from ..runner import LammpsRunner
from ..utils import ExternalCommandError, ensure_dir


def _relpath_or_str(path: Path, base: Path) -> str:
    p = Path(path)
    b = Path(base)
    try:
        return str(p.relative_to(b))
    except Exception:
        return str(p)


def _elastic_cfg(metrics_cfg) -> Any:
    try:
        return getattr(metrics_cfg, "elastic")
    except Exception:
        return None


def should_run_elastic_screen(
    metrics_cfg,
    *,
    runner,
    stage_role: str,
    force_isotropic: bool,
) -> tuple[bool, bool, Any]:
    """Run elastic screen."""

    cfg = _elastic_cfg(metrics_cfg)
    if cfg is None:
        return False, False, cfg
    enabled = getattr(cfg, "enabled", "auto")
    if not isinstance(runner, LammpsRunner):
        if enabled is True:
            raise ValueError("Elastic analysis is supported only for engine='lammps'")
        return False, False, cfg
    if enabled is False:
        return False, False, cfg

    role = str(stage_role or "").strip().lower()
    run = False
    if role == "relax" and bool(getattr(cfg, "run_on_relax", True)):
        run = True
    if force_isotropic and role in {"melt", "hight", "high_t", "high-t", "disorder", "hight_disorder"}:
        if bool(getattr(cfg, "run_on_highT_when_force_isotropic", True)):
            run = True

    strict = bool(run and force_isotropic and bool(getattr(cfg, "strict_when_force_isotropic", True)))
    return bool(run), bool(strict), cfg


def should_collect_elastic_stage_timeseries(
    metrics_cfg,
    *,
    runner,
    stage_role: str,
    force_isotropic: bool,
) -> tuple[bool, bool, Any]:
    """Collect elastic stage."""

    cfg = _elastic_cfg(metrics_cfg)
    if cfg is None:
        return False, False, cfg
    enabled = getattr(cfg, "enabled", "auto")
    if not isinstance(runner, LammpsRunner):
        if enabled is True:
            raise ValueError("Elastic analysis is supported only for engine='lammps'")
        return False, False, cfg
    if enabled is False:
        return False, False, cfg
    if not bool(getattr(cfg, "collect_during_production_stages", True)):
        return False, False, cfg

    role = str(stage_role or "").strip().lower()
    run = role in {"melt", "quench", "relax"}
    strict = bool(run and force_isotropic and bool(getattr(cfg, "strict_when_force_isotropic", True)))
    return bool(run), bool(strict), cfg


def _elastic_cfg_numbers(metrics_cfg) -> tuple[float, bool, float, float, float]:
    escfg = _elastic_cfg(metrics_cfg)
    born_delta = 1.0e-5
    make_plot = True
    isotropy_warn_threshold = 0.15
    coupling_warn_threshold = 0.10
    hotspot_warn_multiple = 5.0
    if escfg is not None:
        try:
            born_delta = float(getattr(escfg, "born_delta", born_delta))
        except Exception:
            pass
        try:
            make_plot = bool(getattr(escfg, "make_plot", make_plot))
        except Exception:
            pass
        try:
            isotropy_warn_threshold = float(getattr(escfg, "isotropy_warn_threshold", isotropy_warn_threshold))
        except Exception:
            pass
        try:
            coupling_warn_threshold = float(getattr(escfg, "coupling_warn_threshold", coupling_warn_threshold))
        except Exception:
            pass
        try:
            hotspot_warn_multiple = float(getattr(escfg, "hotspot_warn_multiple_of_median", hotspot_warn_multiple))
        except Exception:
            pass
    return (
        float(born_delta),
        bool(make_plot),
        float(isotropy_warn_threshold),
        float(coupling_warn_threshold),
        float(hotspot_warn_multiple),
    )


def estimate_diffusion_freeze_temperature(
    temperatures,
    diffusion,
    *,
    Tm: Optional[float],
    time_unit_ps: Optional[float],
    threshold_A2_per_ps: float = 0.1,
) -> float:
    """Diffusion freeze temperature."""

    T = np.asarray(temperatures, dtype=float)
    D = np.asarray(diffusion, dtype=float)
    if T.shape != D.shape:
        raise ValueError("temperatures and diffusion must have the same shape")
    m = np.isfinite(T) & np.isfinite(D)
    if int(np.sum(m)) == 0:
        return float("nan")
    T = T[m]
    D = D[m]
    if time_unit_ps is not None and math.isfinite(float(time_unit_ps)) and float(time_unit_ps) > 0.0:
        D = D / float(time_unit_ps)
    D = np.maximum(D, 0.0)
    order = np.argsort(T)[::-1]
    T = T[order]
    D = D[order]
    if Tm is not None and math.isfinite(float(Tm)):
        keep = T <= float(Tm)
        if np.any(keep):
            T = T[keep]
            D = D[keep]
    if T.size == 0:
        return float("nan")
    threshold = max(float(threshold_A2_per_ps), 0.0)
    below = D <= threshold
    for i, temp in enumerate(T.tolist()):
        if bool(np.all(below[i:])):
            return float(temp)
    return float(np.min(T))


def build_elastic_sampling_hint(
    *,
    Tm: Optional[float],
    freeze_temperature: Optional[float],
    threshold_A2_per_ps: float = 0.1,
) -> Optional[dict[str, float]]:
    if Tm is None or freeze_temperature is None:
        return None
    if not (math.isfinite(float(Tm)) and math.isfinite(float(freeze_temperature))):
        return None
    return {
        "Tm": float(Tm),
        "freeze_temperature": float(freeze_temperature),
        "threshold_A2_per_ps": float(max(threshold_A2_per_ps, 0.0)),
    }


# quench dump module



def _interp_thermo_temp(stage_dir: Path, steps: np.ndarray) -> Optional[np.ndarray]:
    thermo_path = Path(stage_dir) / "thermo.csv"
    if not thermo_path.exists():
        return None
    try:
        tt = parse_thermo_csv(thermo_path)
    except Exception:
        return None
    cols = list(tt.columns)
    if "Step" not in cols or "Temp" not in cols:
        return None
    data = np.asarray(tt.data, dtype=float)
    x = np.asarray(data[:, cols.index("Step")], dtype=float)
    y = np.asarray(data[:, cols.index("Temp")], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 2:
        return None
    return np.interp(steps, x[m], y[m], left=np.nan, right=np.nan)


def _select_elastic_frames(
    frames_all,
    *,
    stage_dir: Path,
    frame_stride: int,
    max_frames: int,
    stage_role: str,
    sampling_hint: Optional[dict[str, float]],
):
    temps = None
    steps = np.asarray([float(fr.timestep) for fr in frames_all], dtype=float)
    interp_t = _interp_thermo_temp(Path(stage_dir), steps)
    if interp_t is not None:
        temps = np.asarray(interp_t, dtype=float)[:: max(1, int(frame_stride))]
    q_window = None
    tm = None
    tfreeze = None
    if sampling_hint is not None:
        tm = sampling_hint.get("Tm")
        tfreeze = sampling_hint.get("freeze_temperature")
        q_window = quench_window_steps(
            T_start=float(np.nanmax(temps)) if (temps is not None and np.isfinite(temps).any()) else 1.0,
            T_stop=float(np.nanmin(temps)) if (temps is not None and np.isfinite(temps).any()) else 0.0,
            total_steps=int(max(float(steps[-1] - steps[0]), 1.0)),
            T_upper=tm,
            T_lower=tfreeze,
        )
        if q_window is not None:
            q_window = (float(steps[0] + q_window[0]), float(steps[0] + q_window[1]))
    frames, meta = select_stage_frames(
        frames_all,
        frame_stride=int(frame_stride),
        max_frames=int(max_frames),
        stage_role=str(stage_role),
        quench_window_steps_range=q_window,
        temperatures=temps,
        tm_temperature=(float(tm) if tm is not None else None),
        diffusion_freeze_temperature=(float(tfreeze) if tfreeze is not None else None),
        quench_tail_fraction=0.75,
        quench_tail_min_frames=max(8, int(max_frames // 2) if int(max_frames) > 1 else 1),
        quench_tail_fallback_fraction=0.40,
    )
    return frames, meta


def run_elastic_screen_lammps(
    runner: LammpsRunner,
    pot_cfg,
    md_cfg,
    *,
    structure_data: Path,
    stage_dir: Path,
    potential_lines: Optional[list[str]] = None,
    metrics_cfg=None,
    force_isotropic: bool = False,
    input_data_for_affine_strain: Optional[Path] = None,
    outdir: Optional[Path] = None,
    make_plot_override: Optional[bool] = None,
) -> dict[str, Any]:
    """Elastic screen lammps."""

    born_delta, make_plot, isotropy_warn_threshold, coupling_warn_threshold, hotspot_warn_multiple = _elastic_cfg_numbers(metrics_cfg)
    if make_plot_override is not None:
        make_plot = bool(make_plot_override)

    stage_dir = Path(stage_dir)
    structure_data = Path(structure_data)
    elastic_dir = stage_dir / "elastic"
    ensure_dir(elastic_dir)

    input_local = elastic_dir / "input.data"
    input_local.write_bytes(structure_data.read_bytes())
    strip_lammps_data_pair_coeff_sections(input_local)

    prepare_potential_files(pot_cfg, elastic_dir, potential_lines)

    log_name = "log.lammps"
    raw_name = "born_raw.txt"
    dump_name = "local_stress.dump"
    summary_path = elastic_dir / "elastic_screen.json"
    born_csv = elastic_dir / "born_matrix.csv"
    stress_csv = elastic_dir / "local_stress.csv"
    plot_path = elastic_dir / "elastic_screen.png"

    summary: dict[str, Any]
    try:
        script = render_elastic_screen(
            pot_cfg,
            md_cfg,
            input_data=input_local,
            born_delta=float(born_delta),
            raw_output_name=raw_name,
            stress_dump_name=dump_name,
            potential_lines=potential_lines,
        )
        runner.run(script, elastic_dir, log_name=log_name)

        raw = parse_born_stress_raw(elastic_dir / raw_name)
        dump = read_single_custom_dump(elastic_dir / dump_name)
        cols = dump["data"]
        required = [
            "id", "type", "x", "y", "z",
            "c_pst[1]", "c_pst[2]", "c_pst[3]", "c_pst[4]", "c_pst[5]", "c_pst[6]",
        ]
        if not all(k in cols for k in required):
            raise ValueError("Elastic-screen dump missing one or more required columns")

        positions = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(float)
        types = np.asarray(cols["type"], dtype=int)
        ids = np.asarray(cols["id"], dtype=int)
        stress_volume = np.column_stack(
            [
                cols["c_pst[1]"],
                cols["c_pst[2]"],
                cols["c_pst[3]"],
                cols["c_pst[4]"],
                cols["c_pst[5]"],
                cols["c_pst[6]"],
            ]
        ).astype(float)

        volume = float(raw.get("volume", float("nan")))
        if not (math.isfinite(volume) and volume > 0.0):
            volume = float(dump.get("volume", float("nan")))
        n_atoms = int(count_atoms_in_datafile(input_local))
        if n_atoms < 1:
            n_atoms = int(positions.shape[0])

        input_cell = None
        if force_isotropic and input_data_for_affine_strain is not None:
            try:
                input_cell = read_datafile_frame(Path(input_data_for_affine_strain), atom_style=str(md_cfg.atom_style)).cell
            except Exception:
                input_cell = None

        units_style = ""
        try:
            units_style = str(getattr(pot_cfg, "user_units", "") or "")
        except Exception:
            units_style = ""

        summary = build_elastic_screen_summary(
            born21=np.asarray(raw["born21"], dtype=float),
            global_stress_voigt=np.asarray(raw["global_stress_voigt"], dtype=float),
            volume=float(volume),
            n_atoms=int(n_atoms),
            local_positions=positions,
            local_types=types,
            local_stress_volume=stress_volume,
            units_style=units_style,
            isotropy_warn_threshold=float(isotropy_warn_threshold),
            coupling_warn_threshold=float(coupling_warn_threshold),
            hotspot_warn_multiple=float(hotspot_warn_multiple),
            force_isotropic=bool(force_isotropic),
            input_cell=input_cell,
        )
        summary["paths"] = {
            "born_csv": str(born_csv.name),
            "local_stress_csv": str(stress_csv.name),
            "raw": str(raw_name),
            "dump": str(dump_name),
            "plot": str(plot_path.name),
        }

        pressure_label = str(summary.get("units", {}).get("pressure_native") or pressure_unit_label(units_style))
        write_born_matrix_csv(
            born_csv,
            np.asarray(summary["born_matrix_native"], dtype=float),
            units_label=pressure_label,
        )
        stress_native = stress_volume_to_pressure_like(stress_volume, volume=float(volume), n_atoms=int(n_atoms))
        write_local_stress_csv(
            stress_csv,
            ids=ids,
            types=types,
            positions=positions,
            stress_volume=stress_volume,
            stress_native=stress_native,
            hydrostatic_native=hydrostatic_from_stress(stress_native),
            von_mises_native=von_mises_from_stress(stress_native),
        )
    except Exception as exc:
        summary = {
            "status": "failed",
            "error": str(exc),
            "force_isotropic": bool(force_isotropic),
        }
        if isinstance(exc, ExternalCommandError):
            summary["returncode"] = int(exc.returncode)
            summary["cmd"] = list(exc.cmd)
            if exc.context is not None:
                summary["screen_tail"] = str(exc.context.screen_tail)
                summary["log_tail"] = str(exc.context.log_tail)
                summary["stderr_tail"] = str(exc.context.stderr_tail)
                summary["stdout_tail"] = str(exc.context.stdout_tail)

    summary_path.write_text(json.dumps(summary, indent=2))

    if summary.get("status") == "ok" and make_plot:
        try:
            from ..plotting import plot_elastic_screen

            plot_elastic_screen(elastic_dir, plot_path)
        except Exception:
            pass

    base = outdir if outdir is not None else stage_dir.parent
    return {
        "dir": _relpath_or_str(elastic_dir, base),
        "summary": _relpath_or_str(summary_path, base),
        "plot": _relpath_or_str(plot_path, base) if plot_path.exists() else None,
        "status": str(summary.get("status", "unknown")),
        "flags": list(summary.get("flags", [])) if isinstance(summary.get("flags", []), list) else [],
    }


def _write_elastic_timeseries_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    hdr = [
        "Step",
        "time",
        "status_ok",
        "flag_count",
        "isotropy_residual",
        "normal_shear_coupling_norm",
        "voigt_bulk_modulus_native",
        "voigt_shear_modulus_native",
        "local_vm_p95",
        "local_vm_max_over_median",
    ]
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for row in rows:
            w.writerow([row.get(k, float("nan")) for k in hdr])


def _plot_elastic_timeseries(csv_path: Path, out_path: Path, *, title: str, dpi: int = 600) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from ..plotting import _apply_publication_style, _style_and_save_figure  # type: ignore

        _apply_publication_style()
    except Exception:
        _style_and_save_figure = None  # type: ignore

    dat = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    if dat.size == 0:
        raise ValueError(f"No rows in elastic timeseries CSV: {csv_path}")
    if dat.ndim == 0:
        dat = np.asarray([dat], dtype=dat.dtype)

    x = np.asarray(dat["time"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)

    ax = axes[0, 0]
    ax.plot(x, np.asarray(dat["isotropy_residual"], dtype=float))
    ax.set_ylabel("isotropy residual")

    ax = axes[0, 1]
    ax.plot(x, np.asarray(dat["normal_shear_coupling_norm"], dtype=float))
    ax.set_ylabel("normal-shear coupling")

    ax = axes[1, 0]
    ax.plot(x, np.asarray(dat["local_vm_max_over_median"], dtype=float))
    ax.set_ylabel("local vm max/median")
    ax.set_xlabel("time")

    ax = axes[1, 1]
    ax.plot(x, np.asarray(dat["voigt_bulk_modulus_native"], dtype=float), label="K")
    ax.plot(x, np.asarray(dat["voigt_shear_modulus_native"], dtype=float), label="G")
    ax.set_ylabel("Born moduli (native)")
    ax.set_xlabel("time")
    ax.legend()

    fig.suptitle(str(title))
    out_path = Path(out_path)
    if _style_and_save_figure is not None:  # type: ignore[name-defined]
        _style_and_save_figure(fig, out_path, dpi=int(dpi))  # type: ignore[misc]
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=int(dpi))
        plt.close(fig)


def run_elastic_screen_timeseries_lammps(
    runner: LammpsRunner,
    pot_cfg,
    md_cfg,
    *,
    stage_dir: Path,
    stage_output_data: Path,
    stage_role: str,
    potential_lines: Optional[list[str]] = None,
    metrics_cfg=None,
    force_isotropic: bool = False,
    outdir: Optional[Path] = None,
    sampling_hint: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Elastic screen timeseries."""

    escfg = _elastic_cfg(metrics_cfg)
    if escfg is None:
        raise ValueError("Elastic timeseries requested without an elastic configuration")

    frame_stride = int(getattr(escfg, "stage_timeseries_frame_stride", 1) or 1)
    max_frames = int(getattr(escfg, "stage_timeseries_max_frames", 8) or 8)
    make_plot = bool(getattr(escfg, "stage_timeseries_make_plot", True))

    stage_dir = Path(stage_dir)
    stage_output_data = Path(stage_output_data)
    elastic_ts_dir = stage_dir / "elastic_timeseries"
    ensure_dir(elastic_ts_dir)

    traj = stage_trajectory_path(stage_dir)
    if traj is None or not Path(traj).exists():
        raise FileNotFoundError(f"No trajectory found for elastic timeseries under {stage_dir}")

    frames_all = list(read_frames_auto(Path(traj)))
    if not frames_all:
        raise ValueError(f"No frames parsed from trajectory: {traj}")
    frames, selection_meta = _select_elastic_frames(
        frames_all,
        stage_dir=stage_dir,
        frame_stride=int(frame_stride),
        max_frames=int(max_frames),
        stage_role=str(stage_role),
        sampling_hint=sampling_hint,
    )
    if not frames:
        raise ValueError(f"No frames selected for elastic timeseries under {stage_dir}")

    masses_by_type = read_datafile_masses(stage_output_data)
    charges_by_id = read_datafile_charges(stage_output_data, atom_style=str(md_cfg.atom_style))

    rows: list[dict[str, Any]] = []
    frame_entries: list[dict[str, Any]] = []
    any_failed = False
    for i, fr in enumerate(frames):
        frame_dir = elastic_ts_dir / f"frame_{i:03d}_step_{int(fr.timestep):010d}"
        ensure_dir(frame_dir)
        frame_data = frame_dir / "input.data"
        write_dumpframe_lammps_data(
            frame_data,
            fr,
            atom_style=str(md_cfg.atom_style),
            masses_by_type=masses_by_type,
            charges_by_id=charges_by_id if str(md_cfg.atom_style).strip().lower() == "charge" else None,
        )
        res = run_elastic_screen_lammps(
            runner,
            pot_cfg,
            md_cfg,
            structure_data=frame_data,
            stage_dir=frame_dir,
            potential_lines=potential_lines,
            metrics_cfg=metrics_cfg,
            force_isotropic=bool(force_isotropic and str(stage_role).strip().lower() == "melt"),
            input_data_for_affine_strain=None,
            outdir=outdir,
            make_plot_override=False,
        )
        summary_path = frame_dir / "elastic" / "elastic_screen.json"
        summary = load_elastic_screen_summary(summary_path)
        status_ok = 1.0 if str(summary.get("status", "")) == "ok" else 0.0
        any_failed = any_failed or not bool(status_ok)
        vm_summary = ((summary.get("local_stress_summary", {}) or {}).get("von_mises_native", {}) or {})
        row = {
            "Step": float(fr.timestep),
            "time": float(fr.timestep) * float(md_cfg.timestep),
            "status_ok": float(status_ok),
            "flag_count": float(len(summary.get("flags", []) or [])),
            "isotropy_residual": float(summary.get("isotropy_residual", float("nan"))),
            "normal_shear_coupling_norm": float(summary.get("normal_shear_coupling_norm", float("nan"))),
            "voigt_bulk_modulus_native": float(summary.get("voigt_bulk_modulus_native", float("nan"))),
            "voigt_shear_modulus_native": float(summary.get("voigt_shear_modulus_native", float("nan"))),
            "local_vm_p95": float(vm_summary.get("p95", float("nan"))),
            "local_vm_max_over_median": float(vm_summary.get("max_over_median", float("nan"))),
        }
        rows.append(row)
        base = outdir if outdir is not None else stage_dir.parent
        frame_entries.append(
            {
                "step": int(fr.timestep),
                "dir": _relpath_or_str(frame_dir, base),
                "summary": _relpath_or_str(summary_path, base),
                "status": str(summary.get("status", "unknown")),
                "flags": list(summary.get("flags", [])) if isinstance(summary.get("flags", []), list) else [],
            }
        )

    csv_path = elastic_ts_dir / "elastic_timeseries.csv"
    _write_elastic_timeseries_csv(csv_path, rows)

    summary = {
        "status": "ok" if not any_failed else "degraded",
        "stage_role": str(stage_role),
        "n_frames": int(len(rows)),
        "frame_stride": int(frame_stride),
        "max_frames": int(max_frames),
        "sampling_hint": dict(sampling_hint or {}),
        "selection": dict(selection_meta or {}),
        "frames": frame_entries,
        "flags_union": sorted({str(f) for fr in frame_entries for f in (fr.get("flags") or [])}),
    }
    summary_path = elastic_ts_dir / "elastic_timeseries.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = elastic_ts_dir / "elastic_timeseries.png"
    if make_plot:
        try:
            _plot_elastic_timeseries(csv_path, plot_path, title=f"Elastic timeseries: {stage_dir.name}")
        except Exception:
            pass

    base = outdir if outdir is not None else stage_dir.parent
    return {
        "status": str(summary.get("status", "unknown")),
        "dir": _relpath_or_str(elastic_ts_dir, base),
        "csv": _relpath_or_str(csv_path, base),
        "summary": _relpath_or_str(summary_path, base),
        "plot": _relpath_or_str(plot_path, base) if plot_path.exists() else None,
        "n_frames": int(len(rows)),
    }


def load_elastic_screen_brief(path: Path) -> dict[str, Any]:
    summary = load_elastic_screen_summary(path)
    return {
        "status": str(summary.get("status", "unknown")),
        "flags": list(summary.get("flags", [])) if isinstance(summary.get("flags", []), list) else [],
        "summary": str(path),
    }
