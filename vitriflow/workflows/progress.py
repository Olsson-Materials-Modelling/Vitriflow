from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomic write json."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    txt = json.dumps(data, indent=2, sort_keys=False)
    tmp.write_text(txt)
    tmp.replace(p)


def make_autotune_compact(results: Mapping[str, Any]) -> dict[str, Any]:
    """Autotune compact."""

    rec = dict(results.get("recommendation", {}) or {})
    tm_scan = dict(results.get("tm_scan", {}) or {})
    tm_est = dict(tm_scan.get("Tm_estimate", {}) or {})
    prod = dict(results.get("production", {}) or {})
    crystal_motifs = dict(prod.get("crystal_motifs", {}) or {})
    crystal_motifs_compact = {
        "used": bool(crystal_motifs.get("used", False)),
        "n_boxes_total": crystal_motifs.get("n_boxes_total"),
        "motifs": [
            {
                "material_id": m.get("material_id"),
                "formula_pretty": m.get("formula_pretty"),
                "n_boxes_detected": m.get("n_boxes_detected"),
                "n_boxes_candidate": m.get("n_boxes_candidate"),
                "max_peak_overlap": m.get("max_peak_overlap"),
                "max_motif_score": m.get("max_motif_score"),
            }
            for m in list(crystal_motifs.get("motifs", []) or [])[:3]
            if isinstance(m, Mapping)
        ],
    }
    compact: dict[str, Any] = {
        "status": str(results.get("status", "ok")),
        "recommendation": rec,
        "tm_scan": {
            "Tm_estimate": tm_est,
            "summary": tm_scan.get("summary"),
            "plot": tm_scan.get("plot"),
        },
        "rate_scan": {
            "decision_density": (results.get("rate_scan", {}) or {}).get("decision_density"),
            "decision_multi": (results.get("rate_scan", {}) or {}).get("decision_multi"),
        },
        "size_scan": {
            "decision_density": (results.get("size_scan", {}) or {}).get("decision_density"),
            "decision_multi": (results.get("size_scan", {}) or {}).get("decision_multi"),
            "skipped": bool((results.get("size_scan", {}) or {}).get("skipped", False)),
            "reason": (results.get("size_scan", {}) or {}).get("reason"),
        },
        "production": {
            "enabled": bool(prod.get("enabled", False)),
            "converged": prod.get("converged"),
            "n_boxes": prod.get("n_boxes"),
            "n_boxes_total": prod.get("n_boxes_total"),
            "convergence": prod.get("convergence"),
            "rejected_boxes": prod.get("rejected_boxes"),
            "crystal_motifs": crystal_motifs_compact,
        },
        "metric_warnings": list(results.get("metric_warnings", []) or []),
        "effective_metrics": dict(results.get("effective_metrics", {}) or {}),
        "paths": dict(results.get("paths", {}) or {}),
    }
    paths = compact.setdefault("paths", {})
    if isinstance(paths, dict):
        paths.setdefault("autotune_results", "autotune_results.json")
        paths.setdefault("autotune", "autotune.json")
        paths.setdefault("condensed_log", "condensed.log")
    return compact


def write_autotune_outputs(outdir: Path, results: Mapping[str, Any]) -> None:
    """Autotune outputs."""

    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(d / "autotune_results.json", dict(results))
    atomic_write_json(d / "autotune.json", make_autotune_compact(results))


def summarise_convergence_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Convergence report."""

    out: dict[str, Any] = {
        "passed": bool(report.get("passed", report.get("converged", False))),
        "groups": {},
        "metrics": {},
        "failed_metrics": [],
    }

    groups = report.get("groups", {})
    if isinstance(groups, Mapping):
        out["groups"] = {str(k): bool(v) for k, v in groups.items()}

    metrics: dict[str, bool] = {}

    scalars = report.get("scalars", {})
    if isinstance(scalars, Mapping):
        for name, payload in scalars.items():
            ok = bool(payload.get("passed", False)) if isinstance(payload, Mapping) else False
            metrics[str(name)] = ok

    distributions = report.get("distributions", {})
    if isinstance(distributions, Mapping):
        for name, payload in distributions.items():
            ok = bool(payload.get("passed", False)) if isinstance(payload, Mapping) else False
            metrics[str(name)] = ok

    # multimetric decisions converged
    met = report.get("metrics", {})
    if isinstance(met, Mapping):
        for name, payload in met.items():
            if isinstance(payload, Mapping):
                passed = payload.get("passed", None)
                if isinstance(passed, list):
                    ok = bool(all(bool(x) for x in passed))
                else:
                    ok = bool(passed) if passed is not None else False
                metrics[str(name)] = ok

    out["metrics"] = metrics
    out["failed_metrics"] = sorted([k for k, ok in metrics.items() if not bool(ok)])
    return out


class CondensedProgressLog:
    """Condensed progress log."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _stamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _write(self, level: str, stage: str, message: str) -> None:
        line = f"[{self._stamp()}] {level.upper():<5s} [{stage}] {message}\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def info(self, stage: str, message: str) -> None:
        self._write("info", stage, message)

    def warn(self, stage: str, message: str) -> None:
        self._write("warn", stage, message)

    def error(self, stage: str, message: str) -> None:
        self._write("error", stage, message)

    def convergence(self, stage: str, report: Mapping[str, Any]) -> None:
        flat = summarise_convergence_report(report)
        groups = flat.get("groups", {}) or {}
        metrics = flat.get("metrics", {}) or {}
        groups_txt = ", ".join(f"{k}={'pass' if bool(v) else 'fail'}" for k, v in sorted(groups.items()))
        self.info(stage, f"total convergence={'pass' if flat['passed'] else 'fail'}" + (f"; groups: {groups_txt}" if groups_txt else ""))
        if metrics:
            ordered = ", ".join(f"{k}={'pass' if bool(v) else 'fail'}" for k, v in sorted(metrics.items()))
            self.info(stage, f"per-metric convergence: {ordered}")
