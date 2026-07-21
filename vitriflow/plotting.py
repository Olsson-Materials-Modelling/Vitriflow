from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Mapping

import numpy as np


SpreadMode = Literal["sd", "se", "p16-84"]


OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#999999",
}

_OKABE_ITO_CYCLE = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluishgreen"],
    OKABE_ITO["purple"],
    OKABE_ITO["orange"],
    OKABE_ITO["skyblue"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]


_PLOT_CDF_ROUNDOFF_ATOL = 1.0e-12


def _require_positive_integer(value: Any, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive integer") from exc
    try:
        exact = float(value)
    except (TypeError, ValueError):
        exact = float(parsed)
    if parsed <= 0 or not np.isfinite(exact) or exact != float(parsed):
        raise ValueError(f"{context} must be a positive integer")
    return parsed


def _strict_production_plot_box_id(value: Any, *, context: str) -> int:
    """Validate one public box identifier without lossy integer coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} has invalid boolean box identifier")
    if isinstance(value, Integral):
        parsed = int(value)
    else:
        try:
            exact = Decimal(str(value).strip())
        except (InvalidOperation, ValueError, AttributeError) as exc:
            raise ValueError(
                f"{context} has invalid box identifier {value!r}"
            ) from exc
        if not exact.is_finite() or exact != exact.to_integral_value():
            raise ValueError(
                f"{context} has non-integral box identifier {value!r}"
            )
        parsed = int(exact)
    if parsed < 0:
        raise ValueError(f"{context} has negative box identifier {parsed}")
    return int(parsed)


def _normalise_production_plot_boxes(
    raw_boxes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return uniquely keyed boxes, preserving the native zero-based contract.

    Current files normally carry either ``box`` or ``box_id``.  When both are
    present they are independent provenance claims and must agree; selecting
    one by truthiness or precedence could otherwise conceal corrupt input.
    """

    boxes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ordinal, raw_box in enumerate(raw_boxes, start=1):
        if not isinstance(raw_box, Mapping):
            raise ValueError(f"Production box {ordinal} must be a mapping")
        box = dict(raw_box)
        has_box = "box" in box and box.get("box") is not None
        has_box_id = "box_id" in box and box.get("box_id") is not None
        if has_box:
            from_box = _strict_production_plot_box_id(
                box.get("box"), context=f"Production box {ordinal}"
            )
        else:
            from_box = None
        if has_box_id:
            from_box_id = _strict_production_plot_box_id(
                box.get("box_id"), context=f"Production box {ordinal}"
            )
        else:
            from_box_id = None
        if from_box is not None and from_box_id is not None and from_box != from_box_id:
            raise ValueError(
                f"Production box {ordinal} has conflicting box identifiers "
                f"box={from_box} and box_id={from_box_id}"
            )
        identifier = (
            from_box_id
            if from_box_id is not None
            else from_box
            if from_box is not None
            else int(ordinal)
        )
        if identifier in seen:
            raise ValueError(
                f"Production section contains duplicate box identifier {identifier}"
            )
        seen.add(identifier)
        box["box_id"] = int(identifier)
        boxes.append(box)
    return boxes


def _validated_plot_cdf_payload(
    payload: Mapping[str, Any],
    *,
    xkey: str = "x",
    allow_implicit_integer_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Validate a stored CDF for plotting without repairing scientific data.

    Only legacy coordination payloads with an entirely absent ``x`` key may
    infer the integer support ``0, 1, ...``.  Current payloads, especially soft
    graph coordination CDFs, must retain and use their explicit physical axis.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("CDF payload must be a mapping")
    try:
        cdf = np.asarray(payload.get("cdf", []), dtype=float)
    except Exception as exc:
        raise ValueError("CDF values must be numeric") from exc
    if cdf.ndim != 1 or cdf.size == 0 or not np.all(np.isfinite(cdf)):
        raise ValueError("CDF must be a non-empty one-dimensional finite array")

    if allow_implicit_integer_axis and xkey not in payload:
        x = np.arange(int(cdf.size), dtype=float)
        axis_source = "implicit_integer_index_legacy"
    else:
        try:
            x = np.asarray(payload.get(xkey, []), dtype=float)
        except Exception as exc:
            raise ValueError(f"CDF axis '{xkey}' must be numeric") from exc
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError(f"CDF axis '{xkey}' must be a non-empty one-dimensional finite array")
        axis_source = "explicit_stored_axis"

    if x.size != cdf.size:
        raise ValueError(
            f"CDF axis/value length mismatch ({int(x.size)} != {int(cdf.size)})"
        )
    if x.size > 1 and not np.all(np.diff(x) > 0.0):
        raise ValueError("CDF axis must be strictly increasing with no duplicates")
    atol = float(_PLOT_CDF_ROUNDOFF_ATOL)
    if np.any(cdf < -atol) or np.any(cdf > 1.0 + atol):
        raise ValueError("CDF values must lie in [0, 1]")
    if cdf.size > 1 and np.any(np.diff(cdf) < -atol):
        raise ValueError("CDF values must be nondecreasing")

    # Only accepted floating-point boundary noise is normalised.
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    return np.asarray(x, dtype=float), np.asarray(cdf, dtype=float), str(axis_source)


def _evaluate_plot_cdf_right_continuous(
    x: np.ndarray,
    cdf: np.ndarray,
    x_ref: np.ndarray,
) -> np.ndarray:
    """Evaluate a validated empirical CDF on a strictly increasing support."""

    x = np.asarray(x, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    if (
        x.ndim != 1
        or cdf.ndim != 1
        or x_ref.ndim != 1
        or x.size == 0
        or x.size != cdf.size
        or x_ref.size == 0
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(cdf))
        or not np.all(np.isfinite(x_ref))
        or (x.size > 1 and not np.all(np.diff(x) > 0.0))
        or (x_ref.size > 1 and not np.all(np.diff(x_ref) > 0.0))
    ):
        raise ValueError("CDF evaluation requires finite, strictly increasing, matching arrays")
    atol = float(_PLOT_CDF_ROUNDOFF_ATOL)
    if np.any(cdf < -atol) or np.any(cdf > 1.0 + atol) or (
        cdf.size > 1 and np.any(np.diff(cdf) < -atol)
    ):
        raise ValueError("CDF evaluation requires nondecreasing values in [0, 1]")

    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    indices = np.searchsorted(x, x_ref, side="right") - 1
    out = np.zeros_like(x_ref, dtype=float)
    valid = indices >= 0
    out[valid] = cdf[indices[valid]]
    out[x_ref >= x[-1]] = float(cdf[-1])
    return out


def _align_plot_cdf_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    xkey: str = "x",
    allow_implicit_integer_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align stored CDFs on their union support using step-CDF semantics."""

    if not payloads:
        raise ValueError("no CDF payloads supplied")
    curves = [
        _validated_plot_cdf_payload(
            payload,
            xkey=xkey,
            allow_implicit_integer_axis=allow_implicit_integer_axis,
        )
        for payload in payloads
    ]
    x_ref = np.unique(np.concatenate([x for x, _cdf, _source in curves]))
    if x_ref.size == 0 or not np.all(np.isfinite(x_ref)) or (
        x_ref.size > 1 and not np.all(np.diff(x_ref) > 0.0)
    ):
        raise ValueError("could not construct a finite strictly increasing CDF union support")
    mat = np.vstack(
        [
            _evaluate_plot_cdf_right_continuous(x, cdf, x_ref)
            for x, cdf, _source in curves
        ]
    ).astype(float)
    first = curves[0][0]
    same_grid = all(
        x.shape == first.shape
        and np.allclose(x, first, rtol=0.0, atol=1.0e-10)
        for x, _cdf, _source in curves
    )
    axis_sources = [source for _x, _cdf, source in curves]
    return x_ref.astype(float), mat, {
        "grid_source": "native_common_grid" if same_grid else "ensemble_union_support_grid",
        "grid_alignment_method": "right_continuous_cdf_evaluation",
        "axis_source": (
            axis_sources[0]
            if len(set(axis_sources)) == 1
            else "mixed_explicit_and_implicit_legacy_axes"
        ),
    }


def _align_sampled_plot_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    family: str,
    xkey: str,
    ykey: str,
    x_ref: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align complete g(r)/S(q) payloads on their common physical support.

    Plotting deliberately keeps this dependency-light instead of importing the
    full production-analysis stack.  It still enforces the same essential
    contract: every box contributes a finite, increasing curve; interpolation
    never extrapolates; and current S(q) representation metadata must be
    present consistently and agree on estimator-defining invariants.
    """

    if not payloads:
        raise ValueError(f"no payloads supplied for {family} curve")
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, payload in enumerate(payloads):
        if not isinstance(payload, Mapping):
            raise ValueError(f"{family} curve payload {idx} must be a mapping")
        try:
            x = np.asarray(payload.get(xkey, []), dtype=float)
            y = np.asarray(payload.get(ykey, []), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{family} curve payload {idx} must be numeric") from exc
        if (
            x.ndim != 1
            or y.ndim != 1
            or x.size < 2
            or y.size != x.size
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or not np.all(np.diff(x) > 0.0)
        ):
            raise ValueError(
                f"{family} curve payload {idx} requires matching finite arrays "
                "on a strictly increasing grid"
            )
        curves.append((x, y))

    representation_status = "not_applicable"
    if str(family) == "sq":
        present = ["representation" in payload for payload in payloads]
        if any(present) and not all(present):
            raise ValueError(
                "mixed S(q) representation metadata availability is not comparable"
            )
        if all(present):
            reps: list[Mapping[str, Any]] = []
            invariant_fields = (
                "schema",
                "observable",
                "estimator",
                "normalization",
                "normalization_family",
                "normalization_formula",
                "self_term",
                "pair",
                "resolved_type_sets",
                "scattering_weights",
                "scattering_weighted",
                "dimensionless",
                "q_unit",
                "r_unit",
                "termination_window",
                "rdf_normalization",
                "termination_window_definition",
                "radial_transform_kernel",
                "radial_quadrature",
                "r_support_requested_A",
                "r_support_policy",
                "n_r_bins",
                "q_min_A^-1",
                "q_max_A^-1",
                "n_q_points",
                "q_zero_semantics",
                "frame_aggregation",
                "density_handling",
                "density_prefactor_unit",
                "partial_kind",
            )
            for idx, payload in enumerate(payloads):
                rep = payload.get("representation")
                if not isinstance(rep, Mapping):
                    raise ValueError(
                        f"S(q) representation in payload {idx} must be a mapping"
                    )
                if rep.get("schema") != "vitriflow.sq_representation.v1":
                    raise ValueError(
                        f"S(q) representation in payload {idx} has an unsupported schema"
                    )
                try:
                    json.dumps(dict(rep), sort_keys=True, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"S(q) representation in payload {idx} is not strict JSON data"
                    ) from exc
                try:
                    n_q = int(rep.get("n_q_points"))
                    n_requested = int(rep.get("n_frames_requested"))
                    n_used = int(rep.get("n_frames_used"))
                    support = float(rep.get("r_support_effective_A"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"S(q) representation in payload {idx} has invalid dimensions"
                    ) from exc
                if (
                    n_q != int(curves[idx][0].size)
                    or n_requested < 1
                    or n_requested != n_used
                    or not (np.isfinite(support) and support > 0.0)
                ):
                    raise ValueError(
                        f"S(q) representation in payload {idx} disagrees with its curve"
                    )
                reps.append(rep)
            reference = reps[0]
            optional_invariants = (
                {"resolved_type_sets", "partial_kind"}
                if reference.get("normalization_family") == "number_number"
                else set()
            )
            required_invariants = [
                field
                for field in invariant_fields
                if field not in optional_invariants
            ]
            for field in invariant_fields:
                if field not in reference and field not in optional_invariants:
                    raise ValueError(
                        f"S(q) representation is missing invariant field {field!r}"
                    )
            for idx, rep in enumerate(reps[1:], start=1):
                for field in required_invariants:
                    if json.dumps(
                        rep.get(field), sort_keys=True, allow_nan=False
                    ) != json.dumps(
                        reference.get(field), sort_keys=True, allow_nan=False
                    ):
                        raise ValueError(
                            f"S(q) representation invariant {field!r} differs in payload {idx}"
                        )
                if int(rep.get("n_frames_used")) != int(
                    reference.get("n_frames_used")
                ):
                    raise ValueError("S(q) frame counts differ across payloads")
                supports = [
                    float(reference.get("r_support_effective_A")),
                    float(rep.get("r_support_effective_A")),
                ]
                tolerance = 128.0 * np.finfo(float).eps * max(
                    1.0, *(abs(value) for value in supports)
                )
                if abs(supports[1] - supports[0]) > tolerance:
                    raise ValueError(
                        "S(q) effective radial support differs across payloads"
                    )
            representation_status = "validated"
        else:
            representation_status = "legacy_unavailable"

    overlap_min = max(float(x[0]) for x, _y in curves)
    overlap_max = min(float(x[-1]) for x, _y in curves)
    if not (
        np.isfinite(overlap_min)
        and np.isfinite(overlap_max)
        and overlap_max > overlap_min
    ):
        raise ValueError(f"{family} curve grids have no common physical support")

    first = curves[0][0]
    same_grid = all(
        x.shape == first.shape
        and np.allclose(x, first, rtol=0.0, atol=1.0e-10)
        for x, _y in curves[1:]
    )
    if x_ref is not None:
        common = np.asarray(x_ref, dtype=float)
        if (
            common.ndim != 1
            or common.size < 2
            or not np.all(np.isfinite(common))
            or not np.all(np.diff(common) > 0.0)
            or float(common[0]) < overlap_min - 1.0e-12
            or float(common[-1]) > overlap_max + 1.0e-12
        ):
            raise ValueError("reference sampled-curve grid lies outside common support")
        grid_source = "provided_ensemble_common_grid"
    elif same_grid:
        common = first.copy()
        grid_source = "native_common_grid"
    else:
        within = [x[(x >= overlap_min) & (x <= overlap_max)] for x, _y in curves]
        common = np.unique(
            np.concatenate(
                [np.asarray([overlap_min, overlap_max], dtype=float), *within]
            )
        )
        common.sort()
        grid_source = "ensemble_union_common_support_grid"
    if common.size < 2:
        raise ValueError(f"cannot construct a common grid for {family} curve")
    matrix = np.vstack([np.interp(common, x, y) for x, y in curves])
    return common, matrix, {
        "grid_source": grid_source,
        "grid_alignment_method": "linear_interpolation_without_extrapolation",
        "representation_validation_status": representation_status,
    }


def _apply_publication_style(*, base_fontsize: float = 10.0) -> None:
    """Apply the VitriFlow publication plotting style.

    The style follows the group guide: Okabe-Ito palette, black boxed axes,
    inward ticks on all sides, visible minor ticks, framed legends, no grids,
    and editable vector-font output.
    """

    import matplotlib as mpl

    fs = float(base_fontsize)
    mpl.rcParams.update(
        {
            # Fonts
            "font.size": fs,
            "axes.labelsize": fs,
            "axes.titlesize": fs,
            "legend.fontsize": max(fs - 1.0, 1.0),
            "xtick.labelsize": max(fs - 1.0, 1.0),
            "ytick.labelsize": max(fs - 1.0, 1.0),
            # Lines/markers
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.prop_cycle": mpl.cycler(color=_OKABE_ITO_CYCLE),
            # Axes
            "axes.linewidth": 1.0,
            "axes.edgecolor": OKABE_ITO["black"],
            "axes.labelcolor": OKABE_ITO["black"],
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.color": OKABE_ITO["black"],
            "ytick.color": OKABE_ITO["black"],
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Legend
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.framealpha": 1.0,
            "legend.facecolor": "white",
            "legend.edgecolor": OKABE_ITO["black"],
            # Output
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _style_axes(ax: Any) -> None:
    """No grid; black spines; inward major+minor ticks on all sides."""

    try:
        if not bool(getattr(ax, "axison", True)):
            return
    except Exception:
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    try:
        ax.grid(False)
    except Exception:
        pass

    # AutoMinorLocator is intended for linear axes.  Log axes keep matplotlib's
    # log minor locator while retaining the same inward tick styling.
    try:
        if str(ax.get_xscale()).lower() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    except Exception:
        pass
    try:
        if str(ax.get_yscale()).lower() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    except Exception:
        pass

    lw = float(plt.rcParams.get("axes.linewidth", 1.0))
    for side in ("top", "right", "bottom", "left"):
        try:
            sp = ax.spines[side]
            sp.set_visible(True)
            sp.set_linewidth(lw)
            sp.set_color(OKABE_ITO["black"])
        except Exception:
            pass
    try:
        ax.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            bottom=True,
            left=True,
            colors=OKABE_ITO["black"],
            width=lw,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            top=True,
            right=True,
            bottom=True,
            left=True,
            colors=OKABE_ITO["black"],
            width=max(0.8 * lw, 0.6),
        )
    except Exception:
        pass


def _style_legend(leg: Any) -> None:
    if leg is None:
        return
    try:
        leg.set_frame_on(True)
        fr = leg.get_frame()
        fr.set_alpha(1.0)
        fr.set_facecolor("white")
        fr.set_edgecolor(OKABE_ITO["black"])
        fr.set_linewidth(0.8)
    except Exception:
        return


def _style_figure(fig: Any) -> None:
    """Apply publication axes/legend styling to every visible axis in a figure."""

    for ax in list(getattr(fig, "axes", []) or []):
        _style_axes(ax)
        try:
            _style_legend(ax.get_legend())
        except Exception:
            pass
    try:
        leg = getattr(fig, "legends", [])
        for lg in list(leg or []):
            _style_legend(lg)
    except Exception:
        pass


def _style_and_save_figure(fig: Any, out_path: Path, *, dpi: int, close: bool = True) -> None:
    import matplotlib.pyplot as plt

    _style_figure(fig)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dpi = int(dpi) if str(out_path).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")) else None
    fig.savefig(str(out_path), dpi=save_dpi)
    if close:
        plt.close(fig)


def _units_from_results(data: dict[str, Any]) -> tuple[str, Optional[float]]:
    if not isinstance(data, Mapping):
        raise ValueError("results root must be a mapping")
    u = data.get("units", {}) or {}
    params = data.get("parameters", {}) or {}
    plan = data.get("production_plan", {}) or {}
    u = u if isinstance(u, Mapping) else {}
    params = params if isinstance(params, Mapping) else {}
    plan = plan if isinstance(plan, Mapping) else {}

    unit_claims = {
        str(value).strip().lower()
        for value in (u.get("lammps_units"), params.get("lammps_units"))
        if value is not None and str(value).strip()
    }
    potential = plan.get("potential_config", {})
    if isinstance(potential, Mapping):
        claim = str(potential.get("user_units", "") or "").strip().lower()
        if claim:
            unit_claims.add(claim)
    if len(unit_claims) > 1:
        raise ValueError(
            f"results contain conflicting LAMMPS unit metadata: {sorted(unit_claims)}"
        )
    units_style = next(iter(unit_claims), "")

    time_claims: list[float] = []
    for value in (
        u.get("time_unit_ps"),
        params.get("time_unit_ps"),
        plan.get("time_unit_ps"),
    ):
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("results time_unit_ps must be numeric") from exc
        if not (np.isfinite(parsed) and parsed > 0.0):
            raise ValueError("results time_unit_ps must be finite and > 0")
        time_claims.append(parsed)
    if time_claims and any(
        not np.isclose(value, time_claims[0], rtol=1.0e-12, atol=0.0)
        for value in time_claims[1:]
    ):
        raise ValueError("results contain conflicting time_unit_ps metadata")
    return units_style, (time_claims[0] if time_claims else None)


def _rate_scan_coordinates(
    results: Sequence[Mapping[str, Any]],
    *,
    time_unit_ps: Optional[float],
) -> tuple[np.ndarray, str]:
    """Return a complete, dimensionally consistent rate-scan coordinate.

    Current results store the native cooling rate as ``rate`` and its explicit
    canonical conversion as ``rate_K_per_ps``.  ``rate_K_per_time`` is accepted
    only as a read-only legacy alias.  A mixture that cannot be placed on one
    common axis is rejected rather than rendered as a blank or partly missing
    plot.
    """

    rows = list(results)
    if not rows:
        raise ValueError("rate_scan contains no rate rows")

    scale = _maybe_float(time_unit_ps)
    has_scale = bool(np.isfinite(scale) and scale > 0.0)
    native: list[float] = []
    per_ps: list[float] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rate_scan row {idx} is not a mapping")

        rate_current = _maybe_float(row.get("rate"))
        rate_legacy = _maybe_float(row.get("rate_K_per_time"))
        if np.isfinite(rate_current) and np.isfinite(rate_legacy) and not np.isclose(
            rate_current,
            rate_legacy,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                f"rate_scan row {idx} has conflicting 'rate' and legacy "
                "'rate_K_per_time' values"
            )
        rate_native = rate_current if np.isfinite(rate_current) else rate_legacy
        rate_ps = _maybe_float(row.get("rate_K_per_ps"))

        if np.isfinite(rate_native) and rate_native <= 0.0:
            raise ValueError(f"rate_scan row {idx} has a non-positive native rate")
        if np.isfinite(rate_ps) and rate_ps <= 0.0:
            raise ValueError(f"rate_scan row {idx} has a non-positive rate_K_per_ps")

        if np.isfinite(rate_native) and np.isfinite(rate_ps) and has_scale:
            expected_ps = float(rate_native) / float(scale)
            if not np.isclose(rate_ps, expected_ps, rtol=1.0e-10, atol=0.0):
                raise ValueError(
                    f"rate_scan row {idx} has inconsistent native and K/ps rates: "
                    f"{rate_native:g} / {scale:g} != {rate_ps:g}"
                )

        if not np.isfinite(rate_ps) and np.isfinite(rate_native) and has_scale:
            rate_ps = float(rate_native) / float(scale)
        if not np.isfinite(rate_native) and np.isfinite(rate_ps) and has_scale:
            rate_native = float(rate_ps) * float(scale)

        native.append(float(rate_native))
        per_ps.append(float(rate_ps))

    per_ps_arr = np.asarray(per_ps, dtype=float)
    native_arr = np.asarray(native, dtype=float)
    if np.all(np.isfinite(per_ps_arr)):
        return per_ps_arr, "cooling rate (K/ps)"
    if np.all(np.isfinite(native_arr)):
        return native_arr, "cooling rate (K / time unit)"
    raise ValueError(
        "rate_scan rows do not provide one complete rate coordinate; expected "
        "rate_K_per_ps or rate/rate_K_per_time with units.time_unit_ps"
    )


def _require_finite_plot_points(x: Any, y: Any, *, context: str) -> None:
    """Reject successful-but-empty public plots."""

    try:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
    except Exception as exc:
        raise ValueError(f"{context} plot data must be numeric") from exc
    if xa.ndim != 1 or ya.ndim != 1 or xa.size != ya.size:
        raise ValueError(
            f"{context} plot coordinate/value arrays must be one-dimensional and equal length"
        )
    if xa.size == 0 or not np.any(np.isfinite(xa) & np.isfinite(ya)):
        raise ValueError(f"{context} contains no finite coordinate/value pairs")


def _strict_optional_bool(value: Any) -> Optional[bool]:
    return value if type(value) is bool else None


def _production_convergence_display(
    production: Mapping[str, Any],
    convergence: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Interpret convergence without coercing null/unassessed states to failure.

    The returned label deliberately distinguishes a terminal fixed-count
    diagnostic from a stopping decision and carries the inference-validity
    qualification emitted by the convergence engine.
    """

    prod = production if isinstance(production, Mapping) else {}
    conv_raw = convergence if convergence is not None else prod.get("convergence", {})
    conv = conv_raw if isinstance(conv_raw, Mapping) else {}

    check_enabled = _strict_optional_bool(prod.get("check_convergence"))
    assessment_role = str(
        conv.get("assessment_role", prod.get("assessment_role", "")) or ""
    ).strip().lower()
    sampling_design = str(
        conv.get("sampling_design", prod.get("sampling_design", "")) or ""
    ).strip().lower()
    status = str(
        conv.get("status", prod.get("convergence_status", "")) or ""
    ).strip().lower()
    inference_status = str(
        prod.get(
            "convergence_inference_status",
            conv.get("convergence_inference_status", conv.get("sequential_inference_status", "")),
        )
        or ""
    ).strip().lower()
    contract = conv.get("inference_contract", {})
    contract = contract if isinstance(contract, Mapping) else {}
    sequentially_valid = _strict_optional_bool(
        contract.get("sequentially_valid", conv.get("sequentially_valid"))
    )

    posthoc = bool(
        assessment_role == "terminal_posthoc_diagnostic"
        or sampling_design == "fixed_n"
        or status.startswith("fixed_n_terminal_posthoc")
    )
    if posthoc:
        criterion = _strict_optional_bool(conv.get("posthoc_criterion_met"))
        if criterion is None:
            criterion = _strict_optional_bool(
                prod.get("posthoc_convergence_criterion_met")
            )
        if criterion is True:
            label = "terminal post-hoc criterion met; not a stopping result"
            state = "posthoc_criterion_met"
        elif criterion is False:
            label = "terminal post-hoc criterion not met; not a stopping result"
            state = "posthoc_criterion_not_met"
        else:
            label = "terminal post-hoc convergence unassessed; not a stopping result"
            state = "unassessed"
        return {
            "state": state,
            "label": label,
            "criterion_met": criterion,
            "sequentially_valid": False,
            "inference_status": inference_status or "not_sequentially_valid",
        }

    overall = conv.get("convergence_degree", {})
    overall = overall.get("overall", {}) if isinstance(overall, Mapping) else {}
    overall_assessed = (
        _strict_optional_bool(overall.get("assessed"))
        if isinstance(overall, Mapping)
        else None
    )
    assessment_performed = _strict_optional_bool(conv.get("assessment_performed"))
    report_has_assessment_evidence = bool(
        isinstance(conv.get("familywise"), Mapping)
        and (
            bool(conv.get("scalars"))
            or bool(conv.get("distributions"))
            or isinstance(conv.get("convergence_degree"), Mapping)
        )
    )
    explicit_unassessed = bool(
        status in {"skipped", "not_evaluated", "not_yet_assessed", "unassessed"}
        or status.endswith("_unassessed")
        or inference_status in {"not_yet_assessed", "fixed_count_unassessed"}
        or overall_assessed is False
        or assessment_performed is False
        or (status == "" and not report_has_assessment_evidence)
    )
    if check_enabled is False:
        explicit_unassessed = True

    criterion = _strict_optional_bool(prod.get("converged"))
    if criterion is None:
        criterion = _strict_optional_bool(conv.get("converged"))
    if criterion is None and status == "converged":
        criterion = True
    if criterion is None and status == "not_converged":
        criterion = False

    if explicit_unassessed or criterion is None:
        reason = (
            "convergence checking disabled"
            if check_enabled is False
            else "no assessed convergence decision"
        )
        return {
            "state": "unassessed",
            "label": f"convergence unassessed; {reason}",
            "criterion_met": None,
            "sequentially_valid": sequentially_valid,
            "inference_status": inference_status or "not_assessed",
        }

    repeated_looks_invalid = bool(
        sequentially_valid is False
        or "not_sequentially_valid" in inference_status
        or "repeated_looks" in inference_status
    )
    if repeated_looks_invalid:
        qualification = "repeated-look inference is not sequentially valid"
    elif sequentially_valid is True:
        qualification = "sequential inference declared valid"
    else:
        qualification = "inference validity not declared"
    return {
        "state": "criterion_met" if criterion else "criterion_not_met",
        "label": f"criterion {'met' if criterion else 'not met'}; {qualification}",
        "criterion_met": criterion,
        "sequentially_valid": sequentially_valid,
        "inference_status": inference_status or "not_declared",
    }


def _familywise_error_annotation(
    familywise: Mapping[str, Any],
    *,
    alpha_per_test: float,
    bounded_ci_method: str,
) -> str:
    """Build a statistically unambiguous familywise-error annotation."""

    fw = familywise if isinstance(familywise, Mapping) else {}
    alpha_family = _maybe_float(fw.get("alpha_family"))
    m_tests_raw = fw.get("m_tests", 0)
    try:
        m_tests = int(m_tests_raw)
    except Exception:
        m_tests = 0
    if np.isfinite(alpha_family) and 0.0 < alpha_family < 1.0:
        family_confidence = 1.0 - float(alpha_family)
        fwer = (
            f"FWER alpha={alpha_family:.3g} "
            f"(family confidence={family_confidence:.3f})"
        )
    else:
        fwer = "FWER alpha=not declared"
    return (
        f"{fwer}  M={m_tests}  alpha_test={float(alpha_per_test):.2e}  "
        f"bounded_CI={bounded_ci_method}"
    )


def _coordination_sweep_matrices(
    boxes: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect complete coordination-cutoff sensitivity curves by metric name."""

    rows = list(boxes)
    names: set[str] = set()
    for box in rows:
        details = box.get("coordination_defect_details", {})
        if not isinstance(details, Mapping):
            continue
        for name, detail in details.items():
            if isinstance(detail, Mapping) and isinstance(
                detail.get("coordination_sweep"), Mapping
            ):
                names.add(str(name))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in sorted(names):
        x_ref: Optional[np.ndarray] = None
        curves: list[np.ndarray] = []
        for box_idx, box in enumerate(rows):
            details = box.get("coordination_defect_details", {})
            detail = details.get(name, {}) if isinstance(details, Mapping) else {}
            sweep = (
                detail.get("coordination_sweep", {})
                if isinstance(detail, Mapping)
                else {}
            )
            if not isinstance(sweep, Mapping):
                raise ValueError(
                    f"coordination sweep {name!r} is missing for box index {box_idx}"
                )
            x = np.asarray(sweep.get("delta_r", []), dtype=float)
            y = np.asarray(sweep.get("defect_fraction", []), dtype=float)
            _require_finite_plot_points(
                x,
                y,
                context=f"coordination sweep {name!r} box index {box_idx}",
            )
            if x.size > 1 and not np.all(np.diff(x) > 0.0):
                raise ValueError(
                    f"coordination sweep {name!r} delta_r grid is not strictly increasing"
                )
            if x_ref is None:
                x_ref = x
            elif x.shape != x_ref.shape or not np.allclose(
                x,
                x_ref,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"coordination sweep {name!r} uses inconsistent grids across boxes"
                )
            curves.append(y)
        if x_ref is not None and len(curves) == len(rows) and curves:
            out[name] = (x_ref, np.vstack(curves))
    return out


def _uses_canonical_report_units(data: Mapping[str, Any]) -> bool:
    """Whether reported physical observables use engine-neutral units.

    Releases before the canonical reporting contract stored several scan
    quantities in the active LAMMPS units.  Keep that legacy plotting path for
    old files, while new files and standalone analysis results are always
    interpreted in their declared physical units.
    """

    units_raw = data.get("units", {}) if isinstance(data, Mapping) else {}
    units = units_raw if isinstance(units_raw, Mapping) else {}
    contract = str(units.get("reporting_contract", "") or "").strip().lower()
    if contract.startswith("vitriflow.canonical_physical_units."):
        return True
    schema = str(data.get("schema", "") or "").strip().lower()
    return schema.startswith("vitriflow.analysis_results.v2")


def _stage_plot_metadata(
    stage_dir: Path,
    *,
    results_json: Optional[Path],
) -> dict[str, Any]:
    """Resolve stage plotting units with stage-local provenance as authority."""

    from .io.stage_manifest import (
        STAGE_ARTIFACT_MANIFEST_NAME,
        load_stage_artifact_manifest,
    )

    manifest_path = Path(stage_dir) / STAGE_ARTIFACT_MANIFEST_NAME
    if manifest_path.exists():
        # A present but malformed manifest must not be silently ignored: that
        # could label native or modified CSV values as canonical physical data.
        manifest = load_stage_artifact_manifest(manifest_path)
        native = manifest.get("native_source_units", {}) or {}
        units_style = (
            str(native.get("lammps_units_style", "") or "").strip().lower()
            if isinstance(native, Mapping)
            else ""
        )
        return {
            "dt": float(manifest["timestep_ps"]),
            "units_style": units_style,
            "time_unit_ps": 1.0,
            "canonical_units": True,
            "manifest": manifest,
            "source": "stage_manifest",
        }

    dt = None
    units_style = ""
    time_unit_ps = None
    canonical_units = False
    if results_json is not None:
        result_path = Path(results_json).resolve()
        stage_resolved = Path(stage_dir).resolve()
        if not stage_resolved.is_relative_to(result_path.parent):
            raise ValueError(
                "plot-stage --dir must be inside the output directory bound to --results"
            )
        try:
            data = json.loads(result_path.read_text())
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"Invalid results JSON for plot-stage: {result_path}") from exc
        if not isinstance(data, dict):
            raise ValueError("plot-stage results JSON root must be an object")
        dt = _infer_dt_from_results(data)
        units_style, time_unit_ps = _units_from_results(data)
        canonical_units = _uses_canonical_report_units(data)
    return {
        "dt": dt,
        "units_style": units_style,
        "time_unit_ps": time_unit_ps,
        "canonical_units": canonical_units,
        "manifest": None,
        "source": "results_json" if results_json is not None else "legacy",
    }


def _verify_stage_plot_artifact(
    *,
    stage_dir: Path,
    manifest: Mapping[str, Any],
    artifact_key: str,
    expected_name: str,
) -> bool:
    from .io.stage_manifest import verify_manifest_artifact

    records = manifest.get("artifacts", {})
    record = records.get(artifact_key) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping) or str(record.get("path", "")) != expected_name:
        raise ValueError(
            f"Stage manifest {artifact_key!r} does not identify expected file {expected_name!r}"
        )
    return verify_manifest_artifact(
        stage_dir=stage_dir,
        manifest=manifest,
        artifact_key=artifact_key,
    )


def _diffusion_for_plot(
    D: np.ndarray,
    *,
    scale: float = 1.0,
    zero_below: float = 0.1,
) -> np.ndarray:
    """Diffusion for plot."""

    arr = np.asarray(D, dtype=float) * float(scale)
    out = np.array(arr, copy=True)
    mfin = np.isfinite(out)
    out[mfin] = np.maximum(out[mfin], 0.0)
    thr = float(zero_below)
    if thr > 0.0:
        m = mfin & (out < thr)
        out[m] = 0.0
    return out


def _infer_dt_from_results(data: dict[str, Any]) -> Optional[float]:
    # Use the selected/replayed timestep, checking redundant claims rather
    # than silently preferring whichever field happens to be inspected first.
    recommendation = data.get("recommendation", {}) or {}
    preflight = data.get("preflight", {}) or {}
    plan = data.get("production_plan", {}) or {}
    parameters = data.get("parameters", {}) or {}
    recommendation = recommendation if isinstance(recommendation, Mapping) else {}
    preflight = preflight if isinstance(preflight, Mapping) else {}
    plan = plan if isinstance(plan, Mapping) else {}
    parameters = parameters if isinstance(parameters, Mapping) else {}
    rec_md = recommendation.get("md", {}) or {}
    plan_md = plan.get("md_use", {}) or {}
    param_md = parameters.get("md", {}) or {}
    rec_md = rec_md if isinstance(rec_md, Mapping) else {}
    plan_md = plan_md if isinstance(plan_md, Mapping) else {}
    param_md = param_md if isinstance(param_md, Mapping) else {}

    claims: list[float] = []
    for value in (
        rec_md.get("timestep"),
        preflight.get("selected_timestep"),
        plan_md.get("timestep"),
        param_md.get("timestep"),
    ):
        if value is None:
            continue
        try:
            dt = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("results timestep metadata must be numeric") from exc
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError("results timestep metadata must be finite and > 0")
        claims.append(dt)
    if claims and any(
        not np.isclose(value, claims[0], rtol=1.0e-12, atol=0.0)
        for value in claims[1:]
    ):
        raise ValueError("results contain conflicting selected timestep metadata")
    return claims[0] if claims else None


def _thermo_unit_label(col: str, units_style: str, *, canonical: bool = False) -> str:
    c = str(col)
    lc = c.strip().lower()
    pressure_columns = {"press", "pressure", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"}
    length_columns = {"lx", "ly", "lz", "xlo", "xhi", "ylo", "yhi", "zlo", "zhi"}
    energy_columns = {
        "pe", "ke", "etotal", "poteng", "kineng", "toteng", "enthalpy",
        "e_pair", "e_mol", "e_vdwl", "e_coul", "e_long", "e_bond",
        "e_angle", "e_dihed", "e_impro", "e_tail", "etot",
    }
    if lc in ("temp", "t"):
        return "K"
    if canonical:
        if lc == "time":
            return "ps"
        if lc in ("density",):
            return "g/cm³"
        if lc in pressure_columns:
            return "GPa"
        if lc in ("vol", "volume"):
            return "Å³"
        if lc in length_columns:
            return "Å"
        if lc.endswith("eng") or lc.endswith("energy") or lc in energy_columns:
            return "eV"
    native_units = {
        "metal": {
            "length": "Å", "volume": "Å³", "density": "g/cm³",
            "pressure": "bar", "energy": "eV", "time": "ps", "msd": "Å²",
        },
        "real": {
            "length": "Å", "volume": "Å³", "density": "g/cm³",
            "pressure": "atm", "energy": "kcal/mol", "time": "fs", "msd": "Å²",
        },
        "electron": {
            "length": "bohr", "volume": "bohr³", "density": "amu/bohr³",
            "pressure": "Pa", "energy": "hartree", "time": "fs", "msd": "bohr²",
        },
        "nano": {
            "length": "nm", "volume": "nm³", "density": "ag/nm³",
            "pressure": "ag/(nm·ns²)", "energy": "ag·nm²/ns²", "time": "ns", "msd": "nm²",
        },
        "si": {
            "length": "m", "volume": "m³", "density": "kg/m³",
            "pressure": "Pa", "energy": "J", "time": "s", "msd": "m²",
        },
        "cgs": {
            "length": "cm", "volume": "cm³", "density": "g/cm³",
            "pressure": "dyne/cm²", "energy": "erg", "time": "s", "msd": "cm²",
        },
        "micro": {
            "length": "µm", "volume": "µm³", "density": "pg/µm³",
            "pressure": "pg/(µm·µs²)", "energy": "pg·µm²/µs²", "time": "µs", "msd": "µm²",
        },
    }.get(str(units_style or "").strip().lower(), {})
    if lc == "time":
        return native_units.get("time", "")
    if lc == "density":
        return native_units.get("density", "")
    if lc in pressure_columns:
        return native_units.get("pressure", "")
    if lc in {"vol", "volume"}:
        return native_units.get("volume", "")
    if lc in length_columns:
        return native_units.get("length", "")
    if lc.endswith("eng") or lc.endswith("energy") or lc in energy_columns:
        return native_units.get("energy", "")
    return ""


def _native_length_unit_label(units_style: str) -> str:
    return {
        "metal": "Å", "real": "Å", "electron": "bohr", "nano": "nm",
        "si": "m", "cgs": "cm", "micro": "µm",
    }.get(str(units_style or "").strip().lower(), "")


def _native_msd_unit_label(units_style: str) -> str:
    return {
        "metal": "Å²", "real": "Å²", "electron": "bohr²", "nano": "nm²",
        "si": "m²", "cgs": "cm²", "micro": "µm²",
    }.get(str(units_style or "").strip().lower(), "")


def _legacy_diffusion_plot_scale(
    *, units_style: str, time_unit_ps: Optional[float]
) -> tuple[float, str]:
    """Return native D -> A^2/ps scaling when legacy units are declared."""

    if time_unit_ps is None:
        return 1.0, "D (length²/time)"
    try:
        time_scale = float(time_unit_ps)
        if not np.isfinite(time_scale) or time_scale <= 0.0:
            return 1.0, "D (length²/time)"
        from .lammps_units import length_to_angstrom_factor

        length_scale = float(length_to_angstrom_factor(units_style))
    except (TypeError, ValueError):
        return 1.0, "D (length²/time)"
    return (length_scale * length_scale) / time_scale, "D (Å²/ps)"


def _time_axis(
    step: np.ndarray,
    *,
    dt: Optional[float],
    time_unit_ps: Optional[float],
    prefer_ps: bool = True,
) -> tuple[np.ndarray, str]:
    step = np.asarray(step, dtype=float)
    if dt is None or (not np.isfinite(float(dt))) or float(dt) <= 0.0:
        return step, "MD step"
    t = step * float(dt)
    if prefer_ps and time_unit_ps is not None and np.isfinite(float(time_unit_ps)):
        return t * float(time_unit_ps), "time (ps)"
    return t, "time (MD time units)"


def plot_stage_timeseries(
    stage_dir: Path,
    out_path: Path,
    *,
    results_json: Optional[Path] = None,
    title: Optional[str] = None,
    thermo_series: Optional[list[str]] = None,
    plot_all_thermo: bool = False,
    include_msd: bool = True,
    xaxis: Literal["time", "step"] = "time",
    dpi: int = 600,
) -> None:
    """Stage timeseries."""

    dpi = _require_positive_integer(dpi, context="dpi")
    stage_dir = Path(stage_dir)
    out_path = Path(out_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()

    metadata = _stage_plot_metadata(stage_dir, results_json=results_json)
    dt = metadata["dt"]
    units_style = str(metadata["units_style"])
    time_unit_ps = metadata["time_unit_ps"]
    canonical_units = bool(metadata["canonical_units"])
    manifest = metadata["manifest"]

    from .io.thermo import parse_thermo_csv, parse_msd_csv

    thermo_path = stage_dir / "thermo.csv"
    if not thermo_path.exists():
        raise FileNotFoundError(f"Missing thermo.csv in stage directory: {stage_dir}")
    if isinstance(manifest, Mapping) and not _verify_stage_plot_artifact(
        stage_dir=stage_dir,
        manifest=manifest,
        artifact_key="thermo_csv",
        expected_name=thermo_path.name,
    ):
        raise ValueError(f"Stage manifest declares thermo.csv unavailable: {stage_dir}")
    table = parse_thermo_csv(thermo_path)
    cols = list(table.columns)
    arr = table.as_dict()
    if "Step" not in arr:
        raise ValueError(f"thermo.csv missing 'Step' column: {thermo_path}")

    step = np.asarray(arr["Step"], dtype=float)
    if xaxis == "time":
        x, xlabel = _time_axis(step, dt=dt, time_unit_ps=time_unit_ps)
    else:
        x, xlabel = step, "MD step"

    # determine thermo plot
    if plot_all_thermo:
        thermo_cols = [c for c in cols if c != "Step"]
    else:
        default_cols = ["Temp", "Press", "Density", "PotEng", "Volume"]
        thermo_cols = list(thermo_series) if thermo_series is not None else default_cols
        thermo_cols = [c for c in thermo_cols if c in cols and c != "Step"]

    # msd
    msd_x = None
    msd_y = None
    msd_path = stage_dir / "msd.csv"
    msd_available = msd_path.exists()
    if include_msd and isinstance(manifest, Mapping):
        msd_available = _verify_stage_plot_artifact(
            stage_dir=stage_dir,
            manifest=manifest,
            artifact_key="msd_csv",
            expected_name=msd_path.name,
        )
    if include_msd and msd_available:
        try:
            ms_step, msd = parse_msd_csv(msd_path)
            if xaxis == "time":
                msd_x, _ = _time_axis(ms_step, dt=dt, time_unit_ps=time_unit_ps)
            else:
                msd_x = ms_step
            msd_y = msd
        except Exception as exc:
            if isinstance(manifest, Mapping):
                # The manifest has asserted that this exact, identity-bound
                # artifact is available and canonical.  A strict parse failure
                # is therefore an integrity error, not an optional missing
                # panel that may be silently dropped.
                raise ValueError(
                    f"Manifest-bound MSD artifact failed strict parsing: {msd_path}"
                ) from exc
            # Explicit legacy path: pre-manifest MSD was optional and malformed
            # files were historically omitted from the plot.
            msd_x = None
            msd_y = None

    n_panels = int(len(thermo_cols)) + (1 if (msd_x is not None and msd_y is not None) else 0)
    if n_panels < 1:
        raise ValueError("No plottable series found (thermo columns missing and/or MSD unavailable).")
    for column in thermo_cols:
        _require_finite_plot_points(
            x,
            np.asarray(arr.get(column, np.full_like(x, np.nan)), dtype=float),
            context=f"stage thermo series {column!r}",
        )
    if msd_x is not None and msd_y is not None:
        _require_finite_plot_points(msd_x, msd_y, context="stage MSD series")

    fig_h = max(2.0, 1.8 * float(n_panels))
    fig, axes = plt.subplots(n_panels, 1, figsize=(6.5, fig_h), sharex=True)
    if n_panels == 1:
        axes = [axes]

    k = 0
    for c in thermo_cols:
        y = np.asarray(arr.get(c, np.full_like(x, np.nan)), dtype=float)
        ax = axes[k]
        ax.plot(x, y)
        unit = _thermo_unit_label(c, units_style, canonical=canonical_units)
        ylabel = f"{c} ({unit})" if unit else str(c)
        ax.set_ylabel(ylabel)
        k += 1

    if msd_x is not None and msd_y is not None:
        ax = axes[k]
        ax.plot(msd_x, msd_y)
        msd_unit = "Å²" if canonical_units else _native_msd_unit_label(units_style)
        ax.set_ylabel(f"MSD ({msd_unit})" if msd_unit else "MSD")
        # msd useful linear

    axes[-1].set_xlabel(xlabel)
    if title is None:
        title = str(stage_dir)
    fig.suptitle(title)

    _style_and_save_figure(fig, out_path, dpi=int(dpi))


def plot_scan_metric(
    json_path: Path,
    out_path: Path,
    *,
    stage: Literal["tm_scan", "rate_scan", "size_scan", "production"],
    metric: str,
    title: Optional[str] = None,
    spread: SpreadMode = "sd",
    show_replicates: bool = False,
    dpi: int = 600,
) -> None:
    """Scan metric."""

    dpi = _require_positive_integer(dpi, context="dpi")
    json_path = Path(json_path)
    out_path = Path(out_path)
    data = json.loads(json_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Scan plotting input must be a JSON object")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()

    units_style, time_unit_ps = _units_from_results(data)
    canonical_units = _uses_canonical_report_units(data)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    st = str(stage)
    mkey = str(metric)

    if st == "tm_scan":
        tm = data.get("tm_scan", {}) or {}
        outcomes = list(tm.get("outcomes", []) or [])
        agg = _aggregate_outcomes_by_T(outcomes, spread=spread)
        x = np.asarray(agg["T"], dtype=float)

        # metric aggregated arrays
        if mkey.lower() in ("d", "diffusion"):
            D_scale = 1.0
            ylab = "Diffusion coefficient (Å²/ps)" if canonical_units else "Diffusion coefficient"
            if not canonical_units:
                D_scale, legacy_label = _legacy_diffusion_plot_scale(
                    units_style=units_style,
                    time_unit_ps=time_unit_ps,
                )
                if legacy_label == "D (Å²/ps)":
                    ylab = "Diffusion coefficient (Å²/ps)"
            y = _diffusion_for_plot(np.asarray(agg["D"], dtype=float), scale=D_scale, zero_below=0.1)
            lo = _diffusion_for_plot(np.asarray(agg["D_lo"], dtype=float), scale=D_scale, zero_below=0.1)
            hi = _diffusion_for_plot(np.asarray(agg["D_hi"], dtype=float), scale=D_scale, zero_below=0.1)
            ax.set_yscale("symlog", linthresh=0.1)
        elif mkey.lower() in ("density", "rho"):
            y = np.asarray(agg["rho"], dtype=float)
            lo = np.asarray(agg["rho_lo"], dtype=float)
            hi = np.asarray(agg["rho_hi"], dtype=float)
            density_unit = (
                "g/cm³" if canonical_units else _thermo_unit_label("Density", units_style)
            ) or "units"
            ylab = f"Density ({density_unit})"
        elif mkey.lower() in ("pe", "poteng", "potential_energy"):
            y = np.asarray(agg["pe"], dtype=float)
            lo = np.asarray(agg["pe_lo"], dtype=float)
            hi = np.asarray(agg["pe_hi"], dtype=float)
            unit = _thermo_unit_label("PotEng", units_style, canonical=canonical_units)
            ylab = f"Potential energy ({unit})" if unit else "Potential energy"
        elif mkey.lower() in ("msd_rms", "msdrms", "rms"):
            y = np.asarray(agg["msdrms"], dtype=float)
            lo = np.asarray(agg["msdrms_lo"], dtype=float)
            hi = np.asarray(agg["msdrms_hi"], dtype=float)
            rms_unit = "Å" if canonical_units else _native_length_unit_label(units_style)
            ylab = f"RMS displacement ({rms_unit})" if rms_unit else "RMS displacement"
        elif mkey.lower() in ("gr_peak_height", "peak_height"):
            y = np.asarray(agg["gH"], dtype=float)
            lo = np.asarray(agg["gH_lo"], dtype=float)
            hi = np.asarray(agg["gH_hi"], dtype=float)
            ylab = "g(r) first-peak height"
        elif mkey.lower() in ("gr_peak_fwhm", "peak_fwhm"):
            y = np.asarray(agg["gW"], dtype=float)
            lo = np.asarray(agg["gW_lo"], dtype=float)
            hi = np.asarray(agg["gW_hi"], dtype=float)
            width_unit = "Å" if canonical_units else _native_length_unit_label(units_style)
            ylab = f"g(r) first-peak FWHM ({width_unit})" if width_unit else "g(r) first-peak FWHM"
        else:
            raise ValueError(
                f"Unsupported tm_scan metric '{mkey}'. "
                "Use one of: D, density, pe, msd_rms, gr_peak_height, gr_peak_fwhm."
            )

        _require_finite_plot_points(x, y, context=f"tm_scan metric {mkey!r}")
        _plot_mean_band(ax, x, y, lo, hi, label=None, show_band=True)
        if show_replicates and outcomes:
            # overlay replica points
            xs = []
            ys = []
            for o in outcomes:
                Ti = _maybe_float(o.get("temperature_start"), default=_maybe_float(o.get("temperature")))
                if not np.isfinite(Ti):
                    continue
                if mkey.lower() in ("d", "diffusion"):
                    vv0 = _maybe_float(o.get("D"))
                    vv = float("nan")
                    if np.isfinite(vv0):
                        vv = float(_diffusion_for_plot(np.asarray([vv0], dtype=float), scale=D_scale, zero_below=0.1)[0])
                elif mkey.lower() in ("density", "rho"):
                    vv = _maybe_float(o.get("density_mean"), default=_maybe_float(o.get("density")))
                elif mkey.lower() in ("pe", "poteng", "potential_energy"):
                    vv = _maybe_float(o.get("pe_mean"), default=_maybe_float(o.get("pe")))
                elif mkey.lower() in ("msd_rms", "msdrms", "rms"):
                    vv = _maybe_float(o.get("msd_rms_last"), default=_maybe_float(o.get("msd_rms")))
                elif mkey.lower() in ("gr_peak_height", "peak_height"):
                    vv = _maybe_float(o.get("gr_peak_height"))
                elif mkey.lower() in ("gr_peak_fwhm", "peak_fwhm"):
                    vv = _maybe_float(o.get("gr_peak_fwhm"))
                else:
                    vv = float("nan")
                if np.isfinite(vv):
                    xs.append(float(Ti))
                    ys.append(float(vv))
            if xs:
                ax.scatter(xs, ys, s=12, alpha=0.7)

        ax.set_xlabel("temperature (K)")
        ax.set_ylabel(ylab)

    elif st in ("rate_scan", "size_scan"):
        key = "rate_scan" if st == "rate_scan" else "size_scan"
        sec = data.get(key, {}) or {}
        if bool(sec.get("skipped", False)):
            raise RuntimeError(f"{key} was skipped: {sec.get('skip_reason', '')}")
        results = list(sec.get("rates" if st == "rate_scan" else "sizes", []) or [])
        if not results:
            raise RuntimeError(f"No entries found under {key}.")

        # x axis
        if st == "rate_scan":
            x, xlabel = _rate_scan_coordinates(
                results,
                time_unit_ps=time_unit_ps,
            )
        else:
            # prefer present multiplier
            x = np.asarray([float(r.get("n_atoms", r.get("multiplier", float("nan")))) for r in results], dtype=float)
            xlabel = "number of atoms" if any("n_atoms" in r for r in results) else "size multiplier"

        # y axis
        if mkey.lower() in ("density", "rho"):
            y = np.asarray([float(r.get("density_mean", float("nan"))) for r in results], dtype=float)
            se = np.asarray([float(r.get("density_stderr", float("nan"))) for r in results], dtype=float)
            unit = (
                "g/cm³" if canonical_units else _thermo_unit_label("Density", units_style)
            ) or "units"
            ylab = f"Density ({unit})"
        else:
            y = np.asarray([float((r.get("metrics_mean", {}) or {}).get(mkey, float("nan"))) for r in results], dtype=float)
            se = np.asarray([float((r.get("metrics_stderr", {}) or {}).get(mkey, float("nan"))) for r in results], dtype=float)
            ylab = str(mkey)

        _require_finite_plot_points(x, y, context=f"{key} metric {mkey!r}")
        lo = y - se
        hi = y + se
        _plot_mean_band(ax, x, y, lo, hi, label=None, show_band=True)

        if show_replicates:
            xs = []
            ys = []
            for row_idx, r in enumerate(results):
                xv = float(x[row_idx]) if st == "rate_scan" else float(r.get("n_atoms", r.get("multiplier", float("nan"))))
                if not np.isfinite(xv):
                    continue
                reps = list(r.get("replicates", []) or [])
                for re in reps:
                    if mkey.lower() in ("density", "rho"):
                        vv = _maybe_float(re.get("density"))
                    else:
                        vv = _maybe_float((re.get("metrics", {}) or {}).get(mkey))
                    if np.isfinite(vv):
                        xs.append(float(xv))
                        ys.append(float(vv))
            if xs:
                ax.scatter(xs, ys, s=12, alpha=0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)

    elif st == "production":
        prod = data.get("production", {}) or {}
        if not bool(prod.get("enabled", False)):
            raise RuntimeError("Production ensemble not present or production.enabled=false")
        boxes = list(prod.get("boxes", []) or [])
        if not boxes:
            raise RuntimeError("No production boxes found")

        # prefix mean stderr
        vals = []
        for b in boxes:
            if mkey.lower() in ("density", "rho"):
                vals.append(_maybe_float(b.get("density")))
            else:
                vals.append(_maybe_float((b.get("metrics", {}) or {}).get(mkey)))
        v = np.asarray(vals, dtype=float)
        n_grid = np.arange(1, len(boxes) + 1, dtype=int)

        mu = np.full_like(n_grid, np.nan, dtype=float)
        se = np.full_like(n_grid, np.nan, dtype=float)
        for i, n in enumerate(n_grid.tolist()):
            vv = v[:n]
            vv = vv[np.isfinite(vv)]
            if vv.size == 0:
                continue
            mu[i] = float(np.mean(vv))
            if vv.size >= 2:
                se[i] = float(np.std(vv, ddof=1) / np.sqrt(float(vv.size)))

        _require_finite_plot_points(
            n_grid,
            mu,
            context=f"production metric {mkey!r}",
        )
        _plot_mean_band(ax, n_grid, mu, mu - se, mu + se, label=None, show_band=True)
        ax.set_xlabel("number of boxes")
        if mkey.lower() in ("density", "rho"):
            unit = (
                "g/cm³" if canonical_units else _thermo_unit_label("Density", units_style)
            ) or "units"
            ax.set_ylabel(f"Density ({unit})")
        else:
            ax.set_ylabel(str(mkey))
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    if title is None:
        title = f"{stage}: {metric}"
    ax.set_title(title)
    _style_and_save_figure(fig, out_path, dpi=int(dpi))


def _maybe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _mean_sd_se(x: np.ndarray) -> tuple[float, float, float, int]:
    """Mean sd se."""

    xf = _finite_1d(x)
    n = int(xf.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mu = float(np.mean(xf))
    if n == 1:
        return mu, 0.0, float("nan"), 1
    sd = float(np.std(xf, ddof=1))
    se = float(sd / np.sqrt(n))
    return mu, sd, se, n


def _median_p16_p84(x: np.ndarray) -> tuple[float, float, float, int]:
    xf = _finite_1d(x)
    n = int(xf.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    med = float(np.median(xf))
    if n == 1:
        return med, med, med, 1
    p16 = float(np.percentile(xf, 16))
    p84 = float(np.percentile(xf, 84))
    return med, p16, p84, n


def _center_band(x: np.ndarray, mode: SpreadMode) -> tuple[float, float, float, int]:
    """Center band."""

    if mode == "p16-84":
        return _median_p16_p84(x)
    mu, sd, se, n = _mean_sd_se(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    s = sd if mode == "sd" else se
    if not np.isfinite(s):
        # e g se
        return mu, float("nan"), float("nan"), n
    return mu, mu - s, mu + s, n


def _center_band_log10(x: np.ndarray, mode: SpreadMode, *, eps: float = 1e-30) -> tuple[float, float, float, int]:
    """Center band log10."""

    xf = _finite_1d(x)
    n = int(xf.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    xf = np.maximum(xf, eps)
    y = np.log10(xf)
    c, lo, hi, n2 = _center_band(y, mode)
    if not np.isfinite(c):
        return float("nan"), float("nan"), float("nan"), n2
    center = float(10.0**c)
    lower = float(10.0**lo) if np.isfinite(lo) else float("nan")
    upper = float(10.0**hi) if np.isfinite(hi) else float("nan")
    return center, lower, upper, n2


def _aggregate_outcomes_by_T(outcomes: list[dict[str, Any]], *, spread: SpreadMode) -> dict[str, np.ndarray]:
    """Aggregate outcomes by."""

    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for o in outcomes:
        Ti = _maybe_float(o.get("temperature_start"), default=_maybe_float(o.get("temperature")))
        if np.isfinite(Ti):
            groups[float(Ti)].append(o)

    Ts = np.array(sorted(groups.keys()), dtype=float)
    nrep = np.zeros_like(Ts)

    def _nanarr() -> np.ndarray:
        return np.full_like(Ts, np.nan)

    D = _nanarr()
    Dlo = _nanarr()
    Dhi = _nanarr()
    rho = _nanarr()
    rholo = _nanarr()
    rhohi = _nanarr()
    pe = _nanarr()
    pelo = _nanarr()
    pehi = _nanarr()
    msdr = _nanarr()
    msdr_lo = _nanarr()
    msdr_hi = _nanarr()
    gH = _nanarr()
    gHlo = _nanarr()
    gHhi = _nanarr()
    gW = _nanarr()
    gWlo = _nanarr()
    gWhi = _nanarr()

    for i, Ti in enumerate(Ts):
        rows = groups[float(Ti)]
        nrep[i] = len(rows)

        Dvals = np.array([_maybe_float(r.get("D")) for r in rows], dtype=float)
        # negative noise space
        Dc, Dl, Du, nn = _center_band_log10(np.maximum(Dvals, 0.0), spread)
        D[i] = Dc
        if nn > 1:
            Dlo[i] = Dl
            Dhi[i] = Du

        rhos = np.array([
            _maybe_float(r.get("density_mean"), default=_maybe_float(r.get("density"))) for r in rows
        ], dtype=float)
        rc, rl, ru, nn = _center_band(rhos, spread)
        rho[i] = rc
        if nn > 1:
            rholo[i] = rl
            rhohi[i] = ru

        pes = np.array([
            _maybe_float(r.get("pe_mean"), default=_maybe_float(r.get("pe"))) for r in rows
        ], dtype=float)
        pc, pl, pu, nn = _center_band(pes, spread)
        pe[i] = pc
        if nn > 1:
            pelo[i] = pl
            pehi[i] = pu

        msd = np.array([
            _maybe_float(r.get("msd_rms_last"), default=_maybe_float(r.get("msd_rms"))) for r in rows
        ], dtype=float)
        mc, ml, mu, nn = _center_band(msd, spread)
        msdr[i] = mc
        if nn > 1:
            msdr_lo[i] = ml
            msdr_hi[i] = mu

        gHv = np.array([_maybe_float(r.get("gr_peak_height")) for r in rows], dtype=float)
        hc, hl, hu, nn = _center_band(gHv, spread)
        gH[i] = hc
        if nn > 1:
            gHlo[i] = hl
            gHhi[i] = hu

        gWv = np.array([_maybe_float(r.get("gr_peak_fwhm")) for r in rows], dtype=float)
        wc, wl, wu, nn = _center_band(gWv, spread)
        gW[i] = wc
        if nn > 1:
            gWlo[i] = wl
            gWhi[i] = wu

    return {
        "T": Ts,
        "nrep": nrep,
        "D": D,
        "D_lo": Dlo,
        "D_hi": Dhi,
        "rho": rho,
        "rho_lo": rholo,
        "rho_hi": rhohi,
        "pe": pe,
        "pe_lo": pelo,
        "pe_hi": pehi,
        "msdrms": msdr,
        "msdrms_lo": msdr_lo,
        "msdrms_hi": msdr_hi,
        "gH": gH,
        "gH_lo": gHlo,
        "gH_hi": gHhi,
        "gW": gW,
        "gW_lo": gWlo,
        "gW_hi": gWhi,
    }


def _plot_mean_band(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    label: Optional[str] = None,
    show_band: bool = True,
) -> None:
    """Mean band."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return
    ax.plot(x[m], y[m], "o-", label=label)
    if show_band:
        mb = m & np.isfinite(lo) & np.isfinite(hi)
        if np.any(mb):
            ax.fill_between(x[mb], lo[mb], hi[mb], alpha=0.2)


def plot_autotune_results(
    json_path: Path,
    out_path: Path,
    *,
    title: Optional[str] = None,
    dpi: int = 600,
    spread: SpreadMode = "sd",
    show_replicates: bool = False,
) -> None:
    """Autotune results."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = _require_positive_integer(dpi, context="dpi")
    _apply_publication_style()

    data = json.loads(Path(json_path).read_text())
    if not isinstance(data, Mapping):
        raise ValueError("Autotune plotting input must be a JSON object")
    canonical_units = _uses_canonical_report_units(data)

    units_style, time_unit_ps = _units_from_results(dict(data))

    # New results are stored in canonical A^2/ps.  Legacy files retain their
    # native-time conversion for backwards-compatible plotting.
    D_scale = float(1.0)
    D_label = "D (Å²/ps)" if canonical_units else "D (length²/time)"
    if not canonical_units:
        D_scale, D_label = _legacy_diffusion_plot_scale(
            units_style=units_style,
            time_unit_ps=time_unit_ps,
        )
    rho_label = "density"
    density_unit = "g/cm³" if canonical_units else _thermo_unit_label("Density", units_style)
    if density_unit:
        rho_label = f"density ({density_unit})"
    pe_unit = _thermo_unit_label("PotEng", units_style, canonical=canonical_units)
    pe_label = f"potential energy ({pe_unit})" if pe_unit else "potential energy"
    rms_unit = "Å" if canonical_units else (_native_length_unit_label(units_style) or "distance units")

    # tm scan series
    # tm scan series
    # tm
    tm = data.get("tm_scan", {}) or {}
    outcomes = tm.get("outcomes", []) or []

    if outcomes:
        agg = _aggregate_outcomes_by_T(outcomes, spread=spread)
        T = agg["T"]
        nrep = agg["nrep"]
        D = agg["D"]
        Dlo = agg["D_lo"]
        Dhi = agg["D_hi"]
        rho = agg["rho"]
        rholo = agg["rho_lo"]
        rhohi = agg["rho_hi"]
        pe = agg["pe"]
        pelo = agg["pe_lo"]
        pehi = agg["pe_hi"]
        msdr = agg["msdrms"]
        msdrlo = agg["msdrms_lo"]
        msdrhi = agg["msdrms_hi"]
        gH = agg["gH"]
        gHlo = agg["gH_lo"]
        gHhi = agg["gH_hi"]
        gW = agg["gW"]
        gWlo = agg["gW_lo"]
        gWhi = agg["gW_hi"]
    else:
        # schema
        T = np.array([_maybe_float(x) for x in tm.get("temps", [])], dtype=float)
        D = np.maximum(np.array([_maybe_float(x) for x in tm.get("D", [])], dtype=float), 0.0)
        Dlo = np.full_like(D, np.nan)
        Dhi = np.full_like(D, np.nan)
        rho = np.full_like(D, np.nan)
        rholo = np.full_like(D, np.nan)
        rhohi = np.full_like(D, np.nan)
        pe = np.full_like(D, np.nan)
        pelo = np.full_like(D, np.nan)
        pehi = np.full_like(D, np.nan)
        msdr = np.full_like(D, np.nan)
        msdrlo = np.full_like(D, np.nan)
        msdrhi = np.full_like(D, np.nan)
        gH = np.full_like(D, np.nan)
        gHlo = np.full_like(D, np.nan)
        gHhi = np.full_like(D, np.nan)
        gW = np.full_like(D, np.nan)
        gWlo = np.full_like(D, np.nan)
        gWhi = np.full_like(D, np.nan)
        nrep = np.zeros_like(D)

    order = np.argsort(T)
    T = T[order]
    D = D[order]
    Dlo = Dlo[order]
    Dhi = Dhi[order]
    rho = rho[order]
    rholo = rholo[order]
    rhohi = rhohi[order]
    pe = pe[order]
    pelo = pelo[order]
    pehi = pehi[order]
    msdr = msdr[order]
    msdrlo = msdrlo[order]
    msdrhi = msdrhi[order]
    gH = gH[order]
    gHlo = gHlo[order]
    gHhi = gHhi[order]
    gW = gW[order]
    gWlo = gWlo[order]
    gWhi = gWhi[order]
    nrep = nrep[order]

    rec = data.get("recommendation", {}) or {}
    Tm = _maybe_float(rec.get("Tm_operational"), default=_maybe_float(tm.get("Tm_estimate", {}).get("Tm")))
    T_liquid = _maybe_float(rec.get("T_liquid"), default=_maybe_float(tm.get("Tm_estimate", {}).get("T_liquid")))
    Thigh = _maybe_float(rec.get("T_high"), default=_maybe_float(data.get("highT", {}).get("T_high")))

    chosen_rate_time = rec.get("cooling_rate_K_per_time", None)
    chosen_rate_time = _maybe_float(chosen_rate_time) if chosen_rate_time is not None else float("nan")
    chosen_rate_ps = rec.get("cooling_rate_K_per_ps", None)
    chosen_rate_ps = _maybe_float(chosen_rate_ps) if chosen_rate_ps is not None else float("nan")
    if not np.isfinite(chosen_rate_ps) and np.isfinite(chosen_rate_time) and time_unit_ps is not None:
        chosen_rate_ps = float(chosen_rate_time) / float(time_unit_ps)

    # rate scan series
    # rate scan series
    # rate scan
    rate_scan = data.get("rate_scan", {}) or {}
    rate_points = rate_scan.get("rates", []) or []
    if rate_points:
        rate_plot_x, rate_xlabel = _rate_scan_coordinates(
            rate_points,
            time_unit_ps=time_unit_ps,
        )
    else:
        rate_plot_x = np.asarray([], dtype=float)
        rate_xlabel = "cooling rate (K/ps)"
    rho_r: list[float] = []
    rho_r_lo: list[float] = []
    rho_r_hi: list[float] = []
    rho_r_n: list[int] = []
    metric_name: Optional[str] = None
    metric_mu: list[float] = []

    # replicate scatter storage
    rate_rep_x: list[float] = []
    rate_rep_rho: list[float] = []

    for rr_idx, rr in enumerate(rate_points):
        # replicate between replica
        reps = rr.get("replicates", None)
        if isinstance(reps, list) and len(reps) > 0:
            dens_vals = np.array([_maybe_float(r.get("density")) for r in reps], dtype=float)
            c, lo, hi, nn = _center_band(dens_vals, spread)
            rate_coord = float(rate_plot_x[rr_idx])
            if np.isfinite(rate_coord):
                for r in reps:
                    rate_rep_x.append(rate_coord)
                    rate_rep_rho.append(_maybe_float(r.get("density")))
        else:
            c = _maybe_float(rr.get("density_mean"))
            lo = float("nan")
            hi = float("nan")
            nn = 1

        rho_r.append(c)
        rho_r_lo.append(lo)
        rho_r_hi.append(hi)
        rho_r_n.append(nn)

        if metric_name is None:
            mm = rr.get("metrics_mean", None)
            if isinstance(mm, dict) and len(mm) > 0:
                keys = sorted(mm.keys())
                pref = [k for k in keys if k.startswith("coord_") and k.endswith("_mean")]
                metric_name = pref[0] if pref else keys[0]

    if metric_name is not None:
        for rr in rate_points:
            mm = rr.get("metrics_mean", {}) or {}
            # rate replica metrics
            val = _maybe_float(mm.get(metric_name))
            metric_mu.append(val)

    rho_r_arr = np.asarray(rho_r, dtype=float)
    rho_r_lo_arr = np.asarray(rho_r_lo, dtype=float)
    rho_r_hi_arr = np.asarray(rho_r_hi, dtype=float)
    metric_mu_arr = np.asarray(metric_mu, dtype=float) if metric_mu else None

    use_rate_ps = rate_xlabel == "cooling rate (K/ps)"
    rate_x = rate_plot_x
    chosen_rate_x = chosen_rate_ps if use_rate_ps else chosen_rate_time

    # size scan series
    # size scan series
    # size scan
    size_scan = data.get("size_scan", {}) or {}
    size_points = size_scan.get("sizes", []) or []
    size_xlabel = "N atoms" if any((s.get("n_atoms") is not None) for s in size_points) else "box multiplier"
    mult: list[float] = []
    rho_s: list[float] = []
    rho_s_lo: list[float] = []
    rho_s_hi: list[float] = []

    size_rep_x: list[float] = []
    size_rep_rho: list[float] = []

    for s in size_points:
        mlt = _maybe_float(s.get("n_atoms")) if s.get("n_atoms") is not None else _maybe_float(s.get("multiplier"))
        mult.append(mlt)
        reps = s.get("replicates", None)
        if isinstance(reps, list) and len(reps) > 0:
            dens_vals = np.array([_maybe_float(r.get("density")) for r in reps], dtype=float)
            c, lo, hi, nn = _center_band(dens_vals, spread)
            if np.isfinite(mlt):
                for r in reps:
                    size_rep_x.append(mlt)
                    size_rep_rho.append(_maybe_float(r.get("density")))
        else:
            c = _maybe_float(s.get("density_mean"))
            lo = float("nan")
            hi = float("nan")
        rho_s.append(c)
        rho_s_lo.append(lo)
        rho_s_hi.append(hi)

    mult_arr = np.asarray(mult, dtype=float)
    rho_s_arr = np.asarray(rho_s, dtype=float)
    rho_s_lo_arr = np.asarray(rho_s_lo, dtype=float)
    rho_s_hi_arr = np.asarray(rho_s_hi, dtype=float)

    # high series panel
    # high series panel
    # high t
    highT = data.get("highT", {}) or {}
    high_out = highT.get("outcomes", []) or []
    hD: list[float] = []
    for o in high_out:
        hD.append(_maybe_float(o.get("D")))
    hD_arr = np.asarray(hD, dtype=float)
    hD_arr_plot = np.maximum(hD_arr, 0.0) * float(D_scale)
    hD_c, hD_lo, hD_hi, hD_n = _center_band_log10(np.maximum(hD_arr_plot, 0.0), spread)

    has_plottable_data = any(
        bool(np.any(np.isfinite(xv) & np.isfinite(yv)))
        for xv, yv in (
            (T, D),
            (T, rho),
            (T, pe),
            (T, msdr),
            (T, gH),
            (T, gW),
            (rate_x, rho_r_arr),
            (mult_arr, rho_s_arr),
            (np.arange(hD_arr_plot.size, dtype=float), hD_arr_plot),
        )
        if np.asarray(xv).shape == np.asarray(yv).shape
    )
    if not has_plottable_data:
        raise ValueError(
            "autotune_results.json contains no finite plottable scan or high-temperature data"
        )

    # plot
    # plot
    # fig axes plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)

    def _decorate_T_lines(ax):
        if np.isfinite(Tm):
            ax.axvline(Tm, linestyle="--")
        if np.isfinite(T_liquid):
            ax.axvline(T_liquid, linestyle="-.")
        if np.isfinite(Thigh):
            ax.axvline(Thigh, linestyle=":")

    # d t
    ax = axes[0, 0]
    _plot_mean_band(
        ax,
        T,
        _diffusion_for_plot(D, scale=float(D_scale), zero_below=0.1),
        _diffusion_for_plot(Dlo, scale=float(D_scale), zero_below=0.1),
        _diffusion_for_plot(Dhi, scale=float(D_scale), zero_below=0.1),
        show_band=True,
    )
    if show_replicates and outcomes:
        Tr = np.array([_maybe_float(o.get("temperature_start")) for o in outcomes], dtype=float)
        Dr = _diffusion_for_plot(
            np.array([_maybe_float(o.get("D")) for o in outcomes], dtype=float),
            scale=float(D_scale),
            zero_below=0.1,
        )
        m = np.isfinite(Tr) & np.isfinite(Dr)
        ax.plot(Tr[m], Dr[m], "o", alpha=0.25, markersize=3)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("T (K)")
    ax.set_ylabel(D_label)
    ax.set_title("diffusion")
    _decorate_T_lines(ax)

    # g peak height
    ax = axes[0, 1]
    if np.isfinite(gH).any():
        _plot_mean_band(ax, T, gH, gHlo, gHhi, show_band=True)
        if show_replicates and outcomes:
            Tr = np.array([_maybe_float(o.get("temperature_start")) for o in outcomes], dtype=float)
            yr = np.array([_maybe_float(o.get("gr_peak_height")) for o in outcomes], dtype=float)
            m = np.isfinite(Tr) & np.isfinite(yr)
            ax.plot(Tr[m], yr[m], "o", alpha=0.25, markersize=3)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("g(r) peak height")
        ax.set_title("structure: peak height")
        _decorate_T_lines(ax)
    else:
        ax.axis("off")

    # g peak width
    ax = axes[0, 2]
    if np.isfinite(gW).any():
        _plot_mean_band(ax, T, gW, gWlo, gWhi, show_band=True)
        if show_replicates and outcomes:
            Tr = np.array([_maybe_float(o.get("temperature_start")) for o in outcomes], dtype=float)
            yr = np.array([_maybe_float(o.get("gr_peak_fwhm")) for o in outcomes], dtype=float)
            m = np.isfinite(Tr) & np.isfinite(yr)
            ax.plot(Tr[m], yr[m], "o", alpha=0.25, markersize=3)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("g(r) peak FWHM")
        ax.set_title("structure: peak width")
        _decorate_T_lines(ax)
    else:
        ax.axis("off")

    # density t
    ax = axes[1, 0]
    if np.isfinite(rho).any():
        _plot_mean_band(ax, T, rho, rholo, rhohi, show_band=True)
        if show_replicates and outcomes:
            Tr = np.array([_maybe_float(o.get("temperature_start")) for o in outcomes], dtype=float)
            yr = np.array([_maybe_float(o.get("density_mean"), default=_maybe_float(o.get("density"))) for o in outcomes], dtype=float)
            m = np.isfinite(Tr) & np.isfinite(yr)
            ax.plot(Tr[m], yr[m], "o", alpha=0.25, markersize=3)
        ax.set_xlabel("T (K)")
        ax.set_ylabel(rho_label)
        ax.set_title("density")
        _decorate_T_lines(ax)
    else:
        ax.axis("off")

    # pe t
    ax = axes[1, 1]
    if np.isfinite(pe).any():
        _plot_mean_band(ax, T, pe, pelo, pehi, show_band=True)
        if show_replicates and outcomes:
            Tr = np.array([_maybe_float(o.get("temperature_start")) for o in outcomes], dtype=float)
            yr = np.array([_maybe_float(o.get("pe_mean"), default=_maybe_float(o.get("pe"))) for o in outcomes], dtype=float)
            m = np.isfinite(Tr) & np.isfinite(yr)
            ax.plot(Tr[m], yr[m], "o", alpha=0.25, markersize=3)
        ax.set_xlabel("T (K)")
        ax.set_ylabel(pe_label)
        ax.set_title("potential energy")
        _decorate_T_lines(ax)
    else:
        ax.axis("off")

    # rms displacement
    ax = axes[1, 2]
    if np.isfinite(msdr).any():
        _plot_mean_band(ax, T, msdr, msdrlo, msdrhi, show_band=True)
        if show_replicates and outcomes:
            Tr = np.array([_maybe_float(o.get("temperature_start")) for o in outcomes], dtype=float)
            yr = np.array([_maybe_float(o.get("msd_rms_last"), default=_maybe_float(o.get("msd_rms"))) for o in outcomes], dtype=float)
            m = np.isfinite(Tr) & np.isfinite(yr)
            ax.plot(Tr[m], yr[m], "o", alpha=0.25, markersize=3)
        ax.set_xlabel("T (K)")
        ax.set_ylabel(f"sqrt(MSD_end) ({rms_unit})")
        ax.set_title("mobility: RMS disp.")
        _decorate_T_lines(ax)
    else:
        ax.axis("off")

    # rate scan density
    ax = axes[2, 0]
    m = np.isfinite(rate_x) & (rate_x > 0) & np.isfinite(rho_r_arr)
    if np.any(m):
        idx = np.argsort(rate_x[m])
        rx = rate_x[m][idx]
        ry = rho_r_arr[m][idx]
        rlo = rho_r_lo_arr[m][idx]
        rhi = rho_r_hi_arr[m][idx]
        _plot_mean_band(ax, rx, ry, rlo, rhi, show_band=True)
        if show_replicates and len(rate_rep_x) > 0:
            xr = np.asarray(rate_rep_x, dtype=float)
            yr = np.asarray(rate_rep_rho, dtype=float)
            mm = np.isfinite(xr) & np.isfinite(yr) & (xr > 0)
            ax.plot(xr[mm], yr[mm], "o", alpha=0.25, markersize=3)
        ax.set_xscale("log")
        ax.set_xlabel(rate_xlabel)
        ax.set_ylabel(rho_label)
        ax.set_title("rate scan: density")
        if np.isfinite(chosen_rate_x) and chosen_rate_x > 0:
            ax.axvline(chosen_rate_x, linestyle="--")
    else:
        ax.axis("off")

    # rate structure metric
    ax = axes[2, 1]
    if metric_name is not None and metric_mu_arr is not None and np.isfinite(metric_mu_arr).any():
        m = np.isfinite(rate_x) & (rate_x > 0) & np.isfinite(metric_mu_arr)
        if np.any(m):
            idx = np.argsort(rate_x[m])
            rx = rate_x[m][idx]
            my = metric_mu_arr[m][idx]
            ax.plot(rx, my, "o-")
            ax.set_xscale("log")
            ax.set_xlabel(rate_xlabel)
            ax.set_ylabel(metric_name)
            ax.set_title("rate scan: structure")
            if np.isfinite(chosen_rate_x) and chosen_rate_x > 0:
                ax.axvline(chosen_rate_x, linestyle="--")
        else:
            ax.axis("off")
    elif np.isfinite(hD_arr_plot).any():
        # plot diffusion distribution
        # axis replica index
        rix = np.arange(1, int(np.isfinite(hD_arr_plot).sum()) + 1, dtype=float)
        vals = np.maximum(_finite_1d(hD_arr_plot), 1e-30)
        ax.plot(rix, vals, "o")
        ax.axhline(max(hD_c, 1e-30), linestyle="-")
        if np.isfinite(hD_lo) and np.isfinite(hD_hi) and hD_n > 1:
            ax.axhline(max(hD_lo, 1e-30), linestyle="--")
            ax.axhline(max(hD_hi, 1e-30), linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("highT replica")
        ax.set_ylabel(D_label)
        ax.set_title("highT: diffusion")
    else:
        ax.axis("off")

    # scan density multiplier
    ax = axes[2, 2]
    m = np.isfinite(mult_arr) & (mult_arr > 0) & np.isfinite(rho_s_arr)
    if np.any(m):
        idx = np.argsort(mult_arr[m])
        mx = mult_arr[m][idx]
        my = rho_s_arr[m][idx]
        mlo = rho_s_lo_arr[m][idx]
        mhi = rho_s_hi_arr[m][idx]
        _plot_mean_band(ax, mx, my, mlo, mhi, show_band=True)
        if show_replicates and len(size_rep_x) > 0:
            xr = np.asarray(size_rep_x, dtype=float)
            yr = np.asarray(size_rep_rho, dtype=float)
            mm = np.isfinite(xr) & np.isfinite(yr) & (xr > 0)
            ax.plot(xr[mm], yr[mm], "o", alpha=0.25, markersize=3)
        ax.set_xscale("log")
        ax.set_xlabel(size_xlabel)
        ax.set_ylabel(rho_label)
        ax.set_title("size scan")
    else:
        ax.axis("off")

    # title
    if title is None:
        title = f"vitriflow calibration: {Path(json_path).name}"
    if np.isfinite(Tm):
        title += f" | Tm≈{Tm:.0f}K"
    if np.isfinite(T_liquid):
        title += f" | T_liquid≈{T_liquid:.0f}K"
    if np.isfinite(Thigh):
        title += f" | T_high≈{Thigh:.0f}K"
    if np.isfinite(chosen_rate_x):
        title += f" | rate≈{chosen_rate_x:g}" + (" K/ps" if use_rate_ps else "")
    if np.isfinite(nrep).any():
        nrep_f = nrep[np.isfinite(nrep) & (nrep > 0)]
        if nrep_f.size > 0:
            title += f" | tm reps≈{int(np.nanmedian(nrep_f))}"
    if spread is not None:
        title += f" | spread={spread}"

    fig.suptitle(title)

    out_path = Path(out_path)
    _style_and_save_figure(fig, out_path, dpi=int(dpi))




def _plot_analysis_results_without_convergence(
    data: Mapping[str, Any],
    prod: Mapping[str, Any],
    json_path: Path,
    out_path: Path,
    *,
    title: Optional[str],
    dpi: int,
    show_boxes: bool,
    max_pages: Optional[int],
) -> None:
    """Plot standalone analysis data even when production convergence is absent.

    Analysis-only datasets may intentionally carry advisory descriptor-set
    convergence diagnostics rather than the production ``familywise`` report.
    Plotting must still expose the analysed structures, scalar descriptors,
    stored distributions and graph sidecar coverage instead of failing.
    """

    import csv
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _finite(v: Any) -> float:
        try:
            x = float(v)
        except Exception:
            return float("nan")
        return x if math.isfinite(x) else float("nan")

    boxes = [dict(b) for b in list(prod.get("boxes", []) or []) if isinstance(b, Mapping)]
    n_boxes = int(len(boxes))
    conv = dict(prod.get("convergence", {}) or {}) if isinstance(prod.get("convergence", {}), Mapping) else {}
    groups = dict(conv.get("groups", {}) or {}) if isinstance(conv.get("groups", {}), Mapping) else {}

    pages: list[tuple[str, Any]] = []
    # Keep analysis-only fallback plotting compact by default.  Users can still
    # request more pages explicitly with --max-pages, but an analysis JSON with
    # hundreds of graph-rule descriptors should never appear to hang while
    # attempting to render every convenience scalar and curve.
    page_budget = 6
    if max_pages is not None:
        if int(max_pages) <= 0:
            raise ValueError("max_pages must be a positive integer when provided")
        # max_pages means extra data pages beyond the mandatory summary page.
        page_budget = 1 + int(max_pages)

    def _page_budget_exhausted() -> bool:
        return len(pages) >= page_budget

    def _add_page(name: str, fig: Any) -> None:
        # Do not retain pages beyond the requested budget.
        if len(pages) < page_budget:
            pages.append((str(name), fig))
        else:
            try:
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception:
                pass

    def _fig_summary():
        fig = plt.figure(figsize=(8.0, 5.0), dpi=int(dpi))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        status = str(data.get("status", "unknown"))
        schema = str(data.get("schema", "unknown"))
        filtering = data.get("filtering", {}) if isinstance(data.get("filtering", {}), Mapping) else {}
        convergence_display = _production_convergence_display(prod, conv)
        lines = [
            title or f"VitriFlow analysis summary: {Path(json_path).name}",
            "",
            f"schema: {schema}",
            f"status: {status}",
            f"boxes analysed: {n_boxes}",
            f"accepted/rejected fields: {prod.get('n_boxes_accepted', n_boxes)}/{prod.get('n_boxes_rejected', 0)}",
            f"would reject under analysis settings: {data.get('n_boxes_would_be_rejected', 0)}",
            f"filtering semantics: {filtering.get('semantics', 'not declared')}",
            f"embedded structures: {data.get('embed_structures', 'not declared')}",
            "",
            "Descriptor-set convergence is advisory for analysis-only inputs.",
            f"convergence assessment: {convergence_display['label']}",
        ]
        if groups:
            lines.append("")
            lines.append("descriptor sets:")
            for name in ("short", "medium", "long"):
                g = groups.get(name, {}) if isinstance(groups.get(name, {}), Mapping) else {}
                items = g.get("items", []) if isinstance(g.get("items", []), list) else []
                passed = g.get("passed", None)
                status_g = g.get("status", "not_evaluated")
                lines.append(f"  {name}: status={status_g}, passed={passed}, n_items={len(items)}")
        if conv.get("reason"):
            lines.append("")
            lines.append(f"convergence note: {conv.get('reason')}")
        ax.text(0.02, 0.98, "\n".join(str(x) for x in lines), va="top", ha="left", family="monospace", fontsize=9)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
        return fig

    _add_page("analysis_summary", _fig_summary())

    # Scalar descriptor trends from boxes[].density and boxes[].metrics.
    scalar_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for idx, b in enumerate(boxes, start=1):
        bid = _finite(b.get("box_id", b.get("box", idx)))
        if not math.isfinite(bid):
            bid = float(idx)
        dens = _finite(b.get("density"))
        if math.isfinite(dens):
            scalar_values["density"].append((bid, dens))
        metrics = b.get("metrics", {}) if isinstance(b.get("metrics", {}), Mapping) else {}
        for key, val in metrics.items():
            fv = _finite(val)
            if math.isfinite(fv):
                scalar_values[str(key)].append((bid, fv))

    scalar_items = [(k, v) for k, v in sorted(scalar_values.items()) if len(v) >= 1]
    if scalar_items:
        # Put density and common topology scalars first, then deterministic alphabetical order.
        priority = {"density": -10, "ring_mean_size": -9, "ring_count": -8, "component_count": -7}
        scalar_items.sort(key=lambda kv: (priority.get(kv[0], 0), kv[0]))
        per_page = 6
        for page_idx in range(0, len(scalar_items), per_page):
            if _page_budget_exhausted():
                break
            chunk = scalar_items[page_idx : page_idx + per_page]

            def _make_scalar_fig(chunk=chunk, page_idx=page_idx):
                n = len(chunk)
                fig = plt.figure(figsize=(8.0, 1.8 * max(1, n)), dpi=int(dpi))
                axes = fig.subplots(n, 1, squeeze=False).reshape(-1)
                for ax, (name, vals) in zip(axes, chunk):
                    arr = np.asarray(vals, dtype=float)
                    order = np.argsort(arr[:, 0])
                    arr = arr[order]
                    ax.plot(arr[:, 0], arr[:, 1], "o-" if show_boxes else "o")
                    if arr.shape[0] >= 2:
                        mu = float(np.mean(arr[:, 1]))
                        ax.axhline(mu, linestyle="--", linewidth=1.0)
                    ax.set_xlabel("box")
                    ax.set_ylabel(str(name))
                    ax.set_title(str(name))
                fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
                return fig

            _add_page(f"scalar_metrics_{page_idx//per_page+1:02d}", _make_scalar_fig())

    # Stored distributions: plot ensemble mean curves directly from boxes[].distributions.
    def _dist_names(kind: str) -> list[str]:
        names: set[str] = set()
        for b in boxes:
            d = b.get("distributions", {}) if isinstance(b.get("distributions", {}), Mapping) else {}
            fam = d.get(kind, {}) if isinstance(d.get(kind, {}), Mapping) else {}
            names.update(str(k) for k in fam.keys())
        return sorted(names)

    def _curve_payload(box: Mapping[str, Any], kind: str, name: str) -> Mapping[str, Any]:
        d = box.get("distributions", {}) if isinstance(box.get("distributions", {}), Mapping) else {}
        fam = d.get(kind, {}) if isinstance(d.get(kind, {}), Mapping) else {}
        payload = fam.get(name, {}) if isinstance(fam.get(name, {}), Mapping) else {}
        return payload

    def _fig_curve(kind: str, name: str):
        x_key = "r" if kind == "gr" else ("q" if kind == "sq" else "x")
        y_key = "g" if kind == "gr" else ("s" if kind == "sq" else "cdf")
        x_ref = None
        curves = []
        for b in boxes:
            payload = _curve_payload(b, kind, name)
            x = np.asarray(payload.get(x_key, []), dtype=float)
            y = np.asarray(payload.get(y_key, []), dtype=float)
            if x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.size != y.size:
                continue
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                continue
            if x_ref is None:
                x_ref = x
                curves.append(y)
            elif x.shape == x_ref.shape and float(np.max(np.abs(x - x_ref))) < 1e-8:
                curves.append(y)
            else:
                curves.append(np.interp(x_ref, x, y))
        fig = plt.figure(figsize=(6.5, 4.0), dpi=int(dpi))
        ax = fig.add_subplot(1, 1, 1)
        if x_ref is not None and curves:
            mat = np.vstack(curves)
            mu = np.mean(mat, axis=0)
            ax.plot(x_ref, mu, label="ensemble mean")
            if show_boxes:
                for row in mat:
                    ax.plot(x_ref, row, alpha=0.25, linewidth=0.8)
            if mat.shape[0] >= 2:
                se = np.std(mat, axis=0, ddof=1) / np.sqrt(float(mat.shape[0]))
                ax.fill_between(x_ref, mu - se, mu + se, alpha=0.18, linewidth=0.0)
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No finite curves available", ha="center", va="center")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key if kind in {"gr", "sq"} else "CDF")
        ax.set_title(f"{kind}: {name}")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
        return fig

    for kind in ("bondlen", "angle", "coord", "void", "gr", "sq"):
        if _page_budget_exhausted():
            break
        for name in _dist_names(kind):
            if _page_budget_exhausted():
                break
            _add_page(f"{kind}_{name}", _fig_curve(kind, name))

    # Optional graph sidecar audit coverage page.
    side = data.get("graph_metric_by_rule", {}) if isinstance(data.get("graph_metric_by_rule", {}), Mapping) else {}
    side_path = side.get("path", None)
    if side_path:
        candidate = Path(json_path).parent / str(side_path)
        if candidate.exists():
            counts: dict[str, int] = defaultdict(int)
            try:
                with candidate.open("r", newline="", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fam = str(row.get("metric_family", "unknown") or "unknown")
                        counts[fam] += 1
            except Exception:
                counts = {}
            if counts and not _page_budget_exhausted():
                fig = plt.figure(figsize=(7.0, 4.0), dpi=int(dpi))
                ax = fig.add_subplot(1, 1, 1)
                names = sorted(counts)
                vals = [counts[n] for n in names]
                ax.bar(np.arange(len(names)), vals)
                ax.set_xticks(np.arange(len(names)))
                ax.set_xticklabels(names, rotation=45, ha="right")
                ax.set_ylabel("rows")
                ax.set_title("graph_metric_by_rule.csv coverage")
                fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
                _add_page("graph_metric_coverage", fig)

    _save_plot_pages(
        pages,
        Path(out_path),
        dpi=int(dpi),
        name_cleaner=lambda nm: str(nm).replace(" ", "_").replace("/", "-"),
    )

def plot_production_results(
    json_path: Path,
    out_path: Path,
    *,
    title: Optional[str] = None,
    dpi: int = 600,
    show_boxes: bool = False,
    max_pages: Optional[int] = None,
) -> None:
    """Production results."""

    import math
    from statistics import NormalDist

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = _require_positive_integer(dpi, context="dpi")
    _apply_publication_style()

    if max_pages is not None:
        max_pages = _require_positive_integer(max_pages, context="max_pages")

    data, prod = _prepare_production_plot_payload(json_path)
    canonical_units = _uses_canonical_report_units(data)

    raw_boxes = prod.get("boxes", []) or []
    if not isinstance(raw_boxes, list) or len(raw_boxes) < 1:
        raise RuntimeError("Production section contains no boxes.")
    boxes = _normalise_production_plot_boxes(raw_boxes)

    conv = prod.get("convergence", {}) or {}
    conv_has_report_payload = (
        isinstance(conv, dict)
        and "familywise" in conv
        and (bool((conv.get("scalars", {}) or {})) or bool((conv.get("distributions", {}) or {})))
    )
    conv_status = str((conv or {}).get("status", "") if isinstance(conv, dict) else "").strip().lower()
    if (not isinstance(conv, dict)) or "familywise" not in conv or (conv_status in {"skipped", "not_evaluated"} and not conv_has_report_payload):
        _plot_analysis_results_without_convergence(
            data,
            {**prod, "boxes": boxes},
            Path(json_path),
            Path(out_path),
            title=title,
            dpi=int(dpi),
            show_boxes=bool(show_boxes),
            max_pages=max_pages,
        )
        return

    # convergence reports ensembles
    conv_md = prod.get("convergence_md", None)
    if not isinstance(conv_md, dict):
        conv_md = None
    conv_dft = prod.get("convergence_dft", None)
    if not isinstance(conv_dft, dict):
        conv_dft = None

    # box metrics distributions
    dft_final_ids_list = [
        _strict_production_plot_box_id(
            value,
            context=f"production.boxes_dft_final[{idx}]",
        )
        for idx, value in enumerate(prod.get("boxes_dft_final", []) or [])
    ]
    if len(dft_final_ids_list) != len(set(dft_final_ids_list)):
        raise ValueError("production.boxes_dft_final contains duplicate box identifier")
    dft_final_ids = set(dft_final_ids_list)
    dft_boxes = []
    for b in boxes:
        d = b.get("dft_opt", None)
        if not isinstance(d, dict):
            continue
        # prefer box acceptance
        if dft_final_ids:
            if int(b.get("box_id", -1)) not in dft_final_ids:
                continue
        else:
            if not bool(d.get("accepted", False)):
                continue
        try:
            dft_boxes.append(
                {
                    "box_id": int(b.get("box_id", -1)),
                    "density": float(d.get("density", float("nan"))),
                    "metrics": d.get("metrics", {}) or {},
                    "distributions": d.get("distributions", {}) or {},
                }
            )
        except Exception:
            continue

    fw = conv.get("familywise", {}) or {}
    alpha_test = float(fw.get("alpha_per_test", float("nan")))
    if not (math.isfinite(alpha_test) and alpha_test > 0.0 and alpha_test < 1.0):
        alpha_test = 0.05

    bounded_ci_method = str(fw.get("bounded_ci_method", "t")).strip().lower()
    if bounded_ci_method not in ("t", "empirical_bernstein", "hoeffding"):
        bounded_ci_method = "t"

    units_style = str((data.get("units", {}) or {}).get("lammps_units", "")).strip().lower()

    def _distance_unit_label() -> str:
        if canonical_units:
            return "Å"
        return _native_length_unit_label(units_style) or "(distance units)"

    def _density_unit_label() -> str:
        if canonical_units:
            return "g/cm³"
        return _thermo_unit_label("Density", units_style) or "(density units)"

    def _critical_value(n: int, alpha: float) -> tuple[float, str]:
        a = float(min(1.0, max(0.0, alpha)))
        if int(n) < 2:
            return float("inf"), "n<2"
        try:
            from scipy.stats import t as _t  # type: ignore

            crit = float(_t.ppf(1.0 - a / 2.0, df=int(n) - 1))
            if math.isfinite(crit):
                return crit, "t"
        except Exception:
            pass
        crit = float(NormalDist().inv_cdf(1.0 - a / 2.0))
        return crit, "z"

    # pretty slugged gr
    # payload includes human
    gr_label_map: dict[str, str] = {}
    try:
        g0 = ((boxes[0].get("distributions", {}) or {}).get("gr", {}) or {})
        for k, v in g0.items():
            lab = (v or {}).get("label", None)
            if lab is not None:
                gr_label_map[str(k)] = str(lab)
    except Exception:
        gr_label_map = {}

    def _clean_name(nm: str) -> str:
        s = str(nm)
        if s.startswith("gr_") and s in gr_label_map:
            lab = str(gr_label_map.get(s, s))
            return f"g(r): {lab.replace('-', '–')}"
        s = s.replace("bondlen_", "Bond length: ")
        s = s.replace("angle_", "Angle: ")
        s = s.replace("coord_", "Coordination: ")
        s = s.replace("gr_", "g(r): ")
        s = s.replace("_", " ")
        s = s.replace("-", "–")
        return s

    # matrices trend
    # matrices trend
    # n

    N = int(len(boxes))

    # helper unbiased sums
    def _mean_sd_from_sums(sum_x: np.ndarray, sum_x2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        mu = sum_x / float(n)
        if n < 2:
            return mu, np.full_like(mu, np.nan, dtype=float)
        var = (sum_x2 - (sum_x * sum_x) / float(n)) / float(n - 1)
        var = np.maximum(var, 0.0)
        return mu, np.sqrt(var)

    # extract convergence present
    spec = (conv.get("convergence_spec_effective") if isinstance(conv, dict) else None) or prod.get("convergence_spec", {}) or {}
    bond_names = list(spec.get("bondlen_names", []))
    angle_names = list(spec.get("angle_names", []))
    coord_names = list(spec.get("coord_names", []))
    ring_keys = list(spec.get("ring_keys", []))
    gr_labels = list(spec.get("gr_labels", []))
    sq_labels = list(spec.get("sq_labels", []))
    void_names = list(spec.get("void_names", []))
    has_ring_mean = bool(spec.get("ring_has_mean_size", False))

    # Per-box distributions are preferred.  Analysis-only results may instead
    # carry authoritative ensemble-level CDF sidecars; in that case the plotting
    # code falls back to the hydrated convergence.distributions entries below.
    if "distributions" not in boxes[0] and not bool((conv.get("distributions", {}) or {})):
        raise RuntimeError(
            "Per-box distributions were not stored and no ensemble CDF sidecar/convergence distribution was available. "
            "Enable store_distributions or keep ensemble_cdfs.json next to analysis_results.json."
        )

    # density
    dens = np.asarray([float(b.get("density", float("nan"))) for b in boxes], dtype=float)
    if dens.size != N or not np.all(np.isfinite(dens)):
        raise RuntimeError(
            "Production density is missing or non-finite for one or more boxes"
        )
    dens_abs = float(((conv.get("scalars", {}) or {}).get("density", {}) or {}).get("abs_tol", 0.0))
    dens_rel = float(((conv.get("scalars", {}) or {}).get("density", {}) or {}).get("rel_tol", 0.0))

    # ring entries convergence
    ring_abs = 0.0
    ring_rel = 0.0
    if ring_keys:
        rep0 = ((conv.get("scalars", {}) or {}).get(ring_keys[0], {}) or {})
        if not isinstance(rep0, dict) or len(rep0) == 0:
            rep0 = ((conv.get("distributions", {}) or {}).get("ring", {}) or {})
        ring_abs = float(rep0.get("abs_tol", 0.0))
        ring_rel = float(rep0.get("rel_tol", 0.0))

    ring_mat = None
    if ring_keys:
        ring_mat = np.zeros((N, len(ring_keys)), dtype=float)
        for i, b in enumerate(boxes):
            m = b.get("metrics", {}) or {}
            for j, k in enumerate(ring_keys):
                ring_mat[i, j] = float(m.get(k, float("nan")))
        if not np.all(np.isfinite(ring_mat)):
            raise RuntimeError(
                "Declared ring-fraction metrics are missing or non-finite for one or more boxes"
            )

    ring_mean_abs = float(((conv.get("scalars", {}) or {}).get("ring_mean_size", {}) or {}).get("abs_tol", 0.0))
    ring_mean_rel = float(((conv.get("scalars", {}) or {}).get("ring_mean_size", {}) or {}).get("rel_tol", 0.0))
    ring_mean_vec = None
    if has_ring_mean:
        ring_mean_vec = np.asarray([float((b.get("metrics", {}) or {}).get("ring_mean_size", float("nan"))) for b in boxes], dtype=float)
        if not np.all(np.isfinite(ring_mean_vec)):
            raise RuntimeError(
                "Declared ring_mean_size is missing or non-finite for one or more boxes"
            )

    # cdf curves
    curve_mats: list[dict[str, Any]] = []

    def _plot_stack_continuous_cdfs(src_boxes: Sequence[Mapping[str, Any]], kind: str, name: str, *, xkey: str = "x") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        payloads: list[Mapping[str, Any]] = []
        for idx, b in enumerate(src_boxes):
            fam = ((b.get("distributions", {}) or {}).get(kind, {}) or {})
            payload = fam.get(name, None) if isinstance(fam, Mapping) else None
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"Missing {kind} CDF '{name}' in box {idx + 1}")
            payloads.append(payload)
        return _align_plot_cdf_payloads(payloads, xkey=xkey)

    def _plot_stack_coord_cdfs(src_boxes: Sequence[Mapping[str, Any]], name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        payloads: list[Mapping[str, Any]] = []
        for idx, b in enumerate(src_boxes):
            fam = ((b.get("distributions", {}) or {}).get("coord", {}) or {})
            payload = fam.get(name, None) if isinstance(fam, Mapping) else None
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"Missing coordination CDF '{name}' in box {idx + 1}")
            payloads.append(payload)
        x_common, mat, meta = _align_plot_cdf_payloads(
            payloads,
            xkey="x",
            allow_implicit_integer_axis=True,
        )
        integer = np.arange(int(x_common.size), dtype=float)
        is_integer_support = bool(
            x_common.shape == integer.shape
            and np.allclose(
                x_common,
                integer,
                rtol=0.0,
                atol=float(_PLOT_CDF_ROUNDOFF_ATOL),
            )
        )
        meta = {
            **dict(meta),
            "grid_source": (
                "ensemble_common_integer_grid"
                if is_integer_support
                else "ensemble_common_explicit_coordination_grid"
            ),
            "coordination_support": (
                "integer" if is_integer_support else "fractional_or_nonuniform"
            ),
        }
        return x_common, mat, meta

    def _stack_cdf(kind: str, name: str, xkey: str = "x") -> None:
        # Continuous CDFs may have been produced on box-specific adaptive grids.
        # Build the same explicit ensemble grid used by the convergence checker
        # instead of requiring the first box's grid to be universal.
        drep = (conv.get("distributions", {}) or {}).get(name, None)
        if not isinstance(drep, dict):
            drep = {}
        abs_tol = float(drep.get("abs_tol", 0.0))
        rel_tol = float(drep.get("rel_tol", 0.0))
        try:
            x0, mats, meta = _plot_stack_continuous_cdfs(boxes, kind, name, xkey=xkey)
        except Exception as exc:
            # A malformed stored box payload must not be hidden by substituting
            # an ensemble mean.  Report-only fallback is reserved for outputs
            # where per-box distributions were deliberately not embedded.
            has_stored_payload = any(
                isinstance(
                    (((b.get("distributions", {}) or {}).get(kind, {}) or {}).get(name, None)),
                    Mapping,
                )
                for b in boxes
            )
            if has_stored_payload:
                raise RuntimeError(
                    f"Malformed stored {kind} CDF {name!r}; refusing to omit its public plot"
                ) from exc
            try:
                x0, y_report, _axis_source = _validated_plot_cdf_payload(
                    {
                        xkey: drep.get(xkey, drep.get("x", [])),
                        "cdf": drep.get("mean", drep.get("ensemble_cdf", [])),
                    },
                    xkey=xkey,
                )
            except ValueError as report_exc:
                raise RuntimeError(
                    f"Declared {kind} CDF {name!r} has neither complete per-box "
                    "payloads nor a valid convergence-report curve"
                ) from report_exc
            mats = np.tile(y_report[None, :], (N, 1))
            meta = {
                "group": drep.get("group", ""),
                "kind": drep.get("kind", f"{kind}_cdf"),
                "name": name,
                "grid_source": "convergence_report_only",
            }
        curve_mats.append(
            {
                "group": drep.get("group", meta.get("group", "")),
                "kind": str(drep.get("kind", meta.get("kind", f"{kind}_cdf"))),
                "name": str(name),
                "x": np.asarray(x0, dtype=float),
                "mat": np.asarray(mats, dtype=float),
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "grid_source": str(meta.get("grid_source", "ensemble_common_grid")),
            }
        )

    for nm in bond_names:
        _stack_cdf("bondlen", nm)
    for nm in angle_names:
        _stack_cdf("angle", nm)

    for nm in void_names:
        _stack_cdf("void", nm)

    # coordination global kmax
    for nm in coord_names:
        drep = (conv.get("distributions", {}) or {}).get(nm, None)
        if not isinstance(drep, dict):
            drep = {}
        abs_tol = float(drep.get("abs_tol", 0.0))
        rel_tol = float(drep.get("rel_tol", 0.0))
        try:
            xk, mats, meta = _plot_stack_coord_cdfs(boxes, nm)
        except Exception as exc:
            has_stored_payload = any(
                isinstance(
                    (((b.get("distributions", {}) or {}).get("coord", {}) or {}).get(nm, None)),
                    Mapping,
                )
                for b in boxes
            )
            if has_stored_payload:
                raise RuntimeError(
                    f"Malformed stored coordination CDF {nm!r}; refusing to omit its public plot"
                ) from exc
            report_payload: dict[str, Any] = {
                "cdf": drep.get("mean", drep.get("ensemble_cdf", [])),
            }
            if "x" in drep:
                report_payload["x"] = drep.get("x")
            try:
                xk, y_report, axis_source = _validated_plot_cdf_payload(
                    report_payload,
                    xkey="x",
                    allow_implicit_integer_axis=True,
                )
            except ValueError as report_exc:
                raise RuntimeError(
                    f"Declared coordination CDF {nm!r} has neither complete per-box "
                    "payloads nor a valid convergence-report curve"
                ) from report_exc
            mats = np.tile(y_report[None, :], (N, 1))
            meta = {
                "grid_source": "convergence_report_only",
                "axis_source": axis_source,
            }
        curve_mats.append(
            {
                "group": drep.get("group", "short"),
                "kind": str(drep.get("kind", "coord_cdf")),
                "name": str(nm),
                "x": np.asarray(xk, dtype=float),
                "mat": np.asarray(mats, dtype=float),
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "grid_source": str(meta.get("grid_source", "ensemble_common_integer_grid")),
            }
        )

    def _stack_sampled_curve(family: str, lab: str, axis_key: str, value_key: str, default_kind: str) -> None:
        drep = (conv.get("distributions", {}) or {}).get(lab, None)
        if not isinstance(drep, Mapping):
            drep = {}
        has_stored_payload = any(
            isinstance(
                (((b.get("distributions", {}) or {}).get(family, {}) or {}).get(lab, None)),
                Mapping,
            )
            for b in boxes
        )
        if has_stored_payload:
            try:
                payloads = [
                    (((b.get("distributions", {}) or {}).get(family, {}) or {}).get(lab, None))
                    for b in boxes
                ]
                x_ref, mats, stack_meta = _align_sampled_plot_payloads(
                    payloads,
                    family=family,
                    xkey=axis_key,
                    ykey=value_key,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Malformed stored {family} curve {lab!r}; refusing to omit its public plot"
                ) from exc
            grid_source = str(stack_meta.get("grid_source", "ensemble_common_axis_overlap"))
        else:
            try:
                x_ref = np.asarray(
                    drep.get(axis_key, drep.get("x", [])),
                    dtype=float,
                )
                y_report = np.asarray(drep.get("mean", []), dtype=float)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Declared {family} curve {lab!r} has no numeric convergence-report curve"
                ) from exc
            _require_finite_plot_points(
                x_ref,
                y_report,
                context=f"declared {family} curve {lab!r}",
            )
            if x_ref.size < 2 or not np.all(np.diff(x_ref) > 0.0):
                raise RuntimeError(
                    f"Declared {family} curve {lab!r} has an invalid report grid"
                )
            mats = np.tile(y_report[None, :], (N, 1))
            grid_source = "convergence_report_only"
        curve_mats.append(
            {
                "group": str(drep.get("group", "long")),
                "kind": str(drep.get("kind", default_kind)),
                "name": str(lab),
                "x": x_ref,
                "mat": mats,
                "abs_tol": float(drep.get("abs_tol", 0.0)),
                "rel_tol": float(drep.get("rel_tol", 0.0)),
                "grid_source": grid_source,
            }
        )

    for lab in gr_labels:
        _stack_sampled_curve("gr", lab, "r", "g", "gr_curve")

    for lab in sq_labels:
        _stack_sampled_curve("sq", lab, "q", "s", "sq_curve")
    # convergence trend figure
    # convergence trend figure
    # n grid

    n_grid = np.arange(2, N + 1, dtype=int)
    trend = {"short": [], "medium": [], "long": [], "all": []}

    # precompute cumulative curves
    mats_sums: list[dict[str, Any]] = []
    for cm in curve_mats:
        X = np.asarray(cm["mat"], dtype=float)
        mats_sums.append({"meta": cm, "S": np.cumsum(X, axis=0), "Q": np.cumsum(X * X, axis=0)})

    dens_S = np.cumsum(dens)
    dens_Q = np.cumsum(dens * dens)
    ring_S = np.cumsum(ring_mat, axis=0) if ring_mat is not None else None
    ring_Q = np.cumsum(ring_mat * ring_mat, axis=0) if ring_mat is not None else None
    rmean_S = np.cumsum(ring_mean_vec) if ring_mean_vec is not None else None
    rmean_Q = np.cumsum(ring_mean_vec * ring_mean_vec) if ring_mean_vec is not None else None

    def _ratio_from_mu_sd(
        mu: np.ndarray,
        sd: np.ndarray,
        n: int,
        abs_tol: float,
        rel_tol: float,
        crit: float,
        *,
        bounded: bool,
    ) -> float:
        mu = np.asarray(mu, dtype=float)
        sd = np.asarray(sd, dtype=float)

        if bounded:
            # consistent production convergence
            a = float(alpha_test)
            if int(n) < 2 or (not math.isfinite(a)) or a <= 0.0 or a >= 1.0:
                half = np.full_like(sd, np.inf, dtype=float)
            elif bounded_ci_method == "t":
                half = float(crit) * (sd / math.sqrt(float(n)))
            elif bounded_ci_method == "hoeffding":
                hw = math.sqrt(math.log(2.0 / a) / (2.0 * float(n)))
                half = np.full_like(sd, float(hw), dtype=float)
            else:
                # empirical bernstein maurer
                v = np.square(sd)
                v = np.minimum(v, 0.25)
                L = math.log(3.0 / a)
                half = np.sqrt(2.0 * v * L / float(n)) + 3.0 * L / float(n)
        else:
            se = sd / math.sqrt(float(n))
            half = float(crit) * se

        tol = np.maximum(float(abs_tol), float(rel_tol) * np.abs(mu))
        tol = np.maximum(tol, 1e-30)  # prevent division zero
        r = half / tol
        return float(np.nanmax(r))

    for n in n_grid.tolist():
        crit, _m = _critical_value(int(n), alpha_test)
        # maxima
        gmax = {"short": 0.0, "medium": 0.0, "long": 0.0}

        # density
        mu, sd = _mean_sd_from_sums(dens_S[n - 1], dens_Q[n - 1], n)
        r = _ratio_from_mu_sd(np.asarray([mu]), np.asarray([sd]), n, dens_abs, dens_rel, crit, bounded=False)
        gmax["long"] = max(gmax["long"], r)

        # rings
        if ring_keys and ring_S is not None and ring_Q is not None:
            mu, sd = _mean_sd_from_sums(ring_S[n - 1, :], ring_Q[n - 1, :], n)
            r = _ratio_from_mu_sd(mu, sd, n, ring_abs, ring_rel, crit, bounded=True)
            gmax["medium"] = max(gmax["medium"], r)

        if ring_mean_vec is not None and rmean_S is not None and rmean_Q is not None:
            mu, sd = _mean_sd_from_sums(rmean_S[n - 1], rmean_Q[n - 1], n)
            r = _ratio_from_mu_sd(np.asarray([mu]), np.asarray([sd]), n, ring_mean_abs, ring_mean_rel, crit, bounded=False)
            gmax["medium"] = max(gmax["medium"], r)

        # curves
        for ms in mats_sums:
            meta = ms["meta"]
            S = ms["S"][n - 1, :]
            Q = ms["Q"][n - 1, :]
            mu, sd = _mean_sd_from_sums(S, Q, n)
            kind = str(meta.get("kind", ""))
            bounded = bool(kind in ("bondlen_cdf", "angle_cdf", "coord_cdf", "void_cdf"))
            r = _ratio_from_mu_sd(mu, sd, n, float(meta["abs_tol"]), float(meta["rel_tol"]), crit, bounded=bounded)
            grp = str(meta.get("group", ""))
            if grp not in gmax:
                continue
            gmax[grp] = max(gmax[grp], r)

        trend["short"].append(gmax["short"])
        trend["medium"].append(gmax["medium"])
        trend["long"].append(gmax["long"])
        trend["all"].append(max(gmax.values()))

    trend = {k: np.asarray(v, dtype=float) for k, v in trend.items()}
    # scale plotting identical
    eps = 1e-12
    for k in list(trend.keys()):
        arr = np.asarray(trend[k], dtype=float)
        m = np.isfinite(arr)
        arr[m] = np.maximum(arr[m], eps)
        trend[k] = arr

    # nonparametric distribution stability
    stab_trend_all: Optional[np.ndarray] = None
    if isinstance(conv.get("stability", None), dict) and bool((conv.get("stability", {}) or {}).get("enabled", False)):
        stab_cfg = conv.get("stability", {}) or {}
        split = str(stab_cfg.get("split", "half")).strip().lower()
        dist_kind = str(stab_cfg.get("distance", "wasserstein")).strip().lower()
        if dist_kind not in {"wasserstein", "ks"}:
            dist_kind = "wasserstein"

        def _w1_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size == 0 or y.size == 0:
                return float("nan")
            xs = np.sort(x)
            ys = np.sort(y)
            z = np.sort(np.concatenate([xs, ys]))
            fx = np.searchsorted(xs, z, side="right") / float(xs.size)
            fy = np.searchsorted(ys, z, side="right") / float(ys.size)
            dz = np.diff(z)
            if dz.size == 0:
                return 0.0
            return float(np.sum(np.abs(fx[:-1] - fy[:-1]) * dz))

        def _ks_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size == 0 or y.size == 0:
                return float("nan")
            xs = np.sort(x)
            ys = np.sort(y)
            z = np.sort(np.concatenate([xs, ys]))
            fx = np.searchsorted(xs, z, side="right") / float(xs.size)
            fy = np.searchsorted(ys, z, side="right") / float(ys.size)
            return float(np.max(np.abs(fx - fy)))

        def _dist_scalar(x: np.ndarray, y: np.ndarray) -> float:
            return _ks_distance_1d(x, y) if dist_kind == "ks" else _w1_distance_1d(x, y)

        def _curve_dist(m1: np.ndarray, m2: np.ndarray) -> float:
            m1 = np.asarray(m1, dtype=float)
            m2 = np.asarray(m2, dtype=float)
            if m1.ndim != 2 or m2.ndim != 2 or m1.shape[1] != m2.shape[1]:
                return float("nan")
            p = int(m1.shape[1])
            dmax = 0.0
            for j in range(p):
                dj = float(_dist_scalar(m1[:, j], m2[:, j]))
                if math.isfinite(dj):
                    dmax = max(dmax, dj)
            return float(dmax)

        def _tol_from_rep(rep: dict[str, Any]) -> float:
            # prefer tol report
            try:
                tol = float(rep.get("tol", float("nan")))
            except Exception:
                tol = float("nan")
            if math.isfinite(tol) and tol > 0.0:
                return float(tol)
            # reconstruct abs mean
            try:
                abs_tol = float(rep.get("abs_tol", 0.0))
                rel_tol = float(rep.get("rel_tol", 0.0))
                mean = float(rep.get("mean", 0.0))
            except Exception:
                return float("nan")
            return float(max(abs_tol, rel_tol * abs(mean)))

        stab_ratios: list[float] = []
        for n in n_grid:
            n = int(n)
            if n < 4:
                stab_ratios.append(float("nan"))
                continue

            if split == "last_batch":
                nb = int(prod.get("batch_boxes", 0) or 0)
                if nb <= 0:
                    nb = max(1, n // 4)
                if n >= 2 * nb:
                    g1 = boxes[n - 2 * nb : n - nb]
                    g2 = boxes[n - nb : n]
                else:
                    n1 = n // 2
                    g1 = boxes[:n1]
                    g2 = boxes[n1:n]
            else:
                n1 = n // 2
                g1 = boxes[:n1]
                g2 = boxes[n1:n]

            ratios: list[float] = []

            # scalars report
            for key, rep in (conv.get("scalars", {}) or {}).items():
                if not isinstance(rep, dict):
                    continue
                tol = _tol_from_rep(rep)
                if not (math.isfinite(tol) and tol > 0.0):
                    continue
                if str(key) == "density":
                    x = np.asarray([float(b.get("density", float("nan"))) for b in g1], dtype=float)
                    y = np.asarray([float(b.get("density", float("nan"))) for b in g2], dtype=float)
                else:
                    x = np.asarray([float((b.get("metrics", {}) or {}).get(str(key), float("nan"))) for b in g1], dtype=float)
                    y = np.asarray([float((b.get("metrics", {}) or {}).get(str(key), float("nan"))) for b in g2], dtype=float)
                d = float(_dist_scalar(x, y))
                if math.isfinite(d):
                    ratios.append(float(d) / float(tol))

            # curves report
            for name, rep in (conv.get("distributions", {}) or {}).items():
                if not isinstance(rep, dict):
                    continue
                kind = str(rep.get("kind", ""))
                if kind not in {"bondlen_cdf", "angle_cdf", "coord_cdf", "void_cdf", "gr_curve", "sq_curve"}:
                    continue
                tol = _tol_from_rep(rep)
                if not (math.isfinite(tol) and tol > 0.0):
                    continue
                try:
                    if kind == "gr_curve":
                        all_payloads = [
                            (((b.get("distributions", {}) or {}).get("gr", {}) or {}).get(name, None))
                            for b in boxes[:n]
                        ]
                        x_ref, _mall, _meta = _align_sampled_plot_payloads(
                            all_payloads, family="gr", xkey="r", ykey="g"
                        )
                        g1_payloads = [
                            (((b.get("distributions", {}) or {}).get("gr", {}) or {}).get(name, None))
                            for b in g1
                        ]
                        g2_payloads = [
                            (((b.get("distributions", {}) or {}).get("gr", {}) or {}).get(name, None))
                            for b in g2
                        ]
                        _x1, m1, _meta1 = _align_sampled_plot_payloads(
                            g1_payloads,
                            family="gr",
                            xkey="r",
                            ykey="g",
                            x_ref=x_ref,
                        )
                        _x2, m2, _meta2 = _align_sampled_plot_payloads(
                            g2_payloads,
                            family="gr",
                            xkey="r",
                            ykey="g",
                            x_ref=x_ref,
                        )
                    elif kind == "sq_curve":
                        all_payloads = [
                            (((b.get("distributions", {}) or {}).get("sq", {}) or {}).get(name, None))
                            for b in boxes[:n]
                        ]
                        x_ref, _mall, _meta = _align_sampled_plot_payloads(
                            all_payloads, family="sq", xkey="q", ykey="s"
                        )
                        g1_payloads = [
                            (((b.get("distributions", {}) or {}).get("sq", {}) or {}).get(name, None))
                            for b in g1
                        ]
                        g2_payloads = [
                            (((b.get("distributions", {}) or {}).get("sq", {}) or {}).get(name, None))
                            for b in g2
                        ]
                        _x1, m1, _meta1 = _align_sampled_plot_payloads(
                            g1_payloads,
                            family="sq",
                            xkey="q",
                            ykey="s",
                            x_ref=x_ref,
                        )
                        _x2, m2, _meta2 = _align_sampled_plot_payloads(
                            g2_payloads,
                            family="sq",
                            xkey="q",
                            ykey="s",
                            x_ref=x_ref,
                        )
                    else:
                        family = {
                            "bondlen_cdf": "bondlen",
                            "angle_cdf": "angle",
                            "coord_cdf": "coord",
                            "void_cdf": "void",
                        }[kind]
                        payloads: list[Mapping[str, Any]] = []
                        for b in [*g1, *g2]:
                            fam = ((b.get("distributions", {}) or {}).get(family, {}) or {})
                            payload = fam.get(name, None) if isinstance(fam, Mapping) else None
                            if not isinstance(payload, Mapping):
                                raise ValueError("missing CDF payload in stability split")
                            payloads.append(payload)
                        _x_ref, mall, _meta = _align_plot_cdf_payloads(
                            payloads,
                            allow_implicit_integer_axis=(kind == "coord_cdf"),
                        )
                        m1 = mall[: len(g1), :]
                        m2 = mall[len(g1) :, :]
                    d = float(_curve_dist(m1, m2))
                except Exception:
                    if kind in {
                        "bondlen_cdf",
                        "angle_cdf",
                        "coord_cdf",
                        "void_cdf",
                    }:
                        ratios.append(float("inf"))
                    continue
                if math.isfinite(d):
                    ratios.append(float(d) / float(tol))

            stab_ratios.append(float(np.nanmax(np.asarray(ratios, dtype=float))) if ratios else float("nan"))

        stab_trend_all = np.asarray(stab_ratios, dtype=float)
        m = np.isfinite(stab_trend_all)
        stab_trend_all[m] = np.maximum(stab_trend_all[m], eps)

    # converged convergence present
    n_conv = None
    mode = str(conv.get("mode", "ci")).strip().lower()
    if mode not in {"ci", "stability", "both"}:
        mode = "ci"
    base = np.asarray(trend.get("all", np.full_like(n_grid, np.nan, dtype=float)), dtype=float)
    if mode == "stability" and stab_trend_all is not None:
        base = np.asarray(stab_trend_all, dtype=float)
    if mode == "both" and stab_trend_all is not None:
        base = np.maximum(base, np.asarray(stab_trend_all, dtype=float))
    ok_mask = np.isfinite(base) & (base <= 1.0)
    if np.any(ok_mask):
        n_conv = int(n_grid[np.argmax(ok_mask)])

    def _fig_convergence() -> "plt.Figure":
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(n_grid, trend["all"], label="all", linestyle="-")

        if stab_trend_all is not None:
            ax.plot(n_grid, stab_trend_all, label="stability", linestyle=(0, (4, 2)))

        # plot actually convergence
        groups_present = set((conv.get("groups", {}) or {}).keys())
        # fallback
        if not groups_present:
            if bond_names or angle_names or coord_names:
                groups_present.add("short")
            if ring_keys or has_ring_mean:
                groups_present.add("medium")
            if True:
                groups_present.add("long")

        styles = {"short": "--", "medium": ":", "long": "-."}
        for g in ["short", "medium", "long"]:
            if g in groups_present:
                ax.plot(n_grid, trend[g], label=g, linestyle=styles.get(g, "-"))
        ax.axhline(1.0, linewidth=1.0)
        if n_conv is not None:
            ax.axvline(int(n_conv), linewidth=1.0)
        ax.set_xlabel("number of boxes")
        ax.set_ylabel("max(ratio / tolerance)")
        ax.set_yscale("log")
        ax.set_title("Retrospective production convergence diagnostic")
        ax.legend(frameon=False)
        # alpha_family is the familywise *error probability*, not confidence.
        txt = _familywise_error_annotation(
            fw,
            alpha_per_test=alpha_test,
            bounded_ci_method=bounded_ci_method,
        )
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, va="bottom", ha="left")
        crossing = (
            f"retrospective first threshold crossing: n={int(n_conv)}"
            if n_conv is not None
            else "retrospective first threshold crossing: none"
        )
        ax.text(0.02, 0.98, crossing, transform=ax.transAxes, va="top", ha="left")
        return fig

    # distribution plot helpers
    # distribution plot helpers
    # curve page

    def _plot_curve_page(
        *,
        x: np.ndarray,
        mu: np.ndarray,
        half: np.ndarray,
        kind: str,
        name: str,
        xlabel: str,
        ylabel: str,
        clip: Optional[tuple[float, float]] = None,
        xlim: Optional[tuple[float, float]] = None,
        box_curves: Optional[np.ndarray] = None,
    ) -> "plt.Figure":
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        half = np.asarray(half, dtype=float)
        lo = mu - half
        hi = mu + half
        if clip is not None:
            lo = np.clip(lo, clip[0], clip[1])
            hi = np.clip(hi, clip[0], clip[1])
        if show_boxes and box_curves is not None:
            for i in range(int(box_curves.shape[0])):
                ax.plot(x, box_curves[i, :], linewidth=0.6, alpha=0.25, color="0.5")
        ax.plot(x, mu, color="black")
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ax.fill_between(x, lo, hi, color="0.8", alpha=0.8, linewidth=0.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None and np.all(np.isfinite(np.asarray(xlim, dtype=float))):
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_title(_clean_name(name))
        return fig

    pages: list[tuple[str, "plt.Figure"]] = []
    pages.append(("convergence", _fig_convergence()))

    # title prefix
    if title is None:
        base = f"vitriflow production ({Path(json_path).name})"
    else:
        base = str(title)
    n_acc = int(prod.get("n_boxes_accepted", prod.get("n_boxes", N)))
    n_rej = int(prod.get("n_boxes_rejected", 0))
    n_tot = int(prod.get("n_boxes_total", n_acc + n_rej))
    convergence_display = _production_convergence_display(prod, conv)
    convergence_label = str(convergence_display["label"])
    if n_rej > 0:
        base2 = (
            f"{base} | accepted={n_acc}/{n_tot} | rejected={n_rej} | "
            f"{convergence_label}"
        )
    else:
        base2 = f"{base} | n={n_acc} | {convergence_label}"
    pages[0][1].suptitle(base2)

    # density page
    dens_rep = ((conv.get("scalars", {}) or {}).get("density", {}) or {})
    dens_mu = float(dens_rep.get("mean", float("nan")))
    dens_half = float(dens_rep.get("ci_halfwidth", float("nan")))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(dens[np.isfinite(dens)], bins=min(30, max(5, N // 2)), density=True, color="0.8", edgecolor="0.2")
    if math.isfinite(dens_mu):
        ax.axvline(dens_mu, color="black", linewidth=1.5, label="mean")
    if math.isfinite(dens_mu) and math.isfinite(dens_half):
        ax.axvspan(dens_mu - dens_half, dens_mu + dens_half, color="0.5", alpha=0.25, label="CI")
    ax.set_xlabel(f"density [{_density_unit_label()}]")
    ax.set_ylabel("probability density")
    ax.set_title("Density across boxes")
    ax.legend(frameon=False)
    fig.suptitle(base2)
    pages.append(("density", fig))

    # ring statistics present
    if ring_keys:
        rrep = (conv.get("distributions", {}) or {}).get("ring", {}) or {}
        mu = np.asarray(rrep.get("mean", []), dtype=float)
        half = np.asarray(rrep.get("ci_halfwidth", []), dtype=float)
        if mu.size != len(ring_keys) and ring_mat is not None:
            # Analysis-only or degraded convergence payloads can declare ring
            # keys before the ensemble PMF has been materialized.  Build a
            # plotting summary directly from boxes[].metrics rather than
            # failing on an empty report entry.
            try:
                mu = np.nanmean(np.asarray(ring_mat, dtype=float), axis=0)
                if np.asarray(ring_mat).shape[0] >= 2:
                    half = np.nanstd(np.asarray(ring_mat, dtype=float), axis=0, ddof=1) / math.sqrt(float(np.asarray(ring_mat).shape[0]))
                else:
                    half = np.zeros_like(mu, dtype=float)
            except Exception:
                mu = np.asarray([], dtype=float)
                half = np.asarray([], dtype=float)
        if mu.size != len(ring_keys):
            mu = np.asarray([], dtype=float)
            half = np.asarray([], dtype=float)
        # parse sizes keys
        sizes = []
        for k in ring_keys:
            try:
                sizes.append(int(str(k).split("ring_frac_")[-1]))
            except Exception:
                sizes.append(len(sizes) + 1)
        sizes_arr = np.asarray(sizes, dtype=int)
        order = np.argsort(sizes_arr)
        sizes_arr = sizes_arr[order]
        if mu.size == sizes_arr.size:
            mu = mu[order]
            half = half[order] if half.size == sizes_arr.size else np.zeros_like(mu, dtype=float)
        else:
            sizes_arr = np.asarray([], dtype=int)
            mu = np.asarray([], dtype=float)
            half = np.asarray([], dtype=float)

        if mu.size == 0:
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            ax.text(0.5, 0.5, "Ring PMF unavailable", ha="center", va="center")
            ax.set_axis_off()
            fig.suptitle(base2)
            pages.append(("rings_unavailable", fig))
            # Continue to other available descriptor plots.
        else:
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            ax.bar(sizes_arr.astype(float), mu, color="0.8", edgecolor="0.2")
            if mu.size == half.size and mu.size > 0 and np.all(np.isfinite(half)):
                ax.errorbar(sizes_arr.astype(float), mu, yerr=half, fmt="none", ecolor="black", capsize=2.5, linewidth=1.0)
            ax.set_xlabel("ring size")
            ax.set_ylabel("fraction")
            ax.set_title("Ring statistics")
            fig.suptitle(base2)
            pages.append(("rings", fig))

    if ring_mean_vec is not None:
        rrep = ((conv.get("scalars", {}) or {}).get("ring_mean_size", {}) or {})
        mu = float(rrep.get("mean", float("nan")))
        half = float(rrep.get("ci_halfwidth", float("nan")))
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.hist(ring_mean_vec[np.isfinite(ring_mean_vec)], bins=min(30, max(5, N // 2)), density=True, color="0.8", edgecolor="0.2")
        if math.isfinite(mu):
            ax.axvline(mu, color="black", linewidth=1.5, label="mean")
        if math.isfinite(mu) and math.isfinite(half):
            ax.axvspan(mu - half, mu + half, color="0.5", alpha=0.25, label="CI")
        ax.set_xlabel("mean ring size")
        ax.set_ylabel("probability density")
        ax.set_title("Mean ring size across boxes")
        ax.legend(frameon=False)
        fig.suptitle(base2)
        pages.append(("ring_mean", fig))

    # Every finite scalar emitted by the configured metric pipeline receives a
    # public production page.  Distribution families are plotted separately
    # below; these pages cover their scalar summaries plus amorphous, graph, and
    # other diagnostic scalar plumbing without silently dropping them.
    scalar_metric_keys: set[str] = set()
    for box in boxes:
        metrics_row = box.get("metrics", {})
        if isinstance(metrics_row, Mapping):
            scalar_metric_keys.update(str(key) for key in metrics_row)
    scalar_metric_keys = {
        key
        for key in scalar_metric_keys
        if not key.startswith("ring_frac_") and key != "ring_mean_size"
    }
    scalar_crit, _scalar_crit_method = _critical_value(N, alpha_test)
    for key in sorted(scalar_metric_keys):
        values = np.asarray(
            [
                _maybe_float(
                    (box.get("metrics", {}) or {}).get(key)
                    if isinstance(box.get("metrics", {}), Mapping)
                    else None
                )
                for box in boxes
            ],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        finite = np.isfinite(values)
        if np.any(finite):
            ids = np.arange(1, N + 1, dtype=float)
            ax.plot(ids[finite], values[finite], "o", label="box value")
            mu = float(np.mean(values[finite]))
            ax.axhline(mu, color="black", linewidth=1.2, label="mean")
            if int(np.sum(finite)) >= 2 and math.isfinite(scalar_crit):
                se = float(
                    np.std(values[finite], ddof=1)
                    / math.sqrt(float(np.sum(finite)))
                )
                half = float(scalar_crit) * se
                ax.axhspan(mu - half, mu + half, color="0.7", alpha=0.2, label="fixed-look CI")
            ax.text(
                0.02,
                0.98,
                f"finite boxes: {int(np.sum(finite))}/{N}",
                transform=ax.transAxes,
                va="top",
                ha="left",
            )
            ax.legend(frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "Metric emitted but unavailable/non-finite in every box",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        ax.set_xlabel("box index")
        ax.set_ylabel(str(key))
        ax.set_title(_clean_name(key))
        fig.suptitle(base2)
        pages.append((f"scalar_{key}", fig))

    # The coordination cutoff sweep is a nested diagnostic rather than a
    # scalar/distribution entry.  Plot it explicitly so enabling the metric
    # cannot leave its application-facing path untested.
    for name, (delta_r, sweep_matrix) in _coordination_sweep_matrices(boxes).items():
        mu = np.mean(sweep_matrix, axis=0)
        if sweep_matrix.shape[0] >= 2:
            se = np.std(sweep_matrix, axis=0, ddof=1) / math.sqrt(
                float(sweep_matrix.shape[0])
            )
            half = float(scalar_crit) * se
        else:
            half = np.full_like(mu, np.nan, dtype=float)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        _plot_mean_band(
            ax,
            delta_r,
            mu,
            mu - half,
            mu + half,
            label="ensemble mean",
            show_band=True,
        )
        if show_boxes:
            for row in sweep_matrix:
                ax.plot(delta_r, row, linewidth=0.6, alpha=0.25, color="0.5")
        ax.set_xlabel(f"cutoff shift [{_distance_unit_label()}]")
        ax.set_ylabel("coordination-defect fraction")
        ax.set_title(f"Coordination cutoff sensitivity: {_clean_name(name)}")
        ax.legend(frameon=False)
        fig.suptitle(base2)
        pages.append((f"coordination_sweep_{name}", fig))

    # curve pages
    # descriptors plot informative
    # range cdfs
    dist_rep = conv.get("distributions", {}) or {}
    crit_N, _ = _critical_value(N, alpha_test)

    def _quantile_xlim(xgrid: np.ndarray, cdf_mu: np.ndarray, qlo: float = 0.01, qhi: float = 0.99) -> Optional[tuple[float, float]]:
        xgrid = np.asarray(xgrid, dtype=float)
        cdf_mu = np.asarray(cdf_mu, dtype=float)
        if xgrid.size < 2 or cdf_mu.size != xgrid.size:
            return None
        if not np.all(np.isfinite(xgrid)):
            return None
        m = np.isfinite(cdf_mu)
        if not np.any(m):
            return None
        c = np.clip(cdf_mu, 0.0, 1.0)
        idx0 = np.where(c >= float(qlo))[0]
        idx1 = np.where(c >= float(qhi))[0]
        if idx0.size == 0 or idx1.size == 0:
            return None
        i0 = int(idx0[0])
        i1 = int(idx1[0])
        if i1 <= i0:
            return None
        x0 = float(xgrid[i0])
        x1 = float(xgrid[i1])
        pad = 0.05 * (x1 - x0)
        lo = max(float(xgrid[0]), x0 - pad)
        hi = min(float(xgrid[-1]), x1 + pad)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            return None
        return (lo, hi)

    for cm in curve_mats:
        nm = str(cm["name"])
        kind = str(cm.get("kind", ""))
        X = np.asarray(cm["mat"], dtype=float)
        xgrid = np.asarray(cm["x"], dtype=float)

        if kind in ("bondlen_cdf", "angle_cdf", "void_cdf"):
            if xgrid.size < 3:
                raise RuntimeError(
                    f"Declared distribution {nm!r} needs at least three CDF grid points for density plotting"
                )
            dx = np.diff(xgrid)
            if not np.all(dx > 0):
                raise RuntimeError(f"Non-increasing grid for '{nm}'.")
            pdf_mat = np.diff(X, axis=1) / dx[None, :]
            xmid = 0.5 * (xgrid[:-1] + xgrid[1:])
            mu = np.nanmean(pdf_mat, axis=0)
            sd = np.nanstd(pdf_mat, axis=0, ddof=1) if N >= 2 else np.full_like(mu, np.nan)
            se = sd / math.sqrt(float(N)) if N >= 2 else np.full_like(mu, np.nan)
            half = crit_N * se

            cdf_mu = np.nanmean(X, axis=0)
            xlim = _quantile_xlim(xgrid, cdf_mu) if kind in ("bondlen_cdf", "void_cdf") else None

            xlabel = (
                f"r [{_distance_unit_label()}]"
                if kind == "bondlen_cdf"
                else (f"clearance r [{_distance_unit_label()}]" if kind == "void_cdf" else "θ [deg]")
            )
            fig = _plot_curve_page(
                x=xmid,
                mu=mu,
                half=half,
                kind=kind,
                name=nm,
                xlabel=xlabel,
                ylabel="probability density",
                clip=(0.0, float("inf")),
                xlim=xlim,
                box_curves=pdf_mat if show_boxes else None,
            )

        elif kind == "coord_cdf":
            if xgrid.size < 2:
                raise RuntimeError(
                    f"Declared coordination distribution {nm!r} has fewer than two support points"
                )
            pmf = np.zeros_like(X, dtype=float)
            pmf[:, 0] = X[:, 0]
            pmf[:, 1:] = X[:, 1:] - X[:, :-1]
            # guard
            pmf = np.maximum(pmf, 0.0)

            mu = np.nanmean(pmf, axis=0)
            sd = np.nanstd(pmf, axis=0, ddof=1) if N >= 2 else np.full_like(mu, np.nan)
            se = sd / math.sqrt(float(N)) if N >= 2 else np.full_like(mu, np.nan)
            half = crit_N * se

            # focus central distribution
            cdf_mu = np.nanmean(X, axis=0)
            xlim = None
            qlim = _quantile_xlim(xgrid, cdf_mu)
            if qlim is not None:
                xlim = (max(float(xgrid[0]) - 0.5, qlim[0] - 0.5), min(float(xgrid[-1]) + 0.5, qlim[1] + 0.5))

            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            ax.bar(xgrid.astype(float), mu, color="0.8", edgecolor="0.2")
            if mu.size == half.size and mu.size > 0 and np.all(np.isfinite(half)):
                ax.errorbar(xgrid.astype(float), mu, yerr=half, fmt="none", ecolor="black", capsize=2.5, linewidth=1.0)
            ax.set_xlabel("k")
            ax.set_ylabel("probability mass")
            if xlim is not None and np.all(np.isfinite(np.asarray(xlim, dtype=float))):
                ax.set_xlim(float(xlim[0]), float(xlim[1]))
            ax.set_title(_clean_name(nm))

        elif kind in ("gr_curve", "sq_curve"):
            rep = dist_rep.get(nm, None)
            use_rep = False
            axis_key = "r" if kind == "gr_curve" else "q"
            y_label = "g(r)" if kind == "gr_curve" else "S(q)"
            x_label = f"r [{_distance_unit_label()}]" if kind == "gr_curve" else f"q [1/{_distance_unit_label()}]"
            if isinstance(rep, dict):
                xr = np.asarray(rep.get(axis_key, rep.get("x", xgrid)), dtype=float)
                mur = np.asarray(rep.get("mean", []), dtype=float)
                halfr = np.asarray(rep.get("ci_halfwidth", []), dtype=float)
                if xr.size >= 2 and mur.size == xr.size and halfr.size == xr.size and np.all(np.isfinite(xr)):
                    x, mu, half = xr, mur, halfr
                    use_rep = True
            if not use_rep:
                x = xgrid
                mu = np.nanmean(X, axis=0)
                sd = np.nanstd(X, axis=0, ddof=1) if N >= 2 else np.full_like(mu, np.nan)
                se = sd / math.sqrt(float(N)) if N >= 2 else np.full_like(mu, np.nan)
                half = crit_N * se
            if mu.size < 2:
                raise RuntimeError(
                    f"Declared sampled curve {nm!r} has fewer than two finite points"
                )
            fig = _plot_curve_page(
                x=x,
                mu=mu,
                half=half,
                kind=kind,
                name=nm,
                xlabel=x_label,
                ylabel=y_label,
                clip=(0.0, float("inf")),
                box_curves=np.asarray(cm["mat"], dtype=float) if show_boxes else None,
            )

        else:
            raise RuntimeError(
                f"Declared distribution {nm!r} has unsupported plotting kind {kind!r}"
            )

        fig.suptitle(base2)
        pages.append((nm, fig))

    # dft comparison pages
    if dft_boxes:
        md_boxes = boxes
        dft_by_id = {int(b.get("box_id", -1)): b for b in dft_boxes}
        common_ids = [int(b.get("box_id", -1)) for b in md_boxes if int(b.get("box_id", -1)) in dft_by_id]

        def _scalar_summary(
            key: str,
            src_boxes: list[dict[str, Any]],
            *,
            conv_report: Optional[dict[str, Any]] = None,
            z_fallback: float = 2.0,
        ) -> tuple[float, float]:
            if isinstance(conv_report, dict):
                rep = (conv_report.get("scalars", {}) or {}).get(key, None)
                if isinstance(rep, dict):
                    mu = float(rep.get("mean", float("nan")))
                    hw = float(rep.get("ci_halfwidth", float("nan")))
                    if math.isfinite(mu) and math.isfinite(hw):
                        return mu, hw
            vals = []
            for b in src_boxes:
                if key == "density":
                    vals.append(float(b.get("density", float("nan"))))
                else:
                    vals.append(float((b.get("metrics", {}) or {}).get(key, float("nan"))))
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size < 2:
                return float("nan"), float("nan")
            mu = float(np.mean(v))
            sd = float(np.std(v, ddof=1))
            z = float((conv_report or {}).get("zscore", z_fallback)) if isinstance(conv_report, dict) else float(z_fallback)
            return mu, float(z * sd / math.sqrt(float(v.size)))

        def _curve_summary(
            kind: str,
            name: str,
            src_boxes: list[dict[str, Any]],
            *,
            conv_report: Optional[dict[str, Any]] = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if isinstance(conv_report, dict):
                rep = (conv_report.get("distributions", {}) or {}).get(name, None)
                if isinstance(rep, dict):
                    try:
                        hw = np.asarray(rep.get("ci_halfwidth", []), dtype=float)
                    except Exception:
                        hw = np.asarray([], dtype=float)
                    if kind in {"bondlen", "void", "angle", "coord"}:
                        report_payload: dict[str, Any] = {"cdf": rep.get("mean", [])}
                        if "x" in rep:
                            report_payload["x"] = rep.get("x")
                        try:
                            x, mu, _axis_source = _validated_plot_cdf_payload(
                                report_payload,
                                allow_implicit_integer_axis=(kind == "coord"),
                            )
                        except ValueError:
                            x = np.asarray([], dtype=float)
                            mu = np.asarray([], dtype=float)
                        if x.size > 0 and hw.size == mu.size and np.all(np.isfinite(hw)):
                            return x, mu, hw
                    else:
                        try:
                            mu = np.asarray(rep.get("mean", []), dtype=float)
                            if kind == "gr":
                                x = np.asarray(rep.get("r", rep.get("x", [])), dtype=float)
                            else:
                                x = np.asarray(rep.get("q", rep.get("x", [])), dtype=float)
                        except Exception:
                            x = np.asarray([], dtype=float)
                            mu = np.asarray([], dtype=float)
                        if (
                            x.ndim == 1
                            and mu.ndim == 1
                            and x.size >= 2
                            and mu.size == x.size
                            and hw.size == mu.size
                            and np.all(np.isfinite(x))
                            and np.all(np.diff(x) > 0.0)
                            and np.all(np.isfinite(mu))
                            and np.all(np.isfinite(hw))
                        ):
                            return x, mu, hw

            # fallback
            if kind == "gr":
                curves = [np.asarray(b["distributions"]["gr"][name]["g"], dtype=float) for b in src_boxes]
                x = np.asarray(src_boxes[0]["distributions"]["gr"][name]["r"], dtype=float)
            elif kind == "sq":
                curves = [np.asarray(b["distributions"]["sq"][name]["s"], dtype=float) for b in src_boxes]
                x = np.asarray(src_boxes[0]["distributions"]["sq"][name]["q"], dtype=float)
            else:
                payloads: list[Mapping[str, Any]] = []
                try:
                    for b in src_boxes:
                        fam = ((b.get("distributions", {}) or {}).get(kind, {}) or {})
                        payload = fam.get(name, None) if isinstance(fam, Mapping) else None
                        if not isinstance(payload, Mapping):
                            raise ValueError("missing CDF payload")
                        payloads.append(payload)
                    x, mat, _meta = _align_plot_cdf_payloads(
                        payloads,
                        allow_implicit_integer_axis=(kind == "coord"),
                    )
                except (KeyError, TypeError, ValueError):
                    return (
                        np.asarray([], dtype=float),
                        np.asarray([], dtype=float),
                        np.asarray([], dtype=float),
                    )
                curves = []
            if kind in {"gr", "sq"}:
                mat = np.vstack(curves)
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0, ddof=1)
            z = float((conv_report or {}).get("zscore", 2.0)) if isinstance(conv_report, dict) else 2.0
            hw = z * sd / math.sqrt(float(mat.shape[0]))
            return np.asarray(x, dtype=float), np.asarray(mu, dtype=float), np.asarray(hw, dtype=float)

        def _fig_md_vs_dft_density() -> "plt.Figure":
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            if not common_ids:
                ax.text(0.5, 0.5, "No common MD↔DFT boxes", ha="center", va="center")
                return fig
            md_ids = np.asarray(common_ids, dtype=int)
            md_vals = np.asarray([float(next(b for b in md_boxes if int(b.get("box_id", -1)) == i)["density"]) for i in md_ids], dtype=float)
            dft_vals = np.asarray([float(dft_by_id[int(i)]["density"]) for i in md_ids], dtype=float)
            ax.plot(md_ids, md_vals, marker="o", linestyle="None", label=f"MD (n={len(md_ids)})")
            ax.plot(md_ids, dft_vals, marker="s", linestyle="None", label=f"DFT-opt (n={len(md_ids)})")
            md_mu, md_hw = _scalar_summary("density", md_boxes, conv_report=conv_md)
            dft_mu, dft_hw = _scalar_summary("density", dft_boxes, conv_report=conv_dft)
            x0, x1 = float(np.min(md_ids)), float(np.max(md_ids))
            if math.isfinite(md_mu) and math.isfinite(md_hw):
                ax.axhline(md_mu, linewidth=1.0)
                ax.fill_between([x0, x1], [md_mu - md_hw, md_mu - md_hw], [md_mu + md_hw, md_mu + md_hw], alpha=0.12)
            if math.isfinite(dft_mu) and math.isfinite(dft_hw):
                ax.axhline(dft_mu, linewidth=1.0)
                ax.fill_between([x0, x1], [dft_mu - dft_hw, dft_mu - dft_hw], [dft_mu + dft_hw, dft_mu + dft_hw], alpha=0.12)
            ax.set_xlabel("box_id")
            ax.set_ylabel("density")
            ax.set_title("MD vs DFT-opt density")
            ax.grid(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            return fig

        def _fig_md_vs_dft_curve(kind: str, name: str, ylabel: str, title: str) -> "plt.Figure":
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            x_md, mu_md, hw_md = _curve_summary(kind, name, md_boxes, conv_report=conv_md)
            x_df, mu_df, hw_df = _curve_summary(kind, name, dft_boxes, conv_report=conv_dft)
            ax.plot(x_md, mu_md, label="MD")
            if mu_md.size == hw_md.size and mu_md.size > 0:
                ax.fill_between(x_md, mu_md - hw_md, mu_md + hw_md, alpha=0.12)
            ax.plot(x_df, mu_df, label="DFT-opt")
            if mu_df.size == hw_df.size and mu_df.size > 0:
                ax.fill_between(x_df, mu_df - hw_df, mu_df + hw_df, alpha=0.12)
            distance_unit = _distance_unit_label()
            ax.set_xlabel(
                f"r [{distance_unit}]"
                if kind in ("gr", "bondlen")
                else (
                    f"q [1/{distance_unit}]"
                    if kind == "sq"
                    else (
                        f"clearance r [{distance_unit}]"
                        if kind == "void"
                        else ("θ [deg]" if kind == "angle" else ("k" if kind == "coord" else "x"))
                    )
                )
            )
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            return fig

        def _fig_md_vs_dft_ring_pmf() -> "plt.Figure":
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            if not ring_keys:
                ax.text(0.5, 0.5, "No ring metrics available", ha="center", va="center")
                return fig
            sizes = []
            for k in ring_keys:
                try:
                    sizes.append(int(str(k).split("_")[-1]))
                except Exception:
                    sizes.append(len(sizes) + 1)
            order = np.argsort(np.asarray(sizes, dtype=int))
            sizes = [sizes[i] for i in order.tolist()]
            keys_sorted = [ring_keys[i] for i in order.tolist()]
            md_mu, md_hw, df_mu, df_hw = [], [], [], []
            for k in keys_sorted:
                mu, hw = _scalar_summary(str(k), md_boxes, conv_report=conv_md)
                md_mu.append(mu)
                md_hw.append(hw)
                mu, hw = _scalar_summary(str(k), dft_boxes, conv_report=conv_dft)
                df_mu.append(mu)
                df_hw.append(hw)
            x = np.asarray(sizes, dtype=float)
            ax.errorbar(x - 0.1, md_mu, yerr=md_hw, fmt="o", label="MD")
            ax.errorbar(x + 0.1, df_mu, yerr=df_hw, fmt="s", label="DFT-opt")
            ax.set_xlabel("ring size")
            ax.set_ylabel("fraction")
            ax.set_title("MD vs DFT-opt ring fractions")
            ax.grid(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            return fig

        pages.append(("MD vs DFT: density", _fig_md_vs_dft_density()))
        if ring_keys:
            pages.append(("MD vs DFT: rings", _fig_md_vs_dft_ring_pmf()))
        for nm in bond_names:
            if nm in (dft_boxes[0].get("distributions", {}) or {}).get("bondlen", {}):
                pages.append((f"MD vs DFT: bondlen {nm}", _fig_md_vs_dft_curve("bondlen", nm, "CDF", f"Bond length CDF: {nm}")))
        for nm in angle_names:
            if nm in (dft_boxes[0].get("distributions", {}) or {}).get("angle", {}):
                pages.append((f"MD vs DFT: angle {nm}", _fig_md_vs_dft_curve("angle", nm, "CDF", f"Angle CDF: {nm}")))
        for nm in void_names:
            if nm in (dft_boxes[0].get("distributions", {}) or {}).get("void", {}):
                pages.append((f"MD vs DFT: void {nm}", _fig_md_vs_dft_curve("void", nm, "CDF", f"Void clearance CDF: {nm}")))
        for nm in coord_names:
            if nm in (dft_boxes[0].get("distributions", {}) or {}).get("coord", {}):
                pages.append((f"MD vs DFT: coord {nm}", _fig_md_vs_dft_curve("coord", nm, "CDF", f"Coordination CDF: {nm}")))
        for lab in gr_labels:
            if lab in (dft_boxes[0].get("distributions", {}) or {}).get("gr", {}):
                pages.append((f"MD vs DFT: g(r) {lab}", _fig_md_vs_dft_curve("gr", lab, "g(r)", f"g(r): {lab}")))
        for lab in sq_labels:
            if lab in (dft_boxes[0].get("distributions", {}) or {}).get("sq", {}):
                pages.append((f"MD vs DFT: S(q) {lab}", _fig_md_vs_dft_curve("sq", lab, "S(q)", f"S(q): {lab}")))

    # enforce pages convergence
    if max_pages is not None and int(max_pages) > 0:
        keep = 1 + int(max_pages)
        pages = pages[:keep]

    _save_plot_pages(
        pages,
        out_path,
        dpi=int(dpi),
        name_cleaner=lambda nm: _clean_name(nm).replace(" ", "_").replace("–", "-"),
    )


def plot_metrics_timeseries(
    input_csv: Path,
    output: Path,
    *,
    xaxis: str = "time",
    metrics: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    dpi: int = 600,
    max_pages: Optional[int] = None,
) -> None:
    """Metrics timeseries."""

    import csv

    from matplotlib import pyplot as plt

    _apply_publication_style(base_fontsize=10.0)

    dpi = _require_positive_integer(dpi, context="dpi")
    in_path = Path(input_csv)
    out_path = Path(output)
    if max_pages is not None:
        max_pages = _require_positive_integer(max_pages, context="max_pages")

    # read numeric csv
    with in_path.open("r", newline="", errors="replace") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration as e:
            raise ValueError(f"Empty CSV: {in_path}") from e
        cols = [str(x).strip() for x in header if str(x).strip() != ""]
        if not cols:
            raise ValueError(f"Invalid CSV header: {in_path}")
        rows: list[list[float]] = []
        for row in r:
            if not row:
                continue
            if len(row) < len(cols):
                row = list(row) + ["nan"] * (len(cols) - len(row))
            try:
                rows.append([float(x) for x in row[: len(cols)]])
            except Exception:
                continue
    if not rows:
        raise ValueError(f"No numeric rows parsed from {in_path}")
    data = np.asarray(rows, dtype=float)
    col_idx = {c: i for i, c in enumerate(cols)}

    if xaxis not in ("time", "step"):
        raise ValueError("xaxis must be 'time' or 'step'")
    xcol = "time" if xaxis == "time" else "Step"
    if xcol not in col_idx:
        raise ValueError(f"Missing required column for xaxis='{xaxis}': {xcol}")
    x = data[:, col_idx[xcol]]
    if not np.any(np.isfinite(x)):
        raise ValueError(
            f"metrics-timeseries x column {xcol!r} contains no finite values"
        )

    # determine metric plot
    skip = {"Step", "time"}
    cand = [c for c in cols if c not in skip]
    if metrics is not None:
        req = [str(m) for m in metrics]
        missing = [name for name in req if name not in cand]
        if missing:
            raise ValueError(
                "Requested metric column(s) are absent from the CSV: "
                + ", ".join(missing)
            )
        req_set = set(req)
        cand = [c for c in cand if c in req_set]
    if not cand:
        raise ValueError("No metric columns selected")

    def _draw_metric(ax: Any, name: str) -> None:
        """Plot available samples or explicitly report an unavailable metric.

        Some valid structural descriptors are undefined for a particular set
        of frames.  For example, a first-peak FWHM has no value unless both
        half-height crossings are bracketed.  The CSV deliberately represents
        that state as NaN.  A requested diagnostic plot must preserve and
        disclose the unavailable state rather than either inventing a value or
        aborting an otherwise valid production box.
        """

        y = np.asarray(data[:, col_idx[name]], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            # Keep unavailable samples as gaps so a line cannot imply evidence
            # across a frame whose descriptor was undefined.
            x_plot = np.where(finite, x, np.nan)
            y_plot = np.where(finite, y, np.nan)
            ax.plot(x_plot, y_plot)
        else:
            ax.text(
                0.5,
                0.5,
                "Metric unavailable: no finite values in selected frames",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        ax.set_xlabel(xcol)
        ax.set_ylabel(str(name))
        ax.grid(False)
        if title is not None:
            ax.set_title(str(title))

    def _slug(s: str) -> str:
        out = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(s))
        out = out.strip("_")
        return out[:180] if len(out) > 180 else out

    # output routing
    if out_path.suffix.lower() == ".pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out_path) as pdf:
            for i, nm in enumerate(cand):
                if max_pages is not None and i >= int(max_pages):
                    break
                fig = plt.figure(figsize=(6.5, 4.0), dpi=int(dpi))
                ax = fig.add_subplot(1, 1, 1)
                _draw_metric(ax, nm)
                _style_figure(fig)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return

    if out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
        for i, nm in enumerate(cand):
            if max_pages is not None and i >= int(max_pages):
                break
            fig = plt.figure(figsize=(6.5, 4.0), dpi=int(dpi))
            ax = fig.add_subplot(1, 1, 1)
            _draw_metric(ax, nm)
            _style_figure(fig)
            fig.tight_layout()
            fig.savefig(out_path / f"{_slug(nm)}.png", dpi=int(dpi))
            plt.close(fig)
        return

    # page output metric
    nm = cand[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.5, 4.0), dpi=int(dpi))
    ax = fig.add_subplot(1, 1, 1)
    _draw_metric(ax, nm)
    _style_figure(fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _prepare_production_plot_payload(json_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load either legacy production results or standalone analysis results.

    ``plot-production`` historically consumed ``autotune_results.json`` files with a
    top-level ``production`` block. ``analyze-output`` now emits standalone
    ``analysis_results.json`` files with the same scientific content but at the
    top level. This helper normalises both layouts into a production-like payload
    so the plotting code can work with either input shape.
    """

    path = Path(json_path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Production plotting input must be a JSON object")

    prod_raw = data.get("production", None)
    if isinstance(prod_raw, Mapping) and bool(prod_raw.get("enabled", False)):
        return data, dict(prod_raw)

    schema = str(data.get("schema", "") or "").strip().lower()
    looks_like_analysis = schema.startswith("vitriflow.analysis_results") or (
        isinstance(data.get("boxes", None), list)
        and isinstance(data.get("convergence", None), Mapping)
        and isinstance(data.get("convergence_spec", None), Mapping)
    )
    if not looks_like_analysis:
        raise RuntimeError(
            "Input is neither a production-enabled autotune_results.json nor a standalone analysis_results.json."
        )

    boxes: list[dict[str, Any]] = []
    for entry in list(data.get("boxes", []) or []):
        if not isinstance(entry, Mapping):
            continue
        box = dict(entry)
        if box.get("box_id", None) is None and box.get("box", None) is not None:
            # Preserve the serialized value for the strict validator.  An
            # eager int() here used to coerce True and fractional ids before
            # validation, masking malformed or ambiguous provenance.
            box["box_id"] = box.get("box")
        boxes.append(box)

    conv_norm = dict(data.get("convergence", {}) or {})

    def _load_ensemble_cdf_sidecar() -> dict[str, Any]:
        ref = data.get("ensemble_cdfs", None)
        if not isinstance(ref, Mapping):
            return {}
        rel = ref.get("path", None)
        if not rel:
            return {}
        sidecar_path = (path.parent / str(rel)).resolve(strict=False)
        if not sidecar_path.exists():
            return {}
        try:
            payload = json.loads(sidecar_path.read_text())
        except Exception:
            return {}
        if not isinstance(payload, Mapping):
            return {}
        out: dict[str, Any] = {}
        direct = payload.get("distributions", {})
        if isinstance(direct, Mapping):
            for name, entry in direct.items():
                if isinstance(entry, Mapping):
                    out[str(name)] = dict(entry)
        families = payload.get("families", {})
        if isinstance(families, Mapping):
            for _family, mapping in families.items():
                if not isinstance(mapping, Mapping):
                    continue
                for name, entry in mapping.items():
                    if not isinstance(entry, Mapping):
                        continue
                    d = dict(entry)
                    if "mean" not in d and "mean_cdf" in d:
                        d["mean"] = d.get("mean_cdf")
                    if "cdf" not in d and "mean_cdf" in d:
                        d["cdf"] = d.get("mean_cdf")
                    out.setdefault(str(name), d)
        return out

    sidecar_dists = _load_ensemble_cdf_sidecar()
    if sidecar_dists:
        conv_d = dict(conv_norm.get("distributions", {}) or {})
        for name, entry in sidecar_dists.items():
            d = dict(entry)
            if "mean" not in d and "mean_cdf" in d:
                d["mean"] = d.get("mean_cdf")
            if "cdf" not in d and "mean" in d:
                d["cdf"] = d.get("mean")
            conv_d.setdefault(str(name), d)
        conv_norm["distributions"] = conv_d
        conv_norm.setdefault("ensemble_cdf_sidecar_loaded", True)

    prod = {
        "enabled": True,
        "boxes": boxes,
        "convergence": conv_norm,
        "convergence_spec": dict(data.get("convergence_spec", {}) or {}),
        # Preserve tri-state convergence.  None/missing means unassessed, not
        # failure, and must remain distinguishable in plot labels.
        "converged": data.get("converged", None),
        "check_convergence": data.get("check_convergence", None),
        "convergence_status": data.get("convergence_status", None),
        "convergence_inference_status": data.get(
            "convergence_inference_status",
            conv_norm.get("convergence_inference_status", None),
        ),
        "posthoc_convergence_criterion_met": data.get(
            "posthoc_convergence_criterion_met",
            conv_norm.get("posthoc_criterion_met", None),
        ),
        "n_boxes": int(data.get("n_boxes", len(boxes)) or len(boxes)),
        "n_boxes_accepted": int(data.get("n_boxes_accepted", len(boxes)) or len(boxes)),
        "n_boxes_rejected": int(data.get("n_boxes_rejected", 0) or 0),
        "n_boxes_total": int(data.get("n_boxes_total", len(boxes)) or len(boxes)),
    }

    normalised = dict(data)
    normalised.setdefault("units", {})
    normalised["production"] = prod
    return normalised, prod


def _default_production_results_label(path: Path) -> str:
    p = Path(path)
    stem = str(p.stem)
    parent = str(p.parent.name)
    if stem in {"analysis_results", "autotune_results", "run_results"} and parent:
        return parent
    if stem:
        return stem
    if parent:
        return parent
    return str(p)


def _save_plot_pages(
    pages: Sequence[tuple[str, Any]],
    out_path: Path,
    *,
    dpi: int,
    name_cleaner: Optional[Callable[[str], str]] = None,
) -> None:
    if not pages:
        raise ValueError("No plot pages were generated")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _fallback_clean_name(name: str) -> str:
        return str(name).replace(" ", "_").replace("/", "-")

    cleaner = name_cleaner or _fallback_clean_name

    if out_path.suffix.lower() == ".pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(out_path) as pdf:
            for _name, fig in pages:
                _style_figure(fig)
                pdf.savefig(fig)
                import matplotlib.pyplot as plt

                plt.close(fig)
    else:
        outdir = out_path
        outdir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        for i, (nm, fig) in enumerate(pages, start=1):
            fn = outdir / f"{i:02d}_{cleaner(str(nm))}.png"
            _style_figure(fig)
            fig.savefig(fn, dpi=int(dpi))
            plt.close(fig)
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except Exception:
        pass


def plot_production_comparison_results(
    json_paths: Sequence[Path],
    out_path: Path,
    *,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    dpi: int = 600,
    max_pages: Optional[int] = None,
) -> None:
    """Compare production/analysis ensembles with exception-safe figures."""

    try:
        _plot_production_comparison_results_impl(
            json_paths,
            out_path,
            labels=labels,
            title=title,
            dpi=dpi,
            max_pages=max_pages,
        )
    except Exception:
        # Comparison materialises several pages before later schema errors can
        # be discovered.  Do not leak those figures into long-lived Python or
        # validation processes when the operation fails.
        import matplotlib.pyplot as plt

        plt.close("all")
        raise


def _plot_production_comparison_results_impl(
    json_paths: Sequence[Path],
    out_path: Path,
    *,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    dpi: int = 600,
    max_pages: Optional[int] = None,
) -> None:
    """Compare production/analysis ensembles across multiple datasets.

    The inputs may be either legacy ``autotune_results.json`` files with a
    production block or standalone ``analysis_results.json`` files emitted by
    ``analyze-output``.
    """

    import math
    from statistics import NormalDist

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = _require_positive_integer(dpi, context="dpi")
    _apply_publication_style()

    if max_pages is not None:
        max_pages = _require_positive_integer(max_pages, context="max_pages")

    paths = [Path(p) for p in list(json_paths)]
    if len(paths) < 2:
        raise RuntimeError("At least two result files are required for comparison plotting.")

    if labels is not None and len(list(labels)) not in {0, len(paths)}:
        raise RuntimeError("When provided, --labels must match the number of input files.")

    user_labels = list(labels) if labels is not None else []

    def _critical_value(n: int, alpha: float) -> tuple[float, str]:
        a = float(min(1.0, max(0.0, alpha)))
        if int(n) < 2:
            return float("inf"), "n<2"
        try:
            from scipy.stats import t as _t  # type: ignore

            crit = float(_t.ppf(1.0 - a / 2.0, df=int(n) - 1))
            if math.isfinite(crit):
                return crit, "t"
        except Exception:
            pass
        crit = float(NormalDist().inv_cdf(1.0 - a / 2.0))
        return crit, "z"

    def _clean_name(nm: str) -> str:
        s = str(nm)
        s = s.replace("bondlen_", "Bond length: ")
        s = s.replace("angle_", "Angle: ")
        s = s.replace("coord_", "Coordination: ")
        s = s.replace("gr_", "g(r): ")
        s = s.replace("sq_", "S(q): ")
        s = s.replace("void_", "Void: ")
        s = s.replace("ring_frac_", "Ring fraction: ")
        s = s.replace("_", " ")
        s = s.replace("-", "–")
        return s

    def _slug_name(nm: str) -> str:
        return _clean_name(nm).replace(" ", "_").replace("–", "-").replace("/", "-")

    def _distance_unit_label(units_style: str, *, canonical: bool = False) -> str:
        if canonical:
            return "Å"
        return _native_length_unit_label(units_style) or "distance units"

    def _density_unit_label(units_style: str, *, canonical: bool = False) -> str:
        if canonical:
            return "g/cm³"
        return _thermo_unit_label("Density", units_style) or "density units"

    datasets: list[dict[str, Any]] = []
    seen_labels: dict[str, int] = {}
    for i, path in enumerate(paths):
        data, prod = _prepare_production_plot_payload(path)
        label = str(user_labels[i]).strip() if i < len(user_labels) and str(user_labels[i]).strip() else _default_production_results_label(path)
        if label in seen_labels:
            seen_labels[label] += 1
            label = f"{label} ({seen_labels[label]})"
        else:
            seen_labels[label] = 1
        raw_boxes = list(prod.get("boxes", []) or [])
        if not raw_boxes:
            raise RuntimeError(f"No production boxes found in {path}")
        boxes = _normalise_production_plot_boxes(raw_boxes)
        datasets.append(
            {
                "label": label,
                "path": path,
                "data": data,
                "prod": prod,
                "boxes": boxes,
                "conv": dict(prod.get("convergence", {}) or {}),
                "spec": dict(prod.get("convergence_spec", {}) or {}),
                "units_style": str((data.get("units", {}) or {}).get("lammps_units", "") or "").strip().lower(),
                "canonical_units": _uses_canonical_report_units(data),
            }
        )

    canonical_flags = {bool(ds["canonical_units"]) for ds in datasets}
    if len(canonical_flags) > 1:
        raise ValueError(
            "Cannot compare a mixture of canonical-reporting and legacy/native-unit "
            "datasets without an explicit conversion contract"
        )
    if canonical_flags == {False}:
        legacy_styles = {
            str(ds["units_style"]) for ds in datasets if str(ds["units_style"])
        }
        if len(legacy_styles) > 1:
            raise ValueError(
                "Cannot compare legacy production datasets with different LAMMPS "
                f"unit styles: {sorted(legacy_styles)}"
            )

    def _metric_keys(ds: dict[str, Any]) -> set[str]:
        keys: set[str] = set()
        for b in ds["boxes"]:
            m = b.get("metrics", {}) or {}
            if isinstance(m, Mapping):
                keys.update(str(k) for k in m.keys())
        return keys

    def _dist_names(ds: dict[str, Any], kind: str, spec_key: str) -> set[str]:
        names: set[str] = set()
        spec_names = ds["spec"].get(spec_key, None)
        if isinstance(spec_names, list) and len(spec_names) > 0:
            names.update(str(x) for x in spec_names)
        for box in ds["boxes"]:
            try:
                family = ((box.get("distributions", {}) or {}).get(kind, {}) or {})
            except Exception:
                family = {}
            if isinstance(family, Mapping):
                names.update(str(key) for key in family)
        return names

    def _dist_entry(box: Mapping[str, Any], kind: str, name: str) -> dict[str, Any]:
        if not isinstance(box, Mapping):
            return {}
        return (((box.get("distributions", {}) or {}).get(kind, {}) or {}).get(name, {}) or {})

    def _scalar_points(ds: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
        ids: list[float] = []
        vals: list[float] = []
        for idx, b in enumerate(ds["boxes"], start=1):
            bid = b.get("box_id", b.get("box", idx))
            try:
                bid_f = float(bid)
            except Exception:
                bid_f = float(idx)
            if str(key).lower() in {"density", "rho"}:
                vv = _maybe_float(b.get("density"))
            else:
                vv = _maybe_float((b.get("metrics", {}) or {}).get(str(key)))
            if np.isfinite(vv):
                ids.append(bid_f)
                vals.append(vv)
        return np.asarray(ids, dtype=float), np.asarray(vals, dtype=float)

    def _scalar_summary(ds: dict[str, Any], key: str) -> tuple[float, float]:
        if str(key).lower() in {"density", "rho"}:
            rep = ((ds["conv"].get("scalars", {}) or {}).get("density", None))
        else:
            rep = ((ds["conv"].get("scalars", {}) or {}).get(str(key), None))
        if isinstance(rep, Mapping):
            mu = _maybe_float(rep.get("mean"))
            half = _maybe_float(rep.get("ci_halfwidth"))
            if np.isfinite(mu) and np.isfinite(half):
                return float(mu), float(half)
        _ids, vals = _scalar_points(ds, key)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), float("nan")
        mu = float(np.mean(vals))
        if vals.size < 2:
            return mu, float("nan")
        alpha = _maybe_float(((ds["conv"].get("familywise", {}) or {}).get("alpha_per_test", 0.05)), default=0.05)
        if not (math.isfinite(alpha) and alpha > 0.0 and alpha < 1.0):
            alpha = 0.05
        crit, _ = _critical_value(int(vals.size), alpha)
        sd = float(np.std(vals, ddof=1))
        return mu, float(crit * sd / math.sqrt(float(vals.size)))

    def _curve_summary(ds: dict[str, Any], kind: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rep = ((ds["conv"].get("distributions", {}) or {}).get(str(name), None))
        if isinstance(rep, Mapping):
            try:
                half = np.asarray(rep.get("ci_halfwidth", []), dtype=float)
            except Exception:
                half = np.asarray([], dtype=float)
            if kind in {"bondlen", "angle", "coord", "void"}:
                report_payload: dict[str, Any] = {"cdf": rep.get("mean", [])}
                if "x" in rep:
                    report_payload["x"] = rep.get("x")
                try:
                    x, mu, _axis_source = _validated_plot_cdf_payload(
                        report_payload,
                        allow_implicit_integer_axis=(kind == "coord"),
                    )
                except ValueError:
                    x = np.asarray([], dtype=float)
                    mu = np.asarray([], dtype=float)
            else:
                try:
                    if kind == "gr":
                        x = np.asarray(rep.get("r", rep.get("x", [])), dtype=float)
                    else:
                        x = np.asarray(rep.get("q", rep.get("x", [])), dtype=float)
                    mu = np.asarray(rep.get("mean", []), dtype=float)
                except Exception:
                    x = np.asarray([], dtype=float)
                    mu = np.asarray([], dtype=float)
            if (
                x.ndim == 1
                and mu.ndim == 1
                and x.size > 0
                and mu.size == x.size
                and half.size == x.size
                and np.all(np.isfinite(x))
                and np.all(np.isfinite(mu))
                and (x.size == 1 or np.all(np.diff(x) > 0.0))
            ):
                return x, mu, half

        boxes = ds["boxes"]
        if kind == "gr":
            curves = [np.asarray(_dist_entry(b, "gr", name).get("g", []), dtype=float) for b in boxes]
            x = np.asarray(_dist_entry(boxes[0], "gr", name).get("r", []), dtype=float)
        elif kind == "sq":
            curves = [np.asarray(_dist_entry(b, "sq", name).get("s", []), dtype=float) for b in boxes]
            x = np.asarray(_dist_entry(boxes[0], "sq", name).get("q", []), dtype=float)
        else:
            payloads = [_dist_entry(b, kind, name) for b in boxes]
            try:
                x, X, _meta = _align_plot_cdf_payloads(
                    payloads,
                    allow_implicit_integer_axis=(kind == "coord"),
                )
            except (TypeError, ValueError):
                return (
                    np.asarray([], dtype=float),
                    np.asarray([], dtype=float),
                    np.asarray([], dtype=float),
                )
            curves = []
        if x.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
        if kind in {"gr", "sq"}:
            mats = [c for c in curves if c.size == x.size]
            if not mats:
                return x, np.asarray([], dtype=float), np.asarray([], dtype=float)
            X = np.vstack(mats)
        mu = np.nanmean(X, axis=0)
        if X.shape[0] < 2:
            return x, mu, np.full_like(mu, np.nan)
        alpha = _maybe_float(((ds["conv"].get("familywise", {}) or {}).get("alpha_per_test", 0.05)), default=0.05)
        if not (math.isfinite(alpha) and alpha > 0.0 and alpha < 1.0):
            alpha = 0.05
        crit, _ = _critical_value(int(X.shape[0]), alpha)
        sd = np.nanstd(X, axis=0, ddof=1)
        half = crit * sd / math.sqrt(float(X.shape[0]))
        return x, mu, half

    def _compute_trend(ds: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Optional[int]]:
        boxes = list(ds["boxes"])
        conv = ds["conv"]
        spec = ds["spec"]
        N = int(len(boxes))
        if N < 2:
            return np.asarray([], dtype=int), np.asarray([], dtype=float), None

        fw = conv.get("familywise", {}) or {}
        alpha_test = float(fw.get("alpha_per_test", float("nan")))
        if not (math.isfinite(alpha_test) and 0.0 < alpha_test < 1.0):
            alpha_test = 0.05
        bounded_ci_method = str(fw.get("bounded_ci_method", "t")).strip().lower()
        if bounded_ci_method not in {"t", "empirical_bernstein", "hoeffding"}:
            bounded_ci_method = "t"

        bond_names = list(spec.get("bondlen_names", []))
        angle_names = list(spec.get("angle_names", []))
        coord_names = list(spec.get("coord_names", []))
        ring_keys = list(spec.get("ring_keys", []))
        gr_labels = list(spec.get("gr_labels", []))
        sq_labels = list(spec.get("sq_labels", []))
        void_names = list(spec.get("void_names", []))
        has_ring_mean = bool(spec.get("ring_has_mean_size", False))

        dens = np.asarray([float(b.get("density", float("nan"))) for b in boxes], dtype=float)
        dens_rep = ((conv.get("scalars", {}) or {}).get("density", {}) or {})
        dens_abs = float(dens_rep.get("abs_tol", 0.0))
        dens_rel = float(dens_rep.get("rel_tol", 0.0))

        ring_rep = ((conv.get("distributions", {}) or {}).get("ring", {}) or {})
        ring_abs = float(ring_rep.get("abs_tol", 0.0))
        ring_rel = float(ring_rep.get("rel_tol", 0.0))

        ring_mat = None
        if ring_keys:
            ring_mat = np.zeros((N, len(ring_keys)), dtype=float)
            for i_box, b in enumerate(boxes):
                m = b.get("metrics", {}) or {}
                for j, k in enumerate(ring_keys):
                    ring_mat[i_box, j] = float(m.get(k, float("nan")))

        ring_mean_rep = ((conv.get("scalars", {}) or {}).get("ring_mean_size", {}) or {})
        ring_mean_abs = float(ring_mean_rep.get("abs_tol", 0.0))
        ring_mean_rel = float(ring_mean_rep.get("rel_tol", 0.0))
        ring_mean_vec = None
        if has_ring_mean:
            ring_mean_vec = np.asarray([float((b.get("metrics", {}) or {}).get("ring_mean_size", float("nan"))) for b in boxes], dtype=float)

        curve_mats: list[dict[str, Any]] = []
        cdf_evidence_blocked = False

        def _stack_cdf(
            kind: str,
            name: str,
            xkey: str = "x",
            *,
            allow_implicit_integer_axis: bool = False,
        ) -> None:
            nonlocal cdf_evidence_blocked
            drep = (conv.get("distributions", {}) or {}).get(name, None)
            if not isinstance(drep, Mapping):
                cdf_evidence_blocked = True
                return
            payloads = [_dist_entry(b, kind, name) for b in boxes]
            try:
                _x0, mats, _meta = _align_plot_cdf_payloads(
                    payloads,
                    xkey=xkey,
                    allow_implicit_integer_axis=allow_implicit_integer_axis,
                )
            except (TypeError, ValueError):
                cdf_evidence_blocked = True
                return
            curve_mats.append(
                {
                    "group": drep.get("group", ""),
                    "kind": str(drep.get("kind", "")),
                    "name": str(name),
                    "mat": mats,
                    "abs_tol": float(drep.get("abs_tol", 0.0)),
                    "rel_tol": float(drep.get("rel_tol", 0.0)),
                }
            )

        for nm in bond_names:
            _stack_cdf("bondlen", nm)
        for nm in angle_names:
            _stack_cdf("angle", nm)
        for nm in void_names:
            _stack_cdf("void", nm)

        for nm in coord_names:
            _stack_cdf(
                "coord",
                nm,
                allow_implicit_integer_axis=True,
            )

        for lab in gr_labels:
            drep = (conv.get("distributions", {}) or {}).get(lab, None)
            if not isinstance(drep, Mapping):
                continue
            mats = []
            for b in boxes:
                g = np.asarray(_dist_entry(b, "gr", lab).get("g", []), dtype=float)
                if g.size == 0:
                    mats = []
                    break
                mats.append(g)
            if not mats:
                continue
            X = np.vstack(mats)
            curve_mats.append(
                {
                    "group": drep.get("group", ""),
                    "kind": str(drep.get("kind", "")),
                    "name": str(lab),
                    "mat": X,
                    "abs_tol": float(drep.get("abs_tol", 0.0)),
                    "rel_tol": float(drep.get("rel_tol", 0.0)),
                }
            )

        for lab in sq_labels:
            drep = (conv.get("distributions", {}) or {}).get(lab, None)
            if not isinstance(drep, Mapping):
                continue
            mats = []
            for b in boxes:
                s = np.asarray(_dist_entry(b, "sq", lab).get("s", []), dtype=float)
                if s.size == 0:
                    mats = []
                    break
                mats.append(s)
            if not mats:
                continue
            X = np.vstack(mats)
            curve_mats.append(
                {
                    "group": drep.get("group", ""),
                    "kind": str(drep.get("kind", "")),
                    "name": str(lab),
                    "mat": X,
                    "abs_tol": float(drep.get("abs_tol", 0.0)),
                    "rel_tol": float(drep.get("rel_tol", 0.0)),
                }
            )

        def _mean_sd_from_sums(sum_x: np.ndarray, sum_x2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
            mu = sum_x / float(n)
            if n < 2:
                return mu, np.full_like(mu, np.nan, dtype=float)
            var = (sum_x2 - (sum_x * sum_x) / float(n)) / float(n - 1)
            var = np.maximum(var, 0.0)
            return mu, np.sqrt(var)

        def _ratio_from_mu_sd(
            mu: np.ndarray,
            sd: np.ndarray,
            n: int,
            abs_tol: float,
            rel_tol: float,
            crit: float,
            *,
            bounded: bool,
        ) -> float:
            mu = np.asarray(mu, dtype=float)
            sd = np.asarray(sd, dtype=float)
            if bounded:
                a = float(alpha_test)
                if int(n) < 2 or (not math.isfinite(a)) or a <= 0.0 or a >= 1.0:
                    half = np.full_like(sd, np.inf, dtype=float)
                elif bounded_ci_method == "t":
                    half = float(crit) * (sd / math.sqrt(float(n)))
                elif bounded_ci_method == "hoeffding":
                    hw = math.sqrt(math.log(2.0 / a) / (2.0 * float(n)))
                    half = np.full_like(sd, float(hw), dtype=float)
                else:
                    v = np.square(sd)
                    v = np.minimum(v, 0.25)
                    L = math.log(3.0 / a)
                    half = np.sqrt(2.0 * v * L / float(n)) + 3.0 * L / float(n)
            else:
                se = sd / math.sqrt(float(n))
                half = float(crit) * se
            tol = np.maximum(float(abs_tol), float(rel_tol) * np.abs(mu))
            tol = np.maximum(tol, 1e-30)
            return float(np.nanmax(half / tol))

        n_grid = np.arange(2, N + 1, dtype=int)
        trend: list[float] = []

        dens_S = np.cumsum(dens)
        dens_Q = np.cumsum(dens * dens)
        ring_S = np.cumsum(ring_mat, axis=0) if ring_mat is not None else None
        ring_Q = np.cumsum(ring_mat * ring_mat, axis=0) if ring_mat is not None else None
        rmean_S = np.cumsum(ring_mean_vec) if ring_mean_vec is not None else None
        rmean_Q = np.cumsum(ring_mean_vec * ring_mean_vec) if ring_mean_vec is not None else None
        mats_sums = [{"meta": cm, "S": np.cumsum(cm["mat"], axis=0), "Q": np.cumsum(cm["mat"] * cm["mat"], axis=0)} for cm in curve_mats]

        for n in n_grid.tolist():
            crit, _ = _critical_value(int(n), alpha_test)
            gmax = float("inf") if cdf_evidence_blocked else 0.0

            mu, sd = _mean_sd_from_sums(dens_S[n - 1], dens_Q[n - 1], n)
            gmax = max(gmax, _ratio_from_mu_sd(np.asarray([mu]), np.asarray([sd]), n, dens_abs, dens_rel, crit, bounded=False))

            if ring_keys and ring_S is not None and ring_Q is not None and (ring_abs > 0.0 or ring_rel > 0.0):
                mu, sd = _mean_sd_from_sums(ring_S[n - 1, :], ring_Q[n - 1, :], n)
                gmax = max(gmax, _ratio_from_mu_sd(mu, sd, n, ring_abs, ring_rel, crit, bounded=True))

            if ring_mean_vec is not None and rmean_S is not None and rmean_Q is not None and (ring_mean_abs > 0.0 or ring_mean_rel > 0.0):
                mu, sd = _mean_sd_from_sums(rmean_S[n - 1], rmean_Q[n - 1], n)
                gmax = max(gmax, _ratio_from_mu_sd(np.asarray([mu]), np.asarray([sd]), n, ring_mean_abs, ring_mean_rel, crit, bounded=False))

            for ms in mats_sums:
                meta = ms["meta"]
                mu, sd = _mean_sd_from_sums(ms["S"][n - 1, :], ms["Q"][n - 1, :], n)
                kind = str(meta.get("kind", ""))
                bounded = bool(kind in ("bondlen_cdf", "angle_cdf", "coord_cdf", "void_cdf"))
                gmax = max(
                    gmax,
                    _ratio_from_mu_sd(
                        mu,
                        sd,
                        n,
                        float(meta.get("abs_tol", 0.0)),
                        float(meta.get("rel_tol", 0.0)),
                        crit,
                        bounded=bounded,
                    ),
                )
            trend.append(gmax)

        arr = np.asarray(trend, dtype=float)
        m = np.isfinite(arr)
        arr[m] = np.maximum(arr[m], 1e-12)
        n_conv = None
        ok_mask = np.isfinite(arr) & (arr <= 1.0)
        if np.any(ok_mask):
            n_conv = int(n_grid[np.argmax(ok_mask)])
        return n_grid, arr, n_conv

    def _declared_scalar_metric_keys(ds: dict[str, Any]) -> set[str]:
        keys = set(_metric_keys(ds))
        scalar_report = ds["conv"].get("scalars", {}) or {}
        if isinstance(scalar_report, Mapping):
            keys.update(str(key) for key in scalar_report)
        return keys

    # Comparison is an audit surface, so use the deterministic union rather
    # than silently discarding metrics that are absent from one dataset.  The
    # individual pages explicitly identify unavailable datasets below.
    comparison_metric_keys = sorted(
        set.union(*[_declared_scalar_metric_keys(ds) for ds in datasets])
        if datasets
        else set()
    )
    comparison_metric_keys = [
        key
        for key in comparison_metric_keys
        if key not in {"density", "ring_mean_size"}
        and not str(key).startswith("ring_frac_")
    ]

    comparison_ring_keys = sorted(
        set.union(
            *[
                {
                    *(
                        str(x)
                        for x in list(ds["spec"].get("ring_keys", []) or [])
                    ),
                    *(
                        key
                        for key in _declared_scalar_metric_keys(ds)
                        if str(key).startswith("ring_frac_")
                    ),
                }
                for ds in datasets
            ]
        )
        if datasets
        else set(),
        key=lambda x: int(str(x).split("ring_frac_")[-1]) if str(x).startswith("ring_frac_") else str(x),
    )

    def _comparison_dist_names(kind: str, spec_key: str) -> list[str]:
        return sorted(
            set.union(*[_dist_names(ds, kind, spec_key) for ds in datasets])
            if datasets
            else set()
        )

    comparison_bond_names = _comparison_dist_names("bondlen", "bondlen_names")
    comparison_angle_names = _comparison_dist_names("angle", "angle_names")
    comparison_coord_names = _comparison_dist_names("coord", "coord_names")
    comparison_void_names = _comparison_dist_names("void", "void_names")
    comparison_gr_labels = _comparison_dist_names("gr", "gr_labels")
    comparison_sq_labels = _comparison_dist_names("sq", "sq_labels")

    pages: list[tuple[str, Any]] = []

    primary_units = next((str(ds["units_style"]) for ds in datasets if str(ds["units_style"])), "")
    all_canonical = all(bool(ds.get("canonical_units", False)) for ds in datasets)
    dist_unit = _distance_unit_label(primary_units, canonical=all_canonical)
    density_ylabel = f"density [{_density_unit_label(primary_units, canonical=all_canonical)}]"

    def _alpha_for_dataset(ds: dict[str, Any]) -> float:
        alpha = _maybe_float(((ds["conv"].get("familywise", {}) or {}).get("alpha_per_test", 0.05)), default=0.05)
        if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
            alpha = 0.05
        return float(alpha)

    def _scalar_values(ds: dict[str, Any], key: str) -> np.ndarray:
        _ids, vals = _scalar_points(ds, key)
        return vals[np.isfinite(vals)]

    def _overlay_line_with_band(
        ax: Any,
        x: np.ndarray,
        mu: np.ndarray,
        half: np.ndarray,
        *,
        label: str,
    ) -> None:
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        half = np.asarray(half, dtype=float)
        m = np.isfinite(x) & np.isfinite(mu)
        if not np.any(m):
            return
        line = ax.plot(x[m], mu[m], label=label)[0]
        col = line.get_color()
        mb = m & np.isfinite(half)
        if np.any(mb):
            ax.fill_between(x[mb], mu[mb] - half[mb], mu[mb] + half[mb], color=col, alpha=0.12, linewidth=0.0)

    def _raw_distribution_matrix(ds: dict[str, Any], kind: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        boxes = list(ds["boxes"])
        if not boxes:
            return np.asarray([], dtype=float), np.empty((0, 0), dtype=float)
        kind = str(kind)
        name = str(name)
        if kind in {"bondlen", "angle", "coord", "void"}:
            try:
                return _align_plot_cdf_payloads(
                    [_dist_entry(b, kind, name) for b in boxes],
                    allow_implicit_integer_axis=(kind == "coord"),
                )[:2]
            except (TypeError, ValueError):
                return np.asarray([], dtype=float), np.empty((0, 0), dtype=float)

        if kind == "gr":
            xkey, ykey = "r", "g"
        elif kind == "sq":
            xkey, ykey = "q", "s"
        else:
            xkey, ykey = "x", "cdf"

        if kind in {"gr", "sq"}:
            try:
                payloads = [_dist_entry(b, kind, name) for b in boxes]
                x_common, matrix, _meta = _align_sampled_plot_payloads(
                    payloads,
                    family=kind,
                    xkey=xkey,
                    ykey=ykey,
                )
                return x_common, matrix
            except (KeyError, TypeError, ValueError, RuntimeError):
                return np.asarray([], dtype=float), np.empty((0, 0), dtype=float)

        return np.asarray([], dtype=float), np.empty((0, 0), dtype=float)

    def _matrix_mean_half(ds: dict[str, Any], mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mat = np.asarray(mat, dtype=float)
        if mat.ndim != 2 or mat.shape[0] < 1:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        mu = np.nanmean(mat, axis=0)
        if mat.shape[0] < 2:
            return mu, np.full_like(mu, np.nan, dtype=float)
        crit, _ = _critical_value(int(mat.shape[0]), _alpha_for_dataset(ds))
        sd = np.nanstd(mat, axis=0, ddof=1)
        half = crit * sd / math.sqrt(float(mat.shape[0]))
        return mu, half

    def _cdf_density_summary(ds: dict[str, Any], kind: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xgrid, cdf_mat = _raw_distribution_matrix(ds, kind, name)
        if xgrid.size >= 3 and cdf_mat.ndim == 2 and cdf_mat.shape[1] == xgrid.size:
            dx = np.diff(xgrid)
            if np.all(dx > 0):
                pdf = np.diff(cdf_mat, axis=1) / dx[None, :]
                pdf = np.maximum(pdf, 0.0)
                xmid = 0.5 * (xgrid[:-1] + xgrid[1:])
                mu, half = _matrix_mean_half(ds, pdf)
                return xmid, mu, half

        # Fallback for results that only carry the convergence summary.
        x, mu_cdf, half_cdf = _curve_summary(ds, kind, name)
        if x.size < 3 or mu_cdf.size != x.size:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
        dx = np.diff(x)
        if not np.all(dx > 0):
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
        xmid = 0.5 * (x[:-1] + x[1:])
        mu = np.maximum(np.diff(mu_cdf) / dx, 0.0)
        if half_cdf.size == mu_cdf.size and np.any(np.isfinite(half_cdf)):
            lo = np.maximum(np.diff(mu_cdf - half_cdf) / dx, 0.0)
            hi = np.maximum(np.diff(mu_cdf + half_cdf) / dx, 0.0)
            half = np.maximum(np.abs(mu - lo), np.abs(hi - mu))
        else:
            half = np.full_like(mu, np.nan)
        return xmid, mu, half

    def _coord_pmf_summary(ds: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xgrid, cdf_mat = _raw_distribution_matrix(ds, "coord", name)
        if xgrid.size >= 1 and cdf_mat.ndim == 2 and cdf_mat.shape[1] == xgrid.size:
            pmf = np.zeros_like(cdf_mat, dtype=float)
            pmf[:, 0] = cdf_mat[:, 0]
            if cdf_mat.shape[1] > 1:
                pmf[:, 1:] = cdf_mat[:, 1:] - cdf_mat[:, :-1]
            pmf = np.maximum(pmf, 0.0)
            mu, half = _matrix_mean_half(ds, pmf)
            return xgrid, mu, half

        x, mu_cdf, half_cdf = _curve_summary(ds, "coord", name)
        if x.size == 0 or mu_cdf.size != x.size:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
        mu = np.zeros_like(mu_cdf, dtype=float)
        mu[0] = mu_cdf[0]
        if mu.size > 1:
            mu[1:] = mu_cdf[1:] - mu_cdf[:-1]
        mu = np.maximum(mu, 0.0)
        if half_cdf.size == mu_cdf.size and np.any(np.isfinite(half_cdf)):
            lo_cdf = mu_cdf - half_cdf
            hi_cdf = mu_cdf + half_cdf
            lo = np.zeros_like(mu)
            hi = np.zeros_like(mu)
            lo[0] = lo_cdf[0]
            hi[0] = hi_cdf[0]
            if mu.size > 1:
                lo[1:] = lo_cdf[1:] - hi_cdf[:-1]
                hi[1:] = hi_cdf[1:] - lo_cdf[:-1]
            half = np.maximum(np.abs(mu - lo), np.abs(hi - mu))
        else:
            half = np.full_like(mu, np.nan)
        return x, mu, half

    def _overlay_histogram_page(key: str, *, page_title: str, xlabel: str) -> Any:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        vals_by_dataset: list[tuple[dict[str, Any], np.ndarray]] = []
        all_vals: list[np.ndarray] = []
        unavailable: list[str] = []
        summary_only: list[str] = []
        for ds in datasets:
            vals = _scalar_values(ds, key)
            if not vals.size:
                mu, half = _scalar_summary(ds, key)
                if np.isfinite(mu):
                    summary_only.append(str(ds["label"]))
                    summary = f"{ds['label']}: summary only (no finite per-box values)"
                    ax.axvline(
                        mu,
                        linestyle="--",
                        linewidth=1.0,
                        label=summary,
                    )
                    if np.isfinite(half) and half > 0.0:
                        ax.axvspan(mu - half, mu + half, alpha=0.10, linewidth=0.0)
                else:
                    unavailable.append(str(ds["label"]))
                    ax.plot(
                        [],
                        [],
                        linestyle="None",
                        marker="x",
                        label=f"{ds['label']}: unavailable (no finite values)",
                    )
                continue
            vals_by_dataset.append((ds, vals))
            all_vals.append(vals)
        if all_vals:
            all_concat = np.concatenate(all_vals)
            bins = np.histogram_bin_edges(all_concat, bins="auto") if all_concat.size >= 2 else 10
            for ds, vals in vals_by_dataset:
                _hvals, _hedges, hpatches = ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.8, label=f"{ds['label']} (n={vals.size})")
                line_color = hpatches[0].get_edgecolor() if hpatches else None
                mu, half = _scalar_summary(ds, key)
                if np.isfinite(mu):
                    ax.axvline(mu, linestyle="--", linewidth=1.0, color=line_color)
                if np.isfinite(mu) and np.isfinite(half) and half > 0.0:
                    ax.axvspan(mu - half, mu + half, color=line_color, alpha=0.10, linewidth=0.0)
        elif unavailable and not summary_only:
            ax.text(
                0.5,
                0.5,
                "No dataset contains finite per-box or summary values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        elif summary_only:
            ax.text(
                0.5,
                0.5,
                "Finite per-box values unavailable; summary markers shown where available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("probability density")
        ax.set_title(page_title)
        ax.legend()
        fig.suptitle((str(title) + " | " if title else "") + page_title)
        return fig

    def _ring_fraction_page() -> Any:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        sizes: list[int] = []
        for k in comparison_ring_keys:
            try:
                sizes.append(int(str(k).split("ring_frac_")[-1]))
            except Exception:
                sizes.append(len(sizes) + 1)
        x = np.asarray(sizes, dtype=float)
        n_ds = max(len(datasets), 1)
        width = min(0.80 / float(n_ds), 0.28)
        offsets = (np.arange(n_ds, dtype=float) - 0.5 * float(n_ds - 1)) * width
        for off, ds in zip(offsets, datasets):
            mu = []
            half = []
            for key in comparison_ring_keys:
                m, h = _scalar_summary(ds, key)
                mu.append(m)
                half.append(h)
            y = np.asarray(mu, dtype=float)
            err = np.asarray(half, dtype=float)
            finite = np.isfinite(y)
            if np.any(finite):
                ax.bar(x[finite] + float(off), y[finite], width=width, label=ds["label"], alpha=0.72, edgecolor=OKABE_ITO["black"], linewidth=0.6)
                finite_err = finite & np.isfinite(err)
                if np.any(finite_err):
                    ax.errorbar(x[finite_err] + float(off), y[finite_err], yerr=err[finite_err], fmt="none", ecolor=OKABE_ITO["black"], capsize=2.5, linewidth=1.0)
            else:
                ax.plot(
                    [],
                    [],
                    linestyle="None",
                    marker="x",
                    label=f"{ds['label']}: unavailable",
                )
            missing = [
                str(key)
                for key, available in zip(comparison_ring_keys, finite.tolist())
                if not available
            ]
            if missing:
                ax.text(
                    0.01,
                    0.98 - 0.055 * float(datasets.index(ds)),
                    f"{ds['label']} unavailable: {', '.join(missing)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                )
        ax.set_xlabel("ring size")
        ax.set_ylabel("fraction")
        ax.set_title("Ring statistics")
        ax.legend()
        fig.suptitle((str(title) + " | " if title else "") + "Ring statistics")
        return fig

    def _curve_overlay_page(
        kind: str,
        name: str,
        *,
        spec_key: str,
        xlabel: str,
        ylabel: str,
    ) -> Any:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        n_available = 0
        for ds in datasets:
            declared = name in _dist_names(ds, kind, spec_key)
            if not declared:
                ax.plot(
                    [],
                    [],
                    linestyle="None",
                    marker="x",
                    label=f"{ds['label']}: unavailable (not declared/emitted)",
                )
                continue
            if kind in {"bondlen", "angle", "void"}:
                x, mu, half = _cdf_density_summary(ds, kind, name)
            elif kind == "coord":
                x, mu, half = _coord_pmf_summary(ds, name)
            else:
                x, mu, half = _curve_summary(ds, kind, name)
            if x.size == 0 or mu.size == 0:
                raise ValueError(
                    f"Comparison distribution {name!r} ({kind}) is declared for "
                    f"dataset {ds['label']!r} but has no usable finite payload"
                )
            _require_finite_plot_points(
                x,
                mu,
                context=f"comparison distribution {name!r} dataset {ds['label']!r}",
            )
            _overlay_line_with_band(ax, x, mu, half, label=ds["label"])
            n_available += 1
        if n_available == 0:
            ax.text(
                0.5,
                0.5,
                "Distribution unavailable in every dataset",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(_clean_name(name))
        ax.legend()
        fig.suptitle((str(title) + " | " if title else "") + _clean_name(name))
        return fig

    # Convergence page: same production convergence-trend plot, overlaid by dataset.
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for ds in datasets:
        n_grid, trend_all, n_conv = _compute_trend(ds)
        assessment = _production_convergence_display(ds["prod"], ds["conv"])
        lab = str(ds["label"])
        if n_grid.size == 0 or trend_all.size == 0:
            ax.plot(
                [],
                [],
                label=f"{lab} — {assessment['label']}; retrospective trend unavailable",
            )
            continue
        crossing = (
            f"retrospective threshold n={int(n_conv)}"
            if n_conv is not None
            else "no retrospective threshold crossing"
        )
        lab_plot = f"{lab} — {assessment['label']}; {crossing}"
        ax.plot(n_grid, trend_all, label=lab_plot)
        if n_conv is not None:
            ax.axvline(int(n_conv), linestyle=":", linewidth=1.0, alpha=0.35)
    ax.axhline(1.0, linewidth=1.0, color=OKABE_ITO["black"])
    ax.set_xlabel("number of boxes")
    ax.set_ylabel("max(ratio / tolerance)")
    ax.set_yscale("log")
    ax.set_title("Retrospective production convergence diagnostic")
    ax.legend()
    ax.text(
        0.02,
        0.02,
        "Threshold crossings are diagnostics, not ensemble stopping decisions.",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )
    fig.suptitle(str(title) if title is not None else "vitriflow production comparison")
    pages.append(("convergence_comparison", fig))

    # Density page mirrors plot-production's density histogram, with overlaid datasets.
    pages.append(("density", _overlay_histogram_page("density", page_title="Density across boxes", xlabel=density_ylabel)))

    if comparison_ring_keys:
        pages.append(("rings", _ring_fraction_page()))

    has_comparison_ring_mean = bool(
        datasets
        and any(
            bool(ds["spec"].get("ring_has_mean_size", False))
            or "ring_mean_size" in _declared_scalar_metric_keys(ds)
            for ds in datasets
        )
    )
    if has_comparison_ring_mean:
        pages.append(("ring_mean", _overlay_histogram_page("ring_mean_size", page_title="Mean ring size across boxes", xlabel="mean ring size")))

    for key in comparison_metric_keys:
        pages.append(
            (
                f"scalar_{key}",
                _overlay_histogram_page(
                    str(key),
                    page_title=_clean_name(str(key)),
                    xlabel=_clean_name(str(key)),
                ),
            )
        )

    sweep_by_dataset = [
        _coordination_sweep_matrices(ds["boxes"]) for ds in datasets
    ]
    comparison_sweep_names = sorted(
        set.union(*[set(sweeps) for sweeps in sweep_by_dataset])
        if sweep_by_dataset
        else set()
    )
    for name in comparison_sweep_names:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for ds, sweeps in zip(datasets, sweep_by_dataset):
            if name not in sweeps:
                ax.plot(
                    [],
                    [],
                    linestyle="None",
                    marker="x",
                    label=f"{ds['label']}: unavailable (not emitted)",
                )
                continue
            delta_r, matrix = sweeps[name]
            mu = np.mean(matrix, axis=0)
            if matrix.shape[0] >= 2:
                crit, _crit_method = _critical_value(
                    int(matrix.shape[0]),
                    _alpha_for_dataset(ds),
                )
                half = (
                    float(crit)
                    * np.std(matrix, axis=0, ddof=1)
                    / math.sqrt(float(matrix.shape[0]))
                )
            else:
                half = np.full_like(mu, np.nan, dtype=float)
            _overlay_line_with_band(
                ax,
                delta_r,
                mu,
                half,
                label=str(ds["label"]),
            )
        ax.set_xlabel(f"cutoff shift [{dist_unit}]")
        ax.set_ylabel("coordination-defect fraction")
        ax.set_title(f"Coordination cutoff sensitivity: {_clean_name(name)}")
        ax.legend()
        fig.suptitle(
            (str(title) + " | " if title else "")
            + f"Coordination cutoff sensitivity: {_clean_name(name)}"
        )
        pages.append((f"coordination_sweep_{name}", fig))

    for name in comparison_bond_names:
        pages.append((name, _curve_overlay_page("bondlen", str(name), spec_key="bondlen_names", xlabel=f"r [{dist_unit}]", ylabel="probability density")))
    for name in comparison_angle_names:
        pages.append((name, _curve_overlay_page("angle", str(name), spec_key="angle_names", xlabel="θ [deg]", ylabel="probability density")))
    for name in comparison_coord_names:
        pages.append((name, _curve_overlay_page("coord", str(name), spec_key="coord_names", xlabel="k", ylabel="probability mass")))
    for name in comparison_void_names:
        pages.append((name, _curve_overlay_page("void", str(name), spec_key="void_names", xlabel=f"clearance r [{dist_unit}]", ylabel="probability density")))
    for name in comparison_gr_labels:
        pages.append((name, _curve_overlay_page("gr", str(name), spec_key="gr_labels", xlabel=f"r [{dist_unit}]", ylabel="g(r)")))
    for name in comparison_sq_labels:
        pages.append((name, _curve_overlay_page("sq", str(name), spec_key="sq_labels", xlabel=f"q [1/{dist_unit}]", ylabel="S(q)")))

    if max_pages is not None and int(max_pages) > 0:
        keep = 1 + int(max_pages)
        pages = pages[:keep]

    _save_plot_pages(pages, out_path, dpi=int(dpi), name_cleaner=_slug_name)


def _length_unit_label(units_style: str) -> str:
    # Analysis frames, void points, and exported EXTXYZ are canonicalised by
    # every supported reader before reaching the plotting layer.
    return "Å"


def _cell_corners0(cell: np.ndarray) -> np.ndarray:
    """Cell corners0."""
    H = np.asarray(cell, dtype=float)
    if H.shape != (3, 3):
        raise ValueError("cell must be 3x3")
    frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    return frac @ H


def _wrap_to_cell0(pos: np.ndarray, cell: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Wrap to cell0."""
    P = np.asarray(pos, dtype=float)
    H = np.asarray(cell, dtype=float)
    org = np.asarray(origin, dtype=float)
    invH = np.linalg.inv(H)
    frac = (P - org) @ invH
    frac = frac - np.floor(frac)
    return frac @ H


def _format_lattice(cell: np.ndarray) -> str:
    H = np.asarray(cell, dtype=float)
    if H.shape != (3, 3):
        raise ValueError("cell must be 3x3")
    if not np.all(np.isfinite(H)):
        raise ValueError("cell must be finite")
    flat = H.reshape(-1)
    return " ".join(f"{float(x):.16g}" for x in flat.tolist())


def _write_void_points_extxyz(
    path: Path,
    *,
    frame_cell: np.ndarray,
    frame_origin: np.ndarray,
    step: int,
    points: np.ndarray,
    clearance: np.ndarray,
) -> None:
    """Void points extxyz."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    pts = _wrap_to_cell0(np.asarray(points, dtype=float), np.asarray(frame_cell, dtype=float), np.asarray(frame_origin, dtype=float))
    r = np.asarray(clearance, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or r.ndim != 1 or r.shape[0] != pts.shape[0]:
        raise ValueError("points must be (n,3) and clearance must be (n,)")

    props = "species:S:1:pos:R:3:clearance:R:1"
    pbc_str = "T T T"

    with p.open("w") as f:
        f.write(f"{int(pts.shape[0])}\n")
        f.write(
            f"Lattice=\"{_format_lattice(frame_cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={int(step)}\n"
        )
        for xyz, rr in zip(pts, r.tolist()):
            f.write(f"V {float(xyz[0]):.12f} {float(xyz[1]):.12f} {float(xyz[2]):.12f} {float(rr):.12f}\n")


def _write_atoms_plus_voids_extxyz(
    path: Path,
    *,
    frame,
    type_to_species: Optional[Sequence[str]],
    void_points: np.ndarray,
    void_clearance: np.ndarray,
) -> None:
    """Atoms plus voids."""
    from .analysis.dump import DumpFrame

    if not isinstance(frame, DumpFrame):
        raise TypeError("frame must be a DumpFrame")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # cell consistent visualisation
    atom_pos = _wrap_to_cell0(frame.positions, frame.cell, frame.origin)
    void_pos = _wrap_to_cell0(np.asarray(void_points, dtype=float), frame.cell, frame.origin)

    nat = int(frame.n_atoms)
    nv = int(void_pos.shape[0])

    # species mapping atoms
    def _sym(t: int) -> str:
        if type_to_species is None:
            return "X"
        i = int(t) - 1
        if i < 0 or i >= len(type_to_species):
            return "X"
        return str(type_to_species[i])

    atom_species = [_sym(int(t)) for t in frame.types.tolist()]
    void_species = ["V"] * nv

    # intact assign voids
    tmax = int(np.max(frame.types)) if nat > 0 else 0
    imax = int(np.max(frame.ids)) if nat > 0 else 0

    atom_types = frame.types.astype(int)
    atom_ids = frame.ids.astype(int)

    void_types = np.full((nv,), int(tmax + 1), dtype=int)
    void_ids = np.arange(imax + 1, imax + 1 + nv, dtype=int)

    is_void = np.concatenate([np.zeros((nat,), dtype=int), np.ones((nv,), dtype=int)], axis=0)
    clearance = np.concatenate([
        np.zeros((nat,), dtype=float),
        np.asarray(void_clearance, dtype=float),
    ], axis=0)

    species = atom_species + void_species
    pos = np.vstack([atom_pos, void_pos])
    types = np.concatenate([atom_types, void_types])
    ids = np.concatenate([atom_ids, void_ids])

    props = "species:S:1:pos:R:3:type:I:1:id:I:1:is_void:I:1:clearance:R:1"
    pbc_str = "T T T"

    with p.open("w") as f:
        f.write(f"{int(nat + nv)}\n")
        f.write(
            f"Lattice=\"{_format_lattice(frame.cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={int(frame.timestep)}\n"
        )
        for sym, xyz, t, i, iv, rr in zip(species, pos, types, ids, is_void, clearance):
            f.write(
                f"{sym} {float(xyz[0]):.12f} {float(xyz[1]):.12f} {float(xyz[2]):.12f} {int(t)} {int(i)} {int(iv)} {float(rr):.12f}\n"
            )


def plot_voids_map(
    stage_dir: Path,
    out_path: Path,
    *,
    n_samples: int,
    sampler: str = "sobol",
    seed: int = 0,
    k_nearest: int = 16,
    type_to_species: Optional[Sequence[str]] = None,
    radii_by_species: Optional[Mapping[str, float]] = None,
    default_radius: float = 0.0,
    min_clearance: Optional[float] = None,
    top_n: int = 2000,
    show_atoms: bool = True,
    units_style: str = "",
    title: Optional[str] = None,
    write_void_extxyz: Optional[Path] = None,
    write_combined_extxyz: Optional[Path] = None,
    dpi: int = 600,
) -> None:
    """Voids map."""

    n_samples = _require_positive_integer(n_samples, context="n_samples")
    k_nearest = _require_positive_integer(k_nearest, context="k_nearest")
    top_n = _require_positive_integer(top_n, context="top_n")
    dpi = _require_positive_integer(dpi, context="dpi")
    if not (np.isfinite(float(default_radius)) and float(default_radius) >= 0.0):
        raise ValueError("default_radius must be finite and >= 0")
    if min_clearance is not None and not np.isfinite(float(min_clearance)):
        raise ValueError("min_clearance must be finite when provided")
    for species, radius in dict(radii_by_species or {}).items():
        if not (np.isfinite(float(radius)) and float(radius) >= 0.0):
            raise ValueError(
                f"radius for species {species!r} must be finite and >= 0"
            )
    stage_dir = Path(stage_dir)
    out_path = Path(out_path)

    from .analysis.trajectory import stage_trajectory_path, read_frames_auto
    from .analysis.voids import sample_void_clearance_points

    traj = stage_trajectory_path(stage_dir)
    if traj is None or not Path(traj).exists():
        raise FileNotFoundError(f"No trajectory found in stage directory: {stage_dir}")

    frames = read_frames_auto(
        Path(traj),
        last_n=1,
        units_style=(
            None
            if units_style is None or not str(units_style).strip()
            else str(units_style)
        ),
    )
    if not frames:
        raise ValueError(f"No frames parsed from trajectory: {traj}")
    fr = frames[-1]

    pts, rad = sample_void_clearance_points(
        fr,
        n_samples=int(n_samples),
        sampler=str(sampler),
        seed=int(seed),
        k_nearest=int(k_nearest),
        type_to_species=type_to_species,
        radii_by_species=dict(radii_by_species or {}),
        default_radius=float(default_radius),
    )

    # selection extxyz threshold
    rad = np.asarray(rad, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if rad.ndim != 1 or pts.ndim != 2 or pts.shape != (rad.size, 3):
        raise ValueError("Void sampler returned inconsistent point/clearance arrays")
    if rad.size == 0 or not np.any(np.isfinite(rad)):
        raise ValueError("Void sampler returned no finite clearance values")
    m = np.isfinite(rad)
    if min_clearance is not None and np.isfinite(float(min_clearance)):
        m &= rad >= float(min_clearance)

    idx = np.where(m)[0]
    if idx.size == 0:
        # fallback
        idx_all = np.where(np.isfinite(rad))[0]
        if idx_all.size == 0:
            idx = np.asarray([], dtype=int)
        else:
            order = np.argsort(rad[idx_all])[::-1]
            idx = idx_all[order[: min(int(top_n), int(idx_all.size))]]

    if idx.size > int(top_n):
        order = np.argsort(rad[idx])[::-1]
        idx = idx[order[: int(top_n)]]

    pts_sel = pts[idx] if idx.size else np.zeros((0, 3), dtype=float)
    rad_sel = rad[idx] if idx.size else np.zeros((0,), dtype=float)

    # exports
    if write_void_extxyz is not None:
        _write_void_points_extxyz(
            Path(write_void_extxyz),
            frame_cell=fr.cell,
            frame_origin=fr.origin,
            step=int(fr.timestep),
            points=pts_sel,
            clearance=rad_sel,
        )

    if write_combined_extxyz is not None:
        _write_atoms_plus_voids_extxyz(
            Path(write_combined_extxyz),
            frame=fr,
            type_to_species=type_to_species,
            void_points=pts_sel,
            void_clearance=rad_sel,
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()

    # points origin plotting
    pts0 = _wrap_to_cell0(pts_sel, fr.cell, fr.origin) if pts_sel.size else pts_sel
    atoms0 = _wrap_to_cell0(fr.positions, fr.cell, fr.origin) if show_atoms else None

    corners0 = _cell_corners0(fr.cell)
    mins = np.min(corners0, axis=0)
    maxs = np.max(corners0, axis=0)

    # marker clearance visibility
    if rad_sel.size and np.isfinite(rad_sel).any():
        rmax = float(np.nanmax(rad_sel))
        if rmax <= 0:
            sizes = np.full_like(rad_sel, 8.0, dtype=float)
        else:
            sizes = 8.0 + 120.0 * (rad_sel / rmax) ** 2
    else:
        sizes = np.asarray([], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))

    # histogram
    axh = axes[0, 0]
    rf = rad[np.isfinite(rad)]
    if rf.size:
        axh.hist(rf, bins=50)
    axh.set_xlabel(f"clearance radius {_length_unit_label(units_style)}")
    axh.set_ylabel("count")
    if min_clearance is not None and np.isfinite(float(min_clearance)):
        axh.axvline(float(min_clearance), linestyle="--")

    # projections
    def _proj(ax, a: int, b: int, lab: str) -> None:
        if show_atoms and atoms0 is not None:
            ax.scatter(atoms0[:, a], atoms0[:, b], s=4.0, alpha=0.35, marker=".", label="atoms")
        if pts0.size:
            ax.scatter(pts0[:, a], pts0[:, b], s=sizes, alpha=0.65, marker="o", label="void samples")
        ax.set_xlabel(["x", "y", "z"][a] + f" {_length_unit_label(units_style)}")
        ax.set_ylabel(["x", "y", "z"][b] + f" {_length_unit_label(units_style)}")
        ax.set_title(lab)
        ax.set_xlim(float(mins[a]), float(maxs[a]))
        ax.set_ylim(float(mins[b]), float(maxs[b]))
        ax.set_aspect("equal", adjustable="box")

    _proj(axes[0, 1], 0, 1, "XY")
    _proj(axes[1, 0], 0, 2, "XZ")
    _proj(axes[1, 1], 1, 2, "YZ")

    # legend labels present
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    if title is None:
        title = f"Void clearance map: {stage_dir} (Step={int(fr.timestep)})"
    fig.suptitle(str(title))

    _style_and_save_figure(fig, out_path, dpi=int(dpi))


def plot_elastic_screen(
    screen_dir: Path,
    out_path: Path,
    *,
    title: Optional[str] = None,
    dpi: int = 600,
) -> None:
    """Elastic screen."""

    dpi = _require_positive_integer(dpi, context="dpi")
    screen_dir = Path(screen_dir)
    if (screen_dir / "elastic_screen.json").exists():
        elastic_dir = screen_dir
    elif (screen_dir / "elastic" / "elastic_screen.json").exists():
        elastic_dir = screen_dir / "elastic"
    else:
        raise FileNotFoundError(f"Could not find elastic_screen.json under: {screen_dir}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_publication_style()

    summary = json.loads((elastic_dir / "elastic_screen.json").read_text())
    if not isinstance(summary, Mapping):
        raise ValueError("elastic_screen.json root must be an object")
    if not isinstance(summary.get("units", {}), Mapping):
        raise ValueError("elastic_screen.json units must be an object")
    if str(summary.get("status", "")) != "ok":
        raise ValueError(f"Elastic screen is not usable: status={summary.get('status')!r}")

    stress_csv = elastic_dir / "virial_proxy.csv"
    if not stress_csv.exists():
        # Read-only compatibility for pre-hardening artifacts.  New output is
        # deliberately named virial_proxy.csv to avoid implying a uniquely
        # partitioned local Cauchy stress.
        stress_csv = elastic_dir / "local_stress.csv"
    if not stress_csv.exists():
        raise FileNotFoundError(
            f"Missing virial_proxy.csv (or legacy local_stress.csv) in {elastic_dir}"
        )
    dat = np.genfromtxt(stress_csv, delimiter=",", names=True, dtype=float)
    if dat.size == 0:
        raise ValueError(f"No local-stress rows found in {stress_csv}")
    if dat.ndim == 0:
        dat = np.asarray([dat], dtype=dat.dtype)

    names = set(dat.dtype.names or ())
    if {
        "x_A",
        "y_A",
        "hydrostatic_virial_proxy_GPa",
        "von_mises_virial_proxy_GPa",
    }.issubset(names):
        x = np.asarray(dat["x_A"], dtype=float)
        y = np.asarray(dat["y_A"], dtype=float)
        hydro = np.asarray(dat["hydrostatic_virial_proxy_GPa"], dtype=float)
        vm = np.asarray(dat["von_mises_virial_proxy_GPa"], dtype=float)
        xy_label = "Å"
        hydro_label = "GPa"
        stress_label = "von Mises (GPa)"
    elif {"x_A", "y_A", "hydrostatic_GPa", "von_mises_GPa"}.issubset(names):
        # Compatibility with the brief pre-release canonical-column format.
        x = np.asarray(dat["x_A"], dtype=float)
        y = np.asarray(dat["y_A"], dtype=float)
        hydro = np.asarray(dat["hydrostatic_GPa"], dtype=float)
        vm = np.asarray(dat["von_mises_GPa"], dtype=float)
        xy_label = "Å"
        hydro_label = "GPa"
        stress_label = "von Mises virial proxy (GPa)"
    else:
        # Legacy release files stored native values.  Convert before applying a
        # GPa label; never relabel unconverted bar/atm/etc. as GPa.
        x = np.asarray(dat["x"], dtype=float)
        y = np.asarray(dat["y"], dtype=float)
        hydro = np.asarray(dat["hydrostatic_native"], dtype=float)
        vm = np.asarray(dat["von_mises_native"], dtype=float)
        p_fac = summary.get("units", {}).get("pressure_to_GPa_factor", None)
        l_fac = summary.get("units", {}).get("length_native_to_A_factor", None)
        if p_fac is not None:
            hydro = hydro * float(p_fac)
            vm = vm * float(p_fac)
            hydro_label = "GPa"
            stress_label = "von Mises (GPa)"
        else:
            hydro_label = str(summary.get("units", {}).get("pressure_native", "native"))
            stress_label = f"von Mises ({hydro_label})"
        if l_fac is not None:
            x = x * float(l_fac)
            y = y * float(l_fac)
            xy_label = "Å"
        else:
            xy_label = str(summary.get("units", {}).get("length_native", "native"))

    if not (
        x.ndim == y.ndim == hydro.ndim == vm.ndim == 1
        and x.size > 0
        and x.size == y.size == hydro.size == vm.size
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(hydro))
        and np.all(np.isfinite(vm))
    ):
        raise ValueError(
            "Elastic virial-proxy table must contain matching finite x/y/hydrostatic/von-Mises columns"
        )

    C = np.asarray(summary.get("born_matrix_GPa") if summary.get("born_matrix_GPa") is not None else summary.get("born_matrix_native"), dtype=float)
    if C.shape != (6, 6) or not np.all(np.isfinite(C)):
        raise ValueError("Elastic screen Born matrix must be a finite 6x6 array")
    c_label = "Affine Born curvature (GPa)" if summary.get("born_matrix_GPa") is not None else f"Affine Born curvature ({summary.get('units', {}).get('pressure_native', 'native')})"

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8))

    ax = axes[0, 0]
    im = ax.imshow(C)
    ax.set_xticks(range(6), ["xx", "yy", "zz", "yz", "xz", "xy"])
    ax.set_yticks(range(6), ["xx", "yy", "zz", "yz", "xz", "xy"])
    ax.set_title(c_label)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    sc = ax.scatter(x, y, c=vm, s=14.0, alpha=0.8)
    ax.set_xlabel(f"x ({xy_label})")
    ax.set_ylabel(f"y ({xy_label})")
    ax.set_title("Mean-volume virial-proxy map (XY)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, shrink=0.8, label=stress_label)

    ax = axes[1, 0]
    hf = hydro[np.isfinite(hydro)]
    if hf.size > 0:
        ax.hist(hf, bins=40)
    ax.set_xlabel(f"hydrostatic virial proxy ({hydro_label})")
    ax.set_ylabel("count")
    ax.set_title("Mean-volume-normalized virial proxy")

    ax = axes[1, 1]
    ax.axis("off")
    flags = summary.get("flags", []) or []
    proxy_summary = summary.get("average_volume_normalized_virial_proxy_summary", {}) or {}
    vm_summary = proxy_summary.get("von_mises_proxy_native", {}) or {}
    if not vm_summary:
        # Pre-hardening summary compatibility; plot wording remains explicit
        # about the physical interpretation of the legacy numbers.
        vm_summary = ((summary.get("local_stress_summary", {}) or {}).get("von_mises_native", {}) or {})
    vm_ratio = float(vm_summary.get("max_over_median", float("nan")))
    txt = [
        f"isotropy residual: {float(summary.get('isotropy_residual', float('nan'))):.3g}",
        f"normal-shear coupling: {float(summary.get('normal_shear_coupling_norm', float('nan'))):.3g}",
        f"K_V Born response: {float(summary.get('voigt_born_bulk_response_GPa', summary.get('voigt_born_bulk_response_native', summary.get('voigt_bulk_modulus_GPa', summary.get('voigt_bulk_modulus_native', float('nan')))))):.3g}",
        f"G_V Born response: {float(summary.get('voigt_born_shear_response_GPa', summary.get('voigt_born_shear_response_native', summary.get('voigt_shear_modulus_GPa', summary.get('voigt_shear_modulus_native', float('nan')))))):.3g}",
        "single affine snapshot; not a thermodynamic modulus",
        "per-atom virial / (V/N); no atomic-volume partition",
        f"virial-proxy VM max/median: {vm_ratio:.3g}",
        f"flags: {', '.join(str(f) for f in flags) if flags else 'none'}",
    ]
    affine = summary.get("affine_isotropization", None)
    if isinstance(affine, dict):
        txt.append("")
        txt.append("force_isotropic affine remap")
        txt.append(
            f"target L: {float(affine.get('target_cubic_length', float('nan'))):.4g} "
            f"{affine.get('length_unit', summary.get('units', {}).get('length_native', 'native'))}"
        )
        txt.append(f"strain ||eps||_F: {float(affine.get('frobenius_norm', float('nan'))):.3g}")
    ax.text(0.0, 1.0, "\n".join(txt), va="top", ha="left")

    if title is None:
        # Output roots differ between reference/reproduction passes.  A
        # filesystem path in the default title made otherwise identical
        # elastic results produce different pixels and therefore different
        # PNG bytes.  The plot already carries all scientific context in its
        # panels and companion JSON, so use a path-independent default.
        title = "Elastic screen"
    fig.suptitle(str(title))
    out_path = Path(out_path)
    _style_and_save_figure(fig, out_path, dpi=int(dpi))
