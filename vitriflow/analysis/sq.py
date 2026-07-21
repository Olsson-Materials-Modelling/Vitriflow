from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple, overload

import numpy as np

from ..config import AtomSelector
from .dump import DumpFrame
from .gr import compute_gr
from .common import resolve_selector as _resolve_selector


@dataclass(frozen=True)
class SqResult:
    q: np.ndarray
    s: np.ndarray
    peak_q: float
    peak_height: float
    peak_fwhm: float


SqRepresentationMetadata = dict[str, Any]


def _canonical_window_kind(value: Any) -> Literal["lorch", "hann", "none"]:
    """Return one documented termination window without truthy coercions."""

    if not isinstance(value, str):
        raise ValueError("window must be one of: 'lorch', 'hann', 'none'")
    kind = value.strip().lower()
    if kind not in {"lorch", "hann", "none"}:
        raise ValueError("window must be one of: 'lorch', 'hann', 'none'")
    return kind  # type: ignore[return-value]


def _exact_positive_integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    try:
        numeric = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer >= {minimum}") from exc
    if not math.isfinite(numeric) or numeric != float(integer) or integer < int(minimum):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(integer)


def _finite_positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be finite and > 0")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and > 0") from exc
    if not (math.isfinite(numeric) and numeric > 0.0):
        raise ValueError(f"{field} must be finite and > 0")
    return float(numeric)


def _sq_normalization_fields(
    pair: Optional[Tuple[AtomSelector, AtomSelector]],
    type_to_species: Optional[Sequence[str]],
) -> dict[str, Any]:
    """Describe the exact normalization implemented by :func:`compute_sq`."""

    if pair is None:
        return {
            "normalization": "unweighted_number_number_total",
            "normalization_family": "number_number",
            "normalization_formula": (
                "S_NN(q) = 1 + 4*pi*rho*integral[r^2*(g_NN(r)-1)*sinc(q*r) dr]"
            ),
            "self_term": 1.0,
            "pair": None,
        }

    a_sel, b_sel = pair
    a_types = set(_resolve_selector(a_sel, type_to_species))
    b_types = set(_resolve_selector(b_sel, type_to_species))
    if a_types != b_types and (a_types & b_types):
        raise ValueError(
            f"Overlapping type selections for S(q) pair {pair} are not supported: "
            f"{sorted(a_types & b_types)}"
        )
    same = bool(a_types == b_types)
    return {
        "normalization": "ashcroft_langreth_partial",
        "normalization_family": "ashcroft_langreth",
        "normalization_formula": (
            "S_ab(q) = delta_ab + 4*pi*sqrt(rho_a*rho_b)*"
            "integral[r^2*(g_ab(r)-1)*sinc(q*r) dr]"
        ),
        "self_term": 1.0 if same else 0.0,
        "partial_kind": "self" if same else "cross",
        "pair": [a_sel, b_sel],
        "resolved_type_sets": [sorted(a_types), sorted(b_types)],
    }


def sq_representation_metadata(
    *,
    frames: Sequence[DumpFrame],
    pair: Optional[Tuple[AtomSelector, AtomSelector]],
    type_to_species: Optional[Sequence[str]],
    window: Literal["lorch", "hann", "none"],
    q_max: float,
    nq: int,
    r_max_requested: float,
    r_max_effective: float,
    nbins: int,
    density_prefactors: Sequence[float],
) -> SqRepresentationMetadata:
    """Build machine-readable provenance for the RDF-transform estimator.

    This helper is intentionally independent of JSON/file-writing code so every
    workflow can attach exactly the same representation contract.  The default
    :func:`compute_sq` API remains the historical two-array return; callers that
    persist curves should request its optional metadata return.
    """

    kind = _canonical_window_kind(window)
    q_limit = _finite_positive_float(q_max, field="q_max")
    q_count = _exact_positive_integer(nq, field="nq", minimum=10)
    r_requested = _finite_positive_float(r_max_requested, field="r_max")
    r_effective = _finite_positive_float(r_max_effective, field="effective r_max")
    r_bins = _exact_positive_integer(nbins, field="nbins", minimum=50)
    densities = np.asarray(list(density_prefactors), dtype=float)
    if densities.ndim != 1 or densities.size != len(frames):
        raise ValueError("density_prefactors must contain exactly one value per frame")
    if not np.all(np.isfinite(densities)) or np.any(densities <= 0.0):
        raise ValueError("density_prefactors must be finite and > 0 for every frame")

    scale = max(1.0, abs(r_requested), abs(r_effective))
    clipped = bool(r_effective < r_requested - 64.0 * np.finfo(float).eps * scale)
    return {
        "schema": "vitriflow.sq_representation.v1",
        "observable": "static_structure_factor",
        "estimator": "isotropic_rdf_fourier_transform",
        **_sq_normalization_fields(pair, type_to_species),
        "rdf_normalization": "finite_population_unordered_pair_shell_volume",
        "scattering_weights": "none",
        "scattering_weighted": False,
        "dimensionless": True,
        "q_unit": "angstrom^-1",
        "r_unit": "angstrom",
        "termination_window": str(kind),
        "termination_window_definition": {
            "lorch": "sinc(pi*r/r_support_effective)",
            "hann": "0.5*(1+cos(pi*r/r_support_effective))",
            "none": "1",
        }[kind],
        "radial_transform_kernel": "4*pi*r^2*sinc(q*r)",
        "radial_quadrature": "uniform_bin_midpoint",
        "r_support_requested_A": float(r_requested),
        "r_support_effective_A": float(r_effective),
        "r_support_clipped_to_unique_image_radius": bool(clipped),
        "r_support_policy": "minimum_half_shortest_lattice_translation_across_frames",
        "n_r_bins": int(r_bins),
        "q_min_A^-1": 0.0,
        "q_max_A^-1": float(q_limit),
        "n_q_points": int(q_count),
        "q_zero_semantics": (
            "finite_r_windowed_rdf_transform_extrapolation_not_thermodynamic_compressibility"
        ),
        "frame_aggregation": "equal_frame_mean_after_per_frame_density_transform",
        "density_handling": "per_frame_number_density_prefactor",
        "density_prefactor_unit": "angstrom^-3",
        "density_prefactors_A^-3": [float(x) for x in densities.tolist()],
        "n_frames_requested": int(len(frames)),
        "n_frames_used": int(len(frames)),
    }


def _window_weights(r: np.ndarray, r_max: float, kind: str) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be 1D")
    if not (math.isfinite(float(r_max)) and float(r_max) > 0.0):
        raise ValueError("r_max must be > 0")

    k = _canonical_window_kind(kind)
    if k == "none":
        return np.ones_like(r, dtype=float)

    x = (math.pi * r) / float(r_max)

    if k == "lorch":
        # lorch window sin
        w = np.ones_like(r, dtype=float)
        m = np.abs(x) > 0
        w[m] = np.sin(x[m]) / x[m]
        return w

    if k == "hann":
        # hann window max
        w = 0.5 * (1.0 + np.cos(x))
        # exactly floating error
        w = np.clip(w, 0.0, 1.0)
        return w

    raise ValueError(f"Unknown window kind: {kind}")


def _effective_density_and_base(
    frames: Sequence[DumpFrame],
    pair: Optional[Tuple[AtomSelector, AtomSelector]],
    type_to_species: Optional[Sequence[str]],
) -> tuple[float, float]:
    if not frames:
        raise ValueError("frames must be non-empty")

    rhos: list[float] = []

    if pair is None:
        for fr in frames:
            V = abs(float(np.linalg.det(fr.cell)))
            if fr.n_atoms > 0 and V > 0:
                rhos.append(float(fr.n_atoms) / V)
        if not rhos:
            return float("nan"), 1.0
        return float(np.mean(rhos)), 1.0

    a_sel, b_sel = pair
    A_types = set(_resolve_selector(a_sel, type_to_species))
    B_types = set(_resolve_selector(b_sel, type_to_species))
    if A_types != B_types and (A_types & B_types):
        raise ValueError(
            f"Overlapping type selections for S(q) pair {pair} are not supported: {sorted(A_types & B_types)}"
        )

    rhoA: list[float] = []
    rhoB: list[float] = []
    for fr in frames:
        V = abs(float(np.linalg.det(fr.cell)))
        if V <= 0:
            continue
        t = fr.types
        Na = float(np.sum(np.isin(t, list(A_types))))
        Nb = float(np.sum(np.isin(t, list(B_types))))
        if Na > 0:
            rhoA.append(Na / V)
        if Nb > 0:
            rhoB.append(Nb / V)

    if A_types == B_types:
        if not rhoA:
            return float("nan"), 1.0
        return float(np.mean(rhoA)), 1.0

    if not rhoA or not rhoB:
        return float("nan"), 0.0
    return float(math.sqrt(float(np.mean(rhoA)) * float(np.mean(rhoB)))), 0.0


@overload
def compute_sq(
    frames: Sequence[DumpFrame],
    *,
    q_max: float,
    nq: int,
    r_max: float,
    nbins: int,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    window: Literal["lorch", "hann", "none"] = "lorch",
    return_metadata: Literal[False] = False,
) -> Tuple[np.ndarray, np.ndarray]: ...


@overload
def compute_sq(
    frames: Sequence[DumpFrame],
    *,
    q_max: float,
    nq: int,
    r_max: float,
    nbins: int,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    window: Literal["lorch", "hann", "none"] = "lorch",
    return_metadata: Literal[True],
) -> Tuple[np.ndarray, np.ndarray, SqRepresentationMetadata]: ...


def compute_sq(
    frames: Sequence[DumpFrame],
    *,
    q_max: float,
    nq: int,
    r_max: float,
    nbins: int,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    window: Literal["lorch", "hann", "none"] = "lorch",
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, SqRepresentationMetadata]:
    """Estimate the equal-frame mean static structure factor.

    Each frame is transformed with its own number density before averaging.
    Transforming an already averaged ``g(r)`` with the mean density is not
    equivalent when an NPT trajectory changes volume and can bias ``S(q)``.
    """
    if not frames:
        raise ValueError("compute_sq requires at least one frame")
    if not isinstance(return_metadata, (bool, np.bool_)):
        raise ValueError("return_metadata must be boolean")
    q_limit = _finite_positive_float(q_max, field="q_max")
    q_count = _exact_positive_integer(nq, field="nq", minimum=10)
    r_limit = _finite_positive_float(r_max, field="r_max")
    r_bins = _exact_positive_integer(nbins, field="nbins", minimum=50)
    window_kind = _canonical_window_kind(window)

    r, g, _l = compute_gr(
        frames,
        r_max=r_limit,
        nbins=r_bins,
        pair=pair,
        type_to_species=type_to_species,
    )
    if r.size < 2:
        raise ValueError("g(r) grid too small")

    dr = float(r[1] - r[0])
    if not (math.isfinite(dr) and dr > 0.0):
        raise ValueError("invalid g(r) grid spacing")

    # effective cutoff normalization
    r_max_eff = float(r[-1] + 0.5 * dr)

    q = np.linspace(0.0, q_limit, q_count, dtype=float)
    w = _window_weights(r, r_max_eff, window_kind)

    # vectorised sinc limit
    qr = np.outer(q, r)
    sinc = np.ones_like(qr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.abs(qr) > 0.0
        sinc[m] = np.sin(qr[m]) / qr[m]

    def _transform(g_values: np.ndarray, rho_eff: float, base: float) -> np.ndarray:
        h = np.asarray(g_values, dtype=float) - 1.0
        A = (r**2) * h * w * dr
        return np.asarray(
            float(base) + 4.0 * math.pi * float(rho_eff) * (sinc @ A),
            dtype=float,
        )

    frame_curves: list[np.ndarray] = []
    density_prefactors: list[float] = []
    if len(frames) == 1:
        rho_eff, base = _effective_density_and_base(frames, pair, type_to_species)
        if not (math.isfinite(rho_eff) and rho_eff > 0.0):
            raise ValueError("S(q) is undefined for frame 0: requested atom population is absent")
        if not np.all(np.isfinite(g)):
            raise ValueError(
                "S(q) is undefined for frame 0: requested RDF population has fewer than two self pairs"
            )
        curve = _transform(g, rho_eff, base)
        if not np.all(np.isfinite(curve)):
            raise ValueError("S(q) transform produced non-finite values for frame 0")
        frame_curves.append(curve)
        density_prefactors.append(float(rho_eff))
    else:
        # ``r_max_eff`` is the minimum geometrically valid cutoff across the
        # full sequence.  Reusing it makes every per-frame g(r) grid identical
        # while retaining the correct density for that frame's transform.
        for frame_index, frame in enumerate(frames):
            r_frame, g_frame, _spacing = compute_gr(
                [frame],
                r_max=r_max_eff,
                nbins=r_bins,
                pair=pair,
                type_to_species=type_to_species,
            )
            if r_frame.shape != r.shape or not np.allclose(
                r_frame, r, rtol=0.0, atol=64.0 * np.finfo(float).eps * max(1.0, r_max_eff)
            ):
                raise RuntimeError("per-frame g(r) grids are inconsistent")
            rho_eff, base = _effective_density_and_base(
                [frame], pair, type_to_species
            )
            if not (math.isfinite(rho_eff) and rho_eff > 0.0):
                raise ValueError(
                    f"S(q) is undefined for frame {frame_index}: requested atom population is absent"
                )
            if not np.all(np.isfinite(g_frame)):
                raise ValueError(
                    f"S(q) is undefined for frame {frame_index}: requested RDF population "
                    "has fewer than two self pairs"
                )
            curve = _transform(g_frame, rho_eff, base)
            if not np.all(np.isfinite(curve)):
                raise ValueError(
                    f"S(q) transform produced non-finite values for frame {frame_index}"
                )
            frame_curves.append(curve)
            density_prefactors.append(float(rho_eff))

    if len(frame_curves) != len(frames):
        raise RuntimeError("S(q) internal error: not every requested frame contributed")
    stack = np.vstack(frame_curves)
    if not np.all(np.isfinite(stack)):
        raise RuntimeError("S(q) internal error: non-finite frame curve escaped validation")
    mean = np.mean(stack, axis=0, dtype=float)
    metadata = sq_representation_metadata(
        frames=frames,
        pair=pair,
        type_to_species=type_to_species,
        window=window_kind,
        q_max=q_limit,
        nq=q_count,
        r_max_requested=r_limit,
        r_max_effective=r_max_eff,
        nbins=r_bins,
        density_prefactors=density_prefactors,
    )
    if bool(return_metadata):
        return q, mean, metadata
    return q, mean


def first_peak_features(
    q: np.ndarray,
    s: np.ndarray,
    *,
    smooth: int = 7,
    q_min: float = 0.5,
    q_max: Optional[float] = 3.0,
) -> tuple[float, float, float]:
    """First peak features."""
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)
    if q.ndim != 1 or s.ndim != 1 or q.size != s.size:
        raise ValueError("q and s must be 1D arrays of equal length")
    if q.size < 10:
        return float("nan"), float("nan"), float("nan")

    # enforce monotone grid
    if not np.all(np.isfinite(q)) or not np.all(np.diff(q) > 0.0) or float(q[0]) < 0.0:
        raise ValueError("q grid must be finite, nonnegative, and strictly increasing")

    if isinstance(q_min, (bool, np.bool_)):
        raise ValueError("q_min must be finite and >= 0")
    try:
        q_min_value = float(q_min)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("q_min must be finite and >= 0") from exc
    if not (math.isfinite(q_min_value) and q_min_value >= 0.0):
        raise ValueError("q_min must be finite and >= 0")
    if q_max is None:
        q_max_value = float(q[-1])
    else:
        if isinstance(q_max, (bool, np.bool_)):
            raise ValueError("q_max must be finite and greater than q_min")
        try:
            q_max_value = float(q_max)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("q_max must be finite and greater than q_min") from exc
        if not math.isfinite(q_max_value):
            raise ValueError("q_max must be finite and greater than q_min")
    if not q_max_value > q_min_value:
        raise ValueError("q_max must be greater than q_min")

    w = _exact_positive_integer(smooth, field="smooth", minimum=1)
    # ``np.convolve(..., mode="same")`` has the length of the longer
    # operand.  Keep the smoothing kernel no longer than the physical q-grid
    # so peak indices and q values cannot become misaligned for an oversized
    # user-supplied window.
    w = min(w, int(q.size))
    if w % 2 == 0:
        w = max(1, w - 1)
    if w > 1:
        ker = np.ones(w, dtype=float)
        finite = np.isfinite(s)
        numerator = np.convolve(np.where(finite, s, 0.0), ker, mode="same")
        denominator = np.convolve(finite.astype(float), ker, mode="same")
        s_s = np.full_like(s, np.nan, dtype=float)
        np.divide(numerator, denominator, out=s_s, where=denominator > 0.0)
    else:
        s_s = np.array(s, dtype=float)

    q0 = max(q_min_value, float(q[0]))
    q1 = min(q_max_value, float(q[-1]))
    if not (q1 > q0):
        return float("nan"), float("nan"), float("nan")

    m = (q >= q0) & (q <= q1) & np.isfinite(s_s)
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")

    idxs = np.where(m)[0]
    idx_peak = int(idxs[int(np.argmax(s_s[m]))])

    q_peak = float(q[idx_peak])
    h = float(s_s[idx_peak])

    # baseline peak window
    m_pre = (q >= q0) & (q <= q_peak) & np.isfinite(s_s)
    baseline = float(np.min(s_s[m_pre])) if np.any(m_pre) else float(s_s[idx_peak])
    prominence = h - baseline
    scale = max(1.0, abs(h), abs(baseline))
    if not (math.isfinite(prominence) and prominence > 32.0 * np.finfo(float).eps * scale):
        return float("nan"), float("nan"), float("nan")
    half = baseline + 0.5 * (h - baseline)

    # crossing
    left = None
    for i in range(idx_peak, 0, -1):
        if not (math.isfinite(float(s_s[i - 1])) and math.isfinite(float(s_s[i]))):
            continue
        if s_s[i - 1] <= half <= s_s[i] or s_s[i] <= half <= s_s[i - 1]:
            x0, x1 = float(q[i - 1]), float(q[i])
            y0, y1 = float(s_s[i - 1]), float(s_s[i])
            if y1 == y0:
                left = x0
            else:
                left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    # crossing
    right = None
    for i in range(idx_peak, q.size - 1):
        if not (math.isfinite(float(s_s[i])) and math.isfinite(float(s_s[i + 1]))):
            continue
        if s_s[i] >= half >= s_s[i + 1] or s_s[i] <= half <= s_s[i + 1]:
            x0, x1 = float(q[i]), float(q[i + 1])
            y0, y1 = float(s_s[i]), float(s_s[i + 1])
            if y1 == y0:
                right = x1
            else:
                right = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            break

    if left is None or right is None:
        fwhm = float("nan")
    else:
        fwhm = float(max(0.0, right - left))

    return q_peak, h, fwhm


def compute_first_peak_sq(
    frames: Sequence[DumpFrame],
    *,
    q_max: float,
    nq: int,
    r_max: float,
    nbins: int,
    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    window: Literal["lorch", "hann", "none"] = "lorch",
    smooth: int = 7,
    peak_search: Tuple[float, float] = (0.5, 3.0),
) -> SqResult:
    q_limit = _finite_positive_float(q_max, field="q_max")
    smooth_count = _exact_positive_integer(smooth, field="smooth", minimum=1)
    if (
        not isinstance(peak_search, Sequence)
        or isinstance(peak_search, (str, bytes, bytearray))
        or len(peak_search) != 2
        or any(isinstance(x, (bool, np.bool_)) for x in peak_search)
    ):
        raise ValueError("peak_search must contain two finite bounds")
    try:
        q0, q1 = float(peak_search[0]), float(peak_search[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("peak_search must contain two finite bounds") from exc
    if not (
        math.isfinite(q0)
        and math.isfinite(q1)
        and 0.0 <= q0 < q1 <= q_limit
    ):
        raise ValueError("peak_search must satisfy 0 <= min < max <= q_max")
    q, s = compute_sq(
        frames,
        q_max=q_limit,
        nq=nq,
        r_max=float(r_max),
        nbins=nbins,
        pair=pair,
        type_to_species=type_to_species,
        window=window,
    )
    q_peak, h, fwhm = first_peak_features(
        q, s, smooth=smooth_count, q_min=q0, q_max=q1
    )
    return SqResult(q=q, s=s, peak_q=q_peak, peak_height=h, peak_fwhm=fwhm)
