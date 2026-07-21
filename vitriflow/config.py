from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union, List, Sequence

import yaml
from pydantic import AliasChoices, BaseModel as PydanticBaseModel, ConfigDict, Field, field_validator, model_validator


# ``+`` is part of the published GAP-20U+gr asset basename and is an ordinary
# non-separator character in both a LAMMPS input token and the supported host
# filesystems.  Keep this deliberately narrow: whitespace, control characters,
# quoting/metacharacters and path separators remain outside the allow-list.
_SAFE_LAMMPS_LOCALIZED_FILENAME = re.compile(r"^[A-Za-z0-9_.+-]+$")
_LAMMPS_RESERVED_EXTRA_ARGS: frozenset[str] = frozenset(
    {
        "-in",
        "-i",
        "-log",
        "-l",
        "-screen",
        "-sc",
        "-partition",
        "-p",
        "-plog",
        "-pscreen",
        "-restart2data",
        "-r2data",
        "-restart2dump",
        "-r2dump",
        "-restart2info",
        "-r2info",
        "-skiprun",
        "-sr",
        "-help",
        "-h",
    }
)


def validated_lammps_extra_args(value: Any) -> list[str]:
    """Validate runtime LAMMPS arguments, including post-model mutations."""

    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("lammps.extra_args must be a list of argument tokens")
    values: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                f"lammps.extra_args[{index}] must be a non-empty string token"
            )
        if raw != raw.strip():
            raise ValueError(
                f"lammps.extra_args[{index}] must not contain leading or trailing whitespace"
            )
        token = raw
        option = token.lower().split("=", 1)[0]
        if option in _LAMMPS_RESERVED_EXTRA_ARGS:
            raise ValueError(
                "lammps.extra_args must not override runner-owned input, "
                f"output, or execution-control switch {token!r}"
            )
        values.append(token)
    return values


def validated_lammps_command(value: Any) -> Union[str, list[str]]:
    """Validate the executable and any optional command-list tail."""

    if isinstance(value, list):
        if not value:
            raise ValueError("lammps_cmd list must be non-empty")
        executable = value[0]
        if (
            not isinstance(executable, str)
            or not executable.strip()
            or executable != executable.strip()
        ):
            raise ValueError(
                "lammps_cmd[0] must be an exact non-empty executable token"
            )
        try:
            tail = validated_lammps_extra_args(value[1:])
        except ValueError as exc:
            raise ValueError(
                "lammps_cmd tokens after the executable must not override "
                "runner-owned execution controls"
            ) from exc
        return [executable, *tail]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lammps_cmd must be a non-empty string")
    if value != value.strip():
        raise ValueError("lammps_cmd must not contain leading or trailing whitespace")
    return value


def validated_lammps_localized_filename(value: Any, *, field_name: str) -> str:
    """Return a portable basename safe for one shared LAMMPS work directory."""

    raw_name = str(value)
    name = raw_name.strip()
    if (
        not name
        or raw_name != name
        or name in {".", ".."}
        or Path(name).name != name
        or _SAFE_LAMMPS_LOCALIZED_FILENAME.fullmatch(name) is None
    ):
        raise ValueError(
            f"{field_name} must be one path-safe basename made from letters, "
            "digits, '_', '-', '+', or '.' (not '.' or '..')"
        )
    return name


class BaseModel(PydanticBaseModel):
    """Strict configuration base.

    A misspelled scientific or execution option must never be silently ignored
    and replaced by a default.  Free-form user metadata remains possible only
    in fields explicitly typed as mappings.
    """

    model_config = ConfigDict(extra="forbid")


# structural metrics melt
AtomSelector = Union[int, str]


_SAFE_CP2K_LOCAL_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_cp2k_extra_args(value: Any) -> list[str]:
    """Validate optional CP2K arguments without ceding runner ownership.

    ``Cp2kRunner`` always supplies the authoritative input and output files.
    CP2K accepts later command-line switches that can replace those files or
    select a non-production utility mode while still returning success.  Keep
    ordinary numerical/runtime options available, but reject every switch that
    changes input/output or whether the calculation is actually executed.

    This function is deliberately shared by configuration validation and the
    runner call boundary: Pydantic models are mutable and ``model_construct``
    can bypass field validators, so configuration-time validation alone is not
    a sufficient execution guarantee.
    """

    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("cp2k.extra_args must be a list of argument tokens")

    reserved_exact = frozenset(
        {
            "-i",
            "--input-file",
            "-o",
            "--output-file",
            "-b",
            "--batch",
            "-c",
            "--check",
            "--check-all",
            "--check-run",
            "-d",
            "--dry-run",
            "-h",
            "--help",
            "--html-manual",
            "--keep-alive",
            "--memory",
            "--mpi-mapping",
            "-r",
            "--run",
            "-s",
            "--shell",
            "--shell-posix",
            "-v",
            "--version",
            "--xml",
        }
    )
    reserved_assignment_prefixes = (
        "--input-file=",
        "--output-file=",
        "--batch=",
        "--check=",
        "--check-all=",
        "--check-run=",
        "--dry-run=",
        "--help=",
        "--html-manual=",
        "--keep-alive=",
        "--memory=",
        "--mpi-mapping=",
        "--run=",
        "--shell=",
        "--version=",
        "--xml=",
    )
    reserved_short_prefixes = ("-i", "-o", "-b", "-c", "-d", "-h", "-r", "-s", "-v")

    values: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise ValueError(
                f"cp2k.extra_args[{index}] must be a non-empty string token "
                "without leading or trailing whitespace"
            )
        token = raw
        lowered = token.lower()
        if (
            lowered in reserved_exact
            or any(lowered.startswith(prefix) for prefix in reserved_assignment_prefixes)
            or any(
                lowered.startswith(prefix) and lowered != prefix
                for prefix in reserved_short_prefixes
            )
        ):
            raise ValueError(
                "cp2k.extra_args must not override runner-owned input/output "
                f"or select a non-production execution mode: {token!r}"
            )
        values.append(token)
    return values


def validate_cp2k_command(value: Any) -> Union[str, List[str]]:
    """Validate the CP2K executable payload, including mutable-model reuse."""

    if isinstance(value, list):
        if not value:
            raise ValueError("cp2k_cmd list must be non-empty")
        try:
            return validate_cp2k_extra_args(value)
        except ValueError as exc:
            raise ValueError(
                "cp2k_cmd tokens must be exact non-empty command tokens and "
                "must not preselect CP2K input/output or utility modes"
            ) from exc
    raw = str(value)
    if not raw or raw != raw.strip():
        raise ValueError(
            "cp2k_cmd must be a non-empty string without leading or trailing whitespace"
        )
    return raw


def _validated_atom_selector(value: Any, *, field_name: str) -> AtomSelector:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a species name or an integer atom type, not boolean")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{field_name} integer atom types must be >= 1")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} species names must be non-empty")
        return stripped
    raise ValueError(f"{field_name} must be a species name or an integer atom type")


def _validated_selector_tuple(value: Any, *, size: int, field_name: str) -> tuple[AtomSelector, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must contain exactly {size} atom selectors")
    items = list(value)
    if len(items) != size:
        raise ValueError(f"{field_name} must contain exactly {size} atom selectors")
    return tuple(
        _validated_atom_selector(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(items)
    )


class LammpsConfig(BaseModel):
    """Lammps config."""
    # executable tokenized command
    # naturally yaml lammps
    lammps_cmd: Union[str, List[str]] = "lmp"
    mpi_cmd: Optional[str] = None
    nprocs: int = 1
    extra_args: list[str] = Field(default_factory=list)

    # timeout lammps invocation
    # terminate timeout expires
    timeout_sec: Optional[float] = None
    # seconds sigterm sigkill
    kill_grace_sec: float = 5.0

    # These switches control the input stream, output namespaces, execution
    # mode, or utility mode owned by ``LammpsRunner``.  Allowing them through
    # ``extra_args`` would place a second occurrence *after* the runner's
    # canonical ``-in/-log/-screen`` arguments.  LAMMPS can then run a
    # different script, suppress the current log, skip the run, or enter a
    # restart-conversion mode while returning success.  Accelerator and
    # package switches remain available through ``extra_args``.
    @field_validator("lammps_cmd")
    @classmethod
    def _lammps_cmd_valid(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        return validated_lammps_command(v)

    @field_validator("extra_args", mode="before")
    @classmethod
    def _extra_args_cannot_override_runner_io(cls, v: Any) -> list[str]:
        return validated_lammps_extra_args(v)

    @field_validator("mpi_cmd")
    @classmethod
    def _mpi_cmd_strip(cls, v: Optional[str]) -> Optional[str]:
        # failures accidental yaml
        if v is None:
            return None
        s = str(v).strip()
        return s if s != "" else None

    @field_validator("nprocs", mode="before")
    @classmethod
    def _nprocs_positive(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("nprocs must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("nprocs must be an integer >= 1")
        return n

    @field_validator("timeout_sec", mode="before")
    @classmethod
    def _timeout_valid(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("timeout_sec must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("timeout_sec must be finite and > 0")
        return x

    @field_validator("kill_grace_sec", mode="before")
    @classmethod
    def _kill_grace_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("kill_grace_sec must be numeric")
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("kill_grace_sec must be finite and >= 0")
        return x


class Cp2kKindConfig(BaseModel):
    """Cp2k kind config."""

    basis_set: str
    potential: str

    @field_validator("basis_set", "potential")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        s = str(v).strip()
        if s == "":
            raise ValueError("value must be a non-empty string")
        return s


class Cp2kConfig(BaseModel):
    """Cp2k config."""

    # execution
    # execution
    # packages different engines
    # packages different engines
    # packages different engines
    # mpi versions open
    # abis installed environment
    # prefix prepended invocation
    # cp2k env
    # cp2k env
    # e g conda
    # yaml
    # yaml
    # cp2k
    # prefix cp2k capture
    # mpi cmd mpiexec
    # cp2k cmd psmp
    # nprocs
    # exec prefix
    exec_prefix: List[str] = Field(default_factory=list)
    # backwards ergonomic naturally
    cp2k_cmd: Union[str, List[str]] = Field(
        default="cp2k.psmp",
        validation_alias=AliasChoices("cp2k_cmd", "exec", "executable", "command"),
    )
    mpi_cmd: Optional[str] = None
    nprocs: int = 1
    extra_args: list[str] = Field(default_factory=list)

    timeout_sec: Optional[float] = None
    kill_grace_sec: float = 5.0

    # directory basis pseudopotentials
    # detect cp2 executable
    data_dir: Optional[str] = None

    # mp thread count
    # mpi nprocs oversubscription
    omp_num_threads: Optional[int] = None

    # dft
    basis_set_file_name: str = "BASIS_MOLOPT"
    potential_file_name: str = "GTH_POTENTIALS"
    xc_functional: Literal["PBE"] = "PBE"

    cutoff_Ry: float = 400.0
    rel_cutoff_Ry: float = 60.0
    ngrids: int = 4

    eps_scf: float = 1.0e-6
    max_scf: int = 50
    scf_guess: Literal["ATOMIC", "RESTART"] = "RESTART"

    # ot robust insulating
    use_ot: bool = True
    ot_minimizer: Literal["DIIS", "CG"] = "DIIS"
    ot_preconditioner: Literal["FULL_SINGLE_INVERSE", "FULL_ALL"] = "FULL_SINGLE_INVERSE"

    # temperature ramp handling
    # requests quench approximates
    # concatenating piecewise constant
    ramp_max_deltaT_K: float = 100.0
    ramp_max_segments: int = 20

    # element keys symbols
    kind_settings: Dict[str, Cp2kKindConfig] = Field(default_factory=dict)

    @field_validator("cp2k_cmd")
    @classmethod
    def _cp2k_cmd_valid(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        return validate_cp2k_command(v)

    @field_validator("extra_args", mode="before")
    @classmethod
    def _cp2k_extra_args_cannot_override_runner(cls, v: Any) -> list[str]:
        return validate_cp2k_extra_args(v)

    @field_validator("data_dir")
    @classmethod
    def _data_dir_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        raw = str(v)
        if not raw or raw != raw.strip():
            raise ValueError(
                "cp2k.data_dir must be omitted/null or an exact non-empty directory path"
            )
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError(
                "cp2k.data_dir must be absolute after YAML path resolution"
            )
        if not path.is_dir():
            raise ValueError(f"cp2k.data_dir is not a directory: {path}")
        return str(path.resolve(strict=False))

    @field_validator("basis_set_file_name", "potential_file_name")
    @classmethod
    def _cp2k_data_filename_valid(cls, v: str) -> str:
        raw = str(v)
        if not raw or raw != raw.strip():
            raise ValueError(
                "CP2K data filename must be non-empty without leading or trailing whitespace"
            )
        path = Path(raw).expanduser()
        qualified = path.is_absolute() or "/" in raw or "\\" in raw
        if qualified:
            if not path.is_absolute():
                raise ValueError(
                    "path-qualified CP2K data filenames must be absolute after YAML path resolution"
                )
            if not path.is_file():
                raise ValueError(f"CP2K data file does not exist: {path}")
            return str(path.resolve(strict=False))
        if (
            raw in {".", ".."}
            or path.name != raw
            or _SAFE_CP2K_LOCAL_FILENAME_RE.fullmatch(raw) is None
        ):
            raise ValueError(
                "bare CP2K data filenames must be path-safe basenames made "
                "from letters, digits, '_', '-' or '.'"
            )
        return raw

    @field_validator("mpi_cmd")
    @classmethod
    def _mpi_cmd_strip_cp2k(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v)
        if not s or s != s.strip():
            raise ValueError(
                "cp2k.mpi_cmd must be omitted/null or an exact non-empty command token"
            )
        return s

    @field_validator("exec_prefix")
    @classmethod
    def _exec_prefix_valid(cls, v: Sequence[str]) -> List[str]:
        values = list(v)
        if any(
            not isinstance(token, str)
            or not token
            or token != token.strip()
            for token in values
        ):
            raise ValueError(
                "cp2k.exec_prefix must contain exact non-empty tokens without "
                "leading or trailing whitespace"
            )
        return values

    @field_validator("nprocs", mode="before")
    @classmethod
    def _nprocs_positive_cp2k(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("nprocs must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("nprocs must be an integer >= 1")
        return n

    @field_validator("omp_num_threads", mode="before")
    @classmethod
    def _omp_threads_valid(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("omp_num_threads must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("omp_num_threads must be an integer >= 1")
        return n

    @field_validator("timeout_sec", mode="before")
    @classmethod
    def _cp2k_timeout_valid(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("timeout_sec must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("timeout_sec must be finite and > 0")
        return x

    @field_validator("kill_grace_sec", mode="before")
    @classmethod
    def _cp2k_kill_grace_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("kill_grace_sec must be numeric")
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("kill_grace_sec must be finite and >= 0")
        return x

    @field_validator("cutoff_Ry", "rel_cutoff_Ry", mode="before")
    @classmethod
    def _cutoffs_pos(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("cutoffs must be numeric")
        x = float(v)
        if not (math.isfinite(x) and x > 0.0):
            raise ValueError("cutoffs must be finite and > 0")
        return x

    @field_validator("ngrids", mode="before")
    @classmethod
    def _ngrids_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("ngrids must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("ngrids must be an integer >= 1")
        return n

    @field_validator("eps_scf", mode="before")
    @classmethod
    def _eps_scf_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("eps_scf must be numeric")
        x = float(v)
        if not (math.isfinite(x) and x > 0.0):
            raise ValueError("eps_scf must be finite and > 0")
        return x

    @field_validator("max_scf", mode="before")
    @classmethod
    def _max_scf_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("max_scf must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("max_scf must be an integer >= 1")
        return n

    @field_validator("ramp_max_deltaT_K", mode="before")
    @classmethod
    def _ramp_deltaT_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("ramp_max_deltaT_K must be numeric")
        x = float(v)
        if not (math.isfinite(x) and x > 0.0):
            raise ValueError("ramp_max_deltaT_K must be finite and > 0")
        return x

    @field_validator("ramp_max_segments", mode="before")
    @classmethod
    def _ramp_segments_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("ramp_max_segments must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("ramp_max_segments must be an integer >= 1")
        return n


class CoreRepulsionConfig(BaseModel):
    """Core repulsion config."""

    enabled: bool = False
    # repulsive shifted purely
    # energy continuous minimum
    # robust relies always
    # cut sub style
    style: Literal["zbl", "lj_repulsive"] = "zbl"

    @field_validator("style", mode="before")
    @classmethod
    def _normalize_style(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s in {"lj", "ljrepulsive", "lj-repulsive"}:
            return "lj_repulsive"
        if s in {"zbl"}:
            return "zbl"
        # pydantic allowed literals
        return str(v)

    # zbl parameters recommended
    # nearest neighbor distance
    r_out_factor: float = 0.5
    r_out_min: float = 0.6
    r_out_max: float = 2.0
    # inner factor outer
    r_in_factor: float = 0.8
    # calibration fails factor
    grow_factor: float = 1.25

    # performance buckingham models
    # enabled calibrates analytic
    # preflight pair interaction
    # table cutoff pair
    # table contains buckingham
    # table contains buckingham
    # interaction
    # table contains buckingham
    # table contains buckingham
    # coulombic cutoff pair
    # table contains buckingham
    # table contains buckingham
    # coulomb contribution parameter
    # reciprocal retained pair
    # compatibility pairwise computation
    # table interacting pair
    # preserving range electrostatics
    # currently style zbl
    # currently style zbl
    # tabulate
    tabulate: bool = False
    table_points: int = 12000
    table_filename: str = "buckingham_core.table"
    table_r_min: float = 0.1
    # parameter distance tabulating
    # coulombic contribution omitted
    # kspace modify present
    # selected successful analytic
    # preflight generating table
    table_gewald: Optional[float] = None
    # preflight table verification
    # production table lammps
    # guard
    # lammps table consistency
    # warn
    table_points_max: int = 96000
    table_verify_points: int = 50001
    table_verify_rel_tol: float = 5.0e-5
    table_verify_abs_tol_frac: float = 1.0e-7
    # Preserve the public field name: true rejects every unaudited or
    # non-inflection table warning. A force/secant warning proven to arise
    # solely at an analytic energy inflection is recorded as advisory.
    table_require_warning_free: bool = True

    # repulsive increase repulsion
    # factor failed
    strength_grow_factor: float = 2.0

    # calibration parameters
    max_attempts: int = 6
    test_equil_steps: int = 0
    test_run_steps: int = 2000
    # integration stability calibration
    # timesteps lammps largest
    # stability preflight generate
    dt_candidates: Optional[List[float]] = None

    # maximum step displacement
    limit_max_disp: float = 0.02  # step

    # length volume stability
    ramp_steps: int = 5000

    # additional displacement switching
    limit_hold_steps: int = 2000

    # damping stability ramp
    langevin_damp: Optional[float] = None


    # target repulsive expressed
    # preflight active lammps
    u_target_kT: float = 50.0

    @field_validator("table_filename")
    @classmethod
    def _table_filename_is_local_basename(cls, v: Any) -> str:
        return validated_lammps_localized_filename(
            v, field_name="core_repulsion.table_filename"
        )

    @field_validator(
        "r_out_factor",
        "r_out_min",
        "r_out_max",
        "r_in_factor",
        "grow_factor",
        "strength_grow_factor",
        "u_target_kT",
        "table_r_min",
        "table_gewald",
        "table_verify_rel_tol",
        "table_verify_abs_tol_frac",
        mode="before",
    )
    @classmethod
    def _pos_float(cls, v: Any) -> Optional[float]:
        if v is None:
            return v
        if isinstance(v, bool):
            raise ValueError("value must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("value must be finite and > 0")
        return x

    @field_validator(
        "max_attempts",
        "test_run_steps",
        "ramp_steps",
        "limit_hold_steps",
        "table_points",
        "table_points_max",
        "table_verify_points",
        mode="before",
    )
    @classmethod
    def _pos_int(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("value must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("value must be an integer >= 1")
        return n

    @field_validator("test_equil_steps", mode="before")
    @classmethod
    def _nonnegative_int(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("test_equil_steps must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("test_equil_steps must be an integer >= 0")
        return n

    @field_validator("dt_candidates", mode="before")
    @classmethod
    def _dt_candidates_valid(cls, v: Any) -> Optional[list[float]]:
        if v is None:
            return None
        if isinstance(v, (str, bytes, Mapping)) or not isinstance(v, Sequence):
            raise ValueError("dt_candidates must be a non-empty sequence of finite values > 0")
        values: list[float] = []
        for raw in v:
            if isinstance(raw, bool):
                raise ValueError("dt_candidates must contain numeric values")
            x = float(raw)
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError("dt_candidates must contain only finite values > 0")
            values.append(x)
        if not values:
            raise ValueError("dt_candidates must not be empty when provided")
        return values

    @field_validator("limit_max_disp", "langevin_damp", mode="before")
    @classmethod
    def _optional_positive_finite(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("value must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("value must be finite and > 0")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "CoreRepulsionConfig":
        if self.r_out_max < self.r_out_min:
            raise ValueError("r_out_max must be >= r_out_min")
        if self.r_in_factor >= 1.0:
            raise ValueError("r_in_factor must be < 1 (inner cutoff must be smaller than outer)")
        # Enabled ZBL autocore is always realized as a replacement table: an
        # analytic Buckingham+ZBL overlay remains unbounded below.  The legacy
        # ``tabulate`` flag is therefore not an authority for whether table
        # controls are consumed and must not be allowed to bypass validation.
        uses_zbl_table = bool(self.enabled) and str(self.style).strip().lower() == "zbl"
        if bool(self.tabulate) or uses_zbl_table:
            if str(self.style).strip().lower() != "zbl":
                raise ValueError("core_repulsion.tabulate supports only style='zbl'")
            if int(self.table_points) < 2000:
                raise ValueError("core_repulsion.table_points must be >= 2000 when ZBL tabulation is used")
            if int(self.table_points_max) < int(self.table_points):
                raise ValueError("core_repulsion.table_points_max must be >= core_repulsion.table_points when ZBL tabulation is used")
            if int(self.table_verify_points) < 2000:
                raise ValueError("core_repulsion.table_verify_points must be >= 2000 when ZBL tabulation is used")
            if str(self.table_filename).strip() == "":
                raise ValueError("core_repulsion.table_filename must be non-empty when ZBL tabulation is used")
            if not (math.isfinite(float(self.table_r_min)) and float(self.table_r_min) > 0.0):
                raise ValueError("core_repulsion.table_r_min must be finite and > 0 when ZBL tabulation is used")
        return self



class KimConfig(BaseModel):
    """Kim config."""
    # discriminator unions yamls
    kind: Literal["kim"] = "kim"

    model: str
    user_units: str = "metal"
    unit_conversion_mode: bool = False
    interactions: Union[list[str], Literal["fixed_types"]] = Field(default_factory=list)

    # repulsion buckingham potentials
    core_repulsion: CoreRepulsionConfig = Field(default_factory=CoreRepulsionConfig)

    @field_validator("model")
    @classmethod
    def _model_nonempty(cls, v: str) -> str:
        value = str(v).strip()
        if not value:
            raise ValueError("kim.model must be non-empty")
        return value

    @field_validator("user_units")
    @classmethod
    def _units_style_supported(cls, v: str) -> str:
        from .lammps_units import normalize_lammps_units_style

        return normalize_lammps_units_style(v)

    @field_validator("interactions", mode="before")
    @classmethod
    def _interactions_nonempty(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v == "fixed_types":
                return v
            raise ValueError("kim.interactions must be a species list or 'fixed_types'")
        if not isinstance(v, Sequence) or isinstance(v, (bytes, bytearray)):
            raise ValueError("kim.interactions must be a species list or 'fixed_types'")
        if any(not isinstance(item, str) for item in v):
            raise ValueError("kim.interactions species names must be strings")
        values = [item.strip() for item in v]
        if not values or any(not item for item in values):
            raise ValueError("kim.interactions must contain only non-empty species names")
        return values

    @model_validator(mode="after")
    def _validate_interactions(self) -> "KimConfig":
        if self.interactions != "fixed_types" and len(self.interactions) == 0:
            raise ValueError("kim.interactions must be a non-empty list or 'fixed_types'")
        return self


class LammpsPotentialConfig(BaseModel):
    """Lammps potential config."""

    kind: Literal["lammps"] = "lammps"
    user_units: str = "metal"
    interactions: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    files: list[Path] = Field(default_factory=list)

    core_repulsion: CoreRepulsionConfig = Field(default_factory=CoreRepulsionConfig)

    @field_validator("user_units")
    @classmethod
    def _units_style_supported(cls, v: str) -> str:
        from .lammps_units import normalize_lammps_units_style

        return normalize_lammps_units_style(v)

    @field_validator("interactions", mode="before")
    @classmethod
    def _interaction_species_valid(cls, v: Any) -> list[str]:
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("potential.interactions must be a non-empty species list")
        if any(not isinstance(item, str) for item in v):
            raise ValueError("potential.interactions species names must be strings")
        values = [item.strip() for item in v]
        if not values or any(not item for item in values):
            raise ValueError("potential.interactions must contain only non-empty species names")
        return values

    @field_validator("commands", mode="before")
    @classmethod
    def _commands_valid(cls, v: Any) -> list[str]:
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("potential.commands must be a non-empty list")
        if any(not isinstance(item, str) for item in v):
            raise ValueError("potential.commands entries must be strings")
        values = list(v)
        if not values or any(not item.strip() for item in values):
            raise ValueError("potential.commands must contain only non-empty commands")
        return values

    @field_validator("files", mode="before")
    @classmethod
    def _files_valid(cls, v: Any) -> list[Path]:
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("potential.files must be a list")
        paths: list[Path] = []
        for index, raw in enumerate(v):
            if raw is None or not str(raw).strip():
                raise ValueError(f"potential.files entry at index {index} must be a non-empty path")
            path = Path(str(raw)).expanduser()
            if not path.is_file():
                raise ValueError(f"potential.files entry is not a file: {path}")
            validated_lammps_localized_filename(
                path.name,
                field_name=f"potential.files[{index}] basename",
            )
            paths.append(path.resolve(strict=False))
        return paths

    @model_validator(mode="after")
    def _validate(self) -> "LammpsPotentialConfig":
        if len(self.interactions) == 0:
            raise ValueError("potential.interactions must be a non-empty list for kind='lammps'")
        if len(self.commands) == 0:
            raise ValueError("potential.commands must be a non-empty list for kind='lammps'")
        return self


class MG2SiNPotentialConfig(BaseModel):
    """Mg2 si npotential."""

    kind: Literal["mg2_sin"]

    user_units: str = "metal"
    # lammps order contain
    interactions: list[str] = Field(default_factory=lambda: ["Si", "N"])

    # si n morse
    D0_eV: float = 3.88461
    alpha_invA: float = 2.32660
    r0_A: float = 1.62136

    # general repulsion exp
    A_SiSi_eVA: float = 177.510
    rho_SiSi_A: float = 0.63685

    # n general repulsion
    A_NN_eVA: float = 2499.01
    rho_NN_A: float = 0.36029

    # n damped dispersion
    C6_NN_eVA6: float = 16691.4
    b6_NN_invA: float = 0.50328

    # p5 taper
    x1_A: float = 4.3
    x0_A: float = 5.8

    # table
    r_min_A: float = 0.2
    table_points: int = 6000
    table_filename: str = "mg2_sin.table"

    core_repulsion: CoreRepulsionConfig = Field(default_factory=CoreRepulsionConfig)

    @field_validator("user_units")
    @classmethod
    def _units_style_supported(cls, v: str) -> str:
        # Imported lazily to keep configuration model import time small.
        from .lammps_units import normalize_lammps_units_style

        return normalize_lammps_units_style(v)

    @field_validator(
        "D0_eV",
        "alpha_invA",
        "r0_A",
        "A_SiSi_eVA",
        "rho_SiSi_A",
        "A_NN_eVA",
        "rho_NN_A",
        "C6_NN_eVA6",
        "b6_NN_invA",
        "x1_A",
        "x0_A",
        "r_min_A",
        mode="before",
    )
    @classmethod
    def _positive_finite_parameter(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("MG2 potential parameters must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("MG2 potential parameters must be finite and > 0")
        return x

    @field_validator("table_points", mode="before")
    @classmethod
    def _table_points_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("MG2 table_points must be an integer >= 1000")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1000:
            raise ValueError("MG2 table_points must be an integer >= 1000")
        return n

    @field_validator("table_filename")
    @classmethod
    def _table_filename_valid(cls, v: str) -> str:
        return validated_lammps_localized_filename(
            v, field_name="MG2 table_filename"
        )

    @model_validator(mode="after")
    def _validate(self) -> "MG2SiNPotentialConfig":
        if len(self.interactions) != 2:
            raise ValueError("potential.interactions must have length 2 for kind='mg2_sin'")
        ss = {str(x) for x in self.interactions}
        if ss != {"Si", "N"}:
            raise ValueError("potential.interactions must contain exactly {'Si','N'} for kind='mg2_sin'")
        if float(self.x0_A) <= float(self.x1_A):
            raise ValueError("MG2 taper requires x0_A > x1_A")
        if float(self.r_min_A) <= 0.0 or float(self.r_min_A) >= float(self.x0_A):
            raise ValueError("MG2 table requires 0 < r_min_A < x0_A")
        if int(self.table_points) < 1000:
            raise ValueError("MG2 table_points must be >= 1000")
        return self


# union potential blocks
PotentialConfig = Union[KimConfig, LammpsPotentialConfig, MG2SiNPotentialConfig]


class StructureGenerateConfig(BaseModel):
    """Structure generate config."""

    method: Literal["cod", "cif_url", "materials_project", "packmol", "random", "builtin", "poscar"] = "cod"
    formula: str

    # cod options
    cod_id: Optional[int] = None

    # cif url option
    cif_url: Optional[str] = None

    # structures packaged
    builtin_name: Optional[str] = None

    # poscar option contcar
    poscar_path: Optional[Path] = None

    # materials project option
    mp_material_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("mp_material_id", "material_id"),
    )
    mp_api_key: Optional[str] = None
    mp_api_key_env: Optional[str] = "MP_API_KEY"

    # packmol option
    packmol_cmd: Union[str, List[str]] = "packmol"
    packing_density_g_cm3: Optional[float] = None
    packing_min_distance_A: Optional[float] = None

    # repetition overrides formula
    repeat: Optional[Tuple[int, int, int]] = None

    # density generated structure
    # cell isotropically density
    # ase melt quench
    # reasonable density catastrophes
    target_density_g_cm3: Optional[float] = None

    # size control
    n_formula_units: int = 32
    min_atoms: Optional[int] = None

    # fallback
    fallback_to_random: bool = True
    random_fallback_density_g_cm3: float = 2.5
    random_min_distance: float = 1.5
    seed: int = 12345

    @field_validator("formula")
    @classmethod
    def _formula_nonempty(cls, v: str) -> str:
        value = str(v).strip()
        if not value:
            raise ValueError("structure.generate.formula must be non-empty for every generation method")
        return value

    @field_validator("poscar_path", mode="before")
    @classmethod
    def _poscar_path_valid(cls, v: Any) -> Optional[Path]:
        if v is None:
            return None
        if not str(v).strip():
            raise ValueError("structure.generate.poscar_path must be a non-empty file path")
        return Path(str(v)).expanduser()

    @field_validator("cod_id", mode="before")
    @classmethod
    def _cod_id_valid(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("structure.generate.cod_id must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("structure.generate.cod_id must be an integer >= 1")
        return n

    @field_validator("repeat", mode="before")
    @classmethod
    def _repeat_valid(cls, v: Any) -> Optional[tuple[int, int, int]]:
        if v is None:
            return None
        values = list(v)
        if len(values) != 3:
            raise ValueError("repeat must contain three integers >= 1")
        out: list[int] = []
        for raw in values:
            if isinstance(raw, bool):
                raise ValueError("repeat must contain three integers >= 1")
            x = float(raw)
            n = int(x)
            if not math.isfinite(x) or x != float(n) or n < 1:
                raise ValueError("repeat must contain three integers >= 1")
            out.append(n)
        return (out[0], out[1], out[2])

    @field_validator("n_formula_units", mode="before")
    @classmethod
    def _nfu_positive_strict(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("n_formula_units must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("n_formula_units must be an integer >= 1")
        return n

    @field_validator("min_atoms", mode="before")
    @classmethod
    def _min_atoms_valid(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("min_atoms must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("min_atoms must be an integer >= 1")
        return n

    @field_validator("seed", mode="before")
    @classmethod
    def _structure_seed_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("structure.generate.seed must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("structure.generate.seed must be an integer >= 0")
        return n

    @field_validator(
        "target_density_g_cm3", "packing_density_g_cm3", "packing_min_distance_A",
        mode="before",
    )
    @classmethod
    def _optional_positive_structure_float(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("structure density/distance values must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("structure density/distance values must be finite and > 0")
        return x

    @field_validator("random_fallback_density_g_cm3", "random_min_distance", mode="before")
    @classmethod
    def _positive_structure_float(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("structure fallback density/distance values must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("structure fallback density/distance values must be finite and > 0")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "StructureGenerateConfig":
        if self.method == "cod":
            # Both COD search and post-fetch composition validation use formula.
            pass
        elif self.method == "cif_url":
            if self.cif_url is None or str(self.cif_url).strip() == "":
                raise ValueError("structure.generate.method='cif_url' requires 'cif_url'")
        elif self.method == "materials_project":
            if self.mp_material_id is None or str(self.mp_material_id).strip() == "":
                raise ValueError("structure.generate.method='materials_project' requires 'mp_material_id'")
            if self.formula is None or str(self.formula).strip() == "":
                raise ValueError("structure.generate.method='materials_project' requires 'formula'")
        elif self.method == "packmol":
            if self.formula is None or str(self.formula).strip() == "":
                raise ValueError("structure.generate.method='packmol' requires 'formula'")
        elif self.method == "random":
            if self.formula is None or str(self.formula).strip() == "":
                raise ValueError("structure.generate.method='random' requires 'formula'")
        elif self.method == "builtin":
            if self.builtin_name is None or str(self.builtin_name).strip() == "":
                raise ValueError("structure.generate.method='builtin' requires 'builtin_name'")
            if self.formula is None or str(self.formula).strip() == "":
                raise ValueError("structure.generate.method='builtin' requires 'formula'")
        elif self.method == "poscar":
            if self.poscar_path is None or str(self.poscar_path).strip() == "":
                raise ValueError("structure.generate.method='poscar' requires 'poscar_path'")
            if self.formula is None or str(self.formula).strip() == "":
                raise ValueError("structure.generate.method='poscar' requires 'formula'")
            if not Path(self.poscar_path).is_file():
                raise FileNotFoundError(f"POSCAR file not found or not a regular file: {self.poscar_path}")
        else:
            raise ValueError(f"Unsupported structure.generate.method: {self.method}")

        if self.repeat is not None:
            nx, ny, nz = self.repeat
            if int(nx) < 1 or int(ny) < 1 or int(nz) < 1:
                raise ValueError("repeat must be positive integers")

        if self.target_density_g_cm3 is not None:
            if float(self.target_density_g_cm3) <= 0:
                raise ValueError("target_density_g_cm3 must be > 0")
        if self.packing_density_g_cm3 is not None and float(self.packing_density_g_cm3) <= 0:
            raise ValueError("packing_density_g_cm3 must be > 0")
        if self.packing_min_distance_A is not None and float(self.packing_min_distance_A) <= 0:
            raise ValueError("packing_min_distance_A must be > 0")
        return self

    @field_validator("packmol_cmd")
    @classmethod
    def _packmol_cmd_strip(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, list):
            vv = [str(x).strip() for x in v if str(x).strip() != ""]
            if len(vv) == 0:
                raise ValueError("packmol_cmd list must be non-empty")
            return vv
        s = str(v).strip()
        if s == "":
            raise ValueError("packmol_cmd must be a non-empty string")
        return s

class StructureConfig(BaseModel):
    """Structure config."""

    lammps_data: Optional[Path] = None
    generate: Optional[StructureGenerateConfig] = None

    # species assignment structures
    # mapping species charge
    charges: Optional[Dict[str, float]] = None

    @field_validator("lammps_data", mode="before")
    @classmethod
    def _lammps_data_valid(cls, v: Any) -> Optional[Path]:
        if v is None:
            return None
        if not str(v).strip():
            raise ValueError("structure.lammps_data must be a non-empty file path")
        return Path(str(v)).expanduser()

    @field_validator("charges", mode="before")
    @classmethod
    def _charges_valid(cls, v: Any) -> Optional[dict[str, float]]:
        if v is None:
            return None
        if not isinstance(v, Mapping) or not v:
            raise ValueError("structure.charges must be a non-empty mapping")
        out: dict[str, float] = {}
        for raw_key, raw_value in v.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("structure.charges keys must be non-empty species names")
            if isinstance(raw_value, bool):
                raise ValueError(f"structure.charges[{key!r}] must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"structure.charges[{key!r}] must be finite")
            out[key] = value
        return out

    @model_validator(mode="after")
    def _validate(self) -> "StructureConfig":
        if (self.lammps_data is None) == (self.generate is None):
            raise ValueError("structure must define exactly one of 'lammps_data' or 'generate'")
        if self.lammps_data is not None:
            if not self.lammps_data.is_file():
                raise FileNotFoundError(f"Structure file not found or not a regular file: {self.lammps_data}")
        if self.charges is not None and len(self.charges) == 0:
            raise ValueError("structure.charges must be a non-empty mapping if provided")
        return self


Ensemble = Literal["npt", "nvt"]


ThermostatStyle = Literal["nose-hoover", "csvr", "langevin", "berendsen"]
BarostatStyle = Literal["nose-hoover", "berendsen"]


class ThermostatConfig(BaseModel):
    # Nose-Hoover is the default robust standard for LAMMPS schedules.
    # LAMMPS also supports explicit CSVR, Langevin, and Berendsen thermostat fixes.
    # CP2K currently supports Nose-Hoover and CSVR through the CP2K driver.
    style: ThermostatStyle = "nose-hoover"
    tdamp: float = 100.0  # time depends on engine units

    @field_validator("style", mode="before")
    @classmethod
    def _style_normalise(cls, v):
        s = str(v).strip().lower().replace("_", "-")
        aliases = {
            "nose": "nose-hoover",
            "nosehoover": "nose-hoover",
            "nose-hoover-chain": "nose-hoover",
            "nh": "nose-hoover",
            "bussi": "csvr",
            "bussi-csvr": "csvr",
            "velocity-rescale": "csvr",
            "stochastic-rescale": "csvr",
        }
        return aliases.get(s, s)

    @field_validator("tdamp", mode="before")
    @classmethod
    def _tdamp_positive(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("thermostat.tdamp must be numeric")
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("thermostat.tdamp must be finite and > 0")
        return v


class BarostatConfig(BaseModel):
    # Nose-Hoover is the default robust standard for pressure-coupled LAMMPS schedules.
    # Berendsen is available for users who explicitly request weak pressure coupling.
    style: BarostatStyle = "nose-hoover"
    pdamp: float = 1000.0  # time depends on engine units
    mode: Literal["iso"] = "iso"

    @field_validator("style", mode="before")
    @classmethod
    def _style_normalise(cls, v):
        s = str(v).strip().lower().replace("_", "-")
        aliases = {
            "nose": "nose-hoover",
            "nosehoover": "nose-hoover",
            "nh": "nose-hoover",
        }
        return aliases.get(s, s)

    @field_validator("pdamp", mode="before")
    @classmethod
    def _pdamp_positive(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("barostat.pdamp must be numeric")
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("barostat.pdamp must be finite and > 0")
        return v


class MDConfig(BaseModel):
    timestep: float = 1.0  # time real metal
    atom_style: Literal["atomic", "charge"] = "atomic"
    ensemble: Ensemble = "npt"
    temperature: float = 300.0
    pressure: float = 0.0
    thermostat: ThermostatConfig = Field(default_factory=ThermostatConfig)
    barostat: BarostatConfig = Field(default_factory=BarostatConfig)

    thermo_every: int = 100
    dump_every: int = 1000  # trajectory dump frequency

    # production melt quench
    # discontinuous initializes velocities
    # discontinuous initializes velocities
    # continuous downstream velocities
    # continuous downstream velocities
    # reading prior output
    # trajectory continuous reinitialization
    # barostat variables lammps
    stage_continuity: Literal["discontinuous", "continuous"] = "discontinuous"

    # temperature melt box
    # volume box begins
    # deformation structure lammps
    force_isotropic: bool = False

    # neighbour skin lammps
    # communication distance critical
    # solvers volume fluctuations
    neighbor_skin: float = 2.5

    # increased skin lammps
    # fails range ensemble
    # dependent choose priori
    neighbor_skin_autotune: bool = True
    neighbor_skin_step: float = 0.5
    neighbor_skin_max: float = 5.0

    # kim interactions
    # contains section commands
    # otherwise species ase
    # kim species overrides
    # never commands masses
    mass_mode: Literal['auto','kim','data'] = 'auto'

    @field_validator("timestep", mode="before")
    @classmethod
    def _dt_positive(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("timestep must be numeric, not boolean")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("timestep must be finite and > 0")
        return x

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature_positive(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("temperature must be numeric, not boolean")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        return x

    @field_validator("pressure", mode="before")
    @classmethod
    def _pressure_finite(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("pressure must be numeric, not boolean")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("pressure must be finite")
        return x



    @field_validator("neighbor_skin", "neighbor_skin_step", "neighbor_skin_max", mode="before")
    @classmethod
    def _skin_positive(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("neighbour skin parameters must be numeric")
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("neighbour skin parameters must be finite and > 0")
        return v

    @model_validator(mode="after")
    def _skin_consistent(self):
        if float(self.neighbor_skin_max) < float(self.neighbor_skin):
            raise ValueError("neighbor_skin_max must be >= neighbor_skin")
        return self
    @field_validator("thermo_every", "dump_every", mode="before")
    @classmethod
    def _positive_int(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("frequency must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("frequency must be an integer >= 1")
        return n


class TmScanConfig(BaseModel):
    t_min: float = 300.0
    t_max: float = 3000.0
    dT: float = 100.0

    # override ensemble scan
    # md ensemble
    ensemble: Optional[Ensemble] = None

    # temperature parameters
    equil_steps: int = 20000

    # independent replicas scan
    # seed
    # metrics selection aggregated
    # backwards ergonomic naturally
    replicates_per_temp: int = Field(
        default=5,
        validation_alias=AliasChoices("replicates_per_temp", "replicates"),
    )

    # equilibrate msd structure
    # equilibrated volume msd
    sample_in_nvt: bool = True

    # high t selection
    # onset high margin
    # temperature fraction diffusion
    # conservative network formers
    highT_mode: Literal['onset','liquid'] = 'liquid'
    liquid_D_frac: float = 0.2
    liquid_top_k: int = 3
    liquid_min_consecutive: int = 2
    sample_steps: int = 40000
    msd_every: int = 100

    class GrIndicatorConfig(BaseModel):
        """Gr indicator config."""

        enabled: bool = True
        nbins: int = 400
        r_max: float = 8.0
        frames: int = 3  # average frames trajectory
        stride: int = 2000  # dump stride sampling
        smooth: int = 7
        # pair selector
        pair: Optional[Tuple[AtomSelector, AtomSelector]] = None
        # heuristic windows spacing
        r_ignore_factor: float = 0.3
        r_search_factor: float = 2.5
        # combined diffusion structure
        w_diffusion: float = 1.0
        w_peak_height: float = 1.0
        w_peak_fwhm: float = 0.5

        @field_validator("pair", mode="before")
        @classmethod
        def _pair_valid(cls, v: Any) -> Optional[tuple[AtomSelector, AtomSelector]]:
            if v is None:
                return None
            return _validated_selector_tuple(v, size=2, field_name="tm_scan.gr.pair")  # type: ignore[return-value]

        @field_validator("nbins", "frames", "stride", "smooth", mode="before")
        @classmethod
        def _pos_int(cls, v: Any) -> int:
            if isinstance(v, bool):
                raise ValueError("value must be an integer >= 1")
            x = float(v)
            n = int(x)
            if not math.isfinite(x) or x != float(n) or n < 1:
                raise ValueError("value must be an integer >= 1")
            return n

        @field_validator("r_max")
        @classmethod
        def _rmax_pos(cls, v: float) -> float:
            x = float(v)
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError("r_max must be finite and > 0")
            return x

        @field_validator("r_ignore_factor", "r_search_factor", mode="before")
        @classmethod
        def _search_factor_valid(cls, v: Any) -> float:
            if isinstance(v, bool):
                raise ValueError("RDF search factors must be numeric")
            x = float(v)
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError("RDF search factors must be finite and > 0")
            return x

        @field_validator("w_diffusion", "w_peak_height", "w_peak_fwhm", mode="before")
        @classmethod
        def _weight_valid(cls, v: Any) -> float:
            if isinstance(v, bool):
                raise ValueError("RDF indicator weights must be numeric")
            x = float(v)
            if not math.isfinite(x) or x < 0.0:
                raise ValueError("RDF indicator weights must be finite and >= 0")
            return x

        @model_validator(mode="after")
        def _at_least_one_weight(self) -> "TmScanConfig.GrIndicatorConfig":
            if max(self.w_diffusion, self.w_peak_height, self.w_peak_fwhm) <= 0.0:
                raise ValueError("At least one RDF indicator weight must be > 0")
            return self

    gr: GrIndicatorConfig = Field(default_factory=GrIndicatorConfig)

    @field_validator("t_min", "t_max", "dT", mode="before")
    @classmethod
    def _temperature_grid_finite(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("temperature scan values must be numeric")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("temperature scan values must be finite")
        return x

    @field_validator(
        "replicates_per_temp", "liquid_top_k",
        "liquid_min_consecutive", "sample_steps", "msd_every",
        mode="before",
    )
    @classmethod
    def _tm_positive_integer(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("temperature-scan counts must be integers >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("temperature-scan counts must be integers >= 1")
        return n

    @field_validator("equil_steps", mode="before")
    @classmethod
    def _tm_nonnegative_integer(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("tm_scan.equil_steps must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("tm_scan.equil_steps must be an integer >= 0")
        return n

    @field_validator("liquid_D_frac", mode="before")
    @classmethod
    def _liquid_fraction_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("liquid_D_frac must be numeric")
        x = float(v)
        if not math.isfinite(x) or not (0.0 < x <= 1.0):
            raise ValueError("liquid_D_frac must be finite and in (0,1]")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "TmScanConfig":
        if self.t_min < 0.0:
            raise ValueError("t_min must be >= 0")
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be > t_min")
        if self.dT <= 0:
            raise ValueError("dT must be > 0")
        if int(self.replicates_per_temp) < 1:
            raise ValueError("replicates_per_temp must be >= 1")
        return self


class HighTConfig(BaseModel):
    margin: float = 200.0  # kelvin estimated tm

    # independent equilibration determine
    # disordering recommended steps
    # disorder observed replicas
    replicates: int = 10

    chunk_steps: int = 50000
    max_chunks: int = 20  # upper multiplier safety

    min_total_steps: int = 100000
    rms_multiple: float = 3.0  # sqrt msd rms
    stationarity_tol: float = 0.02  # relative tolerance density

    # stationarity diagnostics autotune
    # stationarity metrics json
    enforce_stationarity: bool = False

    @field_validator("replicates", "chunk_steps", "max_chunks", "min_total_steps", mode="before")
    @classmethod
    def _high_t_positive_integer(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("highT counts must be integers >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("highT counts must be integers >= 1")
        return n

    @field_validator("margin", "stationarity_tol", mode="before")
    @classmethod
    def _high_t_nonnegative_float(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("highT margin/tolerance values must be numeric")
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("highT margin/tolerance values must be finite and >= 0")
        return x

    @field_validator("rms_multiple", mode="before")
    @classmethod
    def _high_t_rms_multiple(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("highT.rms_multiple must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("highT.rms_multiple must be finite and > 0")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "HighTConfig":
        if int(self.replicates) < 1:
            raise ValueError("highT.replicates must be >= 1")
        if int(getattr(self, 'max_chunks', 20)) < 1:
            raise ValueError("highT.max_chunks must be >= 1")
        return self


class QuenchConfig(BaseModel):
    t_final: float = 300.0
    relax_steps: int = 100000

    # cooling rate specification
    # preferred specification physical
    # lammps based style
    rate_min_K_per_ps: float = 0.1
    rate_max_K_per_ps: float = 100.0
    n_rates: int = 7
    rates_K_per_ps: Optional[list[float]] = None

    # reject outside rate
    enforce_rate_bounds: bool = True

    # specify directly lammps
    # metal time real
    rates_K_per_time: Optional[list[float]] = None

    # seed
    # backwards ergonomic naturally
    replicates_per_rate: int = Field(
        default=10,
        validation_alias=AliasChoices("replicates_per_rate", "replicates"),
    )

    @field_validator("t_final", mode="before")
    @classmethod
    def _t_final_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("quench.t_final must be numeric")
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("quench.t_final must be finite and >= 0")
        return x

    @field_validator("rate_min_K_per_ps", "rate_max_K_per_ps", mode="before")
    @classmethod
    def _rate_bound_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("quench rate bounds must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("quench rate bounds must be finite and > 0")
        return x

    @field_validator("rates_K_per_ps", "rates_K_per_time", mode="before")
    @classmethod
    def _rate_list_valid(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (str, bytes)):
            raise ValueError("quench rate lists must be sequences of finite values > 0")
        values = list(v)
        out: list[float] = []
        for raw in values:
            if isinstance(raw, bool):
                raise ValueError("quench rates must be numeric, not boolean")
            x = float(raw)
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError("quench rates must be finite and > 0")
            out.append(x)
        return out

    @field_validator("relax_steps", mode="before")
    @classmethod
    def _relax_steps_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("quench.relax_steps must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("quench.relax_steps must be an integer >= 0")
        return n

    @field_validator("n_rates", "replicates_per_rate", mode="before")
    @classmethod
    def _quench_positive_count(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("quench counts must be integers >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("quench counts must be integers >= 1")
        return n

    @model_validator(mode="after")
    def _validate(self) -> "QuenchConfig":
        if self.rates_K_per_ps is not None and self.rates_K_per_time is not None:
            raise ValueError("Specify either quench.rates_K_per_ps or quench.rates_K_per_time, not both")
        if self.rates_K_per_ps is not None:
            if len(self.rates_K_per_ps) < 2:
                raise ValueError("quench.rates_K_per_ps must contain >= 2 rates")
            if self.enforce_rate_bounds:
                lo = float(self.rate_min_K_per_ps)
                hi = float(self.rate_max_K_per_ps)
                for r in self.rates_K_per_ps:
                    if not (lo <= float(r) <= hi):
                        raise ValueError(
                            f"Cooling rate {r} K/ps is outside [{lo}, {hi}] K/ps; "
                            "adjust rate_min_K_per_ps/rate_max_K_per_ps or set enforce_rate_bounds=false"
                        )
        if self.rates_K_per_time is not None:
            if len(self.rates_K_per_time) < 2:
                raise ValueError("quench.rates_K_per_time must contain >= 2 rates")
        if self.rates_K_per_ps is None and self.rates_K_per_time is None:
            if self.rate_max_K_per_ps <= self.rate_min_K_per_ps:
                raise ValueError("rate_max_K_per_ps must be > rate_min_K_per_ps")
            if self.n_rates < 2:
                raise ValueError("n_rates must be >= 2")
        return self


class SizeConfig(BaseModel):
    # misread removing effects
    # explicitly repeat box
    enabled: bool = False

    # replicate lammps box
    # lammps replicate multiplies
    replicas: list[tuple[int, int, int]] = Field(default_factory=lambda: [(1, 1, 1), (2, 2, 2)])
    replicates_per_size: int = 10  # seed

    # safety scan autotuning
    # override disable filtering
    max_atoms: int = 1000

    @field_validator("replicas", mode="before")
    @classmethod
    def _replicas_valid(cls, v: Any) -> list[tuple[int, int, int]]:
        if not v:
            raise ValueError("size.replicas must be non-empty")
        out: list[tuple[int, int, int]] = []
        for rep in v:
            if len(rep) != 3:
                raise ValueError("size.replicas entries must contain three integers >= 1")
            parsed: list[int] = []
            for raw in rep:
                if isinstance(raw, bool):
                    raise ValueError("size.replicas entries must contain three integers >= 1")
                x = float(raw)
                n = int(x)
                if not math.isfinite(x) or x != float(n) or n < 1:
                    raise ValueError("size.replicas entries must contain three integers >= 1")
                parsed.append(n)
            out.append((parsed[0], parsed[1], parsed[2]))
        return out

    @field_validator("replicates_per_size", "max_atoms", mode="before")
    @classmethod
    def _size_positive_count(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("size counts must be integers >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("size counts must be integers >= 1")
        return n


class ProductionEnsembleConfig(BaseModel):
    """Production ensemble config."""

    enabled: bool = False
    min_boxes: int = 10
    # generate convergence achieved
    # positive integer impose
    max_boxes: Optional[int] = None
    batch_boxes: int = 5

    # production distinct heating
    # production preflight autotune
    # calibration intentionally stringent
    # directly target t
    warmup_start_temperature: float = 300.0

    # physical production subsequent
    # melt recommended selected
    warmup_duration_ps: float = 5.0

    # trajectory quench relax
    dump_trajectory: bool = True
    dump_every_steps: int = 5000

    # configured metrics target
    check_convergence: bool = True

    # robustness convergence successive
    # occurs adding batch
    consecutive_converged_checks: int = 1


    # distribution production convergence
    # resolution bond distributions
    # simultaneous assertions therefore
    # controlled confidence thresholds
    bondlen_cdf_points: int = 200
    angle_cdf_points: int = 180

    # box distribution json
    # output boxes pairs
    store_distributions: bool = True

    # Optional embedding of full analysed final structures in result JSON.
    # Defaults to historical embedded behaviour; analysis-only YAMLs may set
    # this false to rely on manifest hashes and source paths instead.
    embed_structures: bool = True

    # Analysis-only scalability controls.  Production/run-production keep the
    # historical serial behaviour unless users opt in.  Standalone
    # analyze-output YAMLs for large final-structure ensembles may safely set
    # analysis_workers to the desired core count; heavy graph sidecars are then
    # streamed per box instead of accumulated in memory.
    analysis_workers: int = 1
    analysis_streaming: bool = True
    analysis_max_in_flight: Optional[int] = None

    # coordination handling network
    # box exhibiting coordination
    # ensemble convergence structures
    # production rejects auditability
    exclude_coordination_defects: bool = False
    rejects_subdir: str = "rejects"

    # refinement accepted production
    # disabled expensive block
    # engine lammps
    dft_opt: "DftOptConfig" = Field(default_factory=lambda: DftOptConfig())

    @field_validator("analysis_workers", mode="before")
    @classmethod
    def _analysis_workers_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("analysis_workers must be an integer >= 1")
        x = float(v)
        iv = int(x)
        if not math.isfinite(x) or x != float(iv) or iv < 1:
            raise ValueError("analysis_workers must be an integer >= 1")
        return iv

    @field_validator("analysis_max_in_flight", mode="before")
    @classmethod
    def _analysis_max_in_flight_valid(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("analysis_max_in_flight must be an integer >= 1 when provided")
        x = float(v)
        iv = int(x)
        if not math.isfinite(x) or x != float(iv) or iv < 1:
            raise ValueError("analysis_max_in_flight must be an integer >= 1 when provided")
        return iv

    @field_validator(
        "min_boxes",
        "batch_boxes",
        "consecutive_converged_checks",
        "dump_every_steps",
        "bondlen_cdf_points",
        "angle_cdf_points",
        mode="before",
    )
    @classmethod
    def _positive_production_integer(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("production count/frequency values must be integers >= 1")
        x = float(v)
        iv = int(x)
        if not math.isfinite(x) or x != float(iv) or iv < 1:
            raise ValueError("production count/frequency values must be integers >= 1")
        return iv

    @field_validator("max_boxes", mode="before")
    @classmethod
    def _max_boxes_valid(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("production.max_boxes must be null or an integer >= 1")
        x = float(v)
        iv = int(x)
        if not math.isfinite(x) or x != float(iv) or iv < 1:
            raise ValueError("production.max_boxes must be null or an integer >= 1")
        return iv

    @field_validator("warmup_start_temperature", mode="before")
    @classmethod
    def _warmup_start_temperature_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("warmup_start_temperature must be numeric")
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("warmup_start_temperature must be finite and > 0")
        return v

    @field_validator("warmup_duration_ps", mode="before")
    @classmethod
    def _warmup_duration_ps_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("warmup_duration_ps must be numeric")
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("warmup_duration_ps must be finite and > 0")
        return v

    @model_validator(mode="after")
    def _production_counts_consistent(self) -> "ProductionEnsembleConfig":
        if self.max_boxes is not None and int(self.max_boxes) < int(self.min_boxes):
            raise ValueError("production.max_boxes must be >= production.min_boxes")
        return self


class DftOptConfig(BaseModel):
    """Dft opt config."""

    # backwards ergonomic alias
    enabled: bool = Field(default=False, validation_alias=AliasChoices("enabled", "enable"))

    # cp2 cell parameters
    optimizer: Literal["LBFGS", "BFGS", "CG"] = "LBFGS"
    max_iter: int = 200
    keep_angles: bool = True

    # pressure external bar
    external_pressure_bar: Optional[float] = None

    # trajectory frequency cell
    traj_every: int = 1

    # global print level
    print_level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"

    @field_validator("max_iter", "traj_every", mode="before")
    @classmethod
    def _pos_int(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("value must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("value must be an integer >= 1")
        return n

    @field_validator("external_pressure_bar", mode="before")
    @classmethod
    def _external_pressure_finite(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("dft_opt.external_pressure_bar must be numeric")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("dft_opt.external_pressure_bar must be finite")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "DftOptConfig":
        # angles stability keeping
        # compatible orthorhombic analysis
        if not bool(self.keep_angles):
            raise ValueError("dft_opt.keep_angles must be true (KEEP_ANGLES is enforced)")
        return self


class ConvergenceConfig(BaseModel):
    # An explicitly entered tolerance must be recognised.  This lets scan
    # selection ignore calculated diagnostics that have no tolerance while
    # still rejecting misspelled/unsupported convergence YAML keys.
    model_config = ConfigDict(extra="forbid")

    # production ensemble convergence
    # confidence estimation precision
    # stability distribution nonparametric
    # criteria conservative
    mode: Literal["ci", "stability", "both"] = "both"

    # long range
    density_rel_tol: float = 0.01
    density_abs_tol: float = 1e-3  # lammps density

    # short range
    coord_rel_tol: float = 0.0
    coord_abs_tol: float = 0.05

    bondlen_rel_tol: float = 0.0
    bondlen_abs_tol: float = 0.02  # length angstrom

    angle_rel_tol: float = 0.0
    angle_abs_tol: float = 2.0  # degrees

    # medium range
    ring_rel_tol: float = 0.0
    ring_abs_tol: float = 0.05  # tolerance ring fractions

    ring_size_rel_tol: float = 0.0
    ring_size_abs_tol: float = 0.5  # tolerance ring dimensionless

    # peak descriptors short
    gr_peak_r_rel_tol: float = 0.0
    gr_peak_r_abs_tol: float = 0.02
    gr_peak_height_rel_tol: float = 0.05
    gr_peak_height_abs_tol: float = 0.2
    gr_peak_fwhm_rel_tol: float = 0.05
    gr_peak_fwhm_abs_tol: float = 0.02

    # production distribution metrics
    # tolerances curves points
    # units
    # cdf dimensionless probabilities
    # gr curve dimensionless
    bondlen_cdf_rel_tol: float = 0.0
    bondlen_cdf_abs_tol: float = 0.02

    angle_cdf_rel_tol: float = 0.0
    angle_cdf_abs_tol: float = 0.02

    coord_cdf_rel_tol: float = 0.0
    coord_cdf_abs_tol: float = 0.02

    # g curve convergence
    gr_curve_rel_tol: float = 0.05
    gr_curve_abs_tol: float = 0.05

    # curve convergence dimensionless
    sq_curve_rel_tol: float = 0.05
    sq_curve_abs_tol: float = 0.05

    # clearance convergence dimensionless
    void_cdf_rel_tol: float = 0.0
    void_cdf_abs_tol: float = 0.02

    zscore: float = 1.96  # normal

    # confidence metrics probabilities
    # approximation approximate normality
    # empirical distribution conservative
    # hoeffding distribution conservative
    bounded_ci_method: Literal["t", "empirical_bernstein", "hoeffding"] = "t"

    # rate convergence assertions
    # metric confidence
    # bonferroni family number
    familywise: Literal["none", "bonferroni"] = "bonferroni"

    # distribution diagnostics nonparametric
    # stability compares subsets
    # distance empirical distributions
    stability_split: Literal["half", "last_batch"] = "half"
    stability_distance: Literal["wasserstein", "ks"] = "wasserstein"
    # Dimensionless maximum empirical-CDF separation used only when
    # stability_distance='ks'.  Descriptor-unit absolute tolerances cannot be
    # compared to a Kolmogorov--Smirnov statistic.
    stability_ks_tol: float = 0.10
    stability_bootstrap: int = 200
    stability_quantile: float = 0.95

    # Every reported CI/stability assessment must be supported by strictly
    # more than this fraction of the accepted ensemble.  The strict inequality
    # is intentional: the default 0.5 requires at least six contributors in a
    # ten-box ensemble, rather than allowing a half-ensemble diagnostic to be
    # presented as an ensemble-wide result.
    minimum_evidence_fraction: float = 0.5

    @field_validator(
        "density_rel_tol", "density_abs_tol",
        "coord_rel_tol", "coord_abs_tol",
        "bondlen_rel_tol", "bondlen_abs_tol",
        "angle_rel_tol", "angle_abs_tol",
        "ring_rel_tol", "ring_abs_tol",
        "ring_size_rel_tol", "ring_size_abs_tol",
        "gr_peak_r_rel_tol", "gr_peak_r_abs_tol",
        "gr_peak_height_rel_tol", "gr_peak_height_abs_tol",
        "gr_peak_fwhm_rel_tol", "gr_peak_fwhm_abs_tol",
        "bondlen_cdf_rel_tol", "bondlen_cdf_abs_tol",
        "angle_cdf_rel_tol", "angle_cdf_abs_tol",
        "coord_cdf_rel_tol", "coord_cdf_abs_tol",
        "gr_curve_rel_tol", "gr_curve_abs_tol",
        "sq_curve_rel_tol", "sq_curve_abs_tol",
        "void_cdf_rel_tol", "void_cdf_abs_tol",
        "stability_ks_tol",
        "minimum_evidence_fraction",
        mode="before",
    )
    @classmethod
    def _scientific_tolerance_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("convergence tolerances must be numeric, not boolean")
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("convergence tolerances must be finite and >= 0")
        return x

    @field_validator("stability_ks_tol")
    @classmethod
    def _stability_ks_tol_valid(cls, v: float) -> float:
        x = float(v)
        if x > 1.0:
            raise ValueError("convergence.stability_ks_tol must be in [0,1]")
        return x

    @field_validator("minimum_evidence_fraction")
    @classmethod
    def _minimum_evidence_fraction_valid(cls, v: float) -> float:
        x = float(v)
        if not (0.0 <= x < 1.0):
            raise ValueError(
                "convergence.minimum_evidence_fraction must be in [0,1)"
            )
        return x

    @field_validator("zscore", mode="before")
    @classmethod
    def _zscore_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("convergence.zscore must be numeric, not boolean")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("convergence.zscore must be finite and > 0")
        return x

    @field_validator("stability_bootstrap", mode="before")
    @classmethod
    def _stability_bootstrap_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("convergence.stability_bootstrap must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("convergence.stability_bootstrap must be an integer >= 0")
        return n

    @field_validator("stability_quantile", mode="before")
    @classmethod
    def _stability_quantile_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("convergence.stability_quantile must be numeric")
        x = float(v)
        if not math.isfinite(x) or not (0.0 < x < 1.0):
            raise ValueError("convergence.stability_quantile must be finite and in (0,1)")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "ConvergenceConfig":
        if int(self.stability_bootstrap) < 0:
            raise ValueError("convergence.stability_bootstrap must be >= 0")
        self.stability_bootstrap = int(self.stability_bootstrap)
        if not (0.0 < float(self.stability_quantile) < 1.0):
            raise ValueError("convergence.stability_quantile must be in (0,1)")
        self.stability_quantile = float(self.stability_quantile)
        return self


class AutoCutoffConfig(BaseModel):
    """Configuration for legacy neighbour-cutoff estimation.

    ``pooled_ensemble`` derives one shared cutoff map from all readable
    structures (retaining the historical first-successful-box shared fallback
    when pre-resolution is impossible), ``per_box`` derives an independent map
    for each structure, and ``disabled`` forbids estimation (all required
    legacy cutoffs must then be supplied explicitly by the metrics
    configuration or analysis plan).
    """

    # Reject unknown/misspelled keys so a typo (e.g. `scoep:` or `r_maxx:`) fails
    # loudly at config load instead of being silently ignored and falling back to
    # the default. Matches ConvergenceConfig's strict policy.
    model_config = ConfigDict(extra="forbid")

    scope: Literal["pooled_ensemble", "per_box", "disabled"] = "pooled_ensemble"
    r_max: float = 8.0
    nbins: int = 400
    smooth: int = 7  # moving average preferred
    peak_search: Tuple[float, float] = (0.5, 4.0)
    min_search: Tuple[float, float] = (0.8, 6.0)
    fallback_factor: float = 1.3

    @field_validator("r_max", "fallback_factor", mode="before")
    @classmethod
    def _positive_float_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("auto_cutoff numeric values must not be boolean")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("auto_cutoff numeric values must be finite and > 0")
        return x

    @field_validator("nbins", "smooth", mode="before")
    @classmethod
    def _integer_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("auto_cutoff bin/smoothing values must be integers")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n):
            raise ValueError("auto_cutoff bin/smoothing values must be integers")
        return n

    @field_validator("peak_search", "min_search", mode="before")
    @classmethod
    def _range_valid(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("auto_cutoff search ranges must contain two numeric values")
        values = list(v)
        if len(values) != 2 or any(isinstance(item, bool) for item in values):
            raise ValueError("auto_cutoff search ranges must contain two numeric values")
        lo, hi = float(values[0]), float(values[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("auto_cutoff search ranges must be finite")
        return (lo, hi)

    @model_validator(mode="after")
    def _validate(self) -> "AutoCutoffConfig":
        self.r_max = float(self.r_max)
        if not (math.isfinite(self.r_max) and self.r_max > 0.0):
            raise ValueError("auto_cutoff.r_max must be finite and > 0")
        self.nbins = int(self.nbins)
        if self.nbins < 10:
            raise ValueError("auto_cutoff.nbins must be >= 10")
        self.smooth = int(self.smooth)
        if self.smooth < 1:
            raise ValueError("auto_cutoff.smooth must be >= 1")
        if self.smooth % 2 == 0:
            self.smooth += 1
        for field_name in ("peak_search", "min_search"):
            lo_raw, hi_raw = getattr(self, field_name)
            lo, hi = float(lo_raw), float(hi_raw)
            if not (
                math.isfinite(lo)
                and math.isfinite(hi)
                and 0.0 <= lo < hi <= self.r_max
            ):
                raise ValueError(
                    f"auto_cutoff.{field_name} must be a finite (min,max) "
                    f"with 0 <= min < max <= r_max ({self.r_max:g})"
                )
            setattr(self, field_name, (lo, hi))
        self.fallback_factor = float(self.fallback_factor)
        if not (math.isfinite(self.fallback_factor) and self.fallback_factor > 0.0):
            raise ValueError("auto_cutoff.fallback_factor must be finite and > 0")
        return self


class GrMetricConfig(BaseModel):
    """Gr metric config."""

    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None
    r_max: float = 8.0
    nbins: int = 400
    smooth: int = 7

    @field_validator("pair", mode="before")
    @classmethod
    def _pair_valid(cls, v: Any) -> Optional[tuple[AtomSelector, AtomSelector]]:
        if v is None:
            return None
        return _validated_selector_tuple(v, size=2, field_name="gr.pair")  # type: ignore[return-value]

    @field_validator("r_max", mode="before")
    @classmethod
    def _gr_rmax_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("gr.r_max must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("gr.r_max must be finite and > 0")
        return x

    @field_validator("nbins", mode="before")
    @classmethod
    def _gr_nbins_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("gr.nbins must be an integer >= 10")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 10:
            raise ValueError("gr.nbins must be an integer >= 10")
        return n

    @field_validator("smooth", mode="before")
    @classmethod
    def _gr_smooth_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("gr.smooth must be an integer >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("gr.smooth must be an integer >= 1")
        return n


class SqMetricConfig(BaseModel):
    """Sq metric config."""

    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None

    # q grid
    q_max: float = 20.0
    nq: int = 400

    # g integration grid
    r_max: float = 10.0
    nbins: int = 800

    # termination function transform
    window: Literal["lorch", "hann", "none"] = "lorch"

    # peak diffraction amorphous
    peak_search: Tuple[float, float] = (0.5, 3.0)
    smooth: int = 7

    @field_validator("pair", mode="before")
    @classmethod
    def _pair_valid(cls, v: Any) -> Optional[tuple[AtomSelector, AtomSelector]]:
        if v is None:
            return None
        return _validated_selector_tuple(v, size=2, field_name="sq.pair")  # type: ignore[return-value]

    @field_validator("q_max", "r_max", mode="before")
    @classmethod
    def _positive_float_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("sq q/r limits must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("sq q/r limits must be finite and > 0")
        return x

    @field_validator("nq", "nbins", "smooth", mode="before")
    @classmethod
    def _integer_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("sq grid/smoothing values must be integers")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n):
            raise ValueError("sq grid/smoothing values must be integers")
        return n

    @field_validator("peak_search", mode="before")
    @classmethod
    def _peak_range_valid(cls, v: Any) -> tuple[float, float]:
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("sq.peak_search must contain two numeric values")
        values = list(v)
        if len(values) != 2 or any(isinstance(item, bool) for item in values):
            raise ValueError("sq.peak_search must contain two numeric values")
        lo, hi = float(values[0]), float(values[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("sq.peak_search must contain finite values")
        return (lo, hi)

    @model_validator(mode="after")
    def _validate(self) -> "SqMetricConfig":
        if not (math.isfinite(float(self.q_max)) and float(self.q_max) > 0.0):
            raise ValueError("sq.q_max must be > 0")
        if int(self.nq) < 10:
            raise ValueError("sq.nq must be >= 10")
        self.nq = int(self.nq)
        if not (math.isfinite(float(self.r_max)) and float(self.r_max) > 0.0):
            raise ValueError("sq.r_max must be > 0")
        if int(self.nbins) < 50:
            raise ValueError("sq.nbins must be >= 50")
        self.nbins = int(self.nbins)
        a, b = self.peak_search
        if not (
            math.isfinite(float(a))
            and math.isfinite(float(b))
            and 0.0 <= float(a) < float(b) <= float(self.q_max)
        ):
            raise ValueError(
                "sq.peak_search must be a (min,max) with 0 <= min < max <= q_max"
            )
        if int(self.smooth) < 1:
            raise ValueError("sq.smooth must be >= 1")
        self.smooth = int(self.smooth)
        if self.smooth % 2 == 0:
            self.smooth += 1
        return self




class GraphRuleConfig(BaseModel):
    """Explicit graph-induction rule for graph-derived descriptors."""

    name: str = "graph_rule"
    kind: Literal[
        "hard_cutoff",
        "hard_cutoff_sweep",
        "hard_cutoff_interval",
        "soft_logistic",
        "rdf_adaptive",
        "rdf_adaptive_hard_cutoff",
        "rdf_adaptive_hard_cutoff_sweep",
        "rdf_adaptive_hard_cutoff_interval",
        "rdf_adaptive_soft_logistic",
    ] = "hard_cutoff"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    provenance: Any = "config"

    @model_validator(mode="after")
    def _validate(self) -> "GraphRuleConfig":
        if str(self.name).strip() == "":
            raise ValueError("graph_rules.name must be non-empty")
        params = dict(self.parameters or {})

        def _positive(value: Any, label: str) -> float:
            if isinstance(value, bool):
                raise ValueError(f"graph_rules.parameters.{label} must be numeric")
            x = float(value)
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError(
                    f"graph_rules.parameters.{label} must be finite and > 0"
                )
            return x

        adaptive_kind = self.kind == "rdf_adaptive" or self.kind.startswith("rdf_adaptive_")
        for key in ("cutoff", "r0", "sigma", "r_min", "r_max", "search_radius"):
            if key in params and params[key] not in (None, ""):
                if adaptive_kind and key == "search_radius" and str(params[key]).strip().lower() == "auto":
                    params[key] = "auto"
                    continue
                params[key] = _positive(params[key], key)
        for key in ("points", "n"):
            if key in params and params[key] is not None:
                raw = params[key]
                if isinstance(raw, bool):
                    raise ValueError(f"graph_rules.parameters.{key} must be an integer >= 2")
                numeric = float(raw)
                integer = int(numeric)
                if not math.isfinite(numeric) or numeric != float(integer) or integer < 2:
                    raise ValueError(f"graph_rules.parameters.{key} must be an integer >= 2")
                params[key] = integer

        for key in ("bin_width", "smooth_width"):
            if key in params and params[key] not in (None, ""):
                params[key] = _positive(params[key], key)

        if "min_weight" in params and params["min_weight"] not in (None, ""):
            raw_weight = params["min_weight"]
            if isinstance(raw_weight, bool):
                raise ValueError("graph_rules.parameters.min_weight must be numeric")
            min_weight = float(raw_weight)
            if not math.isfinite(min_weight) or min_weight < 0.0 or min_weight > 1.0:
                raise ValueError("graph_rules.parameters.min_weight must be finite and in [0,1]")
            params["min_weight"] = min_weight

        for key in ("connectivity_fraction", "target_largest_component_fraction"):
            if key not in params or params[key] in (None, ""):
                continue
            raw_fraction = params[key]
            if isinstance(raw_fraction, str) and raw_fraction.strip().lower() == "auto":
                params[key] = "auto"
                continue
            if isinstance(raw_fraction, bool):
                raise ValueError(f"graph_rules.parameters.{key} must be numeric or 'auto'")
            fraction = float(raw_fraction)
            if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
                raise ValueError(
                    f"graph_rules.parameters.{key} must be finite and in (0,1], or 'auto'"
                )
            params[key] = fraction

        def _validate_cutoff_collection(value: Any, label: str) -> Any:
            if isinstance(value, Mapping):
                if not value:
                    raise ValueError(
                        f"graph_rules.parameters.{label} must not be empty"
                    )
                return {
                    str(k): _positive(v, f"{label}.{k}")
                    for k, v in value.items()
                }
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                if len(value) == 0:
                    raise ValueError(
                        f"graph_rules.parameters.{label} must not be empty"
                    )
                out: list[Any] = []
                for index, item in enumerate(value):
                    if isinstance(item, Mapping):
                        row = dict(item)
                        if "cutoff" not in row:
                            raise ValueError(
                                f"graph_rules.parameters.{label}[{index}] requires cutoff"
                            )
                        row["cutoff"] = _positive(
                            row["cutoff"], f"{label}[{index}].cutoff"
                        )
                        out.append(row)
                    else:
                        out.append(_positive(item, f"{label}[{index}]"))
                return out
            raise ValueError(
                f"graph_rules.parameters.{label} must be a cutoff mapping or sequence"
            )

        for key in ("cutoffs", "pair_cutoffs", "values", "r_values"):
            if key in params and params[key] not in (None, ""):
                params[key] = _validate_cutoff_collection(params[key], key)

        if "r_min" in params and "r_max" in params:
            if float(params["r_max"]) <= float(params["r_min"]):
                raise ValueError("graph_rules parameters require r_max > r_min")
        derive = str(params.get("derive_from", "")).strip().lower()
        is_rdf_derived = derive in {
            "rdf",
            "rdf_minimum",
            "rdf_first_minimum",
            "pair_distribution",
            "pair_distribution_function",
            "shell_separability",
        } or self.kind == "rdf_adaptive" or self.kind.startswith("rdf_adaptive_")
        if self.kind == "hard_cutoff":
            if "cutoff" not in params and "cutoffs" not in params and "pair_cutoffs" not in params:
                # Empty hard_cutoff is allowed for legacy injection and for
                # derive_from=pair_distribution_function rules resolved at
                # descriptor time.
                pass
        elif self.kind == "hard_cutoff_sweep":
            vals = params.get("cutoffs", params.get("values", params.get("r_values", None)))
            has_range = "r_min" in params and "r_max" in params
            if vals in (None, "") and not has_range and not is_rdf_derived:
                raise ValueError("hard_cutoff_sweep graph rule requires cutoffs/values/r_values, r_min/r_max, or derive_from=pair_distribution_function")
        elif self.kind == "hard_cutoff_interval":
            if ("r_min" not in params or "r_max" not in params) and not is_rdf_derived:
                raise ValueError("hard_cutoff_interval graph rule requires parameters.r_min and parameters.r_max, or derive_from=pair_distribution_function")
        elif self.kind == "soft_logistic":
            if not is_rdf_derived:
                if "r0" not in params and "cutoff" not in params:
                    raise ValueError("soft_logistic graph rule requires parameters.r0 (or cutoff alias)")
                if "sigma" not in params:
                    raise ValueError("soft_logistic graph rule requires parameters.sigma")
        elif self.kind == "rdf_adaptive" or self.kind.startswith("rdf_adaptive_"):
            # These rules intentionally do not contain a numeric cutoff. They are
            # resolved per analysed structure from the pair distribution function.
            pass
        self.parameters = params
        return self


class PairMetricConfig(BaseModel):
    pair: Tuple[AtomSelector, AtomSelector]
    cutoff: Optional[float] = None

    @field_validator("pair", mode="before")
    @classmethod
    def _pair_valid(cls, v: Any) -> tuple[AtomSelector, AtomSelector]:
        return _validated_selector_tuple(v, size=2, field_name="pairs.pair")  # type: ignore[return-value]

    @field_validator("cutoff", mode="before")
    @classmethod
    def _pair_cutoff_valid(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("pair cutoff must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("pair cutoff must be finite and > 0")
        return x


class CoordinationMetricConfig(BaseModel):
    central: AtomSelector
    neighbor: AtomSelector
    cutoff: Optional[float] = None

    # expectation coordination defects
    # report coordination defect
    # statistics production box
    expected: Optional[int] = None
    allowed: Optional[list[int]] = None
    # box defective fraction
    defect_frac_tol: float = 0.0

    @field_validator("central", "neighbor", mode="before")
    @classmethod
    def _selector_valid(cls, v: Any, info: Any) -> AtomSelector:
        return _validated_atom_selector(v, field_name=f"coordinations.{info.field_name}")

    @field_validator("expected", mode="before")
    @classmethod
    def _expected_valid(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("coordinations.expected must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("coordinations.expected must be an integer >= 0")
        return n

    @field_validator("allowed", mode="before")
    @classmethod
    def _allowed_valid(cls, v: Any) -> Optional[list[int]]:
        if v is None:
            return None
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("coordinations.allowed must be a non-empty list of integers >= 0")
        values: list[int] = []
        for raw in v:
            if isinstance(raw, bool):
                raise ValueError("coordinations.allowed must contain integers >= 0")
            x = float(raw)
            n = int(x)
            if not math.isfinite(x) or x != float(n) or n < 0:
                raise ValueError("coordinations.allowed must contain integers >= 0")
            values.append(n)
        if not values:
            raise ValueError("coordinations.allowed must be a non-empty list of integers >= 0")
        return values

    @field_validator("defect_frac_tol", mode="before")
    @classmethod
    def _defect_fraction_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("coordinations.defect_frac_tol must be numeric")
        x = float(v)
        if not math.isfinite(x) or not (0.0 <= x <= 1.0):
            raise ValueError("coordinations.defect_frac_tol must be finite and in [0,1]")
        return x

    @field_validator("cutoff", mode="before")
    @classmethod
    def _coord_cutoff_valid(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("coordination cutoff must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("coordination cutoff must be finite and > 0")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "CoordinationMetricConfig":
        if self.expected is not None:
            if int(self.expected) < 0:
                raise ValueError("coordinations.expected must be >= 0")
        if self.allowed is not None:
            if not isinstance(self.allowed, list) or len(self.allowed) == 0:
                raise ValueError("coordinations.allowed must be a non-empty list of integers")
            if any(int(x) < 0 for x in self.allowed):
                raise ValueError("coordinations.allowed must contain only integers >= 0")
            # duplicate preserving order
            out: list[int] = []
            seen: set[int] = set()
            for x in self.allowed:
                xi = int(x)
                if xi not in seen:
                    out.append(xi)
                    seen.add(xi)
            self.allowed = out
        if self.expected is not None and self.allowed is not None:
            raise ValueError("Specify either coordinations.expected or coordinations.allowed, not both")
        if not (math.isfinite(float(self.defect_frac_tol)) and float(self.defect_frac_tol) >= 0.0 and float(self.defect_frac_tol) <= 1.0):
            raise ValueError("coordinations.defect_frac_tol must be in [0,1]")
        return self


class CoordinationSweepConfig(BaseModel):
    """Coordination sweep config."""

    enabled: bool = True
    dr: float = 0.02
    n_below: int = 0
    n_above: int = 10
    strained_delta: float = 0.10

    @field_validator("n_below", "n_above", mode="before")
    @classmethod
    def _nonnegative_integer_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("coordination_sweep counts must be integers >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("coordination_sweep counts must be integers >= 0")
        return n

    @field_validator("dr", "strained_delta", mode="before")
    @classmethod
    def _finite_numeric_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("coordination_sweep numeric controls must not be boolean")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("coordination_sweep numeric controls must be finite")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "CoordinationSweepConfig":
        if not (math.isfinite(float(self.dr)) and float(self.dr) > 0.0):
            raise ValueError("coordination_sweep.dr must be > 0")
        if int(self.n_below) < 0 or int(self.n_above) < 0:
            raise ValueError("coordination_sweep.n_below/n_above must be >= 0")
        self.n_below = int(self.n_below)
        self.n_above = int(self.n_above)
        if not (math.isfinite(float(self.strained_delta)) and float(self.strained_delta) >= 0.0):
            raise ValueError("coordination_sweep.strained_delta must be >= 0")
        return self


class AngleMetricConfig(BaseModel):
    # triplet central atom
    triplet: Tuple[AtomSelector, AtomSelector, AtomSelector]

    @field_validator("triplet", mode="before")
    @classmethod
    def _triplet_valid(cls, v: Any) -> tuple[AtomSelector, AtomSelector, AtomSelector]:
        return _validated_selector_tuple(v, size=3, field_name="angles.triplet")  # type: ignore[return-value]


class RingMetricsConfig(BaseModel):
    enabled: bool = False
    mode: Literal["bond_graph", "projected"] = "projected"
    # ring definition
    # primitive franzblau chordless
    # basis networkx unique
    algorithm: Literal["primitive", "cycle_basis"] = "primitive"
    nodes: list[AtomSelector] = Field(default_factory=list)
    bridge: Optional[AtomSelector] = None
    max_cycle_size: int = 12
    max_paths_per_edge: int = 16  # enumeration shortest primitive
    # bond pairs cutoffs
    bond_pairs: list[PairMetricConfig] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _nodes_valid(cls, v: Any) -> list[AtomSelector]:
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("rings.nodes must be a list of atom selectors")
        return [
            _validated_atom_selector(item, field_name=f"rings.nodes[{index}]")
            for index, item in enumerate(v)
        ]

    @field_validator("bridge", mode="before")
    @classmethod
    def _bridge_valid(cls, v: Any) -> Optional[AtomSelector]:
        if v is None:
            return None
        return _validated_atom_selector(v, field_name="rings.bridge")

    @field_validator("max_cycle_size", "max_paths_per_edge", mode="before")
    @classmethod
    def _ring_count_valid(cls, v: Any, info: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("ring limits must be integers")
        x = float(v)
        n = int(x)
        minimum = 3 if info.field_name == "max_cycle_size" else 1
        if not math.isfinite(x) or x != float(n) or n < minimum:
            raise ValueError(f"rings.{info.field_name} must be an integer >= {minimum}")
        return n

    @model_validator(mode="after")
    def _validate(self) -> "RingMetricsConfig":
        if bool(self.enabled):
            if not isinstance(self.nodes, list) or len(self.nodes) == 0:
                raise ValueError(
                    "rings.enabled=true requires rings.nodes to be a non-empty list (e.g. nodes: ['Si'])."
                )
            if str(self.mode) == "projected" and self.bridge is None:
                raise ValueError(
                    "rings.mode='projected' requires rings.bridge (e.g. bridge: 'O'). "
                    "Alternatively set rings.mode='bond_graph'."
                )
            if int(self.max_cycle_size) < 3:
                raise ValueError("rings.max_cycle_size must be >= 3")
            if int(getattr(self, "max_paths_per_edge", 1)) < 1:
                raise ValueError("rings.max_paths_per_edge must be >= 1")
        return self


class VoidMetricsConfig(BaseModel):
    """Void metrics config."""

    enabled: bool = True

    # number sample analysis
    n_samples: int = 8192

    # sample analysis lighter
    n_samples_timeseries: int = 2048

    sampler: Literal["sobol", "random", "grid"] = "sobol"
    seed: int = 0

    # nearest heterogeneous robust
    k_nearest: int = 16

    # excluded distance lammps
    default_radius: float = 0.0
    radii: dict[str, float] = Field(default_factory=dict)

    # clearance production convergence
    r_max: float = 5.0
    cdf_points: int = 200

    # accessible fraction clearance
    probe_radii: list[float] = Field(default_factory=list)

    @field_validator("n_samples", "n_samples_timeseries", "k_nearest", "cdf_points", mode="before")
    @classmethod
    def _void_count_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("void sampling counts must be integers")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n):
            raise ValueError("void sampling counts must be integers")
        return n

    @field_validator("seed", mode="before")
    @classmethod
    def _void_seed_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("voids.seed must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("voids.seed must be an integer >= 0")
        return n

    @model_validator(mode="after")
    def _validate(self) -> "VoidMetricsConfig":
        if int(self.n_samples) < 128:
            raise ValueError("voids.n_samples must be >= 128")
        self.n_samples = int(self.n_samples)
        if int(self.n_samples_timeseries) < 16:
            raise ValueError("voids.n_samples_timeseries must be >= 16")
        self.n_samples_timeseries = int(self.n_samples_timeseries)
        if int(self.k_nearest) < 1:
            raise ValueError("voids.k_nearest must be >= 1")
        self.k_nearest = int(self.k_nearest)
        if not (math.isfinite(float(self.default_radius)) and float(self.default_radius) >= 0.0):
            raise ValueError("voids.default_radius must be >= 0")
        # radii mapping
        if not isinstance(self.radii, dict):
            raise ValueError("voids.radii must be a mapping of species->radius")
        for k, v in self.radii.items():
            if not (isinstance(k, str) and k):
                raise ValueError("voids.radii keys must be non-empty strings")
            if not (math.isfinite(float(v)) and float(v) >= 0.0):
                raise ValueError(f"voids.radii['{k}'] must be >= 0")
        if not (math.isfinite(float(self.r_max)) and float(self.r_max) > 0.0):
            raise ValueError("voids.r_max must be > 0")
        if int(self.cdf_points) < 16:
            raise ValueError("voids.cdf_points must be >= 16")
        self.cdf_points = int(self.cdf_points)
        # probe radii
        if not isinstance(self.probe_radii, list):
            raise ValueError("voids.probe_radii must be a list of floats")
        out: list[float] = []
        for rp in self.probe_radii:
            rpf = float(rp)
            if not (math.isfinite(rpf) and rpf >= 0.0):
                raise ValueError("voids.probe_radii must contain only floats >= 0")
            out.append(rpf)
        self.probe_radii = out
        return self


class AmorphousReferenceConfig(BaseModel):
    """Amorphous reference config."""

    enabled: bool = True
    required: bool = False
    source: Literal["materials_project"] = "materials_project"

    # mp omitted matching
    # lowest candidates filtering
    material_ids: list[str] = Field(default_factory=list)

    # materials project structure
    mp_api_key: Optional[str] = None
    mp_api_key_env: Optional[str] = "MP_API_KEY"

    stable_only: bool = True
    energy_above_hull_max: float = 0.05
    max_candidates: int = 6
    use_conventional_cell: bool = True

    # fingerprints crystal peak
    # box amorphous cell
    min_supercell_length_A: float = 15.0
    min_supercell_atoms: int = 256

    @model_validator(mode="after")
    def _validate(self) -> "AmorphousReferenceConfig":
        if int(self.max_candidates) < 1:
            raise ValueError("amorphous.reference.max_candidates must be >= 1")
        self.max_candidates = int(self.max_candidates)
        if not (math.isfinite(float(self.energy_above_hull_max)) and float(self.energy_above_hull_max) >= 0.0):
            raise ValueError("amorphous.reference.energy_above_hull_max must be >= 0")
        if not (math.isfinite(float(self.min_supercell_length_A)) and float(self.min_supercell_length_A) > 0.0):
            raise ValueError("amorphous.reference.min_supercell_length_A must be > 0")
        if int(self.min_supercell_atoms) < 1:
            raise ValueError("amorphous.reference.min_supercell_atoms must be >= 1")
        self.min_supercell_atoms = int(self.min_supercell_atoms)
        self.material_ids = [str(x).strip() for x in list(self.material_ids or []) if str(x).strip() != ""]
        return self


class AmorphousMetricsConfig(BaseModel):
    """Amorphous metrics config."""

    enabled: bool = False

    # amorphous rate production
    enforce_during_rate_scan: bool = True
    enforce_during_production: bool = True

    # fraction candidate rate
    # amorphous means replicas
    min_pass_fraction: float = 1.0

    # fingerprint sharpness analysis
    q_max: float = 20.0
    nq: int = 400
    r_max: float = 10.0
    nbins: int = 800
    window: Literal["lorch", "hann", "none"] = "lorch"
    smooth: int = 7
    peak_search: Tuple[float, float] = (0.5, 12.0)
    peak_prominence_min: float = 0.15
    peak_height_min: float = 1.05
    max_bragg_sharpness: float = 25.0

    # bond order analysis
    l_values: list[int] = Field(default_factory=lambda: [4, 6])
    solid_like_l: int = 6
    solid_like_bond_threshold: float = 0.5
    ordered_min_neighbors: int = 3
    ordered_min_fraction: float = 0.6
    max_crystalline_fraction: float = 0.15
    max_largest_cluster_fraction: float = 0.10

    # reference fingerprint materials
    reference: AmorphousReferenceConfig = Field(default_factory=AmorphousReferenceConfig)
    reference_peak_match_tol: float = 0.20
    max_reference_peak_overlap: float = 0.65

    @model_validator(mode="after")
    def _validate(self) -> "AmorphousMetricsConfig":
        if not (0.0 <= float(self.min_pass_fraction) <= 1.0):
            raise ValueError("amorphous.min_pass_fraction must be in [0,1]")
        if not (math.isfinite(float(self.q_max)) and float(self.q_max) > 0.0):
            raise ValueError("amorphous.q_max must be > 0")
        if int(self.nq) < 64:
            raise ValueError("amorphous.nq must be >= 64")
        self.nq = int(self.nq)
        if not (math.isfinite(float(self.r_max)) and float(self.r_max) > 0.0):
            raise ValueError("amorphous.r_max must be > 0")
        if int(self.nbins) < 100:
            raise ValueError("amorphous.nbins must be >= 100")
        self.nbins = int(self.nbins)
        if int(self.smooth) < 1:
            raise ValueError("amorphous.smooth must be >= 1")
        self.smooth = int(self.smooth)
        if self.smooth % 2 == 0:
            self.smooth += 1
        a, b = self.peak_search
        if not (
            math.isfinite(float(a))
            and math.isfinite(float(b))
            and 0.0 <= float(a) < float(b) <= float(self.q_max)
        ):
            raise ValueError(
                "amorphous.peak_search must satisfy 0 <= min < max <= q_max"
            )
        if not (math.isfinite(float(self.peak_prominence_min)) and float(self.peak_prominence_min) >= 0.0):
            raise ValueError("amorphous.peak_prominence_min must be >= 0")
        if not (math.isfinite(float(self.peak_height_min)) and float(self.peak_height_min) >= 0.0):
            raise ValueError("amorphous.peak_height_min must be >= 0")
        if not (math.isfinite(float(self.max_bragg_sharpness)) and float(self.max_bragg_sharpness) >= 0.0):
            raise ValueError("amorphous.max_bragg_sharpness must be >= 0")
        ll: list[int] = []
        seen: set[int] = set()
        for x in list(self.l_values or []):
            xi = int(x)
            if xi < 1:
                raise ValueError("amorphous.l_values must contain positive integers")
            if xi not in seen:
                ll.append(xi)
                seen.add(xi)
        if len(ll) == 0:
            raise ValueError("amorphous.l_values must be non-empty")
        self.l_values = ll
        if int(self.solid_like_l) not in set(self.l_values):
            self.l_values = sorted(set(list(self.l_values) + [int(self.solid_like_l)]))
        if not (-1.0 <= float(self.solid_like_bond_threshold) <= 1.0):
            raise ValueError("amorphous.solid_like_bond_threshold must be in [-1,1]")
        if int(self.ordered_min_neighbors) < 1:
            raise ValueError("amorphous.ordered_min_neighbors must be >= 1")
        self.ordered_min_neighbors = int(self.ordered_min_neighbors)
        if not (0.0 <= float(self.ordered_min_fraction) <= 1.0):
            raise ValueError("amorphous.ordered_min_fraction must be in [0,1]")
        if not (0.0 <= float(self.max_crystalline_fraction) <= 1.0):
            raise ValueError("amorphous.max_crystalline_fraction must be in [0,1]")
        if not (0.0 <= float(self.max_largest_cluster_fraction) <= 1.0):
            raise ValueError("amorphous.max_largest_cluster_fraction must be in [0,1]")
        if not (math.isfinite(float(self.reference_peak_match_tol)) and float(self.reference_peak_match_tol) > 0.0):
            raise ValueError("amorphous.reference_peak_match_tol must be > 0")
        if not (0.0 <= float(self.max_reference_peak_overlap) <= 1.0):
            raise ValueError("amorphous.max_reference_peak_overlap must be in [0,1]")
        return self


class ElasticScreenConfig(BaseModel):
    """Elastic screen config."""

    enabled: bool | Literal["auto"] = "auto"
    run_on_relax: bool = True
    run_on_highT_when_force_isotropic: bool = True
    strict_when_force_isotropic: bool = True
    # elastic production quench
    collect_during_production_stages: bool = True
    stage_timeseries_frame_stride: int = 1
    stage_timeseries_max_frames: int = 8
    stage_timeseries_make_plot: bool = True
    quench_tail_min_frames: int = 12
    quench_tail_focus_fraction: float = 0.75
    quench_tail_fallback_fraction: float = 0.40
    diffusion_freeze_threshold_A2_per_ps: float = 0.1
    born_delta: float = 1.0e-5
    make_plot: bool = True
    isotropy_warn_threshold: float = 0.15
    coupling_warn_threshold: float = 0.10
    hotspot_warn_multiple_of_median: float = 5.0

    @field_validator(
        "stage_timeseries_frame_stride",
        "stage_timeseries_max_frames",
        "quench_tail_min_frames",
        mode="before",
    )
    @classmethod
    def _positive_integer_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("elastic sampling counts must be integers >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("elastic sampling counts must be integers >= 1")
        return n

    @field_validator(
        "born_delta",
        "quench_tail_focus_fraction",
        "quench_tail_fallback_fraction",
        "diffusion_freeze_threshold_A2_per_ps",
        "isotropy_warn_threshold",
        "coupling_warn_threshold",
        "hotspot_warn_multiple_of_median",
        mode="before",
    )
    @classmethod
    def _finite_numeric_valid(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("elastic numeric controls must not be boolean")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("elastic numeric controls must be finite")
        return x

    @model_validator(mode="after")
    def _validate(self) -> "ElasticScreenConfig":
        if not (math.isfinite(float(self.born_delta)) and float(self.born_delta) > 0.0):
            raise ValueError("elastic.born_delta must be finite and > 0")
        if int(self.stage_timeseries_frame_stride) < 1:
            raise ValueError("elastic.stage_timeseries_frame_stride must be >= 1")
        self.stage_timeseries_frame_stride = int(self.stage_timeseries_frame_stride)
        if int(self.stage_timeseries_max_frames) < 1:
            raise ValueError("elastic.stage_timeseries_max_frames must be >= 1")
        self.stage_timeseries_max_frames = int(self.stage_timeseries_max_frames)
        if int(self.quench_tail_min_frames) < 1:
            raise ValueError("elastic.quench_tail_min_frames must be >= 1")
        self.quench_tail_min_frames = int(self.quench_tail_min_frames)
        if not (0.0 < float(self.quench_tail_focus_fraction) <= 1.0):
            raise ValueError("elastic.quench_tail_focus_fraction must be in (0,1]")
        if not (0.0 < float(self.quench_tail_fallback_fraction) < 1.0):
            raise ValueError("elastic.quench_tail_fallback_fraction must be in (0,1)")
        if not (
            math.isfinite(float(self.diffusion_freeze_threshold_A2_per_ps))
            and float(self.diffusion_freeze_threshold_A2_per_ps) >= 0.0
        ):
            raise ValueError("elastic.diffusion_freeze_threshold_A2_per_ps must be >= 0")
        if not (
            math.isfinite(float(self.isotropy_warn_threshold))
            and float(self.isotropy_warn_threshold) >= 0.0
        ):
            raise ValueError("elastic.isotropy_warn_threshold must be >= 0")
        if not (
            math.isfinite(float(self.coupling_warn_threshold))
            and float(self.coupling_warn_threshold) >= 0.0
        ):
            raise ValueError("elastic.coupling_warn_threshold must be >= 0")
        if not (
            math.isfinite(float(self.hotspot_warn_multiple_of_median))
            and float(self.hotspot_warn_multiple_of_median) >= 1.0
        ):
            raise ValueError("elastic.hotspot_warn_multiple_of_median must be >= 1")
        return self


class StructureMetricsConfig(BaseModel):
    """Structure metrics config."""

    enabled: bool = False

    # structural metrics relax
    # generate dump accordingly
    # production metric collection
    # metrics quench relax
    time_average_frames: int = 5
    time_average_stride: int = 200
    collect_during_production_stages: bool = True
    stage_timeseries_frame_stride: int = 1
    stage_timeseries_max_frames: int = 64
    stage_timeseries_make_plot: bool = False
    quench_tail_min_frames: int = 24
    quench_tail_focus_fraction: float = 0.60
    quench_tail_fallback_fraction: float = 0.40
    elastic: ElasticScreenConfig = Field(default_factory=ElasticScreenConfig)

    # override lammps species
    # kim interactions
    type_to_species: Optional[list[str]] = None

    # Optional material profile.  Core analysis must not hard-code material
    # chemistry; profile-dependent descriptors report not_applicable when the
    # required assumptions are absent.
    material_profile: Dict[str, Any] = Field(default_factory=dict)

    # graph induction rules for graph-derived descriptors
    graph_rules: list[GraphRuleConfig] = Field(default_factory=list)

    # metrics
    pairs: list[PairMetricConfig] = Field(default_factory=list)
    coordinations: list[CoordinationMetricConfig] = Field(default_factory=list)
    # diagnostics coordination cutoff
    coordination_sweep: CoordinationSweepConfig = Field(default_factory=CoordinationSweepConfig)
    angles: list[AngleMetricConfig] = Field(default_factory=list)
    rings: RingMetricsConfig = Field(default_factory=RingMetricsConfig)
    gr: list[GrMetricConfig] = Field(default_factory=list)
    sq: list[SqMetricConfig] = Field(default_factory=list)
    voids: VoidMetricsConfig = Field(default_factory=VoidMetricsConfig)
    amorphous: AmorphousMetricsConfig = Field(default_factory=AmorphousMetricsConfig)

    auto_cutoff: AutoCutoffConfig = Field(default_factory=AutoCutoffConfig)

    @field_validator("type_to_species", mode="before")
    @classmethod
    def _type_to_species_valid(cls, v: Any) -> Optional[list[str]]:
        if v is None:
            return None
        if not isinstance(v, Sequence) or isinstance(v, (str, bytes, bytearray)):
            raise ValueError("metrics.type_to_species must be a non-empty species list")
        if any(not isinstance(item, str) for item in v):
            raise ValueError("metrics.type_to_species entries must be strings")
        values = [item.strip() for item in v]
        if not values or any(not item for item in values):
            raise ValueError("metrics.type_to_species must contain only non-empty species names")
        return values

    @field_validator(
        "time_average_frames",
        "time_average_stride",
        "stage_timeseries_frame_stride",
        "stage_timeseries_max_frames",
        "quench_tail_min_frames",
        mode="before",
    )
    @classmethod
    def _pos_int(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("metrics sampling counts must be integers >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("metrics sampling counts must be integers >= 1")
        return n

    @field_validator("quench_tail_focus_fraction", "quench_tail_fallback_fraction", mode="before")
    @classmethod
    def _tail_fraction_finite(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("metrics quench-tail fractions must be numeric")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("metrics quench-tail fractions must be finite")
        return x


    @model_validator(mode="after")
    def _voids_mandatory_when_enabled(self) -> "StructureMetricsConfig":
        """Voids mandatory when."""
        if self.enabled and (self.voids is None or not bool(getattr(self.voids, 'enabled', False))):
            raise ValueError("Void analysis is mandatory when structure metrics are enabled (voids.enabled must be true).")
        if not (0.0 < float(self.quench_tail_focus_fraction) <= 1.0):
            raise ValueError("metrics.quench_tail_focus_fraction must be in (0,1]")
        if not (0.0 < float(self.quench_tail_fallback_fraction) < 1.0):
            raise ValueError("metrics.quench_tail_fallback_fraction must be in (0,1)")
        return self





class PreflightConfig(BaseModel):
    """Preflight config."""

    enabled: bool = True

    # candidate ensembles consider
    ensembles: Optional[list[Ensemble]] = None


    # production ensemble candidate
    # select nvt unless
    allow_nvt_fallback: bool = False

    # scan specified timestep
    tdamp_factors: list[float] = Field(default_factory=lambda: [50.0, 100.0, 200.0, 500.0])
    pdamp_factors: list[float] = Field(default_factory=lambda: [500.0, 1000.0, 2000.0, 5000.0])

    # conservative volume neighbor
    # metal barostat pdamp
    min_pdamp_ps_highT: float = 5.0

    # candidate timesteps preflight
    # fallback
    dt_candidates: Optional[list[float]] = None

    # fallback
    # preflight undeclared timesteps
    # appearing generated engine
    allow_implicit_dt_fallback: bool = False

    # temperatures derived scan
    T_low: Optional[float] = None
    T_high: Optional[float] = None

    equil_steps: int = 2000
    run_steps: int = 5000

    # samples preflight statistics
    # temperature volume fluctuation
    # effective thermo frequency
    tail_window: int = 50

    # confirmation length stability
    # instabilities barostat neighbor
    # therefore candidates selection
    confirm_equil_steps: int = 2000
    confirm_run_steps: int = 25000
    confirm_topk: int = 3

    # temperatures confirmation addition
    # silica
    confirm_temps: Optional[list[float]] = None

    # preflight lammps invocation
    # prevents mpi indefinitely
    timeout_sec: float = 600.0

    # acceptance heuristic
    temp_rel_tol: float = 0.05
    press_abs_tol: float = 5000.0

    # pressure convergence objective
    # preflight anharmonic temperature
    # mean pressures target
    # integrator perfectly candidates
    # target prevent selection
    # mean pressure half
    # mean pressure half
    # press mark candidate
    require_pressure_tolerance: bool = False

    # safety failure exceeded
    max_temp_factor: float = 3.0
    max_press_abs: float = 1.0e7
    max_vol_ratio: float = 3.0

    @field_validator("equil_steps", "confirm_equil_steps", mode="before")
    @classmethod
    def _preflight_nonnegative_count(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("preflight equilibration counts must be integers >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("preflight equilibration counts must be integers >= 0")
        return n

    @field_validator("run_steps", "tail_window", "confirm_run_steps", "confirm_topk", mode="before")
    @classmethod
    def _preflight_positive_count(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("preflight counts must be integers >= 1")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 1:
            raise ValueError("preflight counts must be integers >= 1")
        return n

    @field_validator("T_low", "T_high", mode="before")
    @classmethod
    def _preflight_optional_temperature(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("preflight temperatures must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("preflight temperatures must be finite and > 0")
        return x

    @field_validator("confirm_temps", mode="before")
    @classmethod
    def _preflight_confirm_temperatures(cls, v: Any) -> Optional[list[float]]:
        if v is None:
            return None
        values: list[float] = []
        for raw in list(v):
            if isinstance(raw, bool):
                raise ValueError("preflight.confirm_temps must contain numeric temperatures")
            x = float(raw)
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError("preflight.confirm_temps must contain finite values > 0")
            values.append(x)
        if not values:
            raise ValueError("preflight.confirm_temps must be non-empty when provided")
        return values

    @field_validator("min_pdamp_ps_highT", "timeout_sec", mode="before")
    @classmethod
    def _preflight_positive_float(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("preflight duration values must be numeric")
        x = float(v)
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError("preflight duration values must be finite and > 0")
        return x

    @field_validator("temp_rel_tol", "press_abs_tol", mode="before")
    @classmethod
    def _preflight_nonnegative_float(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("preflight tolerances must be numeric")
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("preflight tolerances must be finite and >= 0")
        return x

    @field_validator("max_temp_factor", "max_press_abs", "max_vol_ratio", mode="before")
    @classmethod
    def _preflight_safety_finite(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("preflight safety limits must be numeric")
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("preflight safety limits must be finite")
        return x

    @field_validator("equil_steps")
    @classmethod
    def _equil_nonneg(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("equil_steps must be >= 0")
        return int(v)

    @field_validator("run_steps")
    @classmethod
    def _run_pos(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("run_steps must be >= 1")
        return int(v)

    @field_validator("tail_window")
    @classmethod
    def _tail_pos(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("tail_window must be >= 1")
        return int(v)

    @field_validator("confirm_equil_steps")
    @classmethod
    def _confirm_equil_nonneg(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("confirm_equil_steps must be >= 0")
        return int(v)

    @field_validator("confirm_run_steps")
    @classmethod
    def _confirm_run_pos(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("confirm_run_steps must be >= 1")
        return int(v)

    @field_validator("confirm_topk")
    @classmethod
    def _confirm_topk_pos(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("confirm_topk must be >= 1")
        return int(v)

    @field_validator("tdamp_factors", "pdamp_factors", mode="before")
    @classmethod
    def _factors_valid(cls, v: list[float]) -> list[float]:
        if v is None or len(v) == 0:
            raise ValueError("scan factor list must be non-empty")
        vv = [float(x) for x in v]
        if any((not math.isfinite(x)) or x <= 0.0 for x in vv):
            raise ValueError("scan factors must be finite and > 0")
        return vv

    @field_validator("dt_candidates", mode="before")
    @classmethod
    def _dt_candidates_valid(cls, v: Any) -> Optional[list[float]]:
        if v is None:
            return None
        vv: list[float] = []
        for raw in list(v):
            if isinstance(raw, bool):
                raise ValueError("dt_candidates must contain numeric values")
            vv.append(float(raw))
        if len(vv) == 0:
            return None
        if any((not math.isfinite(x)) or x <= 0.0 for x in vv):
            raise ValueError("dt_candidates must contain only finite values > 0")
        # preserve order duplicates
        out: list[float] = []
        seen: set[float] = set()
        for x in vv:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    @model_validator(mode="after")
    def _validate(self) -> "PreflightConfig":
        if self.max_temp_factor <= 1.0:
            raise ValueError("max_temp_factor must be > 1")
        if self.press_abs_tol < 0:
            raise ValueError("press_abs_tol must be >= 0")
        if self.max_press_abs <= 0:
            raise ValueError("max_press_abs must be > 0")
        if self.max_vol_ratio <= 1.0:
            raise ValueError("max_vol_ratio must be > 1")
        return self

class AutoTuneConfig(BaseModel):
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)
    tm_scan: TmScanConfig = Field(default_factory=TmScanConfig)
    highT: HighTConfig = Field(default_factory=HighTConfig)
    quench: QuenchConfig = Field(default_factory=QuenchConfig)
    size: SizeConfig = Field(default_factory=SizeConfig)
    production: ProductionEnsembleConfig = Field(default_factory=ProductionEnsembleConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    metrics: StructureMetricsConfig = Field(default_factory=StructureMetricsConfig)


class RunConfig(BaseModel):
    # execution backend
    # lammps classical md
    # cp2k aimd quickstep
    engine: Literal["lammps", "cp2k"] = "lammps"

    lammps: LammpsConfig = Field(default_factory=LammpsConfig)
    cp2k: Optional[Cp2kConfig] = None
    # compatible kim potential
    kim: Optional[PotentialConfig] = Field(default=None, validation_alias=AliasChoices("kim", "potential"))
    structure: StructureConfig
    md: MDConfig = Field(default_factory=MDConfig)
    autotune: AutoTuneConfig = Field(default_factory=AutoTuneConfig)

    # Separate run-custom extension blocks. Their complete, alias-aware schema
    # is validated by workflows.custom_schedule before execution; declaring
    # them here keeps the strict top-level model from discarding a supported
    # feature while all other unknown top-level keys remain forbidden.
    custom_schedule: Optional[Dict[str, Any]] = None
    hardcarbon_schedule: Optional[Dict[str, Any]] = None

    random_seed: int = 12345

    @field_validator("random_seed", mode="before")
    @classmethod
    def _random_seed_valid(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("random_seed must be an integer >= 0")
        x = float(v)
        n = int(x)
        if not math.isfinite(x) or x != float(n) or n < 0:
            raise ValueError("random_seed must be an integer >= 0")
        return n

    @model_validator(mode="after")
    def _validate_engine_requirements(self) -> "RunConfig":
        if self.custom_schedule is not None and self.hardcarbon_schedule is not None:
            raise ValueError(
                "Specify only one of custom_schedule or hardcarbon_schedule; neither block is silently ignored"
            )
        eng = str(self.engine).strip().lower()
        if eng == "lammps":
            if self.kim is None:
                raise ValueError("engine='lammps' requires a 'kim:' or 'potential:' block")
        elif eng == "cp2k":
            if self.cp2k is None:
                raise ValueError("engine='cp2k' requires a 'cp2k:' block")
            # cp2 driver scope
            # orthorhombic periodic subset
            # fixed cell isotropic
            ens = str(self.md.ensemble).strip().lower()
            if ens not in ("nvt", "npt"):
                raise ValueError("CP2K driver currently supports md.ensemble in {'nvt','npt'} only")
            if bool(getattr(self.md, "force_isotropic", False)):
                raise ValueError("md.force_isotropic is supported only for engine='lammps'")
            th_style = str(getattr(self.md.thermostat, "style", "nose-hoover")).strip().lower()
            if th_style not in {"nose-hoover", "csvr"}:
                raise ValueError(f"thermostat.style={th_style!r} is supported only for engine='lammps'")
            bar_style = str(getattr(self.md.barostat, "style", "nose-hoover")).strip().lower()
            if bar_style != "nose-hoover":
                raise ValueError("CP2K driver currently supports only barostat.style='nose-hoover'")
            elastic_cfg = getattr(getattr(self.autotune, "metrics", None), "elastic", None)
            elastic_enabled = getattr(elastic_cfg, "enabled", "auto") if elastic_cfg is not None else "auto"
            if elastic_enabled is True:
                raise ValueError("autotune.metrics.elastic.enabled=true is supported only for engine='lammps'")
        else:
            raise ValueError(f"Unknown engine '{self.engine}'")

        # optimisation converged production
        prod = getattr(self.autotune, "production", None)
        try:
            dft_enabled = bool(getattr(getattr(prod, "dft_opt", None), "enabled", False))
        except Exception:
            dft_enabled = False
        if prod is not None and bool(getattr(prod, "enabled", False)) and dft_enabled:
            # cp2 backend lammps
            if self.cp2k is None:
                raise ValueError(
                    "autotune.production.dft_opt.enabled=true requires a 'cp2k:' block "
                    "(used for DFT cell optimisation), even when engine='lammps'"
                )

            # refinement relative convergence
            if not bool(getattr(prod, "check_convergence", True)):
                raise ValueError(
                    "autotune.production.dft_opt.enabled=true requires autotune.production.check_convergence=true"
                )

            # Every accepted LAMMPS potential style is dimensional and has an
            # exact native<->canonical mapping.  DFT refinement operates on
            # canonical ASE/CP2K quantities, so no narrower unit whitelist is
            # scientifically necessary here.
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        data = yaml.safe_load(path.read_text())

        base = path.parent

        def _strip_overlap(base_dir: Path, rel: Path) -> Path:
            bparts = base_dir.parts
            rparts = rel.parts
            kmax = min(len(bparts), len(rparts))
            for k in range(kmax, 0, -1):
                if tuple(rparts[:k]) == tuple(bparts[-k:]):
                    tail = rparts[k:]
                    return Path(*tail) if len(tail) > 0 else Path(".")
            return rel

        def _resolve_with_overlap(raw_path: str) -> str:
            p = Path(raw_path).expanduser()
            if p.is_absolute():
                return str(p.resolve(strict=False))
            p2 = _strip_overlap(base, p)
            candidates = [base / p2]
            if p2 != p:
                candidates.append(base / p)
            resolved: list[Path] = []
            for candidate in candidates:
                normal = candidate.resolve(strict=False)
                if normal not in resolved:
                    resolved.append(normal)
            existing = [candidate for candidate in resolved if candidate.exists()]
            if len(existing) > 1:
                raise ValueError(
                    f"Ambiguous YAML-relative path {raw_path!r}; candidates exist at "
                    + ", ".join(str(candidate) for candidate in existing)
                )
            if existing:
                return str(existing[0])
            # Missing paths retain one deterministic YAML-relative spelling so
            # the owning field validator/execution preflight can report them.
            # Never search ancestor directories or process CWD: doing so can
            # silently bind an unrelated same-name scientific input.
            return str(resolved[0])

        s = data.get("structure", {}) if isinstance(data, dict) else {}
        if isinstance(s, dict) and "lammps_data" in s and s["lammps_data"] is not None:
            if not str(s["lammps_data"]).strip():
                raise ValueError("structure.lammps_data must be a non-empty file path")
            s["lammps_data"] = _resolve_with_overlap(str(s["lammps_data"]))

        g = s.get("generate", {}) if isinstance(s, dict) else {}
        if isinstance(g, dict) and g.get("poscar_path", None) is not None:
            if not str(g.get("poscar_path")).strip():
                raise ValueError("structure.generate.poscar_path must be a non-empty file path")
            g["poscar_path"] = _resolve_with_overlap(str(g.get("poscar_path")))

        # CP2K path-bearing inputs use the same YAML-relative resolution as
        # structures and potential files. Bare BASIS/POTENTIAL filenames are
        # deliberately left unchanged so CP2K_DATA_DIR/package lookup remains
        # available; path-qualified names become absolute and unambiguous.
        cp2k = data.get("cp2k", {}) if isinstance(data, dict) else {}
        if isinstance(cp2k, dict):
            if cp2k.get("data_dir", None) is not None:
                if not str(cp2k.get("data_dir")).strip():
                    raise ValueError("cp2k.data_dir must be omitted/null or a non-empty directory path")
                else:
                    cp2k["data_dir"] = _resolve_with_overlap(str(cp2k.get("data_dir")))
            for field_name in ("basis_set_file_name", "potential_file_name"):
                raw_value = cp2k.get(field_name, None)
                if raw_value is None:
                    continue
                raw_text = str(raw_value)
                raw_path = Path(raw_text).expanduser()
                if not raw_path.is_absolute() and ("/" in raw_text or "\\" in raw_text):
                    cp2k[field_name] = _resolve_with_overlap(raw_text)

        # potential.files: reject null entries before path rewriting. YAML
        # constructs like `files: [null]`,
        # `files: [~]`, or a stray trailing `-` parse as None and used to be
        # silently dropped here, leaving the user with a config that did not
        # match their YAML. Raise instead so the typo is fixed at the source.
        if isinstance(data, dict):
            _pot_key = "potential" if "potential" in data else ("kim" if "kim" in data else None)
            if _pot_key is not None:
                _pot_block = data.get(_pot_key)
                if isinstance(_pot_block, dict):
                    _files_block = _pot_block.get("files")
                    if isinstance(_files_block, (list, tuple)):
                        for _idx, _entry in enumerate(_files_block):
                            if _entry is None:
                                raise ValueError(
                                    f"{_pot_key}.files entry at index {_idx} is null. "
                                    "Remove the entry or provide a path; null is not "
                                    "silently dropped (this would make the loaded config "
                                    "diverge from the YAML you wrote)."
                                )

        # potential.files now uses the same overlap-stripping heuristic as
        # structure.lammps_data and structure.generate.poscar_path so YAML-relative
        # potential file lists behave consistently across fields.
        key = "potential" if isinstance(data, dict) and "potential" in data else "kim"
        pot = data.get(key, {}) if isinstance(data, dict) else {}
        if isinstance(pot, dict) and pot.get("files", None) is not None:
            files = pot.get("files")
            if isinstance(files, (list, tuple)):
                out_files: list[str] = []
                for index, rawf in enumerate(files):
                    # None entries were rejected above; reaching this branch
                    # with one would be a programming error.
                    if not str(rawf).strip():
                        raise ValueError(
                            f"{key}.files entry at index {index} must be a non-empty path"
                        )
                    out_files.append(_resolve_with_overlap(str(rawf)))
                pot["files"] = out_files

        return cls.model_validate(data)


    @model_validator(mode="after")
    def _cross_validate(self) -> "RunConfig":
        # requested structures lammps
        if self.structure.charges is not None and str(self.md.atom_style) != "charge":
            raise ValueError("structure.charges provided but md.atom_style is not 'charge'")

        # buckingham lammps species
        if self.kim is not None and self.kim.core_repulsion.enabled:
            if not isinstance(self.kim.interactions, list) or len(self.kim.interactions) == 0:
                raise ValueError(
                    "kim.core_repulsion.enabled requires kim.interactions to be a non-empty list (species ordering)"
                )
        return self
