from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, List, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices


# structural metrics melt
AtomSelector = Union[int, str]


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

    @field_validator("lammps_cmd")
    @classmethod
    def _lammps_cmd_valid(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, list):
            vv = [str(x).strip() for x in v if str(x).strip() != ""]
            if len(vv) == 0:
                raise ValueError("lammps_cmd list must be non-empty")
            return vv
        if str(v).strip() == "":
            raise ValueError("lammps_cmd must be a non-empty string")
        return str(v).strip()

    @field_validator("mpi_cmd")
    @classmethod
    def _mpi_cmd_strip(cls, v: Optional[str]) -> Optional[str]:
        # failures accidental yaml
        if v is None:
            return None
        s = str(v).strip()
        return s if s != "" else None

    @field_validator("nprocs")
    @classmethod
    def _nprocs_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("nprocs must be >= 1")
        return v


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
        if isinstance(v, list):
            vv = [str(x).strip() for x in v if str(x).strip() != ""]
            if len(vv) == 0:
                raise ValueError("cp2k_cmd list must be non-empty")
            return vv
        if str(v).strip() == "":
            raise ValueError("cp2k_cmd must be a non-empty string")
        return str(v).strip()

    @field_validator("mpi_cmd")
    @classmethod
    def _mpi_cmd_strip_cp2k(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s != "" else None

    @field_validator("exec_prefix")
    @classmethod
    def _exec_prefix_valid(cls, v: Sequence[str]) -> List[str]:
        vv = [str(x) for x in list(v) if str(x).strip() != ""]
        return vv

    @field_validator("nprocs")
    @classmethod
    def _nprocs_positive_cp2k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("nprocs must be >= 1")
        return int(v)

    @field_validator("omp_num_threads")
    @classmethod
    def _omp_threads_valid(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if int(v) < 1:
            raise ValueError("omp_num_threads must be >= 1")
        return int(v)

    @field_validator("cutoff_Ry", "rel_cutoff_Ry")
    @classmethod
    def _cutoffs_pos(cls, v: float) -> float:
        x = float(v)
        if not (math.isfinite(x) and x > 0.0):
            raise ValueError("cutoffs must be finite and > 0")
        return x

    @field_validator("ngrids")
    @classmethod
    def _ngrids_valid(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("ngrids must be >= 1")
        return int(v)

    @field_validator("eps_scf")
    @classmethod
    def _eps_scf_valid(cls, v: float) -> float:
        x = float(v)
        if not (math.isfinite(x) and x > 0.0):
            raise ValueError("eps_scf must be finite and > 0")
        return x

    @field_validator("max_scf")
    @classmethod
    def _max_scf_valid(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("max_scf must be >= 1")
        return int(v)

    @field_validator("ramp_max_deltaT_K")
    @classmethod
    def _ramp_deltaT_valid(cls, v: float) -> float:
        x = float(v)
        if not (math.isfinite(x) and x > 0.0):
            raise ValueError("ramp_max_deltaT_K must be finite and > 0")
        return x

    @field_validator("ramp_max_segments")
    @classmethod
    def _ramp_segments_valid(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("ramp_max_segments must be >= 1")
        return int(v)


class CoreRepulsionConfig(BaseModel):
    """Core repulsion config."""

    enabled: bool = False
    # repulsive shifted purely
    # energy continuous minimum
    # robust relies always
    # cut sub style
    style: Literal["zbl", "lj_repulsive"] = "zbl"

    @field_validator("style")
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
    )
    @classmethod
    def _pos_float(cls, v: float) -> float:
        if v is None:
            return v
        if float(v) <= 0:
            raise ValueError("value must be > 0")
        return float(v)

    @field_validator("max_attempts", "test_run_steps", "table_points", "table_points_max", "table_verify_points")
    @classmethod
    def _pos_int(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("value must be >= 1")
        return int(v)

    @model_validator(mode="after")
    def _validate(self) -> "CoreRepulsionConfig":
        if self.r_out_max < self.r_out_min:
            raise ValueError("r_out_max must be >= r_out_min")
        if self.r_in_factor >= 1.0:
            raise ValueError("r_in_factor must be < 1 (inner cutoff must be smaller than outer)")
        if bool(self.tabulate):
            if str(self.style).strip().lower() != "zbl":
                raise ValueError("core_repulsion.tabulate currently supports only style='zbl'")
            if int(self.table_points) < 2000:
                raise ValueError("core_repulsion.table_points must be >= 2000 when tabulate=true")
            if int(self.table_points_max) < int(self.table_points):
                raise ValueError("core_repulsion.table_points_max must be >= core_repulsion.table_points when tabulate=true")
            if int(self.table_verify_points) < 2000:
                raise ValueError("core_repulsion.table_verify_points must be >= 2000 when tabulate=true")
            if str(self.table_filename).strip() == "":
                raise ValueError("core_repulsion.table_filename must be non-empty when tabulate=true")
            if not (math.isfinite(float(self.table_r_min)) and float(self.table_r_min) > 0.0):
                raise ValueError("core_repulsion.table_r_min must be finite and > 0 when tabulate=true")
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

    @model_validator(mode="after")
    def _validate(self) -> "StructureGenerateConfig":
        if self.method == "cod":
            # cod formula search
            if self.cod_id is None and (self.formula is None or str(self.formula).strip() == ""):
                raise ValueError("structure.generate.method='cod' requires 'formula' or 'cod_id'")
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
            if not Path(self.poscar_path).exists():
                raise FileNotFoundError(f"POSCAR file not found: {self.poscar_path}")
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

    @field_validator("n_formula_units")
    @classmethod
    def _nfu_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_formula_units must be >= 1")
        return v


class StructureConfig(BaseModel):
    """Structure config."""

    lammps_data: Optional[Path] = None
    generate: Optional[StructureGenerateConfig] = None

    # species assignment structures
    # mapping species charge
    charges: Optional[Dict[str, float]] = None

    @model_validator(mode="after")
    def _validate(self) -> "StructureConfig":
        if (self.lammps_data is None) == (self.generate is None):
            raise ValueError("structure must define exactly one of 'lammps_data' or 'generate'")
        if self.lammps_data is not None:
            if not self.lammps_data.exists():
                raise FileNotFoundError(f"Structure file not found: {self.lammps_data}")
        if self.charges is not None and len(self.charges) == 0:
            raise ValueError("structure.charges must be a non-empty mapping if provided")
        return self


Ensemble = Literal["npt", "nvt"]


class ThermostatConfig(BaseModel):
    # lammps currently hoover
    # nose hoover csvr
    style: Literal["nose-hoover", "csvr"] = "nose-hoover"
    tdamp: float = 100.0  # time depends lammps


class BarostatConfig(BaseModel):
    # currently pressure lammps
    # schema
    # yamls specify
    style: Literal["nose-hoover"] = "nose-hoover"
    pdamp: float = 1000.0  # time
    mode: Literal["iso"] = "iso"


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

    @field_validator("timestep")
    @classmethod
    def _dt_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timestep must be > 0")
        return v



    @field_validator("neighbor_skin", "neighbor_skin_step", "neighbor_skin_max")
    @classmethod
    def _skin_positive(cls, v: float) -> float:
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("neighbour skin parameters must be finite and > 0")
        return v

    @model_validator(mode="after")
    def _skin_consistent(self):
        if float(self.neighbor_skin_max) < float(self.neighbor_skin):
            raise ValueError("neighbor_skin_max must be >= neighbor_skin")
        return self
    @field_validator("thermo_every", "dump_every")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("frequency must be >= 1")
        return v


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

        @field_validator("nbins", "frames", "stride", "smooth")
        @classmethod
        def _pos_int(cls, v: int) -> int:
            if v < 1:
                raise ValueError("value must be >= 1")
            return v

        @field_validator("r_max")
        @classmethod
        def _rmax_pos(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("r_max must be > 0")
            return v

    gr: GrIndicatorConfig = Field(default_factory=GrIndicatorConfig)

    @model_validator(mode="after")
    def _validate(self) -> "TmScanConfig":
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

    @field_validator("warmup_start_temperature")
    @classmethod
    def _warmup_start_temperature_valid(cls, v: float) -> float:
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("warmup_start_temperature must be finite and > 0")
        return v

    @field_validator("warmup_duration_ps")
    @classmethod
    def _warmup_duration_ps_valid(cls, v: float) -> float:
        v = float(v)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("warmup_duration_ps must be finite and > 0")
        return v


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

    @field_validator("max_iter", "traj_every")
    @classmethod
    def _pos_int(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("value must be >= 1")
        return int(v)

    @model_validator(mode="after")
    def _validate(self) -> "DftOptConfig":
        # angles stability keeping
        # compatible orthorhombic analysis
        if not bool(self.keep_angles):
            raise ValueError("dft_opt.keep_angles must be true (KEEP_ANGLES is enforced)")
        return self


class ConvergenceConfig(BaseModel):
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
    stability_bootstrap: int = 200
    stability_quantile: float = 0.95

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
    """Auto cutoff config."""

    r_max: float = 8.0
    nbins: int = 400
    smooth: int = 7  # moving average preferred
    peak_search: Tuple[float, float] = (0.5, 4.0)
    min_search: Tuple[float, float] = (0.8, 6.0)
    fallback_factor: float = 1.3


class GrMetricConfig(BaseModel):
    """Gr metric config."""

    pair: Optional[Tuple[AtomSelector, AtomSelector]] = None
    r_max: float = 8.0
    nbins: int = 400
    smooth: int = 7


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
        if not (math.isfinite(float(a)) and math.isfinite(float(b)) and float(b) > float(a) >= 0.0):
            raise ValueError("sq.peak_search must be a (min,max) with 0<=min<max")
        if int(self.smooth) < 1:
            raise ValueError("sq.smooth must be >= 1")
        self.smooth = int(self.smooth)
        if self.smooth % 2 == 0:
            self.smooth += 1
        return self


class PairMetricConfig(BaseModel):
    pair: Tuple[AtomSelector, AtomSelector]
    cutoff: Optional[float] = None


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
        if not (math.isfinite(float(a)) and math.isfinite(float(b)) and 0.0 <= float(a) < float(b)):
            raise ValueError("amorphous.peak_search must satisfy 0 <= min < max")
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

    @field_validator(
        "time_average_frames",
        "time_average_stride",
        "stage_timeseries_frame_stride",
        "stage_timeseries_max_frames",
        "quench_tail_min_frames",
    )
    @classmethod
    def _pos_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("value must be >= 1")
        return v


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

    @field_validator("tdamp_factors", "pdamp_factors")
    @classmethod
    def _factors_valid(cls, v: list[float]) -> list[float]:
        if v is None or len(v) == 0:
            raise ValueError("scan factor list must be non-empty")
        vv = [float(x) for x in v]
        if any(x <= 0 for x in vv):
            raise ValueError("scan factors must be > 0")
        return vv

    @field_validator("dt_candidates")
    @classmethod
    def _dt_candidates_valid(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return None
        vv = [float(x) for x in v]
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

    random_seed: int = 12345

    @model_validator(mode="after")
    def _validate_engine_requirements(self) -> "RunConfig":
        eng = str(self.engine).strip().lower()
        if eng == "lammps":
            if self.kim is None:
                raise ValueError("engine='lammps' requires a 'kim:' or 'potential:' block")
            # lammps backend currently
            # csvr thermostatting driver
            if str(self.md.thermostat.style).strip().lower() == "csvr":
                raise ValueError("thermostat.style='csvr' is supported only for engine='cp2k'")
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

            # density convergence lammps
            # mismatches restrict refinement
            # lammps density convertible
            if eng == "lammps" and self.kim is not None:
                u = str(getattr(self.kim, "user_units", "")).strip().lower()
                if u and u not in ("metal", "real", "si", "cgs"):
                    raise ValueError(
                        "autotune.production.dft_opt is only supported for LAMMPS units styles in "
                        "{metal, real, si, cgs} (density-unit compatibility with CP2K). "
                        f"Got user_units={u!r}."
                    )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        data = yaml.safe_load(path.read_text())

        # relative path
        # specify structure lammps
        # specify structure lammps
        # yaml directory preferred
        # relative project 4x4x4
        # relative path
        # relative path
        # duplicated inside folder
        # candidate resolutions exists
        base = path.parent
        try:
            s = data.get("structure", {})
            if "lammps_data" in s and s["lammps_data"] is not None:
                raw = str(s["lammps_data"])
                p = Path(raw)
                if not p.is_absolute():
                    # overlapping duplicates yaml
                    def _strip_overlap(base_dir: Path, rel: Path) -> Path:
                        bparts = base_dir.parts
                        rparts = rel.parts
                        kmax = min(len(bparts), len(rparts))
                        for k in range(kmax, 0, -1):
                            if tuple(rparts[:k]) == tuple(bparts[-k:]):
                                tail = rparts[k:]
                                return Path(*tail) if len(tail) > 0 else Path(".")
                        return rel

                    p2 = _strip_overlap(base, p)
                    # candidate order yaml
                    candidates: list[Path] = []
                    candidates.append((base / p2))
                    if p2 != p:
                        candidates.append((base / p))
                    for parent in base.parents:
                        candidates.append(parent / p2)
                        if p2 != p:
                            candidates.append(parent / p)
                    candidates.append(Path.cwd() / p2)
                    if p2 != p:
                        candidates.append(Path.cwd() / p)

                    chosen: Optional[Path] = None
                    for c in candidates:
                        if c.exists():
                            chosen = c
                            break
                    if chosen is None:
                        # relative path
                        chosen = base / p2
                    s["lammps_data"] = str(chosen.resolve(strict=False))
        except Exception:
            # yaml parsing heuristics
            pass

        # structure generate present
        try:
            s = data.get("structure", {})
            g = s.get("generate", {}) if isinstance(s, dict) else {}
            if isinstance(g, dict) and g.get("poscar_path", None) is not None:
                raw = str(g.get("poscar_path"))
                p = Path(raw)
                if not p.is_absolute():
                    def _strip_overlap(base_dir: Path, rel: Path) -> Path:
                        bparts = base_dir.parts
                        rparts = rel.parts
                        kmax = min(len(bparts), len(rparts))
                        for k in range(kmax, 0, -1):
                            if tuple(rparts[:k]) == tuple(bparts[-k:]):
                                tail = rparts[k:]
                                return Path(*tail) if len(tail) > 0 else Path(".")
                        return rel

                    p2 = _strip_overlap(base, p)
                    candidates: list[Path] = []
                    candidates.append((base / p2))
                    if p2 != p:
                        candidates.append((base / p))
                    for parent in base.parents:
                        candidates.append(parent / p2)
                        if p2 != p:
                            candidates.append(parent / p)
                    candidates.append(Path.cwd() / p2)
                    if p2 != p:
                        candidates.append(Path.cwd() / p)

                    chosen: Optional[Path] = None
                    for c in candidates:
                        if c.exists():
                            chosen = c
                            break
                    if chosen is None:
                        chosen = base / p2
                    g["poscar_path"] = str(chosen.resolve(strict=False))
        except Exception:
            pass

        # potential kim auxiliary
        try:
            key = "potential" if isinstance(data, dict) and "potential" in data else "kim"
            pot = data.get(key, {}) if isinstance(data, dict) else {}
            if isinstance(pot, dict) and pot.get("files", None) is not None:
                files = pot.get("files")
                if isinstance(files, (list, tuple)):
                    out_files: list[str] = []
                    for rawf in files:
                        pf = Path(str(rawf))
                        if not pf.is_absolute():
                            # prefer yaml relative
                            pf2 = base / pf
                            if pf2.exists():
                                out_files.append(str(pf2.resolve(strict=False)))
                            else:
                                # fall back relative
                                out_files.append(str((Path.cwd() / pf).resolve(strict=False)))
                        else:
                            out_files.append(str(pf.resolve(strict=False)))
                    pot["files"] = out_files
        except Exception:
            pass

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

