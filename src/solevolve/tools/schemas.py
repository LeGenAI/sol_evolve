from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SolverName = Literal["cadical", "kissat", "hcadsbva", "hcadpsbva"]
SolverPreference = Literal["cadical", "kissat"]


class ArtifactCheckArgs(BaseModel):
    manifest_path: str = Field(
        default="artifacts/manifest.json",
        description="Path to the public artifact manifest JSON to validate.",
    )


class SolverCheckArgs(BaseModel):
    root: str = Field(default=".", description="Repository root used to check common local solver build paths.")
    preferred_solver: SolverPreference = Field(
        default="cadical",
        description="Preferred SAT solver checked first; defaults to CaDiCaL.",
    )


class CodetablesLookupArgs(BaseModel):
    q: int = Field(description="Field order q for a codetables.de BKLC lookup.", ge=2, le=9)
    n: int = Field(description="Linear code length n for the BKLC lookup.", ge=1)
    k: int = Field(description="Linear code dimension k for the BKLC lookup.", ge=1)
    artifact_dir: str = Field(default="artifacts", description="Directory used for sanitized codetables cache artifacts.")
    timeout: int = Field(default=15, description="Network timeout in seconds for live codetables.de lookup.", ge=1, le=120)
    use_cache: bool = Field(default=True, description="Use cached sanitized lookup data before live network fetch.")


class LocateSolverArgs(BaseModel):
    name: SolverName = Field(default="cadical", description="SAT solver backend to locate.")
    hint: str | None = Field(default=None, description="Optional explicit solver binary path checked before defaults.")


class RunSolverArgs(BaseModel):
    n: int = Field(description="Binary linear code length n.", ge=1)
    k: int = Field(description="Binary linear code dimension k.", ge=1)
    d: int = Field(description="Target minimum distance d.", ge=1)
    solver: SolverName = Field(default="cadical", description="SAT solver backend to use.")
    solver_path: str | None = Field(default=None, description="Optional explicit solver binary path.")
    num_solutions: int = Field(default=1, description="Number of distinct solutions to request.", ge=1)
    timeout: int = Field(default=120, description="Per-solver timeout in seconds.", ge=1)
    workdir: str | None = Field(default=None, description="Optional output directory for CNF, solver output, and matrices.")
    systematic: bool = Field(default=True, description="Whether to use systematic generator-matrix encoding.")
    verbose: bool = Field(default=False, description="Whether to include verbose solver logs.")


class VerifyCodeArgs(BaseModel):
    matrix_path: str = Field(description="Path to a saved NumPy generator matrix (.npy).")
    n: int = Field(description="Expected binary linear code length n.", ge=1)
    k: int = Field(description="Expected binary linear code dimension k.", ge=1)
    d: int = Field(description="Target minimum distance d to verify against.", ge=1)
    systematic: bool = Field(default=True, description="Whether the matrix is expected to use systematic encoding.")


class RunPythonArgs(BaseModel):
    path: str = Field(description="Path to the local Python script to execute.")
    args: str | None = Field(default=None, description="Optional shell-style CLI arguments passed through shlex.split.")
    cwd: str | None = Field(default=None, description="Optional working directory. Defaults to the script parent directory.")
    timeout: int = Field(default=120, description="Execution timeout in seconds.", ge=1, le=3600)
    v__args: list[str] | None = Field(default=None, description="Compatibility vector of CLI args used when args is absent.")


class Rank1EliminationArgs(BaseModel):
    matrix_path: str = Field(description="Path to a binary NumPy generator matrix (.npy).")
    output_dir: str | None = Field(default=None, description="Optional directory for generated S and extended matrices.")


class SoSatSearchArgs(BaseModel):
    matrix_path: str = Field(description="Path to a binary NumPy generator matrix (.npy).")
    s: int | None = Field(default=None, description="Optional number of embedding columns to add.", ge=0)
    target_d: int | None = Field(default=None, description="Optional target minimum distance for the extended code.", ge=1)
    timeout: int = Field(default=300, description="SAT solver timeout in seconds.", ge=1)
    output_dir: str | None = Field(default=None, description="Optional output directory for CNF and generated matrices.")
    solver: SolverName = Field(default="cadical", description="SAT solver backend to use.")
    solver_path: str | None = Field(default=None, description="Optional explicit solver binary path.")


class VerifySoEmbeddingArgs(BaseModel):
    g_ext_path: str = Field(description="Path to an extended binary generator matrix (.npy).")
    compute_weight_dist: bool = Field(default=True, description="Compute full weight distribution only when k is small enough.")


class CreateHammingGeneratorArgs(BaseModel):
    r: int = Field(description="Hamming parameter r. The generated code has n=2^r-1 and k=n-r.", ge=3)
    output_path: str | None = Field(default=None, description="Optional output .npy path for the generated matrix.")


class FetchUrlArgs(BaseModel):
    url: str = Field(description="HTTP or HTTPS URL to fetch.", pattern=r"^https?://")
    limit: int = Field(default=100000, description="Maximum number of response characters to return.", ge=1, le=1000000)
