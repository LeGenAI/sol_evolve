from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

EvidenceStatus = Literal["OK", "SKIPPED", "UNAVAILABLE", "ERROR", "SAT", "UNSAT", "TIMEOUT", "UNKNOWN"]
Verdict = Literal["PASS", "PARTIAL", "SKIPPED", "INSUFFICIENT_ARTIFACT", "FAIL"]
CoordinatorAction = Literal["continue", "stop"]
CoordinatorMode = Literal["exploration", "exploitation", "repair", "finalize"]
HybridGAMode = Literal["off", "archived", "replay", "live", "frontier_repair"]
HybridGAMethod = Literal["hybrid_sat_ga"]
EvolverActionName = Literal[
    "query_codetables",
    "run_claim_proof",
    "construction_search",
    "propose_repair",
    "request_reflection",
    "stop",
]
InnerProductName = Literal["dot", "hermitian"]
CodetablesFetchStatus = Literal["LIVE", "CACHE", "STALE", "UNAVAILABLE", "ERROR"]
ReflectorActionName = Literal[
    "restart",
    "encoding_switch",
    "blocking_constraints",
    "sat_repair",
    "artifact_repair",
    "construction_search",
    "mutation_operator_adjustment",
    "finalize",
]


class LucasTarget(BaseModel):
    """Lucas-cube-specific target semantics that do not fit linear-code q/n/k/d names."""

    n: int = Field(description="Lucas cube length n in Lambda_n(1^s).", ge=1)
    s: int = Field(description="Forbidden circular run length s in Lambda_n(1^s); this is not a center count.", ge=1)
    expected_vertex_count: int = Field(description="Expected number of vertices in Lambda_n(1^s).", ge=1)
    expected_center_count: int = Field(description="Expected number of centers in the perfect partition.", ge=1)
    expected_ball_size_distribution: dict[int, int] = Field(
        description="Expected distribution mapping closed-neighborhood ball size to number of centers."
    )


class PaperTarget(BaseModel):
    """Paper-facing target specification for one reproducibility claim."""

    claim_id: str | None = Field(default=None, description="Stable paper claim id, if this target comes from the claim registry.")
    q: int = Field(description="Field order q for the target code.", ge=2)
    n: int = Field(description="Target output code length n.", ge=1)
    k: int = Field(
        description=(
            "Target code dimension k for linear-code claims. For Lucas-cube claims this mirrors "
            "lucas.expected_center_count for backward-compatible summaries; use the lucas object for domain semantics."
        ),
        ge=1,
    )
    d: int = Field(description="Target minimum distance d.", ge=1)
    field: str = Field(description="Human-readable field label, e.g. GF(3), GF(4), or GF(5).")
    inner_product: InnerProductName = Field(description="Orthogonality form used for the target.")
    objective: str = Field(description="Paper-level reproduction objective for this target.")
    required_checks: list[str] = Field(description="Required deterministic checks before the target can be marked PASS.")
    solver_preference: str = Field(default="cadical", description="Preferred SAT solver backend for this target.")
    source_n: int | None = Field(default=None, description="Source code length before extension, when applicable.", ge=1)
    t: int | None = Field(default=None, description="Number of appended coordinates, when applicable.", ge=0)
    lucas: LucasTarget | None = Field(
        default=None,
        description="Lucas-cube-specific target fields; present only when field='Lucas cube'.",
    )


class PaperWorkflowInput(BaseModel):
    """Canonical input to the paper-aligned SolEvolve workflow."""

    human_goal: str = Field(description="User or reviewer goal that initiated this workflow.")
    target: PaperTarget | None = Field(default=None, description="Primary target when exactly one target is selected.")
    targets: list[PaperTarget] = Field(default_factory=list, description="All target claims to reproduce in this workflow.")
    codetables_query: dict[str, int] | None = Field(
        default=None,
        description="Primary codetables.de BKLC query parameters with q, n, and k, when singular.",
    )
    solver_budget_sec: int = Field(description="Per-SAT-obligation solver timeout in seconds.", ge=1)
    population_state: dict[str, Any] | None = Field(
        default=None,
        description="Optional paper hybrid SAT-GA population state; absent for SO claim-only reproduction.",
    )
    verifier_state: dict[str, Any] | None = Field(
        default=None,
        description="Optional prior verifier diagnostics injected into the workflow.",
    )
    hybrid_ga_policy: HybridGAPolicy | None = Field(
        default=None,
        description="Optional coordinator-gated Hybrid SAT-GA execution policy enabled by CLI.",
    )


class VerifierDiagnostics(BaseModel):
    """Deterministic verifier diagnostics reported as paper workflow output."""

    claim_id: str | None = Field(default=None, description="Claim id these diagnostics support.")
    rank: int | None = Field(default=None, description="Relevant algebraic rank diagnostic, e.g. Gram rank.", ge=0)
    self_orthogonal: bool | None = Field(default=None, description="Whether the explicit matrix is self-orthogonal.")
    d_min: int | None = Field(default=None, description="Verified minimum distance.", ge=0)
    weight_distribution: dict[int, int] | None = Field(default=None, description="Verified weight distribution.")
    low_weight_count: int | None = Field(default=None, description="Number of codewords below the requested target distance.", ge=0)
    solver_status: str | None = Field(default=None, description="Latest solver status supporting these diagnostics.")
    elapsed_ms: float | None = Field(default=None, description="Elapsed wall-clock time for the associated check.", ge=0)
    memory_mb: float | None = Field(default=None, description="Peak memory in MiB if the backend reported it.", ge=0)
    artifact_hashes: dict[str, str] = Field(default_factory=dict, description="SHA-256 hashes for referenced artifacts.")
    diversity: float | None = Field(default=None, description="Population diversity D_t in [0,1], if available.", ge=0, le=1)
    improvement_rate: float | None = Field(default=None, description="Best-fitness improvement rate over the run, if available.")
    repair_count: int | None = Field(default=None, description="Number of SAT repair events attempted in the run.", ge=0)


class ReflectorAction(BaseModel):
    """Paper vocabulary for bounded Reflector intervention actions."""

    action: ReflectorActionName = Field(description="Bounded paper action class chosen by the Reflector.")
    mode: CoordinatorMode = Field(default="finalize", description="Coordinator mode associated with this action.")
    reason: str = Field(description="Evidence-grounded reason for the action.")
    fixed_target: str | None = Field(default=None, description="Target parameters that must not drift during the intervention.")
    next_instruction: str = Field(default="", description="Instruction to the next role cycle, if continuing.")
    evidence_used: list[str] = Field(default_factory=list, description="Evidence item names used to choose this action.")


class PaperWorkflowOutput(BaseModel):
    """Canonical reviewer-facing output from a paper-aligned LangGraph run."""

    target: PaperTarget | None = Field(default=None, description="Primary target for single-claim workflows.")
    targets: list[PaperTarget] = Field(default_factory=list, description="All target claims considered by the workflow.")
    codetables_results: list[CodetablesLookupResult] = Field(default_factory=list, description="Sanitized codetables.de references.")
    solver_runs: list[dict[str, Any]] = Field(default_factory=list, description="Compact solver execution records.")
    diagnostics: list[VerifierDiagnostics] = Field(default_factory=list, description="Deterministic verifier diagnostics.")
    hybrid_ga_reports: list[HybridGAReport] = Field(
        default_factory=list,
        description="Coordinator-executed Hybrid SAT-GA reports produced during this workflow.",
    )
    reflector_action: ReflectorAction | None = Field(default=None, description="Final paper-vocabulary Reflector action.")
    decision: dict[str, Any] | None = Field(default=None, description="Validated central coordinator decision payload.")
    verdict: Verdict = Field(description="Final evidence-backed workflow verdict.")
    missing_obligations: list[str] = Field(default_factory=list, description="Required proof obligations not yet satisfied.")
    artifact_paths: list[str] = Field(default_factory=list, description="Reviewer-facing artifact paths written by the run.")


class EvidenceItem(BaseModel):
    """Structured evidence produced by deterministic repro checks or tools."""

    name: str = Field(description="Stable evidence identifier, e.g. artifact_manifest or solver_check.")
    status: EvidenceStatus = Field(description="Machine-readable evidence status.")
    command_or_function: str = Field(description="Command or function that produced this evidence.")
    summary: str = Field(description="Concise, secret-free evidence summary.")
    artifacts: list[str] = Field(default_factory=list, description="Artifact paths or checked paths related to this evidence.")
    error: str | None = Field(default=None, description="Error or skip reason, if the evidence is not OK.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured, secret-free details for trace filtering and audit.")


class ProofObligation(BaseModel):
    """A single deterministic proof obligation for a paper claim."""

    name: str = Field(description="Stable proof obligation identifier.")
    field_order: int = Field(description="Field order q used by this obligation.", ge=2)
    n: int = Field(description="Source code length before appended SO coordinates.", ge=1)
    k: int = Field(description="Code dimension.", ge=1)
    t: int = Field(description="Number of appended SO coordinates.", ge=0)
    d: int = Field(description="Minimum-distance target checked by this obligation.", ge=1)
    inner_product: InnerProductName = Field(description="Inner product used for the self-orthogonality check.")
    required_checks: list[str] = Field(description="Deterministic checks required for this obligation.")
    solver_preference: str = Field(default="cadical", description="Preferred solver backend for SAT obligations.")


class EvolverAction(BaseModel):
    """Schema-gated action proposal emitted by the LLM Evolver."""

    action: EvolverActionName = Field(description="Whitelisted action for the deterministic coordinator to execute.")
    claim_id: str | None = Field(default=None, description="Paper claim id targeted by this action, if applicable.")
    method: HybridGAMethod | None = Field(
        default=None,
        description=(
            "Optional bounded construction method requested by the Evolver. "
            "hybrid_sat_ga is valid only with action='construction_search' and "
            "a registered Hybrid SAT-GA claim id."
        ),
    )
    mode: HybridGAMode | None = Field(
        default=None,
        description=(
            "Optional Hybrid SAT-GA execution mode when method='hybrid_sat_ga'. "
            "This is a request only; the deterministic coordinator must reject it unless it is null "
            "or exactly matches the CLI-owned HybridGAPolicy.mode."
        ),
    )
    reason: str = Field(description="Short rationale grounded in deterministic evidence or prior role outputs.")
    timeout_sec: int | None = Field(default=None, description="Optional timeout in seconds for solver-backed actions.", ge=1)
    evidence_used: list[str] = Field(default_factory=list, description="Evidence item names used to choose this action.")

    @model_validator(mode="after")
    def _validate_hybrid_method_surface(self) -> "EvolverAction":
        if self.method == "hybrid_sat_ga":
            if self.action != "construction_search":
                raise ValueError("method='hybrid_sat_ga' is only valid with action='construction_search'.")
            if self.claim_id != "binary_22_11_7_hybrid_ga":
                raise ValueError(
                    "method='hybrid_sat_ga' is only registered for claim_id='binary_22_11_7_hybrid_ga'."
                )
        elif self.mode is not None:
            raise ValueError("mode is only valid when method='hybrid_sat_ga'.")
        return self


class HybridGAPolicy(BaseModel):
    """Coordinator-owned policy for optional Hybrid SAT-GA execution."""

    enabled: bool = Field(description="Whether the Hybrid SAT-GA branch is enabled for this run.")
    mode: HybridGAMode = Field(default="off", description="Execution mode for the Hybrid SAT-GA branch.")
    seed: int = Field(default=0, description="Deterministic random seed used for replay/live population operations.", ge=0)
    population: int = Field(default=100, description="Population size for replay/live GA runs.", ge=1)
    generations: int = Field(default=100, description="Maximum number of GA generations.", ge=0)
    repair_interval: int = Field(default=50, description="SAT repair interval in generations; 0 disables repair.", ge=0)
    timeout_sec: int = Field(default=300, description="Timeout in seconds for each SAT-backed repair or live seed attempt.", ge=1)
    solver_preference: str = Field(default="cadical", description="Preferred SAT solver backend for repair/live modes.")
    target_distance: int = Field(default=7, description="Target minimum distance for frontier-repair workflows.", ge=1)
    frontier_distance: int | None = Field(
        default=None,
        description="Frontier seed minimum distance; defaults to target_distance - 1.",
        ge=1,
    )
    frontier_seed_count: int = Field(default=20, description="Number of frontier seeds requested from SAT.", ge=1)
    frontier_target_timeout_sec: int = Field(default=60, description="Timeout for the direct target-distance SAT attempt.", ge=1)
    frontier_seed_timeout_sec: int = Field(default=30, description="Timeout per frontier seed SAT attempt.", ge=1)
    repair_strategy: str = Field(
        default="low_weight_support_mask",
        description="Coordinator-owned SAT repair strategy for frontier-repair mode.",
    )

    @model_validator(mode="after")
    def _default_frontier_distance(self) -> "HybridGAPolicy":
        if self.frontier_distance is None:
            self.frontier_distance = max(1, self.target_distance - 1)
        return self


class HybridGARepairEvent(BaseModel):
    """One coordinator-executed SAT repair attempt inside Hybrid SAT-GA."""

    generation: int = Field(description="Generation at which the repair was attempted.", ge=0)
    candidate_id: str = Field(description="Stable candidate identifier within the run.")
    pre_repair_diagnostics: dict[str, Any] = Field(description="Compact candidate diagnostics before repair.")
    mutable_columns: list[int] = Field(description="Full generator-matrix column indices left mutable in the SAT repair CNF.")
    solver_status: str = Field(description="SAT solver status for this repair attempt.")
    elapsed_ms: float = Field(description="Elapsed wall-clock time for the repair attempt in milliseconds.", ge=0)
    post_repair_diagnostics: dict[str, Any] | None = Field(
        default=None,
        description="Compact candidate diagnostics after repair, when a repaired matrix was found.",
    )
    post_repair_matrix: list[list[int]] | None = Field(
        default=None,
        exclude=True,
        description="Internal repaired matrix for final verification; excluded from reports and traces.",
    )


class HybridGAReport(BaseModel):
    """Reviewer-facing report for one Hybrid SAT-GA execution."""

    claim_id: str = Field(description="Hybrid SAT-GA claim id.")
    mode: HybridGAMode = Field(description="Execution mode used for this report.")
    config: dict[str, Any] = Field(description="Secret-free execution configuration.")
    seed_status: str = Field(description="How the initial population or archived witness was obtained.")
    seed_bank: dict[str, Any] | None = Field(
        default=None,
        description="Compact frontier seed-bank summary without raw matrices.",
    )
    frontier_solver_runs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Compact direct-target and frontier-seed SAT run records.",
    )
    generation_summaries: list[dict[str, Any]] = Field(default_factory=list, description="Compact per-generation metrics.")
    repair_events: list[HybridGARepairEvent] = Field(default_factory=list, description="SAT repair events attempted.")
    solver_runs: list[dict[str, Any]] = Field(default_factory=list, description="Compact solver execution records.")
    final_diagnostics: dict[str, Any] = Field(default_factory=dict, description="Final deterministic verifier diagnostics.")
    verdict: Verdict = Field(description="Evidence-backed Hybrid SAT-GA verdict.")
    missing_obligations: list[str] = Field(default_factory=list, description="Missing obligations preventing PASS.")
    artifact_paths: list[str] = Field(default_factory=list, description="Reviewer-facing report/artifact paths.")


class CodetablesLookupResult(BaseModel):
    """Compact, cacheable codetables.de lookup result without raw HTML."""

    q: int = Field(description="Field order q for the codetables lookup.", ge=2)
    n: int = Field(description="Linear code length n.", ge=1)
    k: int = Field(description="Linear code dimension k.", ge=1)
    url: str = Field(description="codetables.de BKLC URL queried for these parameters.")
    fetch_status: CodetablesFetchStatus = Field(description="Whether the data came from live fetch, cache, stale fallback, or failed.")
    lower_bound: int | None = Field(default=None, description="Parsed lower bound on d when available.")
    upper_bound: int | None = Field(default=None, description="Parsed upper bound on d when available.")
    construction_d: int | None = Field(default=None, description="Parsed construction distance from a concrete [n,k,d] entry when available.")
    content_sha256: str | None = Field(default=None, description="SHA-256 hash of fetched or cached HTML content.")
    fetched_at: str | None = Field(default=None, description="UTC timestamp for live fetched data.")
    cache_path: str | None = Field(default=None, description="Local JSON cache artifact path.")
    error: str | None = Field(default=None, description="Fetch or parse error, if any.")


class ReflectorMetrics(BaseModel):
    """Quantitative signals used by the deterministic Reflector threshold policy."""

    target_parameters: str | None = Field(default=None, description="Target code parameters under review.")
    d_min: int | None = Field(default=None, description="Current verified minimum distance, if known.")
    low_weight_count: int | None = Field(default=None, description="Count of codewords below target distance, if known.")
    weight_distribution: dict[int, int] | None = Field(default=None, description="Full or truncated weight distribution.")
    rank: int | None = Field(default=None, description="Relevant rank diagnostic, e.g. Gram rank.")
    diversity: float | None = Field(default=None, description="Population diversity D_t in [0,1], if available.", ge=0, le=1)
    improvement_rate: float | None = Field(default=None, description="Improvement rate rho_t over the configured window.")
    stagnation_length: int = Field(default=0, description="Number of consecutive generations below improvement threshold.", ge=0)
    solver_status: str | None = Field(default=None, description="Latest solver status.")
    timeout: bool = Field(default=False, description="Whether the latest solver or repair attempt timed out.")
    memory_mb: float | None = Field(default=None, description="Peak memory usage in MiB, if available.", ge=0)
    best_candidate_id: str | None = Field(default=None, description="Identifier for the current best candidate.")


class CoordinatorDecision(BaseModel):
    """Validated central coordinator decision for routing and final reporting."""

    decision: CoordinatorAction = Field(description="Whether the graph should continue another role cycle or stop.")
    mode: CoordinatorMode = Field(default="finalize", description="Coordinator mode chosen from threshold policy.")
    verdict: Verdict = Field(description="Evidence-backed reproduction verdict.")
    reason: str = Field(description="Short reason grounded in deterministic evidence.")
    next_instruction: str = Field(default="", description="Instruction for the next Generator cycle when decision is continue.")
    evidence_used: list[str] = Field(default_factory=list, description="Evidence item names used to justify the decision.")
    thresholds_applied: dict[str, Any] = Field(default_factory=dict, description="Reflector threshold values and signals applied.")
    reflector_action: ReflectorAction | None = Field(
        default=None,
        description="Paper-vocabulary Reflector action selected for the next bounded intervention or finalization.",
    )


def default_coordinator_decision(evidence: list[EvidenceItem] | None = None) -> CoordinatorDecision:
    """Return the initial decision before any reflector output exists."""
    evidence_names = [item.name for item in evidence or []]
    return CoordinatorDecision(
        decision="continue",
        mode="exploration",
        verdict="PARTIAL",
        reason="Deterministic repro checks collected; role review has not completed yet.",
        next_instruction="Use deterministic evidence to produce an evidence-backed reproduction verdict.",
        evidence_used=evidence_names,
        reflector_action=ReflectorAction(
            action="encoding_switch",
            mode="exploration",
            reason="Initial coordinator state before Reflector review.",
            next_instruction="Use deterministic evidence to produce an evidence-backed reproduction verdict.",
            evidence_used=evidence_names,
        ),
    )


def fallback_coordinator_decision(*, reason: str, evidence: list[EvidenceItem] | None = None) -> CoordinatorDecision:
    """Return a fail-closed stop decision when reflector output cannot be trusted."""
    evidence_names = [item.name for item in evidence or []]
    return CoordinatorDecision(
        decision="stop",
        mode="repair",
        verdict="PARTIAL",
        reason=reason,
        next_instruction="",
        evidence_used=evidence_names,
        reflector_action=ReflectorAction(
            action="finalize",
            mode="repair",
            reason=reason,
            next_instruction="",
            evidence_used=evidence_names,
        ),
    )
