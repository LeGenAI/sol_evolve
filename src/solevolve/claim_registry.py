from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .finite_fields import field_symbol, matrix_from_symbols

InnerProduct = Literal["dot", "hermitian"]


class ClaimSpec(BaseModel):
    """Machine-readable paper claim used by deterministic reproduction runners."""

    claim_id: str = Field(description="Stable identifier for this paper result claim.")
    title: str = Field(description="Human-readable paper result title.")
    field_order: int = Field(description="Field order q for GF(q).")
    start_n: int = Field(description="Length of the source code before appended SO columns.")
    k: int = Field(description="Code dimension.")
    t: int = Field(description="Number of appended self-orthogonal embedding columns.")
    d: int = Field(description="Claimed minimum distance for the extended code.")
    inner_product: InnerProduct = Field(description="Inner product used for self-orthogonality.")
    matrix: list[list[int]] = Field(description="Explicit extended generator matrix in integer field representation.")
    expected_weight_distribution: dict[int, int] = Field(description="Expected full weight distribution.")
    grassl_bound: str | None = Field(default=None, description="Paper-reported Grassl/codetables bound.")
    require_optimality_unsat: bool = Field(
        default=False,
        description="Whether d+1 UNSAT is required for PASS rather than optional extra evidence.",
    )
    solver_preference: str = Field(default="cadical", description="Preferred SAT solver backend.")

    @property
    def n_prime(self) -> int:
        return self.start_n + self.t

    @property
    def base_matrix(self) -> np.ndarray:
        return np.array([row[: self.start_n] for row in self.matrix], dtype=int)

    @property
    def extension_matrix(self) -> np.ndarray:
        return np.array([row[self.start_n :] for row in self.matrix], dtype=int)

    @property
    def extended_matrix(self) -> np.ndarray:
        return np.array(self.matrix, dtype=int)

    @property
    def hermitian(self) -> bool:
        return self.inner_product == "hermitian"


TERNARY_BCH_S = [
    [0, 0, 0, 1, 2, 1, 1],
    [0, 0, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1],
    [2, 1, 0, 0, 2, 0, 1],
    [1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 2, 2],
    [1, 2, 0, 1, 2, 0, 0],
]


def ternary_bch_base_matrix() -> list[list[int]]:
    g = [1, 0, 2, 0, 2, 2, 1]
    rows: list[list[int]] = []
    for shift in range(7):
        row = [0] * 13
        for offset, value in enumerate(g):
            row[shift + offset] = value
        rows.append(row)
    return rows


TERNARY_BCH_EXTENDED = [
    base + extension
    for base, extension in zip(ternary_bch_base_matrix(), TERNARY_BCH_S, strict=True)
]


GF4_EXTENDED = matrix_from_symbols(
    4,
    [
        [0, "w^2", "w", 1, 1, "w^2", 0, "w", 0, 0, 0, 1, 1, 1, "w^2"],
        ["w", "w^2", "w", "w^2", "w", "w^2", "w", 0, "w^2", 1, 1, 0, 1, 0, "w^2"],
        ["w", 1, 0, "w^2", "w^2", "w", 1, "w^2", "w", 1, "w^2", "w", 0, 0, "w^2"],
        [1, 0, 0, "w", "w^2", 0, "w^2", "w^2", 1, "w", 1, "w", 0, 0, "w^2"],
        [0, "w^2", "w", 1, 0, "w^2", 1, "w^2", "w", "w^2", "w", 0, 0, 0, 1],
    ],
).tolist()


GF5_EXTENDED = [
    [0, 3, 2, 0, 4, 1, 1, 0, 1, 0, 1, 4, 1, 2, 0, 0, 0, 1],
    [2, 4, 2, 1, 3, 4, 4, 4, 0, 2, 1, 1, 1, 4, 2, 1, 0, 0],
    [1, 4, 3, 1, 2, 3, 0, 3, 2, 4, 3, 1, 0, 0, 4, 0, 4, 2],
    [4, 3, 1, 2, 3, 1, 1, 0, 0, 2, 0, 2, 1, 3, 4, 2, 1, 0],
    [2, 0, 1, 0, 0, 2, 0, 3, 1, 4, 0, 3, 3, 4, 3, 3, 2, 2],
    [2, 4, 3, 4, 2, 1, 0, 4, 1, 3, 2, 1, 0, 4, 3, 3, 4, 3],
]


CLAIMS: dict[str, ClaimSpec] = {
    "ternary_bch_d9": ClaimSpec(
        claim_id="ternary_bch_d9",
        title="Ternary BCH [13,7,5] SO embedding with [20,7,9]",
        field_order=3,
        start_n=13,
        k=7,
        t=7,
        d=9,
        inner_product="dot",
        matrix=TERNARY_BCH_EXTENDED,
        expected_weight_distribution={0: 1, 9: 166, 12: 972, 15: 954, 18: 94},
        grassl_bound="9",
        require_optimality_unsat=True,
    ),
    "gf4_hermitian_10_5_d6": ClaimSpec(
        claim_id="gf4_hermitian_10_5_d6",
        title="GF(4) Hermitian [10,5] SO embedding with [15,5,6]",
        field_order=4,
        start_n=10,
        k=5,
        t=5,
        d=6,
        inner_product="hermitian",
        matrix=GF4_EXTENDED,
        expected_weight_distribution={0: 1, 6: 12, 8: 69, 10: 372, 12: 402, 14: 168},
        grassl_bound="8",
        require_optimality_unsat=False,
    ),
    "gf5_so_12_6_d8": ClaimSpec(
        claim_id="gf5_so_12_6_d8",
        title="GF(5) dot-product [12,6] SO embedding with [18,6,8]",
        field_order=5,
        start_n=12,
        k=6,
        t=6,
        d=8,
        inner_product="dot",
        matrix=GF5_EXTENDED,
        expected_weight_distribution={
            0: 1,
            8: 4,
            9: 48,
            10: 232,
            11: 436,
            12: 1436,
            13: 2152,
            14: 3752,
            15: 3172,
            16: 2824,
            17: 1304,
            18: 264,
        },
        grassl_bound="10",
        require_optimality_unsat=False,
    ),
}

CLAIM_ALIASES = {
    "ternary_bch_13_7_5_d9": "ternary_bch_d9",
    "gf4_hermitian": "gf4_hermitian_10_5_d6",
    "gf4": "gf4_hermitian_10_5_d6",
    "gf5_so": "gf5_so_12_6_d8",
    "gf5": "gf5_so_12_6_d8",
}


def get_claim(claim_id: str) -> ClaimSpec:
    normalized = claim_id.strip().lower()
    normalized = CLAIM_ALIASES.get(normalized, normalized)
    if normalized not in CLAIMS:
        raise KeyError(f"unknown paper claim id: {claim_id}")
    return CLAIMS[normalized]


def expand_claim_ids(claim_id: str | None) -> list[str]:
    if not claim_id:
        return []
    normalized = claim_id.strip().lower()
    if normalized == "all_so_table":
        return ["ternary_bch_d9", "gf4_hermitian_10_5_d6", "gf5_so_12_6_d8"]
    return [get_claim(normalized).claim_id]


def latex_matrix(spec: ClaimSpec) -> str:
    rows = []
    for row in spec.matrix:
        left = " & ".join(field_symbol(spec.field_order, value) for value in row[: spec.start_n])
        right = " & ".join(field_symbol(spec.field_order, value) for value in row[spec.start_n :])
        rows.append(f"{left} & {right}")
    return " \\\\\n".join(rows)
