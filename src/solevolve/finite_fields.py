from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


GF4_SYMBOLS = {0: "0", 1: "1", 2: "w", 3: "w^2"}
GF4_FROM_SYMBOL = {
    "0": 0,
    "1": 1,
    "w": 2,
    "\\omega": 2,
    "omega": 2,
    "w^2": 3,
    "\\omega^2": 3,
    "omega^2": 3,
}


def field_elements(q: int) -> range:
    if q not in {2, 3, 4, 5}:
        raise ValueError(f"unsupported field order: {q}")
    return range(q)


def field_add(q: int, left: int, right: int) -> int:
    if q == 4:
        return (int(left) ^ int(right)) & 0b11
    return (int(left) + int(right)) % q


def field_neg(q: int, value: int) -> int:
    if q == 4:
        return int(value) & 0b11
    return (-int(value)) % q


def field_sub(q: int, left: int, right: int) -> int:
    return field_add(q, left, field_neg(q, right))


def field_mul(q: int, left: int, right: int) -> int:
    left = int(left)
    right = int(right)
    if q == 4:
        # Polynomial basis 1, w over GF(2), with w^2 = w + 1.
        a0, a1 = left & 1, (left >> 1) & 1
        b0, b1 = right & 1, (right >> 1) & 1
        c0 = (a0 & b0) ^ (a1 & b1)
        c1 = (a0 & b1) ^ (a1 & b0) ^ (a1 & b1)
        return c0 | (c1 << 1)
    return (left * right) % q


def field_pow(q: int, value: int, exponent: int) -> int:
    result = 1
    base = int(value)
    exp = int(exponent)
    while exp:
        if exp & 1:
            result = field_mul(q, result, base)
        base = field_mul(q, base, base)
        exp >>= 1
    return result


def field_inv(q: int, value: int) -> int:
    value = int(value)
    if value == 0:
        raise ZeroDivisionError("zero has no multiplicative inverse")
    for candidate in field_elements(q):
        if field_mul(q, value, candidate) == 1:
            return int(candidate)
    raise ArithmeticError(f"no inverse found in GF({q})")


def field_conjugate(q: int, value: int) -> int:
    if q == 4:
        return field_pow(q, value, 2)
    return int(value) % q


def field_symbol(q: int, value: int) -> str:
    if q == 4:
        return GF4_SYMBOLS[int(value) % 4]
    return str(int(value) % q)


def field_value(q: int, value: int | str) -> int:
    if isinstance(value, str):
        normalized = value.strip()
        if q == 4:
            if normalized not in GF4_FROM_SYMBOL:
                raise ValueError(f"unknown GF(4) symbol: {value}")
            return GF4_FROM_SYMBOL[normalized]
        return int(normalized) % q
    return int(value) % q


def matrix_from_symbols(q: int, rows: Iterable[Iterable[int | str]]) -> np.ndarray:
    return np.array([[field_value(q, entry) for entry in row] for row in rows], dtype=int)


def rank_field(matrix: np.ndarray, q: int) -> int:
    work = np.array(matrix, dtype=int, copy=True) % q
    rows, cols = work.shape
    rank = 0
    pivot_col = 0
    for row in range(rows):
        while pivot_col < cols and not any(int(work[r, pivot_col]) % q for r in range(row, rows)):
            pivot_col += 1
        if pivot_col >= cols:
            break
        pivot_row = next(r for r in range(row, rows) if int(work[r, pivot_col]) % q)
        if pivot_row != row:
            work[[row, pivot_row]] = work[[pivot_row, row]]
        inverse = field_inv(q, int(work[row, pivot_col]))
        work[row] = [field_mul(q, inverse, value) for value in work[row]]
        for other in range(rows):
            factor = int(work[other, pivot_col]) % q
            if other == row or factor == 0:
                continue
            work[other] = [
                field_sub(q, value, field_mul(q, factor, pivot))
                for value, pivot in zip(work[other], work[row], strict=True)
            ]
        rank += 1
        pivot_col += 1
    return rank


def inner_product(row: np.ndarray, other: np.ndarray, q: int, *, hermitian: bool = False) -> int:
    total = 0
    for left, right in zip(row, other, strict=True):
        rhs = field_conjugate(q, int(right)) if hermitian else int(right)
        total = field_add(q, total, field_mul(q, int(left), rhs))
    return total


def gram_matrix(matrix: np.ndarray, q: int, *, hermitian: bool = False) -> np.ndarray:
    rows = int(matrix.shape[0])
    gram = np.zeros((rows, rows), dtype=int)
    for i in range(rows):
        for j in range(rows):
            gram[i, j] = inner_product(matrix[i], matrix[j], q, hermitian=hermitian)
    return gram


def codeword(message: Iterable[int], generator: np.ndarray, q: int) -> np.ndarray:
    coeffs = [int(value) % q for value in message]
    result = np.zeros(generator.shape[1], dtype=int)
    for coefficient, row in zip(coeffs, generator, strict=True):
        if coefficient == 0:
            continue
        result = np.array(
            [field_add(q, current, field_mul(q, coefficient, entry)) for current, entry in zip(result, row, strict=True)],
            dtype=int,
        )
    return result


def all_messages(k: int, q: int):
    return product(field_elements(q), repeat=k)


def canonical_nonzero_messages(k: int, q: int) -> list[tuple[int, ...]]:
    representatives: list[tuple[int, ...]] = []
    for message in product(field_elements(q), repeat=k):
        if not any(message):
            continue
        first_nonzero = next(value for value in message if value)
        if first_nonzero == 1:
            representatives.append(tuple(int(value) for value in message))
    return representatives


def weight_distribution(generator: np.ndarray, q: int) -> dict[int, int]:
    k = int(generator.shape[0])
    distribution: dict[int, int] = {}
    for message in all_messages(k, q):
        word = codeword(message, generator, q)
        weight = int(np.count_nonzero(word))
        distribution[weight] = distribution.get(weight, 0) + 1
    return dict(sorted(distribution.items()))


def minimum_distance(generator: np.ndarray, q: int) -> int | None:
    distribution = weight_distribution(generator, q)
    positive = [weight for weight in distribution if weight > 0]
    return min(positive) if positive else None
