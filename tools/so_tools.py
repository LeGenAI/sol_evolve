#!/usr/bin/env python3
"""
SO Embedding Tools for LangChain/DeepAgents Integration

These tools provide SAT-based self-orthogonal embedding capabilities
integrated with the hitl_demo.py human-in-the-loop workflow.

Tools:
1. run_rank1_elimination: Fast greedy algorithm (optimal for s_min)
2. run_so_sat_search: SAT-based search with A_d optimization
3. verify_so_embedding: Verify self-orthogonality and compute properties
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.tools import tool


ROOT_DIR = Path(__file__).resolve().parents[2]


def _format_error(header: str, exc: Exception) -> str:
    return f"{header}: {exc}\n{traceback.format_exc()}"


def _import_so_embedding():
    """Import SO embedding module."""
    import sys
    sys.path.insert(0, str(ROOT_DIR / "sol_evolve"))
    from so_embedding_sat import (
        SOEmbeddingSATEncoder,
        rank_one_elimination,
        solve_so_embedding_sat
    )
    return SOEmbeddingSATEncoder, rank_one_elimination, solve_so_embedding_sat


@tool
def run_rank1_elimination(
    matrix_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Run Rank-1 Elimination algorithm to find minimum SO embedding.

    This greedy algorithm from SO.tex Section 5 guarantees s = rank(GG^T) columns,
    which is the theoretical minimum for any SO embedding.

    Parameters
    ----------
    matrix_path : str
        Path to .npy file containing generator matrix G (k x n)
    output_dir : str, optional
        Output directory for results

    Returns
    -------
    str
        Summary including S matrix dimensions and SO code parameters
    """
    try:
        # Load generator matrix
        G = np.load(matrix_path)
        k, n = G.shape

        # Import and run
        _, rank_one_elimination, _ = _import_so_embedding()

        S = rank_one_elimination(G, verbose=False)

        # Output
        out_dir = Path(output_dir) if output_dir else Path(matrix_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        s = S.shape[1] if S.size > 0 else 0
        s_path = out_dir / f"S_rank1_{n}_{k}_s{s}.npy"
        g_ext_path = out_dir / f"G_ext_rank1_{n}_{k}_s{s}.npy"

        np.save(s_path, S)
        G_ext = np.hstack([G, S]) if S.size > 0 else G
        np.save(g_ext_path, G_ext)

        # Verify
        gram_ext = (G_ext @ G_ext.T) % 2
        is_so = np.all(gram_ext == 0)

        # Compute minimum distance
        min_dist = n + s + 1
        for msg_int in range(1, 2**k):
            msg = np.array([int(b) for b in format(msg_int, f'0{k}b')], dtype=int)
            cw = (msg @ G_ext) % 2
            w = int(np.sum(cw))
            if 0 < w < min_dist:
                min_dist = w

        lines = [
            f"Rank-1 Elimination Result",
            f"Input code: [{n}, {k}]",
            f"Gram rank: rank(GG^T) = {np.linalg.matrix_rank((G @ G.T) % 2)}",
            f"Columns added: s = {s}",
            f"Output code: [{n + s}, {k}, {min_dist}]",
            f"Self-orthogonal: {is_so}",
            f"S matrix saved: {s_path}",
            f"G_ext saved: {g_ext_path}",
        ]

        return "\n".join(lines)

    except Exception as exc:
        return _format_error("ERROR in rank1_elimination", exc)


@tool
def run_so_sat_search(
    matrix_path: str,
    s: Optional[int] = None,
    target_d: Optional[int] = None,
    timeout: int = 300,
    output_dir: Optional[str] = None,
    solver: str = "cadical",
    solver_path: Optional[str] = None,
) -> str:
    """
    Run SAT-based search for SO embedding with optional A_d minimization.

    This extends the Rank-1 algorithm by using SAT to search for S matrices
    that additionally satisfy minimum distance constraints or minimize A_d.

    Parameters
    ----------
    matrix_path : str
        Path to .npy file containing generator matrix G (k x n)
    s : int, optional
        Number of columns to add. If None, uses minimum = rank(GG^T)
    target_d : int, optional
        Target minimum distance for extended code. If set, adds constraints
        to ensure all codewords have weight >= target_d
    timeout : int
        SAT solver timeout in seconds (default: 300)
    output_dir : str, optional
        Output directory for CNF and results
    solver : str
        SAT solver type: "cadical" or "kissat"
    solver_path : str, optional
        Explicit path to solver binary

    Returns
    -------
    str
        Summary of SAT result including SO code parameters if SAT
    """
    try:
        import subprocess
        import time

        # Load generator matrix
        G = np.load(matrix_path)
        k, n = G.shape

        # Import encoder
        SOEmbeddingSATEncoder, _, _ = _import_so_embedding()

        # Setup
        out_dir = Path(output_dir) if output_dir else Path(matrix_path).parent / "so_sat"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create encoder
        encoder = SOEmbeddingSATEncoder(
            G, s=s, target_d=target_d, working_dir=str(out_dir)
        )
        encoder.encode_all_constraints()
        cnf_file = encoder.export_to_dimacs()

        # Find solver
        if solver_path is None:
            candidates = [
                ROOT_DIR / "sat_solvers" / "cadical" / "build" / "cadical",
                ROOT_DIR / "sat_solvers" / "kissat" / "build" / "kissat",
            ]
            solver_path = next((str(p) for p in candidates if p.exists()), solver)

        # Run solver
        start_time = time.time()
        try:
            result = subprocess.run(
                [solver_path, cnf_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            solve_time = time.time() - start_time
            output = result.stdout
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {timeout}s. CNF saved at {cnf_file}"

        lines = [
            f"SO SAT Search Result",
            f"Input code: [{n}, {k}]",
            f"Columns to add: s = {encoder.s}",
            f"Target distance: {target_d or 'none'}",
            f"Solver time: {solve_time:.2f}s",
        ]

        if 's SATISFIABLE' in output:
            # Parse model
            model = {}
            for line in output.split('\n'):
                if line.startswith('v '):
                    literals = line[2:].split()
                    for lit in literals:
                        if lit == '0':
                            break
                        var = abs(int(lit))
                        model[var] = int(lit) > 0

            S = encoder.parse_solution(model)
            G_ext = np.hstack([G, S])

            # Save results
            s_path = out_dir / f"S_sat_{n}_{k}_s{encoder.s}.npy"
            g_ext_path = out_dir / f"G_ext_sat_{n+encoder.s}_{k}.npy"
            np.save(s_path, S)
            np.save(g_ext_path, G_ext)

            # Verify
            gram_ext = (G_ext @ G_ext.T) % 2
            is_so = np.all(gram_ext == 0)

            # Compute weight distribution
            weight_dist = {}
            min_dist = n + encoder.s + 1
            for msg_int in range(2**k):
                msg = np.array([int(b) for b in format(msg_int, f'0{k}b')], dtype=int)
                cw = (msg @ G_ext) % 2
                w = int(np.sum(cw))
                weight_dist[w] = weight_dist.get(w, 0) + 1
                if 0 < w < min_dist:
                    min_dist = w

            A_d = weight_dist.get(min_dist, 0)

            lines.extend([
                f"Status: SAT",
                f"Output code: [{n + encoder.s}, {k}, {min_dist}]",
                f"Self-orthogonal: {is_so}",
                f"A_d (weight {min_dist}): {A_d}",
                f"Weight dist: {dict(sorted(weight_dist.items()))}",
                f"S matrix: {s_path}",
                f"G_ext: {g_ext_path}",
            ])

            # Check if self-dual
            if 2 * k == n + encoder.s:
                lines.append(f"Note: This is a self-dual [{n + encoder.s}, {k}] code!")

        elif 's UNSATISFIABLE' in output:
            lines.extend([
                f"Status: UNSAT",
                f"No SO embedding exists with s={encoder.s}" +
                (f" and d>={target_d}" if target_d else ""),
                f"Try increasing s or relaxing distance constraint.",
            ])
        else:
            lines.extend([
                f"Status: UNKNOWN",
                f"Solver output: {output[:300]}...",
            ])

        return "\n".join(lines)

    except Exception as exc:
        return _format_error("ERROR in so_sat_search", exc)


@tool
def verify_so_embedding(
    g_ext_path: str,
    compute_weight_dist: bool = True,
) -> str:
    """
    Verify that a generator matrix generates a self-orthogonal code.

    Parameters
    ----------
    g_ext_path : str
        Path to .npy file containing extended generator matrix G_ext (k x n')
    compute_weight_dist : bool
        Whether to compute full weight distribution (can be slow for large k)

    Returns
    -------
    str
        Verification summary including self-orthogonality check and weight distribution
    """
    try:
        G_ext = np.load(g_ext_path)
        k, n_prime = G_ext.shape

        # Self-orthogonality check
        gram = (G_ext @ G_ext.T) % 2
        is_so = np.all(gram == 0)
        hull_dim = k - np.linalg.matrix_rank(gram)

        lines = [
            f"SO Embedding Verification",
            f"Code: [{n_prime}, {k}]",
            f"Self-orthogonal: {is_so}",
            f"Hull dimension: {hull_dim}",
        ]

        if is_so:
            lines.append("GG^T = 0 (verified)")
        else:
            lines.append(f"GG^T has {np.sum(gram)} non-zero entries")

        # Check if self-dual
        if 2 * k == n_prime:
            lines.append(f"Potentially self-dual (n' = 2k = {n_prime})")

        # Weight distribution
        if compute_weight_dist and k <= 20:
            weight_dist = {}
            min_dist = n_prime + 1
            for msg_int in range(2**k):
                msg = np.array([int(b) for b in format(msg_int, f'0{k}b')], dtype=int)
                cw = (msg @ G_ext) % 2
                w = int(np.sum(cw))
                weight_dist[w] = weight_dist.get(w, 0) + 1
                if 0 < w < min_dist:
                    min_dist = w

            A_d = weight_dist.get(min_dist, 0)

            lines.extend([
                f"Minimum distance: {min_dist}",
                f"A_d: {A_d}",
                f"Weight distribution: {dict(sorted(weight_dist.items()))}",
            ])

            # Check weight enumerator properties
            # Self-complementary: A_w = A_{n'-w} iff all-one in code
            is_self_comp = all(
                weight_dist.get(w, 0) == weight_dist.get(n_prime - w, 0)
                for w in range(n_prime // 2 + 1)
            )
            if is_self_comp:
                lines.append("Self-complementary: YES (all-one vector in code)")
        else:
            lines.append(f"Weight distribution: skipped (k={k} > 20)")

        return "\n".join(lines)

    except Exception as exc:
        return _format_error("ERROR in verify_so_embedding", exc)


@tool
def create_hamming_generator(
    r: int,
    output_path: Optional[str] = None,
) -> str:
    """
    Create generator matrix for Hamming [2^r - 1, 2^r - 1 - r, 3] code.

    From SO.tex: Hamming codes are important test cases for SO embedding.
    The shortest SO embedding adds k - r = n - 2r columns.

    Parameters
    ----------
    r : int
        Hamming parameter (r >= 3). Code has n = 2^r - 1, k = n - r.
    output_path : str, optional
        Path to save generator matrix. If None, uses ./hamming_r_G.npy

    Returns
    -------
    str
        Summary with code parameters and file path
    """
    try:
        if r < 3:
            return f"ERROR: r must be >= 3 (got {r})"

        n = (1 << r) - 1
        k = n - r

        # Construct parity check matrix H (r x n)
        # Columns are all non-zero r-bit vectors
        H = np.array([
            [int(b) for b in format(i, f'0{r}b')]
            for i in range(1, n + 1)
        ], dtype=int).T

        # Generator matrix in systematic form: G = [I_k | P]
        # where H = [A | I_r] and G = [I_k | A^T]
        # Rearrange H to put identity at end
        # Find columns that form identity
        identity_cols = []
        for bit_pos in range(r):
            for col in range(n):
                if H[:, col].tolist() == [1 if i == bit_pos else 0 for i in range(r)]:
                    identity_cols.append(col)
                    break

        # Remaining columns form A (or P^T)
        data_cols = [c for c in range(n) if c not in identity_cols]

        # Systematic form: G = [I_k | P] where P = A^T
        # A is r x k, so P is k x r
        A = H[:, data_cols]  # r x k
        P = A.T  # k x r

        G = np.zeros((k, n), dtype=int)
        G[:, :k] = np.eye(k, dtype=int)
        G[:, k:] = P

        # Verify: GH^T = 0
        H_reordered = np.zeros((r, n), dtype=int)
        H_reordered[:, :k] = A
        H_reordered[:, k:] = np.eye(r, dtype=int)
        check = (G @ H_reordered.T) % 2
        is_valid = np.all(check == 0)

        # Save
        out_path = Path(output_path) if output_path else Path(f"hamming_{r}_G.npy")
        np.save(out_path, G)

        # Compute Gram matrix rank
        gram = (G @ G.T) % 2
        gram_rank = np.linalg.matrix_rank(gram)
        s_min = gram_rank

        lines = [
            f"Hamming Code H_{r}",
            f"Parameters: [{n}, {k}, 3]",
            f"Generator matrix: {k} x {n}",
            f"Valid (GH^T = 0): {is_valid}",
            f"",
            f"SO Embedding Info (from SO.tex Theorem 3.1):",
            f"  Gram rank = r = {gram_rank}",
            f"  Minimum columns to add: s = k - r = {k - r}",
            f"  Resulting code: [{n + k - r}, {k}] = [{2*(n-r)}, {n-r}]",
            f"  This is a self-dual code!",
            f"",
            f"Saved to: {out_path}",
        ]

        return "\n".join(lines)

    except Exception as exc:
        return _format_error("ERROR in create_hamming_generator", exc)


# Convenience function for testing
def _test_tools():
    """Test the SO tools locally."""
    print("Testing SO tools...")

    # Create test matrix
    np.random.seed(42)
    k, n = 4, 7
    G = np.zeros((k, n), dtype=int)
    G[:, :k] = np.eye(k, dtype=int)
    G[:, k:] = np.random.randint(0, 2, (k, n - k))

    test_path = ROOT_DIR / "sol_evolve" / "test_G.npy"
    np.save(test_path, G)

    # Test rank-1 elimination
    result = run_rank1_elimination.invoke({"matrix_path": str(test_path)})
    print(result)

    # Cleanup
    test_path.unlink()


if __name__ == "__main__":
    _test_tools()
