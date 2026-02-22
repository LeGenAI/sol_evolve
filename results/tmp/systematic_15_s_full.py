#!/usr/bin/env python3
"""
Systematic FULL SAT search for Lambda_15(1^s) - ignores greedy feasibility check.

We know s=11, s=12 are SAT even though greedy says otherwise.
Run full SAT to find the actual boundary.

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import Counter, defaultdict
import time
import subprocess
import sys

def has_circular_consecutive_ones(n_val, bits, s):
    doubled = (n_val << bits) | n_val
    mask = (1 << s) - 1
    for i in range(bits):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def generate_lambda_vertices(n, s):
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def get_ball(center, n, vertices_set):
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

def create_sat_encoding(n, s):
    vertices = generate_lambda_vertices(n, s)
    if not vertices:
        return None, "No vertices"

    vertices_set = set(vertices)

    balls = {}
    ball_membership = defaultdict(list)
    for v in vertices:
        ball = get_ball(v, n, vertices_set)
        balls[v] = ball
        for u in ball:
            ball_membership[u].append(v)

    candidates = vertices
    var_map = {c: i + 1 for i, c in enumerate(candidates)}
    num_vars = len(candidates)
    aux_counter = [num_vars + 1]

    clauses = []

    def new_aux():
        var = aux_counter[0]
        aux_counter[0] += 1
        return var

    def at_most_one_sequential(lits):
        if len(lits) <= 1:
            return
        if len(lits) <= 4:
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    clauses.append([-lits[i], -lits[j]])
            return

        n_lits = len(lits)
        s_vars = [new_aux() for _ in range(n_lits - 1)]

        clauses.append([-lits[0], s_vars[0]])
        for i in range(1, n_lits - 1):
            clauses.append([-s_vars[i-1], s_vars[i]])
            clauses.append([-lits[i], s_vars[i]])
        clauses.append([-s_vars[-1], -lits[-1]])

        for i in range(n_lits - 1):
            clauses.append([-lits[i+1], -s_vars[i]])

    # Covering constraints
    for v in vertices:
        covering = ball_membership[v]
        lits = [var_map[c] for c in covering]
        clauses.append(lits)  # At least one
        at_most_one_sequential(lits)  # At most one

    # Packing constraints
    for i, c1 in enumerate(candidates):
        v1 = var_map[c1]
        for c2 in candidates[i+1:]:
            if hamming_distance(c1, c2) < 3:
                v2 = var_map[c2]
                clauses.append([-v1, -v2])

    # Symmetry breaking: vertex 0 as center if valid
    if 0 in var_map:
        clauses.append([var_map[0]])

    return {
        'vertices': vertices,
        'vertices_set': vertices_set,
        'candidates': candidates,
        'var_map': var_map,
        'num_vars': aux_counter[0] - 1,
        'clauses': clauses,
        'ball_sizes': {v: len(balls[v]) for v in vertices}
    }, None

def write_dimacs(encoder, filename):
    with open(filename, 'w') as f:
        f.write(f"p cnf {encoder['num_vars']} {len(encoder['clauses'])}\n")
        for clause in encoder['clauses']:
            f.write(" ".join(map(str, clause)) + " 0\n")

def run_solver(cnf_file, timeout=300):
    solver = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"

    try:
        proc = subprocess.run(
            [solver, cnf_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if proc.returncode == 10:
            return "SAT", proc.stdout
        elif proc.returncode == 20:
            return "UNSAT", None
        else:
            return f"UNKNOWN({proc.returncode})", None

    except subprocess.TimeoutExpired:
        return "TIMEOUT", None

def parse_solution(output, candidates, num_vars):
    centers = []
    for line in output.split('\n'):
        if line.startswith('v '):
            parts = line.split()[1:]
            for p in parts:
                if p == '0':
                    continue
                var = int(p)
                if var > 0 and var <= len(candidates):
                    centers.append(candidates[var - 1])
    return centers

def verify_solution(centers, n, s):
    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)

    for c in centers:
        if has_circular_consecutive_ones(c, n, s):
            return False, "Invalid center"

    for i, c1 in enumerate(centers):
        for c2 in centers[i+1:]:
            if hamming_distance(c1, c2) < 3:
                return False, "Centers too close"

    covered = set()
    for c in centers:
        ball = get_ball(c, n, vertices_set)
        for v in ball:
            if v in covered:
                return False, "Overlap"
            covered.add(v)

    if covered != vertices_set:
        return False, f"Missing {len(vertices_set - covered)} vertices"

    return True, "OK"

def main():
    n = 15

    start_s = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    end_s = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    results = []

    print("=" * 70)
    print(f"Full SAT Search: Lambda_{n}(1^s) for s = {start_s} to {end_s}")
    print(f"Timeout: {timeout}s per case")
    print("=" * 70)

    for s in range(start_s, end_s + 1):
        print(f"\n{'='*60}")
        print(f"Testing s = {s}")
        print("=" * 60)

        # Create SAT encoding
        print(f"Creating SAT encoding...")
        t0 = time.time()
        encoder, err = create_sat_encoding(n, s)
        if err:
            print(f"Error: {err}")
            results.append({'s': s, 'status': 'ERROR', 'error': err})
            continue

        num_v = len(encoder['vertices'])
        size_dist = Counter(encoder['ball_sizes'].values())
        expected = sum(1.0 / encoder['ball_sizes'][v] for v in encoder['vertices'])

        print(f"|V| = {num_v}")
        print(f"Ball sizes: {dict(size_dist)}")
        print(f"Expected centers: {expected:.1f}")
        print(f"Variables: {encoder['num_vars']}")
        print(f"Clauses: {len(encoder['clauses'])}")

        cnf_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s}.cnf"
        write_dimacs(encoder, cnf_file)
        encode_time = time.time() - t0
        print(f"Encoding time: {encode_time:.1f}s")

        # Run solver
        print(f"\nRunning CaDiCaL (timeout: {timeout}s)...")
        t1 = time.time()
        status, output = run_solver(cnf_file, timeout)
        solve_time = time.time() - t1

        print(f"Result: {status} ({solve_time:.1f}s)")

        if status == "SAT":
            centers = parse_solution(output, encoder['candidates'], len(encoder['candidates']))
            print(f"Centers found: {len(centers)}")

            ok, msg = verify_solution(centers, n, s)
            print(f"Verification: {msg}")

            if ok:
                np.save(f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s}_centers.npy",
                        np.array(centers))
                print(f"Solution saved!")

            results.append({
                's': s,
                'status': 'SAT',
                'vertices': num_v,
                'centers': len(centers),
                'expected': expected,
                'time': solve_time,
                'verified': ok
            })

            # Quick break on SAT to see if we can find the boundary
            # (comment out if you want to run all)
            # continue

        else:
            results.append({
                's': s,
                'status': status,
                'vertices': num_v,
                'expected': expected,
                'time': solve_time if status != "TIMEOUT" else timeout
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'s':>3} | {'|V|':>8} | {'Expected':>10} | {'Status':>12} | {'Centers':>8} | {'Time':>8}")
    print("-" * 70)
    for r in results:
        centers = r.get('centers', '-')
        exp = f"{r.get('expected', 0):.1f}"
        time_str = f"{r.get('time', 0):.1f}s"
        print(f"{r['s']:>3} | {r.get('vertices', '-'):>8} | {exp:>10} | {r['status']:>12} | {str(centers):>8} | {time_str:>8}")

    # Save summary
    with open("/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_full_results.txt", 'w') as f:
        f.write("Full SAT Search Results: Lambda_15(1^s)\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"s={r['s']}: {r['status']}\n")
            for k, v in r.items():
                if k != 's' and k != 'status':
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
