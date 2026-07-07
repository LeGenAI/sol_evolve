#!/usr/bin/env python3
"""Monte Carlo power analysis for the paired exact McNemar design (n=90).

Reproduces the power figures quoted in the manuscript's common-interface
comparison: rejection rate of the two-sided exact McNemar test at alpha=0.05
for a variant success rate p_a against the observed random-baseline rate
p_b = 7/90, under independent per-case outcomes. Deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    trials = a_only + b_only
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, i) for i in range(0, min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def power(pa: float, pb: float, *, n: int, sims: int, alpha: float, rng: random.Random) -> float:
    rejections = 0
    for _ in range(sims):
        a_only = b_only = 0
        for _ in range(n):
            a = rng.random() < pa
            b = rng.random() < pb
            if a and not b:
                a_only += 1
            if b and not a:
                b_only += 1
        if exact_mcnemar_p(a_only, b_only) < alpha:
            rejections += 1
    return rejections / sims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=90)
    parser.add_argument("--baseline", type=float, default=7 / 90)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--variant-rates",
        default="0.10,0.15,0.178,0.20,0.228,0.25,0.30",
        help="Comma-separated variant success rates to evaluate.",
    )
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for token in args.variant_rates.split(","):
        pa = float(token)
        est = power(pa, args.baseline, n=args.n, sims=args.sims, alpha=args.alpha, rng=rng)
        rows.append(
            {
                "variant_rate": pa,
                "difference_pp": round(100 * (pa - args.baseline), 1),
                "power": est,
            }
        )
        print(f"pa={pa:.3f} (diff={rows[-1]['difference_pp']:+.1f}pp): power={est:.3f}")

    if args.out_json:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": "Monte Carlo power of two-sided exact McNemar at alpha vs the random baseline, independent per-case outcomes.",
            "args": vars(args),
            "rows": rows,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
