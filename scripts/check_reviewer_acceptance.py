"""Offline regression checks for the public reviewer interface."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from solevolve.claim_registry import get_claim
from solevolve.so_claim_repro import verify_claim_matrix


ROOT = Path(__file__).resolve().parents[1]


class ReviewerAcceptanceTests(unittest.TestCase):
    def test_fixed_base_witness(self):
        claim = get_claim("gf5_so_12_6_d8")
        self.assertTrue(verify_claim_matrix(claim)["matrix_ok"])
        archive = json.loads((ROOT / "artifacts/gf5_fixed_base_d9.json").read_text())
        witness = claim.model_copy(update={"matrix": archive["generator_matrix"]})
        self.assertEqual(witness.base_matrix.tolist(), claim.base_matrix.tolist())
        result = verify_claim_matrix(witness, sat_target=9)
        self.assertTrue(result["matrix_ok"])
        self.assertEqual(result["minimum_distance"], 9)
        self.assertEqual(result["weight_distribution"], {int(k): v for k, v in archive["weight_distribution"].items()})
        self.assertFalse(verify_claim_matrix(claim, sat_target=9)["matrix_ok"])

    def test_missing_manifest_fails_with_or_without_strict_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            for flags in ([], ["--require-pass"]):
                with self.subTest(flags=flags):
                    run = subprocess.run(
                        ["solevolve-reviewer-reproduce", "--paper-claim-id", "binary_ad_43_10_16",
                         "--manifest", str(Path(directory) / "missing.json"), "--output-dir", directory, *flags],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(run.returncode, 1, run.stderr)
                    self.assertEqual(json.loads(run.stdout)["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
