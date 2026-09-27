"""
STEP T4 - INTERNAL SUBMISSION VALIDATOR (full test scale)

Checks output/matching_results.tsv and output/candidate_pairs.tsv
against the format rules in the project brief:

    - both files: exactly one row per test S1, no duplicates, no
      unexpected ids
    - matched_entity_ids / candidate_entity_ids: only S2-/S3- prefixed
      ids, no duplicates within a row, no malformed cells
      ("nan"/"None"/"[]"), only ids that actually exist in test
      source2/source3
    - every predicted match is contained in that S1's candidate list

This reuses the exact same check function (submission_checks.py) that
run_test_inference.py's automatic smoke-test check uses, so the two
can never silently drift apart.

No utils/validate_submission.py (an official Amazon validator) was
found anywhere in this repository at the time this script was written.
This script is NOT a substitute for the official validator - it is an
internal sanity check only, and does not claim otherwise. If an
official validator is added to the repo later under
utils/validate_submission.py, re-run it before submitting; this file
does not know how to invoke it.

USAGE:
    python validate_submission.py
    python validate_submission.py --data-dir data/test --output-dir output
"""

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submission_checks import validate_outputs, print_report  # noqa: E402


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=os.path.join(BASE_DIR, "data", "test"),
        help="Directory containing test_source1/2/3.tsv (default: data/test)",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(BASE_DIR, "output"),
        help="Directory containing matching_results.tsv / "
             "candidate_pairs.tsv (default: output)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP T4 - INTERNAL SUBMISSION VALIDATOR")
    print("=" * 70)

    s1_path = os.path.join(args.data_dir, "test_source1.tsv")
    s2_path = os.path.join(args.data_dir, "test_source2.tsv")
    s3_path = os.path.join(args.data_dir, "test_source3.tsv")
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidates_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    for p in (s1_path, s2_path, s3_path, matching_path, candidates_path):
        if not os.path.exists(p):
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"\nLoading expected S1 ids from: {s1_path}", flush=True)
    test_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    expected_s1_ids = set(test_s1["entity_id"])
    print(f"Expected test S1 entities: {len(expected_s1_ids):,}", flush=True)

    print(f"\nLoading valid candidate-id universe from S2/S3 "
          f"entity_id columns...", flush=True)
    s2_ids = pd.read_csv(
        s2_path, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"]
    )["entity_id"]
    s3_ids = pd.read_csv(
        s3_path, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"]
    )["entity_id"]
    valid_candidate_ids = set(s2_ids) | set(s3_ids)
    print(f"Valid S2+S3 id universe: {len(valid_candidate_ids):,}", flush=True)

    print(f"\nValidating:\n  {matching_path}\n  {candidates_path}", flush=True)
    passed, summary, problems = validate_outputs(
        matching_path, candidates_path, expected_s1_ids, valid_candidate_ids
    )

    print_report(summary, problems)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
