"""
STEP T4 - INTERNAL SUBMISSION VALIDATOR

Checks matching_results.tsv against the rules in MASTER_PROMPT.txt
Section 24, before you run Amazon's official validator:

    - every test S1 appears exactly once
    - zero matches are allowed (empty matched_entity_ids)
    - multiple matches are allowed
    - no duplicate matched IDs within a row
    - only S2-/S3- prefixed IDs may appear
    - every predicted match must exist in candidate_pairs.tsv

This is a sanity check, not a replacement for the official validator.

INPUT:
    data/processed/matching_results.tsv
    data/processed/candidate_pairs.tsv
    data/test/test_source1.tsv

Exits with a non-zero status if any check fails, and prints a
line-by-line summary of what failed and how many rows were affected.
"""

import os
import sys

import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESULTS_PATH = os.path.join(
    BASE_DIR, "data", "processed", "matching_results.tsv"
)

CANDIDATE_PATH = os.path.join(
    BASE_DIR, "data", "processed", "candidate_pairs.tsv"
)

TEST_S1_PATH = os.path.join(BASE_DIR, "data", "test", "test_source1.tsv")


def main():

    print("=" * 70)
    print("STEP T4 - INTERNAL SUBMISSION VALIDATOR")
    print("=" * 70)

    problems = []

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print(f"\nLoading: {RESULTS_PATH}", flush=True)
    results = pd.read_csv(
        RESULTS_PATH, sep="\t", dtype=str, keep_default_na=False
    )

    expected_columns = ["source1_entity_id", "matched_entity_ids"]
    if list(results.columns) != expected_columns:
        problems.append(
            f"matching_results.tsv columns are {list(results.columns)}, "
            f"expected {expected_columns}"
        )

    print(f"Loading: {TEST_S1_PATH}", flush=True)
    test_s1 = pd.read_csv(
        TEST_S1_PATH, sep="\t", dtype=str, keep_default_na=False
    )
    expected_s1_ids = set(test_s1["entity_id"])

    print(f"Loading: {CANDIDATE_PATH}", flush=True)
    candidates = pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    candidate_pairs = set(
        zip(candidates["source1_entity_id"], candidates["candidate_entity_id"])
    )

    # --------------------------------------------------------
    # Check 1: every test S1 appears exactly once
    # --------------------------------------------------------

    result_ids = results["source1_entity_id"].tolist()
    result_id_set = set(result_ids)

    duplicate_count = len(result_ids) - len(result_id_set)
    if duplicate_count > 0:
        problems.append(
            f"{duplicate_count:,} duplicate source1_entity_id rows "
            f"found in matching_results.tsv"
        )

    missing = expected_s1_ids - result_id_set
    if missing:
        problems.append(
            f"{len(missing):,} test S1 entities are missing from "
            f"matching_results.tsv (e.g. {sorted(list(missing))[:5]})"
        )

    extra = result_id_set - expected_s1_ids
    if extra:
        problems.append(
            f"{len(extra):,} rows in matching_results.tsv have an "
            f"S1 id not present in test_source1.tsv "
            f"(e.g. {sorted(list(extra))[:5]})"
        )

    # --------------------------------------------------------
    # Check 2-4: matched id format, duplicates, candidate membership
    # --------------------------------------------------------

    rows_with_bad_prefix = 0
    rows_with_dup_ids = 0
    rows_with_non_candidate_match = 0
    total_matches = 0
    s1_with_matches = 0

    example_bad_prefix = None
    example_dup = None
    example_non_candidate = None

    for row in results.itertuples(index=False):

        s1_id = row.source1_entity_id
        raw = row.matched_entity_ids

        if not raw:
            continue

        ids = [x.strip() for x in raw.split(",") if x.strip()]

        if not ids:
            continue

        s1_with_matches += 1
        total_matches += len(ids)

        if len(set(ids)) != len(ids):
            rows_with_dup_ids += 1
            if example_dup is None:
                example_dup = (s1_id, raw)

        bad_prefix = [
            x for x in ids if not (x.startswith("S2-") or x.startswith("S3-"))
        ]
        if bad_prefix:
            rows_with_bad_prefix += 1
            if example_bad_prefix is None:
                example_bad_prefix = (s1_id, bad_prefix)

        non_candidate = [
            x for x in ids if (s1_id, x) not in candidate_pairs
        ]
        if non_candidate:
            rows_with_non_candidate_match += 1
            if example_non_candidate is None:
                example_non_candidate = (s1_id, non_candidate)

    if rows_with_dup_ids:
        problems.append(
            f"{rows_with_dup_ids:,} rows contain duplicate matched IDs "
            f"(e.g. {example_dup})"
        )

    if rows_with_bad_prefix:
        problems.append(
            f"{rows_with_bad_prefix:,} rows contain an ID that is not "
            f"S2-/S3- prefixed (e.g. {example_bad_prefix})"
        )

    if rows_with_non_candidate_match:
        problems.append(
            f"{rows_with_non_candidate_match:,} rows contain a matched "
            f"ID that does not exist in candidate_pairs.tsv for that "
            f"S1 (e.g. {example_non_candidate})"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Test S1 entities (expected)      : {len(expected_s1_ids):,}")
    print(f"Rows in matching_results.tsv     : {len(results):,}")
    print(f"S1 entities with >=1 match        : {s1_with_matches:,}")
    print(f"S1 entities with zero matches      : "
          f"{len(results) - s1_with_matches:,}")
    print(f"Total matched pairs                : {total_matches:,}")

    print()

    if problems:
        print(f"FAILED - {len(problems)} problem(s) found:\n")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("PASSED - matching_results.tsv looks structurally valid.")
        print("(This is an internal check only - still run the official "
              "validator before final submission.)")


if __name__ == "__main__":
    main()
