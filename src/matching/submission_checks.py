"""
SHARED SUBMISSION VALIDATION LOGIC

Both the automatic smoke-test check (run inline by run_test_inference.py
after every smoke-test run) and the standalone CLI validator
(validate_submission.py, run after the full test run) use this module,
so the two checks can never silently drift apart.

This checks structural / format correctness only:

    1. Correct headers on both files.
    2. Exactly one row per expected source1_entity_id, in both files.
    3. No duplicate source1_entity_id rows, in both files.
    4. No unexpected source1_entity_id rows.
    5. Every id in matched_entity_ids / candidate_entity_ids starts with
       "S2-" or "S3-".
    6. No duplicate ids within a single row's list.
    7. Every id in matched_entity_ids / candidate_entity_ids actually
       belongs to the supplied universe of valid test S2/S3 ids.
    8. Every predicted match is contained in that same row's candidate
       list (predicted_matches(S1) subset of candidate_entities(S1)).
    9. No malformed list cells: no literal "nan", "None", "[]", or
       stray whitespace-only entries.
    10. Empty lists are represented as an empty string, not "nan"/"None".

This is an internal sanity check, NOT the official Amazon validator.
Nothing here is claimed to be the official validator.
"""

import pandas as pd


BAD_LIST_TOKENS = {"nan", "none", "null", "[]", "na", "n/a"}


def _split_ids(raw):
    """Split a comma-separated id cell into a clean list of ids.

    Returns (ids, malformed_flag). `raw` is assumed to already be a
    string (pandas read with dtype=str, keep_default_na=False), so a
    genuinely empty cell is "" and should split to [].
    """

    if raw is None:
        return [], True

    raw_stripped = raw.strip()

    if raw_stripped == "":
        return [], False

    if raw_stripped.lower() in BAD_LIST_TOKENS:
        return [], True

    parts = [p.strip() for p in raw_stripped.split(",")]

    # Any empty part (e.g. trailing comma "A,B,") or a part that is
    # itself one of the bad tokens is malformed.
    malformed = any(p == "" or p.lower() in BAD_LIST_TOKENS for p in parts)

    ids = [p for p in parts if p != "" and p.lower() not in BAD_LIST_TOKENS]

    return ids, malformed


def _check_one_file(df, filename, id_column, expected_s1_ids, problems):
    """Checks common to both matching_results.tsv and candidate_pairs.tsv:
    headers, one-row-per-S1, no duplicates, no unexpected ids.
    Returns dict s1_id -> raw list column value for later use.
    """

    expected_columns = ["source1_entity_id", id_column]
    if list(df.columns) != expected_columns:
        problems.append(
            f"{filename}: columns are {list(df.columns)}, "
            f"expected {expected_columns}"
        )
        return {}

    ids = df["source1_entity_id"].tolist()
    id_set = set(ids)

    duplicate_count = len(ids) - len(id_set)
    if duplicate_count > 0:
        problems.append(
            f"{filename}: {duplicate_count:,} duplicate "
            f"source1_entity_id rows"
        )

    missing = expected_s1_ids - id_set
    if missing:
        example = sorted(list(missing))[:5]
        problems.append(
            f"{filename}: {len(missing):,} expected S1 entities are "
            f"missing (e.g. {example})"
        )

    extra = id_set - expected_s1_ids
    if extra:
        example = sorted(list(extra))[:5]
        problems.append(
            f"{filename}: {len(extra):,} rows have an S1 id outside "
            f"the expected set (e.g. {example})"
        )

    return dict(zip(df["source1_entity_id"], df[id_column]))


def validate_outputs(
    matching_results_path,
    candidate_pairs_path,
    expected_s1_ids,
    valid_candidate_ids=None,
):
    """Run all structural checks.

    Parameters
    ----------
    matching_results_path, candidate_pairs_path : str
        Paths to the two TSVs to check.
    expected_s1_ids : set[str]
        The exact set of source1_entity_id values that must appear,
        exactly once each, in both files.
    valid_candidate_ids : set[str] or None
        Universe of valid test S2/S3 entity ids. If None, the
        S2-/S3- prefix check still runs but membership-in-universe is
        skipped (used by the smoke test, where building the full id
        universe is not worth the extra I/O for a 1-10k row check;
        the full-run CLI validator always supplies this).

    Returns
    -------
    (passed: bool, summary: dict, problems: list[str])
    """

    problems = []

    try:
        results = pd.read_csv(
            matching_results_path, sep="\t", dtype=str, keep_default_na=False
        )
    except Exception as exc:
        return False, {}, [f"Could not read {matching_results_path}: {exc}"]

    try:
        candidates = pd.read_csv(
            candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False
        )
    except Exception as exc:
        return False, {}, [f"Could not read {candidate_pairs_path}: {exc}"]

    matches_by_s1 = _check_one_file(
        results, "matching_results.tsv", "matched_entity_ids",
        expected_s1_ids, problems,
    )
    candidates_by_s1 = _check_one_file(
        candidates, "candidate_pairs.tsv", "candidate_entity_ids",
        expected_s1_ids, problems,
    )

    rows_with_bad_prefix = 0
    rows_with_dup_ids = 0
    rows_with_malformed = 0
    rows_with_non_universe_match = 0
    rows_with_match_not_in_candidates = 0

    total_matches = 0
    total_candidates = 0
    s1_with_matches = 0
    s1_with_candidates = 0

    example_bad_prefix = None
    example_dup = None
    example_malformed = None
    example_non_universe = None
    example_not_subset = None

    for s1_id in expected_s1_ids:

        match_raw = matches_by_s1.get(s1_id)
        cand_raw = candidates_by_s1.get(s1_id)

        match_ids, match_malformed = _split_ids(match_raw)
        cand_ids, cand_malformed = _split_ids(cand_raw)

        if match_malformed or cand_malformed:
            rows_with_malformed += 1
            if example_malformed is None:
                example_malformed = (s1_id, match_raw, cand_raw)

        if match_ids:
            s1_with_matches += 1
            total_matches += len(match_ids)

        if cand_ids:
            s1_with_candidates += 1
            total_candidates += len(cand_ids)

        if len(set(match_ids)) != len(match_ids):
            rows_with_dup_ids += 1
            if example_dup is None:
                example_dup = (s1_id, "matched_entity_ids", match_raw)

        if len(set(cand_ids)) != len(cand_ids):
            rows_with_dup_ids += 1
            if example_dup is None:
                example_dup = (s1_id, "candidate_entity_ids", cand_raw)

        bad_prefix = [
            x for x in (match_ids + cand_ids)
            if not (x.startswith("S2-") or x.startswith("S3-"))
        ]
        if bad_prefix:
            rows_with_bad_prefix += 1
            if example_bad_prefix is None:
                example_bad_prefix = (s1_id, bad_prefix[:5])

        if valid_candidate_ids is not None:
            not_in_universe = [
                x for x in (match_ids + cand_ids)
                if x not in valid_candidate_ids
            ]
            if not_in_universe:
                rows_with_non_universe_match += 1
                if example_non_universe is None:
                    example_non_universe = (s1_id, not_in_universe[:5])

        not_subset = [x for x in match_ids if x not in set(cand_ids)]
        if not_subset:
            rows_with_match_not_in_candidates += 1
            if example_not_subset is None:
                example_not_subset = (s1_id, not_subset[:5])

    if rows_with_malformed:
        problems.append(
            f"{rows_with_malformed:,} rows contain a malformed list "
            f"cell (e.g. {example_malformed})"
        )

    if rows_with_dup_ids:
        problems.append(
            f"{rows_with_dup_ids:,} rows contain duplicate ids within "
            f"a single list (e.g. {example_dup})"
        )

    if rows_with_bad_prefix:
        problems.append(
            f"{rows_with_bad_prefix:,} rows contain an id that is not "
            f"S2-/S3- prefixed (e.g. {example_bad_prefix})"
        )

    if valid_candidate_ids is not None and rows_with_non_universe_match:
        problems.append(
            f"{rows_with_non_universe_match:,} rows contain an id that "
            f"is not a real test S2/S3 entity id "
            f"(e.g. {example_non_universe})"
        )

    if rows_with_match_not_in_candidates:
        problems.append(
            f"{rows_with_match_not_in_candidates:,} rows have a "
            f"predicted match that is NOT in that row's candidate "
            f"list (e.g. {example_not_subset})"
        )

    summary = {
        "expected_s1": len(expected_s1_ids),
        "matching_results_rows": len(results),
        "candidate_pairs_rows": len(candidates),
        "s1_with_matches": s1_with_matches,
        "s1_with_zero_matches": len(expected_s1_ids) - s1_with_matches,
        "s1_with_candidates": s1_with_candidates,
        "s1_with_zero_candidates": len(expected_s1_ids) - s1_with_candidates,
        "total_matches": total_matches,
        "total_candidates": total_candidates,
    }

    return (len(problems) == 0), summary, problems


def print_report(summary, problems):
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key:<28}: {value:,}")
    print()
    if problems:
        print(f"FAILED - {len(problems)} problem(s) found:\n")
        for p in problems:
            print(f"  - {p}")
    else:
        print("PASSED - output looks structurally valid.")
        print(
            "(This is an internal check only - it is not the official "
            "Amazon validator. No utils/validate_submission.py was "
            "found in this repository, so run the official validator "
            "separately if/when you have access to it.)"
        )
