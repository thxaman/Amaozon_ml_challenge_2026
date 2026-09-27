# Test-Time Submission Pipeline

This documents the production scripts under `src/matching/` that take
the already-trained model (`data/processed/final_match_classifier.joblib`)
from "trained on train data" to "submission-ready on test data".

Nothing here retrains the model or touches the frozen 0.30 threshold
except by explicit opt-in flag (`--use-model-threshold`, and even then
only using a threshold that was itself selected on train/validation
data, never on test data).

## Prerequisites

```
pip install -r requirements.txt
```

Expected on disk before running anything here:

```
data/processed/final_match_classifier.joblib   (already trained by you)
data/test/test_source1.tsv
data/test/test_source2.tsv
data/test/test_source3.tsv
```

## IMPORTANT - pipeline was rewritten (memory-safety fix)

The original three-script pipeline (`build_test_candidates.py` ->
`build_test_features.py` -> `run_test_inference.py`, run in sequence)
is **no longer the production path**. It is kept in the repo for
reference/audit only, with a header note in each file saying so.

That three-script design was correct in its blocking and feature
*logic*, but each script materialized a full-test-scale intermediate
file to disk:

- `build_test_candidates.py` wrote one row **per candidate pair** to
  `data/processed/candidate_pairs.tsv` (not one row per S1 - a format
  mismatch with the required submission format).
- `build_test_features.py` wrote `data/processed/test_pair_features.tsv`,
  one row per candidate pair.
- `run_test_inference.py` (old version) wrote
  `data/processed/test_pair_scores.tsv`, one row per candidate pair.

At full test scale (~1.73M test S1, ~1,489 candidates/S1 on average),
that is on the order of **~2.58 billion rows** for each of those two
intermediate files. This is exactly what the project brief prohibits.

**`run_test_inference.py` was rewritten** to fuse candidate generation,
feature engineering, and scoring into a single streaming, batched pass
that never writes a per-pair file to disk. It is now a single script
that does everything T1+T2+T3 used to do, safely.

## Run order (current)

```bash
# Smoke test first - real data, real model, real blocking/feature
# logic, but only the first N test S1 rows. Writes to
# output/smoke_test/ and validates automatically. Does NOT touch
# output/matching_results.tsv or output/candidate_pairs.tsv.
python src/matching/run_test_inference.py --max-s1 1000

# Full run - processes every test S1 row. Writes to temporary
# .partial files and atomically renames them to the final names only
# after the run completes without error.
python src/matching/run_test_inference.py --full
#   -> output/matching_results.tsv   <-- final submission file
#   -> output/candidate_pairs.tsv    <-- final candidate file

# Full-scale validation against the real S2/S3 id universe (also runs
# automatically, against a smaller id universe, at the end of the
# smoke test above).
python src/matching/validate_submission.py
```

All three commands accept `--data-dir`, `--model-path`,
`--output-dir` (the two runner commands also accept `--batch-size`);
see `python src/matching/run_test_inference.py --help`.

## What the new script reuses from the existing codebase (unchanged)

- The four-channel blocker (country + exact name / country + address
  token, freq<=2500 / country + name-token pair, freq<=1000 / country +
  address-token pair, freq<=50), copied verbatim from
  `build_test_candidates.py`, including its two-pass
  (frequency-then-index) approach - unavoidable, since applying a
  frequency cap requires knowing the frequency first.
- The exact pairwise feature logic (all 40 features - see
  `compute_pair_features()` in `run_test_inference.py`), copied
  verbatim from `build_test_features.py` / `build_full_pair_features.py`.
- The frozen model and the frozen 0.30 threshold.
- `source_is_s3` encoding (S3=1, S2=0).
- The two dev-scale spec reconciliations already made in
  `build_test_candidates.py` (channel 4 includes country in its key;
  generic name/address token lists match the master spec) - preserved
  in the new script's copies of those constants/functions.

Two distinct `normalize_text()` functions exist for two different
purposes in the original codebase and are both preserved, now under
distinct names so they can never be accidentally swapped:
`blocking_normalize_text()` (Unicode-aware, used for blocking keys) and
`feature_normalize_text()` (ASCII a-z0-9 only, used for pairwise text
features).

## How the memory-safety fix works

For each batch of `--batch-size` (default 1,000) S1 entities:

1. Generate candidates for just this batch (four-channel blocker).
2. Compute pairwise features for just this batch's candidate pairs,
   in memory, transiently.
3. Score with the frozen classifier; apply the frozen 0.30 threshold.
4. Write **one aggregated row per S1** to each output file
   (`source1_entity_id \t comma,separated,ids`).
5. Discard the batch and continue.

No full candidate-pair file, feature file, or score file is ever
created. Peak memory is bounded by the blocking indexes (already
frequency-capped by the four-channel design) plus the *filtered*
S2/S3 feature lookups (only entities that are candidate-eligible for
at least one S1 are loaded - not the full 4.89M/5.08M-row files) plus
one batch's candidate pairs (controlled by `--batch-size`, independent
of total test-set size).

`predicted_matches(S1) ⊆ candidate_entities(S1)` holds by construction,
since both are derived from the exact same in-memory candidate set for
that S1 within the same batch iteration - there is no way for them to
diverge.

## Failure safety

The full run writes to `output/matching_results.tsv.partial` and
`output/candidate_pairs.tsv.partial` throughout, and only calls
`os.replace()` to move them to the final names after the entire run
completes without raising. If the process crashes or is killed, the
final `output/*.tsv` files (if they exist from a previous successful
run) are left completely untouched, and the `.partial` files are left
in place for inspection with a clear error message - never a
half-written file at the final path.

**Not implemented:** true checkpointed resume (restarting a killed run
partway through without redoing already-processed S1 batches). The
pipeline is deterministic and fast enough per batch that a full
restart is the current recovery path; see the risk list in the hand-off
notes for a sketch of how resume could be added if a run turns out to
be multi-hour and interruption-prone in practice.

## Legacy scripts (kept for reference only)

- `build_test_candidates.py` - superseded. Its blocking logic was
  copied into `run_test_inference.py`; do not run it for a final
  submission (it writes a billions-of-rows per-pair file).
- `build_test_features.py` - superseded, same reason.

Both now have a header note pointing here.
