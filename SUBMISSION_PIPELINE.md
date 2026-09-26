# Test-Time Submission Pipeline

This documents the four new scripts added under `src/matching/` to take
the already-trained model (`data/processed/final_match_classifier.joblib`)
from "trained on train data" to "submission-ready on test data".

Nothing here retrains the model or touches the frozen 0.30 threshold
except by explicit opt-in flag. Test data is used strictly for
inference, as required by MASTER_PROMPT.txt Section 19.

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

## Run order

```bash
# T1 - four-channel blocking over the FULL test set
# (not a 10k sample - this processes all test S1/S2/S3)
python src/matching/build_test_candidates.py
#   -> data/processed/candidate_pairs.tsv

# T2 - pairwise feature engineering on the candidates from T1
# (identical feature logic to build_full_pair_features.py, no label)
python src/matching/build_test_features.py
#   -> data/processed/test_pair_features.tsv

# T3 - score with the frozen classifier and apply the frozen threshold
python src/matching/run_test_inference.py
#   -> data/processed/matching_results.tsv   <-- final submission file
#   -> data/processed/test_pair_scores.tsv   <-- per-pair scores, for audit

# T4 - internal sanity checks before the official validator
python src/matching/validate_submission.py
```

After T4 passes, run Amazon's official validator on
`data/processed/matching_results.tsv` (and submit
`data/processed/candidate_pairs.tsv` alongside it, since
MASTER_PROMPT.txt Section 24 requires the candidate file too).

## What each script reuses from the existing codebase

- `build_test_candidates.py` reimplements the four-channel blocker
  validated in `evaluate_improved_blocking.py` +
  `evaluate_address_pair_blocking.py` (96.79% pair recall on the 10k-S1
  train dev sample), scaled to the full test set instead of a sample.
  Same channels, same frequency limits (2500 / 1000 / 50).

- `build_test_features.py` copies the feature logic in
  `build_full_pair_features.py` verbatim (40 features), because the
  trained model's behavior depends on features being computed exactly
  the same way as during training.

- `run_test_inference.py` loads the joblib package as-is
  (`model`, `feature_columns`, `threshold`) and hardcodes the frozen
  competition threshold (0.30) per MASTER_PROMPT.txt Section 18,
  independent of whatever threshold happens to be stored in the
  package, unless you pass `--use-model-threshold` explicitly.

## Two spec reconciliations made in `build_test_candidates.py`

The dev-scale scripts that validated the four-channel blocker had two
small internal inconsistencies relative to MASTER_PROMPT.txt Section 6/7.
The production script follows the master prompt (declared as source of
truth) in both cases:

1. **Channel 4 (address-token pair) now includes country in its key**,
   matching Section 7's "country + meaningful address-token pair" for
   every channel. (`evaluate_address_pair_blocking.py`, the dev-scale
   script, built this channel's frequency table without country.)
2. **Generic name/address token lists match Section 6 exactly**
   (e.g. added `groups`, `partners`, `partner`, `lp` to the generic
   name-token list, which were missing from `evaluate_improved_blocking.py`).

Neither change touches the four-channel architecture, the union logic,
or the frequency thresholds - only internal consistency of the
normalization already specified in the master prompt.

## Performance note

`build_test_candidates.py` does two full passes over test S2 + S3
(~4.89M + ~5.08M rows) for every one of the ~1.73M test S1 entities'
blocking keys, the same two-pass frequency-then-index approach already
used in the codebase's dev-scale scripts, just applied to the full
test set instead of a 10k-S1 sample. Expect this to be the longest-
running step; it prints progress every ~1M rows scanned/indexed and
every 50k S1 processed during candidate generation.
