# Amazon ML Challenge 2026 - Business Entity Resolution
## Submission Documentation

---

### 1. Executive Summary

This submission matches noisy business records in Source 2 and
Source 3 against the deduplicated reference records in Source 1. Each
Source 1 entity can have zero, one, or multiple matches, and there is
no shared ID across sources. The approach is a standard entity
resolution pipeline: normalize -> block (candidate generation) ->
pairwise feature engineering -> binary classification -> threshold ->
aggregate to multi-match output. The classifier is a frozen
`HistGradientBoostingClassifier` at a frozen decision threshold of
0.30, selected during development and not tuned on test data. The
final inference pipeline was re-engineered for test scale (~1.73M
Source 1 records) to run in bounded memory without ever materializing
the full candidate-pair universe (~2.5+ billion pairs at naive
materialization) as an intermediate file.

### 2. Problem Analysis

- Source 1: ~1.73M reference/deduplicated entities (test).
- Source 2: ~4.89M noisy entities (test).
- Source 3: ~5.08M noisy entities (test).
- No common key across sources; matching must be done on
  `business_name`, `business_address`, and `country` alone.
- Multiplicity: a Source 1 entity may match zero, one, or several
  Source 2/3 entities, so this is a one-to-many retrieval-then-classify
  problem, not a one-to-one assignment problem.
- Scale is the dominant engineering constraint: an unrestricted
  cross-join of Source 1 against Source 2+3 is combinatorially
  infeasible, so a blocking (candidate generation) step is mandatory
  before any pairwise comparison.

### 3. Solution Strategy

A five-stage pipeline:

1. **Normalization** - lowercase, punctuation stripped, `&` expanded
   to `and`, whitespace collapsed (two slightly different
   normalization functions are used for two different purposes - see
   Section 4).
2. **Blocking / candidate generation** - four independent blocking
   channels, unioned, to maximize recall while keeping the candidate
   set per Source 1 entity tractable.
3. **Pairwise feature engineering** - 40 features per (Source 1,
   candidate) pair covering exact/fuzzy string similarity, token
   overlap, shared numeric tokens, and several hand-built interaction
   features.
4. **Binary classification** - a single frozen
   `HistGradientBoostingClassifier` scores every candidate pair.
5. **Thresholding and aggregation** - a frozen threshold (0.30) is
   applied per pair, and accepted pairs are grouped back up to one
   row per Source 1 entity for the final submission format.

### 4. Normalization

Two distinct normalization functions are used, for two different
purposes, and both are preserved exactly as implemented in
development (not changed for this submission):

- **Blocking-key normalization** (`blocking_normalize_text`):
  lowercases, expands `&` to `and`, strips any character that is not a
  Unicode word character or whitespace (`[^\w\s]`), collapses
  whitespace. Unicode-aware, so non-Latin scripts retain their letters
  as blocking-key material.
- **Feature-value normalization** (`feature_normalize_text`):
  lowercases, expands `&` to `and`, strips any character that is not
  ASCII `a-z0-9` (`[^a-z0-9]+`), collapses whitespace. ASCII-only.

Generic tokens are removed before blocking-key tokenization (e.g.
`ltd`, `inc`, `pvt`, `group`, `road`, `street`, `suite` - see
`GENERIC_NAME_TOKENS` / `GENERIC_ADDRESS_TOKENS` in the code) so that
extremely common boilerplate words do not dominate blocking keys.

### 5. Candidate Generation / Blocking

Four blocking channels, unioned per Source 1 entity, each keyed by
`country` plus a signal derived from `business_name` or
`business_address`:

| Channel | Key | Frequency cap |
|---|---|---|
| 1 | country + exact normalized business name | none |
| 2 | country + meaningful address token | freq <= 2,500 |
| 3 | country + meaningful name-token pair | freq <= 1,000 |
| 4 | country + meaningful address-token pair | freq <= 50 |

Frequency caps exclude overly generic keys (e.g. a single common
address token shared by thousands of records) from contributing
candidates, which controls candidate-set size without an arbitrary
top-K cutoff. These four channels and their caps were experimentally
evaluated during development and are unchanged here.

**Development-sample blocking result** (10,000-S1 development sample,
using train ground truth - this figure is a *development/validation*
metric, not a test-set metric, since test labels are never available):

- Pair recall: **≈96.79%**
- Full-S1 recall: **≈91.04%**

### 6. Feature Engineering

40 features per (Source 1, candidate) pair:

- **Base similarity**: `country_match`, `name_exact`, `name_ratio`,
  `name_token_ratio`, `name_partial_ratio`, `name_length_ratio`,
  `shared_name_tokens`, `address_exact`, `address_ratio`,
  `address_token_ratio`, `address_partial_ratio`,
  `address_length_ratio`, `shared_address_tokens`, `shared_numbers`.
- **Threshold indicators**: `name_strong_90/80`,
  `address_strong_90/80`, `both_strong_90/80`.
- **Interaction features**: `name_strong_address_weak`,
  `address_strong_name_weak`, `name_address_mean/min/max/gap/product`,
  `exact_name_strong_address`, `exact_address_strong_name`.
- **Evidence aggregates**: `name_tokens_strong`, `address_tokens_strong`,
  `address_tokens_very_strong`, `has_shared_number`,
  `multiple_shared_numbers`, `strong_address_with_number`,
  `very_strong_address_with_number`, `strong_name_with_number`,
  `strong_evidence_count`, `very_strong_evidence_count`.
- **Source indicator**: `source_is_s3` (S3=1, S2=0).

String similarity is computed with RapidFuzz (`ratio`,
`token_sort_ratio`, `partial_ratio`); shared numeric tokens are
extracted from addresses with a digit regex. These are exactly the
features the frozen classifier was trained on; the final inference
pipeline computes them identically (verified against the training
feature-generation code as the source of truth).

### 7. Matching Model

`sklearn.ensemble.HistGradientBoostingClassifier`
(`learning_rate=0.08`, `max_iter=250`, `max_leaf_nodes=31`,
`min_samples_leaf=50`, `l2_regularization=1.0`, `random_state=42`),
trained on a sampled mix of positive pairs, hard negatives (near-miss
pairs with high name/address similarity but no true match), and
ordinary negatives, with a grouped (by Source 1 entity)
train/validation split to avoid leaking a Source 1 entity's pairs
across the split. The model is frozen for this submission; it is not
retrained as part of productionizing the inference pipeline.

### 8. Threshold Selection

The decision threshold is frozen at **0.30**, selected during
development by sweeping candidate thresholds and choosing the one that
maximized per-Source-1 macro F0.5 on the held-out validation split
(grouped by Source 1 entity). This threshold is hardcoded in the
inference pipeline and is not re-tuned using test data or test
predictions, per the project's no-leakage requirement.

### 9. Validation

An internal (non-official) validator checks, for both output files:
correct headers and TSV format; exactly one row per test Source 1
entity with no duplicates and no unexpected IDs; every listed ID is
`S2-`/`S3-` prefixed and actually exists in test Source 2/3; no
duplicate IDs within a row; no malformed cells (`nan`/`None`/`[]`);
correct empty-list representation; and that every predicted match is
contained in that row's candidate list. No official Amazon validator
(`utils/validate_submission.py`) was present in this repository at
submission time; this internal check does not claim to replace it.

### 10. Error Analysis

Error analysis (false positives/negatives at the frozen threshold,
by evidence-feature bucket) was performed during development against
the labeled training data and hard-negative mining used those
findings to build the training sample. Because test labels do not
exist, no equivalent test-set error analysis is possible or claimed
here; only the development/validation-split analysis is available.

### 11. Final Inference Pipeline

Test-time inference reuses the exact blocking and feature logic above
but is engineered to run within bounded memory at test scale
(~1.73M Source 1 x ~4.89M/5.08M Source 2/3), where naive
materialization of every (Source 1, candidate) pair would produce on
the order of ~2.5+ billion rows.

The pipeline processes Source 1 entities in configurable batches
(default 1,000). For each batch: candidates are generated with the
four blocking channels; pairwise features are computed for just that
batch's candidates, in memory; the frozen classifier scores them; the
frozen 0.30 threshold is applied; and one aggregated row per Source 1
entity is written to each output file. No full candidate-pair file,
feature file, or score file is ever created - peak memory is governed
by the (frequency-capped) blocking indexes and one batch's candidate
pairs, not by total test-set size. A smoke-test mode
(`--max-s1 N`) runs the identical logic on a small slice, writes to an
isolated directory, and validates automatically before any full run is
attempted. The full run writes to temporary files and only atomically
renames them to the final submission names after completing without
error, so a crash never corrupts or replaces a previously valid
submission file.

**Final test-run counts** (to be filled in with the actual numbers
printed by `run_test_inference.py --full` after that run is executed -
not invented here, since the full run has not yet been executed as of
this document):

- Test Source 1 records processed: `<fill in from run log>`
- Total candidate pairs generated: `<fill in from run log>`
- Total predicted matches: `<fill in from run log>`
- Source 1 entities with zero candidates: `<fill in from run log>`
- Source 1 entities with zero predicted matches: `<fill in from run log>`
- Wall-clock runtime: `<fill in from run log>`

No test-set F0.5, precision, or recall is reported, because the
competition test set has no available ground truth to compute one.

### 12. Conclusion

The submission reuses an already-validated blocking strategy
(≈96.79% pair recall / ≈91.04% full-S1 recall on a 10k-S1 development
sample) and an already-trained, frozen classifier at a frozen
threshold, and focuses the remaining engineering effort on making test
inference run safely at full competition scale without materializing
billions of intermediate rows, while preserving the exact blocking and
feature logic the model was trained and validated against.
