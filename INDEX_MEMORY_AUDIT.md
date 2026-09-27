# Memory audit: `run_test_inference.py` blocking-index storage

Scope: this is a **follow-up** to the existing fused/memory-safe pipeline.
It touches only the *storage layer* of the four blocking-channel posting
indexes. The model, threshold, blocking rules, frequency caps,
normalization functions, feature definitions (40-feature schema), and
output formats are unchanged. See "What was NOT changed" below and the
file-diff summary at the end.

## 1. What the audit found

Before this change, `run_test_inference.py` built its blocking index in
two in-RAM structures, both living for the whole run:

- **`freq_tables`** (Phase 2, `scan_frequencies`): four `Counter()`
  objects (`exact_name_freq`, `address_freq`, `name_pair_freq`,
  `address_pair_freq`), keyed by `(country, key)` / `(country, key_pair)`,
  restricted to the vocabulary derived from test S1 but otherwise
  unbounded in cardinality.
- **`indexes`** (Phase 3, `build_indexes_per_source`): for each of S2 and
  S3, four `defaultdict(set)` posting indexes (`exact_name`, `address`,
  `name_pair`, `address_pair`), each mapping `(country, key)` →
  `set(entity_id)`. Eight such dict-of-sets structures total, held in RAM
  for the entire run (used by every batch in Phase 3's candidate
  generation).

Two things make this not fit reliably at full scale (1.73M S1 / 4.89M S2
/ 5.08M S3, ~2.58B candidate relationships):

**(a) The three frequency caps bound each individual posting list, not
the number of posting lists.** `BROAD_ADDRESS_FREQ<=2500`,
`NAME_PAIR_FREQ<=1000`, `ADDRESS_PAIR_FREQ<=50` limit how many entity ids
can sit under any *one* key. They do nothing to limit how many distinct
`(country, key)` keys exist. That count is driven by the S1-derived
vocabulary, and for the two pair-based channels it is combinatorial:
`token_pairs()` is `itertools.combinations(tokens, 2)`, so a business
name/address with `k` meaningful tokens contributes `C(k,2)` pair-keys.
Summed across 1.73M S1 records, `all_name_pairs` and (especially)
`all_address_pairs` in `vocab` can run into the tens of millions of
distinct keys even before S2/S3 are scanned. Every one of those keys that
gets even one qualifying S2/S3 hit becomes its own dict entry with its
own (small) set.

**(b) The exact-name channel has no cap at all.** In the original code,
`if exact_name_freq[key] > 0: exact_name_index[key].add(entity_id)` looks
like a threshold check, but by construction `exact_name_freq[key]` was
already incremented for exactly this row during the scan pass, so the
check is always true once a name is in `vocab["exact_names"]`. It is pure
membership, not a real frequency cap — so this channel's postings can
grow with real-world name collisions with nothing bounding them.

**(c) Python per-entry overhead, not just the raw ids.** Each
`defaultdict(set)` entry costs a dict slot (key = a 2-tuple object, itself
~56 bytes plus its two string refs) plus a `set` object (a *small*
Python `set` is not "free" — CPython allocates a hash table sized to the
next power of two above `len/0.6`, i.e. a set with 1 element still
reserves 8 slots). Summed over tens of millions of keys across 8 index
structures, this per-entry overhead — not the entity-id strings
themselves, which are shared references — is what pushes total memory
well past what a Colab-class runtime (roughly 12–25 GB standard/high-RAM,
up to ~51 GB on Colab Pro+, never guaranteed) can hold *for this one
structure*, on top of everything else the process needs (the model,
pandas chunks, the batch's transient feature frame).

**(d) Double-booking during index construction.** The original
`build_indexes_per_source` received `freq_tables` and only the caller
does `del freq_tables` *after* both `scan_frequencies` and
`build_indexes_per_source` return. That means for the whole of Phase 3,
the freq `Counter()`s (same key cardinality as the indexes being built)
and the indexes themselves are simultaneously resident — roughly
doubling the peak of an already-too-large structure.

**Conclusion:** at the real test scale, the complete in-memory posting
index cannot be relied on to fit in the intended Colab/runtime
environment. A disk-backed or otherwise compact index-storage mechanism
is required. (Note: `build_filtered_feature_lookup`, which loads only
candidate-eligible S2/S3 *rows* — as opposed to the *index* — was already
memory-safe in the original code and did not need to change; see below.)

## 2. What was changed

Only the index **storage and lookup** mechanism changed. The four
channels, their exact keys, and their exact frequency thresholds are
unchanged.

- **`open_index_db()`**: creates a fresh on-disk SQLite database
  (stdlib `sqlite3`, no new dependency) with two tables:
  - `freq(channel, country, key, cnt)` — replaces the three `Counter()`s
    for the address / name-pair / address-pair channels. (The exact-name
    channel needs no frequency table at all — see point (b) above — so
    it was dropped from the counting pass entirely rather than kept as
    dead weight.)
  - `postings(channel, source, country, key, entity_id)` — replaces the
    eight `defaultdict(set)` indexes.

  A `(token_a, token_b)` pair key is encoded as a single TEXT string
  `token_a + "\x1f" + token_b` (`encode_pair()`) so it can be a SQLite
  key; `token_pairs()` already returns pairs sorted via
  `itertools.combinations(sorted(tokens), 2)`, so the encoding is stable
  and lossless (0x1F cannot appear in a token, since
  `blocking_normalize_text()` only keeps `\w` characters).

- **`scan_and_store_frequencies()`** (was `scan_frequencies`): same
  chunked scan over S2 then S3, same per-row channel logic and vocab
  restriction, but the `Counter()` for each channel is now **chunk-local**
  (bounded by `SCAN_CHUNK_SIZE`, not by total corpus vocabulary) and is
  flushed into the on-disk `freq` table via a batched
  `INSERT ... ON CONFLICT DO UPDATE SET cnt = cnt + excluded.cnt` after
  each chunk, then discarded.

- **`build_indexes_on_disk()`** (was `build_indexes_per_source`): same
  per-row channel logic and the same three threshold comparisons, but
  (1) frequency lookups for the current chunk are fetched from the
  on-disk `freq` table via one batched join per channel
  (`_lookup_chunk_freq()`, using a temp table of just this chunk's
  needed keys) instead of a full-corpus Python `Counter`, and
  (2) qualifying rows are inserted into the on-disk `postings` table
  instead of an in-RAM `set`. Only one chunk's worth of parsed rows and
  frequency lookups are ever held in Python objects at a time. The
  `needed_ids` set (entity ids that appear in at least one posting, used
  to build the filtered S2/S3 feature lookups) is unchanged in
  mechanism — it is still a Python `set`, because it is bounded by the
  number of *eligible rows* (≤ |S2|+|S3| ≈ 10M in the worst case), not by
  posting-list cardinality, and was never the memory problem.

- **`find_candidates_for_batch()`** (was `find_candidates_for_s1()`,
  called once per S1): for a whole S1-batch at once, builds a reverse
  map `(country, key) -> [s1_id, ...]` per channel, loads those keys into
  a small SQLite temp table, and does one indexed `JOIN` per channel
  against `postings` to retrieve exactly the postings that batch needs.
  Same 4-channel union-of-postings result — `{s1_id: {candidate_id:
  source}}` — as the original per-S1 loop, just computed with 4 queries
  per batch instead of `O(batch_size × keys_per_s1)` Python dict
  lookups against a multi-GB structure. This keeps candidate generation
  batch-scale, not test-scale, in line with the existing memory-safety
  design of Phase 3 (never materializing all candidates/features/scores
  at once — untouched).

- **`report_index_stats()`** (new): prints the instrumentation requested
  below, once, right after the index is built.

- **`IncrementalStats`**: added `peak_batch_candidates` (max candidates
  summed over a single S1-batch), reported alongside the existing
  per-S1 `max_candidates`, average candidates, and total candidate pairs.

- **`main()`**: wires the SQLite connection through instead of the old
  `indexes` dict; the DB file defaults to
  `<output-dir>/_blocking_index.sqlite3` (overridable with
  `--index-db-path`) and is deleted on exit unless `--keep-index-db` is
  passed (kept for debugging/instrumentation inspection). Cleanup runs
  in a `finally` block, so it happens on both success and failure (the
  index DB is a disposable, rebuildable intermediate artifact, not test
  output — it is never one of the two required deliverable files).

### What was NOT changed

- `final_match_classifier.joblib` loading and the frozen `0.30`
  threshold logic (`FROZEN_THRESHOLD`, `--use-model-threshold`).
- The 40-feature schema and `compute_pair_features()` body (feature
  logic, including `source_is_s3`) — byte-for-byte untouched.
- `blocking_normalize_text()` / `feature_normalize_text()` (the two
  distinct, intentionally-different normalization functions) and all
  Unicode-blocking behavior.
- The four channels' keys and their three frequency thresholds
  (`BROAD_ADDRESS_FREQ=2500`, `NAME_PAIR_FREQ=1000`,
  `ADDRESS_PAIR_FREQ=50`).
- Missing-value handling (`pd.isna` checks in both normalizers,
  `keep_default_na=False` + `dtype=str` reads).
- `build_filtered_feature_lookup()` — already scanned S2/S3 in chunks and
  kept only candidate-eligible rows; this was not part of the audited
  problem and is unchanged.
- Output format: one row per S1 in both `candidate_pairs.tsv`
  (`source1_entity_id\tcandidate_entity_ids`) and `matching_results.tsv`
  (`source1_entity_id\tmatched_entity_ids`); atomic `.partial` +
  `os.replace()` write for `--full`; `--max-s1` smoke-test mode writing
  to an isolated `smoke_test/` subdirectory without touching final
  files; internal validation via `submission_checks.validate_outputs`
  (also unmodified).
- Candidate generation, feature computation, and scoring remain
  batched per S1-batch; no full candidate-pair file, feature file, or
  score file is ever materialized (this was already true and remains
  true).

## 3. Instrumentation added

Printed once after the on-disk index is built (`report_index_stats`),
and in the final run-statistics block (`IncrementalStats.report`):

- Posting entries (row count in the `postings` table)
- Frequency-table row count (`freq` table)
- On-disk index DB size (bytes and GB)
- Maximum candidates for a single S1 (existing, kept)
- Peak candidates summed across a single S1-batch (new)
- Total candidate pairs processed across the whole run (existing, kept
  and relabeled for clarity)

## 4. Testing performed

Per the brief, **the real full test was not run** (real test data was not
present in the uploaded project — `data/test/` and the frozen `.joblib`
are both absent from this repository snapshot). All testing below is
synthetic, plus one smoke-test-mode dry run.

### 4.1 Exact-equivalence test (correctness)

A synthetic S1/S2/S3 dataset (1,500 / 2,200 / 2,100 rows) was generated
to exercise all four blocking channels, including exact-name collisions,
shared address/name tokens, shared token pairs, multiple countries,
Unicode business names (`Café Müller`, `北京餐馆`, `José García & Sons`,
`O'Brien's Pub`), and missing name/address values. A synthetic frozen
model package (`{"model", "feature_columns" (40 cols, exact order used
by `compute_pair_features`), "threshold": 0.30}`) was built with
`scikit-learn`'s `LogisticRegression` so `predict_proba` behaves
realistically.

The **unmodified original script** and the **modified script** were run
end-to-end (`--full`) against the identical input files and model:

```
Total candidate pairs   : 633,059   (identical, both versions)
Total predicted matches : 624,467   (identical, both versions)
S1 with zero candidates : 5         (identical, both versions)
S1 with zero predictions: 13        (identical, both versions)
```

`matching_results.tsv` and `candidate_pairs.tsv` from both runs were
sorted and diffed: **byte-for-byte identical**, confirming the on-disk
index produces exactly the same candidate sets and match decisions as
the original in-RAM index.

### 4.2 Smoke-test mode

`--max-s1 200 --batch-size 50` was run against the same synthetic data.
Output landed under `smoke_test/`, the (pre-existing) `--full` output
files were left untouched, and `submission_checks.validate_outputs`
reported **PASSED**. Index instrumentation was confirmed present in the
output (posting-entry count, freq-row count, on-disk DB size).

### 4.3 Failure-path / atomicity check

A deliberately broken model package (an extra, non-existent feature
column) was used to force a `ValueError` mid-`--full`-run. Confirmed:
the run exits non-zero with a clear message; `matching_results.tsv` /
`candidate_pairs.tsv` at the final path are left untouched (a
pre-existing file at that path was **not** overwritten); the `.partial`
files are left in place for inspection; and the on-disk index DB is
still cleaned up via the `finally` block despite the exception.

### 4.4 Index-build memory profile (checkpoint instrumentation)

RSS checkpoints (`resource.getrusage(...).ru_maxrss`) were added to
throwaway copies of both scripts (not part of the delivered file) at
"index build complete". On a 20k-S1 / ~30k-S2 / ~30k-S3 synthetic corpus,
the on-disk index build for the modified script completed using ~230 MB
RSS and produced a ~13 MB SQLite file, with posting counts confirmed via
`report_index_stats`.

**Limitation, stated plainly:** the sandbox this audit ran in has ~3.9 GB
of total RAM and a 300-second per-command ceiling, so it cannot host a
literal 1.73M/4.89M/5.08M-row run, nor cleanly reproduce a "before: OOM,
after: succeeds" moment at that scale — any synthetic corpus large enough
to overflow 4 GB of *index* memory in the original in-RAM design (tens of
millions of posting entries) takes far longer than 300 seconds to
generate and scan in this environment. Two adjacent synthetic runs at
20k–30k rows/source did hit the container's memory ceiling, but in
**Phase 3 (batched candidate generation/scoring)**, not in index
construction, and hit it in *both* the original and the modified script
equally — that phase's memory is a function of `--batch-size` and of how
combinatorially collision-heavy the synthetic vocabulary is (an
artificially small word list was used to keep the corpus small, which
inflates per-key posting sizes toward the frequency caps and is not
representative of the diversity of real business names/addresses). That
is expected, pre-existing, batch-size-dependent behavior documented in
the file's own module docstring ("batch-size dependent, NOT test-scale
dependent") and is not the mechanism this follow-up was asked to change.
The conclusion that the *index* no longer needs to fit in RAM rests on
(a) the complexity argument in Section 1, and (b) the confirmed
disk-backed, chunk-bounded construction demonstrated in 4.1–4.4, rather
than on a full-scale reproduction that this sandbox cannot host.

## 5. Files changed

| File | Change | Why |
|---|---|---|
| `src/matching/run_test_inference.py` | Modified | Replaced the in-RAM `defaultdict(set)` posting indexes and in-RAM `Counter()` frequency tables with an on-disk SQLite database (`freq` + `postings` tables), built and queried in chunk-/batch-bounded steps. Added index-size/posting/disk-usage/peak-batch-candidate instrumentation. Added `--index-db-path` / `--keep-index-db` CLI flags. No change to the model, threshold, blocking rules/frequency caps, normalization functions, the 40-feature schema, `source_is_s3`, missing-value handling, output row format, atomic-write behavior, or smoke-test/validation behavior. |
| `INDEX_MEMORY_AUDIT.md` | Added | This document. |

No other file in the repository was modified.
