"""
PRODUCTION TEST PIPELINE (FUSED, MEMORY-SAFE) - T1+T2+T3 in one pass

WHY THIS FILE WAS REWRITTEN
----------------------------
The previous three-script pipeline (build_test_candidates.py ->
build_test_features.py -> run_test_inference.py) was correct in its
blocking and feature LOGIC, but materialized two full-test-scale
intermediate files to disk:

    data/processed/test_pair_features.tsv   (one row per candidate pair)
    data/processed/test_pair_scores.tsv     (one row per candidate pair)

At full test scale (~1.73M test S1, ~1,489 candidates/S1 on average),
that is on the order of ~2.58 BILLION rows. This is exactly the failure
mode the project brief prohibits ("DO NOT create test_pair_features.tsv
/ test_pair_scores.tsv containing all test candidate pairs").

It was also a format mismatch: the required candidate_pairs.tsv is ONE
ROW PER TEST SOURCE-1 ENTITY with a comma-separated candidate list, not
one row per (S1, candidate) pair.

This script fixes both problems by fusing candidate generation, feature
engineering, and scoring into a single per-S1-batch loop:

    for each batch of S1 entities:
        1. generate candidates for this batch (existing 4-channel blocker)
        2. compute the existing pairwise features for just this batch's
           candidate pairs (kept in memory transiently, never written
           to disk)
        3. score with the frozen classifier, apply the frozen 0.30
           threshold
        4. write ONE aggregated row per S1 to each output file
        5. release the batch and continue

No full candidate-pair file, no full feature file, no full score file
is ever created. Peak memory is bounded by (a) the blocking indexes,
which are already frequency-capped by the existing 4-channel design,
and (b) one batch's worth of candidate pairs (batch-size dependent,
NOT test-scale dependent).

WHAT WAS DELIBERATELY KEPT UNCHANGED
-------------------------------------
- The four-channel blocking design and its frequency thresholds
  (2500 / 1000 / 50), copied verbatim from build_test_candidates.py.
- The exact pairwise feature logic (all 40 features), copied verbatim
  from build_test_features.py / build_full_pair_features.py.
- The frozen model (data/processed/final_match_classifier.joblib) and
  the frozen 0.30 threshold. Nothing here retrains or re-tunes.
- source_is_s3 encoding (S3=1, S2=0).

Two distinct normalize_text() functions exist in the original codebase
for two different purposes and are BOTH preserved exactly, under
distinct names so they can never be accidentally swapped:

    blocking_normalize_text()  - used for blocking keys (Unicode-aware,
                                  strips non-word chars, keeps digits
                                  and letters in any script). Copied
                                  verbatim from build_test_candidates.py.
    feature_normalize_text()   - used for pairwise text features
                                  (ASCII a-z0-9 only). Copied verbatim
                                  from build_test_features.py /
                                  build_full_pair_features.py.

INPUT:
    <data-dir>/test_source1.tsv
    <data-dir>/test_source2.tsv
    <data-dir>/test_source3.tsv
    <model-path> (default data/processed/final_match_classifier.joblib)

OUTPUT (full run):
    <output-dir>/matching_results.tsv
        source1_entity_id \t matched_entity_ids
    <output-dir>/candidate_pairs.tsv
        source1_entity_id \t candidate_entity_ids

OUTPUT (smoke test, --max-s1 N):
    <output-dir>/smoke_test/matching_results.tsv
    <output-dir>/smoke_test/candidate_pairs.tsv
    (final files in <output-dir>/ are never touched)

Both final-run files are written to temp paths and only atomically
os.replace()'d into their final names after the run completes without
error, so a crash never leaves a partial file at the final path.

USAGE:
    # smoke test (real data, real model, real logic, tiny slice)
    python run_test_inference.py --max-s1 1000

    # full run
    python run_test_inference.py --full

    # explicit paths / batch size
    python run_test_inference.py \\
        --data-dir data/test \\
        --model-path data/processed/final_match_classifier.joblib \\
        --output-dir output \\
        --batch-size 1000 \\
        --full
"""

import os
import re
import sys
import time
import sqlite3
import argparse
import itertools
import tempfile
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import joblib

from rapidfuzz.fuzz import ratio, token_sort_ratio, partial_ratio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from submission_checks import validate_outputs, print_report  # noqa: E402


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Frozen per the project brief. Never tuned on test data, never changed
# based on test predictions.
FROZEN_THRESHOLD = 0.30

# Same frequency caps as the validated 4-channel blocker.
BROAD_ADDRESS_FREQ = 2500       # Channel 2
NAME_PAIR_FREQ = 1000           # Channel 3
ADDRESS_PAIR_FREQ = 50          # Channel 4

# Rows read at a time while scanning the (large) S2/S3 test files for
# frequency counting / index building / filtered lookups. Independent
# of --batch-size, which controls S1 batching.
SCAN_CHUNK_SIZE = 250_000

DEFAULT_S1_BATCH_SIZE = 1_000

# ------------------------------------------------------------------
# MEMORY-AUDIT FOLLOW-UP (see module docstring section further down
# and the accompanying INDEX_MEMORY_AUDIT.md for the full writeup):
#
# The four blocking channels below are unchanged - same keys, same
# frequency caps. What changed is WHERE their posting lists live.
# They used to live in four `defaultdict(set)` per source (8 Python
# dict-of-sets total), fully in RAM for the whole run. At full test
# scale that in-RAM structure does not reliably fit in a Colab-class
# runtime (see audit). It is now a single on-disk SQLite database
# (`postings` table below), which is disk-backed by construction -
# SQLite only pages in what a query touches, so RAM use is governed
# by SQLite's own bounded page cache, not by total posting count.
# ------------------------------------------------------------------

CHANNEL_EXACT_NAME = 1     # Channel 1 - country + exact normalized name
CHANNEL_ADDRESS = 2        # Channel 2 - country + meaningful address token
CHANNEL_NAME_PAIR = 3      # Channel 3 - country + meaningful name-token pair
CHANNEL_ADDRESS_PAIR = 4   # Channel 4 - country + meaningful address-token pair

# Separator used to encode a (token_a, token_b) pair as a single SQLite
# TEXT key. 0x1F (ASCII "unit separator") cannot appear in a pair's
# tokens, since blocking_normalize_text() strips everything except
# \w and collapses whitespace, so this encoding is lossless and
# collision-free.
PAIR_KEY_SEP = "\x1f"


def encode_pair(pair):
    """(token_a, token_b) -> single string key for the on-disk index.
    `pair` is already sorted by token_pairs() via
    itertools.combinations(sorted(tokens), 2), so encoding is stable."""
    return PAIR_KEY_SEP.join(pair)


# ============================================================
# BLOCKING NORMALIZATION - copied verbatim from
# build_test_candidates.py. DO NOT change.
# ============================================================

def blocking_normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


GENERIC_NAME_TOKENS = {
    "private", "limited", "pvt", "ltd", "llc", "inc", "incorporated",
    "corp", "corporation", "company", "companies", "co", "plc",
    "llp", "lp", "group", "groups", "holdings", "holding",
    "partners", "partner", "services", "service", "solutions",
    "solution", "enterprises", "enterprise", "international", "global",
}

GENERIC_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd",
    "lane", "ln", "drive", "dr", "parkway", "pkwy", "highway", "hwy",
    "way", "place", "pl", "court", "ct", "circle", "cir", "terrace",
    "ter", "square", "sq", "building", "bldg", "floor", "fl", "unit",
    "suite", "ste", "room", "rm", "block", "blk", "sector",
    "near", "opposite",
}


def meaningful_name_tokens(text):
    normalized = blocking_normalize_text(text)
    return {
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in GENERIC_NAME_TOKENS
    }


def meaningful_address_tokens(text):
    normalized = blocking_normalize_text(text)
    return {
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in GENERIC_ADDRESS_TOKENS
    }


def token_pairs(tokens):
    if len(tokens) < 2:
        return set()
    return set(itertools.combinations(sorted(tokens), 2))


# ============================================================
# FEATURE NORMALIZATION - copied verbatim from
# build_test_features.py / build_full_pair_features.py. DO NOT change.
# This is intentionally a DIFFERENT function from
# blocking_normalize_text() above - that inconsistency already existed
# between the two original dev scripts and is preserved as-is, per the
# instruction not to invent different normalization rules.
# ============================================================

def feature_normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def feature_tokenize(value):
    if not value:
        return set()
    return set(value.split())


def extract_numbers(value):
    if not value:
        return set()
    return set(re.findall(r"\d+", value))


def safe_ratio(a, b):
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return ratio(a, b)


def safe_token_ratio(a, b):
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return token_sort_ratio(a, b)


def safe_partial_ratio(a, b):
    if not a or not b:
        return 0.0
    return partial_ratio(a, b)


def length_ratio(a, b):
    la = len(a)
    lb = len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def shared_token_count(a, b):
    if not a or not b:
        return 0
    return len(a.intersection(b))


def shared_number_count(a, b):
    if not a or not b:
        return 0
    return len(a.intersection(b))


def compute_pair_features(s1_id, candidate_id, source, s1_record, candidate_record):
    """One pair's feature dict. Logic copied verbatim (row body only,
    restructured out of the chunk loop) from build_test_features.py's
    build_features(). Do not change the feature logic itself."""

    s1_name = s1_record["business_name"]
    c_name = candidate_record["business_name"]

    s1_address = s1_record["business_address"]
    c_address = candidate_record["business_address"]

    s1_country = s1_record["country"]
    c_country = candidate_record["country"]

    n1 = feature_normalize_text(s1_name)
    n2 = feature_normalize_text(c_name)

    a1 = feature_normalize_text(s1_address)
    a2 = feature_normalize_text(c_address)

    name_tokens_1 = feature_tokenize(n1)
    name_tokens_2 = feature_tokenize(n2)

    address_tokens_1 = feature_tokenize(a1)
    address_tokens_2 = feature_tokenize(a2)

    numbers_1 = extract_numbers(a1)
    numbers_2 = extract_numbers(a2)

    country_match = int(str(s1_country).lower() == str(c_country).lower())

    name_exact = int(bool(n1) and bool(n2) and n1 == n2)

    name_ratio = safe_ratio(n1, n2)
    name_token_ratio = safe_token_ratio(n1, n2)
    name_partial_ratio = safe_partial_ratio(n1, n2)
    name_length_ratio = length_ratio(n1, n2)

    shared_name_tokens = shared_token_count(name_tokens_1, name_tokens_2)

    address_exact = int(bool(a1) and bool(a2) and a1 == a2)

    address_ratio = safe_ratio(a1, a2)
    address_token_ratio = safe_token_ratio(a1, a2)
    address_partial_ratio = safe_partial_ratio(a1, a2)
    address_length_ratio = length_ratio(a1, a2)

    shared_address_tokens = shared_token_count(address_tokens_1, address_tokens_2)

    shared_numbers = shared_number_count(numbers_1, numbers_2)

    name_strong_90 = int(name_token_ratio >= 90)
    address_strong_90 = int(address_token_ratio >= 90)

    name_strong_80 = int(name_token_ratio >= 80)
    address_strong_80 = int(address_token_ratio >= 80)

    both_strong_90 = int(name_strong_90 and address_strong_90)
    both_strong_80 = int(name_strong_80 and address_strong_80)

    name_strong_address_weak = int(name_strong_90 and address_token_ratio < 70)
    address_strong_name_weak = int(address_strong_90 and name_token_ratio < 70)

    name_address_mean = (name_token_ratio + address_token_ratio) / 200.0
    name_address_min = min(name_token_ratio, address_token_ratio) / 100.0
    name_address_max = max(name_token_ratio, address_token_ratio) / 100.0
    name_address_gap = abs(name_token_ratio - address_token_ratio) / 100.0
    name_address_product = (name_token_ratio * address_token_ratio) / 10000.0

    exact_name_strong_address = int(name_exact and address_token_ratio >= 80)
    exact_address_strong_name = int(address_exact and name_token_ratio >= 80)

    name_tokens_strong = int(name_token_ratio >= 90)
    address_tokens_strong = int(address_token_ratio >= 80)
    address_tokens_very_strong = int(address_token_ratio >= 90)

    has_shared_number = int(shared_numbers >= 1)
    multiple_shared_numbers = int(shared_numbers >= 2)

    strong_address_with_number = int(address_token_ratio >= 90 and shared_numbers >= 1)
    very_strong_address_with_number = int(address_token_ratio >= 95 and shared_numbers >= 1)
    strong_name_with_number = int(name_token_ratio >= 90 and shared_numbers >= 1)

    strong_evidence_count = (
        int(name_token_ratio >= 90)
        + int(address_token_ratio >= 80)
        + int(shared_address_tokens >= 3)
        + int(shared_name_tokens >= 2)
        + int(shared_numbers >= 1)
    )

    very_strong_evidence_count = (
        int(name_token_ratio >= 95)
        + int(address_token_ratio >= 90)
        + int(shared_address_tokens >= 5)
        + int(shared_numbers >= 2)
        + int(name_exact)
    )

    return {
        "source1_entity_id": s1_id,
        "candidate_entity_id": candidate_id,
        "source": source,

        "country_match": country_match,

        "name_exact": name_exact,
        "name_ratio": name_ratio,
        "name_token_ratio": name_token_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_length_ratio": name_length_ratio,
        "shared_name_tokens": shared_name_tokens,

        "address_exact": address_exact,
        "address_ratio": address_ratio,
        "address_token_ratio": address_token_ratio,
        "address_partial_ratio": address_partial_ratio,
        "address_length_ratio": address_length_ratio,
        "shared_address_tokens": shared_address_tokens,
        "shared_numbers": shared_numbers,

        "name_strong_90": name_strong_90,
        "address_strong_90": address_strong_90,
        "name_strong_80": name_strong_80,
        "address_strong_80": address_strong_80,
        "both_strong_90": both_strong_90,
        "both_strong_80": both_strong_80,

        "name_strong_address_weak": name_strong_address_weak,
        "address_strong_name_weak": address_strong_name_weak,

        "name_address_mean": name_address_mean,
        "name_address_min": name_address_min,
        "name_address_max": name_address_max,
        "name_address_gap": name_address_gap,
        "name_address_product": name_address_product,

        "exact_name_strong_address": exact_name_strong_address,
        "exact_address_strong_name": exact_address_strong_name,

        "name_tokens_strong": name_tokens_strong,
        "address_tokens_strong": address_tokens_strong,
        "address_tokens_very_strong": address_tokens_very_strong,

        "has_shared_number": has_shared_number,
        "multiple_shared_numbers": multiple_shared_numbers,

        "strong_address_with_number": strong_address_with_number,
        "very_strong_address_with_number": very_strong_address_with_number,
        "strong_name_with_number": strong_name_with_number,

        "strong_evidence_count": strong_evidence_count,
        "very_strong_evidence_count": very_strong_evidence_count,

        "source_is_s3": int(source == "S3"),
    }


# ============================================================
# PHASE 1 - load test S1 (small: ~1.7M rows worst case) and
# build blocking keys + vocab. Copied/adapted from
# build_test_candidates.py's load_s1_and_build_keys().
# ============================================================

def load_s1_and_build_keys(s1_path, max_s1):

    print("1. Loading test S1 and building blocking keys...", flush=True)

    read_kwargs = dict(sep="\t", dtype=str, keep_default_na=False)
    if max_s1 is not None:
        read_kwargs["nrows"] = max_s1

    s1 = pd.read_csv(s1_path, **read_kwargs)

    print(f"   Test S1 rows loaded: {len(s1):,}"
          + (f"  (--max-s1 {max_s1} applied)" if max_s1 is not None else ""),
          flush=True)

    s1_data = {}
    s1_feature_lookup = {}

    all_exact_names = set()
    all_address_tokens = set()
    all_name_tokens = set()
    all_name_pairs = set()
    all_address_pairs = set()

    for row in s1.itertuples(index=False):

        s1_id = row.entity_id
        country = row.country

        s1_feature_lookup[s1_id] = {
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": country,
        }

        name = blocking_normalize_text(row.business_name)
        address_tokens = meaningful_address_tokens(row.business_address)
        name_tokens = meaningful_name_tokens(name)

        name_pairs = token_pairs(name_tokens)
        address_pairs = token_pairs(address_tokens)

        s1_data[s1_id] = {
            "country": country,
            "name": name,
            "address_tokens": address_tokens,
            "name_pairs": name_pairs,
            "address_pairs": address_pairs,
        }

        if name:
            all_exact_names.add(name)
        all_address_tokens.update(address_tokens)
        all_name_tokens.update(name_tokens)
        all_name_pairs.update(name_pairs)
        all_address_pairs.update(address_pairs)

    print(f"   Relevant exact names        : {len(all_exact_names):,}", flush=True)
    print(f"   Relevant address tokens     : {len(all_address_tokens):,}", flush=True)
    print(f"   Relevant name tokens        : {len(all_name_tokens):,}", flush=True)
    print(f"   Relevant name-token pairs   : {len(all_name_pairs):,}", flush=True)
    print(f"   Relevant address-token pairs: {len(all_address_pairs):,}", flush=True)

    vocab = {
        "exact_names": all_exact_names,
        "address_tokens": all_address_tokens,
        "name_tokens": all_name_tokens,
        "name_pairs": all_name_pairs,
        "address_pairs": all_address_pairs,
    }

    return list(s1["entity_id"]), s1_data, s1_feature_lookup, vocab


# ============================================================
# PHASE 2 - on-disk (SQLite) two-pass frequency counting + index
# building over S2/S3, restricted to the S1-derived vocab.
#
# This replaces the former all-in-RAM `defaultdict(set)` posting
# indexes (4 channels x 2 sources) and the former all-in-RAM
# Counter() frequency tables. The two-pass STRUCTURE and all
# blocking/frequency LOGIC are unchanged - see INDEX_MEMORY_AUDIT.md
# for why the storage layer had to change. Same as before, this also
# collects the set of entity ids that ever land in a posting, so
# Phase 3 can build FILTERED (not full-file) feature lookups for
# S2/S3 (that part was already memory-safe and is untouched).
# ============================================================

def open_index_db(db_path):
    """Create a fresh on-disk SQLite database to hold the blocking
    frequency table and posting index. Disk-backed by construction:
    SQLite pages data to/from `db_path` through its own bounded page
    cache (set below), so this never requires holding the full
    posting set in Python objects."""

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    # This DB is a disposable intermediate build artifact (rebuilt
    # from test_source2/3.tsv every run), not a record of truth, so
    # durability/crash-safety is deliberately traded for build speed.
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -262144")   # ~256MB page cache cap

    conn.execute(
        "CREATE TABLE freq ("
        " channel INTEGER NOT NULL,"
        " country TEXT NOT NULL,"
        " key TEXT NOT NULL,"
        " cnt INTEGER NOT NULL,"
        " PRIMARY KEY (channel, country, key)"
        ")"
    )
    conn.execute(
        "CREATE TABLE postings ("
        " channel INTEGER NOT NULL,"
        " source TEXT NOT NULL,"
        " country TEXT NOT NULL,"
        " key TEXT NOT NULL,"
        " entity_id TEXT NOT NULL"
        ")"
    )
    conn.commit()
    return conn


def _flush_freq_counter(conn, channel, counter):
    """Merge one chunk's local Counter into the on-disk `freq` table
    and discard the Python Counter. Counter() itself never exceeds
    one chunk's worth of distinct keys (bounded by SCAN_CHUNK_SIZE),
    unlike the original single process-lifetime Counter that grew to
    full-corpus-vocab size."""
    if not counter:
        return
    conn.executemany(
        "INSERT INTO freq(channel, country, key, cnt) VALUES (?,?,?,?) "
        "ON CONFLICT(channel, country, key) "
        "DO UPDATE SET cnt = cnt + excluded.cnt",
        [(channel, country, key, cnt) for (country, key), cnt in counter.items()],
    )
    conn.commit()


def scan_and_store_frequencies(conn, source_files, vocab):
    """Pass 1 of 2. Same traversal and same restriction to
    vocab-relevant keys as the original scan_frequencies(), but
    counts land in the on-disk `freq` table instead of a growing
    in-RAM Counter().

    Channel 1 (exact name) needs no frequency table at all: the
    original code's `if exact_name_freq[key] > 0` check in the
    indexing pass is always true once a name has been seen once (it
    is vocab membership, not a real threshold - there is no
    EXACT_NAME_FREQ cap anywhere in the frozen blocking design), so
    it is handled as pure membership in Phase 2's second pass below,
    with no counting step needed."""

    print("\n2. Scanning test S2 + S3 to build the on-disk frequency "
          "table (channels 2/3/4; channel 1 needs no frequency pass "
          "- see comment above)...", flush=True)

    for source_name, path in source_files.items():

        print(f"   Scanning {source_name}...", flush=True)
        rows = 0

        for chunk in pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False,
            chunksize=SCAN_CHUNK_SIZE,
        ):
            address_freq = Counter()
            name_pair_freq = Counter()
            address_pair_freq = Counter()

            for row in chunk.itertuples(index=False):

                country = row.country
                name = blocking_normalize_text(row.business_name)

                address_tokens = (
                    meaningful_address_tokens(row.business_address)
                    & vocab["address_tokens"]
                )
                for token in address_tokens:
                    address_freq[(country, token)] += 1

                name_tokens = meaningful_name_tokens(name) & vocab["name_tokens"]
                for pair in token_pairs(name_tokens):
                    if pair in vocab["name_pairs"]:
                        name_pair_freq[(country, encode_pair(pair))] += 1

                for pair in token_pairs(address_tokens):
                    if pair in vocab["address_pairs"]:
                        address_pair_freq[(country, encode_pair(pair))] += 1

            _flush_freq_counter(conn, CHANNEL_ADDRESS, address_freq)
            _flush_freq_counter(conn, CHANNEL_NAME_PAIR, name_pair_freq)
            _flush_freq_counter(conn, CHANNEL_ADDRESS_PAIR, address_pair_freq)
            del address_freq, name_pair_freq, address_pair_freq

            rows += len(chunk)
            print(f"     {source_name}: {rows:,} rows scanned...", flush=True)

    conn.execute("CREATE INDEX idx_freq_lookup ON freq(channel, country, key)")
    conn.commit()

    n_freq_rows = conn.execute("SELECT COUNT(*) FROM freq").fetchone()[0]
    print(f"   On-disk frequency rows (channels 2+3+4): {n_freq_rows:,}",
          flush=True)
    return n_freq_rows


def _lookup_chunk_freq(conn, channel, keys):
    """Batched replacement for a Counter[key] read: given the
    (country, key) pairs actually needed by ONE chunk, fetch their
    counts from the on-disk `freq` table in a single join and return
    them as a small chunk-scoped Python dict. Never touches more than
    one chunk's worth of keys at a time."""
    keys = list(keys)
    if not keys:
        return {}
    conn.execute("DROP TABLE IF EXISTS temp.chunk_keys")
    conn.execute("CREATE TEMP TABLE chunk_keys(country TEXT, key TEXT)")
    conn.executemany("INSERT INTO chunk_keys VALUES (?,?)", keys)
    cur = conn.execute(
        "SELECT f.country, f.key, f.cnt FROM freq f "
        "JOIN chunk_keys c ON f.country = c.country AND f.key = c.key "
        "WHERE f.channel = ?",
        (channel,),
    )
    result = {(c, k): cnt for c, k, cnt in cur.fetchall()}
    conn.execute("DROP TABLE chunk_keys")
    return result


def build_indexes_on_disk(conn, source_files, vocab):
    """Pass 2 of 2. Same per-row channel logic and the same three
    frequency thresholds (BROAD_ADDRESS_FREQ / NAME_PAIR_FREQ /
    ADDRESS_PAIR_FREQ) as the original build_indexes_per_source(),
    but qualifying (channel, source, country, key, entity_id) rows
    are written to the on-disk `postings` table instead of being
    added to an in-RAM `defaultdict(set)`. Only one chunk's worth of
    rows is ever held in Python objects at a time."""

    print("\n3. Building the on-disk posting index (`postings` "
          "table)...", flush=True)

    needed_ids = {}

    for source_name, path in source_files.items():

        print(f"   Indexing {source_name}...", flush=True)
        rows = 0
        source_needed = set()

        for chunk in pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False,
            chunksize=SCAN_CHUNK_SIZE,
        ):
            parsed_rows = []
            addr_keys_needed = set()
            name_pair_keys_needed = set()
            addr_pair_keys_needed = set()

            for row in chunk.itertuples(index=False):

                entity_id = row.entity_id
                country = row.country
                name = blocking_normalize_text(row.business_name)

                address_tokens = (
                    meaningful_address_tokens(row.business_address)
                    & vocab["address_tokens"]
                )
                name_tokens = meaningful_name_tokens(name) & vocab["name_tokens"]
                name_pairs_here = {
                    p for p in token_pairs(name_tokens) if p in vocab["name_pairs"]
                }
                addr_pairs_here = {
                    p for p in token_pairs(address_tokens)
                    if p in vocab["address_pairs"]
                }

                parsed_rows.append((
                    entity_id, country, name,
                    address_tokens, name_pairs_here, addr_pairs_here,
                ))

                for token in address_tokens:
                    addr_keys_needed.add((country, token))
                for pair in name_pairs_here:
                    name_pair_keys_needed.add((country, encode_pair(pair)))
                for pair in addr_pairs_here:
                    addr_pair_keys_needed.add((country, encode_pair(pair)))

            addr_freq = _lookup_chunk_freq(conn, CHANNEL_ADDRESS, addr_keys_needed)
            name_pair_freq = _lookup_chunk_freq(
                conn, CHANNEL_NAME_PAIR, name_pair_keys_needed
            )
            addr_pair_freq = _lookup_chunk_freq(
                conn, CHANNEL_ADDRESS_PAIR, addr_pair_keys_needed
            )
            del addr_keys_needed, name_pair_keys_needed, addr_pair_keys_needed

            posting_rows = []
            for (entity_id, country, name, address_tokens,
                 name_pairs_here, addr_pairs_here) in parsed_rows:

                if name and name in vocab["exact_names"]:
                    # No frequency cap on this channel - see the
                    # docstring on scan_and_store_frequencies().
                    posting_rows.append(
                        (CHANNEL_EXACT_NAME, source_name, country, name, entity_id)
                    )
                    source_needed.add(entity_id)

                for token in address_tokens:
                    if addr_freq.get((country, token), 0) <= BROAD_ADDRESS_FREQ:
                        posting_rows.append(
                            (CHANNEL_ADDRESS, source_name, country, token, entity_id)
                        )
                        source_needed.add(entity_id)

                for pair in name_pairs_here:
                    enc = encode_pair(pair)
                    if name_pair_freq.get((country, enc), 0) <= NAME_PAIR_FREQ:
                        posting_rows.append(
                            (CHANNEL_NAME_PAIR, source_name, country, enc, entity_id)
                        )
                        source_needed.add(entity_id)

                for pair in addr_pairs_here:
                    enc = encode_pair(pair)
                    if addr_pair_freq.get((country, enc), 0) <= ADDRESS_PAIR_FREQ:
                        posting_rows.append(
                            (CHANNEL_ADDRESS_PAIR, source_name, country, enc, entity_id)
                        )
                        source_needed.add(entity_id)

            if posting_rows:
                conn.executemany(
                    "INSERT INTO postings"
                    "(channel, source, country, key, entity_id) "
                    "VALUES (?,?,?,?,?)",
                    posting_rows,
                )
                conn.commit()
            del parsed_rows, posting_rows, addr_freq, name_pair_freq, addr_pair_freq

            rows += len(chunk)
            if rows % (SCAN_CHUNK_SIZE * 4) == 0:
                print(f"     {source_name}: {rows:,} rows indexed...", flush=True)

        print(f"     {source_name}: {rows:,} rows indexed (done)", flush=True)

        needed_ids[source_name] = source_needed
        print(f"     {source_name}: {len(source_needed):,} distinct "
              f"entities appear in at least one index (candidate-eligible)",
              flush=True)

    print("   Building lookup index on postings(channel, country, "
          "key)...", flush=True)
    conn.execute("CREATE INDEX idx_postings_lookup ON postings(channel, country, key)")
    conn.commit()

    return needed_ids


def report_index_stats(conn, db_path):
    """Instrumentation requested for the memory audit: on-disk index
    size, row/posting counts. Printed once, right after the index is
    built and before Phase 3 (candidate generation) begins."""

    n_postings = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
    n_freq_rows = conn.execute("SELECT COUNT(*) FROM freq").fetchone()[0]
    try:
        disk_bytes = os.path.getsize(db_path)
    except OSError:
        disk_bytes = -1

    print("\n   --- Index storage instrumentation ---")
    print(f"   Posting entries (rows in `postings`)  : {n_postings:,}")
    print(f"   Frequency-table rows (`freq`)         : {n_freq_rows:,}")
    if disk_bytes >= 0:
        print(f"   Index DB file size on disk            : "
              f"{disk_bytes / (1024 ** 3):.3f} GB ({disk_bytes:,} bytes)")
    else:
        print("   Index DB file size on disk            : unavailable")
    print("   --------------------------------------\n")

    return {
        "n_postings": n_postings,
        "n_freq_rows": n_freq_rows,
        "disk_bytes": disk_bytes,
    }


def build_filtered_feature_lookup(path, needed_ids):
    """Load ONLY the rows whose entity_id is candidate-eligible
    (appears in at least one blocking index), instead of the entire
    S2/S3 file. Bounded by the (already frequency-capped) index size,
    not by the raw file size."""

    lookup = {}
    rows = 0

    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        chunksize=SCAN_CHUNK_SIZE,
    ):
        mask = chunk["entity_id"].isin(needed_ids)
        selected = chunk[mask]
        for row in selected.itertuples(index=False):
            lookup[row.entity_id] = {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }
        rows += len(chunk)

    return lookup


# ============================================================
# PHASE 3 - batched candidate generation + feature computation +
# scoring + incremental output. This is the memory-safety fix: only
# ONE batch's worth of candidate pairs ever exists in memory, and
# nothing per-pair is ever written to disk.
# ============================================================

def find_candidates_for_batch(conn, batch_ids, s1_data):
    """Batched, on-disk-index replacement for the old
    find_candidates_for_s1(). Produces the exact same result shape -
    {s1_id: {candidate_id: source}} for every s1_id in the batch,
    unioned over the same four channels - but sources it from the
    `postings` SQLite table via one indexed JOIN query per channel
    (4 queries total per batch) instead of a Python dict .get() per
    key against a fully-materialized in-RAM index.

    Because the join is scoped to "keys this batch actually needs",
    the amount of data pulled from disk per call is bounded by this
    one batch's candidates, matching the existing memory-safety
    design (batch-scale, not test-scale, memory)."""

    found = {s1_id: {} for s1_id in batch_ids}

    # channel -> {(country, key): [s1_id, s1_id, ...]}
    channel_reverse = {
        CHANNEL_EXACT_NAME: defaultdict(list),
        CHANNEL_ADDRESS: defaultdict(list),
        CHANNEL_NAME_PAIR: defaultdict(list),
        CHANNEL_ADDRESS_PAIR: defaultdict(list),
    }

    for s1_id in batch_ids:
        data = s1_data[s1_id]
        country = data["country"]

        if data["name"]:
            channel_reverse[CHANNEL_EXACT_NAME][(country, data["name"])].append(s1_id)
        for token in data["address_tokens"]:
            channel_reverse[CHANNEL_ADDRESS][(country, token)].append(s1_id)
        for pair in data["name_pairs"]:
            key = (country, encode_pair(pair))
            channel_reverse[CHANNEL_NAME_PAIR][key].append(s1_id)
        for pair in data["address_pairs"]:
            key = (country, encode_pair(pair))
            channel_reverse[CHANNEL_ADDRESS_PAIR][key].append(s1_id)

    for channel, reverse_map in channel_reverse.items():
        if not reverse_map:
            continue

        conn.execute("DROP TABLE IF EXISTS temp.batch_keys")
        conn.execute("CREATE TEMP TABLE batch_keys(country TEXT, key TEXT)")
        conn.executemany(
            "INSERT INTO batch_keys VALUES (?,?)", list(reverse_map.keys())
        )

        cur = conn.execute(
            "SELECT p.country, p.key, p.entity_id, p.source FROM postings p "
            "JOIN batch_keys b ON p.country = b.country AND p.key = b.key "
            "WHERE p.channel = ?",
            (channel,),
        )
        for country, key, entity_id, source in cur.fetchall():
            for s1_id in reverse_map[(country, key)]:
                found[s1_id][entity_id] = source

        conn.execute("DROP TABLE batch_keys")

    return found


class IncrementalStats:

    def __init__(self):
        self.n_s1 = 0
        self.n_s1_zero_candidates = 0
        self.total_candidate_pairs = 0
        self.max_candidates = 0
        self.n_predicted_matches = 0
        self.n_s1_zero_predictions = 0
        self.peak_batch_candidates = 0

    def update(self, n_candidates, n_matches):
        self.n_s1 += 1
        if n_candidates == 0:
            self.n_s1_zero_candidates += 1
        self.total_candidate_pairs += n_candidates
        self.max_candidates = max(self.max_candidates, n_candidates)
        self.n_predicted_matches += n_matches
        if n_matches == 0:
            self.n_s1_zero_predictions += 1

    def update_batch(self, batch_candidate_count):
        self.peak_batch_candidates = max(
            self.peak_batch_candidates, batch_candidate_count
        )

    def report(self, elapsed_seconds):
        avg_candidates = (
            self.total_candidate_pairs / self.n_s1 if self.n_s1 else 0.0
        )
        avg_matches = self.n_predicted_matches / self.n_s1 if self.n_s1 else 0.0
        print()
        print("=" * 70)
        print("RUN STATISTICS")
        print("=" * 70)
        print(f"S1 processed                  : {self.n_s1:,}")
        print(f"S1 with zero candidates        : {self.n_s1_zero_candidates:,}")
        print(f"Total candidate pairs processed: {self.total_candidate_pairs:,}")
        print(f"Average candidates per S1      : {avg_candidates:.2f}")
        print(f"Maximum candidates for one S1   : {self.max_candidates:,}")
        print(f"Peak candidates in a single batch: {self.peak_batch_candidates:,}")
        print(f"Total predicted matches        : {self.n_predicted_matches:,}")
        print(f"Average predicted matches/S1    : {avg_matches:.4f}")
        print(f"S1 with zero predictions        : {self.n_s1_zero_predictions:,}")
        print(f"Runtime                         : {elapsed_seconds / 60:.2f} minutes")


def process_and_write(
    s1_ids, s1_data, s1_feature_lookup, conn,
    s2_lookup, s3_lookup, model, feature_columns, threshold,
    batch_size, matching_out, candidates_out, stats, log_every_s1=20_000,
):
    """Iterate S1 in batches; for each batch, generate candidates,
    compute features for just that batch, score, and write ONE
    aggregated row per S1 to each open output file handle."""

    n = len(s1_ids)

    for batch_start in range(0, n, batch_size):

        batch_ids = s1_ids[batch_start:batch_start + batch_size]

        # 1. candidate generation for this batch only, sourced from
        # the on-disk posting index (see find_candidates_for_batch).
        batch_found = find_candidates_for_batch(conn, batch_ids, s1_data)
        pair_records = []  # flat list for this batch's feature frame

        for s1_id in batch_ids:
            for candidate_id, source in batch_found[s1_id].items():
                pair_records.append((s1_id, candidate_id, source))

        stats.update_batch(len(pair_records))

        # 2. feature computation for this batch only (transient)
        matches_by_s1 = {s1_id: set() for s1_id in batch_ids}

        if pair_records:

            feature_rows = []
            for s1_id, candidate_id, source in pair_records:
                candidate_record = (
                    s2_lookup.get(candidate_id) if source == "S2"
                    else s3_lookup.get(candidate_id)
                )
                if candidate_record is None:
                    # Should not happen: candidate_id came from an
                    # index built off the same source file. Guard
                    # anyway rather than crash a multi-hour run.
                    continue
                feature_rows.append(
                    compute_pair_features(
                        s1_id, candidate_id, source,
                        s1_feature_lookup[s1_id], candidate_record,
                    )
                )

            if feature_rows:

                features_df = pd.DataFrame(feature_rows)

                missing = [c for c in feature_columns if c not in features_df.columns]
                if missing:
                    raise ValueError(
                        f"Computed features are missing model feature "
                        f"columns: {missing}. The feature logic no "
                        f"longer matches final_match_classifier.joblib."
                    )

                X = features_df[feature_columns].astype(np.float32)
                scores = model.predict_proba(X)[:, 1]

                accepted = features_df.loc[
                    scores >= threshold, ["source1_entity_id", "candidate_entity_id"]
                ]
                for row in accepted.itertuples(index=False):
                    matches_by_s1[row.source1_entity_id].add(row.candidate_entity_id)

                del features_df, X, scores, feature_rows

        # 3. write this batch's aggregated rows and update stats
        for s1_id in batch_ids:
            candidate_ids = sorted(batch_found[s1_id].keys())
            matched_ids = sorted(matches_by_s1[s1_id])

            candidates_out.write(f"{s1_id}\t{','.join(candidate_ids)}\n")
            matching_out.write(f"{s1_id}\t{','.join(matched_ids)}\n")

            stats.update(len(candidate_ids), len(matched_ids))

        # 4. release batch memory explicitly before continuing
        del batch_found, pair_records, matches_by_s1

        if stats.n_s1 % log_every_s1 < batch_size:
            print(
                f"   {stats.n_s1:,}/{n:,} S1 processed | "
                f"{stats.total_candidate_pairs:,} candidate pairs so far | "
                f"{stats.n_predicted_matches:,} matches so far",
                flush=True,
            )


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Fused, memory-safe production test inference pipeline."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(BASE_DIR, "data", "test"),
        help="Directory containing test_source1/2/3.tsv "
             "(default: data/test)",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(
            BASE_DIR, "data", "processed", "final_match_classifier.joblib"
        ),
        help="Path to the frozen trained classifier joblib "
             "(default: data/processed/final_match_classifier.joblib)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_DIR, "output"),
        help="Directory for matching_results.tsv / candidate_pairs.tsv "
             "(default: output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_S1_BATCH_SIZE,
        help=f"Number of S1 entities processed per in-memory batch "
             f"(default: {DEFAULT_S1_BATCH_SIZE}). Controls the only "
             f"unbounded-with-scale memory in this pipeline (one "
             f"batch's candidate pairs); does not need to grow with "
             f"the size of the test set.",
    )
    parser.add_argument(
        "--max-s1",
        type=int,
        default=None,
        help="Smoke-test mode: only process the first N test S1 rows, "
             "write to <output-dir>/smoke_test/, validate, and leave "
             "the final output files untouched. Mutually exclusive "
             "with --full.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full-run mode: process every test S1 row and write "
             "directly (atomically) to <output-dir>/. Mutually "
             "exclusive with --max-s1.",
    )
    parser.add_argument(
        "--use-model-threshold",
        action="store_true",
        help="Use the threshold stored inside the model package "
             "instead of the frozen 0.30. Only use this if you "
             "deliberately retrained and refroze a new threshold on "
             "TRAIN/VALIDATION data (never on test data).",
    )
    parser.add_argument(
        "--index-db-path",
        default=None,
        help="Path for the on-disk SQLite blocking index built from "
             "test S2/S3 (default: <output-dir>/_blocking_index.sqlite3). "
             "This replaces the old in-RAM posting indexes; see "
             "INDEX_MEMORY_AUDIT.md.",
    )
    parser.add_argument(
        "--keep-index-db",
        action="store_true",
        help="Do not delete the on-disk index DB after the run "
             "finishes (useful for inspecting index-size "
             "instrumentation or re-using the index across runs). "
             "By default it is removed on exit.",
    )

    args = parser.parse_args()

    if args.max_s1 is not None and args.full:
        parser.error("--max-s1 and --full are mutually exclusive. "
                      "Run a smoke test first, then --full separately.")
    if args.max_s1 is None and not args.full:
        parser.error("Specify either --max-s1 N (smoke test) or --full "
                      "(complete run).")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    return args


def main():

    args = parse_args()
    start = time.time()

    print("=" * 70)
    print("PRODUCTION TEST INFERENCE (fused, memory-safe)")
    print("=" * 70)

    s1_path = os.path.join(args.data_dir, "test_source1.tsv")
    s2_path = os.path.join(args.data_dir, "test_source2.tsv")
    s3_path = os.path.join(args.data_dir, "test_source3.tsv")

    for p in (s1_path, s2_path, s3_path):
        if not os.path.exists(p):
            print(f"ERROR: expected input file not found: {p}", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"ERROR: model file not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading frozen model: {args.model_path}", flush=True)
    package = joblib.load(args.model_path)
    model = package["model"]
    feature_columns = package["feature_columns"]
    stored_threshold = package.get("threshold")

    print(f"Model feature count: {len(feature_columns)}", flush=True)
    print(f"Threshold stored in model package: {stored_threshold}", flush=True)

    if args.use_model_threshold and stored_threshold is not None:
        threshold = float(stored_threshold)
    else:
        threshold = FROZEN_THRESHOLD

    if stored_threshold is not None and abs(float(stored_threshold) - threshold) > 1e-9:
        print(
            f"WARNING: using threshold {threshold}, which differs from "
            f"the threshold stored in the model package "
            f"({stored_threshold}). The frozen competition threshold "
            f"(0.30) is used unless --use-model-threshold was passed.",
            flush=True,
        )
    print(f"Using threshold: {threshold}", flush=True)

    source_files = {"S2": s2_path, "S3": s3_path}

    os.makedirs(args.output_dir, exist_ok=True)

    s1_ids, s1_data, s1_feature_lookup, vocab = load_s1_and_build_keys(
        s1_path, args.max_s1
    )

    # ------------------------------------------------------------
    # On-disk blocking index (replaces the old in-RAM
    # defaultdict(set) posting indexes + Counter() frequency
    # tables). See open_index_db() / INDEX_MEMORY_AUDIT.md.
    # ------------------------------------------------------------
    index_db_path = args.index_db_path or os.path.join(
        args.output_dir, "_blocking_index.sqlite3"
    )
    conn = open_index_db(index_db_path)
    try:
        scan_and_store_frequencies(conn, source_files, vocab)
        needed_ids = build_indexes_on_disk(conn, source_files, vocab)
        index_stats = report_index_stats(conn, index_db_path)

        print("\n4. Building filtered feature lookups for candidate-eligible "
              "S2/S3 entities only...", flush=True)
        s2_lookup = build_filtered_feature_lookup(s2_path, needed_ids["S2"])
        s3_lookup = build_filtered_feature_lookup(s3_path, needed_ids["S3"])
        print(f"   S2 filtered lookup: {len(s2_lookup):,} entities "
              f"(candidate-eligible only)", flush=True)
        print(f"   S3 filtered lookup: {len(s3_lookup):,} entities "
              f"(candidate-eligible only)", flush=True)
        del needed_ids

        _run_inference_body(
            args, conn, s1_ids, s1_data, s1_feature_lookup,
            s2_lookup, s3_lookup, model, feature_columns, threshold, start,
        )
    finally:
        conn.close()
        if args.keep_index_db:
            print(f"\n--keep-index-db set: leaving index DB at "
                  f"{index_db_path}", flush=True)
        else:
            try:
                os.remove(index_db_path)
            except OSError:
                pass


def _run_inference_body(
    args, conn, s1_ids, s1_data, s1_feature_lookup,
    s2_lookup, s3_lookup, model, feature_columns, threshold, start,
):
    if args.max_s1 is not None:
        # ------------------------------------------------------
        # SMOKE TEST: isolated subdirectory, final files untouched.
        # ------------------------------------------------------
        smoke_dir = os.path.join(args.output_dir, "smoke_test")
        os.makedirs(smoke_dir, exist_ok=True)
        matching_path = os.path.join(smoke_dir, "matching_results.tsv")
        candidates_path = os.path.join(smoke_dir, "candidate_pairs.tsv")

        print(f"\n5. SMOKE TEST - writing to {smoke_dir} "
              f"(final output/ files are NOT touched)", flush=True)

        stats = IncrementalStats()
        with open(matching_path, "w", encoding="utf-8") as matching_out, \
             open(candidates_path, "w", encoding="utf-8") as candidates_out:

            matching_out.write("source1_entity_id\tmatched_entity_ids\n")
            candidates_out.write("source1_entity_id\tcandidate_entity_ids\n")

            process_and_write(
                s1_ids, s1_data, s1_feature_lookup, conn,
                s2_lookup, s3_lookup, model, feature_columns, threshold,
                args.batch_size, matching_out, candidates_out, stats,
            )

        stats.report(time.time() - start)

        print("\n6. Validating smoke-test output...", flush=True)
        expected_ids = set(s1_ids)
        # Universe of valid ids: everything the (filtered) lookups
        # contain, since only candidate-eligible entities were loaded
        # for this smoke slice. This still exercises every check.
        valid_ids = set(s2_lookup.keys()) | set(s3_lookup.keys())
        passed, summary, problems = validate_outputs(
            matching_path, candidates_path, expected_ids, valid_ids
        )
        print_report(summary, problems)

        print(f"\nSmoke-test files written to:\n  {matching_path}\n  {candidates_path}")
        if not passed:
            print("\nSmoke test FAILED validation. Fix issues before --full.",
                  file=sys.stderr)
            sys.exit(1)
        print("\nSmoke test PASSED. Final output files were not touched.")

    else:
        # ------------------------------------------------------
        # FULL RUN: write to temp paths, atomic rename at the end.
        # ------------------------------------------------------
        matching_final = os.path.join(args.output_dir, "matching_results.tsv")
        candidates_final = os.path.join(args.output_dir, "candidate_pairs.tsv")
        matching_tmp = matching_final + ".partial"
        candidates_tmp = candidates_final + ".partial"

        print(f"\n5. FULL RUN - writing to temp files, atomic rename on "
              f"success:\n   {matching_tmp}\n   {candidates_tmp}", flush=True)

        stats = IncrementalStats()
        try:
            with open(matching_tmp, "w", encoding="utf-8") as matching_out, \
                 open(candidates_tmp, "w", encoding="utf-8") as candidates_out:

                matching_out.write("source1_entity_id\tmatched_entity_ids\n")
                candidates_out.write("source1_entity_id\tcandidate_entity_ids\n")

                process_and_write(
                    s1_ids, s1_data, s1_feature_lookup, conn,
                    s2_lookup, s3_lookup, model, feature_columns, threshold,
                    args.batch_size, matching_out, candidates_out, stats,
                )
        except BaseException:
            print(
                "\nERROR: run did not complete. Final output files were "
                "NOT written/updated. The incomplete .partial files are "
                f"left at:\n  {matching_tmp}\n  {candidates_tmp}\n"
                "for inspection; delete them and re-run when ready.",
                file=sys.stderr,
            )
            raise

        os.replace(matching_tmp, matching_final)
        os.replace(candidates_tmp, candidates_final)

        stats.report(time.time() - start)

        print(f"\nFinal files written:\n  {matching_final}\n  {candidates_final}")
        print("\nRun `validate_submission.py` next for full-scale validation "
              "against the official id universe.")

    print(f"\nTotal wall-clock time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
