"""
STEP T1 - PRODUCTION TEST CANDIDATE GENERATION (FOUR-CHANNEL BLOCKING)

This is the full-scale counterpart of the dev-scale blocking scripts
(evaluate_improved_blocking.py + evaluate_address_pair_blocking.py,
sanity-checked for volume in evaluate_test_blocking_scale.py).

It applies the SAME four-channel blocking strategy described in
MASTER_PROMPT.txt Section 7 to the FULL test set (not a 10k sample):

    Channel 1: country + exact normalized business name
    Channel 2: country + meaningful address token   (freq <= 2500)
    Channel 3: country + meaningful name-token pair  (freq <= 1000)
    Channel 4: country + meaningful address-token pair (freq <= 50)

Two reconciliations vs. the dev-scale scripts (both required by
MASTER_PROMPT.txt, treated as source of truth):

1. Channel 4 keys include country. In evaluate_address_pair_blocking.py
   (the dev-scale script), the address-pair frequency/index keys did NOT
   include country, while Section 7 explicitly specifies "country +
   meaningful address-token pair" for every channel, and
   evaluate_test_blocking_scale.py already does include country for
   channel 4. This script follows the master prompt and includes country
   for channel 4, consistent with channels 1-3.

2. The GENERIC_NAME_TOKENS / GENERIC_ADDRESS_TOKENS lists here are taken
   directly from MASTER_PROMPT.txt Section 6, reconciling small
   differences between the two dev-scale scripts that implemented
   channels 1-3 (evaluate_improved_blocking.py) and channel 4
   (evaluate_address_pair_blocking.py) with slightly different generic
   token sets.

Neither change alters the four-channel architecture, the frequency
limits, or the union logic. They only make the production run internally
consistent and consistent with the written spec.

INPUT (test set, inference-only):
    data/test/test_source1.tsv
    data/test/test_source2.tsv
    data/test/test_source3.tsv

OUTPUT:
    data/processed/candidate_pairs.tsv
        columns: source1_entity_id, candidate_entity_id, source
        (source is "S2" or "S3")

    This is "the candidate file containing the actual candidates
    passed to the classifier" required by MASTER_PROMPT.txt Section 24.

This script does NOT touch train_ground_truth.tsv, does NOT train
anything, and does NOT tune any threshold. It only generates candidates.
"""

import os
import re
import time
import itertools
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 250_000

BROAD_ADDRESS_FREQ = 2500      # Channel 2 limit (Section 7)
NAME_PAIR_FREQ = 1000          # Channel 3 limit (Section 7)
ADDRESS_PAIR_FREQ = 50         # Channel 4 limit (Section 7)

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

S1_PATH = os.path.join(BASE_DIR, "data", "test", "test_source1.tsv")
S2_PATH = os.path.join(BASE_DIR, "data", "test", "test_source2.tsv")
S3_PATH = os.path.join(BASE_DIR, "data", "test", "test_source3.tsv")

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "candidate_pairs.tsv"
)

SOURCE_FILES = {
    "S2": S2_PATH,
    "S3": S3_PATH,
}


# ============================================================
# NORMALIZATION (matches evaluate_improved_blocking.py /
# evaluate_address_pair_blocking.py blocking-key normalization)
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# Generic token lists - taken from MASTER_PROMPT.txt Section 6
# ------------------------------------------------------------

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
    normalized = normalize_text(text)
    return {
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in GENERIC_NAME_TOKENS
    }


def meaningful_address_tokens(text):
    normalized = normalize_text(text)
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
# 1. LOAD FULL TEST S1 AND BUILD BLOCKING KEYS
# ============================================================

def load_s1_and_build_keys():

    print("1. Loading full test S1 and building blocking keys...",
          flush=True)

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"   Test S1 rows: {len(s1):,}", flush=True)

    s1_data = {}

    all_exact_names = set()
    all_address_tokens = set()
    all_name_tokens = set()
    all_name_pairs = set()
    all_address_pairs = set()

    for row in s1.itertuples(index=False):

        s1_id = row.entity_id
        country = row.country

        name = normalize_text(row.business_name)
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

    print(f"   Relevant exact names       : {len(all_exact_names):,}",
          flush=True)
    print(f"   Relevant address tokens    : {len(all_address_tokens):,}",
          flush=True)
    print(f"   Relevant name tokens       : {len(all_name_tokens):,}",
          flush=True)
    print(f"   Relevant name-token pairs  : {len(all_name_pairs):,}",
          flush=True)
    print(f"   Relevant address-token pairs: {len(all_address_pairs):,}",
          flush=True)

    return (
        s1,
        s1_data,
        all_exact_names,
        all_address_tokens,
        all_name_tokens,
        all_name_pairs,
        all_address_pairs,
    )


# ============================================================
# 2. FIRST PASS OVER S2/S3: FREQUENCY COUNTING
# ============================================================

def build_frequency_tables(
    all_exact_names,
    all_address_tokens,
    all_name_tokens,
    all_name_pairs,
    all_address_pairs,
):

    print("\n2. Building frequency tables over test S2 + S3...",
          flush=True)

    exact_name_freq = Counter()
    address_freq = Counter()
    name_pair_freq = Counter()
    address_pair_freq = Counter()

    for source_name, path in SOURCE_FILES.items():

        print(f"   Scanning {source_name}...", flush=True)
        rows = 0

        for chunk in pd.read_csv(
            path, sep="\t", dtype=str,
            keep_default_na=False, chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(index=False):

                country = row.country
                name = normalize_text(row.business_name)

                if name in all_exact_names:
                    exact_name_freq[(country, name)] += 1

                address_tokens = (
                    meaningful_address_tokens(row.business_address)
                    & all_address_tokens
                )

                for token in address_tokens:
                    address_freq[(country, token)] += 1

                name_tokens = (
                    meaningful_name_tokens(name) & all_name_tokens
                )

                for pair in token_pairs(name_tokens):
                    if pair in all_name_pairs:
                        name_pair_freq[(country, pair)] += 1

                for pair in token_pairs(address_tokens):
                    if pair in all_address_pairs:
                        address_pair_freq[(country, pair)] += 1

            rows += len(chunk)
            print(f"     {source_name}: {rows:,} rows scanned...",
                  flush=True)

    print(f"   Exact-name freq keys   : {len(exact_name_freq):,}",
          flush=True)
    print(f"   Address freq keys      : {len(address_freq):,}",
          flush=True)
    print(f"   Name-pair freq keys    : {len(name_pair_freq):,}",
          flush=True)
    print(f"   Address-pair freq keys : {len(address_pair_freq):,}",
          flush=True)

    return exact_name_freq, address_freq, name_pair_freq, address_pair_freq


# ============================================================
# 3. SECOND PASS OVER S2/S3: BUILD PER-SOURCE BLOCKING INDEXES
# ============================================================
#
# Indexes are built separately per source (S2, S3) rather than
# merged, because the output file needs to record which source
# each candidate came from.

def build_indexes_per_source(
    all_exact_names,
    all_address_tokens,
    all_name_tokens,
    all_name_pairs,
    all_address_pairs,
    exact_name_freq,
    address_freq,
    name_pair_freq,
    address_pair_freq,
):

    print("\n3. Building per-source blocking indexes...", flush=True)

    indexes = {}

    for source_name, path in SOURCE_FILES.items():

        print(f"   Indexing {source_name}...", flush=True)

        exact_name_index = defaultdict(set)
        address_index = defaultdict(set)
        name_pair_index = defaultdict(set)
        address_pair_index = defaultdict(set)

        rows = 0

        for chunk in pd.read_csv(
            path, sep="\t", dtype=str,
            keep_default_na=False, chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(index=False):

                entity_id = row.entity_id
                country = row.country
                name = normalize_text(row.business_name)

                if name in all_exact_names:
                    key = (country, name)
                    if exact_name_freq[key] > 0:
                        exact_name_index[key].add(entity_id)

                address_tokens = (
                    meaningful_address_tokens(row.business_address)
                    & all_address_tokens
                )

                for token in address_tokens:
                    key = (country, token)
                    if address_freq[key] <= BROAD_ADDRESS_FREQ:
                        address_index[key].add(entity_id)

                name_tokens = (
                    meaningful_name_tokens(name) & all_name_tokens
                )

                for pair in token_pairs(name_tokens):
                    if pair in all_name_pairs:
                        key = (country, pair)
                        if name_pair_freq[key] <= NAME_PAIR_FREQ:
                            name_pair_index[key].add(entity_id)

                for pair in token_pairs(address_tokens):
                    if pair in all_address_pairs:
                        key = (country, pair)
                        if address_pair_freq[key] <= ADDRESS_PAIR_FREQ:
                            address_pair_index[key].add(entity_id)

            rows += len(chunk)

            if rows % (CHUNK_SIZE * 4) == 0:
                print(f"     {source_name}: {rows:,} rows indexed...",
                      flush=True)

        print(f"     {source_name}: {rows:,} rows indexed (done)",
              flush=True)

        indexes[source_name] = {
            "exact_name": exact_name_index,
            "address": address_index,
            "name_pair": name_pair_index,
            "address_pair": address_pair_index,
        }

    return indexes


# ============================================================
# 4. GENERATE CANDIDATES AND STREAM TO DISK
# ============================================================

def generate_and_write_candidates(s1_data, indexes, output_path):

    print("\n4. Generating candidates and writing output...", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    channel_usage = {
        "exact_name": 0, "address": 0,
        "name_pair": 0, "address_pair": 0,
    }

    candidate_counts = []
    zero_candidates = 0
    total_pairs_written = 0

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("source1_entity_id\tcandidate_entity_id\tsource\n")

        for i, (s1_id, data) in enumerate(s1_data.items(), start=1):

            country = data["country"]

            # candidate_entity_id -> source, deduplicated per S1
            found = {}

            any_exact = any_address = any_name_pair = any_address_pair = False

            for source_name, source_indexes in indexes.items():

                if data["name"]:
                    ids = source_indexes["exact_name"].get(
                        (country, data["name"]), set()
                    )
                    if ids:
                        any_exact = True
                    for eid in ids:
                        found[eid] = source_name

                addr_found = False
                for token in data["address_tokens"]:
                    ids = source_indexes["address"].get(
                        (country, token), set()
                    )
                    if ids:
                        addr_found = True
                    for eid in ids:
                        found[eid] = source_name
                any_address = any_address or addr_found

                np_found = False
                for pair in data["name_pairs"]:
                    ids = source_indexes["name_pair"].get(
                        (country, pair), set()
                    )
                    if ids:
                        np_found = True
                    for eid in ids:
                        found[eid] = source_name
                any_name_pair = any_name_pair or np_found

                ap_found = False
                for pair in data["address_pairs"]:
                    ids = source_indexes["address_pair"].get(
                        (country, pair), set()
                    )
                    if ids:
                        ap_found = True
                    for eid in ids:
                        found[eid] = source_name
                any_address_pair = any_address_pair or ap_found

            if any_exact:
                channel_usage["exact_name"] += 1
            if any_address:
                channel_usage["address"] += 1
            if any_name_pair:
                channel_usage["name_pair"] += 1
            if any_address_pair:
                channel_usage["address_pair"] += 1

            candidate_counts.append(len(found))

            if not found:
                zero_candidates += 1
            else:
                for candidate_id, source_name in found.items():
                    f.write(f"{s1_id}\t{candidate_id}\t{source_name}\n")
                    total_pairs_written += 1

            if i % 50_000 == 0:
                print(f"   {i:,}/{len(s1_data):,} S1 processed | "
                      f"{total_pairs_written:,} pairs written so far",
                      flush=True)

    return channel_usage, candidate_counts, zero_candidates, total_pairs_written


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 70)
    print("STEP T1 - PRODUCTION TEST CANDIDATE GENERATION")
    print("(four-channel blocking, full test scale)")
    print("=" * 70)

    (
        s1,
        s1_data,
        all_exact_names,
        all_address_tokens,
        all_name_tokens,
        all_name_pairs,
        all_address_pairs,
    ) = load_s1_and_build_keys()

    (
        exact_name_freq,
        address_freq,
        name_pair_freq,
        address_pair_freq,
    ) = build_frequency_tables(
        all_exact_names,
        all_address_tokens,
        all_name_tokens,
        all_name_pairs,
        all_address_pairs,
    )

    indexes = build_indexes_per_source(
        all_exact_names,
        all_address_tokens,
        all_name_tokens,
        all_name_pairs,
        all_address_pairs,
        exact_name_freq,
        address_freq,
        name_pair_freq,
        address_pair_freq,
    )

    (
        channel_usage,
        candidate_counts,
        zero_candidates,
        total_pairs_written,
    ) = generate_and_write_candidates(s1_data, indexes, OUTPUT_PATH)

    series = pd.Series(candidate_counts)

    print()
    print("=" * 70)
    print("CANDIDATE GENERATION RESULTS")
    print("=" * 70)

    print(f"Test S1 processed        : {len(s1_data):,}")
    print(f"Total candidate pairs    : {total_pairs_written:,}")
    print(f"Average candidates/S1    : {series.mean():.2f}")
    print(f"Median candidates/S1     : {series.median():.0f}")
    print(f"P95 candidates/S1        : {series.quantile(0.95):.2f}")
    print(f"Maximum candidates/S1    : {series.max():,}")
    print(f"Zero-candidate S1        : {zero_candidates:,}")

    print()
    print("Channel usage (S1s where the channel contributed):")
    for channel, count in channel_usage.items():
        print(f"  {channel:<15}: {count:,}")

    print()
    print(f"Saved candidate file to: {OUTPUT_PATH}")
    print(f"Runtime: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
