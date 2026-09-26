"""
STEP T2 - PRODUCTION TEST FEATURE ENGINEERING

Builds pairwise features for every test candidate pair produced by
build_test_candidates.py.

CRITICAL: the feature logic here is copied VERBATIM from
build_full_pair_features.py (the script that produced
full_training_features.tsv, which final_match_classifier.joblib was
trained on). It must stay byte-for-byte identical to that logic,
because the trained model's decision boundary depends on features
being computed exactly the same way at train and test time. The only
difference is that there is no "label" column, since test data has
no ground truth.

INPUT:
    data/processed/candidate_pairs.tsv
        (source1_entity_id, candidate_entity_id, source)
    data/test/test_source1.tsv
    data/test/test_source2.tsv
    data/test/test_source3.tsv

OUTPUT:
    data/processed/test_pair_features.tsv
"""

import os
import re
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_sort_ratio, partial_ratio


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PAIR_FILE = os.path.join(BASE_DIR, "data", "processed", "candidate_pairs.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "test_pair_features.tsv")

S1_PATH = os.path.join(BASE_DIR, "data", "test", "test_source1.tsv")
S2_PATH = os.path.join(BASE_DIR, "data", "test", "test_source2.tsv")
S3_PATH = os.path.join(BASE_DIR, "data", "test", "test_source3.tsv")

CHUNK_SIZE = 100_000


# ============================================================
# NORMALIZATION
# (identical to build_full_pair_features.py - DO NOT change)
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = value.replace("&", " and ")

    value = re.sub(r"[^a-z0-9]+", " ", value)

    value = re.sub(r"\s+", " ", value).strip()

    return value


def tokenize(value):
    if not value:
        return set()

    return set(value.split())


def extract_numbers(value):
    if not value:
        return set()

    return set(re.findall(r"\d+", value))


# ============================================================
# STRING FEATURES
# (identical to build_full_pair_features.py - DO NOT change)
# ============================================================

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


# ============================================================
# FEATURE CREATION
# (identical to build_full_pair_features.py, minus "label" - DO NOT
# change the feature logic itself)
# ============================================================

def build_features(chunk, s1_lookup, s2_lookup, s3_lookup):

    records = []

    for row in chunk.itertuples(index=False):

        s1_id = row.source1_entity_id
        candidate_id = row.candidate_entity_id
        source = row.source

        s1 = s1_lookup.get(s1_id)

        if source == "S2":
            candidate = s2_lookup.get(candidate_id)
        else:
            candidate = s3_lookup.get(candidate_id)

        if s1 is None or candidate is None:
            continue

        s1_name = s1["business_name"]
        s2_name = candidate["business_name"]

        s1_address = s1["business_address"]
        s2_address = candidate["business_address"]

        s1_country = s1["country"]
        s2_country = candidate["country"]

        n1 = normalize_text(s1_name)
        n2 = normalize_text(s2_name)

        a1 = normalize_text(s1_address)
        a2 = normalize_text(s2_address)

        name_tokens_1 = tokenize(n1)
        name_tokens_2 = tokenize(n2)

        address_tokens_1 = tokenize(a1)
        address_tokens_2 = tokenize(a2)

        numbers_1 = extract_numbers(a1)
        numbers_2 = extract_numbers(a2)

        country_match = int(
            str(s1_country).lower() == str(s2_country).lower()
        )

        name_exact = int(bool(n1) and bool(n2) and n1 == n2)

        name_ratio = safe_ratio(n1, n2)
        name_token_ratio = safe_token_ratio(n1, n2)
        name_partial_ratio = safe_partial_ratio(n1, n2)
        name_length_ratio = length_ratio(n1, n2)

        shared_name_tokens = shared_token_count(
            name_tokens_1, name_tokens_2
        )

        address_exact = int(bool(a1) and bool(a2) and a1 == a2)

        address_ratio = safe_ratio(a1, a2)
        address_token_ratio = safe_token_ratio(a1, a2)
        address_partial_ratio = safe_partial_ratio(a1, a2)
        address_length_ratio = length_ratio(a1, a2)

        shared_address_tokens = shared_token_count(
            address_tokens_1, address_tokens_2
        )

        shared_numbers = shared_number_count(numbers_1, numbers_2)

        name_strong_90 = int(name_token_ratio >= 90)
        address_strong_90 = int(address_token_ratio >= 90)

        name_strong_80 = int(name_token_ratio >= 80)
        address_strong_80 = int(address_token_ratio >= 80)

        both_strong_90 = int(name_strong_90 and address_strong_90)
        both_strong_80 = int(name_strong_80 and address_strong_80)

        name_strong_address_weak = int(
            name_strong_90 and address_token_ratio < 70
        )

        address_strong_name_weak = int(
            address_strong_90 and name_token_ratio < 70
        )

        name_address_mean = (
            name_token_ratio + address_token_ratio
        ) / 200.0

        name_address_min = (
            min(name_token_ratio, address_token_ratio) / 100.0
        )

        name_address_max = (
            max(name_token_ratio, address_token_ratio) / 100.0
        )

        name_address_gap = (
            abs(name_token_ratio - address_token_ratio) / 100.0
        )

        name_address_product = (
            name_token_ratio * address_token_ratio
        ) / 10000.0

        exact_name_strong_address = int(
            name_exact and address_token_ratio >= 80
        )

        exact_address_strong_name = int(
            address_exact and name_token_ratio >= 80
        )

        name_tokens_strong = int(name_token_ratio >= 90)
        address_tokens_strong = int(address_token_ratio >= 80)
        address_tokens_very_strong = int(address_token_ratio >= 90)

        has_shared_number = int(shared_numbers >= 1)
        multiple_shared_numbers = int(shared_numbers >= 2)

        strong_address_with_number = int(
            address_token_ratio >= 90 and shared_numbers >= 1
        )

        very_strong_address_with_number = int(
            address_token_ratio >= 95 and shared_numbers >= 1
        )

        strong_name_with_number = int(
            name_token_ratio >= 90 and shared_numbers >= 1
        )

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

        records.append({
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
            "very_strong_address_with_number":
                very_strong_address_with_number,
            "strong_name_with_number": strong_name_with_number,

            "strong_evidence_count": strong_evidence_count,
            "very_strong_evidence_count": very_strong_evidence_count,

            "source_is_s3": int(source == "S3"),

            # NOTE: no "label" column - test data has no ground truth.
        })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP T2 - PRODUCTION TEST FEATURE ENGINEERING")
    print("=" * 70)

    print("\nLoading test source datasets...", flush=True)

    s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(S2_PATH, sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(S3_PATH, sep="\t", dtype=str, keep_default_na=False)

    print("Building lookup dictionaries...", flush=True)

    s1_lookup = s1.set_index("entity_id").to_dict("index")
    s2_lookup = s2.set_index("entity_id").to_dict("index")
    s3_lookup = s3.set_index("entity_id").to_dict("index")

    print(f"S1: {len(s1_lookup):,}", flush=True)
    print(f"S2: {len(s2_lookup):,}", flush=True)
    print(f"S3: {len(s3_lookup):,}", flush=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    first_chunk = True
    total_rows = 0

    reader = pd.read_csv(
        PAIR_FILE,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
            "source": str,
        },
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        features = build_features(chunk, s1_lookup, s2_lookup, s3_lookup)

        if first_chunk:
            features.to_csv(OUTPUT_FILE, sep="\t", index=False, mode="w")
            first_chunk = False
        else:
            features.to_csv(
                OUTPUT_FILE, sep="\t", index=False, mode="a", header=False
            )

        total_rows += len(features)

        print(
            f"Chunk {chunk_number:04d} | "
            f"input={len(chunk):,} | "
            f"features={len(features):,} | "
            f"total={total_rows:,}",
            flush=True,
        )

    print("\nFeature generation complete.", flush=True)
    print(f"Total feature rows: {total_rows:,}", flush=True)
    print(f"Saved to: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
