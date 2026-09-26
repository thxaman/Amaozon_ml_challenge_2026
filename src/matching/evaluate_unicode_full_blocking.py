import os
import re
import time
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

S1_PATH = os.path.join(BASE_DIR, "data", "raw", "train_source1.tsv")
S2_PATH = os.path.join(BASE_DIR, "data", "raw", "train_source2.tsv")
S3_PATH = os.path.join(BASE_DIR, "data", "raw", "train_source3.tsv")
GT_PATH = os.path.join(BASE_DIR, "data", "raw", "train_ground_truth.tsv")

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "unicode_full_candidate_pairs.tsv"
)

CHUNK_SIZE = 250_000
MAX_TOKEN_FREQ = 500

GENERIC_NAME_TOKENS = {
    "private",
    "limited",
    "pvt",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "companies",
    "co",
    "plc",
    "llp",
    "lp",
    "group",
    "groups",
    "holdings",
    "holding",
    "partners",
    "partner",
    "services",
    "service",
    "solutions",
    "solution",
    "enterprises",
    "enterprise",
    "international",
    "global",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_country(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def unicode_tokens(value):
    """
    Unicode-aware tokenization.

    Unlike the old normalization, this preserves:
    Hindi, Bengali, Tamil, Telugu, Kannada, etc.
    """
    if pd.isna(value):
        return []

    value = str(value).lower()

    tokens = re.findall(r"[^\W_]+", value, flags=re.UNICODE)

    result = []

    for token in tokens:
        if token in GENERIC_NAME_TOKENS:
            continue

        # Avoid tiny tokens.
        if len(token) < 3:
            continue

        result.append(token)

    return result


# ============================================================
# LOAD S1
# ============================================================

print("Loading Source 1...", flush=True)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
)

s1["country_norm"] = s1["country"].map(normalize_country)

print(
    f"S1 rows: {len(s1):,}",
    flush=True
)


# ============================================================
# BUILD S1 TOKEN FREQUENCY
# ============================================================

print("\nBuilding Unicode token frequencies...", flush=True)

token_freq = Counter()

for row in s1.itertuples(index=False):
    tokens = set(unicode_tokens(row.business_name))

    for token in tokens:
        token_freq[(row.country_norm, token)] += 1


print(
    f"Total country-token keys: {len(token_freq):,}",
    flush=True
)


# ============================================================
# KEEP ONLY RARE / USEFUL TOKENS
# ============================================================

print(
    f"\nKeeping tokens with frequency <= {MAX_TOKEN_FREQ}...",
    flush=True
)

valid_keys = {
    key
    for key, freq in token_freq.items()
    if freq <= MAX_TOKEN_FREQ
}

print(
    f"Valid country-token keys: {len(valid_keys):,}",
    flush=True
)


# ============================================================
# BUILD INVERTED INDEX
#
# (country, token) -> S1 entity IDs
# ============================================================

print("\nBuilding S1 Unicode index...", flush=True)

token_index = defaultdict(list)

for row in s1.itertuples(index=False):

    tokens = set(unicode_tokens(row.business_name))

    for token in tokens:

        key = (row.country_norm, token)

        if key in valid_keys:
            token_index[key].append(row.entity_id)


print(
    f"Index keys: {len(token_index):,}",
    flush=True
)


# ============================================================
# GROUND TRUTH
#
# IMPORTANT:
# We DO NOT build the entire candidate set in RAM.
# ============================================================

print("\nLoading ground truth...", flush=True)

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

print(
    f"Ground-truth S1 rows: {len(gt):,}",
    flush=True
)


# ------------------------------------------------------------
# Instead of storing every pair in a giant Python set,
# create a dictionary only for true pairs.
# ------------------------------------------------------------

true_pairs = set()

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    matched = row.matched_entity_ids

    if pd.isna(matched) or not str(matched).strip():
        continue

    for entity_id in str(matched).split(","):

        entity_id = entity_id.strip()

        if entity_id:
            true_pairs.add((s1_id, entity_id))


print(
    f"True positive pairs: {len(true_pairs):,}",
    flush=True
)


# ============================================================
# PREPARE OUTPUT
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

# Remove old output.
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)


# ============================================================
# STATISTICS
# ============================================================

candidate_pair_count = 0
recovered_pairs = set()

candidate_s1 = set()

s2_candidate_count = 0
s3_candidate_count = 0

start_time = time.time()


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(path, source_name):

    global candidate_pair_count
    global s2_candidate_count
    global s3_candidate_count

    print(
        f"\nProcessing {source_name}...",
        flush=True
    )

    first_write = True

    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk["country_norm"] = chunk["country"].map(
            normalize_country
        )

        output_rows = []

        for row in chunk.itertuples(index=False):

            processed += 1

            country = row.country_norm

            tokens = set(
                unicode_tokens(row.business_name)
            )

            if not tokens:
                continue

            matched_s1 = set()

            for token in tokens:

                key = (country, token)

                ids = token_index.get(key)

                if ids:
                    matched_s1.update(ids)

            if not matched_s1:
                continue

            entity_id = row.entity_id

            for s1_id in matched_s1:

                candidate_pair_count += 1
                candidate_s1.add(s1_id)

                if source_name == "S2":
                    s2_candidate_count += 1
                else:
                    s3_candidate_count += 1

                if (s1_id, entity_id) in true_pairs:
                    recovered_pairs.add(
                        (s1_id, entity_id)
                    )

                output_rows.append(
                    (
                        s1_id,
                        entity_id
                    )
                )

        # ----------------------------------------------------
        # Write this chunk immediately.
        # ----------------------------------------------------

        if output_rows:

            out_df = pd.DataFrame(
                output_rows,
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id"
                ]
            )

            out_df.to_csv(
                OUTPUT_PATH,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

            del out_df
            del output_rows

        elapsed = time.time() - start_time

        print(
            f"{source_name}: "
            f"{processed:,} / "
            f"{'?' if source_name else ''} "
            f"records | "
            f"candidates={candidate_pair_count:,} | "
            f"recovered={len(recovered_pairs):,} | "
            f"time={elapsed / 60:.1f} min",
            flush=True
        )

    print(
        f"\nFinished {source_name}. "
        f"Processed {processed:,} records.",
        flush=True
    )


# ============================================================
# RUN
# ============================================================

process_source(
    S2_PATH,
    "S2"
)

process_source(
    S3_PATH,
    "S3"
)


# ============================================================
# FINAL REPORT
# ============================================================

elapsed = time.time() - start_time

pair_recall = (
    len(recovered_pairs) / len(true_pairs)
    if true_pairs
    else 0
)

print("\n" + "=" * 70)
print("UNICODE FULL BLOCKING RESULTS")
print("=" * 70)

print(
    f"Candidate pairs:       {candidate_pair_count:,}"
)

print(
    f"True pairs:             {len(true_pairs):,}"
)

print(
    f"Recovered true pairs:   {len(recovered_pairs):,}"
)

print(
    f"Pair recall:            {pair_recall:.6f}"
)

print(
    f"S1 with candidates:     {len(candidate_s1):,}"
)

print(
    f"S2 candidate pairs:     {s2_candidate_count:,}"
)

print(
    f"S3 candidate pairs:     {s3_candidate_count:,}"
)

print(
    f"Runtime:                {elapsed / 60:.2f} minutes"
)

print(
    f"\nOutput written to:\n{OUTPUT_PATH}"
)

print("=" * 70)