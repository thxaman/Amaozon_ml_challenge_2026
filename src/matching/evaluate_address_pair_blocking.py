import os
import re
import itertools
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42
CHUNK_SIZE = 250_000

ADDRESS_PAIR_FREQ = 50

S1_PATH = "data/raw/train_source1.tsv"
S2_PATH = "data/raw/train_source2.tsv"
S3_PATH = "data/raw/train_source3.tsv"
GT_PATH = "data/raw/train_ground_truth.tsv"

BASELINE_CANDIDATE_PATH = (
    "data/processed/improved_candidate_pairs.tsv"
)

OUTPUT_PATH = (
    "data/processed/address_pair_candidate_pairs.tsv"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# ADDRESS TOKENIZATION
# ============================================================

GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "lane",
    "ln",
    "drive",
    "dr",
    "parkway",
    "pkwy",
    "highway",
    "hwy",
    "way",
    "place",
    "pl",
    "court",
    "ct",
    "circle",
    "cir",
    "terrace",
    "ter",
    "square",
    "sq",
    "building",
    "bldg",
    "floor",
    "fl",
    "unit",
    "suite",
    "ste",
    "room",
    "rm",
    "block",
    "blk",
    "sector",
    "near",
    "opposite",
    "opp",
    "no",
    "number",
    "street",
    "state",
    "county",
    "district",
    "city",
}


def meaningful_address_tokens(value):

    normalized = normalize_text(value)

    if not normalized:
        return set()

    tokens = normalized.split()

    result = set()

    for token in tokens:

        if token in GENERIC_ADDRESS_TOKENS:
            continue

        # Ignore extremely short alphabetic tokens.
        if token.isalpha() and len(token) < 3:
            continue

        result.add(token)

    return result


def address_token_pairs(value):

    tokens = meaningful_address_tokens(value)

    if len(tokens) < 2:
        return set()

    return set(
        itertools.combinations(
            sorted(tokens),
            2
        )
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

sample_ids = (
    gt["source1_entity_id"]
    .sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
    .tolist()
)

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_ids)
]

true_matches = {}

for _, row in gt_sample.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value) or not str(value).strip():

        true_matches[
            row["source1_entity_id"]
        ] = set()

    else:

        true_matches[
            row["source1_entity_id"]
        ] = set(
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        )

print(
    f"Sampled S1 entities: {len(sample_ids):,}"
)


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
)

s1 = s1[
    s1["entity_id"].isin(sample_ids)
].copy()

print(
    f"Loaded S1: {len(s1):,}"
)


# ============================================================
# LOAD CURRENT IMPROVED CANDIDATES
# ============================================================

print("\nLoading current improved candidates...")

base_candidates_df = pd.read_csv(
    BASELINE_CANDIDATE_PATH,
    sep="\t",
    dtype=str
)

base_candidates = {}

for _, row in base_candidates_df.iterrows():

    s1_id = row["source1_entity_id"]

    value = row["candidate_entity_ids"]

    if pd.isna(value) or not str(value).strip():

        base_candidates[s1_id] = set()

    else:

        base_candidates[s1_id] = set(
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        )

print(
    f"Loaded current candidate rows: "
    f"{len(base_candidates):,}"
)


# ============================================================
# PREPARE S1 ADDRESS PAIRS
# ============================================================

print("\nPreparing S1 address-token pairs...")

s1_address_pairs = {}

relevant_pairs = set()

for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    pairs = address_token_pairs(
        row["business_address"]
    )

    s1_address_pairs[s1_id] = pairs

    relevant_pairs.update(pairs)

print(
    f"Relevant address pairs: "
    f"{len(relevant_pairs):,}"
)


# ============================================================
# COUNT ADDRESS-PAIR FREQUENCY
# ============================================================

def count_address_pairs(path):

    counts = Counter()

    processed = 0

    print(f"  Scanning {os.path.basename(path)}...")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        for address in chunk["business_address"]:

            pairs = (
                address_token_pairs(address)
                &
                relevant_pairs
            )

            for pair in pairs:
                counts[pair] += 1

        processed += len(chunk)

        print(
            f"    Processed {processed:,} rows...",
            flush=True
        )

    return counts


print("\nBuilding address-pair frequency tables...")

s2_pair_freq = count_address_pairs(
    S2_PATH
)

s3_pair_freq = count_address_pairs(
    S3_PATH
)

s2_usable_pairs = {
    pair
    for pair, freq in s2_pair_freq.items()
    if freq <= ADDRESS_PAIR_FREQ
}

s3_usable_pairs = {
    pair
    for pair, freq in s3_pair_freq.items()
    if freq <= ADDRESS_PAIR_FREQ
}

print()
print(
    f"S2 usable address pairs: "
    f"{len(s2_usable_pairs):,}"
)

print(
    f"S3 usable address pairs: "
    f"{len(s3_usable_pairs):,}"
)


# ============================================================
# BUILD ADDRESS-PAIR INDEX
# ============================================================

def build_pair_index(
    path,
    usable_pairs,
    source_name
):

    index = defaultdict(set)

    processed = 0

    print(
        f"\n  Indexing {source_name}..."
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            pairs = (
                address_token_pairs(
                    row["business_address"]
                )
                &
                usable_pairs
            )

            for pair in pairs:

                index[pair].add(
                    entity_id
                )

        processed += len(chunk)

        print(
            f"    Indexed {processed:,} rows...",
            flush=True
        )

    return index


print("\nBuilding address-pair indexes...")

s2_pair_index = build_pair_index(
    S2_PATH,
    s2_usable_pairs,
    "S2"
)

s3_pair_index = build_pair_index(
    S3_PATH,
    s3_usable_pairs,
    "S3"
)

print()
print(
    f"S2 address-pair index keys: "
    f"{len(s2_pair_index):,}"
)

print(
    f"S3 address-pair index keys: "
    f"{len(s3_pair_index):,}"
)


# ============================================================
# GENERATE NEW CANDIDATES
# ============================================================

print("\nGenerating candidates...")

final_candidates = {}

address_pair_candidates = {}

for i, s1_id in enumerate(sample_ids, start=1):

    candidates = set(
        base_candidates.get(
            s1_id,
            set()
        )
    )

    pair_candidates = set()

    for pair in s1_address_pairs.get(
        s1_id,
        set()
    ):

        pair_candidates.update(
            s2_pair_index.get(
                pair,
                set()
            )
        )

        pair_candidates.update(
            s3_pair_index.get(
                pair,
                set()
            )
        )

    address_pair_candidates[s1_id] = (
        pair_candidates
    )

    candidates.update(
        pair_candidates
    )

    final_candidates[s1_id] = candidates

    if i % 1000 == 0:

        print(
            f"  Generated candidates for "
            f"{i:,}/{len(sample_ids):,} S1...",
            flush=True
        )


# ============================================================
# EVALUATION
# ============================================================

total_true_pairs = 0

base_recovered_pairs = 0
new_recovered_pairs = 0

base_full_s1 = 0
new_full_s1 = 0

matched_s1 = 0

base_candidate_counts = []
new_candidate_counts = []

unique_new_true_pairs = 0
new_candidate_pairs = 0

new_recovered_by_source = {
    "S2": 0,
    "S3": 0,
}


for s1_id in sample_ids:

    true_set = true_matches.get(
        s1_id,
        set()
    )

    base_set = base_candidates.get(
        s1_id,
        set()
    )

    new_set = final_candidates.get(
        s1_id,
        set()
    )

    address_pair_set = (
        address_pair_candidates.get(
            s1_id,
            set()
        )
    )

    total_true_pairs += len(true_set)

    base_recovered = (
        true_set & base_set
    )

    new_recovered = (
        true_set & new_set
    )

    base_recovered_pairs += len(
        base_recovered
    )

    new_recovered_pairs += len(
        new_recovered
    )

    base_candidate_counts.append(
        len(base_set)
    )

    new_candidate_counts.append(
        len(new_set)
    )

    new_only_candidates = (
        new_set - base_set
    )

    new_candidate_pairs += len(
        new_only_candidates
    )

    new_only_true = (
        true_set
        &
        new_only_candidates
    )

    unique_new_true_pairs += len(
        new_only_true
    )

    for entity_id in new_only_true:

        if entity_id.startswith("S2-"):
            new_recovered_by_source["S2"] += 1

        elif entity_id.startswith("S3-"):
            new_recovered_by_source["S3"] += 1

    if true_set:

        matched_s1 += 1

        if base_recovered == true_set:
            base_full_s1 += 1

        if new_recovered == true_set:
            new_full_s1 += 1


base_pair_recall = (
    base_recovered_pairs /
    total_true_pairs
)

new_pair_recall = (
    new_recovered_pairs /
    total_true_pairs
)

base_full_recall = (
    base_full_s1 /
    matched_s1
)

new_full_recall = (
    new_full_s1 /
    matched_s1
)

avg_base = (
    sum(base_candidate_counts) /
    len(base_candidate_counts)
)

avg_new = (
    sum(new_candidate_counts) /
    len(new_candidate_counts)
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("ADDRESS TOKEN-PAIR BLOCKING RESULTS")
print("=" * 70)

print(
    f"Address pair frequency limit : "
    f"{ADDRESS_PAIR_FREQ:,}"
)

print(
    f"S1 sample                   : "
    f"{len(sample_ids):,}"
)

print(
    f"True pairs                  : "
    f"{total_true_pairs:,}"
)

print()
print("CURRENT IMPROVED BLOCKER")

print(
    f"Recovered pairs             : "
    f"{base_recovered_pairs:,}"
)

print(
    f"Pair recall                 : "
    f"{base_pair_recall:.4f}"
)

print(
    f"Fully recovered S1          : "
    f"{base_full_s1:,}"
)

print(
    f"Full S1 recall              : "
    f"{base_full_recall:.4f}"
)

print(
    f"Average candidates/S1       : "
    f"{avg_base:.2f}"
)

print()
print("WITH ADDRESS TOKEN-PAIR CHANNEL")

print(
    f"Recovered pairs             : "
    f"{new_recovered_pairs:,}"
)

print(
    f"Pair recall                 : "
    f"{new_pair_recall:.4f}"
)

print(
    f"Fully recovered S1          : "
    f"{new_full_s1:,}"
)

print(
    f"Full S1 recall              : "
    f"{new_full_recall:.4f}"
)

print(
    f"Average candidates/S1       : "
    f"{avg_new:.2f}"
)

print()
print("=" * 70)
print("ADDRESS-PAIR UNIQUE CONTRIBUTION")
print("=" * 70)

print(
    f"New candidate pairs added  : "
    f"{new_candidate_pairs:,}"
)

print(
    f"New true pairs recovered   : "
    f"{unique_new_true_pairs:,}"
)

print(
    f"New S2 true pairs          : "
    f"{new_recovered_by_source['S2']:,}"
)

print(
    f"New S3 true pairs          : "
    f"{new_recovered_by_source['S3']:,}"
)

print()
print("=" * 70)
print("IMPROVEMENT")
print("=" * 70)

print(
    f"Pair recall improvement    : "
    f"{new_pair_recall - base_pair_recall:+.4f}"
)

print(
    f"Full S1 recall improvement : "
    f"{new_full_recall - base_full_recall:+.4f}"
)

print(
    f"Average candidates change  : "
    f"{avg_new - avg_base:+.2f}"
)

print(
    f"Candidate multiplier       : "
    f"{avg_new / avg_base:.2f}x"
)


# ============================================================
# SAVE
# ============================================================

output_rows = []

for s1_id in sample_ids:

    candidates = final_candidates.get(
        s1_id,
        set()
    )

    output_rows.append(
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids":
                ",".join(sorted(candidates))
        }
    )

output_df = pd.DataFrame(
    output_rows
)

output_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print()
print(
    f"Saved candidates to: "
    f"{OUTPUT_PATH}"
)