import re
import time

import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

S1_PATH = "data/raw/train_source1.tsv"
S2_PATH = "data/raw/train_source2.tsv"
S3_PATH = "data/raw/train_source3.tsv"
GT_PATH = "data/raw/train_ground_truth.tsv"

CANDIDATE_PATH = (
    "data/processed/address_pair_candidate_pairs.tsv"
)

OUTPUT_PATH = (
    "data/processed/training_pair_features.tsv"
)

CHUNK_SIZE = 250_000


# ============================================================
# NORMALIZATION
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


def tokenize(value):

    normalized = normalize_text(value)

    if not normalized:
        return set()

    return set(normalized.split())


def extract_numbers(value):

    if pd.isna(value):
        return set()

    return set(
        re.findall(
            r"\d+",
            str(value)
        )
    )


# ============================================================
# FEATURE CALCULATION
# ============================================================

def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b)


def safe_token_ratio(a, b):

    if not a or not b:
        return 0.0

    return fuzz.token_sort_ratio(a, b)


def safe_partial_ratio(a, b):

    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b)


def length_ratio(a, b):

    if not a or not b:
        return 0.0

    return (
        min(len(a), len(b))
        /
        max(len(a), len(b))
    )


def shared_token_count(a, b):

    if not a or not b:
        return 0

    return len(a & b)


def shared_number_count(a, b):

    if not a or not b:
        return 0

    return len(a & b)


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    s1_id,
    s1,
    s2,
    source
):

    # --------------------------------------------------------
    # Normalize names
    # --------------------------------------------------------

    name1 = normalize_text(
        s1["business_name"]
    )

    name2 = normalize_text(
        s2["business_name"]
    )

    # --------------------------------------------------------
    # Normalize addresses
    # --------------------------------------------------------

    address1 = normalize_text(
        s1["business_address"]
    )

    address2 = normalize_text(
        s2["business_address"]
    )

    # --------------------------------------------------------
    # Tokenize names
    # --------------------------------------------------------

    name_tokens1 = tokenize(
        s1["business_name"]
    )

    name_tokens2 = tokenize(
        s2["business_name"]
    )

    # --------------------------------------------------------
    # Tokenize addresses
    # --------------------------------------------------------

    address_tokens1 = tokenize(
        s1["business_address"]
    )

    address_tokens2 = tokenize(
        s2["business_address"]
    )

    # --------------------------------------------------------
    # Extract address numbers
    # --------------------------------------------------------

    numbers1 = extract_numbers(
        s1["business_address"]
    )

    numbers2 = extract_numbers(
        s2["business_address"]
    )

    # --------------------------------------------------------
    # Normalize countries
    # --------------------------------------------------------

    country1 = normalize_text(
        s1["country"]
    )

    country2 = normalize_text(
        s2["country"]
    )

    # ========================================================
    # RETURN FEATURES
    # ========================================================

    return {

        # ----------------------------------------------------
        # Identification
        # ----------------------------------------------------

        "source1_entity_id":
            s1_id,

        "candidate_entity_id":
            s2.name
            if s2.name is not None
            else "",

        "source":
            source,

        # ----------------------------------------------------
        # Country
        # ----------------------------------------------------

        "country_match":
            int(
                country1 == country2
            ),

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        "name_exact":
            int(
                bool(name1)
                and name1 == name2
            ),

        "name_ratio":
            safe_ratio(
                name1,
                name2
            ),

        "name_token_ratio":
            safe_token_ratio(
                name1,
                name2
            ),

        "name_partial_ratio":
            safe_partial_ratio(
                name1,
                name2
            ),

        "name_length_ratio":
            length_ratio(
                name1,
                name2
            ),

        "shared_name_tokens":
            shared_token_count(
                name_tokens1,
                name_tokens2
            ),

        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        "address_exact":
            int(
                bool(address1)
                and address1 == address2
            ),

        "address_ratio":
            safe_ratio(
                address1,
                address2
            ),

        "address_token_ratio":
            safe_token_ratio(
                address1,
                address2
            ),

        "address_partial_ratio":
            safe_partial_ratio(
                address1,
                address2
            ),

        "address_length_ratio":
            length_ratio(
                address1,
                address2
            ),

        "shared_address_tokens":
            shared_token_count(
                address_tokens1,
                address_tokens2
            ),

        "shared_numbers":
            shared_number_count(
                numbers1,
                numbers2
            ),
    }


# ============================================================
# LOAD DATA
# ============================================================

start = time.time()

print("Loading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
)

s1 = s1.set_index(
    "entity_id"
)

print(
    f"S1 loaded: {len(s1):,}"
)


print("\nLoading S2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str
)

s2 = s2.set_index(
    "entity_id"
)

print(
    f"S2 loaded: {len(s2):,}"
)


print("\nLoading S3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str
)

s3 = s3.set_index(
    "entity_id"
)

print(
    f"S3 loaded: {len(s3):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

ground_truth = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    if (
        pd.isna(value)
        or not str(value).strip()
    ):

        ground_truth[
            row["source1_entity_id"]
        ] = set()

    else:

        ground_truth[
            row["source1_entity_id"]
        ] = set(
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        )

print(
    f"Ground truth loaded: "
    f"{len(ground_truth):,}"
)


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading candidates...")

candidates_df = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str
)

print(
    f"Candidate S1 rows: "
    f"{len(candidates_df):,}"
)


# ============================================================
# GENERATE FEATURES
# ============================================================

print("\nGenerating pairwise features...")

rows = []

positive_count = 0
negative_count = 0
processed_s1 = 0
processed_pairs = 0


for _, candidate_row in candidates_df.iterrows():

    s1_id = candidate_row[
        "source1_entity_id"
    ]

    candidate_value = candidate_row[
        "candidate_entity_ids"
    ]

    # --------------------------------------------------------
    # No candidates
    # --------------------------------------------------------

    if (
        pd.isna(candidate_value)
        or not str(candidate_value).strip()
    ):

        processed_s1 += 1
        continue

    # --------------------------------------------------------
    # Candidate IDs
    # --------------------------------------------------------

    candidate_ids = [
        x.strip()
        for x in str(candidate_value).split(",")
        if x.strip()
    ]

    # --------------------------------------------------------
    # Get S1 record
    # --------------------------------------------------------

    if s1_id not in s1.index:

        processed_s1 += 1
        continue

    s1_row = s1.loc[s1_id]

    # --------------------------------------------------------
    # Ground-truth IDs
    # --------------------------------------------------------

    true_ids = ground_truth.get(
        s1_id,
        set()
    )

    # ========================================================
    # PROCESS CANDIDATES
    # ========================================================

    for candidate_id in candidate_ids:

        # ----------------------------------------------------
        # S2
        # ----------------------------------------------------

        if candidate_id.startswith("S2-"):

            if candidate_id not in s2.index:
                continue

            candidate_row_data = s2.loc[
                candidate_id
            ]

            source = "S2"

        # ----------------------------------------------------
        # S3
        # ----------------------------------------------------

        elif candidate_id.startswith("S3-"):

            if candidate_id not in s3.index:
                continue

            candidate_row_data = s3.loc[
                candidate_id
            ]

            source = "S3"

        # ----------------------------------------------------
        # Invalid candidate
        # ----------------------------------------------------

        else:

            continue

        # ----------------------------------------------------
        # Build pair features
        # ----------------------------------------------------

        features = build_features(
            s1_id,
            s1_row,
            candidate_row_data,
            source
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = int(
            candidate_id in true_ids
        )

        features["candidate_entity_id"] = (
            candidate_id
        )

        features["label"] = label

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        rows.append(
            features
        )

        processed_pairs += 1

        if label:

            positive_count += 1

        else:

            negative_count += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    processed_s1 += 1

    if processed_s1 % 500 == 0:

        print(
            f"  Processed "
            f"{processed_s1:,}/"
            f"{len(candidates_df):,} S1 | "
            f"pairs={processed_pairs:,} | "
            f"positives={positive_count:,}",
            flush=True
        )


# ============================================================
# SAVE
# ============================================================

print("\nCreating DataFrame...")

features_df = pd.DataFrame(
    rows
)

print(
    f"Total pairs: "
    f"{len(features_df):,}"
)

print(
    f"Positive pairs: "
    f"{positive_count:,}"
)

print(
    f"Negative pairs: "
    f"{negative_count:,}"
)

if len(features_df) > 0:

    print(
        f"Positive rate: "
        f"{positive_count / len(features_df):.6%}"
    )


features_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)


# ============================================================
# FINISH
# ============================================================

elapsed = (
    time.time() - start
) / 60

print()

print(
    f"Saved features to: "
    f"{OUTPUT_PATH}"
)

print(
    f"Runtime: "
    f"{elapsed:.2f} minutes"
)