import re
import numpy as np
import pandas as pd

from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
GROUND_TRUTH = "data/raw/train_ground_truth.tsv"
CANDIDATES = "data/processed/baseline_candidate_pairs.tsv"

SOURCE_FILES = {
    "S2": "data/raw/train_source2.tsv",
    "S3": "data/raw/train_source3.tsv",
}

CHUNK_SIZE = 50_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # & -> and
    text = text.replace("&", " and ")

    # keep letters/numbers, remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


GENERIC_NAME_TOKENS = {
    "company",
    "companies",
    "corporation",
    "corp",
    "corporation",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "private",
    "pvt",
    "llp",
    "plc",
    "co",
    "group",
    "holdings",
    "holding",
    "services",
    "service",
    "solutions",
    "solution",
    "enterprise",
    "enterprises",
    "industries",
    "industry",
    "international",
    "global",
}


GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "lane",
    "ln",
    "drive",
    "dr",
    "highway",
    "hwy",
    "boulevard",
    "blvd",
    "parkway",
    "pkwy",
    "place",
    "pl",
    "building",
    "bldg",
    "floor",
    "fl",
    "suite",
    "ste",
    "unit",
    "block",
    "sector",
    "near",
    "opposite",
}


def meaningful_tokens(text, generic_tokens):
    normalized = normalize_text(text)

    tokens = normalized.split()

    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in generic_tokens
    }


def extract_numbers(text):
    if pd.isna(text):
        return set()

    return set(re.findall(r"\d+", str(text)))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

# Same sample used during baseline blocking
sample_s1 = (
    gt.sample(
        n=min(SAMPLE_SIZE, len(gt)),
        random_state=42
    )
    .copy()
)

sample_ids = set(sample_s1["source1_entity_id"])

print(f"Sample S1 entities: {len(sample_ids):,}")


# ============================================================
# BUILD TRUE MATCH LOOKUP
# ============================================================

true_matches = {}

for _, row in sample_s1.iterrows():

    s1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"]

    if not value:
        true_matches[s1_id] = set()
    else:
        true_matches[s1_id] = {
            x.strip()
            for x in value.split(",")
            if x.strip()
        }


# ============================================================
# LOAD CANDIDATES
# ============================================================

print("Loading candidate pairs...")

candidates = pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

candidates = candidates[
    candidates["source1_entity_id"].isin(sample_ids)
].copy()

print(f"Candidate S1 rows: {len(candidates):,}")


# ============================================================
# CONVERT CANDIDATES INTO PAIRS
# ============================================================

pairs = []

for _, row in candidates.iterrows():

    s1_id = row["source1_entity_id"]

    candidate_string = row["candidate_entity_ids"]

    if not candidate_string:
        continue

    for candidate_id in candidate_string.split(","):

        candidate_id = candidate_id.strip()

        if candidate_id:
            pairs.append(
                (s1_id, candidate_id)
            )


pairs = pd.DataFrame(
    pairs,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
    ],
)

print(f"Total candidate pairs: {len(pairs):,}")


# ============================================================
# LABEL POSITIVE / NEGATIVE
# ============================================================

def is_positive(row):

    return (
        row["candidate_entity_id"]
        in true_matches.get(
            row["source1_entity_id"],
            set()
        )
    )


pairs["label"] = pairs.apply(
    is_positive,
    axis=1
)

positive_count = int(pairs["label"].sum())
negative_count = len(pairs) - positive_count

print()
print("Candidate quality:")
print(f"  Positive candidates : {positive_count:,}")
print(f"  Negative candidates : {negative_count:,}")
print(
    f"  Positive rate       : "
    f"{positive_count / len(pairs) * 100:.4f}%"
)


# ============================================================
# LOAD ONLY REQUIRED SOURCE RECORDS
# ============================================================

needed_s2 = set(
    pairs.loc[
        pairs["candidate_entity_id"].str.startswith("S2-"),
        "candidate_entity_id",
    ]
)

needed_s3 = set(
    pairs.loc[
        pairs["candidate_entity_id"].str.startswith("S3-"),
        "candidate_entity_id",
    ]
)

print()
print(f"Required S2 records: {len(needed_s2):,}")
print(f"Required S3 records: {len(needed_s3):,}")


def load_required_records(path, needed_ids):

    records = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        selected = chunk[
            chunk["entity_id"].isin(needed_ids)
        ]

        if len(selected):
            records.append(selected)

    if not records:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        )

    return pd.concat(
        records,
        ignore_index=True
    )


print("\nLoading required S2 records...")

s2 = load_required_records(
    SOURCE_FILES["S2"],
    needed_s2,
)

print(f"Loaded S2: {len(s2):,}")

print("\nLoading required S3 records...")

s3 = load_required_records(
    SOURCE_FILES["S3"],
    needed_s3,
)

print(f"Loaded S3: {len(s3):,}")


# ============================================================
# LOAD S1 RECORDS
# ============================================================

needed_s1 = set(sample_ids)

print("\nLoading S1 records...")

s1_parts = []

for chunk in pd.read_csv(
    "data/raw/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE,
):

    selected = chunk[
        chunk["entity_id"].isin(needed_s1)
    ]

    if len(selected):
        s1_parts.append(selected)


s1 = pd.concat(
    s1_parts,
    ignore_index=True
)

print(f"Loaded S1: {len(s1):,}")


# ============================================================
# CREATE LOOKUPS
# ============================================================

s1_lookup = s1.set_index("entity_id").to_dict("index")
s2_lookup = s2.set_index("entity_id").to_dict("index")
s3_lookup = s3.set_index("entity_id").to_dict("index")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_features(s1_record, candidate_record):

    name1 = normalize_text(
        s1_record["business_name"]
    )

    name2 = normalize_text(
        candidate_record["business_name"]
    )

    addr1 = normalize_text(
        s1_record["business_address"]
    )

    addr2 = normalize_text(
        candidate_record["business_address"]
    )

    name_tokens_1 = meaningful_tokens(
        name1,
        GENERIC_NAME_TOKENS
    )

    name_tokens_2 = meaningful_tokens(
        name2,
        GENERIC_NAME_TOKENS
    )

    addr_tokens_1 = meaningful_tokens(
        addr1,
        GENERIC_ADDRESS_TOKENS
    )

    addr_tokens_2 = meaningful_tokens(
        addr2,
        GENERIC_ADDRESS_TOKENS
    )

    numbers1 = extract_numbers(
        s1_record["business_address"]
    )

    numbers2 = extract_numbers(
        candidate_record["business_address"]
    )

    return {

        "name_exact":
            int(
                bool(name1)
                and name1 == name2
            ),

        "name_ratio":
            fuzz.ratio(name1, name2),

        "name_token_ratio":
            fuzz.token_set_ratio(name1, name2),

        "address_exact":
            int(
                bool(addr1)
                and addr1 == addr2
            ),

        "address_ratio":
            fuzz.ratio(addr1, addr2),

        "address_token_ratio":
            fuzz.token_set_ratio(addr1, addr2),

        "country_match":
            int(
                s1_record["country"]
                == candidate_record["country"]
            ),

        "shared_name_tokens":
            len(name_tokens_1 & name_tokens_2),

        "shared_address_tokens":
            len(addr_tokens_1 & addr_tokens_2),

        "shared_numbers":
            len(numbers1 & numbers2),

    }


# ============================================================
# COMPUTE FEATURES
# ============================================================

print("\nComputing pair features...")

results = []

missing_records = 0

for index, row in pairs.iterrows():

    s1_id = row["source1_entity_id"]
    candidate_id = row["candidate_entity_id"]

    s1_record = s1_lookup.get(s1_id)

    if candidate_id.startswith("S2-"):
        candidate_record = s2_lookup.get(candidate_id)
    else:
        candidate_record = s3_lookup.get(candidate_id)

    if s1_record is None or candidate_record is None:
        missing_records += 1
        continue

    features = calculate_features(
        s1_record,
        candidate_record
    )

    features["label"] = int(row["label"])

    features["source"] = (
        "S2"
        if candidate_id.startswith("S2-")
        else "S3"
    )

    results.append(features)


features_df = pd.DataFrame(results)

print(
    f"Computed features for "
    f"{len(features_df):,} pairs"
)

if missing_records:
    print(
        f"Missing records skipped: "
        f"{missing_records:,}"
    )


# ============================================================
# FEATURE SUMMARY
# ============================================================

feature_columns = [
    "name_exact",
    "name_ratio",
    "name_token_ratio",
    "address_exact",
    "address_ratio",
    "address_token_ratio",
    "shared_name_tokens",
    "shared_address_tokens",
    "shared_numbers",
]


def print_summary(df, title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for feature in feature_columns:

        values = df[feature]

        print(
            f"\n{feature}"
        )

        print(
            f"  mean   : {values.mean():.2f}"
        )

        print(
            f"  p10    : {values.quantile(.10):.2f}"
        )

        print(
            f"  p25    : {values.quantile(.25):.2f}"
        )

        print(
            f"  median : {values.median():.2f}"
        )

        print(
            f"  p75    : {values.quantile(.75):.2f}"
        )

        print(
            f"  p90    : {values.quantile(.90):.2f}"
        )

        print(
            f"  p95    : {values.quantile(.95):.2f}"
        )


positive_df = features_df[
    features_df["label"] == 1
]

negative_df = features_df[
    features_df["label"] == 0
]

print_summary(
    positive_df,
    "POSITIVE CANDIDATES"
)

print_summary(
    negative_df,
    "NEGATIVE CANDIDATES"
)


# ============================================================
# HIGH-SIMILARITY NEGATIVE ANALYSIS
# ============================================================

print()
print("=" * 70)
print("HIGH-SIMILARITY NEGATIVES")
print("=" * 70)

hard_negatives = negative_df[
    (
        negative_df["name_token_ratio"] >= 90
    )
    |
    (
        negative_df["address_token_ratio"] >= 90
    )
]

print(
    f"Negatives with name/address token "
    f"similarity >= 90: "
    f"{len(hard_negatives):,}"
)

print(
    f"Percentage of negatives: "
    f"{len(hard_negatives) / len(negative_df) * 100:.2f}%"
)


# ============================================================
# SAVE FEATURE SAMPLE
# ============================================================

output_path = (
    "data/processed/"
    "candidate_quality_features.tsv"
)

features_df.to_csv(
    output_path,
    sep="\t",
    index=False
)

print()
print(f"Saved features to: {output_path}")