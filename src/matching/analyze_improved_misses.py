import os
import re
import unicodedata
import pandas as pd
import numpy as np
from rapidfuzz import fuzz


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42
CHUNK_SIZE = 250_000

S1_PATH = "data/raw/train_source1.tsv"
S2_PATH = "data/raw/train_source2.tsv"
S3_PATH = "data/raw/train_source3.tsv"

GT_PATH = "data/raw/train_ground_truth.tsv"

CANDIDATE_PATH = (
    "data/processed/improved_candidate_pairs.tsv"
)

OUTPUT_PATH = (
    "data/processed/improved_missed_pairs.tsv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.lower()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def tokenize(value):
    if not value:
        return set()

    return set(value.split())


def extract_numbers(value):
    if not value:
        return set()

    return set(
        re.findall(r"\d+", value)
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
].copy()

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

s1 = s1.set_index("entity_id")

print(
    f"Loaded S1: {len(s1):,}"
)


# ============================================================
# LOAD IMPROVED CANDIDATES
# ============================================================

print("\nLoading improved candidates...")

candidates_df = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str
)

candidate_map = {}

for _, row in candidates_df.iterrows():

    s1_id = row["source1_entity_id"]

    value = row["candidate_entity_ids"]

    if pd.isna(value) or not str(value).strip():

        candidate_map[s1_id] = set()

    else:

        candidate_map[s1_id] = set(
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        )

print(
    f"Loaded candidate rows: "
    f"{len(candidate_map):,}"
)


# ============================================================
# FIND MISSED TRUE PAIRS
# ============================================================

missed_pairs = []

total_true = 0
total_recovered = 0

for s1_id in sample_ids:

    true_set = true_matches.get(
        s1_id,
        set()
    )

    candidate_set = candidate_map.get(
        s1_id,
        set()
    )

    total_true += len(true_set)

    recovered = (
        true_set & candidate_set
    )

    total_recovered += len(recovered)

    missed = (
        true_set - candidate_set
    )

    for target_id in missed:

        missed_pairs.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_id": target_id
            }
        )


print()
print("=" * 70)
print("MISSED PAIRS")
print("=" * 70)

print(
    f"True pairs      : {total_true:,}"
)

print(
    f"Recovered pairs : {total_recovered:,}"
)

print(
    f"Missed pairs    : "
    f"{total_true - total_recovered:,}"
)

print(
    f"Recall          : "
    f"{total_recovered / total_true:.4f}"
)


# ============================================================
# LOAD ONLY NEEDED S2/S3 RECORDS
# ============================================================

missed_df = pd.DataFrame(
    missed_pairs
)

if missed_df.empty:

    print("\nNo missed pairs!")

    raise SystemExit


missed_ids = set(
    missed_df["matched_entity_id"]
)


print()
print(
    f"Unique missed source entities: "
    f"{len(missed_ids):,}"
)


def load_matching_records(
    path,
    target_ids
):

    records = []

    target_ids = set(target_ids)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        selected = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        if not selected.empty:
            records.append(selected)

    if not records:

        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        )

    return pd.concat(
        records,
        ignore_index=True
    )


print("\nLoading missed S2/S3 records...")

s2 = load_matching_records(
    S2_PATH,
    missed_ids
)

s3 = load_matching_records(
    S3_PATH,
    missed_ids
)

print(
    f"Loaded S2 missed records: "
    f"{len(s2):,}"
)

print(
    f"Loaded S3 missed records: "
    f"{len(s3):,}"
)


# ============================================================
# COMBINE S2 + S3
# ============================================================

source_records = pd.concat(
    [
        s2.assign(source="S2"),
        s3.assign(source="S3")
    ],
    ignore_index=True
)

source_records = source_records.set_index(
    "entity_id"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

results = []

for _, row in missed_df.iterrows():

    s1_id = row[
        "source1_entity_id"
    ]

    target_id = row[
        "matched_entity_id"
    ]

    if s1_id not in s1.index:
        continue

    if target_id not in source_records.index:
        continue

    s1_row = s1.loc[s1_id]
    target = source_records.loc[target_id]

    # --------------------------------------------------------
    # Text normalization
    # --------------------------------------------------------

    s1_name = normalize_text(
        s1_row["business_name"]
    )

    target_name = normalize_text(
        target["business_name"]
    )

    s1_address = normalize_text(
        s1_row["business_address"]
    )

    target_address = normalize_text(
        target["business_address"]
    )

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    s1_name_tokens = tokenize(
        s1_name
    )

    target_name_tokens = tokenize(
        target_name
    )

    s1_address_tokens = tokenize(
        s1_address
    )

    target_address_tokens = tokenize(
        target_address
    )

    # --------------------------------------------------------
    # Similarities
    # --------------------------------------------------------

    name_ratio = fuzz.ratio(
        s1_name,
        target_name
    )

    name_token_ratio = fuzz.token_set_ratio(
        s1_name,
        target_name
    )

    address_ratio = fuzz.ratio(
        s1_address,
        target_address
    )

    address_token_ratio = fuzz.token_set_ratio(
        s1_address,
        target_address
    )

    # --------------------------------------------------------
    # Shared tokens
    # --------------------------------------------------------

    shared_name_tokens = (
        s1_name_tokens
        & target_name_tokens
    )

    shared_address_tokens = (
        s1_address_tokens
        & target_address_tokens
    )

    # --------------------------------------------------------
    # Shared numbers
    # --------------------------------------------------------

    s1_numbers = extract_numbers(
        s1_address
    )

    target_numbers = extract_numbers(
        target_address
    )

    shared_numbers = (
        s1_numbers
        & target_numbers
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    same_country = int(
        str(s1_row["country"]).strip().lower()
        ==
        str(target["country"]).strip().lower()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.append(
        {
            "source1_entity_id": s1_id,
            "matched_entity_id": target_id,
            "source": target["source"],

            "s1_name": s1_row["business_name"],
            "matched_name": target["business_name"],

            "s1_address": s1_row["business_address"],
            "matched_address": target["business_address"],

            "name_ratio": name_ratio,
            "name_token_ratio": name_token_ratio,

            "address_ratio": address_ratio,
            "address_token_ratio": address_token_ratio,

            "shared_name_tokens":
                len(shared_name_tokens),

            "shared_address_tokens":
                len(shared_address_tokens),

            "shared_numbers":
                len(shared_numbers),

            "same_country":
                same_country
        }
    )


result_df = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("MISSED-PAIR SIMILARITY ANALYSIS")
print("=" * 70)

print(
    f"Missed pairs analyzed: "
    f"{len(result_df):,}"
)

metrics = [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "shared_name_tokens",
    "shared_address_tokens",
    "shared_numbers"
]

for metric in metrics:

    series = result_df[metric]

    print()
    print(metric)

    print(
        f"  mean   : {series.mean():.2f}"
    )

    print(
        f"  p10    : {series.quantile(.10):.2f}"
    )

    print(
        f"  p25    : {series.quantile(.25):.2f}"
    )

    print(
        f"  median : {series.median():.2f}"
    )

    print(
        f"  p75    : {series.quantile(.75):.2f}"
    )

    print(
        f"  p90    : {series.quantile(.90):.2f}"
    )

    print(
        f"  p95    : {series.quantile(.95):.2f}"
    )

    print(
        f"  max    : {series.max():.2f}"
    )


# ============================================================
# THRESHOLD COUNTS
# ============================================================

print()
print("=" * 70)
print("HIGH-SIMILARITY MISSED PAIRS")
print("=" * 70)

for threshold in [50, 60, 70, 80, 90]:

    print()
    print(
        f"Threshold >= {threshold}"
    )

    for metric in [
        "name_ratio",
        "name_token_ratio",
        "address_ratio",
        "address_token_ratio"
    ]:

        count = (
            result_df[metric] >= threshold
        ).sum()

        percentage = (
            count / len(result_df) * 100
        )

        print(
            f"  {metric:<24}"
            f"{count:>7,}"
            f" ({percentage:>6.2f}%)"
        )


# ============================================================
# COMBINED SIGNALS
# ============================================================

print()
print("=" * 70)
print("COMBINED HIGH-SIMILARITY SIGNALS")
print("=" * 70)

conditions = {

    "name_token >= 90":
        result_df["name_token_ratio"] >= 90,

    "address_token >= 90":
        result_df["address_token_ratio"] >= 90,

    "name_token >= 90 OR address_token >= 90":
        (
            (result_df["name_token_ratio"] >= 90)
            |
            (result_df["address_token_ratio"] >= 90)
        ),

    "name_token >= 80 AND address_token >= 80":
        (
            (result_df["name_token_ratio"] >= 80)
            &
            (result_df["address_token_ratio"] >= 80)
        ),

    "name >= 80 AND address >= 80":
        (
            (result_df["name_ratio"] >= 80)
            &
            (result_df["address_ratio"] >= 80)
        ),

    "shared address tokens >= 3":
        result_df["shared_address_tokens"] >= 3,

    "shared numbers >= 1":
        result_df["shared_numbers"] >= 1,
}

for name, condition in conditions.items():

    count = condition.sum()

    percentage = (
        count / len(result_df) * 100
    )

    print(
        f"{name:<45}"
        f"{count:>7,}"
        f" ({percentage:>6.2f}%)"
    )


# ============================================================
# SOURCE BREAKDOWN
# ============================================================

print()
print("=" * 70)
print("MISSED PAIRS BY SOURCE")
print("=" * 70)

source_counts = (
    result_df["source"]
    .value_counts()
)

for source, count in source_counts.items():

    percentage = (
        count / len(result_df) * 100
    )

    print(
        f"{source}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# EXAMPLES
# ============================================================

print()
print("=" * 70)
print("HIGH-SIMILARITY MISSED EXAMPLES")
print("=" * 70)

examples = (
    result_df
    .assign(
        combined_score=lambda x:
            (
                x["name_token_ratio"]
                +
                x["address_token_ratio"]
            ) / 2
    )
    .sort_values(
        "combined_score",
        ascending=False
    )
    .head(20)
)

for _, row in examples.iterrows():

    print()
    print(
        f"S1: {row['source1_entity_id']}"
    )

    print(
        f"Matched: {row['matched_entity_id']}"
    )

    print(
        f"Source: {row['source']}"
    )

    print(
        f"Name: "
        f"{row['s1_name']} "
        f"<-> "
        f"{row['matched_name']}"
    )

    print(
        f"Address: "
        f"{row['s1_address']} "
        f"<-> "
        f"{row['matched_address']}"
    )

    print(
        f"Name ratio: "
        f"{row['name_ratio']:.2f}"
    )

    print(
        f"Name token ratio: "
        f"{row['name_token_ratio']:.2f}"
    )

    print(
        f"Address ratio: "
        f"{row['address_ratio']:.2f}"
    )

    print(
        f"Address token ratio: "
        f"{row['address_token_ratio']:.2f}"
    )

    print(
        f"Shared name tokens: "
        f"{row['shared_name_tokens']}"
    )

    print(
        f"Shared address tokens: "
        f"{row['shared_address_tokens']}"
    )

    print(
        f"Shared numbers: "
        f"{row['shared_numbers']}"
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print()
print(
    f"Saved missed-pair analysis to: "
    f"{OUTPUT_PATH}"
)