import pandas as pd
import time


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = (
    "data/processed/training_pair_features.tsv"
)


# ============================================================
# START
# ============================================================

start = time.time()

print("Loading feature dataset...")

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t"
)

print(
    f"Loaded: {len(df):,} rows"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(f"  {column}")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

label_counts = df["label"].value_counts()

print(
    f"Negative pairs : "
    f"{label_counts.get(0, 0):,}"
)

print(
    f"Positive pairs : "
    f"{label_counts.get(1, 0):,}"
)

positive_rate = (
    df["label"].mean()
)

print(
    f"Positive rate  : "
    f"{positive_rate:.6%}"
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

missing_found = False

for column, count in missing.items():

    if count > 0:

        missing_found = True

        print(
            f"{column:30s}: "
            f"{count:,}"
        )

if not missing_found:

    print("No missing values found.")


# ============================================================
# DUPLICATE PAIRS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE PAIRS")
print("=" * 70)

duplicate_columns = [
    "source1_entity_id",
    "candidate_entity_id"
]

duplicate_count = df.duplicated(
    subset=duplicate_columns
).sum()

print(
    f"Duplicate candidate pairs: "
    f"{duplicate_count:,}"
)


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("S2 / S3 DISTRIBUTION")
print("=" * 70)

source_stats = (
    df.groupby("source")["label"]
    .agg(
        total_pairs="count",
        positives="sum"
    )
)

source_stats["positive_rate"] = (
    source_stats["positives"]
    /
    source_stats["total_pairs"]
)

print(source_stats)


# ============================================================
# FEATURE LIST
# ============================================================

feature_columns = [
    "country_match",

    "name_exact",
    "name_ratio",
    "name_token_ratio",
    "name_partial_ratio",
    "name_length_ratio",
    "shared_name_tokens",

    "address_exact",
    "address_ratio",
    "address_token_ratio",
    "address_partial_ratio",
    "address_length_ratio",
    "shared_address_tokens",
    "shared_numbers",
]


# ============================================================
# POSITIVE / NEGATIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SEPARATION")
print("=" * 70)

positive_df = df[
    df["label"] == 1
]

negative_df = df[
    df["label"] == 0
]

statistics = []

for feature in feature_columns:

    positive_values = positive_df[
        feature
    ]

    negative_values = negative_df[
        feature
    ]

    statistics.append({

        "feature": feature,

        "positive_mean":
            positive_values.mean(),

        "negative_mean":
            negative_values.mean(),

        "positive_median":
            positive_values.median(),

        "negative_median":
            negative_values.median(),

        "positive_p90":
            positive_values.quantile(0.90),

        "negative_p90":
            negative_values.quantile(0.90),

    })


stats_df = pd.DataFrame(
    statistics
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print(
    stats_df.to_string(
        index=False
    )
)


# ============================================================
# HIGH-SIMILARITY NEGATIVES
# ============================================================

print("\n" + "=" * 70)
print("HARD NEGATIVES")
print("=" * 70)

hard_negative_mask = (

    (df["name_token_ratio"] >= 90)

    |

    (df["address_token_ratio"] >= 90)

)

hard_negatives = df[
    (df["label"] == 0)
    &
    hard_negative_mask
]

print(
    f"Hard negatives "
    f"(name_token_ratio >= 90 OR "
    f"address_token_ratio >= 90): "
    f"{len(hard_negatives):,}"
)

print(
    f"Percentage of all negatives: "
    f"{len(hard_negatives) / len(negative_df):.4%}"
)


# ============================================================
# VERY STRONG CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("VERY STRONG CANDIDATES")
print("=" * 70)

strong_mask = (

    (df["name_token_ratio"] >= 90)

    &

    (df["address_token_ratio"] >= 90)

)

strong_positive = df[
    (df["label"] == 1)
    &
    strong_mask
]

strong_negative = df[
    (df["label"] == 0)
    &
    strong_mask
]

print(
    f"Positive pairs: "
    f"{len(strong_positive):,}"
)

print(
    f"Negative pairs: "
    f"{len(strong_negative):,}"
)

strong_total = (
    len(strong_positive)
    +
    len(strong_negative)
)

if strong_total:

    print(
        f"Precision of this simple rule: "
        f"{len(strong_positive) / strong_total:.4%}"
    )


# ============================================================
# COUNTRY MISMATCH
# ============================================================

print("\n" + "=" * 70)
print("COUNTRY MATCH ANALYSIS")
print("=" * 70)

country_stats = (
    df.groupby(
        ["label", "country_match"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(country_stats)


# ============================================================
# EXACT MATCH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EXACT MATCH ANALYSIS")
print("=" * 70)

for feature in [
    "name_exact",
    "address_exact"
]:

    print(f"\n{feature}")

    result = (
        df.groupby(
            ["label", feature]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(result)


# ============================================================
# FEATURE CORRELATION WITH LABEL
# ============================================================

print("\n" + "=" * 70)
print("FEATURE / LABEL CORRELATION")
print("=" * 70)

correlations = (
    df[
        feature_columns + ["label"]
    ]
    .corr()["label"]
    .drop("label")
    .sort_values(
        ascending=False
    )
)

print(correlations)


# ============================================================
# SAMPLE POSITIVES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE POSITIVE PAIRS")
print("=" * 70)

print(
    positive_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "source",
            "country_match",
            "name_ratio",
            "name_token_ratio",
            "address_ratio",
            "address_token_ratio",
            "shared_name_tokens",
            "shared_address_tokens",
            "shared_numbers"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SAMPLE HARD NEGATIVES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE HARD NEGATIVES")
print("=" * 70)

print(
    hard_negatives[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "source",
            "country_match",
            "name_ratio",
            "name_token_ratio",
            "address_ratio",
            "address_token_ratio",
            "shared_name_tokens",
            "shared_address_tokens",
            "shared_numbers"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# FINISH
# ============================================================

elapsed = (
    time.time() - start
) / 60

print("\n" + "=" * 70)

print(
    f"Analysis completed in "
    f"{elapsed:.2f} minutes"
)

print("=" * 70)