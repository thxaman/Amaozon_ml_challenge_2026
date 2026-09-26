import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "training_pair_features.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "training_pair_features_improved.tsv"
)


# ============================================================
# LOAD EXISTING FEATURES
# ============================================================

print(
    "Loading existing pair features...",
    flush=True
)

df = pd.read_csv(
    INPUT_PATH,
    sep="\t"
)

print(
    f"Loaded rows: {len(df):,}",
    flush=True
)

print(
    f"Existing columns: {len(df.columns)}",
    flush=True
)


# ============================================================
# CONVERT SIMILARITY FEATURES
# ============================================================

print(
    "\nCreating interaction features...",
    flush=True
)

# ------------------------------------------------------------
# Strong evidence flags
# ------------------------------------------------------------

df["name_strong_90"] = (
    df["name_token_ratio"] >= 90
).astype(np.int8)

df["address_strong_90"] = (
    df["address_token_ratio"] >= 90
).astype(np.int8)

df["name_strong_80"] = (
    df["name_token_ratio"] >= 80
).astype(np.int8)

df["address_strong_80"] = (
    df["address_token_ratio"] >= 80
).astype(np.int8)


# ------------------------------------------------------------
# Joint strong evidence
# ------------------------------------------------------------

df["both_strong_90"] = (
    (df["name_token_ratio"] >= 90)
    &
    (df["address_token_ratio"] >= 90)
).astype(np.int8)

df["both_strong_80"] = (
    (df["name_token_ratio"] >= 80)
    &
    (df["address_token_ratio"] >= 80)
).astype(np.int8)


# ------------------------------------------------------------
# One-sided strong evidence
# ------------------------------------------------------------

df["name_strong_address_weak"] = (
    (df["name_token_ratio"] >= 90)
    &
    (df["address_token_ratio"] < 70)
).astype(np.int8)

df["address_strong_name_weak"] = (
    (df["address_token_ratio"] >= 90)
    &
    (df["name_token_ratio"] < 70)
).astype(np.int8)


# ============================================================
# COMBINED SIMILARITY FEATURES
# ============================================================

name_score = (
    df["name_token_ratio"]
    / 100.0
)

address_score = (
    df["address_token_ratio"]
    / 100.0
)


# ------------------------------------------------------------
# Average evidence
# ------------------------------------------------------------

df["name_address_mean"] = (
    name_score + address_score
) / 2.0


# ------------------------------------------------------------
# Minimum evidence
# ------------------------------------------------------------

df["name_address_min"] = np.minimum(
    name_score,
    address_score
)


# ------------------------------------------------------------
# Maximum evidence
# ------------------------------------------------------------

df["name_address_max"] = np.maximum(
    name_score,
    address_score
)


# ------------------------------------------------------------
# Difference between name and address evidence
# ------------------------------------------------------------

df["name_address_gap"] = np.abs(
    name_score - address_score
)


# ------------------------------------------------------------
# Combined product
#
# High only when BOTH pieces of evidence are high.
# ------------------------------------------------------------

df["name_address_product"] = (
    name_score * address_score
)


# ============================================================
# EXACT + STRONG EVIDENCE COMBINATIONS
# ============================================================

df["exact_name_strong_address"] = (
    (df["name_exact"] == 1)
    &
    (df["address_token_ratio"] >= 80)
).astype(np.int8)

df["exact_address_strong_name"] = (
    (df["address_exact"] == 1)
    &
    (df["name_token_ratio"] >= 80)
).astype(np.int8)


# ============================================================
# TOKEN EVIDENCE
# ============================================================

df["name_tokens_strong"] = (
    df["shared_name_tokens"] >= 2
).astype(np.int8)

df["address_tokens_strong"] = (
    df["shared_address_tokens"] >= 3
).astype(np.int8)

df["address_tokens_very_strong"] = (
    df["shared_address_tokens"] >= 5
).astype(np.int8)


# ============================================================
# NUMBER EVIDENCE
# ============================================================

df["has_shared_number"] = (
    df["shared_numbers"] >= 1
).astype(np.int8)

df["multiple_shared_numbers"] = (
    df["shared_numbers"] >= 2
).astype(np.int8)


# Number evidence becomes more useful when address similarity
# is also reasonably strong.

df["strong_address_with_number"] = (
    (df["address_token_ratio"] >= 80)
    &
    (df["shared_numbers"] >= 1)
).astype(np.int8)

df["very_strong_address_with_number"] = (
    (df["address_token_ratio"] >= 90)
    &
    (df["shared_numbers"] >= 1)
).astype(np.int8)


# Number evidence + name evidence

df["strong_name_with_number"] = (
    (df["name_token_ratio"] >= 80)
    &
    (df["shared_numbers"] >= 1)
).astype(np.int8)


# ============================================================
# EVIDENCE COUNT
# ============================================================

df["strong_evidence_count"] = (
    (df["name_token_ratio"] >= 80).astype(np.int8)
    +
    (df["address_token_ratio"] >= 80).astype(np.int8)
    +
    (df["shared_name_tokens"] >= 2).astype(np.int8)
    +
    (df["shared_address_tokens"] >= 3).astype(np.int8)
    +
    (df["shared_numbers"] >= 1).astype(np.int8)
)


# ============================================================
# VERY STRONG EVIDENCE COUNT
# ============================================================

df["very_strong_evidence_count"] = (
    (df["name_token_ratio"] >= 90).astype(np.int8)
    +
    (df["address_token_ratio"] >= 90).astype(np.int8)
    +
    (df["name_exact"] == 1).astype(np.int8)
    +
    (df["address_exact"] == 1).astype(np.int8)
    +
    (df["shared_numbers"] >= 2).astype(np.int8)
)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print(
    "\nChecking generated features...",
    flush=True
)

new_columns = [
    "name_strong_90",
    "address_strong_90",
    "name_strong_80",
    "address_strong_80",
    "both_strong_90",
    "both_strong_80",
    "name_strong_address_weak",
    "address_strong_name_weak",
    "name_address_mean",
    "name_address_min",
    "name_address_max",
    "name_address_gap",
    "name_address_product",
    "exact_name_strong_address",
    "exact_address_strong_name",
    "name_tokens_strong",
    "address_tokens_strong",
    "address_tokens_very_strong",
    "has_shared_number",
    "multiple_shared_numbers",
    "strong_address_with_number",
    "very_strong_address_with_number",
    "strong_name_with_number",
    "strong_evidence_count",
    "very_strong_evidence_count",
]


print(
    f"New features: {len(new_columns)}",
    flush=True
)

print(
    f"Total columns: {len(df.columns)}",
    flush=True
)


# ============================================================
# CHECK NULLS
# ============================================================

null_count = int(
    df[new_columns].isna().sum().sum()
)

print(
    f"New-feature null values: {null_count:,}",
    flush=True
)

if null_count > 0:

    raise ValueError(
        "New features contain missing values."
    )


# ============================================================
# FEATURE SUMMARY
# ============================================================

print(
    "\nNew feature statistics:",
    flush=True
)

print(
    df[new_columns].describe().T[
        [
            "mean",
            "50%",
            "max"
        ]
    ].to_string()
)


# ============================================================
# SAVE
# ============================================================

print(
    "\nSaving improved feature file...",
    flush=True
)

df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print(
    "\nImproved feature file saved to:",
    flush=True
)

print(
    OUTPUT_PATH,
    flush=True
)

print(
    f"\nRows: {len(df):,}",
    flush=True
)

print(
    f"Columns: {len(df.columns):,}",
    flush=True
)

print(
    "\nDone.",
    flush=True
)