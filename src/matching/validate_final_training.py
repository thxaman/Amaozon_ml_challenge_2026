import os
import hashlib
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = "data/processed/full_training_features.tsv"

RANDOM_STATE = 42

# Group split:
# same S1 entity can NEVER appear in both train and validation.
VAL_PERCENT = 0.10

# Keep validation manageable while preserving positives.
MAX_TRAIN_POSITIVES = 1_000_000
MAX_TRAIN_NEGATIVES = 1_000_000

MAX_VAL_POSITIVES = 250_000
MAX_VAL_NEGATIVES = 250_000


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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

    "source_is_s3",
]


# ============================================================
# GROUP ASSIGNMENT
# ============================================================

def is_validation_group(entity_id):
    """
    Deterministic group split.

    IMPORTANT:
    The split is based on S1 entity_id, so every pair belonging
    to the same S1 goes entirely into either train or validation.
    """

    digest = hashlib.md5(
        str(entity_id).encode("utf-8")
    ).hexdigest()

    value = int(digest[:8], 16)

    return (value % 100) < int(VAL_PERCENT * 100)


# ============================================================
# STORAGE
# ============================================================

train_pos = []
train_neg = []

val_pos = []
val_neg = []


train_pos_count = 0
train_neg_count = 0
val_pos_count = 0
val_neg_count = 0


# ============================================================
# STREAM FULL TRAINING FEATURES
# ============================================================

print("=" * 70)
print("FINAL TRAINING VALIDATION")
print("=" * 70)

print("\nReading full training features...")

usecols = [
    "source1_entity_id",
    "label"
] + FEATURE_COLUMNS


for chunk_idx, chunk in enumerate(
    pd.read_csv(
        FEATURE_PATH,
        sep="\t",
        usecols=usecols,
        chunksize=250_000
    ),
    start=1
):

    # --------------------------------------------------------
    # Group split
    # --------------------------------------------------------

    val_mask = chunk["source1_entity_id"].map(
        is_validation_group
    )

    train_chunk = chunk.loc[~val_mask]
    val_chunk = chunk.loc[val_mask]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_positive = train_chunk[
        train_chunk["label"] == 1
    ]

    train_negative = train_chunk[
        train_chunk["label"] == 0
    ]

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_positive = val_chunk[
        val_chunk["label"] == 1
    ]

    val_negative = val_chunk[
        val_chunk["label"] == 0
    ]

    # --------------------------------------------------------
    # Collect training rows
    # --------------------------------------------------------

    if len(train_positive) > 0:

        remaining = (
            MAX_TRAIN_POSITIVES
            - train_pos_count
        )

        if remaining > 0:

            take = train_positive.iloc[
                :remaining
            ]

            train_pos.append(take)

            train_pos_count += len(take)

    if len(train_negative) > 0:

        remaining = (
            MAX_TRAIN_NEGATIVES
            - train_neg_count
        )

        if remaining > 0:

            take = train_negative.iloc[
                :remaining
            ]

            train_neg.append(take)

            train_neg_count += len(take)

    # --------------------------------------------------------
    # Collect validation rows
    # --------------------------------------------------------

    if len(val_positive) > 0:

        remaining = (
            MAX_VAL_POSITIVES
            - val_pos_count
        )

        if remaining > 0:

            take = val_positive.iloc[
                :remaining
            ]

            val_pos.append(take)

            val_pos_count += len(take)

    if len(val_negative) > 0:

        remaining = (
            MAX_VAL_NEGATIVES
            - val_neg_count
        )

        if remaining > 0:

            take = val_negative.iloc[
                :remaining
            ]

            val_neg.append(take)

            val_neg_count += len(take)

    if chunk_idx % 10 == 0:

        print(
            f"Chunks: {chunk_idx:,} | "
            f"Train +: {train_pos_count:,} | "
            f"Train -: {train_neg_count:,} | "
            f"Val +: {val_pos_count:,} | "
            f"Val -: {val_neg_count:,}"
        )


# ============================================================
# COMBINE
# ============================================================

train_positive = pd.concat(
    train_pos,
    ignore_index=True
)

train_negative = pd.concat(
    train_neg,
    ignore_index=True
)

val_positive = pd.concat(
    val_pos,
    ignore_index=True
)

val_negative = pd.concat(
    val_neg,
    ignore_index=True
)


train_df = pd.concat(
    [train_positive, train_negative],
    ignore_index=True
)

val_df = pd.concat(
    [val_positive, val_negative],
    ignore_index=True
)


# Shuffle

train_df = train_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(drop=True)

val_df = val_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(drop=True)


X_train = train_df[FEATURE_COLUMNS].astype(np.float32)
y_train = train_df["label"].astype(np.int8)

X_val = val_df[FEATURE_COLUMNS].astype(np.float32)
y_val = val_df["label"].astype(np.int8)


print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print(f"Train rows : {len(X_train):,}")
print(f"  positives: {int(y_train.sum()):,}")
print(f"  negatives: {int((y_train == 0).sum()):,}")

print(f"\nVal rows   : {len(X_val):,}")
print(f"  positives: {int(y_val.sum()):,}")
print(f"  negatives: {int((y_val == 0).sum()):,}")


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

val_probability = model.predict_proba(
    X_val
)[:, 1]


roc_auc = roc_auc_score(
    y_val,
    val_probability
)

average_precision = average_precision_score(
    y_val,
    val_probability
)

print(f"ROC-AUC : {roc_auc:.6f}")
print(f"AP      : {average_precision:.6f}")


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):

    beta = 0.5

    denominator = (
        beta * beta * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta * beta)
        * precision
        * recall
        / denominator
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


results = []

for threshold in thresholds:

    predicted = (
        val_probability >= threshold
    )

    tp = np.sum(
        (predicted == 1) &
        (y_val.to_numpy() == 1)
    )

    fp = np.sum(
        (predicted == 1) &
        (y_val.to_numpy() == 0)
    )

    fn = np.sum(
        (predicted == 0) &
        (y_val.to_numpy() == 1)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    score = f05(
        precision,
        recall
    )

    results.append(
        (
            threshold,
            precision,
            recall,
            score
        )
    )

    print(
        f"{threshold:>5.2f} | "
        f"Precision {precision:.6f} | "
        f"Recall {recall:.6f} | "
        f"F0.5 {score:.6f}"
    )


best = max(
    results,
    key=lambda x: x[3]
)

print("\n" + "=" * 70)

print(
    f"BEST THRESHOLD: {best[0]:.2f}"
)

print(
    f"Precision      : {best[1]:.6f}"
)

print(
    f"Recall         : {best[2]:.6f}"
)

print(
    f"F0.5           : {best[3]:.6f}"
)

print("=" * 70)

print(
    "\nIMPORTANT:"
    "\nThis threshold is selected using TRAINING DATA only."
    "\nNo test data was used."
)