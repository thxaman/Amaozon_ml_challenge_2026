import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = (
    "data/processed/training_pair_features.tsv"
)

RANDOM_STATE = 42

NEGATIVE_SAMPLE_SIZE = 200_000


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
    f"Loaded rows: {len(df):,}"
)


# ============================================================
# FEATURE COLUMNS
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
]


# ============================================================
# SPLIT POSITIVES / NEGATIVES
# ============================================================

print("\nSeparating positives and negatives...")

positive_df = df[
    df["label"] == 1
].copy()

negative_df = df[
    df["label"] == 0
].copy()

print(
    f"Positive rows: "
    f"{len(positive_df):,}"
)

print(
    f"Negative rows: "
    f"{len(negative_df):,}"
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

print(
    f"\nSampling "
    f"{NEGATIVE_SAMPLE_SIZE:,} negatives..."
)

negative_sample = negative_df.sample(
    n=min(
        NEGATIVE_SAMPLE_SIZE,
        len(negative_df)
    ),
    random_state=RANDOM_STATE
)


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

model_df = pd.concat(
    [
        positive_df,
        negative_sample
    ],
    ignore_index=True
)

# Shuffle
model_df = model_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"\nModel dataset: "
    f"{len(model_df):,} rows"
)

print(
    f"Positives: "
    f"{(model_df['label'] == 1).sum():,}"
)

print(
    f"Negatives: "
    f"{(model_df['label'] == 0).sum():,}"
)


# ============================================================
# X / y
# ============================================================

X = model_df[
    FEATURE_COLUMNS
].astype(np.float32)

y = model_df[
    "label"
].astype(np.int8)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train/validation split...")

X_train, X_valid, y_train, y_valid = (
    train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print(
    f"Training rows  : "
    f"{len(X_train):,}"
)

print(
    f"Validation rows: "
    f"{len(X_valid):,}"
)


# ============================================================
# MODEL
# ============================================================

print("\nTraining HistGradientBoostingClassifier...")

model = HistGradientBoostingClassifier(

    learning_rate=0.08,

    max_iter=250,

    max_leaf_nodes=31,

    min_samples_leaf=50,

    l2_regularization=1.0,

    random_state=RANDOM_STATE
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation probabilities...")

probabilities = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

roc_auc = roc_auc_score(
    y_valid,
    probabilities
)

average_precision = (
    average_precision_score(
        y_valid,
        probabilities
    )
)

print(
    f"ROC-AUC       : "
    f"{roc_auc:.6f}"
)

print(
    f"Average Precision: "
    f"{average_precision:.6f}"
)


# ============================================================
# PRECISION / RECALL CURVE
# ============================================================

precision, recall, thresholds = (
    precision_recall_curve(
        y_valid,
        probabilities
    )
)


# ============================================================
# F0.5
# ============================================================

beta = 0.5

f_beta = (
    (1 + beta ** 2)
    * precision
    * recall
    /
    (
        beta ** 2 * precision
        + recall
        + 1e-12
    )
)


best_index = np.argmax(
    f_beta
)

best_f05 = f_beta[
    best_index
]

# precision_recall_curve has one
# extra precision/recall point
if best_index < len(thresholds):

    best_threshold = thresholds[
        best_index
    ]

else:

    best_threshold = 0.5


print(
    f"\nBest F0.5      : "
    f"{best_f05:.6f}"
)

print(
    f"Best threshold : "
    f"{best_threshold:.6f}"
)

print(
    f"Precision       : "
    f"{precision[best_index]:.6f}"
)

print(
    f"Recall          : "
    f"{recall[best_index]:.6f}"
)


# ============================================================
# THRESHOLD CHECK
# ============================================================

print("\nThreshold comparison:")

for threshold in [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99
]:

    predictions = (
        probabilities >= threshold
    )

    tp = np.sum(
        (predictions == 1)
        &
        (y_valid.to_numpy() == 1)
    )

    fp = np.sum(
        (predictions == 1)
        &
        (y_valid.to_numpy() == 0)
    )

    fn = np.sum(
        (predictions == 0)
        &
        (y_valid.to_numpy() == 1)
    )

    precision_value = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall_value = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f05_value = (
        (1 + beta ** 2)
        * precision_value
        * recall_value
        /
        (
            beta ** 2 * precision_value
            + recall_value
            + 1e-12
        )
        if precision_value + recall_value > 0
        else 0
    )

    print(
        f"  threshold={threshold:.2f} "
        f"precision={precision_value:.4f} "
        f"recall={recall_value:.4f} "
        f"F0.5={f05_value:.4f}"
    )


# ============================================================
# FINISH
# ============================================================

elapsed = (
    time.time() - start
) / 60

print("\n" + "=" * 70)

print(
    f"Completed in "
    f"{elapsed:.2f} minutes"
)

print("=" * 70)