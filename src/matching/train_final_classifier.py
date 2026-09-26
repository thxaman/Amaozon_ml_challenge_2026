import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = "data/processed/full_training_features.tsv"

MODEL_FILE = "data/processed/final_match_classifier.joblib"
TRAIN_SAMPLE_FILE = "data/processed/final_classifier_training_sample.tsv"
THRESHOLD_FILE = "data/processed/final_threshold_results.tsv"

RANDOM_STATE = 42

POSITIVE_SAMPLE = 1_000_000
HARD_NEGATIVE_SAMPLE = 33_610
ORDINARY_NEGATIVE_SAMPLE = 1_000_000

# Threshold used later for final test matching.
THRESHOLDS = [
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
    0.95,
]


# ============================================================
# FEATURES
# ============================================================

EXCLUDED_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "source",
    "label",
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading full feature dataset...", flush=True)

df = pd.read_csv(
    FEATURE_FILE,
    sep="\t",
)

print(
    f"Rows loaded: {len(df):,}",
    flush=True
)


# ============================================================
# IDENTIFY HARD NEGATIVES
# ============================================================

positive_mask = df["label"] == 1
negative_mask = df["label"] == 0

hard_negative_mask = (
    negative_mask
    &
    (
        (df["name_token_ratio"] >= 90)
        |
        (df["address_token_ratio"] >= 90)
        |
        (
            (df["name_token_ratio"] >= 80)
            &
            (df["address_token_ratio"] >= 80)
        )
    )
)

ordinary_negative_mask = (
    negative_mask
    & ~hard_negative_mask
)

print("\nAvailable data:", flush=True)

print(
    f"Positives: {positive_mask.sum():,}",
    flush=True
)

print(
    f"Hard negatives: {hard_negative_mask.sum():,}",
    flush=True
)

print(
    f"Ordinary negatives: {ordinary_negative_mask.sum():,}",
    flush=True
)


# ============================================================
# SAMPLE TRAINING DATA
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)


def sample_rows(mask, n, name):

    indices = np.flatnonzero(mask.to_numpy())

    if len(indices) < n:
        raise ValueError(
            f"Not enough rows for {name}: "
            f"requested {n:,}, available {len(indices):,}"
        )

    selected = rng.choice(
        indices,
        size=n,
        replace=False,
    )

    print(
        f"{name}: selected {len(selected):,}",
        flush=True
    )

    return df.iloc[selected].copy()


print("\nSampling training data...", flush=True)

positive_sample = sample_rows(
    positive_mask,
    POSITIVE_SAMPLE,
    "Positive examples",
)

hard_negative_sample = sample_rows(
    hard_negative_mask,
    HARD_NEGATIVE_SAMPLE,
    "Hard negative examples",
)

ordinary_negative_sample = sample_rows(
    ordinary_negative_mask,
    ORDINARY_NEGATIVE_SAMPLE,
    "Ordinary negative examples",
)


train_df = pd.concat(
    [
        positive_sample,
        hard_negative_sample,
        ordinary_negative_sample,
    ],
    ignore_index=True,
)

# Shuffle.
train_df = train_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE,
).reset_index(drop=True)


print(
    f"\nFinal sampled dataset: {len(train_df):,}",
    flush=True
)

print(
    "\nLabel distribution:",
    flush=True
)

print(
    train_df["label"].value_counts(),
    flush=True
)

print(
    "\nSource distribution:",
    flush=True
)

print(
    train_df["source"].value_counts(),
    flush=True
)


# Save the exact training sample.
train_df.to_csv(
    TRAIN_SAMPLE_FILE,
    sep="\t",
    index=False,
)

print(
    f"\nTraining sample saved to: {TRAIN_SAMPLE_FILE}",
    flush=True
)


# ============================================================
# PREPARE MODEL MATRIX
# ============================================================

feature_columns = [
    column
    for column in train_df.columns
    if column not in EXCLUDED_COLUMNS
]

X = train_df[feature_columns].astype(np.float32)
y = train_df["label"].astype(np.int8)

groups = train_df["source1_entity_id"]


print(
    f"\nModel features: {len(feature_columns)}",
    flush=True
)


# ============================================================
# GROUPED TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating grouped validation split...", flush=True)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

train_groups = groups.iloc[train_idx]
val_groups = groups.iloc[val_idx]

overlap = len(
    set(train_groups).intersection(
        set(val_groups)
    )
)

print(
    f"Training rows: {len(X_train):,}",
    flush=True
)

print(
    f"Validation rows: {len(X_val):,}",
    flush=True
)

print(
    f"Training S1 groups: {train_groups.nunique():,}",
    flush=True
)

print(
    f"Validation S1 groups: {val_groups.nunique():,}",
    flush=True
)

print(
    f"S1 group overlap: {overlap}",
    flush=True
)


# ============================================================
# TRAIN DEVELOPMENT MODEL
# ============================================================

print("\nTraining classifier...", flush=True)

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=250,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)

model.fit(
    X_train,
    y_train,
)

print("Training complete.", flush=True)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation probabilities...", flush=True)

val_probability = model.predict_proba(
    X_val
)[:, 1]


roc_auc = roc_auc_score(
    y_val,
    val_probability,
)

average_precision = average_precision_score(
    y_val,
    val_probability,
)

print(
    f"ROC-AUC: {roc_auc:.6f}",
    flush=True
)

print(
    f"Average Precision: {average_precision:.6f}",
    flush=True
)


# ============================================================
# EXACT S1 MACRO F0.5
# ============================================================

validation_s1 = groups.iloc[val_idx].to_numpy()
validation_labels = y_val.to_numpy()


def f05(precision, recall):

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = 0.25

    return (
        (1 + beta_squared)
        * precision
        * recall
        /
        (
            beta_squared * precision
            + recall
        )
    )


results = []


print(
    "\nThreshold evaluation:",
    flush=True
)

print(
    "threshold | macro_f05 | pair_precision | pair_recall | pair_f05",
    flush=True
)


for threshold in THRESHOLDS:

    predicted = (
        val_probability >= threshold
    )

    # --------------------------------------------------------
    # Pair metrics
    # --------------------------------------------------------

    tp = np.sum(
        predicted & (validation_labels == 1)
    )

    fp = np.sum(
        predicted & (validation_labels == 0)
    )

    fn = np.sum(
        (~predicted) & (validation_labels == 1)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    pair_f05 = f05(
        precision,
        recall,
    )

    # --------------------------------------------------------
    # Exact per-S1 Macro F0.5
    # --------------------------------------------------------

    validation_frame = pd.DataFrame({
        "s1": validation_s1,
        "label": validation_labels,
        "predicted": predicted,
    })

    s1_scores = []

    for _, group in validation_frame.groupby("s1"):

        true_values = group["label"].to_numpy()
        pred_values = group["predicted"].to_numpy()

        tp_s1 = np.sum(
            pred_values & (true_values == 1)
        )

        fp_s1 = np.sum(
            pred_values & (true_values == 0)
        )

        fn_s1 = np.sum(
            (~pred_values) & (true_values == 1)
        )

        precision_s1 = (
            tp_s1 / (tp_s1 + fp_s1)
            if (tp_s1 + fp_s1) > 0
            else 0.0
        )

        recall_s1 = (
            tp_s1 / (tp_s1 + fn_s1)
            if (tp_s1 + fn_s1) > 0
            else 0.0
        )

        s1_scores.append(
            f05(
                precision_s1,
                recall_s1,
            )
        )

    macro_f05 = float(
        np.mean(s1_scores)
    )

    results.append({
        "threshold": threshold,
        "macro_f05": macro_f05,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f05": pair_f05,
    })

    print(
        f"{threshold:.2f}       | "
        f"{macro_f05:.6f}   | "
        f"{precision:.6f}        | "
        f"{recall:.6f}     | "
        f"{pair_f05:.6f}",
        flush=True
    )


# ============================================================
# SELECT THRESHOLD
# ============================================================

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["macro_f05"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

print(
    "\nBest validation threshold:",
    best_threshold,
    flush=True
)

print(
    f"Best S1 Macro F0.5: "
    f"{best_row['macro_f05']:.6f}",
    flush=True
)


results_df.to_csv(
    THRESHOLD_FILE,
    sep="\t",
    index=False,
)


# ============================================================
# RETRAIN FINAL MODEL ON ENTIRE SAMPLED DATASET
# ============================================================

print(
    "\nRetraining final classifier on all "
    "sampled training data...",
    flush=True
)

final_model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=250,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)

final_model.fit(
    X,
    y,
)

print(
    "Final model training complete.",
    flush=True
)


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

model_package = {
    "model": final_model,
    "feature_columns": feature_columns,
    "threshold": best_threshold,
    "random_state": RANDOM_STATE,
}

joblib.dump(
    model_package,
    MODEL_FILE,
)

print(
    f"\nFinal model saved to: {MODEL_FILE}",
    flush=True
)

print(
    f"Final threshold: {best_threshold}",
    flush=True
)

print("\nDONE.", flush=True)