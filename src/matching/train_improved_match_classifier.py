import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
NEGATIVE_SAMPLE_SIZE = 200_000

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "training_pair_features_improved.tsv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "improved_classifier_threshold_results.tsv"
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading improved pair features...",
    flush=True
)

df = pd.read_csv(
    INPUT_PATH,
    sep="\t"
)

print(
    f"Rows: {len(df):,}",
    flush=True
)

print(
    f"Columns: {len(df.columns):,}",
    flush=True
)


# ============================================================
# TARGET
# ============================================================

TARGET = "label"

if TARGET not in df.columns:
    raise ValueError(
        f"Missing target column: {TARGET}"
    )


# ============================================================
# ENCODE SOURCE
# ============================================================
#
# Original source column contains:
#
# S2
# S3
#
# HistGradientBoosting requires numerical features.
#
# S2 -> 0
# S3 -> 1
#
# ============================================================

print(
    "\nEncoding source...",
    flush=True
)

unknown_sources = set(
    df["source"].dropna().unique()
) - {"S2", "S3"}

if unknown_sources:

    raise ValueError(
        f"Unexpected source values: {unknown_sources}"
    )

df["source_is_s3"] = (
    df["source"] == "S3"
).astype(np.int8)


print(
    "Source encoding:",
    flush=True
)

print(
    df["source_is_s3"].value_counts().sort_index().to_string(),
    flush=True
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "source",
    TARGET,
}


feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]


print(
    f"\nFeatures used: {len(feature_columns)}",
    flush=True
)

print(
    feature_columns,
    flush=True
)


# ============================================================
# CHECK FEATURE TYPES
# ============================================================

non_numeric_features = [
    column
    for column in feature_columns
    if not pd.api.types.is_numeric_dtype(
        df[column]
    )
]

if non_numeric_features:

    raise ValueError(
        "Non-numeric features found:\n"
        + "\n".join(non_numeric_features)
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

feature_nulls = int(
    df[feature_columns].isna().sum().sum()
)

if feature_nulls > 0:

    raise ValueError(
        f"Feature matrix contains "
        f"{feature_nulls:,} missing values."
    )

print(
    "\nFeature validation passed.",
    flush=True
)


# ============================================================
# POSITIVE / NEGATIVE PAIRS
# ============================================================

positive_df = df[
    df[TARGET] == 1
].copy()

negative_df = df[
    df[TARGET] == 0
].copy()


print(
    f"\nPositive pairs: {len(positive_df):,}",
    flush=True
)

print(
    f"Negative pairs: {len(negative_df):,}",
    flush=True
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================
#
# Keep the exact same sampling strategy as the baseline
# classifier so that the comparison remains fair.
#
# ============================================================

if len(negative_df) > NEGATIVE_SAMPLE_SIZE:

    negative_sample = negative_df.sample(
        n=NEGATIVE_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

else:

    negative_sample = negative_df.copy()


# ============================================================
# BUILD MODEL DATASET
# ============================================================

model_df = pd.concat(
    [
        positive_df,
        negative_sample
    ],
    ignore_index=True
)


# Shuffle rows.

model_df = model_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"\nModel dataset: {len(model_df):,}",
    flush=True
)

print(
    "Class distribution:",
    flush=True
)

print(
    model_df[TARGET]
    .value_counts()
    .sort_index()
    .to_string(),
    flush=True
)


# ============================================================
# GROUPED TRAIN / VALIDATION SPLIT
# ============================================================
#
# Important:
#
# All candidate pairs belonging to the same S1 entity stay
# in either train OR validation.
#
# This prevents the same S1 business from appearing in both.
#
# ============================================================

print(
    "\nCreating grouped validation split...",
    flush=True
)

groups = model_df[
    "source1_entity_id"
]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)


train_idx, val_idx = next(
    splitter.split(
        model_df,
        model_df[TARGET],
        groups=groups
    )
)


train_df = model_df.iloc[
    train_idx
]

val_df = model_df.iloc[
    val_idx
]


# ============================================================
# GROUP LEAKAGE CHECK
# ============================================================

train_groups = set(
    train_df[
        "source1_entity_id"
    ]
)

val_groups = set(
    val_df[
        "source1_entity_id"
    ]
)

overlap = (
    train_groups
    .intersection(val_groups)
)


print(
    f"Training rows: {len(train_df):,}",
    flush=True
)

print(
    f"Validation rows: {len(val_df):,}",
    flush=True
)

print(
    f"Training S1 groups: {len(train_groups):,}",
    flush=True
)

print(
    f"Validation S1 groups: {len(val_groups):,}",
    flush=True
)

print(
    f"S1 group overlap: {len(overlap)}",
    flush=True
)


if overlap:

    raise ValueError(
        "Data leakage detected: "
        "S1 groups overlap between train "
        "and validation."
    )


# ============================================================
# TRAIN / VALIDATION MATRICES
# ============================================================

X_train = train_df[
    feature_columns
]

y_train = train_df[
    TARGET
]

X_val = val_df[
    feature_columns
]

y_val = val_df[
    TARGET
]


# ============================================================
# MODEL
# ============================================================

print(
    "\nTraining improved "
    "HistGradientBoostingClassifier...",
    flush=True
)

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
    y_train
)


print(
    "Training complete.",
    flush=True
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print(
    "\nGenerating validation probabilities...",
    flush=True
)

val_prob = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# GLOBAL METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_val,
    val_prob
)

average_precision = average_precision_score(
    y_val,
    val_prob
)


print(
    f"\nROC-AUC: {roc_auc:.6f}",
    flush=True
)

print(
    f"Average Precision: "
    f"{average_precision:.6f}",
    flush=True
)


# ============================================================
# EXACT S1-LEVEL MACRO F0.5
# ============================================================

def calculate_s1_macro_f05(
    validation_df,
    probabilities,
    threshold
):
    """
    Calculate the challenge-style S1-level macro F0.5.

    Each S1 entity receives its own F0.5 score.
    The final score is the mean across S1 entities.
    """

    temp = validation_df[
        [
            "source1_entity_id",
            TARGET
        ]
    ].copy()

    temp["probability"] = probabilities

    temp["prediction"] = (
        temp["probability"] >= threshold
    ).astype(int)


    f_scores = []


    for _, group in temp.groupby(
        "source1_entity_id"
    ):

        y_true = group[
            TARGET
        ].to_numpy()

        y_pred = group[
            "prediction"
        ].to_numpy()


        score = fbeta_score(
            y_true,
            y_pred,
            beta=0.5,
            zero_division=0
        )


        f_scores.append(
            score
        )


    if not f_scores:

        return 0.0


    return float(
        np.mean(f_scores)
    )


# ============================================================
# THRESHOLDS
# ============================================================

thresholds = [
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
# EVALUATE THRESHOLDS
# ============================================================

print(
    "\nThreshold evaluation:",
    flush=True
)

print(
    "\n"
    "threshold | "
    "S1_macro_F0.5 | "
    "pair_precision | "
    "pair_recall | "
    "pair_F0.5",
    flush=True
)

print(
    "-" * 78,
    flush=True
)


results = []


for threshold in thresholds:

    predictions = (
        val_prob >= threshold
    ).astype(int)


    pair_precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )


    pair_recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )


    pair_f05 = fbeta_score(
        y_val,
        predictions,
        beta=0.5,
        zero_division=0
    )


    s1_macro_f05 = calculate_s1_macro_f05(
        val_df,
        val_prob,
        threshold
    )


    result = {
        "threshold": threshold,
        "s1_macro_f05": s1_macro_f05,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f05": pair_f05,
    }


    results.append(
        result
    )


    print(
        f"{threshold:9.2f} | "
        f"{s1_macro_f05:13.6f} | "
        f"{pair_precision:14.6f} | "
        f"{pair_recall:11.6f} | "
        f"{pair_f05:9.6f}",
        flush=True
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST THRESHOLD
# ============================================================

best_index = results_df[
    "s1_macro_f05"
].idxmax()


best_row = results_df.loc[
    best_index
]


# ============================================================
# BEST RESULT
# ============================================================

print(
    "\n============================================================",
    flush=True
)

print(
    "BEST IMPROVED MODEL RESULT",
    flush=True
)

print(
    "============================================================",
    flush=True
)

print(
    f"Threshold: "
    f"{best_row['threshold']:.2f}",
    flush=True
)

print(
    f"S1 Macro F0.5: "
    f"{best_row['s1_macro_f05']:.6f}",
    flush=True
)

print(
    f"Pair Precision: "
    f"{best_row['pair_precision']:.6f}",
    flush=True
)

print(
    f"Pair Recall: "
    f"{best_row['pair_recall']:.6f}",
    flush=True
)

print(
    f"Pair F0.5: "
    f"{best_row['pair_f05']:.6f}",
    flush=True
)


# ============================================================
# BASELINE COMPARISON
# ============================================================

BASELINE_S1_F05 = 0.916919


improvement = (
    best_row["s1_macro_f05"]
    - BASELINE_S1_F05
)


print(
    "\n============================================================",
    flush=True
)

print(
    "BASELINE COMPARISON",
    flush=True
)

print(
    "============================================================",
    flush=True
)

print(
    f"Baseline S1 Macro F0.5: "
    f"{BASELINE_S1_F05:.6f}",
    flush=True
)

print(
    f"Improved S1 Macro F0.5: "
    f"{best_row['s1_macro_f05']:.6f}",
    flush=True
)

print(
    f"Difference: "
    f"{improvement:+.6f}",
    flush=True
)


if improvement > 0:

    print(
        "\nRESULT: "
        "Improved features beat the baseline.",
        flush=True
    )

elif improvement < 0:

    print(
        "\nRESULT: "
        "Improved features did NOT beat the baseline.",
        flush=True
    )

else:

    print(
        "\nRESULT: "
        "No measurable change.",
        flush=True
    )


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    sep="\t",
    index=False
)


print(
    "\nThreshold results saved to:",
    flush=True
)

print(
    RESULTS_PATH,
    flush=True
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n============================================================",
    flush=True
)

print(
    "EXPERIMENT COMPLETE",
    flush=True
)

print(
    "============================================================",
    flush=True

)

print(
    f"Best threshold: "
    f"{best_row['threshold']:.2f}",
    flush=True
)

print(
    f"Best S1 Macro F0.5: "
    f"{best_row['s1_macro_f05']:.6f}",
    flush=True
)

print(
    f"Baseline S1 Macro F0.5: "
    f"{BASELINE_S1_F05:.6f}",
    flush=True
)

print(
    f"Improvement: "
    f"{improvement:+.6f}",
    flush=True
)

print(
    "\nDone.",
    flush=True
)