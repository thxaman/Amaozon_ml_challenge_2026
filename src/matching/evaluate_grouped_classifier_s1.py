import os
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "training_pair_features.tsv"
)

GT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_ground_truth.tsv"
)

RANDOM_STATE = 42

NEGATIVE_SAMPLE_SIZE = 200_000

THRESHOLDS = [
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
# F0.5
# ============================================================

def f05(tp, fp, fn):

    precision_denominator = tp + fp
    recall_denominator = tp + fn

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = tp / precision_denominator

    if recall_denominator == 0:
        recall = 1.0
    else:
        recall = tp / recall_denominator

    if precision == 0 and recall == 0:
        return 0.0

    beta2 = 0.25

    return (
        (1 + beta2)
        * precision
        * recall
        / (beta2 * precision + recall)
    )


# ============================================================
# LOAD FEATURES
# ============================================================

print("Loading pair features...", flush=True)

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t"
)

print(
    f"Rows: {len(df):,}",
    flush=True
)


# ============================================================
# LABEL
# ============================================================

df["label"] = df["label"].astype(int)

positives = df[df["label"] == 1]

negatives = df[df["label"] == 0]

print(
    f"Positive pairs: {len(positives):,}",
    flush=True
)

print(
    f"Negative pairs: {len(negatives):,}",
    flush=True
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

print(
    f"\nSampling {NEGATIVE_SAMPLE_SIZE:,} negatives...",
    flush=True
)

negative_sample = negatives.sample(
    n=min(
        NEGATIVE_SAMPLE_SIZE,
        len(negatives)
    ),
    random_state=RANDOM_STATE
)

model_df = pd.concat(
    [
        positives,
        negative_sample
    ],
    ignore_index=True
)

print(
    f"Model rows: {len(model_df):,}",
    flush=True
)


# ============================================================
# GROUPED TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating grouped S1 split...",
    flush=True
)

groups = model_df["source1_entity_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter.split(
        model_df,
        model_df["label"],
        groups=groups
    )
)

train_df = model_df.iloc[train_idx].copy()
val_df = model_df.iloc[val_idx].copy()


print(
    f"Training rows:   {len(train_df):,}",
    flush=True
)

print(
    f"Validation rows: {len(val_df):,}",
    flush=True
)

print(
    f"Training S1 groups:   "
    f"{train_df['source1_entity_id'].nunique():,}",
    flush=True
)

print(
    f"Validation S1 groups: "
    f"{val_df['source1_entity_id'].nunique():,}",
    flush=True
)


# ============================================================
# CHECK GROUP OVERLAP
# ============================================================

train_groups = set(
    train_df["source1_entity_id"]
)

val_groups = set(
    val_df["source1_entity_id"]
)

overlap = train_groups.intersection(
    val_groups
)

print(
    f"S1 group overlap: {len(overlap):,}",
    flush=True
)


# ============================================================
# TRAIN MODEL
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    "label"
]

X_val = val_df[
    FEATURE_COLUMNS
]

y_val = val_df[
    "label"
]


print(
    "\nTraining HistGradientBoostingClassifier...",
    flush=True
)

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=250,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
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
# PREDICT VALIDATION PROBABILITIES
# ============================================================

print(
    "\nPredicting validation probabilities...",
    flush=True
)

val_df["probability"] = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

print(
    "\nLoading ground truth...",
    flush=True
)

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

true_matches = {}

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    matched = row.matched_entity_ids

    if pd.isna(matched) or not str(matched).strip():

        true_matches[s1_id] = set()

    else:

        true_matches[s1_id] = {
            x.strip()
            for x in str(matched).split(",")
            if x.strip()
        }


# ============================================================
# VALIDATION S1s
# ============================================================

validation_s1_ids = set(
    val_df["source1_entity_id"]
)

print(
    f"Validation S1 entities: "
    f"{len(validation_s1_ids):,}",
    flush=True
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_threshold(threshold):

    macro_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    zero_prediction_s1 = 0

    exact_count_s1 = 0

    over_predicted_s1 = 0
    under_predicted_s1 = 0

    predicted_match_counts = []

    # --------------------------------------------------------
    # Evaluate one S1 at a time
    # --------------------------------------------------------

    for s1_id, group in val_df.groupby(
        "source1_entity_id",
        sort=False
    ):

        true_set = true_matches.get(
            s1_id,
            set()
        )

        predicted_set = set(
            group.loc[
                group["probability"] >= threshold,
                "candidate_entity_id"
            ]
        )

        tp = len(
            predicted_set.intersection(true_set)
        )

        fp = len(
            predicted_set - true_set
        )

        fn = len(
            true_set - predicted_set
        )

        score = f05(
            tp,
            fp,
            fn
        )

        macro_scores.append(score)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        predicted_count = len(
            predicted_set
        )

        predicted_match_counts.append(
            predicted_count
        )

        if predicted_count == 0:
            zero_prediction_s1 += 1

        if predicted_count == len(true_set):
            exact_count_s1 += 1

        if predicted_count > len(true_set):
            over_predicted_s1 += 1

        elif predicted_count < len(true_set):
            under_predicted_s1 += 1

    # --------------------------------------------------------
    # Macro F0.5
    # --------------------------------------------------------

    macro_f05 = np.mean(
        macro_scores
    )

    # --------------------------------------------------------
    # Pair-level precision / recall
    # --------------------------------------------------------

    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = (
            total_tp
            / (total_tp + total_fp)
        )

    if total_tp + total_fn == 0:
        recall = 1.0
    else:
        recall = (
            total_tp
            / (total_tp + total_fn)
        )

    if precision == 0 and recall == 0:
        pair_f05 = 0.0
    else:
        pair_f05 = (
            1.25
            * precision
            * recall
            / (0.25 * precision + recall)
        )

    return {
        "threshold": threshold,
        "macro_f05": macro_f05,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f05": pair_f05,
        "zero_prediction_s1": zero_prediction_s1,
        "exact_count_s1": exact_count_s1,
        "over_predicted_s1": over_predicted_s1,
        "under_predicted_s1": under_predicted_s1,
        "mean_predicted_matches": np.mean(
            predicted_match_counts
        ),
        "median_predicted_matches": np.median(
            predicted_match_counts
        ),
    }


# ============================================================
# RUN THRESHOLDS
# ============================================================

print(
    "\nEvaluating thresholds...",
    flush=True
)

results = []

for threshold in THRESHOLDS:

    result = evaluate_threshold(
        threshold
    )

    results.append(result)

    print(
        f"\nThreshold {threshold:.2f}",
        flush=True
    )

    print(
        f"  S1 Macro F0.5: "
        f"{result['macro_f05']:.6f}",
        flush=True
    )

    print(
        f"  Pair Precision: "
        f"{result['pair_precision']:.6f}",
        flush=True
    )

    print(
        f"  Pair Recall: "
        f"{result['pair_recall']:.6f}",
        flush=True
    )

    print(
        f"  Pair F0.5: "
        f"{result['pair_f05']:.6f}",
        flush=True
    )

    print(
        f"  Zero predictions: "
        f"{result['zero_prediction_s1']:,}",
        flush=True
    )

    print(
        f"  Exact prediction count: "
        f"{result['exact_count_s1']:,}",
        flush=True
    )

    print(
        f"  Over-predicted: "
        f"{result['over_predicted_s1']:,}",
        flush=True
    )

    print(
        f"  Under-predicted: "
        f"{result['under_predicted_s1']:,}",
        flush=True
    )

    print(
        f"  Mean predicted matches: "
        f"{result['mean_predicted_matches']:.3f}",
        flush=True
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

results_df = pd.DataFrame(
    results
)

best_idx = results_df[
    "macro_f05"
].idxmax()

best = results_df.loc[
    best_idx
]


# ============================================================
# FINAL TABLE
# ============================================================

print("\n" + "=" * 90)
print("S1-LEVEL CLASSIFIER EVALUATION")
print("=" * 90)

print(
    results_df[
        [
            "threshold",
            "macro_f05",
            "pair_precision",
            "pair_recall",
            "pair_f05",
            "zero_prediction_s1",
            "exact_count_s1",
            "over_predicted_s1",
            "under_predicted_s1",
            "mean_predicted_matches",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "-" * 90)

print(
    f"BEST THRESHOLD: {best['threshold']:.2f}"
)

print(
    f"BEST S1 MACRO F0.5: "
    f"{best['macro_f05']:.6f}"
)

print("-" * 90)