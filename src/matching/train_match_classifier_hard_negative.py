import time
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


FEATURE_FILE = "data/processed/training_pair_features.tsv"


FEATURES = [
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


# =============================================================
# S1-LEVEL MACRO F0.5
# =============================================================

def calculate_macro_f05(valid_results, threshold):

    scores = []

    for s1_id, group in valid_results.groupby(
        "source1_entity_id"
    ):

        true_ids = set(
            group.loc[
                group["label"] == 1,
                "candidate_entity_id"
            ]
        )

        predicted_ids = set(
            group.loc[
                group["probability"] >= threshold,
                "candidate_entity_id"
            ]
        )

        tp = len(
            true_ids.intersection(predicted_ids)
        )

        fp = len(
            predicted_ids.difference(true_ids)
        )

        fn = len(
            true_ids.difference(predicted_ids)
        )

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        # Correctly predicting no matches.
        if tp == 0 and fp == 0 and fn == 0:
            f05 = 1.0

        elif precision == 0 and recall == 0:
            f05 = 0.0

        else:
            f05 = (
                1.25 * precision * recall
                / (0.25 * precision + recall)
            )

        scores.append(f05)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# =============================================================
# MAIN
# =============================================================

def main():

    start = time.time()

    # =========================================================
    # 1. LOAD FEATURES
    # =========================================================

    print("Loading feature dataset...")

    df = pd.read_csv(
        FEATURE_FILE,
        sep="\t"
    )

    print(
        f"Loaded rows: {len(df):,}"
    )

    # =========================================================
    # 2. SEPARATE POSITIVES / NEGATIVES
    # =========================================================

    print()
    print("Separating positives and negatives...")

    positives = df[
        df["label"] == 1
    ].copy()

    negatives = df[
        df["label"] == 0
    ].copy()

    print(
        f"Positive rows: {len(positives):,}"
    )

    print(
        f"Negative rows: {len(negatives):,}"
    )

    # =========================================================
    # 3. IDENTIFY HARD NEGATIVES
    # =========================================================
    #
    # A negative pair is considered difficult when either
    # the business name OR address looks very similar.
    #
    # These are exactly the examples where false positives
    # are likely to occur.
    #

    print()
    print("Identifying hard negatives...")

    hard_negative_mask = (
        (negatives["name_token_ratio"] >= 90)
        |
        (negatives["address_token_ratio"] >= 90)
        |
        (
            (negatives["name_token_ratio"] >= 80)
            &
            (negatives["address_token_ratio"] >= 80)
        )
    )

    hard_negatives = negatives[
        hard_negative_mask
    ]

    easy_negatives = negatives[
        ~hard_negative_mask
    ]

    print(
        f"Hard negatives: "
        f"{len(hard_negatives):,}"
    )

    print(
        f"Other negatives: "
        f"{len(easy_negatives):,}"
    )

    # =========================================================
    # 4. SAMPLE NEGATIVES
    # =========================================================
    #
    # Total negative training examples = 200,000
    #
    # 100,000 hard negatives
    # 100,000 other/random negatives
    #
    # This gives the model much more exposure to difficult
    # false-positive cases than pure random sampling.
    #

    HARD_NEGATIVE_SAMPLE_SIZE = 100_000
    RANDOM_NEGATIVE_SAMPLE_SIZE = 100_000

    print()
    print(
        f"Sampling "
        f"{HARD_NEGATIVE_SAMPLE_SIZE:,} hard negatives..."
    )

    hard_sample = hard_negatives.sample(
        n=min(
            HARD_NEGATIVE_SAMPLE_SIZE,
            len(hard_negatives)
        ),
        random_state=42
    )

    print(
        f"Sampling "
        f"{RANDOM_NEGATIVE_SAMPLE_SIZE:,} other negatives..."
    )

    random_sample = easy_negatives.sample(
        n=RANDOM_NEGATIVE_SAMPLE_SIZE,
        random_state=42
    )

    negatives_sampled = pd.concat(
        [
            hard_sample,
            random_sample
        ],
        ignore_index=True
    )

    # =========================================================
    # 5. BUILD MODEL DATASET
    # =========================================================

    model_df = pd.concat(
        [
            positives,
            negatives_sampled
        ],
        ignore_index=True
    )

    # Shuffle.
    model_df = model_df.sample(
        frac=1.0,
        random_state=42
    ).reset_index(drop=True)

    print()
    print("=" * 70)
    print("MODEL DATASET")
    print("=" * 70)

    print(
        f"Total rows : {len(model_df):,}"
    )

    print(
        f"Positives  : "
        f"{(model_df['label'] == 1).sum():,}"
    )

    print(
        f"Negatives  : "
        f"{(model_df['label'] == 0).sum():,}"
    )

    # =========================================================
    # 6. GROUPED TRAIN / VALIDATION SPLIT
    # =========================================================

    print()
    print(
        "Creating GROUPED train/validation split..."
    )

    groups = model_df[
        "source1_entity_id"
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, valid_idx = next(
        splitter.split(
            model_df,
            model_df["label"],
            groups=groups
        )
    )

    train_df = model_df.iloc[
        train_idx
    ]

    valid_df = model_df.iloc[
        valid_idx
    ]

    print(
        f"Training rows  : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(valid_df):,}"
    )

    print(
        f"Training S1 groups  : "
        f"{train_df['source1_entity_id'].nunique():,}"
    )

    print(
        f"Validation S1 groups: "
        f"{valid_df['source1_entity_id'].nunique():,}"
    )

    # =========================================================
    # 7. VERIFY NO S1 LEAKAGE
    # =========================================================

    train_groups = set(
        train_df["source1_entity_id"]
    )

    valid_groups = set(
        valid_df["source1_entity_id"]
    )

    overlap = train_groups.intersection(
        valid_groups
    )

    print(
        f"S1 group overlap    : "
        f"{len(overlap):,}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "ERROR: source1_entity_id leakage detected!"
        )

    # =========================================================
    # 8. TRAIN MODEL
    # =========================================================

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "label"
    ]

    X_valid = valid_df[
        FEATURES
    ]

    y_valid = valid_df[
        "label"
    ]

    print()
    print(
        "Training HistGradientBoostingClassifier..."
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    # =========================================================
    # 9. VALIDATION PROBABILITIES
    # =========================================================

    print()
    print(
        "Generating validation probabilities..."
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    # =========================================================
    # 10. BASIC METRICS
    # =========================================================

    roc_auc = roc_auc_score(
        y_valid,
        probabilities
    )

    average_precision = average_precision_score(
        y_valid,
        probabilities
    )

    print()
    print("=" * 70)
    print("GROUPED VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC          : "
        f"{roc_auc:.6f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.6f}"
    )

    # =========================================================
    # 11. VALIDATION RESULT TABLE
    # =========================================================

    valid_results = valid_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "label",
        ]
    ].copy()

    valid_results[
        "probability"
    ] = probabilities

    # =========================================================
    # 12. PAIR-LEVEL THRESHOLD SEARCH
    # =========================================================

    print()
    print("=" * 70)
    print("PAIR-LEVEL THRESHOLD COMPARISON")
    print("=" * 70)

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
    ]

    best_pair_f05 = -1
    best_pair_threshold = None

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_valid,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_valid,
            predictions,
            zero_division=0
        )

        f05 = fbeta_score(
            y_valid,
            predictions,
            beta=0.5,
            zero_division=0
        )

        print(
            f"  threshold={threshold:.2f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"F0.5={f05:.4f}"
        )

        if f05 > best_pair_f05:
            best_pair_f05 = f05
            best_pair_threshold = threshold

    # =========================================================
    # 13. S1-LEVEL MACRO F0.5
    # =========================================================

    print()
    print("=" * 70)
    print("S1-LEVEL MACRO F0.5")
    print("=" * 70)

    macro_thresholds = [
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

    best_macro_f05 = -1
    best_macro_threshold = None

    for threshold in macro_thresholds:

        macro_f05 = calculate_macro_f05(
            valid_results,
            threshold
        )

        print(
            f"  threshold={threshold:.2f} "
            f"Macro-F0.5={macro_f05:.6f}"
        )

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_macro_threshold = threshold

    # =========================================================
    # 14. FINAL SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("FINAL HARD-NEGATIVE MODEL SUMMARY")
    print("=" * 70)

    print()
    print("Pair-level:")
    print(
        f"  Best F0.5      : "
        f"{best_pair_f05:.6f}"
    )

    print(
        f"  Best threshold : "
        f"{best_pair_threshold:.2f}"
    )

    print()
    print("S1-level:")
    print(
        f"  Best Macro F0.5: "
        f"{best_macro_f05:.6f}"
    )

    print(
        f"  Best threshold : "
        f"{best_macro_threshold:.2f}"
    )

    print()
    print(
        f"Completed in "
        f"{(time.time() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()