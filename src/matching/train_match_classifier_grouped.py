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


def calculate_macro_f05(valid_results, threshold):
    """
    Calculate macro F0.5 across S1 entities.

    Each S1 entity is evaluated independently.

    For each S1:
        true_ids      = actual matching S2/S3 IDs
        predicted_ids = IDs whose model probability >= threshold

    Then calculate F0.5 for that S1.

    Finally return the mean F0.5 across all S1 entities.
    """

    scores = []

    for s1_id, group in valid_results.groupby(
        "source1_entity_id"
    ):

        # Actual positive candidate IDs.
        true_ids = set(
            group.loc[
                group["label"] == 1,
                "candidate_entity_id"
            ]
        )

        # Model-predicted positive candidate IDs.
        predicted_ids = set(
            group.loc[
                group["probability"] >= threshold,
                "candidate_entity_id"
            ]
        )

        # True positives.
        tp = len(
            true_ids.intersection(predicted_ids)
        )

        # False positives.
        fp = len(
            predicted_ids.difference(true_ids)
        )

        # False negatives.
        fn = len(
            true_ids.difference(predicted_ids)
        )

        # -----------------------------------------------------
        # Calculate precision and recall for this S1.
        # -----------------------------------------------------

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        # -----------------------------------------------------
        # Special case:
        #
        # No true matches and no predictions.
        #
        # This means the model correctly predicted that this
        # S1 entity has no matching S2/S3 entity.
        # -----------------------------------------------------

        if tp == 0 and fp == 0 and fn == 0:
            f05 = 1.0

        elif precision == 0 and recall == 0:
            f05 = 0.0

        else:
            # F_beta formula with beta = 0.5
            #
            # F0.5 = (1 + beta^2) * P * R
            #        ---------------------
            #        beta^2 * P + R
            #
            # beta^2 = 0.25
            #
            f05 = (
                1.25 * precision * recall
                / (0.25 * precision + recall)
            )

        scores.append(f05)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def main():

    start = time.time()

    # =========================================================
    # 1. LOAD FEATURE DATASET
    # =========================================================

    print("Loading feature dataset...")

    df = pd.read_csv(
        FEATURE_FILE,
        sep="\t"
    )

    print(f"Loaded rows: {len(df):,}")

    # =========================================================
    # 2. SEPARATE POSITIVES AND NEGATIVES
    # =========================================================

    print()
    print("Separating positives and negatives...")

    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    print(
        f"Positive rows: {len(positives):,}"
    )

    print(
        f"Negative rows: {len(negatives):,}"
    )

    # =========================================================
    # 3. SAMPLE NEGATIVES
    # =========================================================

    NEGATIVE_SAMPLE_SIZE = 200_000

    print()
    print(
        f"Sampling {NEGATIVE_SAMPLE_SIZE:,} negatives..."
    )

    negatives_sampled = negatives.sample(
        n=NEGATIVE_SAMPLE_SIZE,
        random_state=42
    )

    # Keep all positives.
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
    print("Model dataset:")
    print(
        f"Total    : {len(model_df):,}"
    )

    print(
        f"Positives: "
        f"{(model_df['label'] == 1).sum():,}"
    )

    print(
        f"Negatives: "
        f"{(model_df['label'] == 0).sum():,}"
    )

    # =========================================================
    # 4. GROUPED TRAIN / VALIDATION SPLIT
    # =========================================================

    print()
    print(
        "Creating GROUPED train/validation split..."
    )

    # IMPORTANT:
    #
    # Every S1 entity must belong entirely to either
    # training or validation.
    #
    # This prevents the same business from appearing in
    # both sets.

    groups = model_df["source1_entity_id"]

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

    train_df = model_df.iloc[train_idx]
    valid_df = model_df.iloc[valid_idx]

    print(
        f"Training rows  : {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(valid_df):,}"
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
    # 5. VERIFY NO S1 LEAKAGE
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
        f"S1 group overlap    : {len(overlap):,}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "ERROR: source1_entity_id leakage detected!"
        )

    # =========================================================
    # 6. PREPARE TRAINING DATA
    # =========================================================

    X_train = train_df[FEATURES]
    y_train = train_df["label"]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df["label"]

    # =========================================================
    # 7. TRAIN MODEL
    # =========================================================

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
    # 8. GENERATE VALIDATION PROBABILITIES
    # =========================================================

    print()
    print(
        "Generating validation probabilities..."
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    # =========================================================
    # 9. BASIC PAIR-LEVEL METRICS
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
        f"ROC-AUC          : {roc_auc:.6f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.6f}"
    )

    # =========================================================
    # 10. CREATE VALIDATION RESULT TABLE
    # =========================================================
    #
    # We need this because the actual challenge evaluates
    # matches grouped by S1 entity.
    #

    valid_results = valid_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "label",
        ]
    ].copy()

    valid_results["probability"] = probabilities

    # =========================================================
    # 11. PAIR-LEVEL THRESHOLD SEARCH
    # =========================================================

    print()
    print("=" * 70)
    print("PAIR-LEVEL THRESHOLD COMPARISON")
    print("=" * 70)

    best_pair_threshold = None
    best_pair_f05 = -1

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
    ]

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
    # 12. S1-LEVEL MACRO F0.5
    # =========================================================
    #
    # THIS IS THE IMPORTANT PART.
    #
    # We evaluate each S1 independently and then average
    # the F0.5 scores.
    #

    print()
    print("=" * 70)
    print("S1-LEVEL MACRO F0.5")
    print("=" * 70)

    macro_thresholds = [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    best_macro_threshold = None
    best_macro_f05 = -1

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
    # 13. FINAL RESULTS
    # =========================================================

    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
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