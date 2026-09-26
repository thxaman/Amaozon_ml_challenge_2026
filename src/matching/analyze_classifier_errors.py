import os
import pandas as pd
import numpy as np

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

ERROR_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "classifier_errors.tsv"
)

S1_ERROR_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "classifier_s1_errors.tsv"
)

RANDOM_STATE = 42

NEGATIVE_SAMPLE_SIZE = 200_000

# Current best threshold from exact S1-level evaluation
THRESHOLD = 0.55


# ============================================================
# FEATURES USED BY THE CLASSIFIER
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
# COLUMNS TO SAVE FOR ERROR ANALYSIS
# ============================================================

ERROR_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "label",
    "probability",

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
# HELPER
# ============================================================

def print_separator(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD PAIR FEATURES
# ============================================================

print("Loading pair features...", flush=True)

df = pd.read_csv(
    FEATURE_PATH,
    sep="\t"
)

df["label"] = df["label"].astype(int)

positives = df[
    df["label"] == 1
]

negatives = df[
    df["label"] == 0
]

print(
    f"Total rows:       {len(df):,}",
    flush=True
)

print(
    f"Positive pairs:   {len(positives):,}",
    flush=True
)

print(
    f"Negative pairs:   {len(negatives):,}",
    flush=True
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

print(
    "\nSampling negatives...",
    flush=True
)

negative_sample_size = min(
    NEGATIVE_SAMPLE_SIZE,
    len(negatives)
)

negative_sample = negatives.sample(
    n=negative_sample_size,
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
    f"Sampled negatives: {len(negative_sample):,}",
    flush=True
)

print(
    f"Model dataset:     {len(model_df):,}",
    flush=True
)


# ============================================================
# GROUPED TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating grouped S1 split...",
    flush=True
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter.split(
        model_df,
        model_df["label"],
        groups=model_df["source1_entity_id"]
    )
)

train_df = model_df.iloc[train_idx].copy()
val_df = model_df.iloc[val_idx].copy()

train_groups = set(
    train_df["source1_entity_id"]
)

val_groups = set(
    val_df["source1_entity_id"]
)

group_overlap = train_groups.intersection(
    val_groups
)

print(
    f"Training rows:   {len(train_df):,}",
    flush=True
)

print(
    f"Validation rows: {len(val_df):,}",
    flush=True
)

print(
    f"Training S1s:    {len(train_groups):,}",
    flush=True
)

print(
    f"Validation S1s:  {len(val_groups):,}",
    flush=True
)

print(
    f"S1 group overlap: {len(group_overlap)}",
    flush=True
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print(
    "\nTraining classifier...",
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
    train_df[FEATURE_COLUMNS],
    train_df["label"]
)

print(
    "Training complete.",
    flush=True
)


# ============================================================
# GENERATE VALIDATION PROBABILITIES
# ============================================================

print(
    "\nGenerating validation probabilities...",
    flush=True
)

val_df["probability"] = model.predict_proba(
    val_df[FEATURE_COLUMNS]
)[:, 1]

val_df["prediction"] = (
    val_df["probability"] >= THRESHOLD
).astype(int)


# ============================================================
# BASIC CONFUSION COUNTS
# ============================================================

tp = int(
    (
        (val_df["label"] == 1)
        &
        (val_df["prediction"] == 1)
    ).sum()
)

tn = int(
    (
        (val_df["label"] == 0)
        &
        (val_df["prediction"] == 0)
    ).sum()
)

fp = int(
    (
        (val_df["label"] == 0)
        &
        (val_df["prediction"] == 1)
    ).sum()
)

fn = int(
    (
        (val_df["label"] == 1)
        &
        (val_df["prediction"] == 0)
    ).sum()
)

print_separator("PAIR-LEVEL CONFUSION MATRIX")

print(
    f"True positives:  {tp:,}"
)

print(
    f"True negatives:  {tn:,}"
)

print(
    f"False positives: {fp:,}"
)

print(
    f"False negatives: {fn:,}"
)


# ============================================================
# FALSE NEGATIVES
# ============================================================

false_negatives = val_df[
    (val_df["label"] == 1)
    &
    (val_df["prediction"] == 0)
].copy()


# ============================================================
# FALSE POSITIVES
# ============================================================

false_positives = val_df[
    (val_df["label"] == 0)
    &
    (val_df["prediction"] == 1)
].copy()


print_separator("ERROR COUNTS")

print(
    f"False negatives: {len(false_negatives):,}"
)

print(
    f"False positives: {len(false_positives):,}"
)


# ============================================================
# SAVE PAIR-LEVEL ERRORS
# ============================================================

os.makedirs(
    os.path.dirname(ERROR_OUTPUT),
    exist_ok=True
)

errors = pd.concat(
    [
        false_negatives,
        false_positives
    ],
    ignore_index=True
)

errors = errors[
    ERROR_COLUMNS
]

# Sort by probability:
# highest-confidence false positives first,
# then highest-confidence false negatives.
errors["error_type"] = np.where(
    errors["label"] == 1,
    "false_negative",
    "false_positive"
)

errors = errors.sort_values(
    [
        "error_type",
        "probability"
    ],
    ascending=[
        True,
        False
    ]
)

errors.to_csv(
    ERROR_OUTPUT,
    sep="\t",
    index=False
)

print(
    f"\nPair-level errors saved to:"
)

print(
    ERROR_OUTPUT
)


# ============================================================
# LOAD GROUND TRUTH
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

    if pd.isna(matched):

        true_matches[s1_id] = set()

    elif not str(matched).strip():

        true_matches[s1_id] = set()

    else:

        true_matches[s1_id] = {
            x.strip()
            for x in str(matched).split(",")
            if x.strip()
        }


# ============================================================
# S1-LEVEL ERROR ANALYSIS
# ============================================================

print(
    "\nAnalyzing S1-level errors...",
    flush=True
)

s1_rows = []

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
            group["probability"] >= THRESHOLD,
            "candidate_entity_id"
        ]
    )

    correct_set = (
        true_set
        .intersection(predicted_set)
    )

    missed_set = (
        true_set
        -
        predicted_set
    )

    wrong_set = (
        predicted_set
        -
        true_set
    )

    s1_rows.append(
        {
            "source1_entity_id": s1_id,

            "true_match_count": len(
                true_set
            ),

            "predicted_match_count": len(
                predicted_set
            ),

            "correct_match_count": len(
                correct_set
            ),

            "missed_match_count": len(
                missed_set
            ),

            "wrong_match_count": len(
                wrong_set
            ),

            "candidate_count": len(
                group
            ),
        }
    )


s1_errors = pd.DataFrame(
    s1_rows
)

s1_errors.to_csv(
    S1_ERROR_OUTPUT,
    sep="\t",
    index=False
)

print(
    f"S1-level errors saved to:"
)

print(
    S1_ERROR_OUTPUT
)


# ============================================================
# MOST PROBLEMATIC S1 ENTITIES
# ============================================================

print_separator(
    "MOST PROBLEMATIC S1 ENTITIES"
)

worst_s1 = s1_errors.sort_values(
    [
        "missed_match_count",
        "wrong_match_count"
    ],
    ascending=False
).head(30)

print(
    worst_s1.to_string(
        index=False
    )
)


# ============================================================
# FALSE NEGATIVE PROBABILITY DISTRIBUTION
# ============================================================

print_separator(
    "FALSE NEGATIVE PROBABILITY DISTRIBUTION"
)

if len(false_negatives) > 0:

    print(
        false_negatives[
            "probability"
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        ).to_string()
    )

else:

    print("No false negatives.")


# ============================================================
# FALSE POSITIVE PROBABILITY DISTRIBUTION
# ============================================================

print_separator(
    "FALSE POSITIVE PROBABILITY DISTRIBUTION"
)

if len(false_positives) > 0:

    print(
        false_positives[
            "probability"
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        ).to_string()
    )

else:

    print("No false positives.")


# ============================================================
# FALSE NEGATIVE FEATURE SUMMARY
# ============================================================

print_separator(
    "FALSE NEGATIVE FEATURE SUMMARY"
)

if len(false_negatives) > 0:

    fn_summary = false_negatives[
        FEATURE_COLUMNS
    ].describe().T

    print(
        fn_summary.to_string()
    )

else:

    print("No false negatives.")


# ============================================================
# FALSE POSITIVE FEATURE SUMMARY
# ============================================================

print_separator(
    "FALSE POSITIVE FEATURE SUMMARY"
)

if len(false_positives) > 0:

    fp_summary = false_positives[
        FEATURE_COLUMNS
    ].describe().T

    print(
        fp_summary.to_string()
    )

else:

    print("No false positives.")


# ============================================================
# HIGH-CONFIDENCE FALSE POSITIVES
# ============================================================

print_separator(
    "TOP HIGH-CONFIDENCE FALSE POSITIVES"
)

if len(false_positives) > 0:

    top_fp = false_positives.sort_values(
        "probability",
        ascending=False
    ).head(30)

    print(
        top_fp[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "probability",
                "name_ratio",
                "name_token_ratio",
                "name_partial_ratio",
                "address_ratio",
                "address_token_ratio",
                "address_partial_ratio",
                "shared_name_tokens",
                "shared_address_tokens",
                "shared_numbers"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print("No false positives.")


# ============================================================
# HARDEST FALSE NEGATIVES
# ============================================================

print_separator(
    "TOP HARDEST FALSE NEGATIVES"
)

if len(false_negatives) > 0:

    top_fn = false_negatives.sort_values(
        "probability",
        ascending=False
    ).head(30)

    print(
        top_fn[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "probability",
                "name_ratio",
                "name_token_ratio",
                "name_partial_ratio",
                "address_ratio",
                "address_token_ratio",
                "address_partial_ratio",
                "shared_name_tokens",
                "shared_address_tokens",
                "shared_numbers"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print("No false negatives.")


# ============================================================
# ERROR COUNTS BY SOURCE
# ============================================================

print_separator(
    "ERRORS BY SOURCE"
)

if len(errors) > 0:

    errors["candidate_source"] = (
        errors["candidate_entity_id"]
        .astype(str)
        .str[:3]
    )

    source_summary = (
        errors
        .groupby(
            [
                "candidate_source",
                "error_type"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        source_summary.to_string()
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_separator(
    "FINAL ERROR ANALYSIS SUMMARY"
)

print(
    f"Threshold:              {THRESHOLD}"
)

print(
    f"Validation S1 groups:   {len(val_groups):,}"
)

print(
    f"Validation pairs:       {len(val_df):,}"
)

print(
    f"True positives:         {tp:,}"
)

print(
    f"True negatives:         {tn:,}"
)

print(
    f"False positives:        {fp:,}"
)

print(
    f"False negatives:        {fn:,}"
)

print(
    "\nAnalysis complete.",
    flush=True
)