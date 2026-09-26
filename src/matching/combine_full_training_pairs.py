import os
import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

POSITIVE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "full_positive_pairs.tsv"
)

NEGATIVE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "full_negative_pairs.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "full_training_pairs.tsv"
)


print("Loading positive pairs...", flush=True)

positive = pd.read_csv(
    POSITIVE_PATH,
    sep="\t",
    dtype=str
)

print(
    f"Positive pairs: {len(positive):,}",
    flush=True
)


print("\nLoading negative pairs...", flush=True)

negative = pd.read_csv(
    NEGATIVE_PATH,
    sep="\t",
    dtype=str
)

print(
    f"Negative pairs: {len(negative):,}",
    flush=True
)


training = pd.concat(
    [
        positive,
        negative
    ],
    ignore_index=True
)


# Remove accidental duplicate pairs.
training = training.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)


# Shuffle once.
training = training.sample(
    frac=1.0,
    random_state=42
).reset_index(drop=True)


print(
    "\nFinal training dataset:",
    flush=True
)

print(
    f"Rows: {len(training):,}",
    flush=True
)

print(
    "\nLabel distribution:",
    flush=True
)

print(
    training["label"].value_counts(),
    flush=True
)

print(
    "\nSource distribution:",
    flush=True
)

print(
    training["source"].value_counts(),
    flush=True
)


training.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)


print(
    f"\nSaved to:\n{OUTPUT_PATH}",
    flush=True
)