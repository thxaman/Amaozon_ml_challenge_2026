import os
import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

GROUND_TRUTH_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_ground_truth.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "full_positive_pairs.tsv"
)


print("Loading ground truth...", flush=True)

gt = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    f"Ground truth rows: {len(gt):,}",
    flush=True
)


records = []

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    matched_ids = row.matched_entity_ids

    if not matched_ids:
        continue

    for entity_id in matched_ids.split(","):

        entity_id = entity_id.strip()

        if not entity_id:
            continue

        source = (
            "S2"
            if entity_id.startswith("S2-")
            else "S3"
        )

        records.append(
            (
                s1_id,
                entity_id,
                source,
                1
            )
        )


positive_pairs = pd.DataFrame(
    records,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "source",
        "label"
    ]
)

print(
    f"Positive pairs: "
    f"{len(positive_pairs):,}",
    flush=True
)

print(
    "\nBy source:",
    flush=True
)

print(
    positive_pairs["source"].value_counts(),
    flush=True
)

print(
    "\nPositive pairs per S1:",
    flush=True
)

print(
    positive_pairs
    .groupby("source1_entity_id")
    .size()
    .describe(),
    flush=True
)


positive_pairs.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}",
    flush=True
)