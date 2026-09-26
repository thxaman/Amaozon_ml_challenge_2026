import os
import re
import random
from collections import defaultdict

import pandas as pd


SEED = 42

TARGET_S2 = 7_500_000
TARGET_S3 = 7_500_000

CHUNK_SIZE = 250_000

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

S1_PATH = os.path.join(
    BASE_DIR, "data", "raw", "train_source1.tsv"
)

S2_PATH = os.path.join(
    BASE_DIR, "data", "raw", "train_source2.tsv"
)

S3_PATH = os.path.join(
    BASE_DIR, "data", "raw", "train_source3.tsv"
)

POSITIVE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "full_positive_pairs.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "full_negative_pairs.tsv"
)

random.seed(SEED)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = value.replace("&", " and ")

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def get_tokens(value):

    value = normalize(value)

    if not value:
        return []

    return value.split()


# ============================================================
# LOAD S1
# ============================================================

print("Loading S1...", flush=True)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
)

print(
    f"S1 rows: {len(s1):,}",
    flush=True
)


# ============================================================
# BUILD LIGHTWEIGHT S1 INDEX
# ============================================================

print(
    "\nBuilding lightweight S1 indexes...",
    flush=True
)

# We only need a small number of S1 IDs per token.
#
# This prevents extremely common tokens such as:
# "road", "india", "company", etc.
# from exploding candidate generation.

MAX_POSTINGS = 100


name_index = defaultdict(list)
address_index = defaultdict(list)

for row in s1.itertuples(index=False):

    s1_id = row.entity_id
    country = row.country

    name_tokens = set(
        get_tokens(row.business_name)
    )

    address_tokens = set(
        get_tokens(row.business_address)
    )

    for token in name_tokens:

        key = (
            country,
            token
        )

        if len(name_index[key]) < MAX_POSTINGS:

            name_index[key].append(
                s1_id
            )

    for token in address_tokens:

        key = (
            country,
            token
        )

        if len(address_index[key]) < MAX_POSTINGS:

            address_index[key].append(
                s1_id
            )


print(
    f"Name index keys: {len(name_index):,}",
    flush=True
)

print(
    f"Address index keys: {len(address_index):,}",
    flush=True
)


# ============================================================
# LOAD POSITIVE PAIRS
# ============================================================

print(
    "\nLoading positive pairs...",
    flush=True
)

positive_df = pd.read_csv(
    POSITIVE_PATH,
    sep="\t",
    dtype=str
)

positive_pairs = set(
    zip(
        positive_df[
            "source1_entity_id"
        ],
        positive_df[
            "candidate_entity_id"
        ]
    )
)

print(
    f"Positive pairs: "
    f"{len(positive_pairs):,}",
    flush=True
)


# ============================================================
# NEGATIVE SAMPLING
# ============================================================

def sample_source(
    source_path,
    source_name,
    target
):

    print(
        f"\nProcessing {source_name}...",
        flush=True
    )

    negatives = set()

    rows_seen = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        for row in chunk.itertuples(
            index=False
        ):

            rows_seen += 1

            candidate_id = row.entity_id
            country = row.country

            name_tokens = set(
                get_tokens(
                    row.business_name
                )
            )

            address_tokens = set(
                get_tokens(
                    row.business_address
                )
            )

            # ------------------------------------------------
            # Get a SMALL candidate pool
            # ------------------------------------------------

            candidate_ids = set()

            # Prefer name overlap.
            for token in name_tokens:

                ids = name_index.get(
                    (
                        country,
                        token
                    )
                )

                if ids:
                    candidate_ids.update(ids)


            # Add address overlap only if
            # the name pool is small.
            if len(candidate_ids) < 10:

                for token in address_tokens:

                    ids = address_index.get(
                        (
                            country,
                            token
                        )
                    )

                    if ids:
                        candidate_ids.update(ids)


            if not candidate_ids:
                continue


            # ------------------------------------------------
            # Randomly sample from candidate pool
            # ------------------------------------------------

            candidate_ids = list(
                candidate_ids
            )

            random.shuffle(
                candidate_ids
            )

            # At most 2 negatives per source record.
            #
            # This prevents one noisy source record
            # from producing thousands of pairs.

            for s1_id in candidate_ids[:2]:

                pair = (
                    s1_id,
                    candidate_id
                )

                if pair in positive_pairs:
                    continue

                negatives.add(pair)

                if len(negatives) >= target:
                    break


            if len(negatives) >= target:

                break


        if chunk_number % 5 == 0:

            print(
                f"{source_name}: "
                f"{rows_seen:,} rows processed | "
                f"{len(negatives):,} negatives",
                flush=True
            )


    print(
        f"{source_name} negatives: "
        f"{len(negatives):,}",
        flush=True
    )

    return negatives


# ============================================================
# S2
# ============================================================

negative_s2 = sample_source(
    S2_PATH,
    "S2",
    TARGET_S2
)


# ============================================================
# S3
# ============================================================

negative_s3 = sample_source(
    S3_PATH,
    "S3",
    TARGET_S3
)


# ============================================================
# SAVE
# ============================================================

records = []

for s1_id, candidate_id in negative_s2:

    records.append(
        (
            s1_id,
            candidate_id,
            "S2",
            0
        )
    )

for s1_id, candidate_id in negative_s3:

    records.append(
        (
            s1_id,
            candidate_id,
            "S3",
            0
        )
    )


negative_df = pd.DataFrame(
    records,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "source",
        "label"
    ]
)

negative_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)


print(
    "\n============================================================",
    flush=True
)

print(
    "NEGATIVE SAMPLING COMPLETE",
    flush=True
)

print(
    "============================================================",
    flush=True
)

print(
    f"S2 negatives: {len(negative_s2):,}",
    flush=True
)

print(
    f"S3 negatives: {len(negative_s3):,}",
    flush=True
)

print(
    f"Total negatives: {len(negative_df):,}",
    flush=True
)

print(
    f"Saved to:\n{OUTPUT_PATH}",
    flush=True
)