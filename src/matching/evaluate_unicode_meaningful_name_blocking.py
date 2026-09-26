import re
import time
from collections import defaultdict

import pandas as pd


S1_FILE = "data/raw/train_source1.tsv"
S2_FILE = "data/raw/train_source2.tsv"
S3_FILE = "data/raw/train_source3.tsv"

MISSED_FILE = (
    "data/processed/final_blocking_misses.tsv"
)

CHUNK_SIZE = 250_000

FREQUENCY_LIMITS = [
    50,
    100,
    250,
    500,
    1000,
]


GENERIC_NAME_TOKENS = {
    "private",
    "limited",
    "pvt",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "companies",
    "co",
    "plc",
    "llp",
    "lp",
    "group",
    "groups",
    "holdings",
    "holding",
    "partners",
    "partner",
    "services",
    "service",
    "solutions",
    "solution",
    "enterprises",
    "enterprise",
    "international",
    "global",
}


def unicode_tokens(value):

    if pd.isna(value):
        return set()

    value = str(value).lower()

    tokens = re.findall(
        r"[^\W_]+",
        value,
        flags=re.UNICODE
    )

    return {
        token.strip()
        for token in tokens
        if token.strip()
    }


def meaningful_tokens(value):

    tokens = unicode_tokens(value)

    return {
        token
        for token in tokens
        if token not in GENERIC_NAME_TOKENS
    }


def main():

    start = time.time()

    print("Loading missed pairs...")

    missed = pd.read_csv(
        MISSED_FILE,
        sep="\t"
    )

    print(
        f"Missed pairs: {len(missed):,}"
    )

    missed_s1_ids = set(
        missed["source1_entity_id"]
    )

    missed_candidate_ids = set(
        missed["candidate_entity_id"]
    )

    missed_s2_ids = {
        x for x in missed_candidate_ids
        if str(x).startswith("S2-")
    }

    missed_s3_ids = {
        x for x in missed_candidate_ids
        if str(x).startswith("S3-")
    }

    # ========================================================
    # LOAD S1
    # ========================================================

    print()
    print("Loading S1...")

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t"
    )

    s1 = s1[
        s1["entity_id"].isin(
            missed_s1_ids
        )
    ].copy()

    # ========================================================
    # LOAD REQUIRED S2/S3
    # ========================================================

    print("Loading required S2 records...")

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t"
    )

    s2 = s2[
        s2["entity_id"].isin(
            missed_s2_ids
        )
    ].copy()

    print(
        f"S2 records: {len(s2):,}"
    )

    print("Loading required S3 records...")

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t"
    )

    s3 = s3[
        s3["entity_id"].isin(
            missed_s3_ids
        )
    ].copy()

    print(
        f"S3 records: {len(s3):,}"
    )

    # ========================================================
    # BUILD TOKEN MAPS
    # ========================================================

    s1_tokens = {}

    for _, row in s1.iterrows():

        s1_tokens[
            row["entity_id"]
        ] = meaningful_tokens(
            row["business_name"]
        )

    candidate_tokens = {}

    for _, row in pd.concat(
        [s2, s3],
        ignore_index=True
    ).iterrows():

        candidate_tokens[
            row["entity_id"]
        ] = meaningful_tokens(
            row["business_name"]
        )

    # ========================================================
    # FREQUENCY OF MEANINGFUL TOKENS
    #
    # Frequency is measured across all records involved
    # in this diagnostic sample.
    # ========================================================

    token_frequency = defaultdict(int)

    for tokens in s1_tokens.values():

        for token in tokens:
            token_frequency[token] += 1

    for tokens in candidate_tokens.values():

        for token in tokens:
            token_frequency[token] += 1

    # ========================================================
    # TEST FREQUENCY LIMITS
    # ========================================================

    print()
    print("=" * 75)
    print("UNICODE MEANINGFUL-TOKEN BLOCKING")
    print("=" * 75)

    for limit in FREQUENCY_LIMITS:

        recovered = 0

        candidate_counts = []

        recovered_examples = []

        for _, row in missed.iterrows():

            s1_id = row[
                "source1_entity_id"
            ]

            candidate_id = row[
                "candidate_entity_id"
            ]

            left = s1_tokens.get(
                s1_id,
                set()
            )

            right = candidate_tokens.get(
                candidate_id,
                set()
            )

            shared = left & right

            useful = {
                token
                for token in shared
                if token_frequency[token]
                <= limit
            }

            if useful:

                recovered += 1

                if len(
                    recovered_examples
                ) < 10:

                    recovered_examples.append({
                        "s1": s1_id,
                        "candidate": candidate_id,
                        "tokens": " ".join(
                            sorted(useful)
                        )
                    })

        recall = (
            recovered
            /
            len(missed)
        )

        print()
        print(
            f"Frequency <= {limit}"
        )

        print(
            f"  Misses recovered : "
            f"{recovered:,} / {len(missed):,}"
        )

        print(
            f"  Recovery rate    : "
            f"{recall:.4%}"
        )

        # ====================================================
        # EXAMPLES
        # ====================================================

        for example in recovered_examples:

            print(
                f"    "
                f"{example['s1']} -> "
                f"{example['candidate']} : "
                f"{example['tokens']}"
            )

    print()
    print(
        f"Completed in "
        f"{(time.time() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()