import re
import time
from collections import defaultdict

import pandas as pd


# ============================================================
# FILES
# ============================================================

S1_FILE = "data/raw/train_source1.tsv"
S2_FILE = "data/raw/train_source2.tsv"
S3_FILE = "data/raw/train_source3.tsv"

GROUND_TRUTH_FILE = "data/raw/train_ground_truth.tsv"

MISSED_FILE = (
    "data/processed/final_blocking_misses.tsv"
)


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 250_000

# Ignore extremely common tokens.
MAX_TOKEN_FREQ = 500


# ============================================================
# UNICODE TOKENIZATION
# ============================================================

def unicode_tokens(value):

    if pd.isna(value):
        return set()

    value = str(value).lower()

    # Keep Unicode letters and numbers.
    tokens = re.findall(
        r"[^\W_]+",
        value,
        flags=re.UNICODE
    )

    return set(
        token.strip()
        for token in tokens
        if token.strip()
    )


# ============================================================
# MAIN
# ============================================================

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

    # ========================================================
    # 1. BUILD REQUIRED ENTITY SETS
    # ========================================================

    missed_s1 = set(
        missed[
            "source1_entity_id"
        ]
    )

    missed_candidates = set(
        missed[
            "candidate_entity_id"
        ]
    )

    missed_s2 = {
        x for x in missed_candidates
        if str(x).startswith("S2-")
    }

    missed_s3 = {
        x for x in missed_candidates
        if str(x).startswith("S3-")
    }

    # ========================================================
    # 2. LOAD S1
    # ========================================================

    print()
    print("Loading S1...")

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t"
    )

    s1 = s1[
        s1["entity_id"].isin(
            missed_s1
        )
    ].copy()

    print(
        f"S1 records: {len(s1):,}"
    )

    # ========================================================
    # 3. CREATE S1 TOKEN MAP
    # ========================================================

    s1_token_map = {}

    token_frequency = defaultdict(int)

    for _, row in s1.iterrows():

        entity_id = row[
            "entity_id"
        ]

        tokens = unicode_tokens(
            row["business_name"]
        )

        s1_token_map[
            entity_id
        ] = tokens

        for token in tokens:
            token_frequency[token] += 1

    # ========================================================
    # 4. LOAD REQUIRED S2
    # ========================================================

    print()
    print("Scanning S2...")

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t"
    )

    s2 = s2[
        s2["entity_id"].isin(
            missed_s2
        )
    ].copy()

    print(
        f"S2 records: {len(s2):,}"
    )

    # ========================================================
    # 5. LOAD REQUIRED S3
    # ========================================================

    print()
    print("Scanning S3...")

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t"
    )

    s3 = s3[
        s3["entity_id"].isin(
            missed_s3
        )
    ].copy()

    print(
        f"S3 records: {len(s3):,}"
    )

    # ========================================================
    # 6. BUILD CANDIDATE TOKEN INDEX
    # ========================================================

    source_records = pd.concat(
        [
            s2,
            s3
        ],
        ignore_index=True
    )

    candidate_token_map = {}

    for _, row in source_records.iterrows():

        entity_id = row[
            "entity_id"
        ]

        tokens = unicode_tokens(
            row["business_name"]
        )

        candidate_token_map[
            entity_id
        ] = tokens

        for token in tokens:
            token_frequency[token] += 0

    # ========================================================
    # 7. FIND SHARED UNICODE TOKENS
    # ========================================================

    print()
    print(
        "Analyzing Unicode token overlap..."
    )

    recovered = 0
    recovered_pairs = []

    no_shared_token = 0
    shared_rare_token = 0

    for _, row in missed.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        candidate_id = row[
            "candidate_entity_id"
        ]

        s1_tokens = s1_token_map.get(
            s1_id,
            set()
        )

        candidate_tokens = candidate_token_map.get(
            candidate_id,
            set()
        )

        shared = (
            s1_tokens
            &
            candidate_tokens
        )

        # Remove very common tokens based
        # on S1-side frequency.
        useful_shared = {
            token
            for token in shared
            if token_frequency[token]
            <= MAX_TOKEN_FREQ
        }

        if shared:
            recovered += 1

        if useful_shared:
            shared_rare_token += 1

            recovered_pairs.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "shared_tokens": " ".join(
                    sorted(useful_shared)
                ),
            })
        else:
            no_shared_token += 1

    # ========================================================
    # 8. RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("UNICODE NAME BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"Missed true pairs       : "
        f"{len(missed):,}"
    )

    print(
        f"Any shared Unicode token: "
        f"{recovered:,}"
    )

    print(
        f"Useful shared token     : "
        f"{shared_rare_token:,}"
    )

    print(
        f"No useful shared token  : "
        f"{no_shared_token:,}"
    )

    print()

    if missed.empty is False:

        recall = (
            shared_rare_token
            /
            len(missed)
        )

        print(
            f"Recovery of misses     : "
            f"{recall:.4%}"
        )

    # ========================================================
    # 9. SHOW EXAMPLES
    # ========================================================

    if recovered_pairs:

        recovered_df = pd.DataFrame(
            recovered_pairs
        )

        print()
        print(
            "Examples of recoverable "
            "missed pairs:"
        )

        print(
            recovered_df.head(30).to_string(
                index=False
            )
        )

        output = (
            "data/processed/"
            "unicode_name_recoverable.tsv"
        )

        recovered_df.to_csv(
            output,
            sep="\t",
            index=False
        )

        print()
        print(
            f"Saved: {output}"
        )

    print()
    print(
        f"Completed in "
        f"{(time.time() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()