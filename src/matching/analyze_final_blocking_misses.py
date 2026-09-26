import re
import time
import pandas as pd

from rapidfuzz import fuzz


# ============================================================
# FILES
# ============================================================

S1_FILE = "data/raw/train_source1.tsv"
S2_FILE = "data/raw/train_source2.tsv"
S3_FILE = "data/raw/train_source3.tsv"
GROUND_TRUTH_FILE = "data/raw/train_ground_truth.tsv"

CANDIDATE_FILE = (
    "data/processed/address_pair_candidate_pairs.tsv"
)


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 250_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def normalize_tokens(value):

    text = normalize_text(value)

    if not text:
        return set()

    return set(text.split())


def extract_numbers(value):

    if pd.isna(value):
        return set()

    return set(
        re.findall(
            r"\d+",
            str(value)
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    # ========================================================
    # 1. LOAD GROUND TRUTH
    # ========================================================

    print("Loading ground truth...")

    gt = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t"
    )

    print(
        f"Ground truth rows: {len(gt):,}"
    )

    # ========================================================
    # 2. LOAD FINAL CANDIDATES
    # ========================================================

    print()
    print("Loading final candidate pairs...")

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        sep="\t"
    )

    print(
        f"Candidate rows: {len(candidates):,}"
    )

    # ========================================================
    # 3. BUILD CANDIDATE SET
    # ========================================================

    candidate_set = set()

    for _, row in candidates.iterrows():

        s1_id = row["source1_entity_id"]

        candidate_ids = str(
            row["candidate_entity_ids"]
        )

        if (
            candidate_ids == ""
            or candidate_ids.lower() == "nan"
        ):
            continue

        for candidate_id in candidate_ids.split(","):

            candidate_id = candidate_id.strip()

            if candidate_id:
                candidate_set.add(
                    (s1_id, candidate_id)
                )

    print(
        f"Candidate pairs in set: "
        f"{len(candidate_set):,}"
    )

    # ========================================================
    # 4. FIND TRUE PAIRS FOR THE S1 ENTITIES IN CANDIDATES
    # ========================================================

    candidate_s1_ids = set(
        candidates["source1_entity_id"]
    )

    print(
        f"S1 entities analyzed: "
        f"{len(candidate_s1_ids):,}"
    )

    true_pairs = []

    for _, row in gt.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        if s1_id not in candidate_s1_ids:
            continue

        matched = row[
            "matched_entity_ids"
        ]

        if pd.isna(matched):
            continue

        matched = str(matched).strip()

        if not matched:
            continue

        for candidate_id in matched.split(","):

            candidate_id = candidate_id.strip()

            if candidate_id:
                true_pairs.append(
                    (
                        s1_id,
                        candidate_id
                    )
                )

    print(
        f"True pairs in analyzed sample: "
        f"{len(true_pairs):,}"
    )

    # ========================================================
    # 5. FIND MISSED TRUE PAIRS
    # ========================================================

    missed_pairs = [
        pair
        for pair in true_pairs
        if pair not in candidate_set
    ]

    print()
    print("=" * 70)
    print("FINAL BLOCKING MISS ANALYSIS")
    print("=" * 70)

    print(
        f"True pairs       : {len(true_pairs):,}"
    )

    print(
        f"Recovered        : "
        f"{len(true_pairs) - len(missed_pairs):,}"
    )

    print(
        f"Missed           : "
        f"{len(missed_pairs):,}"
    )

    if true_pairs:
        recall = (
            len(true_pairs) - len(missed_pairs)
        ) / len(true_pairs)

        print(
            f"Pair recall      : "
            f"{recall:.6f}"
        )

    if not missed_pairs:
        print()
        print("No missed pairs found.")
        return

    # ========================================================
    # 6. ORGANIZE MISSED IDS
    # ========================================================

    missed_s1_ids = set()
    missed_s2_ids = set()
    missed_s3_ids = set()

    for s1_id, candidate_id in missed_pairs:

        missed_s1_ids.add(s1_id)

        if str(candidate_id).startswith("S2-"):
            missed_s2_ids.add(candidate_id)

        elif str(candidate_id).startswith("S3-"):
            missed_s3_ids.add(candidate_id)

    print()
    print(
        f"Missed S1 entities: "
        f"{len(missed_s1_ids):,}"
    )

    print(
        f"Missed S2 entities: "
        f"{len(missed_s2_ids):,}"
    )

    print(
        f"Missed S3 entities: "
        f"{len(missed_s3_ids):,}"
    )

    # ========================================================
    # 7. LOAD REQUIRED S1 RECORDS
    # ========================================================

    print()
    print("Loading required S1 records...")

    s1_lookup = {}

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t"
    )

    s1 = s1[
        s1["entity_id"].isin(
            missed_s1_ids
        )
    ]

    for _, row in s1.iterrows():

        s1_lookup[
            row["entity_id"]
        ] = row

    print(
        f"S1 records loaded: "
        f"{len(s1_lookup):,}"
    )

    del s1

    # ========================================================
    # 8. LOAD ONLY REQUIRED S2 RECORDS
    # ========================================================

    print()
    print("Scanning S2 in chunks...")

    s2_lookup = {}

    for chunk in pd.read_csv(
        S2_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):

        needed = chunk[
            chunk["entity_id"].isin(
                missed_s2_ids
            )
        ]

        for _, row in needed.iterrows():

            s2_lookup[
                row["entity_id"]
            ] = row

        if len(s2_lookup) == len(missed_s2_ids):
            break

    print(
        f"S2 records loaded: "
        f"{len(s2_lookup):,}"
    )

    # ========================================================
    # 9. LOAD ONLY REQUIRED S3 RECORDS
    # ========================================================

    print()
    print("Scanning S3 in chunks...")

    s3_lookup = {}

    for chunk in pd.read_csv(
        S3_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):

        needed = chunk[
            chunk["entity_id"].isin(
                missed_s3_ids
            )
        ]

        for _, row in needed.iterrows():

            s3_lookup[
                row["entity_id"]
            ] = row

        if len(s3_lookup) == len(missed_s3_ids):
            break

    print(
        f"S3 records loaded: "
        f"{len(s3_lookup):,}"
    )

    # ========================================================
    # 10. CALCULATE FEATURES FOR MISSED PAIRS
    # ========================================================

    print()
    print("Analyzing missed pairs...")

    results = []

    for index, (s1_id, candidate_id) in enumerate(
        missed_pairs,
        start=1
    ):

        s1_row = s1_lookup.get(
            s1_id
        )

        if s1_row is None:
            continue

        if candidate_id.startswith("S2-"):
            candidate_row = s2_lookup.get(
                candidate_id
            )
            source = "S2"

        else:
            candidate_row = s3_lookup.get(
                candidate_id
            )
            source = "S3"

        if candidate_row is None:
            continue

        s1_name = normalize_text(
            s1_row["business_name"]
        )

        candidate_name = normalize_text(
            candidate_row["business_name"]
        )

        s1_address = normalize_text(
            s1_row["business_address"]
        )

        candidate_address = normalize_text(
            candidate_row["business_address"]
        )

        s1_name_tokens = set(
            s1_name.split()
        ) if s1_name else set()

        candidate_name_tokens = set(
            candidate_name.split()
        ) if candidate_name else set()

        s1_address_tokens = set(
            s1_address.split()
        ) if s1_address else set()

        candidate_address_tokens = set(
            candidate_address.split()
        ) if candidate_address else set()

        shared_name_tokens = (
            s1_name_tokens
            & candidate_name_tokens
        )

        shared_address_tokens = (
            s1_address_tokens
            & candidate_address_tokens
        )

        s1_numbers = extract_numbers(
            s1_row["business_address"]
        )

        candidate_numbers = extract_numbers(
            candidate_row["business_address"]
        )

        shared_numbers = (
            s1_numbers
            & candidate_numbers
        )

        name_ratio = (
            fuzz.ratio(
                s1_name,
                candidate_name
            )
            if s1_name and candidate_name
            else 0
        )

        name_token_ratio = (
            fuzz.token_set_ratio(
                s1_name,
                candidate_name
            )
            if s1_name and candidate_name
            else 0
        )

        name_partial_ratio = (
            fuzz.partial_ratio(
                s1_name,
                candidate_name
            )
            if s1_name and candidate_name
            else 0
        )

        address_ratio = (
            fuzz.ratio(
                s1_address,
                candidate_address
            )
            if s1_address and candidate_address
            else 0
        )

        address_token_ratio = (
            fuzz.token_set_ratio(
                s1_address,
                candidate_address
            )
            if s1_address and candidate_address
            else 0
        )

        address_partial_ratio = (
            fuzz.partial_ratio(
                s1_address,
                candidate_address
            )
            if s1_address and candidate_address
            else 0
        )

        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": candidate_id,
            "source": source,

            "s1_name": s1_row["business_name"],
            "candidate_name": candidate_row[
                "business_name"
            ],

            "s1_address": s1_row[
                "business_address"
            ],
            "candidate_address": candidate_row[
                "business_address"
            ],

            "country": s1_row["country"],

            "name_ratio": name_ratio,
            "name_token_ratio": name_token_ratio,
            "name_partial_ratio": name_partial_ratio,

            "address_ratio": address_ratio,
            "address_token_ratio": address_token_ratio,
            "address_partial_ratio": address_partial_ratio,

            "shared_name_tokens": len(
                shared_name_tokens
            ),

            "shared_address_tokens": len(
                shared_address_tokens
            ),

            "shared_numbers": len(
                shared_numbers
            ),

            "name_exact": int(
                s1_name == candidate_name
                and s1_name != ""
            ),

            "address_exact": int(
                s1_address == candidate_address
                and s1_address != ""
            ),

            "name_tokens": " ".join(
                sorted(shared_name_tokens)
            ),

            "address_tokens": " ".join(
                sorted(shared_address_tokens)
            ),

            "numbers": " ".join(
                sorted(shared_numbers)
            ),
        })

        if index % 500 == 0:
            print(
                f"  Processed "
                f"{index:,}/"
                f"{len(missed_pairs):,}",
                flush=True
            )

    # ========================================================
    # 11. CREATE RESULT DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    print()
    print(
        f"Analyzed missed pairs: "
        f"{len(result_df):,}"
    )

    if result_df.empty:
        print("No records could be analyzed.")
        return

    # ========================================================
    # 12. SUMMARY STATISTICS
    # ========================================================

    print()
    print("=" * 70)
    print("MISSED PAIR STATISTICS")
    print("=" * 70)

    numeric_columns = [
        "name_ratio",
        "name_token_ratio",
        "name_partial_ratio",
        "address_ratio",
        "address_token_ratio",
        "address_partial_ratio",
        "shared_name_tokens",
        "shared_address_tokens",
        "shared_numbers",
    ]

    print(
        result_df[
            numeric_columns
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        ).round(2)
    )

    # ========================================================
    # 13. SOURCE BREAKDOWN
    # ========================================================

    print()
    print("Misses by source:")

    print(
        result_df[
            "source"
        ].value_counts()
    )

    # ========================================================
    # 14. HIGH-SIMILARITY MISSES
    # ========================================================

    print()
    print("High-similarity missed pairs:")

    high_similarity = result_df[
        (
            result_df["name_token_ratio"] >= 90
        )
        |
        (
            result_df["address_token_ratio"] >= 90
        )
    ]

    print(
        f"Name token >=90 OR "
        f"address token >=90: "
        f"{len(high_similarity):,}"
    )

    both_high = result_df[
        (
            result_df["name_token_ratio"] >= 80
        )
        &
        (
            result_df["address_token_ratio"] >= 80
        )
    ]

    print(
        f"Both name/address token >=80: "
        f"{len(both_high):,}"
    )

    # ========================================================
    # 15. ADDRESS STRUCTURE
    # ========================================================

    print()
    print("Address-related misses:")

    print(
        f"Shared address tokens >=3: "
        f"{(
            result_df['shared_address_tokens'] >= 3
        ).sum():,}"
    )

    print(
        f"Shared address tokens >=5: "
        f"{(
            result_df['shared_address_tokens'] >= 5
        ).sum():,}"
    )

    print(
        f"Shared numbers >=1: "
        f"{(
            result_df['shared_numbers'] >= 1
        ).sum():,}"
    )

    print(
        f"Shared numbers >=2: "
        f"{(
            result_df['shared_numbers'] >= 2
        ).sum():,}"
    )

    # ========================================================
    # 16. SAVE RESULTS
    # ========================================================

    output_file = (
        "data/processed/"
        "final_blocking_misses.tsv"
    )

    result_df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print()
    print(
        f"Saved detailed analysis to:"
    )

    print(
        output_file
    )

    # ========================================================
    # 17. SHOW EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("EXAMPLE MISSED PAIRS")
    print("=" * 70)

    display_columns = [
        "source",
        "candidate_entity_id",
        "name_ratio",
        "name_token_ratio",
        "address_ratio",
        "address_token_ratio",
        "shared_name_tokens",
        "shared_address_tokens",
        "shared_numbers",
        "s1_name",
        "candidate_name",
        "s1_address",
        "candidate_address",
    ]

    print(
        result_df[
            display_columns
        ].head(20).to_string(
            index=False
        )
    )

    print()
    print(
        f"Completed in "
        f"{(time.time() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()