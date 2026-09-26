import time
import re
import unicodedata
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
CHUNK_SIZE = 250_000
RANDOM_SEED = 42

ADDRESS_TOKEN_FREQUENCY_LIMIT = 500

S1_PATH = "data/raw/train_source1.tsv"
S2_PATH = "data/raw/train_source2.tsv"
S3_PATH = "data/raw/train_source3.tsv"
GT_PATH = "data/raw/train_ground_truth.tsv"

OUTPUT_PATH = (
    "data/processed/baseline_candidate_pairs.tsv"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    value = value.lower()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# ADDRESS TOKENIZATION
# ============================================================

GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "lane",
    "ln",
    "drive",
    "dr",
    "parkway",
    "pkwy",
    "boulevard",
    "blvd",
    "highway",
    "hwy",
    "route",
    "rt",
    "building",
    "bldg",
    "floor",
    "fl",
    "unit",
    "suite",
    "ste",
    "block",
    "sector",
    "district",
    "city",
    "state",
    "country",
    "india",
    "usa",
    "us",
}


def address_tokens(value):

    text = normalize_text(value)

    if not text:
        return set()

    tokens = set()

    for token in text.split():

        # Ignore tiny alphabetic tokens.
        if len(token) < 3 and token.isalpha():
            continue

        if token in GENERIC_ADDRESS_TOKENS:
            continue

        tokens.add(token)

    return tokens


# ============================================================
# GROUND TRUTH
# ============================================================

def parse_ground_truth(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


# ============================================================
# LOAD DEVELOPMENT SAMPLE
# ============================================================

def load_sample():

    print()
    print("1. Loading development sample...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    sample = gt.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(
        f"   Sampled S1 entities: "
        f"{len(sample):,}"
    )

    return sample


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

def load_s1_records(sample):

    required_ids = set(
        sample["source1_entity_id"]
    )

    records = {}

    print()
    print("2. Loading sampled S1 records...")

    for chunk in pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        relevant = chunk[
            chunk["entity_id"].isin(
                required_ids
            )
        ]

        for _, row in relevant.iterrows():

            records[row["entity_id"]] = {
                "entity_id": row["entity_id"],
                "business_name": row["business_name"],
                "business_address": row[
                    "business_address"
                ],
                "country": row["country"],
            }

    print(
        f"   Loaded S1 records: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# PREPARE S1 BLOCKING KEYS
# ============================================================

def prepare_s1_keys(s1_records):

    name_keys = {}
    address_tokens = {}

    relevant_address_tokens = set()

    for entity_id, record in s1_records.items():

        country = record["country"]

        normalized_name = normalize_text(
            record["business_name"]
        )

        name_keys[entity_id] = (
            country,
            normalized_name
        )

        tokens = address_tokens_func(
            record["business_address"]
        )

        address_tokens[entity_id] = (
            country,
            tokens
        )

        relevant_address_tokens.update(
            tokens
        )

    return (
        name_keys,
        address_tokens,
        relevant_address_tokens
    )


# Small wrapper so the variable and function
# names don't collide.
def address_tokens_func(value):

    return address_tokens(value)


# ============================================================
# COUNT ADDRESS TOKEN FREQUENCY
# ============================================================

def count_address_tokens(
    path,
    relevant_tokens,
    source_name
):

    counter = Counter()

    print()
    print(
        f"3. Counting relevant address tokens "
        f"in {source_name}..."
    )

    rows_scanned = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        rows_scanned += len(chunk)

        for value in chunk[
            "business_address"
        ]:

            tokens = (
                address_tokens_func(value)
                &
                relevant_tokens
            )

            for token in tokens:
                counter[token] += 1

    print(
        f"   Rows scanned: "
        f"{rows_scanned:,}"
    )

    print(
        f"   Relevant tokens found: "
        f"{len(counter):,}"
    )

    return counter


# ============================================================
# BUILD EXACT NAME INDEX
# ============================================================

def build_name_index(
    path,
    relevant_name_keys,
    source_name
):

    index = defaultdict(set)

    print()
    print(
        f"4. Building exact-name index "
        f"for {source_name}..."
    )

    rows_scanned = 0
    indexed_records = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        rows_scanned += len(chunk)

        for _, row in chunk.iterrows():

            normalized_name = normalize_text(
                row["business_name"]
            )

            if not normalized_name:
                continue

            key = (
                row["country"],
                normalized_name
            )

            if key not in relevant_name_keys:
                continue

            index[key].add(
                row["entity_id"]
            )

            indexed_records += 1

    print(
        f"   Rows scanned: "
        f"{rows_scanned:,}"
    )

    print(
        f"   Indexed records: "
        f"{indexed_records:,}"
    )

    print(
        f"   Index keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# BUILD ADDRESS INDEX
# ============================================================

def build_address_index(
    path,
    usable_tokens,
    source_name
):

    index = defaultdict(set)

    print()
    print(
        f"5. Building rare-address-token "
        f"index for {source_name}..."
    )

    rows_scanned = 0
    indexed_records = 0
    postings = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        rows_scanned += len(chunk)

        for _, row in chunk.iterrows():

            tokens = (
                address_tokens_func(
                    row["business_address"]
                )
                &
                usable_tokens
            )

            if not tokens:
                continue

            indexed_records += 1

            country = row["country"]
            entity_id = row["entity_id"]

            for token in tokens:

                index[
                    (country, token)
                ].add(entity_id)

                postings += 1

    print(
        f"   Rows scanned: "
        f"{rows_scanned:,}"
    )

    print(
        f"   Indexed records: "
        f"{indexed_records:,}"
    )

    print(
        f"   Posting entries: "
        f"{postings:,}"
    )

    print(
        f"   Index keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    sample,
    s1_records,
    name_keys,
    address_keys,
    s2_name_index,
    s3_name_index,
    s2_address_index,
    s3_address_index
):

    candidate_rows = []

    total_true_pairs = 0
    recovered_true_pairs = 0

    total_s2_pairs = 0
    recovered_s2_pairs = 0

    total_s3_pairs = 0
    recovered_s3_pairs = 0

    full_recovered = 0
    matched_s1 = 0

    candidate_counts = []

    zero_candidates = 0
    matched_zero_candidates = 0

    name_found_count = 0
    address_found_count = 0
    both_found_count = 0

    for _, gt_row in sample.iterrows():

        s1_id = gt_row[
            "source1_entity_id"
        ]

        record = s1_records[s1_id]

        country = record["country"]

        # ----------------------------------------------------
        # Channel 1: exact normalized name
        # ----------------------------------------------------

        name_key = name_keys[s1_id]

        name_candidates = set()

        name_candidates.update(
            s2_name_index.get(
                name_key,
                set()
            )
        )

        name_candidates.update(
            s3_name_index.get(
                name_key,
                set()
            )
        )

        # ----------------------------------------------------
        # Channel 2: rare address token
        # ----------------------------------------------------

        address_candidates = set()

        _, tokens = address_keys[s1_id]

        for token in tokens:

            key = (
                country,
                token
            )

            address_candidates.update(
                s2_address_index.get(
                    key,
                    set()
                )
            )

            address_candidates.update(
                s3_address_index.get(
                    key,
                    set()
                )
            )

        # ----------------------------------------------------
        # UNION
        # ----------------------------------------------------

        candidates = (
            name_candidates
            |
            address_candidates
        )

        candidate_counts.append(
            len(candidates)
        )

        if name_candidates:
            name_found_count += 1

        if address_candidates:
            address_found_count += 1

        if (
            name_candidates
            and address_candidates
        ):
            both_found_count += 1

        if not candidates:
            zero_candidates += 1

        # ----------------------------------------------------
        # Ground truth evaluation
        # ----------------------------------------------------

        true_matches = parse_ground_truth(
            gt_row["matched_entity_ids"]
        )

        total_true_pairs += len(
            true_matches
        )

        if true_matches:
            matched_s1 += 1

        if (
            true_matches
            and not candidates
        ):
            matched_zero_candidates += 1

        for entity_id in true_matches:

            if entity_id.startswith("S2-"):

                total_s2_pairs += 1

            elif entity_id.startswith("S3-"):

                total_s3_pairs += 1

            if entity_id in candidates:

                recovered_true_pairs += 1

                if entity_id.startswith(
                    "S2-"
                ):
                    recovered_s2_pairs += 1

                elif entity_id.startswith(
                    "S3-"
                ):
                    recovered_s3_pairs += 1

        if (
            true_matches
            and true_matches.issubset(
                candidates
            )
        ):
            full_recovered += 1

        # ----------------------------------------------------
        # Save candidate row
        # ----------------------------------------------------

        # Deterministic ordering makes the output
        # reproducible.
        sorted_candidates = sorted(
            candidates
        )

        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(
                sorted_candidates
            )
        })

    candidate_series = pd.Series(
        candidate_counts
    )

    metrics = {
        "total_true_pairs": total_true_pairs,
        "recovered_true_pairs": recovered_true_pairs,

        "pair_recall": (
            recovered_true_pairs
            / total_true_pairs
            if total_true_pairs
            else 0
        ),

        "s2_recall": (
            recovered_s2_pairs
            / total_s2_pairs
            if total_s2_pairs
            else 0
        ),

        "s3_recall": (
            recovered_s3_pairs
            / total_s3_pairs
            if total_s3_pairs
            else 0
        ),

        "full_s1_recall": (
            full_recovered
            / matched_s1
            if matched_s1
            else 0
        ),

        "avg_candidates":
            candidate_series.mean(),

        "median_candidates":
            candidate_series.median(),

        "p95_candidates":
            candidate_series.quantile(0.95),

        "max_candidates":
            candidate_series.max(),

        "zero_candidates":
            zero_candidates,

        "matched_zero_candidates":
            matched_zero_candidates,

        "name_found":
            name_found_count,

        "address_found":
            address_found_count,

        "both_found":
            both_found_count,
    }

    return candidate_rows, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 60)
    print(
        "STEP 5A - BASELINE CANDIDATE GENERATION"
    )
    print("=" * 60)

    print()
    print(
        "Baseline:"
    )

    print(
        "  country + exact normalized name"
    )

    print(
        "  UNION"
    )

    print(
        "  country + rare address token"
    )

    print()
    print(
        f"Sample size: {SAMPLE_SIZE:,}"
    )

    # --------------------------------------------------------
    # 1. Sample
    # --------------------------------------------------------

    sample = load_sample()

    # --------------------------------------------------------
    # 2. S1
    # --------------------------------------------------------

    s1_records = load_s1_records(
        sample
    )

    # --------------------------------------------------------
    # 3. S1 keys
    # --------------------------------------------------------

    (
        name_keys,
        address_keys,
        relevant_address_tokens
    ) = prepare_s1_keys(
        s1_records
    )

    relevant_name_keys = set(
        name_keys.values()
    )

    print()
    print(
        f"Relevant exact-name keys: "
        f"{len(relevant_name_keys):,}"
    )

    print(
        f"Relevant address tokens: "
        f"{len(relevant_address_tokens):,}"
    )

    # --------------------------------------------------------
    # 4. Count address token frequencies
    # --------------------------------------------------------

    s2_address_counts = count_address_tokens(
        S2_PATH,
        relevant_address_tokens,
        "S2"
    )

    s3_address_counts = count_address_tokens(
        S3_PATH,
        relevant_address_tokens,
        "S3"
    )

    # A token must be rare in the corresponding source.
    usable_s2_tokens = {
        token
        for token, count in s2_address_counts.items()
        if count <= ADDRESS_TOKEN_FREQUENCY_LIMIT
    }

    usable_s3_tokens = {
        token
        for token, count in s3_address_counts.items()
        if count <= ADDRESS_TOKEN_FREQUENCY_LIMIT
    }

    print()
    print(
        f"S2 usable address tokens: "
        f"{len(usable_s2_tokens):,}"
    )

    print(
        f"S3 usable address tokens: "
        f"{len(usable_s3_tokens):,}"
    )

    # --------------------------------------------------------
    # 5. Build exact name indexes
    # --------------------------------------------------------

    s2_name_index = build_name_index(
        S2_PATH,
        relevant_name_keys,
        "S2"
    )

    s3_name_index = build_name_index(
        S3_PATH,
        relevant_name_keys,
        "S3"
    )

    # --------------------------------------------------------
    # 6. Build address indexes
    # --------------------------------------------------------

    s2_address_index = build_address_index(
        S2_PATH,
        usable_s2_tokens,
        "S2"
    )

    s3_address_index = build_address_index(
        S3_PATH,
        usable_s3_tokens,
        "S3"
    )

    # --------------------------------------------------------
    # 7. Generate candidates
    # --------------------------------------------------------

    print()
    print(
        "6. Generating baseline candidates..."
    )

    (
        candidate_rows,
        metrics
    ) = generate_candidates(
        sample,
        s1_records,
        name_keys,
        address_keys,
        s2_name_index,
        s3_name_index,
        s2_address_index,
        s3_address_index
    )

    # --------------------------------------------------------
    # 8. Save candidates
    # --------------------------------------------------------

    output = pd.DataFrame(
        candidate_rows
    )

    output.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # 9. Print metrics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "STEP 5A RESULTS"
    )
    print("=" * 60)

    print()
    print(
        f"True pairs:              "
        f"{metrics['total_true_pairs']:,}"
    )

    print(
        f"Recovered pairs:         "
        f"{metrics['recovered_true_pairs']:,}"
    )

    print(
        f"Pair recall:             "
        f"{metrics['pair_recall']:.4f}"
    )

    print(
        f"S2 recall:               "
        f"{metrics['s2_recall']:.4f}"
    )

    print(
        f"S3 recall:               "
        f"{metrics['s3_recall']:.4f}"
    )

    print(
        f"Full S1 recall:          "
        f"{metrics['full_s1_recall']:.4f}"
    )

    print()
    print(
        f"Average candidates:      "
        f"{metrics['avg_candidates']:.2f}"
    )

    print(
        f"Median candidates:       "
        f"{metrics['median_candidates']:.2f}"
    )

    print(
        f"P95 candidates:          "
        f"{metrics['p95_candidates']:.2f}"
    )

    print(
        f"Maximum candidates:      "
        f"{metrics['max_candidates']:.0f}"
    )

    print()
    print(
        f"Zero candidates:         "
        f"{metrics['zero_candidates']:,}"
    )

    print(
        f"Matched zero candidates: "
        f"{metrics['matched_zero_candidates']:,}"
    )

    print()
    print(
        f"Name channel found:      "
        f"{metrics['name_found']:,}"
    )

    print(
        f"Address channel found:   "
        f"{metrics['address_found']:,}"
    )

    print(
        f"Both channels found:     "
        f"{metrics['both_found']:,}"
    )

    print()
    print(
        f"Saved candidate file:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print()
    print(
        f"Runtime: "
        f"{(time.time() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()