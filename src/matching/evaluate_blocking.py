import time
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
CHUNK_SIZE = 250_000

# We will compare several frequency limits.
FREQUENCY_LIMITS = [
    500,
    1000,
    2500,
    5000,
]

RANDOM_SEED = 42

S1_PATH = "data/raw/train_source1.tsv"
S2_PATH = "data/raw/train_source2.tsv"
S3_PATH = "data/raw/train_source3.tsv"
GT_PATH = "data/raw/train_ground_truth.tsv"


# ============================================================
# NORMALIZATION
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
# NAME TOKENIZATION
# ============================================================

GENERIC_NAME_TOKENS = {
    "the",
    "and",
    "of",
    "for",
    "in",
    "at",
    "on",
    "to",
    "a",
    "an",

    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
    "plc",
    "pvt",
    "private",

    "group",
    "holdings",
    "international",

    "services",
    "service",

    "business",
    "businesses",

    "enterprise",
    "enterprises",

    "india",
    "indian",
}


def name_tokens(value):

    text = normalize_text(value)

    if not text:
        return set()

    tokens = set()

    for token in text.split():

        if len(token) < 3:
            continue

        if token in GENERIC_NAME_TOKENS:
            continue

        tokens.add(token)

    return tokens


# ============================================================
# GROUND TRUTH
# ============================================================

def parse_ground_truth(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


# ============================================================
# LOAD SAMPLE
# ============================================================

def load_sample():

    print()
    print("1. Loading and sampling ground truth...")

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

    return sample


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

def load_s1_sample(sample):

    required_ids = set(
        sample["source1_entity_id"]
    )

    records = {}

    print()
    print("2. Loading sampled S1 records...")

    rows_scanned = 0

    for chunk in pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        rows_scanned += len(chunk)

        relevant = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in relevant.iterrows():

            records[row["entity_id"]] = {
                "entity_id": row["entity_id"],
                "business_name": row["business_name"],
                "country": row["country"],
            }

    print(
        f"   Rows scanned: {rows_scanned:,}"
    )

    print(
        f"   S1 records loaded: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# PREPARE S1 TOKENS
# ============================================================

def prepare_s1_tokens(s1_records):

    result = {}
    relevant_tokens = set()

    for entity_id, record in s1_records.items():

        tokens = name_tokens(
            record["business_name"]
        )

        result[entity_id] = tokens
        relevant_tokens.update(tokens)

    return result, relevant_tokens


# ============================================================
# COUNT TOKEN FREQUENCIES
# ============================================================

def count_tokens(
    path,
    relevant_tokens,
    source_name
):

    counter = Counter()

    print()
    print(
        f"3. Counting relevant name tokens in "
        f"{source_name}..."
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

        for value in chunk["business_name"]:

            tokens = (
                name_tokens(value)
                &
                relevant_tokens
            )

            # Each record contributes once
            # to each token's document frequency.
            for token in tokens:
                counter[token] += 1

    print(
        f"   Rows scanned: {rows_scanned:,}"
    )

    print(
        f"   Relevant unique tokens found: "
        f"{len(counter):,}"
    )

    return counter


# ============================================================
# BUILD INDEX FOR ONE FREQUENCY LIMIT
# ============================================================

def build_index(
    path,
    usable_tokens,
    source_name,
    frequency_limit
):

    index = defaultdict(set)

    rows_scanned = 0
    indexed_records = 0
    posting_entries = 0

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
                name_tokens(
                    row["business_name"]
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
                    (
                        country,
                        token
                    )
                ].add(entity_id)

                posting_entries += 1

    print(
        f"      {source_name}: "
        f"records={indexed_records:,}, "
        f"postings={posting_entries:,}, "
        f"keys={len(index):,}"
    )

    return index


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    sample,
    s1_records,
    s1_tokens,
    s2_index,
    s3_index,
    frequency_limit
):

    total_true_pairs = 0
    total_s2_pairs = 0
    total_s3_pairs = 0

    recovered_pairs = 0
    recovered_s2 = 0
    recovered_s3 = 0

    matched_s1 = 0
    fully_recovered_s1 = 0

    candidate_counts = []

    zero_candidates = 0
    matched_zero_candidates = 0

    for _, row in sample.iterrows():

        s1_id = row["source1_entity_id"]

        s1 = s1_records[s1_id]

        country = s1["country"]

        tokens = s1_tokens[s1_id]

        candidates = set()

        for token in tokens:

            key = (
                country,
                token
            )

            candidates.update(
                s2_index.get(
                    key,
                    set()
                )
            )

            candidates.update(
                s3_index.get(
                    key,
                    set()
                )
            )

        candidate_count = len(candidates)

        candidate_counts.append(
            candidate_count
        )

        if candidate_count == 0:
            zero_candidates += 1

        true_matches = set(
            parse_ground_truth(
                row["matched_entity_ids"]
            )
        )

        if true_matches:
            matched_s1 += 1

        if (
            true_matches
            and candidate_count == 0
        ):
            matched_zero_candidates += 1

        total_true_pairs += len(
            true_matches
        )

        for entity_id in true_matches:

            if entity_id.startswith("S2-"):
                total_s2_pairs += 1

            elif entity_id.startswith("S3-"):
                total_s3_pairs += 1

            if entity_id in candidates:

                recovered_pairs += 1

                if entity_id.startswith("S2-"):
                    recovered_s2 += 1

                elif entity_id.startswith("S3-"):
                    recovered_s3 += 1

        if (
            true_matches
            and true_matches.issubset(candidates)
        ):
            fully_recovered_s1 += 1

    candidate_series = pd.Series(
        candidate_counts
    )

    return {
        "frequency_limit": frequency_limit,
        "pair_recall": (
            recovered_pairs / total_true_pairs
            if total_true_pairs
            else 0
        ),
        "s2_recall": (
            recovered_s2 / total_s2_pairs
            if total_s2_pairs
            else 0
        ),
        "s3_recall": (
            recovered_s3 / total_s3_pairs
            if total_s3_pairs
            else 0
        ),
        "full_s1_recall": (
            fully_recovered_s1 / matched_s1
            if matched_s1
            else 0
        ),
        "avg_candidates": candidate_series.mean(),
        "median_candidates": candidate_series.median(),
        "p95_candidates": candidate_series.quantile(0.95),
        "max_candidates": candidate_series.max(),
        "zero_candidates": zero_candidates,
        "matched_zero_candidates": (
            matched_zero_candidates
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 60)
    print(
        "STEP 4H - NAME TOKEN FREQUENCY SWEEP"
    )
    print("=" * 60)

    print()
    print(
        "Testing country + business-name token "
        "blocking at multiple frequency limits."
    )

    print()
    print(
        f"Sample size:       {SAMPLE_SIZE:,}"
    )

    print(
        f"Frequency limits:   "
        f"{FREQUENCY_LIMITS}"
    )

    print(
        f"Random seed:        {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # 1. Sample GT
    # --------------------------------------------------------

    sample = load_sample()

    # --------------------------------------------------------
    # 2. S1
    # --------------------------------------------------------

    s1_records = load_s1_sample(
        sample
    )

    # --------------------------------------------------------
    # 3. S1 tokens
    # --------------------------------------------------------

    (
        s1_tokens,
        relevant_tokens
    ) = prepare_s1_tokens(
        s1_records
    )

    print()
    print(
        f"Relevant S1 name tokens: "
        f"{len(relevant_tokens):,}"
    )

    # --------------------------------------------------------
    # 4. Count source frequencies ONCE
    # --------------------------------------------------------

    s2_counts = count_tokens(
        S2_PATH,
        relevant_tokens,
        "S2"
    )

    s3_counts = count_tokens(
        S3_PATH,
        relevant_tokens,
        "S3"
    )

    # --------------------------------------------------------
    # 5. Evaluate each frequency threshold
    # --------------------------------------------------------

    results = []

    for limit in FREQUENCY_LIMITS:

        print()
        print("=" * 60)
        print(
            f"TESTING FREQUENCY LIMIT: {limit}"
        )
        print("=" * 60)

        usable_s2 = {
            token
            for token, count in s2_counts.items()
            if count <= limit
        }

        usable_s3 = {
            token
            for token, count in s3_counts.items()
            if count <= limit
        }

        print()
        print(
            f"Usable S2 tokens: "
            f"{len(usable_s2):,}"
        )

        print(
            f"Usable S3 tokens: "
            f"{len(usable_s3):,}"
        )

        print()
        print(
            "Building indexes..."
        )

        s2_index = build_index(
            S2_PATH,
            usable_s2,
            "S2",
            limit
        )

        s3_index = build_index(
            S3_PATH,
            usable_s3,
            "S3",
            limit
        )

        result = evaluate(
            sample,
            s1_records,
            s1_tokens,
            s2_index,
            s3_index,
            limit
        )

        results.append(result)

        print()
        print(
            f"Results for frequency <= {limit}"
        )

        print(
            f"Pair recall:              "
            f"{result['pair_recall']:.4f}"
        )

        print(
            f"S2 recall:                "
            f"{result['s2_recall']:.4f}"
        )

        print(
            f"S3 recall:                "
            f"{result['s3_recall']:.4f}"
        )

        print(
            f"Full S1 recall:           "
            f"{result['full_s1_recall']:.4f}"
        )

        print(
            f"Average candidates:       "
            f"{result['avg_candidates']:.2f}"
        )

        print(
            f"Median candidates:        "
            f"{result['median_candidates']:.2f}"
        )

        print(
            f"P95 candidates:           "
            f"{result['p95_candidates']:.2f}"
        )

        print(
            f"Maximum candidates:       "
            f"{result['max_candidates']:.0f}"
        )

        print(
            f"Zero candidates:          "
            f"{result['zero_candidates']:,}"
        )

        print(
            f"Matched zero candidates:  "
            f"{result['matched_zero_candidates']:,}"
        )

        # Release the large indexes before next threshold.
        del s2_index
        del s3_index

    # --------------------------------------------------------
    # 6. Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "STEP 4H - FREQUENCY SWEEP SUMMARY"
    )
    print("=" * 60)

    print()

    print(
        f"{'Limit':>8} "
        f"{'PairRec':>10} "
        f"{'S2Rec':>10} "
        f"{'S3Rec':>10} "
        f"{'FullS1':>10} "
        f"{'AvgCand':>10} "
        f"{'P95Cand':>10} "
        f"{'MaxCand':>10}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['frequency_limit']:>8} "
            f"{result['pair_recall']:>10.4f} "
            f"{result['s2_recall']:>10.4f} "
            f"{result['s3_recall']:>10.4f} "
            f"{result['full_s1_recall']:>10.4f} "
            f"{result['avg_candidates']:>10.2f} "
            f"{result['p95_candidates']:>10.2f} "
            f"{result['max_candidates']:>10.0f}"
        )

    elapsed = time.time() - start

    print()
    print(
        f"Total runtime: "
        f"{elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()