import time
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd


SAMPLE_SIZE = 10_000
CHUNK_SIZE = 250_000
RANDOM_SEED = 42

# Pair frequency limit.
# We will test a few values.
PAIR_LIMITS = [100, 500, 1000]

S1_PATH = "data/raw/train_source1.tsv"
S2_PATH = "data/raw/train_source2.tsv"
S3_PATH = "data/raw/train_source3.tsv"
GT_PATH = "data/raw/train_ground_truth.tsv"


GENERIC_NAME_TOKENS = {
    "the", "and", "of", "for", "in", "at", "on", "to",
    "a", "an",
    "co", "company", "corp", "corporation",
    "inc", "incorporated", "ltd", "limited",
    "llc", "plc", "pvt", "private",
    "group", "holdings", "international",
    "services", "service",
    "business", "businesses",
    "enterprise", "enterprises",
    "india", "indian",
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def name_tokens(value):
    text = normalize_text(value)

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_NAME_TOKENS
    }


def name_pairs(value):
    tokens = sorted(name_tokens(value))

    if len(tokens) < 2:
        return set()

    return set(combinations(tokens, 2))


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


def load_sample():

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return gt.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)


def load_s1_sample(sample):

    required_ids = set(
        sample["source1_entity_id"]
    )

    records = {}

    for chunk in pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        relevant = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in relevant.iterrows():

            records[row["entity_id"]] = {
                "business_name": row["business_name"],
                "country": row["country"],
            }

    return records


def prepare_s1_pairs(s1_records):

    result = {}
    relevant_pairs = set()

    for entity_id, record in s1_records.items():

        pairs = name_pairs(
            record["business_name"]
        )

        result[entity_id] = pairs
        relevant_pairs.update(pairs)

    return result, relevant_pairs


def count_source_pairs(
    path,
    relevant_pairs,
    source_name
):

    counter = Counter()

    print(
        f"\nCounting relevant name pairs in {source_name}..."
    )

    rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        rows += len(chunk)

        for value in chunk["business_name"]:

            pairs = (
                name_pairs(value)
                & relevant_pairs
            )

            for pair in pairs:
                counter[pair] += 1

    print(
        f"{source_name} rows scanned: {rows:,}"
    )

    print(
        f"{source_name} relevant pairs: "
        f"{len(counter):,}"
    )

    return counter


def build_index(
    path,
    usable_pairs,
    source_name
):

    index = defaultdict(set)

    indexed_records = 0
    postings = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        for _, row in chunk.iterrows():

            pairs = (
                name_pairs(row["business_name"])
                & usable_pairs
            )

            if not pairs:
                continue

            indexed_records += 1

            country = row["country"]
            entity_id = row["entity_id"]

            for pair in pairs:

                index[
                    (country, pair)
                ].add(entity_id)

                postings += 1

    print(
        f"{source_name}: "
        f"records={indexed_records:,}, "
        f"postings={postings:,}, "
        f"keys={len(index):,}"
    )

    return index


def evaluate(
    sample,
    s1_records,
    s1_pairs,
    s2_index,
    s3_index
):

    total_true = 0
    recovered = 0

    total_s2 = 0
    recovered_s2 = 0

    total_s3 = 0
    recovered_s3 = 0

    full_recovered = 0
    matched_s1 = 0

    candidate_counts = []

    zero_candidates = 0
    matched_zero = 0

    for _, row in sample.iterrows():

        s1_id = row["source1_entity_id"]

        record = s1_records[s1_id]

        country = record["country"]

        candidates = set()

        for pair in s1_pairs[s1_id]:

            key = (country, pair)

            candidates.update(
                s2_index.get(key, set())
            )

            candidates.update(
                s3_index.get(key, set())
            )

        candidate_counts.append(
            len(candidates)
        )

        true_matches = parse_ground_truth(
            row["matched_entity_ids"]
        )

        total_true += len(true_matches)

        if true_matches:
            matched_s1 += 1

        if not candidates:
            zero_candidates += 1

            if true_matches:
                matched_zero += 1

        for entity_id in true_matches:

            if entity_id.startswith("S2-"):

                total_s2 += 1

                if entity_id in candidates:
                    recovered_s2 += 1

            elif entity_id.startswith("S3-"):

                total_s3 += 1

                if entity_id in candidates:
                    recovered_s3 += 1

            if entity_id in candidates:
                recovered += 1

        if (
            true_matches
            and true_matches.issubset(candidates)
        ):
            full_recovered += 1

    series = pd.Series(candidate_counts)

    return {
        "pair_recall": recovered / total_true,
        "s2_recall": recovered_s2 / total_s2,
        "s3_recall": recovered_s3 / total_s3,
        "full_recall": full_recovered / matched_s1,
        "avg": series.mean(),
        "median": series.median(),
        "p95": series.quantile(.95),
        "max": series.max(),
        "zero": zero_candidates,
        "matched_zero": matched_zero,
    }


def main():

    start = time.time()

    print("=" * 60)
    print("STEP 4I - NAME TOKEN PAIR BLOCKING")
    print("=" * 60)

    sample = load_sample()

    print(
        f"\nSample size: {len(sample):,}"
    )

    s1_records = load_s1_sample(
        sample
    )

    s1_pairs, relevant_pairs = (
        prepare_s1_pairs(
            s1_records
        )
    )

    print(
        f"S1 relevant name pairs: "
        f"{len(relevant_pairs):,}"
    )

    s2_counts = count_source_pairs(
        S2_PATH,
        relevant_pairs,
        "S2"
    )

    s3_counts = count_source_pairs(
        S3_PATH,
        relevant_pairs,
        "S3"
    )

    for limit in PAIR_LIMITS:

        print()
        print("=" * 60)
        print(
            f"PAIR FREQUENCY LIMIT: {limit}"
        )
        print("=" * 60)

        usable_s2 = {
            pair
            for pair, count in s2_counts.items()
            if count <= limit
        }

        usable_s3 = {
            pair
            for pair, count in s3_counts.items()
            if count <= limit
        }

        print(
            f"Usable S2 pairs: {len(usable_s2):,}"
        )

        print(
            f"Usable S3 pairs: {len(usable_s3):,}"
        )

        print("\nBuilding indexes...")

        s2_index = build_index(
            S2_PATH,
            usable_s2,
            "S2"
        )

        s3_index = build_index(
            S3_PATH,
            usable_s3,
            "S3"
        )

        result = evaluate(
            sample,
            s1_records,
            s1_pairs,
            s2_index,
            s3_index
        )

        print()
        print(
            f"Pair recall:       "
            f"{result['pair_recall']:.4f}"
        )

        print(
            f"S2 recall:         "
            f"{result['s2_recall']:.4f}"
        )

        print(
            f"S3 recall:         "
            f"{result['s3_recall']:.4f}"
        )

        print(
            f"Full S1 recall:    "
            f"{result['full_recall']:.4f}"
        )

        print(
            f"Average candidates:"
            f" {result['avg']:.2f}"
        )

        print(
            f"Median candidates: "
            f"{result['median']:.2f}"
        )

        print(
            f"P95 candidates:    "
            f"{result['p95']:.2f}"
        )

        print(
            f"Maximum candidates:"
            f" {result['max']:.0f}"
        )

        print(
            f"Zero candidates:   "
            f"{result['zero']:,}"
        )

        print(
            f"Matched zero:      "
            f"{result['matched_zero']:,}"
        )

        del s2_index
        del s3_index

    print()
    print(
        f"Runtime: "
        f"{(time.time() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()