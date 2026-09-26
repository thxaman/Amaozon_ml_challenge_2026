from pathlib import Path
import re
import random

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio


RAW_DIR = Path("data/raw")

SOURCE_FILES = {
    "S1": RAW_DIR / "train_source1.tsv",
    "S2": RAW_DIR / "train_source2.tsv",
    "S3": RAW_DIR / "train_source3.tsv",
}

GROUND_TRUTH_PATH = RAW_DIR / "train_ground_truth.tsv"

SAMPLE_S1 = 10_000


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Keep Unicode characters, remove punctuation/symbol noise.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def load_source(path):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string",
        },
    ).set_index("entity_id")


def similarity_stats(s1_row, match_row):
    s1_name = normalize_text(s1_row["business_name"])
    match_name = normalize_text(match_row["business_name"])

    s1_address = normalize_text(s1_row["business_address"])
    match_address = normalize_text(match_row["business_address"])

    return {
        "name_exact": int(
            bool(s1_name) and bool(match_name) and s1_name == match_name
        ),
        "name_ratio": ratio(s1_name, match_name),
        "name_token_ratio": token_set_ratio(s1_name, match_name),
        "address_exact": int(
            bool(s1_address)
            and bool(match_address)
            and s1_address == match_address
        ),
        "address_ratio": ratio(s1_address, match_address),
        "address_token_ratio": token_set_ratio(
            s1_address,
            match_address,
        ),
        "country_same": int(
            s1_row["country"] == match_row["country"]
        ),
    }


def main():
    print("=" * 70)
    print("POSITIVE PAIR ANALYSIS")
    print("=" * 70)

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        keep_default_na=False,
        dtype="string",
    )

    # Sample Source-1 entities so we don't need to analyze
    # all 7.6M positive pairs in the first iteration.
    if len(gt) > SAMPLE_S1:
        gt_sample = gt.sample(
            n=SAMPLE_S1,
            random_state=42,
        )
    else:
        gt_sample = gt

    print(f"Ground truth rows: {len(gt):,}")
    print(f"S1 entities sampled: {len(gt_sample):,}")

    # Collect only IDs required by this sample.
    required_ids = {
        "S1": set(gt_sample["source1_entity_id"])
    }

    required_ids["S2"] = set()
    required_ids["S3"] = set()

    for value in gt_sample["matched_entity_ids"]:
        if not value.strip():
            continue

        for entity_id in value.split(","):
            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                required_ids["S2"].add(entity_id)

            elif entity_id.startswith("S3-"):
                required_ids["S3"].add(entity_id)

    print(f"S1 records required: {len(required_ids['S1']):,}")
    print(f"S2 records required: {len(required_ids['S2']):,}")
    print(f"S3 records required: {len(required_ids['S3']):,}")

    # ------------------------------------------------------------
    # Read only required records from each source.
    # ------------------------------------------------------------

    source_data = {}

    for source, path in SOURCE_FILES.items():
        print(f"\nLoading required {source} records...")

        chunks = []

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype={
                "entity_id": "string",
                "business_name": "string",
                "business_address": "string",
                "country": "string",
            },
            chunksize=250_000,
        ):
            selected = chunk[
                chunk["entity_id"].isin(required_ids[source])
            ]

            if not selected.empty:
                chunks.append(selected)

        if chunks:
            source_data[source] = pd.concat(
                chunks,
                ignore_index=True,
            ).set_index("entity_id")
        else:
            source_data[source] = pd.DataFrame()

        print(
            f"Loaded {len(source_data[source]):,} {source} records."
        )

    # ------------------------------------------------------------
    # Analyze positive pairs.
    # ------------------------------------------------------------

    results = []

    print("\nAnalyzing positive pairs...")

    for _, gt_row in gt_sample.iterrows():

        s1_id = gt_row["source1_entity_id"]

        if s1_id not in source_data["S1"].index:
            continue

        s1 = source_data["S1"].loc[s1_id]

        matched_ids = gt_row["matched_entity_ids"]

        if not matched_ids.strip():
            continue

        for match_id in matched_ids.split(","):
            match_id = match_id.strip()

            if match_id.startswith("S2-"):
                source = "S2"
            elif match_id.startswith("S3-"):
                source = "S3"
            else:
                continue

            if match_id not in source_data[source].index:
                continue

            match = source_data[source].loc[match_id]

            stats = similarity_stats(s1, match)

            stats["source"] = source
            results.append(stats)

    result_df = pd.DataFrame(results)

    print(f"\nPositive pairs analyzed: {len(result_df):,}")

    if result_df.empty:
        print("No positive pairs found.")
        return

    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nCountry agreement:")
    print(
        result_df["country_same"]
        .value_counts()
        .sort_index()
        .rename(index={0: "Different", 1: "Same"})
    )

    print("\nExact normalized name:")
    print(
        result_df["name_exact"]
        .value_counts()
        .sort_index()
        .rename(index={0: "No", 1: "Yes"})
    )

    print("\nExact normalized address:")
    print(
        result_df["address_exact"]
        .value_counts()
        .sort_index()
        .rename(index={0: "No", 1: "Yes"})
    )

    print("\nSimilarity statistics:")
    print(
        result_df[
            [
                "name_ratio",
                "name_token_ratio",
                "address_ratio",
                "address_token_ratio",
            ]
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
                0.99,
            ]
        )
    )

    print("\nBy source:")
    print(
        result_df.groupby("source")[
            [
                "name_ratio",
                "name_token_ratio",
                "address_ratio",
                "address_token_ratio",
                "name_exact",
                "address_exact",
            ]
        ].mean()
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION HELP")
    print("=" * 70)

    print(
        """
Use these results to decide our blocking strategy.

Important things we are looking for:

1. How often does a true match have the same normalized name?
2. How often does a true match have a highly similar name?
3. How often does a true match have a highly similar address?
4. How useful is exact country equality?
5. Are S2 and S3 behaving differently?

We will use these observations to design candidate generation.
"""
    )


if __name__ == "__main__":
    main()