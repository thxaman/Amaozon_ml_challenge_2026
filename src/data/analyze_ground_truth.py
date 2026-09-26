from pathlib import Path
from collections import Counter
import pandas as pd


GROUND_TRUTH_PATH = Path("data/raw/train_ground_truth.tsv")


def main():
    df = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        keep_default_na=False
    )

    print("=" * 70)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 70)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    # Number of matches for every Source 1 entity
    match_counts = df["matched_entity_ids"].apply(
        lambda x: 0 if not x.strip() else len(x.split(","))
    )

    print("\nMatch count distribution:")
    print(match_counts.value_counts().sort_index())

    print("\nBasic statistics:")
    print(match_counts.describe())

    # Singleton / matched entity statistics
    print("\nSource 1 entities:")
    print(f"Total: {len(df):,}")

    print(f"No matches: {(match_counts == 0).sum():,}")
    print(f"At least one match: {(match_counts > 0).sum():,}")

    # Source distribution of matches
    source_counter = Counter()

    for matched_ids in df["matched_entity_ids"]:
        if not matched_ids.strip():
            continue

        for entity_id in matched_ids.split(","):
            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                source_counter["S2"] += 1
            elif entity_id.startswith("S3-"):
                source_counter["S3"] += 1
            else:
                source_counter["UNKNOWN"] += 1

    print("\nMatched entity source distribution:")
    for source, count in source_counter.items():
        print(f"{source}: {count:,}")

    # How many S1 entities match S2, S3, or both?
    s2_only = 0
    s3_only = 0
    both = 0

    for matched_ids in df["matched_entity_ids"]:
        if not matched_ids.strip():
            continue

        ids = [x.strip() for x in matched_ids.split(",")]

        has_s2 = any(x.startswith("S2-") for x in ids)
        has_s3 = any(x.startswith("S3-") for x in ids)

        if has_s2 and has_s3:
            both += 1
        elif has_s2:
            s2_only += 1
        elif has_s3:
            s3_only += 1

    print("\nMatch-source combinations:")
    print(f"S2 only : {s2_only:,}")
    print(f"S3 only : {s3_only:,}")
    print(f"Both    : {both:,}")


if __name__ == "__main__":
    main()