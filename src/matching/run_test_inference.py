"""
STEP T3 - PRODUCTION TEST INFERENCE

Loads the already-trained, frozen classifier
(data/processed/final_match_classifier.joblib) and scores every test
candidate pair produced by build_test_features.py.

This script does NOT train anything and does NOT tune the threshold.
Per MASTER_PROMPT.txt Section 18/19, the frozen threshold is 0.30 and
must not be changed based on test predictions. That value is hardcoded
below as FROZEN_THRESHOLD. If the threshold stored inside the model
package differs from 0.30 (e.g. a newer retrain picked a different
validation-selected threshold), this script prints a clear warning so
you notice, but still applies FROZEN_THRESHOLD unless you explicitly
pass --use-model-threshold.

INPUT:
    data/processed/final_match_classifier.joblib
    data/processed/test_pair_features.tsv
    data/test/test_source1.tsv   (to guarantee every test S1 appears
                                   exactly once, including zero-match
                                   S1 entities)

OUTPUT:
    data/processed/matching_results.tsv
        columns: source1_entity_id, matched_entity_ids
        (matched_entity_ids is a comma-separated list, or empty)
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_FILE = os.path.join(
    BASE_DIR, "data", "processed", "final_match_classifier.joblib"
)

FEATURE_FILE = os.path.join(
    BASE_DIR, "data", "processed", "test_pair_features.tsv"
)

TEST_S1_PATH = os.path.join(BASE_DIR, "data", "test", "test_source1.tsv")

OUTPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed", "matching_results.tsv"
)

SCORED_OUTPUT_FILE = os.path.join(
    BASE_DIR, "data", "processed", "test_pair_scores.tsv"
)

# Frozen per MASTER_PROMPT.txt Section 18. Do not tune on test data.
FROZEN_THRESHOLD = 0.30

CHUNK_SIZE = 200_000


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-model-threshold",
        action="store_true",
        help=(
            "Use the threshold stored inside the model package instead "
            "of the frozen 0.30. Only use this if you deliberately "
            "retrained and refroze a new threshold on TRAIN/VALIDATION "
            "data (never on test data)."
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP T3 - PRODUCTION TEST INFERENCE")
    print("=" * 70)

    print(f"\nLoading frozen model: {MODEL_FILE}", flush=True)

    package = joblib.load(MODEL_FILE)

    model = package["model"]
    feature_columns = package["feature_columns"]
    stored_threshold = package.get("threshold")

    print(f"Model feature count: {len(feature_columns)}", flush=True)
    print(f"Threshold stored in model package: {stored_threshold}",
          flush=True)

    if args.use_model_threshold and stored_threshold is not None:
        threshold = float(stored_threshold)
    else:
        threshold = FROZEN_THRESHOLD

    if stored_threshold is not None and abs(
        float(stored_threshold) - threshold
    ) > 1e-9:
        print(
            f"WARNING: using threshold {threshold}, which differs from "
            f"the threshold stored in the model package "
            f"({stored_threshold}). This is expected if the frozen "
            f"competition threshold (0.30) differs from whatever "
            f"validation-selected threshold happened to be saved "
            f"last. The frozen 0.30 threshold is being used unless "
            f"--use-model-threshold was passed.",
            flush=True,
        )

    print(f"\nUsing threshold: {threshold}", flush=True)

    # --------------------------------------------------------
    # Score candidates in chunks (test_pair_features.tsv can be
    # large - millions of rows)
    # --------------------------------------------------------

    print(f"\nScoring candidates from: {FEATURE_FILE}", flush=True)

    os.makedirs(os.path.dirname(SCORED_OUTPUT_FILE), exist_ok=True)

    matches_by_s1 = {}

    first_chunk = True
    total_rows = 0
    total_accepted = 0

    reader = pd.read_csv(FEATURE_FILE, sep="\t", chunksize=CHUNK_SIZE)

    for chunk_number, chunk in enumerate(reader, start=1):

        missing = [c for c in feature_columns if c not in chunk.columns]
        if missing:
            raise ValueError(
                f"test_pair_features.tsv is missing model feature "
                f"columns: {missing}. Did build_test_features.py run "
                f"with the same feature logic as build_full_pair_"
                f"features.py?"
            )

        X = chunk[feature_columns].astype(np.float32)

        scores = model.predict_proba(X)[:, 1]

        chunk_out = chunk[
            ["source1_entity_id", "candidate_entity_id", "source"]
        ].copy()
        chunk_out["score"] = scores

        chunk_out.to_csv(
            SCORED_OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False

        accepted = chunk_out[chunk_out["score"] >= threshold]

        for row in accepted.itertuples(index=False):
            matches_by_s1.setdefault(row.source1_entity_id, set()).add(
                row.candidate_entity_id
            )

        total_rows += len(chunk)
        total_accepted += len(accepted)

        print(
            f"Chunk {chunk_number:04d} | "
            f"scored={len(chunk):,} | "
            f"accepted={len(accepted):,} | "
            f"total_scored={total_rows:,} | "
            f"total_accepted={total_accepted:,}",
            flush=True,
        )

    print(f"\nTotal candidate pairs scored  : {total_rows:,}", flush=True)
    print(f"Total accepted (score>=thr)   : {total_accepted:,}", flush=True)
    print(f"S1 entities with >=1 match    : {len(matches_by_s1):,}",
          flush=True)

    # --------------------------------------------------------
    # Ensure every test S1 appears exactly once, including
    # zero-match S1 entities.
    # --------------------------------------------------------

    print(f"\nLoading full test S1 list from: {TEST_S1_PATH}", flush=True)

    test_s1 = pd.read_csv(
        TEST_S1_PATH, sep="\t", dtype=str, keep_default_na=False
    )

    all_s1_ids = test_s1["entity_id"].tolist()

    print(f"Total test S1 entities: {len(all_s1_ids):,}", flush=True)

    print(f"\nWriting: {OUTPUT_FILE}", flush=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        f.write("source1_entity_id\tmatched_entity_ids\n")

        for s1_id in all_s1_ids:

            matches = sorted(matches_by_s1.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(matches)}\n")

    print("\nDONE.", flush=True)
    print(f"Submission file: {OUTPUT_FILE}")
    print(f"Per-pair scores (for auditing): {SCORED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
