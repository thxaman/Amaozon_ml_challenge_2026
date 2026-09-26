import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

GT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_ground_truth.tsv"
)

CANDIDATE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "address_pair_candidate_pairs.tsv"
)


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):

    beta2 = 0.5 ** 2

    if precision == 0 and recall == 0:
        return 0.0

    return (
        (1 + beta2) * precision * recall
        / (beta2 * precision + recall)
    )


# ============================================================
# LOAD CANDIDATES FIRST
# ============================================================

print("Loading candidate file...", flush=True)

candidates = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str
)

candidate_s1_ids = set(
    candidates["source1_entity_id"]
)

print(
    f"Candidate S1 rows: {len(candidates):,}",
    flush=True
)

print(
    f"Candidate S1 IDs:  {len(candidate_s1_ids):,}",
    flush=True
)


# ============================================================
# BUILD CANDIDATE SETS
# ============================================================

print("\nBuilding candidate sets...", flush=True)

candidate_matches = {}

for row in candidates.itertuples(index=False):

    s1_id = row.source1_entity_id
    candidate_ids = row.candidate_entity_ids

    if pd.isna(candidate_ids) or not str(candidate_ids).strip():

        candidate_matches[s1_id] = set()

    else:

        candidate_matches[s1_id] = {
            x.strip()
            for x in str(candidate_ids).split(",")
            if x.strip()
        }


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...", flush=True)

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

print(
    f"Ground truth rows: {len(gt):,}",
    flush=True
)


# ============================================================
# ONLY KEEP GROUND TRUTH FOR OUR 10K S1
# ============================================================

print(
    "\nFiltering ground truth to development S1 sample...",
    flush=True
)

gt_sample = gt[
    gt["source1_entity_id"].isin(candidate_s1_ids)
].copy()

print(
    f"Ground truth S1 rows in sample: {len(gt_sample):,}",
    flush=True
)


# ============================================================
# BUILD TRUE MATCH SETS
# ============================================================

true_matches = {}

for row in gt_sample.itertuples(index=False):

    s1_id = row.source1_entity_id
    matched = row.matched_entity_ids

    if pd.isna(matched) or not str(matched).strip():

        true_matches[s1_id] = set()

    else:

        true_matches[s1_id] = {
            x.strip()
            for x in str(matched).split(",")
            if x.strip()
        }


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nEvaluating S1-level candidate coverage...",
    flush=True
)

f05_scores = []

fully_recovered = 0
partially_recovered = 0
zero_recovered = 0

matched_s1_count = 0
zero_match_s1_count = 0

true_pairs_total = 0
recovered_pairs_total = 0

s1_recalls = []

zero_candidate_s1 = 0
matched_zero_candidate = 0


for s1_id in candidate_s1_ids:

    true_set = true_matches.get(
        s1_id,
        set()
    )

    candidate_set = candidate_matches.get(
        s1_id,
        set()
    )

    true_count = len(true_set)

    recovered_set = (
        true_set.intersection(candidate_set)
    )

    recovered_count = len(recovered_set)

    true_pairs_total += true_count
    recovered_pairs_total += recovered_count

    # --------------------------------------------------------
    # S1 match statistics
    # --------------------------------------------------------

    if true_count > 0:

        matched_s1_count += 1

        if len(candidate_set) == 0:
            matched_zero_candidate += 1

        if recovered_count == true_count:
            fully_recovered += 1

        elif recovered_count > 0:
            partially_recovered += 1

        else:
            zero_recovered += 1

        s1_recall = (
            recovered_count / true_count
        )

    else:

        zero_match_s1_count += 1

        # No true matches => candidate recall is effectively 1.
        s1_recall = 1.0

    s1_recalls.append(s1_recall)

    # --------------------------------------------------------
    # Candidate set treated as prediction
    # --------------------------------------------------------

    tp = recovered_count

    fp = len(
        candidate_set - true_set
    )

    fn = len(
        true_set - candidate_set
    )

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    f05_scores.append(
        f05(
            precision,
            recall
        )
    )

    if len(candidate_set) == 0:
        zero_candidate_s1 += 1


# ============================================================
# FINAL METRICS
# ============================================================

pair_recall = (
    recovered_pairs_total / true_pairs_total
    if true_pairs_total > 0
    else 0.0
)

mean_s1_recall = np.mean(
    s1_recalls
)

macro_f05 = np.mean(
    f05_scores
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("EXACT S1-LEVEL CANDIDATE EVALUATION")
print("=" * 70)

print(
    f"Development S1 entities:         {len(candidate_s1_ids):,}"
)

print(
    f"S1 with >=1 true match:          {matched_s1_count:,}"
)

print(
    f"S1 with zero true matches:       {zero_match_s1_count:,}"
)

print()

print(
    f"True pairs:                       {true_pairs_total:,}"
)

print(
    f"Recovered true pairs:             {recovered_pairs_total:,}"
)

print(
    f"Pair recall:                      {pair_recall:.6f}"
)

print(
    f"Mean S1 recall:                   {mean_s1_recall:.6f}"
)

print()

print(
    f"Fully recovered matched S1:       {fully_recovered:,}"
)

print(
    f"Partially recovered matched S1:   {partially_recovered:,}"
)

print(
    f"Zero recovered matched S1:        {zero_recovered:,}"
)

print(
    f"S1 with zero candidates:          {zero_candidate_s1:,}"
)

print(
    f"Matched S1 with zero candidates:  {matched_zero_candidate:,}"
)

print()

print(
    f"S1 macro F0.5 of candidate set:  {macro_f05:.6f}"
)

print("=" * 70)