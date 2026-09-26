import os
import re
import time
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
SAMPLE_SIZE = 10_000

# We will test several frequency limits.
FREQUENCY_LIMITS = [25, 50, 100, 250, 500]

CHUNK_SIZE = 250_000

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

S1_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_source1.tsv"
)

S2_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_source2.tsv"
)

S3_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_source3.tsv"
)

GROUND_TRUTH_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_ground_truth.tsv"
)

CURRENT_CANDIDATE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "address_pair_candidate_pairs.tsv"
)

MISS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_blocking_misses.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "targeted_unicode_recovery_results.tsv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def unicode_tokens(value):
    """
    Unicode-aware tokenization.

    Unlike the existing ASCII normalization, this preserves
    tokens from non-Latin scripts.
    """

    if pd.isna(value):
        return []

    value = str(value).lower()

    return re.findall(
        r"[^\W_]+",
        value,
        flags=re.UNICODE
    )


# ============================================================
# GENERIC NAME TOKENS
# ============================================================

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


def meaningful_tokens(value):
    """
    Keep Unicode tokens but remove generic legal/business words.
    """

    tokens = unicode_tokens(value)

    return [
        token
        for token in tokens
        if token not in GENERIC_NAME_TOKENS
        and len(token) >= 2
    ]


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

start = time.time()

print(
    "Loading S1...",
    flush=True
)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
)

s1_sample = s1.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).copy()

s1_sample = s1_sample.set_index(
    "entity_id"
)

sample_ids = set(
    s1_sample.index
)

print(
    f"S1 total rows: {len(s1):,}",
    flush=True
)

print(
    f"S1 sample rows: {len(s1_sample):,}",
    flush=True
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print(
    "\nLoading ground truth...",
    flush=True
)

gt = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

gt = gt[
    gt["source1_entity_id"].isin(
        sample_ids
    )
].copy()


# ============================================================
# BUILD TRUE PAIR SET
# ============================================================

true_pairs = set()

true_matches_by_s1 = defaultdict(set)


for row in gt.itertuples(
    index=False
):

    s1_id = row.source1_entity_id

    matched = row.matched_entity_ids

    if not matched:
        continue

    for candidate_id in matched.split(","):

        candidate_id = candidate_id.strip()

        if not candidate_id:
            continue

        pair = (
            s1_id,
            candidate_id
        )

        true_pairs.add(pair)

        true_matches_by_s1[
            s1_id
        ].add(candidate_id)


print(
    f"True pairs in sample: "
    f"{len(true_pairs):,}",
    flush=True
)


# ============================================================
# LOAD CURRENT CANDIDATES
# ============================================================

print(
    "\nLoading current candidate set...",
    flush=True
)

current_candidates = pd.read_csv(
    CURRENT_CANDIDATE_PATH,
    sep="\t",
    dtype=str
)

current_candidates = current_candidates[
    current_candidates[
        "source1_entity_id"
    ].isin(sample_ids)
]


current_pair_set = set()

for row in current_candidates.itertuples(
    index=False
):

    s1_id = row.source1_entity_id

    candidate_ids = (
        str(row.candidate_entity_ids)
        .split(",")
    )

    for candidate_id in candidate_ids:

        candidate_id = candidate_id.strip()

        if candidate_id:

            current_pair_set.add(
                (
                    s1_id,
                    candidate_id
                )
            )


print(
    f"Current candidate pairs: "
    f"{len(current_pair_set):,}",
    flush=True
)


# ============================================================
# CURRENT MISSED TRUE PAIRS
# ============================================================

missed_true_pairs = (
    true_pairs
    - current_pair_set
)


print(
    f"Current missed true pairs: "
    f"{len(missed_true_pairs):,}",
    flush=True
)


# ============================================================
# MISS IDS
# ============================================================

miss_s1_ids = {
    s1_id
    for s1_id, _ in missed_true_pairs
}

miss_candidate_ids = {
    candidate_id
    for _, candidate_id in missed_true_pairs
}


# ============================================================
# BUILD S1 UNICODE TOKEN INDEX
# ============================================================

print(
    "\nBuilding S1 Unicode meaningful-token index...",
    flush=True
)

s1_token_map = {}

s1_token_frequency = Counter()


for s1_id, row in s1_sample.iterrows():

    tokens = set(
        meaningful_tokens(
            row["business_name"]
        )
    )

    s1_token_map[s1_id] = tokens

    for token in tokens:

        s1_token_frequency[token] += 1


print(
    f"Unique meaningful S1 tokens: "
    f"{len(s1_token_frequency):,}",
    flush=True
)


# ============================================================
# EVALUATE DIFFERENT FREQUENCY LIMITS
# ============================================================

results = []


for frequency_limit in FREQUENCY_LIMITS:

    print(
        "\n============================================================",
        flush=True
    )

    print(
        f"Testing Unicode frequency limit: "
        f"{frequency_limit}",
        flush=True
    )

    print(
        "============================================================",
        flush=True
    )

    # --------------------------------------------------------
    # Build valid token set
    # --------------------------------------------------------

    valid_tokens = {
        token
        for token, frequency
        in s1_token_frequency.items()
        if frequency <= frequency_limit
    }

    print(
        f"Valid tokens: {len(valid_tokens):,}",
        flush=True
    )


    # --------------------------------------------------------
    # Build S1 index
    # --------------------------------------------------------

    s1_index = defaultdict(list)

    for s1_id, tokens in s1_token_map.items():

        for token in tokens:

            if token in valid_tokens:

                s1_index[token].append(
                    s1_id
                )


    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    recovered_pairs = set()

    total_candidates = 0

    s1_with_candidates = set()

    candidate_s1_ids = set()

    candidate_counts = []


    def process_source(
        source_path,
        source_name
    ):

        nonlocal_placeholder = None

        print(
            f"\nScanning {source_name}...",
            flush=True
        )

        local_candidates = 0

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                source_path,
                sep="\t",
                dtype=str,
                chunksize=CHUNK_SIZE
            ),
            start=1
        ):

            for row in chunk.itertuples(
                index=False
            ):

                candidate_id = row.entity_id

                tokens = set(
                    meaningful_tokens(
                        row.business_name
                    )
                )

                if not tokens:
                    continue


                matching_s1_ids = set()

                for token in tokens:

                    if token not in valid_tokens:
                        continue

                    matching_s1_ids.update(
                        s1_index.get(
                            token,
                            []
                        )
                    )


                if not matching_s1_ids:
                    continue


                for s1_id in matching_s1_ids:

                    # We only need the additional
                    # candidates for S1s where the
                    # current blocker missed at least
                    # one true pair.

                    if s1_id not in miss_s1_ids:
                        continue


                    pair = (
                        s1_id,
                        candidate_id
                    )

                    local_candidates += 1

                    candidate_s1_ids.add(
                        s1_id
                    )

                    s1_with_candidates.add(
                        s1_id
                    )

                    if pair in missed_true_pairs:

                        recovered_pairs.add(
                            pair
                        )


            if chunk_number % 10 == 0:

                print(
                    f"{source_name}: "
                    f"processed {chunk_number * CHUNK_SIZE:,} rows",
                    flush=True
                )


        return local_candidates


    s2_candidates = process_source(
        S2_PATH,
        "S2"
    )

    s3_candidates = process_source(
        S3_PATH,
        "S3"
    )

    total_candidates = (
        s2_candidates
        + s3_candidates
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    recovery = (
        len(recovered_pairs)
        / len(missed_true_pairs)
        if missed_true_pairs
        else 0.0
    )


    overall_recovered = (
        len(true_pairs.intersection(
            current_pair_set
            | recovered_pairs
        ))
    )


    overall_recall = (
        overall_recovered
        / len(true_pairs)
        if true_pairs
        else 0.0
    )


    print(
        "\nResults:",
        flush=True
    )

    print(
        f"New candidate pairs: "
        f"{total_candidates:,}",
        flush=True
    )

    print(
        f"Recovered missed true pairs: "
        f"{len(recovered_pairs):,}",
        flush=True
    )

    print(
        f"Miss recovery: "
        f"{recovery:.6f}",
        flush=True
    )

    print(
        f"Overall pair recall: "
        f"{overall_recall:.6f}",
        flush=True
    )

    print(
        f"S1s with recovery candidates: "
        f"{len(candidate_s1_ids):,}",
        flush=True
    )


    results.append(
        {
            "frequency_limit": frequency_limit,
            "valid_tokens": len(valid_tokens),
            "new_candidate_pairs": total_candidates,
            "missed_true_pairs": len(missed_true_pairs),
            "recovered_missed_pairs": len(recovered_pairs),
            "miss_recovery": recovery,
            "overall_pair_recall": overall_recall,
            "s1_with_recovery_candidates": len(
                candidate_s1_ids
            ),
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n============================================================",
    flush=True
)

print(
    "TARGETED UNICODE RECOVERY RESULTS",
    flush=True
)

print(
    "============================================================",
    flush=True
)

print(
    results_df.to_string(
        index=False
    ),
    flush=True
)

print(
    "\nSaved to:",
    flush=True
)

print(
    OUTPUT_PATH,
    flush=True
)

print(
    f"\nRuntime: "
    f"{(time.time() - start) / 60:.2f} minutes",
    flush=True
)

print(
    "\nDone.",
    flush=True
)