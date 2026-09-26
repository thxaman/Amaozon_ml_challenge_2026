import re
import time
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42

GROUND_TRUTH = "data/raw/train_ground_truth.tsv"

SOURCE_FILES = {
    "S2": "data/raw/train_source2.tsv",
    "S3": "data/raw/train_source3.tsv",
}

CHUNK_SIZE = 250_000

BASE_ADDRESS_FREQ = 500
BROAD_ADDRESS_FREQ = 2500
NAME_PAIR_FREQ = 1000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


GENERIC_NAME_TOKENS = {
    "company", "companies", "corporation", "corp",
    "limited", "ltd", "llc", "inc", "incorporated",
    "private", "pvt", "llp", "plc", "co", "group",
    "holdings", "holding", "services", "service",
    "solutions", "solution", "enterprise", "enterprises",
    "industries", "industry", "international", "global",
}


GENERIC_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "drive", "dr", "highway", "hwy",
    "boulevard", "blvd", "parkway", "pkwy", "place",
    "pl", "building", "bldg", "floor", "fl", "suite",
    "ste", "unit", "block", "sector", "near", "opposite",
}


def meaningful_name_tokens(text):
    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
        and token not in GENERIC_NAME_TOKENS
    }


def meaningful_address_tokens(text):
    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
        and token not in GENERIC_ADDRESS_TOKENS
    }


# ============================================================
# LOAD SAMPLE + GROUND TRUTH
# ============================================================

print("Loading ground truth...", flush=True)

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

sample = gt.sample(
    n=min(SAMPLE_SIZE, len(gt)),
    random_state=RANDOM_STATE,
)

sample_ids = set(sample["source1_entity_id"])

true_matches = {}

for _, row in sample.iterrows():

    value = row["matched_entity_ids"]

    if value:
        true_matches[row["source1_entity_id"]] = {
            x.strip()
            for x in value.split(",")
            if x.strip()
        }
    else:
        true_matches[row["source1_entity_id"]] = set()


print(
    f"Sampled S1 entities: {len(sample_ids):,}",
    flush=True,
)


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...", flush=True)

s1_parts = []

for chunk in pd.read_csv(
    "data/raw/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE,
):

    selected = chunk[
        chunk["entity_id"].isin(sample_ids)
    ]

    if len(selected):
        s1_parts.append(selected)


s1 = pd.concat(
    s1_parts,
    ignore_index=True,
)

print(
    f"Loaded S1: {len(s1):,}",
    flush=True,
)


# ============================================================
# PREPARE S1 BLOCKING KEYS
# ============================================================

print(
    "\nPreparing S1 blocking keys...",
    flush=True,
)

s1_exact_name = {}
s1_address_tokens = {}
s1_name_pairs = {}

all_exact_names = set()
all_address_tokens = set()
all_name_pairs = set()

# Precomputed once instead of rebuilding it for every S2/S3 row.
all_name_tokens = set()


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    name = normalize_text(
        row["business_name"]
    )

    address = row["business_address"]

    exact_name = name

    address_tokens = meaningful_address_tokens(
        address
    )

    name_tokens = sorted(
        meaningful_name_tokens(name)
    )

    name_pairs = set()

    for i in range(len(name_tokens)):

        for j in range(i + 1, len(name_tokens)):

            pair = (
                name_tokens[i],
                name_tokens[j],
            )

            name_pairs.add(pair)

    s1_exact_name[s1_id] = exact_name
    s1_address_tokens[s1_id] = address_tokens
    s1_name_pairs[s1_id] = name_pairs

    if exact_name:
        all_exact_names.add(exact_name)

    all_address_tokens.update(
        address_tokens
    )

    all_name_pairs.update(
        name_pairs
    )

    all_name_tokens.update(
        name_tokens
    )


print(
    f"Relevant exact names: "
    f"{len(all_exact_names):,}",
    flush=True,
)

print(
    f"Relevant address tokens: "
    f"{len(all_address_tokens):,}",
    flush=True,
)

print(
    f"Relevant name-token pairs: "
    f"{len(all_name_pairs):,}",
    flush=True,
)

print(
    f"Relevant name tokens: "
    f"{len(all_name_tokens):,}",
    flush=True,
)


# ============================================================
# FIRST PASS:
# FIND FREQUENCIES IN S2/S3
# ============================================================

print(
    "\nBuilding frequency tables...",
    flush=True,
)

exact_name_freq = defaultdict(int)
address_freq = defaultdict(int)
name_pair_freq = defaultdict(int)


def process_frequency_chunk(chunk):

    for _, row in chunk.iterrows():

        country = row["country"]

        name = normalize_text(
            row["business_name"]
        )

        address = row["business_address"]

        # ----------------------------------------------------
        # Exact name
        # ----------------------------------------------------

        if name in all_exact_names:

            exact_name_freq[
                (
                    country,
                    name,
                )
            ] += 1

        # ----------------------------------------------------
        # Address tokens
        # ----------------------------------------------------

        address_tokens = (
            meaningful_address_tokens(address)
            & all_address_tokens
        )

        for token in address_tokens:

            address_freq[
                (
                    country,
                    token,
                )
            ] += 1

        # ----------------------------------------------------
        # Name token pairs
        # ----------------------------------------------------

        # Use the precomputed token set.
        name_tokens = sorted(
            meaningful_name_tokens(name)
            & all_name_tokens
        )

        for i in range(len(name_tokens)):

            for j in range(i + 1, len(name_tokens)):

                pair = (
                    name_tokens[i],
                    name_tokens[j],
                )

                if pair in all_name_pairs:

                    name_pair_freq[
                        (
                            country,
                            pair,
                        )
                    ] += 1


for source_name, path in SOURCE_FILES.items():

    print(
        f"  Scanning {source_name}...",
        flush=True,
    )

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        process_frequency_chunk(chunk)

        total_rows += len(chunk)

        print(
            f"    Processed {total_rows:,} rows...",
            flush=True,
        )


print(
    f"\nExact-name frequency keys: "
    f"{len(exact_name_freq):,}",
    flush=True,
)

print(
    f"Address frequency keys: "
    f"{len(address_freq):,}",
    flush=True,
)

print(
    f"Name-pair frequency keys: "
    f"{len(name_pair_freq):,}",
    flush=True,
)


# ============================================================
# CREATE BLOCKING INDEXES
# ============================================================

print(
    "\nBuilding blocking indexes...",
    flush=True,
)

exact_name_index = defaultdict(set)
address_index = defaultdict(set)
name_pair_index = defaultdict(set)


def add_to_indexes(chunk):

    for _, row in chunk.iterrows():

        entity_id = row["entity_id"]
        country = row["country"]

        name = normalize_text(
            row["business_name"]
        )

        address = row["business_address"]

        # ----------------------------------------------------
        # Exact name
        # ----------------------------------------------------

        if name in all_exact_names:

            key = (
                country,
                name,
            )

            if exact_name_freq[key] > 0:

                exact_name_index[key].add(
                    entity_id
                )

        # ----------------------------------------------------
        # Address tokens
        # ----------------------------------------------------

        address_tokens = (
            meaningful_address_tokens(address)
            & all_address_tokens
        )

        for token in address_tokens:

            key = (
                country,
                token,
            )

            if (
                address_freq[key]
                <= BROAD_ADDRESS_FREQ
            ):

                address_index[key].add(
                    entity_id
                )

        # ----------------------------------------------------
        # Name token pairs
        # ----------------------------------------------------

        name_tokens = sorted(
            meaningful_name_tokens(name)
            & all_name_tokens
        )

        for i in range(len(name_tokens)):

            for j in range(i + 1, len(name_tokens)):

                pair = (
                    name_tokens[i],
                    name_tokens[j],
                )

                if pair not in all_name_pairs:
                    continue

                key = (
                    country,
                    pair,
                )

                if (
                    name_pair_freq[key]
                    <= NAME_PAIR_FREQ
                ):

                    name_pair_index[key].add(
                        entity_id
                    )


for source_name, path in SOURCE_FILES.items():

    print(
        f"  Indexing {source_name}...",
        flush=True,
    )

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        add_to_indexes(chunk)

        total_rows += len(chunk)

        print(
            f"    Indexed {total_rows:,} rows...",
            flush=True,
        )


print(
    f"\nExact-name index keys: "
    f"{len(exact_name_index):,}",
    flush=True,
)

print(
    f"Address index keys: "
    f"{len(address_index):,}",
    flush=True,
)

print(
    f"Name-pair index keys: "
    f"{len(name_pair_index):,}",
    flush=True,
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print(
    "\nGenerating candidates...",
    flush=True,
)

candidate_sets = {
    s1_id: set()
    for s1_id in sample_ids
}


# Track which blocking channels generated each candidate.
channel_candidates = {
    s1_id: {
        "exact_name": set(),
        "address": set(),
        "name_pair": set(),
    }
    for s1_id in sample_ids
}


channel_stats = {
    "exact_name": 0,
    "address": 0,
    "name_pair": 0,
}

start = time.time()


for index, (_, row) in enumerate(
    s1.iterrows(),
    start=1,
):

    s1_id = row["entity_id"]
    country = row["country"]

    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    name = s1_exact_name[s1_id]

    if name:

        key = (
            country,
            name,
        )

        ids = exact_name_index.get(
            key,
            set(),
        )

        candidate_sets[s1_id].update(ids)

        channel_candidates[s1_id][
            "exact_name"
        ].update(ids)

        if ids:
            channel_stats["exact_name"] += 1

    # --------------------------------------------------------
    # Address token
    # --------------------------------------------------------

    for token in s1_address_tokens[s1_id]:

        key = (
            country,
            token,
        )

        ids = address_index.get(
            key,
            set(),
        )

        candidate_sets[s1_id].update(ids)

        channel_candidates[s1_id][
            "address"
        ].update(ids)

        if ids:
            channel_stats["address"] += 1

    # --------------------------------------------------------
    # Name token pair
    # --------------------------------------------------------

    for pair in s1_name_pairs[s1_id]:

        key = (
            country,
            pair,
        )

        ids = name_pair_index.get(
            key,
            set(),
        )

        candidate_sets[s1_id].update(ids)

        channel_candidates[s1_id][
            "name_pair"
        ].update(ids)

        if ids:
            channel_stats["name_pair"] += 1

    if index % 1000 == 0:

        print(
            f"  Generated candidates for "
            f"{index:,}/{len(s1):,} S1...",
            flush=True,
        )


elapsed = time.time() - start


# ============================================================
# EVALUATE
# ============================================================

total_true_pairs = 0
recovered_pairs = 0

matched_s1 = 0
fully_recovered_s1 = 0

candidate_counts = []

zero_candidates = 0
matched_zero_candidates = 0


# ------------------------------------------------------------
# Channel contribution statistics
# ------------------------------------------------------------

channel_pair_recovery = {
    "exact_name": 0,
    "address": 0,
    "name_pair": 0,
}

combination_counts = defaultdict(int)


for s1_id in sample_ids:

    candidates = candidate_sets[s1_id]

    candidate_count = len(candidates)

    candidate_counts.append(
        candidate_count
    )

    if candidate_count == 0:
        zero_candidates += 1

    true_set = true_matches[s1_id]

    total_true_pairs += len(true_set)

    recovered = true_set & candidates

    recovered_pairs += len(recovered)

    if true_set:

        matched_s1 += 1

        if recovered == true_set:
            fully_recovered_s1 += 1

        if candidate_count == 0:
            matched_zero_candidates += 1

    # --------------------------------------------------------
    # Analyze every recovered TRUE pair
    # --------------------------------------------------------

    for candidate_id in recovered:

        exact = (
            candidate_id
            in channel_candidates[s1_id]["exact_name"]
        )

        address = (
            candidate_id
            in channel_candidates[s1_id]["address"]
        )

        name_pair = (
            candidate_id
            in channel_candidates[s1_id]["name_pair"]
        )

        # Individual channel recovery
        if exact:
            channel_pair_recovery["exact_name"] += 1

        if address:
            channel_pair_recovery["address"] += 1

        if name_pair:
            channel_pair_recovery["name_pair"] += 1

        # ----------------------------------------------------
        # Exact combination
        # ----------------------------------------------------

        channels = []

        if exact:
            channels.append("exact_name")

        if address:
            channels.append("address")

        if name_pair:
            channels.append("name_pair")

        combination = "+".join(channels)

        combination_counts[combination] += 1


pair_recall = (
    recovered_pairs / total_true_pairs
    if total_true_pairs
    else 0
)

full_s1_recall = (
    fully_recovered_s1 / matched_s1
    if matched_s1
    else 0
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("IMPROVED BLOCKING RESULTS")
print("=" * 70)

print(
    f"S1 sample              : "
    f"{len(sample_ids):,}"
)

print(
    f"True pairs             : "
    f"{total_true_pairs:,}"
)

print(
    f"Recovered pairs        : "
    f"{recovered_pairs:,}"
)

print(
    f"Pair recall            : "
    f"{pair_recall:.4f}"
)

print(
    f"S1 with matches        : "
    f"{matched_s1:,}"
)

print(
    f"Fully recovered S1     : "
    f"{fully_recovered_s1:,}"
)

print(
    f"Full S1 recall         : "
    f"{full_s1_recall:.4f}"
)

print()

print(
    f"Average candidates/S1  : "
    f"{sum(candidate_counts) / len(candidate_counts):.2f}"
)

print(
    f"Median candidates/S1   : "
    f"{pd.Series(candidate_counts).median():.0f}"
)

print(
    f"P95 candidates/S1      : "
    f"{pd.Series(candidate_counts).quantile(.95):.2f}"
)

print(
    f"Maximum candidates/S1 : "
    f"{max(candidate_counts):,}"
)

print(
    f"Zero-candidate S1      : "
    f"{zero_candidates:,}"
)

print(
    f"Matched S1 with zero   : "
    f"{matched_zero_candidates:,}"
)

print()

print("Channel usage:")

print(
    f"  Exact name           : "
    f"{channel_stats['exact_name']:,}"
)

print(
    f"  Address              : "
    f"{channel_stats['address']:,}"
)

print(
    f"  Name token pair      : "
    f"{channel_stats['name_pair']:,}"
)


# ============================================================
# CHANNEL CONTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TRUE-PAIR CHANNEL CONTRIBUTION")
print("=" * 70)

print(
    f"Recovered by exact name : "
    f"{channel_pair_recovery['exact_name']:,}"
)

print(
    f"Recovered by address    : "
    f"{channel_pair_recovery['address']:,}"
)

print(
    f"Recovered by name pair  : "
    f"{channel_pair_recovery['name_pair']:,}"
)


# ============================================================
# CHANNEL COMBINATIONS
# ============================================================

print()
print("=" * 70)
print("TRUE-PAIR CHANNEL COMBINATIONS")
print("=" * 70)

combination_order = [
    "exact_name",
    "address",
    "name_pair",
    "exact_name+address",
    "exact_name+name_pair",
    "address+name_pair",
    "exact_name+address+name_pair",
]

for combination in combination_order:

    count = combination_counts.get(
        combination,
        0,
    )

    percentage = (
        count / recovered_pairs * 100
        if recovered_pairs
        else 0
    )

    print(
        f"{combination:<35}"
        f"{count:>8,}"
        f"  ({percentage:>6.2f}%)"
    )


# ============================================================
# UNIQUE CONTRIBUTION
# ============================================================

print()
print("=" * 70)
print("UNIQUE CHANNEL CONTRIBUTION")
print("=" * 70)

unique_exact = combination_counts.get(
    "exact_name",
    0,
)

unique_address = combination_counts.get(
    "address",
    0,
)

unique_name_pair = combination_counts.get(
    "name_pair",
    0,
)

print(
    f"Only exact name : "
    f"{unique_exact:,}"
)

print(
    f"Only address    : "
    f"{unique_address:,}"
)

print(
    f"Only name pair  : "
    f"{unique_name_pair:,}"
)

print()

print("Interpretation:")

print(
    "  'Only X' means the true pair was "
    "recovered exclusively by that channel."
)

print(
    "  Combinations show true pairs "
    "recovered by multiple channels."
)


# ============================================================
# RUNTIME
# ============================================================

print()
print(
    f"Runtime                : "
    f"{elapsed / 60:.2f} minutes"
)


# ============================================================
# SAVE CANDIDATES
# ============================================================

output_path = (
    "data/processed/"
    "improved_candidate_pairs.tsv"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "source1_entity_id\t"
        "candidate_entity_ids\n"
    )

    for s1_id in sorted(sample_ids):

        ids = sorted(
            candidate_sets[s1_id]
        )

        f.write(
            f"{s1_id}\t"
            f"{','.join(ids)}\n"
        )


print()

print(
    f"Saved candidates to: "
    f"{output_path}"
)