import os
import re
import pandas as pd
from collections import defaultdict, Counter

# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42
CHUNK_SIZE = 250_000

ADDRESS_TOKEN_FREQ = 2500
NAME_PAIR_FREQ = 1000
ADDRESS_PAIR_FREQ = 50

BASE = "data/raw"

S1_PATH = os.path.join(BASE, "train_source1.tsv")
S2_PATH = os.path.join(BASE, "train_source2.tsv")
S3_PATH = os.path.join(BASE, "train_source3.tsv")
GT_PATH = os.path.join(BASE, "train_ground_truth.tsv")


# ============================================================
# NORMALIZATION
# ============================================================

GENERIC_NAME_TOKENS = {
    "company", "companies", "corporation", "corp",
    "limited", "ltd", "llc", "inc", "incorporated",
    "private", "pvt", "llp", "plc", "co",
    "group", "groups", "holdings", "holding",
    "services", "service", "solutions", "solution",
    "enterprise", "enterprises", "industries", "industry",
    "international", "global"
}

GENERIC_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave",
    "boulevard", "blvd", "lane", "ln", "drive", "dr",
    "parkway", "pkwy", "highway", "hwy", "way",
    "place", "pl", "court", "ct", "circle", "cir",
    "terrace", "ter", "square", "sq",
    "building", "bldg", "floor", "fl",
    "unit", "suite", "ste", "room", "rm",
    "block", "blk", "sector",
    "near", "opposite", "opp",
    "no", "number", "state", "county",
    "district", "city"
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def meaningful_name_tokens(value):
    text = normalize_text(value)

    return [
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_NAME_TOKENS
    ]


def meaningful_address_tokens(value):
    text = normalize_text(value)

    return [
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_ADDRESS_TOKENS
    ]


def token_pairs(tokens):
    tokens = sorted(set(tokens))

    pairs = []

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            pairs.append((tokens[i], tokens[j]))

    return pairs


# ============================================================
# LOAD SAMPLE S1
# ============================================================

print("=" * 70)
print("COMPACT BLOCKING EXPERIMENT — TRAINING DATA")
print("=" * 70)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

s1_sample = s1.sample(
    n=min(SAMPLE_SIZE, len(s1)),
    random_state=RANDOM_STATE
).reset_index(drop=True)

sample_ids = set(s1_sample["entity_id"])

print(f"S1 sample: {len(s1_sample):,}")


# ============================================================
# PREPARE S1 KEYS
# ============================================================

s1_info = {}

exact_names = set()
address_tokens = set()
name_pairs = set()
address_pairs = set()

for row in s1_sample.itertuples(index=False):

    sid = row.entity_id
    country = row.country

    name = normalize_text(row.business_name)
    addr = normalize_text(row.business_address)

    nt = meaningful_name_tokens(name)
    at = meaningful_address_tokens(addr)

    npairs = token_pairs(nt)
    apairs = token_pairs(at)

    s1_info[sid] = {
        "country": country,
        "name": name,
        "address": addr,
        "address_tokens": at,
        "name_pairs": npairs,
        "address_pairs": apairs,
    }

    if name:
        exact_names.add((country, name))

    for token in at:
        address_tokens.add((country, token))

    for pair in npairs:
        name_pairs.add((country, pair))

    for pair in apairs:
        address_pairs.add((country, pair))


print(f"Exact names : {len(exact_names):,}")
print(f"Address tokens : {len(address_tokens):,}")
print(f"Name pairs : {len(name_pairs):,}")
print(f"Address pairs : {len(address_pairs):,}")


# ============================================================
# FREQUENCY TABLES
# ============================================================

print("\nBuilding frequencies...")

address_freq = Counter()
name_pair_freq = Counter()
address_pair_freq = Counter()
exact_name_freq = Counter()


def scan_source(path, source_name):

    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        total += len(chunk)

        for row in chunk.itertuples(index=False):

            country = row.country

            name = normalize_text(row.business_name)
            addr = normalize_text(row.business_address)

            # Exact name
            if (country, name) in exact_names and name:
                exact_name_freq[(country, name)] += 1

            # Address tokens
            at = meaningful_address_tokens(addr)

            for token in set(at):
                key = (country, token)

                if key in address_tokens:
                    address_freq[key] += 1

            # Name pairs
            nt = meaningful_name_tokens(name)

            for pair in token_pairs(nt):
                key = (country, pair)

                if key in name_pairs:
                    name_pair_freq[key] += 1

            # Address pairs
            for pair in token_pairs(at):
                key = (country, pair)

                if key in address_pairs:
                    address_pair_freq[key] += 1

        print(
            f"  {source_name}: {total:,} rows",
            end="\r"
        )

    print()


scan_source(S2_PATH, "S2")
scan_source(S3_PATH, "S3")


# ============================================================
# BUILD COMPACT INDEXES
# ============================================================

print("\nBuilding compact indexes...")

# Each key stores entity IDs.
# Only keys that pass frequency thresholds are retained.

exact_index = defaultdict(set)
address_index = defaultdict(set)
name_pair_index = defaultdict(set)
address_pair_index = defaultdict(set)


def index_source(path, source_name):

    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        total += len(chunk)

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = row.country

            name = normalize_text(row.business_name)
            addr = normalize_text(row.business_address)

            # ------------------------------------------------
            # Exact name
            # ------------------------------------------------

            if name:
                key = (country, name)

                if exact_name_freq.get(key, 0) > 0:
                    exact_index[key].add(
                        (entity_id, source_name)
                    )

            # ------------------------------------------------
            # Address tokens
            # ------------------------------------------------

            at = meaningful_address_tokens(addr)

            for token in set(at):

                key = (country, token)

                if address_freq.get(key, 10**18) <= ADDRESS_TOKEN_FREQ:
                    address_index[key].add(
                        (entity_id, source_name)
                    )

            # ------------------------------------------------
            # Name pairs
            # ------------------------------------------------

            nt = meaningful_name_tokens(name)

            for pair in token_pairs(nt):

                key = (country, pair)

                if name_pair_freq.get(key, 10**18) <= NAME_PAIR_FREQ:
                    name_pair_index[key].add(
                        (entity_id, source_name)
                    )

            # ------------------------------------------------
            # Address pairs
            # ------------------------------------------------

            for pair in token_pairs(at):

                key = (country, pair)

                if address_pair_freq.get(key, 10**18) <= ADDRESS_PAIR_FREQ:
                    address_pair_index[key].add(
                        (entity_id, source_name)
                    )

        print(
            f"  {source_name}: {total:,} rows",
            end="\r"
        )

    print()


index_source(S2_PATH, "S2")
index_source(S3_PATH, "S3")


print("\nIndexes ready.")


# ============================================================
# GROUND TRUTH
# ============================================================

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_ids)
].copy()

true_pairs = defaultdict(set)

for row in gt_sample.itertuples(index=False):

    sid = row.source1_entity_id
    value = row.matched_entity_ids

    if pd.isna(value) or not str(value).strip():
        continue

    for candidate in str(value).split(","):
        candidate = candidate.strip()

        if candidate:
            true_pairs[sid].add(candidate)


total_true_pairs = sum(len(v) for v in true_pairs.values())


# ============================================================
# CANDIDATE GENERATION
# ============================================================

print("\nGenerating compact candidates...")

candidate_counts = []
recovered_pairs = 0
fully_recovered = 0
matched_s1 = 0

for idx, sid in enumerate(s1_sample["entity_id"], start=1):

    info = s1_info[sid]

    candidates = set()

    country = info["country"]

    # --------------------------------------------------------
    # Channel 1 — exact name
    # --------------------------------------------------------

    name = info["name"]

    if name:
        candidates.update(
            entity_id
            for entity_id, source in
            exact_index.get((country, name), set())
        )

    # --------------------------------------------------------
    # Channel 2 — choose the RAREST address tokens
    # --------------------------------------------------------

    address_keys = []

    for token in set(info["address_tokens"]):

        key = (country, token)
        freq = address_freq.get(key)

        if freq is not None and freq <= ADDRESS_TOKEN_FREQ:
            address_keys.append((freq, key))

    # IMPORTANT:
    # Only use the 2 rarest address tokens.

    address_keys.sort(key=lambda x: x[0])

    for _, key in address_keys[:2]:

        candidates.update(
            entity_id
            for entity_id, source in
            address_index.get(key, set())
        )

    # --------------------------------------------------------
    # Channel 3 — rarest name pairs
    # --------------------------------------------------------

    name_keys = []

    for pair in info["name_pairs"]:

        key = (country, pair)
        freq = name_pair_freq.get(key)

        if freq is not None and freq <= NAME_PAIR_FREQ:
            name_keys.append((freq, key))

    name_keys.sort(key=lambda x: x[0])

    # Two rarest name pairs.

    for _, key in name_keys[:2]:

        candidates.update(
            entity_id
            for entity_id, source in
            name_pair_index.get(key, set())
        )

    # --------------------------------------------------------
    # Channel 4 — rarest address pair
    # --------------------------------------------------------

    address_pair_keys = []

    for pair in info["address_pairs"]:

        key = (country, pair)
        freq = address_pair_freq.get(key)

        if freq is not None and freq <= ADDRESS_PAIR_FREQ:
            address_pair_keys.append((freq, key))

    address_pair_keys.sort(key=lambda x: x[0])

    # Use only the two rarest address pairs.

    for _, key in address_pair_keys[:2]:

        candidates.update(
            entity_id
            for entity_id, source in
            address_pair_index.get(key, set())
        )

    candidate_counts.append(len(candidates))

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    truth = true_pairs.get(sid, set())

    recovered = truth.intersection(candidates)

    recovered_pairs += len(recovered)

    if truth:
        matched_s1 += 1

        if len(recovered) == len(truth):
            fully_recovered += 1

    if idx % 1000 == 0:
        print(f"  {idx:,}/{len(s1_sample):,}")


# ============================================================
# RESULTS
# ============================================================

import numpy as np

candidate_counts = np.array(candidate_counts)

print("\n" + "=" * 70)
print("COMPACT BLOCKING RESULTS")
print("=" * 70)

print(f"S1 sample                  : {len(s1_sample):,}")
print(f"True pairs                 : {total_true_pairs:,}")
print(f"Recovered pairs            : {recovered_pairs:,}")

if total_true_pairs:
    print(
        f"Pair recall                : "
        f"{recovered_pairs / total_true_pairs:.6f}"
    )

print(
    f"Matched S1 fully recovered : "
    f"{fully_recovered:,}"
)

if matched_s1:
    print(
        f"Full S1 recall             : "
        f"{fully_recovered / matched_s1:.6f}"
    )

print(
    f"Average candidates/S1      : "
    f"{candidate_counts.mean():,.2f}"
)

print(
    f"Median candidates/S1       : "
    f"{np.median(candidate_counts):,.0f}"
)

print(
    f"P95 candidates/S1          : "
    f"{np.percentile(candidate_counts, 95):,.2f}"
)

print(
    f"Maximum candidates/S1      : "
    f"{candidate_counts.max():,}"
)

print(
    f"Zero-candidate S1          : "
    f"{(candidate_counts == 0).sum():,}"
)

estimated_full = (
    candidate_counts.mean() * len(s1)
)

print(
    f"\nEstimated full candidate pairs: "
    f"{estimated_full:,.0f}"
)

print("=" * 70)