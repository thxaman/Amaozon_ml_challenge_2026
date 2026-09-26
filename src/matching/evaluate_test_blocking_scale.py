import re
import time
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10_000
CHUNK_SIZE = 250_000
RANDOM_STATE = 42

ADDRESS_PAIR_FREQ = 50
BROAD_ADDRESS_FREQ = 2500
NAME_PAIR_FREQ = 1000

S1_PATH = "data/test/test_source1.tsv"
S2_PATH = "data/test/test_source2.tsv"
S3_PATH = "data/test/test_source3.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# NAME TOKENS
# ============================================================

GENERIC_NAME_TOKENS = {
    "company",
    "companies",
    "corporation",
    "corp",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "private",
    "pvt",
    "llp",
    "plc",
    "co",
    "group",
    "holdings",
    "holding",
    "services",
    "service",
    "solutions",
    "solution",
    "enterprise",
    "enterprises",
    "industries",
    "industry",
    "international",
    "global",
}


def meaningful_name_tokens(text):

    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
        and token not in GENERIC_NAME_TOKENS
    }


# ============================================================
# ADDRESS TOKENS
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
    "highway",
    "hwy",
    "boulevard",
    "blvd",
    "parkway",
    "pkwy",
    "place",
    "pl",
    "building",
    "bldg",
    "floor",
    "fl",
    "suite",
    "ste",
    "unit",
    "block",
    "sector",
    "near",
    "opposite",
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
# LOAD TEST S1 SAMPLE
# ============================================================

print("=" * 70)
print("TEST BLOCKING SCALE CHECK")
print("=" * 70)

start = time.time()

print("\nLoading test S1 sample...", flush=True)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

s1 = s1.sample(
    n=min(SAMPLE_SIZE, len(s1)),
    random_state=RANDOM_STATE,
).reset_index(drop=True)

print(
    f"Sampled S1: {len(s1):,}",
    flush=True,
)


# ============================================================
# PREPARE S1 KEYS
# ============================================================

print("\nPreparing S1 keys...", flush=True)

s1_data = {}

all_exact_names = set()
all_address_tokens = set()
all_name_pairs = set()
all_name_tokens = set()

for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = row["country"]

    name = normalize_text(
        row["business_name"]
    )

    address = row["business_address"]

    name_tokens = meaningful_name_tokens(name)

    address_tokens = meaningful_address_tokens(
        address
    )

    sorted_name_tokens = sorted(
        name_tokens
    )

    name_pairs = set()

    for i in range(
        len(sorted_name_tokens)
    ):
        for j in range(
            i + 1,
            len(sorted_name_tokens)
        ):
            name_pairs.add(
                (
                    sorted_name_tokens[i],
                    sorted_name_tokens[j],
                )
            )

    s1_data[s1_id] = {
        "country": country,
        "name": name,
        "address_tokens": address_tokens,
        "name_pairs": name_pairs,
    }

    if name:
        all_exact_names.add(name)

    all_address_tokens.update(
        address_tokens
    )

    all_name_tokens.update(
        name_tokens
    )

    all_name_pairs.update(
        name_pairs
    )


print(
    f"Exact names       : {len(all_exact_names):,}"
)

print(
    f"Address tokens    : {len(all_address_tokens):,}"
)

print(
    f"Name tokens       : {len(all_name_tokens):,}"
)

print(
    f"Name pairs        : {len(all_name_pairs):,}"
)


# ============================================================
# FREQUENCY TABLES
# ============================================================

print("\nBuilding frequency tables...", flush=True)

exact_name_freq = Counter()
address_freq = Counter()
name_pair_freq = Counter()
address_pair_freq = Counter()


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
        # Name pairs
        # ----------------------------------------------------

        name_tokens = sorted(
            meaningful_name_tokens(name)
            & all_name_tokens
        )

        for i in range(
            len(name_tokens)
        ):

            for j in range(
                i + 1,
                len(name_tokens)
            ):

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

        # ----------------------------------------------------
        # Address pairs
        # ----------------------------------------------------

        address_tokens = sorted(
            meaningful_address_tokens(address)
            & all_address_tokens
        )

        for i in range(
            len(address_tokens)
        ):

            for j in range(
                i + 1,
                len(address_tokens)
            ):

                pair = (
                    address_tokens[i],
                    address_tokens[j],
                )

                address_pair_freq[
                    (
                        country,
                        pair,
                    )
                ] += 1


for source_name, path in [
    ("S2", S2_PATH),
    ("S3", S3_PATH),
]:

    print(
        f"\nScanning {source_name}...",
        flush=True,
    )

    rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        process_frequency_chunk(chunk)

        rows += len(chunk)

        print(
            f"  {rows:,} rows...",
            flush=True,
        )


# ============================================================
# BUILD INDEXES
# ============================================================

print("\nBuilding indexes...", flush=True)

exact_name_index = defaultdict(set)
address_index = defaultdict(set)
name_pair_index = defaultdict(set)
address_pair_index = defaultdict(set)


def index_chunk(chunk):

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

            exact_name_index[key].add(
                entity_id
            )

        # ----------------------------------------------------
        # Address tokens
        # ----------------------------------------------------

        address_tokens = sorted(
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
        # Name pairs
        # ----------------------------------------------------

        name_tokens = sorted(
            meaningful_name_tokens(name)
            & all_name_tokens
        )

        for i in range(
            len(name_tokens)
        ):

            for j in range(
                i + 1,
                len(name_tokens)
            ):

                pair = (
                    name_tokens[i],
                    name_tokens[j],
                )

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

        # ----------------------------------------------------
        # Address pairs
        # ----------------------------------------------------

        for i in range(
            len(address_tokens)
        ):

            for j in range(
                i + 1,
                len(address_tokens)
            ):

                pair = (
                    address_tokens[i],
                    address_tokens[j],
                )

                key = (
                    country,
                    pair,
                )

                if (
                    address_pair_freq[key]
                    <= ADDRESS_PAIR_FREQ
                ):

                    address_pair_index[key].add(
                        entity_id
                    )


for source_name, path in [
    ("S2", S2_PATH),
    ("S3", S3_PATH),
]:

    print(
        f"\nIndexing {source_name}...",
        flush=True,
    )

    rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        index_chunk(chunk)

        rows += len(chunk)

        print(
            f"  {rows:,} rows...",
            flush=True,
        )


print(
    f"\nExact-name index keys : "
    f"{len(exact_name_index):,}"
)

print(
    f"Address index keys    : "
    f"{len(address_index):,}"
)

print(
    f"Name-pair index keys  : "
    f"{len(name_pair_index):,}"
)

print(
    f"Address-pair keys     : "
    f"{len(address_pair_index):,}"
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\nGenerating candidates...", flush=True)

candidate_counts = []

zero_candidates = 0

channel_usage = {
    "exact_name": 0,
    "address": 0,
    "name_pair": 0,
    "address_pair": 0,
}


for index, (s1_id, data) in enumerate(
    s1_data.items(),
    start=1,
):

    country = data["country"]

    candidates = set()

    # --------------------------------------------------------
    # 1. Exact name
    # --------------------------------------------------------

    if data["name"]:

        key = (
            country,
            data["name"],
        )

        ids = exact_name_index.get(
            key,
            set(),
        )

        if ids:
            channel_usage["exact_name"] += 1
            candidates.update(ids)

    # --------------------------------------------------------
    # 2. Address token
    # --------------------------------------------------------

    found = False

    for token in data["address_tokens"]:

        key = (
            country,
            token,
        )

        ids = address_index.get(
            key,
            set(),
        )

        if ids:
            found = True
            candidates.update(ids)

    if found:
        channel_usage["address"] += 1

    # --------------------------------------------------------
    # 3. Name token pair
    # --------------------------------------------------------

    found = False

    for pair in data["name_pairs"]:

        key = (
            country,
            pair,
        )

        ids = name_pair_index.get(
            key,
            set(),
        )

        if ids:
            found = True
            candidates.update(ids)

    if found:
        channel_usage["name_pair"] += 1

    # --------------------------------------------------------
    # 4. Address token pair
    # --------------------------------------------------------

    address_tokens = sorted(
        data["address_tokens"]
    )

    found = False

    for i in range(
        len(address_tokens)
    ):

        for j in range(
            i + 1,
            len(address_tokens)
        ):

            pair = (
                address_tokens[i],
                address_tokens[j],
            )

            key = (
                country,
                pair,
            )

            ids = address_pair_index.get(
                key,
                set(),
            )

            if ids:
                found = True
                candidates.update(ids)

    if found:
        channel_usage["address_pair"] += 1

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    count = len(candidates)

    candidate_counts.append(
        count
    )

    if count == 0:
        zero_candidates += 1

    if index % 1000 == 0:

        print(
            f"  {index:,}/{len(s1_data):,}",
            flush=True,
        )


# ============================================================
# RESULTS
# ============================================================

series = pd.Series(
    candidate_counts
)

average = series.mean()
median = series.median()
p95 = series.quantile(0.95)
maximum = series.max()

estimated_total = (
    average * len(s1_data)
)


print()
print("=" * 70)
print("TEST BLOCKING SCALE RESULTS")
print("=" * 70)

print(
    f"S1 sample              : "
    f"{len(s1_data):,}"
)

print(
    f"Average candidates/S1  : "
    f"{average:,.2f}"
)

print(
    f"Median candidates/S1   : "
    f"{median:,.0f}"
)

print(
    f"P95 candidates/S1      : "
    f"{p95:,.2f}"
)

print(
    f"Maximum candidates/S1  : "
    f"{maximum:,}"
)

print(
    f"Zero-candidate S1      : "
    f"{zero_candidates:,}"
)

print()
print(
    f"Estimated full test "
    f"candidate pairs       : "
    f"{estimated_total:,.0f}"
)

print()
print("Channel usage:")

for channel, count in channel_usage.items():

    print(
        f"  {channel:<20}: "
        f"{count:,}"
    )

print()
print(
    f"Runtime: "
    f"{(time.time() - start) / 60:.2f} minutes"
)