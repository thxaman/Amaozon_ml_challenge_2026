import numpy as np
import pandas as pd

FILE = "data/processed/full_training_features.tsv"

print("Loading feature file...", flush=True)

df = pd.read_csv(
    FILE,
    sep="\t"
)

print("\nShape:")
print(df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSource distribution:")
print(df["source"].value_counts())

print("\nMissing values:")

missing = df.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)

print("\nDuplicate candidate pairs:")

duplicates = df.duplicated(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
).sum()

print(duplicates)

print("\nInfinite numeric values:")

numeric = df.select_dtypes(include=np.number)

inf_count = np.isinf(numeric.to_numpy()).sum()

print(inf_count)

print("\nFeature columns:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")

print("\nPositive rate:")

positive_rate = df["label"].mean()

print(f"{positive_rate:.6%}")

print("\nValidation complete.")