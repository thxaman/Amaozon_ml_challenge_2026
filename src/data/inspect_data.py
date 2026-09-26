from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")


def inspect_file(file_path: Path):
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(file_path, sep="\t")

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nUnique values:")
    for column in df.columns:
        print(f"  {column}: {df[column].nunique(dropna=True)}")


def main():
    files = sorted(DATA_DIR.glob("*.tsv"))

    if not files:
        print(f"No TSV files found in {DATA_DIR}")
        return

    for file_path in files:
        inspect_file(file_path)


if __name__ == "__main__":
    main()