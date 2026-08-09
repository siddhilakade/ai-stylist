"""One-off exploration of the source dataset.

Prints the schema and the distributions that actually drive our design decisions:
which article types exist, how they map to outfit slots, and whether we have
enough coverage per (gender x slot x usage) cell to build complete outfits.

Run:  python scripts/inspect_dataset.py
"""

from __future__ import annotations

import io
from collections import Counter

import pandas as pd
from datasets import load_dataset

DATASET_ID = "ashraq/fashion-product-images-small"


def main() -> None:
    print(f"Loading {DATASET_ID} (first run downloads ~270 MB)...")
    ds = load_dataset(DATASET_ID, split="train")

    print("\n=== SCHEMA ===")
    print(ds.features)
    print(f"\nTotal rows: {len(ds):,}")

    # Drop the image column before going to pandas — otherwise we materialise
    # 44k decoded images in memory for no reason.
    meta_cols = [c for c in ds.column_names if c != "image"]
    df = ds.select_columns(meta_cols).to_pandas()

    print("\n=== NULLS ===")
    print(df.isna().sum())

    for col in ["gender", "masterCategory", "subCategory", "usage", "season"]:
        print(f"\n=== {col} ===")
        print(df[col].value_counts(dropna=False).to_string())

    print("\n=== articleType (top 60) ===")
    print(df["articleType"].value_counts().head(60).to_string())
    print(f"\nDistinct articleType: {df['articleType'].nunique()}")

    print("\n=== baseColour ===")
    print(df["baseColour"].value_counts(dropna=False).to_string())

    print("\n=== gender x subCategory (top cells) ===")
    cross = df.groupby(["gender", "subCategory"]).size().sort_values(ascending=False)
    print(cross.head(40).to_string())

    print("\n=== usage x gender for Apparel/Footwear ===")
    wearable = df[df["masterCategory"].isin(["Apparel", "Footwear", "Accessories"])]
    print(pd.crosstab(wearable["usage"], wearable["gender"]).to_string())

    # Sanity-check that images actually decode.
    print("\n=== IMAGE CHECK ===")
    sample = ds.select(range(3))
    for row in sample:
        img = row["image"]
        print(f"id={row['id']} size={img.size} mode={img.mode}")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        print(f"  encodes to {len(buf.getvalue()):,} bytes as JPEG")

    print("\n=== DUPLICATE IDS ===")
    print(Counter(df["id"]).most_common(3))


if __name__ == "__main__":
    main()
