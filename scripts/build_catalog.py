"""Build the deployment catalog from the 44k-row source dataset.

Why a subset: the full dataset is ~270 MB of images and adds nothing to a
styling demo. What the demo needs is coverage, not volume.

Why not random: the source is severely skewed - 77% "Casual", 7,065 t-shirts,
only 109 women's "Formal" items. A uniform sample would leave whole cells empty
and the recommender would fail for reasons unrelated to the recommender.

So: stratified quotas per (gender x slot), split across occasion groups, then
round-robin across (articleType x colour) buckets. Ordered by product id, never
by random draw, so re-running produces an identical catalog. A final pass
guarantees staple combinations (black trousers, white shirts) that proportional
sampling misses.

Run:  python scripts/build_catalog.py
Out:  data/catalog.csv, data/catalog_stats.json, static/products/<id>.jpg
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    SLOT_ACCESSORY,
    SLOT_BOTTOM,
    SLOT_FOOTWEAR,
    SLOT_ONEPIECE,
    SLOT_OUTERWEAR,
    SLOT_TOP,
    add_derived_features,
    wearable_mask,
)

DATASET_ID = "ashraq/fashion-product-images-small"
DATA_DIR = ROOT / "data"
# Served over HTTP by Streamlit (see .streamlit/config.toml), which is why this
# lives under static/ rather than assets/.
IMAGE_DIR = ROOT / "static" / "products"

# Source images are 60x80 - genuinely low resolution. We resample 3x with
# Lanczos and apply a mild unsharp mask, which recovers apparent edge definition
# lost to interpolation. This adds no information; it is purely presentational,
# and the UI caps display size at roughly the resampled size so images are never
# blown up further in the browser.
IMAGE_UPSCALE = 3
JPEG_QUALITY = 90
UNSHARP = ImageFilter.UnsharpMask(radius=1.2, percent=85, threshold=2)

# --------------------------------------------------------------------------
# Quotas
# --------------------------------------------------------------------------
# Sized so that every slot a template can ask for has enough candidates to
# survive hard filtering. Tops and footwear get the largest share because they
# appear in every outfit template and carry the most style variation.
SLOT_QUOTAS: dict[str, dict[str, int]] = {
    "Men": {
        SLOT_TOP: 70,
        SLOT_BOTTOM: 50,
        SLOT_FOOTWEAR: 50,
        SLOT_OUTERWEAR: 18,
        SLOT_ACCESSORY: 40,
    },
    "Women": {
        SLOT_TOP: 65,
        SLOT_BOTTOM: 45,
        SLOT_FOOTWEAR: 50,
        SLOT_OUTERWEAR: 12,
        SLOT_ACCESSORY: 40,
        SLOT_ONEPIECE: 40,
    },
    "Unisex": {
        SLOT_TOP: 5,
        SLOT_FOOTWEAR: 10,
        SLOT_ACCESSORY: 25,
    },
}

# The dataset's `usage` values, collapsed into four occasion groups.
USAGE_GROUPS: dict[str, set[str]] = {
    "casual": {"Casual"},
    "formal": {"Formal", "Smart Casual", "Party"},
    "ethnic": {"Ethnic"},
    "sports": {"Sports", "Travel"},
}

# Share of each slot's quota per occasion group. Deliberately over-samples
# formal and ethnic wear relative to the source distribution - that is the whole
# point of stratifying.
GROUP_SHARES: dict[str, dict[str, float]] = {
    SLOT_TOP: {"casual": 0.45, "formal": 0.30, "ethnic": 0.15, "sports": 0.10},
    SLOT_BOTTOM: {"casual": 0.45, "formal": 0.30, "ethnic": 0.15, "sports": 0.10},
    SLOT_FOOTWEAR: {"casual": 0.45, "formal": 0.30, "ethnic": 0.05, "sports": 0.20},
    SLOT_OUTERWEAR: {"casual": 0.50, "formal": 0.35, "ethnic": 0.05, "sports": 0.10},
    SLOT_ACCESSORY: {"casual": 0.50, "formal": 0.30, "ethnic": 0.10, "sports": 0.10},
    SLOT_ONEPIECE: {"casual": 0.35, "formal": 0.20, "ethnic": 0.45, "sports": 0.00},
}

# The dataset labels some children's products with an adult gender - e.g.
# "Gini and Jony Girl's Valerie Kidswear" carries gender="Women". Those items
# would be recommended to adult users, which looks broken. The gender column
# cannot catch them, so we screen on the product name as well.
KIDSWEAR_NAME_PATTERN = r"\b(kids?|kidswear|girl'?s|boy'?s|infant|toddler|junior)\b"

# --------------------------------------------------------------------------
# Staple coverage
# --------------------------------------------------------------------------
# Quota sampling balances article types and colours *proportionally*, which is
# right on average and wrong in the tail: the first build produced 28 pairs of
# trousers and not one black pair. "Black trousers" is a wardrobe staple, so a
# user asking for it hit an honest-but-useless dead end caused by our sampling,
# not by the source data.
#
# These combinations are therefore guaranteed: after the quotas are filled, any
# missing staple is added explicitly. Deterministic (lowest product id wins) and
# a handful of extra rows, but it means the obvious requests work.
STAPLES: dict[str, dict[str, tuple[str, ...]]] = {
    "Men": {
        "Shirts": ("Black", "White", "Blue", "Navy Blue", "Grey"),
        "Tshirts": ("Black", "White", "Navy Blue", "Grey"),
        "Trousers": ("Black", "Navy Blue", "Grey", "Brown", "Khaki"),
        "Jeans": ("Blue", "Black", "Navy Blue"),
        "Formal Shoes": ("Black", "Brown"),
        "Casual Shoes": ("White", "Black", "Brown"),
        "Jackets": ("Black", "Navy Blue"),
        "Kurtas": ("White", "Blue", "Maroon"),
    },
    "Women": {
        "Tops": ("Black", "White", "Blue", "Red"),
        "Shirts": ("Black", "White", "Blue"),
        "Tshirts": ("Black", "White"),
        "Trousers": ("Black", "Navy Blue", "Grey"),
        "Jeans": ("Blue", "Black"),
        "Dresses": ("Black", "Red", "Blue", "White"),
        "Kurtas": ("Black", "White", "Blue", "Pink"),
        "Sarees": ("Red", "Pink", "Blue"),
        "Heels": ("Black", "Brown"),
        "Flats": ("Black", "Brown"),
        "Skirts": ("Black",),
    },
}


def add_staples(df: pd.DataFrame, selected: list[int]) -> tuple[list[int], list[str]]:
    """Guarantee the staple (gender x articleType x colour) combinations."""
    chosen = set(selected)
    added: list[str] = []

    for gender, by_type in STAPLES.items():
        for article_type, colours in by_type.items():
            for colour in colours:
                already = df[
                    df["id"].isin(chosen)
                    & (df["gender"] == gender)
                    & (df["articleType"] == article_type)
                    & (df["baseColour"] == colour)
                ]
                if not already.empty:
                    continue

                candidates = df[
                    (df["gender"] == gender)
                    & (df["articleType"] == article_type)
                    & (df["baseColour"] == colour)
                ].sort_values("id")
                if candidates.empty:
                    added.append(f"MISSING UPSTREAM: {gender} {colour} {article_type}")
                    continue

                pick = int(candidates.iloc[0]["id"])
                chosen.add(pick)
                added.append(f"added: {gender} {colour} {article_type} (id {pick})")

    return sorted(chosen), added


CATALOG_COLUMNS = [
    "id", "productDisplayName", "brand", "gender", "masterCategory", "subCategory",
    "articleType", "baseColour", "season", "year", "usage",
    "outfit_slot", "formality", "formality_tier", "color_family",
    "is_neutral", "is_ethnic", "price", "mrp", "discount_pct", "image_file",
]


def round_robin_pick(candidates: pd.DataFrame, quota: int) -> list[int]:
    """Pick up to `quota` ids, cycling across (articleType, colour) buckets.

    Deterministic: buckets are visited in sorted key order and each bucket
    yields its items in ascending id order.
    """
    if quota <= 0 or candidates.empty:
        return []

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in candidates.sort_values("id").itertuples(index=False):
        buckets[(row.articleType, row.color_family)].append(int(row.id))

    ordered_keys = sorted(buckets)
    picked: list[int] = []
    depth = 0
    while len(picked) < quota:
        added_this_pass = False
        for key in ordered_keys:
            if depth < len(buckets[key]):
                picked.append(buckets[key][depth])
                added_this_pass = True
                if len(picked) == quota:
                    return picked
        if not added_this_pass:  # every bucket exhausted
            break
        depth += 1
    return picked


def select_ids(df: pd.DataFrame) -> tuple[list[int], list[dict]]:
    """Run the stratified selection. Returns (ids, per-cell report rows)."""
    selected: list[int] = []
    report: list[dict] = []

    for gender, slot_quotas in SLOT_QUOTAS.items():
        for slot, quota in slot_quotas.items():
            cell = df[(df["gender"] == gender) & (df["outfit_slot"] == slot)]
            shares = GROUP_SHARES[slot]

            picked_for_cell: list[int] = []
            shortfall = 0
            for group in ("formal", "ethnic", "sports", "casual"):
                # Casual is processed last so it can absorb any shortfall.
                if group == "casual":
                    group_quota = quota - len(picked_for_cell)
                else:
                    group_quota = int(round(quota * shares[group]))

                group_rows = cell[cell["usage"].isin(USAGE_GROUPS[group])]
                group_rows = group_rows[~group_rows["id"].isin(picked_for_cell)]
                got = round_robin_pick(group_rows, group_quota)
                picked_for_cell.extend(got)
                if group != "casual" and len(got) < group_quota:
                    shortfall += group_quota - len(got)

            selected.extend(picked_for_cell)
            report.append({
                "gender": gender,
                "slot": slot,
                "quota": quota,
                "selected": len(picked_for_cell),
                "unfilled_specialist_quota": shortfall,
            })

    return selected, report


def main() -> None:
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")

    meta_cols = [c for c in ds.column_names if c != "image"]
    df = ds.select_columns(meta_cols).to_pandas()
    print(f"  source rows: {len(df):,}")

    df = df[wearable_mask(df)]
    df = df[df["gender"].isin(["Men", "Women", "Unisex"])]
    df = df.dropna(subset=["baseColour", "articleType", "productDisplayName"])
    df = df.drop_duplicates(subset=["id"])

    mislabelled_kids = df["productDisplayName"].str.contains(
        KIDSWEAR_NAME_PATTERN, case=False, regex=True, na=False
    )
    df = df[~mislabelled_kids]
    print(f"  dropped {int(mislabelled_kids.sum()):,} children's items mislabelled as adult")
    print(f"  wearable rows for Men/Women/Unisex: {len(df):,}")

    df = add_derived_features(df)

    selected_ids, report = select_ids(df)
    print(f"\n  selected {len(selected_ids)} products by quota")
    print(pd.DataFrame(report).to_string(index=False))

    selected_ids, staple_log = add_staples(df, selected_ids)
    filled = [line for line in staple_log if line.startswith("added")]
    missing = [line for line in staple_log if line.startswith("MISSING")]
    print(f"\n  staple coverage: {len(filled)} added, {len(missing)} unavailable upstream")
    for line in filled + missing:
        print(f"    {line}")
    print(f"\n  catalog size after staples: {len(selected_ids)}")

    catalog = df[df["id"].isin(selected_ids)].copy()
    catalog = catalog.sort_values("id").reset_index(drop=True)
    catalog["image_file"] = catalog["id"].astype(str) + ".jpg"

    # --- export images -----------------------------------------------------
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in IMAGE_DIR.glob("*.jpg"):
        stale.unlink()

    wanted = set(catalog["id"].tolist())
    written = 0
    id_to_index = {int(v): i for i, v in enumerate(ds["id"])}
    for pid in sorted(wanted):
        image = ds[id_to_index[pid]]["image"].convert("RGB")
        image = image.resize(
            (image.width * IMAGE_UPSCALE, image.height * IMAGE_UPSCALE),
            resample=Image.Resampling.LANCZOS,
        ).filter(UNSHARP)
        image.save(
            IMAGE_DIR / f"{pid}.jpg",
            format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True,
        )
        written += 1
    print(f"\n  wrote {written} images to {IMAGE_DIR.relative_to(ROOT)}")

    # --- export catalog ----------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog[CATALOG_COLUMNS].to_csv(DATA_DIR / "catalog.csv", index=False)
    print(f"  wrote {DATA_DIR.relative_to(ROOT)}/catalog.csv")

    # --- export stats used by the documentation ----------------------------
    stats = {
        "source_dataset": DATASET_ID,
        "source_rows": int(len(ds)),
        "catalog_rows": int(len(catalog)),
        "images_written": written,
        "by_gender": catalog["gender"].value_counts().to_dict(),
        "by_slot": catalog["outfit_slot"].value_counts().to_dict(),
        "by_usage": catalog["usage"].value_counts().to_dict(),
        "by_formality_tier": catalog["formality_tier"].value_counts().to_dict(),
        "by_color_family": catalog["color_family"].value_counts().to_dict(),
        "article_types": int(catalog["articleType"].nunique()),
        "price_inr": {
            "min": int(catalog["price"].min()),
            "median": int(catalog["price"].median()),
            "max": int(catalog["price"].max()),
        },
        "price_by_slot": {
            slot: {
                "min": int(g["price"].min()),
                "median": int(g["price"].median()),
                "max": int(g["price"].max()),
                "count": int(len(g)),
            }
            for slot, g in catalog.groupby("outfit_slot")
        },
        "coverage_report": report,
    }
    (DATA_DIR / "catalog_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  wrote {DATA_DIR.relative_to(ROOT)}/catalog_stats.json")

    print("\n=== CATALOG SUMMARY ===")
    print(catalog.groupby(["gender", "outfit_slot"]).size().to_string())
    print("\nformality tiers:")
    print(catalog["formality_tier"].value_counts().to_string())
    print(
        "\nNOTE: images above are the 60x80 originals. Run "
        "`python scripts/upgrade_images.py` to replace them with the 384x512 "
        "re-export of the same catalog."
    )
    print("\nprice per slot (INR):")
    print(catalog.groupby("outfit_slot")["price"].describe()[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
